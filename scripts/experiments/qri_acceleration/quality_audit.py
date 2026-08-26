#!/usr/bin/env python3
"""Audit PASD candidates for LR consistency and ReID identity preservation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from easydict import EasyDict
from PIL import Image
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PLUGIN_ROOT = PROJECT_ROOT / "plugins" / "qwen_imagination"
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PLUGIN_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qwen_imagination.regional.composite import lr_cycle_energy  # noqa: E402
from salt_vi.engine import build_model  # noqa: E402


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_identity_model(config_path: Path, checkpoint: Path, device: torch.device):
    with config_path.open("r", encoding="utf-8") as handle:
        config = EasyDict(yaml.load(handle, Loader=yaml.UnsafeLoader))
    config.gpu_id = "0"
    model = build_model(config)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, config


def image_tensor(image: Image.Image, height: int, width: int) -> torch.Tensor:
    pipeline = transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return pipeline(image.convert("RGB")).unsqueeze(0)


@torch.inference_mode()
def identity_feature(model, image: Image.Image, modality: str, config, device):
    tensor = image_tensor(image, int(config.img_h), int(config.img_w)).to(device)
    mode = "IR" if modality.lower() == "ir" else "RGB"
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        visual = model.encode_image_featmap(tensor, mode.lower())
        feature = model.classifier(model.extract_global_feat(visual), mode)
    return feature.float().cpu()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--identity-config", required=True, type=Path)
    parser.add_argument("--identity-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    base = json.loads((run_root / "metrics" / "prepare_pilot.json").read_text())
    pasd = json.loads((run_root / "metrics" / "pasd_pilot.json").read_text())
    base_by_key = {item["source_key"]: item for item in base["records"]}
    device = torch.device(args.device)
    model, identity_config = load_identity_model(
        args.identity_config, args.identity_checkpoint, device
    )

    reference_features = {}
    reference_cycles = {}
    records = []
    for row in pasd["records"]:
        source_key = row["source_key"]
        base_row = base_by_key[source_key]
        lr = Image.open(run_root / base_row["lr"]).convert("RGB")
        reference = Image.open(run_root / base_row["swin"]).convert("RGB")
        candidate = Image.open(run_root / row["composite"]).convert("RGB")
        if source_key not in reference_features:
            reference_features[source_key] = identity_feature(
                model, reference, base_row["modality"], identity_config, device
            )
            reference_cycles[source_key] = float(
                lr_cycle_energy(reference, lr, base_row["modality"])
            )
        candidate_feature = identity_feature(
            model, candidate, base_row["modality"], identity_config, device
        )
        identity_cosine = float(
            torch.nn.functional.cosine_similarity(
                reference_features[source_key], candidate_feature
            ).item()
        )
        cycle = float(row["metrics"]["lr_cycle_energy"])
        baseline_cycle = reference_cycles[source_key]
        records.append(
            {
                **row,
                "quality": {
                    "reference_lr_cycle_energy": baseline_cycle,
                    "lr_cycle_ratio_to_reference": cycle / max(baseline_cycle, 1e-12),
                    "identity_cosine_to_swin": identity_cosine,
                },
            }
        )
        print(
            json.dumps(
                {
                    "source": source_key,
                    "variant": row["variant"]["name"],
                    "cycle_ratio": round(cycle / max(baseline_cycle, 1e-12), 4),
                    "identity_cosine": round(identity_cosine, 6),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    output = {
        "schema_version": 1,
        "identity_config": str(args.identity_config),
        "identity_checkpoint": str(args.identity_checkpoint),
        "records": records,
    }
    path = run_root / "metrics" / "quality_audit.json"
    atomic_json(path, output)
    print(json.dumps({"metrics": str(path), "record_count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
