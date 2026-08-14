#!/usr/bin/env python3
"""Recluster a completed atomic sample pool without rerunning the VLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from semantic_imagination import cluster_hypothesis_samples, to_pasd_record
from transformers import AutoModel, AutoTokenizer


def canonical_state(category: str, value: str) -> str:
    value = value.casefold().replace("_", " ").strip()
    tokens = set(value.split())
    if value in {"absent", "no additional detail"}:
        return value.replace(" ", "_")
    rules = {
        "eyewear": [
            ("frame_style", {"frame", "frameless"}),
            ("lens_detail", {"lens", "transparent"}),
            ("eyewear_type", {"glass", "glasses", "sunglasses"}),
        ],
        "wrist_accessory": [
            ("watch", {"watch"}),
            ("bracelet", {"bracelet"}),
            ("wristband", {"wristband"}),
        ],
        "headwear": [
            ("cap", {"cap"}),
            ("hat", {"hat"}),
            ("hood", {"hood"}),
            ("other_headwear", {"beanie", "helmet", "visor"}),
        ],
        "body_marking": [
            ("tattoo", {"tattoo", "tattoos"}),
            ("scar", {"scar", "scars"}),
        ],
        "clothing_detail": [
            ("graphic", {"graphic", "design"}),
            ("pattern", {"patterned", "checkered", "plaid", "stripe", "striped"}),
            ("sleeve_detail", {"sleeve", "sleeves"}),
            ("color_detail", {"red", "blue", "green", "black", "white", "gray"}),
            ("other_clothing_detail", {"collar", "pocket"}),
        ],
        "carried_object": [
            ("backpack", {"backpack"}),
            ("shoulder_bag", {"shoulder", "bag"}),
            ("bottle", {"bottle"}),
            ("bag_accessory", {"strap"}),
            ("other_carried_object", {"umbrella", "book"}),
        ],
        "pocket_item": [
            ("zipper", {"zipper"}),
            ("phone", {"phone"}),
            ("keys", {"key", "keys"}),
            ("wallet", {"wallet"}),
            ("other_pocket_item", {"card"}),
        ],
        "footwear_detail": [
            ("laces", {"lace", "laces"}),
            ("strap", {"strap", "straps"}),
            ("toe_style", {"toe", "open"}),
            ("sock_detail", {"sock", "socks"}),
            ("color_detail", {"red", "blue", "green", "black", "white", "gray"}),
            ("texture", {"texture"}),
            ("other_footwear_detail", {"heel", "sole"}),
        ],
    }
    for state, keywords in rules.get(category, []):
        if tokens & keywords:
            return state
    return "invalid_output"


def add_canonical_state(text: str) -> str:
    fields = dict(part.split("=", 1) for part in text.split("; ") if "=" in part)
    category = fields.get("category", "invalid")
    state = canonical_state(category, fields.get("value", "invalid"))
    return (
        f"category={category}; state={state}; value={fields.get('value', 'invalid')}; "
        f"location={fields.get('location', 'none')}"
    )


def compose_atomic(observed: str, hypothesis: str) -> str:
    fields = dict(part.split("=", 1) for part in hypothesis.split("; ") if "=" in part)
    category = fields.get("category", "detail").replace("_", " ")
    value = fields.get("value", "no_additional_detail")
    location = fields.get("location", "none")
    if value == "no_additional_detail":
        possible = f"The pedestrian may have no additional {category} detail"
    elif value == "absent":
        possible = f"The pedestrian may have no {category}"
    else:
        possible = f"The pedestrian may have {value.replace('_', ' ')}"
    if location not in {"none", "absent"}:
        possible += f" at {location.replace('_', ' ')}"
    return f"{observed} {possible}."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embed-model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--pasd-seed", type=int, default=20260813)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.embed_model, local_files_only=True)
    model = AutoModel.from_pretrained(args.embed_model, local_files_only=True).eval().cpu()

    def embed(texts: list[str]) -> list[list[float]]:
        embedding_texts = []
        for text in texts:
            fields = dict(
                part.split("=", 1) for part in text.split("; ") if "=" in part
            )
            value = fields.get("value", text).replace("_", " ")
            location = fields.get("location", "none").replace("_", " ")
            embedding_texts.append(f"{value} at {location}")
        batch = tokenizer(
            embedding_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1).tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    summary_items = []
    for manifest_path in sorted(args.artifacts.glob("sample-*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        texts = [add_canonical_state(sample["text"]) for sample in manifest["samples"]]
        clusters = cluster_hypothesis_samples(
            texts, embed(texts), args.threshold, "complete"
        )
        samples = [
            {**sample, "text": text}
            for sample, text in zip(manifest["samples"], texts)
        ]
        hypotheses = []
        for cluster in clusters:
            hypothesis = dict(cluster)
            hypothesis["caption"] = compose_atomic(
                manifest["observed"], hypothesis["representative"]
            )
            hypotheses.append(hypothesis)
            for member in hypothesis["member_indices"]:
                samples[member]["cluster_id"] = hypothesis["cluster_id"]

        contract = dict(manifest["sampling_contract"])
        contract["cluster_linkage"] = "complete"
        backend_contract = dict(contract.get("backend_contract", {}))
        backend_contract["embedding_projection"] = "atomic-value-location-v1"
        backend_contract["atomic_hard_states"] = ["absent", "no_additional_detail"]
        backend_contract["canonical_state_taxonomy"] = "closed-category-specific-v1"
        contract["backend_contract"] = backend_contract
        encoded = json.dumps(
            contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        manifest.update(
            {
                "sampling_contract": contract,
                "sampling_contract_sha256": hashlib.sha256(encoded).hexdigest(),
                "samples": samples,
                "hypotheses": hypotheses,
                "cluster_linkage": "complete",
            }
        )
        item_dir = args.output_dir / manifest_path.parent.name
        item_dir.mkdir(parents=True, exist_ok=True)
        target = item_dir / "manifest.json"
        target.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        record = to_pasd_record(
            manifest,
            item_dir / "pasd_views",
            pasd_seed=args.pasd_seed,
            modality="rgb",
        )
        (item_dir / "pasd_record.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        records.append(record)
        summary_items.append(
            {
                "source": manifest["source_key"],
                "samples": len(samples),
                "clusters": len(hypotheses),
                "categories": sorted({item["category"] for item in hypotheses}),
                "weights_valid": abs(sum(item["weight"] for item in hypotheses) - 1.0)
                < 1e-9,
            }
        )

    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "source_artifacts": str(args.artifacts.resolve()),
        "similarity_threshold": args.threshold,
        "cluster_linkage": "complete",
        "embedding_projection": "atomic-value-location-v1",
        "atomic_hard_states": ["absent", "no_additional_detail"],
        "canonical_state_taxonomy": "closed-category-specific-v1",
        "items": summary_items,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
