#!/usr/bin/env python
"""Single-process SYSU evaluator supporting MER, TTA, re-ranking, and ensembles."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as torch_functional


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import EXPECTED_E4, git_commit_sha, load_yaml, protocol_guard, sha256_file, utc_now
from eval_ops import aggregate_tta, l2_normalize, rerank_cosine, similarity, weighted_mer_score
from salt_vi.config.config_rn import get_args  # noqa: F401 - ensures configuration dependencies are importable.
from salt_vi.engine import build_model
from salt_vi.data.loader import Loader
from salt_vi.utils import eval_sysu
from salt_vi.utils.utils import load_train_configs


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must resolve to a state dictionary, got {type(payload).__name__}")
    return payload


def _resize_positional_state(
    state_dict: Mapping[str, torch.Tensor], model: torch.nn.Module, source_size: Sequence[int]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    prepared = dict(state_dict)
    key = "base_model.visual.vit.pos_embed"
    target = model.state_dict().get(key)
    source = prepared.get(key)
    audit = {"resized": False, "key": key}
    if source is None or target is None or tuple(source.shape) == tuple(target.shape):
        return prepared, audit
    visual = model.base_model.visual.vit
    patch_height, patch_width = visual.patch_embed.patch_size
    stride_height, stride_width = visual.patch_embed.stride_size
    old_height = (int(source_size[0]) - patch_height) // stride_height + 1
    old_width = (int(source_size[1]) - patch_width) // stride_width + 1
    new_height, new_width = visual.base_grid_size
    if source.shape[1] != old_height * old_width + 1:
        raise ValueError(
            f"Cannot reshape source positional embedding {tuple(source.shape)} as "
            f"{old_height}x{old_width}+CLS"
        )
    token, grid = source[:, :1], source[:, 1:]
    grid = grid.reshape(1, old_height, old_width, -1).permute(0, 3, 1, 2).float()
    grid = torch_functional.interpolate(
        grid, size=(new_height, new_width), mode="bilinear", align_corners=False
    )
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_height * new_width, -1).to(source.dtype)
    prepared[key] = torch.cat([token, grid], dim=1)
    audit = {
        "resized": True,
        "key": key,
        "source_shape": list(source.shape),
        "target_shape": list(target.shape),
        "source_grid": [old_height, old_width],
        "target_grid": [new_height, new_width],
    }
    return prepared, audit


def load_model_checkpoint(model: torch.nn.Module, checkpoint: Path, source_size=(288, 144)) -> Dict[str, Any]:
    payload = torch.load(str(checkpoint), map_location="cpu")
    state_dict, resize_audit = _resize_positional_state(unwrap_state_dict(payload), model, source_size)
    model_state = model.state_dict()
    incompatible_shapes = {
        key: {"checkpoint": list(value.shape), "model": list(model_state[key].shape)}
        for key, value in state_dict.items()
        if key in model_state and hasattr(value, "shape") and tuple(value.shape) != tuple(model_state[key].shape)
    }
    if incompatible_shapes:
        raise RuntimeError(f"Checkpoint contains incompatible tensor shapes: {incompatible_shapes}")
    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {result.unexpected_keys}")
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "sha256": sha256_file(checkpoint),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "positional_embedding": resize_audit,
    }


def _view_tensors(images: torch.Tensor, scales: Sequence[Sequence[int]], flip: bool) -> List[torch.Tensor]:
    result = []
    for height, width in scales:
        size = (int(height), int(width))
        view = images if tuple(images.shape[-2:]) == size else torch_functional.interpolate(
            images, size=size, mode="bilinear", align_corners=False
        )
        result.append(view)
        if flip:
            result.append(torch.flip(view, dims=[3]))
    return result


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _classified_image(model, images, mode: str) -> torch.Tensor:
    feature_map = model.encode_image_featmap(images, mode.lower())
    embedding = model.extract_global_feat(feature_map)
    return model.classifier(embedding, mode)


def _classified_text(model, text: torch.Tensor) -> torch.Tensor:
    if text.ndim == 2:
        return model.classifier(model.encode_text_feat(text), "Text")
    if text.ndim != 3:
        raise ValueError(f"Expected text shape [B,L] or [B,K,L], got {tuple(text.shape)}")
    batch, captions, length = text.shape
    flattened = text.reshape(batch * captions, length)
    values = model.classifier(model.encode_text_feat(flattened), "Text")
    values = values.reshape(batch, captions, -1)
    return torch_functional.normalize(values.mean(dim=1), dim=-1)


def extract_feature_bundle(model, loader: Loader, config, device: torch.device) -> Dict[str, Any]:
    model.set_eval()
    scales = [tuple(item) for item in getattr(config, "test_multi_scale", [[config.img_h, config.img_w]])]
    flip = bool(getattr(config, "test_flip_tta", False))
    use_mer = bool(config.CAT_EVAL)
    query_parts = {"Fusion": [], "IR": [], "Text": []}
    with torch.no_grad():
        for batch in loader.query_loader:
            images = batch["img"].to(device)
            text = batch["text"].to(device).long()
            fusion_views = []
            ir_views = []
            for view in _view_tensors(images, scales, flip):
                fusion = model.classifier(model.encode_fusion(text, view, "ir"), "Fusion")
                fusion_views.append(_as_numpy(fusion))
                if use_mer:
                    ir_views.append(_as_numpy(_classified_image(model, view, "IR")))
            query_parts["Fusion"].append(aggregate_tta(fusion_views))
            if use_mer:
                query_parts["IR"].append(aggregate_tta(ir_views))
                query_parts["Text"].append(_as_numpy(_classified_text(model, text)))

    queries = {"Fusion": np.concatenate(query_parts["Fusion"], axis=0)}
    if use_mer:
        queries["IR"] = np.concatenate(query_parts["IR"], axis=0)
        queries["Text"] = l2_normalize(np.concatenate(query_parts["Text"], axis=0))

    galleries = []
    with torch.no_grad():
        for gallery_loader in loader.gallery_loaders:
            chunks = []
            for batch in gallery_loader:
                images = batch["img"].to(device)
                views = [
                    _as_numpy(_classified_image(model, view, "RGB"))
                    for view in _view_tensors(images, scales, flip)
                ]
                chunks.append(aggregate_tta(views))
            galleries.append(np.concatenate(chunks, axis=0))
    return {"queries": queries, "galleries": galleries, "scales": scales, "flip": flip}


def _score(features: Mapping[str, Any], gallery: np.ndarray, config) -> np.ndarray:
    if bool(config.CAT_EVAL):
        weights = (
            float(getattr(config, "mer_fusion_weight", 1.0)),
            float(getattr(config, "mer_ir_weight", 1.0)),
            float(getattr(config, "mer_text_weight", 1.0)),
        )
        return weighted_mer_score(
            features["queries"]["Fusion"],
            features["queries"]["IR"],
            features["queries"]["Text"],
            gallery,
            weights=weights,
            normalize=bool(getattr(config, "mer_l2_normalize", False)),
        )
    return similarity(features["queries"]["Fusion"], gallery, normalize=False)


def _rerank_features(features: Mapping[str, Any], gallery: np.ndarray, config) -> Tuple[np.ndarray, np.ndarray]:
    if not bool(config.CAT_EVAL):
        return features["queries"]["Fusion"], gallery
    weights = np.asarray(
        [
            float(getattr(config, "mer_fusion_weight", 1.0)),
            float(getattr(config, "mer_ir_weight", 1.0)),
            float(getattr(config, "mer_text_weight", 1.0)),
        ],
        dtype=np.float32,
    )
    if np.any(weights < 0):
        raise ValueError("Re-ranking requires non-negative MER weights")
    roots = np.sqrt(weights)
    query = np.concatenate(
        [
            l2_normalize(features["queries"][name]) * roots[index]
            for index, name in enumerate(("Fusion", "IR", "Text"))
        ],
        axis=1,
    )
    combined_gallery = np.concatenate(
        [l2_normalize(gallery) * roots[index] for index in range(3)], axis=1
    )
    return query, combined_gallery


def _average_feature_bundles(bundles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    queries = {}
    for name in bundles[0]["queries"]:
        queries[name] = aggregate_tta([bundle["queries"][name] for bundle in bundles])
    galleries = [
        aggregate_tta([bundle["galleries"][trial] for bundle in bundles])
        for trial in range(len(bundles[0]["galleries"]))
    ]
    return {"queries": queries, "galleries": galleries}


def evaluate_bundles(bundles: Sequence[Mapping[str, Any]], loader: Loader, config) -> Dict[str, Any]:
    ensemble_mode = str(getattr(config, "ensemble_mode", "none"))
    if len(bundles) == 1:
        ensemble_mode = "none"
    if ensemble_mode not in {"none", "feature", "score"}:
        raise ValueError(f"Unsupported ensemble_mode: {ensemble_mode}")
    if ensemble_mode == "feature":
        bundles = [_average_feature_bundles(bundles)]
    if ensemble_mode == "score" and bool(getattr(config, "rerank", False)):
        raise ValueError("Score ensemble with re-ranking is not supported; keep results separate")

    all_cmc = None
    all_map = 0.0
    all_minp = 0.0
    trials = []
    labels_list = getattr(loader, "gallery_labels", [loader.gall_label] * 10)
    cams_list = getattr(loader, "gallery_cams", [loader.gall_cam] * 10)
    for trial in range(10):
        gallery = bundles[0]["galleries"][trial]
        if ensemble_mode == "score":
            score = np.mean([_score(bundle, bundle["galleries"][trial], config) for bundle in bundles], axis=0)
            distance = -score
        elif bool(getattr(config, "rerank", False)):
            query, rerank_gallery = _rerank_features(bundles[0], gallery, config)
            distance = rerank_cosine(
                query,
                rerank_gallery,
                k1=int(getattr(config, "rerank_k1", 20)),
                k2=int(getattr(config, "rerank_k2", 6)),
                lambda_value=float(getattr(config, "rerank_lambda", 0.3)),
            )
        else:
            distance = -_score(bundles[0], gallery, config)
        cmc, map_value, minp = eval_sysu(
            distance,
            loader.query_label,
            labels_list[trial],
            loader.query_cam,
            cams_list[trial],
        )
        all_cmc = cmc.copy() if all_cmc is None else all_cmc + cmc
        all_map += float(map_value)
        all_minp += float(minp)
        trials.append({"trial": trial, "Rank-1": float(cmc[0]), "mAP": float(map_value), "mINP": float(minp)})
    all_cmc /= 10.0
    return {
        "Rank-1": float(all_cmc[0]),
        "mAP": all_map / 10.0,
        "mINP": all_minp / 10.0,
        "trials": trials,
        "gallery_trials": 10,
        "ensemble_mode": ensemble_mode,
    }


def run(config_path: Path) -> Dict[str, Any]:
    config = load_train_configs(str(config_path))
    config.pid_num = 395
    protocol_payload = dict(vars(config))
    protocol_payload.setdefault("gallery_trials", 10)
    protocol_guard(protocol_payload)
    seed_everything(int(config.seed))
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal SYSU evaluation requires an explicitly leased CUDA GPU")
    loader = Loader(config)
    checkpoints = list(getattr(config, "ensemble_checkpoints", []) or [])
    if not checkpoints:
        checkpoints = [config.test_model_path]
    if len(checkpoints) > 5:
        raise ValueError("Checkpoint ensemble is limited to at most five models")
    bundles = []
    audits = []
    for checkpoint_value in checkpoints:
        checkpoint = Path(str(checkpoint_value)).resolve()
        model = build_model(config)
        audits.append(load_model_checkpoint(model, checkpoint, source_size=(288, 144)))
        model = model.to(device)
        bundles.append(extract_feature_bundle(model, loader, config, device))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    metrics = evaluate_bundles(bundles, loader, config)
    deltas = {key: (metrics[key] - EXPECTED_E4[key]) * 100.0 for key in EXPECTED_E4}
    return {
        "status": "succeeded",
        "generated_at": utc_now(),
        "git_commit_sha": git_commit_sha(),
        "config": str(Path(config_path).resolve()),
        "checkpoints": audits,
        "metrics": metrics,
        "delta_percentage_points": deltas,
        "protocol": {
            "dataset": "SYSU-MM01",
            "test_mode": "all-search",
            "gall_mode": "single-shot",
            "gallery_trials": 10,
            "test_labels_used_for_tuning": False,
            "MER": bool(config.CAT_EVAL),
            "TTA": bool(getattr(config, "test_flip_tta", False)) or len(getattr(config, "test_multi_scale", [])) > 1,
            "re_ranking": bool(getattr(config, "rerank", False)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = run(args.config)
    if args.output_json:
        from common import atomic_write_json

        atomic_write_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

