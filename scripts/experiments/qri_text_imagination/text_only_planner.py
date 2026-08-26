#!/usr/bin/env python3
"""Text-only regional imagination with probability sampling and ROI reinspection."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PLUGIN_ROOT = PROJECT_ROOT / "plugins" / "qwen_imagination"
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src", PLUGIN_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from qwen_imagination.regional.qwen import _data_url, _json_object  # noqa: E402
from qwen_imagination.regional.schema import Region  # noqa: E402
from qwen_imagination.regional.visual_context import roi_comparison_board  # noqa: E402


EVIDENCE_BASES = {"visual_evidence", "world_knowledge", "mixed", "unresolved"}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_bbox(value: object, size: tuple[int, int]) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox_xyxy must contain four integers")
    left, top, right, bottom = (int(item) for item in value)
    if not (0 <= left < right <= size[0] and 0 <= top < bottom <= size[1]):
        raise ValueError(f"bbox_xyxy {value} lies outside image size {size}")
    return left, top, right, bottom


def normalize_hypotheses(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("hypotheses")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("VLM must return at least two hypotheses")
    normalized = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"hypothesis {index} must be an object")
        description = str(raw.get("description", "")).strip()
        if not description:
            raise ValueError(f"hypothesis {index} has no description")
        basis = str(raw.get("basis", "unresolved")).strip().lower()
        if basis not in EVIDENCE_BASES:
            raise ValueError(f"hypothesis {index} has unsupported basis {basis}")
        probability = max(0.0, float(raw.get("probability", 0.0)))
        normalized.append(
            {
                "description": description,
                "probability": probability,
                "basis": basis,
                "observable_support": str(raw.get("observable_support", "")).strip(),
                "uncertainty": str(raw.get("uncertainty", "")).strip(),
            }
        )
    total = sum(row["probability"] for row in normalized)
    if total <= 0.0:
        raise ValueError("hypothesis probabilities must contain positive mass")
    for row in normalized:
        row["probability"] = row["probability"] / total
    return normalized


def sample_text_worlds(
    hypotheses: list[dict[str, Any]], sample_count: int, seed: int
) -> dict[str, Any]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    rng = random.Random(int(seed))
    indices = rng.choices(
        range(len(hypotheses)),
        weights=[row["probability"] for row in hypotheses],
        k=int(sample_count),
    )
    counts = Counter(indices)
    return {
        "sample_count": int(sample_count),
        "seed": int(seed),
        "draws": indices,
        "empirical_worlds": [
            {
                "hypothesis_index": index,
                "description": hypotheses[index]["description"],
                "basis": hypotheses[index]["basis"],
                "sample_count": counts.get(index, 0),
                "sample_frequency": counts.get(index, 0) / float(sample_count),
                "caption": (
                    f"regional text hypothesis: {hypotheses[index]['description']} "
                    f"[basis={hypotheses[index]['basis']}]"
                ),
            }
            for index in range(len(hypotheses))
        ],
    }


def normalized_entropy(hypotheses: list[dict[str, Any]]) -> float:
    if len(hypotheses) <= 1:
        return 0.0
    entropy = -sum(
        row["probability"] * math.log(max(row["probability"], 1e-12))
        for row in hypotheses
    )
    return float(entropy / math.log(len(hypotheses)))


def planner_instruction(sample: dict[str, Any]) -> str:
    return (
        "Act as a text-only regional imagination reasoner for a low-resolution person image. "
        "You will never edit or generate pixels. Image A is the authoritative low-resolution "
        "measurement. Image B is a SwinIR proposal whose added detail is not fact. The ROI board "
        "contains A-tight, A-context, B-tight and B-context views. Use two complementary abilities: "
        "(1) abstract world knowledge: recognize a known object, visual motif, cultural character "
        "or conventional design when enough evidence exists, and explain which visible abstraction "
        "supports it; (2) reasonable imagination: preserve several mutually exclusive explanations "
        "when the ROI is blurry. Do not force a positive object. Never convert a generic prior into "
        "an observation. Separate observations, world-knowledge inference and unresolved content. "
        "Return JSON only with this contract: "
        '{"region_summary":"...","observations":[{"claim":"...","evidence_strength":'
        '"strong|weak","evidence":"..."}],"world_knowledge":[{"inference":"...",'
        '"knowledge_used":"...","probability":0.0,"evidence_relation":"..."}],'
        '"hypotheses":[{"description":"...","probability":0.0,"basis":'
        '"visual_evidence|world_knowledge|mixed|unresolved","observable_support":"...",'
        '"uncertainty":"..."}],"unresolved":["..."]}. Return 3-5 hypotheses whose '
        "probabilities sum to one and include an unresolved alternative whenever pixels do not "
        "select a unique interpretation. Descriptions should be specific enough for a text encoder "
        "but must not contain image-generation instructions. Focus question: "
        + str(sample["focus_question"])
    )


def request_json(
    *,
    endpoint: str,
    model_id: str,
    content: list[dict[str, Any]],
    instruction: str,
    enable_thinking: bool,
    reasoning_effort: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": content},
        ],
        "temperature": float(temperature),
        "top_p": 0.9,
        "seed": int(seed),
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        "reasoning_effort": str(reasoning_effort),
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
        document = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    message = document["choices"][0]["message"]
    parsed = None
    response_field = None
    errors = []
    for field in ("content", "reasoning_content"):
        text = message.get(field)
        if not text:
            continue
        try:
            parsed = _json_object(str(text))
            response_field = field
            break
        except ValueError as error:
            errors.append(f"{field}:{type(error).__name__}")
    if parsed is None:
        raise ValueError(f"Qwen returned no final JSON ({', '.join(errors)})")
    telemetry = {
        "elapsed_seconds": elapsed,
        "response_field": response_field,
        "usage": document.get("usage", {}),
        "reasoning_characters": len(str(message.get("reasoning_content", ""))),
        "content_characters": len(str(message.get("content", ""))),
    }
    return parsed, telemetry


def record_metrics(record: dict[str, Any]) -> dict[str, Any]:
    hypotheses = record["hypotheses"]
    observations = record["response"].get("observations", [])
    knowledge = record["response"].get("world_knowledge", [])
    return {
        "elapsed_seconds": record["telemetry"]["elapsed_seconds"],
        "observation_count": len(observations) if isinstance(observations, list) else 0,
        "world_knowledge_count": len(knowledge) if isinstance(knowledge, list) else 0,
        "hypothesis_count": len(hypotheses),
        "mean_hypothesis_characters": sum(len(row["description"]) for row in hypotheses)
        / len(hypotheses),
        "normalized_entropy": normalized_entropy(hypotheses),
        "unresolved_probability": sum(
            row["probability"] for row in hypotheses if row["basis"] == "unresolved"
        ),
        "has_unresolved_hypothesis": any(
            row["basis"] == "unresolved" for row in hypotheses
        ),
        "complete_support_fields": all(
            bool(row["observable_support"]) and bool(row["uncertainty"])
            for row in hypotheses
        ),
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if int(spec.get("schema_version", 0)) != 2:
        raise ValueError("text-only experiment requires schema_version=2")
    output_root = (args.output_root or Path(spec["output_root"])).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    prepared = []
    for sample in spec["samples"]:
        lr = Image.open(sample["lr"]).convert("RGB")
        swin = Image.open(sample["swin"]).convert("RGB")
        bbox = parse_bbox(sample["bbox_xyxy"], swin.size)
        mask = np.zeros((swin.height, swin.width), dtype=bool)
        mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = True
        region = Region(str(sample["region_id"]), str(sample["category"]), bbox, mask)
        board = roi_comparison_board(lr, swin, region, int(spec["board_size_px"]))
        board_path = output_root / "boards" / f"{sample['name']}.png"
        board_path.parent.mkdir(parents=True, exist_ok=True)
        board.save(board_path, compress_level=2)
        content = [
            {
                "type": "text",
                "text": (
                    "Image A is authoritative LR; Image B and the lower ROI tiles are "
                    "non-authoritative SwinIR proposals."
                ),
            },
            {"type": "image_url", "image_url": {"url": _data_url(lr)}},
            {"type": "image_url", "image_url": {"url": _data_url(swin)}},
            {
                "type": "text",
                "text": f"Enlarged ROI board for {sample['region_id']} / {sample['category']}",
            },
            {"type": "image_url", "image_url": {"url": _data_url(board)}},
        ]
        prepared.append((sample, content, board_path))

    records = []
    for mode in spec["modes"]:
        for sample_index, (sample, content, board_path) in enumerate(prepared):
            response, telemetry = request_json(
                endpoint=str(spec["endpoint"]),
                model_id=str(spec["model_id"]),
                content=content,
                instruction=planner_instruction(sample),
                enable_thinking=bool(mode["enable_thinking"]),
                reasoning_effort=str(mode["reasoning_effort"]),
                seed=int(spec["seed"]) + sample_index,
                temperature=float(spec["temperature"]),
                max_tokens=int(spec["max_tokens"]),
                timeout_seconds=float(spec["timeout_seconds"]),
            )
            hypotheses = normalize_hypotheses(response)
            worlds = sample_text_worlds(
                hypotheses,
                int(spec["world_sample_count"]),
                int(spec["seed"]) + sample_index,
            )
            record = {
                "sample": sample,
                "mode": mode,
                "board": str(board_path),
                "response": response,
                "hypotheses": hypotheses,
                "sampled_text_worlds": worlds,
                "telemetry": telemetry,
            }
            record["metrics"] = record_metrics(record)
            record_path = output_root / "records" / mode["name"] / f"{sample['name']}.json"
            atomic_json(record_path, record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "mode": mode["name"],
                        "sample": sample["name"],
                        "seconds": round(telemetry["elapsed_seconds"], 3),
                        "hypotheses": len(hypotheses),
                        "knowledge": record["metrics"]["world_knowledge_count"],
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

    mode_summaries = {}
    for mode in spec["modes"]:
        selected = [row for row in records if row["mode"]["name"] == mode["name"]]
        mode_summaries[mode["name"]] = {
            "sample_count": len(selected),
            "mean_elapsed_seconds": mean(
                [row["metrics"]["elapsed_seconds"] for row in selected]
            ),
            "mean_observation_count": mean(
                [row["metrics"]["observation_count"] for row in selected]
            ),
            "mean_world_knowledge_count": mean(
                [row["metrics"]["world_knowledge_count"] for row in selected]
            ),
            "mean_hypothesis_count": mean(
                [row["metrics"]["hypothesis_count"] for row in selected]
            ),
            "mean_hypothesis_characters": mean(
                [row["metrics"]["mean_hypothesis_characters"] for row in selected]
            ),
            "mean_normalized_entropy": mean(
                [row["metrics"]["normalized_entropy"] for row in selected]
            ),
            "mean_unresolved_probability": mean(
                [row["metrics"]["unresolved_probability"] for row in selected]
            ),
            "valid_unresolved_rate": mean(
                [float(row["metrics"]["has_unresolved_hypothesis"]) for row in selected]
            ),
            "complete_support_rate": mean(
                [float(row["metrics"]["complete_support_fields"]) for row in selected]
            ),
        }
    names = [mode["name"] for mode in spec["modes"]]
    comparison = {}
    if len(names) == 2:
        left, right = (mode_summaries[name] for name in names)
        comparison = {
            "left_mode": names[0],
            "right_mode": names[1],
            "right_to_left_latency_ratio": right["mean_elapsed_seconds"]
            / max(left["mean_elapsed_seconds"], 1e-12),
            "delta_mean_world_knowledge_count": right["mean_world_knowledge_count"]
            - left["mean_world_knowledge_count"],
            "delta_mean_hypothesis_characters": right["mean_hypothesis_characters"]
            - left["mean_hypothesis_characters"],
            "delta_mean_unresolved_probability": right["mean_unresolved_probability"]
            - left["mean_unresolved_probability"],
        }
    payload = {
        "schema_version": 2,
        "experiment_id": spec["experiment_id"],
        "config": str(args.config.resolve()),
        "record_count": len(records),
        "mode_summaries": mode_summaries,
        "comparison": comparison,
        "records": records,
    }
    atomic_json(output_root / "metrics.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
