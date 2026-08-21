from __future__ import annotations

import base64
import json
import math
import random
import re
import urllib.request
from collections import Counter
from io import BytesIO
from typing import Any, Protocol

from PIL import Image

from qwen_imagination.taxonomy import CATEGORY_STATES, normalize_symbol

from .schema import Candidate, JointSample, Region
from .visual_context import RESAMPLING, normalized_tight_crop


CRITIC_LABELS = {
    "pixel_supported",
    "compatible_prior_only",
    "contradicted",
    "abstain",
}


class RegionalReasoner(Protocol):
    model_id: str

    def propose(
        self, lr: Image.Image, swin: Image.Image, regions: list[Region]
    ) -> dict[str, list[Candidate]]: ...

    def sample_world(
        self,
        lr: Image.Image,
        swin: Image.Image,
        regions: list[Region],
        proposals: dict[str, list[Candidate]],
        seed: int,
    ) -> dict[str, Candidate]: ...

    def critique(
        self,
        lr: Image.Image,
        swin: Image.Image,
        regions: list[Region],
        assignments: list[dict[str, Candidate]],
    ) -> list[dict[str, dict[str, Any]]]: ...


def _json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for match in re.finditer(r"\{", text):
        try:
            value, consumed = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((match.start() + consumed, -match.start(), value))
    if not candidates:
        raise ValueError("Qwen response must contain one complete JSON object")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG", compress_level=3)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_content(
    lr: Image.Image,
    swin: Image.Image,
    regions: list[Region],
    crop_size_px: int = 384,
) -> list[dict]:
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Image A is the authoritative low-resolution measurement. Image B is a "
                "SwinIR proposal whose added high-frequency detail is not ground truth."
            ),
        },
        {"type": "image_url", "image_url": {"url": _data_url(lr)}},
        {"type": "image_url", "image_url": {"url": _data_url(swin)}},
    ]
    lr_aligned = lr.resize(swin.size, RESAMPLING.NEAREST)
    for region in regions:
        content.extend(
            (
                {
                    "type": "text",
                    "text": (
                        f"ROI {region.region_id}: normalized LR crop followed by the "
                        "corresponding SwinIR crop. Padding preserves crop aspect ratio."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url(
                            normalized_tight_crop(
                                lr_aligned,
                                region,
                                crop_size_px,
                                resample=RESAMPLING.NEAREST,
                            )
                        )
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url(
                            normalized_tight_crop(
                                swin,
                                region,
                                crop_size_px,
                                resample=RESAMPLING.LANCZOS,
                            )
                        )
                    },
                },
            )
        )
    return content


class LlamaServerQwenReasoner:
    """OpenAI-compatible llama.cpp client for the local GGUF + mmproj server."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8080/v1/chat/completions",
        model_id: str = "third-party-qwen3.8-27b-ud-q4-k-xl",
        timeout_seconds: float = 180.0,
        enable_thinking: bool = True,
        reasoning_effort: str = "high",
        roi_crop_size_px: int = 384,
    ):
        self.endpoint = endpoint
        self.model_id = model_id
        self.timeout_seconds = float(timeout_seconds)
        self.enable_thinking = bool(enable_thinking)
        self.reasoning_effort = str(reasoning_effort)
        self.roi_crop_size_px = int(roi_crop_size_px)

    def _complete(
        self,
        content: list[dict],
        instruction: str,
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": content},
            ],
            "temperature": float(temperature),
            "top_p": 0.9,
            "seed": int(seed),
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            "reasoning_effort": self.reasoning_effort,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            document = json.loads(response.read().decode("utf-8"))
        message = document["choices"][0]["message"]
        errors = []
        for field in ("content", "reasoning_content"):
            text = message.get(field)
            if not text:
                continue
            try:
                return _json_object(str(text))
            except (json.JSONDecodeError, ValueError) as error:
                errors.append(f"{field}={type(error).__name__}")
        raise ValueError(
            "Qwen response did not contain a parseable final JSON object "
            f"({', '.join(errors) or 'empty response'})"
        )

    def propose(
        self, lr: Image.Image, swin: Image.Image, regions: list[Region]
    ) -> dict[str, list[Candidate]]:
        board = [
            {
                "region_id": region.region_id,
                "category": region.category,
                "allowed_states": sorted(CATEGORY_STATES[region.category]),
            }
            for region in regions
        ]
        instruction = (
            "Act as a cautious visual reasoner for low-resolution person re-identification. "
            "Separate evidence visible in authoritative Image A from structures suggested only "
            "by Image B. For every ROI propose 2-4 mutually exclusive states compatible with A. "
            "Always include no_additional_detail as an unresolved control state. Do not equate "
            "a detail that is outside the visible view with absence. Do not invent identity, "
            "pose, body shape, color, or a detail contradicted by A. Return JSON only as "
            '{"regions":[{"region_id":...,"candidates":[{"state":...,'
            '"value":...,"evidence":...,"evidence_source":"pixel_supported|'
            'compatible_prior_only|abstain"}]}]}. Internal reasoning must not be returned. '
            f"Closed taxonomy: {json.dumps(board, separators=(',', ':'))}"
        )
        result = self._complete(
            _image_content(lr, swin, regions, self.roi_crop_size_px),
            instruction,
            seed=0,
            temperature=0.2,
            max_tokens=4096,
        )
        proposals: dict[str, list[Candidate]] = {}
        region_by_id = {region.region_id: region for region in regions}
        for item in result.get("regions", []):
            region_id = str(item.get("region_id", ""))
            if region_id not in region_by_id:
                continue
            region = region_by_id[region_id]
            candidates = []
            for raw in item.get("candidates", []):
                state = normalize_symbol(str(raw.get("state", "")))
                if state not in CATEGORY_STATES[region.category]:
                    continue
                source = normalize_symbol(
                    str(raw.get("evidence_source", "compatible_prior_only"))
                )
                if source not in CRITIC_LABELS - {"contradicted"}:
                    source = "compatible_prior_only"
                candidates.append(
                    Candidate(
                        state=state,
                        value=str(raw.get("value", state)).strip() or state,
                        evidence=str(raw.get("evidence", "")).strip(),
                        evidence_source=source,
                    )
                )
            unique = {candidate.key(): candidate for candidate in candidates}
            sentinel = "no_additional_detail"
            sentinel_key = (sentinel, sentinel)
            unique.setdefault(
                sentinel_key,
                Candidate(sentinel, sentinel, "not resolved by LR", "abstain"),
            )
            ordered = list(unique.values())
            if len(ordered) > 4:
                ordered = [
                    candidate
                    for candidate in ordered
                    if candidate.key() != sentinel_key
                ][:3]
                ordered.append(unique[sentinel_key])
            proposals[region_id] = ordered
        missing = sorted(set(region_by_id).difference(proposals))
        if missing:
            raise ValueError(f"Qwen proposal omitted regional ROIs: {missing}")
        return proposals

    def sample_world(
        self,
        lr: Image.Image,
        swin: Image.Image,
        regions: list[Region],
        proposals: dict[str, list[Candidate]],
        seed: int,
    ) -> dict[str, Candidate]:
        board = {
            region.region_id: [
                {"state": candidate.state, "value": candidate.value}
                for candidate in proposals[region.region_id]
            ]
            for region in regions
        }
        instruction = (
            "Select one jointly compatible candidate for every ROI. Image A is authoritative; "
            "Image B is only a proposal. Explore jointly compatible candidates without preferring "
            "abstention or treating an unobserved detail as absent. "
            'Return JSON only as {"assignments":[{"region_id":...,"state":...,'
            '"value":...}]}. Do not return reasoning. Candidate board: '
            + json.dumps(board, ensure_ascii=False, separators=(",", ":"))
        )
        result = self._complete(
            _image_content(lr, swin, regions, self.roi_crop_size_px),
            instruction,
            seed=seed,
            temperature=0.75,
            max_tokens=2048,
        )
        selected: dict[str, Candidate] = {}
        for raw in result.get("assignments", []):
            region_id = str(raw.get("region_id", ""))
            state = normalize_symbol(str(raw.get("state", "")))
            candidates = proposals.get(region_id, [])
            matches = [
                candidate for candidate in candidates if candidate.state == state
            ]
            if matches:
                value = str(raw.get("value", "")).strip()
                selected[region_id] = next(
                    (candidate for candidate in matches if candidate.value == value),
                    matches[0],
                )
        if set(selected) != set(proposals):
            raise ValueError("Qwen joint sample did not assign every ROI")
        return selected

    def critique(
        self,
        lr: Image.Image,
        swin: Image.Image,
        regions: list[Region],
        assignments: list[dict[str, Candidate]],
    ) -> list[dict[str, dict[str, Any]]]:
        if len(assignments) > 4:
            checked = []
            for start in range(0, len(assignments), 4):
                checked.extend(
                    self.critique(lr, swin, regions, assignments[start : start + 4])
                )
            return checked
        worlds = [
            {
                "world_index": index,
                "assignments": [
                    {
                        "region_id": region_id,
                        "state": candidate.state,
                        "value": candidate.value,
                    }
                    for region_id, candidate in sorted(world.items())
                ],
            }
            for index, world in enumerate(assignments)
        ]
        instruction = (
            "Independently verify each proposed regional state against authoritative Image A; "
            "use Image B only as a non-authoritative hint. Label each assignment exactly one of "
            "pixel_supported, compatible_prior_only, contradicted, abstain. A prior-only state is "
            "allowed; contradicted means inconsistent with visible pixels, geometry, or another "
            'assignment. Return JSON only as {"worlds":[{"world_index":0,"regions":['
            '{"region_id":...,"label":...,"score":0..1,"evidence":...}]}]}. '
            "Do not return internal reasoning. Worlds: "
            + json.dumps(worlds, ensure_ascii=False, separators=(",", ":"))
        )
        result = self._complete(
            _image_content(lr, swin, regions, self.roi_crop_size_px),
            instruction,
            seed=0,
            temperature=0.1,
            max_tokens=4096,
        )
        checked: list[dict[str, dict[str, Any]]] = [dict() for _ in assignments]
        for item in result.get("worlds", []):
            index = int(item.get("world_index", -1))
            if not 0 <= index < len(checked):
                continue
            for raw in item.get("regions", []):
                region_id = str(raw.get("region_id", ""))
                label = normalize_symbol(str(raw.get("label", "abstain")))
                if label not in CRITIC_LABELS:
                    label = "abstain"
                checked[index][region_id] = {
                    "label": label,
                    "score": min(1.0, max(0.0, float(raw.get("score", 0.0)))),
                    "evidence": str(raw.get("evidence", "")).strip(),
                }
        for index, world in enumerate(assignments):
            for region_id in world:
                checked[index].setdefault(
                    region_id,
                    {
                        "label": "abstain",
                        "score": 0.0,
                        "evidence": "critic omitted ROI",
                    },
                )
        return checked


def regional_entropy(
    samples: list[JointSample], region_id: str, candidate_count: int
) -> float:
    if not samples or candidate_count <= 1:
        return 0.0
    counts = Counter(sample.assignments[region_id].key() for sample in samples)
    total = sum(counts.values())
    entropy = -sum(
        (count / total) * math.log(count / total) for count in counts.values()
    )
    return float(entropy / math.log(max(candidate_count, len(counts), 2)))


def sample_joint_worlds(
    reasoner: RegionalReasoner,
    lr: Image.Image,
    swin: Image.Image,
    regions: list[Region],
    proposals: dict[str, list[Candidate]],
    sample_count: int,
    seed: int,
    *,
    coverage_first: bool = False,
) -> list[JointSample]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    rng = random.Random(seed)
    samples: list[JointSample] = []
    if coverage_first:
        control_preference = ("unresolved", "no_additional_detail", "absent")
        baseline = {}
        for region in regions:
            candidates = proposals[region.region_id]
            baseline[region.region_id] = next(
                (
                    candidate
                    for state in control_preference
                    for candidate in candidates
                    if candidate.state == state
                ),
                candidates[0],
            )
        samples.append(JointSample(baseline, "coverage"))
        covered = {
            region.region_id: {baseline[region.region_id].key()} for region in regions
        }
        maximum = max(len(proposals[region.region_id]) for region in regions)
        for index in range(maximum):
            world = {
                region.region_id: proposals[region.region_id][
                    index % len(proposals[region.region_id])
                ]
                for region in regions
            }
            if any(
                candidate.key() not in covered[region_id]
                for region_id, candidate in world.items()
            ):
                samples.append(JointSample(world, "coverage"))
                for region_id, candidate in world.items():
                    covered[region_id].add(candidate.key())
        if len(samples) > sample_count:
            raise ValueError(
                "sample budget is smaller than the candidate coverage schedule"
            )
    samples.extend(
        JointSample(
            reasoner.sample_world(lr, swin, regions, proposals, rng.randrange(2**31)),
            "free",
        )
        for _ in range(sample_count - len(samples))
    )
    return samples
