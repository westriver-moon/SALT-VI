"""Reusable fast-path components for the QRI acceleration pilot.

The module intentionally lives outside the production QRI package until the
pilot establishes a useful quality/throughput point.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
from PIL import Image

from qwen_imagination.regional.qwen import (
    LlamaServerQwenReasoner,
)
from qwen_imagination.regional.qwen_v2 import (
    V2_NON_EDIT_STATES,
    _bounded_candidates,
    _v2_image_content,
    v2_category_states,
)
from qwen_imagination.regional.schema import Candidate, Region
from qwen_imagination.taxonomy import normalize_symbol


class CachedSamBackend:
    """Reuse one SAM image embedding for every ROI belonging to one image."""

    def __init__(self, backend: Any):
        self.backend = backend
        self.predictor = backend.predictor
        self._image_ref: Image.Image | None = None
        self.set_image_calls = 0
        self.predict_calls = 0

    def refine(
        self,
        image: Image.Image,
        bbox_xyxy: tuple[int, int, int, int],
        seed_mask: np.ndarray,
    ) -> np.ndarray:
        if self._image_ref is not image:
            self.predictor.set_image(np.asarray(image.convert("RGB")))
            self._image_ref = image
            self.set_image_calls += 1
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(bbox_xyxy, dtype=np.float32),
            multimask_output=True,
        )
        self.predict_calls += 1
        candidates = [np.asarray(mask, dtype=bool) for mask in masks]
        if seed_mask.any():
            overlaps = [
                float((mask & seed_mask).sum())
                / max(1.0, float((mask | seed_mask).sum()))
                for mask in candidates
            ]
            best = int(
                np.argmax(
                    np.asarray(overlaps, dtype=np.float32)
                    + 0.05 * np.asarray(scores, dtype=np.float32)
                )
            )
            if overlaps[best] <= 0:
                return np.asarray(seed_mask, dtype=bool)
        else:
            best = int(np.argmax(scores))
        selected = candidates[best]
        bounded = np.zeros_like(selected, dtype=bool)
        left, top, right, bottom = bbox_xyxy
        bounded[top:bottom, left:right] = True
        selected &= bounded
        return selected if selected.any() else np.asarray(seed_mask, dtype=bool)


def _candidate_payload(candidate: Candidate) -> dict[str, str]:
    return {
        "state": candidate.state,
        "value": candidate.value,
        "evidence": candidate.evidence,
        "evidence_source": candidate.evidence_source,
    }


class OneShotQwenPlanner:
    """Produce regional candidates and a compact world set in one VLM call."""

    def __init__(
        self,
        endpoint: str,
        model_id: str,
        timeout_seconds: float = 300.0,
        board_size_px: int = 512,
    ) -> None:
        self.client = LlamaServerQwenReasoner(
            endpoint=endpoint,
            model_id=model_id,
            timeout_seconds=timeout_seconds,
            enable_thinking=False,
            reasoning_effort="none",
        )
        self.board_size_px = int(board_size_px)

    def plan(
        self,
        lr: Image.Image,
        swin: Image.Image,
        regions: list[Region],
        *,
        max_worlds: int = 3,
        seed: int = 0,
    ) -> dict[str, Any]:
        contract = [
            {
                "region_id": region.region_id,
                "category": region.category,
                "allowed_states": sorted(v2_category_states(region.category)),
            }
            for region in regions
        ]
        instruction = (
            "For low-resolution person re-identification, produce a compact set of "
            "mutually exclusive regional hypotheses and up to three jointly compatible "
            "visual worlds in one response. Image A is authoritative LR evidence; Image B "
            "is only a super-resolution proposal. For every region include exactly three "
            "short candidates: one concrete drawable positive candidate, absent when allowed "
            "(otherwise a second positive), and unresolved. Keep value and evidence under "
            "eight words each. "
            "Do not change identity, pose, body shape, or observed clothing. Return JSON only "
            "with keys regions and worlds. regions items contain region_id and candidates; "
            "candidate fields are state, value, evidence, evidence_source. worlds items contain "
            "assignments with region_id, state and value. Evidence source must be one of "
            "strong_pixel_supported, weak_pixel_supported, prior_plausible, unresolved. "
            "Contract: "
            + json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
        )
        raw = self.client._complete(
            _v2_image_content(lr, swin, regions, self.board_size_px),
            instruction,
            seed=int(seed),
            temperature=0.45,
            max_tokens=1024,
        )
        region_by_id = {region.region_id: region for region in regions}
        proposals: dict[str, list[Candidate]] = {
            region.region_id: [] for region in regions
        }
        for item in raw.get("regions", []):
            region_id = str(item.get("region_id", ""))
            if region_id not in region_by_id:
                continue
            allowed = v2_category_states(region_by_id[region_id].category)
            for value in item.get("candidates", []):
                state = normalize_symbol(str(value.get("state", "")))
                if state not in allowed:
                    continue
                source = normalize_symbol(
                    str(value.get("evidence_source", "prior_plausible"))
                )
                if source not in {
                    "strong_pixel_supported",
                    "weak_pixel_supported",
                    "prior_plausible",
                    "unresolved",
                }:
                    source = "prior_plausible"
                proposals[region_id].append(
                    Candidate(
                        state=state,
                        value=str(value.get("value", state)).strip() or state,
                        evidence=str(value.get("evidence", "")).strip(),
                        evidence_source=source,
                    )
                )
        proposals = {
            region_id: _bounded_candidates(region_by_id[region_id], candidates)
            for region_id, candidates in proposals.items()
        }

        def match(region_id: str, state: str, value: str) -> Candidate | None:
            state = normalize_symbol(state)
            matches = [
                candidate
                for candidate in proposals.get(region_id, [])
                if candidate.state == state
            ]
            if not matches:
                return None
            return next(
                (candidate for candidate in matches if candidate.value == value),
                matches[0],
            )

        worlds: list[dict[str, Candidate]] = []
        for item in raw.get("worlds", []):
            selected: dict[str, Candidate] = {}
            for assignment in item.get("assignments", []):
                region_id = str(assignment.get("region_id", ""))
                candidate = match(
                    region_id,
                    str(assignment.get("state", "")),
                    str(assignment.get("value", "")).strip(),
                )
                if candidate is not None:
                    selected[region_id] = candidate
            if set(selected) == set(proposals):
                key = tuple(
                    (region_id, candidate.state, candidate.value)
                    for region_id, candidate in sorted(selected.items())
                )
                if all(
                    key
                    != tuple(
                        (rid, candidate.state, candidate.value)
                        for rid, candidate in sorted(existing.items())
                    )
                    for existing in worlds
                ):
                    worlds.append(selected)
            if len(worlds) >= max_worlds:
                break

        control = {
            region_id: next(
                (
                    candidate
                    for preferred in ("unresolved", "no_additional_detail", "absent")
                    for candidate in candidates
                    if candidate.state == preferred
                ),
                candidates[-1],
            )
            for region_id, candidates in proposals.items()
        }
        if not worlds:
            worlds.append(dict(control))
        if not any(
            all(candidate.state in V2_NON_EDIT_STATES for candidate in world.values())
            for world in worlds
        ):
            worlds.insert(0, dict(control))
        for region in regions:
            if len(worlds) >= max_worlds:
                break
            positive = next(
                (
                    candidate
                    for candidate in proposals[region.region_id]
                    if candidate.state not in V2_NON_EDIT_STATES
                ),
                None,
            )
            if positive is None:
                continue
            if not any(
                world[region.region_id].state not in V2_NON_EDIT_STATES
                for world in worlds
            ):
                world = dict(control)
                world[region.region_id] = positive
                worlds.append(world)

        return {
            "raw": raw,
            "regions": {
                region_id: [_candidate_payload(item) for item in candidates]
                for region_id, candidates in proposals.items()
            },
            "worlds": [
                {
                    region_id: _candidate_payload(candidate)
                    for region_id, candidate in sorted(world.items())
                }
                for world in worlds[:max_worlds]
            ],
        }


def mask_metrics(
    reference: Image.Image,
    candidate: Image.Image,
    mask: Image.Image,
) -> dict[str, float]:
    left = np.asarray(reference.convert("RGB"), dtype=np.float32)
    right = np.asarray(candidate.convert("RGB"), dtype=np.float32)
    selected = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    delta = np.mean(np.abs(right - left), axis=2)
    inside_weight = max(float(selected.sum()), 1.0)
    outside = 1.0 - selected
    outside_weight = max(float(outside.sum()), 1.0)
    return {
        "inside_mean_abs_change": float((delta * selected).sum() / inside_weight),
        "inside_fraction_changed_gt10": float(
            (((delta > 10).astype(np.float32) * selected).sum()) / inside_weight
        ),
        "outside_mean_abs_change": float((delta * outside).sum() / outside_weight),
    }
