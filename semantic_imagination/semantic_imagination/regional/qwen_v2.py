from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from PIL import Image

from semantic_imagination.taxonomy import CATEGORY_STATES, normalize_symbol

from .qwen import LlamaServerQwenReasoner, _data_url
from .schema import Candidate, Region
from .visual_context import roi_comparison_board


V2_CRITIC_LABELS = {
    "strong_pixel_supported",
    "weak_pixel_supported",
    "prior_plausible",
    "contradicted",
    "unresolved",
}
V2_NON_EDIT_STATES = {"absent", "no_additional_detail", "unresolved"}
GENERIC_POSITIVE_STATE: Mapping[str, str] = {
    "eyewear": "eyewear_present",
    "wrist_accessory": "wrist_accessory_present",
    "headwear": "headwear_present",
    "body_marking": "body_marking_present",
    "clothing_detail": "clothing_detail_present",
    "carried_object": "carried_object_present",
    "pocket_item": "pocket_item_present",
    "footwear_detail": "footwear_detail_present",
}
GENERIC_POSITIVE_VALUE: Mapping[str, str] = {
    "eyewear": "possible eyewear; subtype unresolved",
    "wrist_accessory": "possible wrist accessory; subtype unresolved",
    "headwear": "possible headwear; subtype unresolved",
    "body_marking": "possible body marking; subtype unresolved",
    "clothing_detail": "possible clothing detail; subtype unresolved",
    "carried_object": "possible carried object; subtype unresolved",
    "pocket_item": "possible pocket item; subtype unresolved",
    "footwear_detail": "possible footwear detail; subtype unresolved",
}


def v2_category_states(category: str) -> frozenset[str]:
    return frozenset(
        set(CATEGORY_STATES[category])
        | {"unresolved", GENERIC_POSITIVE_STATE[category]}
    )


def _v2_image_content(
    lr: Image.Image,
    swin: Image.Image,
    regions: list[Region],
    board_size_px: int,
) -> list[dict]:
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Image A is the authoritative LR measurement. Image B is a SwinIR proposal; "
                "its added high-frequency details are hypotheses, not facts. Each following ROI "
                "board contains TL=A tight, TR=A context, BL=B tight, BR=B context. Board "
                "resizing preserves aspect ratio."
            ),
        },
        {"type": "image_url", "image_url": {"url": _data_url(lr)}},
        {"type": "image_url", "image_url": {"url": _data_url(swin)}},
    ]
    for region in regions:
        content.extend(
            (
                {
                    "type": "text",
                    "text": f"ROI comparison board: {region.region_id} / {region.category}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url(
                            roi_comparison_board(
                                lr,
                                swin,
                                region,
                                size_px=board_size_px,
                            )
                        )
                    },
                },
            )
        )
    return content


def _state_exists(candidates: list[Candidate], state: str) -> bool:
    return any(candidate.state == state for candidate in candidates)


def _bounded_candidates(region: Region, candidates: list[Candidate]) -> list[Candidate]:
    unique: dict[tuple[str, str], Candidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.key(), candidate)
    merged = list(unique.values())
    positive = [
        candidate for candidate in merged if candidate.state not in V2_NON_EDIT_STATES
    ]
    if not positive:
        positive = [
            Candidate(
                GENERIC_POSITIVE_STATE[region.category],
                GENERIC_POSITIVE_VALUE[region.category],
                "positive candidate inserted by the imagination coverage contract after Qwen "
                "omitted every non-contradicted positive interpretation",
                "prior_plausible",
            )
        ]
        merged.extend(positive)
    if "absent" in CATEGORY_STATES[region.category] and not _state_exists(
        merged, "absent"
    ):
        merged.append(
            Candidate(
                "absent",
                "absent",
                "symmetric absence alternative retained for imaginative comparison",
                "prior_plausible",
            )
        )
    if not _state_exists(merged, "unresolved"):
        merged.append(
            Candidate(
                "unresolved",
                "unresolved",
                "control world: the authoritative LR does not select a unique interpretation",
                "unresolved",
            )
        )

    positives = [
        candidate for candidate in merged if candidate.state not in V2_NON_EDIT_STATES
    ]
    absence = next(
        (candidate for candidate in merged if candidate.state == "absent"), None
    )
    unresolved = next(
        candidate for candidate in merged if candidate.state == "unresolved"
    )
    selected = positives[:3]
    if absence is not None:
        selected.append(absence)
    selected.append(unresolved)
    for candidate in merged:
        if len(selected) >= 5:
            break
        if candidate not in selected:
            selected.insert(max(0, len(selected) - 1), candidate)
    return selected[:5]


class ImaginativeQwenReasoner(LlamaServerQwenReasoner):
    """QRI-v2 recall-first proposer with a contradiction-only compatibility critic."""

    def __init__(
        self,
        *args,
        proposal_rounds: int = 3,
        roi_board_size_px: int = 512,
        **kwargs,
    ):
        super().__init__(*args, roi_crop_size_px=384, **kwargs)
        self.proposal_rounds = int(proposal_rounds)
        self.roi_board_size_px = int(roi_board_size_px)
        if self.proposal_rounds < 2:
            raise ValueError(
                "imagination-first proposal requires at least two proposal rounds"
            )

    def _content(
        self, lr: Image.Image, swin: Image.Image, regions: list[Region]
    ) -> list[dict]:
        return _v2_image_content(lr, swin, regions, self.roi_board_size_px)

    def propose(
        self, lr: Image.Image, swin: Image.Image, regions: list[Region]
    ) -> dict[str, list[Candidate]]:
        board = [
            {
                "region_id": region.region_id,
                "category": region.category,
                "allowed_states": sorted(v2_category_states(region.category)),
                "required_coverage": (
                    "one or more positive interpretations, an absence interpretation when "
                    "allowed, and unresolved"
                ),
            }
            for region in regions
        ]
        instruction = (
            "Act as an imagination-first visual hypothesis proposer for low-resolution person "
            "re-identification. Maximize recall of mutually exclusive visual worlds; this stage "
            "must not decide which world is true. Use weak edges, silhouettes, reflections, "
            "temple-arm geometry and contextual structure. Never equate unobservable with absent. "
            "For every ROI return 3-5 candidates, including at least one positive interpretation, "
            "an absent interpretation when allowed, and unresolved. A weak or prior-only candidate "
            "is valid whenever Image A does not contradict it. Image B may suggest candidates but "
            "cannot prove them. Do not change identity, pose, body shape or observed clothing. "
            'Return JSON only as {"regions":[{"region_id":...,"candidates":[{"state":...,'
            '"value":...,"evidence":...,"evidence_source":"strong_pixel_supported|'
            'weak_pixel_supported|prior_plausible|unresolved"}]}]}. Do not return internal '
            "reasoning. Closed candidate contract: "
            + json.dumps(board, ensure_ascii=False, separators=(",", ":"))
        )
        region_by_id = {region.region_id: region for region in regions}
        merged: dict[str, list[Candidate]] = {
            region.region_id: [] for region in regions
        }
        allowed = {
            region.region_id: v2_category_states(region.category) for region in regions
        }
        for round_index in range(self.proposal_rounds):
            result = self._complete(
                self._content(lr, swin, regions),
                instruction,
                seed=round_index,
                temperature=0.65,
                max_tokens=4096,
            )
            for item in result.get("regions", []):
                region_id = str(item.get("region_id", ""))
                if region_id not in region_by_id:
                    continue
                for raw in item.get("candidates", []):
                    state = normalize_symbol(str(raw.get("state", "")))
                    if state not in allowed[region_id]:
                        continue
                    source = normalize_symbol(
                        str(raw.get("evidence_source", "prior_plausible"))
                    )
                    if source not in V2_CRITIC_LABELS - {"contradicted"}:
                        source = "prior_plausible"
                    merged[region_id].append(
                        Candidate(
                            state=state,
                            value=str(raw.get("value", state)).strip() or state,
                            evidence=str(raw.get("evidence", "")).strip(),
                            evidence_source=source,
                        )
                    )
        return {
            region_id: _bounded_candidates(region_by_id[region_id], candidates)
            for region_id, candidates in merged.items()
        }

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
            "Select one jointly compatible candidate for every ROI to instantiate one plausible "
            "semantic world. Explore positive imaginative interpretations as well as absence and "
            "unresolved controls. Do not prefer abstention. Low visibility is a reason to preserve "
            "multiple worlds, not a reason to collapse them to absent. Reject only combinations "
            "that conflict with Image A, identity, pose or each other. Return JSON only as "
            '{"assignments":[{"region_id":...,"state":...,"value":...}]}. Candidate board: '
            + json.dumps(board, ensure_ascii=False, separators=(",", ":"))
        )
        result = self._complete(
            self._content(lr, swin, regions),
            instruction,
            seed=seed,
            temperature=0.85,
            max_tokens=2048,
        )
        selected: dict[str, Candidate] = {}
        for raw in result.get("assignments", []):
            region_id = str(raw.get("region_id", ""))
            state = normalize_symbol(str(raw.get("state", "")))
            matches = [
                candidate
                for candidate in proposals.get(region_id, [])
                if candidate.state == state
            ]
            if matches:
                value = str(raw.get("value", "")).strip()
                selected[region_id] = next(
                    (candidate for candidate in matches if candidate.value == value),
                    matches[0],
                )
        if set(selected) != set(proposals):
            raise ValueError("QRI-v2 joint sample did not assign every ROI")
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
            "Act as a compatibility critic, not a truth verifier. Compare every assignment with "
            "authoritative Image A and use Image B only as a non-authoritative hint. Label each "
            "assignment strong_pixel_supported, weak_pixel_supported, prior_plausible, "
            "contradicted, or unresolved. Weak and prior-only imaginative worlds must survive when "
            "A does not contradict them. Use contradicted only for a visible pixel, geometry, "
            "identity, pose, or cross-assignment conflict. Score compatibility from 0 to 1. Return "
            'JSON only as {"worlds":[{"world_index":0,"regions":[{"region_id":...,'
            '"label":...,"score":0..1,"evidence":...}]}]}. Do not return internal '
            "reasoning. Worlds: "
            + json.dumps(worlds, ensure_ascii=False, separators=(",", ":"))
        )
        result = self._complete(
            self._content(lr, swin, regions),
            instruction,
            seed=0,
            temperature=0.15,
            max_tokens=4096,
        )
        checked: list[dict[str, dict[str, Any]]] = [dict() for _ in assignments]
        for item in result.get("worlds", []):
            index = int(item.get("world_index", -1))
            if not 0 <= index < len(checked):
                continue
            for raw in item.get("regions", []):
                region_id = str(raw.get("region_id", ""))
                label = normalize_symbol(str(raw.get("label", "unresolved")))
                if label not in V2_CRITIC_LABELS:
                    label = "unresolved"
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
                        "label": "unresolved",
                        "score": 0.0,
                        "evidence": "critic omitted ROI",
                    },
                )
        return checked
