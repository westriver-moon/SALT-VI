from __future__ import annotations

import json
import math
import random
import re
import time
import urllib.request
from collections import Counter
from typing import Any

from PIL import Image

from ..regional.qwen import _data_url, _json_object
from ..regional.schema import Region
from ..regional.visual_context import roi_comparison_board, swin_roi_board


EVIDENCE_BASES = {"visual_evidence", "world_knowledge", "mixed", "unresolved"}
EVIDENCE_STRENGTHS = {"strong", "weak"}
REID_ATTRIBUTE_ALIASES = {
    "head": "hd",
    "upper": "up",
    "lower": "lo",
    "footwear": "ft",
    "carried": "ca",
    "distinctive": "ds",
}
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
UNCERTAINTY_PATTERN = re.compile(
    r"\b(possible|possibly|may|might|uncertain|appears)\b", re.IGNORECASE
)


def _field(value: dict[str, Any], long_name: str, short_name: str, default=None):
    if long_name in value:
        return value[long_name]
    if short_name in value:
        return value[short_name]
    return default


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:3]


def _observations(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    rows = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        claim = str(_field(raw, "claim", "c", "")).strip()
        if not claim:
            continue
        strength = str(_field(raw, "evidence_strength", "s", "weak")).strip().lower()
        rows.append(
            {
                "claim": claim,
                "evidence_strength": (
                    strength if strength in EVIDENCE_STRENGTHS else "weak"
                ),
                "evidence": str(_field(raw, "evidence", "e", "")).strip(),
            }
        )
    return rows[:4]


def normalize_hypotheses(
    value: object,
    *,
    probability_mode: str = "required",
    exact_count: int | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("each selected ROI must return at least two hypotheses")
    if len(value) > 3:
        raise ValueError("each selected ROI must return at most three hypotheses")
    if exact_count is not None and len(value) != int(exact_count):
        raise ValueError(
            f"each selected ROI must return exactly {int(exact_count)} hypotheses"
        )
    if probability_mode not in {"required", "forbidden"}:
        raise ValueError(f"unsupported hypothesis probability mode {probability_mode}")
    rows = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"hypothesis {index} must be an object")
        description = str(_field(raw, "description", "d", "")).strip()
        if not description:
            raise ValueError(f"hypothesis {index} has no description")
        basis = str(_field(raw, "basis", "b", "unresolved")).strip().lower()
        if basis not in EVIDENCE_BASES:
            raise ValueError(f"hypothesis {index} has unsupported basis {basis}")
        if probability_mode == "forbidden" and (
            "probability" in raw or "p" in raw
        ):
            raise ValueError(
                "Swin-only separated hypotheses must not contain VLM-reported probabilities"
            )
        row = {
            "description": description,
            "basis": basis,
            "observable_support": str(
                _field(raw, "observable_support", "e", "")
            ).strip(),
            "uncertainty": str(_field(raw, "uncertainty", "u", "")).strip(),
        }
        if probability_mode == "required":
            row["probability"] = max(
                0.0, float(_field(raw, "probability", "p", 0.0))
            )
        rows.append(row)
    if probability_mode == "required":
        total = sum(float(row["probability"]) for row in rows)
        if total <= 0.0:
            raise ValueError("hypothesis probabilities must contain positive mass")
        for row in rows:
            row["probability"] = float(row["probability"]) / total
    return rows


def _reid_attributes(value: object, *, required: bool) -> dict[str, str]:
    if not isinstance(value, dict):
        if required:
            raise ValueError("global re-identification attributes must be an object")
        return {}
    result = {}
    for name, alias in REID_ATTRIBUTE_ALIASES.items():
        text = str(_field(value, name, alias, "")).strip()
        if text:
            result[name] = text
    missing = [name for name in REID_ATTRIBUTE_ALIASES if name not in result]
    if required and missing:
        raise ValueError(f"global re-identification attributes omit: {missing}")
    return result


def caption_word_count(caption: str) -> int:
    return len(WORD_PATTERN.findall(str(caption)))


def _contains_token_phrase(caption: str, phrase: str) -> bool:
    caption_tokens = [token.lower() for token in WORD_PATTERN.findall(caption)]
    phrase_tokens = [token.lower() for token in WORD_PATTERN.findall(phrase)]
    if not phrase_tokens:
        return False
    width = len(phrase_tokens)
    return any(
        caption_tokens[index : index + width] == phrase_tokens
        for index in range(len(caption_tokens) - width + 1)
    )


def _regional_addenda(
    value: object, expected_ids: list[str], *, required: bool
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        if required:
            raise ValueError("global regional addenda must be an array")
        return []
    by_id = {}
    for raw in value:
        if not isinstance(raw, dict):
            continue
        region_id = str(_field(raw, "region_id", "id", "")).strip()
        text = str(_field(raw, "text", "t", "")).strip()
        if not region_id or not text:
            continue
        if region_id in by_id:
            raise ValueError(f"duplicate global regional addendum {region_id}")
        if required:
            if not UNCERTAINTY_PATTERN.match(text):
                raise ValueError(
                    f"regional addendum {region_id} must begin with possible/may/might language"
                )
            phrase_words = caption_word_count(text)
            if not 2 <= phrase_words <= 5:
                raise ValueError(
                    f"regional addendum {region_id} must contain 2-5 words, got {phrase_words}"
                )
        by_id[region_id] = text
    missing = [region_id for region_id in expected_ids if region_id not in by_id]
    if required and missing:
        raise ValueError(f"global regional addenda omit selected ROIs: {missing}")
    return [
        {"region_id": region_id, "text": by_id[region_id]}
        for region_id in expected_ids
        if region_id in by_id
    ]


def normalize_annotation(
    payload: dict[str, Any],
    selected_regions: list[Region],
    *,
    require_reid_caption: bool = False,
    require_swin_separated: bool = False,
) -> dict[str, Any]:
    global_raw = _field(payload, "global", "g")
    if not isinstance(global_raw, dict):
        raise ValueError("Qwen annotation must contain a global object")
    caption = str(_field(global_raw, "caption", "c", "")).strip()
    if not caption:
        raise ValueError("global caption must be non-empty")
    attributes = _reid_attributes(
        _field(global_raw, "attributes", "a"),
        required=require_reid_caption or require_swin_separated,
    )
    expected_region_ids = [region.region_id for region in selected_regions]
    regional_addenda = _regional_addenda(
        _field(global_raw, "regional_addenda", "x"),
        expected_region_ids,
        required=require_reid_caption,
    )
    word_count = caption_word_count(caption)
    if (require_reid_caption or require_swin_separated) and not 22 <= word_count <= 35:
        raise ValueError(
            f"global re-identification caption must contain 22-35 words, got {word_count}"
        )
    if require_swin_separated:
        if _field(global_raw, "regional_addenda", "x") is not None:
            raise ValueError(
                "Swin-only separated global output must not contain regional addenda"
            )
        if UNCERTAINTY_PATTERN.search(caption):
            raise ValueError(
                "Swin-only separated global caption must not contain modal qualifiers"
            )

    raw_regions = _field(payload, "regions", "r")
    if not isinstance(raw_regions, list):
        raise ValueError("Qwen annotation must contain a regions array")
    by_id = {}
    for raw in raw_regions:
        if not isinstance(raw, dict):
            continue
        region_id = str(_field(raw, "region_id", "id", "")).strip()
        if region_id:
            if region_id in by_id:
                raise ValueError(f"duplicate regional annotation {region_id}")
            by_id[region_id] = raw
    missing = [region_id for region_id in expected_region_ids if region_id not in by_id]
    if missing:
        raise ValueError(f"Qwen omitted selected ROI annotations: {missing}")

    normalized_regions = []
    for region in selected_regions:
        raw = by_id[region.region_id]
        region_summary = str(_field(raw, "region_summary", "s", "")).strip()
        if require_swin_separated:
            if not region_summary:
                raise ValueError(
                    f"Swin-only separated ROI {region.region_id} has no independent caption"
                )
            summary_words = caption_word_count(region_summary)
            if not 3 <= summary_words <= 16:
                raise ValueError(
                    f"Swin-only separated ROI {region.region_id} caption must contain "
                    f"3-16 words, got {summary_words}"
                )
            if UNCERTAINTY_PATTERN.search(region_summary):
                raise ValueError(
                    f"Swin-only separated ROI {region.region_id} caption must not "
                    "contain modal qualifiers"
                )
        knowledge = []
        for item in (_field(raw, "world_knowledge", "k", []) or [])[:2]:
            if not isinstance(item, dict):
                continue
            inference = str(_field(item, "inference", "i", "")).strip()
            if inference:
                knowledge.append(
                    {
                        "inference": inference,
                        "knowledge_used": str(_field(item, "knowledge_used", "k", "")).strip(),
                        "evidence_relation": str(
                            _field(item, "evidence_relation", "r", "")
                        ).strip(),
                    }
                )
        hypotheses = normalize_hypotheses(
            _field(raw, "hypotheses", "h"),
            probability_mode=(
                "forbidden" if require_swin_separated else "required"
            ),
            exact_count=2 if require_swin_separated else None,
        )
        if require_swin_separated:
            for hypothesis in hypotheses:
                description = hypothesis["description"]
                description_words = caption_word_count(description)
                if not 1 <= description_words <= 10:
                    raise ValueError(
                        f"Swin-only separated ROI {region.region_id} hypothesis must "
                        f"contain 1-10 words, got {description_words}"
                    )
                if UNCERTAINTY_PATTERN.search(description):
                    raise ValueError(
                        f"Swin-only separated ROI {region.region_id} hypothesis must "
                        "not contain modal qualifiers"
                    )
        normalized_regions.append(
            {
                "region_id": region.region_id,
                "category": region.category,
                "region_summary": region_summary,
                "observations": _observations(_field(raw, "observations", "o")),
                "world_knowledge": knowledge,
                "hypotheses": hypotheses,
                "unresolved": _strings(_field(raw, "unresolved", "u")),
            }
        )
    if require_reid_caption:
        absent = [
            row["region_id"]
            for row in regional_addenda
            if not _contains_token_phrase(caption, row["text"])
        ]
        if absent:
            raise ValueError(
                "global caption does not contain regional addenda verbatim for ROIs: "
                f"{absent}"
            )
    global_result = {
        "caption": caption,
        "caption_word_count": word_count,
        "caption_profile": (
            "swin_only_separated_22_35_v2"
            if require_swin_separated
            else "reid_balanced_22_35_v3"
            if require_reid_caption
            else "legacy"
        ),
        "observations": _observations(_field(global_raw, "observations", "o")),
        "unresolved": _strings(_field(global_raw, "unresolved", "u")),
    }
    if attributes:
        global_result["attributes"] = attributes
    if regional_addenda:
        global_result["regional_addenda"] = regional_addenda
    return {"global": global_result, "regions": normalized_regions}


def sample_joint_text_worlds(
    regions: list[dict[str, Any]],
    *,
    sample_count: int,
    max_worlds: int,
    seed: int,
) -> dict[str, Any]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not 1 <= max_worlds <= sample_count:
        raise ValueError("max_worlds must be within [1, sample_count]")
    rng = random.Random(int(seed))
    draws = []
    for _ in range(int(sample_count)):
        assignment = []
        for region in regions:
            hypotheses = region["hypotheses"]
            index = rng.choices(
                range(len(hypotheses)),
                weights=[float(row["probability"]) for row in hypotheses],
                k=1,
            )[0]
            assignment.append(index)
        draws.append(tuple(assignment))
    counts = Counter(draws)
    accepted = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
        : int(max_worlds)
    ]
    selected_count = sum(count for _, count in accepted)
    worlds = []
    for world_index, (indices, count) in enumerate(accepted):
        assignments = []
        for region, hypothesis_index in zip(regions, indices):
            hypothesis = region["hypotheses"][hypothesis_index]
            assignments.append(
                {
                    "region_id": region["region_id"],
                    "category": region["category"],
                    "hypothesis_index": int(hypothesis_index),
                    "description": hypothesis["description"],
                    "basis": hypothesis["basis"],
                    "source_probability": float(hypothesis["probability"]),
                }
            )
        worlds.append(
            {
                "world_id": f"w{world_index:02d}",
                "assignments": assignments,
                "sample_count": int(count),
                "sample_frequency": count / float(sample_count),
                "selected_weight": count / float(selected_count),
            }
        )
    return {
        "seed": int(seed),
        "sample_count": int(sample_count),
        "unique_world_count": len(counts),
        "retained_world_count": len(worlds),
        "retained_sample_mass": selected_count / float(sample_count),
        "worlds": worlds,
    }


def normalized_entropy(hypotheses: list[dict[str, Any]]) -> float:
    if len(hypotheses) <= 1:
        return 0.0
    entropy = -sum(
        float(row["probability"])
        * math.log(max(float(row["probability"]), 1e-12))
        for row in hypotheses
    )
    return float(entropy / math.log(len(hypotheses)))


class TextAnnotationReasoner:
    """One-call global plus multi-ROI text annotation client."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_id: str,
        timeout_seconds: float = 360.0,
        enable_thinking: bool = False,
        reasoning_effort: str = "none",
        temperature: float = 0.35,
        max_tokens: int = 2048,
        response_profile: str = "detailed_v1",
        roi_board_size_px: int = 512,
    ):
        self.endpoint = str(endpoint)
        self.model_id = str(model_id)
        self.timeout_seconds = float(timeout_seconds)
        self.enable_thinking = bool(enable_thinking)
        self.reasoning_effort = str(reasoning_effort)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.response_profile = str(response_profile)
        if self.response_profile not in {
            "detailed_v1",
            "compact_v1",
            "compact_reid_v3",
            "swin_separated_v1",
        }:
            raise ValueError(
                "response_profile must be detailed_v1, compact_v1, compact_reid_v3 "
                "or swin_separated_v1"
            )
        self.roi_board_size_px = int(roi_board_size_px)

    def _instruction(self, regions: list[Region], modality: str) -> str:
        board = [
            {
                "region_id": region.region_id,
                "category": region.category,
                "bbox_xyxy": list(region.bbox_xyxy),
            }
            for region in regions
        ]
        color_rule = (
            "Describe defensible colors from Image A."
            if modality == "rgb"
            else "This is infrared input: do not infer visible-spectrum colors."
        )
        if self.response_profile == "swin_separated_v1":
            swin_color_rule = (
                "For RGB input, describe only colors visible in the SwinIR image."
                if modality == "rgb"
                else "This is infrared input: do not infer visible-spectrum colors."
            )
            return (
                "Annotate a SwinIR-restored low-resolution person image for person "
                "re-identification. The SwinIR full image and SwinIR ROI boards are the "
                "sole visual inputs. Produce two independent outputs and never merge them. "
                "First, write one natural English global caption from the full SwinIR image. "
                "Target 26-30 words and enforce a hard limit of 22-35 words; count words before "
                "returning JSON. Use this single-sentence shape without repeating slots: "
                "Person with [head] wears [upper], [lower], and [footwear]; [carried item]; "
                "[distinctive detail]. It must cover head/hair/headwear; upper garment color, "
                "type, sleeves and visible design; lower garment color, type and length; "
                "footwear; carried item; and distinctive detail. If a slot is not visible, "
                "state that it is not clearly visible. The global caption must not contain "
                "ROI hypotheses, modal qualifiers, or any final composed regional text. "
                "For body sides, use the person's anatomical left or right; when orientation "
                "is unclear, say one wrist or one hand instead of guessing a side. "
                "Second, for every listed ROI, independently return one 4-14 word visual "
                "caption and exactly two mutually exclusive 2-8 word candidate hypotheses. "
                "Describe the ROI itself; do not rewrite the whole person caption. Candidate "
                "descriptions must be direct neutral phrases without modal qualifiers. Both "
                "candidates must answer the same ROI question and cannot both be true. Never "
                "split an object and its attribute into separate candidates; for example use "
                "thin eyeglasses versus facial shadow without eyewear. "
                "Do not assign probabilities or "
                "confidence scores; probability estimation is a separate repeated-sampling "
                "and semantic-clustering stage. Do not rewrite or append the global caption "
                "using ROI content. Output JSON only and no reasoning or prose. "
                + swin_color_rule
                + " Return only the keys in this minimal contract: "
                '{"g":{"c":"22-35 word independent global caption","a":{'
                '"hd":"head phrase","up":"upper phrase","lo":"lower phrase",'
                '"ft":"footwear phrase","ca":"carried-item phrase",'
                '"ds":"distinctive phrase"}},"r":[{"id":"region_id",'
                '"s":"4-14 word independent ROI caption","h":[{"d":"candidate one"},'
                '{"d":"candidate two"}]}]}. Omit every other key. Keep each global '
                "attribute phrase at most 8 words. ROIs: "
                + json.dumps(board, separators=(",", ":"))
            )
        if self.response_profile == "compact_reid_v3":
            return (
                "Annotate this low-resolution person for re-identification. Image A is "
                "authoritative; Image B and B tiles are non-authoritative SwinIR proposals. "
                "Never edit pixels or treat SwinIR-only detail as fact. Produce one natural "
                "English global caption containing 22-35 words in total. Its stable main "
                "clause must cover all "
                "six slots: head/hair/headwear; upper garment color, type, sleeves and visible "
                "design; lower garment color, type and length; footwear; carried item; and "
                "distinctive detail. If a slot is not visible, explicitly say it is not "
                "clearly visible instead of omitting or guessing it. Do not promote ambiguous "
                "dark hair or head pixels into a cap or hat. After the main clause, append one "
                "concise hypothesis for EVERY listed ROI, even when its top probability is "
                "high. Each addendum must begin with possible, may, might, or appears, must be "
                "2-5 words, and must remain inside the 22-35 word total. Copy the exact same "
                "addenda into g.x, one per ROI; validation requires every g.x phrase to occur "
                "verbatim in g.c. Also return each slot as a grammar-ready phrase in g.a. Give exactly "
                "2 mutually exclusive hypotheses per ROI with probabilities summing to "
                "one. Do not turn a vague edge, band, shadow, or SwinIR-only texture into an "
                "accessory. Output no reasoning or prose. "
                + color_rule
                + " JSON only, using this compact contract: "
                '{"g":{"c":"22-35 word caption","a":{"hd":"head phrase",'
                '"up":"upper phrase","lo":"lower phrase","ft":"footwear phrase",'
                '"ca":"carried-item phrase","ds":"distinctive phrase"},"x":['
                '{"id":"region_id","t":"possible short phrase copied in caption"}],'
                '"o":[{"c":"claim","s":"strong|weak","e":"evidence"}],'
                '"u":["unresolved"]},"r":[{"id":"region_id","s":"summary",'
                '"o":[{"c":"claim","s":"strong|weak","e":"evidence"}],'
                '"k":[{"i":"inference","k":"knowledge","r":"relation"}],'
                '"h":[{"d":"description","p":0.0,'
                '"b":"visual_evidence|world_knowledge|mixed|unresolved",'
                '"e":"support","u":"uncertainty"}],"u":["unresolved"]}]}. '
                "Use <=2 global observations, <=1 observation and <=1 knowledge item per "
                "ROI, and <=1 unresolved item globally or per ROI. Keep attribute phrases <=10 words and every "
                "other short field <=12 words. ROIs: "
                + json.dumps(board, separators=(",", ":"))
            )
        if self.response_profile == "compact_v1":
            return (
                "Annotate this low-resolution person for re-identification. Image A is "
                "authoritative; Image B and B tiles are non-authoritative SwinIR proposals. "
                "Never edit pixels. Give one <=30-word global caption, concise visible "
                "observations, and exactly 2 or 3 mutually exclusive hypotheses per listed "
                "ROI with probabilities summing to one. Separate evidence from world "
                "knowledge, keep blur alternatives, and output no reasoning or prose. "
                "Do not turn a vague edge, band, shadow, or SwinIR-only texture into an "
                "accessory. When direct evidence is weak, prefer a no-object or unresolved "
                "alternative and lower the positive-object probability. "
                + color_rule
                + " JSON only, using this compact contract: "
                '{"g":{"c":"caption","o":[{"c":"claim","s":"strong|weak",'
                '"e":"evidence"}],"u":["unresolved"]},"r":[{"id":"region_id",'
                '"s":"summary","o":[{"c":"claim","s":"strong|weak",'
                '"e":"evidence"}],"k":[{"i":"inference","k":"knowledge",'
                '"r":"relation"}],"h":[{"d":"description","p":0.0,'
                '"b":"visual_evidence|world_knowledge|mixed|unresolved",'
                '"e":"support","u":"uncertainty"}],"u":["unresolved"]}]}. '
                "Use <=3 global observations, <=2 observations and <=1 knowledge item per "
                "ROI, and <=2 unresolved items. Keep every short field <=12 words. ROIs: "
                + json.dumps(board, separators=(",", ":"))
            )
        return (
            "Act as a text-only annotator for low-resolution person re-identification. "
            "Never generate or edit pixels. Image A is the authoritative LR measurement; "
            "Image B and every B tile are non-authoritative SwinIR proposals. First produce "
            "a compact English global person caption covering stable hair, clothing, footwear, "
            "carried items and distinctive visible structure. Limit it to 35 words. Then annotate every listed ROI. "
            "For each ROI separate observations, world knowledge and unresolved content, and "
            "return exactly 2 or 3 mutually exclusive hypotheses with probabilities summing to one. Use "
            "abstract world knowledge when visible abstraction supports a familiar object or "
            "design, while retaining plausible alternatives for blur. Never force a positive "
            "object or treat SwinIR detail as fact. Keep claims concise and do not return chain "
            "of thought. Use no more than 4 observations globally or per ROI, 2 world-knowledge "
            "items per ROI, and 3 unresolved items. Keep every claim, support, uncertainty, "
            "knowledge and hypothesis description under 18 words. "
            + color_rule
            + " Return JSON only with contract: "
            '{"global":{"caption":"...","observations":[{"claim":"...",'
            '"evidence_strength":"strong|weak","evidence":"..."}],'
            '"unresolved":["..."]},"regions":[{"region_id":"...",'
            '"region_summary":"...","observations":[{"claim":"...",'
            '"evidence_strength":"strong|weak","evidence":"..."}],'
            '"world_knowledge":[{"inference":"...","knowledge_used":"...",'
            '"evidence_relation":"..."}],"hypotheses":[{"description":"...",'
            '"probability":0.0,"basis":"visual_evidence|world_knowledge|mixed|unresolved",'
            '"observable_support":"...","uncertainty":"..."}],'
            '"unresolved":["..."]}]}. Selected ROIs: '
            + json.dumps(board, separators=(",", ":"))
        )

    def _content(
        self, lr: Image.Image, swin: Image.Image, regions: list[Region]
    ) -> list[dict[str, Any]]:
        if self.response_profile == "swin_separated_v1":
            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": "SwinIR full image, the sole full-image visual input.",
                },
                {"type": "image_url", "image_url": {"url": _data_url(swin)}},
            ]
            for region in regions:
                board = swin_roi_board(
                    swin, region, size_px=self.roi_board_size_px
                )
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": (
                                f"SwinIR-only ROI board {region.region_id} / "
                                f"{region.category}: tight and context."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(board)},
                        },
                    ]
                )
            return content
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Image A (authoritative LR), followed by Image B (SwinIR proposal).",
            },
            {"type": "image_url", "image_url": {"url": _data_url(lr)}},
            {"type": "image_url", "image_url": {"url": _data_url(swin)}},
        ]
        for region in regions:
            board = roi_comparison_board(
                lr, swin, region, size_px=self.roi_board_size_px
            )
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"ROI board {region.region_id} / {region.category}: "
                            "A-tight, A-context, B-tight, B-context."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": _data_url(board)}},
                ]
            )
        return content

    def annotate(
        self,
        lr: Image.Image,
        swin: Image.Image,
        regions: list[Region],
        *,
        modality: str,
        seed: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self._instruction(regions, modality)},
                {"role": "user", "content": self._content(lr, swin, regions)},
            ],
            "temperature": self.temperature,
            "top_p": 0.9,
            "seed": int(seed),
            "max_tokens": self.max_tokens,
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
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
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
        require_swin_separated = self.response_profile == "swin_separated_v1"
        caption_repair = None
        try:
            annotation = normalize_annotation(
                parsed,
                regions,
                require_reid_caption=self.response_profile == "compact_reid_v3",
                require_swin_separated=require_swin_separated,
            )
        except ValueError as error:
            if not require_swin_separated or "22-35 words" not in str(error):
                raise
            global_raw = _field(parsed, "global", "g")
            repair_input = {
                "caption": str(_field(global_raw, "caption", "c", "")).strip(),
                "attributes": _field(global_raw, "attributes", "a", {}),
            }
            repair_payload = {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Repair only an overlength person caption. Return JSON with exactly "
                            "one key c. Rewrite the supplied caption into 26-30 English words "
                            "while preserving all six supplied attribute facts. Use one sentence, "
                            "do not add facts, do not include ROI content, and do not use modal "
                            "qualifiers. JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(repair_input, separators=(",", ":")),
                    },
                ],
                "temperature": 0.0,
                "top_p": 0.9,
                "seed": int(seed),
                "max_tokens": min(220, self.max_tokens),
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_effort": "none",
            }
            repair_request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(repair_payload, separators=(",", ":")).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            repair_started = time.perf_counter()
            with urllib.request.urlopen(
                repair_request, timeout=self.timeout_seconds
            ) as response:
                repair_document = json.loads(response.read().decode("utf-8"))
            repair_message = repair_document["choices"][0]["message"]
            repair_parsed = None
            for field in ("content", "reasoning_content"):
                text = repair_message.get(field)
                if not text:
                    continue
                try:
                    repair_parsed = _json_object(str(text))
                    break
                except ValueError:
                    continue
            if repair_parsed is None:
                raise ValueError("Qwen caption repair returned no final JSON") from error
            repaired_caption = str(
                _field(repair_parsed, "caption", "c", "")
            ).strip()
            if not repaired_caption:
                raise ValueError("Qwen caption repair returned an empty caption") from error
            caption_key = "caption" if "caption" in global_raw else "c"
            global_raw[caption_key] = repaired_caption
            annotation = normalize_annotation(
                parsed,
                regions,
                require_reid_caption=False,
                require_swin_separated=True,
            )
            caption_repair = {
                "trigger": str(error),
                "elapsed_seconds": time.perf_counter() - repair_started,
                "usage": repair_document.get("usage", {}),
            }
        usage = dict(document.get("usage", {}))
        if caption_repair:
            for key, value in caption_repair["usage"].items():
                if isinstance(value, (int, float)) and isinstance(usage.get(key, 0), (int, float)):
                    usage[key] = usage.get(key, 0) + value
        elapsed = time.perf_counter() - started
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        telemetry = {
            "elapsed_seconds": elapsed,
            "response_field": response_field,
            "usage": usage,
            "response_profile": self.response_profile,
            "completion_tokens_per_second": (
                completion_tokens / elapsed if elapsed > 0.0 else None
            ),
            "reasoning_characters": len(str(message.get("reasoning_content", ""))),
            "content_characters": len(str(message.get("content", ""))),
            "caption_repair": caption_repair,
        }
        return annotation, telemetry
