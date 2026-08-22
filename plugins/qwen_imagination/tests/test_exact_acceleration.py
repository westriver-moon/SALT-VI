import base64
import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from qwen_imagination.text_annotation.reasoner import (
    TextAnnotationReasoner,
    normalize_annotation,
)


def test_compact_annotation_expands_to_canonical_schema():
    regions = [
        SimpleNamespace(region_id="eyes", category="eyewear"),
        SimpleNamespace(region_id="head", category="headwear"),
        SimpleNamespace(region_id="upper_torso", category="clothing_detail"),
    ]
    compact_regions = []
    for region in regions:
        compact_regions.append(
            {
                "id": region.region_id,
                "s": "unclear detail",
                "o": [{"c": "dark shape", "s": "weak", "e": "visible blur"}],
                "k": [{"i": "possible object", "k": "common shape", "r": "similar"}],
                "h": [
                    {
                        "d": "ordinary item",
                        "p": 0.75,
                        "b": "mixed",
                        "e": "shape support",
                        "u": "blur",
                    },
                    {
                        "d": "image artifact",
                        "p": 0.25,
                        "b": "unresolved",
                        "e": "weak edge",
                        "u": "low resolution",
                    },
                ],
                "u": ["fine structure"],
            }
        )
    result = normalize_annotation(
        {
            "g": {
                "c": "Person in dark shirt and shorts.",
                "o": [{"c": "dark shirt", "s": "strong", "e": "visible torso"}],
                "u": ["logo"],
            },
            "r": compact_regions,
        },
        regions,
    )
    assert result["global"]["caption"].startswith("Person")
    assert [row["region_id"] for row in result["regions"]] == [
        "eyes",
        "head",
        "upper_torso",
    ]
    assert result["regions"][0]["hypotheses"][0] == {
        "description": "ordinary item",
        "probability": 0.75,
        "basis": "mixed",
        "observable_support": "shape support",
        "uncertainty": "blur",
    }


def test_compact_reasoner_instruction_uses_alias_contract():
    reasoner = TextAnnotationReasoner(
        endpoint="http://127.0.0.1:1/v1/chat/completions",
        model_id="test",
        response_profile="compact_v1",
    )
    regions = [SimpleNamespace(region_id="eyes", category="eyewear", bbox_xyxy=(1, 2, 3, 4))]
    instruction = reasoner._instruction(regions, "rgb")
    assert '"g"' in instruction
    assert '"h"' in instruction
    assert "probabilities summing to one" in instruction
    assert "prefer a no-object or unresolved" in instruction
    assert "lower the positive-object probability" in instruction


def _separated_payload(caption: str):
    regions = []
    for region_id in ("eyes", "left_foot", "upper_torso"):
        regions.append(
            {
                "id": region_id,
                "s": "standalone regional description from SwinIR pixels",
                "h": [
                    {"d": "specific visible item"},
                    {"d": "no additional item"},
                ],
            }
        )
    return {
        "g": {
            "c": caption,
            "a": {
                "hd": "short dark hair",
                "up": "grey short-sleeved graphic T-shirt",
                "lo": "beige knee-length shorts",
                "ft": "dark open-toe sandals",
                "ca": "no clearly visible carried item",
                "ds": "red chest graphic",
            },
        },
        "r": regions,
    }


def test_swin_separated_caption_requires_slots_and_probability_free_rois():
    selected = [
        SimpleNamespace(region_id="eyes", category="eyewear"),
        SimpleNamespace(region_id="left_foot", category="footwear_detail"),
        SimpleNamespace(region_id="upper_torso", category="clothing_detail"),
    ]
    caption = (
        "A person with short dark hair wears a grey short-sleeved graphic T-shirt, "
        "beige knee-length shorts, and dark open-toe sandals, with no clearly visible "
        "carried item and a red chest design."
    )
    result = normalize_annotation(
        _separated_payload(caption), selected, require_swin_separated=True
    )
    assert 22 <= result["global"]["caption_word_count"] <= 35
    assert result["global"]["caption_profile"] == "swin_only_separated_22_35_v2"
    assert set(result["global"]["attributes"]) == {
        "head",
        "upper",
        "lower",
        "footwear",
        "carried",
        "distinctive",
    }
    assert "regional_addenda" not in result["global"]
    assert all(
        "probability" not in hypothesis
        for region in result["regions"]
        for hypothesis in region["hypotheses"]
    )


def test_swin_separated_caption_rejects_combination_and_vlm_probabilities():
    selected = [
        SimpleNamespace(region_id="eyes", category="eyewear"),
        SimpleNamespace(region_id="left_foot", category="footwear_detail"),
        SimpleNamespace(region_id="upper_torso", category="clothing_detail"),
    ]
    with pytest.raises(ValueError, match="22-35 words"):
        normalize_annotation(
            _separated_payload("Person in a grey shirt and shorts."),
            selected,
            require_swin_separated=True,
        )
    payload = _separated_payload(
        "A person with short dark hair wears a grey short-sleeved graphic T-shirt, "
        "beige knee-length shorts, and dark open-toe sandals, with no clearly visible "
        "carried item and a red chest design."
    )
    del payload["g"]["a"]["ca"]
    with pytest.raises(ValueError, match="attributes omit"):
        normalize_annotation(payload, selected, require_swin_separated=True)

    payload = _separated_payload(
        "A person with short dark hair wears a grey short-sleeved graphic T-shirt, "
        "beige knee-length shorts, and dark open-toe sandals, with no clearly visible "
        "carried item and a red chest design."
    )
    payload["g"]["x"] = [{"id": "eyes", "t": "eyeglasses"}]
    with pytest.raises(ValueError, match="must not contain regional addenda"):
        normalize_annotation(payload, selected, require_swin_separated=True)

    payload = _separated_payload(
        "A person with short dark hair wears a grey short-sleeved graphic T-shirt, "
        "beige knee-length shorts, and dark open-toe sandals, with no clearly visible "
        "carried item and a red chest design."
    )
    payload["r"][0]["h"][0]["p"] = 0.6
    with pytest.raises(ValueError, match="must not contain VLM-reported probabilities"):
        normalize_annotation(payload, selected, require_swin_separated=True)


def test_swin_separated_instruction_keeps_global_and_roi_outputs_independent():
    reasoner = TextAnnotationReasoner(
        endpoint="http://127.0.0.1:1/v1/chat/completions",
        model_id="test",
        response_profile="swin_separated_v1",
    )
    regions = [
        SimpleNamespace(
            region_id="eyes", category="eyewear", bbox_xyxy=(1, 2, 3, 4)
        )
    ]
    instruction = reasoner._instruction(regions, "rgb")
    assert "22-35 words" in instruction
    assert "head/hair/headwear" in instruction
    assert "carried item" in instruction
    assert "sole visual inputs" in instruction
    assert "two independent outputs and never merge them" in instruction
    assert "Do not assign probabilities" in instruction
    assert "Do not rewrite or append the global caption" in instruction
    assert "possible" not in instruction.lower()
    assert "may" not in instruction.lower()
    assert "might" not in instruction.lower()
    assert "Omit every other key" in instruction
    assert '"s":"4-14 word independent ROI caption"' in instruction
    assert '"p":' not in instruction
    assert "without repeating slots" in instruction
    assert "cannot both be true" in instruction
    assert "object and its attribute" in instruction


def test_swin_separated_rejects_missing_or_modal_roi_caption():
    selected = [SimpleNamespace(region_id="eyes", category="eyewear")]
    caption = (
        "A person with short dark hair wears a grey short-sleeved graphic T-shirt, "
        "beige knee-length shorts, and dark open-toe sandals, with no clearly visible "
        "carried item and a red chest design."
    )
    payload = _separated_payload(caption)
    payload["r"] = [payload["r"][0]]
    payload["r"][0]["s"] = ""
    with pytest.raises(ValueError, match="has no independent caption"):
        normalize_annotation(payload, selected, require_swin_separated=True)
    payload["r"][0]["s"] = "might show a thin dark frame"
    with pytest.raises(ValueError, match="must not contain modal qualifiers"):
        normalize_annotation(payload, selected, require_swin_separated=True)


def test_swin_separated_content_contains_no_lr_image():
    reasoner = TextAnnotationReasoner(
        endpoint="http://127.0.0.1:1/v1/chat/completions",
        model_id="test",
        response_profile="swin_separated_v1",
        roi_board_size_px=256,
    )
    lr = Image.new("RGB", (64, 128), (210, 10, 10))
    swin = Image.new("RGB", (128, 256), (20, 30, 40))
    regions = [
        SimpleNamespace(
            region_id="eyes", category="eyewear", bbox_xyxy=(40, 20, 88, 52)
        )
    ]
    content = reasoner._content(lr, swin, regions)
    images = [item for item in content if item["type"] == "image_url"]
    text = " ".join(
        item["text"] for item in content if item["type"] == "text"
    )
    assert len(images) == 2
    assert "LR" not in text
    assert "Image A" not in text
    encoded = images[0]["image_url"]["url"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as decoded:
        assert decoded.convert("RGB").getpixel((0, 0)) == (20, 30, 40)


def test_swin_separated_overlength_caption_gets_text_only_qwen_repair(monkeypatch):
    region = SimpleNamespace(
        region_id="eyes", category="eyewear", bbox_xyxy=(40, 20, 88, 52)
    )
    first = {
        "g": {
            "c": (
                "A person with short dark hair and no visible headwear wears a grey "
                "short-sleeved T-shirt with a prominent red chest graphic, beige "
                "knee-length shorts, and dark open-toe sandals, carries no clearly "
                "visible item, and has no other distinctive detail clearly visible."
            ),
            "a": {
                "hd": "short dark hair without visible headwear",
                "up": "grey short-sleeved red-graphic T-shirt",
                "lo": "beige knee-length shorts",
                "ft": "dark open-toe sandals",
                "ca": "no clearly visible carried item",
                "ds": "prominent red chest graphic",
            },
        },
        "r": [
            {
                "id": "eyes",
                "s": "dark horizontal structure around the eye area",
                "h": [{"d": "thin dark eyeglasses"}, {"d": "facial shadow"}],
            }
        ],
    }
    repaired = {
        "c": (
            "Person with short dark hair wears a grey short-sleeved red-graphic "
            "T-shirt, beige knee-length shorts, and dark open-toe sandals; carries "
            "nothing clearly visible; distinctive red chest graphic."
        )
    }
    documents = iter(
        [
            {
                "choices": [{"message": {"content": json.dumps(first)}}],
                "usage": {"completion_tokens": 100},
            },
            {
                "choices": [{"message": {"content": json.dumps(repaired)}}],
                "usage": {"completion_tokens": 40},
            },
        ]
    )
    requests = []

    class _Response:
        def __init__(self, document):
            self.document = document

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(self.document).encode("utf-8")

    def _urlopen(request, timeout):
        requests.append(json.loads(request.data.decode("utf-8")))
        return _Response(next(documents))

    monkeypatch.setattr(
        "qwen_imagination.text_annotation.reasoner.urllib.request.urlopen", _urlopen
    )
    reasoner = TextAnnotationReasoner(
        endpoint="http://127.0.0.1:1/v1/chat/completions",
        model_id="test",
        response_profile="swin_separated_v1",
        roi_board_size_px=256,
    )
    annotation, telemetry = reasoner.annotate(
        Image.new("RGB", (64, 128), (200, 10, 10)),
        Image.new("RGB", (128, 256), (20, 30, 40)),
        [region],
        modality="rgb",
        seed=7,
    )
    assert 22 <= annotation["global"]["caption_word_count"] <= 35
    assert telemetry["caption_repair"] is not None
    assert telemetry["usage"]["completion_tokens"] == 140
    assert len(requests) == 2
    assert all(
        isinstance(message["content"], str) for message in requests[1]["messages"]
    )


def test_unknown_response_profile_rejected():
    with pytest.raises(ValueError, match="response_profile"):
        TextAnnotationReasoner(
            endpoint="http://127.0.0.1:1/v1/chat/completions",
            model_id="test",
            response_profile="unknown",
        )
