#!/usr/bin/env python3
"""Send one real image through the QRI llama.cpp multimodal endpoint."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
import urllib.request


def _json_object(text: str) -> dict:
    text = str(text).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("visual smoke response is not one JSON object")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8080/v1/chat/completions"
    )
    parser.add_argument(
        "--model", default="third-party-qwen3.8-27b-ud-q4-k-xl"
    )
    args = parser.parse_args(argv)
    image = args.image.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Inspect the supplied surveillance image. Use visual evidence only and "
                    "return JSON with keys person_visible, modality, and observations. "
                    "Observations must be a short list of directly visible facts."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Audit this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                ],
            },
        ],
        "temperature": 0.1,
        "seed": 20260819,
        "max_tokens": 768,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_effort": "high",
    }
    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        document = json.loads(response.read().decode("utf-8"))
    message = document["choices"][0]["message"]
    content = None
    for field in ("content", "reasoning_content"):
        if not message.get(field):
            continue
        try:
            content = _json_object(message[field])
            break
        except (json.JSONDecodeError, ValueError):
            continue
    if content is None:
        raise ValueError("visual smoke response contains no parseable final JSON object")
    if set(content) != {"person_visible", "modality", "observations"}:
        raise ValueError(f"unexpected visual smoke response: {content}")
    print(json.dumps(content, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
