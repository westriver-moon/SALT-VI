from __future__ import annotations

import base64
import io
import json
import urllib.request
from dataclasses import dataclass

from PIL import Image

from .schema import Region


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + payload


@dataclass
class OpenAICompatibleQwenBackend:
    """Thin llama-server adapter for the audited local Qwen GGUF deployment."""

    endpoint: str = "http://127.0.0.1:18080/v1/chat/completions"
    model_id: str = "third-party-qwen3.8-27b-ud-q4-k-xl"
    timeout_seconds: float = 360.0
    max_tokens: int = 220

    def sample_atomic(
        self,
        full_swin: Image.Image,
        roi_board: Image.Image,
        region: Region,
        *,
        modality: str,
        observed: str,
        system_prompt: str,
        seed: int,
        temperature: float,
        thinking: bool,
    ) -> str:
        del modality
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Observed facts: {observed or 'none recorded.'}"},
                        {"type": "text", "text": "SwinIR full image."},
                        {"type": "image_url", "image_url": {"url": _data_url(full_swin)}},
                        {
                            "type": "text",
                            "text": f"Target ROI {region.region_id} / {region.category}.",
                        },
                        {"type": "image_url", "image_url": {"url": _data_url(roi_board)}},
                    ],
                },
            ],
            "temperature": float(temperature),
            "top_p": 0.9,
            "seed": int(seed),
            "max_tokens": int(self.max_tokens),
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            document = json.loads(response.read().decode("utf-8"))
        content = str(document["choices"][0]["message"]["content"]).strip()
        if content.startswith("{"):
            parsed = json.loads(content)
            content = str(parsed.get("atom", "")).strip()
        if not content:
            raise ValueError("Qwen returned an empty atomic response")
        return content
