from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F


SUPPORTED_ATTENTION_BACKENDS = ("manual", "sdpa", "flash")


def normalize_attention_backend(value) -> str:
    backend = str(value or "manual").strip().lower()
    if backend not in SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError(
            f"Unsupported attention backend {backend!r}; expected one of "
            f"{list(SUPPORTED_ATTENTION_BACKENDS)}"
        )
    return backend


def _sdpa():
    function = getattr(F, "scaled_dot_product_attention", None)
    if function is None:
        raise RuntimeError(
            "The sdpa and flash attention backends require PyTorch 2.0 or newer"
        )
    return function


def validate_attention_backend_runtime(value) -> str:
    backend = normalize_attention_backend(value)
    if backend != "manual":
        _sdpa()
    return backend


@contextmanager
def _flash_only_kernel():
    attention = getattr(torch.nn, "attention", None)
    if attention is not None and hasattr(attention, "sdpa_kernel"):
        with attention.sdpa_kernel(attention.SDPBackend.FLASH_ATTENTION):
            yield
        return

    cuda_backends = getattr(torch.backends, "cuda", None)
    sdp_kernel = getattr(cuda_backends, "sdp_kernel", None)
    if sdp_kernel is None:
        raise RuntimeError(
            "Forcing the flash attention backend requires a CUDA PyTorch build "
            "with SDPA kernel controls"
        )
    with sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False,
    ):
        yield


def run_scaled_dot_product_attention(
    query,
    key,
    value,
    *,
    scale: float,
    dropout_p: float,
    training: bool,
    backend: str,
):
    """Run exact attention without changing QKV or projection parameters."""
    backend = normalize_attention_backend(backend)
    dropout_p = float(dropout_p) if training else 0.0

    if backend == "manual":
        weights = (query @ key.transpose(-2, -1)) * float(scale)
        weights = weights.softmax(dim=-1)
        weights = F.dropout(weights, p=dropout_p, training=training)
        return weights @ value

    sdpa = _sdpa()
    default_scale = query.shape[-1] ** -0.5
    if float(scale) != float(default_scale):
        query = query * (float(scale) / float(default_scale))

    if backend == "sdpa":
        return sdpa(
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=False,
        )

    if not query.is_cuda:
        raise RuntimeError("The flash attention backend requires CUDA tensors")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "The flash attention backend requires float16 or bfloat16 QKV tensors"
        )
    try:
        with _flash_only_kernel():
            return sdpa(
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=False,
            )
    except RuntimeError as error:
        raise RuntimeError(
            "The requested flash attention kernel could not run for the current "
            f"QKV tensors: {error}"
        ) from error
