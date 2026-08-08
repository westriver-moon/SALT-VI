from . import ir_to_rgb_text


_BACKENDS = {
    ir_to_rgb_text.NAME: ir_to_rgb_text,
}
SUPPORTED_RETRIEVAL_BACKENDS = ("legacy", *_BACKENDS)


def get_retrieval_backend(name):
    normalized = str(name or "legacy").lower()
    if normalized == "legacy":
        return None
    try:
        return _BACKENDS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported retrieval_backend {normalized!r}; "
            f"expected one of {list(SUPPORTED_RETRIEVAL_BACKENDS)}"
        ) from exc
