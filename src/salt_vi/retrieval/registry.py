from . import ir_to_rgb_text, legacy


_BACKENDS = {
    legacy.NAME: legacy,
    ir_to_rgb_text.NAME: ir_to_rgb_text,
}
SUPPORTED_RETRIEVAL_BACKENDS = tuple(_BACKENDS)


def get_retrieval_protocol(name):
    normalized = str(name or "legacy").lower()
    try:
        return _BACKENDS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported retrieval_backend {normalized!r}; "
            f"expected one of {list(SUPPORTED_RETRIEVAL_BACKENDS)}"
        ) from exc


get_retrieval_backend = get_retrieval_protocol
