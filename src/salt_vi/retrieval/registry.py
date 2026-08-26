from . import identity_text, ir_to_rgb_text


_BACKENDS = {
    identity_text.NAME: identity_text,
    ir_to_rgb_text.NAME: ir_to_rgb_text,
}
_ALIASES = {"legacy": identity_text.NAME}
SUPPORTED_RETRIEVAL_BACKENDS = tuple(_BACKENDS)


def get_retrieval_protocol(name):
    normalized = str(name or identity_text.NAME).lower()
    normalized = _ALIASES.get(normalized, normalized)
    try:
        return _BACKENDS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported retrieval_backend {normalized!r}; "
            f"expected one of {list(SUPPORTED_RETRIEVAL_BACKENDS)}"
        ) from exc
