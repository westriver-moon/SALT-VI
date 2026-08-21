from .quadruple_patch import QuadruplePatchInput


QUADRUPLE_BRANCH_ORDER = (
    "visible_global",
    "visible_channel",
    "infrared_global",
    "infrared_channel",
)
SUPPORTED_VISUAL_INPUT_BACKENDS = ("single", "quadruple_patch")


def normalize_visual_input_backend(name):
    normalized = str(name or "single").strip().lower()
    if normalized not in SUPPORTED_VISUAL_INPUT_BACKENDS:
        raise ValueError(
            f"Unsupported visual_input_backend {normalized!r}; expected one of "
            f"{list(SUPPORTED_VISUAL_INPUT_BACKENDS)}"
        )
    return normalized


def build_visual_input_plugin(name, patch_embed, branch_order=None):
    backend = normalize_visual_input_backend(name)
    if backend == "single":
        return None
    order = tuple(branch_order or QUADRUPLE_BRANCH_ORDER)
    if order != QUADRUPLE_BRANCH_ORDER:
        raise ValueError(
            "quadruple_patch requires the canonical branch order "
            f"{list(QUADRUPLE_BRANCH_ORDER)}, got {list(order)}"
        )
    return QuadruplePatchInput(patch_embed, branch_order=order)
