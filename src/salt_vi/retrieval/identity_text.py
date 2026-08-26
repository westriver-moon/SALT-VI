from collections.abc import Mapping


NAME = "identity_text"
RESULT_KEYS = ("IR", "Fusion", "Text")
RESULT_KEY = "Fusion"
QUERY_NAME = "infrared"
GALLERY_NAME = "visible"


def _value(config, name, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _test_modalities(config):
    raw = str(_value(config, "test_modality", ""))
    modalities = tuple(part.strip() for part in raw.split(",") if part.strip())
    supported = set(RESULT_KEYS)
    if (
        not modalities
        or len(set(modalities)) != len(modalities)
        or not set(modalities) <= supported
    ):
        raise ValueError(
            "identity_text retrieval requires a comma-separated non-empty subset of "
            "IR, Fusion, Text"
        )
    return modalities


def train_text_modalities(config):
    return ("rgb", "ir")


def query_caption_lookup(config):
    test_modality = str(_value(config, "test_modality", ""))
    return "identity" if any(name in test_modality for name in ("Fusion", "Text")) else None


def gallery_caption_lookup(config):
    return None


def training_recipe(config):
    training_mode = str(_value(config, "training_mode", ""))
    if training_mode == "RGB_IR_Text":
        return "identity_text_rgb_ir_text"
    if training_mode == "RGB_IR":
        return "identity_text_rgb_ir"
    return None


def validate(config, sr_backend=None, sr_modalities=None):
    dataset = str(_value(config, "dataset", "")).lower()
    if dataset not in {"sysu", "regdb", "llcm"}:
        raise ValueError(
            "identity_text retrieval supports only SYSU-MM01, RegDB, and LLCM"
        )
    modalities = _test_modalities(config)
    caption_lookup = query_caption_lookup(config)
    if any(name in modalities for name in ("Fusion", "Text")):
        if caption_lookup != "identity":
            raise ValueError(
                "identity_text Fusion/Text retrieval requires identity caption lookup"
            )
    elif caption_lookup is not None:
        raise ValueError("identity_text IR retrieval must not require captions")
    if dataset == "regdb":
        direction = str(_value(config, "regdb_test_mode", ""))
        if direction not in {"t-v", "v-t"}:
            raise ValueError("RegDB retrieval direction must be 't-v' or 'v-t'")
        first_trial = int(_value(config, "trial", 1))
        trial_count = int(_value(config, "eval_num_regdb", 1))
        if first_trial < 1 or trial_count < 1 or first_trial + trial_count - 1 > 10:
            raise ValueError("RegDB numbered trials must stay within 1-10")
    return config


def evaluate(model, loader, config, device):
    from salt_vi.engine.test import evaluate_identity_text

    return evaluate_identity_text(model, loader, config, device)
