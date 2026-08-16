from collections.abc import Mapping


NAME = "legacy"
RESULT_KEY = "Fusion"
IS_LEGACY = True
QUERY_NAME = "infrared"
GALLERY_NAME = "visible"


def _value(config, name, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _test_modalities(config):
    raw = str(_value(config, "test_modality", ""))
    modalities = tuple(part.strip() for part in raw.split(",") if part.strip())
    supported = {"IR", "Fusion", "Text"}
    if (
        not modalities
        or len(set(modalities)) != len(modalities)
        or not set(modalities) <= supported
    ):
        raise ValueError(
            "legacy retrieval requires a comma-separated non-empty subset of "
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
    return None


def validate(config, sr_backend=None, sr_modalities=None):
    dataset = str(_value(config, "dataset", "")).lower()
    if dataset not in {"sysu", "regdb", "llcm"}:
        raise ValueError(
            "legacy retrieval supports only SYSU-MM01, RegDB, and LLCM"
        )
    modalities = _test_modalities(config)
    caption_lookup = query_caption_lookup(config)
    if any(name in modalities for name in ("Fusion", "Text")):
        if caption_lookup != "identity":
            raise ValueError(
                "legacy Fusion/Text retrieval requires identity caption lookup"
            )
    elif caption_lookup is not None:
        raise ValueError("legacy IR retrieval must not require captions")
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
    from salt_vi.engine.test import evaluate_legacy

    return evaluate_legacy(model, loader, config, device)
