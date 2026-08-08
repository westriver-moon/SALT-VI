NAME = "legacy"
RESULT_KEY = "Fusion"
IS_LEGACY = True
QUERY_NAME = "infrared"
GALLERY_NAME = "visible"


def train_text_modalities(config):
    return ("rgb", "ir")


def query_caption_lookup(config):
    test_modality = str(getattr(config, "test_modality", ""))
    return "identity" if any(name in test_modality for name in ("Fusion", "Text")) else None


def gallery_caption_lookup(config):
    return None


def training_recipe(config):
    return None


def validate(config, sr_backend=None, sr_modalities=None):
    return config


def evaluate(model, loader, config, device):
    from salt_vi.engine.test import evaluate_legacy

    return evaluate_legacy(model, loader, config, device)
