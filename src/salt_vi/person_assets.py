"""Optional bridge from SALT core to the repository-owned person asset package."""

import sys
from pathlib import Path


def _load_package():
    try:
        import person_preprocessing
    except ModuleNotFoundError:
        source_checkout = Path(__file__).resolve().parents[2] / "person_preprocessing"
        if not source_checkout.is_dir():
            raise
        sys.path.insert(0, str(source_checkout))
        import person_preprocessing
    return person_preprocessing


_package = _load_package()
PersonAssetStore = _package.PersonAssetStore
PersonVisualSource = _package.PersonVisualSource

__all__ = ["PersonAssetStore", "PersonVisualSource"]
