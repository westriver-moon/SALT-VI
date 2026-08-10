#!/usr/bin/env python3
from pathlib import Path
import sys


FEATURE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FEATURE_ROOT.parent
for source in (FEATURE_ROOT / "src", PROJECT_ROOT / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from salt_feature_analysis.cli import main


if __name__ == "__main__":
    main()

