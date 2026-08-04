#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if __name__ == '__main__':
    runpy.run_module('salt_vi.entrypoints.train', run_name='__main__')
