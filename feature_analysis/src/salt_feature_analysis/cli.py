from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

from .config import load_analysis_config, validate_enabled_representations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SALT-VI reusable feature analysis")
    parser.add_argument("command", choices=("validate", "extract", "analyze", "all"))
    parser.add_argument("--config", required=True, help="Analysis YAML path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    config = load_analysis_config(args.config)
    validate_enabled_representations(config)
    if args.command == "validate":
        print(json.dumps({"status": "valid", "run_id": config["run_id"]}, indent=2))
        return

    # CUDA visibility must be selected before torch or SALT modules are imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["runtime"]["cuda_visible_devices"])
    if args.command in {"extract", "all"}:
        from .extraction import extract_all

        catalog = extract_all(config)
        print(json.dumps({"extracted": len(catalog["artifacts"]), "run_id": config["run_id"]}, indent=2))
    if args.command in {"analyze", "all"}:
        from .analysis import analyze_all

        print(json.dumps(analyze_all(config), indent=2))

