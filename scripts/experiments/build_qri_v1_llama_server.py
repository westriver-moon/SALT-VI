#!/usr/bin/env python3
"""Reproduce the pinned CUDA llama-server build used by QRI-v1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


LLAMA_REVISION = "8ef78e644f559db4e8716b59bf76b8e11619337d"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda-12.4"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    source = args.source.expanduser().resolve()
    revision = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={source}",
            "-C",
            str(source),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    if revision != LLAMA_REVISION:
        raise ValueError(f"expected llama.cpp {LLAMA_REVISION}, got {revision}")
    if args.jobs < 1 or args.jobs > 16:
        raise ValueError("QRI build jobs must be in [1, 16]")
    nvcc = args.cuda_root / "bin" / "nvcc"
    if not nvcc.is_file():
        raise FileNotFoundError(nvcc)
    build = source / "build"
    configure = [
        args.cmake,
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DGGML_CUDA=ON",
        "-DGGML_CUDA_FA=ON",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_UI=OFF",
        "-DLLAMA_USE_PREBUILT_UI=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    compile_command = [
        args.cmake,
        "--build",
        str(build),
        "--target",
        "llama-server",
        "-j",
        str(args.jobs),
    ]
    plan = {
        "revision": revision,
        "source": str(source),
        "build": str(build),
        "cuda_root": str(args.cuda_root),
        "configure": configure,
        "compile": compile_command,
        "execute": bool(args.execute),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    environment = dict(os.environ)
    environment["CUDACXX"] = str(nvcc)
    subprocess.run(configure, check=True, env=environment)
    subprocess.run(compile_command, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
