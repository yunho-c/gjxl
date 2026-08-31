#!/usr/bin/env python3
"""Prepares a normal still image for gjxl_encoding_benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from encode_image import (
    WrapperError,
    convert_to_pfm,
    find_executable,
    is_pfm,
)


def run_benchmark(
    benchmark_command: str,
    benchmark_args: list[str],
    source: Path,
) -> int:
    benchmark = find_executable(benchmark_command, "GJXL benchmark")
    try:
        result = subprocess.run(
            [benchmark, "--input", str(source), *benchmark_args],
            check=False,
        )
    except OSError as exc:
        raise WrapperError(f"Unable to run GJXL benchmark: {exc}") from exc
    return result.returncode


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a PFM directly, or use optional ImageMagick to prepare "
            "a single-frame linear-sRGB RGB PFM before benchmarking."
        )
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--magick",
        default=os.environ.get("GJXL_MAGICK", "magick"),
        help="ImageMagick executable for non-PFM input (default: %(default)s)",
    )
    parser.add_argument(
        "--background",
        choices=("white", "black"),
        default=os.environ.get("GJXL_ALPHA_BACKGROUND", "white"),
        help="linear RGB background used to composite alpha (default: %(default)s)",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to the benchmark after an optional -- separator",
    )
    args = parser.parse_args(argv)
    if args.benchmark_args[:1] == ["--"]:
        args.benchmark_args = args.benchmark_args[1:]
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    source = args.input.resolve()
    if not source.is_file():
        raise WrapperError(
            f"Input image does not exist or is not a file: {args.input}"
        )

    if is_pfm(source):
        return run_benchmark(args.benchmark, args.benchmark_args, source)

    with tempfile.TemporaryDirectory(prefix="gjxl-benchmark-image-") as temporary:
        prepared = Path(temporary) / "input.pfm"
        width, height = convert_to_pfm(
            args.magick,
            source,
            prepared,
            args.background,
        )
        print(
            f"Prepared {width}x{height} linear RGB from {args.input} "
            f"with ImageMagick",
            flush=True,
        )
        return run_benchmark(args.benchmark, args.benchmark_args, prepared)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except WrapperError as exc:
        print(f"Input preparation error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
