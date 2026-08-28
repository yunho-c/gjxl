#!/usr/bin/env python3
"""Runs gjxl_encode after preparing an optional ImageMagick PFM input."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


class WrapperError(RuntimeError):
    """An actionable command-line preparation failure."""


def find_executable(command: str, description: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise WrapperError(f"{description} executable was not found: {command}")
    return resolved


def run_checked(command: list[str], description: str) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise WrapperError(f"Unable to run {description}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise WrapperError(f"{description} failed{suffix}")
    return result.stdout


def require_single_frame(magick: str, source: Path) -> None:
    output = run_checked(
        [magick, "identify", "-ping", "-format", "%n\n", str(source)],
        "ImageMagick input inspection",
    )
    try:
        frame_counts = [int(line) for line in output.splitlines() if line]
    except ValueError as exc:
        raise WrapperError(
            "ImageMagick returned an invalid frame count"
        ) from exc
    if not frame_counts or any(count != frame_counts[0] for count in frame_counts):
        raise WrapperError("ImageMagick did not report a consistent frame count")
    if frame_counts[0] != 1:
        raise WrapperError(
            f"Animated or multi-page input is unsupported ({frame_counts[0]} frames)"
        )


def validate_pfm(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as stream:
            if stream.readline() != b"PF\n":
                raise WrapperError(
                    "ImageMagick did not produce a three-channel RGB PFM"
                )
            dimensions = stream.readline().split()
            if len(dimensions) != 2:
                raise WrapperError("ImageMagick produced invalid PFM dimensions")
            width, height = (int(value) for value in dimensions)
            scale = float(stream.readline())
            header_bytes = stream.tell()
    except (OSError, ValueError) as exc:
        raise WrapperError(f"Unable to validate ImageMagick PFM output: {exc}") from exc

    if width <= 0 or height <= 0 or not math.isfinite(scale) or scale == 0.0:
        raise WrapperError("ImageMagick produced an invalid PFM header")
    expected_bytes = header_bytes + width * height * 3 * 4
    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise WrapperError(f"Unable to inspect ImageMagick PFM output: {exc}") from exc
    if actual_bytes != expected_bytes:
        raise WrapperError(
            "ImageMagick produced an incomplete or malformed PFM payload"
        )
    return width, height


def convert_to_pfm(
    magick_command: str,
    source: Path,
    destination: Path,
    background: str,
) -> tuple[int, int]:
    magick = find_executable(magick_command, "ImageMagick")
    require_single_frame(magick, source)
    run_checked(
        [
            magick,
            str(source),
            "-auto-orient",
            "-colorspace",
            "RGB",
            "-background",
            background,
            "-alpha",
            "remove",
            "-alpha",
            "off",
            "-type",
            "TrueColor",
            f"PFM:{destination}",
        ],
        "ImageMagick conversion",
    )
    return validate_pfm(destination)


def is_pfm(path: Path) -> bool:
    if path.suffix.lower() == ".pfm":
        return True
    try:
        with path.open("rb") as stream:
            return stream.read(2) in (b"PF", b"Pf")
    except OSError as exc:
        raise WrapperError(f"Unable to read input image: {exc}") from exc


def run_encoder(
    encoder_command: str,
    encoder_args: list[str],
    source: Path,
    destination: Path,
) -> int:
    encoder = find_executable(encoder_command, "GJXL encoder")
    try:
        result = subprocess.run(
            [encoder, *encoder_args, str(source), str(destination)],
            check=False,
        )
    except OSError as exc:
        raise WrapperError(f"Unable to run GJXL encoder: {exc}") from exc
    return result.returncode


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode a PFM directly, or use optional ImageMagick to prepare a "
            "single-frame linear-sRGB RGB PFM before running gjxl_encode."
        )
    )
    parser.add_argument("--encoder", required=True, help="path to gjxl_encode")
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
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "encoder_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to gjxl_encode after an optional -- separator",
    )
    args = parser.parse_args(argv)
    if args.encoder_args[:1] == ["--"]:
        args.encoder_args = args.encoder_args[1:]
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    source = args.input.resolve()
    if not source.is_file():
        raise WrapperError(f"Input image does not exist or is not a file: {args.input}")

    if is_pfm(source):
        return run_encoder(args.encoder, args.encoder_args, source, args.output)

    with tempfile.TemporaryDirectory(prefix="gjxl-image-") as temporary:
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
        return run_encoder(args.encoder, args.encoder_args, prepared, args.output)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except WrapperError as exc:
        print(f"Input preparation error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
