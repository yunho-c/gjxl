#!/usr/bin/env python3
"""Generates the deterministic linear-RGB PFM used by the public CLI gate."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


WIDTH = 17
HEIGHT = 13


def sample_bytes(width: int = WIDTH, height: int = HEIGHT) -> bytes:
    header = f"PF\n{width} {height}\n-1.0\n".encode("ascii")
    pixels = bytearray()
    for y in reversed(range(height)):
        fy = y / (height - 1)
        for x in range(width):
            fx = x / (width - 1)
            texture = ((5 * x + 3 * y) % 11) / 110.0
            values = (
                0.03 + 0.72 * fx,
                0.04 + 0.64 * fy + texture,
                0.02 + 0.28 * fx + 0.48 * fy,
            )
            pixels.extend(struct.pack("<3f", *values))
    return header + pixels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    args = parser.parse_args()
    if args.width < 2 or args.height < 2:
        parser.error("--width and --height must both be at least 2")

    expected = sample_bytes(args.width, args.height)
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != expected:
            raise SystemExit(f"generated sample is stale: {args.output}")
        print(f"Verified deterministic sample: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(expected)
    print(f"Wrote {args.width}x{args.height} linear-RGB PFM: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
