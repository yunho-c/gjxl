#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

"""Runs the scalar libjxl fixture generator and checks its committed header."""

from __future__ import annotations

import argparse
import difflib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "tests/butteraugli_goldens_generated.h"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    generated = subprocess.run(
        [str(args.generator)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    if args.check:
        committed = args.output.read_text() if args.output.exists() else ""
        if committed == generated:
            return 0
        diff = difflib.unified_diff(
            committed.splitlines(keepends=True),
            generated.splitlines(keepends=True),
            fromfile=str(args.output),
            tofile="scalar generator output",
        )
        print("".join(diff), end="")
        return 1

    args.output.write_text(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
