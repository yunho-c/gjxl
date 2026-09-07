#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Check the real header/configuration boundary, including unsupported modes."""

import argparse
from pathlib import Path
import subprocess
import tempfile


def run(command, *, source=None, diagnostic=None):
    result = subprocess.run(command, input=source, text=True, capture_output=True, timeout=45)
    output = result.stdout + result.stderr
    if diagnostic is None:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0 and diagnostic in output, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--cmake", required=True)
    args = parser.parse_args()
    root = args.source.resolve()
    compile_command = [args.compiler, "-fsyntax-only", "-x", "c++", "-", "-I" + str(root / "src")]
    public_header = '#include "codec/adaptive_quantization.h"\nint main() {}\n'
    for mode in ("c++20", "c++23"):
        run(compile_command + ["-std=" + mode], source=public_header)
    run(compile_command + ["-std=c++2c"], source=public_header,
        diagnostic="GJXL storage contract requires C++20 or C++23")
    # Load libc++ first, then alter its identity in this test translation unit.
    # No production escape hatch is introduced to admit an unknown library.
    cases = [
        ("_LIBCPP_VERSION", "999999", "audited libc++ 200100"),
        ("_LIBCPP_VERSION", None, "audited libc++ 200100"),
        ("__apple_build_version__", "1", "audited Apple Clang build 17000604"),
        ("_LIBCPP_ABI_VERSION", "2", "stable libc++ ABI version 1"),
        ("_LIBCPP_ABI_UNSTABLE", "1", "stable libc++ ABI version 1"),
    ]
    for macro, value, diagnostic in cases:
        source = '#include <version>\n#undef ' + macro + '\n'
        if value is not None:
            source += '#define ' + macro + ' ' + value + '\n'
        source += public_header
        for mode in ("c++20", "c++23"):
            run(compile_command + ["-std=" + mode], source=source, diagnostic=diagnostic)
    with tempfile.TemporaryDirectory(prefix="gjxl-storage-toolchain-") as directory:
        # Actual production configuration, before any shader or encoder build.
        for language in ("CXX", "OBJCXX"):
            run([args.cmake, "-S", str(root), "-B", str(Path(directory) / language),
                 "-DGJXL_BUILD_TESTS=OFF", "-DGJXL_BUILD_BENCHMARKS=OFF",
                 f"-DCMAKE_{language}_STANDARD=26"],
                diagnostic="GJXL storage contract requires C++20 or C++23")
    print("Storage toolchain: supported public header and unsupported identities/modes checked")


if __name__ == "__main__":
    main()
