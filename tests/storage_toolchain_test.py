#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Check the real header/configuration boundary, including unsupported modes."""

import argparse
from pathlib import Path
import subprocess
import sys
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
    public_header = '#include "codec/adaptive_quantization.h"\nint main() {}\n'
    msvc = Path(args.compiler).name.lower() in ("cl", "cl.exe")
    gnu = False
    if not msvc:
        macros = subprocess.run([args.compiler, "-std=c++20", "-dM", "-E", "-x", "c++", "-"],
                                input="#include <version>\n", text=True, capture_output=True,
                                check=True, timeout=45).stdout
        gnu = "#define __GLIBCXX__ " in macros
    # Load the library first, then alter its identity in this translation unit.
    # No production escape hatch is introduced to admit an unknown library.
    cases = ([
        ("_MSC_VER", "1938", "audited MSVC 19.37/STL 143 update 202305"),
        ("_MSVC_STL_VERSION", "999", "audited MSVC 19.37/STL 143 update 202305"),
        ("_MSVC_STL_UPDATE", "999999L", "audited MSVC 19.37/STL 143 update 202305"),
        ("_MSVC_STL_VERSION", None, "an audited standard library"),
        ("_ITERATOR_DEBUG_LEVEL", "2", "MSVC iterator debug level 0"),
    ] if msvc else [
        ("__GNUC_MINOR__", "4", "audited GCC 13.3.0/libstdc++ headers 20240904"),
        ("_GLIBCXX_RELEASE", "14", "audited GCC 13.3.0/libstdc++ headers 20240904"),
        ("__GLIBCXX__", "20250101", "audited GCC 13.3.0/libstdc++ headers 20240904"),
        ("__GLIBCXX__", None, "an audited standard library"),
        ("_GLIBCXX_USE_CXX11_ABI", "0", "release libstdc++ with C++11 ABI"),
        ("_GLIBCXX_DEBUG", "1", "release libstdc++ with C++11 ABI"),
        ("_GLIBCXX_PARALLEL", "1", "release libstdc++ with C++11 ABI"),
    ] if gnu else [
        ("_LIBCPP_VERSION", "999999", "audited libc++ 200100"),
        ("_LIBCPP_VERSION", None, "an audited standard library"),
        ("__apple_build_version__", "1", "audited Apple Clang build 17000604"),
        ("_LIBCPP_ABI_VERSION", "2", "stable libc++ ABI version 1"),
        ("_LIBCPP_ABI_UNSTABLE", "1", "stable libc++ ABI version 1"),
    ])
    with tempfile.TemporaryDirectory(prefix="gjxl-storage-header-") as directory:
        source_path = Path(directory) / "probe.cpp"

        def compile_header(source, mode, diagnostic=None):
            source_path.write_text(source)
            if msvc:
                command = [args.compiler, "/nologo", "/Zs", "/EHsc", "/MD",
                           "/DNDEBUG", "/std:" + ("c++latest" if mode == "c++23" else mode),
                           "/I" + str(root / "src"), str(source_path)]
            else:
                command = [args.compiler, "-fsyntax-only", "-std=" + mode,
                           "-I" + str(root / "src"), str(source_path)]
            run(command, diagnostic=diagnostic)

        for mode in ("c++20", "c++23"):
            compile_header(public_header, mode)
        # MSVC 19.37 has no C++26 mode and does not permit spoofing _MSVC_LANG.
        # Exercise its real C++17 mode here; CMake rejects C++26 below.
        compile_header(public_header, "c++17" if msvc or gnu else "c++2c",
                       "GJXL storage contract requires C++20 or C++23")
        for macro, value, diagnostic in cases:
            source = ('#include <version>\n#include <span>\n#include <string>\n'
                      '#include <unordered_map>\n#include <vector>\n#undef ' + macro + '\n')
            if value is not None:
                source += '#define ' + macro + ' ' + value + '\n'
            source += public_header
            for mode in ("c++20", "c++23"):
                compile_header(source, mode, diagnostic)
    with tempfile.TemporaryDirectory(prefix="gjxl-storage-toolchain-") as directory:
        # Actual production configuration, before any shader or encoder build.
        for language in (("CXX", "OBJCXX") if sys.platform == "darwin" else ("CXX",)):
            run([args.cmake, "-S", str(root), "-B", str(Path(directory) / language),
                 "-DGJXL_BUILD_TESTS=OFF", "-DGJXL_BUILD_BENCHMARKS=OFF",
                 "-DGJXL_ENABLE_CUDA=OFF", "-DCMAKE_BUILD_TYPE=Release",
                 f"-DCMAKE_{language}_STANDARD=26"],
                diagnostic="GJXL storage contract requires C++20 or C++23")
    print("Storage toolchain: supported public header and unsupported identities/modes checked")


if __name__ == "__main__":
    main()
