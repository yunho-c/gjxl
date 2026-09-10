#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Exercise production shader dependencies under Make/Ninja and incremental builds."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


def run(args, log):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True, timeout=45)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Command failed: {args}\n{result.stdout}\n{result.stderr}")


def verify(directory, profiling, expected_count):
    library = (directory / "metal/gjxl.metallib").read_bytes()
    include = (directory / "metal/gjxl_embedded_metallib.inc").read_text()
    embedded = bytes(int(x, 16) for x in re.findall(r"0x([0-9a-f]{2})", include))
    declared = int(re.search(r"gjxl_embedded_metallib_len = (\d+)", include)[1])
    assert library.endswith(b"COMPLETE\n") and embedded == library and declared == len(library)
    assert int((directory / "metal/gjxl.link-count").read_text()) == expected_count
    if profiling:
        assert (directory / "metal/gjxl.metallibsym").read_text() == hashlib.sha256(library).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--make-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    fixture = Path(__file__).resolve().parent
    scratch = args.artifacts.resolve() if args.artifacts else Path(tempfile.mkdtemp(prefix="gjxl-metal-graph-"))
    scratch.mkdir(parents=True, exist_ok=True)
    generators = ["Unix Makefiles"]
    if not args.make_only and shutil.which("ninja"):
        generators.append("Ninja")
    succeeded = False
    try:
        for generator in generators:
            for profiling in (False, True):
                directory = scratch / (generator.replace(" ", "-") + f"-profile{int(profiling)}")
                # Every run starts fresh; retained evidence is never overwritten.
                directory.mkdir()
                run(["cmake", "-S", source, "-B", directory, "-G", generator,
                     "-DGJXL_BUILD_TESTS=OFF", "-DGJXL_BUILD_BENCHMARKS=OFF", "-DGJXL_ENABLE_LIBJXL_REFERENCE=OFF",
                     f"-DGJXL_ENABLE_METAL_PROFILING={'ON' if profiling else 'OFF'}",
                     f"-DXCRUN_EXECUTABLE:FILEPATH={fixture / 'fake_metal_toolchain.py'}",
                     f"-DCMAKE_PROJECT_INCLUDE={fixture / 'metal_build_graph_fixture.cmake'}"], directory / "configure.log")
                command = ["cmake", "--build", directory, "--target", "gjxl_metal_graph_test_consumer", "--parallel", "16"]
                run(command, directory / "initial.log")
                verify(directory, profiling, 1)
                run(command, directory / "unchanged.log")
                verify(directory, profiling, 1)
                # Rebuild after an input IR changes. A target-only dependency
                # would avoid races but fail to refresh embedded bytes/symbols.
                time.sleep(1.05)
                ir = next((directory / "metal").glob("*.ir"))
                os.utime(ir, None)
                run(command, directory / "changed.log")
                verify(directory, profiling, 2)
                print(f"{generator} profiling={profiling}: single producer, complete and refreshed payload", flush=True)
        succeeded = True
    finally:
        if succeeded and args.artifacts is None:
            shutil.rmtree(scratch)
        else:
            print(f"Build-graph artifacts: {scratch}", flush=True)


if __name__ == "__main__":
    main()
