#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Verify that native-only edits rebuild Cargo's codec and enabled GPU kernels."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def build(workspace, features):
    command = ["cargo", "build", "--workspace", "--message-format=json"]
    if features:
        command += ["--features", features]
    result = subprocess.run(command, cwd=workspace, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    outputs = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    native = [row for row in outputs if row.get("reason") == "build-script-executed"
              and "gjxl-sys" in row.get("package_id", "")]
    if len(native) != 1:
        raise RuntimeError("Expected one gjxl-sys native build script")
    build_dir = Path(native[0]["out_dir"]) / "build"
    objects = [p for p in build_dir.rglob("dct.cpp.*") if p.suffix in (".o", ".obj")]
    if len(objects) != 1:
        raise RuntimeError(f"Expected one scalar DCT object below {build_dir}")
    tracked = {"src/codec/dct.cpp": objects[0]}
    if "cuda" in features:
        kernels = [p for p in build_dir.rglob("cuda_aq_cpu_order_kernels.cu.*")
                   if p.suffix in (".o", ".obj")]
        if len(kernels) != 1:
            raise RuntimeError("Expected one exact-arithmetic CUDA object")
        tracked["src/gpu/cuda/cuda_aq_cpu_order_kernels.cu"] = kernels[0]
    if sys.platform == "darwin":
        shaders = list(build_dir.rglob("aq_reconstruction.ir"))
        if len(shaders) != 1:
            raise RuntimeError("Expected one Metal reconstruction shader object")
        tracked["src/gpu/metal/kernels/aq_reconstruction.metal"] = shaders[0]
    return tracked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parent
    source = Path(os.environ.get("GJXL_SOURCE_DIR", workspace.parent)).resolve()
    tracked = build(workspace, args.features)
    before = {name: obj.stat().st_mtime_ns for name, obj in tracked.items()}
    original = {name: (source / name).stat() for name in tracked}
    contents = {name: (source / name).read_bytes() for name in tracked}
    # A full second also handles source trees on filesystems with coarse mtimes.
    time.sleep(1.1)
    try:
        for name in tracked:
            os.utime(source / name, None)
        rebuilt = build(workspace, args.features)
        for name, obj in tracked.items():
            if rebuilt.get(name) != obj or obj.stat().st_mtime_ns <= before[name]:
                raise RuntimeError(f"A native-only edit did not rebuild {name}")
    finally:
        # No source bytes change; restore the caller's file metadata as well.
        changed = []
        for name, metadata in original.items():
            path = source / name
            if path.is_file() and path.read_bytes() == contents[name]:
                os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
            else:
                changed.append(name)
        if changed:
            raise RuntimeError(f"Native sources changed during rebuild check: {changed}")
    print(f"PASS native-only edits rebuild {len(tracked)} Cargo native objects")

if __name__ == "__main__":
    main()
