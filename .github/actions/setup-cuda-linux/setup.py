#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Install the pinned Linux CUDA build components without changing the driver."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request


def download(url, destination, expected_hash):
    if not destination.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError(f"Checksum mismatch: {destination}")


def copy_component(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        target = destination / entry.name
        if entry.is_symlink():
            if target.is_symlink() and target.readlink() == entry.readlink():
                continue
            target.symlink_to(entry.readlink())
        elif entry.is_dir():
            copy_component(entry, target)
        else:
            shutil.copy2(entry, target)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: setup.py INSTALL_ROOT")
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = "https://developer.download.nvidia.com/compute/cuda/redist/"
    manifest = root / "redistrib_12.6.3.json"
    download(base + manifest.name, manifest,
             "9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf")
    components = json.loads(manifest.read_text())
    prefix = root / "toolkit"
    for name in ("cuda_nvcc", "cuda_cudart", "cuda_cccl", "cuda_profiler_api"):
        component = components[name]
        record = component["linux-x86_64"]
        archive = root / Path(record["relative_path"]).name
        download(base + record["relative_path"], archive, record["sha256"])
        extracted = root / name
        extracted.mkdir(exist_ok=True)
        with tarfile.open(archive) as source:
            source.extractall(extracted, filter="data")
        [directory] = list(extracted.iterdir())
        copy_component(directory, prefix)
        print(f"Verified {name} {component['version']}", flush=True)
    # Redistributable archives use lib; nvcc's Linux profile expects lib64.
    if not (prefix / "lib64").exists():
        (prefix / "lib64").symlink_to("lib", target_is_directory=True)
    print(prefix, flush=True)


if __name__ == "__main__":
    main()
