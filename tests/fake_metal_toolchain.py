#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Build-graph fixture, not a Metal compiler. Detect duplicate link producers."""
import hashlib
import os
from pathlib import Path
import sys
import time

args = sys.argv[1:]
output = Path(args[args.index("-o") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
if "-c" in args:
    output.write_bytes(b"fake-ir\n")
elif "metal-dsymutil" in args:
    library = Path(args[-1]).read_bytes()
    assert library.endswith(b"COMPLETE\n"), "Symbols read an incomplete library"
    output.write_text(hashlib.sha256(library).hexdigest())
else:
    assert output.name == "gjxl.metallib"
    lock = output.with_suffix(".producer-lock")
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("Concurrent producers for gjxl.metallib") from None
    try:
        count_file = output.with_suffix(".link-count")
        count = int(count_file.read_text()) + 1 if count_file.exists() else 1
        count_file.write_text(str(count))
        payload = (f"complete-payload-{count}\n".encode() * 16384) + b"COMPLETE\n"
        with output.open("wb") as stream:
            stream.write(payload[:128])
            stream.flush()
            # Enlarge the vulnerable window: another recursive Make target
            # must not link the same file or read this partial payload.
            time.sleep(0.2)
            stream.write(payload[128:])
    finally:
        os.close(handle)
        lock.unlink()
