#!/usr/bin/env python3
"""Parser-only checks: every command exits before creating a CUDA backend."""
from pathlib import Path
import subprocess
import sys
import tempfile

binary = sys.argv[1]
result = subprocess.run([binary, "--help"], capture_output=True, text=True)
assert result.returncode == 0 and "--raw-samples FILE.json" in result.stdout
with tempfile.TemporaryDirectory(prefix="gjxl-cuda-export-cli-") as directory:
    root = Path(directory)
    output = root / "profile.json"
    output.write_text("keep")
    for args in [
        ["--raw-samples"],
        ["--raw-samples", ""],
        ["--raw-samples", output, "--input", output],
        ["--raw-samples", output, "--gpu-profile", "stage", "--gpu-profile-output", output],
        ["--raw-samples", root / "missing" / ".." / "profile.json", "--input", output],
    ]:
        result = subprocess.run([binary, *map(str, args)], capture_output=True, text=True)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "Benchmark error:" in result.stderr
        assert output.read_text() == "keep"
        assert "CUDA unavailable" not in result.stdout and "device=" not in result.stdout
        assert not list(root.rglob("*.tmp-*"))
print("CUDA host-export parser checks passed without GPU initialization")
