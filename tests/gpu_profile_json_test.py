#!/usr/bin/env python3
"""Shared Metal/CUDA JSON writer: escaping, locale and atomic replacement."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


with tempfile.TemporaryDirectory(prefix="gjxl-profile-json-") as directory:
    root = Path(directory)
    destination = root / "nested" / "profile.json"
    for _ in range(2):
        result = subprocess.run([sys.argv[1], str(destination)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        document = json.loads(destination.read_text())
        assert document["scope"] == 'fixture\n"\\\b\f\r\t\x01'
        assert document["distance"] == 1.25
        assert document["schema_version"] == 4
        assert document["execution_path"] == "fixture-path\n"
        assert len(document["workloads"]) == 3
        workload = document["workloads"][0]
        assert workload["name"] == "workload\n" and workload["source_width"] == 1234
        sample = workload["samples"][0]
        assert sample["wall_stages"][0]["wall_nanoseconds"] == 12345
        submission = sample["submissions"][0]
        assert submission["submission_id"] == "submission\r"
        stage = submission["stages"][0]
        assert stage["stage_id"] == 'stage"' and stage["group_id"] == "group\\"
        assert stage["dispatches"][0]["kernel_id"] == "kernel\t"
        assert stage["dispatches"][0]["gpu_nanoseconds"] == 11111
        assert stage["timestamp_valid"] is True
        assert stage["dispatches"][0]["timestamp_valid"] is True
        empty = submission["stages"][1]
        assert empty["timestamp_valid"] is False and empty["gpu_nanoseconds"] == 0
        assert empty["dispatches"][0]["kind"] == "indirect_threadgroups"
        assert empty["dispatches"][0]["grid"] == [0, 1, 1]
        assert empty["dispatches"][0]["timestamp_valid"] is False
        assert document["workloads"][1]["samples"][0]["submissions"] == []
        assert document["workloads"][2]["samples"] == []
        assert not list(root.rglob("*.tmp-*"))
    blocked = root / "blocked.json"
    blocked.mkdir()
    (blocked / "sentinel").write_text("keep")
    result = subprocess.run([sys.argv[1], str(blocked)], capture_output=True, text=True)
    assert result.returncode == 1
    assert (blocked / "sentinel").read_text() == "keep"
    assert not list(root.rglob("*.tmp-*"))
    sentinel = destination.read_bytes()
    result = subprocess.run([sys.argv[1], str(destination / "child")], capture_output=True, text=True)
    assert result.returncode == 1
    assert destination.read_bytes() == sentinel
    assert not list(root.rglob("*.tmp-*"))
print("Shared GPU-profile JSON escaping, locale, replacement and failure cleanup passed")
