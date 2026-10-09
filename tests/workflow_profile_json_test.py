#!/usr/bin/env python3
"""Exercise the host-profile writer with fabricated counters, without a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="gjxl-workflow-json-") as directory:
    root = Path(directory)
    output = root / "nested" / "profile.json"
    for _ in range(2):
        result = subprocess.run([sys.argv[1], str(output)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        data = json.loads(output.read_text())
        assert data["schema_version"] == 1
        assert data["scope"] == "cuda-public-workflow-host-profile"
        assert data["timing_semantics"] == "instrumented-workflow-elapsed"
        assert data["substage_work_timing"] == "aggregate-worker-time"
        assert data["gpu_profile_mode"] == "dispatch"
        assert data["distance"] == 1.25 and data["device"] == 'device\n"'
        assert data["adaptive_dc_smoothing"] is False
        assert data["cpu_threads"] == 8 and data["effort"] == 4
        assert data["sample_count"] == 1 and data["warmups"] == 2
        workload = data["workloads"][0]
        assert workload["name"] == 'fixture\n"\\\t\x01'
        assert workload["source_width"] == 1234 and workload["source_height"] == 9
        cuda, cpu = workload["samples"]
        assert cuda["sample_index"] == cpu["sample_index"] == 0
        assert cuda["backend"] == "cuda" and cpu["backend"] == "cpu"
        phases = cuda["phase_nanoseconds"]
        assert len(phases) == 45
        assert all(type(v) is int and v >= 0 for v in phases.values())
        assert phases["total"] == 9007199254740993
        assert phases["input_resident_preparation"] == 101
        assert phases["codestream_encoding"] == 500
        assert phases["codestream_validation"] == 17
        assert phases["codestream_dc_tokenization"] == 31
        assert phases["codestream_ac_tokenization"] == 43
        assert phases["codestream_coefficient_order_work"] == 53
        assert phases["codestream_entropy_optimization"] == 71
        assert phases["codestream_entropy_ans_histogram_build_work"] == 1001
        assert phases["codestream_entropy_prefix_clustering_work"] == 1003
        assert phases["codestream_section_token_write_work"] == 1007
        assert phases["codestream_assembly_output_copy"] == 13
        assert cuda["codestream_total_nanoseconds"] == 490
        assert cuda["serializer_counters"]["coefficient_token_count"] == 12345
        assert cuda["serializer_counters"]["ans_uint_config_candidate_count"] == 29
        assert cuda["selected_balanced_fallback"] is True
        assert cuda["entropy_behavior"] == "rate-optimized"
        assert cuda["entropy_coding"] == {"dc": "ans", "ac": "prefix", "coefficient_order": "prefix"}
        assert cuda["ac_coefficient_bytes"] == 222 and cuda["ac_storage_bytes"] == 333
        assert cpu["entropy_coding"]["coefficient_order"] == "none"
        assert all(v == 0 for v in cpu["phase_nanoseconds"].values())
        assert data["workloads"][1]["samples"] == []
        assert not list(root.rglob("*.tmp-*"))
    blocked = root / "blocked.json"
    blocked.mkdir()
    (blocked / "sentinel").write_text("keep")
    result = subprocess.run([sys.argv[1], str(blocked)], capture_output=True, text=True)
    assert result.returncode == 1
    assert (blocked / "sentinel").read_text() == "keep"
    assert not list(root.rglob("*.tmp-*"))
    saved = output.read_bytes()
    result = subprocess.run([sys.argv[1], str(output / "child")], capture_output=True, text=True)
    assert result.returncode == 1 and output.read_bytes() == saved
    assert not list(root.rglob("*.tmp-*"))
print("Host profile JSON values, timing semantics, locale, escaping and atomic output passed")
