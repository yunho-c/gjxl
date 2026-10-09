#!/usr/bin/env python3
"""Metal/CUDA host schemas, using fabricated profiles without a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(output, *args):
    return subprocess.run([sys.argv[1], str(output), *args],
                          capture_output=True, text=True, timeout=10)


# These contracts were captured from commit 2491a36's original writers using
# the same fabricated profiles. Metal's writer was extracted verbatim into a
# CPU-only harness. Compare parsed documents so whitespace is not an API.
fixtures = Path(__file__).parent / "fixtures"
with tempfile.TemporaryDirectory(prefix="gjxl-workflow-json-") as directory:
    root = Path(directory)
    documents = {}
    for format_name, version in [("cuda", 1), ("metal", 17)]:
        expected = json.loads(
            (fixtures / f"workflow_profile_{format_name}_v{version}.json").read_text())
        output = root / format_name / "nested" / "profile.json"
        for _ in range(2):
            result = run(output, format_name)
            assert result.returncode == 0, result.stderr
            document = json.loads(output.read_text())
            assert document == expected, format_name
            samples = document["workloads"][0]["samples"]
            phases = samples[0]["phase_nanoseconds"]
            assert len(phases) == 45
            assert all(type(v) is int for v in phases.values())
            assert phases["total"] == 9007199254740993
            assert phases["codestream_entropy_ans_histogram_build_work"] > phases["codestream_encoding"]
            assert all(v == 0 for v in samples[1]["phase_nanoseconds"].values())
            assert document["workloads"][1]["samples"] == []
            assert not list(root.rglob("*.tmp-*"))
        documents[format_name] = document

        blocked = root / format_name / "blocked.json"
        blocked.mkdir()
        (blocked / "sentinel").write_text("keep")
        result = run(blocked, format_name)
        assert result.returncode == 1
        assert (blocked / "sentinel").read_text() == "keep"
        saved = output.read_bytes()
        result = run(output / "child", format_name)
        assert result.returncode == 1 and output.read_bytes() == saved

        # Serialization failures, before publication, must also preserve output.
        result = run(output, "fail")
        assert result.returncode == 1
        assert "Injected serialization failure" in result.stderr
        assert output.read_bytes() == saved
        assert not list(root.rglob("*.tmp-*"))

    cuda_sample = documents["cuda"]["workloads"][0]["samples"][0]
    metal_sample = documents["metal"]["workloads"][0]["samples"][0]
    assert cuda_sample["phase_nanoseconds"] == metal_sample["phase_nanoseconds"]
    assert cuda_sample["entropy_coding"] == metal_sample["entropy_coding"]
    assert metal_sample["final_score"] == 1.2345678901234567
    assert documents["metal"]["workloads"][0]["samples"][1]["final_score"] is None

    # Cover all policy names and the ANS/custom-order variant in both adapters.
    for variant, behavior in enumerate(["balanced", "high-density", "maximum", "rate-optimized"]):
        for format_name in ("cuda", "metal"):
            output = root / f"{format_name}-variant.json"
            result = run(output, format_name, str(variant))
            assert result.returncode == 0, result.stderr
            sample = json.loads(output.read_text())["workloads"][0]["samples"][0]
            assert sample["entropy_behavior"] == behavior
            assert sample["entropy_coding"] == {"dc": "ans", "ac": "ans", "coefficient_order": "ans"}
            if format_name == "metal":
                assert sample["ac_tokenization"]["path"] == ("template" if variant == 2 else "direct")

print("Metal/CUDA host schema compatibility, policy labels, locale and atomic output passed")
