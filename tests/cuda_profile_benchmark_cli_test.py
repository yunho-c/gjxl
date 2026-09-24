#!/usr/bin/env python3
"""CUDA diagnostic export and ordinary benchmark mode remain distinct."""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def run(binary, *args):
    return subprocess.run([binary, *map(str, args)], capture_output=True,
                          text=True, timeout=180)


def validate(document, mode, samples):
    assert document["schema_version"] == 4
    assert document["scope"] == "cuda-public-workflow"
    assert document["timestamp_origin"] == "submission-local"
    assert document["mode"] == mode
    assert document["sample_count"] == samples
    assert document["workloads"]
    for workload in document["workloads"]:
        assert workload["source_width"] > 0 and workload["source_height"] > 0
        assert len(workload["samples"]) == samples
        for index, sample in enumerate(workload["samples"]):
            assert sample["sample_index"] == index
            assert all(sample["capabilities"].values())
            assert sample["wall_stages"] and sample["submissions"]
            for wall in sample["wall_stages"]:
                assert wall["stage_id"] and wall["wall_nanoseconds"] >= 0
            for si, submission in enumerate(sample["submissions"]):
                assert submission["submission_index"] == si
                assert submission["submission_id"] and submission["stages"]
                assert submission["command_buffer_gpu_nanoseconds"] > 0
                invocations = {}
                for stage in submission["stages"]:
                    assert stage["stage_id"] and stage["begin_timestamp"] == 0
                    assert stage["end_timestamp"] == stage["gpu_nanoseconds"] > 0
                    assert stage["timestamp_valid"] is True
                    assert stage["dispatches"]
                    previous_end = 0
                    for dispatch in stage["dispatches"]:
                        kernel = dispatch["kernel_id"]
                        assert kernel and dispatch["kind"] == "threadgroups"
                        assert dispatch["invocation"] == invocations.get(kernel, 0)
                        invocations[kernel] = dispatch["invocation"] + 1
                        assert len(dispatch["grid"]) == len(dispatch["threads_per_threadgroup"]) == 3
                        assert all(n > 0 for n in dispatch["grid"] + dispatch["threads_per_threadgroup"])
                        begin, end = dispatch["begin_timestamp"], dispatch["end_timestamp"]
                        if mode == "stage":
                            assert dispatch["timestamp_valid"] is False
                            assert begin == end == dispatch["gpu_nanoseconds"] == 0
                        else:
                            assert dispatch["timestamp_valid"] is True
                            assert previous_end <= begin <= end <= stage["end_timestamp"]
                            assert dispatch["gpu_nanoseconds"] == end - begin
                            previous_end = end


def main():
    binary = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="gjxl-cuda-profile-") as directory:
        root = Path(directory)
        output = root / "nested" / "profile.json"
        base = ["--workload", "synthetic_128x96", "--warmups", "1", "--samples", "2",
                "--cpu-threads", "1"]
        for mode, aq, effort in [
            ("stage", "fully-resident", 1), ("stage", "fully-resident", 7),
            ("dispatch", "fully-resident", 7), ("stage", "throughput", 7),
            ("dispatch", "throughput", 7), ("dispatch", "throughput", 10),
        ]:
            result = run(binary, *base, "--gpu-profile", mode, "--gpu-profile-output", output,
                         "--gpu-aq", aq, "--effort", effort, "--collect-final-score")
            if result.returncode == 77:
                print(result.stdout)
                return 77
            assert result.returncode == 0, result.stdout + result.stderr
            assert "scope=gpu-profile" in result.stdout
            assert "codestream_comparison=exact" in result.stdout
            assert "ratio " not in result.stdout and "timing_ms " not in result.stdout
            document = json.loads(output.read_text())
            validate(document, mode, 2)
            assert document["gpu_aq"] == aq and document["effort"] == effort
            assert document["cpu_threads"] == 1 and document["warmups"] == 1
            assert document["collect_final_score"] is True
            assert document["adaptive_dc_smoothing"] is None
            assert document["dc_quantization"] == "auto"

        # Odd external dimensions and explicit DC overrides reach both encode paths.
        pfm = root / "odd.pfm"
        pfm.write_bytes(b"PF\n17 9\n-1.0\n" + struct.pack("<459f", *[0.25] * 459))
        result = run(binary, "--input", pfm, "--warmups", 0, "--samples", 1,
                     "--gpu-profile", "stage", "--gpu-profile-output", output,
                     "--dc-quantization", "round", "--dc-prediction", "gradient",
                     "--no-adaptive-dc-smoothing", "--effort", 8)
        assert result.returncode == 0, result.stdout + result.stderr
        document = json.loads(output.read_text())
        validate(document, "stage", 1)
        assert document["dc_quantization"] == "round"
        assert document["dc_prediction"] == "gradient"
        assert document["adaptive_dc_smoothing"] is False
        assert document["collect_final_score"] is False
        assert document["workloads"][0]["source_width"] == 17
        assert document["workloads"][0]["source_height"] == 9

        result = run(binary, "--workload", "all", "--effort", 1, "--warmups", 0,
                     "--samples", 1, "--gpu-profile", "stage", "--gpu-profile-output", output)
        assert result.returncode == 0, result.stdout + result.stderr
        document = json.loads(output.read_text())
        validate(document, "stage", 1)
        assert [w["name"] for w in document["workloads"]] == [
            "synthetic_128x96", "padded_1080p", "padded_4k"]

        sentinel = output.read_bytes()
        for args in [
            ["--gpu-profile-output", output],
            ["--gpu-profile", "stage"],
            ["--gpu-profile", "unknown", "--gpu-profile-output", output],
            ["--gpu-profile", "stage", "--gpu-profile-output", output, "--profile-range", "--gpu-only"],
            ["--gpu-profile", "stage", "--gpu-profile-output", output, "--gpu-aq", "maximum-throughput"],
            ["--gpu-profile", "stage", "--gpu-profile-output", output, "--gpu-aq", "exact-coefficients"],
            ["--gpu-profile", "stage", "--gpu-profile-output", output, "--input", root / "missing.pfm"],
        ]:
            result = run(binary, *base, *args)
            assert result.returncode == 1, result.stdout + result.stderr
            assert output.read_bytes() == sentinel
            assert not list(root.rglob("*.tmp-*"))

        # A failed final rename preserves an existing directory and cleans staging.
        blocked = root / "blocked.json"
        blocked.mkdir()
        (blocked / "sentinel").write_text("keep")
        result = run(binary, *base, "--gpu-profile", "stage", "--gpu-profile-output", blocked)
        assert result.returncode == 1, result.stdout + result.stderr
        assert (blocked / "sentinel").read_text() == "keep"
        assert not list(root.rglob("*.tmp-*"))

        result = run(binary, "--workload", "synthetic_128x96", "--warmups", 0,
                     "--samples", 1, "--gpu-aq", "exact-coefficients", "--cpu-threads", 1)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ratio cpu_to_cuda_total" in result.stdout
        assert "comparison=exact" in result.stdout
        assert "scope=gpu-profile" not in result.stdout
    print("CUDA benchmark stage/dispatch exports and failure preservation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
