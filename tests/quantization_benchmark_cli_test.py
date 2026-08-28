#!/usr/bin/env python3
"""CLI coverage for reproducible quantization benchmark samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


PHASES = {
    "total",
    "input_preparation",
    "backend_selection",
    "quantization_pipeline",
    "codestream_encoding",
    "summary_assembly",
    "codestream_validation",
    "codestream_dc_tokenization",
    "codestream_ac_tokenization",
    "codestream_entropy_optimization",
    "codestream_section_writing",
    "codestream_assembly",
}


class QuantizationBenchmarkCliTest(unittest.TestCase):
    benchmark: Path
    metallib: Path

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="gjxl-raw-test-")
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_benchmark(
        self, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        command = [
            str(self.benchmark),
            "--scope",
            "metal-public-workflow",
            "--workload",
            "synthetic_128x96",
            "--implementation",
            "simd",
            "--gpu-aq",
            "fully-resident",
            "--warmups",
            "0",
            "--samples",
            "1",
            *arguments,
        ]
        return subprocess.run(
            command, check=False, capture_output=True, text=True
        )

    def test_external_metallib_writes_integer_raw_samples_atomically(self) -> None:
        destination = self.directory / "samples.json"
        destination.write_text("sentinel", encoding="utf-8")

        result = self.run_benchmark(
            "--validation",
            "metal-only",
            "--metallib",
            str(self.metallib),
            "--raw-samples",
            str(destination),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("codestream=not-compared", result.stdout)
        self.assertNotIn("cpu_bytes=", result.stdout)
        document = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["validation"], "metal-only")
        self.assertEqual(document["sample_count"], 1)
        workload = document["workloads"][0]
        self.assertEqual(workload["codestream_comparison"], "not-compared")
        self.assertEqual(len(workload["samples"]), 1)
        sample = workload["samples"][0]
        self.assertEqual(sample["sample_index"], 0)
        self.assertEqual(sample["backend"], "metal")
        self.assertIsInstance(sample["encoded_bytes"], int)
        self.assertGreater(sample["encoded_bytes"], 2)
        self.assertEqual(set(sample["phase_nanoseconds"]), PHASES)
        for value in sample["phase_nanoseconds"].values():
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)
        self.assertFalse(list(self.directory.glob("samples.json.tmp-*")))

    def test_failed_external_metallib_preserves_existing_output(self) -> None:
        destination = self.directory / "samples.json"
        destination.write_text("sentinel", encoding="utf-8")

        result = self.run_benchmark(
            "--validation",
            "metal-only",
            "--metallib",
            str(self.directory / "missing.metallib"),
            "--raw-samples",
            str(destination),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(destination.read_text(encoding="utf-8"), "sentinel")

    def test_stage_profile_records_ordered_gpu_intervals_and_dispatches(self) -> None:
        destination = self.directory / "gpu-stages.json"

        result = self.run_benchmark(
            "--validation",
            "metal-only",
            "--metallib",
            str(self.metallib),
            "--gpu-profile",
            "stage",
            "--gpu-profile-output",
            str(destination),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["mode"], "stage")
        sample = document["workloads"][0]["samples"][0]
        self.assertTrue(sample["capabilities"]["timestamp_counter"])
        self.assertTrue(sample["capabilities"]["stage_boundary"])
        submission = sample["submissions"][0]
        self.assertGreater(submission["command_buffer_gpu_nanoseconds"], 0)
        stages = submission["stages"]
        self.assertIn("aq.reconstruction", {stage["stage_id"] for stage in stages})
        self.assertIn("aq.epf.pass_1", {stage["stage_id"] for stage in stages})
        self.assertIn(
            "butteraugli.malta.main", {stage["stage_id"] for stage in stages}
        )
        for stage in stages:
            self.assertGreaterEqual(stage["end_timestamp"], stage["begin_timestamp"])
            self.assertEqual(
                stage["gpu_nanoseconds"],
                stage["end_timestamp"] - stage["begin_timestamp"],
            )
            self.assertTrue(stage["dispatches"])

    def test_metal_only_validation_rejects_other_scopes(self) -> None:
        result = subprocess.run(
            [
                str(self.benchmark),
                "--scope",
                "public-workflow",
                "--validation",
                "metal-only",
                "--samples",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Metal-only validation requires metal-public-workflow scope",
            result.stderr,
        )

    def test_default_text_output_retains_cpu_metal_validation(self) -> None:
        result = self.run_benchmark()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("codestream=", result.stdout)
        self.assertIn("cpu_bytes=", result.stdout)
        self.assertIn("gpu_bytes=", result.stdout)
        header = result.stdout.splitlines()[0]
        self.assertNotIn(" validation=", header)
        self.assertNotIn(" metallib=", header)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--metallib", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    QuantizationBenchmarkCliTest.benchmark = arguments.benchmark.resolve()
    QuantizationBenchmarkCliTest.metallib = arguments.metallib.resolve()
    unittest.main(argv=[__file__, *remaining])


if __name__ == "__main__":
    main()
