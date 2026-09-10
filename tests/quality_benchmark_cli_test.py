#!/usr/bin/env python3
"""Tiny-image quality harness protocol checks; no calibration or corpus jobs."""

import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


class QualityBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gjxl-quality-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.pfm"
        self.input.write_bytes(
            b"PF\n13 11\n-1.0\n" + struct.pack("<429f", *([0.25] * 429))
        )
        self.output = self.root / "output.jxl"
        self.raw = self.root / "raw.json"

    def run_harness(self, *args):
        return subprocess.run(
            [
                str(BENCHMARK),
                "--input",
                str(self.input),
                "--output",
                str(self.output),
                "--raw-samples",
                str(self.raw),
                *args,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_warm_uninstrumented_complete_call_and_retained_output(self):
        result = self.run_harness(
            "--distance",
            "1.2",
            "--effort",
            "3",
            "--num-threads",
            "8",
            "--warmups",
            "1",
            "--samples",
            "2",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.raw.read_text())
        for key, value in {
            "schema_version": 1,
            "encoder": "gjxl",
            "backend": "metal",
            "metal_aq_mode": "fully-resident",
            "stage_profile_enabled": False,
            "timing_semantics": "complete-encode-wall-time",
            "input_width": 13,
            "input_height": 11,
            "validation_encodes": 1,
            "warmups": 1,
            "sample_count": 2,
            "thread_count": 8,
            "collect_final_score": False,
        }.items():
            self.assertEqual(report[key], value, key)
        self.assertEqual([s["sample_index"] for s in report["samples"]], [0, 1])
        for sample in report["samples"]:
            self.assertGreater(sample["elapsed_nanoseconds"], 0)
            self.assertEqual(sample["encoded_bytes"], self.output.stat().st_size)
        self.assertTrue(self.output.read_bytes().startswith(b"\xff\x0a"))
        if DECODER:
            decoded = self.root / "decoded.pfm"
            subprocess.run(
                [
                    str(DECODER),
                    str(self.output),
                    str(decoded),
                    "--color_space=RGB_D65_SRG_Rel_Lin",
                ],
                check=True,
                capture_output=True,
            )
            with decoded.open("rb") as stream:
                self.assertEqual(stream.readline().strip(), b"PF")
                self.assertEqual(stream.readline().strip(), b"13 11")

    def test_invalid_options_do_not_write_outputs(self):
        for args in [
            ("--effort", "0"),
            ("--distance", "nan"),
            ("--samples", "0"),
            ("--warmups", "-1"),
            ("--num-threads", "0"),
            ("--distance", "1.2junk"),
            ("--backend", "cpu"),
        ]:
            with self.subTest(args=args):
                self.assertNotEqual(self.run_harness(*args).returncode, 0)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.raw.exists())

    def test_cannot_overwrite_input(self):
        before = self.input.read_bytes()
        self.assertNotEqual(self.run_harness("--output", str(self.input)).returncode, 0)
        self.assertEqual(before, self.input.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--decoder", type=Path)
    args, remaining = parser.parse_known_args()
    BENCHMARK, DECODER = args.benchmark.resolve(), args.decoder
    unittest.main(argv=[__file__, *remaining])
