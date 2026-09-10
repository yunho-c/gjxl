#!/usr/bin/env python3
"""CLI checks for per-image batch measurements and raw timing export."""

import argparse
import csv
import io
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import unittest


class ImageBatchBenchmarkTest(unittest.TestCase):
    binary: Path

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.raw = self.root / "samples.csv"

    def image(self, name="image.pfm", width=17, height=13):
        path = self.root / name
        values = [(i * 13 % 97) / 100 for i in range(width * height * 3)]
        path.write_bytes(
            f"PF\n{width} {height}\n-1\n".encode()
            + struct.pack(f"<{len(values)}f", *values)
        )
        return path

    def run_benchmark(self, *arguments):
        return subprocess.run(
            [
                str(self.binary),
                "--batch-sizes",
                "1,4",
                "--samples",
                "2",
                "--warmups",
                "1",
                *map(str, arguments),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def raw_rows(self):
        with self.raw.open(newline="") as stream:
            return list(csv.DictReader(stream))

    def summary_rows(self, result):
        start = result.stdout.index("\nworkload,width,height,batch_size,")
        return list(csv.DictReader(io.StringIO(result.stdout[start + 1 :])))

    def test_single_image_repeated_batches_and_raw_summary_agree(self):
        source = self.image()
        result = self.run_benchmark("--input", source, "--raw-samples", self.raw)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = self.raw_rows()
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {row["encoded_bytes_per_image"] for row in rows},
            {rows[0]["encoded_bytes_per_image"]},
        )
        self.assertGreater(int(rows[0]["encoded_bytes_per_image"]), 0)
        for row in rows:
            self.assertEqual(
                (row["codec"], row["backend"], row["requested_backend"]),
                ("gjxl", "metal", "metal"),
            )
            self.assertEqual(row["aq_mode"], "fully-resident")
            self.assertEqual(row["effort"], "7")
            self.assertAlmostEqual(float(row["distance"]), 1.2, places=6)
            self.assertEqual(row["source"], str(source))
            self.assertEqual((row["width"], row["height"]), ("17", "13"))
            self.assertEqual(row["thread_policy"], "automatic_per_image")
            self.assertEqual(
                row["timing_boundary"], "linear_rgb_to_in_memory_codestream"
            )
            self.assertEqual(
                row["order"], "serial-first" if row["sample"] == "0" else "batch-first"
            )
            self.assertGreater(int(row["serial_ns"]), 0)
            self.assertGreater(int(row["batch_ns"]), 0)
        for summary in self.summary_rows(result):
            samples = [
                row for row in rows if row["batch_size"] == summary["batch_size"]
            ]
            elapsed_ms = (
                statistics.median(int(row["batch_ns"]) for row in samples) / 1e6
            )
            ratios = [int(row["serial_ns"]) / int(row["batch_ns"]) for row in samples]
            self.assertAlmostEqual(
                float(summary["batch_median_ms"]), elapsed_ms, delta=0.00051
            )
            self.assertAlmostEqual(
                float(summary["batch_ms_per_image"]),
                elapsed_ms / int(summary["batch_size"]),
                delta=0.00051,
            )
            self.assertAlmostEqual(
                float(summary["paired_speedup_median"]),
                statistics.median(ratios),
                delta=0.00051,
            )

    def test_directory_sort_dedup_dimensions_and_csv_escaping(self):
        second = self.image('b,"line\n2.pfm')
        first = self.image("a.PFM", 16, 16)
        alias = self.root / "alias.pfm"
        alias.symlink_to(second)
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "invalid.pfm").write_text("must not be visited")
        (self.root / "ignored.png").write_text("not a PFM")
        result = self.run_benchmark(
            "--input",
            self.root,
            "--input",
            first,
            "--input",
            alias,
            "--workload",
            "overridden-by-inputs",
            "--raw-samples",
            self.raw,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = self.raw_rows()
        self.assertEqual(
            [row["source"] for row in rows], [str(first)] * 4 + [str(second)] * 4
        )
        summaries = self.summary_rows(result)
        self.assertEqual(
            [row["workload"] for row in summaries], [str(first)] * 2 + [str(second)] * 2
        )
        self.assertEqual(
            [(row["width"], row["height"]) for row in summaries],
            [("16", "16")] * 2 + [("17", "13")] * 2,
        )

    def test_synthetic_and_explicit_modes_remain_available(self):
        for arguments, backend, mode in (
            ([], "metal", "fully-resident"),
            (["--metal-aq", "maximum-throughput"], "metal", "maximum-throughput"),
            (["--backend", "cpu"], "cpu", "n/a"),
        ):
            with self.subTest(arguments=arguments):
                raw = self.root / f"{backend}-{mode.replace('/', '-')}.csv"
                result = self.run_benchmark(
                    "--workload",
                    "thumbnail_64x64",
                    "--batch-sizes",
                    "1",
                    "--samples",
                    "1",
                    "--raw-samples",
                    raw,
                    *arguments,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                with raw.open() as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(
                    (row["workload"], row["source"]), ("thumbnail_64x64", "")
                )
                self.assertEqual((row["backend"], row["aq_mode"]), (backend, mode))

    def test_rejects_invalid_inputs_and_preserves_existing_output(self):
        source = self.image()
        before = source.read_bytes()
        result = self.run_benchmark("--input", source, "--raw-samples", source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)
        self.assertEqual(source.read_bytes(), before)
        empty = self.root / "empty"
        empty.mkdir()
        invalid = self.root / "invalid.pfm"
        invalid.write_bytes(b"PF\n17 13\n-1\n")
        nonfinite = self.root / "nonfinite.pfm"
        nonfinite.write_bytes(
            b"PF\n1 1\n-1\n" + struct.pack("<fff", float("nan"), 0, 0)
        )
        for args in (
            ("--input", self.root / "missing.pfm"),
            ("--input", empty),
            ("--input", invalid),
            ("--input", nonfinite),
            ("--samples", "0"),
            ("--workload", "unknown"),
            ("--raw-samples", self.root / "missing-parent/out.csv"),
        ):
            with self.subTest(args=args):
                result = self.run_benchmark(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("reference workload=", result.stdout)

    @unittest.skipUnless(shutil.which("magick"), "Optional ImageMagick unavailable")
    def test_existing_normal_image_wrapper(self):
        source = self.root / "photo.png"
        subprocess.run(
            ["magick", "-size", "17x13", "gradient:red-blue", str(source)], check=True
        )
        wrapper = (
            Path(__file__).resolve().parents[1] / "tools/benchmark_encoding_image.py"
        )
        result = subprocess.run(
            [
                sys.executable,
                str(wrapper),
                "--benchmark",
                str(self.binary),
                str(source),
                "--",
                "--batch-sizes",
                "1,4",
                "--samples",
                "1",
                "--warmups",
                "1",
                "--raw-samples",
                str(self.raw),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Prepared 17x13", result.stdout)
        self.assertEqual(len(self.raw_rows()), 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    ImageBatchBenchmarkTest.binary = args.benchmark.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
