#!/usr/bin/env python3
"""Tests for the stock-libjxl versus hybrid comparison tooling."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
TOOL_PATH = REPOSITORY / "tools/libjxl_stock_hybrid.py"
SPEC = importlib.util.spec_from_file_location("libjxl_stock_hybrid", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)

STOCK_BENCHMARK: Path | None = None


def write_pfm(path: Path, width: int = 4, height: int = 3) -> None:
    values = [float((index * 17) % 29) / 28.0 for index in range(width * height * 3)]
    with path.open("wb") as output:
        output.write(f"PF\n{width} {height}\n-1.0\n".encode("ascii"))
        output.write(struct.pack(f"<{len(values)}f", *values))


class ComparisonToolTest(unittest.TestCase):
    def test_process_order_alternates(self) -> None:
        self.assertEqual(TOOL.process_order(0), ("hybrid", "stock"))
        self.assertEqual(TOOL.process_order(1), ("stock", "hybrid"))
        self.assertEqual(TOOL.process_order(2), ("hybrid", "stock"))

    def test_calibration_finds_an_absolute_match(self) -> None:
        selected, evaluations = TOOL.calibrate_distance(
            1.5,
            lambda distance: {
                "butteraugli": 2.0 * distance,
                "encoded_bytes": 100,
                "codestream_sha256": "a" * 64,
            },
            initial=1.0,
            minimum=0.25,
            maximum=2.0,
            tolerance=0.002,
            maximum_evaluations=16,
            maximum_relative_error=0.025,
        )
        self.assertLessEqual(selected["absolute_error"], 0.002)
        self.assertEqual(selected["match_kind"], "absolute-tolerance")
        self.assertGreaterEqual(len(evaluations), 3)

    def test_raw_parsers_keep_wall_time_boundaries(self) -> None:
        arguments = SimpleNamespace(
            threads=8,
            samples=2,
            warmups=1,
            frontend_effort=7,
            libjxl_effort=7,
            gjxl_distance=1.2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            hybrid = directory / "hybrid.json"
            hybrid.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "scope": "hybrid-workflow",
                        "timing": "elapsed-wall-time",
                        "tail_boundary": "warm-context",
                        "frontend": "metal",
                        "frontend_effort": 7,
                        "libjxl_effort": 7,
                        "gpu_aq": "fully-resident",
                        "cpu_threads": 8,
                        "libjxl_threads": 8,
                        "distance": 1.20000005,
                        "warmups": 1,
                        "paired_sample_count": 2,
                        "workloads": [
                            {
                                "decoded_validation": "exact-float-equal",
                                "correctness_outputs": {
                                    "gjxl": {"bytes": 91, "sha256": "a" * 64},
                                    "libjxl": {"bytes": 90, "sha256": "b" * 64},
                                },
                                "pairs": [
                                    {
                                        "gjxl": {
                                            "backend": "gjxl",
                                            "wall_nanoseconds": 230,
                                            "encoded_bytes": 91,
                                            "codestream_sha256": "a" * 64,
                                            "workflow_phase_nanoseconds": {
                                                "input_preparation": 20,
                                                "quantization_pipeline": 100,
                                                "codestream_encoding": 80,
                                            },
                                        },
                                        "libjxl": {
                                            "backend": "libjxl",
                                            "wall_nanoseconds": 200,
                                            "encoded_bytes": 90,
                                            "codestream_sha256": "b" * 64,
                                            "workflow_phase_nanoseconds": {
                                                "input_preparation": 20,
                                                "quantization_pipeline": 100,
                                                "codestream_encoding": 50,
                                            },
                                        },
                                    },
                                    {
                                        "gjxl": {
                                            "backend": "gjxl",
                                            "wall_nanoseconds": 250,
                                            "encoded_bytes": 91,
                                            "codestream_sha256": "a" * 64,
                                            "workflow_phase_nanoseconds": {
                                                "input_preparation": 30,
                                                "quantization_pipeline": 110,
                                                "codestream_encoding": 90,
                                            },
                                        },
                                        "libjxl": {
                                            "backend": "libjxl",
                                            "wall_nanoseconds": 220,
                                            "encoded_bytes": 90,
                                            "codestream_sha256": "b" * 64,
                                            "workflow_phase_nanoseconds": {
                                                "input_preparation": 30,
                                                "quantization_pipeline": 110,
                                                "codestream_encoding": 70,
                                            },
                                        },
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            parsed_hybrid = TOOL.parse_hybrid(hybrid, arguments)
            self.assertEqual(parsed_hybrid["gjxl"]["median_nanoseconds"], 240.0)
            self.assertEqual(parsed_hybrid["hybrid"]["median_nanoseconds"], 210.0)
            self.assertEqual(
                parsed_hybrid["hybrid"]["input_preparation_median_nanoseconds"],
                25.0,
            )
            self.assertEqual(
                parsed_hybrid["hybrid"]["quantization_pipeline_median_nanoseconds"],
                105.0,
            )
            self.assertEqual(parsed_hybrid["hybrid"]["tail_median_nanoseconds"], 60.0)
            self.assertEqual(parsed_hybrid["gjxl"]["tail_median_nanoseconds"], 85.0)

            stock = directory / "stock.json"
            stock.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "timing_semantics": "complete-encode-wall-time",
                        "encoder": "libjxl",
                        "encoder_path": "ordinary-public-api",
                        "thread_count": 8,
                        "effort": 7,
                        "validation_encodes": 1,
                        "requested_distance": 1.25,
                        "warmups": 1,
                        "sample_count": 2,
                        "samples": [
                            {"elapsed_nanoseconds": 100, "encoded_bytes": 110},
                            {"elapsed_nanoseconds": 120, "encoded_bytes": 110},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            parsed_stock = TOOL.parse_stock(stock, arguments, 1.25)
            self.assertEqual(parsed_stock["median_nanoseconds"], 110.0)
            self.assertEqual(parsed_stock["output_bytes"], 110)

    def test_stock_benchmark_writes_a_decodable_raw_schema(self) -> None:
        if STOCK_BENCHMARK is None:
            self.skipTest("stock benchmark was not supplied")
        assert STOCK_BENCHMARK is not None
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image = directory / "input.pfm"
            raw = directory / "raw.json"
            codestream = directory / "output.jxl"
            write_pfm(image)
            result = subprocess.run(
                [
                    str(STOCK_BENCHMARK),
                    "--input",
                    str(image),
                    "--raw-samples",
                    str(raw),
                    "--output",
                    str(codestream),
                    "--distance",
                    "1.0",
                    "--effort",
                    "7",
                    "--num-threads",
                    "1",
                    "--warmups",
                    "0",
                    "--samples",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            document = json.loads(raw.read_text(encoding="utf-8"))
            self.assertEqual(document["schema_version"], 1)
            self.assertEqual(document["encoder"], "libjxl")
            self.assertEqual(document["encoder_path"], "ordinary-public-api")
            self.assertTrue(document["revision"])
            self.assertTrue(document["linked_revision"])
            self.assertEqual(document["validation_encodes"], 1)
            self.assertEqual(document["sample_count"], 1)
            self.assertGreater(document["samples"][0]["elapsed_nanoseconds"], 0)
            self.assertEqual(
                document["samples"][0]["encoded_bytes"], codestream.stat().st_size
            )


def main() -> int:
    global STOCK_BENCHMARK
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stock-benchmark", type=Path)
    arguments, remaining = parser.parse_known_args()
    STOCK_BENCHMARK = arguments.stock_benchmark
    program = unittest.main(argv=[sys.argv[0], *remaining], exit=False)
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
