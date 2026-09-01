#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

"""Focused tests for the no-touch libjxl comparison driver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import libjxl_comparison as comparison


def write_pfm(path: Path, width: int = 4, height: int = 3) -> None:
    path.write_bytes(
        f"PF\n{width} {height}\n-1.0\n".encode("ascii")
        + struct.pack("<3f", 0.1, 0.2, 0.3) * width * height
    )


class CorpusTest(unittest.TestCase):
    def test_identity_linear_pfm_is_hashed_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gjxl-corpus-test-") as temporary:
            root = Path(temporary)
            source = root / "source.pfm"
            write_pfm(source)
            source_manifest = root / "sources.json"
            source_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "inputs": [
                            {
                                "name": "fixture",
                                "path": source.name,
                                "source": "generated test fixture",
                                "license": "test-only",
                                "source_color": "linear-sRGB",
                                "category": "synthetic",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "corpus"
            args = argparse.Namespace(
                source_manifest=source_manifest,
                output=output,
                magick="magick",
                background="white",
            )
            comparison.prepare_corpus(args)

            manifest_path = output / "manifest.json"
            entries = comparison.validate_corpus(manifest_path)
            self.assertEqual(len(entries), 1)
            self.assertEqual((entries[0]["width"], entries[0]["height"]), (4, 3))
            self.assertEqual(entries[0]["category"], "synthetic")
            self.assertEqual(
                entries[0]["canonical_sha256"],
                comparison.sha256_file(output / "canonical/fixture.pfm"),
            )
            with self.assertRaisesRegex(
                comparison.ComparisonError, "already exists"
            ):
                comparison.prepare_corpus(args)

    def test_identity_pfm_requires_declared_linear_pixels(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gjxl-corpus-test-") as temporary:
            root = Path(temporary)
            source = root / "source.pfm"
            write_pfm(source)
            manifest = root / "sources.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "inputs": [
                            {
                                "name": "nonlinear",
                                "path": source.name,
                                "source": "fixture",
                                "license": "test-only",
                                "source_color": "sRGB",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                comparison.ComparisonError, "must declare linear-sRGB"
            ):
                comparison.prepare_corpus(
                    argparse.Namespace(
                        source_manifest=manifest,
                        output=root / "corpus",
                        magick="magick",
                        background="white",
                    )
                )
            self.assertFalse((root / "corpus").exists())


class RawSummaryTest(unittest.TestCase):
    def test_process_order_alternates_between_pairs(self) -> None:
        self.assertEqual(comparison.encoder_order(0), ("gjxl", "libjxl"))
        self.assertEqual(comparison.encoder_order(1), ("libjxl", "gjxl"))
        self.assertEqual(comparison.encoder_order(2), ("gjxl", "libjxl"))

    def test_native_schemas_produce_normalized_process_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gjxl-summary-test-") as temporary:
            root = Path(temporary)
            entry = {"name": "fixture", "width": 100, "height": 50}
            gjxl = root / "gjxl.json"
            gjxl.write_text(
                json.dumps(
                    {
                        "schema_version": 10,
                        "serializer_workers": 1,
                        "workloads": [
                            {
                                "samples": [
                                    {
                                        "backend": "metal",
                                        "encoded_bytes": 1000,
                                        "phase_nanoseconds": {
                                            "total": 3_000_000,
                                            "codestream_encoding": 1_000_000,
                                        },
                                    },
                                    {
                                        "backend": "metal",
                                        "encoded_bytes": 1000,
                                        "phase_nanoseconds": {
                                            "total": 5_000_000,
                                            "codestream_encoding": 2_000_000,
                                        },
                                    },
                                ]
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            libjxl = root / "libjxl.json"
            libjxl.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "encoder": "libjxl",
                        "revision": comparison.PINNED_LIBJXL_REVISION,
                        "thread_count": 0,
                        "samples": [
                            {"elapsed_nanoseconds": 2_000_000, "encoded_bytes": 800},
                            {"elapsed_nanoseconds": 4_000_000, "encoded_bytes": 800},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            gjxl_row = comparison.median_process_row(
                "gjxl", "serial", entry, gjxl
            )
            libjxl_row = comparison.median_process_row(
                "libjxl", "serial", entry, libjxl
            )
            self.assertEqual(gjxl_row["median_nanoseconds"], 4_000_000)
            self.assertEqual(gjxl_row["milliseconds_per_megapixel"], 800.0)
            self.assertEqual(gjxl_row["nanoseconds_per_pixel"], 800.0)
            self.assertEqual(
                gjxl_row["codestream_median_nanoseconds"], 1_500_000
            )
            self.assertEqual(libjxl_row["median_nanoseconds"], 3_000_000)
            self.assertIsNone(libjxl_row["codestream_median_nanoseconds"])
            aggregate = comparison.aggregate_rows(
                [gjxl_row, dict(gjxl_row, median_nanoseconds=6_000_000)]
            )[0]
            self.assertEqual(
                aggregate["median_of_process_medians_nanoseconds"], 5_000_000
            )
            self.assertEqual(
                aggregate["process_median_range_nanoseconds"],
                [4_000_000, 6_000_000],
            )


if __name__ == "__main__":
    unittest.main()
