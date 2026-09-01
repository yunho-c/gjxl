#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

"""Focused tests for the no-touch libjxl comparison driver."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import libjxl_comparison as comparison


def write_pfm(path: Path, width: int = 4, height: int = 3) -> None:
    path.write_bytes(
        f"PF\n{width} {height}\n-1.0\n".encode("ascii")
        + struct.pack("<3f", 0.1, 0.2, 0.3) * width * height
    )


class CorpusTest(unittest.TestCase):
    def test_fetch_deduplicates_and_hashes_manifest_sources(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gjxl-corpus-test-") as temporary:
            root = Path(temporary)
            payload = b"pinned source bytes"
            digest = comparison.hashlib.sha256(payload).hexdigest()
            manifest = root / "sources.json"
            common = {
                "path": "shared/source.bin",
                "source": "https://example.invalid/source.bin",
                "sha256": digest,
            }
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "inputs": [
                            {"name": "first", **common},
                            {"name": "second", **common},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "sources"
            response = io.BytesIO(payload)
            response.__enter__ = lambda value: value
            response.__exit__ = lambda *unused: None
            with mock.patch.object(comparison, "urlopen", return_value=response) as get:
                comparison.fetch_corpus_sources(
                    argparse.Namespace(
                        source_manifest=manifest,
                        output=output,
                        timeout=1,
                    )
                )
            self.assertEqual((output / "shared/source.bin").read_bytes(), payload)
            get.assert_called_once()
            with mock.patch.object(comparison, "urlopen") as get:
                comparison.fetch_corpus_sources(
                    argparse.Namespace(
                        source_manifest=manifest,
                        output=output,
                        timeout=1,
                    )
                )
            get.assert_not_called()

    def test_fetch_generates_builtin_workload_once_and_hashes_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gjxl-corpus-test-") as temporary:
            root = Path(temporary)
            payload = b"generated source bytes"
            digest = comparison.hashlib.sha256(payload).hexdigest()
            manifest = root / "sources.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "inputs": [
                            {
                                "name": "stress",
                                "path": "padded.pfm",
                                "source": "GJXL built-in workload",
                                "sha256": digest,
                                "generator": {
                                    "kind": "gjxl-encoding-benchmark-source-v1",
                                    "workload": "padded_1080p",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            benchmark = root / "benchmark"
            benchmark.write_text("fixture", encoding="utf-8")

            def generate(command: list[str], **unused: object) -> object:
                Path(command[-1]).write_bytes(payload)
                return mock.Mock(stdout="")

            output = root / "sources"
            with mock.patch.object(
                comparison, "run_capture", side_effect=generate
            ) as run:
                comparison.fetch_corpus_sources(
                    argparse.Namespace(
                        source_manifest=manifest,
                        output=output,
                        timeout=1,
                        gjxl_benchmark=benchmark,
                    )
                )
            self.assertEqual((output / "padded.pfm").read_bytes(), payload)
            run.assert_called_once()

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

    @unittest.skipUnless(shutil.which("magick"), "ImageMagick is unavailable")
    def test_explicit_crop_and_resize_are_retained(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gjxl-corpus-test-") as temporary:
            root = Path(temporary)
            source = root / "source.ppm"
            source.write_bytes(
                b"P6\n8 6\n255\n"
                + bytes((x * 13 + y * 17) % 256 for y in range(6) for x in range(24))
            )
            source_manifest = root / "sources.json"
            transform = {
                "crop": {"x": 1, "y": 1, "width": 6, "height": 4},
                "resize": {"width": 3, "height": 2},
            }
            source_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "inputs": [
                            {
                                "name": "transformed",
                                "path": source.name,
                                "source": "generated test fixture",
                                "license": "test-only",
                                "source_color": "sRGB",
                                **transform,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "corpus"
            comparison.prepare_corpus(
                argparse.Namespace(
                    source_manifest=source_manifest,
                    output=output,
                    magick="magick",
                    background="white",
                )
            )

            entry = comparison.validate_corpus(output / "manifest.json")[0]
            self.assertEqual((entry["width"], entry["height"]), (3, 2))
            self.assertEqual(entry["source_transform"]["crop"], transform["crop"])
            self.assertEqual(
                entry["source_transform"]["resize"],
                {"width": 3, "height": 2, "filter": "Lanczos"},
            )
            self.assertIn("-crop 6x4+1+1 +repage", entry["conversion"])
            self.assertIn("-filter Lanczos -resize 3x2!", entry["conversion"])

    def test_transform_contract_rejects_implicit_or_invalid_geometry(self) -> None:
        with self.assertRaisesRegex(comparison.ComparisonError, "exactly"):
            comparison.parse_source_transform(
                {"crop": {"x": 0, "y": 0, "width": 4}}
            )
        with self.assertRaisesRegex(comparison.ComparisonError, "positive"):
            comparison.parse_source_transform(
                {"resize": {"width": 0, "height": 2}}
            )

    def test_input_filter_preserves_corpus_order_and_rejects_unknowns(self) -> None:
        entries = [{"name": "first"}, {"name": "second"}, {"name": "third"}]
        self.assertEqual(
            comparison.select_corpus_entries(entries, ["third", "first"]),
            [entries[0], entries[2]],
        )
        with self.assertRaisesRegex(comparison.ComparisonError, "Unknown"):
            comparison.select_corpus_entries(entries, ["missing"])
        with self.assertRaisesRegex(comparison.ComparisonError, "duplicates"):
            comparison.select_corpus_entries(entries, ["first", "first"])

    def test_declared_source_hash_is_enforced(self) -> None:
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
                                "name": "fixture",
                                "path": source.name,
                                "sha256": "0" * 64,
                                "source": "generated test fixture",
                                "license": "test-only",
                                "source_color": "linear-sRGB",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(comparison.ComparisonError, "differs"):
                comparison.prepare_corpus(
                    argparse.Namespace(
                        source_manifest=manifest,
                        output=root / "corpus",
                        magick="magick",
                        background="white",
                    )
                )
            self.assertFalse((root / "corpus").exists())

    def test_source_root_can_be_independent_of_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gjxl-corpus-test-") as temporary:
            root = Path(temporary)
            source_root = root / "source-cache"
            source_root.mkdir()
            source = source_root / "source.pfm"
            write_pfm(source)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            manifest = manifest_dir / "sources.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "inputs": [
                            {
                                "name": "fixture",
                                "path": source.name,
                                "sha256": comparison.sha256_file(source),
                                "source": "generated test fixture",
                                "license": "test-only",
                                "source_color": "linear-sRGB",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "corpus"
            comparison.prepare_corpus(
                argparse.Namespace(
                    source_manifest=manifest,
                    source_root=source_root,
                    output=output,
                    magick="magick",
                    background="white",
                )
            )
            retained = comparison.validate_corpus(output / "manifest.json")[0]
            self.assertEqual(retained["source_path"], str(source.resolve()))


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
                        "distance": 1.0,
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
                        "requested_distance": 0.9,
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
            self.assertEqual(gjxl_row["requested_distance"], 1.0)
            self.assertEqual(libjxl_row["requested_distance"], 0.9)
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


class QualityCalibrationTest(unittest.TestCase):
    def test_distance_search_converges_and_records_every_evaluation(self) -> None:
        calls = []

        def evaluate(distance: float) -> dict[str, float]:
            calls.append(distance)
            return {"butteraugli": 2.0 * distance}

        selected, evaluations = comparison.calibrate_distance(
            1.25,
            evaluate,
            minimum_distance=0.1,
            maximum_distance=2.0,
            initial_distance=1.0,
            tolerance=0.001,
            maximum_evaluations=16,
        )
        self.assertLessEqual(selected["absolute_error"], 0.001)
        self.assertEqual(calls, [item["distance"] for item in evaluations])
        self.assertEqual(len(calls), len(set(calls)))

    def test_distance_search_rejects_an_unbracketed_target(self) -> None:
        with self.assertRaisesRegex(
            comparison.ComparisonError, "did not converge"
        ):
            comparison.calibrate_distance(
                3.0,
                lambda distance: {"butteraugli": distance},
                minimum_distance=0.1,
                maximum_distance=2.0,
                initial_distance=1.0,
                tolerance=0.01,
                maximum_evaluations=8,
            )

    def test_distance_search_accepts_a_quantized_policy_transition(self) -> None:
        selected, _ = comparison.calibrate_distance(
            1.413,
            lambda distance: {
                "butteraugli": 1.401 if distance < 0.999 else 1.446
            },
            minimum_distance=0.1,
            maximum_distance=2.0,
            initial_distance=1.0,
            tolerance=0.015,
            maximum_evaluations=12,
        )
        self.assertEqual(selected["butteraugli"], 1.401)
        self.assertLessEqual(selected["absolute_error"], 0.015)

    def test_distance_search_labels_a_boundary_limited_match(self) -> None:
        selected, _ = comparison.calibrate_distance(
            3.813,
            lambda distance: {
                "butteraugli": 3.89 if distance <= 0.05 else 3.95
            },
            minimum_distance=0.05,
            maximum_distance=2.0,
            initial_distance=1.0,
            tolerance=0.015,
            maximum_evaluations=12,
            maximum_relative_error=0.025,
        )
        self.assertEqual(selected["distance"], 0.05)
        self.assertEqual(
            selected["match_kind"], "boundary-limited-relative-tolerance"
        )
        self.assertLessEqual(selected["relative_error"], 0.025)

    def test_distance_search_finds_a_nonmonotonic_interior_bracket(self) -> None:
        selected, evaluations = comparison.calibrate_distance(
            1.95,
            lambda distance: {
                "butteraugli": 1.92 + (distance - 1.25) ** 2
            },
            minimum_distance=0.05,
            maximum_distance=2.0,
            initial_distance=1.0,
            tolerance=0.005,
            maximum_evaluations=12,
            maximum_relative_error=0.0,
        )
        self.assertLessEqual(selected["absolute_error"], 0.005)
        self.assertTrue(any(item["distance"] > 1.0 for item in evaluations))

    def test_nominal_targets_are_bound_to_the_corpus(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gjxl-quality-test-") as temporary:
            root = Path(temporary)
            corpus = root / "corpus.json"
            corpus.write_text('{"schema_version": 1}\n', encoding="utf-8")
            run = root / "nominal"
            run.mkdir()
            (run / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "corpus_manifest_sha256": comparison.sha256_file(corpus),
                        "libjxl_revision": comparison.PINNED_LIBJXL_REVISION,
                        "gjxl_revision": "a" * 40,
                        "parameters": {"distance": 1.0, "effort": 7},
                    }
                ),
                encoding="utf-8",
            )
            summary = run / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "validations": [
                            {
                                "input": "fixture",
                                "encoder": "gjxl",
                                "configuration": "serial",
                                "butteraugli": 1.25,
                            },
                            {
                                "input": "fixture",
                                "encoder": "gjxl",
                                "configuration": "production",
                                "butteraugli": 1.25,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            targets, metadata = comparison.load_nominal_quality_targets(
                summary, corpus, [{"name": "fixture"}]
            )
            self.assertEqual(targets, {"fixture": 1.25})
            self.assertEqual(metadata["target_gjxl_distance"], 1.0)
            self.assertEqual(metadata["effort"], 7)

    def test_quality_map_requires_complete_in_tolerance_corpus(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gjxl-quality-test-") as temporary:
            root = Path(temporary)
            corpus = root / "corpus.json"
            corpus.write_text('{"schema_version": 1}\n', encoding="utf-8")
            calibration = root / "calibration.json"
            document = {
                "schema_version": comparison.CALIBRATION_SCHEMA_VERSION,
                "kind": "libjxl-butteraugli-calibration",
                "libjxl_revision": comparison.PINNED_LIBJXL_REVISION,
                "corpus_manifest_sha256": comparison.sha256_file(corpus),
                "parameters": {
                    "target_gjxl_distance": 1.0,
                    "effort": 7,
                    "tolerance": 0.01,
                    "maximum_relative_error": 0.01,
                },
                "inputs": [
                    {
                        "input": "fixture",
                        "canonical_sha256": "fixture-sha256",
                        "target_gjxl_distance": 1.0,
                        "target_butteraugli": 1.25,
                        "achieved_butteraugli": 1.245,
                        "libjxl_distance": 0.875,
                        "absolute_error": 0.005,
                        "relative_error": 0.004,
                        "match_kind": "within-absolute-tolerance",
                    }
                ],
            }
            calibration.write_text(json.dumps(document), encoding="utf-8")
            distances, metadata = comparison.load_quality_calibration(
                calibration,
                corpus,
                [{"name": "fixture", "canonical_sha256": "fixture-sha256"}],
                target_distance=1.0,
                effort=7,
            )
            self.assertEqual(distances, {"fixture": 0.875})
            self.assertEqual(metadata["tolerance"], 0.01)

            document["inputs"][0]["achieved_butteraugli"] = 1.23
            document["inputs"][0]["absolute_error"] = 0.02
            document["inputs"][0]["relative_error"] = 0.016
            document["inputs"][0]["match_kind"] = (
                "boundary-limited-relative-tolerance"
            )
            calibration.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                comparison.ComparisonError, "relative match is inconsistent"
            ):
                comparison.load_quality_calibration(
                    calibration,
                    corpus,
                    [{"name": "fixture", "canonical_sha256": "fixture-sha256"}],
                    target_distance=1.0,
                    effort=7,
                )


if __name__ == "__main__":
    unittest.main()
