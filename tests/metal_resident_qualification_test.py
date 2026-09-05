#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Portable tests of qualification decisions, independent of Metal availability."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import metal_resident_qualification as q
from metal_qualification_support import calibrate_distance, ComparisonError


def row():
    return {
        "id": "image/e7/1",
        "status": "pass",
        "failures": [],
        "cpu": {"butteraugli": 1.0, "bytes": 1000},
        "metal": {"butteraugli": 1.0, "bytes": 1000},
        "cpu_size_ratio": 1.0,
        "libjxl_size_ratio": 1.0,
    }


class QualificationTests(unittest.TestCase):
    def test_score_zero_and_positive(self):
        self.assertEqual(q.parse_score("0\nadditional diagnostics"), 0)
        self.assertEqual(q.parse_score("1.25\n"), 1.25)

    def test_invalid_scores(self):
        for value in ("", "NaN", "inf", "-inf", "-0.1", "text"):
            with self.subTest(value=value), self.assertRaises(ComparisonError):
                q.parse_score(value)

    def test_backend_must_be_actual_requested_mode(self):
        q.check_backend(
            "Encoded using Metal fully-resident AQ and balanced entropy.", "metal"
        )
        q.check_backend("Encoded using CPU and balanced entropy.", "cpu")
        for output in (
            "Encoded using CPU",
            "Encoded using Metal exact-coefficient AQ",
            "",
        ):
            with self.subTest(output=output), self.assertRaises(ComparisonError):
                q.check_backend(output, "metal")

    def test_initial_limits_are_per_case_and_inclusive(self):
        r = row()
        r["metal"]["butteraugli"] = 1.1
        r["cpu_size_ratio"] = 1.1
        self.assertEqual(q.acceptance(r), [])
        r["metal"]["butteraugli"] = math.nextafter(1.1, math.inf)
        r["cpu_size_ratio"] = math.nextafter(1.1, math.inf)
        self.assertEqual(len(q.acceptance(r)), 2)

    def test_absolute_quality_floor_near_zero(self):
        self.assertEqual(q.quality_limit(0), 0.1)
        self.assertEqual(q.quality_limit(0.1), 0.2)
        self.assertEqual(q.quality_limit(10), 11)

    def test_libjxl_is_not_initial_superiority_requirement(self):
        r = row()
        r["libjxl_size_ratio"] = 2.5
        self.assertEqual(q.acceptance(r), [])

    def test_dynamic_matching_cannot_hide_fixed_distance_regression(self):
        r = row()
        r["metal"]["butteraugli"] = 1.2
        r["cpu"]["butteraugli"] = 1.2
        self.assertEqual(q.acceptance(r), [])
        self.assertIn(
            "fixed-distance quality regressed from baseline", q.acceptance(r, row())
        )

    def test_baseline_bytes_and_libjxl_ratio_are_independent(self):
        r = row()
        r["metal"]["bytes"] = 1101
        r["libjxl_size_ratio"] = 1.101
        self.assertEqual(len(q.acceptance(r, row())), 2)

    def test_baseline_requires_exact_configuration_and_complete_cases(self):
        contract = {
            "metric": q.METRIC,
            "inputs": ["input-hash"],
            "reference_tools": ["tool-hash"],
        }
        baseline = {
            "kind": "metal-resident-baseline",
            "schema_version": q.SCHEMA,
            "contract": contract,
            "passed": True,
            "rows": [row()],
        }
        expected = [row()["id"]]
        self.assertEqual(
            q.validate_baseline(baseline, contract, expected), {row()["id"]: row()}
        )
        for field in ("metric", "inputs", "reference_tools"):
            changed = copy.deepcopy(contract)
            changed[field] = "changed"
            with self.assertRaises(ComparisonError):
                q.validate_baseline(baseline, changed, expected)
        for rows in (
            [],
            [row(), row()],
            [{**row(), "status": "incomplete"}],
            [{**row(), "failures": ["regression"]}],
        ):
            with self.subTest(rows=rows), self.assertRaises(ComparisonError):
                q.validate_baseline({**baseline, "rows": rows}, contract, expected)

    def test_candidate_and_failed_reports_cannot_be_baselines(self):
        for kind, passed in (
            ("metal-resident-qualification", True),
            ("metal-resident-baseline", False),
        ):
            with self.assertRaises(ComparisonError):
                q.validate_baseline(
                    {
                        "kind": kind,
                        "schema_version": q.SCHEMA,
                        "contract": {},
                        "passed": passed,
                        "rows": [],
                    },
                    {},
                    [],
                )

    def test_same_case_id_is_stable_across_json_roundtrip(self):
        d = 2.499
        self.assertEqual(
            q.case_id("gradient", "e7", d),
            q.case_id("gradient", "e7", json.loads(json.dumps(d))),
        )

    def test_nonfinite_baseline_cannot_disable_comparisons(self):
        for value in (math.nan, math.inf, -1, True):
            r = row()
            r["metal"]["butteraugli"] = value
            with self.subTest(value=value), self.assertRaises(ComparisonError):
                q.validate_baseline(
                    {
                        "kind": "metal-resident-baseline",
                        "schema_version": q.SCHEMA,
                        "contract": {},
                        "passed": True,
                        "rows": [r],
                    },
                    {},
                    [r["id"]],
                )

    def test_matrix_contains_interiors_and_threshold_neighbors(self):
        self.assertEqual(len(q.POLICIES), 11)
        self.assertEqual(len(q.FULL_DISTANCES), 32)
        for d in q.COMPACT_DISTANCES:
            self.assertIn(d, q.FULL_DISTANCES)

    def test_corpus_cannot_silently_replace_or_omit_a_fixture(self):
        locked = json.loads(
            (q.ROOT / "tests/metal_qualification/inputs.json").read_text()
        )
        entries = [e for e in locked["inputs"] if e["name"] in locked["compact"]]
        q.validate_corpus_entries(entries, "compact")
        with self.assertRaises(ComparisonError):
            q.validate_corpus_entries(entries[:-1], "compact")
        changed = copy.deepcopy(entries)
        changed[0]["sha256"] = "0" * 64
        with self.assertRaises(ComparisonError):
            q.validate_corpus_entries(changed, "compact")

    def test_baseline_inside_report_directory_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            baseline = path / "report.json"
            baseline.write_text("sentinel")
            with self.assertRaisesRegex(ComparisonError, "outside the report"):
                q.run(argparse.Namespace(baseline=baseline, output=path))
            self.assertEqual(baseline.read_text(), "sentinel")
        for d in (0.999, 1.001, 1.199, 1.201, 2.499, 2.5, 2.501):
            self.assertIn(d, q.FULL_DISTANCES)

    def test_pfm_endian_and_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pfm"
            q.write_pfm(path, 1, 2, [1, 2, 3, 4, 5, 6])
            self.assertEqual(list(q.read_pfm(path)[2]), [4, 5, 6, 1, 2, 3])
            path.write_bytes(b"PF\n1 1\n1.0\n" + struct.pack(">3f", 0.5, 0.25, 0.75))
            self.assertEqual(list(q.read_pfm(path)[2]), [0.5, 0.25, 0.75])

    def test_pfm_rejects_nonfinite_and_truncated_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pfm"
            for values in ((math.nan, 0, 0), (math.inf, -math.inf, 0), (1, 2)):
                path.write_bytes(
                    b"PF\n1 1\n-1\n" + struct.pack(f"<{len(values)}f", *values)
                )
                with self.assertRaises(ComparisonError):
                    q.read_pfm(path)

    def test_large_finite_pfm_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pfm"
            q.write_pfm(path, 1, 1, [3e38, 3e38, -3e38])
            self.assertEqual(q.read_pfm(path)[:2], (1, 1))

    def calibrate(self, target, function, **kwargs):
        return calibrate_distance(
            target,
            lambda d: {"butteraugli": function(d), "bytes": 100},
            minimum_distance=0.03,
            maximum_distance=15,
            initial_distance=1,
            tolerance=0.015,
            maximum_relative_error=0.02,
            maximum_evaluations=24,
            **kwargs,
        )

    def test_calibration_linear(self):
        best, evaluations = self.calibrate(2.3, lambda d: d * 2)
        self.assertLessEqual(abs(best["butteraugli"] - 2.3), 0.015)
        self.assertLessEqual(len(evaluations), 24)
        self.assertEqual(len(evaluations), len({e["distance"] for e in evaluations}))

    def test_calibration_handles_nonmonotonic_bracket(self):
        best, _ = self.calibrate(3, lambda d: 1 if d < 2 else (3 if d < 7 else 0.5))
        self.assertEqual(best["butteraugli"], 3)

    def test_unmatchable_discontinuity_is_incomplete(self):
        with self.assertRaisesRegex(ComparisonError, "did not converge"):
            self.calibrate(1.5, lambda d: 1 if d < 1 else 2)

    def test_calibration_zero_score_uses_absolute_tolerance(self):
        best, _ = self.calibrate(0, lambda d: 0)
        self.assertEqual(best["match_kind"], "within-absolute-tolerance")

    def test_calibration_rejects_nan_score(self):
        with self.assertRaises(ComparisonError):
            self.calibrate(1, lambda d: math.nan)

    def test_alias_mismatch_is_a_failure(self):
        class Fake:
            def evaluate(self, *args, **kwargs):
                return {"sha256": "wrong"}

        with self.assertRaisesRegex(ComparisonError, "alias differs"):
            q.check_aliases(
                Fake(), {}, "e10", 1.2, {"sha256": "right"}, {"sha256": "right"}
            )


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        args = argparse.Namespace(
            cache=self.root / "cache",
            encoder=Path("gjxl"),
            cjxl=Path("cjxl"),
            djxl=Path("djxl"),
            butteraugli=Path("metric"),
        )
        self.evaluator = q.Evaluator(args, {"binary_sha256": "build-a"})
        self.entry = {
            "resolved": str(self.root / "input.pfm"),
            "sha256": "input-a",
            "width": 2,
            "height": 2,
        }
        self.calls = []
        self.wrong_backend = False
        self.different_repeat = False
        self.wrong_dimensions = False
        self.nonfinite = False

        def command(argv, directory, label):
            self.calls.append(label)
            if label in ("encode", "repeat"):
                data = (
                    b"different"
                    if label == "repeat" and self.different_repeat
                    else b"codestream"
                )
                output = Path(argv[-1]) if argv[0] == "gjxl" else Path(argv[2])
                output.write_bytes(data)
                return (
                    "Encoded using CPU"
                    if self.wrong_backend or "cpu" in argv
                    else "Encoded using Metal fully-resident AQ"
                )
            if label == "decode":
                width = 3 if self.wrong_dimensions else 2
                q.write_pfm(
                    Path(argv[2]),
                    width,
                    2,
                    [math.nan if self.nonfinite else 0.25] * (width * 2 * 3),
                )
                return ""
            if label == "metric":
                return "1.25\n"
            raise AssertionError(label)

        self.evaluator.command = command

    def tearDown(self):
        self.temp.cleanup()

    def evaluate(self, **kwargs):
        return self.evaluator.evaluate(
            self.entry, "metal", "e7", kwargs.get("distance", 1.0)
        )

    def test_real_evaluation_sequence_and_cache_reuse(self):
        result = self.evaluate()
        self.assertEqual(self.calls, ["encode", "repeat", "decode", "metric"])
        self.assertEqual(self.evaluate(), result)
        self.assertEqual(len(self.calls), 4)
        self.evaluate(distance=1.1)
        self.assertEqual(self.calls[4:], ["encode", "repeat", "decode"])

    def test_different_build_does_not_reuse_evaluation(self):
        self.evaluate()
        other = q.Evaluator(self.evaluator.args, {"binary_sha256": "build-b"})
        other.command = self.evaluator.command
        other.evaluate(self.entry, "metal", "e7", 1.0)
        self.assertEqual(self.calls.count("encode"), 2)

    def test_fallback_is_rejected_before_scoring(self):
        self.wrong_backend = True
        with self.assertRaisesRegex(ComparisonError, "requested metal"):
            self.evaluate()
        self.assertEqual(self.calls, ["encode"])

    def test_nondeterminism_is_not_cached_as_success(self):
        self.different_repeat = True
        with self.assertRaisesRegex(ComparisonError, "not repeatable"):
            self.evaluate()
        self.assertEqual(list(self.evaluator.cache.rglob("result.json")), [])

    def test_decoded_dimensions_and_pixels_are_checked(self):
        self.wrong_dimensions = True
        with self.assertRaisesRegex(ComparisonError, "dimensions changed"):
            self.evaluate()
        self.wrong_dimensions = False
        self.nonfinite = True
        with self.assertRaisesRegex(ComparisonError, "Non-finite"):
            self.evaluate()
        self.assertNotIn("metric", self.calls)

    def test_corrupt_cached_measurement_is_rejected(self):
        result = self.evaluate()
        path = Path(result["artifact"]) / "result.json"
        result["butteraugli"] = math.nan
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ComparisonError, "Invalid cached"):
            self.evaluate()


if __name__ == "__main__":
    unittest.main()
