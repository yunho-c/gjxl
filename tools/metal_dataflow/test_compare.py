"""Reject damaged measurement evidence and preserve paired sample boundaries."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from compare import extract, quiet


class ExtractTest(unittest.TestCase):
    def extract(self, samples, profile=False):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.json"
            path.write_text(
                json.dumps(
                    {"sample_count": len(samples), "workloads": [{"samples": samples}]}
                )
            )
            return extract(path, profile, len(samples))

    def workflow(self):
        phases = dict(
            total=100,
            input_preparation=5,
            backend_selection=1,
            quantization_pipeline=60,
            codestream_encoding=30,
            summary_assembly=1,
        )
        return [
            dict(
                sample_index=i,
                backend="metal",
                encoded_bytes=12,
                phase_nanoseconds=phases.copy(),
            )
            for i in range(3)
        ]

    def test_workflow(self):
        self.assertEqual(self.extract(self.workflow())["total"], 0.0001)

    def test_reject_corruption(self):
        cases = []
        value = self.workflow()
        value[1]["sample_index"] = 0
        cases.append(value)
        value = self.workflow()
        value[1]["encoded_bytes"] = 13
        cases.append(value)
        value = self.workflow()
        value[1]["backend"] = "cpu"
        cases.append(value)
        value = self.workflow()
        del value[1]["phase_nanoseconds"]["total"]
        cases.append(value)
        for duration in (-1, float("nan"), float("inf")):
            value = self.workflow()
            value[1]["phase_nanoseconds"]["total"] = duration
            cases.append(value)
        value = self.workflow()
        value[1]["phase_nanoseconds"]["extra"] = 1
        cases.append(value)
        for samples in cases:
            with self.subTest(samples=samples), self.assertRaises(RuntimeError):
                self.extract(samples)

    def test_sum_stages_before_median(self):
        samples = [
            dict(
                sample_index=i,
                capabilities={"stage_boundary": True},
                submissions=[
                    {
                        "stages": [
                            {"stage_id": "aq.epf.pass_1", "gpu_nanoseconds": a},
                            {"stage_id": "aq.epf.pass_2", "gpu_nanoseconds": b},
                        ]
                    }
                ],
            )
            for i, (a, b) in enumerate(((100, 0), (0, 100), (0, 0)))
        ]
        result = self.extract(samples, True)
        self.assertEqual(result["group.epf"], 0.0001)
        self.assertEqual(result["aq.epf.pass_1"] + result["aq.epf.pass_2"], 0)
        broken = copy.deepcopy(samples)
        broken[0]["capabilities"]["stage_boundary"] = False
        with self.assertRaises(RuntimeError):
            self.extract(broken, True)

    def test_fused_epf_uses_the_same_timed_work(self):
        def samples(fused):
            return [dict(sample_index=i, capabilities={"stage_boundary": True},
                submissions=[{"stages": [
                    {"stage_id": "aq.epf.pass_1", "gpu_nanoseconds": 10},
                    *([{"stage_id": "aq.epf_linear.pass_2", "gpu_nanoseconds": a + b}]
                      if fused else [
                          {"stage_id": "aq.epf.pass_2", "gpu_nanoseconds": a},
                          {"stage_id": "aq.opsin_to_linear", "gpu_nanoseconds": b}]),
                ]}]) for i, (a, b) in enumerate(((100, 0), (0, 100), (0, 0)))]
        baseline = self.extract(samples(False), True)
        candidate = self.extract(samples(True), True)
        self.assertEqual(baseline["group.epf_linear"], 0.00011)
        self.assertEqual(baseline["group.epf_linear"], candidate["group.epf_linear"])
        self.assertEqual(baseline["group.all_stages"], candidate["group.all_stages"])

    def test_paused_controller_requires_a_quiet_child(self):
        with patch("compare.subprocess.check_output", return_value=
                   "900001 T /usr/bin/python3\n900002 R /tmp/gjxl_encoding_benchmark\n"):
            with self.assertRaises(RuntimeError):
                quiet()
        with patch("compare.subprocess.check_output", return_value=
                   "900001 T /usr/bin/python3\n900002 Z /tmp/gjxl_encoding_benchmark\n"):
            quiet()
        with patch("compare.subprocess.check_output", return_value=
                   "900001 S /usr/bin/python3\n"), patch(
                   "compare.subprocess.run", return_value=SimpleNamespace(stdout=
                   "python3 tools/scripts/cjxl_runtime_characterization.py run")):
            with self.assertRaises(RuntimeError):
                quiet()


if __name__ == "__main__":
    unittest.main()
