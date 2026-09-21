#!/usr/bin/env python3
"""Guards against accepting silent fallbacks or corrupted resumable evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "tools/ablation/run.py"
SPEC = importlib.util.spec_from_file_location("ablation_run", MODULE)
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class AblationRunnerTest(unittest.TestCase):
    def report(self, variant="production"):
        return {"stage_profile_enabled": False, "ablation": {
            "variant": variant, "counters": {
                "aq_evaluations": 3, "gpu_selector": 1, "deferred_search": 1,
                "gjxl_butteraugli_malta_fixed_f32": 36,
                "gjxl_aq_epf_pass2_linear_direct": 3,
                "gjxl_ac_strategy_dct8_candidate_loss_local": 1,
                "gjxl_dct8_forward_simdgroup_2d_matmul_image": 1}}}

    def test_production_path_contract(self):
        run.validate_audit(self.report(), "production", 8)

    def test_rejects_missing_aq_updates(self):
        report = self.report()
        report["ablation"]["counters"]["aq_evaluations"] = 2
        with self.assertRaisesRegex(ValueError, "AQ evaluations"):
            run.validate_audit(report, "production", 8)

    def test_rejects_profiling_and_unknown_kernels(self):
        report = self.report()
        report["stage_profile_enabled"] = True
        with self.assertRaisesRegex(ValueError, "profiling"):
            run.validate_audit(report, "production", 8)
        report["stage_profile_enabled"] = False
        report["ablation"]["counters"]["unknown_kernel"] = 1
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            run.validate_audit(report, "production", 8)

    def test_rejects_silent_fusion_fallback(self):
        with self.assertRaisesRegex(ValueError, "Split Malta"):
            run.validate_audit(self.report("split-malta"), "split-malta", 8)

    def test_rejects_host_selector_in_resident_arm(self):
        report = self.report()
        report["ablation"]["counters"]["cpu_selector"] = 1
        with self.assertRaisesRegex(ValueError, "CPU selector"):
            run.validate_audit(report, "production", 8)

    def test_rejects_wrong_dct_implementation(self):
        report = self.report("packed-scalar")
        report["ablation"]["counters"] = {
            "aq_evaluations": 3, "cpu_selector": 1,
            "gjxl_aq_gather_transform_pixels": 1,
            "gjxl_aq_scatter_reconstructed_pixels": 1,
            "gjxl_dct8_forward_simdgroup_2d_matmul": 1}
        with self.assertRaisesRegex(ValueError, "Scalar DCT"):
            run.validate_audit(report, "packed-scalar", 8)

    def test_resume_rejects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "output.jxl"
            path.write_bytes(b"before")
            record = {"artifacts": {"output.jxl": run.sha(path)}}
            path.write_bytes(b"after")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                run.verify_record(directory, record)

    def test_new_run_cannot_reuse_existing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            run.validate_output_directory(directory, False)
            (directory / "inputs").mkdir()
            with self.assertRaisesRegex(ValueError, "empty output"):
                run.validate_output_directory(directory, False)
            with self.assertRaisesRegex(ValueError, "existing manifest"):
                run.validate_output_directory(directory, True)

    def test_generated_pfm_and_decoded_extent_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = run.make_trial_inputs(Path(tmp))
            self.assertEqual(run.read_pfm(paths[0])[0], (128, 96))
            self.assertEqual(run.pixel_delta(paths[0], paths[0]),
                             {"rmse": 0., "max_abs": 0.})
            with self.assertRaisesRegex(ValueError, "extent"):
                run.pixel_delta(*paths)

    def test_controlled_environment(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"GJXL_ABLATION_VARIANT": "bad", "MTL_CAPTURE_ENABLED": "1"}):
            env = run.clean_env()
        self.assertNotIn("GJXL_ABLATION_VARIANT", env)
        self.assertNotIn("MTL_CAPTURE_ENABLED", env)

    def test_full_run_refused_on_battery_before_creating_output(self):
        import contextlib
        import io
        import sys
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "must-not-exist"
            argv = ["run.py", "--full", "--output", str(destination),
                    "--input", "unused.pfm", "--decoder", "unused-djxl",
                    "--ssimulacra2", "unused-metric"]
            with patch.object(sys, "argv", argv), patch.object(run, "battery", return_value="Battery Power"), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    run.main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
