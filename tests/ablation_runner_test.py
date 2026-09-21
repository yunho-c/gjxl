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
STORAGE_SPEC = importlib.util.spec_from_file_location("ablation_storage", MODULE.with_name("storage.py"))
storage = importlib.util.module_from_spec(STORAGE_SPEC)
try:
    STORAGE_SPEC.loader.exec_module(storage)
except ImportError:
    storage = None


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


@unittest.skipIf(storage is None, "Compact corpus collection requires NumPy")
class CompactStoreTest(unittest.TestCase):
    def test_numpy_error_matches_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = run.make_trial_inputs(Path(tmp))[0]
            altered = Path(tmp) / "altered.pfm"
            data = bytearray(source.read_bytes())
            import struct
            data[-4:] = struct.pack("<f", .42)
            altered.write_bytes(data)
            expected = run.pixel_delta(source, altered)
            actual = storage.pixel_delta(source, altered)
            self.assertAlmostEqual(expected["rmse"], actual["rmse"], places=14)
            self.assertEqual(expected["max_abs"], actual["max_abs"])

    def test_nonfinite_pixels_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = run.make_trial_inputs(Path(tmp))[0]
            import struct
            data = source.read_bytes()
            source.write_bytes(data[:-4] + struct.pack("<f", float("nan")))
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                storage.pixel_delta(source, source)

    def test_identical_codestreams_share_validation_and_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = run.make_trial_inputs(root)[0]
            decoder, metric = root / "decoder", root / "metric"
            decoder.write_bytes(b"decoder identity")
            metric.write_bytes(b"metric identity")
            calls = []
            def command(args, **kwargs):
                calls.append(args)
                if args[0] == decoder:
                    Path(args[2]).write_bytes(source.read_bytes())
                    return ""
                return "100"
            store = storage.CompactStore(root, decoder, metric, command)
            results, paths = [], []
            for name in ("first", "second"):
                directory = root / name
                directory.mkdir()
                (directory / "output.jxl").write_bytes(b"same codestream")
                results.append(store.validate(source, run.sha(source), directory))
                paths.append(directory / "output.jxl")
            self.assertEqual(len(calls), 2)  # one decode and one quality score
            self.assertEqual(results[0], results[1])
            self.assertEqual(paths[0].stat().st_ino, paths[1].stat().st_ino)
            self.assertEqual(list((root / "scratch").iterdir()), [])
            cache = root / results[0]["validation_cache"]
            cached = json.loads(cache.read_text())
            cached["identity"]["tools"]["decoder"] = "changed"
            cache.write_text(json.dumps(cached))
            with self.assertRaisesRegex(ValueError, "cache identity"):
                store.validate(source, run.sha(source), root / "first")


if __name__ == "__main__":
    unittest.main()
