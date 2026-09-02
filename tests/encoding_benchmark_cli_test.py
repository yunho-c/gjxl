#!/usr/bin/env python3
"""CLI coverage for reproducible encoding benchmark samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
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
    "codestream_block_context_map_work",
    "codestream_coefficient_order_work",
    "codestream_coefficient_tokenization_work",
    "codestream_coefficient_context_materialization_work",
    "codestream_entropy_optimization",
    "codestream_entropy_prefix_histogram_build_work",
    "codestream_entropy_prefix_histogram_cost_work",
    "codestream_entropy_prefix_clustering_work",
    "codestream_entropy_prefix_code_build_work",
    "codestream_entropy_prefix_value_collection_work",
    "codestream_entropy_prefix_config_search_work",
    "codestream_entropy_prefix_exact_measurement_work",
    "codestream_entropy_ans_prefix_validation_work",
    "codestream_entropy_ans_value_collection_work",
    "codestream_entropy_ans_value_aggregation_work",
    "codestream_entropy_ans_prepared_value_validation_work",
    "codestream_entropy_ans_uint_config_work",
    "codestream_entropy_ans_histogram_build_work",
    "codestream_entropy_ans_model_build_work",
    "codestream_entropy_ans_token_cost_work",
    "codestream_entropy_selection_work",
    "codestream_section_writing",
    "codestream_section_model_and_header_work",
    "codestream_section_token_write_work",
    "codestream_section_candidate_measure_work",
    "codestream_assembly",
    "codestream_assembly_candidate_selection",
    "codestream_assembly_section_size",
    "codestream_assembly_frame_header",
    "codestream_assembly_toc_and_sections",
    "codestream_assembly_output_copy",
}

ELIMINATED_WORK_PHASES = {
    "codestream_entropy_ans_value_collection_work",
    "codestream_entropy_ans_value_aggregation_work",
}


class EncodingBenchmarkCliTest(unittest.TestCase):
    benchmark: Path
    metallib: Path
    libjxl_tail_enabled: bool

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

    def run_tail_benchmark(
        self, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        command = [
            str(self.benchmark),
            "--scope",
            "codestream-tail",
            "--workload",
            "synthetic_128x96",
            "--tail-frontend",
            "cpu",
            "--warmups",
            "0",
            "--samples",
            "1",
            *arguments,
        ]
        return subprocess.run(
            command, check=False, capture_output=True, text=True
        )

    def test_tail_scope_validates_backend_names_and_effort_bounds(self) -> None:
        invalid_backend = self.run_tail_benchmark(
            "--tail-backends", "mystery"
        )
        self.assertNotEqual(invalid_backend.returncode, 0)
        self.assertIn("Unknown tail backend selection", invalid_backend.stderr)

        for effort in ("0", "11"):
            result = self.run_tail_benchmark(
                "--tail-backends", "gjxl", "--libjxl-effort", effort
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Libjxl", result.stderr)
            frontend = self.run_tail_benchmark(
                "--tail-backends", "gjxl", "--frontend-effort", effort
            )
            self.assertNotEqual(frontend.returncode, 0)
            self.assertIn("Frontend effort", frontend.stderr)

        native = self.run_tail_benchmark("--tail-backends", "gjxl")
        self.assertEqual(native.returncode, 0, native.stderr)
        self.assertIn("timing_ms gjxl_tail", native.stdout)

    def test_tail_scope_raw_samples_artifacts_and_availability(self) -> None:
        destination = self.directory / "tail.json"
        artifact_directory = self.directory / "artifacts"
        if not self.libjxl_tail_enabled:
            destination.write_text("sentinel", encoding="utf-8")
            unavailable = self.run_tail_benchmark(
                "--tail-backends",
                "both",
                "--raw-samples",
                str(destination),
            )
            self.assertNotEqual(unavailable.returncode, 0)
            self.assertIn("libjxl tail is unavailable", unavailable.stderr)
            self.assertEqual(
                destination.read_text(encoding="utf-8"), "sentinel"
            )

            native = self.run_tail_benchmark(
                "--tail-backends",
                "gjxl",
                "--raw-samples",
                str(destination),
            )
            self.assertEqual(native.returncode, 0, native.stderr)
            document = json.loads(destination.read_text(encoding="utf-8"))
            workload = document["workloads"][0]
            self.assertEqual(workload["decoded_validation"], "not-run")
            self.assertEqual(len(workload["samples"]), 1)
            self.assertEqual(workload["samples"][0]["backend"], "gjxl")
            return

        result = self.run_tail_benchmark(
            "--tail-backends",
            "both",
            "--libjxl-effort",
            "5",
            "--libjxl-threads",
            "1",
            "--raw-samples",
            str(destination),
            "--artifacts",
            str(artifact_directory),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("decoded=exact-float-equal", result.stdout)
        document = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["scope"], "codestream-tail")
        self.assertEqual(document["timing"], "elapsed-wall-time")
        self.assertEqual(document["tail_boundary"], "warm-context")
        self.assertEqual(document["frontend"], "cpu")
        self.assertEqual(document["backends"], "both")
        self.assertEqual(document["libjxl_effort"], 5)
        self.assertEqual(document["libjxl_threads"], 1)
        self.assertTrue(document["libjxl_calling_thread_participates"])
        self.assertNotEqual(document["libjxl_base_revision"], "unavailable")
        self.assertNotEqual(document["libjxl_patch_revision"], "unavailable")
        workload = document["workloads"][0]
        fingerprint = workload["frame_fingerprint_sha256"]
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(workload["decoded_validation"], "exact-float-equal")
        self.assertGreater(workload["libjxl_context_setup_nanoseconds"], 0)
        self.assertEqual(
            set(workload["correctness_outputs"]), {"gjxl", "libjxl"}
        )
        self.assertIsInstance(workload["size_delta"]["percent"], float)
        samples = workload["samples"]
        self.assertEqual(len(samples), 2)
        self.assertEqual(
            [(sample["order_index"], sample["backend"]) for sample in samples],
            [(0, "gjxl"), (1, "libjxl")],
        )
        for sample in samples:
            self.assertEqual(sample["frame_fingerprint_sha256"], fingerprint)
            self.assertEqual(len(sample["codestream_sha256"]), 64)
            self.assertGreater(sample["wall_nanoseconds"], 0)
            self.assertGreater(sample["backend_total_nanoseconds"], 0)
            self.assertGreater(sample["encoded_bytes"], 2)
        libjxl = samples[1]
        self.assertEqual(libjxl["phase_nanoseconds"]["context_setup"], 0)
        self.assertGreater(
            libjxl["phase_nanoseconds"]["adapter_validation_and_copy"], 0
        )
        self.assertGreater(
            libjxl["phase_nanoseconds"]["libjxl_internal"], 0
        )
        for artifact in workload["artifacts"].values():
            path = Path(artifact)
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 0)
        self.assertFalse(list(self.directory.glob("tail.json.tmp-*")))

    def test_hybrid_workflow_reports_paired_amdahl_results(self) -> None:
        destination = self.directory / "hybrid.json"
        command = [
            str(self.benchmark),
            "--scope",
            "hybrid-workflow",
            "--workload",
            "workflow_gradient_128x96",
            "--tail-frontend",
            "cpu",
            "--tail-backends",
            "both",
            "--frontend-effort",
            "6",
            "--libjxl-effort",
            "5",
            "--libjxl-threads",
            "1",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--raw-samples",
            str(destination),
        ]
        if not self.libjxl_tail_enabled:
            destination.write_text("sentinel", encoding="utf-8")
            result = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("libjxl tail is unavailable", result.stderr)
            self.assertEqual(
                destination.read_text(encoding="utf-8"), "sentinel"
            )
            return

        result = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ratio perfect_tail_bound", result.stdout)
        self.assertIn("ratio measured_hybrid_speedup", result.stdout)
        document = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["scope"], "hybrid-workflow")
        self.assertEqual(document["frontend_effort"], 6)
        self.assertEqual(document["libjxl_effort"], 5)
        workload = document["workloads"][0]
        self.assertEqual(workload["decoded_validation"], "exact-float-equal")
        self.assertEqual(len(workload["pairs"]), 1)
        pair = workload["pairs"][0]
        self.assertEqual(pair["first_backend"], "gjxl")
        self.assertGreater(pair["perfect_tail_bound"], 1.0)
        self.assertGreater(pair["measured_outer_speedup"], 0.0)
        self.assertGreater(pair["measured_profiled_speedup"], 0.0)
        for backend in ("gjxl", "libjxl"):
            sample = pair[backend]
            self.assertEqual(sample["backend"], backend)
            self.assertGreater(sample["wall_nanoseconds"], 0)
            self.assertGreater(sample["profile_total_nanoseconds"], 0)
            self.assertGreater(
                sample["workflow_phase_nanoseconds"]["quantization_pipeline"],
                0,
            )
            self.assertGreater(
                sample["workflow_phase_nanoseconds"]["codestream_encoding"],
                0,
            )
        self.assertEqual(
            pair["libjxl"]["tail_phase_nanoseconds"]["context_setup"], 0
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
        self.assertEqual(document["schema_version"], 10)
        self.assertEqual(
            document["substage_work_timing"], "aggregate-worker-time"
        )
        self.assertEqual(document["validation"], "metal-only")
        self.assertEqual(document["density"], "default")
        self.assertEqual(document["cpu_threads"], 0)
        self.assertFalse(document["collect_final_score"])
        self.assertEqual(document["sample_count"], 1)
        workload = document["workloads"][0]
        self.assertEqual(workload["codestream_comparison"], "not-compared")
        self.assertEqual(len(workload["samples"]), 1)
        self.assertGreater(
            workload["samples"][0]["peak_cpu_participants"], 0
        )
        sample = workload["samples"][0]
        self.assertEqual(sample["sample_index"], 0)
        self.assertEqual(sample["backend"], "metal")
        self.assertIsInstance(sample["encoded_bytes"], int)
        self.assertGreater(sample["encoded_bytes"], 2)
        self.assertEqual(set(sample["entropy_bits"]), {"model", "tokens"})
        self.assertGreater(sample["entropy_bits"]["model"], 0)
        self.assertGreater(sample["entropy_bits"]["tokens"], 0)
        self.assertIsNone(sample["final_score"])
        self.assertEqual(set(sample["entropy_clusters"]), {"dc", "ac"})
        self.assertGreater(sample["entropy_clusters"]["dc"], 0)
        self.assertGreater(sample["entropy_clusters"]["ac"], 0)
        self.assertEqual(
            set(sample["ac_tokenization"]),
            {
                "template_count",
                "template_tokens",
                "context_materialization_count",
                "materialized_tokens",
            },
        )
        self.assertIn(sample["ac_tokenization"]["template_count"], {1, 2})
        self.assertGreater(sample["ac_tokenization"]["template_tokens"], 0)
        self.assertGreater(
            sample["ac_tokenization"]["context_materialization_count"], 0
        )
        self.assertGreater(
            sample["ac_tokenization"]["materialized_tokens"], 0
        )
        self.assertEqual(
            set(sample["entropy_coding"]),
            {"dc", "ac", "coefficient_order"},
        )
        for mode in sample["entropy_coding"].values():
            self.assertIn(mode, {"prefix", "ans", "none"})
        self.assertEqual(
            set(sample["coefficient_order"]),
            {"natural_bytes", "custom_bytes", "selected_mask"},
        )
        self.assertGreater(sample["coefficient_order"]["natural_bytes"], 2)
        self.assertGreaterEqual(
            sample["coefficient_order"]["custom_bytes"], 0
        )
        self.assertGreaterEqual(
            sample["coefficient_order"]["selected_mask"], 0
        )
        self.assertEqual(
            set(sample["block_context"]),
            {
                "candidate_count",
                "compact_bytes",
                "selected_index",
                "selected_contexts",
                "qf_thresholds",
            },
        )
        self.assertGreater(sample["block_context"]["candidate_count"], 0)
        self.assertGreater(sample["block_context"]["compact_bytes"], 2)
        self.assertGreaterEqual(sample["block_context"]["selected_index"], 0)
        self.assertGreater(sample["block_context"]["selected_contexts"], 0)
        self.assertGreaterEqual(sample["block_context"]["qf_thresholds"], 0)
        self.assertEqual(set(sample["phase_nanoseconds"]), PHASES)
        for value in sample["phase_nanoseconds"].values():
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)
        for phase in PHASES:
            if phase in ELIMINATED_WORK_PHASES:
                self.assertEqual(sample["phase_nanoseconds"][phase], 0)
            elif phase.endswith("_work"):
                self.assertGreater(sample["phase_nanoseconds"][phase], 0)
        self.assertFalse(list(self.directory.glob("samples.json.tmp-*")))

    def test_external_pfm_input_uses_its_source_extent(self) -> None:
        source = self.directory / "input.pfm"
        width = 128
        height = 96
        source.write_bytes(
            f"PF\n{width} {height}\n-1.0\n".encode("ascii")
            + struct.pack("<3f", 0.1, 0.2, 0.3) * (width * height)
        )

        result = self.run_benchmark(
            "--validation",
            "metal-only",
            "--metallib",
            str(self.metallib),
            "--input",
            str(source),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("workload external_input source=128x96", result.stdout)

    def test_high_density_is_explicit_in_raw_samples(self) -> None:
        destination = self.directory / "high-density.json"
        result = self.run_benchmark(
            "--density",
            "high",
            "--validation",
            "metal-only",
            "--raw-samples",
            str(destination),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], 10)
        self.assertEqual(document["density"], "high")
        self.assertIn("density=high", result.stdout)

    def test_cpu_thread_budget_is_recorded_and_enforced(self) -> None:
        destination = self.directory / "thread-budget.json"
        result = self.run_benchmark(
            "--scope",
            "public-workflow",
            "--cpu-threads",
            "2",
            "--raw-samples",
            str(destination),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], 10)
        self.assertEqual(document["cpu_threads"], 2)
        self.assertIn("cpu_threads=2", result.stdout)
        for sample in document["workloads"][0]["samples"]:
            self.assertGreater(sample["peak_cpu_participants"], 0)
            self.assertLessEqual(sample["peak_cpu_participants"], 2)

    def test_final_score_collection_is_explicit(self) -> None:
        destination = self.directory / "scored.json"
        result = self.run_benchmark(
            "--collect-final-score",
            "--validation",
            "metal-only",
            "--raw-samples",
            str(destination),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(destination.read_text(encoding="utf-8"))
        self.assertTrue(document["collect_final_score"])
        sample = document["workloads"][0]["samples"][0]
        self.assertIsInstance(sample["final_score"], float)
        self.assertIn("final_score=collect", result.stdout)

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
        self.assertEqual(document["schema_version"], 3)
        self.assertEqual(document["mode"], "stage")
        self.assertFalse(document["collect_final_score"])
        sample = document["workloads"][0]["samples"][0]
        self.assertTrue(sample["capabilities"]["timestamp_counter"])
        self.assertTrue(sample["capabilities"]["stage_boundary"])
        submission_ids = [
            submission["submission_id"] for submission in sample["submissions"]
        ]
        self.assertEqual(
            submission_ids,
            [
                "frontend.prepare_aq.reference",
                "frontend.initial_quantization",
                "frontend.ac_strategy",
                "frontend.quant_adjustment",
                "resident.aq",
            ],
        )
        for submission in sample["submissions"]:
            self.assertEqual(submission["invocation"], 0)
            self.assertGreater(submission["command_buffer_gpu_nanoseconds"], 0)
        wall_stages = {
            (stage["stage_id"], stage["kind"]): stage
            for stage in sample["wall_stages"]
        }
        self.assertIn(
            ("frontend.prepare_evaluator", "preparation"), wall_stages
        )
        self.assertIn(
            ("frontend.initial_quantization", "operation"), wall_stages
        )
        self.assertIn(("frontend.ac_strategy.wait", "wait"), wall_stages)
        self.assertIn(("frontend.reconfigure_aq", "preparation"), wall_stages)
        self.assertNotIn(("frontend.prepare_aq", "preparation"), wall_stages)
        self.assertIn(("frontend.fixed_cfl", "host"), wall_stages)
        self.assertNotIn(("frontend.cfl_upload", "upload"), wall_stages)
        self.assertIn(("resident.aq", "operation"), wall_stages)
        for wall_stage in sample["wall_stages"]:
            self.assertEqual(wall_stage["invocation"], 0)
            self.assertGreaterEqual(wall_stage["wall_nanoseconds"], 0)
        submission = next(
            item
            for item in sample["submissions"]
            if item["submission_id"] == "resident.aq"
        )
        self.assertGreater(submission["command_buffer_gpu_nanoseconds"], 0)
        stages = submission["stages"]
        reconstruction_stages = {
            stage["stage_id"]
            for stage in stages
            if stage["group_id"] == "aq.reconstruction"
        }
        self.assertIn("aq.reconstruction.reset", reconstruction_stages)
        self.assertIn("aq.reconstruction.quantizer", reconstruction_stages)
        self.assertTrue(
            any(
                stage.startswith("aq.reconstruction.dct")
                for stage in reconstruction_stages
            )
        )
        self.assertIn("aq.epf.pass_1", {stage["stage_id"] for stage in stages})
        self.assertIn(
            "butteraugli.malta.main", {stage["stage_id"] for stage in stages}
        )
        self.assertEqual(
            sum(
                dispatch["kernel_id"] == "gjxl_aq_final_cfl"
                for stage in stages
                for dispatch in stage["dispatches"]
            ),
            1,
        )
        for stage in stages:
            self.assertTrue(stage["group_id"])
            if stage["group_id"] == "aq.reconstruction":
                self.assertEqual(stage["invocation"], stage["iteration"])
            self.assertGreaterEqual(
                stage["end_timestamp"], stage["begin_timestamp"]
            )
            self.assertEqual(
                stage["gpu_nanoseconds"],
                stage["end_timestamp"] - stage["begin_timestamp"],
            )
            self.assertTrue(stage["dispatches"])
        ac_submission = next(
            item
            for item in sample["submissions"]
            if item["submission_id"] == "frontend.ac_strategy"
        )
        ac_stages = ac_submission["stages"]
        self.assertEqual(
            {stage["stage_id"] for stage in ac_stages},
            {
                "frontend.ac_strategy.dct8",
                "frontend.ac_strategy.dct16x8",
                "frontend.ac_strategy.dct8x16",
                "frontend.ac_strategy.dct16",
                "frontend.ac_strategy.dct32x16",
                "frontend.ac_strategy.dct16x32",
                "frontend.ac_strategy.dct32",
            },
        )
        self.assertTrue(
            all(
                stage["group_id"] == "frontend.ac_strategy"
                and stage["invocation"] == 0
                for stage in ac_stages
            )
        )
        self.assertTrue(all(len(stage["dispatches"]) == 3 for stage in ac_stages))
        for stage in ac_stages:
            kernel_ids = {
                dispatch["kernel_id"] for dispatch in stage["dispatches"]
            }
            self.assertNotIn("gjxl_ac_strategy_gather", kernel_ids)
            self.assertEqual(
                sum(
                    kernel_id.startswith("gjxl_ac_strategy_dct")
                    and kernel_id.endswith("_forward_fused")
                    for kernel_id in kernel_ids
                ),
                1,
            )
            self.assertNotIn("gjxl_ac_strategy_residual", kernel_ids)
            self.assertEqual(
                sum(
                    kernel_id.startswith("gjxl_ac_strategy_dct")
                    and kernel_id.endswith("_residual_inverse_fused")
                    for kernel_id in kernel_ids
                ),
                1,
            )
            self.assertIn("gjxl_ac_strategy_cost", kernel_ids)

    def test_unsupported_dispatch_profile_preserves_existing_output(self) -> None:
        capability_output = self.directory / "capabilities.json"
        stage_result = self.run_benchmark(
            "--validation",
            "metal-only",
            "--metallib",
            str(self.metallib),
            "--gpu-profile",
            "stage",
            "--gpu-profile-output",
            str(capability_output),
        )
        self.assertEqual(stage_result.returncode, 0, stage_result.stderr)
        capability_document = json.loads(
            capability_output.read_text(encoding="utf-8")
        )
        capabilities = capability_document["workloads"][0]["samples"][0][
            "capabilities"
        ]
        if capabilities["dispatch_boundary"]:
            self.skipTest("device supports dispatch-boundary timestamps")

        destination = self.directory / "dispatch.json"
        destination.write_text("sentinel", encoding="utf-8")
        result = self.run_benchmark(
            "--validation",
            "metal-only",
            "--metallib",
            str(self.metallib),
            "--gpu-profile",
            "dispatch",
            "--gpu-profile-output",
            str(destination),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "GPU dispatch-boundary timestamp sampling is unavailable",
            result.stderr,
        )
        self.assertEqual(destination.read_text(encoding="utf-8"), "sentinel")

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
    parser.add_argument("--libjxl-tail-enabled", choices=("0", "1"), required=True)
    arguments, remaining = parser.parse_known_args()
    EncodingBenchmarkCliTest.benchmark = arguments.benchmark.resolve()
    EncodingBenchmarkCliTest.metallib = arguments.metallib.resolve()
    EncodingBenchmarkCliTest.libjxl_tail_enabled = (
        arguments.libjxl_tail_enabled == "1"
    )
    unittest.main(argv=[__file__, *remaining])


if __name__ == "__main__":
    main()
