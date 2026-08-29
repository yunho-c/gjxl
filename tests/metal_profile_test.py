#!/usr/bin/env python3
"""Unit tests for the reproducible Metal profiling driver."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import metal_profile  # noqa: E402


class FakeCommandRunner:
    def __init__(
        self,
        records: list[dict[str, object]],
        *,
        fail_trace: bool = False,
        mutate_source: Path | None = None,
    ) -> None:
        self.records = records
        self.fail_trace = fail_trace
        self.mutate_source = mutate_source

    def run(
        self,
        command: list[str],
        *,
        cwd: Path,
        log_path: Path,
    ) -> None:
        self.records.append(
            {
                "command": list(command),
                "cwd": str(cwd),
                "log": log_path.name,
                "exit_code": 0,
            }
        )
        log_path.write_text("mock command\n", encoding="utf-8")
        if command[:2] == ["cmake", "--build"]:
            build_dir = Path(command[2])
            (build_dir / "metal").mkdir(parents=True, exist_ok=True)
            (build_dir / "gjxl_quantization_benchmark").write_bytes(b"binary")
            (build_dir / "metal" / "gjxl.metallib").write_bytes(b"metallib")
            (build_dir / "metal" / "gjxl.metallibsym").write_bytes(b"symbols")
            return
        if Path(command[0]).name == "gjxl_quantization_benchmark":
            if "--gpu-profile-output" in command:
                output = Path(command[command.index("--gpu-profile-output") + 1])
                output.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "mode": "stage",
                            "gpu_aq": "fully-resident",
                            "distance": 1.2,
                            "workloads": [],
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                raw_path = Path(command[command.index("--raw-samples") + 1])
                raw_path.write_text(
                    json.dumps({"schema_version": 1, "workloads": []}),
                    encoding="utf-8",
                )
            return
        if command[:3] == ["xcrun", "xctrace", "record"]:
            if self.mutate_source is not None:
                self.mutate_source.write_text("changed\n", encoding="utf-8")
            if self.fail_trace:
                self.records[-1]["exit_code"] = 9
                raise RuntimeError("mock trace failure")
            Path(command[command.index("--output") + 1]).mkdir()
            Path(command[command.index("--target-stdout") + 1]).write_text(
                "trace target\n", encoding="utf-8"
            )
            raw_path = Path(command[command.index("--raw-samples") + 1])
            raw_path.write_text(
                json.dumps({"schema_version": 1, "workloads": []}),
                encoding="utf-8",
            )
            return
        if command[:3] == ["xcrun", "xctrace", "export"]:
            Path(command[command.index("--output") + 1]).write_text(
                "<trace-toc/>\n", encoding="utf-8"
            )


class MetalProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="gjxl-profile-test-")
        self.directory = Path(self.temporary.name)
        self.repo = self.directory / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Profile Test"],
            cwd=self.repo,
            check=True,
        )
        (self.repo / ".gitignore").write_text(
            "build/\nlogs/\n", encoding="utf-8"
        )
        self.source = self.repo / "source.txt"
        self.source.write_text("original\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "fixture"], cwd=self.repo, check=True
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def arguments(self, output: Path) -> list[str]:
        return [
            "--repo",
            str(self.repo),
            "--build-dir",
            "build/metal-profile",
            "--output",
            str(output),
            "--workload",
            "synthetic_128x96",
            "--samples",
            "1",
            "--warmups",
            "0",
        ]

    def run_mocked(
        self,
        output: Path,
        *,
        fail_trace: bool = False,
        mutate_source: bool = False,
    ) -> int:
        def runner_factory(records: list[dict[str, object]]) -> FakeCommandRunner:
            return FakeCommandRunner(
                records,
                fail_trace=fail_trace,
                mutate_source=self.source if mutate_source else None,
            )

        with (
            mock.patch.object(metal_profile.platform, "system", return_value="Darwin"),
            mock.patch.object(metal_profile.shutil, "which", return_value="/mock/tool"),
            mock.patch.object(metal_profile, "system_metadata", return_value={"mock": True}),
            mock.patch.object(metal_profile, "CommandRunner", side_effect=runner_factory),
        ):
            return metal_profile.main(self.arguments(output))

    def test_success_manifest_records_exact_commands_and_hashes(self) -> None:
        output = self.directory / "artifact"

        result = self.run_mocked(output)

        self.assertEqual(result, 0)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(manifest["git"]["source_changed_during_run"])
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(len(manifest["commands"]), 6)
        benchmark = manifest["commands"][2]["command"]
        gpu_profile = manifest["commands"][3]["command"]
        trace = manifest["commands"][4]["command"]
        self.assertEqual(benchmark[benchmark.index("--validation") + 1], "cpu-metal")
        self.assertEqual(
            gpu_profile[gpu_profile.index("--gpu-profile") + 1], "stage"
        )
        self.assertEqual(trace[trace.index("--validation") + 1], "metal-only")
        self.assertIn("gjxl_metal_profile_symbols", manifest["commands"][1]["command"])
        for artifact in manifest["build_artifacts"].values():
            self.assertEqual(len(artifact["sha256"]), 64)
        self.assertTrue((output / "worktree.patch").exists())
        self.assertTrue((output / "index.patch").exists())
        self.assertTrue((output / "untracked-files.json").exists())
        self.assertTrue((output / "gpu-stage-samples.json").exists())
        self.assertTrue((output / "gpu-stage-summary.json").exists())

    def test_stage_aggregation_reports_cumulative_medians(self) -> None:
        payload = {
            "schema_version": 2,
            "mode": "stage",
            "gpu_aq": "fully-resident",
            "distance": 1.2,
            "workloads": [
                {
                    "name": "fixture",
                    "source_width": 16,
                    "source_height": 16,
                    "samples": [
                        {
                            "wall_stages": [
                                {
                                    "stage_id": "frontend.ac_strategy.wait",
                                    "kind": "wait",
                                    "wall_nanoseconds": 40,
                                }
                            ],
                            "submissions": [
                                {
                                    "submission_id": "resident.aq",
                                    "command_buffer_gpu_nanoseconds": 100,
                                    "stages": [
                                        {
                                            "stage_id": "aq.epf.pass_1",
                                            "iteration": 0,
                                            "gpu_nanoseconds": 20,
                                            "dispatches": [{}, {}],
                                        },
                                        {
                                            "stage_id": "aq.epf.pass_1",
                                            "iteration": 1,
                                            "gpu_nanoseconds": 30,
                                            "dispatches": [{}, {}],
                                        },
                                    ],
                                }
                            ]
                        }
                    ],
                }
            ],
        }

        summary = metal_profile.aggregate_gpu_stage_samples(payload)

        workload = summary["workloads"][0]
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["source_schema_version"], 2)
        self.assertEqual(workload["median_sampled_stage_coverage_percent"], 50)
        submission = workload["submissions"][0]
        self.assertEqual(submission["submission_id"], "resident.aq")
        self.assertEqual(submission["median_cumulative_gpu_nanoseconds"], 100)
        self.assertEqual(submission["median_stage_count"], 2)
        wall_stage = workload["wall_stages"][0]
        self.assertEqual(wall_stage["stage_id"], "frontend.ac_strategy.wait")
        self.assertEqual(wall_stage["kind"], "wait")
        self.assertEqual(wall_stage["median_cumulative_wall_nanoseconds"], 40)
        stage = workload["stages"][0]
        self.assertEqual(stage["median_cumulative_gpu_nanoseconds"], 50)
        self.assertEqual(stage["median_call_count"], 2)
        self.assertEqual(stage["median_dispatch_count"], 4)
        self.assertEqual(stage["median_percent_of_command_buffer_gpu_time"], 50)

    def test_stage_aggregation_keeps_schema_one_artifacts_readable(self) -> None:
        payload = {
            "schema_version": 1,
            "mode": "stage",
            "gpu_aq": "fully-resident",
            "distance": 1.2,
            "workloads": [
                {
                    "name": "legacy",
                    "source_width": 8,
                    "source_height": 8,
                    "samples": [
                        {
                            "submissions": [
                                {
                                    "command_buffer_gpu_nanoseconds": 10,
                                    "stages": [],
                                }
                            ]
                        }
                    ],
                }
            ],
        }

        summary = metal_profile.aggregate_gpu_stage_samples(payload)

        self.assertEqual(summary["source_schema_version"], 1)
        workload = summary["workloads"][0]
        self.assertEqual(
            workload["submissions"][0]["submission_id"], "submission_0"
        )
        self.assertEqual(workload["wall_stages"], [])

    def test_trace_failure_preserves_partial_failed_manifest(self) -> None:
        output = self.directory / "failed-artifact"

        result = self.run_mocked(output, fail_trace=True)

        self.assertEqual(result, 1)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("mock trace failure", manifest["error"])
        self.assertTrue((output / "raw-samples.json").exists())

    def test_source_change_marks_otherwise_complete_capture_failed(self) -> None:
        output = self.directory / "changed-artifact"

        result = self.run_mocked(output, mutate_source=True)

        self.assertEqual(result, 1)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue(manifest["git"]["source_changed_during_run"])
        self.assertEqual(manifest["error"], "Source checkout changed during profiling")

    def test_existing_artifact_is_never_overwritten(self) -> None:
        output = self.directory / "existing"
        output.mkdir()
        sentinel = output / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")

        result = self.run_mocked(output)

        self.assertEqual(result, 2)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_generated_roots_do_not_trigger_source_change(self) -> None:
        generated = self.repo / "generated-profile"
        generated.mkdir()
        payload = generated / "capture.log"
        payload.write_text("first\n", encoding="utf-8")
        before = metal_profile.capture_git_state(
            self.repo, excluded_roots=(generated,)
        )

        payload.write_text("second\n", encoding="utf-8")
        after = metal_profile.capture_git_state(
            self.repo, excluded_roots=(generated,)
        )

        self.assertEqual(
            before["source_fingerprint"], after["source_fingerprint"]
        )
        self.assertNotIn("generated-profile", after["status_porcelain_v2"])


if __name__ == "__main__":
    unittest.main()
