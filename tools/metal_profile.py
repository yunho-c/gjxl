#!/usr/bin/env python3
"""Build and capture a reproducible Metal encoder profiling artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence


ENVIRONMENT_ALLOWLIST = (
    "CMAKE_BUILD_PARALLEL_LEVEL",
    "CMAKE_GENERATOR",
    "DEVELOPER_DIR",
    "MTL_CAPTURE_ENABLED",
    "SDKROOT",
)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def probe(command: Sequence[str], cwd: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": list(command), "error": str(error)}
    return {
        "command": list(command),
        "exit_code": result.returncode,
        "output": (result.stdout + result.stderr).strip(),
    }


def probe_json(command: Sequence[str], cwd: Path) -> dict[str, Any]:
    result = probe(command, cwd)
    if result.get("exit_code") == 0:
        try:
            result["json"] = json.loads(result["output"])
        except json.JSONDecodeError as error:
            result["parse_error"] = str(error)
    return result


def git_bytes(repo: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    return result.stdout


def is_excluded(path: Path, excluded_roots: Sequence[Path]) -> bool:
    absolute = Path(os.path.abspath(path))
    resolved = absolute.parent.resolve(strict=False) / absolute.name
    return any(
        resolved == root or resolved.is_relative_to(root)
        for root in excluded_roots
    )


def untracked_inventory(
    repo: Path, excluded_roots: Sequence[Path] = ()
) -> list[dict[str, Any]]:
    paths = git_bytes(
        repo, "ls-files", "--others", "--exclude-standard", "-z"
    ).split(b"\0")
    inventory: list[dict[str, Any]] = []
    for encoded_path in paths:
        if not encoded_path:
            continue
        relative = encoded_path.decode("utf-8", errors="surrogateescape")
        path = repo / relative
        if is_excluded(path, excluded_roots):
            continue
        entry: dict[str, Any] = {"path": relative}
        if path.is_symlink():
            entry["kind"] = "symlink"
            entry["target"] = os.readlink(path)
            entry["sha256"] = sha256_bytes(
                os.fsencode(entry["target"])
            )
        elif path.is_file():
            entry["kind"] = "file"
            entry["size"] = path.stat().st_size
            entry["sha256"] = sha256_file(path)
        else:
            entry["kind"] = "missing"
        inventory.append(entry)
    return inventory


def capture_git_state(
    repo: Path,
    artifact_dir: Path | None = None,
    excluded_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    excluded_roots = tuple(
        path.resolve(strict=False) for path in excluded_roots
    )
    head = git_bytes(repo, "rev-parse", "HEAD").decode().strip()
    branch = git_bytes(repo, "branch", "--show-current").decode().strip()
    status_parts = git_bytes(
        repo, "status", "--porcelain=v2", "--untracked-files=all", "-z"
    ).split(b"\0")
    filtered_status_parts: list[bytes] = []
    for status_part in status_parts:
        if not status_part:
            continue
        if status_part.startswith(b"? "):
            relative = status_part[2:].decode(
                "utf-8", errors="surrogateescape"
            )
            if is_excluded(repo / relative, excluded_roots):
                continue
        filtered_status_parts.append(status_part)
    status = b"\0".join(filtered_status_parts)
    worktree_patch = git_bytes(repo, "diff", "--binary", "--no-ext-diff")
    index_patch = git_bytes(
        repo, "diff", "--cached", "--binary", "--no-ext-diff"
    )
    untracked = untracked_inventory(repo, excluded_roots)
    untracked_payload = json.dumps(
        untracked, sort_keys=True, ensure_ascii=True
    ).encode()
    fingerprint_payload = b"\0".join(
        (
            head.encode(),
            worktree_patch,
            index_patch,
            untracked_payload,
        )
    )
    state = {
        "head": head,
        "short_head": head[:12],
        "branch": branch,
        "status_porcelain_v2": status.decode(
            "utf-8", errors="backslashreplace"
        ),
        "status_sha256": sha256_bytes(status),
        "worktree_patch_sha256": sha256_bytes(worktree_patch),
        "index_patch_sha256": sha256_bytes(index_patch),
        "untracked_inventory": untracked,
        "untracked_inventory_sha256": sha256_bytes(untracked_payload),
        "source_fingerprint": sha256_bytes(fingerprint_payload),
    }
    if artifact_dir is not None:
        (artifact_dir / "worktree.patch").write_bytes(worktree_patch)
        (artifact_dir / "index.patch").write_bytes(index_patch)
        atomic_write_json(
            artifact_dir / "untracked-files.json", untracked
        )
    return state


class CommandRunner:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        log_path: Path,
    ) -> None:
        started = utc_now()
        begin = time.monotonic()
        record: dict[str, Any] = {
            "command": list(command),
            "cwd": str(cwd),
            "started_at": started,
            "log": log_path.name,
        }
        self.records.append(record)
        try:
            with log_path.open("wb") as log:
                result = subprocess.run(
                    command,
                    cwd=cwd,
                    check=False,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            record["exit_code"] = result.returncode
        except OSError as error:
            record["exit_code"] = None
            record["launch_error"] = str(error)
            raise RuntimeError(
                f"Could not launch {command[0]}: {error}"
            ) from error
        finally:
            record["finished_at"] = utc_now()
            record["duration_seconds"] = round(
                time.monotonic() - begin, 6
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}: "
                + " ".join(command)
                + f" (see {log_path})"
            )


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not slug:
        raise ValueError("Artifact name component is empty after sanitizing")
    return slug


def create_artifact_dir(
    output_root: Path,
    workload: str,
    gpu_aq: str,
    short_head: str,
    explicit_output: Path | None,
) -> Path:
    if explicit_output is None:
        timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        destination = output_root / (
            f"{timestamp}-{safe_slug(workload)}-{safe_slug(gpu_aq)}-"
            f"{safe_slug(short_head)}"
        )
    else:
        destination = explicit_output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    return destination.resolve()


def make_commands(
    args: argparse.Namespace,
    repo: Path,
    build_dir: Path,
    artifact_dir: Path,
) -> dict[str, list[str]]:
    benchmark = (build_dir / "gjxl_encoding_benchmark").resolve()
    metallib = (build_dir / "metal" / "gjxl.metallib").resolve()
    common = [
        str(benchmark),
        "--scope",
        "metal-public-workflow",
        "--workload",
        args.workload,
        "--implementation",
        args.implementation,
        "--gpu-aq",
        args.gpu_aq,
        "--distance",
        str(args.distance),
        "--metallib",
        str(metallib),
    ]
    benchmark_command = [
        *common,
        "--validation",
        "cpu-metal",
        "--warmups",
        str(args.warmups),
        "--samples",
        str(args.samples),
        "--raw-samples",
        str(artifact_dir / "raw-samples.json"),
    ]
    trace_target = [
        *common,
        "--validation",
        "metal-only",
        "--warmups",
        "0",
        "--samples",
        "1",
        "--raw-samples",
        str(artifact_dir / "trace-sample.json"),
    ]
    return {
        "configure": [
            "cmake",
            "-S",
            str(repo),
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGJXL_BUILD_TESTS=OFF",
            "-DGJXL_BUILD_BENCHMARKS=ON",
            "-DGJXL_ENABLE_LIBJXL_REFERENCE=OFF",
            "-DGJXL_ENABLE_METAL_PROFILING=ON",
            "-DHWY_ENABLE_TESTS=OFF",
        ],
        "build": [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "gjxl_encoding_benchmark",
            "gjxl_metal_profile_symbols",
        ],
        "benchmark": benchmark_command,
        "gpu_profile": [
            *common,
            "--validation",
            "metal-only",
            "--warmups",
            str(args.warmups),
            "--samples",
            str(args.samples),
            "--gpu-profile",
            "stage",
            "--gpu-profile-output",
            str(artifact_dir / "gpu-stage-samples.json"),
        ],
        "trace": [
            "xcrun",
            "xctrace",
            "record",
            "--template",
            "Metal System Trace",
            "--output",
            str(artifact_dir / "capture.trace"),
            "--no-prompt",
            "--target-stdout",
            str(artifact_dir / "trace.stdout"),
            "--launch",
            "--",
            *trace_target,
        ],
        "export": [
            "xcrun",
            "xctrace",
            "export",
            "--input",
            str(artifact_dir / "capture.trace"),
            "--toc",
            "--output",
            str(artifact_dir / "trace-toc.xml"),
        ],
    }


def aggregate_gpu_stage_samples(payload: dict[str, Any]) -> dict[str, Any]:
    workloads: list[dict[str, Any]] = []
    for workload in payload["workloads"]:
        per_stage: dict[str, list[dict[str, float]]] = {}
        per_stage_group: dict[str, list[dict[str, float]]] = {}
        per_iteration: dict[tuple[str, int], list[dict[str, float]]] = {}
        per_group_iteration: dict[tuple[str, int], list[dict[str, float]]] = {}
        per_submission: dict[str, list[dict[str, float]]] = {}
        per_wall_stage: dict[tuple[str, str], list[dict[str, float]]] = {}
        coverage_percent: list[float] = []
        for sample in workload["samples"]:
            total_gpu = sum(
                submission["command_buffer_gpu_nanoseconds"]
                for submission in sample["submissions"]
            )
            stage_totals: dict[str, dict[str, float]] = {}
            stage_group_totals: dict[str, dict[str, float]] = {}
            iteration_totals: dict[tuple[str, int], dict[str, float]] = {}
            group_iteration_totals: dict[
                tuple[str, int], dict[str, float]
            ] = {}
            submission_totals: dict[str, dict[str, float]] = {}
            wall_totals: dict[tuple[str, str], dict[str, float]] = {}
            sampled_gpu = 0
            for submission_index, submission in enumerate(sample["submissions"]):
                submission_id = submission.get(
                    "submission_id", f"submission_{submission_index}"
                )
                submission_entry = submission_totals.setdefault(
                    submission_id,
                    {"duration": 0.0, "calls": 0.0, "stages": 0.0,
                     "dispatches": 0.0},
                )
                submission_entry["duration"] += submission[
                    "command_buffer_gpu_nanoseconds"
                ]
                submission_entry["calls"] += 1
                submission_entry["stages"] += len(submission["stages"])
                for stage in submission["stages"]:
                    duration = stage["gpu_nanoseconds"]
                    sampled_gpu += duration
                    submission_entry["dispatches"] += len(stage["dispatches"])
                    stage_entry = stage_totals.setdefault(
                        stage["stage_id"],
                        {"duration": 0.0, "calls": 0.0, "dispatches": 0.0},
                    )
                    stage_entry["duration"] += duration
                    stage_entry["calls"] += 1
                    stage_entry["dispatches"] += len(stage["dispatches"])
                    group_id = stage.get("group_id", stage["stage_id"])
                    group_entry = stage_group_totals.setdefault(
                        group_id,
                        {"duration": 0.0, "calls": 0.0, "dispatches": 0.0},
                    )
                    group_entry["duration"] += duration
                    group_entry["calls"] += 1
                    group_entry["dispatches"] += len(stage["dispatches"])
                    key = (stage["stage_id"], stage["iteration"])
                    iteration_entry = iteration_totals.setdefault(
                        key,
                        {"duration": 0.0, "calls": 0.0, "dispatches": 0.0},
                    )
                    iteration_entry["duration"] += duration
                    iteration_entry["calls"] += 1
                    iteration_entry["dispatches"] += len(stage["dispatches"])
                    group_key = (group_id, stage["iteration"])
                    group_iteration_entry = group_iteration_totals.setdefault(
                        group_key,
                        {"duration": 0.0, "calls": 0.0, "dispatches": 0.0},
                    )
                    group_iteration_entry["duration"] += duration
                    group_iteration_entry["calls"] += 1
                    group_iteration_entry["dispatches"] += len(
                        stage["dispatches"]
                    )
            for wall_stage in sample.get("wall_stages", []):
                key = (wall_stage["stage_id"], wall_stage["kind"])
                wall_entry = wall_totals.setdefault(
                    key, {"duration": 0.0, "calls": 0.0}
                )
                wall_entry["duration"] += wall_stage["wall_nanoseconds"]
                wall_entry["calls"] += 1
            for stage_id, values in stage_totals.items():
                values["percent"] = (
                    100.0 * values["duration"] / total_gpu if total_gpu else 0.0
                )
                per_stage.setdefault(stage_id, []).append(values)
            for group_id, values in stage_group_totals.items():
                values["percent"] = (
                    100.0 * values["duration"] / total_gpu if total_gpu else 0.0
                )
                per_stage_group.setdefault(group_id, []).append(values)
            for key, values in iteration_totals.items():
                values["percent"] = (
                    100.0 * values["duration"] / total_gpu if total_gpu else 0.0
                )
                per_iteration.setdefault(key, []).append(values)
            for key, values in group_iteration_totals.items():
                values["percent"] = (
                    100.0 * values["duration"] / total_gpu if total_gpu else 0.0
                )
                per_group_iteration.setdefault(key, []).append(values)
            for submission_id, values in submission_totals.items():
                values["percent"] = (
                    100.0 * values["duration"] / total_gpu if total_gpu else 0.0
                )
                per_submission.setdefault(submission_id, []).append(values)
            for key, values in wall_totals.items():
                per_wall_stage.setdefault(key, []).append(values)
            coverage_percent.append(
                100.0 * sampled_gpu / total_gpu if total_gpu else 0.0
            )

        def summarize(values: list[dict[str, float]]) -> dict[str, Any]:
            return {
                "median_cumulative_gpu_nanoseconds": int(
                    statistics.median(value["duration"] for value in values)
                ),
                "median_call_count": statistics.median(
                    value["calls"] for value in values
                ),
                "median_dispatch_count": statistics.median(
                    value["dispatches"] for value in values
                ),
                "median_percent_of_command_buffer_gpu_time": statistics.median(
                    value["percent"] for value in values
                ),
            }

        stages = [
            {"stage_id": stage_id, **summarize(values)}
            for stage_id, values in sorted(per_stage.items())
        ]
        stage_groups = [
            {"group_id": group_id, **summarize(values)}
            for group_id, values in sorted(per_stage_group.items())
        ]
        iterations = [
            {"stage_id": key[0], "iteration": key[1], **summarize(values)}
            for key, values in sorted(per_iteration.items())
        ]
        group_iterations = [
            {"group_id": key[0], "iteration": key[1], **summarize(values)}
            for key, values in sorted(per_group_iteration.items())
        ]
        submissions = [
            {
                "submission_id": submission_id,
                "median_cumulative_gpu_nanoseconds": int(
                    statistics.median(value["duration"] for value in values)
                ),
                "median_call_count": statistics.median(
                    value["calls"] for value in values
                ),
                "median_stage_count": statistics.median(
                    value["stages"] for value in values
                ),
                "median_dispatch_count": statistics.median(
                    value["dispatches"] for value in values
                ),
                "median_percent_of_command_buffer_gpu_time": statistics.median(
                    value["percent"] for value in values
                ),
            }
            for submission_id, values in sorted(per_submission.items())
        ]
        wall_stages = [
            {
                "stage_id": key[0],
                "kind": key[1],
                "median_cumulative_wall_nanoseconds": int(
                    statistics.median(value["duration"] for value in values)
                ),
                "median_call_count": statistics.median(
                    value["calls"] for value in values
                ),
            }
            for key, values in sorted(per_wall_stage.items())
        ]
        workloads.append(
            {
                "name": workload["name"],
                "source_width": workload["source_width"],
                "source_height": workload["source_height"],
                "sample_count": len(workload["samples"]),
                "median_sampled_stage_coverage_percent": statistics.median(
                    coverage_percent
                ),
                "submissions": submissions,
                "wall_stages": wall_stages,
                "stages": stages,
                "stage_groups": stage_groups,
                "iterations": iterations,
                "group_iterations": group_iterations,
            }
        )
    return {
        "schema_version": 3,
        "source_schema_version": payload["schema_version"],
        "mode": payload["mode"],
        "gpu_aq": payload["gpu_aq"],
        "collect_final_score": payload.get("collect_final_score", False),
        "distance": payload["distance"],
        "workloads": workloads,
    }


def required_artifacts(build_dir: Path) -> dict[str, Path]:
    return {
        "benchmark": build_dir / "gjxl_encoding_benchmark",
        "metallib": build_dir / "metal" / "gjxl.metallib",
        "metallib_symbols": build_dir / "metal" / "gjxl.metallibsym",
    }


def verify_artifacts(paths: dict[str, Path]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required profiling artifact is missing: {path}")
        verified[name] = {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return verified


def system_metadata(repo: Path) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos": probe(["sw_vers"], repo),
        "xcode": probe(["xcodebuild", "-version"], repo),
        "metal": probe(["xcrun", "-sdk", "macosx", "metal", "--version"], repo),
        "metallib": probe(
            ["xcrun", "-sdk", "macosx", "metallib", "--version"], repo
        ),
        "xctrace": probe(["xcrun", "xctrace", "version"], repo),
        "displays": probe_json(
            ["system_profiler", "SPDisplaysDataType", "-json"], repo
        ),
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-dir", type=Path, default=Path("build/metal-profile"))
    parser.add_argument("--output-root", type=Path, default=Path("logs/metal-profile"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workload", default="padded_4k")
    parser.add_argument(
        "--implementation", choices=("scalar", "simd", "factored"), default="simd"
    )
    parser.add_argument(
        "--gpu-aq",
        choices=(
            "exact-coefficients",
            "fully-resident",
            "throughput",
            "maximum-throughput",
        ),
        default="fully-resident",
    )
    parser.add_argument("--distance", type=float, default=1.2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args(argv)
    if args.distance <= 0.0:
        parser.error("--distance must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    gpu_stage_enabled = args.gpu_aq in ("fully-resident", "throughput")
    repo = args.repo.resolve()
    if platform.system() != "Darwin":
        print("Metal profiling requires macOS", file=sys.stderr)
        return 2
    if not (repo / ".git").exists():
        print(f"Not a Git checkout: {repo}", file=sys.stderr)
        return 2
    missing = [tool for tool in ("cmake", "git", "ninja", "xcrun") if shutil.which(tool) is None]
    if missing:
        print("Missing required tools: " + ", ".join(missing), file=sys.stderr)
        return 2

    build_dir = args.build_dir
    if not build_dir.is_absolute():
        build_dir = repo / build_dir
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo / output_root
    explicit_output = args.output
    if explicit_output is not None and not explicit_output.is_absolute():
        explicit_output = repo / explicit_output

    head = git_bytes(repo, "rev-parse", "HEAD").decode().strip()
    try:
        artifact_dir = create_artifact_dir(
            output_root,
            args.workload,
            args.gpu_aq,
            head[:12],
            explicit_output,
        )
    except FileExistsError as error:
        print(f"Profiling artifact already exists: {error.filename}", file=sys.stderr)
        return 2

    state_exclusions = (artifact_dir, build_dir)
    pre_state = capture_git_state(
        repo, artifact_dir, excluded_roots=state_exclusions
    )
    started_at = utc_now()
    commands: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "status": "running",
        "started_at": started_at,
        "artifact_directory": str(artifact_dir),
        "configuration": {
            "workload": args.workload,
            "implementation": args.implementation,
            "gpu_aq": args.gpu_aq,
            "distance": args.distance,
            "warmups": args.warmups,
            "samples": args.samples,
            "build_directory": str(build_dir.resolve()),
            "cmake_build_type": "Release",
            "cmake_generator": "Ninja",
            "metal_trace_template": "Metal System Trace",
            "gpu_diagnostic_profile": (
                "stage" if gpu_stage_enabled else "unsupported-for-mode"
            ),
        },
        "environment": {
            name: os.environ[name]
            for name in ENVIRONMENT_ALLOWLIST
            if name in os.environ
        },
        "system": system_metadata(repo),
        "git": {"before": pre_state},
        "commands": commands,
    }
    manifest_path = artifact_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)

    runner = CommandRunner(commands)
    commands_by_phase = make_commands(args, repo, build_dir, artifact_dir)
    failure: str | None = None
    source_changed = False
    try:
        runner.run(
            commands_by_phase["configure"],
            cwd=repo,
            log_path=artifact_dir / "configure.log",
        )
        runner.run(
            commands_by_phase["build"],
            cwd=repo,
            log_path=artifact_dir / "build.log",
        )
        manifest["build_artifacts"] = verify_artifacts(
            required_artifacts(build_dir)
        )
        atomic_write_json(manifest_path, manifest)
        runner.run(
            commands_by_phase["benchmark"],
            cwd=repo,
            log_path=artifact_dir / "benchmark.stdout",
        )
        if gpu_stage_enabled:
            runner.run(
                commands_by_phase["gpu_profile"],
                cwd=repo,
                log_path=artifact_dir / "gpu-stage-profile.stdout",
            )
            gpu_stage_payload = json.loads(
                (artifact_dir / "gpu-stage-samples.json").read_text(
                    encoding="utf-8"
                )
            )
            atomic_write_json(
                artifact_dir / "gpu-stage-summary.json",
                aggregate_gpu_stage_samples(gpu_stage_payload),
            )
        runner.run(
            commands_by_phase["trace"],
            cwd=repo,
            log_path=artifact_dir / "xctrace.log",
        )
        runner.run(
            commands_by_phase["export"],
            cwd=repo,
            log_path=artifact_dir / "xctrace-export.log",
        )
        expected_outputs = [
            artifact_dir / "raw-samples.json",
            artifact_dir / "trace-sample.json",
            artifact_dir / "capture.trace",
            artifact_dir / "trace-toc.xml",
            artifact_dir / "trace.stdout",
        ]
        if gpu_stage_enabled:
            expected_outputs.extend(
                (
                    artifact_dir / "gpu-stage-samples.json",
                    artifact_dir / "gpu-stage-summary.json",
                )
            )
        for path in expected_outputs:
            if not path.exists() or (path.is_file() and path.stat().st_size == 0):
                raise RuntimeError(f"Expected profiling output is missing: {path}")
        json.loads((artifact_dir / "raw-samples.json").read_text(encoding="utf-8"))
        json.loads((artifact_dir / "trace-sample.json").read_text(encoding="utf-8"))
        if gpu_stage_enabled:
            json.loads(
                (artifact_dir / "gpu-stage-samples.json").read_text(
                    encoding="utf-8"
                )
            )
            json.loads(
                (artifact_dir / "gpu-stage-summary.json").read_text(
                    encoding="utf-8"
                )
            )
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        failure = str(error)
    finally:
        try:
            post_state = capture_git_state(
                repo, excluded_roots=state_exclusions
            )
            manifest["git"]["after"] = post_state
            source_changed = (
                post_state["source_fingerprint"]
                != pre_state["source_fingerprint"]
            )
            manifest["git"]["source_changed_during_run"] = source_changed
        except (OSError, RuntimeError) as error:
            manifest["git"]["after_error"] = str(error)
            source_changed = True
        if source_changed and failure is None:
            failure = "Source checkout changed during profiling"
        manifest["finished_at"] = utc_now()
        manifest["status"] = "failed" if failure else "complete"
        if failure:
            manifest["error"] = failure
        atomic_write_json(manifest_path, manifest)

    if failure:
        print(f"Metal profiling failed: {failure}", file=sys.stderr)
        print(f"Partial artifact preserved at {artifact_dir}", file=sys.stderr)
        return 1
    print(f"Metal profiling artifact: {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
