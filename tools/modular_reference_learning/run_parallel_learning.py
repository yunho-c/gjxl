#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Matched local-learning scaling, with exact bytes and frozen build provenance."""
import argparse
import json
import platform
import shutil
import statistics
import subprocess
import tarfile
import time
from pathlib import Path

from run_histogram_attribution import snapshot, source_files
from run_hybrid_qualification import command, digest, mac_dependencies, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Prior gpu-histogram-learner-integration-results.json")
    parser.add_argument("--participants", nargs="+", type=int, default=[1, 4, 14])
    parser.add_argument("--cpu-ceiling", type=int, default=14)
    parser.add_argument("--route", choices=["cpu", "shared-cpu", "hybrid", "auto"], default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    require(args.repeats > 0 and args.warmups >= 0 and 1 <= args.cpu_ceiling <= 256,
            "Invalid repetition/CPU limit")
    require(1 in args.participants and len(set(args.participants)) == len(args.participants) and
            all(1 <= p <= args.cpu_ceiling for p in args.participants), "Invalid participant sweep")
    repo = Path(__file__).resolve().parents[2]
    binary, out = args.binary.resolve(), args.out.resolve()
    inputs = [p.resolve() for p in args.input]
    require(binary.is_file() and all(p.is_file() for p in inputs), "Missing binary/input")
    require(len(set(p.stem for p in inputs)) == len(inputs), "Input names must be unique")
    reference = {}
    for row in json.loads(args.reference.read_text())["full_resolution"]["runs"]:
        require(reference.setdefault(row["input"], row["output_sha256"]) == row["output_sha256"],
                "Conflicting prior output hashes")
    require(all(p.stem in reference for p in inputs), "Prior qualification is missing an input")
    initial = snapshot()
    require(platform.system() != "Darwin" or "'AC Power'" in initial["power"], "AC power required")
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / "hybrid_profile"
    if platform.system() == "Darwin":
        subprocess.run(["cp", "-c", str(binary), str(frozen)], check=True)
    else:
        shutil.copy2(binary, frozen)
    files = source_files(repo)
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
        for name in files:
            archive.add(repo / name, arcname=name, recursive=False)
    (out / "source.patch").write_text(command(["git", "-C", str(repo), "diff", "--binary", "HEAD"]) + "\n")
    shutil.copy2(args.reference, out / "reference-results.json")
    for name in ["CMakeCache.txt", "compile_commands.json"]:
        if (binary.parent / name).is_file():
            shutil.copy2(binary.parent / name, out / name)
    manifest = {
        "schema": 1, "status": "running", "started_unix": time.time(),
        "source_revision": command(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "source_status": command(["git", "-C", str(repo), "status", "--porcelain=v1"]),
        "submodules": command(["git", "-C", str(repo), "submodule", "status"]),
        "source_files_sha256": files, "binary_source": str(binary), "binary_sha256": digest(frozen),
        "source_archive_sha256": digest(out / "source.tar.gz"), "environment_start": initial,
        "participants": args.participants, "cpu_ceiling": args.cpu_ceiling, "route": args.route,
        "warmups": args.warmups, "repeats": args.repeats,
        "inputs": [{"name": p.stem, "path": str(p), "sha256": digest(p)} for p in inputs],
        "stages": ["input", "transforms", "training", "tokens", "model", "emission", "assembly", "gpu_input_tokens"],
        "scope": "Identical binary, automatic local trees, Best predictors, estimated direct ANS/64 clusters. "
                 "Fixed CPU domain ceiling and maximum four CPU feature participants; vary local "
                 "prepare-and-fit participants. Cohort alignment may change feature batch boundaries "
                 "and automatic producer routing; retain routing per condition. Fresh serial processes in rotating participant order, "
                 "one independently decoded output per call; all bytes checked against prior qualification. "
                 "Encoder-call wall/process CPU include admission, selection, features, local learning, "
                 "tokens, entropy and output publication/teardown; exclude input/output IO, backend "
                 "creation and independent decode. Detailed fitting probes disabled. detail_wall is "
                 "elapsed caller time; detail_work sums overlapping per-job wall intervals. "
                 "Power checked before/after each call; all observations retained. Managed memory "
                 "excludes borrowed packed input, decode buffers, stacks, allocator metadata and backend setup.",
    }
    if platform.system() == "Darwin":
        manifest["linked_libraries"], manifest["dependency_sha256"] = mac_dependencies(frozen)
        manifest["power_settings"] = command(["pmset", "-g", "custom"])
    manifest_path = out / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    rows, controls = [], {}
    try:
        with (out / "results.jsonl").open("x", buffering=1) as ledger:
            for image_index, source in enumerate(inputs):
                for iteration in range(args.warmups + args.repeats):
                    offset = (image_index + iteration) % len(args.participants)
                    order = args.participants[offset:] + args.participants[:offset]
                    for participants in order:
                        name = f"{source.stem}-p{participants}-{iteration}"
                        directory = out / name
                        argv = [str(frozen), str(source), str(directory), args.route, "auto", "best",
                                "fit-off", str(participants), str(args.cpu_ceiling)]
                        before = snapshot()
                        require(platform.system() != "Darwin" or "'AC Power'" in before["power"],
                                "Power changed before encode; capture stopped")
                        started = time.time()
                        with (out / (name + ".stdout")).open("x") as stdout, (out / (name + ".stderr")).open("x") as stderr:
                            subprocess.run(argv, stdout=stdout, stderr=stderr, check=True)
                        finished, after = time.time(), snapshot()
                        result = json.loads((directory / "result.json").read_text())
                        checksum = digest(directory / "output.jxl")
                        row = {"input": source.stem, "participants": participants, "iteration": iteration,
                               "warmup": iteration < args.warmups, "started_unix": started,
                               "finished_unix": finished, "command": argv, "output_sha256": checksum,
                               "environment_before": before, "environment_after": after, "result": result}
                        # Preserve an observation even if a subsequent qualification gate fails.
                        ledger.write(json.dumps(row, sort_keys=True) + "\n")
                        rows.append(row)
                        require(platform.system() != "Darwin" or "'AC Power'" in after["power"],
                                "Power changed during encode; capture stopped")
                        require(result["schema"] == 2 and result["exact_decode"] and
                                result["route"] == args.route and result["buffering"] == "auto" and
                                result["predictors"] == "best" and not result["histogram_fitting"]["enabled"],
                                "Wrong profile/policy result")
                        require(checksum == reference[source.stem], "Bytes differ from prior qualification")
                        control = (checksum, result["bytes"], result["work"])
                        require(controls.setdefault(source.stem, control) == control,
                                "Participant count changed bytes or learning work")
                        require(controls.setdefault((source.stem, participants), result["routing"]) == result["routing"],
                                "Repeated participant condition changed feature/token routing")
                        learning = result["learning"]
                        require(learning["requested_participants"] == participants and
                                learning["cpu_ceiling"] == args.cpu_ceiling and
                                learning["peak_participants"] <= participants,
                                "Wrong CPU participation")
                        require(not learning["memory_limited_waves"], "Sweep became memory constrained")
                        for kind in ["wall", "cpu"]:
                            values = result["profile_" + kind]
                            require(len(values) == len(manifest["stages"]) and min(values) >= -1e-6 and
                                    sum(values) <= result["total"][kind] + 1e-4,
                                    "Invalid top-level elapsed/process attribution")
                        require(result["detail_wall"]["learning"] + result["detail_wall"]["features"] <=
                                result["profile_wall"][2] + 1e-4, "Overlapping training wall intervals")
                        require(result["memory"]["peak_feature_samples"] <= 8 * 1024 * 1024,
                                "Live feature batch bound exceeded")
                        print(name, f"total={result['total']['wall']:.6f}s",
                              f"learning={result['detail_wall']['learning']:.6f}s", "exact", flush=True)
        require(source_files(repo) == files, "Build/tool source changed during capture")
        require(command(["git", "-C", str(repo), "submodule", "status"]) == manifest["submodules"],
                "Submodule revision changed")
        require(digest(frozen) == manifest["binary_sha256"], "Frozen binary changed")
        for item in manifest["inputs"]:
            require(digest(Path(item["path"])) == item["sha256"], "Input changed")
        for path, checksum in manifest.get("dependency_sha256", {}).items():
            require(digest(Path(path)) == checksum, "Linked dependency changed")
        summary = []
        for source in inputs:
            for participants in args.participants:
                values = [r["result"] for r in rows if r["input"] == source.stem and
                          r["participants"] == participants and not r["warmup"]]
                summary.append({
                    "input": source.stem, "participants": participants, "repeats": len(values),
                    "bytes": values[0]["bytes"], "output_sha256": reference[source.stem],
                    "median_wall": statistics.median(r["total"]["wall"] for r in values),
                    "range_wall": [min(r["total"]["wall"] for r in values), max(r["total"]["wall"] for r in values)],
                    "median_cpu": statistics.median(r["total"]["cpu"] for r in values),
                    "median_stage_wall": dict(zip(manifest["stages"],
                        [statistics.median(r["profile_wall"][i] for r in values) for i in range(8)])),
                    "median_detail_wall": {k: statistics.median(r["detail_wall"][k] for r in values)
                                           for k in values[0]["detail_wall"]},
                    "median_detail_work": {k: statistics.median(r["detail_work"][k] for r in values)
                                           for k in values[0]["detail_work"]},
                    "maximum_peak_backing_bytes": max(r["memory"]["peak_backing_bytes"] for r in values),
                    "maximum_peak_committed_bytes": max(r["memory"]["peak_committed_bytes"] for r in values),
                    "learning": values[0]["learning"], "routing": values[0]["routing"],
                })
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest["status"] = "passed"
    except BaseException as error:
        manifest["status"], manifest["error"] = "failed", str(error)
        raise
    finally:
        manifest["finished_unix"] = time.time()
        manifest["environment_finish"] = snapshot()
        manifest["completed_calls"] = len(rows)
        manifest["artifacts"] = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*"))
                                 if p.is_file() and p != manifest_path}
        save()


if __name__ == "__main__":
    main()
