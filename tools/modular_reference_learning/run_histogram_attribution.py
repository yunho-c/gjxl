#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Paired on/off attribution of the native histogram fitter in the hybrid route."""
import argparse
import json
import math
import platform
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from run_hybrid_qualification import command, digest, environment, mac_dependencies, require


def source_files(repo):
    names = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard",
         "--", "CMakeLists.txt", "cmake", "include", "src", "tests", "tools"], text=True)
    return {name: digest(repo / name) for name in sorted(set(names.split("\0")))
            if name and (repo / name).is_file()}


def snapshot():
    result = environment()
    if platform.system() == "Darwin":
        result["thermal"] = command(["pmset", "-g", "therm"])
        rows = command(["ps", "-axo", "pid=,pcpu=,comm="]).splitlines()
        result["top_processes"] = sorted(rows, key=lambda r: float(r.split()[1]), reverse=True)[:12]
    return result


def validate_profile(result):
    p = result["histogram_fitting"]
    require(p["enabled"] and p["schema"] in (1, 2, 3), "Missing fitting profile")
    require(p["calls"] == p["successes"] == result["work"]["fits"] == len(p["fits"])
            and not p["records_dropped"], "Incomplete fit coverage")
    require(all(f["success"] for f in p["fits"]), "Failed fit included")
    # Parallel profiles expose the summed per-job fitting envelope as work,
    # since those intervals overlap rather than subdividing caller wall time.
    fitting = result.get("detail_work", result["detail_wall"])["fitting"]
    require(p["fit_seconds"] <= fitting + 1e-6,
            "Fitter envelope exceeds workflow fitting interval")
    parallel = p.get("parallel_property_nodes", 0) > 0
    if not parallel:
        require(sum(p["stages"].values()) <= p["fit_seconds"] + 1e-6,
                "Overlapping serial fitting phases")
    require(all(math.isfinite(v) and v >= 0 for v in p["stages"].values()), "Invalid timing")
    nodes = sum(b["work"]["nodes"] for b in p["by_rows"])
    splits = sum(b["work"]["splits"] for b in p["by_rows"])
    require(nodes == sum(f["nodes"] for f in p["fits"]) == 2 * splits + p["calls"],
            "Node/split conservation failed")
    require(nodes == sum(b["nodes"] for b in p["by_depth"])
            == sum(b["count"] for b in p["nodes_by_symbol_span"]), "Node distributions differ")
    require(sum(f["rows"] for f in p["fits"]) == result["work"]["distinct_samples"],
            "Fit records lost sample rows")
    require(sum(f["population"] for f in p["fits"]) == result["work"]["selected_samples"],
            "Fit records lost weighted population")
    for b in p["by_rows"]:
        if not parallel:
            require(sum(b["stages"].values()) <= b["node_seconds"] + 1e-6, "Node phase overlap")
        w = b["work"]
        require(w["active_bins"] == w["candidate_thresholds"] + w["properties"],
                "Active bins and candidate thresholds disagree")
        if p["schema"] >= 2:
            require(w["fused_property_row_visits"] == w["property_row_visits"]
                    <= w["joint_histogram_row_visits"], "Invalid fused histogram visits")
            require(w["skipped_constant_properties"] <= w["constant_properties"]
                    <= w["properties"], "Invalid constant-property skips")
    for key in p["stages"]:
        require(abs(sum(f["stages"][key] for f in p["fits"]) - p["stages"][key]) < 1e-6,
                "Fit-stage records do not sum")


def signature(p):
    return {"rows": [(b["bucket"], b["work"]) for b in p["by_rows"]],
            "depth": [(b["depth"], b["nodes"], b["rows"]) for b in p["by_depth"]],
            "symbols": p["nodes_by_symbol_span"],
            "active_bins": p["properties_by_active_bins"], "bin_span": p["properties_by_bin_span"],
            "fits": [{k: v for k, v in f.items() if k not in ("seconds", "stages")} for f in p["fits"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Prior hybrid qualification JSON for exact-byte checks")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    require(args.repeats > 0 and args.warmups >= 0, "Invalid repetition count")
    repo = Path(__file__).resolve().parents[2]
    binary, out = args.binary.resolve(), args.out.resolve()
    inputs = [p.resolve() for p in args.input]
    require(binary.is_file() and all(p.is_file() for p in inputs), "Missing binary/input")
    require(len(set(p.stem for p in inputs)) == len(inputs), "Input names must be unique")
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / "hybrid_profile"
    if platform.system() == "Darwin":
        subprocess.run(["cp", "-c", str(binary), str(frozen)], check=True)
    else:
        shutil.copy2(binary, frozen)
    files = source_files(repo)
    # Uncommitted work is reviewable and reproducible: archive every build/tool
    # source and hash it before and after. No clean-checkout claim is made.
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
        for name in files:
            archive.add(repo / name, arcname=name, recursive=False)
    (out / "source.patch").write_text(command(["git", "-C", str(repo), "diff", "--binary", "HEAD"]) + "\n")
    reference = {}
    if args.reference:
        data = json.loads(args.reference.read_text())
        for row in data["full_resolution"]["runs"]:
            reference.setdefault(row["input"], row["output_sha256"])
        require(all(p.stem in reference for p in inputs), "Reference is missing an input")
        shutil.copy2(args.reference, out / "reference-results.json")
    manifest = {
        "schema": 1, "status": "running", "started_unix": time.time(),
        "source_revision": command(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "source_status": command(["git", "-C", str(repo), "status", "--porcelain=v1"]),
        "submodules": command(["git", "-C", str(repo), "submodule", "status"]),
        "source_files_sha256": files, "binary_sha256": digest(frozen),
        "binary_source": str(binary), "environment_start": snapshot(),
        "warmups": args.warmups, "repeats": args.repeats,
        "inputs": [{"name": p.stem, "path": str(p), "sha256": digest(p)} for p in inputs],
        "scope": "Auto hybrid, auto/local buffering, Best predictors, four CPU participants ceiling. "
                 "Identical binary with histogram attribution disabled/enabled. Fresh serial processes, "
                 f"rotating on/off order, {args.warmups} warmups per mode/input. Native encoder-call wall and process "
                 "CPU exclude IO/backend creation/independent decoding. Fitting phases are exclusive "
                 "wall intervals nested within the native fitter envelope and workflow fitting time. "
                 "Node size means distinct weighted rows. Candidate evaluation includes prefix histogram "
                 "updates/clears, entropy estimates and per-side predictor comparison. No per-threshold "
                 "clock calls. Uninstrumented control/cleanup and instrumentation overhead remain in "
                 "the envelope residual. Caller-owned diagnostic memory is outside codec admission.",
    }
    if platform.system() == "Darwin":
        manifest["linked_libraries"], manifest["dependency_sha256"] = mac_dependencies(frozen)
        manifest["power_settings"] = command(["pmset", "-g", "custom"])
    for name in ["CMakeCache.txt", "compile_commands.json"]:
        if (binary.parent / name).is_file():
            shutil.copy2(binary.parent / name, out / name)
    manifest_path = out / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    controls, work, rows = {}, {}, []
    try:
        with (out / "results.jsonl").open("x", buffering=1) as ledger:
            for image_index, source in enumerate(inputs):
                for iteration in range(args.warmups + args.repeats):
                    modes = ["fit-off", "fit-on"]
                    if (image_index + iteration) % 2:
                        modes.reverse()
                    for mode in modes:
                        name = f"{source.stem}-{mode}-{iteration}"
                        directory = out / name
                        argv = [str(frozen), str(source), str(directory), "auto", "auto", "best", mode]
                        before = snapshot()
                        started = time.time()
                        with (out / (name + ".stdout")).open("x") as stdout, (out / (name + ".stderr")).open("x") as stderr:
                            subprocess.run(argv, stdout=stdout, stderr=stderr, check=True)
                        finished = time.time()
                        after = snapshot()
                        result = json.loads((directory / "result.json").read_text())
                        checksum = digest(directory / "output.jxl")
                        require(result["exact_decode"] and result["route"] == "auto" and
                                result["buffering"] == "auto" and result["predictors"] == "best", "Wrong result policy")
                        require(result["histogram_fitting"]["enabled"] == (mode == "fit-on"), "Wrong profile mode")
                        routing = {k: v for k, v in result["routing"].items()
                                   if k != "cpu_feature_peak_participants"}
                        control = (checksum, result["bytes"], result["work"], routing)
                        require(controls.setdefault(source.stem, control) == control,
                                "Profiling changed bytes, routing or learning work")
                        if reference:
                            require(reference[source.stem] == checksum, "Bytes differ from prior qualified implementation")
                        if mode == "fit-on":
                            validate_profile(result)
                            sig = signature(result["histogram_fitting"])
                            require(work.setdefault(source.stem, sig) == sig, "Fitting work changed across repetitions")
                        row = {"input": source.stem, "mode": mode, "iteration": iteration,
                               "warmup": iteration < args.warmups, "started_unix": started,
                               "finished_unix": finished, "command": argv, "output_sha256": checksum,
                               "environment_before": before, "environment_after": after, "result": result}
                        ledger.write(json.dumps(row, sort_keys=True) + "\n")
                        rows.append(row)
                        print(name, f"total={result['total']['wall']:.6f}s",
                              f"fit={result['detail_wall']['fitting']:.6f}s", "exact", flush=True)
        require(source_files(repo) == files, "Build/tool source changed during capture")
        require(command(["git", "-C", str(repo), "submodule", "status"]) == manifest["submodules"],
                "Submodule revision changed")
        require(digest(frozen) == manifest["binary_sha256"], "Frozen binary changed")
        for item in manifest["inputs"]:
            require(digest(Path(item["path"])) == item["sha256"], "Input changed")
        for path, checksum in manifest.get("dependency_sha256", {}).items():
            require(digest(Path(path)) == checksum, "Linked dependency changed")
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
