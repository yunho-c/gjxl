#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Qualify the gathered Modular branches against a frozen staging baseline."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess as sp
import tarfile
import time

from run_histogram_attribution import snapshot, source_files
from run_hybrid_qualification import mac_dependencies, require
from run_sample_rows_qualification import competing, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--baseline-repo", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--large-inputs", type=Path, required=True)
    p.add_argument("--small-inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    require(args.repeats > 0, "At least one measured repetition is required")
    repo = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", started_unix=time.time(), repeats=args.repeats,
                  boundary="Complete private encoder call including cleanup; excludes backend/domain creation, file IO and independent pinned-libjxl decode. 14 participants, 8 GiB managed domain, Best predictors unless stated. One warmup and interleaved measured rounds for large inputs; small cases are correctness-only.",
                  binaries={}, sources={}, cases=[], rejected=[])
    dependencies = {}
    for label, binary, root in [("baseline", args.baseline, args.baseline_repo),
                                ("candidate", args.candidate, repo)]:
        root = root.resolve()
        files = source_files(root)
        archive = out / f"{label}-source.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name in files:
                tar.add(root / name, arcname=name, recursive=False)
        report["sources"][label] = dict(
            root=str(root), files=files, archive_sha256=sha(archive),
            revision=sp.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            tree=sp.check_output(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True).strip())
        (out / f"{label}-source.patch").write_bytes(sp.check_output(
            ["git", "-C", str(root), "diff", "--binary", "HEAD"]))
        shutil.copy2(binary.parent / "CMakeCache.txt", out / f"{label}-CMakeCache.txt")
        frozen = out / label
        sp.run(["cp", "-c", str(binary.resolve()), str(frozen)], check=True)
        listing, hashes = mac_dependencies(frozen)
        dependencies.update(hashes)
        report["binaries"][label] = dict(path=str(frozen), sha256=sha(frozen), libraries=listing)
    report["dependencies"] = dependencies
    variants = [
        ("baseline-cpu", "baseline", "cpu", []),
        ("integrated-cpu", "candidate", "cpu", []),
        ("baseline-metal", "baseline", "auto", []),
        ("integrated-metal", "candidate", "auto", []),
        ("cpu-unique", "candidate", "cpu", ["--sample-rows=raster-unique"]),
        ("gpu-fit-rows", "candidate", "auto",
         ["fitter-gpu", "16384", "64", "--sample-rows=metal-unique"]),
        ("pipeline-rows", "candidate", "auto",
         ["--learning-schedule=pipeline", "--feature-window=16", "--sample-rows=metal-unique"]),
    ]
    report["variants"] = variants
    inputs = [(f"photo-large-{i}", args.large_inputs / f"photo-large-{i}.pam", "auto", "best", True)
              for i in (1029, 1039, 1049)]
    inputs += [(name, args.small_inputs / f"{name}.pam", buffering, predictors, False)
               for name, buffering, predictors in [("photo-macan", "auto", "best"),
                   ("alpha-ria", "auto", "best"), ("palette-358", "auto", "best"),
                   ("alpha16-ramp", "auto", "all"), ("photo-macan", "buffered", "all")]]

    def save():
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(case, variant, repeat):
        label, binary, route, tail = variant
        tag = f"{case['name']}-{case['buffering']}-{case['predictors']}-{label}-r{repeat}"
        for attempt in range(3):
            require(not competing(), "Competing codec/compiler work before capture")
            before = snapshot()
            require("'AC Power'" in before["power"], "Capture requires AC power")
            directory = out / f"{tag}-a{attempt}"
            command = [report["binaries"][binary]["path"], case["input"], str(directory),
                       route, case["buffering"], case["predictors"], "fit-off", *(["14"] * 5), *tail]
            activity = []
            with (out / f"{tag}-a{attempt}.log").open("x") as log:
                process = sp.Popen(command, stdout=log, stderr=sp.STDOUT)
                while process.poll() is None:
                    activity.extend(competing({process.pid}))
                    time.sleep(.25)
            after = snapshot()
            require(process.returncode == 0, f"Encode failed: {tag}")
            require("'AC Power'" in after["power"], "Power changed during capture")
            result = json.loads((directory / "result.json").read_text())
            output = directory / "output.jxl"
            digest = sha(output)
            require(result["exact_decode"], f"Decode mismatch: {tag}")
            if "output_sha256" in case:
                require(case["output_sha256"] == digest and case["work"] == result["work"],
                        f"Codestream or learning decisions changed: {tag}")
            else:
                case.update(output_sha256=digest, work=result["work"], canonical=str(output))
            observation = dict(variant=label, repetition=repeat, warmup=case["timed"] and repeat == 0,
                               command=command, directory=str(directory), before=before, after=after,
                               competitors=activity, output_sha256=digest, result=result)
            if str(output) != case["canonical"]:
                # Only replace this run's verified duplicate with an APFS clone.
                output.unlink()
                sp.run(["cp", "-c", case["canonical"], str(output)], check=True)
            if activity:
                report["rejected"].append(observation)
                save()
                print("Excluded overlapping call", tag, flush=True)
                continue
            case["runs"].append(observation)
            save()
            print(tag, f"{result['total']['wall']:.6f}s exact", flush=True)
            return
        raise RuntimeError(f"No isolated observation for {tag}")

    try:
        for name, source, buffering, predictors, timed in inputs:
            case = dict(name=name, input=str(source.resolve()), input_sha256=sha(source),
                        buffering=buffering, predictors=predictors, timed=timed, runs=[])
            report["cases"].append(case)
            for repeat in range(args.repeats + 1 if timed else 1):
                order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
                if repeat % 2:
                    order = list(reversed(order))
                for variant in order:
                    run(case, variant, repeat)
            if timed:
                case["summary"] = {}
                for label, *_ in variants:
                    results = [r["result"] for r in case["runs"]
                               if r["variant"] == label and not r["warmup"]]
                    walls = [r["total"]["wall"] for r in results]
                    case["summary"][label] = dict(
                        wall=statistics.median(walls), wall_range=[min(walls), max(walls)],
                        bytes=results[0]["bytes"],
                        peak_backing=max(r["memory"]["peak_backing_bytes"] for r in results),
                        peak_committed=max(r["memory"]["peak_committed_bytes"] for r in results),
                        peak_feature_bytes=max(r["memory"]["peak_feature_bytes"] for r in results))
            save()
        for label, provenance in report["sources"].items():
            require(source_files(Path(provenance["root"])) == provenance["files"], f"{label} sources changed")
        for provenance in report["binaries"].values():
            require(sha(provenance["path"]) == provenance["sha256"], "Frozen binary changed")
        for path, digest in dependencies.items():
            require(sha(path) == digest, "Dependency changed")
        for case in report["cases"]:
            require(sha(case["input"]) == case["input_sha256"], "Input changed")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["finished_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
