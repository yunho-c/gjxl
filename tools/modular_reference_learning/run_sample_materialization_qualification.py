#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Interleave frozen before/after binaries for native sample materialization."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile
import time

from run_histogram_attribution import snapshot, source_files
from run_hybrid_qualification import mac_dependencies, require
from run_sample_rows_qualification import competing, quiet, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--modes", nargs="+", choices=["raster", "selected", "metal"],
                        default=["raster", "metal"])
    parser.add_argument("--route", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--predictors", choices=["best", "all"], default="best")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--participants", type=int, default=14)
    parser.add_argument("--candidate-metal-mode", choices=["metal", "metal-deferred"],
                        default="metal")
    parser.add_argument("--baseline-metal-mode", choices=["metal", "metal-deferred"],
                        default="metal")
    parser.add_argument("--candidate-unique-rows", action="store_true")
    args = parser.parse_args()
    require(args.repeats > 0 and args.warmups >= 0 and args.participants > 0,
            "Invalid repetition or participant count")
    require(args.route != "cpu" or "metal" not in args.modes,
            "CPU route cannot generate Metal rows")
    require(len(set(args.modes)) == len(args.modes), "Duplicate modes")
    repo = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = source_files(repo)
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
        for name in sources:
            archive.add(repo / name, arcname=name, recursive=False)
    (out / "source.patch").write_bytes(subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"]))
    shutil.copy2(args.binary.parent / "CMakeCache.txt", out / "CMakeCache.txt")
    binaries, dependencies = {}, {}
    for label, source in [("baseline", args.baseline), ("candidate", args.binary)]:
        frozen = out / f"gjxl_sample_materialization_{label}"
        shutil.copy2(source, frozen)
        listing, hashes = mac_dependencies(frozen)
        dependencies.update(hashes)
        binaries[label] = dict(path=str(frozen), sha256=sha(frozen), libraries=listing)
    manifest = dict(
        schema=1, status="running", started_unix=time.time(),
        baseline_revision=args.baseline_revision,
        candidate_revision=subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        source_files=sources, source_archive_sha256=sha(out / "source.tar.gz"),
        binaries=binaries, dependencies=dependencies,
        inputs=[dict(path=str(p.resolve()), sha256=sha(p)) for p in args.input],
        modes=args.modes, route=args.route, predictors=args.predictors,
        repeats=args.repeats, warmups=args.warmups, participants=args.participants,
        candidate_metal_mode=args.candidate_metal_mode,
        baseline_metal_mode=args.baseline_metal_mode,
        candidate_unique_rows=args.candidate_unique_rows,
        timing="EncodeReferenceModularImage caller wall, excluding backend creation, input/output I/O and independent decode; preparation/replay are overlapping per-scope wall sums",
        observations=[])
    expected = {}

    def save():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def call(source, mode, label, repetition):
        attempt = 0
        while True:
            quiet()
            directory = out / f"{source.stem}-{mode}-{label}-{repetition}-{attempt}"
            actual_mode = mode
            if mode == "metal":
                actual_mode = args.candidate_metal_mode if label == "candidate" else args.baseline_metal_mode
            if label == "candidate" and args.candidate_unique_rows:
                actual_mode = mode + "-unique"
            argv = [binaries[label]["path"], str(source), str(directory), args.route,
                    "auto", args.predictors, "fit-off", *([str(args.participants)] * 5), actual_mode]
            before = snapshot()
            require("'AC Power'" in before["power"], f"AC power required: {before['power']}")
            collisions = []
            with (out / (directory.name + ".log")).open("x") as log:
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    collisions.extend(competing({process.pid}))
                    time.sleep(.5)
            after = snapshot()
            require(process.returncode == 0, f"Encode failed: {directory.name}")
            require("'AC Power'" in after["power"], f"Power changed during encode: {after['power']}")
            result = json.loads((directory / "result.json").read_text())
            digest = sha(directory / "output.jxl")
            require(result["exact_decode"] and digest == expected.setdefault(str(source), digest),
                    f"Bytes or independent decode changed: {directory.name}")
            manifest["observations"].append(dict(
                input=str(source), mode=mode, build=label, repetition=repetition,
                warmup=repetition < args.warmups, attempt=attempt, command=argv,
                directory=str(directory), before=before, after=after,
                competitors=collisions, output_sha256=digest, result=result))
            save()
            # Retain identical output artifacts as APFS clones of our own first
            # output, after hashing and independent decoding each new encode.
            canonical = out / f"{source.stem}-{sha(source)[:12]}.jxl"
            if canonical.exists():
                require(sha(canonical) == digest, "Canonical output changed")
                (directory / "output.jxl").unlink()
                subprocess.run(["cp", "-c", str(canonical), str(directory / "output.jxl")], check=True)
            else:
                subprocess.run(["cp", "-c", str(directory / "output.jxl"), str(canonical)], check=True)
            print(directory.name, f"wall={result['total']['wall']:.3f}s "
                  f"prepare={result['detail_work']['sample_preparation']:.3f}s "
                  f"replay={result['sample_rows']['replay_work']:.3f}s exact", flush=True)
            if not collisions:
                return
            print("Excluded overlapping timing; waiting and retrying", flush=True)
            attempt += 1

    save()
    try:
        for repetition in range(args.warmups + args.repeats):
            for i, source in enumerate(args.input):
                modes = args.modes[repetition % len(args.modes):] + args.modes[:repetition % len(args.modes)]
                for j, mode in enumerate(modes):
                    order = ["baseline", "candidate"]
                    if (repetition + i + j) % 2:
                        order.reverse()
                    for label in order:
                        call(source.resolve(), mode, label, repetition)
        require(source_files(repo) == sources, "Source changed during capture")
        for info in binaries.values():
            require(sha(Path(info["path"])) == info["sha256"], "Frozen binary changed")
        for path, digest in dependencies.items():
            require(sha(Path(path)) == digest, "Linked dependency changed")
        for info in manifest["inputs"]:
            require(sha(Path(info["path"])) == info["sha256"], "Input changed")
        manifest["timing_summary"] = []
        for source in args.input:
            for mode in args.modes:
                row = dict(input=str(source.resolve()), mode=mode, variants={})
                for label in binaries:
                    results = [r["result"] for r in manifest["observations"]
                               if not r["warmup"] and not r["competitors"]
                               and (r["input"], r["mode"], r["build"]) ==
                               (str(source.resolve()), mode, label)]
                    columns = dict(wall=[r["total"]["wall"] for r in results],
                                   preparation=[r["detail_work"]["sample_preparation"] for r in results],
                                   gather=[r["sample_rows"]["gather_work"] for r in results],
                                   replay=[r["sample_rows"]["replay_work"] for r in results])
                    row["variants"][label] = {key: dict(values=values, median=statistics.median(values))
                                              for key, values in columns.items()}
                manifest["timing_summary"].append(row)
        manifest["status"] = "passed"
    except BaseException as error:
        manifest["status"], manifest["error"] = "failed", str(error)
        raise
    finally:
        manifest["finished_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
