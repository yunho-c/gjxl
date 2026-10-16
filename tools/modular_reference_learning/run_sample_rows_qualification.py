#!/usr/bin/env python3
"""Paired private-workflow row experiment; preserves raw runs and exact hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def competing(exclude=()):
    """Detect other codec qualifications/builds; ordinary desktop work is allowed."""
    found = []
    for line in subprocess.check_output(["ps", "-axo", "pid=,ppid=,args="], text=True).splitlines():
        pid, ppid, args = line.strip().split(None, 2)
        if int(pid) in {os.getpid(), *exclude} or "ps -axo" in args:
            continue
        executable = Path(args.split()[0]).name
        runner = ("python" in executable.lower() or executable == "Python") and (
            "qualification.py" in args or "run_histogram_attribution.py" in args)
        binary = executable.startswith("gjxl_") or executable in {"ctest", "ninja", "clang", "clang++", "metal"}
        build = executable == "cmake" and "--build" in args
        if runner or binary or build:
            found.append({"pid": int(pid), "command": args})
    return found


def quiet():
    while active := competing():
        print("Waiting for other work:", json.dumps(active), flush=True)
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--participants", type=int, default=14)
    parser.add_argument("--predictors", choices=["best", "all"], default="best")
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.input = [p.resolve() for p in args.input]
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "binary": str(args.binary), "binary_sha256": sha(args.binary),
        "inputs": [{"path": str(p), "sha256": sha(p)} for p in args.input],
        "power": subprocess.check_output(["pmset", "-g", "batt"], text=True),
        "device": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "participants": args.participants, "repeats": args.repeats, "predictors": args.predictors,
        "timing": "EncodeReferenceModularImage caller wall; excludes input I/O, backend creation, output I/O and independent decode",
        "features": "same auto Metal weighted-feature route and GPU tokenization for all row variants",
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.out / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"]))
    # Preserve newly introduced implementation files too, without staging them.
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
    for name in untracked:
        source = Path(name)
        if source.is_file() and name.startswith(("src/", "tests/", "tools/")):
            destination = args.out / "new_sources" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    runs, hashes = [], {}
    modes = ["raster", "selected", "metal"]
    for repeat in range(args.repeats):
        for source in args.input:
            for mode in modes[repeat % 3:] + modes[:repeat % 3]:
                attempt = 0
                while True:
                    quiet()
                    directory = args.out / f"{source.stem}-{mode}-{repeat}-{attempt}"
                    command = [str(args.binary), str(source), str(directory), "auto", "auto", args.predictors,
                               "fit-off", *([str(args.participants)] * 5), mode]
                    print("Running", source.name, mode, repeat, "attempt", attempt, flush=True)
                    collisions = []
                    started = time.time()
                    with (args.out / f"{directory.name}.log").open("w") as log:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                        while process.poll() is None:
                            collisions.extend(competing({process.pid}))
                            time.sleep(.5)
                    if process.returncode:
                        raise RuntimeError(f"Failed {command}: see {directory.name}.log")
                    result = json.loads((directory / "result.json").read_text())
                    digest = sha(directory / "output.jxl")
                    reference = hashes.setdefault(str(source), digest)
                    if digest != reference or not result["exact_decode"]:
                        raise RuntimeError(f"Row route changed encoded bytes: {directory}")
                    item = {"input": str(source), "mode": mode, "repeat": repeat, "attempt": attempt,
                            "directory": str(directory), "command": command, "started_unix": started,
                            "output_sha256": digest, "competitors": collisions, "result": result}
                    runs.append(item)
                    (args.out / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
                    if not collisions:
                        break
                    print("Discarding overlapping timing; retaining evidence and retrying after idle", flush=True)
                    attempt += 1
    summary = []
    for source in args.input:
        row = {"input": str(source), "variants": {}}
        for mode in modes:
            results = [r["result"] for r in runs if r["input"] == str(source) and r["mode"] == mode and not r["competitors"]]
            walls = [r["total"]["wall"] for r in results]
            row["variants"][mode] = {
                "wall_seconds": walls, "median_wall": statistics.median(walls),
                "median_preparation_work": statistics.median(r["detail_work"]["sample_preparation"] for r in results),
                "median_learning_wall": statistics.median(r["detail_wall"]["learning"] for r in results),
                "median_row_gather_work": statistics.median(r["sample_rows"]["gather_work"] for r in results),
                "peak_backing_bytes": max(r["memory"]["peak_backing_bytes"] for r in results),
                "gpu_rows": results[0]["sample_rows"]["gpu_rows"],
                "gpu_calls": results[0]["sample_rows"]["gpu_calls"],
            }
        baseline = row["variants"]["raster"]["median_wall"]
        for mode in modes:
            row["variants"][mode]["wall_reduction_percent"] = 100 * (1 - row["variants"][mode]["median_wall"] / baseline)
        summary.append(row)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
