#!/usr/bin/env python3
"""Interleaved complete-call qualification; keep every output and raw result."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess as sp
import time


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--large-inputs", type=Path, required=True)
    parser.add_argument("--small-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    root = Path.cwd()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("control", "candidate"):
        shutil.copy2(getattr(args, name), out / name)
    inputs = [(f"photo-large-{i}", args.large_inputs / f"photo-large-{i}.pam", "auto", "best")
              for i in (1029, 1039, 1049)]
    inputs += [(name, args.small_inputs / f"{name}.pam", buffering, predictors)
               for name, buffering, predictors in (("photo-macan", "auto", "best"),
                   ("alpha-ria", "auto", "best"), ("palette-358", "auto", "best"),
                   ("alpha16-ramp", "auto", "all"), ("photo-macan", "buffered", "all"))]
    variants = [("frozen-cohorts", "control", "auto", None, 8),
                ("cohorts", "candidate", "auto", "cohorts", 8),
                ("pipeline8", "candidate", "auto", "pipeline", 8),
                ("pipeline16", "candidate", "auto", "pipeline", 16),
                ("cpu", "candidate", "cpu", "cohorts", 8)]
    report = {"status": "running", "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "base_commit": sp.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "binaries": {n: sha(out / n) for n in ("control", "candidate")},
              "repetitions": args.repetitions,
              "machine": sp.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
              "boundary": "Complete private encode including cleanup; excludes backend/domain creation, file IO and independent exact decode. 8 GiB domain, 14 CPU participants for each stage. Feature CPU request 4; admitted counts are recorded. One warmup per variant/input, alternating measured order.",
              "cases": [], "rejected": []}
    (out / "source.patch").write_bytes(sp.check_output(["git", "diff", "--binary", "HEAD"]))
    paths = sp.check_output(["git", "ls-files", "-m", "-o", "--exclude-standard"], text=True).splitlines()
    report["sources"] = {}
    for path in sorted(set(paths)):
        if Path(path).is_file():
            destination = out / "sources" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            report["sources"][path] = sha(path)

    def save():
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    controllers = set()

    def competitors(excluded=0):
        table = {}
        for line in sp.check_output(["ps", "-ww", "-axo", "pid=,ppid=,args="], text=True).splitlines():
            fields = line.strip().split(None, 2)
            if len(fields) == 3:
                pid, parent, command = fields
                table[int(pid)] = (int(parent), command, line)
        busy = []
        for pid, (parent, command, line) in table.items():
            if pid in (os.getpid(), excluded):
                continue
            executable = command.split(None, 1)[0]
            name = Path(executable).name
            heavy = ("gjxl" in executable and ("/build/" in executable or name.startswith("gjxl_")))
            heavy |= name in ("cjxl", "djxl", "reference_learning", "clang", "clang++", "ninja", "ld", "ctest")
            if heavy:
                ancestor = parent
                while ancestor in table and ancestor != os.getpid():
                    ancestor_parent, ancestor_command, _ = table[ancestor]
                    if "python" in Path(ancestor_command.split(None, 1)[0]).name.lower():
                        controllers.add(ancestor)
                    ancestor = ancestor_parent
            if heavy or pid in controllers:
                busy.append(line)
        return busy

    def power():
        state = sp.check_output(["pmset", "-g", "batt"], text=True)
        if "AC Power" not in state:
            raise RuntimeError("Qualification requires AC power")
        return state

    def run(source, buffering, predictors, variant, tag):
        label, binary, route, schedule, window = variant
        for attempt in range(10):
            start_wait = time.monotonic()
            while competitors():
                if time.monotonic() - start_wait > 1800:
                    raise RuntimeError("No quiet qualification interval within 30 minutes")
                time.sleep(1)
            destination = out / f"{tag}-attempt{attempt}"
            destination.mkdir()
            command = [str(out / binary), str(source.resolve()), str(destination / "encode"),
                       route, buffering, predictors, "fit-off", "14", "14", "14", "14", "14",
                       "fitter-cpu", "16384", "64"]
            if schedule:
                command += [schedule, str(window)]
            before = power()
            activity = []
            with (destination / "run.log").open("w") as log:
                process = sp.Popen(command, stdout=log, stderr=sp.STDOUT)
                while process.poll() is None:
                    busy = competitors(process.pid)
                    if busy:
                        activity.append({"time": time.time(), "processes": busy})
                    time.sleep(.2)
            (destination / "activity.json").write_text(json.dumps(activity, indent=2) + "\n")
            after = power()
            if process.returncode:
                raise RuntimeError(f"Encode failed: {destination}")
            result = json.loads((destination / "encode/result.json").read_text())
            if not result["exact_decode"]:
                raise RuntimeError(f"Decode mismatch: {destination}")
            result.update({"variant": label, "command": command, "power_before": before,
                           "power_after": after, "artifact": str(destination),
                           "output_sha256": sha(destination / "encode/output.jxl")})
            if not activity:
                return result
            report["rejected"].append(result)
            save()
            print("Rejected overlapping run", tag, flush=True)
        raise RuntimeError("Unable to obtain an isolated run")

    try:
        for name, source, buffering, predictors in inputs:
            case = {"name": name, "buffering": buffering, "predictors": predictors,
                    "input": str(source.resolve()), "input_sha256": sha(source), "runs": []}
            report["cases"].append(case)
            expected = None
            for repeat in range(args.repetitions + 1):
                for variant in (variants if repeat % 2 == 0 else list(reversed(variants))):
                    tag = f"{name}-{buffering}-{predictors}-{variant[0]}-r{repeat}"
                    result = run(source, buffering, predictors, variant, tag)
                    expected = expected or result["output_sha256"]
                    if result["output_sha256"] != expected:
                        raise RuntimeError(f"Codestream mismatch: {tag}")
                    result.update({"repeat": repeat, "warmup": repeat == 0})
                    case["runs"].append(result)
                    save()
                    print(tag, round(result["total"]["wall"], 6), flush=True)
            case["medians"] = {}
            for label, *_ in variants:
                runs = [r for r in case["runs"] if r["variant"] == label and not r["warmup"]]
                walls = [r["total"]["wall"] for r in runs]
                case["medians"][label] = {"wall": statistics.median(walls),
                    "wall_range": [min(walls), max(walls)], "bytes": runs[0]["bytes"],
                    "peak_backing": max(r["memory"]["peak_backing_bytes"] for r in runs),
                    "peak_committed": max(r["memory"]["peak_committed_bytes"] for r in runs),
                    "peak_feature_bytes": max(r["memory"]["peak_feature_bytes"] for r in runs)}
            save()
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save()
    print("Complete", out, flush=True)


if __name__ == "__main__":
    main()
