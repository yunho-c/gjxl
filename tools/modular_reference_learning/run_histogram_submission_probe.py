#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the fixed synthetic histogram batches under a recorded power condition."""
import argparse
import json
from pathlib import Path
import subprocess as sp
import time

from run_gpu_histogram_wait_study import competitors, sha
from run_histogram_attribution import snapshot, source_files
from run_hybrid_qualification import mac_dependencies, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--power", choices=("ac", "battery"), default="ac")
    args = parser.parse_args()
    require(0 < args.repeats <= 1000, "Invalid repetitions")
    repo = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    _, dependencies = mac_dependencies(binary)
    report = dict(status="running", binary=str(binary), binary_sha256=sha(binary),
                  sources=source_files(repo), dependencies=dependencies,
                  power=args.power, attempts=[])
    expected_power = "'AC Power'" if args.power == "ac" else "'Battery Power'"
    def save():
        (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        for attempt in range(10):
            while active := competitors():
                print("Waiting for competing work", json.dumps(active), flush=True)
                time.sleep(10)
            before = snapshot()
            require(expected_power in before["power"], "Wrong power condition")
            result_path = out / f"attempt-{attempt}.json"
            command = [str(binary), str(result_path), str(args.repeats)]
            activity = []
            with (out / f"attempt-{attempt}.log").open("x") as log:
                process = sp.Popen(command, stdout=log, stderr=sp.STDOUT)
                while process.poll() is None:
                    activity.extend(competitors({process.pid}))
                    time.sleep(.1)
            after = snapshot()
            observation = dict(command=command, before=before, after=after,
                               competitors=activity, returncode=process.returncode,
                               result=str(result_path),
                               sha256=sha(result_path) if result_path.exists() else None)
            report["attempts"].append(observation)
            save()
            require(process.returncode == 0, "Probe failed")
            require(expected_power in after["power"], "Power changed during probe")
            result = json.loads(result_path.read_text())
            require(result["passed"] and len(result["records"]) == 12 * (args.repeats + 3),
                    "Probe coverage differs")
            for r in result["records"]:
                t = r["wait_trace"]
                require(r["exact"] and t["commit_begin"] <= t["kernel_begin"] <=
                        t["kernel_end"] <= t["blocking_end"], "Invalid driver timing")
                require(t["commit_begin"] <= t["gpu_begin"] <= t["gpu_end"] <=
                        t["blocking_end"], "Invalid GPU timing")
                require(r["compute_encoders"] == (1 + 4*r["waves"] if r["variant"] & 2 else 1),
                        "Wrong encoder count")
                if not r["variant"] & 1:
                    require(r["reused_buffers"] == 0, "Fresh variant reused backing")
                elif not r["warmup"]:
                    require(r["reused_buffers"] == 7, "Fixed reuse variant lost backing")
            if activity:
                continue
            report["accepted"] = str(result_path)
            require(report["sources"] == source_files(repo), "Source changed during probe")
            require(report["binary_sha256"] == sha(binary), "Binary changed during probe")
            for path, digest in dependencies.items():
                require(sha(path) == digest, "Dependency changed during probe")
            report["status"] = "passed"
            return
        raise RuntimeError("No isolated probe attempt")
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
