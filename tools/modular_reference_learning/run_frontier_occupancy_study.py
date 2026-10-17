#!/usr/bin/env python3
"""Bounded, matched prepared-sample frontier admission experiment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess as sp
import time

from run_gpu_histogram_wait_study import competitors, sha
from run_histogram_attribution import snapshot, source_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--power", choices=("AC", "Battery"), default="AC")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.output / "probe"
    shutil.copy2(args.probe, binary)
    binary = binary.resolve()
    archive = args.archive.resolve()
    record = dict(schema=1, status="running", head=sp.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        source_hashes=source_files(repo), binary_sha256=sha(binary),
        archive_sha256=sha(archive), power=args.power, before=snapshot(), runs=[])
    def save():
        (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GJXL_", "MTL_"))}
    orders = ((32, 64, 128, 256, 512), (512, 256, 128, 64, 32), (128, 32, 512, 64, 256))
    try:
        save()
        for block, order in enumerate(orders):
            for cap in order:
                if competitors():
                    raise RuntimeError("Another benchmark or build is active")
                if f"'{args.power} Power'" not in sp.check_output(["pmset", "-g", "batt"], text=True):
                    raise RuntimeError("Requested power source is not active")
                result_path = (args.output / f"block-{block}-cap-{cap}.json").resolve()
                command = [str(binary), str(archive), str(result_path), "16", "2", "128",
                           "cpu-frontier", str(cap), "16", "256"]
                run = dict(block=block, active_limit=cap, command=command, before=snapshot(),
                           started=time.time(), power_samples=[], competitors=[])
                record["runs"].append(run)
                save()
                with result_path.with_suffix(".log").open("w") as log:
                    child = sp.Popen(command, env=env, stdout=log, stderr=sp.STDOUT)
                    try:
                        deadline = time.monotonic() + 90
                        while child.poll() is None:
                            if time.monotonic() > deadline:
                                raise RuntimeError("Fitting replay timed out")
                            run["power_samples"].append(sp.check_output(["pmset", "-g", "batt"], text=True))
                            run["competitors"] += competitors(exclude=[child.pid])
                            time.sleep(0.5)
                        run["exit_code"] = child.returncode
                    finally:
                        if child.poll() is None:
                            child.kill()
                            child.wait()
                if run["exit_code"]:
                    raise RuntimeError(f"Replay failed: {result_path}")
                result = json.loads(result_path.read_text())
                if not result["passed"] or not all(x["exact"] for x in result["observations"]):
                    raise RuntimeError("Tree equality failed")
                if result["planned_active_nodes"] != cap:
                    raise RuntimeError("Requested admission was clipped by storage limits")
                if any(f"'{args.power} Power'" not in x for x in run["power_samples"]):
                    raise RuntimeError("Power source changed during replay")
                if run["competitors"]:
                    raise RuntimeError("Competing work detected during replay")
                run.update(finished=time.time(), after=snapshot(), result_sha256=sha(result_path))
                save()
                times = [x["wall_seconds"] for x in result["observations"]
                         if x["mode"] == "frontier" and not x["warmup"]]
                print(f"block {block}, cap {cap}: {statistics.median(times):.6f} s, exact", flush=True)
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["after"] = snapshot()
        save()


if __name__ == "__main__":
    main()
