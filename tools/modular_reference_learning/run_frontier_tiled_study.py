#!/usr/bin/env python3
"""Matched complete-fit replays of the monolithic and tiled frontier executors."""
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
    binary = (args.output / "probe").resolve()
    shutil.copy2(args.probe, binary)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    archive = args.archive.resolve(strict=True)
    configurations = [
        ("frontier128", "cpu-frontier", 128, 64),
        ("frontier512", "cpu-frontier", 512, 256),
        ("tiled128", "cpu-tiled", 128, 64),
        ("tiled512", "cpu-tiled", 512, 64),
        ("tiled1024", "cpu-tiled", 1024, 64),
    ]
    record = dict(schema=1, status="running", binary=str(binary), binary_sha256=sha(binary),
                  archive=str(archive), archive_sha256=sha(archive), power=args.power,
                  source_hashes=source_files(repo), before=snapshot(), configurations=configurations, runs=[])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GJXL_", "MTL_"))}

    def save():
        (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    try:
        for block, order in enumerate((configurations, list(reversed(configurations)), configurations[2:] + configurations[:2])):
            for name, mode, cap, scratch in order:
                if competitors():
                    raise RuntimeError("Competing codec/build job is active")
                before = snapshot()
                if f"'{args.power} Power'" not in before["power"]:
                    raise RuntimeError("Requested power condition is not active")
                path = (args.output / f"block-{block}-{name}.json").resolve()
                command = [str(binary), str(archive), str(path), "16", "2", "128", mode, str(cap), "16", str(scratch)]
                run = dict(block=block, name=name, command=command, before=before,
                           started=time.time(), power=[], competitors=[])
                record["runs"].append(run)
                save()
                with path.with_suffix(".log").open("x") as log:
                    child = sp.Popen(command, env=env, stdout=log, stderr=sp.STDOUT)
                    try:
                        deadline = time.monotonic() + 120
                        while child.poll() is None:
                            if time.monotonic() > deadline:
                                raise RuntimeError("Replay timed out")
                            run["power"].append(sp.check_output(["pmset", "-g", "batt"], text=True))
                            run["competitors"] += competitors({child.pid})
                            time.sleep(.5)
                    finally:
                        if child.poll() is None:
                            child.kill()
                            child.wait()
                run.update(exit_code=child.returncode, after=snapshot(), finished=time.time())
                save()
                if child.returncode or run["competitors"]:
                    raise RuntimeError("Failed or interfered replay")
                if any(f"'{args.power} Power'" not in x for x in run["power"] + [run["after"]["power"]]):
                    raise RuntimeError("Power changed")
                result = json.loads(path.read_text())
                gpu_mode = "frontier" if mode == "cpu-frontier" else "tiled_frontier"
                gpu = [x for x in result["observations"] if x["mode"] == gpu_mode]
                if not result["passed"] or not all(x["exact"] for x in result["observations"]):
                    raise RuntimeError("Tree parity failed")
                if result["planned_active_nodes"] != cap or not gpu or any(not 0 < x["maximum_active_nodes"] <= cap for x in gpu):
                    raise RuntimeError("Active capacity differs from admission")
                # A fixed histogram pool can limit how quickly children become
                # ready; capacity is not a promise that all slots will fill.
                run.update(result=str(path), result_sha256=sha(path),
                           filled_capacity=all(x["maximum_active_nodes"] == cap for x in gpu))
                save()
                median = statistics.median(x["wall_seconds"] for x in gpu if not x["warmup"])
                print(f"block {block}, {name}: {median:.6f} s, exact", flush=True)
        if sha(binary) != record["binary_sha256"] or sha(archive) != record["archive_sha256"]:
            raise RuntimeError("Frozen inputs changed")
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["after"] = snapshot()
        save()


if __name__ == "__main__":
    main()
