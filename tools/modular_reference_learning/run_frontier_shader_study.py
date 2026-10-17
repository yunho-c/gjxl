#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Interleave exact complete-fit shader/admission comparisons on one archive.

Configurations JSON: [{"name": ..., "metallib": ..., "active": 128,
"scratch_mib": 64, "rounds": 16}]. Each process includes CPU14 controls,
one warmup and the requested measured repetitions. Diagnostic captures and
validation layers are excluded. Outputs and frozen inputs are never replaced.
"""
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
from run_hybrid_qualification import mac_dependencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("configurations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--power", choices=("AC", "Battery"), default="AC")
    args = parser.parse_args()
    if not 1 <= args.blocks <= 10 or not 1 <= args.repeats <= 30:
        parser.error("Invalid repetition count")
    repo = Path(__file__).resolve().parents[2]
    configurations = json.loads(args.configurations.read_text())
    names = [c["name"] for c in configurations]
    if not names or len(names) != len(set(names)) or any(not n.replace("-", "").replace("_", "").isalnum() for n in names):
        parser.error("Configuration names must be unique safe filenames")
    args.output.mkdir(parents=True, exist_ok=False)
    binary = (args.output / "gjxl_frontier_probe").resolve()
    shutil.copy2(args.probe, binary)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    frozen = {str(binary): sha(binary)}
    for c in configurations:
        library = (args.output / (c["name"] + ".metallib")).resolve()
        shutil.copy2(c["metallib"], library)
        c.update(metallib_source=str(Path(c["metallib"]).resolve()), metallib=str(library))
        frozen[str(library)] = sha(library)
    archive = args.archive.resolve(strict=True)
    frozen[str(archive)] = sha(archive)
    listing, dependencies = mac_dependencies(binary)
    frozen.update(dependencies)
    record = dict(schema=1, status="running", head=sp.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  binary=str(binary), archive=str(archive), frozen_sha256=frozen, configurations=configurations,
                  dependency_listing=listing, source_hashes=source_files(repo), before=snapshot(),
                  power=args.power, blocks=args.blocks, repeats=args.repeats, runs=[])
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("GJXL_", "MTL_"))}

    def save():
        (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    expected = None
    try:
        for block in range(args.blocks):
            order = configurations if block % 3 == 0 else list(reversed(configurations)) if block % 3 == 1 else configurations[1:] + configurations[:1]
            for c in order:
                if competitors():
                    raise RuntimeError("Competing codec/build/profiler job")
                before = snapshot()
                if f"'{args.power} Power'" not in before["power"]:
                    raise RuntimeError("Requested power condition is not active")
                output = (args.output / f"block-{block}-{c['name']}.json").resolve()
                command = [str(binary), str(archive), str(output), "16", str(args.repeats), "128", "cpu-tiled",
                           str(c.get("active", 128)), str(c.get("rounds", 16)), str(c.get("scratch_mib", 64))]
                run = dict(block=block, name=c["name"], command=command, before=before,
                           started=time.time(), power=[], competitors=[])
                record["runs"].append(run)
                save()
                with output.with_suffix(".log").open("x") as log:
                    child = sp.Popen(command, env=clean | {"GJXL_FRONTIER_PROBE_METALLIB": c["metallib"]}, stdout=log, stderr=sp.STDOUT)
                    try:
                        deadline = time.monotonic() + 180
                        while child.poll() is None:
                            if time.monotonic() > deadline:
                                raise RuntimeError("Replay exceeded 180 seconds")
                            run["competitors"] += competitors({child.pid})
                            run["power"].append(sp.check_output(["pmset", "-g", "batt"], text=True))
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
                result = json.loads(output.read_text())
                calls = result["observations"]
                counts = (result["fits"], result["rows"], result["population"], calls[0]["tree_nodes"])
                if expected is None:
                    expected = counts
                if counts != expected or not result["passed"] or len(calls) != 2 * (args.repeats + 1) or any(not x["exact"] or x["tree_nodes"] != expected[3] for x in calls):
                    raise RuntimeError("Work counts or exact trees differ")
                gpu = [x for x in calls if x["mode"] == "tiled_frontier"]
                if len(gpu) != args.repeats + 1 or any(not 0 < x["maximum_active_nodes"] <= result["planned_active_nodes"] for x in gpu):
                    raise RuntimeError("Invalid GPU coverage/admission")
                run.update(result=str(output), result_sha256=sha(output))
                save()
                print(f"block {block}, {c['name']}: {statistics.median(x['wall_seconds'] for x in gpu if not x['warmup']):.6f} s, exact", flush=True)
        if any(sha(p) != h for p, h in frozen.items()):
            raise RuntimeError("Frozen inputs or dependencies changed")
        record.update(status="passed", work=dict(zip(("fits", "rows", "population", "tree_nodes"), expected)))
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["after"] = snapshot()
        save()


if __name__ == "__main__":
    main()
