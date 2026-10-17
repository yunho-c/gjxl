#!/usr/bin/env python3
"""Matched full encodes: CPU cohorts/pipeline, monolithic frontier, tiled frontier."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess as sp
import time

from run_gpu_histogram_wait_study import competitors, sha
from run_histogram_attribution import snapshot, source_files
from run_hybrid_qualification import mac_dependencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("oracle_manifest", type=Path, help="Prior full-study results with cases/input/output hashes and work counts")
    parser.add_argument("output", type=Path)
    parser.add_argument("--power", choices=("AC", "Battery"), default="AC")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    binary = (args.output / "gjxl_modular_hybrid_profile").resolve()
    shutil.copy2(args.binary, binary)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    prior = json.loads(args.oracle_manifest.read_text())
    variants = {
        "cpu_cohorts": ("fitter-cpu", "cohorts", 8),
        "cpu_pipeline": ("fitter-cpu", "pipeline", 16),
        "frontier": ("fitter-frontier", "cohorts", 8),
        "tiled": ("fitter-tiled", "cohorts", 8),
    }
    dependency_listing, dependencies = mac_dependencies(binary)
    record = dict(schema=1, status="running", binary=str(binary), binary_sha256=sha(binary),
                  oracle_manifest=str(args.oracle_manifest.resolve()), oracle_sha256=sha(args.oracle_manifest),
                  source_hashes=source_files(repo), power=args.power, before=snapshot(),
                  dependency_listing=dependency_listing, dependencies=dependencies,
                  boundary=prior["boundary"], variants=variants, cases=[], observations=[])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GJXL_", "MTL_"))}

    def save():
        (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    try:
        for case_index, case in enumerate(prior["cases"]):
            if sha(case["input"]) != case["input_sha256"]:
                raise RuntimeError("Oracle input changed")
            record["cases"].append({k: case[k] for k in ("name", "input", "input_sha256", "output_sha256", "work")})
            for repetition in range(4):
                names = list(variants)
                rotation = (case_index + repetition) % len(names)
                for name in names[rotation:] + names[:rotation]:
                    if competitors():
                        raise RuntimeError("Competing job before encode")
                    before = snapshot()
                    if f"'{args.power} Power'" not in before["power"]:
                        raise RuntimeError("Requested power condition is not active")
                    fitter, schedule, window = variants[name]
                    tag = f"{case['name']}-{repetition}-{name}"
                    dest = (args.output / tag).resolve()
                    command = [str(binary), case["input"], str(dest), "auto", "auto", "best", "fit-off",
                               "14", "14", "14", "14", "14", fitter, "16384", "64",
                               "--sample-rows=metal-unique", f"--learning-schedule={schedule}", f"--feature-window={window}"]
                    run = dict(case=case["name"], variant=name, repetition=repetition, warmup=repetition == 0,
                               command=command, artifact=str(dest), before=before, started=time.time(),
                               competitors=[], power=[])
                    record["observations"].append(run)
                    save()
                    with (args.output / (tag + ".log")).open("x") as log:
                        child = sp.Popen(command, env=env, stdout=log, stderr=sp.STDOUT)
                        try:
                            deadline = time.monotonic() + 180
                            while child.poll() is None:
                                if time.monotonic() > deadline:
                                    raise RuntimeError("Encode timed out")
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
                        raise RuntimeError("Failed or interfered encode")
                    if any(f"'{args.power} Power'" not in x for x in run["power"] + [run["after"]["power"]]):
                        raise RuntimeError("Power changed")
                    result = json.loads((dest / "result.json").read_text())
                    if not result["exact_decode"] or result["work"] != case["work"] or sha(dest / "output.jxl") != case["output_sha256"]:
                        raise RuntimeError("Decode, bytes or work differs")
                    if result["learning"]["schedule"] != schedule:
                        raise RuntimeError("Learning schedule differs")
                    if name in ("frontier", "tiled"):
                        if result["resident_fitting"]["fits"] != case["work"]["fits"] or result["histogram_gpu"]["cpu_nodes"] or result["histogram_gpu"]["admission_fallbacks"]:
                            raise RuntimeError("Incomplete resident fitting coverage")
                    run.update(result_sha256=sha(dest / "result.json"), output_sha256=sha(dest / "output.jxl"),
                               wall=result["total"]["wall"], cpu=result["total"]["cpu"],
                               training_wall=result["profile_wall"][2], memory=result["memory"],
                               resident_fitting=result.get("resident_fitting"))
                    save()
                    print(f"{tag}: {run['wall']:.6f} s, exact", flush=True)
        if sha(binary) != record["binary_sha256"] or any(sha(p) != h for p, h in dependencies.items()):
            raise RuntimeError("Frozen binary/dependencies changed")
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["after"] = snapshot()
        save()


if __name__ == "__main__":
    main()
