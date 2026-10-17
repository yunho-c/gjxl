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
    parser.add_argument("--caps", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--scratch-mib", type=int, default=256)
    parser.add_argument("--diagnostic-large-scratch", action="store_true",
                        help="Request the private diagnostic build's 320 MiB device ceiling")
    parser.add_argument("--variant", action="append", default=[], metavar="NAME=METALLIB")
    args = parser.parse_args()
    if not args.caps or len(set(args.caps)) != len(args.caps) or not all(1 <= c <= 1024 for c in args.caps):
        parser.error("Caps must be unique integers from 1 to 1024")
    if not 1 <= args.scratch_mib <= 512:
        parser.error("Scratch must be between 1 and 512 MiB")
    libraries = {}
    for variant in args.variant:
        name, separator, path = variant.partition("=")
        if not separator or not name or not all(c.isalnum() or c in "_-" for c in name) or name in libraries:
            parser.error("Each variant must have a unique simple NAME=METALLIB")
        libraries[name] = Path(path).resolve(strict=True)
    repo = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.output / "probe"
    shutil.copy2(args.probe, binary)
    binary = binary.resolve()
    archive = args.archive.resolve()
    frozen = {}
    for name, path in libraries.items():
        target = args.output / f"{name}.metallib"
        shutil.copy2(path, target)
        frozen[name] = target.resolve()
    record = dict(schema=2, status="running", head=sp.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        source_hashes=source_files(repo), binary_sha256=sha(binary),
        archive_sha256=sha(archive), power=args.power, before=snapshot(), runs=[],
        scratch_mib=args.scratch_mib, diagnostic_large_scratch=args.diagnostic_large_scratch,
        libraries={name: dict(source=str(libraries[name]), frozen=str(path), sha256=sha(path))
                   for name, path in frozen.items()})
    def save():
        (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GJXL_", "MTL_"))}
    if args.diagnostic_large_scratch:
        env["GJXL_FRONTIER_DIAGNOSTIC_LARGE_SCRATCH"] = "1"
    caps = args.caps
    orders = (caps, list(reversed(caps)), caps[len(caps)//2:] + caps[:len(caps)//2])
    variants = list(frozen) or ["embedded"]
    configurations = [(block, cap, name) for block, order in enumerate(orders)
                      for position, cap in enumerate(order)
                      for name in (variants if (block + position) % 2 == 0 else list(reversed(variants)))]
    record["order"] = configurations
    try:
        save()
        for block, cap, name in configurations:
            if competitors():
                raise RuntimeError("Another benchmark or build is active")
            if f"'{args.power} Power'" not in sp.check_output(["pmset", "-g", "batt"], text=True):
                raise RuntimeError("Requested power source is not active")
            suffix = f"-{name}" if frozen else ""
            result_path = (args.output / f"block-{block}-cap-{cap}{suffix}.json").resolve()
            command = [str(binary), str(archive), str(result_path), "16", "2", "128",
                       "cpu-frontier", str(cap), "16", str(args.scratch_mib)]
            child_env = dict(env)
            if frozen:
                child_env["GJXL_FRONTIER_PROBE_METALLIB"] = str(frozen[name])
            run = dict(block=block, active_limit=cap, variant=name, command=command, before=snapshot(),
                       started=time.time(), power_samples=[], competitors=[])
            record["runs"].append(run)
            save()
            with result_path.with_suffix(".log").open("w") as log:
                child = sp.Popen(command, env=child_env, stdout=log, stderr=sp.STDOUT)
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
            gpu = [x for x in result["observations"] if x["mode"] == "frontier"]
            if not gpu or any(x["maximum_active_nodes"] != cap for x in gpu):
                raise RuntimeError("Replay did not exercise the requested admission")
            if any(f"'{args.power} Power'" not in x for x in run["power_samples"]):
                raise RuntimeError("Power source changed during replay")
            if run["competitors"]:
                raise RuntimeError("Competing work detected during replay")
            run.update(finished=time.time(), after=snapshot(), result_sha256=sha(result_path))
            save()
            times = [x["wall_seconds"] for x in result["observations"]
                     if x["mode"] == "frontier" and not x["warmup"]]
            print(f"block {block}, cap {cap}, {name}: {statistics.median(times):.6f} s, exact", flush=True)
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["after"] = snapshot()
        save()


if __name__ == "__main__":
    main()
