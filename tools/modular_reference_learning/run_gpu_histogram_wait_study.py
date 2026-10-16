#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Interleaved GPU fitting experiments with bounded traces and exact outputs.

Variants JSON: [{"name": ..., "binary": ..., "args": [...]}]. Args follow the
input and output paths. Each phase has one warmup and N measured repetitions.
Raw records remain in the output directory; the summary keeps artifact hashes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess as sp
import time

from run_histogram_attribution import snapshot, source_files
from run_hybrid_qualification import mac_dependencies, require


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


CONTROLLERS = set()


def competitors(exclude=()):
    table = {}
    for line in sp.check_output(["ps", "-ww", "-axo", "pid=,ppid=,args="], text=True).splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3:
            table[int(fields[0])] = (int(fields[1]), fields[2])
    ignored = {os.getpid(), *exclude}
    found = []
    for pid, (parent, command) in table.items():
        if pid in ignored:
            continue
        executable = command.split()[0]
        name = Path(executable).name.lower()
        heavy = ("gjxl" in executable and "/build/" in executable) or name.startswith("gjxl_")
        heavy |= name in {"cjxl", "djxl", "ninja", "clang", "clang++", "ld", "ctest", "samply", "metal",
                          "gpudebug", "gpucapture", "xctrace"}
        heavy |= name == "cmake" and "--build" in command
        controller = "python" in name and any(s in command for s in (
            "qualify_", "qualification.py", "benchmark", "run_gpu_histogram_wait_study.py",
            "run_histogram_submission_probe.py",
            "storage-audit", "run_weighted_presets", "run_histogram_attribution",
            "run_weighted_statistics", "capture_statistics"))
        if heavy:
            ancestor = parent
            while ancestor in table and ancestor not in ignored:
                ancestor_parent, ancestor_command = table[ancestor]
                if "python" in Path(ancestor_command.split()[0]).name.lower():
                    CONTROLLERS.add(ancestor)
                ancestor = ancestor_parent
        if heavy or controller or pid in CONTROLLERS:
            found.append(dict(pid=pid, command=command))
    return found


def summarize(result):
    profile = result.get("histogram_gpu_profile", {})
    batches = profile.get("batches", [])
    summary = dict(wall=result["total"]["wall"], cpu=result["total"]["cpu"],
                   batches=len(batches), stages={}, phases={}, trace={}, queue={})
    for key, field in (("stages", "gpu_ns"), ("phases", "wall_ns")):
        names = profile.get("gpu_stages" if key == "stages" else "wall_phases", [])
        summary[key] = {name: sum(b[field][i] for b in batches) * 1e-9 for i, name in enumerate(names)}
    summary["gpu_span"] = sum(b["command_buffer_gpu_ns"] for b in batches) * 1e-9
    summary["waves"] = sum(b["waves"] for b in batches)
    summary["scratch_max"] = max((b["scratch_bytes"] for b in batches), default=0)
    metrics = {name: [] for name in ("commit_to_gpu", "commit_call", "pre_gpu_wait",
               "gpu_in_wait", "post_gpu_wait", "suspend", "cpu_resume", "blocking",
               "next", "oldest_ready", "publish", "between_commands",
               "commit_to_kernel", "kernel_scheduling", "kernel_to_gpu")}
    reasons = {str(i): 0 for i in range(4)}
    previous = None
    for b in batches:
        if "wait_trace" not in b:
            continue
        t, q = b["wait_trace"], b["queue_trace"]
        require(0 < t["commit_begin"] <= t["commit_end"] <= t["wait_begin"] <=
                t["blocking_begin"] <= t["blocking_end"] <= t["cpu_resumed"], "Invalid host trace order")
        require(t["commit_begin"] <= t["gpu_begin"] <= t["gpu_end"] <= t["blocking_end"],
                "GPU timestamps do not share the host mach clock")
        begin, end = t["blocking_begin"], t["blocking_end"]
        values = dict(commit_to_gpu=t["gpu_begin"] - t["commit_begin"],
                      commit_call=t["commit_end"] - t["commit_begin"],
                      pre_gpu_wait=max(0, min(end, t["gpu_begin"]) - begin),
                      gpu_in_wait=max(0, min(end, t["gpu_end"]) - max(begin, t["gpu_begin"])),
                      post_gpu_wait=max(0, end - max(begin, t["gpu_end"])),
                      suspend=begin - t["wait_begin"], cpu_resume=t["cpu_resumed"] - end,
                      blocking=end - begin, next=q["next_ns"], oldest_ready=q["oldest_ready_ns"],
                      publish=q["publish_ns"])
        require(values["pre_gpu_wait"] + values["gpu_in_wait"] + values["post_gpu_wait"] ==
                values["blocking"], "Wait partition does not conserve duration")
        if "kernel_begin" in t:
            require(t["commit_begin"] <= t["kernel_begin"] <= t["kernel_end"] <= t["blocking_end"],
                    "Invalid CPU-driver scheduling timestamps")
            # Keep the final difference signed: CPU scheduling and GPU work
            # may overlap. Do not invent an exclusive driver/GPU partition.
            values.update(commit_to_kernel=t["kernel_begin"] - t["commit_begin"],
                          kernel_scheduling=t["kernel_end"] - t["kernel_begin"],
                          kernel_to_gpu=t["gpu_begin"] - t["kernel_end"])
        for name, value in values.items():
            metrics[name].append(value)
        if previous is not None:
            metrics["between_commands"].append(t["commit_begin"] - previous)
        previous = t["gpu_end"]
        reasons[str(q["reason"])] += 1
    summary["trace"] = {name: dict(seconds=sum(values) * 1e-9,
                                  median_us=statistics.median(values) * 1e-3,
                                  p95_us=sorted(values)[min(len(values)-1, int(len(values)*.95))] * 1e-3)
                        for name, values in metrics.items() if values}
    traced = [b for b in batches if "queue_trace" in b]
    if traced:
        summary["queue"] = dict(reasons=reasons,
            mean_nodes=statistics.mean(b["nodes"] for b in traced),
            ready_at_resume=sum(b["queue_trace"]["resume_ready"] > 0 for b in traced),
            cpu_idle_at_resume=sum(b["queue_trace"]["resume_cpu_queued"] == 0 and
                                   b["queue_trace"]["resume_cpu_running"] == 0 for b in traced))
    summary["fitter"] = result.get("histogram_gpu", {})
    summary["memory"] = result.get("memory", {})
    summary["reuse"] = dict(bytes=sum(b.get("reused_buffer_bytes", 0) for b in batches),
        batches=sum(bool(b.get("reused_buffers", 0)) for b in batches),
        masks={str(mask): sum(b.get("reused_buffers", 0) == mask for b in batches) for mask in range(8)},
        compute_encoders=sum(b.get("compute_encoders", 0) for b in batches))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--power", choices=("ac", "battery"), default="ac")
    args = parser.parse_args()
    require(args.repeats > 0, "At least one measured repetition required")
    repo = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    variants = json.loads(args.variants.read_text())
    require(len({v["name"] for v in variants}) == len(variants), "Duplicate variant names")
    report = dict(status="running", revision=sp.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  started_unix=time.time(), variants=variants, repetitions=args.repeats,
                  boundary="Complete private encode; excludes backend/domain creation, file IO, independent decode and hashing",
                  power=args.power, sources=source_files(repo), dependencies={}, binaries={}, cases=[], rejected=[])
    expected_power = "'AC Power'" if args.power == "ac" else "'Battery Power'"
    (out / "source.patch").write_bytes(sp.check_output(["git", "diff", "--binary", "HEAD"]))
    for name in sp.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True).splitlines():
        if name in report["sources"]:
            destination = out / "new_sources" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / name, destination)
    for v in variants:
        binary = str(Path(v["binary"]).resolve())
        v["binary"] = binary
        listing, hashes = mac_dependencies(Path(binary))
        report["binaries"][binary] = dict(sha256=sha(binary), dependencies=listing)
        report["dependencies"].update(hashes)
    def save():
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        for source in args.input:
            case = dict(input=str(source.resolve()), name=source.stem, input_sha256=sha(source), runs=[])
            report["cases"].append(case)
            for repeat in range(args.repeats + 1):
                if len(variants) == 4:
                    orders = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
                    order = [variants[i] for i in orders[repeat % 4]]
                else:
                    order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
                for v in order:
                    tag = f"{source.stem}-{v['name']}-r{repeat}"
                    for attempt in range(10):
                        while active := competitors():
                            print("Waiting for competing work", json.dumps(active), flush=True)
                            time.sleep(10)
                        before = snapshot()
                        require(expected_power in before["power"], f"Capture requires {args.power} power")
                        directory = out / f"{tag}-a{attempt}"
                        command = [v["binary"], str(source.resolve()), str(directory), *v["args"]]
                        activity = []
                        with (out / f"{tag}-a{attempt}.log").open("x") as log:
                            process = sp.Popen(command, stdout=log, stderr=sp.STDOUT)
                            while process.poll() is None:
                                activity.extend(competitors({process.pid}))
                                time.sleep(.25)
                        after = snapshot()
                        require(process.returncode == 0, f"Encode failed: {directory}")
                        require(expected_power in after["power"], "Power changed during capture")
                        result = json.loads((directory / "result.json").read_text())
                        require(result["exact_decode"], "Independent decode differs")
                        output = directory / "output.jxl"
                        digest = sha(output)
                        if "output_sha256" not in case:
                            case.update(output_sha256=digest, work=result["work"], canonical=str(output))
                        require(digest == case["output_sha256"] and result["work"] == case["work"],
                                f"Changed bytes or logical learning work: {tag}")
                        fitter = result.get("histogram_gpu", {})
                        if fitter.get("route") == "fitter-gpu":
                            require(fitter["nodes"] > 0 and fitter["admission_fallbacks"] == 0,
                                    "GPU experiment did not execute without admission fallback")
                            work = {key: fitter[key] for key in ("nodes", "rows", "properties", "cpu_nodes")}
                            case.setdefault("gpu_work", work)
                            require(case["gpu_work"] == work, "GPU/CPU fitting work changed")
                        if str(output) != case["canonical"]:
                            # Only newly created, verified duplicate artifacts are
                            # replaced; APFS clones preserve every run's output.
                            output.unlink()
                            sp.run(["cp", "-c", case["canonical"], str(output)], check=True)
                        observation = dict(variant=v["name"], repetition=repeat, warmup=repeat == 0,
                                           command=command, artifact=str(directory), before=before, after=after,
                                           competitors=activity, output_sha256=digest,
                                           result_sha256=sha(directory / "result.json"), summary=summarize(result))
                        if activity:
                            report["rejected"].append(observation)
                            save()
                            continue
                        case["runs"].append(observation)
                        save()
                        print(tag, round(result["total"]["wall"], 6), "s exact", flush=True)
                        break
                    else:
                        raise RuntimeError(f"No isolated run: {tag}")
        require(report["sources"] == source_files(repo), "Source changed during capture")
        for binary, info in report["binaries"].items():
            require(sha(binary) == info["sha256"], "Binary changed during capture")
        for dependency, digest in report["dependencies"].items():
            require(sha(dependency) == digest, "Dependency changed during capture")
        for case in report["cases"]:
            require(sha(case["input"]) == case["input_sha256"], "Input changed during capture")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
