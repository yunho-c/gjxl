#!/usr/bin/env python3
"""Alternate frozen Metal builds, retaining raw samples and runtime identities.

This measures the encoding benchmark's documented public-workflow boundary.
Stage mode uses instrumented submissions and is a separate experiment.
"""

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def quiet():
    busy = []
    for line in subprocess.check_output(
        ["ps", "-axo", "pid=,comm="], text=True
    ).splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        pid, command = fields
        name = Path(command).name
        if int(pid) == os.getpid():
            continue
        if name.startswith("gjxl_") or name in (
            "ctest",
            "ninja",
            "make",
            "clang",
            "clang++",
            "cc1",
        ):
            busy.append(line)
    if busy:
        raise RuntimeError("Competing measurement/build processes: " + "; ".join(busy))


def extract(path, profile, expected_samples):
    result = json.loads(path.read_text())
    if len(result["workloads"]) != 1 or result["sample_count"] != expected_samples:
        raise RuntimeError("Unexpected benchmark sample inventory")
    samples = result["workloads"][0]["samples"]
    if len(samples) != expected_samples:
        raise RuntimeError("Incomplete raw samples")
    if [sample["sample_index"] for sample in samples] != list(range(expected_samples)):
        raise RuntimeError("Missing or duplicate sample indices")
    values = collections.defaultdict(list)

    def duration(value):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuntimeError("Invalid duration")
        return value

    if profile:
        for sample in samples:
            if (
                not sample["capabilities"]["stage_boundary"]
                or not sample["submissions"]
            ):
                raise RuntimeError("Stage timing unavailable or incomplete")
            totals = collections.defaultdict(int)
            for submission in sample["submissions"]:
                for stage in submission["stages"]:
                    totals[stage["stage_id"]] += duration(stage["gpu_nanoseconds"])
            if not totals:
                raise RuntimeError("No GPU stage measurements")
            groups = {
                "group.ac_strategy": lambda k: k.startswith("frontend.ac_strategy."),
                "group.coefficients": lambda k: k.startswith(
                    "aq.reconstruction.coefficients."
                )
                or k.startswith("aq.final_frame.dct"),
                "group.dct_io": lambda k: k.startswith(
                    (
                        "aq.reconstruction.dct",
                        "aq.reconstruction.forward.",
                        "aq.reconstruction.scatter.",
                    )
                ),
                "group.epf": lambda k: k.startswith("aq.epf."),
                "group.all_stages": lambda k: True,
            }
            grouped = {
                group: sum(v for k, v in totals.items() if select(k))
                for group, select in groups.items()
            }
            totals.update(grouped)
            for key, value in totals.items():
                values[key].append(value / 1e6)
    else:
        if len({s["encoded_bytes"] for s in samples}) != 1:
            raise RuntimeError("Nondeterministic encoded size")
        for sample in samples:
            phases = sample["phase_nanoseconds"]
            required = {
                "total",
                "input_preparation",
                "backend_selection",
                "quantization_pipeline",
                "codestream_encoding",
                "summary_assembly",
            }
            if (
                sample["backend"] != "metal"
                or sample["encoded_bytes"] <= 0
                or not required <= phases.keys()
            ):
                raise RuntimeError("Missing Metal workflow evidence")
            if phases["total"] <= 0:
                raise RuntimeError("Nonpositive workflow time")
            for key, value in phases.items():
                values[key].append(duration(value) / 1e6)
    if any(len(v) != expected_samples for v in values.values()):
        raise RuntimeError("Metric missing from some samples")
    return {key: statistics.median(v) for key, v in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-build", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.pairs, args.warmups, args.samples) < 1:
        parser.error("Positive pairs, warmups and samples required")
    jobs = [(name, ["--workload", name]) for name in args.workload]
    jobs += [(p.stem, ["--input", str(p.resolve())]) for p in args.input]
    if not jobs or len({name for name, _ in jobs}) != len(jobs):
        parser.error("Supply workloads/inputs with unique names")
    builds = {
        "baseline": args.baseline_build.resolve(),
        "candidate": args.candidate_build.resolve(),
    }
    runtime = {}
    for label, build in builds.items():
        binary = build / "gjxl_encoding_benchmark"
        library = build / "metal/gjxl.metallib"
        if not library.exists():
            library = build / "gjxl.metallib"
        runtime[label] = {str(path): sha(path) for path in (binary, library)}
    identity = {
        "runtime": runtime,
        "inputs": {str(p.resolve()): sha(p) for p in args.input},
        "jobs": jobs,
        "pairs": args.pairs,
        "warmups": args.warmups,
        "samples": args.samples,
        "profile": args.profile,
        "platform": platform.platform(),
        "protocol": sha(Path(__file__)),
    }
    # Normalize tuples to the JSON form for resume equality.
    identity = json.loads(json.dumps(identity))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "identity.json").exists():
        if json.loads((out / "identity.json").read_text()) != identity:
            raise RuntimeError("Identity changed: use a new output directory")
    else:
        save(out / "identity.json", identity)
    rows = []
    for name, workload in jobs:
        for pair in range(args.pairs):
            medians = {}
            order = (
                ("baseline", "candidate")
                if pair % 2 == 0
                else ("candidate", "baseline")
            )
            for label in order:
                stem = f"{name}-pair{pair}-{label}"
                raw, log, record = [
                    out / (stem + suffix) for suffix in (".raw.json", ".log", ".json")
                ]
                binary, library = list(runtime[label])
                for path, expected in runtime[label].items():
                    if sha(path) != expected:
                        raise RuntimeError("Runtime changed during experiment: " + path)
                command = [
                    binary,
                    "--metallib",
                    library,
                    *workload,
                    "--scope",
                    "metal-public-workflow",
                    "--implementation",
                    "simd",
                    "--ac-residual-inverse",
                    "fused-tuned",
                    "--gpu-aq",
                    "fully-resident",
                    "--validation",
                    "metal-only",
                    "--distance",
                    "1.2",
                    "--effort",
                    "7",
                    "--warmups",
                    str(args.warmups),
                    "--samples",
                    str(args.samples),
                ]
                command += (
                    ["--gpu-profile", "stage", "--gpu-profile-output", str(raw)]
                    if args.profile
                    else ["--raw-samples", str(raw)]
                )
                if record.exists():
                    old = json.loads(record.read_text())
                    if (
                        old["command"] != command
                        or old["raw_sha256"] != sha(raw)
                        or old["log_sha256"] != sha(log)
                    ):
                        raise RuntimeError("Recorded artifacts changed: " + stem)
                else:
                    if raw.exists() or log.exists():
                        raise RuntimeError(
                            "Incomplete attempt retained; use a new output directory: "
                            + stem
                        )
                    quiet()
                    started = time.time()
                    with log.open("w") as stream:
                        process = subprocess.run(
                            command,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            timeout=600,
                        )
                    if process.returncode:
                        raise RuntimeError(
                            f"Benchmark failed ({process.returncode}): {log}"
                        )
                    save(
                        record,
                        {
                            "command": command,
                            "start_unix": started,
                            "end_unix": time.time(),
                            "raw_sha256": sha(raw),
                            "log_sha256": sha(log),
                        },
                    )
                medians[label] = extract(raw, args.profile, args.samples)
            if args.profile:
                removed = medians["baseline"].keys() - medians["candidate"].keys()
                if any(
                    not key.startswith("aq.reconstruction.scatter.") for key in removed
                ):
                    raise RuntimeError("Unexpected removed GPU stages")
                # Image I/O eliminates these dispatches and their stage records.
                medians["candidate"].update({key: 0.0 for key in removed})
            if medians["baseline"].keys() != medians["candidate"].keys():
                raise RuntimeError("Baseline/candidate metric inventories differ")
            shared = medians["baseline"].keys()
            row = {
                "name": name,
                "pair": pair,
                "process_medians_ms": medians,
                "paired_changes_percent": {
                    key: 100
                    * (medians["candidate"][key] / medians["baseline"][key] - 1)
                    for key in shared
                    if medians["baseline"][key] > 0
                },
            }
            rows.append(row)
            save(out / "pairs.json", rows)
            print(f"{name} pair {pair} complete", flush=True)
    summary = []
    for name, _ in jobs:
        chosen = [row for row in rows if row["name"] == name]
        keys = set.intersection(*(set(row["paired_changes_percent"]) for row in chosen))
        summary.append(
            {
                "name": name,
                "metrics": {
                    key: {
                        "paired_changes_percent": [
                            row["paired_changes_percent"][key] for row in chosen
                        ],
                        "median_change_percent": statistics.median(
                            row["paired_changes_percent"][key] for row in chosen
                        ),
                        "baseline_ms": statistics.median(
                            row["process_medians_ms"]["baseline"][key] for row in chosen
                        ),
                        "candidate_ms": statistics.median(
                            row["process_medians_ms"]["candidate"][key]
                            for row in chosen
                        ),
                    }
                    for key in sorted(keys)
                },
            }
        )
    save(out / "summary.json", summary)


if __name__ == "__main__":
    main()
