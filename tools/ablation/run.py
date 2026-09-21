#!/usr/bin/env python3
"""Frozen, resumable conditional ablations; a small trial is the default.

No third-party Python packages required. See docs/paper-ablation.md for scope.
"""
from __future__ import annotations

import argparse
import array
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("production", "aq-sync", "ac-handoff", "split-malta", "split-epf",
            "host-fused", "host-split-ac", "packed-simd", "packed-scalar")
# Positive ratio = slower without the optimization. Only these pairs isolate
# the named factor; do not treat all arms as an orthogonal factorial design.
PAIRS = (
    ("aq-boundaries", "production", "aq-sync", "exact"),
    ("ac-aq-handoff", "production", "ac-handoff", "exact"),
    ("malta-fusion", "production", "split-malta", "exact"),
    ("epf-linear-fusion", "production", "split-epf", "exact"),
    ("ac-fusion", "host-fused", "host-split-ac", "exact"),
    ("dct-image-io", "host-split-ac", "packed-simd", "exact"),
    ("dct-arithmetic", "packed-simd", "packed-scalar", "numerical"),
)


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def clean_env():
    return {k: v for k, v in os.environ.items()
            if not k.startswith(("GJXL_", "MTL_", "DYLD_"))}


def command(args, *, env=None, timeout=300):
    result = subprocess.run([str(a) for a in args], cwd=ROOT,
                            env=env or clean_env(), capture_output=True,
                            text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {args}\n"
                           f"{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def battery():
    return command(["pmset", "-g", "batt"])


def source_hashes():
    names = command(["git", "ls-files", "--cached", "--others",
                     "--exclude-standard", "--", "src", "cmake", "CMakeLists.txt",
                     "benchmarks", "tools/ablation", "tests"])
    return {name: sha(ROOT / name) for name in sorted(set(names.splitlines()))
            if (ROOT / name).is_file()}


def make_trial_inputs(directory):
    directory.mkdir(exist_ok=True)
    paths = []
    for name, width, height in (("smooth", 128, 96), ("edges", 257, 193)):
        path = directory / f"{name}.pfm"
        if not path.exists():
            values = array.array("f")
            for y in reversed(range(height)):
                for x in range(width):
                    u, v = x / (width - 1), y / (height - 1)
                    texture = 0.08 * math.sin(x * 0.73) * math.cos(y * 0.51)
                    edge = 0.2 * ((x // 13 + y // 11) % 2) if name == "edges" else 0
                    values.extend(max(0., min(1., c + texture + edge))
                                  for c in (0.05 + .7*u, .1 + .6*v, .08 + .5*u*v))
            if sys.byteorder != "little":
                values.byteswap()
            path.write_bytes(f"PF\n{width} {height}\n-1.0\n".encode() + values.tobytes())
        paths.append(path)
    return paths


def read_pfm(path):
    with Path(path).open("rb") as stream:
        def line():
            value = stream.readline().strip()
            while value.startswith(b"#"):
                value = stream.readline().strip()
            return value
        if line() != b"PF":
            raise ValueError("Expected RGB PFM")
        width, height = map(int, line().split())
        scale = float(line())
        if not math.isfinite(scale) or scale == 0 or width <= 0 or height <= 0:
            raise ValueError("Invalid PFM header")
        values = array.array("f")
        values.frombytes(stream.read())
        if len(values) != width * height * 3:
            raise ValueError("Invalid PFM payload size")
        if (scale < 0) != (sys.byteorder == "little"):
            values.byteswap()
        if abs(scale) != 1:
            values = array.array("f", (v * abs(scale) for v in values))
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Non-finite decoded pixels")
        return (width, height), values


def pixel_delta(left, right):
    le, lv = read_pfm(left)
    re, rv = read_pfm(right)
    if le != re:
        raise ValueError("Decoded extent mismatch")
    total, maximum = 0., 0.
    for a, b in zip(lv, rv):
        difference = abs(a - b)
        total += difference * difference
        maximum = max(maximum, difference)
    return {"rmse": math.sqrt(total / len(lv)), "max_abs": maximum}


def validate_audit(report, variant, effort):
    if report["ablation"]["variant"] != variant or report["stage_profile_enabled"]:
        raise ValueError("Wrong variant or profiling enabled")
    counters = report["ablation"]["counters"]
    expected_updates = {5: 1, 6: 1, 7: 2, 8: 3, 9: 4}[effort]
    if counters.get("aq_evaluations") != expected_updates:
        raise ValueError("Unexpected number of AQ evaluations")
    if counters.get("unknown_kernel", 0):
        raise ValueError("Unregistered kernel in execution audit")
    def has(fragment):
        return any(fragment in k and v for k, v in counters.items())
    host = variant in VARIANTS[5:]
    if counters.get("cpu_selector", 0) != int(host):
        raise ValueError("Unexpected CPU selector path")
    if counters.get("gpu_selector", 0) != int(not host):
        raise ValueError("Unexpected GPU selector path")
    if counters.get("deferred_search", 0) != int(not host and variant != "ac-handoff"):
        raise ValueError("Unexpected combined ACS/AQ path")
    if variant == "split-malta" and (not has("malta_scale") or has("malta_fixed") or has("malta_fused")):
        raise ValueError("Split Malta was not exercised")
    if variant == "split-epf" and any("epf" in k and "linear" in k for k in counters):
        raise ValueError("Final EPF conversion remained fused")
    if variant in ("host-split-ac", "packed-simd", "packed-scalar"):
        if has("candidate_loss") or any("ac_strategy" in k and "fused" in k for k in counters):
            raise ValueError("AC fusion remained enabled")
    if variant.startswith("packed-"):
        if not has("gather_transform_pixels") or not has("scatter_reconstructed_pixels"):
            raise ValueError("Packed DCT path was not exercised")
        if any("dct" in k and k.endswith("_image") for k in counters):
            raise ValueError("Direct image DCT remained enabled")
    if variant == "packed-scalar":
        if not has("_scalar_2d_matmul") or has("_simdgroup_2d_matmul"):
            raise ValueError("Scalar DCT control was not exercised")
    elif has("_scalar_2d_matmul") or (variant == "packed-simd" and not has("_simdgroup_2d_matmul")):
        raise ValueError("Unexpected DCT arithmetic implementation")
    if variant != "split-malta" and not (has("malta_fixed") or has("malta_fused")):
        raise ValueError("Fused Malta baseline was not exercised")
    if variant != "split-epf" and not any("epf" in k and "linear" in k for k in counters):
        raise ValueError("Fused EPF baseline was not exercised")
    if variant in ("production", "host-fused") and not has("candidate_loss"):
        raise ValueError("Fused AC baseline was not exercised")
    if variant in ("production", "host-fused", "host-split-ac") and not any(
            "dct" in k and k.endswith("_image") for k in counters):
        raise ValueError("Direct image DCT baseline was not exercised")
    return counters


def verify_record(directory, record):
    for name, digest in record["artifacts"].items():
        if not (directory / name).is_file() or sha(directory / name) != digest:
            raise ValueError(f"Retained artifact changed: {directory / name}")
    raw = json.loads((directory / "raw.json").read_text())
    if (record["times_ns"] != [s["elapsed_nanoseconds"] for s in raw["samples"]] or
        record["counters"] != raw["ablation"]["counters"] or
        record["variant"] != raw["ablation"]["variant"] or
        record["bytes"] != (directory / "output.jxl").stat().st_size or
        record["quality"] != float((directory / "metric.txt").read_text())):
        raise ValueError("Retained record disagrees with raw artifacts")


def validate_output_directory(out, resume):
    if (out / "manifest.json").exists() != resume:
        raise ValueError("Use a fresh output directory, or --resume for an existing manifest")
    if not resume and any(p.name != "run.lock" for p in out.iterdir()):
        raise ValueError("New runs require an empty output directory")


def compare_records(records, out):
    comparisons = []
    for key in sorted({r["case"] for r in records.values()}):
        arms = {r["variant"]: r for r in records.values() if r["case"] == key}
        for name, baseline, disabled, contract in PAIRS:
            a, b = arms[baseline], arms[disabled]
            same = a["artifacts"]["output.jxl"] == b["artifacts"]["output.jxl"]
            delta = pixel_delta(out / a["id"] / "decoded.pfm", out / b["id"] / "decoded.pfm")
            ca, cb = a["counters"], b["counters"]
            if name == "aq-boundaries":
                expected = ca["aq_evaluations"] + 1
                if (cb["submissions"] - ca["submissions"] != expected or
                    cb["completion_waits"] - ca["completion_waits"] != expected):
                    raise ValueError("AQ boundary ablation did not add expected submissions/waits")
                if {k: v for k, v in ca.items() if k.startswith("gjxl_")} != {
                        k: v for k, v in cb.items() if k.startswith("gjxl_")}:
                    raise ValueError("AQ boundary ablation changed kernel invocation counts")
            comparisons.append({"case": key, "factor": name, "baseline": baseline,
                "disabled": disabled, "contract": contract, "byte_identical": same,
                "pixel_delta": delta, "quality_delta_ssimulacra2": b["quality"] - a["quality"],
                "byte_ratio": b["bytes"] / a["bytes"],
                "time_ratio": statistics.median(b["times_ns"]) / statistics.median(a["times_ns"]),
                "status": "pass" if same else ("requires-quality-review" if contract == "numerical" else "failed-exactness")})
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-ablation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--ssimulacra2", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--full", action="store_true", help="Explicitly enable corpus measurement; requires AC power")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--efforts", default="5,8")
    parser.add_argument("--distances", default="1")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.build_dir = args.build_dir.resolve()
    args.decoder = args.decoder.resolve()
    args.ssimulacra2 = args.ssimulacra2.resolve()
    efforts = [int(x) for x in args.efforts.split(",")]
    distances = [float(x) for x in args.distances.split(",")]
    if not efforts or len(efforts) != len(set(efforts)) or any(e not in range(5, 10) for e in efforts):
        parser.error("This study supports mixed-transform efforts 5..9")
    if not distances or len(distances) != len(set(distances)) or any(not math.isfinite(d) or d <= 0 for d in distances):
        parser.error("Distances must be finite and positive")
    if args.full and (not args.input or "AC Power" not in battery()):
        parser.error("Full runs require explicit corpus inputs and AC power")
    if not args.full and (args.input or efforts != [5, 8] or distances != [1.]):
        parser.error("Trial is bounded to generated tiny inputs, efforts 5,8 and distance 1")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, efforts, distances)


def run(args, efforts, distances):
    out = args.output
    manifest_path = out / "manifest.json"
    validate_output_directory(out, args.resume)
    cache_path = args.build_dir / "CMakeCache.txt"
    cache = cache_path.read_text()
    if (f"CMAKE_HOME_DIRECTORY:INTERNAL={ROOT}" not in cache or
        "GJXL_BUILD_ABLATION_EXPERIMENT:BOOL=ON" not in cache or
        "GJXL_BUILD_FRONTIER_EXPERIMENT:BOOL=OFF" not in cache):
        raise ValueError("Wrong source tree or experimental build configuration")
    # Incremental build before freezing the executable prevents stale-source runs.
    sources = source_hashes()
    build_log = command(["cmake", "--build", args.build_dir, "--target", "gjxl_ablation_benchmark", "-j", "2"], timeout=1200)
    if source_hashes() != sources:
        raise ValueError("Source changed during build; retry with stable sources")
    binary = args.build_dir / "gjxl_ablation_benchmark"
    if tuple(json.loads(command([binary, "--variants"]))) != VARIANTS:
        raise ValueError("Driver and runner disagree on variants")
    inputs = [p.resolve() for p in args.input] if args.full else make_trial_inputs(out / "inputs")
    protocol = {"mode": "full" if args.full else "trial", "efforts": efforts, "distances": distances,
                "rounds": 3 if args.full else 1, "warmups": 3 if args.full else 0,
                "samples": 7 if args.full else 1, "threads": 8 if args.full else 2}
    fingerprint = {"protocol": protocol, "variants": VARIANTS, "pairs": PAIRS,
        "revision": command(["git", "rev-parse", "HEAD"]), "sources": sources,
        "submodules": command(["git", "submodule", "status"]),
        "binary_sha256": sha(binary), "cache_sha256": sha(cache_path),
        "decoder": {"path": str(args.decoder), "sha256": sha(args.decoder)},
        "metric": {"path": str(args.ssimulacra2), "sha256": sha(args.ssimulacra2)},
        "inputs": [{"path": str(p), "sha256": sha(p)} for p in inputs],
        "driver_version": json.loads(command([binary, "--version"])),
        "system": platform.platform(), "hardware": command(["sysctl", "-n", "machdep.cpu.brand_string"])}
    fingerprint = json.loads(json.dumps(fingerprint))
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Resume refused: source, binary, tools, inputs, machine or protocol changed")
        if sha(out / "encoder") != fingerprint["binary_sha256"]:
            raise ValueError("Frozen encoder changed")
    else:
        manifest = {"schema": 1, "fingerprint": fingerprint, "created": time.time(),
                    "initial_power": battery(), "timing": "complete public call, warm process, no stage profiling",
                    "audit": "separate untimed encode; no GPU counter sampling"}
        shutil.copy2(binary, out / "encoder")
        shutil.copy2(cache_path, out / "CMakeCache.txt")
        (out / "build.log").write_text(build_log)
        (out / "source.patch").write_text(command(["git", "diff", "HEAD", "--", "src", "CMakeLists.txt", "benchmarks"]))
        with tarfile.open(out / "sources.tar.gz", "w:gz") as archive:
            for name in fingerprint["sources"]:
                archive.add(ROOT / name, arcname=name)
        if source_hashes() != sources:
            raise ValueError("Source changed while freezing the run")
        if sha(out / "encoder") != fingerprint["binary_sha256"]:
            raise ValueError("Binary changed while freezing the run")
        atomic_json(manifest_path, manifest)
    records = {}
    case_index = 0
    for round_index in range(protocol["rounds"]):
        for input_index, image in enumerate(inputs):
            for effort in efforts:
                for distance in distances:
                    case = f"r{round_index}-i{input_index}-e{effort}-d{distance}"
                    shift = (case_index + round_index) % len(VARIANTS)
                    order = VARIANTS[shift:] + VARIANTS[:shift]
                    case_index += 1
                    for variant in order:
                        if args.full and "AC Power" not in battery():
                            raise RuntimeError("AC power lost; resume after reconnecting")
                        job_id = f"{case}-{variant}"
                        directory = out / job_id
                        directory.mkdir(exist_ok=True)
                        record_path = directory / "record.json"
                        if record_path.exists():
                            record = json.loads(record_path.read_text())
                            verify_record(directory, record)
                            records[job_id] = record
                            continue
                        print(job_id, flush=True)
                        env = clean_env() | {"GJXL_ABLATION_VARIANT": variant}
                        power_before = battery()
                        command([out / "encoder", "--input", image, "--output", directory / "output.jxl",
                                 "--raw-samples", directory / "raw.json", "--effort", effort, "--distance", distance,
                                 "--num-threads", protocol["threads"], "--warmups", protocol["warmups"],
                                 "--samples", protocol["samples"]], env=env, timeout=600)
                        report = json.loads((directory / "raw.json").read_text())
                        counters = validate_audit(report, variant, effort)
                        command([args.decoder, directory / "output.jxl", directory / "decoded.pfm", "--num_threads=2",
                                 "--color_space=RGB_D65_SRG_Rel_Lin"])
                        distortion = pixel_delta(image, directory / "decoded.pfm")
                        metric_text = command([args.ssimulacra2, image, directory / "decoded.pfm"])
                        quality = float(metric_text)
                        if not math.isfinite(quality):
                            raise ValueError("Non-finite quality score")
                        (directory / "metric.txt").write_text(metric_text + "\n")
                        record = {"id": job_id, "case": case, "variant": variant, "quality": quality,
                                  "input_pixel_error": distortion, "counters": counters,
                                  "bytes": (directory / "output.jxl").stat().st_size,
                                  "times_ns": [s["elapsed_nanoseconds"] for s in report["samples"]],
                                  "power_before": power_before, "power_after": battery(),
                                  "artifacts": {name: sha(directory / name) for name in
                                                ("output.jxl", "raw.json", "decoded.pfm", "metric.txt")}}
                        atomic_json(record_path, record)
                        records[job_id] = record
                        with (out / "ledger.jsonl").open("a") as ledger:
                            ledger.write(json.dumps(record) + "\n")
    comparisons = compare_records(records, out)
    failed = [c for c in comparisons if c["status"] == "failed-exactness"]
    review = [c for c in comparisons if c["status"] == "requires-quality-review"]
    summary = {"mode": protocol["mode"], "jobs": len(records), "comparisons": comparisons,
               "exactness_failures": len(failed), "numerical_reviews": len(review),
               "status": "failed-exactness" if failed else "complete-with-numerical-review" if review else "complete",
               "performance_qualified": False if not args.full or failed or review else None}
    atomic_json(out / "summary.json", summary)
    lines = ["# GJXL ablation " + protocol["mode"], "", f"Status: {summary['status']}; {len(records)} jobs.",
             "", "Trial timings are plumbing checks on battery, not performance evidence." if not args.full else
             "Review cohort coverage, numerical cases and power records before qualifying results.", "",
             "| Case | Factor | Byte identical | Pixel max abs | SSIMULACRA2 delta | Disabled/baseline time | Status |",
             "|---|---|---:|---:|---:|---:|---|"]
    for c in comparisons:
        lines.append(f"| {c['case']} | {c['factor']} | {c['byte_identical']} | {c['pixel_delta']['max_abs']:.6g} | "
                     f"{c['quality_delta_ssimulacra2']:.6g} | {c['time_ratio']:.4f} | {c['status']} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "comparisons"}, indent=2))
    if failed:
        raise RuntimeError("Exactness gates failed; see summary.json")


if __name__ == "__main__":
    main()
