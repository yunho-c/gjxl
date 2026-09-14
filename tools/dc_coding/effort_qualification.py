#!/usr/bin/env python3
"""Bounded e4 DC qualification; explicit collection, timing, and report phases.

Uses an unchanged source export and the existing complete-call benchmark.
The manifest freezes the protocol, inputs, sources, and external tools.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import visual_fixtures
from visual_fixtures import linear, srgb

MODES = {
    "default": ("weighted", "round", False),
    "quantize": ("weighted", "prediction-aware", False),
    "smooth": ("weighted", "round", True),
    "both": ("weighted", "prediction-aware", True),
    "control_a": ("weighted", "round", False),
    "control_b": ("weighted", "round", False),
}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check(value, message):
    if not value:
        raise RuntimeError(message)


def capture(*args):
    return subprocess.check_output(list(map(str, args)), text=True).strip()


def pfm(path):
    with Path(path).open("rb") as f:
        check(f.readline().strip() == b"PF", "Expected RGB PFM")
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        pixels = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4")
    return np.ascontiguousarray(pixels.reshape(h, w, 3)[::-1]) * abs(scale)


def write_pfm(path, pixels):
    with Path(path).open("wb") as f:
        f.write(f"PF\n{pixels.shape[1]} {pixels.shape[0]}\n-1.0\n".encode())
        np.asarray(pixels[::-1], dtype="<f4").tofile(f)


def picture(path, pixels):
    Image.fromarray(np.uint8(np.clip(pixels, 0, 1) * 255 + .5)).save(path)


def helpers(out):
    sys.path.insert(0, str(out / "source/tools/dc_coding"))
    import processing_study as study
    study.CONTROLS = MODES
    return study


def state():
    rows = []
    for line in capture("ps", "-axo", "pid=,pcpu=,comm=").splitlines():
        pid, cpu, command = line.strip().split(None, 2)
        if int(pid) != os.getpid():
            rows.append({"pid": int(pid), "cpu": float(cpu), "command": command})
    return {"time": time.time(), "cpu_total": sum(r["cpu"] for r in rows),
            "top_cpu": sorted(rows, key=lambda r: -r["cpu"])[:20],
            "power": capture("pmset", "-g", "batt"),
            "thermal": capture("pmset", "-g", "therm")}


def initialize(out, previous, synthetic_only=False):
    check(not (out / "manifest.json").exists(), "Manifest exists")
    prior = read(previous / "manifest.json")
    inputs = out / "inputs"
    inputs.mkdir(exist_ok=True)
    images = []
    for old in ([] if synthetic_only else prior["images"]):
        check(sha(old["path"]) == old["sha256"], f"Input changed: {old['path']}")
        if old["size_class"] != "12mp":
            continue
        original = pfm(old["path"])
        width = 2048
        height = round(original.shape[0] * width / original.shape[1])
        resized = np.stack([np.array(Image.fromarray(original[:, :, c]).resize(
            (width, height), Image.Resampling.LANCZOS)) for c in range(3)], axis=2)
        pixels = np.clip(resized, 0, 1)
        name = old["photo"] + "-visual"
        path = inputs / (name + ".pfm")
        write_pfm(path, pixels)
        images.append({"name": name, "kind": "photographic", "path": str(path),
                       "width": width, "height": height, "sha256": sha(path),
                       "parent": old, "preparation": "Pillow per-channel float Lanczos, 2048 wide, clip [0,1]",
                       "clipped_samples": int(np.count_nonzero(resized != pixels))})
    signals = visual_fixtures.make_gradients()
    for name, pixels in signals.items():
        path = inputs / (name + ".pfm")
        write_pfm(path, pixels)
        images.append({"name": name, "kind": "synthetic-gradient", "path": str(path),
                       "width": 2048, "height": 1024, "sha256": sha(path),
                       "preparation": "Deterministic analytic float signal; see frozen runner"})
    for image in images:
        picture(inputs / (image["name"] + ".png"), srgb(pfm(image["path"])))
    cases = [{"id": f"{i['name']}-e{e}-d{d:g}", "image": i, "effort": e, "distance": d}
             for i in images for e in (3, 4, 7) for d in (1.2, 3., 6.)]
    timing = [{"id": f"{i['name']}-e{e}", "image": i, "effort": e, "distance": 1.2}
              for i in ([] if synthetic_only else prior["images"]) for e in (3, 4, 7) if e != 3 or i["size_class"] == "12mp"]
    random.Random(20260913).shuffle(timing)
    benchmark = out / "release/gjxl_dc_processing_benchmark"
    check("no work to do" in capture("ninja", "-C", benchmark.parent, "-n", benchmark.name), "Pending native build")
    files = [p for p in (out / "source").rglob("*") if p.is_file()]
    files += [p for p in (out / "external").rglob("*") if p.is_file()]
    files += [benchmark, out / "release/gjxl_quality_benchmark", out / "PROTOCOL.md", Path(__file__).resolve(), Path(visual_fixtures.__file__).resolve()]
    m = {"revision": read(out / "source-identity.json"), "identities": {str(p): sha(p) for p in files},
         "benchmark": str(benchmark), "quality_benchmark": str(out / "release/gjxl_quality_benchmark"),
         "djxl": str(out / "external/decoder/tools/djxl"), "cjxl": str(out / "external/stock/tools/cjxl"),
         "scorer": str(out / "external/cjxl-quality-metric"), "scorer_version": prior["scorer_version"],
         "backend": "metal", "threads": 8, "warmups": 3, "rounds": 20,
         "visual_cases": cases, "timing_cases": timing, "images": images,
         "size_tolerance": .01, "max_size_probes": 10, "distance_bounds": [.1, 12.],
         "machine": state(), "python": sys.version, "numpy": np.__version__,
         "pillow": Image.__version__, "source_build_commands": capture("ninja", "-C", benchmark.parent, "-t", "commands", benchmark.name)}
    save(out / "manifest.json", m)
    print(f"Initialized {len(cases)} visual cases and {len(timing)} timing cases", flush=True)


def verify(m):
    for path, digest in m["identities"].items():
        check(sha(path) == digest, f"Frozen identity changed: {path}")


def ready(out, label):
    check(shutil.disk_usage(out).free > 4 * 1024**3, "Less than 4 GiB available")
    # Fail before creating an attempt if another encoder/build is active.
    before = state()
    blockers = [r for r in before["top_cpu"] if Path(r["command"]).name in
                ("cjxl", "djxl", "cjxl.real", "djxl.real", "ninja", "ctest", "clang", "clang++")
                or Path(r["command"]).name.startswith("gjxl_") or
                (r["cpu"] > 100 and Path(r["command"]).name != "kernel_task")]
    if blockers:
        save(out / "prelaunch-stops" / f"{time.time_ns()}.json", {"case": label, "state": before, "blockers": blockers})
        raise RuntimeError(f"Prelaunch background-work stop: {blockers}")
    return before


def measure(out, m, case, distances, parent, timing=False):
    study = helpers(out)
    check(sha(case["image"]["path"]) == case["image"]["sha256"], "Input changed")
    before = ready(out, case["id"])
    folder = study.common.new_attempt(parent)
    rows = study.encode(m, case, distances, folder, m["warmups"] if timing else 0, m["rounds"] if timing else 1)
    encoder = read(folder / "encode.stdout")
    check(encoder["revision"] == m["revision"]["revision"], "Wrong compiled revision")
    after = state()
    if not timing:
        study.score(m, case, rows, folder)
    record = {"case": case, "rows": rows, "encoder": encoder, "machine_before": before, "machine_after": after}
    save(folder / "complete.json", record)
    return record


def collect(out, m):
    for case in m["visual_cases"]:
        base = out / "visual" / case["id"]
        if (base / "complete.json").exists():
            helpers(out).audit(read(base / "complete.json"))
            continue
        result = measure(out, m, case, {n: case["distance"] for n in list(MODES)[:4]}, base)
        save(base / "complete.json", result)
        print("VISUAL", case["id"], {n: (r["encoded_bytes"], round(r["quality"], 4)) for n, r in result["rows"].items()}, flush=True)


def next_size_distance(rows, target):
    rows = sorted(rows, key=lambda r: r["distance"])
    brackets = [(a, b) for a, b in zip(rows, rows[1:]) if (a["encoded_bytes"] - target) * (b["encoded_bytes"] - target) < 0]
    if brackets:
        a, b = min(brackets, key=lambda p: p[1]["distance"] - p[0]["distance"])
        f = (math.log(target)-math.log(a["encoded_bytes"]))/(math.log(b["encoded_bytes"])-math.log(a["encoded_bytes"]))
        return a["distance"] + min(.8, max(.2, f)) * (b["distance"]-a["distance"])
    if all(r["encoded_bytes"] > target for r in rows):
        return rows[-1]["distance"] * 1.2
    if all(r["encoded_bytes"] < target for r in rows):
        return rows[0]["distance"] / 1.2
    raise RuntimeError("Cannot select matched-size probe")


def calibrate(out, m):
    for case in m["visual_cases"]:
        if case["effort"] == 3 or case["distance"] == 1.2:
            continue
        base = out / "matched-size" / case["id"]
        if (base / "selection.json").exists():
            continue
        initial = read(out / "visual" / case["id"] / "complete.json")
        target = initial["rows"]["default"]["encoded_bytes"]
        rows = [initial["rows"]["both"]]
        for folder in sorted(base.glob("attempt-*")):
            check((folder / "complete.json").exists(), f"Inspect incomplete attempt: {folder}")
            rows.append(read(folder / "complete.json")["rows"]["both"])
        error = lambda r: abs(r["encoded_bytes"] / target - 1)
        reason = "probe-budget-exhausted"
        while min(map(error, rows)) > m["size_tolerance"] and len(rows) < m["max_size_probes"]:
            distance = next_size_distance(rows, target)
            if not m["distance_bounds"][0] <= distance <= m["distance_bounds"][1]:
                reason = "distance-bound"; break
            if any(abs(distance-r["distance"]) < 1e-7 for r in rows):
                reason = "distance-stalled"; break
            rows.append(measure(out, m, case, {"both": distance}, base)["rows"]["both"])
        selected = min(rows, key=error)
        passing = error(selected) <= m["size_tolerance"]
        save(base / "selection.json", {"case": case, "default": initial["rows"]["default"], "both": selected,
                                       "complete": passing, "reason": "within-tolerance" if passing else reason,
                                       "probes": len(rows), "relative_size_error": error(selected)})
        print("MATCHED-SIZE", case["id"], passing, f"error={100*error(selected):.3f}%", flush=True)


def timing(out, m):
    study = helpers(out)
    control = next(c for c in m["timing_cases"] if c["effort"] == 4 and "alpine_lake--12mp" in c["id"])
    for number in range(3):
        base = out / "controls" / str(number)
        if (base / "complete.json").exists():
            continue
        result = measure(out, m, control, {"control_a": 1.2, "control_b": 1.2}, base, timing=True)
        check(result["rows"]["control_a"]["sha256"] == result["rows"]["control_b"]["sha256"], "Control differs")
        save(base / "complete.json", result)
        print("CONTROL", number, flush=True)
    for case in m["timing_cases"]:
        base = out / "timing" / case["id"]
        if (base / "complete.json").exists():
            study.audit(read(base / "complete.json"))
            continue
        result = measure(out, m, case, {"default": 1.2, "both": 1.2}, base, timing=True)
        folder = Path(result["rows"]["default"]["artifact"]).parent.parent
        study.score(m, case, result["rows"], folder)
        save(base / "complete.json", result)
        print("TIMED", case["id"], {n: round(statistics.median(r["samples_ns"])/1e6, 3) for n, r in result["rows"].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "collect", "calibrate", "timing"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("build/dc-photo-large-20260913-v2"))
    parser.add_argument("--synthetic-only", action="store_true", help="Initialize only the four diagnostic gradients; no photographic timing")
    args = parser.parse_args()
    out = args.output.resolve()
    with (out / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "init":
            initialize(out, args.previous.resolve(), args.synthetic_only)
        else:
            m = read(out / "manifest.json")
            verify(m)
            {"collect": collect, "calibrate": calibrate, "timing": timing}[args.command](out, m)


if __name__ == "__main__":
    main()
