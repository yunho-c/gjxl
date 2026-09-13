"""Source-bound DC rate/quality curves and matched-quality complete-call timing.

init freezes the experiment; collect, calibrate, and timing are explicit,
resumable commands. report consumes retained evidence without running codecs.
"""
import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import zipfile

import study as common

ROOT = common.ROOT
CONTROLS = {
    "default": ("gradient", "round", False),
    "weighted": ("weighted", "round", False),
    "quantize": ("weighted", "prediction-aware", False),
    "smooth": ("weighted", "round", True),
    "both": ("weighted", "prediction-aware", True),
}
sha, load, save, check = common.sha, common.load, common.save, common.check


def run(argv, folder, name, stdin=None):
    argv = list(map(str, argv))
    start = time.monotonic()
    if stdin is not None:
        (folder / f"{name}.stdin").write_text(stdin)
    env = {**os.environ, "RAYON_NUM_THREADS": "8"}
    with (folder / f"{name}.stdout").open("w") as out, (folder / f"{name}.stderr").open("w") as err:
        result = subprocess.run(argv, input=stdin, text=True, stdout=out, stderr=err,
                                timeout=900, env=env)
    save(folder / f"{name}.command.json", {"argv": argv, "exit_code": result.returncode,
         "elapsed_seconds": time.monotonic() - start, "RAYON_NUM_THREADS": "8"})
    check(result.returncode == 0, f"Command failed: {folder / (name + '.stderr')}")


def initialize(args):
    output = args.output.resolve()
    check(not output.exists(), "Output already exists; resume or choose a fresh directory")
    old = load(args.weighted_manifest)
    unique = {c["image"]["name"]: c["image"] for c in old["cases"]}
    images = []
    for stratum in ("clic2024_test", "kodak", "photo_4k"):
        pool = sorted((x for x in unique.values() if x["stratum"] == stratum), key=lambda x: x["name"])
        count = (1 if stratum != "photo_4k" else 0) if args.pilot else (2 if stratum == "photo_4k" else 6)
        images.extend(pool[round(i * (len(pool) - 1) / max(1, count - 1))] for i in range(count))
    images.extend(unique[n] for n in (["edge", "gradient"] if args.pilot else ["edge", "gradient", "saturated", "texture"]))
    for item in images:
        check(sha(item["path"]) == item["sha256"], f"Input changed: {item['name']}")
    benchmark = args.benchmark.resolve()
    check("no work to do" in common.capture(["ninja", "-C", benchmark.parent, "-n", "gjxl_dc_processing_benchmark"]),
          "Benchmark needs rebuilding")
    decoder = load(args.decoder_build)
    paths = [benchmark, args.djxl.resolve(), args.scorer.resolve(), Path(decoder["binary"])]
    paths.extend(Path(p) for p in decoder["libraries"])
    check(sha(decoder["binary"]) == decoder["binary_sha256"], "Decoder helper identity changed")
    check(sha(ROOT / "tools/dc_coding/decode_benchmark.cpp") == decoder["source_sha256"], "Decoder helper needs rebuilding")
    distances = [0.7, 1.0, 1.4] if args.pilot else [0.5, 0.7, 1.0, 1.4, 2.0]
    cases = [{"id": f"{image['name']}-e{effort}-d{distance:g}", "image": image,
              "effort": effort, "distance": distance}
             for image in images for effort in (3, 4, 7) for distance in distances]
    defaults = {}
    for case in cases:
        previous = args.weighted_manifest.parent / "cases" / case["id"] / "complete.json"
        if previous.exists():
            record = load(previous)
            row = next(r for r in record["rows"] if r["variant"] == "gradient")
            check(sha(row["artifact"]) == row["sha256"], "Retained default artifact changed")
            defaults[case["id"]] = row["sha256"]
    equivalent = {}
    if args.equivalent_study is not None:
        previous_manifest = load(args.equivalent_study / "manifest.json")
        check(previous_manifest["controls"] == {n: list(v) for n, v in CONTROLS.items()},
              "Equivalent study used different controls")
        for case in cases:
            complete = args.equivalent_study / "curves" / case["id"] / "complete.json"
            if complete.exists():
                record = load(complete)
                audit(record)
                check(record["case"] == case, "Equivalent study case differs")
                equivalent[case["id"]] = {n: r["sha256"] for n, r in record["rows"].items()}
    timing_images = {next(i["name"] for i in images if i["stratum"] == s) for s in ("clic2024_test", "kodak")}
    timing_images.add("edge")
    if not args.pilot:
        timing_images.add(next(i["name"] for i in images if i["stratum"] == "photo_4k"))
    timing_cases = [c for c in cases if c["distance"] == 1 and c["image"]["name"] in timing_images]
    manifest = {"schema_version": 1, "purpose": "DC lossy matched-quality qualification", "pilot": args.pilot,
        "source_revision": common.capture(["git", "-C", ROOT, "rev-parse", "HEAD"]),
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in common.source_files()},
        "tools": {str(p): sha(p) for p in paths}, "benchmark": str(benchmark), "decoder": decoder,
        "djxl": str(args.djxl.resolve()), "scorer": str(args.scorer.resolve()),
        "scorer_version": json.loads(common.capture([args.scorer, "--version"])),
        "controls": CONTROLS, "images": images, "cases": cases, "timing_cases": timing_cases,
        "distances": distances, "default_regressions": defaults, "threads": 8, "backend": "metal",
        "equivalent_regressions": equivalent,
        "equivalent_study": None if args.equivalent_study is None else {
            "path": str(args.equivalent_study.resolve()),
            "manifest_sha256": sha(args.equivalent_study / "manifest.json")},
        "warmups": 3, "rounds": 10 if args.pilot else 20, "decode_pairs": 5 if args.pilot else 20,
        "quality_tolerance": 0.05, "maximum_calibration_probes": 10,
        "quality_metric": "SSIMULACRA2 on independently decoded linear-sRGB float pixels",
        "build_cache_sha256": sha(benchmark.parent / "CMakeCache.txt"),
        "build_commands": common.capture(["ninja", "-C", benchmark.parent, "-t", "commands", "gjxl_dc_processing_benchmark"]),
        "input_manifest": {"path": str(args.weighted_manifest.resolve()), "sha256": sha(args.weighted_manifest)},
        "machine": {"os": common.capture(["sw_vers"]), "hardware": common.capture(["sysctl", "-n", "hw.model"]),
                    "initial_state": common.machine_state()}}
    output.mkdir(parents=True)
    save(output / "manifest.json", manifest)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in common.source_files():
            archive.write(path, str(path.relative_to(ROOT)))
    (output / "source.diff").write_text(common.capture(["git", "-C", ROOT, "diff", "HEAD", "--"]))
    print(f"Initialized {len(cases)} curve cases ({len(cases) * len(CONTROLS)} outputs) and {len(timing_cases)} timing cases", flush=True)


def encode(manifest, case, distances, folder, warmups=0, rounds=1):
    specification = folder / "variants.txt"
    specification.write_text("".join(f"{name} {distance:.9g} {p} {q} {int(s)}\n"
        for name, distance in distances.items() for p, q, s in [CONTROLS[name]]))
    output = folder / "encoded"
    run([manifest["benchmark"], case["image"]["path"], output, specification,
         case["effort"], manifest["threads"], warmups, rounds, manifest["backend"]], folder, "encode")
    report = load(folder / "encode.stdout")
    check(report["backend"] == manifest["backend"] and report["thread_count"] == manifest["threads"] and
          report["effort"] == case["effort"] and report["input_width"] == case["image"]["width"] and
          report["input_height"] == case["image"]["height"] and not report["stage_profile_enabled"] and
          not report["collect_final_score"] and len(report["samples"]) == rounds * len(distances), "Encode provenance mismatch")
    check({v["name"] for v in report["variants"]} == distances.keys(), "Variant set changed")
    rows = {}
    for variant in report["variants"]:
        name = variant["name"]
        p, q, s = CONTROLS[name]
        artifact = output / f"{name}.jxl"
        check(variant["dc_prediction"] == p and variant["dc_quantization"] == q and
              variant["adaptive_dc_smoothing"] == s and math.isclose(variant["distance"], distances[name], rel_tol=1e-6) and
              artifact.stat().st_size == variant["encoded_bytes"], "DC control/size mismatch")
        samples = [v["elapsed_nanoseconds"] for v in report["samples"] if v["variant"] == name]
        check(len(samples) == rounds and all(t > 0 for t in samples), "Incomplete timing samples")
        rows[name] = {**variant, "artifact": str(artifact), "sha256": sha(artifact), "samples_ns": samples}
    return rows


def pixel_identity(path, width, height):
    import numpy as np
    with path.open("rb") as stream:
        def line():
            while True:
                value = stream.readline()
                check(value, "Truncated PFM header")
                value = value.split(b"#")[0].strip()
                if value:
                    return value
        check(line() == b"PF", "Expected RGB float PFM")
        check(list(map(int, line().split())) == [width, height], "Decoded geometry changed")
        scale = float(line())
        check(math.isfinite(scale) and scale != 0, "Invalid PFM scale")
        raw = stream.read()
    check(len(raw) == width * height * 12, "Decoded payload size changed")
    pixels = np.frombuffer(raw, dtype="<f4" if scale < 0 else ">f4").reshape(height, width, 3)
    pixels = np.ascontiguousarray(pixels[::-1], dtype=np.float32) * np.float32(abs(scale))
    check(bool(np.isfinite(pixels).all()), "Decoded pixels are not finite")
    return {"sha256": hashlib.sha256(pixels.tobytes()).hexdigest(), "pfm_sha256": sha(path), "finite": True}


def score(manifest, case, rows, folder):
    requests, decoded = [], {}
    image = case["image"]
    # Both lossless residual predictors must preserve exactly the same pixels.
    equality = None
    if "default" in rows and "weighted" in rows:
        run([manifest["decoder"]["binary"], rows["default"]["artifact"], rows["weighted"]["artifact"],
             image["width"], image["height"], 0, 0], folder, "predictor-equality")
        equality = load(folder / "predictor-equality.stdout")
        check(equality["decoded_equal"] and equality["decoded_finite"], "Lossless predictor changed pixels")
    for name, row in rows.items():
        if name == "weighted" and equality is not None:
            continue
        path = folder / f"{name}.pfm"
        run([manifest["djxl"], "--quiet", "--num_threads=1", "--bits_per_sample=32",
             "--color_space=RGB_D65_SRG_Rel_Lin", row["artifact"], path], folder, f"decode-{name}")
        decoded[name] = pixel_identity(path, image["width"], image["height"])
        requests.append({"path": str(path)})
    run([manifest["scorer"], image["path"], "auto"], folder, "metric",
        "".join(json.dumps(r) + "\n" for r in requests))
    result = [json.loads(line) for line in (folder / "metric.stdout").read_text().splitlines()]
    ready = result[0]
    check(ready.get("ready") and ready["width"] == image["width"] and ready["height"] == image["height"] and
          ready["fast_ssim2_revision"] == manifest["scorer_version"]["fast_ssim2_revision"] and
          len(result) == len(requests) + 1, "Metric provenance mismatch")
    for request, value in zip(requests, result[1:]):
        check(value.get("path") == request["path"] and math.isfinite(value.get("score", float("nan"))), "Metric failed")
        name = Path(request["path"]).stem
        rows[name].update({"quality": value["score"], "decoded": decoded[name], "metric_mode": ready["mode"]})
    if equality is not None:
        check(rows["default"]["decoded"]["sha256"] == equality["decoded_sha256"], "djxl and helper float output disagree")
        rows["weighted"].update({k: rows["default"][k] for k in ("quality", "decoded", "metric_mode")})
    # Retain JXL, commands, metric output and pixel hashes. Only these generated
    # scratch PFMs are removed after successful scoring; they can be recreated.
    for request in requests:
        Path(request["path"]).unlink()
    return equality


def audit(record):
    for row in record["rows"].values():
        check(sha(row["artifact"]) == row["sha256"] and math.isfinite(row["quality"]) and row["decoded"]["finite"],
              "Saved rate/quality artifact changed or is invalid")


def collect(args, manifest):
    added = 0
    for index, case in enumerate(manifest["cases"]):
        complete = args.output / "curves" / case["id"] / "complete.json"
        if complete.exists():
            audit(load(complete))
            continue
        if args.max_cases is not None and added >= args.max_cases:
            break
        check(sha(case["image"]["path"]) == case["image"]["sha256"], "Input changed")
        state = common.machine_state()
        folder = common.new_attempt(complete.parent)
        names = list(CONTROLS)
        names = names[index % len(names):] + names[:index % len(names)]
        rows = encode(manifest, case, {n: case["distance"] for n in names}, folder)
        expected = manifest["default_regressions"].get(case["id"])
        check(expected is None or rows["default"]["sha256"] == expected, "Default output regression")
        equivalent = manifest.get("equivalent_regressions", {}).get(case["id"])
        check(equivalent is None or all(rows[n]["sha256"] == expected for n, expected in equivalent.items()),
              "Equivalent implementation changed output bytes")
        equality = score(manifest, case, rows, folder)
        save(complete, {"case": case, "rows": rows, "equality": equality,
                        "default_regression_checked": expected is not None,
                        "equivalent_regression_checked": equivalent is not None, "machine": state})
        added += 1
        print(f"Curve {index + 1}/{len(manifest['cases'])}: {case['id']}", flush=True)


def calibration(args, manifest):
    added = 0
    for case in manifest["timing_cases"]:
        base = args.output / "calibration" / case["id"]
        complete = base / "complete.json"
        if complete.exists():
            audit(load(complete))
            continue
        if args.max_cases is not None and added >= args.max_cases:
            break
        check(sha(case["image"]["path"]) == case["image"]["sha256"], "Calibration input changed")
        curves = [load(args.output / "curves" / c["id"] / "complete.json") for c in manifest["cases"]
                  if c["image"]["name"] == case["image"]["name"] and c["effort"] == case["effort"]]
        for record in curves:
            audit(record)
        anchor = next(r for r in curves if r["case"]["distance"] == 1)
        target = anchor["rows"]["default"]["quality"]
        selected, unresolved = {}, []
        for name in CONTROLS:
            if name in ("default", "weighted"):
                selected[name] = anchor["rows"][name]
                continue
            evidence = [r["rows"][name] for r in curves]
            saved = base / name / "complete.json"
            if saved.exists():
                saved_row = load(saved)
                audit({"rows": {name: saved_row}})
                selected[name] = saved_row
                continue
            # Retain and reuse every completed probe, including across resumes.
            for probe in sorted((base / name).glob("attempt-*/complete.json")):
                record = load(probe)
                audit(record)
                evidence.append(record["rows"][name])
            for _ in range(manifest["maximum_calibration_probes"] + 1):
                best = min(evidence, key=lambda r: abs(r["quality"] - target))
                if abs(best["quality"] - target) <= manifest["quality_tolerance"]:
                    save(saved, best)
                    selected[name] = best
                    break
                probes = len(evidence) - len(curves)
                if probes >= manifest["maximum_calibration_probes"]:
                    unresolved.append({"variant": name, "reason": "probe-budget-exhausted", "best": best})
                    break
                ordered = sorted(evidence, key=lambda r: r["distance"])
                brackets = [(a, b) for a, b in zip(ordered, ordered[1:])
                            if (a["quality"] - target) * (b["quality"] - target) < 0]
                if not brackets:
                    unresolved.append({"variant": name, "reason": "target-not-bracketed", "best": best})
                    break
                a, b = min(brackets, key=lambda pair: pair[1]["distance"] - pair[0]["distance"])
                fraction = (target - a["quality"]) / (b["quality"] - a["quality"])
                fraction = min(0.8, max(0.2, fraction))
                distance = a["distance"] + fraction * (b["distance"] - a["distance"])
                state = common.machine_state()
                folder = common.new_attempt(base / name)
                rows = encode(manifest, case, {name: distance}, folder)
                score(manifest, case, rows, folder)
                record = {"case": case, "rows": rows, "target": target, "machine": state}
                save(folder / "complete.json", record)
                evidence.append(rows[name])
            check(name in selected or any(r["variant"] == name for r in unresolved), "Calibration lost a variant")
        if unresolved:
            save(base / "unresolved.json", {"case": case, "target": target, "unresolved": unresolved})
            print(f"Unresolved calibration: {case['id']}", flush=True)
        else:
            save(complete, {"case": case, "target": target, "rows": selected})
            print(f"Calibrated: {case['id']}", flush=True)
        added += 1


def timing(args, manifest):
    added = 0
    for case in manifest["timing_cases"]:
        complete = args.output / "timing" / case["id"] / "complete.json"
        if complete.exists():
            record = load(complete)
            audit(record)
            continue
        if args.max_cases is not None and added >= args.max_cases:
            break
        check(sha(case["image"]["path"]) == case["image"]["sha256"], "Timing input changed")
        calibrated = args.output / "calibration" / case["id"] / "complete.json"
        if not calibrated.exists():
            print(f"Timing awaits calibration: {case['id']}", flush=True)
            continue
        reference = load(calibrated)
        audit(reference)
        check(all(abs(r["quality"] - reference["target"]) <= manifest["quality_tolerance"] for r in reference["rows"].values()),
              "Calibration misses declared quality tolerance")
        state = common.machine_state()
        folder = common.new_attempt(complete.parent)
        rows = encode(manifest, case, {n: r["distance"] for n, r in reference["rows"].items()}, folder,
                      manifest["warmups"], manifest["rounds"])
        for name, row in rows.items():
            check(row["sha256"] == reference["rows"][name]["sha256"], "Timed encode changed calibrated pixels")
            row.update({k: reference["rows"][name][k] for k in ("quality", "decoded", "metric_mode")})
        decodes = {}
        for name in CONTROLS:
            if name == "default":
                continue
            run([manifest["decoder"]["binary"], rows["default"]["artifact"], rows[name]["artifact"],
                 case["image"]["width"], case["image"]["height"], manifest["warmups"], manifest["decode_pairs"],
                 "--independent-pixels"], folder, f"decode-time-{name}")
            result = load(folder / f"decode-time-{name}.stdout")
            check(result["decoded_finite"] and result["decoded_sha256"] == rows["default"]["decoded"]["sha256"] and
                  result["second_decoded_sha256"] == rows[name]["decoded"]["sha256"], "Timed decode changed pixels")
            decodes[name] = result
        save(complete, {"case": case, "target": reference["target"], "rows": rows, "decodes": decodes,
                        "encoder": load(folder / "encode.stdout"), "machine_before": state,
                        "machine_after": common.machine_state()})
        added += 1
        print(f"Timed: {case['id']}", flush=True)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(args, manifest):
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    points, groups, missing = [], defaultdict(lambda: defaultdict(list)), []
    for case in manifest["cases"]:
        complete = args.output / "curves" / case["id"] / "complete.json"
        if not complete.exists():
            missing.append(case["id"])
            continue
        record = load(complete)
        audit(record)
        for name, row in record["rows"].items():
            point = {"image": case["image"]["name"], "stratum": case["image"]["stratum"], "effort": case["effort"],
                     "variant": name, "distance": row["distance"], "bytes": row["encoded_bytes"], "quality": row["quality"]}
            points.append(point)
            groups[(point["image"], point["stratum"], point["effort"])][name].append(point)
    rates, excluded = [], []
    for key, variants in groups.items():
        if any(len(variants[n]) != len(manifest["distances"]) for n in CONTROLS):
            excluded.append({"group": key, "reason": "incomplete-curve"})
            continue
        frontiers = {}
        for name, rows in variants.items():
            # Remove strictly dominated rate/quality points; retain every raw
            # observation in curves.csv. No quality extrapolation or polynomial fit.
            frontier = []
            for row in sorted(rows, key=lambda r: (r["bytes"], -r["quality"])):
                if not frontier or row["quality"] > frontier[-1]["quality"] + 1e-9:
                    frontier.append(row)
            frontiers[name] = frontier
        low = max(v[0]["quality"] for v in frontiers.values())
        high = min(v[-1]["quality"] for v in frontiers.values())
        if min(map(len, frontiers.values())) < 3 or high - low < 1e-6:
            excluded.append({"group": key, "reason": "insufficient-common-quality-span",
                             "frontier_points": {n: len(v) for n, v in frontiers.items()}, "low": low, "high": high})
            continue
        integrals = {n: float(PchipInterpolator([r["quality"] for r in v],
                         np.log([r["bytes"] for r in v]), extrapolate=False).integrate(low, high))
                     for n, v in frontiers.items()}
        for reference in ("default", "weighted"):
            for name in CONTROLS:
                if name == reference:
                    continue
                rate = 100 * math.expm1((integrals[name] - integrals[reference]) / (high - low))
                rates.append({"image": key[0], "stratum": key[1], "effort": key[2], "reference": reference,
                              "variant": name, "bd_rate_percent": rate, "quality_low": low, "quality_high": high})
    times, timing_missing, calibration_unresolved = [], [], []
    for case in manifest["timing_cases"]:
        unresolved = args.output / "calibration" / case["id"] / "unresolved.json"
        calibrated = args.output / "calibration" / case["id"] / "complete.json"
        if unresolved.exists() and not calibrated.exists():
            calibration_unresolved.append(load(unresolved))
        complete = args.output / "timing" / case["id"] / "complete.json"
        if not complete.exists():
            timing_missing.append(case["id"])
            continue
        record = load(complete)
        audit(record)
        samples = {(s["round"], s["variant"]): s["elapsed_nanoseconds"] for s in record["encoder"]["samples"]}
        baseline = record["rows"]["default"]
        for name, row in record["rows"].items():
            ratios = [samples[r, name] / samples[r, "default"] for r in range(manifest["rounds"])]
            decode = record["decodes"].get(name)
            decode_ratio, decode_ms = None, None
            if decode:
                pairs = {(s["pair"], s["variant"]): s["elapsed_nanoseconds"] for s in decode["samples"]}
                decode_ratio = statistics.median(pairs[p, "weighted"] / pairs[p, "gradient"] for p in range(manifest["decode_pairs"]))
                decode_ms = statistics.median(s["elapsed_nanoseconds"] for s in decode["samples"] if s["variant"] == "weighted") / 1e6
            times.append({"image": case["image"]["name"], "stratum": case["image"]["stratum"], "effort": case["effort"],
                "variant": name, "distance": row["distance"], "target_quality": record["target"], "quality": row["quality"],
                "bytes": row["encoded_bytes"], "byte_change_percent": 100 * (row["encoded_bytes"] / baseline["encoded_bytes"] - 1),
                "encode_ms": statistics.median(row["samples_ns"]) / 1e6, "paired_encode_change_percent": 100 * (statistics.median(ratios) - 1),
                "decode_ms": decode_ms, "paired_decode_change_percent": None if decode_ratio is None else 100 * (decode_ratio - 1)})
    write_csv(args.output / "curves.csv", points)
    write_csv(args.output / "bd-rate.csv", rates)
    write_csv(args.output / "timing.csv", times)
    summary = {"curve_cases_complete": len(manifest["cases"]) - len(missing), "curve_cases_expected": len(manifest["cases"]),
               "curve_missing": missing, "bd_rate_groups_excluded": excluded,
               "timing_complete": len(manifest["timing_cases"]) - len(timing_missing), "timing_expected": len(manifest["timing_cases"]),
               "timing_missing": timing_missing, "calibration_unresolved": calibration_unresolved, "rate_summaries": []}
    bins = defaultdict(list)
    for row in rates:
        bins[(row["stratum"], row["effort"], row["reference"], row["variant"])].append(row["bd_rate_percent"])
    for key, values in sorted(bins.items()):
        summary["rate_summaries"].append({"stratum": key[0], "effort": key[1], "reference": key[2], "variant": key[3],
                                         "images": len(values), "median_percent": statistics.median(values),
                                         "minimum_percent": min(values), "maximum_percent": max(values)})
    save(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("curve_cases_complete", "curve_cases_expected", "timing_complete", "timing_expected")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "collect", "calibrate", "timing", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weighted-manifest", type=Path, default=ROOT / "build/dc-weighted-qualification-v1/manifest.json")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "build/release/gjxl_dc_processing_benchmark")
    parser.add_argument("--decoder-build", type=Path, default=ROOT / "build/dc-processing-tools/build.json")
    parser.add_argument("--djxl", type=Path, default=ROOT / "build/dc-libjxl-reference/tools/djxl")
    parser.add_argument("--scorer", type=Path, default=Path("/Users/yunhocho/GitHub/libjxl/tools/scripts/quality_metric/target/release/cjxl-quality-metric"))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--equivalent-study", type=Path)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    check(args.max_cases is None or args.max_cases > 0, "max-cases must be positive")
    if args.command == "init":
        initialize(args)
        return
    manifest = load(args.output / "manifest.json")
    if args.command == "report":
        report(args, manifest)
        return
    common.verify_identity(manifest)
    with (args.output / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"collect": collect, "calibrate": calibration, "timing": timing}[args.command](args, manifest)


if __name__ == "__main__":
    main()
