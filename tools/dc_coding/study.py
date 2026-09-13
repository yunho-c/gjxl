"""Resumable, source-bound qualification of lossless weighted DC prediction.

Collection and timing are explicit commands. Reporting reads saved artifacts only.
Use a new output directory whenever source, tools, inputs, or settings change.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE_REVISION = "2c936fa96a334d7f67e04abf85fb38ccfcdf73f5"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def source_files():
    prefixes = ("src", "include", "cmake", "benchmarks", "tools/dc_coding")
    files = [ROOT / "CMakeLists.txt"]
    for prefix in prefixes:
        files.extend(p for p in (ROOT / prefix).rglob("*") if p.is_file() and
                     p.suffix in (".cpp", ".h", ".hpp", ".mm", ".metal", ".cmake", ".py", ".inc"))
    return sorted(files)


def capture(args):
    return subprocess.check_output(list(map(str, args)), text=True).strip()


def machine_state():
    text = capture(["ps", "-axo", "pid=,pcpu=,comm="])
    busy = []
    rows = []
    for line in text.splitlines():
        pid, cpu, command = line.strip().split(None, 2)
        if int(pid) == os.getpid():
            continue
        row = {"pid": int(pid), "cpu_percent": float(cpu), "command": command}
        rows.append(row)
        name = Path(command).name
        if (name.startswith("gjxl_") or name in ("djxl", "cjxl", "decode_benchmark", "ctest", "ninja", "cargo", "clang", "clang++")
                or (float(cpu) > 100 and name != "kernel_task")):
            busy.append(row)
    check(not busy, f"Other measurement/build or sustained CPU work is active: {busy}")
    return {"time_unix": time.time(), "top_cpu": sorted(rows, key=lambda r: r["cpu_percent"], reverse=True)[:12],
            "power": capture(["pmset", "-g", "batt"])}


def inputs_from(args):
    metadata = load(args.metadata)
    images = [{"name": x["image_id"].replace("/", "--"), "stratum": x["corpus"],
               "path": x["pfm_path"], "sha256": x["pfm_sha256"],
               "width": x["width"], "height": x["height"]}
              for x in metadata["images"] if x["corpus"] in ("kodak", "clic2024_test")]
    old = load(args.historical)
    compact = {"flat", "gradient", "edge", "saturated", "texture", "noise", "flower", "grayscale"}
    photo = {"imazen26-1029-planter-4k", "imazen26-1039-sun-forest-4k",
             "imazen26-1049-river-city-4k", "imazen26-1207-bedroom-noise-4k"}
    found = {}
    for record in old["records"]:
        item = record["input"]
        if item["name"] in compact | photo:
            found[item["name"]] = {**item, "stratum": "compact" if item["name"] in compact else "photo_4k"}
    check(found.keys() == compact | photo, "Missing historical controls")
    images.extend(found.values())
    images.sort(key=lambda x: (x["stratum"], x["name"]))
    check(len(images) == 68, "Expected 32 CLIC, 24 Kodak, four 4K, eight compact inputs")
    if args.pilot:
        images = [next(x for x in images if x["stratum"] == "clic2024_test"),
                  found["imazen26-1029-planter-4k"], found["edge"], found["gradient"]]
    for item in images:
        check(sha(item["path"]) == item["sha256"], f"Input hash mismatch: {item['name']}")
    return images


def initialize(args):
    output = args.output.resolve()
    check(not output.exists(), "Output exists; use collect/timing to resume or a fresh namespace")
    images = inputs_from(args)
    candidate, baseline = args.candidate.resolve(), args.baseline.resolve()
    paired_encoder = args.paired_encoder.resolve()
    decoder_manifest = load(args.decoder_build)
    baseline_source = args.baseline_source.resolve()
    check(capture(["git", "-C", baseline_source, "rev-parse", "HEAD"]) == BASE_REVISION,
          "Unexpected baseline revision")
    subprocess.run(["git", "-C", baseline_source, "diff", "--quiet", "HEAD", "--"], check=True)
    for binary in (candidate, baseline):
        pending = capture(["ninja", "-C", binary.parent, "-n", "gjxl_quality_benchmark"])
        check("no work to do" in pending, f"Rebuild required: {binary}\n{pending}")
    check("no work to do" in capture(["ninja", "-C", paired_encoder.parent, "-n", "gjxl_dc_prediction_benchmark"]),
          "Paired encoder needs rebuilding")
    source = {str(p.relative_to(ROOT)): sha(p) for p in source_files()}
    tool_paths = [candidate, baseline, paired_encoder, Path(decoder_manifest["binary"])]
    tool_paths.extend(Path(p) for p in decoder_manifest["libraries"])
    distances = [1.0] if args.pilot else [0.6, 1.0, 2.0]
    cases = [{"id": f"{image['name']}-e{effort}-d{distance:g}", "image": image,
              "effort": effort, "distance": distance}
             for image in images for effort in (3, 4, 7) for distance in distances]
    timing_images = {next(x["name"] for x in images if x["stratum"] == "clic2024_test"),
                     "imazen26-1029-planter-4k", "edge", "gradient"}
    if not args.pilot:
        timing_images.update({next(x["name"] for x in images if x["stratum"] == "kodak"),
                              "imazen26-1039-sun-forest-4k"})
    manifest = {"schema_version": 1, "purpose": "current-effort lossless weighted DC qualification",
                "pilot": args.pilot, "source_revision": capture(["git", "-C", ROOT, "rev-parse", "HEAD"]),
                "source_root": str(ROOT), "sources": source, "baseline_revision": BASE_REVISION,
                "candidate": str(candidate), "baseline": str(baseline), "decoder": decoder_manifest,
                "paired_encoder": str(paired_encoder),
                "tools": {str(p): sha(p) for p in tool_paths}, "cases": cases,
                "timing_cases": [c["id"] for c in cases if c["image"]["name"] in timing_images and c["distance"] == 1],
                "threads": 8, "timing_warmups": 3,
                "timing_pairs": 5 if args.pilot else 20,
                "input_manifests": {str(p.resolve()): sha(p) for p in (args.metadata, args.historical)},
                "system": {"os": capture(["sw_vers"]), "hardware": capture(["sysctl", "-n", "hw.model"]),
                           "initial_state": machine_state()},
                "builds": {str(p.parent): {"cache": sha(p.parent / "CMakeCache.txt"),
                             "commands": capture(["ninja", "-C", p.parent, "-t", "commands", "gjxl_quality_benchmark"])}
                           for p in (candidate, baseline)}}
    output.mkdir(parents=True)
    save(output / "manifest.json", manifest)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            archive.write(path, str(path.relative_to(ROOT)))
    (output / "source.diff").write_text(capture(["git", "-C", ROOT, "diff", "HEAD", "--"]))
    print(f"Initialized {len(cases)} cases and {len(manifest['timing_cases'])} timing cases in {output}", flush=True)


def verify_identity(manifest):
    for relative, expected in manifest["sources"].items():
        check(sha(ROOT / relative) == expected, f"Source changed; use a fresh namespace: {relative}")
    for path, expected in manifest["tools"].items():
        check(sha(path) == expected, f"Tool changed; use a fresh namespace: {path}")


def command(argv, folder, name):
    argv = list(map(str, argv))
    start = time.monotonic()
    with (folder / f"{name}.stdout").open("w") as stdout, (folder / f"{name}.stderr").open("w") as stderr:
        result = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=900)
    save(folder / f"{name}.command.json", {"argv": argv, "exit_code": result.returncode,
                                          "elapsed_seconds": time.monotonic() - start})
    check(result.returncode == 0, f"Command failed: {folder / (name + '.stderr')}")


def encode(manifest, case, variant, folder, stem, warmups, samples):
    baseline = variant == "baseline"
    binary = manifest["baseline" if baseline else "candidate"]
    output, raw = folder / f"{stem}.jxl", folder / f"{stem}.json"
    argv = [binary, "--input", case["image"]["path"], "--output", output, "--raw-samples", raw,
            "--distance", case["distance"], "--effort", case["effort"], "--num-threads", manifest["threads"],
            "--warmups", warmups, "--samples", samples]
    if not baseline:
        argv += ["--dc-prediction", variant]
    command(argv, folder, stem)
    report = load(raw)
    check(report["backend"] == "metal" and report["metal_aq_mode"] == "fully-resident" and
          report["stage_profile_enabled"] is False and report["effort"] == case["effort"] and
          math.isclose(report["requested_distance"], case["distance"], rel_tol=1e-6) and
          report["input_width"] == case["image"]["width"] and
          report["input_height"] == case["image"]["height"] and
          report.get("dc_prediction", "gradient") == ("gradient" if baseline else variant) and
          report["thread_count"] == manifest["threads"] and len(report["samples"]) == samples,
          "Encoder provenance mismatch")
    size = output.stat().st_size
    check(all(s["encoded_bytes"] == size and s["elapsed_nanoseconds"] > 0 for s in report["samples"]),
          "Invalid measured encode")
    return {"variant": variant, "artifact": str(output), "sha256": sha(output), "bytes": size,
            "samples_ns": [s["elapsed_nanoseconds"] for s in report["samples"]]}


def audit_complete(record):
    for row in record["rows"]:
        check(sha(row["artifact"]) == row["sha256"], "Saved codestream changed")
    check(record["decoded"]["decoded_equal"] and record["decoded"]["decoded_finite"], "Invalid decode evidence")


def new_attempt(folder):
    folder.mkdir(parents=True, exist_ok=True)
    attempt = folder / f"attempt-{len(list(folder.glob('attempt-*'))):04d}"
    attempt.mkdir()
    return attempt


def collect(args, manifest):
    done = 0
    for index, case in enumerate(manifest["cases"]):
        folder = args.output / "cases" / case["id"]
        complete = folder / "complete.json"
        if complete.exists():
            audit_complete(load(complete))
            continue
        if args.max_cases is not None and done >= args.max_cases:
            break
        check(sha(case["image"]["path"]) == case["image"]["sha256"], "Input changed")
        state = machine_state()
        attempt = new_attempt(folder)
        variants = ("baseline", "gradient", "weighted") if index % 2 == 0 else ("weighted", "gradient", "baseline")
        rows = {v: encode(manifest, case, v, attempt, v, 0, 1) for v in variants}
        check(rows["baseline"]["sha256"] == rows["gradient"]["sha256"], "Default gradient bytes regressed")
        command([manifest["decoder"]["binary"], rows["gradient"]["artifact"], rows["weighted"]["artifact"],
                 case["image"]["width"], case["image"]["height"], 0, 0], attempt, "decode")
        decoded = load(attempt / "decode.stdout")
        check(decoded["gradient_sha256"] == rows["gradient"]["sha256"] and
              decoded["weighted_sha256"] == rows["weighted"]["sha256"], "Decoded wrong codestream")
        result = {"case": case, "rows": [rows[v] for v in ("baseline", "gradient", "weighted")],
                  "decoded": decoded, "machine": state}
        audit_complete(result)
        save(complete, result)
        done += 1
        print(f"{index+1}/{len(manifest['cases'])} {case['id']} saved "
              f"{100 * (1 - rows['weighted']['bytes'] / rows['gradient']['bytes']):+.3f}%", flush=True)


def timing(args, manifest):
    done = 0
    for case in manifest["cases"]:
        if case["id"] not in manifest["timing_cases"]:
            continue
        reference = load(args.output / "cases" / case["id"] / "complete.json")
        audit_complete(reference)
        check(sha(case["image"]["path"]) == case["image"]["sha256"], "Input changed")
        refs = {r["variant"]: r for r in reference["rows"]}
        folder = args.output / "timing" / case["id"]
        if (folder / "complete.json").exists():
            saved = load(folder / "complete.json")
            for row in saved["artifacts"]:
                check(sha(row["artifact"]) == row["sha256"] == refs[row["variant"]]["sha256"],
                      "Saved timing artifact changed")
            continue
        if args.max_cases is not None and done >= args.max_cases:
            break
        state = machine_state()
        attempt = new_attempt(folder)
        command([manifest["paired_encoder"], case["image"]["path"], attempt / "artifacts",
                 case["distance"], case["effort"], manifest["threads"], manifest["timing_warmups"],
                 manifest["timing_pairs"]], attempt, "encode-timing")
        encoded = load(attempt / "encode-timing.stdout")
        check(encoded["backend"] == "metal" and encoded["metal_aq_mode"] == "fully-resident" and
              encoded["pairing"] == "alternating-calls-in-one-process" and
              encoded["stage_profile_enabled"] is False and encoded["collect_final_score"] is False and
              encoded["effort"] == case["effort"] and encoded["thread_count"] == manifest["threads"] and
              math.isclose(encoded["requested_distance"], case["distance"], rel_tol=1e-6) and
              encoded["input_width"] == case["image"]["width"] and encoded["input_height"] == case["image"]["height"],
              "Paired encoder provenance mismatch")
        artifacts = []
        for variant in ("gradient", "weighted"):
            path = attempt / "artifacts" / f"{variant}.jxl"
            check(sha(path) == refs[variant]["sha256"], "Paired encoder output changed")
            artifacts.append({"variant": variant, "artifact": str(path), "sha256": sha(path)})
        command([manifest["decoder"]["binary"], refs["gradient"]["artifact"], refs["weighted"]["artifact"],
                 case["image"]["width"], case["image"]["height"], manifest["timing_warmups"],
                 manifest["timing_pairs"]], attempt, "decode-timing")
        decoded = load(attempt / "decode-timing.stdout")
        check(decoded["decoded_sha256"] == reference["decoded"]["decoded_sha256"], "Timed decoded pixels changed")
        for report in (encoded, decoded):
            check(len(report["samples"]) == 2 * manifest["timing_pairs"], "Missing timing samples")
            for index, row in enumerate(report["samples"]):
                pair, position = divmod(index, 2)
                variant = ("gradient", "weighted")[(pair + position) % 2]
                check(row["pair"] == pair and row["position"] == position and row["variant"] == variant and
                      row["elapsed_nanoseconds"] > 0, "Invalid timing order or value")
                if report is encoded:
                    check(row["encoded_bytes"] == refs[variant]["bytes"], "Timed size changed")
        save(folder / "complete.json", {"case": case, "encode": encoded, "decode": decoded,
                                         "machine": state, "artifacts": artifacts})
        done += 1
        print(f"Timing complete: {case['id']}", flush=True)


def report(args, manifest):
    rows = []
    timing_rows = []
    missing = []
    for case in manifest["cases"]:
        path = args.output / "cases" / case["id"] / "complete.json"
        if not path.exists():
            missing.append(case["id"])
            continue
        record = load(path)
        audit_complete(record)
        _, gradient, weighted = record["rows"]
        rows.append({"image": case["image"]["name"], "stratum": case["image"]["stratum"],
                     "effort": case["effort"], "distance": case["distance"],
                     "gradient_bytes": gradient["bytes"], "weighted_bytes": weighted["bytes"],
                     "saving_percent": 100 * (1 - weighted["bytes"] / gradient["bytes"])})
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stratum"], row["effort"])].append(row["saving_percent"])
    size_summary = [{"stratum": key[0], "effort": key[1], "points": len(values),
                     "wins": sum(v > 0 for v in values), "losses": sum(v < 0 for v in values),
                     "median_saving_percent": statistics.median(values), "minimum_saving_percent": min(values),
                     "maximum_saving_percent": max(values)} for key, values in sorted(groups.items())]
    for case_id in manifest["timing_cases"]:
        path = args.output / "timing" / case_id / "complete.json"
        if not path.exists():
            continue
        data = load(path)
        for row in data.get("artifacts", []):
            check(sha(row["artifact"]) == row["sha256"], "Timed codestream changed")
        for boundary in ("encode", "decode"):
            pairs = defaultdict(dict)
            if boundary == "encode" and "encode_pairs" in data:
                for pair in data["encode_pairs"]:
                    for row in pair["rows"]:
                        check(sha(row["artifact"]) == row["sha256"], "Timed codestream changed")
                        pairs[pair["pair"]][row["variant"]] = statistics.median(row["samples_ns"])
            else:
                for row in data[boundary]["samples"]:
                    pairs[row["pair"]][row["variant"]] = row["elapsed_nanoseconds"]
            deltas = [(p["weighted"] - p["gradient"]) / 1e6 for p in pairs.values()]
            ratios = [100 * (p["weighted"] / p["gradient"] - 1) for p in pairs.values()]
            timing_rows.append({"case": case_id, "boundary": boundary, "pairs": len(pairs),
                "gradient_ms": statistics.median(p["gradient"] for p in pairs.values()) / 1e6,
                "weighted_ms": statistics.median(p["weighted"] for p in pairs.values()) / 1e6,
                "paired_delta_ms": statistics.median(deltas), "paired_delta_percent": statistics.median(ratios),
                "slower_pairs": sum(d > 0 for d in deltas)})
    for name, values in (("size", rows), ("size-summary", size_summary), ("timing", timing_rows)):
        if values:
            with (args.output / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    result = {"expected_cases": len(manifest["cases"]), "completed_cases": len(rows), "missing": missing,
              "expected_timing_cases": len(manifest["timing_cases"]), "completed_timing_cases": len(timing_rows) // 2,
              "size_summary": size_summary, "timing_summary": timing_rows}
    save(args.output / "summary.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "collect", "timing", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--candidate", type=Path, default=ROOT / "build/release/gjxl_quality_benchmark")
    parser.add_argument("--baseline", type=Path, default=ROOT / "build/baseline/gjxl_quality_benchmark")
    parser.add_argument("--paired-encoder", type=Path, default=ROOT / "build/release/gjxl_dc_prediction_benchmark")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--decoder-build", type=Path, default=ROOT / "build/dc-tools/build.json")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command == "init":
        check(args.metadata and args.historical and args.baseline_source, "init requires metadata, historical, baseline-source")
        initialize(args)
    else:
        manifest = load(args.output / "manifest.json")
        if args.command != "report":
            verify_identity(manifest)
        {"collect": collect, "timing": timing, "report": report}[args.command](args, manifest)


if __name__ == "__main__":
    main()
