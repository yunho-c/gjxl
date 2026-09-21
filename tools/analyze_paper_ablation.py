#!/usr/bin/env python3
"""Audit coverage and report paired ablation timings from retained evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
from ablation import run as collector


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--verify-artifacts", action="store_true")
    args = parser.parse_args()
    root = args.run.resolve()
    output = (args.output or root / "analysis").resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    fingerprint = manifest["fingerprint"]
    protocol = fingerprint["protocol"]
    if digest(collector.__file__) != fingerprint["sources"]["tools/ablation/run.py"]:
        raise ValueError("Restore the frozen collector source before auditing its path contracts")
    if args.verify_artifacts and digest(root / "encoder") != fingerprint["binary_sha256"]:
        raise ValueError("Frozen encoder changed")
    corpus = json.loads(args.corpus.read_text())
    metadata = {str(Path(row["pfm_path"]).resolve()): row for row in corpus["images"]}
    inputs = []
    for item in fingerprint["inputs"]:
        row = metadata[item["path"]]
        if row["pfm_sha256"] != item["sha256"]:
            raise ValueError("Corpus provenance mismatch")
        inputs.append(row)
    expected = {}
    for rnd in range(protocol["rounds"]):
        for index in range(len(inputs)):
            for effort in protocol["efforts"]:
                for distance in protocol["distances"]:
                    case = f"r{rnd}-i{index}-e{effort}-d{distance}"
                    for variant in fingerprint["variants"]:
                        expected[f"{case}-{variant}"] = (rnd, index, effort, distance)
    records, hashes, problems = {}, {}, []
    verified_inodes = {}
    def verify(path, expected_hash):
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        actual = verified_inodes.get(key)
        if actual is None:
            actual = digest(path)
            verified_inodes[key] = actual
        if actual != expected_hash:
            raise ValueError(f"Artifact changed: {path}")
    for job, context in expected.items():
        directory = root / job
        path = directory / "record.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record["id"] != job:
            raise ValueError("Record identity mismatch")
        raw = json.loads((directory / "raw.json").read_text())
        if (raw["revision"] != fingerprint["revision"] or raw["stage_profile_enabled"] or
            raw["timing_semantics"] != "complete-encode-wall-time" or
            raw["thread_count"] != protocol["threads"] or raw["warmups"] != protocol["warmups"] or
            raw["sample_count"] != protocol["samples"] or raw["collect_final_score"] or
            raw["validation_encodes"] != 2 or
            raw["ablation"]["audit_scope"] != "untimed-warm-validation-call" or
            raw["backend"] != "metal" or raw["metal_aq_mode"] != "fully-resident" or
            raw["ablation"]["variant"] != record["variant"] or
            raw["ablation"]["counters"] != record["counters"] or
            raw["effort"] != context[2] or raw["requested_distance"] != context[3]):
            raise ValueError(f"Timing/configuration contract mismatch: {job}")
        samples = [sample["elapsed_nanoseconds"] for sample in raw["samples"]]
        if samples != record["times_ns"] or len(samples) != protocol["samples"] or min(samples) <= 0:
            raise ValueError(f"Invalid timing samples: {job}")
        collector.validate_audit(raw, record["variant"], context[2])
        image = inputs[context[1]]
        if raw["input_width"] != image["width"] or raw["input_height"] != image["height"]:
            raise ValueError(f"Image extent mismatch: {job}")
        if any(sample["encoded_bytes"] != record["bytes"] for sample in raw["samples"]):
            raise ValueError(f"Encoded size mismatch: {job}")
        if not math.isfinite(record["quality"]):
            raise ValueError(f"Invalid quality score: {job}")
        if "validation_cache" in record:
            cached = json.loads((root / record["validation_cache"]).read_text())
            expected_identity = {"input_sha256": fingerprint["inputs"][context[1]]["sha256"],
                                 "codestream_sha256": record["artifacts"]["output.jxl"],
                                 "tools": {"decoder": fingerprint["decoder"]["sha256"],
                                           "metric": fingerprint["metric"]["sha256"]}}
            if (cached["identity"] != expected_identity or cached["quality"] != record["quality"] or
                cached["decoded_sha256"] != record["decoded_sha256"] or
                cached["input_pixel_error"] != record["input_pixel_error"]):
                raise ValueError(f"Decoded validation identity mismatch: {job}")
        if "AC Power" not in record["power_before"] or "AC Power" not in record["power_after"]:
            problems.append({"job": job, "problem": "power-boundary"})
        if args.verify_artifacts:
            for name, value in record["artifacts"].items():
                verify(directory / name, value)
            if "validation_cache" in record:
                verify(root / record["validation_cache"], record["validation_sha256"])
        identity = (context[1], context[2], context[3], record["variant"])
        hashes.setdefault(identity, set()).add(record["artifacts"]["output.jxl"])
        records[job] = record | {"context": context}
    unstable = [str(key) for key, values in hashes.items() if len(values) != 1]
    complete = len(records) == len(expected)
    if not complete and not args.partial:
        raise ValueError(f"Collection incomplete: {len(records)}/{len(expected)} jobs")
    summary = json.loads((root / "summary.json").read_text()) if (root / "summary.json").exists() else None
    if complete and not args.partial:
        if not summary or summary["jobs"] != len(expected):
            raise ValueError("Final collection summary missing or incomplete")
        expected_pairs = len(expected) // len(fingerprint["variants"]) * len(fingerprint["pairs"])
        if len(summary["comparisons"]) != expected_pairs:
            raise ValueError("Final pair coverage mismatch")
    paired = []
    for job, a in records.items():
        for factor, baseline, disabled, contract in fingerprint["pairs"]:
            if a["variant"] != baseline:
                continue
            b = records.get(f"{a['case']}-{disabled}")
            if b is None:
                continue
            rnd, index, effort, distance = a["context"]
            ca, cb = a["counters"], b["counters"]
            if factor == "aq-boundaries":
                for key in ("submissions", "completion_waits"):
                    if cb[key] - ca[key] != ca["aq_evaluations"] + 1:
                        raise ValueError(f"AQ synchronization control mismatch: {a['case']}")
                if {k: v for k, v in ca.items() if k.startswith("gjxl_")} != {
                        k: v for k, v in cb.items() if k.startswith("gjxl_")}:
                    raise ValueError("AQ synchronization changed kernel invocations")
            same = a["artifacts"]["output.jxl"] == b["artifacts"]["output.jxl"]
            row = inputs[index]
            scene = row["image_id"].rsplit("/", 1)[0] if row["corpus"].startswith("unsplash") else row["image_id"]
            baseline_ns, disabled_ns = statistics.median(a["times_ns"]), statistics.median(b["times_ns"])
            paired.append({"factor": factor, "effort": effort, "distance": distance,
                "round": rnd, "image_index": index, "image_id": row["image_id"],
                "scene": scene, "cohort": row["resolution_class"], "pixels": row["pixels"],
                "baseline_ms": baseline_ns / 1e6, "disabled_ms": disabled_ns / 1e6,
                "ratio": disabled_ns / baseline_ns, "byte_identical": same,
                "decoded_identical": a.get("decoded_sha256") == b.get("decoded_sha256") if "decoded_sha256" in a else same,
                "byte_ratio": b["bytes"] / a["bytes"], "ssimulacra2_delta": b["quality"] - a["quality"],
                "contract": contract})
    if complete and not args.partial:
        reported = {(row["case"], row["factor"]): row for row in summary["comparisons"]}
        if len(reported) != len(paired):
            raise ValueError("Duplicate or missing final comparisons")
        for row in paired:
            case = f"r{row['round']}-i{row['image_index']}-e{row['effort']}-d{row['distance']}"
            value = reported[(case, row["factor"])]
            if (value["byte_identical"] != row["byte_identical"] or
                not math.isclose(value["time_ratio"], row["ratio"], rel_tol=1e-12) or
                not math.isclose(value["byte_ratio"], row["byte_ratio"], rel_tol=1e-12) or
                not math.isclose(value["quality_delta_ssimulacra2"], row["ssimulacra2_delta"], abs_tol=1e-12)):
                raise ValueError(f"Final comparison disagrees with raw evidence: {case}")
    output.mkdir(parents=True, exist_ok=True)
    def write_csv(name, rows):
        if not rows:
            return
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_csv("paired.csv", paired)
    aggregates = []
    conditions = sorted({(p["factor"], p["effort"], p["distance"]) for p in paired})
    rng = np.random.default_rng(20260921)
    for factor, effort, distance in conditions:
        selected = [p for p in paired if (p["factor"], p["effort"], p["distance"]) == (factor, effort, distance)]
        for cohort in ["all"] + sorted({p["cohort"] for p in selected}):
            subset = selected if cohort == "all" else [p for p in selected if p["cohort"] == cohort]
            images = {}
            for p in subset:
                images.setdefault(p["image_id"], []).append(p)
            # Equal image weight; geometric mean of paired process-round ratios.
            logs = {name: statistics.mean(math.log(p["ratio"]) for p in rows) for name, rows in images.items()}
            clusters = {}
            for name, rows in images.items():
                clusters.setdefault(rows[0]["scene"], []).append(logs[name])
            sums = np.array([sum(v) for v in clusters.values()])
            counts = np.array([len(v) for v in clusters.values()])
            picks = rng.integers(0, len(clusters), size=(2000, len(clusters)))
            resampled = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
            lo, hi = np.exp(np.quantile(resampled, [.025, .975]))
            ratio = math.exp(statistics.mean(logs.values()))
            round_ratios = [math.exp(statistics.mean(math.log(p["ratio"]) for p in subset if p["round"] == rnd))
                            for rnd in sorted({p["round"] for p in subset})]
            aggregates.append({"factor": factor, "effort": effort, "distance": distance,
                "cohort": cohort, "images": len(images), "scenes": len(clusters), "process_pairs": len(subset),
                "disabled_over_baseline": ratio, "ci95_low": float(lo), "ci95_high": float(hi),
                "optimized_time_reduction_percent": 100 * (1 - 1 / ratio),
                "round_ratio_min": min(round_ratios), "round_ratio_max": max(round_ratios),
                "nonidentical_pairs": sum(not p["byte_identical"] for p in subset),
                "max_abs_ssimulacra2_delta": max(abs(p["ssimulacra2_delta"]) for p in subset),
                "max_abs_byte_ratio_delta": max(abs(p["byte_ratio"] - 1) for p in subset)})
    write_csv("aggregate.csv", aggregates)
    audit = {"complete": complete, "expected_jobs": len(expected), "completed_jobs": len(records),
             "timed_samples": sum(len(r["times_ns"]) for r in records.values()),
             "paired_observations": len(paired), "verified_artifacts": args.verify_artifacts,
             "verified_unique_files": len(verified_inodes), "power_problems": problems,
             "unstable_process_outputs": unstable,
             "nonidentical_exact_pairs": sum(not p["byte_identical"] and p["contract"] == "exact" for p in paired),
             "nonidentical_dct_pairs": sum(not p["byte_identical"] and p["contract"] == "numerical" for p in paired),
             "source_manifest_sha256": digest(root / "manifest.json"),
             "corpus_manifest_sha256": digest(args.corpus), "analysis_script_sha256": digest(__file__),
             "generated_at": time.time(),
             "method": "Job medians; geometric mean across process rounds within each image; equal-image geometric mean. 2000 source-scene cluster bootstrap draws; 95% percentile interval. Round spread reported separately."}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = ["# GJXL paper ablation measurements", "",
             f"{'Complete' if complete else 'PARTIAL'}: {len(records)}/{len(expected)} jobs; {audit['timed_samples']} timed samples.",
             "", f"Encoder revision: `{fingerprint['revision']}`. Hardware: {fingerprint['hardware']}.",
             "", audit["method"], "",
             "Ratios are disabled / optimized: values above 1 favor the optimization. Comparisons use the conditional baselines in the protocol; gains must not be added together.",
             "", "| Factor | Effort | Distance | Images | Ratio | 95% cluster interval | Nonidentical pairs |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for a in aggregates:
        if a["cohort"] == "all":
            lines.append(f"| {a['factor']} | {a['effort']} | {a['distance']} | {a['images']} | {a['disabled_over_baseline']:.4f} | {a['ci95_low']:.4f}–{a['ci95_high']:.4f} | {a['nonidentical_pairs']} |")
    lines += ["", "See aggregate.csv for size-cohort results and paired.csv for every process-round comparison.",
              "", "Any nonidentical outputs require rate/quality interpretation; timings alone do not establish an equal-output gain. The 12/24/48 MP cohorts each contain only three scenes, so their uncertainty estimates have limited generality.",
              "", "Validation audit:", "", "```json", json.dumps(audit, indent=2), "```"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
