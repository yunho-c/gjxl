#!/usr/bin/env python3
"""Review timing variability without removing or replacing original measurements."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_run(root):
    fingerprint = json.loads((root / "manifest.json").read_text())["fingerprint"]
    audit = json.loads((root / "analysis/audit.json").read_text())
    if not audit["complete"] or not audit["verified_artifacts"]:
        raise ValueError("Timing review requires complete, artifact-verified runs")
    if audit["power_problems"] or audit["unstable_process_outputs"]:
        raise ValueError("Unresolved power or output repeatability problem")
    protocol = fingerprint["protocol"]
    expected = {f"r{rnd}-i{i}-e{e}-d{d}-{v}"
                for rnd in range(protocol["rounds"])
                for i in range(len(fingerprint["inputs"]))
                for e in protocol["efforts"] for d in protocol["distances"]
                for v in fingerprint["variants"]}
    paths = list(root.glob("r*/record.json"))
    if {p.parent.name for p in paths} != expected:
        raise ValueError("Unexpected or missing retained job directories")
    records = {}
    for path in paths:
        row = json.loads(path.read_text())
        if row["id"] != path.parent.name or row["id"] != f"{row['case']}-{row['variant']}":
            raise ValueError("Record identity mismatch")
        records[row["id"]] = row
    with (root / "analysis/paired.csv").open() as stream:
        paired = list(csv.DictReader(stream))
    return fingerprint, records, paired


def estimate(rows, rng, median_rounds=False):
    images = defaultdict(list)
    scenes = {}
    for row in rows:
        images[row["image_id"]].append(math.log(float(row["ratio"])))
        scenes[row["image_id"]] = row["scene"]
    reduce = statistics.median if median_rounds else statistics.mean
    logs = {name: reduce(values) for name, values in images.items()}
    clusters = defaultdict(list)
    for name, value in logs.items():
        clusters[scenes[name]].append(value)
    sums = np.array([sum(values) for values in clusters.values()])
    counts = np.array([len(values) for values in clusters.values()])
    picks = rng.integers(0, len(clusters), size=(2000, len(clusters)))
    draws = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.exp(np.quantile(draws, [.025, .975]))
    return math.exp(statistics.mean(logs.values())), float(low), float(high)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    args = parser.parse_args()
    root, repeat = args.run.resolve(), args.repeat.resolve()
    original, records, paired = read_run(root)
    repeated, repeat_records, repeat_pairs = read_run(repeat)
    contract = ["binary_sha256", "decoder", "metric", "protocol", "revision", "sources",
                "execution_environment", "system", "hardware", "variants", "pairs"]
    if any(original[key] != repeated[key] for key in contract):
        raise ValueError("Diagnostic repeat changed the frozen experiment contract")
    if len(repeated["inputs"]) != 1:
        raise ValueError("This review expects a balanced repeat of one entire image")
    matching = [i for i, row in enumerate(original["inputs"])
                if row == repeated["inputs"][0]]
    if len(matching) != 1:
        raise ValueError("Repeat input does not uniquely match the original corpus")
    index = matching[0]
    groups = defaultdict(list)
    for row in records.values():
        groups[(row["case"].split("-", 1)[1], row["variant"])].append(row)
    anomalies = []
    for group in groups.values():
        center = statistics.median(statistics.median(row["times_ns"]) for row in group)
        for row in group:
            ratio = statistics.median(row["times_ns"]) / center
            if ratio > 2:
                anomalies.append({"job": row["id"], "relative_to_round_median": ratio,
                                  "record_completed_epoch": (root / row["id"] / "record.json").stat().st_mtime})
    repeat_groups = defaultdict(list)
    repeated_outputs_equal = True
    for row in repeat_records.values():
        suffix = row["case"].split("-i0-", 1)[1]
        reference = groups[(f"i{index}-{suffix}", row["variant"])]
        repeated_outputs_equal &= all(row["artifacts"]["output.jxl"] == x["artifacts"]["output.jxl"]
                                      for x in reference)
        repeat_groups[(suffix, row["variant"])].append(row)
    if not repeated_outputs_equal:
        raise ValueError("Diagnostic repeat changed encoded output")
    repeat_ratios, repeat_spikes = [], []
    for (suffix, variant), group in repeat_groups.items():
        center = statistics.median(statistics.median(row["times_ns"]) for row in group)
        old_center = statistics.median(statistics.median(row["times_ns"])
                                       for row in groups[(f"i{index}-{suffix}", variant)])
        repeat_ratios.append(center / old_center)
        repeat_spikes.extend(row["id"] for row in group
                             if statistics.median(row["times_ns"]) / center > 2)
    rng = np.random.default_rng(20260921)
    results = []
    with (root / "analysis/aggregate.csv").open() as stream:
        primary = {(x["factor"], int(x["effort"]), float(x["distance"])): x
                   for x in csv.DictReader(stream) if x["cohort"] == "all"}
    conditions = sorted({(x["factor"], int(x["effort"]), float(x["distance"])) for x in paired})
    for factor, effort, distance in conditions:
        def selected(rows):
            return [x for x in rows if (x["factor"], int(x["effort"]), float(x["distance"])) ==
                    (factor, effort, distance)]
        rows = selected(paired)
        diagnostic_rows = [x for x in rows if int(x["image_index"]) != index] + selected(repeat_pairs)
        if len(rows) != len(diagnostic_rows):
            raise ValueError("Unbalanced whole-image sensitivity comparison")
        inclusive, _, _ = estimate(rows, rng)
        if not math.isclose(inclusive, float(primary[(factor, effort, distance)]["disabled_over_baseline"]),
                            rel_tol=1e-12):
            raise ValueError("Sensitivity calculation disagrees with the original aggregate")
        median, mlo, mhi = estimate(rows, rng, median_rounds=True)
        replaced, rlo, rhi = estimate(diagnostic_rows, rng)
        results.append({"factor": factor, "effort": effort, "distance": distance,
                        "images": len({x["image_id"] for x in rows}),
                        "all_collected_ratio": inclusive,
                        "median_round_ratio": median, "median_ci95_low": mlo, "median_ci95_high": mhi,
                        "whole_image_repeat_ratio": replaced, "repeat_ci95_low": rlo, "repeat_ci95_high": rhi})
    output = root / "analysis"
    with (output / "sensitivity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    cases = defaultdict(dict)
    for row in records.values():
        cases[row["case"]][row["variant"]] = row["counters"]
    for case, arms in cases.items():
        production, handoff, split = arms["production"], arms["ac-handoff"], arms["split-epf"]
        if not (production.get("indirect_dispatches", 0) > 0 and
                handoff.get("indirect_dispatches", 0) == 0 and handoff.get("immediate_search") == 1 and
                handoff.get("gpu_selector") == 1 and
                handoff["readback_copy_bytes"] > production["readback_copy_bytes"] and
                split.get("gjxl_aq_opsin_to_linear_rgb_f32") == production["aq_evaluations"]):
            raise ValueError("Additional handoff/conversion manipulation check failed: " + case)
    controller = root.with_name(root.name + "-controller")
    telemetry = [json.loads(line) for line in (controller / "telemetry.jsonl").read_text().splitlines()]
    review = {"original_records_unchanged": True, "original_jobs": len(records),
              "original_manifest_sha256": sha(root / "manifest.json"),
              "repeat_run": str(repeat), "repeat_jobs": len(repeat_records),
              "repeat_manifest_sha256": sha(repeat / "manifest.json"),
              "repeat_image_index": index, "repeat_outputs_equal": repeated_outputs_equal,
              "repeat_three_round_median_vs_original_median_range": [min(repeat_ratios), max(repeat_ratios)],
              "repeat_jobs_above_twice_round_median": repeat_spikes,
              "original_jobs_above_twice_round_median": sorted(anomalies, key=lambda x: x["record_completed_epoch"]),
              "additional_manipulation_checks": len(cases),
              "telemetry_samples": len(telemetry), "telemetry_first_epoch": telemetry[0]["time"],
              "telemetry_last_epoch": telemetry[-1]["time"],
              "telemetry_non_ac_samples": sum("AC Power" not in x["power"] for x in telemetry),
              "telemetry_thermal_messages": sorted({x["thermal"] for x in telemetry}),
              "script_sha256": sha(__file__), "sensitivity_sha256": sha(output / "sensitivity.csv"),
              "interpretation": "Post hoc diagnostics only. Original all-collected estimates and records are retained. Median-round sensitivity uses all three paired ratios per image. Whole-image repeat sensitivity uses the complete balanced repeat for the one image, not selected arms or samples. Neither changes the original primary aggregate. The cause of the transient timing disturbance is not established."}
    (output / "timing-review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
