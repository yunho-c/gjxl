#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Verify a completed fitting-attribution ledger and write additive reports."""
import argparse
import json
import statistics
from pathlib import Path

from run_histogram_attribution import signature, validate_profile
from run_hybrid_qualification import digest, require


def summarize(rows):
    images = []
    for name in sorted({r["input"] for r in rows}):
        measured = [r for r in rows if r["input"] == name and not r["warmup"]]
        off = [r for r in measured if r["mode"] == "fit-off"]
        on = [r for r in measured if r["mode"] == "fit-on"]
        require(off and len(off) == len(on), "Incomplete paired measurements")
        median = lambda r, key: statistics.median(x["result"][key[0]][key[1]] for x in r)
        off_fit, on_fit = median(off, ("detail_wall", "fitting")), median(on, ("detail_wall", "fitting"))
        chosen = sorted(on, key=lambda r: r["result"]["histogram_fitting"]["fit_seconds"])[len(on) // 2]
        profile = chosen["result"]["histogram_fitting"]
        require(not profile.get("parallel_property_nodes", 0),
                "Exclusive phase attribution requires serial property evaluation")
        stages = dict(profile["stages"])
        stages["control_cleanup_and_probe_overhead"] = profile["fit_seconds"] - sum(stages.values())
        work = {key: sum(b["work"][key] for b in profile["by_rows"])
                for key in profile["by_rows"][0]["work"]}
        fit_times = [f["seconds"] for f in profile["fits"]]
        images.append({
            "input": name, "repeats_per_mode": len(on), "bytes": chosen["result"]["bytes"],
            "power_source": cohort_power([r for r in rows if r["input"] == name]),
            "median_total_off": median(off, ("total", "wall")),
            "median_total_on": median(on, ("total", "wall")),
            "median_workflow_fitting_off": off_fit, "median_workflow_fitting_on": on_fit,
            "fitting_overhead_percent": 100 * (on_fit / off_fit - 1),
            "paired_fitting_ratios": [a["result"]["detail_wall"]["fitting"] /
                                      b["result"]["detail_wall"]["fitting"]
                                      for a, b in zip(sorted(on, key=lambda r: r["iteration"]),
                                                      sorted(off, key=lambda r: r["iteration"]))],
            "fitting_off_range": [min(r["result"]["detail_wall"]["fitting"] for r in off),
                                  max(r["result"]["detail_wall"]["fitting"] for r in off)],
            "fitting_on_range": [min(r["result"]["detail_wall"]["fitting"] for r in on),
                                 max(r["result"]["detail_wall"]["fitting"] for r in on)],
            "selected_iteration": chosen["iteration"], "selected_fitter_seconds": profile["fit_seconds"],
            "selected_total_seconds": chosen["result"]["total"]["wall"], "stages": stages,
            "work": work, "mean_active_bins": work["active_bins"] / work["properties"],
            "mean_bin_span": work["bin_span"] / work["properties"],
            "maximum_depth": max(b["depth"] for b in profile["by_depth"]),
            "fit_median_seconds": statistics.median(fit_times), "fit_maximum_seconds": max(fit_times),
            "profile": profile,
        })
    return images


def markdown(report):
    images = report["images"]
    labels = [x["input"].removeprefix("photo-large-") for x in images]
    lines = ["# Native histogram fitting attribution", "", report["scope"], "",
             "## Instrumentation overhead", "",
             "Same frozen binary and hybrid policy, with fitting attribution disabled/enabled. "
             "Values are medians of complete calls; repetitions and ranges remain in the JSON.", "",
             "| Image | Power | Encoder off / on (s) | Workflow fitting off / on (s) | Fitting change |",
             "| --- | --- | ---: | ---: | ---: |"]
    for x, label in zip(images, labels):
        lines.append(f"| {label} | {x['power_source']} | {x['median_total_off']:.3f} / {x['median_total_on']:.3f} | "
                     f"{x['median_workflow_fitting_off']:.3f} / {x['median_workflow_fitting_on']:.3f} | "
                     f"{x['fitting_overhead_percent']:+.2f}% |")
    lines += ["", "## Exclusive fitting phases", "",
              "Each column uses the individual profiled call with median fitter time. The rows add "
              "to its fitter envelope; they are not independently computed stage medians. These "
              "are new instrumented measurements, not an exact subdivision of the earlier 52/38/35 s calls.", "",
              "| Phase | " + " | ".join(labels) + " |",
              "| --- | " + " | ".join("---:" for _ in images) + " |"]
    names = {"validation": "Input validation", "setup": "Workspace/result allocation and setup",
             "symbol_range": "Find residual symbol range", "parent_histograms": "Parent histograms + base cost",
             "property_population": "Property occupancy + cost-array initialization",
             "joint_histograms": "Joint property/residual histograms",
             "candidate_evaluation": "Prefix updates + entropy/predictor evaluation",
             "split_selection": "Choose threshold within each property", "partition": "Partition sample rows",
             "output": "Validate/publish tree", "control_cleanup_and_probe_overhead": "Other control, cleanup, probe overhead"}
    if any(x["profile"]["schema"] >= 2 for x in images):
        names["property_population"] = "Bin endpoints + occupancy/cost initialization"
        names["joint_histograms"] = "Joint histograms + fused property occupancy"
    for key, title in names.items():
        lines.append("| " + title + " | " + " | ".join(
            f"{x['stages'][key]:.3f} s ({100*x['stages'][key]/x['selected_fitter_seconds']:.1f}%)" for x in images) + " |")
    lines.append("| **Fitter envelope** | " + " | ".join(f"**{x['selected_fitter_seconds']:.3f} s**" for x in images) + " |")
    if any(x["profile"]["schema"] >= 2 for x in images):
        lines += ["", "Schema 2 includes first-predictor occupancy in the joint-histogram phase. "
                  "Physical property/histogram row passes are property_row_visits + "
                  "joint_histogram_row_visits - fused_property_row_visits; these counters "
                  "must not be added without subtracting the shared visits."]
    lines += ["", "## Node-size distribution", "",
              "Rows are distinct weighted training rows, not image pixels. Time is inclusive node "
              "processing time as a share of each image's fitter envelope. Node counts combine the selected images. "
              "Each node is assigned once; ancestor and descendant row visits repeat work on the same samples.", "",
              "| Rows per node | Nodes | " + " | ".join(labels) + " |",
              "| --- | ---: | " + " | ".join("---:" for _ in images) + " |"]
    bounds = [(0, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 16384), (16384, 65536), (65536, 1 << 24)]
    for low, high in bounds:
        buckets = [b for x in images for b in x["profile"]["by_rows"] if low < b["maximum_rows"] <= high]
        count = sum(b["work"]["nodes"] for b in buckets)
        label = f"{low+1:,}–{high:,}" if high < 1 << 24 else f">{low:,}"
        shares = [100 * sum(b["node_seconds"] for b in x["profile"]["by_rows"]
                            if low < b["maximum_rows"] <= high) / x["selected_fitter_seconds"]
                  for x in images]
        lines.append(f"| {label} | {count:,} | " + " | ".join(f"{v:.2f}%" for v in shares) + " |")
    lines += ["", "## Workload", "",
              "| Image | Fits | Visited nodes | Joint-histogram row visits | Entropy evaluations | Max depth | Mean occupied bins |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for x, label in zip(images, labels):
        w = x["work"]
        lines.append(f"| {label} | {x['profile']['calls']} | {w['nodes']:,} | {w['joint_histogram_row_visits']:,} | "
                     f"{w['entropy_evaluations']:,} | {x['maximum_depth']} | {x['mean_active_bins']:.2f} |")
    lines += ["", "The JSON also retains exact depth counts/times, padded residual-symbol spans, "
              "occupied-bin/span distributions, row swaps, and individual fit times/shapes. Symbol "
              "span is the padded maximum token plus one used by the cost loop, not nonzero-symbol count.", "",
              "## Measurement boundaries and reproduction", "",
              "Instrumentation is opt-in and allocates no fitting storage. Disabled hot loops are a "
              "separate compile-time instantiation. Timers surround node/property/predictor blocks, "
              "never individual thresholds. All phase times are wall time; the fit stays serial. "
              "Candidate evaluation deliberately combines prefix updates/clears, entropy estimation, "
              "and per-side predictor comparison. It does not isolate the logarithm polynomial. "
              "The remainder includes final category preference, tree bookkeeping, destruction, and "
              "unassigned probe overhead. The workflow's fitting timer additionally includes storage "
              "planning and prepared-column release/compaction outside LearnHistogramTree.", "",
              f"All **{report['completed_calls']} encodes** decoded exactly and preserved complete output "
              "bytes, routing, and learning work across on/off modes and repetitions. Enabled runs "
              "also preserved all workload counters. "
              f"Timing uses {report['timing_calls']} calls including warmups; "
              f"{report['superseded_calls']} superseded calls remain in the JSON and raw archives. "
              "Each retained image cohort has a single observed power source at every call boundary. "
              "Reference-byte verification: " +
              ("matched the earlier qualified hybrid files." if report["prior_reference_checked"] else "not requested."), "",
              "Build and input identities, every run, before/after power/load observations, and "
              "dependencies are retained. Source was archived and hashed including uncommitted changes; "
              "the recorded base revision alone does not identify the instrumented source. "
              "Desktop activity was not isolated. This attribution is not a CPU/GPU speedup claim.", "",
              f"Artifacts: `{report['artifact_directory']}`. Manifest SHA-256: `{report['manifest_sha256']}`.", "",
              "```sh", "python3 tools/modular_reference_learning/analyze_histogram_attribution.py \\",
              f"  {report['artifact_directory']} \\",
              "  --json docs/modular-histogram-fitting-profile-results.json \\",
              "  --markdown docs/modular-histogram-fitting-profile.md", "```", ""]
    return "\n".join(lines)


def cohort_power(rows):
    sources = set()
    for row in rows:
        for when in ["environment_before", "environment_after"]:
            text = row[when].get("power", "unrecorded")
            sources.add(text.split("'")[1] if text.startswith("Now drawing from '") else "unrecorded")
    require(len(sources) == 1, "Power changed within an image cohort; supply a stable replacement capture")
    return next(iter(sources))


def read_capture(root):
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["status"] == "passed", "Capture is incomplete or failed")
    for name, checksum in manifest["artifacts"].items():
        require(digest(root / name) == checksum, "Artifact changed: " + name)
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    expected = {(p["name"], mode, iteration) for p in manifest["inputs"]
                for mode in ["fit-off", "fit-on"]
                for iteration in range(manifest["warmups"] + manifest["repeats"])}
    observed = {(r["input"], r["mode"], r["iteration"]) for r in rows}
    require(observed == expected and len(rows) == manifest["completed_calls"] == len(expected),
            "Incomplete run ledger")
    require(all(r["warmup"] == (r["iteration"] < manifest["warmups"]) for r in rows),
            "Wrong warmup classification")
    return manifest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--replacement", type=Path, action="append", default=[],
                        help="Replace whole input cohorts with this completed capture; keep every raw run")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    roots = [args.capture.resolve()] + [p.resolve() for p in args.replacement]
    captures, cohorts, all_rows, signatures, controls, inputs = [], {}, [], {}, {}, {}
    for index, root in enumerate(roots):
        manifest, rows = read_capture(root)
        if captures:
            require(manifest["binary_sha256"] == captures[0]["manifest"]["binary_sha256"],
                    "Replacement binary differs")
        for item in manifest["inputs"]:
            require(inputs.setdefault(item["name"], item["sha256"]) == item["sha256"],
                    "Replacement input differs")
        for row in rows:
            result = row["result"]
            require(result["exact_decode"], "An encode did not decode exactly")
            routing = {k: v for k, v in result["routing"].items()
                       if k != "cpu_feature_peak_participants"}
            control = (row["output_sha256"], result["bytes"], result["work"], routing)
            require(controls.setdefault(row["input"], control) == control,
                    "Outputs or work differ across captures")
            if row["mode"] == "fit-on":
                validate_profile(result)
                sig = signature(result["histogram_fitting"])
                require(signatures.setdefault(row["input"], sig) == sig, "Work counters changed")
            row["capture_index"] = index
        for name in {r["input"] for r in rows}:
            cohorts[name] = [r for r in rows if r["input"] == name]
        all_rows.extend(rows)
        captures.append({"root": root, "manifest": manifest})
    rows = [r for name in sorted(cohorts) for r in cohorts[name]]
    chosen = {id(r) for r in rows}
    for row in all_rows:
        row["used_for_timing"] = id(row) in chosen
    # Do not report medians that combine AC and battery observations. A whole
    # replacement cohort avoids selective removal of inconvenient individual calls.
    images = summarize(rows)
    root, manifest = roots[0], captures[0]["manifest"]
    report = {"schema": 2, "scope": manifest["scope"], "artifact_directory": str(root),
              "manifest_sha256": digest(root / "manifest.json"), "completed_calls": len(all_rows),
              "timing_calls": len(rows), "superseded_calls": len(all_rows) - len(rows),
              "prior_reference_checked": all((p / "reference-results.json").exists() for p in roots),
              "base_revision": manifest["source_revision"], "binary_sha256": manifest["binary_sha256"],
              "source_archive_sha256": manifest["artifacts"]["source.tar.gz"],
              "environment_start": manifest["environment_start"], "environment_finish": manifest["environment_finish"],
              "captures": [{"artifact_directory": str(c["root"]),
                            "manifest_sha256": digest(c["root"] / "manifest.json"),
                            "source_archive_sha256": c["manifest"]["artifacts"]["source.tar.gz"],
                            "inputs": c["manifest"]["inputs"],
                            "completed_calls": c["manifest"]["completed_calls"],
                            "used_inputs": sorted({r["input"] for r in rows if r["capture_index"] == i})}
                           for i, c in enumerate(captures)],
              "images": images, "runs": all_rows}
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    md = markdown(report)
    if args.replacement:
        additions = "".join("  --replacement " + str(p.resolve()) + " \\\n" for p in args.replacement)
        marker = "  --json docs/modular-histogram-fitting-profile-results.json"
        md = md.replace(marker, additions + marker)
    args.markdown.write_text(md)
    print(f"Verified {len(all_rows)} calls; timing uses {len(rows)} calls; reports written")


if __name__ == "__main__":
    main()
