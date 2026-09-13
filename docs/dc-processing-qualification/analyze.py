"""Saved-evidence DC qualification tables; never runs an encoder or a scorer."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

import numpy as np


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    root, output = args.study.resolve(), args.output.resolve()
    manifest = load(root / "manifest.json")
    summary = load(root / "summary.json")
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(lambda: defaultdict(list))
    artifacts, regressions = {}, Counter()
    for case in manifest["cases"]:
        record = load(root / "curves" / case["id"] / "complete.json")
        require(record["equality"]["decoded_equal"] and record["equality"]["decoded_finite"],
                "Residual predictor equality check missing")
        for name, row in record["rows"].items():
            require(sha(row["artifact"]) == row["sha256"] and math.isfinite(row["quality"])
                    and row["decoded"]["finite"], "Invalid retained curve artifact")
            artifacts[row["artifact"]] = row["sha256"]
            groups[(case["image"]["name"], case["image"]["stratum"], case["effort"])][name].append(row)
        regressions["default_cases"] += record["default_regression_checked"]
        regressions["equivalent_cases"] += record["equivalent_regression_checked"]

    sensitivity = []
    with (root / "bd-rate.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = row["image"], row["stratum"], int(row["effort"])
            low, high = float(row["quality_low"]), float(row["quality_high"])
            integrals = {}
            for name in (row["reference"], row["variant"]):
                frontier = []
                for point in sorted(groups[key][name], key=lambda p: (p["encoded_bytes"], -p["quality"])):
                    if not frontier or point["quality"] > frontier[-1]["quality"] + 1e-9:
                        frontier.append(point)
                qualities = [p["quality"] for p in frontier]
                require(qualities[0] <= low < high <= qualities[-1], "Extrapolation requested")
                knots = np.array([low] + [q for q in qualities if low < q < high] + [high])
                values = np.interp(knots, qualities, np.log([p["encoded_bytes"] for p in frontier]))
                integrals[name] = float(np.trapezoid(values, knots))
            linear = 100 * math.expm1((integrals[row["variant"]] - integrals[row["reference"]]) / (high - low))
            pchip = float(row.pop("bd_rate_percent"))
            sensitivity.append({**row, "pchip_percent": pchip, "linear_percent": linear,
                                "difference_percentage_points": linear - pchip})
    bins = defaultdict(list)
    for row in sensitivity:
        bins[(row["stratum"], row["effort"], row["reference"], row["variant"])].append(row)
    sensitivity_summary = []
    for key, rows in sorted(bins.items()):
        sensitivity_summary.append({"stratum": key[0], "effort": key[1], "reference": key[2],
            "variant": key[3], "images": len(rows),
            "pchip_median_percent": statistics.median(r["pchip_percent"] for r in rows),
            "linear_median_percent": statistics.median(r["linear_percent"] for r in rows),
            "maximum_absolute_difference_pp": max(abs(r["difference_percentage_points"]) for r in rows)})

    paired, encode_count, decode_count, maximum_error = [], 0, 0, 0.0
    for case in manifest["timing_cases"]:
        complete = root / "timing" / case["id"] / "complete.json"
        if not complete.exists():
            continue
        record = load(complete)
        samples = record["encoder"]["samples"]
        encode_count += len(samples)
        by_round = {(s["round"], s["variant"]): s["elapsed_nanoseconds"] for s in samples}
        require(len(by_round) == manifest["rounds"] * len(manifest["controls"]), "Incomplete encode samples")
        positions = Counter((s["variant"], s["position"]) for s in samples)
        require(set(positions.values()) == {manifest["rounds"] // len(manifest["controls"])},
                "Unbalanced encode positions")
        for name, row in record["rows"].items():
            require(sha(row["artifact"]) == row["sha256"], "Changed timing artifact")
            error = abs(row["quality"] - record["target"])
            require(error <= manifest["quality_tolerance"], "Unmatched timing quality")
            maximum_error = max(error, maximum_error)
        for decode in record["decodes"].values():
            require(len(decode["samples"]) == 2 * manifest["decode_pairs"] and decode["decoded_finite"],
                    "Incomplete decode samples")
            decode_count += len(decode["samples"])
        for reference in ("default", "weighted"):
            for name, row in record["rows"].items():
                if name == reference:
                    continue
                ratios = [by_round[r, name] / by_round[r, reference] for r in range(manifest["rounds"])]
                q25, q75 = np.quantile(ratios, [0.25, 0.75])
                baseline = record["rows"][reference]
                paired.append({"image": case["image"]["name"], "stratum": case["image"]["stratum"],
                    "effort": case["effort"], "reference": reference, "variant": name,
                    "byte_change_percent": 100 * (row["encoded_bytes"] / baseline["encoded_bytes"] - 1),
                    "quality_delta_from_target": row["quality"] - record["target"],
                    "encode_median_change_percent": 100 * (statistics.median(ratios) - 1),
                    "encode_q25_change_percent": 100 * (q25 - 1),
                    "encode_q75_change_percent": 100 * (q75 - 1),
                    "slower_rounds": sum(r > 1 for r in ratios), "rounds": len(ratios)})

    for name in ("curves.csv", "bd-rate.csv", "timing.csv", "summary.json"):
        shutil.copyfile(root / name, output / name)
    write_csv(output / "rate-summary.csv", summary["rate_summaries"])
    write_csv(output / "sensitivity.csv", sensitivity)
    write_csv(output / "sensitivity-summary.csv", sensitivity_summary)
    write_csv(output / "timing-paired.csv", paired)
    audit = {"raw_study": str(root), "manifest_sha256": sha(root / "manifest.json"),
        "source_zip_sha256": sha(root / "source.zip"), "source_revision": manifest["source_revision"],
        "analysis_sha256": sha(__file__), "measurement_tools": manifest["tools"],
        "machine": manifest["machine"], "scorer_version": manifest["scorer_version"],
        "curve_artifacts_audited": len(artifacts), "regressions": dict(regressions),
        "accepted_encode_samples": encode_count, "accepted_decode_samples": decode_count,
        "maximum_timing_quality_error": maximum_error,
        "calibration_probes_completed": len(list((root / "calibration").glob("*/*/attempt-*/complete.json"))),
        "timing_complete": summary["timing_complete"], "timing_expected": summary["timing_expected"],
        "timing_missing": summary["timing_missing"], "bd_rate_groups_excluded": summary["bd_rate_groups_excluded"]}
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: audit[k] for k in ("curve_artifacts_audited", "regressions",
        "accepted_encode_samples", "accepted_decode_samples", "maximum_timing_quality_error")}))


if __name__ == "__main__":
    main()
