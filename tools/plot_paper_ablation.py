#!/usr/bin/env python3
"""Render a standalone ablation figure from verified saved-data aggregates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FACTORS = [
    ("aq-boundaries", "AQ synchronization"),
    ("ac-aq-handoff", "ACS–AQ residency"),
    ("malta-fusion", "Malta fusion / locality"),
    ("epf-linear-fusion", "EPF–RGB fusion"),
    ("ac-fusion", "AC candidate fusion"),
    ("dct-image-io", "Direct image DCT I/O"),
    ("dct-arithmetic", "SIMD DCT arithmetic"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis.resolve()
    audit = json.loads((root / "audit.json").read_text())
    if not audit["complete"] or not audit["verified_artifacts"]:
        raise ValueError("Figure requires complete, artifact-verified analysis")
    if audit["power_problems"] or audit["unstable_process_outputs"]:
        raise ValueError("Resolve power or process-repeatability problems before rendering the paper figure")
    review_path = root / "timing-review.json"
    timing_disturbance = False
    if review_path.exists():
        review = json.loads(review_path.read_text())
        if review["original_manifest_sha256"] != audit["source_manifest_sha256"]:
            raise ValueError("Timing review belongs to a different measurement run")
        timing_disturbance = bool(review["original_jobs_above_twice_round_median"])
    with (root / "aggregate.csv").open() as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [row for row in all_rows if row["cohort"] == "all"]
    lookup = {(r["factor"], int(r["effort"]), float(r["distance"])): r for r in rows}
    efforts = sorted({key[1] for key in lookup})
    distances = sorted({key[2] for key in lookup})
    if len(efforts) != 2 or len(distances) != 3:
        raise ValueError("Figure layout expects two efforts and three distances")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.5), sharex=True, sharey=True)
    colors = ["#235789", "#B35B24", "#32806A"]
    offsets = [-.22, 0., .22]
    minimum, maximum = 1., 1.
    nonidentical = False
    for ax, effort in zip(axes, efforts):
        for i, (factor, label) in enumerate(FACTORS):
            for distance, color, offset in zip(distances, colors, offsets):
                row = lookup[(factor, effort, distance)]
                value = float(row["disabled_over_baseline"])
                low, high = float(row["ci95_low"]), float(row["ci95_high"])
                minimum, maximum = min(minimum, low), max(maximum, high)
                changed = int(row["nonidentical_pairs"]) != 0
                nonidentical |= changed
                y = len(FACTORS) - 1 - i + offset
                ax.hlines(y, low, high, color=color, linewidth=1)
                ax.vlines([low, high], y - .045, y + .045, color=color, linewidth=.8)
                ax.plot(value, y, "o", markersize=3.7, markeredgewidth=1,
                        color=color, markerfacecolor="white" if changed else color,
                        label=f"Distance {distance:g}" if i == 0 else None)
        ax.axvline(1, color="#59626C", linewidth=.8, linestyle="--", zorder=0)
        ax.set_title(f"Effort {effort}", fontsize=9, weight="bold", pad=10)
        ax.set_xlabel("Disabled / optimized encode time")
        ax.grid(axis="x", color="#E5E9ED", linewidth=.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#AEB7C0")
        ax.tick_params(axis="y", length=0, pad=8)
        ax.set_yticks(list(range(len(FACTORS))))
        ax.set_yticklabels([label for _, label in reversed(FACTORS)])
    padding = max(.025, (maximum - minimum) * .1)
    axes[0].set_xlim(minimum - padding, maximum + padding)
    axes[0].set_ylim(-.6, len(FACTORS) - .4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.61, 1.0), handletextpad=.35, columnspacing=1.5)
    fig.subplots_adjust(left=.24, right=.98, top=.82,
                        bottom=.24 if timing_disturbance else .2, wspace=.18)
    image_counts = {int(row["images"]) for row in rows}
    round_counts = {int(row["process_pairs"]) // int(row["images"]) for row in rows}
    if len(image_counts) != 1 or len(round_counts) != 1:
        raise ValueError("Figure conditions have unequal coverage")
    image_count, round_count = image_counts.pop(), round_counts.pop()
    caption = (f"{image_count} image{'s' if image_count != 1 else ''}; "
               f"{round_count} process round{'s' if round_count != 1 else ''}; conditional ablations; complete public call.\n"
               "Whiskers: 95% source-scene cluster bootstrap intervals.")
    if nonidentical:
        caption += "\nOpen markers include nonidentical outputs; consult the rate/quality audit."
    if timing_disturbance:
        caption += "\nTiming disturbance retained; see sensitivity.csv."
    fig.text(.24, .035, caption, fontsize=6.4, color="#444C55", va="bottom")
    artifacts = []
    for suffix in ("pdf", "svg", "png"):
        path = root / f"ablation-effects.{suffix}"
        fig.savefig(path, dpi=220, facecolor="white")
        artifacts.append({"path": path.name, "sha256": sha(path)})
    plt.close(fig)
    cohort_labels = [("kodak_0_4mp", "Kodak\n0.4 MP"),
                     ("clic_1_8_to_3_4mp", "CLIC\n1.8–3.4 MP"),
                     ("12mp", "12 MP"), ("24mp", "24 MP"), ("48mp", "48 MP")]
    available = {r["cohort"] for r in all_rows}
    if all(key in available for key, _ in cohort_labels) and 1.0 in distances:
        cohort_lookup = {(r["factor"], int(r["effort"]), r["cohort"]): r for r in all_rows
                         if float(r["distance"]) == 1.0}
        grids = []
        for effort in efforts:
            grids.append(np.array([[100 * (float(cohort_lookup[(factor, effort, cohort)]["disabled_over_baseline"]) - 1)
                                    for cohort, _ in cohort_labels] for factor, _ in FACTORS]))
        extent = max(1., max(float(np.max(np.abs(grid))) for grid in grids))
        fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.7), sharey=True)
        changed_cells = False
        for ax, effort, grid in zip(axes, efforts, grids):
            mesh = ax.imshow(grid, cmap="RdBu_r", vmin=-extent, vmax=extent, aspect="auto")
            for i, (factor, _) in enumerate(FACTORS):
                for j, (cohort, _) in enumerate(cohort_labels):
                    row = cohort_lookup[(factor, effort, cohort)]
                    changed = int(row["nonidentical_pairs"]) != 0
                    changed_cells |= changed
                    ax.text(j, i, f"{grid[i, j]:+.1f}" + ("†" if changed else ""),
                            ha="center", va="center", fontsize=7,
                            color="white" if abs(grid[i, j]) > .65 * extent else "#202833")
            ax.set_title(f"Effort {effort}", fontsize=9, weight="bold", pad=9)
            ax.set_xticks(range(len(cohort_labels)), [label for _, label in cohort_labels], fontsize=6.7)
            ax.set_yticks(range(len(FACTORS)), [label for _, label in FACTORS])
            ax.tick_params(length=0, pad=6)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.subplots_adjust(left=.24, right=.875, top=.88, bottom=.25, wspace=.08)
        cax = fig.add_axes([.90, .25, .013, .63])
        bar = fig.colorbar(mesh, cax=cax)
        bar.ax.tick_params(labelsize=6.5, length=2)
        bar.set_label("Extra encode time when disabled (%)", fontsize=6.5, labelpad=5)
        text = ("Distance 1; equal-image geometric mean of paired time ratios. Positive values favor the optimization.\n"
                "12/24/48 MP each contain three scenes. Per-cohort intervals and round variation are in aggregate.csv.")
        if changed_cells:
            text += "\n† Includes nonidentical outputs; consult the rate/quality audit."
        if timing_disturbance:
            text += "\nTiming disturbance retained; see sensitivity.csv."
        fig.text(.24, .035, text, fontsize=6.3, color="#444C55", va="bottom")
        for suffix in ("pdf", "svg", "png"):
            path = root / f"ablation-by-size.{suffix}"
            fig.savefig(path, dpi=220, facecolor="white")
            artifacts.append({"path": path.name, "sha256": sha(path)})
        plt.close(fig)
    provenance = {"aggregate_sha256": sha(root / "aggregate.csv"),
                  "audit_sha256": sha(root / "audit.json"), "script_sha256": sha(__file__),
                  "matplotlib": matplotlib.__version__, "python": sys.version,
                  "timing_review_sha256": sha(review_path) if review_path.exists() else None,
                  "nonidentical_output_markers": nonidentical, "artifacts": artifacts}
    (root / "figure-manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
