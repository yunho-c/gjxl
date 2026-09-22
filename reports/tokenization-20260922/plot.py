#!/usr/bin/env python3
"""Publication-style figures from retained summary statistics only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ['Kodak 01 · 0.39 MP', 'CLIC A · 3.09 MP', 'CLIC B · 3.36 MP',
         'Campus · 12 MP', 'Alpine · 24 MP', 'Forest · 48 MP']


def main():
    data = json.loads((ROOT / 'summary.json').read_text())
    rows = data['profile']
    images = list(dict.fromkeys(r['image'] for r in rows))
    names = dict(zip(images, NAMES))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'savefig.dpi': 180})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8), sharex=True, sharey=True)
    for ax, effort in zip(axes, [1, 7, 8]):
        subset = [r for r in rows if r['effort'] == effort]
        y = np.arange(len(subset))
        for label, color, offset, title in [('combined', '#16697a', -.13, 'vs original main'), ('gpu_over_cpu', '#d46a40', .13, 'vs improved CPU')]:
            center = np.array([r[label + '_saving_percent_median'] for r in subset])
            lower = np.array([r[label + '_saving_percent_minimum'] for r in subset])
            upper = np.array([r[label + '_saving_percent_maximum'] for r in subset])
            ax.errorbar(center, y + offset, xerr=[center-lower, upper-center], fmt='o',
                        color=color, capsize=3, markersize=5, linewidth=1.4, label=title)
        ax.axvline(0, color='#9b9b9b', linewidth=.8)
        ax.grid(axis='x', color='#eeeeee')
        ax.set_title(f'Effort {effort}', loc='left', fontweight='bold')
        ax.set_yticks(y, [names[r['image']] for r in subset])
        ax.set_xlabel('Complete-call latency reduction (%)')
    axes[0].invert_yaxis()
    axes[-1].legend(loc='lower right', frameon=False, fontsize=9)
    fig.suptitle('Compact GPU tokenization + parallel CPU DC + overlap', x=.02, ha='left', fontsize=15, fontweight='bold')
    fig.text(.02, .01, 'M4 Pro · distance 1.9 · 8 CPU participants. Dots: median paired process-round reduction; whiskers: observed range across 5 rounds, not a confidence interval.', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .92))
    fig.savefig(ROOT/'complete-call.png');fig.savefig(ROOT/'complete-call.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
