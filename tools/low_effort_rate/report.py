#!/usr/bin/env python3
"""Audit and visualize saved low-effort rate results; never runs encoders."""
from collections import Counter
import itertools
from pathlib import Path
import statistics

import study


def main():
    root = Path('build/e3-e4-rate-20260914').resolve()
    manifest = study.read(root / 'manifest.json')
    observations = study.rows(root / 'observations.jsonl')
    expected = {f'{image}|{arm}|q{quality}' for image, arm, quality in itertools.product(
        manifest['image_order'], manifest['arms'], manifest['qualities'])}
    assert {r['id'] for r in observations} == expected
    assert len(observations) == len(expected)
    for path, digest in manifest['files'].items():
        assert study.sha(path) == digest, path
    for row in observations:
        assert study.sha(row['output_path']) == row['output_sha256'], row['id']
        if 'raw_sha256' in row:
            assert study.sha(Path(row['output_path']).parent / 'raw.json') == row['raw_sha256']
    commands = study.rows(root / 'commands.jsonl')
    assert all(r['returncode'] == 0 for r in commands)
    kinds = Counter(r['source'] for r in observations)
    assert kinds['fresh-native-control'] == 2 * len(manifest['image_order'])
    study.analyze(root)
    analysis = study.read(root / 'dc-factorial-analysis.json')
    study.save(root / 'audit.json', {
        'status': 'passed', 'observations': len(observations),
        'source_counts': kinds, 'commands': len(commands),
        'frozen_files': len(manifest['files']),
        'source_output_hashes_verified_at_freeze': manifest['verified_source_outputs'],
        'outputs_rehashed': len(observations),
        'native_control_contract': 'Codestream, pinned decoded PFM hash, and score reproduce source; benchmark checks repeated bytes',
        'curve_statuses': Counter(r['status'] for r in analysis['images']),
    })
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout='constrained')
    order = ['all', 'kodak_0_4mp', 'clic_1_8_to_3_4mp', '12mp', '24mp', '48mp']
    labels = ['All 65', 'Kodak (24)', 'CLIC (32)', '12 MP (3)', '24 MP (3)', '48 MP (3)']
    for effort, offset, color in [(3, -.13, '#4477AA'), (4, .13, '#CC6677')]:
        direct = study.read(root / f'direct-e{effort}.json')
        values = {p['scope']: p['bd_rate_pchip'] for p in direct['points'] if p['encoder'] == 'gjxl'}
        axes[0].scatter([values[key] for key in order], np.arange(6)+offset,
                        s=45, color=color, label=f'Effort {effort}')
    axes[0].set(yticks=np.arange(6), yticklabels=labels, xlabel='BD-rate vs same-effort libjxl (%)',
                title='The actual same-effort gap\nOriginal 65-image sweep')
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False)
    arm_order = [study.arm(e,q,s) for e,q,s in itertools.product(
        (3,4), ('round','prediction-aware'), (False,True))]
    arm_labels = [f"e{manifest['arms'][a]['effort']} · "
                  f"{'ordinary DC' if manifest['arms'][a]['quantization']=='round' else 'prediction DC + extra bit'} · "
                  f"smooth {'on' if manifest['arms'][a]['smoothing'] else 'off'}" for a in arm_order]
    for n, key in enumerate(arm_order):
        row = next(r for r in analysis['summary'] if r['arm']==key and r['reference']=='stock')
        if 'pchip' not in row:
            axes[1].text(0, n, f"Incomplete: {row['ready']}/{row['planned']}")
            continue
        color = '#4477AA' if key.startswith('e3') else '#CC6677'
        axes[1].scatter(row['pchip'], n, color=color, s=45)
        axes[1].annotate(f"{row['pchip']:+.2f}%", (row['pchip'],n), xytext=(6,0),
                         textcoords='offset points', va='center', fontsize=9)
    axes[1].set(yticks=np.arange(8), yticklabels=arm_labels,
                xlabel='BD-rate vs same-effort libjxl (%)',
                title='DC and AQ factorial\nTen selected diagnostic inputs')
    axes[1].invert_yaxis()
    for ax in axes:
        ax.axvline(0, color='#666666', linewidth=.8)
        ax.grid(axis='x', alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        ax.margins(x=.28)
    fig.suptitle('GJXL b1fbfc1 · measured SSIMULACRA2 75–85 · lower is better', fontsize=13)
    for suffix in ('png', 'svg'):
        fig.savefig(root / f'dc-factorial.{suffix}', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
