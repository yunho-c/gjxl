#!/usr/bin/env python3
"""Audit the complete DC-grid confirmation and analyze saved curves only."""
import argparse
from collections import Counter, defaultdict
import itertools
from pathlib import Path
import statistics

import study


def audit(root, manifest, observations):
    expected = {f'{image}|{arm}|q{quality}' for image, arm, quality in itertools.product(
        manifest['image_order'], manifest['arms'], manifest['qualities'])}
    assert len(observations) == len(expected)
    assert {r['id'] for r in observations} == expected
    for path, digest in manifest['files'].items():
        assert study.sha(path) == digest, path
    saved = {(r['image_id'],r['effort'],r['requested_quality']):r
             for r in study.rows(study.GJXL/'scores.jsonl') if r['effort'] in (3,4)}
    native_controls = 0
    for row in observations:
        policy = manifest['arms'][row['arm']]
        assert study.sha(row['output_path']) == row['output_sha256'], row['id']
        assert Path(row['output_path']).stat().st_size == row['encoded_bytes']
        reference = saved[row['image_id'],row['effort'],row['requested_quality']]
        assert row['distance'] == reference['distance']
        if row['source'] == 'verified-saved-native':
            for key, value in reference.items():
                assert row[key] == value, (row['id'],key)
            continue
        rawpath = Path(row['output_path']).parent/'raw.json'
        assert study.sha(rawpath) == row['raw_sha256']
        raw = study.read(rawpath)
        assert raw['dc_quantization'] == policy['quantization']
        assert raw['adaptive_dc_smoothing'] == policy['smoothing']
        assert raw['revision'] == manifest['revision']
        assert raw['effort'] == policy['effort'] and raw['dc_prediction']=='weighted'
        assert raw['validation_encodes'] == raw['sample_count'] == 1
        assert raw['samples'][0]['encoded_bytes'] == row['encoded_bytes']
        assert raw['thread_count']==8 and raw['metal_aq_mode']=='fully-resident'
        assert row['extra_dc_precision_override'] == policy.get('precision_override')
        if row['source'] == 'fresh-native-control':
            native_controls += 1
            for key in ('output_sha256','decoded_sha256'):
                assert row[key] == reference[key], (row['id'],key)
            assert abs(row['score']-reference['score']) < 1e-9
    assert native_controls == 2*len(manifest['images'])
    commands = study.rows(root/'commands.jsonl')
    assert all(r['returncode']==0 for r in commands)
    return {'status':'passed','observations':len(observations),
            'source_counts':dict(Counter(r['source'] for r in observations)),
            'native_byte_decode_score_controls':native_controls,
            'verified_frozen_files':len(manifest['files']),
            'verified_output_hashes':len(observations),'successful_commands':len(commands),
            'metric':'pinned fast-SSIMULACRA2 on linear-sRGB float decodes',
            'timing':'diagnostic, not isolated; no speed claim'}


def analyze(manifest, observations):
    curves = defaultdict(list)
    for row in observations:
        curves[row['image_id'],row['arm']].append(row)
    for row in study.rows(study.LIBJXL/'scores.jsonl'):
        if row['effort'] in (3,4) and row['resampling']==1:
            curves[row['image_id'],f'libjxl-e{row["effort"]}'].append(row)
    pairs = []
    for e in (3,4):
        candidate = f'e{e}-round-smooth0-precision1'
        pairs += [(f'native-e{e}-vs-libjxl-e{e}', study.native(e),f'libjxl-e{e}'),
                  (f'candidate-e{e}-vs-libjxl-e{e}',candidate,f'libjxl-e{e}'),
                  (f'candidate-e{e}-vs-native-e{e}',candidate,study.native(e))]
    pairs += [('candidate-zero-AQ-vs-libjxl-e4','e3-round-smooth0-precision1','libjxl-e4'),
              ('candidate-one-vs-zero-AQ','e4-round-smooth0-precision1','e3-round-smooth0-precision1')]
    comparisons = []
    for low, high in [(75,85),(45,55),(25,35)]:
        for label, candidate, reference in pairs:
            for image in manifest['image_order']:
                row = {'comparison':label,'candidate':candidate,'reference':reference,
                       'image_id':image,'scope':manifest['images'][image]['resolution_class'],
                       'range':[low,high],'status':'ready'}
                try:
                    for key in (candidate,reference):
                        if key in manifest['arms'] and len(curves[image,key])!=len(manifest['qualities']):
                            raise ValueError('Incomplete candidate/native curve')
                    for method in ('pchip','akima'):
                        row[method] = study.bd.bd_rate(*study.curve(curves[image,reference]),
                            *study.curve(curves[image,candidate]), (low,high), method)
                except ValueError as error:
                    row['status'] = str(error)
                comparisons.append(row)
    groups = defaultdict(list)
    for row in comparisons:
        for scope in ('all',row['scope']):
            groups[row['comparison'],tuple(row['range']),scope].append(row)
    summary = []
    for (label, quality, scope), group in sorted(groups.items()):
        ready = [r for r in group if r['status']=='ready']
        row = {'comparison':label,'range':list(quality),'scope':scope,
               'ready':len(ready),'expected':len(group),
               'failures':dict(Counter(r['status'] for r in group if r['status']!='ready'))}
        if len(ready)==len(group):
            row.update({method:statistics.mean(r[method] for r in ready) for method in ('pchip','akima')})
            row.update(wins=sum(r['pchip']<0 for r in ready),
                       worst=max(r['pchip'] for r in ready),best=min(r['pchip'] for r in ready),
                       max_interpolation_difference_pp=max(abs(r['pchip']-r['akima']) for r in ready))
        summary.append(row)
    return {'aggregation':'arithmetic mean of per-image BD percentages; full cohort required',
            'curve_policy':'Original setting order; no extrapolation or removal of reversals',
            'summary':summary,'images':comparisons}


def plot(root, analysis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    names = ['all','kodak_0_4mp','clic_1_8_to_3_4mp','12mp','24mp','48mp']
    labels = ['All 65','Kodak (24)','CLIC (32)','12 MP (3)','24 MP (3)','48 MP (3)']
    fig, axes = plt.subplots(1,2,figsize=(10,5),layout='constrained')
    for ax,e in zip(axes,(3,4)):
        for prefix, offset, color, label in [('native',-.12,'#999999','Native'),
                                             ('candidate',.12,'#4477AA','Ordinary DC, extra bit, smoothing off')]:
            for n,scope in enumerate(names):
                row = next(r for r in analysis['summary'] if r['comparison']==f'{prefix}-e{e}-vs-libjxl-e{e}'
                           and r['scope']==scope and r['range']==[75,85])
                if 'pchip' in row:
                    ax.scatter(row['pchip'],n+offset,color=color,s=35,label=label if n==0 else None)
                else:
                    ax.text(0,n+offset,f"{row['ready']}/{row['expected']} supported",fontsize=8)
        ax.set(yticks=np.arange(6),yticklabels=labels,xlabel=f'BD-rate vs libjxl e{e} (%)',
               title=f'Effort {e}')
        ax.invert_yaxis();ax.axvline(0,color='#777777',lw=.8)
        ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside lower center',ncols=2,frameon=False)
    fig.suptitle('DC-grid confirmation · all original images · SSIMULACRA2 75–85\nLower is better; diagnostic policy, not a production default',fontsize=12)
    for extension in ('png','svg'):
        fig.savefig(root/f'dc-grid-confirmation.{extension}',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('build/e3-e4-dc-grid-confirm-20260914'))
    args=parser.parse_args();root=args.root.resolve()
    manifest=study.read(root/'manifest.json');observations=study.rows(root/'observations.jsonl')
    audited=audit(root,manifest,observations)
    analysis=analyze(manifest,observations)
    audited['curve_statuses']=dict(Counter(r['status'] for r in analysis['images']))
    study.save(root/'audit.json',audited)
    study.save(root/'analysis.json',analysis)
    plot(root,analysis)
    for row in analysis['summary']:
        if row['scope']=='all':print(row)


if __name__=='__main__':main()
