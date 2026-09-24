#!/usr/bin/env python3
"""Saved-artifact analysis; never launches an encoder."""
import csv
import hashlib
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def median_range(values):
    return dict(median=st.median(values), minimum=min(values), maximum=max(values))


def write_csv(name, rows):
    with (ROOT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def vm(p):
    return {k: int(v) for k, v in re.findall(r'^([^:\n]+):\s+(\d+)\.', p.read_text(), re.M)}


def main():
    profile = json.loads((ROOT / 'final-profile/results.json').read_text())
    batch = json.loads((ROOT / 'final-batch/results.json').read_text())
    assert len(profile) == 360 and len(batch) == 108
    expected = {}
    grouped = defaultdict(dict)
    gpu_groups = defaultdict(list)
    unnamed_dispatches = []
    for r in profile:
        key = (r['image'], r['effort'])
        expected.setdefault(key, r['output_sha256'])
        assert expected[key] == r['output_sha256']
        p = ROOT / 'final-profile' / f"r{r['round']}-{r['image']}-e{r['effort']}-{r['mode']}"
        assert sha(p / 'reference.jxl') == r['output_sha256']
        pairs = json.loads((p / 'paired.json').read_text())['rows']
        assert all(x['byte_equal'] and x['summary_equal'] for x in pairs)
        assert len({x['committed_submissions'] for x in pairs}) == 1
        profiled = [x for x in pairs if x['sample_index'] >= 0 and x['mode'] == 'profiled']
        gpu = json.loads((p / 'gpu.json').read_text())
        for sample in gpu['workloads'][0]['samples']:
            totals = defaultdict(float)
            for submission in sample['submissions']:
                for stage in submission['stages']:
                    totals[stage['group_id']] += stage['gpu_nanoseconds'] / 1e6
                    for dispatch in stage['dispatches']:
                        if not dispatch['kernel_id'].startswith('gjxl_'):
                            unnamed_dispatches.append(dict(path=str(p), kernel_id=dispatch['kernel_id']))
            for group, milliseconds in totals.items():
                gpu_groups[key + (r['mode'], group)].append(milliseconds)
        r['profiled_ms'] = st.median(x['complete_call_nanoseconds'] / 1e6 for x in profiled)
        for phase in ['codestream_encoding', 'quantization_pipeline', 'codestream_entropy_optimization', 'codestream_section_writing']:
            r[phase + '_ms'] = st.median(x['phase_nanoseconds'][phase] / 1e6 for x in profiled)
        grouped[key][r['round'], r['mode']] = r
    profile_summary = []
    for (image, effort), rows in grouped.items():
        assert len(rows) == 20
        out = dict(image=image, effort=effort)
        for mode in ['base', 'cpu', 'release', 'joined1']:
            for metric in ['ordinary_ms', 'dc_ms', 'ac_ms', 'profiled_ms', 'codestream_encoding_ms', 'quantization_pipeline_ms', 'codestream_entropy_optimization_ms', 'codestream_section_writing_ms']:
                out[mode + '_' + metric] = st.median(rows[i, mode][metric] for i in range(5))
        for control, candidate, label in [('base', 'cpu', 'dc'), ('cpu', 'release', 'cleanup'), ('base', 'joined1', 'combined'), ('release', 'joined1', 'gpu_over_cpu')]:
            saving = [100 * (1 - rows[i, candidate]['ordinary_ms'] / rows[i, control]['ordinary_ms']) for i in range(5)]
            for k, v in median_range(saving).items():
                out[label + '_saving_percent_' + k] = v
        profile_summary.append(out)
    write_csv('profile-summary.csv', profile_summary)
    write_csv('gpu-stage-summary.csv', [dict(image=k[0], effort=k[1], mode=k[2], group=k[3], milliseconds=st.median(v)) for k, v in gpu_groups.items()])
    (ROOT / 'unnamed-dispatches.json').write_text(json.dumps(unnamed_dispatches, indent=2))

    bgroups = defaultdict(dict)
    vm_rows = []
    for r in batch:
        key = (r['image'], r['effort'])
        assert expected[key] == r['output_sha256']
        p = ROOT / 'final-batch' / r['artifacts']
        assert sha(p / 'reference.jxl') == r['output_sha256']
        plan = json.loads((p / 'plan.json').read_text())
        samples = list(csv.DictReader((p / 'samples.csv').open()))
        assert max(int(x['peak_cpu_slots']) for x in samples) <= 8
        assert max(int(x['peak_committed_bytes']) for x in samples) <= plan['memory_limit']
        r['first_encode_ms'] = int(samples[0]['wall_ns']) / 1e6
        r['in_flight'] = plan['in_flight']
        r['trim_after_each_image'] = plan['trim_after_each_image']
        attempt = p.name.removeprefix('attempt')
        before, after = vm(p.parent / f'vm-before{attempt}.txt'), vm(p.parent / f'vm-after{attempt}.txt')
        for metric in ['Pageouts', 'Swapouts', 'Swapins', 'Compressions']:
            r[metric] = after[metric] - before[metric]
        vm_rows.append({k: r[k] for k in ['round', 'image', 'effort', 'batch', 'mode', 'in_flight', 'trim_after_each_image', 'peak_backing_bytes', 'peak_cpu_slots', 'first_encode_ms', 'Pageouts', 'Swapouts', 'Swapins', 'Compressions']})
        bgroups[key + (r['batch'],)][r['round'], r['mode']] = r
    write_csv('batch-process-details.csv', vm_rows)
    batch_summary = []
    for (image, effort, count), rows in bgroups.items():
        assert len(rows) == 9
        out = dict(image=image, effort=effort, batch=count)
        for mode in ['base', 'release', 'joined']:
            for metric in ['wall_ms', 'fps', 'first_encode_ms']:
                out[mode + '_' + metric] = st.median(rows[i, mode][metric] for i in range(3))
            out[mode + '_peak_backing_gib'] = max(rows[i, mode]['peak_backing_bytes'] for i in range(3)) / 2**30
            out[mode + '_in_flight'] = rows[0, mode]['in_flight']
            out[mode + '_trim'] = rows[0, mode]['trim_after_each_image']
        for control, label in [('base', 'combined'), ('release', 'gpu_over_cpu')]:
            ratios = [rows[i, control]['wall_ms'] / rows[i, 'joined']['wall_ms'] for i in range(3)]
            for k, v in median_range(ratios).items():
                out[label + '_throughput_ratio_' + k] = v
        batch_summary.append(out)
    write_csv('batch-summary.csv', batch_summary)
    output = dict(profile=profile_summary, batch=batch_summary, validation=dict(profile_processes=len(profile), ordinary_samples=4*len(profile), profiled_samples=4*len(profile), batch_processes=len(batch), batch_timed_calls=5*len(batch), distinct_outputs=len(expected), unnamed_dispatches=len(unnamed_dispatches), max_cpu_slots=max(r['peak_cpu_slots'] for r in batch), pageout_processes=sum(r['Pageouts'] > 0 for r in batch), swapout_processes=sum(r['Swapouts'] > 0 for r in batch)))
    (ROOT / 'summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output['validation'], indent=2))


if __name__ == '__main__':
    main()
