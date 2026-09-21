"""Validate same-call host/GPU captures; no benchmark or plotting dependencies."""
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def gpu_intervals(sample):
    previous = 0
    result = []
    caps = sample['capabilities']
    if not (caps['timestamp_counter'] and caps['stage_boundary']):
        raise ValueError('Stage timestamps unavailable')
    for submission in sample['submissions']:
        for stage in submission['stages']:
            duration = stage['gpu_nanoseconds']
            begin, end = stage['begin_timestamp'], stage['end_timestamp']
            if not stage['timestamp_valid']:
                dispatches = stage['dispatches']
                if (duration != 0 or begin != 0 or end != 0 or not dispatches or
                    not all(d['kind'] == 'indirect_threadgroups' and 0 in d['grid']
                            for d in dispatches)):
                    raise ValueError('Unexplained missing timestamp')
            else:
                if (not all(math.isfinite(v) for v in (begin, end, duration)) or
                    begin <= 0 or end < begin or end - begin != duration or begin < previous):
                    raise ValueError('Invalid or overlapping GPU interval')
                previous = end
            result.append((stage['stage_id'], duration))
    if not result or sum(value for _, value in result) <= 0:
        raise ValueError('No measured GPU intervals')
    return result


def validate(directory, job, config):
    directory = Path(directory)
    gpu = load(directory / 'gpu.json')
    paired = load(directory / 'paired.json')
    required = dict(schema_version=4, execution_path='production-aligned-resident-v1',
                    scope='metal-public-workflow', mode='stage', gpu_aq='fully-resident',
                    collect_final_score=False, warmups=config['warmups'], sample_count=config['samples'])
    if any(gpu.get(k) != v for k, v in required.items()):
        raise ValueError('GPU capture metadata mismatch')
    if not math.isclose(gpu['distance'], job['distance'], rel_tol=1e-6):
        raise ValueError('Distance mismatch')
    if paired.get('capture_kind') != 'paired-same-call-v1':
        raise ValueError('Host and GPU profiles are not from the same call')
    if len(gpu['workloads']) != 1:
        raise ValueError('Expected one input')
    work = gpu['workloads'][0]
    for data in (work, paired):
        if (data['source_width'], data['source_height']) != (job['width'], job['height']):
            raise ValueError('Input geometry mismatch')
    rows = paired['rows']
    n, w = config['samples'], config['warmups']
    expected = {(i, mode) for i in range(-w, n) for mode in ('ordinary', 'profiled')}
    if len(rows) != len(expected) or {(r['sample_index'], r['mode']) for r in rows} != expected:
        raise ValueError('Incomplete paired samples')
    if not all(r['byte_equal'] is True and r['summary_equal'] is True and
               r['complete_call_nanoseconds'] > 0 for r in rows):
        raise ValueError('Encode validation failed')
    if len({r['committed_submissions'] for r in rows}) != 1:
        raise ValueError('Submission boundaries changed')
    samples = work['samples']
    if len(samples) != n or {s['sample_index'] for s in samples} != set(range(n)):
        raise ValueError('Incomplete GPU samples')
    by_key = {(r['sample_index'], r['mode']): r for r in rows}
    for sample in samples:
        row = by_key[sample['sample_index'], 'profiled']
        p = row['phase_nanoseconds']
        if not all(math.isfinite(v) and v >= 0 for v in p.values()):
            raise ValueError('Invalid host durations')
        checks = (
            (p['input_preparation'], sum(p[k] for k in ('input_geometry_and_storage',
                'input_color_transform', 'input_matrix_scale_stats', 'input_resident_preparation',
                'input_quantization_preparation'))),
            (p['codestream_encoding'], sum(p['codestream_' + k] for k in
                ('validation', 'dc_tokenization', 'ac_tokenization', 'entropy_optimization',
                 'section_writing', 'assembly'))),
            (p['total'], p['input_preparation'] + p['quantization_pipeline'] + p['codestream_encoding']),
            (row['complete_call_nanoseconds'], p['total']),
            (p['quantization_pipeline'], sum(v for _, v in gpu_intervals(sample))))
        if any(parent < children for parent, children in checks):
            raise ValueError('Negative additive-partition residual')
    deltas = [(by_key[i, 'profiled']['complete_call_nanoseconds'] -
               by_key[i, 'ordinary']['complete_call_nanoseconds']) / 1e6 for i in range(n)]
    return {'ordinary_median_ms': statistics.median(by_key[i, 'ordinary']['complete_call_nanoseconds']/1e6 for i in range(n)),
            'profiled_median_ms': statistics.median(by_key[i, 'profiled']['complete_call_nanoseconds']/1e6 for i in range(n)),
            'paired_delta_median_ms': statistics.median(deltas), 'paired_deltas_ms': deltas,
            'encoded_bytes': paired['encoded_bytes'], 'submissions': rows[0]['committed_submissions'],
            'reference_sha256': sha(directory/'reference.jxl'),
            'gpu_sha256': sha(directory/'gpu.json'), 'paired_sha256': sha(directory/'paired.json')}
