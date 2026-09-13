"""Source-bound, resumable small-image DC tree qualification.

Explicit init/collect/timing commands perform work; report reads artifacts only.
The lossless writer change requires exact decoded equality, not quality fitting.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import zipfile

import numpy as np
import study as common

ROOT = common.ROOT
sha, load, save, check = common.sha, common.load, common.save, common.check
BASE = "6ca9d5ff0b313f2380099b72af390b56c3202b41"


def read_pfm(path):
    with Path(path).open('rb') as stream:
        check(stream.readline().strip() == b'PF', 'Expected RGB PFM')
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        data = np.frombuffer(stream.read(), dtype='<f4' if scale < 0 else '>f4')
    check(len(data) == width * height * 3 and np.isfinite(data).all(), 'Invalid PFM')
    return data.reshape(height, width, 3)[::-1].astype(np.float64) * abs(scale)


def area_weights(source, target):
    edges = np.linspace(0, source, target + 1)
    left = np.arange(source)
    weights = np.maximum(0, np.minimum(edges[1:, None], left + 1) -
                         np.maximum(edges[:-1, None], left))
    return weights / (source / target)


def thumbnail(image, side, folder):
    source = read_pfm(image['path'])
    height, width = source.shape[:2]
    scale = side / max(width, height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    wx, wy = area_weights(width, w), area_weights(height, h)
    # Sparse area sums avoid backend-dependent BLAS execution and FP-status
    # warnings from the macOS matrix-multiply implementation.
    horizontal = []
    for weights in wx:
        indices = np.flatnonzero(weights)
        horizontal.append(np.sum(source[:, indices, :] * weights[indices][None, :, None], axis=1))
    intermediate = np.stack(horizontal, axis=1)
    vertical = []
    for weights in wy:
        indices = np.flatnonzero(weights)
        vertical.append(np.sum(intermediate[indices, :, :] * weights[indices][:, None, None], axis=0))
    result = np.stack(vertical)
    check(np.isfinite(result).all() and result.min() >= source.min()-1e-12 and
          result.max() <= source.max()+1e-12, 'Invalid area-resampled thumbnail')
    name = f"{image['name']}-thumb{side}"
    path = folder / f'{name}.pfm'
    with path.open('wb') as stream:
        stream.write(f'PF\n{w} {h}\n-1.0\n'.encode())
        stream.write(result[::-1].astype('<f4').tobytes())
    return {'name': name, 'path': str(path), 'sha256': sha(path), 'width': w, 'height': h,
            'stratum': 'thumbnail', 'source': image, 'long_side': side,
            'resampling': 'separable exact area average in linear RGB, float64 accumulation, float32 PFM'}


def source_files():
    return sorted(set(common.source_files()) | set((ROOT / 'tests').glob('*.cpp')))


def initialize(args):
    output = args.output.resolve()
    check(not output.exists(), 'Use a new output directory')
    check(common.capture(['git', '-C', ROOT, 'rev-parse', 'HEAD']) == BASE, 'Unexpected base revision')
    prior = load(args.inputs)
    unique = {c['image']['name']: c['image'] for c in prior['cases']}
    selected = [x for x in unique.values() if x['stratum'] in ('compact', 'photo_4k')]
    parents = []
    for stratum in ('clic2024_test', 'kodak'):
        candidates = sorted((x for x in unique.values() if x['stratum'] == stratum), key=lambda x: x['name'])
        parents.extend(candidates[i] for i in (0, len(candidates)//2, len(candidates)-1))
    for item in selected + parents:
        check(sha(item['path']) == item['sha256'], 'Input hash changed')
    check(len(selected) == 12 and len(parents) == 6, 'Missing cohort inputs')
    binary = (ROOT / 'build/release/gjxl_dc_tree_benchmark').resolve()
    check('no work to do' in common.capture(['ninja', '-C', binary.parent, '-n', binary.name]), 'Rebuild required')
    decoder = prior['decoder']
    tools = {str(binary): sha(binary), decoder['binary']: sha(decoder['binary']), **decoder['libraries']}
    output.mkdir(parents=True)
    (output / 'inputs').mkdir()
    selected += [thumbnail(p, side, output / 'inputs') for p in parents for side in (256, 384)]
    selected.sort(key=lambda x: x['name'])
    cases = [{'id': f"{image['name']}-e{e}-d{d:g}-{prediction}", 'image': image,
              'effort': e, 'distance': d, 'prediction': prediction, 'backend': 'metal', 'flags': 0}
             for image in selected for e in (3, 4, 7) for d in (.6, 1., 2.)
             for prediction in ('gradient', 'weighted')]
    timing_names = {'edge', 'imazen26-1029-planter-4k', f"{parents[0]['name']}-thumb256"}
    integration_names = {'flat', 'edge', f"{parents[0]['name']}-thumb384"}
    integration = [{'id': f"{image['name']}-{backend}-{prediction}-f{flags}", 'image': image,
                    'effort': 4, 'distance': 1., 'prediction': prediction, 'backend': backend, 'flags': flags}
                   for image in selected if image['name'] in integration_names
                   for backend in ('cpu', 'metal') for prediction in ('gradient', 'weighted')
                   for flags in range(4)]
    manifest = {'schema_version': 1, 'purpose': 'libjxl-style fixed DC tree pruning', 'base_revision': BASE,
                'source_root': str(ROOT), 'sources': {str(p.relative_to(ROOT)): sha(p) for p in source_files()},
                'binary': str(binary), 'decoder': decoder, 'tools': tools, 'threads': 8,
                'input_manifest': {'path': str(args.inputs.resolve()), 'sha256': sha(args.inputs)},
                'images': selected, 'cases': cases, 'integration': integration,
                'timing': [c for c in cases if c['image']['name'] in timing_names and c['distance'] == 1],
                'warmups': 3, 'pairs': 20,
                'gate': {'no_case_growth': True, 'affected_aggregate_savings_per_predictor': True,
                         'time_threshold_ms': .1, 'time_threshold_fraction': .02,
                         'slower_pairs_required': 16, 'repeat_offender_once': True},
                'build_cache_sha256': sha(binary.parent / 'CMakeCache.txt'),
                'build_commands': common.capture(['ninja', '-C', binary.parent, '-t', 'commands', binary.name]),
                'system': {'os': common.capture(['sw_vers']), 'hardware': common.capture(['sysctl', '-n', 'hw.model'])}}
    save(output / 'manifest.json', manifest)
    with zipfile.ZipFile(output / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in source_files(): archive.write(path, str(path.relative_to(ROOT)))
    (output / 'source.diff').write_text(common.capture(['git', '-C', ROOT, 'diff', 'HEAD', '--']))
    print(f"Initialized {len(cases)} predictor cases (864 streams), {len(integration)} integration pairs, {len(manifest['timing'])} timing cases", flush=True)


def validate_manifest(manifest):
    for path, digest in manifest['sources'].items(): check(sha(ROOT / path) == digest, f'Source changed: {path}')
    for path, digest in manifest['tools'].items(): check(sha(path) == digest, f'Tool changed: {path}')


def audit(record):
    check(record['decode']['decoded_equal'] and record['decode']['decoded_finite'], 'Invalid decode')
    for path, digest in record['artifacts'].items(): check(sha(path) == digest, 'Artifact changed')


def run_pair(manifest, case, folder, timed):
    image = case['image']
    check(sha(image['path']) == image['sha256'], 'Input changed')
    warmups, pairs = (manifest['warmups'], manifest['pairs']) if timed else (0, 1)
    common.command([manifest['binary'], image['path'], folder / 'streams', case['distance'], case['effort'],
                    manifest['threads'], warmups, pairs, case['prediction'], case['backend'], case['flags']], folder, 'encode')
    encoded = load(folder / 'encode.stdout')
    samples = 3 * math.ceil(image['width']/8) * math.ceil(image['height']/8)
    leaves = 1 if samples <= 512 else 2 if samples <= 4096 else 4 if samples <= 8192 else 34
    check(encoded['dc_samples'] == samples and encoded['adaptive_dc_leaves'] == leaves and
          encoded['adaptive_contexts'] == leaves + 11 and encoded['backend'] == case['backend'] and
          encoded['dc_prediction'] == case['prediction'] and encoded['flags'] == case['flags'] and
          encoded['input_width'] == image['width'] and encoded['input_height'] == image['height'] and
          encoded['effort'] == case['effort'] and math.isclose(encoded['requested_distance'], case['distance'], rel_tol=1e-6) and
          encoded['thread_count'] == manifest['threads'] and encoded['warmups_per_variant'] == warmups and
          not encoded['stage_profile_enabled'] and len(encoded['samples']) == pairs*2, 'Encode provenance mismatch')
    paths = [folder / 'streams' / f'{variant}.jxl' for variant in ('legacy', 'adaptive')]
    check(all(row['elapsed_nanoseconds'] > 0 and row['encoded_bytes'] ==
              paths[row['variant'] == 'adaptive'].stat().st_size for row in encoded['samples']), 'Invalid encoder samples')
    check(leaves != 34 or sha(paths[0]) == sha(paths[1]), 'Full tree bytes changed')
    common.command([manifest['decoder']['binary'], *paths, image['width'], image['height'], warmups, pairs if timed else 0], folder, 'decode')
    decoded = load(folder / 'decode.stdout')
    check(decoded['decoded_equal'] and decoded['decoded_finite'] and
          decoded['thread_count'] == 1 and len(decoded['samples']) == (pairs*2 if timed else 0) and
          decoded['gradient_sha256'] == sha(paths[0]) and decoded['weighted_sha256'] == sha(paths[1]),
          'Decoded pixels or provenance differ')
    record = {'case': case, 'encode': encoded, 'decode': decoded,
              'artifacts': {str(p): sha(p) for p in paths}, 'bytes': [p.stat().st_size for p in paths]}
    audit(record)
    return record


def quiet_state(timed):
    state = common.machine_state()
    if timed:
        busy = [p for p in state['top_cpu'] if p['cpu_percent'] > 80 and Path(p['command']).name != 'kernel_task']
        check(not busy, f'Not quiet enough for timing: {busy}')
    return state


def collect(args, manifest):
    added = 0
    for section in ('cases', 'integration'):
        for i, case in enumerate(manifest[section]):
            complete = args.output / section / case['id'] / 'complete.json'
            if complete.exists(): audit(load(complete)); continue
            if args.max_cases is not None and added >= args.max_cases: return
            state = quiet_state(False)
            folder = common.new_attempt(complete.parent)
            save(folder / 'machine-before.json', state)
            record = run_pair(manifest, case, folder, False)
            save(complete, record)
            added += 1
            print(f"{section} {i+1}/{len(manifest[section])}: {case['id']} {record['bytes']}", flush=True)


def timing_stats(record, kind):
    groups = defaultdict(dict)
    for row in record[kind]['samples']:
        variant = row['variant']
        if kind == 'decode': variant = {'gradient': 'legacy', 'weighted': 'adaptive'}[variant]
        groups[row['pair']][variant] = row['elapsed_nanoseconds']/1e6
    check(len(groups) == 20 and all(set(x) == {'legacy', 'adaptive'} for x in groups.values()), 'Invalid timed pairs')
    a = [v['legacy'] for v in groups.values()]; b = [v['adaptive'] for v in groups.values()]
    delta = [y-x for x,y in zip(a,b)]
    median_a, median_b = statistics.median(a), statistics.median(b)
    d = statistics.median(delta)
    slower = sum(x > 0 for x in delta)
    return {'legacy_ms': median_a, 'adaptive_ms': median_b, 'paired_delta_ms': d,
            'delta_percent': 100*d/median_a, 'slower_pairs': slower,
            'offender': d > max(.1, .02*median_a) and slower >= 16}


def timing(args, manifest):
    added = 0
    for i, case in enumerate(manifest['timing']):
        complete = args.output / 'timing' / case['id'] / 'complete.json'
        if complete.exists():
            for record in load(complete)['runs']: audit(record)
            continue
        if args.max_cases is not None and added >= args.max_cases: return
        runs = []
        for repeat in range(2):
            state = quiet_state(True)
            folder = common.new_attempt(complete.parent)
            save(folder / 'machine-before.json', state)
            record = run_pair(manifest, case, folder, True)
            save(folder / 'result.json', record)
            # Reject busy attempts while keeping all raw calls and artifacts.
            after = quiet_state(True)
            save(folder / 'machine-after.json', after)
            record['stats'] = {kind: timing_stats(record, kind) for kind in ('encode', 'decode')}
            runs.append(record)
            if not any(s['offender'] for s in record['stats'].values()): break
            if repeat == 0: time.sleep(10)
        save(complete, {'runs': runs})
        added += 1
        print(f"Timing {i+1}/{len(manifest['timing'])}: {case['id']} {runs[-1]['stats']}", flush=True)


def write_csv(path, rows):
    if not rows: return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def report(args, manifest):
    destination = args.report_dir or args.output
    destination.mkdir(parents=True, exist_ok=True)
    sizes, times, incomplete, integration = [], [], [], []
    for section in ('cases', 'integration'):
        for case in manifest[section]:
            path = args.output / section / case['id'] / 'complete.json'
            if not path.exists(): incomplete.append(str(path)); continue
            r = load(path); audit(r)
            a,b = r['bytes']
            row = {'case': case['id'], 'image': case['image']['name'], 'stratum': case['image']['stratum'],
                   'effort': case['effort'], 'distance': case['distance'], 'prediction': case['prediction'],
                   'backend': case['backend'], 'flags': case['flags'], 'dc_samples': r['encode']['dc_samples'],
                   'dc_leaves': r['encode']['adaptive_dc_leaves'], 'contexts': r['encode']['adaptive_contexts'],
                   'legacy_bytes': a, 'adaptive_bytes': b, 'delta_bytes': b-a, 'delta_percent': 100*(b-a)/a,
                   'decoded_equal': True, 'byte_identical': a == b and len(set(r['artifacts'].values())) == 1}
            (sizes if section == 'cases' else integration).append(row)
    for case in manifest['timing']:
        path = args.output / 'timing' / case['id'] / 'complete.json'
        if not path.exists(): incomplete.append(str(path)); continue
        for repeat, r in enumerate(load(path)['runs']):
            audit(r)
            for kind in ('encode','decode'):
                times.append({'case': case['id'], 'kind': kind, 'repeat': repeat, **r['stats'][kind]})
    affected = [x for x in sizes if x['dc_leaves'] < 34]
    aggregate = {p: {'legacy_bytes': sum(x['legacy_bytes'] for x in affected if x['prediction'] == p),
                     'delta_bytes': sum(x['delta_bytes'] for x in affected if x['prediction'] == p)}
                 for p in ('gradient','weighted')}
    repeat_failures = []
    for x in times:
        if x['repeat'] == 1 and x['offender'] and any(y['case'] == x['case'] and y['kind'] == x['kind'] and y['repeat'] == 0 and y['offender'] for y in times):
            repeat_failures.append(x)
    grown = [x for x in sizes + integration if x['delta_bytes'] > 0]
    summary = {'complete': not incomplete, 'case_pairs': len(sizes), 'integration_pairs': len(integration),
               'timing_rows': len(times), 'incomplete': incomplete, 'affected_aggregate': aggregate,
               'grown_cases': grown, 'repeatable_timing_regressions': repeat_failures,
               'measurement_gate_pass': not incomplete and not grown and not repeat_failures and
                     all(x['delta_bytes'] < 0 for x in aggregate.values()),
               'correctness_and_resource_tests': 'Review separate retained test logs before promotion',
               'raw_root': str(args.output.resolve())}
    write_csv(destination / 'sizes.csv', sizes)
    write_csv(destination / 'integration.csv', integration)
    write_csv(destination / 'timing.csv', times)
    save(destination / 'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('grown_cases','incomplete')}, indent=2))
    print(f'Grown cases: {len(grown)}; incomplete artifacts: {len(incomplete)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('init','collect','timing','report'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--max-cases', type=int)
    parser.add_argument('--report-dir', type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command == 'init': initialize(args); return
    manifest = load(args.output / 'manifest.json')
    if args.command != 'report': validate_manifest(manifest)
    globals()[args.command](args, manifest)


if __name__ == '__main__': main()
