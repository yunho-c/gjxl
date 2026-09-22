"""Read and validate the frozen Q80 study; never run an encoder."""
import collections
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
STUDY = Path('/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/gjxl-stages-q80-20260921')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def write_csv(name, rows):
    with (HERE / name).open('w') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    cfg = json.loads((STUDY / 'config.json').read_text())
    spec = importlib.util.spec_from_file_location('saved_validation', STUDY / 'code/validate.py')
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    for path, expected in cfg['frozen_sha256'].items():
        assert sha(STUDY / path) == expected, path
    for im in cfg['images']:
        assert sha(im['input_path']) == im['input_sha256'], im['image_id']
    cases = {}
    for job in cfg['jobs']:
        directory = STUDY / 'cases' / job['case_id']
        completion = json.loads((directory / 'complete.json').read_text())
        attempt = directory / completion['attempt']
        result = validator.validate(attempt, job, cfg)
        for key in ('gpu_sha256', 'paired_sha256', 'reference_sha256'):
            assert result[key] == completion['summary'][key], (job['case_id'], key)
        command_path = attempt / 'command.json'
        assert sha(command_path) == completion['command_sha256']
        command = json.loads(command_path.read_text())
        argv = command['argv']
        assert command['returncode'] == 0 and not command['interference']
        assert argv.count('--input') == 1 and argv[argv.index('--input') + 1] == job['input_path']
        cases[job['case_id']] = result

    rows = list(csv.DictReader((STUDY / 'plots/gjxl-stage-samples.csv').open()))
    raw = collections.defaultdict(dict)
    for row in rows:
        if row['kind'] != 'flat':
            continue
        key = (row['image_id'], int(row['effort']), int(row['sample_index']))
        assert row['stage'] not in raw[key]
        raw[key][row['stage']] = float(row['ms'])
    assert len(raw) == 360
    overhead = list(csv.DictReader((STUDY / 'plots/gjxl-stage-overhead.csv').open()))
    pairs = {(r['image_id'], int(r['effort']), int(r['sample_index'])): r for r in overhead}
    assert raw.keys() == pairs.keys()
    for key, stages in raw.items():
        assert all(v >= 0 for v in stages.values())
        assert math.isclose(sum(stages.values()), float(pairs[key]['profiled_ms']), abs_tol=1e-7)

    image_class = {im['image_id']: im['resolution_class'] for im in cfg['images']}
    cohorts = collections.defaultdict(list)
    for key, stages in raw.items():
        cohorts[(image_class[key[0]], key[1])].append((key, stages))
    summary, scenarios = [], []
    serializer = {'Validation', 'DC tokenization', 'AC tokenization', 'Entropy optimization',
                  'Section writing', 'Assembly', 'Other serializer'}
    for (res, effort), records in sorted(cohorts.items()):
        # All images have exactly six samples; pooling within this cohort is equal-image weighting.
        stages = {s: statistics.mean(r.get(s, 0) for _, r in records)
                  for s in set().union(*(set(r) for _, r in records))}
        total = sum(stages.values())
        values = {
            'all_gpu': sum(v for k, v in stages.items() if k.startswith('GPU:')),
            'perceptual': stages.get('GPU: Butteraugli comparison', 0) + stages.get('GPU: Reference features', 0),
            'comparison': stages.get('GPU: Butteraugli comparison', 0),
            'ac_search': stages.get('GPU: AC strategy search', 0),
            'serializer': sum(stages.get(s, 0) for s in serializer),
            'entropy': stages.get('Entropy optimization', 0),
            'orchestration': stages.get('Pipeline orchestration / gaps', 0),
        }
        row = {'resolution_class': res, 'effort': effort, 'sample_count': len(records),
               'ordinary_ms': statistics.mean(float(pairs[k]['ordinary_ms']) for k, _ in records),
               'profiled_ms': total}
        for k, v in values.items():
            row[k + '_ms'] = v
            row[k + '_percent'] = 100 * v / total
            f = v / total
            scenarios.append({'resolution_class': res, 'effort': effort, 'target': k,
                              'profiled_fraction': f, 'speedup_if_target_2x': 1 / (1 - f / 2),
                              'latency_reduction_if_target_2x_percent': 50 * f,
                              'speedup_if_target_free': 1 / (1 - f),
                              'interpretation': 'Amdahl sensitivity, not an achieved or predicted gain'})
        summary.append(row)
    write_csv('summary.csv', summary)
    write_csv('amdahl-scenarios.csv', scenarios)
    fine = collections.defaultdict(float)
    for r in csv.DictReader((STUDY / 'plots/gjxl-stage-gpu-stages.csv').open()):
        fine[(r['resolution_class'], int(r['effort']), r['stage'])] += float(r['ms'])
    fine_rows = []
    for (res, effort, stage), total in sorted(fine.items()):
        n = len(cohorts[res, effort])
        fine_rows.append({'resolution_class': res, 'effort': effort, 'stage': stage, 'mean_ms': total/n})
    write_csv('gpu-stage-means.csv', fine_rows)
    manifest = {'source': str(STUDY), 'source_revision': cfg['source_revision'],
                'config_sha256': sha(STUDY / 'config.json'), 'validated_cases': len(cases),
                'paired_samples': len(raw), 'frozen_hashes_valid': True, 'input_hashes_valid': True,
                'completion_hashes_valid': True, 'additive_partitions_valid': True,
                'case_results': cases,
                'analysis_inputs': {str(p): sha(p) for p in (STUDY / 'plots').glob('*.csv')}}
    (HERE / 'saved-data-validation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Validated 60 cases, 360 pairs, input/frozen/capture hashes and additive partitions.')
    for r in summary:
        if r['resolution_class'] in ('24mp', '48mp', 'clic_1_8_to_3_4mp') and r['effort'] in (1, 7, 10):
            print(r['resolution_class'], r['effort'],
                  'ordinary/profiled', round(r['ordinary_ms'], 2), round(r['profiled_ms'], 2),
                  {k: round(r[k + '_percent'], 1) for k in values})

if __name__ == '__main__':
    main()
