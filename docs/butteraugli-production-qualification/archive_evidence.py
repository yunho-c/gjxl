#!/usr/bin/env python3
"""Archive completed qualification records; never launch measurements."""
from pathlib import Path
import collections
import hashlib
import json
import shutil
import statistics
import subprocess

P = Path(__file__).resolve().parent
W = Path('/Users/yunhocho/GitHub/gjxl-butteraugli-production')
D = W / 'docs/butteraugli-production-qualification'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + '\n')


def copy(relative):
    source = P / relative
    dest = D / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    return {'source': str(source), 'sha256': sha(source)}


def main():
    assert read(P / 'qualification-state.json')['status'] == 'complete'
    assert read(P / 'production-validation-state.json')['status'] == 'complete'
    copied = {}
    roots = [
        'hardware.json', 'timing-plan.json', 'qualification-state.json',
        'known-failure.json', 'freeze-recheck.json',
        'qualification-initial-failure.json', 'qualification-launch-failure.json',
        'canonical-parity-launch-failure.log', 'executable-permission-correction.json',
        'finite-record-copy.json',
        'focused-command.json', 'focused.log',
        'integration-tests-command.json', 'integration-tests.log',
        'integration-tests.xml', 'integration-test-build.log',
        'full-suite-final-command.json', 'full-suite-final.log',
        'full-suite-final.xml', 'full-suite-final-exit.json',
        'baseline-failure-control-command.json',
        'baseline-failure-control.log', 'baseline-failure-control.xml',
        'full-suite-command.json', 'full-suite.log', 'full-suite.xml',
        'candidate-configure-command.json', 'candidate-configure.log',
        'candidate-build-command.json', 'candidate-build.log',
        'baseline-configure-command.json', 'baseline-configure.log',
        'baseline-build-command.json', 'baseline-build.log',
        'parity.py', 'compare.py', 'run_paired.py', 'run_qualification.py',
        'encode_probe.cpp', 'verify_evidence.py', 'archive_evidence.py', 'write_report.py',
        'canonical-parity-command.json', 'canonical-parity.log',
        'canonical-parity-v2/identity.json', 'canonical-parity-v2/summary.json',
        'canonical-parity-v2/finite-pixels.json',
    ]
    roots.extend(f.name for f in P.iterdir() if f.is_file()
                 and f.suffix in ('.json', '.log', '.cpp', '.py')
                 and f.name.startswith(('metadata-', 'finish_validation',
                                        'production-validation', 'full-suite-production',
                                        'final-freeze', 'ac-', 'full-suite-release')))
    roots.append('full-suite-production.xml')
    roots.append('full-suite-release.xml')
    roots.append('metadata-candidate.xml')
    for label in ('baseline', 'candidate'):
        roots.extend(f'{label}/{name}' for name in
                     ('identity.json', 'build-command.json', 'build.log'))
    for relative in roots:
        copied[relative] = copy(relative)

    audit = {'cohorts': [], 'external_root': str(P)}
    observed_kernels = {'baseline': set(), 'candidate': set()}
    raw_file_checks = 0
    for cohort in read(P / 'timing-plan.json')['cohorts']:
        name = cohort['name']
        folder = P / name
        identity, pairs = read(folder / 'identity.json'), read(folder / 'pairs.json')
        assert read(folder / 'state.json')['status'] == 'complete'
        records = []
        for row in pairs:
            for side, result in row['sides'].items():
                case = folder / row['job'] / f'{row["pair"]:02d}-{side}'
                assert read(case / 'complete.json') == result
                for filename, digest in result['files'].items():
                    assert sha(case / filename) == digest, case / filename
                    raw_file_checks += 1
                assert sha(case / 'reference.jxl') == result['codestream_sha256']
                samples, execution = read(case / 'samples.json'), read(case / 'execution.json')
                assert samples['byte_equal'] and samples['summary_equal']
                assert execution['returncode'] == 0 and not execution['interference']
                assert statistics.median(samples['samples']) / 1e6 == result['metrics']['complete_call_ms']
                if identity['profile']:
                    profile = read(case / 'gpu.json')
                    totals = collections.defaultdict(list)
                    for sample in profile['workloads'][0]['samples']:
                        stages = [s for sub in sample['submissions'] for s in sub['stages']]
                        for stage in stages:
                            observed_kernels[side].update(d['kernel_id'] for d in stage['dispatches'])
                            assert stage['timestamp_valid']
                            assert stage['gpu_nanoseconds'] == stage['end_timestamp'] - stage['begin_timestamp'] >= 0
                        intervals = sorted((s['begin_timestamp'], s['end_timestamp']) for s in stages)
                        assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
                        totals['gpu_total_ms'].append(sum(s['gpu_nanoseconds'] for s in stages) / 1e6)
                        totals['butteraugli_all_ms'].append(sum(s['gpu_nanoseconds'] for s in stages
                            if s['stage_id'].startswith(('butteraugli.', 'frontend.prepare_aq.reference.'))) / 1e6)
                    for metric, values in totals.items():
                        assert statistics.median(values) == result['metrics'][metric]
                records.append({'job': row['job'], 'pair': row['pair'], 'side': side,
                                'files': {n: (case / n).read_text() for n in
                                          ('samples.json', 'execution.json', 'command.json')}})
            assert row['sides']['baseline']['codestream_sha256'] == row['sides']['candidate']['codestream_sha256']
        for filename in ('identity.json', 'pairs.json', 'summary.json', 'state.json', 'protocol.py'):
            relative = f'{name}/{filename}'
            copied[relative] = copy(relative)
        save(D / name / 'case-records.json', records)
        audit['cohorts'].append({'cohort': name, 'pairs': len(pairs),
                                'summary_sha256': sha(folder / 'summary.json'),
                                'profile': identity['profile']})
    audit['raw_case_file_hash_checks'] = raw_file_checks
    audit['candidate_only_profiled_kernel_ids'] = sorted(observed_kernels['candidate'] - observed_kernels['baseline'])
    assert {'gjxl_butteraugli_low_medium_shared_f32', 'gjxl_butteraugli_low_medium_packed_dc_f32'} <= observed_kernels['candidate']
    assert 'gjxl_butteraugli_low_medium_shared_f32' not in observed_kernels['baseline']
    save(D / 'artifact-audit.json', audit)
    save(D / 'archive-copy-provenance.json', copied)
    base = read(P / 'initial-state.json')['base']
    revision = subprocess.check_output(['git', '-C', str(W), 'rev-parse', 'HEAD'], text=True).strip()
    changed = subprocess.check_output(['git', '-C', str(W), 'diff', '--name-only', base, revision], text=True).splitlines()
    assert changed and all(not x.startswith('docs/') for x in changed)
    save(D / 'git-provenance.json', {
        'base': base, 'source_commit': revision,
        'source_files': {name: sha(W / name) for name in changed},
        'study_branch': 'perf/butteraugli-traffic',
        'study_head': 'c7549eaed63e3a84d0fd86a4b113d36e64764aeb',
        'study_bundle': 'db8a4d67d554bcb04b651243c3660c42ec245bb8',
        'excluded_refinement': 'ef5f29003c52ec788d339f00b6f2e50a6f1818e7',
    })
    print('Archived', len(copied), 'copied files and', sum(c['pairs'] for c in audit['cohorts']), 'pairs')


if __name__ == '__main__':
    main()
