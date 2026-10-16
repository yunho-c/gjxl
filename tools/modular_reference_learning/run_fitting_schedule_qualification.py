#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Interleaved public encoder qualification for fitting scheduling and integration."""
import argparse
import json
import shutil
import statistics
import subprocess
import tarfile
import time
from pathlib import Path

from run_histogram_attribution import snapshot, source_files, signature, validate_profile
from run_hybrid_qualification import command, digest, mac_dependencies, require
from run_parallel_profile import validate
from report_parallel_profile import schedule_metrics


def logical_fitting_signature(profile):
    value = signature(profile)
    physical = {'property_row_visits', 'joint_histogram_row_visits',
                'fused_property_row_visits', 'skipped_constant_properties'}
    value['rows'] = [(bucket, {k:v for k,v in work.items() if k not in physical})
                     for bucket,work in value['rows']]
    return value


def fitting_summary(profile):
    work = {}
    for bucket in profile['by_rows']:
        for name,value in bucket['work'].items():
            work[name] = work.get(name, 0) + value
    return dict(schema=profile['schema'], work=work,
                physical_property_histogram_passes=work['property_row_visits'] +
                    work['joint_histogram_row_visits'] - work.get('fused_property_row_visits', 0),
                work_seconds=profile['fit_seconds'], phase_work_seconds=profile['stages'])


def require_no_competing_work(state):
    # Boundary snapshots already retain these observations. Refuse a known
    # overlapping GJXL experiment/build instead of treating it as desktop noise.
    competing = []
    for row in state.get('top_processes', []):
        fields = row.split(None, 2)
        if len(fields) != 3:
            continue
        _, cpu, path = fields
        name = Path(path).name
        gjxl_path = any(part == 'gjxl' or part.startswith('gjxl-') for part in Path(path).parts)
        if float(cpu) > 1 and (gjxl_path or name.startswith(('gjxl_', 'clang'))):
            competing.append(row)
    require(not competing, 'Concurrent GJXL/compiler work; wait before restarting the complete matrix: ' + repr(competing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-capture', type=Path, required=True)
    parser.add_argument('--baseline-variant', choices=['before', 'after'],
                        help='Select an executable from a completed before/after capture')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--validation-dir', type=Path, help='Completed test logs to archive')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--diagnostic-mode', choices=['schedule', 'fit-on'], default='schedule',
                        help='Separate captures; fit-on also verifies logical fitter-work parity')
    args = parser.parse_args()
    require(args.repeats > 0, 'Positive measured repetition count required')
    repo = Path(__file__).resolve().parents[2]
    baseline = args.baseline_capture.resolve()
    prior = json.loads((baseline/'manifest.json').read_text())
    require(prior['status'] == 'passed', 'Baseline capture is incomplete')
    checksum = prior['binary_sha256']
    if isinstance(checksum, dict):
        require(args.baseline_variant is not None, 'Select a before/after baseline variant')
        baseline_binary = baseline/args.baseline_variant
        checksum = checksum[args.baseline_variant]
    else:
        require(args.baseline_variant is None, 'Single-executable baseline has no variants')
        baseline_binary = baseline/'hybrid_profile'
    require(digest(baseline_binary) == checksum, 'Baseline executable changed')
    for name, checksum in prior['dependency_sha256'].items():
        require(digest(Path(name)) == checksum, 'Baseline dependency changed')
    expected = {}
    for line in (baseline/'results.jsonl').read_text().splitlines():
        r = json.loads(line)
        require(expected.setdefault(r['input'], r['output_sha256']) == r['output_sha256'],
                'Conflicting baseline output hashes')
    inputs = [p.resolve() for p in args.input]
    require(len({p.stem for p in inputs}) == len(inputs), 'Duplicate input names')
    for p in inputs:
        old = next((v for v in prior['inputs'] if v['name'] == p.stem), None)
        require(old and old['sha256'] == digest(p), 'Input differs from baseline')
    initial = snapshot()
    require("'AC Power'" in initial['power'], 'AC power required')
    require_no_competing_work(initial)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.validation_dir:
        shutil.copytree(args.validation_dir, out/'validation')
    binaries = {}
    for name, source in [('before', baseline_binary), ('after', args.binary.resolve())]:
        binaries[name] = out/name
        subprocess.run(['cp', '-c', str(source), str(binaries[name])], check=True)
    shutil.copy2(baseline/'manifest.json', out/'baseline-manifest.json')
    for name in ['CMakeCache.txt', 'compile_commands.json']:
        shutil.copy2(args.binary.parent/name, out/name)
    files = source_files(repo)
    with tarfile.open(out/'source.tar.gz', 'w:gz') as archive:
        for name in files:
            archive.add(repo/name, arcname=name, recursive=False)
    (out/'source.patch').write_text(command(['git', '-C', str(repo), 'diff', '--binary', 'HEAD'])+'\n')
    listing, dependencies = mac_dependencies(binaries['after'])
    require(dependencies == prior['dependency_sha256'], 'Before/after dependencies differ')
    manifest = dict(schema=1, status='running', started_unix=time.time(),
                    revision=command(['git', '-C', str(repo), 'rev-parse', 'HEAD']),
                    source_status=command(['git', '-C', str(repo), 'status', '--short']),
                    source_files_sha256=files, source_archive_sha256=digest(out/'source.tar.gz'),
                    binary_sha256={k:digest(p) for k,p in binaries.items()},
                    baseline_directory=str(baseline), baseline_manifest_sha256=digest(out/'baseline-manifest.json'),
                    baseline_variant=args.baseline_variant, diagnostic_mode=args.diagnostic_mode,
                    linked_libraries=listing, dependency_sha256=dependencies,
                    submodules=command(['git', '-C', str(repo), 'submodule', 'status']),
                    repeats=args.repeats, environment_start=initial,
                    inputs=[dict(path=str(p),name=p.stem,sha256=digest(p)) for p in inputs],
                    scope='Explicit public CPU and Metal, 14 CPU participants, default learned policy. '
                          'Rotating fresh-process before/after controls, one discarded call per condition. '
                          f'Top-level timers only for medians; separate {args.diagnostic_mode} captures. '
                          'Exact historical bytes and independent decoding on every call. '
                          'Input/output IO, backend setup and decoding excluded from encoder timer.',
                    completed_calls=0)
    rows, work, routing, logical_fits, physical_fits = [], {}, {}, {}, {}
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    def call(source, revision, route, repetition, mode='fit-off'):
        label = f'{source.stem}-{revision}-{route}-{mode}-{repetition}'
        directory = out/label
        argv = [str(binaries[revision]), str(source), str(directory), route, 'auto', 'best', mode,
                '14', '14', '14', '14', '14', 'public']
        before = snapshot()
        require("'AC Power'" in before['power'], 'Power changed before encode')
        require_no_competing_work(before)
        started = time.time()
        with (out/(label+'.stdout')).open('x') as stdout, (out/(label+'.stderr')).open('x') as stderr:
            subprocess.run(argv, stdout=stdout, stderr=stderr, check=True)
        finished, after = time.time(), snapshot()
        result = json.loads((directory/'result.json').read_text())
        checksum = digest(directory/'output.jxl')
        row = dict(input=source.stem, revision=revision, route=route, repetition=repetition, mode=mode,
                   warmup=repetition == 0, diagnostic=mode != 'fit-off', directory=label, command=argv,
                   started_unix=started, finished_unix=finished, output_sha256=checksum,
                   environment_before=before, environment_after=after, result=result)
        with (out/'results.jsonl').open('a') as ledger:
            ledger.write(json.dumps(row)+'\n')
        rows.append(row)
        manifest['completed_calls'] = len(rows)
        save()
        require("'AC Power'" in after['power'], 'Power changed during encode; observation retained')
        require_no_competing_work(after)
        validate(result)
        require(checksum == expected[source.stem], 'Bytes differ from qualified baseline')
        require(work.setdefault(source.stem, result['work']) == result['work'], 'Learning decisions/work changed')
        key = (source.stem, revision, route)
        require(routing.setdefault(key, result['routing']) == result['routing'],
                'Feature routing changed within a repeated condition')
        require(result['memory']['peak_feature_samples'] <= 8*1024*1024, 'Feature cohort exceeded cap')
        require(not result['routing'].get('feature_memory_limited_cohorts', 0), 'Feature cohort memory pressure')
        require(sum(result['profile_wall']) <= result['total']['wall']+1e-4, 'Elapsed stages overlap')
        require(sum(result['detail_wall'].values()) <= result['total']['wall']+1e-4, 'Detail intervals overlap')
        if result['histogram_fitting']['enabled']:
            validate_profile(result)
            profile = result['histogram_fitting']
            logical = logical_fitting_signature(profile)
            require(logical_fits.setdefault(source.stem, logical) == logical,
                    'Logical fitter work or canonical fit records changed')
            physical = signature(profile)
            require(physical_fits.setdefault((source.stem, revision), physical) == physical,
                    'CPU/Metal fitter work differs within the same implementation')
        canonical = out/(source.stem+'.jxl')
        if not canonical.exists():
            subprocess.run(['cp', '-c', str(directory/'output.jxl'), str(canonical)], check=True)
        else:
            # Newly generated outputs have already passed exact decode and hash checks.
            (directory/'output.jxl').unlink()
            subprocess.run(['cp', '-c', str(canonical), str(directory/'output.jxl')], check=True)
        print(label, f"total={result['total']['wall']:.3f}s learning={result['detail_wall']['learning']:.3f}s "
                    f"waves={result['learning']['waves']} exact", flush=True)
    try:
        variants = [('before','cpu'), ('after','cpu'), ('before','auto'), ('after','auto')]
        for repetition in range(args.repeats+1):
            for index, source in enumerate(inputs):
                offset = (repetition+index) % len(variants)
                for revision, route in variants[offset:]+variants[:offset]:
                    call(source, revision, route, repetition)
        for index, source in enumerate(inputs):
            offset = index % len(variants)
            for revision, route in variants[offset:]+variants[:offset]:
                call(source, revision, route, 1, args.diagnostic_mode)
        require(source_files(repo) == files, 'Build/tool source changed during capture')
        for k,p in binaries.items():
            require(digest(p) == manifest['binary_sha256'][k], 'Frozen binary changed')
        for item in manifest['inputs']:
            require(digest(Path(item['path'])) == item['sha256'], 'Input changed')
        for name,checksum in dependencies.items():
            require(digest(Path(name)) == checksum, 'Dependency changed')
        summary = []
        for source in inputs:
            for revision,route in variants:
                values = [r['result'] for r in rows if r['input']==source.stem and r['revision']==revision
                          and r['route']==route and not r['warmup'] and not r['diagnostic']]
                median = lambda f:statistics.median(f(r) for r in values)
                diagnostic = next(r['result'] for r in rows if r['input']==source.stem and
                                  r['revision']==revision and r['route']==route and r['diagnostic'])
                item = dict(input=source.stem, revision=revision, route=route,
                                    seconds=median(lambda r:r['total']['wall']),
                                    seconds_range=[min(r['total']['wall'] for r in values), max(r['total']['wall'] for r in values)],
                                    cpu_seconds=median(lambda r:r['total']['cpu']),
                                    stages=[median(lambda r:r['profile_wall'][i]) for i in range(9)],
                                    details={k:median(lambda r:r['detail_wall'][k]) for k in values[0]['detail_wall']},
                                    memory={k:max(r['memory'][k] for r in values) for k in values[0]['memory']},
                                    work=values[0]['work'], routing=values[0]['routing'],
                                    learning=values[0]['learning'], schedule=schedule_metrics(diagnostic),
                                    bytes=values[0]['bytes'], output_sha256=expected[source.stem])
                if diagnostic['histogram_fitting']['enabled']:
                    item['fitting'] = fitting_summary(diagnostic['histogram_fitting'])
                summary.append(item)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        manifest['status'] = 'passed'
    except BaseException as error:
        manifest['status'],manifest['error'] = 'failed',str(error)
        raise
    finally:
        manifest['finished_unix'] = time.time()
        manifest['environment_finish'] = snapshot()
        manifest['artifacts_sha256'] = {str(p.relative_to(out)):digest(p) for p in out.rglob('*')
                                       if p.is_file() and p.name!='manifest.json'}
        save()


if __name__=='__main__':
    main()
