#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Paired public CPU/Metal ANS allocation qualification with frozen executables."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from run_histogram_attribution import snapshot, source_files
from run_hybrid_qualification import command, digest, mac_dependencies, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True,
                        help='Frozen baseline directory containing profile, probe and source.json')
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--samply-analyzer', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    require(args.repeats > 0, 'Need measured repetitions')
    repo = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidate = out / 'candidate'
    candidate.mkdir()
    baseline = args.baseline.resolve()
    baseline_source = json.loads((baseline / 'source.json').read_text())
    for name, checksum in baseline_source['binaries'].items():
        require(digest(baseline / name) == checksum, 'Baseline binary changed')
    for original, name in [('gjxl_ans_allocation_profile', 'profile'), ('gjxl_ans_search_probe', 'probe')]:
        subprocess.run(['cp', '-c', str(args.build / original), str(candidate / name)], check=True)
    subprocess.run(['dsymutil', str(candidate / 'profile'), '-o', str(candidate / 'profile.dSYM')], check=True)
    files = source_files(repo)
    with tarfile.open(candidate / 'source.tar.gz', 'w:gz') as archive:
        for name in files:
            archive.add(repo / name, arcname=name, recursive=False)
    (candidate / 'source.patch').write_text(command(['git', '-C', str(repo), 'diff', '--binary', 'HEAD']))
    for name in ['CMakeCache.txt', 'compile_commands.json']:
        if (args.build / name).exists():
            shutil.copy2(args.build / name, candidate / name)
    # Freeze the same attribution method as the integrated profiling effort.
    analyzer = out / 'analyzer'
    analyzer.mkdir()
    for name in ['analyze_parallel_samply.py', 'analyze_fitting_samply.py', 'run_hybrid_qualification.py']:
        shutil.copy2(args.samply_analyzer.parent / name, analyzer / name)
    builds = {'baseline': baseline, 'reuse': candidate}
    dependencies = {}
    binaries = {}
    for label, directory in builds.items():
        listing, hashes = mac_dependencies(directory / 'profile')
        dependencies.update(hashes)
        binaries[label] = dict(directory=str(directory), profile_sha256=digest(directory / 'profile'),
                               probe_sha256=digest(directory / 'probe'), libraries=listing)
    expected = {row['input']: row['output_sha256'] for row in
                json.loads((repo / 'docs/parallel-tree-fitting-results.json').read_text())['runs']}
    manifest = dict(schema=1, status='running', started=time.time(),
                    revision=command(['git', '-C', str(repo), 'rev-parse', 'HEAD']),
                    source_status=command(['git', '-C', str(repo), 'status', '--short']),
                    source_files=files, source_archive_sha256=digest(candidate / 'source.tar.gz'),
                    baseline_source=baseline_source, binaries=binaries, dependencies=dependencies,
                    inputs=[dict(path=str(p.resolve()), name=p.stem, sha256=digest(p)) for p in args.input],
                    repeats=args.repeats, samply_version=command(['samply', '--version']),
                    environment_start=snapshot(), observations=[])

    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def call(source, route, participants, label, repetition, sampled=False):
        name = f'{source.stem}-{route}-{participants}-{label}-{repetition}' + ('-samply' if sampled else '')
        directory = out / name
        argv = [str(builds[label] / 'profile'), str(source.resolve()), str(directory), route, str(participants)]
        if sampled:
            argv = ['samply', 'record', '--save-only', '--unstable-presymbolicate', '--rate', '1000',
                    '--output', str(out / (name + '.json.gz')), '--', *argv]
        before = snapshot()
        require("'AC Power'" in before['power'], 'AC power required')
        started = time.time()
        with (out / (name + '.stdout')).open('x') as stdout, (out / (name + '.stderr')).open('x') as stderr:
            subprocess.run(argv, stdout=stdout, stderr=stderr, check=True)
        after = snapshot()
        result = json.loads((directory / 'result.json').read_text())
        checksum = digest(directory / 'output.jxl')
        row = dict(input=source.stem, route=route, participants=participants, build=label,
                   repetition=repetition, warmup=repetition == 0, sampled=sampled, command=argv,
                   started=started, finished=time.time(), before=before, after=after,
                   result=result, output_sha256=checksum, directory=name,
                   stream_wall=result['stages'][8] if participants > 1
                     else result['stages'][4] + result['stages'][5])
        manifest['observations'].append(row)
        save()
        require("'AC Power'" in after['power'], 'Power changed during call')
        require(checksum == expected[source.stem] and result['exact_decode'], 'Qualified bytes/decode differ')
        if participants > 1:
            require(result['stream_jobs'] == 192 and result['entropy_participants'] == participants,
                    'Unexpected entropy job/concurrency coverage')
        else:
            require(result['stream_jobs'] == result['entropy_participants'] == 0,
                    'Serial control launched entropy workers')
        canonical = out / (source.stem + '.jxl')
        if not canonical.exists():
            subprocess.run(['cp', '-c', str(directory / 'output.jxl'), str(canonical)], check=True)
        else:
            # Only replace this new, independently validated duplicate output.
            (directory / 'output.jxl').unlink()
            subprocess.run(['cp', '-c', str(canonical), str(directory / 'output.jxl')], check=True)
        print(name, f"wall={result['wall']:.3f}s stream={row['stream_wall']:.3f}s exact", flush=True)
        if sampled:
            subprocess.run(['python3', str(analyzer / 'analyze_parallel_samply.py'),
                            str(out / (name + '.json.gz')), '--binary', str(builds[label] / 'profile'),
                            '--output', str(out / (name + '-analysis.json'))], check=True,
                           stdout=subprocess.DEVNULL)

    save()
    try:
        probes = {}
        for label, directory in builds.items():
            destination = out / (label + '-probe')
            subprocess.run([str(directory / 'probe'), str(destination)], check=True)
            probes[label] = [json.loads(line) for line in (destination / 'results.jsonl').read_text().splitlines()]
        require(len(probes['baseline']) == len(probes['reuse']) == 42, 'Probe coverage differs')
        for a, b in zip(probes['baseline'], probes['reuse']):
            require(all(a[k] == b[k] for k in ['case', 'bits', 'configs', 'histograms']), 'Search decisions differ')
            name = a['case'] + '.bin'
            require(digest(out / 'baseline-probe' / name) == digest(out / 'reuse-probe' / name), 'Model bytes differ')
        manifest['probes'] = probes
        conditions = [(source, route, 14) for source in args.input for route in ['cpu', 'metal']]
        # One full-resolution serial control, with the same default model search.
        conditions.append((args.input[0], 'cpu', 1))
        for repetition in range(args.repeats + 1):
            for index, (source, route, participants) in enumerate(conditions):
                order = ['baseline', 'reuse'] if (index + repetition) % 2 == 0 else ['reuse', 'baseline']
                for label in order:
                    call(source, route, participants, label, repetition)
        for source in args.input:
            for route in ['cpu', 'metal']:
                for label in builds:
                    call(source, route, 14, label, 1, sampled=True)
        require(source_files(repo) == files, 'Build/tool source changed during capture')
        for label, directory in builds.items():
            for name in ['profile', 'probe']:
                require(digest(directory / name) == binaries[label][name + '_sha256'], 'Frozen executable changed')
        for name, checksum in dependencies.items():
            require(digest(Path(name)) == checksum, 'Dependency changed')
        for item in manifest['inputs']:
            require(digest(Path(item['path'])) == item['sha256'], 'Input changed')
        manifest['status'] = 'passed'
    except BaseException as error:
        manifest['status'], manifest['error'] = 'failed', str(error)
        raise
    finally:
        manifest['finished'] = time.time()
        manifest['environment_finish'] = snapshot()
        manifest['artifacts'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*')
                                 if p.is_file() and p.name != 'manifest.json'}
        save()


if __name__ == '__main__':
    main()
