#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Compare public encodes against the frozen ANS-reuse qualification binary."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import tarfile
import time

from run_histogram_attribution import snapshot, source_files
from run_hybrid_qualification import command, digest, mac_dependencies, require


def other_codec_jobs(out):
    jobs = []
    for line in command(['ps', '-axo', 'pid=,comm=']).splitlines():
        pid, name = line.strip().split(None, 1)
        if int(pid) == os.getpid():
            continue
        if '/GitHub/gjxl' in name and not name.startswith(str(out) + '/'):
            jobs.append(line.strip())
        elif Path(name).name in ('ctest', 'samply'):
            jobs.append(line.strip())
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-capture', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    require(args.repeats > 0, 'Need measured repetitions')
    repo = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    previous_path = args.baseline_capture / 'manifest.json'
    previous = json.loads(previous_path.read_text())
    require(previous['status'] == 'passed', 'Baseline qualification did not pass')
    baseline = Path(previous['binaries']['reuse']['directory']) / 'profile'
    require(digest(baseline) == previous['binaries']['reuse']['profile_sha256'],
            'Frozen baseline changed')
    files = source_files(repo)
    production_changes = [p for p, h in previous['source_files'].items()
                          if p.startswith(('src/', 'include/')) and files.get(p) != h]
    require(production_changes == ['src/codec/modular/histogram_sample_preparation.cpp'],
            'Unexpected production changes against frozen baseline')
    binaries, dependencies = {}, {}
    for label, original in [('baseline', baseline), ('reuse', args.binary)]:
        target = out / label
        subprocess.run(['cp', '-c', str(original), str(target)], check=True)
        listing, hashes = mac_dependencies(target)
        dependencies.update(hashes)
        binaries[label] = dict(sha256=digest(target), libraries=listing)
    subprocess.run(['cp', '-c', str(previous_path), str(out / 'baseline-manifest.json')], check=True)
    with tarfile.open(out / 'source.tar.gz', 'w:gz') as archive:
        for name in files:
            archive.add(repo / name, arcname=name, recursive=False)
    (out / 'source.patch').write_text(command(['git', '-C', str(repo), 'diff', '--binary', 'HEAD']))
    subprocess.run(['cp', '-c', str(args.binary.parent / 'CMakeCache.txt'),
                    str(out / 'CMakeCache.txt')], check=True)
    expected = {r['input']: r['output_sha256'] for r in previous['observations']}
    manifest = dict(schema=1, status='running', started=time.time(),
                    revision=command(['git', '-C', str(repo), 'rev-parse', 'HEAD']),
                    source_files=files, source_archive_sha256=digest(out / 'source.tar.gz'),
                    baseline_manifest_sha256=digest(previous_path), binaries=binaries,
                    dependencies=dependencies, repeats=args.repeats, participants=14,
                    inputs=[dict(path=str(p.resolve()), sha256=digest(p)) for p in args.input],
                    observations=[], interrupted=[])

    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def call(source, route, label, repetition):
        attempt = 0
        while True:
            jobs = other_codec_jobs(out)
            while jobs:
                print('Waiting for other codec work:', jobs, flush=True)
                time.sleep(10)
                jobs = other_codec_jobs(out)
            name = f'{source.stem}-{route}-{label}-{repetition}-{attempt}'
            directory = out / name
            argv = [str(out / label), str(source.resolve()), str(directory), route, '14']
            before = snapshot()
            require("'AC Power'" in before['power'], 'AC power required')
            with (out / (name + '.stdout')).open('x') as stdout, (out / (name + '.stderr')).open('x') as stderr:
                process = subprocess.Popen(argv, stdout=stdout, stderr=stderr)
                while process.poll() is None:
                    time.sleep(0.5)
                    jobs = other_codec_jobs(out)
                    if jobs and process.poll() is None:
                        process.terminate()
                        process.wait()
                        break
            if jobs:
                manifest['interrupted'].append(dict(name=name, competing_jobs=jobs))
                save()
                attempt += 1
                continue
            require(process.returncode == 0, f'Encode failed: {name}')
            after = snapshot()
            require("'AC Power'" in after['power'], 'Power changed during encode')
            result = json.loads((directory / 'result.json').read_text())
            checksum = digest(directory / 'output.jxl')
            require(result['exact_decode'] and checksum == expected[source.stem], 'Bytes/decode changed')
            require(result['stream_jobs'] == 192 and result['entropy_participants'] == 14,
                    'Parallel stream coverage changed')
            manifest['observations'].append(dict(input=source.stem, route=route, build=label,
                repetition=repetition, warmup=repetition == 0, command=argv,
                before=before, after=after, result=result, output_sha256=checksum))
            save()
            # Replace only newly generated, byte-verified duplicate files with clones.
            canonical = out / (source.stem + '.jxl')
            if canonical.exists():
                require(digest(canonical) == checksum, 'Canonical output differs')
                (directory / 'output.jxl').unlink()
                subprocess.run(['cp', '-c', str(canonical), str(directory / 'output.jxl')], check=True)
            else:
                subprocess.run(['cp', '-c', str(directory / 'output.jxl'), str(canonical)], check=True)
            print(name, f"wall={result['wall']:.3f}s training={result['stages'][2]:.3f}s exact", flush=True)
            return

    save()
    try:
        conditions = [(source, route) for source in args.input for route in ['cpu', 'metal']]
        for repetition in range(args.repeats + 1):
            for i, (source, route) in enumerate(conditions):
                order = ['baseline', 'reuse'] if (i + repetition) % 2 == 0 else ['reuse', 'baseline']
                for label in order:
                    call(source, route, label, repetition)
        require(source_files(repo) == files, 'Source changed during capture')
        for label, info in binaries.items():
            require(digest(out / label) == info['sha256'], 'Executable changed')
        for name, checksum in dependencies.items():
            require(digest(Path(name)) == checksum, 'Dependency changed')
        for item in manifest['inputs']:
            require(digest(Path(item['path'])) == item['sha256'], 'Input changed')
        manifest['timing'] = []
        for source, route in conditions:
            row = dict(input=source.stem, route=route)
            for label in binaries:
                results = [r['result'] for r in manifest['observations'] if not r['warmup']
                           and (r['input'], r['route'], r['build']) == (source.stem, route, label)]
                row[label] = {key: dict(median=statistics.median(values), minimum=min(values), maximum=max(values))
                              for key, values in [('wall', [r['wall'] for r in results]),
                                                  ('cpu', [r['cpu'] for r in results]),
                                                  ('training', [r['stages'][2] for r in results])]}
            manifest['timing'].append(row)
        manifest['status'] = 'passed'
    except BaseException as error:
        manifest['status'], manifest['error'] = 'failed', str(error)
        raise
    finally:
        manifest['finished'] = time.time()
        save()


if __name__ == '__main__':
    main()
