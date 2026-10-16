#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Public CPU/Metal timing, learning-wave diagnostics and all-thread Samply capture."""
import argparse
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from run_histogram_attribution import snapshot, source_files, signature
from run_hybrid_qualification import command, digest, mac_dependencies, require


def validate(result):
    require(result['schema'] == 6 and result['public_api'] and result['exact_decode'], 'Wrong route')
    require(result['learning']['peak_participants'] == 14, 'Learning participant mismatch')
    require(not result['learning']['memory_limited_waves'], 'Learning memory pressure')
    require(not result['selection']['memory_limited_waves'], 'Selection memory pressure')
    require(not result['entropy']['memory_reductions'] and
            not result['tokenization']['memory_reductions'], 'Encoding memory pressure')
    trace = result['learning_schedule']
    require(not trace['dropped'], 'Schedule records lost')
    if trace['enabled']:
        jobs = trace['jobs']
        require(sum(j['fits'] for j in jobs) == result['work']['fits'] and
                sum(j['calls'] for j in jobs) == result['work']['learning_calls'], 'Unexpected local job coverage')
        cursor = 0
        for w in trace['waves']:
            require(w['first_job'] == cursor and 0 < w['participants'] <= 14, 'Invalid wave range')
            require(w['start'] <= w['dispatch'] <= w['joined'] <= w['finish'], 'Invalid wave interval')
            for j in jobs[cursor:cursor+w['jobs']]:
                require(w['dispatch'] <= j['start'] <= j['finish'] <= w['joined'], 'Job escaped wave')
                require(j['worker'] < w['participants'], 'Job exceeded participant grant')
                require(j['preparation'] + j['fitting'] <= j['finish']-j['start']+1e-6, 'Job phase overlap')
            cursor += w['jobs']
        require(cursor == len(jobs), 'Wave coverage differs')
        require(len({j['stream'] for j in jobs}) == len(jobs), 'Stream repeated')
    p = result['histogram_fitting']
    if p['enabled']:
        require(p['calls'] == p['successes'] == result['work']['fits'] == len(p['fits']) and
                not p['records_dropped'], 'Fitter coverage differs')
        require(sum(p['stages'].values()) <= p['fit_seconds']+1e-6, 'Fit phase overlap')
        require(p['fit_seconds'] <= result['detail_work']['fitting']+1e-6, 'Fit work exceeds parent')
        require(sum(b['work']['nodes'] for b in p['by_rows']) == sum(f['nodes'] for f in p['fits']),
                'Fitter node conservation differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--samply', default='samply')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = out / 'hybrid_profile'
    subprocess.run(['cp', '-c', str(args.binary.resolve()), str(binary)], check=True)
    subprocess.run(['dsymutil', str(binary), '-o', str(binary)+'.dSYM'], check=True)
    files = source_files(repo)
    with tarfile.open(out/'source.tar.gz', 'w:gz') as archive:
        for name in files:
            archive.add(repo/name, arcname=name, recursive=False)
    (out/'source.patch').write_text(command(['git', '-C', str(repo), 'diff', '--binary', 'HEAD']))
    for name in ['CMakeCache.txt', 'compile_commands.json']:
        shutil.copy2(args.binary.parent/name, out/name)
    prior = json.loads((repo/'docs/parallel-tree-fitting-results.json').read_text())
    expected = {r['input']: r['output_sha256'] for r in prior['runs']}
    listing, dependencies = mac_dependencies(binary)
    manifest = dict(schema=1, status='running', started_unix=time.time(),
                    revision=command(['git', '-C', str(repo), 'rev-parse', 'HEAD']),
                    source_status=command(['git', '-C', str(repo), 'status', '--short']),
                    source_files_sha256=files, binary_sha256=digest(binary),
                    source_archive_sha256=digest(out/'source.tar.gz'),
                    submodules=command(['git', '-C', str(repo), 'submodule', 'status']),
                    linked_libraries=listing, dependency_sha256=dependencies,
                    environment_start=snapshot(), repeats=args.repeats,
                    samply_version=command([args.samply, '--version']),
                    inputs=[dict(path=str(p.resolve()), name=p.stem, sha256=digest(p)) for p in args.input],
                    completed_calls=0)
    records, signatures = [], {}
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    def call(source, route, mode, repetition, sampled=False):
        label = f'{source.stem}-{route}-{mode}-{repetition}' + ('-samply' if sampled else '')
        directory = out/label
        argv = [str(binary), str(source.resolve()), str(directory), route, 'auto', 'best', mode,
                '14', '14', '14', '14', '14', 'public']
        if sampled:
            argv = [args.samply, 'record', '--save-only', '--unstable-presymbolicate', '--rate', '1000',
                    '--output', str(out/(label+'.json.gz')), '--', *argv]
        before = snapshot()
        require("'AC Power'" in before['power'], 'AC power required; capture stopped')
        started = time.time()
        with (out/(label+'.stdout')).open('x') as stdout, (out/(label+'.stderr')).open('x') as stderr:
            subprocess.run(argv, stdout=stdout, stderr=stderr, check=True)
        after = snapshot()
        result = json.loads((directory/'result.json').read_text())
        checksum = digest(directory/'output.jxl')
        row = dict(input=source.stem, route=route, mode=mode, repetition=repetition,
                   sampled=sampled, warmup=repetition == 0, command=argv, started_unix=started,
                   finished_unix=time.time(), environment_before=before, environment_after=after,
                   output_sha256=checksum, directory=label, result=result)
        with (out/'results.jsonl').open('a') as ledger:
            ledger.write(json.dumps(row)+'\n')
        records.append(row)
        manifest['completed_calls'] = len(records)
        save()
        require("'AC Power'" in after['power'], 'Power changed; retained observation excluded by failed capture')
        require(checksum == expected[source.stem], 'Complete bytes differ from qualified parent')
        validate(result)
        if result['histogram_fitting']['enabled']:
            sig = signature(result['histogram_fitting'])
            require(signatures.setdefault(source.stem, sig) == sig, 'CPU/Metal fitting work differs')
        canonical = out/(source.stem+'.jxl')
        if not canonical.exists():
            subprocess.run(['cp', '-c', str(directory/'output.jxl'), str(canonical)], check=True)
        else:
            # These are newly generated, already decoded/hash-verified outputs.
            (directory/'output.jxl').unlink()
            subprocess.run(['cp', '-c', str(canonical), str(directory/'output.jxl')], check=True)
        print(label, f"wall={result['total']['wall']:.3f}s learning={result['detail_wall']['learning']:.3f}s exact", flush=True)
    try:
        variants = [(route, mode) for route in ['cpu', 'auto'] for mode in ['fit-off', 'schedule', 'fit-on']]
        # Separate fresh processes, one discarded process per configuration,
        # followed by measured calls. Backend initialization is outside encoding.
        for repetition in range(args.repeats+1):
            for image_index, source in enumerate(args.input):
                order = (repetition + image_index*2) % len(variants)
                for route, mode in variants[order:] + variants[:order]:
                    call(source, route, mode, repetition)
        for source in args.input:
            for route in ['cpu', 'auto']:
                call(source, route, 'fit-off', 1, sampled=True)
        require(source_files(repo) == files, 'Build/tool source changed during capture')
        require(digest(binary) == manifest['binary_sha256'], 'Frozen binary changed')
        for item in manifest['inputs']:
            require(digest(Path(item['path'])) == item['sha256'], 'Input changed')
        for name, checksum in dependencies.items():
            require(digest(Path(name)) == checksum, 'Dependency changed')
        manifest['status'] = 'passed'
    except BaseException as error:
        manifest['status'], manifest['error'] = 'failed', str(error)
        raise
    finally:
        manifest['finished_unix'] = time.time()
        manifest['environment_finish'] = snapshot()
        manifest['artifacts_sha256'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*')
                                       if p.is_file() and p.name != 'manifest.json'}
        save()


if __name__ == '__main__':
    main()
