#!/usr/bin/env python3
"""Check native GPU-greedy encodes against retained diagnostic CPU controls."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from run_experiment import ROOT, run, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    controls = args.controls.resolve()
    source = json.loads((controls / 'manifest.json').read_text())
    rows = [json.loads(x) for x in (controls / 'observations.jsonl').read_text().splitlines()]
    images = {x['id']: x['path'] for x in source['images']}
    executable = output / 'gjxl_quality_benchmark'
    shutil.copy2(ROOT / 'build/frontier/gjxl_quality_benchmark', executable)
    files = ['CMakeLists.txt', 'src/gpu/ops/ac_strategy_search.cpp', 'src/gpu/ops/ac_strategy_search.h',
             'src/gpu/ops/ac_strategy_selection.h', 'src/gpu/metal/metal_ac_strategy.cpp',
             'src/gpu/metal/metal_backend_internal.h', 'src/gpu/metal/kernels/ac_strategy_select.metal',
             'tools/frontier_dp/check_native_encodes.py']
    for path in files:
        target = output / 'source' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    manifest = dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    sources={p: sha(ROOT / p) for p in files}, binary_sha256=sha(executable),
                    control_manifest_sha256=sha(controls / 'manifest.json'),
                    observations_sha256=sha(controls / 'observations.jsonl'), commands=[], completed=False)
    (output / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff'], cwd=ROOT))
    def save():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    env = {k: v for k, v in os.environ.items() if not k.startswith('GJXL_')}
    count = 0
    for row in rows:
        if row['arm'] != 'greedy':
            continue
        image = images[row['image']]
        assert sha(image) == source['files'][image]
        assert sha(row['output']) == row['output_sha256']
        folder = output / 'cases' / row['id']
        folder.mkdir(parents=True)
        codestream = folder / 'output.jxl'
        argv = [executable, '--input', image, '--output', codestream, '--raw-samples', folder / 'raw.json',
                '--effort', row['effort'], '--distance', row['distance'], '--num-threads', 8,
                '--warmups', 0, '--samples', 1]
        manifest['commands'].append(run(argv, folder / 'encode.log', env))
        digest = sha(codestream)
        check = dict(id=row['id'], cpu_sha256=row['output_sha256'], gpu_sha256=digest,
                     exact=digest == row['output_sha256'], output=str(codestream))
        with (output / 'checks.jsonl').open('a') as f:
            f.write(json.dumps(check) + '\n')
        save()
        if not check['exact']:
            raise RuntimeError(f"Native selection changed codestream: {row['id']}")
        count += 1
        print('Byte identical', count, row['id'], flush=True)
    assert count == 60
    manifest.update(completed=True, byte_identical=count)
    save()


if __name__ == '__main__':
    main()
