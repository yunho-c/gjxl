#!/usr/bin/env python3
"""Small resumable DC-worker screen, alternating independent processes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/tokenization-20260922/dc-screen'
STUDY = Path('/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/gjxl-stages-q80-20260921')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads((STUDY / 'config.json').read_text())
    images = [config['images'][i] for i in (1, 4, 5)]
    hashes = json.loads((RUN / 'binary-hashes.json').read_text())
    for name, expected in hashes.items():
        assert sha(RUN / name) == expected
    results = []
    expected_outputs = {}
    for im in images:
        assert sha(Path(im['input_path'])) == im['input_sha256']
    for round_id in range(2):
        for image_index, im in enumerate(images):
            for effort in (1, 7):
                modes = ['base', '1', '2', '4', '8'] if round_id == 0 else ['8', '4', '2', '1', 'base']
                for mode in modes:
                    out = RUN / f'r{round_id}-i{image_index}-e{effort}-w{mode}'
                    out.mkdir(exist_ok=True)
                    done = out / 'complete.json'
                    if done.exists():
                        record = json.loads(done.read_text())
                        assert sha(out / 'reference.jxl') == record['output_sha256']
                    else:
                        env = {k:v for k,v in os.environ.items() if not k.startswith('GJXL_EXPERIMENT_')}
                        if mode != 'base':
                            env['GJXL_EXPERIMENT_DC_WORKERS'] = mode
                        command = [str(RUN / ('base' if mode == 'base' else 'dc')),
                            '--input', im['input_path'], '--scope', 'metal-public-workflow',
                            '--validation', 'metal-only', '--gpu-aq', 'fully-resident',
                            '--effort', str(effort), '--distance', '1.9', '--cpu-threads', '8',
                            '--warmups', '1', '--samples', '2', '--gpu-profile', 'stage',
                            '--gpu-profile-output', str(out / 'gpu.json')]
                        (out / 'command.json').write_text(json.dumps({'argv':command,'dc_workers':mode},indent=2))
                        start = time.monotonic()
                        with (out / 'stdout.txt').open('w') as stdout, (out / 'stderr.txt').open('w') as stderr:
                            subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=True, timeout=180)
                        data = json.loads((out / 'paired.json').read_text())
                        rows = [r for r in data['rows'] if r['sample_index'] >= 0]
                        assert len(rows) == 4 and all(r['byte_equal'] and r['summary_equal'] for r in rows)
                        ordinary = [r['complete_call_nanoseconds']/1e6 for r in rows if r['mode']=='ordinary']
                        profiles = [r['phase_nanoseconds'] for r in rows if r['mode']=='profiled']
                        record = {'round':round_id,'image':im['image_id'],'effort':effort,'dc_workers':mode,
                            'ordinary_ms':statistics.median(ordinary),'ordinary_samples_ms':ordinary,
                            'dc_ms':statistics.mean(r['codestream_dc_tokenization'] for r in profiles)/1e6,
                            'ac_ms':statistics.mean(r['codestream_ac_tokenization'] for r in profiles)/1e6,
                            'output_sha256':sha(out/'reference.jxl'),'elapsed_s':time.monotonic()-start}
                        done.write_text(json.dumps(record,indent=2)+'\n')
                    key = (im['image_id'],effort)
                    if key in expected_outputs:
                        assert expected_outputs[key] == record['output_sha256'], (key,mode,'byte mismatch')
                    else:
                        expected_outputs[key] = record['output_sha256']
                    results.append(record)
                    print(round_id,im['resolution_class'],effort,mode,round(record['ordinary_ms'],2),round(record['dc_ms'],2),flush=True)
    (RUN / 'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Complete: 60 processes; every reference identical across modes and rounds.',flush=True)


if __name__ == '__main__':
    main()
