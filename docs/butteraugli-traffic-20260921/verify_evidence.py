#!/usr/bin/env python3
"""Verify committed evidence without running an encoder, GPU job or raw-trace audit."""
from pathlib import Path
import hashlib
import json
import re
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
REPO = DOCS.parent


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = read(HERE / 'evidence-manifest.json')
    for relative, expected in manifest['files'].items():
        path = DOCS / relative
        require(sha(path.read_bytes()) == expected, f'Package hash mismatch: {relative}')

    audit = read(HERE / 'short-artifact-audit.json')
    pair_count = 0
    sample_count = 0
    for cohort in audit['cohorts']:
        label = cohort['cohort']
        folder = HERE / label
        identity = read(folder / 'identity.json')
        require(read(folder / 'state.json')['status'] == 'complete', label)
        require(sha((folder / 'protocol.py').read_bytes()) == identity['protocol_sha256'], label)
        require(sha((folder / 'summary.json').read_bytes()) == cohort['summary_sha256'], label)
        pairs = read(folder / 'pairs.json')
        summary = read(folder / 'summary.json')
        records = read(folder / 'case-records.json')
        cases = {(r['job'], r['pair'], r['side']): r['files'] for r in records}
        require(len(cases) == len(records) == 2 * len(pairs), f'{label}: duplicate/missing cases')
        require(len(pairs) == cohort['pairs'] == identity['pairs'] * len(identity['jobs']), label)
        require(len({(r['job'], r['pair']) for r in pairs}) == len(pairs), label)
        for row in pairs:
            sides = row['sides']
            require(sides['baseline']['codestream_sha256'] == sides['candidate']['codestream_sha256'], label)
            require(sides['baseline']['submissions'] == sides['candidate']['submissions'], label)
            for side, result in sides.items():
                files = cases[(row['job'], row['pair'], side)]
                for name, content in files.items():
                    require(sha(content.encode()) == result['files'][name], f'{label}: {name}')
                samples = json.loads(files['samples.json'])
                execution = json.loads(files['execution.json'])
                require(execution['returncode'] == 0 and not execution['interference'], label)
                require(samples['byte_equal'] and samples['summary_equal'], label)
                require(samples['profiled'] == identity['profile'], label)
                require(len(samples['samples']) == identity['samples'], label)
                require(statistics.median(samples['samples']) / 1e6 == result['metrics']['complete_call_ms'], label)
                command = json.loads(files['command.json'])
                require(command[0] == identity['binaries'][side], label)
                require(command[command.index('--input') + 1].endswith('/' + identity['jobs'][row['job']][0]), label)
                sample_count += len(samples['samples'])
            for metric, value in row['changes_percent'].items():
                expected = 100 * (sides['candidate']['metrics'][metric] / sides['baseline']['metrics'][metric] - 1)
                require(value == expected, f'{label}: paired ratio {metric}')
        for job, metrics in summary.items():
            rows = [row for row in pairs if row['job'] == job]
            require(len(rows) == identity['pairs'], label)
            for metric, record in metrics.items():
                expected = {
                    'median_change_percent': statistics.median(r['changes_percent'][metric] for r in rows),
                    'wins': sum(r['changes_percent'][metric] < 0 for r in rows),
                    'pairs': len(rows),
                    'baseline_ms': statistics.median(r['sides']['baseline']['metrics'][metric] for r in rows),
                    'candidate_ms': statistics.median(r['sides']['candidate']['metrics'][metric] for r in rows),
                }
                require(record == expected, f'{label}: aggregate {metric}')
        pair_count += len(pairs)

    provenance = read(HERE / 'git-provenance.json')
    for candidate in ['short-bundle', 'short-final']:
        identity = read(HERE / 'integrated' / candidate / 'identity.json')
        require(identity['test_returncode'] == 0, candidate)
        for name, path in provenance['source_paths'].items():
            blob = subprocess.check_output(['git', '-C', str(REPO), 'show', provenance[candidate] + ':' + path])
            require(sha(blob) == identity['files'][name], f'{candidate}: committed source {path}')
        parity = read(HERE / (candidate + '-canonical-parity') / 'summary.json')
        require(parity['cases'] == len(parity['rows']) == 56, candidate)
        for row in parity['rows']:
            outputs = row['outputs']
            require(outputs['baseline']['sha256'] == outputs['candidate']['sha256'], candidate)
            require(outputs['baseline']['bytes'] == outputs['candidate']['bytes'], candidate)
        require(len(parity['decodes']) == 3, candidate)
        for row in parity['decodes']:
            require(len({out['sha256'] for out in row['outputs']}) == 1, candidate)
        finite = read(HERE / (candidate + '-canonical-parity') / 'finite-pixels.json')
        require(len(finite['checks']) == 3, candidate)
        for row in finite['checks']:
            require(row['both_decodes_byte_equal'] and row['nonfinite'] == 0, candidate)
            require(row['floats'] == row['width'] * row['height'] * 3, candidate)

    for root in [HERE, DOCS / 'performance-headroom-20260921']:
        for path in root.rglob('*.md'):
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
                if target.startswith(('http:', 'https:', '#', '/')):
                    continue  # Absolute historical artifact references remain external.
                require((path.parent / target.split('#')[0]).exists(), f'Broken local link: {path}: {target}')
    require(len(audit['cohorts']) == 23 and pair_count == 256, 'Unexpected cohort scope')
    print(f'PASS: {len(manifest["files"])} package hashes; 23 cohorts, 256 pairs, {sample_count} call samples; '
          'two committed source identities; recorded byte/decoder equality; local document links.')
    print('This saved-data audit does not re-run GPU timing, decode pixels or read excluded raw traces/binaries.')


if __name__ == '__main__':
    main()
