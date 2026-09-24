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
        path = HERE / relative
        require(sha(path.read_bytes()) == expected, f'Package hash mismatch: {relative}')

    audit = read(HERE / 'artifact-audit.json')
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
    for label, revision in [('baseline', provenance['base']), ('candidate', provenance['source_commit'])]:
        identity = read(HERE / label / 'identity.json')
        for path, digest in identity['sources'].items():
            blob = subprocess.check_output(['git', '-C', str(REPO), 'show', revision + ':' + path])
            require(sha(blob) == digest, f'{label}: committed source {path}')
    for path, digest in provenance['source_files'].items():
        blob = subprocess.check_output(['git', '-C', str(REPO), 'show', provenance['source_commit'] + ':' + path])
        require(sha(blob) == digest, f'Production source {path}')
    parity = read(HERE / 'canonical-parity-v2/summary.json')
    require(parity['cases'] == len(parity['rows']) == 56, 'Canonical case count')
    for row in parity['rows']:
        outputs = row['outputs']
        require(outputs['baseline']['sha256'] == outputs['candidate']['sha256'], row['name'])
        require(outputs['baseline']['bytes'] == outputs['candidate']['bytes'], row['name'])
    require(len(parity['decodes']) == 3, 'Decode count')
    for row in parity['decodes']:
        require(len({out['sha256'] for out in row['outputs']}) == 1, row['name'])
    finite = read(HERE / 'canonical-parity-v2/finite-pixels.json')
    require(len(finite['checks']) == 3, 'Finite decoded count')
    for row in finite['checks']:
        require(row['both_decodes_byte_equal'] and row['nonfinite'] == 0, row['name'])
        require(row['width'] > 0 and row['height'] > 0, row['name'])
    import xml.etree.ElementTree as ET
    suite = ET.parse(HERE / 'full-suite-final.xml').getroot()
    cases = suite.findall('testcase')
    failures = [c.get('name') for c in cases if c.find('failure') is not None]
    require(len(cases) == 156 and failures == ['metal_aq_strategy_metadata'], 'Full suite result')
    control = ET.parse(HERE / 'baseline-failure-control.xml').getroot()
    require([c.get('name') for c in control.findall('testcase') if c.find('failure') is not None] == failures,
            'Inherited failure control')
    final = ET.parse(HERE / 'full-suite-release.xml').getroot()
    require(len(final.findall('testcase')) == 156 and final.get('failures') == '0', 'Final production suite')
    incident = ET.parse(HERE / 'full-suite-production.xml').getroot()
    require([c.get('name') for c in incident.findall('testcase') if c.find('failure') is not None]
            == ['metal_ac_strategy'], 'Preserved AC failure incident')
    repeats = read(HERE / 'ac-original-results.json')
    require(len(repeats) == 40 and all(r['exit'] == 0 for r in repeats), 'Original AC repeat results')
    metadata = ET.parse(HERE / 'metadata-candidate.xml').getroot()
    require(len(metadata.findall('testcase')) == 1 and metadata.get('failures') == '0', 'Corrected metadata test')
    require(read(HERE / 'metadata-corrected-baseline-v3-exit.json')['exit'] == 0,
            'Corrected metadata test on unchanged baseline')
    require(all(read(HERE / 'final-freeze-recheck.json').values()), 'Measured runtime unchanged')
    focused = ET.parse(HERE / 'integration-tests.xml').getroot()
    require(len(focused.findall('testcase')) == 3 and focused.get('failures') == '0', 'Updated integration checks')

    for root in [HERE]:
        for path in root.rglob('*.md'):
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
                if target.startswith(('http:', 'https:', '#', '/')):
                    continue  # Absolute historical artifact references remain external.
                require((path.parent / target.split('#')[0]).exists(), f'Broken local link: {path}: {target}')
    require(len(audit['cohorts']) == 4 and pair_count == 72 and sample_count == 768, 'Unexpected cohort scope')
    print(f'PASS: {len(manifest["files"])} package hashes; 4 cohorts, 72 pairs, {sample_count} call samples; '
          'current-main and production source identities; recorded byte/decoder equality; local document links.')
    print('This saved-data audit does not re-run GPU timing, decode pixels or read excluded raw traces/binaries.')


if __name__ == '__main__':
    main()
