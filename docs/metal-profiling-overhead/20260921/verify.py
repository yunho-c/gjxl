"""Verify the frozen overhead evidence without building or encoding."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def analyze(script, directory):
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze(directory)


def main():
    expected = read(ROOT / 'bundle-sha256.json')
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts
              and p.name != 'bundle-sha256.json'}
    assert actual == set(expected), 'Bundle file coverage changed'
    for name, digest in expected.items():
        assert sha(ROOT / name) == digest, name
    original = read(ROOT / 'original-manifest.json')['sha256']
    for path in (ROOT / 'harness').rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        name = path.relative_to(ROOT / 'harness').as_posix()
        # The original top-level manifest retained the archived header hash
        # in evidence-sha256.json rather than as a standalone entry.
        digest = original.get(name)
        if digest is None:
            digest = read(ROOT / 'harness/evidence-sha256.json')[
                'build/profiling-overhead-20260921/' + name]
        assert sha(path) == digest, name

    measured = logged = 0
    outputs = {}
    with tempfile.TemporaryDirectory(prefix='gjxl-overhead-check-') as temp:
        root = Path(temp)
        with tarfile.open(ROOT / 'measurements.tar.gz', 'r:gz') as archive:
            seen = set()
            for member in archive:
                name = Path(member.name)
                assert member.isfile() and not name.is_absolute()
                assert '..' not in name.parts and len(name.parts) == 2
                assert name.parts[0] in ('ablation', 'followup')
                assert member.name not in seen
                seen.add(member.name)
                data = archive.extractfile(member).read()
                assert hashlib.sha256(data).hexdigest() == original[member.name]
                dest = root / name
                dest.parent.mkdir(exist_ok=True)
                dest.write_bytes(data)
        assert len(seen) == 110
        for cohort, case_count in (('ablation', 20), ('followup', 6)):
            directory = root / cohort
            identity = read(directory / 'identity.json')
            assert identity['revision'] == '3e1ef9e8e31bc809a6c3f767af3eed23f4ce69bb'
            code = ROOT / 'harness'
            if cohort == 'followup':
                code /= 'followup-code'
            assert sha(code / 'probe.cpp') == identity['source_sha256']
            assert sha(code / 'run.py') == identity['runner_sha256']
            probe = 'probe-ablation' if cohort == 'ablation' else 'followup-code/probe'
            assert identity['probe_sha256'] == original[probe]
            assert read(directory / 'complete.json')['cases'] == case_count
            markers = sorted(directory.glob('*.ok.json'))
            assert len(markers) == len(identity['jobs']) == case_count
            for marker, job in zip(markers, identity['jobs']):
                stem = marker.name.removesuffix('.ok.json')
                record = read(marker)
                raw = directory / (stem + '.jsonl')
                assert sha(raw) == record['raw_sha256']
                rows = [json.loads(line) for line in raw.read_text().splitlines()]
                expected_rows = len(identity['modes']) * (
                    identity['pairs'] * (identity['samples'] + identity['warmups'])
                    if cohort == 'ablation' else
                    identity['pairs'] * identity['samples'] + identity['warmups'])
                assert len(rows) == record['rows'] == expected_rows
                assert all(r['byte_equal'] is True and r['effort'] == job['effort'] for r in rows)
                assert {r['mode'] for r in rows} == set(identity['modes'])
                assert len({r['submissions'] for r in rows}) == 1
                assert len({r['encoded_bytes'] for r in rows}) == 1
                assert all(r['encoders'] == r['submissions'] for r in rows
                           if r['mode'] in ('ordinary', 'host', 'graph'))
                assert len({r['encoders'] for r in rows
                            if r['mode'] in ('split', 'record', 'full')}) == 1
                for pair in range(identity['pairs']):
                    for mode in identity['modes']:
                        assert sorted(r['rep'] for r in rows
                                      if r['pair'] == pair and r['mode'] == mode
                                      and r['rep'] >= 0) == list(range(identity['samples']))
                command = read(directory / (stem + '.command.json'))
                assert command['returncode'] == 0 and not command['interference']
                key = (job['label'], job['effort'])
                if cohort == 'followup':
                    assert outputs[key] == record['output_sha256']
                else:
                    outputs[key] = record['output_sha256']
                logged += len(rows)
                measured += sum(r['rep'] >= 0 for r in rows)
            script = 'analyze.py' if cohort == 'ablation' else 'analyze_followup.py'
            assert analyze(ROOT / 'harness' / script, directory) == read(directory / 'summary.json')
            print(f'{cohort}: {case_count} cases, raw hashes and recomputed paired summaries match')
    verification = read(ROOT / 'verification.json')
    assert measured == verification['measured_calls'] == 3456
    assert logged == verification['all_logged_calls'] == 4968
    print(f'Verified {measured} measured calls and {logged - measured} logged warmups; no GPU work run')


if __name__ == '__main__':
    main()
