#!/usr/bin/env python3
"""Opt-in native context-map qualification on the retained 65-image corpus.

Uses the completed DC uint search as its baseline. Collection and reporting
reuse its exact-pixel, complete-call and non-extrapolating BD-rate protocol.
Every run freezes its own source, executables, inputs and helper versions.
"""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dc_uint_search'))
import qualify as q


def freeze(root, baseline, candidate, reference):
    path = root / 'manifest.json'
    if not path.exists():
        old = q.read(reference / 'manifest.json')
        previous = q.rows(reference / 'observations.jsonl')
        assert q.read(reference / 'audit.json')['status'] == 'passed'
        assert len(previous) == len({r['id'] for r in previous}) == 455
        lookup = {(r['image_id'], r['requested_quality']): r for r in previous}
        records = []
        for row in old['rows']:
            r = lookup[row['image_id'], row['requested_quality']]
            assert r['exact_decoded_identity'] and r['decoded_sha256'] == row['decoded_sha256']
            output = Path(r['folder']) / 'candidate/out.jxl'
            assert q.sha(output) == r['candidate_sha256']
            records.append({**row, 'output_path': str(output),
                            'output_sha256': r['candidate_sha256'],
                            'encoded_bytes': r['candidate_bytes']})
        assert q.sha(baseline) == old['files'][old['candidate']]
        files = {}
        for binary in [baseline, candidate]:
            files[str(binary)] = q.sha(binary)
            for metal in binary.parent.rglob('*.metallib'):
                files[str(metal)] = q.sha(metal)
        sources = subprocess.check_output(
            ['git', 'ls-files', '--cached', '--others', '--exclude-standard',
             'src', 'include', 'CMakeLists.txt', 'cmake/InstalledHeaders.cmake'], text=True).splitlines()
        with zipfile.ZipFile(root / 'source-snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for source in sorted(set(sources)):
                file = Path(source).resolve()
                files[str(file)] = q.sha(file)
                archive.write(file, source)
        for file in [Path(__file__).resolve(), Path(q.__file__), Path(q.study.__file__),
                     Path(q.cross_metric.__file__), Path(q.study.bd.__file__),
                     reference / 'manifest.json', reference / 'observations.jsonl',
                     reference / 'audit.json', Path(old['djxl']),
                     q.study.LIBJXL / 'scores.jsonl', root / 'source-snapshot.zip']:
            files[str(file)] = q.sha(file)
        for image in old['images'].values():
            assert q.sha(image['pfm_path']) == image['pfm_sha256']
            files[image['pfm_path']] = image['pfm_sha256']
        (root / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD']))
        files[str(root / 'source.patch')] = q.sha(root / 'source.patch')
        q.save(path, {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      'baseline': str(baseline), 'candidate': str(candidate),
                      'reference': str(reference), 'files': files, 'images': old['images'],
                      'rows': records, 'djxl': old['djxl'], 'expected': 455,
                      'timing_images': old['timing_images'],
                      'protocol': 'DC uint search baseline; 65 images x seven retained distances. '
                                  'Fresh baseline byte identities and exact candidate decoded PFM hashes. '
                                  'BD-rate at SSIMULACRA2 75-85 with PCHIP/Akima, no extrapolation. '
                                  'Separate complete-call timing: 12 images, four alternating process '
                                  'pairs, two warmups and five samples. Model budgets unchanged.'})
    m = q.read(path)
    assert m['baseline'] == str(baseline) and m['candidate'] == str(candidate)
    assert m['reference'] == str(reference)
    for file, digest in m['files'].items():
        assert q.sha(file) == digest, file
    return m


def timing(root, manifest):
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            result = subprocess.run(['ps', '-axo', 'pid,ppid,pcpu,comm'], capture_output=True, text=True)
            q.append(root / 'timing-environment.jsonl', {'timestamp': time.time(), 'processes': result.stdout})
            stop.wait(5)

    q.save(root / 'environment.json', {
        'power': subprocess.check_output(['pmset', '-g', 'batt'], text=True),
        'system': subprocess.check_output(['uname', '-a'], text=True),
        'started': time.time()})
    worker = threading.Thread(target=monitor)
    worker.start()
    try:
        q.timing(root, manifest)
    finally:
        stop.set()
        worker.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['rate', 'timing', 'report'])
    parser.add_argument('--root', type=Path, default=Path('build/context-map-qualification-20260914'))
    parser.add_argument('--reference', type=Path, default=Path('build/dc-uint-fast-qualification-20260914'))
    parser.add_argument('--baseline', type=Path, default=Path('build/dc-uint-fast-candidate/gjxl_quality_benchmark'))
    parser.add_argument('--candidate', type=Path, default=Path('build/context-map-native/gjxl_quality_benchmark'))
    parser.add_argument('--limit', type=int, default=455)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m = freeze(root, args.baseline.resolve(), args.candidate.resolve(), args.reference.resolve())
        if args.action == 'rate':
            q.rate(root, m, args.limit)
        elif args.action == 'timing':
            timing(root, m)
        else:
            q.report(root, m)


if __name__ == '__main__':
    main()
