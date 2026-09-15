#!/usr/bin/env python3
"""Opt-in paired latency qualification using the already audited rate binaries.

Retains every attempt. Admission/retry decisions use environment evidence only,
never observed latency. No compilation, decoding or scoring during collection.
"""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dc_uint_search'))
import qualify as q

PROTOCOL = {
    'blocks': 2, 'pairs_per_block': 4, 'warmups': 2, 'samples_per_process': 5,
    'threads': 8, 'effort': 4, 'monitor_interval_seconds': 1,
    'quiet_seconds': 20, 'maximum_idle_wait_seconds': 600,
    'maximum_attempts_per_pair': 3,
    'maximum_other_process_cpu_percent': 80,
    'maximum_aggregate_other_cpu_percent': 200,
    'cpu_measurement': 'Process CPU-time deltas over at least 0.75 seconds; ps percent CPU for new processes and shorter intervals.',
    'order': 'Alternate baseline/candidate order; rotate image order by pair index.',
    'exclusion': 'Reject the entire pair on observed competing codec/build work, '
                 'CPU pressure above the declared limits, loss of AC power, '
                 'thermal/performance warning, or monitoring failure.',
    'boundary': 'complete-encode-wall-time',
}


def output(argv):
    return subprocess.check_output(argv, text=True, timeout=10)


def freeze(root, reference):
    path = root / 'manifest.json'
    old = q.read(reference / 'manifest.json')
    assert q.read(reference / 'audit.json')['status'] == 'passed'
    for file, digest in old['files'].items():
        assert q.sha(file) == digest, file
    if not path.exists():
        rows = [r for r in q.rows(reference / 'observations.jsonl')
                if r['requested_quality'] == 80 and r['image_id'] in old['timing_images']]
        assert len(rows) == len({r['image_id'] for r in rows}) == 12
        files = {str(reference / name): q.sha(reference / name)
                 for name in ['manifest.json', 'audit.json', 'observations.jsonl']}
        for file in [Path(__file__).resolve(), Path(q.__file__)]:
            files[str(file)] = q.sha(file)
        for r in rows:
            assert r['exact_decoded_identity']
            for arm in ['baseline', 'candidate']:
                file = Path(r['folder']) / arm / 'out.jxl'
                assert q.sha(file) == r[arm + '_sha256']
                files[str(file)] = r[arm + '_sha256']
        q.save(path, {
            'reference': str(reference), 'files': files, 'protocol': PROTOCOL,
            'head': output(['git', 'rev-parse', 'HEAD']).strip(),
            'baseline': old['baseline'], 'candidate': old['candidate'],
            'binary_sha256': {old[arm]: old['files'][old[arm]] for arm in ['baseline', 'candidate']},
            'images': old['images'], 'timing_images': old['timing_images'], 'rows': rows,
            'system': output(['uname', '-a']),
            'hardware': output(['sysctl', '-n', 'hw.model', 'hw.memsize', 'hw.ncpu', 'machdep.cpu.brand_string']),
        })
    m = q.read(path)
    assert m['reference'] == str(reference) and m['protocol'] == PROTOCOL
    for file, digest in m['files'].items():
        assert q.sha(file) == digest, file
    return m


def cpu_seconds(value):
    parts = value.split(':')
    return sum(float(part) * 60 ** i for i, part in enumerate(reversed(parts)))


def competing(comm):
    name = Path(comm).name.lower()
    return (name.startswith(('gjxl_', 'gjxl-', 'clang', 'gcc', 'g++', 'rustc')) or
            name in {'cjxl', 'djxl', 'ssimulacra2', 'butteraugli', 'cjxl-quality-metric',
                     'cmake', 'ctest', 'ninja', 'make', 'xcodebuild', 'cargo', 'cc', 'c++'})


class Monitor:
    def __init__(self, root, manifest):
        self.root, self.manifest = root, manifest
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.snapshots = []
        self.previous = {}
        self.previous_time = None
        self.thread = threading.Thread(target=self.run)

    def sample(self):
        with self.lock:
            now = time.time()
            row = {'timestamp': now, 'issues': []}
            try:
                raw = output(['ps', '-axo', 'pid,ppid,pcpu,time,comm'])
                power = output(['pmset', '-g', 'batt'])
                thermal = output(['pmset', '-g', 'therm'])
                row.update(processes=raw, power=power, thermal=thermal)
                current, other = {}, []
                for line in raw.splitlines()[1:]:
                    fields = line.strip().split(None, 4)
                    if len(fields) != 5:
                        continue
                    pid, ppid, pcpu, elapsed, comm = fields
                    pid, ppid = int(pid), int(ppid)
                    seconds = cpu_seconds(elapsed)
                    current[pid] = (seconds, comm)
                    own = (pid == os.getpid() or
                           (ppid == os.getpid() and comm in [self.manifest['baseline'], self.manifest['candidate']]))
                    if own:
                        continue
                    cpu = float(pcpu)
                    previous = self.previous.get(pid)
                    if (previous and previous[1] == comm and self.previous_time is not None
                            and now - self.previous_time >= 0.75):
                        cpu = max(0.0, 100 * (seconds - previous[0]) / (now - self.previous_time))
                    other.append({'pid': pid, 'cpu_percent': cpu, 'command': comm})
                    if competing(comm):
                        row['issues'].append(f'competing process: {pid} {comm}')
                    if cpu > PROTOCOL['maximum_other_process_cpu_percent']:
                        row['issues'].append(f'other CPU pressure: {pid} {cpu:.1f}% {comm}')
                if self.previous_time is None or now - self.previous_time >= 0.75:
                    self.previous, self.previous_time = current, now
                row['aggregate_other_cpu_percent'] = sum(p['cpu_percent'] for p in other)
                row['top_other_cpu'] = sorted(other, key=lambda p: p['cpu_percent'], reverse=True)[:8]
                if row['aggregate_other_cpu_percent'] > PROTOCOL['maximum_aggregate_other_cpu_percent']:
                    row['issues'].append('aggregate other CPU pressure')
                if "Now drawing from 'AC Power'" not in power:
                    row['issues'].append('not on AC power')
                if ('No thermal warning level has been recorded' not in thermal or
                        'No performance warning level has been recorded' not in thermal):
                    row['issues'].append('thermal/performance status requires review')
            except Exception as error:
                row['issues'].append(f'monitor failure: {error!r}')
            q.append(self.root / 'environment.jsonl', row)
            self.snapshots.append(row)
            return row

    def run(self):
        while not self.stop.wait(PROTOCOL['monitor_interval_seconds']):
            self.sample()

    def quiet(self):
        start = time.time()
        clean_since = start
        while time.time() - start < PROTOCOL['maximum_idle_wait_seconds']:
            row = self.sample()
            if row['issues']:
                clean_since = time.time()
                print('waiting for idle:', '; '.join(row['issues'][:2]), flush=True)
            elif time.time() - clean_since >= PROTOCOL['quiet_seconds']:
                return
            time.sleep(1)
        raise RuntimeError('Idle admission timed out; retained run can be resumed.')


def collect(root, m):
    accepted = {r['id'] for r in q.rows(root / 'attempts.jsonl') if r['accepted']}
    refs = {r['image_id']: r for r in m['rows']}
    monitor = Monitor(root, m)
    monitor.thread.start()
    try:
        monitor.quiet()
        for pair in range(PROTOCOL['blocks'] * PROTOCOL['pairs_per_block']):
            images = m['timing_images']
            images = images[pair % len(images):] + images[:pair % len(images)]
            for image in images:
                identifier = hashlib.sha256(image.encode()).hexdigest()[:16] + f'-{pair}'
                if identifier in accepted:
                    continue
                old_attempts = [r for r in q.rows(root / 'attempts.jsonl') if r['id'] == identifier]
                for attempt in range(len(old_attempts), PROTOCOL['maximum_attempts_per_pair']):
                    pre = monitor.sample()
                    if pre['issues']:
                        monitor.quiet()
                    start = time.time()
                    arms = {}
                    order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
                    for arm in order:
                        folder = root / 'cases' / f'{identifier}-{attempt}-{arm}'
                        raw = q.encode(root, m, refs[image], arm, folder,
                                       PROTOCOL['warmups'], PROTOCOL['samples_per_process'])
                        digest = q.sha(folder / 'out.jxl')
                        assert digest == refs[image][arm + '_sha256'], (image, arm)
                        assert raw['backend'] == 'metal' and raw['thread_count'] == PROTOCOL['threads']
                        assert raw['warmups'] == PROTOCOL['warmups'] and not raw['stage_profile_enabled']
                        values = [s['elapsed_nanoseconds'] for s in raw['samples']]
                        arms[arm] = {'samples_ns': values, 'median_ns': statistics.median(values),
                                     'raw_path': str(folder / 'raw.json'), 'raw_sha256': q.sha(folder / 'raw.json'),
                                     'output_path': str(folder / 'out.jxl'), 'output_sha256': digest}
                    post = monitor.sample()
                    end = post['timestamp']
                    with monitor.lock:
                        environment = [r for r in monitor.snapshots if start - 1 <= r['timestamp'] <= end]
                    issues = sorted({issue for r in environment for issue in r['issues']})
                    row = {'id': identifier, 'image_id': image, 'pair': pair,
                           'block': pair // PROTOCOL['pairs_per_block'], 'attempt': attempt,
                           'started': start, 'ended': end, 'arms': arms, 'order': order,
                           'accepted': not issues, 'issues': issues,
                           'environment_samples': len(environment)}
                    q.append(root / 'attempts.jsonl', row)
                    print('accepted' if not issues else 'excluded', len(accepted) + (not issues),
                          '/96', image, 'pair', pair, 'attempt', attempt, issues[:2], flush=True)
                    if not issues:
                        accepted.add(identifier)
                        break
                    monitor.quiet()
                else:
                    raise RuntimeError(f'Attempt limit reached for {identifier}; inspect environment evidence.')
    finally:
        monitor.stop.set()
        monitor.thread.join()


def report(root, m):
    attempts = q.rows(root / 'attempts.jsonl')
    accepted = [r for r in attempts if r['accepted']]
    assert len(accepted) == len({r['id'] for r in accepted}) == 96
    env = q.rows(root / 'environment.jsonl')
    refs = {r['image_id']: r for r in m['rows']}
    for r in attempts:
        relevant = [e for e in env if r['started'] - 1 <= e['timestamp'] <= r['ended']]
        assert len(relevant) == r['environment_samples'] and relevant
        issues = sorted({issue for e in relevant for issue in e['issues']})
        assert issues == r['issues'] and r['accepted'] == (not issues)
        for arm, a in r['arms'].items():
            assert q.sha(a['raw_path']) == a['raw_sha256']
            assert q.sha(a['output_path']) == a['output_sha256'] == refs[r['image_id']][arm + '_sha256']
            raw = q.read(a['raw_path'])
            assert [s['elapsed_nanoseconds'] for s in raw['samples']] == a['samples_ns']
            assert statistics.median(a['samples_ns']) == a['median_ns']
    results = []
    for image in m['timing_images']:
        selected = sorted([r for r in accepted if r['image_id'] == image], key=lambda r: r['pair'])
        assert [r['pair'] for r in selected] == list(range(8))
        ratios = [r['arms']['candidate']['median_ns'] / r['arms']['baseline']['median_ns'] for r in selected]
        results.append({'image_id': image, 'paired_ratios': ratios,
                        'median_paired_ratio': statistics.median(ratios),
                        'block_median_paired_ratios': [statistics.median(ratios[:4]), statistics.median(ratios[4:])],
                        'median_ms': {arm: statistics.median(v for r in selected for v in r['arms'][arm]['samples_ns']) / 1e6
                                      for arm in ['baseline', 'candidate']}})
    ratios = [r['median_paired_ratio'] for r in results]
    commands = q.rows(root / 'commands.jsonl')
    assert len(commands) == 2 * len(attempts) and all(c['returncode'] == 0 for c in commands)
    result = {'status': 'passed', 'protocol': PROTOCOL,
              'accepted_pairs': len(accepted), 'excluded_pairs': len(attempts) - len(accepted),
              'accepted_processes': 2 * len(accepted), 'measured_calls': 10 * len(accepted),
              'codestream_identities': 2 * len(attempts),
              'median_per_image_paired_ratio': statistics.median(ratios),
              'per_image_ratio_range': [min(ratios), max(ratios)],
              'block_medians': [statistics.median(r['block_median_paired_ratios'][b] for r in results) for b in range(2)],
              'images': results, 'environment_snapshots': len(env),
              'limitation': 'No contention detected under the declared sampled checks. Desktop applications remain open; this is local warm-call qualification, not hardware-wide or cold-start qualification.',
              'provenance': {str(root / name): q.sha(root / name) for name in
                             ['manifest.json', 'attempts.jsonl', 'commands.jsonl', 'environment.jsonl']}}
    q.save(root / 'analysis.json', result)
    print({k: v for k, v in result.items() if k not in ['images', 'provenance', 'protocol']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['collect', 'report'])
    parser.add_argument('--root', type=Path, default=Path('build/context-map-latency-20260915'))
    parser.add_argument('--reference', type=Path, default=Path('build/context-map-qualification-20260914'))
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m = freeze(root, args.reference.resolve())
        if args.action == 'collect':
            collect(root, m)
        else:
            report(root, m)


if __name__ == '__main__':
    main()
