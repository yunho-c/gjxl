"""Resumable paired resident-profile collection from a frozen capture executable."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from validate import load, sha, validate


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def competitors(exclude=None):
    found = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,stat=,comm='], text=True).splitlines():
        pid, state, command = line.strip().split(None, 2)
        if int(pid) == exclude or any(s in state for s in 'TZ'):
            continue
        name = Path(command).name
        if name.startswith(('gjxl_', 'cjxl')) or name in ('ctest', 'ninja', 'clang', 'clang++', 'metal', 'xctrace'):
            found.append(line.strip())
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    config = load(root/'config.json')
    for name, digest in config['frozen_sha256'].items():
        assert sha(root/name) == digest, name
    for image in config['images']:
        assert sha(image['input_path']) == image['input_sha256'], image['image_id']
    lock = root/'collector.lock'
    # A crashed run leaves an auditable lock. Resume only after confirming its
    # recorded PID is no longer running; never run two collectors on one ledger.
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode());os.close(descriptor)
    started = utc()
    completed = []
    try:
        for number, job in enumerate(config['jobs']):
            case = root/'cases'/job['case_id']
            case.mkdir(parents=True, exist_ok=True)
            done = case/'complete.json'
            if done.exists():
                previous = load(done)
                attempt = case/previous['attempt']
                checked = validate(attempt, job, config)
                assert checked == previous['summary'], job['case_id']
                assert sha(attempt/'command.json') == previous['command_sha256']
                completed.append(job['case_id'])
                continue
            while busy := competitors():
                save(root/'status.json', dict(state='waiting-for-competing-work', current=job['case_id'],
                    completed=len(completed), total=len(config['jobs']), processes=busy, updated=utc()))
                time.sleep(10)
            attempts = len(list(case.glob('attempt-*')))
            attempt = case/f'attempt-{attempts:03d}'
            attempt.mkdir()
            command = [str(root/'bin/capture'), '--input', job['input_path'], '--scope', 'metal-public-workflow',
                '--validation', 'metal-only', '--gpu-aq', 'fully-resident', '--effort', str(job['effort']),
                '--distance', str(job['distance']), '--cpu-threads', str(config['cpu_threads']),
                '--warmups', str(config['warmups']), '--samples', str(config['samples']),
                '--gpu-profile', 'stage', '--gpu-profile-output', str(attempt/'gpu.json')]
            save(root/'status.json', dict(state='running', current=job['case_id'], completed=len(completed),
                total=len(config['jobs']), started=started, updated=utc()))
            print('RUN', number + 1, '/', len(config['jobs']), job['case_id'], flush=True)
            begin=time.monotonic();interference=set()
            with (attempt/'stdout.txt').open('w') as stdout, (attempt/'stderr.txt').open('w') as stderr:
                proc=subprocess.Popen(command, stdout=stdout, stderr=stderr)
                while proc.poll() is None:
                    time.sleep(2)
                    interference.update(competitors(proc.pid))
            save(attempt/'command.json', dict(argv=command, returncode=proc.returncode,
                elapsed_seconds=time.monotonic()-begin, interference=sorted(interference), finished=utc()))
            if proc.returncode or interference:
                raise RuntimeError(f'Rejected {job["case_id"]}: exit={proc.returncode}, competitors={sorted(interference)}')
            summary = validate(attempt, job, config)
            save(done, dict(attempt=attempt.name, summary=summary,
                command_sha256=sha(attempt/'command.json'), completed=utc()))
            completed.append(job['case_id'])
            print('DONE', job['case_id'], 'ordinary/profiled ms',
                  round(summary['ordinary_median_ms'], 3), round(summary['profiled_median_ms'], 3), flush=True)
        for name, digest in config['frozen_sha256'].items():
            assert sha(root/name) == digest, name
        for image in config['images']:
            assert sha(image['input_path']) == image['input_sha256'], image['image_id']
        save(root/'status.json', dict(state='complete', completed=len(completed), total=len(config['jobs']),
             started=started, updated=utc()))
    except BaseException as error:
        save(root/'status.json', dict(state='failed', completed=len(completed), total=len(config['jobs']),
             error=str(error), started=started, updated=utc()))
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
