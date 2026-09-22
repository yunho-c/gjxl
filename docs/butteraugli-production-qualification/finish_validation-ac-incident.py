#!/usr/bin/env python3
"""Diagnose the inherited test and validate the test-only correction after timing."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time

P = Path(__file__).resolve().parent
W = Path('/Users/yunhocho/GitHub/gjxl-butteraugli-production')


def save(name, value):
    (P / name).write_text(json.dumps(value, indent=2) + '\n')


def state(step, status='running'):
    save('production-validation-state.json', {
        'status': status, 'step': step,
        'at': datetime.datetime.now(datetime.timezone.utc).isoformat()})


def run(name, command, debug=False, expected=0):
    state(name)
    env = dict(os.environ, DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer')
    if debug:
        env.update(MTL_DEBUG_LAYER='1', MTL_SHADER_VALIDATION='1')
    save(name + '-command.json', {'command': command, 'metal_validation': debug})
    with (P / (name + '.log')).open('w') as f:
        result = subprocess.run(command, cwd=W, env=env, stdout=f, stderr=subprocess.STDOUT)
    save(name + '-exit.json', {'exit': result.returncode})
    if result.returncode != expected:
        raise RuntimeError(f'{name}: exit {result.returncode}, expected {expected}')


try:
    state('waiting-for-timing')
    while True:
        qualification = json.loads((P / 'qualification-state.json').read_text())
        if qualification['status'] == 'complete':
            break
        if qualification['status'] == 'failed':
            raise RuntimeError('Timing qualification failed')
        time.sleep(2)

    assert json.loads((P / 'metadata-corrected-baseline-v3-exit.json').read_text())['exit'] == 0
    assert 'Diagnostic false at source line 560' in (P / 'metadata-diagnostic-v2.log').read_text()
    run('metadata-candidate-build', ['cmake', '--build', 'build/production', '-j8',
                                    '--target', 'gjxl_metal_aq_strategy_metadata_test'])
    run('metadata-candidate', ['ctest', '--test-dir', 'build/production',
                              '-R', '^metal_aq_strategy_metadata$', '--output-on-failure',
                              '--output-junit', str(P / 'metadata-candidate.xml')], debug=True)
    run('full-suite-production', ['ctest', '--test-dir', 'build/production',
                                 '--output-on-failure', '--output-junit',
                                 str(P / 'full-suite-production.xml')])
    identity = json.loads((P / 'candidate/identity.json').read_text())
    checks = {name: hashlib.sha256((W / name).read_bytes()).hexdigest() == digest
              for name, digest in identity['sources'].items()}
    for name in ('gjxl_encode', 'gjxl.metallib'):
        path = W / 'build/production' / name
        if not path.exists():
            hits = [f for f in (W / 'build/production').rglob(name)
                    if 'installed-consumer-test' not in str(f)]
            assert len(hits) == 1
            path = hits[0]
        checks[name] = hashlib.sha256(path.read_bytes()).hexdigest() == identity['files'][name]
    assert all(checks.values()), checks
    save('final-freeze-recheck.json', checks)
    state('complete', 'complete')
    print('Production validation complete', flush=True)
except Exception as e:
    state(repr(e), 'failed')
    raise
