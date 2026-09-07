"""Fresh CTest and scoped sanitizer evidence, without archived-log dependencies."""
from pathlib import Path
import json
import re
import signal
import xml.etree.ElementTree as ET
from session import PACKAGE, check, read, save, sha

GOLDEN = 'actual=0.24919039011001587 expected=0.24914586544036865'
REPEATED = ['cpu_budget', 'cpu_execution', 'batch_lifecycle', 'batch_lifecycle_cpu',
            'batch_scheduling', 'batch_scheduling_cpu', 'workflow_cpu_coordination',
            'workflow_cpu_coordination_cpu', 'workflow_admission', 'workflow_admission_cpu',
            'workflow_publication', 'metal_completed_frame', 'metal_cache_admission']
CPU_GROUPS = ['startup', 'forward', 'cpu_color', 'cpu_aq', 'cpu_orders', 'cpu_sections', 'cpu_drains']
METAL_GROUPS = ['metal_forward', 'metal_orders', 'metal_sections', 'metal_drains']


def suite(session, directory, stem, names=None, repeat=1, environment=None, golden=False):
    sidecar = session.out / (stem + '.json')
    fingerprint = {'build_sha256': sha(session.out / 'build.json'), 'directory': str(directory),
                   'names': names, 'repeat': repeat, 'environment': environment, 'golden': golden}
    if sidecar.exists():
        result = read(sidecar)
        check(result['fingerprint'] == fingerprint, 'Stale test configuration: ' + stem)
        validate_suite(session, result)
        return result
    inventory_command = session.run(['ctest', '--test-dir', directory, '--show-only=json-v1'], stem + '-inventory')
    inventory = json.loads((session.out / inventory_command['log']).read_text())
    available = [t['name'] for t in inventory['tests']]
    selected = names or available
    check(selected and set(selected) <= set(available), 'Required tests missing: ' + stem)
    xml = session.unique(stem, '.xml')
    command = ['ctest', '--test-dir', directory, '--output-on-failure', '--parallel', '1', '--output-junit', xml]
    if names:
        command += ['-R', '^(' + '|'.join(map(re.escape, names)) + ')$']
    if repeat > 1:
        command += ['--repeat', 'until-fail:' + str(repeat)]
    result = session.run(command, stem, expected=(0, 8) if golden else (0,), environment=environment, timeout=3600)
    cases = ET.parse(xml).getroot().findall('.//testcase')
    check(sorted(c.get('name') for c in cases) == sorted(selected), 'Incomplete CTest inventory: ' + stem)
    check(not any(c.find('skipped') is not None for c in cases), 'Skipped required test: ' + stem)
    failed = [c for c in cases if c.find('failure') is not None]
    known = None
    if failed:
        check(golden and len(failed) == 1 and failed[0].get('name') == 'quantization_pipeline', 'Unexpected test failure: ' + stem)
        diagnostic = ''.join(failed[0].itertext())
        check(GOLDEN in diagnostic, 'Quantization failure differs from documented mismatch')
        known = GOLDEN
    check(result['exit'] == (8 if failed else 0), 'CTest exit contradicts its results')
    log = (session.out / result['log']).read_text()
    passed_lines = len(re.findall(r'\.\.\.\s+Passed', log))
    check(passed_lines == (len(selected) - len(failed)) * repeat, 'Incomplete repetition count: ' + stem)
    record = {'status': 'complete', 'fingerprint': fingerprint, 'command': result,
              'inventory': inventory_command, 'xml': str(xml.relative_to(session.out)), 'xml_sha256': sha(xml),
              'tests': selected, 'repeat': repeat, 'passed': len(selected) - len(failed), 'known_failure': known}
    save(sidecar, record)
    return record


def validate_suite(session, record):
    check(record['status'] == 'complete', 'Incomplete test record')
    for key in ('command', 'inventory'):
        data = record[key]
        check(sha(session.out / data['log']) == data['log_sha256'], 'Test log changed')
    check(sha(session.out / record['xml']) == record['xml_sha256'], 'Test XML changed')


def validate_metal_diagnostic(session, record):
    check(sha(session.out / record['log']) == record['log_sha256'], 'Unsuppressed diagnostic changed')
    text = (session.out / record['log']).read_text()
    # Darwin's combined ASan/UBSan runtime may abort on halt_on_error.
    # Accept that signal only for the independently retained known diagnostic.
    check(record['exit'] == 0 or (record['exit'] in (1, -signal.SIGABRT) and
          'third_party/metal-cpp/' in text and 'runtime error: member call on null pointer' in text),
          'Unexpected unsuppressed Metal failure')


def tests(session):
    session.identities()
    controller = session.run([__import__('sys').executable, PACKAGE/'test_runner.py'], 'controller-tests')
    records = {label: suite(session, path, label + '-tests', golden=True) for label, path in session.builds.items()}
    if records['candidate']['known_failure']:
        check(records['integrated']['known_failure'] == records['candidate']['known_failure'],
              'Candidate quantization failure was not reproduced on this baseline')
    repeated = suite(session, session.builds['candidate'], 'candidate-repeated', names=REPEATED, repeat=3)
    save(session.out / 'tests.json', {'status': 'complete', 'controller': controller, 'records': records, 'repeated': repeated})


def sanitizers(session):
    session.identities()
    result = {'status': 'running', 'records': [], 'artifacts': {}, 'limitations': {
        'leak_detection': False, 'asan_objcxx_instrumentation': False,
        'metal_ubsan_suppression': (PACKAGE / 'metal-ubsan.supp').read_text().strip()}}
    for mode, flags in [('tsan', 'thread'), ('asan', 'address,undefined')]:
        directory = session.out / 'builds' / mode
        stamp = session.out / (mode + '-build.json')
        if not stamp.exists():
            session.compile('candidate', mode, flags)
            from session import package_hashes
            artifacts = {str(p.relative_to(session.out)): sha(p) for p in directory.rglob('*')
                         if p.is_file() and (p.suffix == '.a' or p.name == 'CMakeCache.txt' or
                                            (p.name.startswith('gjxl_') and p.stat().st_mode & 0o111))}
            save(stamp, {'package': package_hashes(), 'artifacts': artifacts})
        build = read(stamp)
        from session import package_hashes
        check(build['package'] == package_hashes(), 'Sanitizer harness changed')
        for name, digest in build['artifacts'].items():
            check(sha(session.out / name) == digest, 'Sanitizer binary changed: ' + name)
        result['artifacts'].update(build['artifacts'])
        environment = {'TSAN_OPTIONS': 'halt_on_error=1'} if mode == 'tsan' else {
            'ASAN_OPTIONS': 'detect_leaks=0:halt_on_error=1', 'UBSAN_OPTIONS': 'halt_on_error=1'}
        result['records'].append(suite(session, directory / 'codec', mode + '-cpu-groups',
            names=['worker_launch_failure_' + n for n in CPU_GROUPS], environment=environment))
        result['records'].append(suite(session, directory / 'codec', mode + '-cpu-repeat',
            names=['cpu_execution', 'batch_lifecycle_cpu', 'batch_scheduling_cpu'],
            repeat=10 if mode == 'tsan' else 3, environment=environment))
        if mode == 'asan':
            diagnostic_path = session.out / 'asan-unsuppressed.json'
            if diagnostic_path.exists():
                diagnostic = read(diagnostic_path)
            else:
                diagnostic = session.run([directory / 'codec/gjxl_worker_launch_failure_test', '--group', 'metal_orders'],
                    'asan-unsuppressed', expected=(0, 1, -signal.SIGABRT), environment=environment, timeout=900)
                validate_metal_diagnostic(session, diagnostic)
                save(diagnostic_path, diagnostic)
            validate_metal_diagnostic(session, diagnostic)
            result['unsuppressed'] = diagnostic
            metal_env = {**environment, 'UBSAN_OPTIONS': 'halt_on_error=1:suppressions=' + str(PACKAGE / 'metal-ubsan.supp')}
            result['records'].append(suite(session, directory / 'codec', 'asan-metal-groups',
                names=['worker_launch_failure_' + n for n in METAL_GROUPS], environment=metal_env))
            result['records'].append(suite(session, directory / 'codec', 'asan-metal-repeat',
                names=['batch_lifecycle', 'batch_scheduling'], repeat=3, environment=metal_env))
        save(session.out / 'sanitizers.json', result)
    result['status'] = 'complete'
    save(session.out / 'sanitizers.json', result)
