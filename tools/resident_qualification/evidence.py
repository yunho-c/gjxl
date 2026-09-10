"""Read-only semantic audit and sealing of the selected protocol profile."""
from session import check, read, save, sha
import gates
import protocol as p


def check_command(session, record):
    check(sha(session.out / record['log']) == record['log_sha256'], 'Command log changed')


def check_outputs(value):
    if isinstance(value, dict):
        for key in ('compressed', 'decoded'):
            if key in value and key + '_sha256' in value:
                check(sha(value[key]) == value[key + '_sha256'], 'Retained output changed')
        for child in value.values():
            check_outputs(child)
    elif isinstance(value, list):
        for child in value:
            check_outputs(child)


def audit(session):
    session.identities()
    out = session.out
    required = ('tests', 'sanitizers', 'parity', 'conformance', 'retained', 'performance', 'pressure', 'performance-report')
    records = {name: read(out / (name + '.json')) for name in required}
    check(all(r['status'] == 'complete' for r in records.values()), 'Required phase is incomplete')
    tests = records['tests']
    check_command(session, tests['controller'])
    for record in [*tests['records'].values(), tests['repeated']]:
        gates.validate_suite(session, record)
    if tests['records']['candidate']['known_failure']:
        check(tests['records']['candidate']['known_failure'] == tests['records']['integrated']['known_failure'], 'Baseline failure differs')
    sanitizer = records['sanitizers']
    check(len(sanitizer['records']) == 6, 'Incomplete sanitizer matrix')
    for record in sanitizer['records']:
        gates.validate_suite(session, record)
    gates.validate_metal_diagnostic(session, sanitizer['unsuppressed'])
    for name, digest in sanitizer['artifacts'].items():
        check(sha(out / name) == digest, 'Sanitizer artifact changed')
    parity = records['parity']
    tasks = {name: (source, args) for name, source, args in p.parity_tasks()}
    check(len(parity['rows']) == len(tasks) and {r['name'] for r in parity['rows']} == set(tasks), 'Incomplete parity matrix')
    for row in parity['rows']:
        source, args = tasks[row['name']]
        check(row['source'] == str(source) and row['source_sha256'] == sha(source) and row['args'] == args, 'Parity task changed')
        check(set(row['outputs']) == set(session.builds), 'Missing comparison output')
        check(len({v['sha256'] for v in row['outputs'].values()}) == 1, 'Codestreams differ')
        for label, value in row['outputs'].items():
            check(sha(out / value['file']) == value['sha256'], 'Codestream changed')
            check(sha(session.builds[label] / 'gjxl_encode') == value['encoder_sha256'], 'Encoder changed')
            check_command(session, value['command'])
    check(len(parity['decodes']) == 6, 'Missing pinned decodes')
    check_outputs(parity)
    for name in ('kodak-kodim17', 'imazen26-1029-planter-4k', 'padded-stress-4k'):
        decodes = [r for r in parity['decodes'] if r['name'] == name]
        check({r['label'] for r in decodes} == set(session.builds) and
              len({r['decoded_sha256'] for r in decodes}) == 1, 'Decoded pixels differ')
        for row in decodes:
            check(row['decoder_sha256'] == sha(session.decoder), 'Decoder changed')
            check_command(session, row['command'])
    retained = records['retained']
    expected = {(name, sample, caller, image) for name, callers in [('mixed', 1), ('cohort', 2)]
                for sample in range(2) for caller in range(callers) for image in range(4)}
    check(len(retained['rows']) == 24 and
          {(r['scenario'], r['sample'], r['caller'], r['image']) for r in retained['rows']} == expected, 'Incomplete retained-output matrix')
    check_outputs(retained)
    for row in retained['rows']:
        check(set(row['outputs']) == set(session.builds), 'Missing retained comparison')
        for field in ('compressed_sha256', 'decoded_sha256'):
            check(len({v[field] for v in row['outputs'].values()}) == 1, 'Retained outputs differ')
        for value in row['outputs'].values():
            check_command(session, value['command'])
    for scenario in ('mixed', 'cohort'):
        jobs = {}
        for label in session.builds:
            path = out / f'retained-{scenario}-{label}.json'
            args = read(path)['fingerprint']['args']
            jobs[label] = p.audit_job(path, label, args, 2, measured=False)
        check(p.signature(jobs['integrated']) == p.signature(jobs['candidate']), 'Retained job signature differs')
    conformance = records['conformance']
    check([r['label'] for r in conformance['rows']] == list(session.builds), 'Incomplete conformance matrix')
    for row in conformance['rows']:
        check_command(session, row['command'])
        check(row['command']['exit'] == 0 and 'All 22 pinned codestream conformance fixtures passed.' in
              (out / row['command']['log']).read_text(), 'Conformance failed')
    performance = records['performance']
    expected = {(name, pair) for name, _ in p.cases() for pair in range(p.pairs())}
    check(len(performance['rows']) == len(expected) and
          {(r['name'], r['pair']) for r in performance['rows']} == expected, 'Incomplete performance matrix')
    check(performance['profile'] == session.config['profile'] and performance['pairs'] == p.pairs() and performance['warmups'] == 2, 'Performance profile changed')
    for row in performance['rows']:
        check(row['args'] == dict(p.cases())[row['name']], 'Performance arguments changed')
        jobs = {label: p.audit_job(out / row['jobs'][label], label, row['args'], 9) for label in session.builds}
        check(jobs['integrated']['setup']['sources'] == jobs['candidate']['setup']['sources'] and
              p.signature(jobs['integrated']) == p.signature(jobs['candidate']), 'Measured output differs')
    pressure = records['pressure']
    repeats = 3 if session.config['profile'] == 'full' else 1
    expected = {(name, cpu, limit, repeat) for name, _, limits in p.pressure_tasks()
                for cpu in (1, 3) for limit in limits for repeat in range(repeats)}
    check(len(pressure['rows']) == len(expected) and
          {(r['name'], r['cpu'], r['limit'], r['repeat']) for r in pressure['rows']} == expected, 'Incomplete pressure matrix')
    signatures = {}
    for row in pressure['rows']:
        args = next(args for name, args, _ in p.pressure_tasks() if name == row['name'])
        job = p.audit_job(out / row['job'], 'candidate', args + ['--cpu-limit', str(row['cpu']), '--limit', row['limit']], 5)
        if row['limit'] == 'tight' and job['setup']['batch']:
            check(job['setup']['planned_slots'] == 1, 'Tight batch admitted extra slots')
        signature = p.signature(job)
        check(signature == signatures.setdefault(row['name'], signature), 'Pressure changed outputs')
    check(p.report(write=False) == records['performance-report'], 'Report arithmetic changed')
    print('All selected-profile gates and raw records audited.', flush=True)


def seal(session):
    audit(session)
    files = {}
    for path in sorted(session.out.rglob('*')):
        rel = path.relative_to(session.out)
        if (path.is_file() and rel.parts[0] not in ('builds', 'sources', 'metal-cpp-repository') and
                path.name not in ('validation.json', 'execution-events.jsonl') and '__pycache__' not in rel.parts):
            files[str(rel)] = sha(path)
    save(session.out / 'validation.json', {'status': 'complete', 'profile': session.config['profile'],
        'full_qualification': session.config['profile'] == 'full', 'hashes': files})


def verify(session):
    record = read(session.out / 'validation.json')
    check(record['profile'] == session.config['profile'] and
          record['full_qualification'] == (session.config['profile'] == 'full'), 'Qualification scope changed')
    for name, digest in record['hashes'].items():
        check(sha(session.out / name) == digest, 'Sealed artifact changed: ' + name)
    audit(session)
    print('Sealed ' + record['profile'] + ' qualification verified without rewriting evidence.', flush=True)
