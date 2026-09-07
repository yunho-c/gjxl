"""Historical workload semantics with explicit session dependencies."""
import json
import math
import statistics
from pathlib import Path
from session import check, read, save, sha


def initialize(session):
    global S, out, root, corpus, decoder, info, builds, sources, run, identities, quiet
    global small, natural, p1080, p4k, planter, mixed
    S = session
    out, root, corpus, decoder, info = S.out, S.sources['candidate'], S.corpus, S.decoder, S.info
    builds, sources, run, identities, quiet = S.builds, S.sources, S.run, S.identities, S.quiet
    small = ['--synthetic', '17x9']
    natural = ['--input', str(corpus/'kodak-kodim17.pfm')]
    p1080 = ['--synthetic', '1919x1079']
    p4k = ['--synthetic', '3839x2159']
    planter = ['--input', str(corpus/'imazen26-1029-planter-4k.pfm')]
    mixed = small + natural + p1080 + p4k


def pairs():
    return 7 if S.config['profile'] == 'full' else 2


def pressure_tasks():
    tasks = [('single4k', p4k, ('tight',)), ('batch4k', p4k + ['--batch','4','--in-flight','4'], ('tight','full')),
             ('mixed', mixed + ['--batch','4','--in-flight','4'], ('tight','full')),
             ('two-callers', natural + ['--batch','4','--in-flight','4','--callers','2'], ('tight','full'))]
    return tasks if S.config['profile'] == 'full' else [tasks[0], tasks[3]]


def attempt_directory(stem):
    directory = S.unique(stem, '')
    directory.mkdir()
    return directory


def sample_job(stem, label, args, count=9, measured=True):
    driver = out/f'{label}-driver'
    fingerprint = {'driver_sha256': sha(driver), 'args': [str(x) for x in args], 'count': count, 'measured': measured, 'build_sha256': sha(out/'build.json')}
    path = out/(stem + '.json')
    if path.exists():
        saved = json.loads(path.read_text())
        assert saved['fingerprint'] == fingerprint and sha(out/saved['command']['log']) == saved['command']['log_sha256']
        return saved
    before = quiet() if measured else None
    command = run([driver, *args, '--count', count], stem, timeout=600)
    records = [json.loads(line) for line in (out/command['log']).read_text().splitlines() if line.startswith('{')]
    assert [r['type'] for r in records] == ['setup'] + ['sample'] * count + ['trim']
    result = {'fingerprint': fingerprint, 'before_processes': before, 'command': command,
              'setup': records[0], 'samples': records[1:-1], 'trim': records[-1]}
    assert [r['index'] for r in result['samples']] == list(range(count))
    save(path, result)
    return result

def signature(job):
    return [[[(i['source'], i['phase'], i['bytes'], i['fnv64']) for i in call['images']]
             for call in sample['calls']] for sample in job['samples']]

def smoke():
    identities()
    for name, args in [('small', small), ('batch', small + ['--batch', '4', '--in-flight', '2']),
                       ('cohort', small + natural + ['--batch', '4', '--in-flight', '2', '--callers', '2'])]:
        a = sample_job('smoke-' + name + '-integrated', 'integrated', args, count=3, measured=False)
        b = sample_job('smoke-' + name + '-candidate', 'candidate', args, count=3, measured=False)
        assert a['setup']['sources'] == b['setup']['sources'] and signature(a) == signature(b)
    sample_job('smoke-tight', 'candidate', small + natural + ['--batch', '4', '--in-flight', '2',
               '--callers', '2', '--limit', 'tight', '--cpu-limit', '1'], count=3, measured=False)
    print('Both revision-correct drivers and constrained cohort smoke checks passed.', flush=True)

def cases():
    result = [(name, args) for name, args in [('small', small), ('kodak17', natural), ('padded1080', p1080),
                                               ('padded4k', p4k), ('planter4k', planter)]]
    for name, args, batch, counts in [('small', small, 8, (1,4,8)), ('kodak17', natural, 8, (1,2,4,8)),
         ('padded1080', p1080, 4, (1,2,4)), ('padded4k', p4k, 4, (1,2,4)), ('mixed', mixed, 4, (1,2,4))]:
        for n in counts: result.append((f'{name}-batch{batch}-slots{n}', args + ['--batch', str(batch), '--in-flight', str(n)]))
    for threads in (1,4):
        result.append((f'padded4k-threads{threads}', p4k + ['--threads', str(threads)]))
        result.append((f'kodak17-batch8-slots4-threads{threads}', natural + ['--batch','8','--in-flight','4','--threads',str(threads)]))
    for name, args in [('kodak17', natural), ('mixed', mixed)]:
        result.append((f'{name}-two-callers', args + ['--batch','4','--in-flight','4','--callers','2']))
    return result if S.config['profile'] == 'full' else [item for item in result if item[0] in ('small', 'kodak17', 'padded4k', 'mixed-batch4-slots1', 'kodak17-two-callers')]

def performance():
    record = identities()
    rows = []
    for name, args in cases():
        for pair in range(pairs()):
            jobs = {}
            for label in (['integrated','candidate'] if pair % 2 == 0 else ['candidate','integrated']):
                jobs[label] = sample_job(f'perf-{name}-pair{pair}-{label}', label, args)
            assert jobs['integrated']['setup']['sources'] == jobs['candidate']['setup']['sources']
            assert signature(jobs['integrated']) == signature(jobs['candidate']), name
            rows.append({'name': name, 'pair': pair, 'args': args,
                         'jobs': {label: f'perf-{name}-pair{pair}-{label}.json' for label in jobs}})
            save(out/'performance.json', {'status': 'running', 'rows': rows, 'profile': S.config['profile'], 'warmups': 2, 'pairs': pairs(),
                                          'build_revision': record['candidate_revision']})
            print(f'Performance {name} pair {pair + 1}/{pairs()}', flush=True)
    save(out/'performance.json', {'status': 'complete', 'rows': rows, 'profile': S.config['profile'], 'warmups': 2, 'pairs': pairs(),
                                  'build_revision': record['candidate_revision']})
    report()

def pressure():
    identities()
    rows = []
    tasks = pressure_tasks()
    repeats = 3 if S.config['profile'] == 'full' else 1
    for name, args, limits in tasks:
        for cpu in (1,3):
            for limit in limits:
                for repeat in range(repeats):
                    stem = f'pressure-{name}-cpu{cpu}-{limit}-{repeat}'
                    job = sample_job(stem, 'candidate', args + ['--cpu-limit',str(cpu),'--limit',limit], count=5)
                    assert all(s['managed']['cpu_limit'] == cpu and s['managed']['cpu_protected'] <= cpu and
                               s['managed']['peak_committed'] <= job['setup']['managed_limit'] for s in job['samples'])
                    if limit == 'tight' and job['setup']['batch']:
                        assert job['setup']['planned_slots'] == 1
                    rows.append({'name':name,'cpu':cpu,'limit':limit,'repeat':repeat,'job':stem+'.json'})
                    save(out/'pressure.json', {'status':'running','rows':rows,'warmups':2})
                    print(f'Pressure {name} CPU {cpu} {limit} repeat {repeat+1}/{repeats}', flush=True)
    for name, _, _ in tasks:
        jobs = [json.loads((out/r['job']).read_text()) for r in rows if r['name'] == name]
        assert all(signature(job) == signature(jobs[0]) for job in jobs)
    save(out/'pressure.json', {'status':'complete','rows':rows,'warmups':2})

def parity_tasks():
    defaults = ['--backend','metal','--metal-aq','fully-resident','--distance','1.2','--effort','7']
    tasks = [(Path(name).stem, corpus/name, defaults) for name in sorted(S.config['inputs'])]
    for effort in range(1,11): tasks.append((f'policy-effort-{effort}',corpus/'kodak-kodim17.pfm',defaults[:-1]+[str(effort)]))
    for name, extra in [('high-density',['--high-density']),('maximum-compression',['--maximum-compression']),
                        ('final-score',['--collect-final-score'])]:
        tasks.append((f'policy-{name}',corpus/'kodak-kodim17.pfm',(defaults[:-2] if name == 'high-density' else defaults)+extra))
    for mode in ('exact-coefficients','throughput','maximum-throughput'):
        tasks.append(('policy-'+mode,corpus/'kodak-kodim17.pfm',
                     ['--backend','metal','--metal-aq',mode,'--distance','1.2','--effort','7']))
    tasks += [('policy-target',root/'testdata/codestream_sample.pfm',
               ['--backend','metal','--target-bytes','280','--size-tolerance','0.1','--max-attempts','8']),
              ('policy-maximum-error',corpus/'kodak-kodim17.pfm',['--backend','cpu','--maximum-error','0.05','0.05','0.05'])]
    assert len(tasks) == 56
    return tasks


def parity():
    identities()
    tasks = parity_tasks()
    directory = out/'parity'
    directory.mkdir(exist_ok=True)
    rows = []
    for name, source, args in tasks:
        path = directory/(name+'.json')
        if path.exists():
            row = json.loads(path.read_text())
            assert row['source_sha256'] == sha(source) and row['args'] == args
            for label, output in row['outputs'].items():
                assert sha(out/output['file']) == output['sha256'] and sha(builds[label]/'gjxl_encode') == output['encoder_sha256']
        else:
            row = {'name':name,'source':str(source),'source_sha256':sha(source),'args':args,'outputs':{}}
            attempt = attempt_directory('parity-' + name)
            for label, build in builds.items():
                dest = attempt/(label+'.jxl')
                assert not dest.exists(), f'Review incomplete retained output before retrying: {dest}'
                command = run([build/'gjxl_encode',*args,source,dest],f'parity-{name}-{label}')
                row['outputs'][label] = {'file':str(dest.relative_to(out)),'sha256':sha(dest),'bytes':dest.stat().st_size,'command':command,
                                         'encoder_sha256':sha(build/'gjxl_encode')}
            assert row['outputs']['integrated']['sha256'] == row['outputs']['candidate']['sha256'], name
            save(path,row)
        rows.append(row)
        save(out/'parity.json',{'status':'running','rows':rows})
        print('Canonical/policy bytes match:',name,flush=True)
    decodes = []
    for name in ('kodak-kodim17','imazen26-1029-planter-4k','padded-stress-4k'):
        hashes = []
        for label in builds:
            compressed = out/next(r for r in rows if r['name'] == name)['outputs'][label]['file']
            decoded = compressed.with_suffix('.decoded.pfm')
            sidecar = decoded.with_suffix('.json')
            if sidecar.exists():
                record = json.loads(sidecar.read_text())
                assert sha(decoded) == record['decoded_sha256'] and sha(compressed) == record['compressed_sha256']
                assert sha(decoder) == record['decoder_sha256']
            else:
                if decoded.exists(): decoded.rename(S.unique('incomplete-decoded-' + name + '-' + label, '.pfm'))
                command = run([decoder,'--quiet','--num_threads=0',compressed,decoded],f'decode-{name}-{label}')
                record = {'name':name,'label':label,'command':command,'decoded':str(decoded),
                          'decoded_sha256':sha(decoded),'compressed_sha256':sha(compressed),'decoder_sha256':sha(decoder)}
                save(sidecar,record)
            hashes.append(record['decoded_sha256']); decodes.append(record)
        assert hashes[0] == hashes[1], name
    save(out/'parity.json',{'status':'complete','rows':rows,'decodes':decodes})

def conformance():
    identities()
    if (out/'conformance.json').exists():
        previous = read(out/'conformance.json')
        assert previous['status'] == 'complete' and len(previous['rows']) == 2
        for row in previous['rows']:
            command = row['command']
            assert sha(out/command['log']) == command['log_sha256']
            assert sha(builds[row['label']]/'gjxl_encode') == row['encoder_sha256']
            assert sha(builds[row['label']]/'gjxl_codestream_conformance_test') == row['test_sha256']
        return
    rows = []
    for label, build in builds.items():
        artifacts = attempt_directory('conformance-'+label)
        command = run([build/'gjxl_codestream_conformance_test','--decoder',decoder,'--info',info,
                       '--encoder',build/'gjxl_encode','--sample',root/'testdata/codestream_sample.pfm',
                       '--artifacts',artifacts],f'conformance-{label}')
        assert 'All 22 pinned codestream conformance fixtures passed.' in (out/command['log']).read_text()
        rows.append({'label':label,'command':command,'encoder_sha256':sha(build/'gjxl_encode'),
                     'test_sha256':sha(build/'gjxl_codestream_conformance_test')})
    save(out/'conformance.json',{'status':'complete','rows':rows,'decoder_sha256':sha(decoder)})

def retained():
    identities()
    rows = []
    scenarios = [('mixed',mixed+['--batch','4','--in-flight','4']),
                 ('cohort',natural+['--batch','4','--in-flight','4','--callers','2'])]
    for name, args in scenarios:
        jobs = {}
        for label in builds:
            cached = out/f'retained-{name}-{label}.json'
            if cached.exists():
                args_saved = read(cached)['fingerprint']['args']
                directory = Path(args_saved[args_saved.index('--retain') + 1])
            else:
                directory = attempt_directory(f'retained-{name}-{label}')
            jobs[label] = sample_job(f'retained-{name}-{label}',label,args+['--retain',str(directory)],count=2,measured=False)
        assert signature(jobs['integrated']) == signature(jobs['candidate'])
        for sample in range(2):
            for caller in range(jobs['candidate']['setup']['callers']):
                for image in range(jobs['candidate']['setup']['batch']):
                    stem = f'sample-{sample}-caller-{caller}-image-{image}'
                    outputs = {}
                    for label in builds:
                        saved_args = jobs[label]['fingerprint']['args']
                        directory = Path(saved_args[saved_args.index('--retain') + 1])
                        compressed, decoded = directory/(stem+'.jxl'), directory/(stem+'.decoded.pfm')
                        sidecar = directory/(stem+'.decode.json')
                        if sidecar.exists():
                            record = json.loads(sidecar.read_text())
                            assert sha(compressed) == record['compressed_sha256'] and sha(decoded) == record['decoded_sha256']
                            assert sha(decoder) == record['decoder_sha256']
                        else:
                            if decoded.exists(): decoded.rename(S.unique('incomplete-decoded-' + name + '-' + label, '.pfm'))
                            command = run([decoder,'--quiet','--num_threads=0',compressed,decoded],f'retained-{name}-{label}-{stem}-decode')
                            record = {'command':command,'compressed':str(compressed),'compressed_sha256':sha(compressed),
                                      'decoded':str(decoded),'decoded_sha256':sha(decoded),'decoder_sha256':sha(decoder)}
                            save(sidecar,record)
                        outputs[label] = record
                    assert outputs['integrated']['compressed_sha256'] == outputs['candidate']['compressed_sha256']
                    assert outputs['integrated']['decoded_sha256'] == outputs['candidate']['decoded_sha256']
                    rows.append({'scenario':name,'sample':sample,'caller':caller,'image':image,'outputs':outputs})
        print('Retained changed-image batch/cohort decoded parity:',name,flush=True)
    save(out/'retained.json',{'status':'complete','rows':rows})

def distribution(values):
    values = sorted(values)
    return {'min': values[0], 'median': statistics.median(values), 'p95': values[math.ceil(len(values)*0.95)-1],
            'max': values[-1], 'count': len(values)}

def audit_job(path, label, args, count, measured=True):
    job = json.loads(path.read_text())
    assert job['fingerprint'] == {'driver_sha256':sha(out/f'{label}-driver'),
                                  'args':[str(x) for x in args], 'count':count, 'measured':measured, 'build_sha256':sha(out/'build.json')}, path
    command = job['command']
    assert command['exit'] == 0 and command['command'] == [str(out/f'{label}-driver'), *map(str,args), '--count',str(count)]
    assert sha(out/command['log']) == command['log_sha256']
    raw = [json.loads(line) for line in (out/command['log']).read_text().splitlines() if line.startswith('{')]
    assert raw == [job['setup'],*job['samples'],job['trim']]
    assert [r['type'] for r in raw] == ['setup'] + ['sample'] * count + ['trim']
    assert (job['before_processes'] is not None) == measured
    setup = job['setup']
    for index, sample in enumerate(job['samples']):
        assert sample['index'] == index and sample['cohort_ns'] > 0
        assert len(sample['calls']) == setup['callers']
        for caller, call in enumerate(sample['calls']):
            assert call['caller'] == caller and 0 < call['wall_ns'] <= sample['cohort_ns']
            assert len(call['images']) == (setup['batch'] or 1)
            for number, image in enumerate(call['images']):
                size = len(setup['sources'])
                source = (number+caller) % size if setup['batch'] else index % size
                phase = (index+number+caller) % 2 if setup['batch'] else (index//size) % 2
                assert image['source'] == source and image['phase'] == phase and image['bytes'] > 0
                timing = image['scheduling']
                if S.config['capabilities'][label] == 'current' and setup['batch']:
                    assert timing['queue_ns'] >= 0 and timing['service_ns'] > 0
                    assert timing['queue_ns'] + timing['service_ns'] == timing['ready_ns'] <= call['wall_ns']
                else: assert timing is None
        managed = sample['managed']
        if S.config['capabilities'][label] == 'current':
            assert 0 < managed['cpu_peak'] <= managed['cpu_protected'] <= managed['cpu_limit']
            if setup['cpu_limit_requested']: assert managed['cpu_limit'] == setup['cpu_limit_requested']
            if setup['managed_limit']: assert managed['peak_committed'] <= setup['managed_limit']
        else: assert managed is None
    return job

def report(write=True):
    data = json.loads((out/'performance.json').read_text())
    assert data['status'] == 'complete' and len(data['rows']) == len(cases()) * pairs()
    report_rows = []
    for name, _ in cases():
        selected = [r for r in data['rows'] if r['name'] == name]
        medians = {label: [] for label in builds}
        spans = {key: [] for key in ('queue_ns','service_ns','ready_ns')}
        memory = {label: [] for label in builds}
        setups = {label: [] for label in builds}
        public_calls = {label: [] for label in builds}
        by_source = {}
        for row in selected:
            jobs = {label: json.loads((out/path).read_text()) for label,path in row['jobs'].items()}
            assert signature(jobs['integrated']) == signature(jobs['candidate'])
            for label, job in jobs.items():
                medians[label].append(statistics.median(s['cohort_ns']/1e6 for s in job['samples'][2:]))
                memory[label].append(job['trim'])
                setups[label].append(job['setup']['setup_ns']/1e6)
                public_calls[label].extend(call['wall_ns']/1e6 for s in job['samples'][2:] for call in s['calls'])
                if label == 'candidate':
                    for sample in job['samples'][2:]:
                        for call in sample['calls']:
                            for image in call['images']:
                                if image['scheduling']:
                                    for key in spans: spans[key].append(image['scheduling'][key]/1e6)
                                    source = job['setup']['sources'][image['source']]['name']
                                    source_spans = by_source.setdefault(source,{key:[] for key in spans})
                                    for key in spans: source_spans[key].append(image['scheduling'][key]/1e6)
        changes = [(b/a-1)*100 for a,b in zip(medians['integrated'],medians['candidate'])]
        report_rows.append({'name':name,'process_medians_ms':medians,'paired_changes_percent':changes,
                            'median_change_percent':statistics.median(changes),
                            'image_spans_ms':{key.removesuffix('_ns'):distribution(values) for key,values in spans.items() if values},
                            'source_image_spans_ms':{source:{key.removesuffix('_ns'):distribution(values) for key,values in values_by_key.items()}
                                                    for source,values_by_key in by_source.items()},
                            'public_calls_ms':{label:distribution(values) for label,values in public_calls.items()},
                            'memory':memory,'backend_setup_ms':setups})
    result = {'status':'complete', 'profile':S.config['profile'], 'rows':report_rows}
    if write: save(out/'performance-report.json', result)
    for row in report_rows:
        print(f"{row['name']}: {row['median_change_percent']:+.2f}% median paired change", flush=True)

    return result
