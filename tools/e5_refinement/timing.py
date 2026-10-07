#!/usr/bin/env python3
"""AC-powered paired timing of the frozen e5 variants at measured near-80 quality.

Adapted from tools/context_map/latency.py. All collection is explicit, serial,
resumable, and hash-checked. Analysis consumes retained evidence only.
"""
import argparse
from collections import defaultdict
import fcntl
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time

import numpy as np

import qualify as q

PROTOCOL = {
    'version': 1, 'effort': 5, 'threads': 8, 'rounds': 8, 'blocks': 2,
    'aa_rounds': [1, 5], 'warmups': 2, 'samples': 5,
    'boundary': 'warm complete synchronous EncodeLinearRgbVarDctCodestream public call; '
                'excludes input loading, process/Metal initialization, output hashing, and file writing',
    'quality': 'Eight original strict target pairs plus four secondary saved-probe matches. '
               'Each score in [79.9,80.1], pair difference <=0.05; '
               'the four secondary pair differences <=0.01. Distances independently measured per arm.',
    'order': 'Rotate image order by round; alternate reference/test process order by round plus original image index. '
             'One same-baseline A/A pair per image before its A/B pair in rounds 1 and 5.',
    'monitor_interval_seconds': 1, 'quiet_seconds': 20, 'maximum_idle_wait_seconds': 600,
    'maximum_attempts_per_pair': 3, 'maximum_other_process_cpu_percent': 80,
    'maximum_aggregate_other_cpu_percent': 200, 'minimum_free_bytes': 5 * 1024**3,
    'exclusion': 'Reject whole pair on competing codec/build processes, declared CPU pressure, '
                 'non-AC power, thermal/performance warnings, swap growth from session start, '
                 'or monitor error. Decisions never use measured latency.',
    'aggregation': 'Median of five calls per process; median of eight paired ratios per image; '
                   'equal-image geometric mean across the fixed cohort. Also report both blocks '
                   'and strict/secondary strata separately. No pooling of absolute image times.',
    'uncertainty': '10000 stratified bootstrap draws resampling process pairs within each fixed image; '
                   'conditional 95% interval, not a population confidence interval. Seed 20261007.',
    'timing_gates': {'maximum_aggregate_ratio': .85, 'maximum_image_ratio': 1.05,
                     'upper_conditional_95_ratio_below': 1.0,
                     'aa_aggregate_ratio_interval': [.97, 1.03],
                     'aa_image_ratio_interval': [.90, 1.10]},
    'scope': 'Single-image warm calls on this M4 Pro. No batch, cold-start, CUDA, or paper-row refresh.',
}


def output(argv):
    return subprocess.check_output(argv, text=True, timeout=10)


def freeze(root, reference):
    path = root / 'manifest.json'
    if path.exists():
        m = q.read(path)
        q.check(m['reference'] == str(reference) and m['protocol'] == PROTOCOL, 'Timing configuration changed')
    else:
        study = q.Study(reference)
        m0 = study.m
        q.check(q.read(reference / 'analysis.json')['summary']['rate_complete'], 'Incomplete rate study')
        pairs = defaultdict(dict)
        for row in q.rows(reference / 'near-target-matches.jsonl'):
            q.check(row['arm'] not in pairs[row['image_id']], 'Duplicate quality arm')
            pairs[row['image_id']][row['arm']] = row
        q.check(set(pairs) == set(m0['selected']) and len(pairs) == 12, 'Wrong timing cohort')
        strict = secondary = 0
        for name, arms in pairs.items():
            q.check(set(arms) == {'baseline', 'candidate'}, 'Missing quality arm')
            a, b = arms['baseline'], arms['candidate']
            q.check(abs(a['score']-80) <= .1 and abs(b['score']-80) <= .1, 'Target neighborhood changed')
            q.check(a['matching_kind'] == b['matching_kind'], 'Matching kind differs')
            tolerance = .01 if a['matching_kind'] == 'secondary-saved-probe-match' else .05
            q.check(abs(a['score']-b['score']) <= tolerance, 'Quality mismatch')
            secondary += a['matching_kind'] == 'secondary-saved-probe-match'
            strict += a['matching_kind'] == 'strict-target'
            for row in arms.values():
                q.check(study.data[row['id']]['output_sha256'] == row['output_sha256'], 'Quality identity changed')
        q.check((strict, secondary) == (8, 4), 'Changed matching strata')
        files = dict(m0['files'])
        for name in ['manifest.json', 'near-target-selection.json', 'near-target-matches.jsonl',
                     'analysis.json', 'controls.json', 'candidate-tests.xml', 'batch-checks.json']:
            files[str(reference/name)] = q.sha(reference/name)
        for arms in pairs.values():
            for row in arms.values():
                files[row['output_path']] = row['output_sha256']
                files[row['raw_path']] = row['raw_sha256']
        for p in [Path(__file__).resolve(), Path(q.__file__).resolve()]:
            files[str(p)] = q.sha(p)
        q.check(output(['git', 'rev-parse', 'HEAD']).strip() == m0['source']['baseline_revision'], 'HEAD changed')
        q.check(subprocess.check_output(['git', 'diff', 'HEAD']) == (reference/'source.patch').read_bytes(), 'Policy patch changed')
        m = dict(reference=str(reference), protocol=PROTOCOL, files=files, pairs=pairs,
                 images={i['image_id']: i for i in m0['images'] if i['image_id'] in pairs},
                 image_order=m0['selected'], baseline=m0['baseline'], candidate=m0['candidate'],
                 source=m0['source'], hardware=output(['sysctl','-n','hw.model','hw.memsize','hw.ncpu','machdep.cpu.brand_string']),
                 system=output(['uname','-a']), power_settings=output(['pmset','-g','custom']),
                 created=time.time(), numpy=np.__version__)
        q.save(path, m)
    for p, digest in m['files'].items():
        q.check(q.sha(p) == digest, f'Frozen identity changed: {p}')
    return m


def cpu_seconds(value):
    return sum(float(part) * 60**i for i, part in enumerate(reversed(value.split(':'))))


def competing(comm):
    name = Path(comm).name.lower()
    return (name.startswith(('gjxl_', 'gjxl-', 'clang', 'gcc', 'g++', 'rustc')) or
            name in {'cjxl','djxl','ssimulacra2','butteraugli','butteraugli_main','cjxl-quality-metric',
                     'cmake','ctest','ninja','make','xcodebuild','cargo','cc','c++'})


class Monitor:
    def __init__(self, root, manifest):
        self.root, self.manifest = root, manifest
        self.stop, self.lock = threading.Event(), threading.Lock()
        self.snapshots, self.previous = [], {}
        self.previous_time, self.initial_swap = None, None
        self.thread = threading.Thread(target=self.run)

    def sample(self):
        with self.lock:
            now = time.time()
            row = {'timestamp': now, 'issues': []}
            try:
                raw = output(['ps','-axo','pid,ppid,pcpu,time,comm'])
                power = output(['pmset','-g','batt'])
                thermal = output(['pmset','-g','therm'])
                swap = output(['sysctl','-n','vm.swapusage'])
                used = float(re.search(r'used\s*=\s*([\d.]+)M', swap).group(1))
                if self.initial_swap is None:
                    self.initial_swap = used
                row.update(processes=raw, power=power, thermal=thermal, swap=swap,
                           swap_used_mb=used, initial_swap_mb=self.initial_swap)
                if used > self.initial_swap:
                    row['issues'].append('swap grew from session start')
                current, other = {}, []
                for line in raw.splitlines()[1:]:
                    fields = line.strip().split(None, 4)
                    if len(fields) != 5:
                        continue
                    pid, ppid, pcpu, elapsed, comm = fields
                    pid, ppid, seconds = int(pid), int(ppid), cpu_seconds(elapsed)
                    current[pid] = (seconds, comm)
                    if pid == os.getpid() or (ppid == os.getpid() and comm in [self.manifest['baseline'], self.manifest['candidate']]):
                        continue
                    cpu = float(pcpu)
                    previous = self.previous.get(pid)
                    if previous and previous[1] == comm and self.previous_time is not None and now-self.previous_time >= .75:
                        cpu = max(0., 100*(seconds-previous[0])/(now-self.previous_time))
                    other.append(dict(pid=pid, cpu_percent=cpu, command=comm))
                    if competing(comm):
                        row['issues'].append(f'competing process: {pid} {comm}')
                    if cpu > PROTOCOL['maximum_other_process_cpu_percent']:
                        row['issues'].append(f'other CPU pressure: {pid} {cpu:.1f}% {comm}')
                if self.previous_time is None or now-self.previous_time >= .75:
                    self.previous, self.previous_time = current, now
                row['aggregate_other_cpu_percent'] = sum(r['cpu_percent'] for r in other)
                row['top_other_cpu'] = sorted(other, key=lambda r:-r['cpu_percent'])[:8]
                if row['aggregate_other_cpu_percent'] > PROTOCOL['maximum_aggregate_other_cpu_percent']:
                    row['issues'].append('aggregate other CPU pressure')
                if "Now drawing from 'AC Power'" not in power:
                    row['issues'].append('not on AC power')
                if ('No thermal warning level has been recorded' not in thermal or
                    'No performance warning level has been recorded' not in thermal):
                    row['issues'].append('thermal/performance status requires review')
            except Exception as error:
                row['issues'].append(f'monitor failure: {error!r}')
            q.append(self.root/'environment.jsonl', row)
            self.snapshots.append(row)
            return row

    def run(self):
        while not self.stop.wait(PROTOCOL['monitor_interval_seconds']):
            self.sample()

    def quiet(self):
        start = clean_since = time.time()
        last_message = 0
        while time.time()-start < PROTOCOL['maximum_idle_wait_seconds']:
            row = self.sample()
            if row['issues']:
                clean_since = time.time()
                if time.time()-last_message >= 10:
                    print('waiting for idle:', '; '.join(row['issues'][:2]), flush=True)
                    last_message = time.time()
            elif time.time()-clean_since >= PROTOCOL['quiet_seconds']:
                return
            time.sleep(1)
        raise RuntimeError('Idle admission timed out; retain evidence and resume after contention ends.')


def encode(root, m, image, variant, folder):
    q.check(shutil_free(root) > PROTOCOL['minimum_free_bytes'], 'Free disk floor reached')
    folder.mkdir(parents=True, exist_ok=False)
    selected = m['pairs'][image][variant]
    argv = [m[variant], '--input', m['images'][image]['pfm_path'], '--output', str(folder/'out.jxl'),
            '--raw-samples', str(folder/'raw.json'), '--effort', '5', '--num-threads', '8',
            '--distance', format(selected['distance'], '.9g'), '--warmups', '2', '--samples', '5']
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GJXL_','RCA_','MTL_','METAL_','DYLD_'))}
    start = time.time()
    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=180)
    q.append(root/'commands.jsonl', dict(argv=argv, started=start, seconds=time.time()-start,
             returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    result.check_returncode()
    raw = q.read(folder/'raw.json')
    q.check(raw['timing_semantics']=='complete-encode-wall-time' and not raw['stage_profile_enabled'] and
            raw['backend']=='metal' and raw['gpu_aq_mode']=='fully-resident' and raw['effort']==5 and
            raw['thread_count']==8 and raw['warmups']==2 and raw['sample_count']==5 and raw['validation_encodes']==1 and
            raw['density']=='default' and raw['compression']=='automatic' and not raw['collect_final_score'] and
            raw['resampling']==1 and raw['dc_quantization']=='prediction-aware' and raw['adaptive_dc_smoothing'] and
            raw['dc_prediction']=='weighted' and
            (raw['input_width'],raw['input_height'])==(m['images'][image]['width'],m['images'][image]['height']) and
            abs(raw['requested_distance']-selected['distance'])<1e-8, 'Unexpected encoder settings')
    digest = q.sha(folder/'out.jxl')
    q.check(digest==selected['output_sha256'], 'Timing output differs from qualified bytes')
    q.check(all(r['encoded_bytes']==selected['encoded_bytes'] and r['elapsed_nanoseconds']>0 for r in raw['samples']), 'Invalid sample')
    values = [r['elapsed_nanoseconds'] for r in raw['samples']]
    return dict(variant=variant, distance=selected['distance'], score=selected['score'],
                samples_ns=values, median_ns=statistics.median(values),
                raw_path=str(folder/'raw.json'),raw_sha256=q.sha(folder/'raw.json'),
                output_path=str(folder/'out.jxl'),output_sha256=digest)


def shutil_free(root):
    import shutil
    return shutil.disk_usage(root).free


def schedule(m):
    for round_ in range(PROTOCOL['rounds']):
        images=m['image_order'][round_%12:]+m['image_order'][:round_%12]
        for image in images:
            for kind in (['aa','ab'] if round_ in PROTOCOL['aa_rounds'] else ['ab']):
                yield dict(id=q.key(kind,image,round_),kind=kind,image_id=image,round=round_,block=round_//4,
                           order=['reference','test'] if (round_+m['image_order'].index(image))%2==0 else ['test','reference'])


def collect(root,m):
    old=q.rows(root/'attempts.jsonl')
    accepted={r['id'] for r in old if r['accepted']}
    monitor=Monitor(root,m)
    monitor.thread.start()
    try:
        monitor.quiet()
        for case in schedule(m):
            if case['id'] in accepted:
                continue
            previous=[r for r in q.rows(root/'attempts.jsonl') if r['id']==case['id']]
            for attempt in range(len(previous),PROTOCOL['maximum_attempts_per_pair']):
                pre=monitor.sample()
                if pre['issues']:
                    monitor.quiet()
                    pre=monitor.sample()
                start=pre['timestamp']
                arms={}
                for slot in case['order']:
                    variant='candidate' if slot=='test' and case['kind']=='ab' else 'baseline'
                    folder=root/'cases'/f"{case['id']}-{attempt}-{slot}"
                    arms[slot]=encode(root,m,case['image_id'],variant,folder)
                post=monitor.sample()
                end=post['timestamp']
                with monitor.lock:
                    evidence=[r for r in monitor.snapshots if start<=r['timestamp']<=end]
                issues=sorted({issue for r in evidence for issue in r['issues']})
                record={**case,'attempt':attempt,'started':start,'ended':end,'arms':arms,
                        'accepted':not issues,'issues':issues,'environment_samples':len(evidence)}
                q.append(root/'attempts.jsonl',record)
                if not issues:
                    accepted.add(case['id'])
                q.save(root/'progress.json',dict(accepted=len(accepted),expected=120,case=case,issues=issues,time=time.time()))
                print('accepted' if not issues else 'excluded',len(accepted),'/120',case['kind'],case['image_id'],
                      'round',case['round'],'attempt',attempt,issues[:2],flush=True)
                if not issues:
                    break
                monitor.quiet()
            else:
                raise RuntimeError(f"Attempt budget exhausted for {case['id']}; retain all attempts")
    finally:
        monitor.stop.set()
        monitor.thread.join()


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def summarize(root,m):
    attempts=q.rows(root/'attempts.jsonl')
    accepted=[r for r in attempts if r['accepted']]
    expected={r['id']:r for r in schedule(m)}
    q.check(len(accepted)==len({r['id'] for r in accepted})==120 and {r['id'] for r in accepted}==set(expected),'Incomplete accepted coverage')
    environment=q.rows(root/'environment.jsonl')
    for row in attempts:
        evidence=[r for r in environment if row['started']<=r['timestamp']<=row['ended']]
        issues=sorted({issue for r in evidence for issue in r['issues']})
        q.check(evidence and len(evidence)==row['environment_samples'] and issues==row['issues'] and row['accepted']==(not issues),'Environment audit failed')
        for k in ('kind','image_id','round','block','order'):
            q.check(row[k]==expected[row['id']][k],'Schedule changed')
        for slot, arm in row['arms'].items():
            variant='candidate' if slot=='test' and row['kind']=='ab' else 'baseline'
            q.check(arm['variant']==variant and q.sha(arm['raw_path'])==arm['raw_sha256'] and
                    q.sha(arm['output_path'])==arm['output_sha256']==m['pairs'][row['image_id']][variant]['output_sha256'],'Artifact audit failed')
            raw=q.read(arm['raw_path'])
            q.check([r['elapsed_nanoseconds'] for r in raw['samples']]==arm['samples_ns'] and statistics.median(arm['samples_ns'])==arm['median_ns'],'Sample audit failed')
    commands=q.rows(root/'commands.jsonl')
    q.check(len(commands)==len(attempts)*2 and all(r['returncode']==0 for r in commands),'Unaccounted encoder invocation')
    images=[]
    for name in m['image_order']:
        ab=sorted([r for r in accepted if r['kind']=='ab' and r['image_id']==name],key=lambda r:r['round'])
        aa=sorted([r for r in accepted if r['kind']=='aa' and r['image_id']==name],key=lambda r:r['round'])
        q.check([r['round'] for r in ab]==list(range(8)) and [r['round'] for r in aa]==[1,5],'Wrong per-image coverage')
        ratios=[r['arms']['test']['median_ns']/r['arms']['reference']['median_ns'] for r in ab]
        controls=[r['arms']['test']['median_ns']/r['arms']['reference']['median_ns'] for r in aa]
        images.append(dict(image_id=name,matching_kind=m['pairs'][name]['baseline']['matching_kind'],
                           paired_ratios=ratios,median_ratio=statistics.median(ratios),
                           blocks=[statistics.median(ratios[:4]),statistics.median(ratios[4:])],
                           aa_ratios=controls,aa_median_ratio=statistics.median(controls),
                           baseline_ms=statistics.median(r['arms']['reference']['median_ns'] for r in ab)/1e6,
                           candidate_ms=statistics.median(r['arms']['test']['median_ns'] for r in ab)/1e6))
    rng=np.random.default_rng(20261007)
    boot=[]
    for _ in range(10000):
        boot.append(geometric_mean([float(np.median(rng.choice(r['paired_ratios'],size=8,replace=True))) for r in images]))
    ci=list(map(float,np.quantile(boot,[.025,.975])))
    ratio=geometric_mean([r['median_ratio'] for r in images])
    aa_ratio=geometric_mean([r['aa_median_ratio'] for r in images])
    gates=PROTOCOL['timing_gates']
    checks=dict(aggregate_gain=ratio<=gates['maximum_aggregate_ratio'],
                conditional_interval=ci[1]<gates['upper_conditional_95_ratio_below'],
                no_material_image_regression=max(r['median_ratio'] for r in images)<=gates['maximum_image_ratio'],
                aa_aggregate=gates['aa_aggregate_ratio_interval'][0]<=aa_ratio<=gates['aa_aggregate_ratio_interval'][1],
                aa_images=all(gates['aa_image_ratio_interval'][0]<=r['aa_median_ratio']<=gates['aa_image_ratio_interval'][1] for r in images))
    result=dict(status='passed' if all(checks.values()) else 'complete-with-failed-timing-gates',checks=checks,
                accepted_pairs=120,ab_pairs=96,aa_pairs=24,excluded_pairs=len(attempts)-120,
                accepted_processes=240,timed_calls=1200,total_accepted_encodes=1920,
                aggregate_ratio=ratio,time_change_percent=100*(ratio-1),throughput_factor=1/ratio,
                conditional_95_ratio=ci,conditional_95_time_change_percent=[100*(x-1) for x in ci],
                aa_aggregate_ratio=aa_ratio,aa_time_change_percent=100*(aa_ratio-1),
                block_ratios=[geometric_mean([r['blocks'][b] for r in images]) for b in (0,1)],
                strata={kind:dict(images=len([r for r in images if r['matching_kind']==kind]),
                                  ratio=geometric_mean([r['median_ratio'] for r in images if r['matching_kind']==kind]))
                        for kind in ('strict-target','secondary-saved-probe-match')},
                images=images,protocol=PROTOCOL,environment_snapshots=len(environment),
                accepted_swap_growth_mb=max(e['swap_used_mb']-e['initial_swap_mb'] for r in accepted for e in environment if r['started']<=e['timestamp']<=r['ended']),
                provenance={str(root/name):q.sha(root/name) for name in ['manifest.json','attempts.jsonl','commands.jsonl','environment.jsonl']})
    q.save(root/'analysis.json',result)
    print({k:v for k,v in result.items() if k not in ('images','protocol','provenance','strata')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','collect','report'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('build/e5-qualification'))
    args=p.parse_args()
    root,reference=args.root.resolve(),args.reference.resolve()
    root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock, (reference/'run.lock').open('a') as source_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(source_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=freeze(root,reference)
        if args.action=='collect':
            collect(root,m)
        elif args.action=='report':
            summarize(root,m)
        else:
            print('Frozen 96 A/B and 24 A/A pairs across 12 measured quality pairs',flush=True)


if __name__=='__main__':
    main()
