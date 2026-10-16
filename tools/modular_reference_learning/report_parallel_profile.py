#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Summarize public workflow controls, wave lifetimes, fitting probes and Samply."""
import argparse
import json
import statistics
from pathlib import Path

from run_parallel_profile import validate
from run_hybrid_qualification import require, digest

STAGES = ['input', 'selection', 'training', 'tokens', 'model', 'emission', 'assembly',
          'gpu_input_tokens', 'stream_encoding']


def schedule_metrics(result):
    trace = result['learning_schedule']
    wall = active = slots = 0.0
    last_jobs = []
    rows = []
    for w in trace['waves']:
        jobs = trace['jobs'][w['first_job']:w['first_job']+w['jobs']]
        duration = w['joined']-w['dispatch']
        work = sum(j['finish']-j['start'] for j in jobs)
        wall += duration
        active += work
        slots += w['participants']*duration
        last = max(jobs, key=lambda j:j['finish'])
        last_jobs.append(last)
        rows.append(dict(jobs=w['jobs'], participants=w['participants'], seconds=duration,
                         active_job_seconds=work, last_stream=last['stream'],
                         last_preparation=last['preparation'], last_fitting=last['fitting']))
    return dict(waves=len(rows), slots=len(trace['jobs']), real_fits=sum(j['fits'] for j in trace['jobs']),
                joined_wave_seconds=wall, active_job_seconds=active,
                average_active_jobs=active/wall,
                active_fraction_of_granted_slots=active/slots,
                active_fraction_of_14_slots=active/(14*wall),
                underfilled_waves=sum(w['participants']<14 for w in rows),
                reservation_and_cleanup_seconds=sum(w['finish']-w['start']-(w['joined']-w['dispatch']) for w in trace['waves']),
                last_job_preparation_seconds=sum(j['preparation'] for j in last_jobs),
                last_job_fitting_seconds=sum(j['fitting'] for j in last_jobs),
                wave_details=rows)


def critical_chain(result):
    """Observed last-finishing worker in each wave, not a counterfactual speedup."""
    trace = result['learning_schedule']
    fits = iter(result['histogram_fitting']['fits'])
    by_stream = {j['stream']:next(fits) for j in trace['jobs'] if j['fits']}
    require(next(fits,None) is None, 'Schedule/fit mapping lost a record')
    phases = {}
    preparation = job_fitting = fit_envelope = joined = 0
    for w in trace['waves']:
        jobs = trace['jobs'][w['first_job']:w['first_job']+w['jobs']]
        worker = max(jobs,key=lambda j:j['finish'])['worker']
        joined += w['joined']-w['dispatch']
        for j in jobs:
            if j['worker'] != worker:
                continue
            preparation += j['preparation']
            job_fitting += j['fitting']
            if j['fits']:
                fit = by_stream[j['stream']]
                fit_envelope += fit['seconds']
                for k,v in fit['stages'].items():
                    phases[k] = phases.get(k,0)+v
    require(preparation+job_fitting <= joined+1e-6, 'Last-worker chain exceeded joined waves')
    return dict(joined_wave_seconds=joined,preparation_seconds=preparation,
                fitting_seconds=job_fitting,native_fitter_seconds=fit_envelope,
                fitter_phase_seconds=phases,
                note='Per wave, phases of jobs on its observed last-finishing worker. Includes preemption; '
                     'does not predict which worker becomes critical after optimization.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--plot', type=Path, help='Optional SVG of representative 1039 wave occupancy')
    args = parser.parse_args()
    capture = args.capture.resolve()
    manifest = json.loads((capture/'manifest.json').read_text())
    require(manifest['status']=='passed', 'Capture did not complete')
    rows = [json.loads(line) for line in (capture/'results.jsonl').read_text().splitlines()]
    for r in rows:
        validate(r['result'])
    summary = []
    for image in sorted({r['input'] for r in rows}):
        for route in ['cpu', 'auto']:
            for mode in ['fit-off', 'schedule', 'fit-on']:
                selected = [r for r in rows if r['input']==image and r['route']==route and r['mode']==mode
                            and not r['warmup'] and not r['sampled']]
                require(len(selected)==manifest['repeats'], 'Incomplete measured condition')
                results = [r['result'] for r in selected]
                median = lambda fn: statistics.median(fn(x) for x in results)
                value = dict(input=image, route=route, mode=mode, repeats=len(selected),
                             seconds=median(lambda x:x['total']['wall']),
                             seconds_range=[min(x['total']['wall'] for x in results), max(x['total']['wall'] for x in results)],
                             cpu_seconds=median(lambda x:x['total']['cpu']),
                             stages={name:median(lambda x:x['profile_wall'][i]) for i,name in enumerate(STAGES)},
                             stage_cpu={name:median(lambda x:x['profile_cpu'][i]) for i,name in enumerate(STAGES)},
                             learning_seconds=median(lambda x:x['detail_wall']['learning']),
                             feature_seconds=median(lambda x:x['detail_wall']['features']),
                             preparation_work_seconds=median(lambda x:x['detail_work']['sample_preparation']),
                             fitting_work_seconds=median(lambda x:x['detail_work']['fitting']),
                             routing=results[0]['routing'], work=results[0]['work'],
                             bytes=results[0]['bytes'], output_sha256=selected[0]['output_sha256'],
                             peak_backing_bytes=max(x['memory']['peak_backing_bytes'] for x in results),
                             peak_committed_bytes=max(x['memory']['peak_committed_bytes'] for x in results))
                if mode != 'fit-off':
                    representative = sorted(selected,key=lambda r:r['result']['detail_wall']['learning'])[len(selected)//2]
                    value['schedule_observation'] = representative['directory']
                    value['schedule'] = schedule_metrics(representative['result'])
                if mode == 'fit-on':
                    value['observed_last_worker_chain'] = critical_chain(representative['result'])
                    profiles = [x['histogram_fitting'] for x in results]
                    total = sum(x['fit_seconds'] for x in profiles)
                    value['fitting_phases_percent'] = {k:100*sum(x['stages'][k] for x in profiles)/total for k in profiles[0]['stages']}
                    buckets = {}
                    for p in profiles:
                        for b in p['by_rows']:
                            item = buckets.setdefault(b['maximum_rows'], dict(nodes=0, seconds=0))
                            item['nodes'] += b['work']['nodes']
                            item['seconds'] += b['node_seconds']
                    value['nodes_by_rows'] = [dict(maximum_rows=k,nodes_per_call=v['nodes']/len(profiles),
                                                   fitting_work_percent=100*v['seconds']/total) for k,v in sorted(buckets.items())]
                summary.append(value)
    sampling = []
    for r in rows:
        if r['sampled']:
            path = capture/(r['directory']+'-analysis.json')
            data = json.loads(path.read_text())
            # Full per-thread records remain in the hashed analysis artifact.
            for a in data['attribution'].values():
                a['attributed_threads'] = len(a.pop('threads'))
            sampling.append(dict(input=r['input'],route=r['route'],artifact=str(path),
                                 sha256=digest(path),**data))
    observations = []
    for row in rows:
        item = {k:v for k,v in row.items() if k!='result'}
        item['result'] = {k:v for k,v in row['result'].items() if k not in ['learning_schedule','histogram_fitting']}
        item['full_result_sha256'] = digest(capture/row['directory']/'result.json')
        observations.append(item)
    args.json.write_text(json.dumps(dict(schema=1,artifact_directory=str(capture),
                                        analyzer_sha256=digest(Path(__file__)),
                                        manifest_sha256=digest(capture/'manifest.json'),
                                        manifest=manifest,summary=summary,sampling=sampling,runs=observations),indent=2)+'\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True, sharey=True)
        for ax, route, color in zip(axes, ['cpu','auto'], ['#226b80','#b45a28']):
            s = next(s for s in summary if s['input']=='photo-large-1039' and s['route']==route and s['mode']=='schedule')
            result = next(r['result'] for r in rows if r['directory']==s['schedule_observation'])
            trace = result['learning_schedule']
            origin = trace['waves'][0]['dispatch']
            events = sorted([(j['start']-origin,1) for j in trace['jobs']] + [(j['finish']-origin,-1) for j in trace['jobs']])
            times, counts, count = [0], [0], 0
            for timestamp, change in events:
                count += change
                times.append(timestamp); counts.append(count)
            ax.fill_between(times, counts, step='post', color=color, alpha=.25)
            ax.step(times, counts, where='post', color=color, linewidth=1.15)
            for w in trace['waves']:
                ax.axvline(w['dispatch']-origin, color='#888888', linewidth=.5, alpha=.35)
            ax.axhline(14, color='#777777', linewidth=.7, linestyle='--')
            ax.set_title(f"{'CPU' if route=='cpu' else 'Metal'}: {s['schedule']['waves']} waves; "
                         f"{s['schedule']['average_active_jobs']:.2f} average active jobs during joined waves",loc='left',fontsize=11)
            ax.set_ylim(-.1,15); ax.set_yticks([0,4,8,12,14]); ax.set_ylabel('Active jobs')
            ax.spines[['right','top']].set_visible(False)
        axes[-1].set_xlabel('Seconds since the first learning wave dispatched')
        fig.suptitle('1039: fitting waves leave capacity unused',fontsize=15,x=.08,ha='left')
        fig.text(.08,.025,'Schedule-only captures; jobs include preparation and fitting. Job lifetimes are not hardware CPU utilization.',fontsize=9)
        fig.tight_layout(rect=(0,.055,1,.94))
        args.plot.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.plot,metadata={'Date':None})
        fig.savefig(args.plot.with_suffix('.png'),dpi=150)
        plt.close(fig)
    for s in summary:
        if s['mode']=='fit-off':
            print(s['input'],s['route'],round(s['seconds'],3),s['stages'])


if __name__=='__main__':
    main()
