#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Package fitting-scheduler controls and render separate activity diagnostics."""
import argparse
import json
from pathlib import Path

from run_hybrid_qualification import digest, require
from run_parallel_profile import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    parser.add_argument('--plot-title', default='1039: continuous queues and shared feature cohorts')
    args = parser.parse_args()
    capture = args.capture.resolve()
    manifest = json.loads((capture/'manifest.json').read_text())
    require(manifest['status'] == 'passed', 'Incomplete qualification')
    summary = json.loads((capture/'summary.json').read_text())
    rows = [json.loads(line) for line in (capture/'results.jsonl').read_text().splitlines()]
    require(len(rows) == manifest['completed_calls'], 'Missing observations')
    for r in rows:
        validate(r['result'])
        r['full_result_sha256'] = digest(capture/r['directory']/'result.json')
    args.json.write_text(json.dumps(dict(schema=1, artifact_directory=str(capture),
                                        analyzer_sha256=digest(Path(__file__)),
                                        manifest_sha256=digest(capture/'manifest.json'),
                                        manifest=manifest, summary=summary, runs=rows),indent=2)+'\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True, sharey=True)
        for ax,(route,revision) in zip(axes, [('cpu','before'),('cpu','after'),('auto','before'),('auto','after')]):
            item = next(s for s in summary if s['input']=='photo-large-1039' and s['revision']==revision and s['route']==route)
            r = next(r for r in rows if r['input']==item['input'] and r['revision']==revision and r['route']==route and r['diagnostic'])
            trace = r['result']['learning_schedule']
            origin = trace['waves'][0]['dispatch']
            events = sorted([(j['start']-origin,1) for j in trace['jobs']] + [(j['finish']-origin,-1) for j in trace['jobs']])
            times, counts, count = [0], [0], 0
            for timestamp, delta in events:
                count += delta
                times.append(timestamp)
                counts.append(count)
            color = '#226b80' if revision=='before' else '#a74d28'
            ax.fill_between(times, counts, step='post', color=color, alpha=.25)
            ax.step(times, counts, where='post', color=color, linewidth=1)
            for w in trace['waves']:
                ax.axvline(w['dispatch']-origin,color='#888888',linewidth=.5,alpha=.3)
            ax.axhline(14,color='#777777',linewidth=.7,linestyle='--')
            ax.set_title(f"{'CPU' if route=='cpu' else 'Metal'} {revision}: {item['schedule']['waves']} joined queues/waves; "
                         f"{item['schedule']['average_active_jobs']:.2f} average active jobs",loc='left',fontsize=11)
            ax.set_ylim(-.1,15)
            ax.set_yticks([0,4,8,12,14])
            ax.set_ylabel('Active jobs')
            ax.spines[['right','top']].set_visible(False)
        axes[-1].set_xlabel('Seconds since the first learning dispatch')
        fig.suptitle(args.plot_title,fontsize=15,x=.08,ha='left')
        fig.text(.08,.025,'Separate schedule captures. Job lifetimes include preemption; this is not hardware CPU utilization.',fontsize=9)
        fig.tight_layout(rect=(0,.05,1,.95))
        args.plot.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.plot,metadata={'Date':None})
        fig.savefig(args.plot.with_suffix('.png'),dpi=150)
        plt.close(fig)
    for item in summary:
        print(item['input'],item['revision'],item['route'],round(item['seconds'],4),item['seconds_range'],
              'learning',round(item['details']['learning'],4),
              'active',round(item['schedule']['average_active_jobs'],2))


if __name__=='__main__':
    main()
