#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Summarize paired elapsed times separately from Samply CPU attribution."""
import argparse
import json
from pathlib import Path
import statistics
from run_hybrid_qualification import digest, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.artifacts / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require(manifest['status'] == 'passed', 'Capture did not complete successfully')
    observations = manifest['observations']
    rows = []
    for name, route, participants in sorted({(r['input'], r['route'], r['participants']) for r in observations}):
        row = dict(input=name, route=route, participants=participants)
        for label in ['baseline', 'reuse']:
            calls = [r for r in observations if r['input'] == name and r['route'] == route
                     and r['participants'] == participants and r['build'] == label
                     and not r['warmup'] and not r['sampled']]
            require(len(calls) == manifest['repeats'], 'Missing measured calls')
            row[label] = dict(wall=statistics.median(r['result']['wall'] for r in calls),
                              wall_range=[min(r['result']['wall'] for r in calls),
                                          max(r['result']['wall'] for r in calls)],
                              stream_wall=statistics.median(r['stream_wall'] for r in calls),
                              cpu=statistics.median(r['result']['cpu'] for r in calls),
                              peak_committed=max(r['result']['peak_committed'] for r in calls),
                              peak_backing=max(r['result']['peak_backing'] for r in calls))
        row['complete_speedup'] = row['baseline']['wall'] / row['reuse']['wall']
        row['stream_speedup'] = row['baseline']['stream_wall'] / row['reuse']['stream_wall']
        rows.append(row)
    sampling = []
    for call in observations:
        if not call['sampled']:
            continue
        path = args.artifacts / (call['directory'] + '-analysis.json')
        analysis = json.loads(path.read_text())
        row = {k: call[k] for k in ['input', 'route', 'build']}
        row['analysis_sha256'] = digest(path)
        row['attribution'] = {}
        for method, attribution in analysis['attribution'].items():
            stream = attribution['stages_cpu_seconds'].get('stream_encoding', 0)
            mutex = attribution['managed_allocation_mutex_cpu_seconds_by_stage'].get('stream_encoding', 0)
            row['attribution'][method] = dict(stream_sampled_cpu=stream, mutex_sampled_cpu=mutex,
                                             mutex_share=mutex / stream if stream else None)
        sampling.append(row)
    result = dict(schema=1, revision=manifest['revision'], artifact_directory=str(args.artifacts.resolve()),
                  manifest_sha256=digest(manifest_path), reporter_sha256=digest(Path(__file__)),
                  scope='Fresh-process public learned estimated-ANS calls, AC power, active desktop. '
                    'Three measured calls per condition after one discarded call. '
                    'At 14 participants stream time is joined local model/emission wall; at one it is '
                    'serial model plus emission wall. Sampled CPU is separate diagnostic evidence.',
                  complete_encodes=len(observations), exact_decodes=all(r['result']['exact_decode'] for r in observations),
                  inputs=manifest['inputs'], binaries=manifest['binaries'], timing=rows,
                  sampling=sampling, probes=manifest['probes'])
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print(row['input'], row['route'], row['participants'],
              f"total {row['baseline']['wall']:.3f} -> {row['reuse']['wall']:.3f}, "
              f"stream {row['baseline']['stream_wall']:.3f} -> {row['reuse']['stream_wall']:.3f}")
    for row in sampling:
        print(row['input'], row['route'], row['build'], 'mutex share', row['attribution']['current']['mutex_share'])


if __name__ == '__main__':
    main()
