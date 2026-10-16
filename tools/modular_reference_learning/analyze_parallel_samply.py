#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""All-thread sampled CPU attribution; never treat these totals as wall stages."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import subprocess

from analyze_fitting_samply import resolve, sample_weights
from run_hybrid_qualification import digest, require


def category(names):
    text = '\n'.join(names)
    if 'LearnHistogramTree(' in text or 'FindBestSplit<' in text:
        return 'fitting'
    if 'PrepareHistogramSamples' in text:
        return 'sample_preparation'
    if 'SelectReferenceModularEncoding' in text or 'SelectLocalReferenceScopes' in text:
        return 'selection'
    if 'PrepareModularStreamFeatures' in text:
        return 'feature_preparation'
    if any(name in text for name in ['TokenizePreparedModularStreams', 'GenerateModularStreamTokens',
                                    'TokenizeValidatedModularChannel',
                                    '::Tokenize(gjxl::modular_internal::PreparedModularStreamsView']):
        return 'tokenization'
    if 'EncodePreparedModularStreams' in text:
        return 'stream_encoding'
    if 'gjxl::modular_internal::' in text or 'gjxl::EncodeModularImage(' in text:
        return 'other_encoder'
    return 'outside_encoder_or_unresolved'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--symbolizer', default='llvm-symbolizer')
    args = parser.parse_args()
    with gzip.open(args.profile, 'rt') as stream:
        profile = json.load(stream)
    sidecar = Path(str(args.profile).removesuffix('.json.gz')+'.json.syms.json')
    symbols = json.loads(sidecar.read_text())
    require(profile['meta']['sampleUnits']['threadCPUDelta'] == 'µs', 'Unknown sampled CPU units')
    contexts, addresses = [], set()
    for thread in profile['threads']:
        frames = resolve(profile, symbols, thread)
        stacks = {}
        table = thread['stackTable']
        for index in set(thread['samples']['stack']):
            current, stack, seen = index, [], set()
            while current is not None:
                require(current not in seen, 'Cyclic stack')
                seen.add(current)
                stack.append(frames[table['frame'][current]])
                current = table['prefix'][current]
            stacks[index] = stack
        kinds = {i: category(f['name'] for f in stack) for i, stack in stacks.items()}
        contexts.append((thread, stacks, kinds))
        for index, stack in stacks.items():
            if kinds[index] == 'outside_encoder_or_unresolved':
                continue
            for f in stack:
                if f['lib'] is not None and f['lib'] >= 0 and profile['libs'][f['lib']]['name'] == args.binary.name:
                    addresses.add(f['address'])
    addresses = sorted(addresses)
    require(addresses, 'No native encoder stacks')
    commands = subprocess.check_output(['otool', '-l', str(args.binary)], text=True)
    match = re.search(r'segname __TEXT\s+vmaddr (0x[0-9a-fA-F]+)', commands)
    require(match, 'Missing image base')
    image_base = int(match.group(1), 16)
    symbolizer = [args.symbolizer, '--inlines', '--output-style=JSON', '--obj='+str(args.binary.resolve())]
    output = subprocess.check_output(symbolizer, input=''.join(f'0x{a+image_base:x}\n' for a in addresses), text=True)
    resolved = [json.loads(line) for line in output.splitlines()]
    require(len(resolved) == len(addresses), 'Symbolizer lost addresses')
    inline = dict(zip(addresses, resolved))
    result = dict(scope='All captured threads. CPU deltas assigned to current and previous stacks. '
                       'Stages are disjoint sampled CPU buckets; inclusive function rows overlap. '
                       'Outside-encoder samples include input, independent decoding, Metal service threads '
                       'and unresolved stacks. These are not wall times or GPU execution times.',
                  profile_sha256=digest(args.profile), sidecar_sha256=digest(sidecar),
                  binary_sha256=digest(args.binary), captured_threads=len(contexts),
                  analyzer_sha256=digest(Path(__file__)),
                  symbolizer_command=symbolizer, image_base=image_base, attribution={})
    for previous in [False, True]:
        stages, flat, fit_flat, fit_inclusive, lines = (Counter() for _ in range(5))
        mutex_stages, managed_mutex_stages, mutex_stacks = Counter(), Counter(), Counter()
        threads = []
        resolved_fit = native_fit = inline_fit = fitting = positive = 0
        for thread, stacks, kinds in contexts:
            totals = Counter()
            weights = sample_weights(thread['samples']['threadCPUDelta'], previous)
            for index, weight in zip(thread['samples']['stack'], weights):
                if weight <= 0:
                    continue
                positive += 1
                kind, stack = kinds[index], stacks[index]
                totals[kind] += weight
                stages[kind] += weight
                if stack and '__psynch_mutex' in stack[0]['name']:
                    mutex_stages[kind] += weight
                    if any('ResourceAllocation::' in f['name'] or 'ResourceReservation::' in f['name'] for f in stack):
                        managed_mutex_stages[kind] += weight
                        mutex_stacks[tuple(f['name'] for f in stack[:10])] += weight
                if kind == 'outside_encoder_or_unresolved' or not stack:
                    continue
                leaf = stack[0]
                native = leaf['lib'] is not None and leaf['lib'] >= 0 and profile['libs'][leaf['lib']]['name'] == args.binary.name
                info = inline.get(leaf['address'], {}).get('Symbol', []) if native else []
                name = (info[0].get('FunctionName') if info else None) or leaf['name']
                flat[name] += weight
                if kind != 'fitting':
                    continue
                fitting += weight
                fit_flat[name] += weight
                names = {f['name'] for f in stack}
                for f in stack:
                    if f['lib'] is not None and f['lib'] >= 0 and profile['libs'][f['lib']]['name'] == args.binary.name:
                        names.update(x['FunctionName'] for x in inline.get(f['address'], {}).get('Symbol', []) if x.get('FunctionName'))
                for n in names:
                    fit_inclusive[n] += weight
                if name and not name.startswith('0x'):
                    resolved_fit += weight
                if native:
                    native_fit += weight
                    if info and info[0].get('FileName') and info[0].get('Line', 0):
                        inline_fit += weight
                        lines[f"{info[0]['FileName']}:{info[0]['Line']}"] += weight
            if totals:
                threads.append(dict(tid=thread['tid'], name=thread['name'], main=thread.get('isMainThread', False),
                                    cpu_seconds={k: v/1e6 for k,v in totals.items()}))
        encoder = sum(v for k,v in stages.items() if k != 'outside_encoder_or_unresolved')
        require(encoder > 0 and fitting > 0, 'No encoder/fitter CPU attributed')
        require(sum(flat.values()) == encoder and sum(fit_flat.values()) == fitting, 'CPU conservation failed')
        rows = lambda c, total: [dict(name=k, cpu_seconds=v/1e6, percent=100*v/total) for k,v in c.most_common(40)]
        result['attribution']['previous' if previous else 'current'] = dict(
            positive_samples=positive, encoder_cpu_seconds=encoder/1e6, fitting_cpu_seconds=fitting/1e6,
            stages_cpu_seconds={k:v/1e6 for k,v in stages.items()},
            mutex_cpu_seconds_by_stage={k:v/1e6 for k,v in mutex_stages.items()},
            managed_allocation_mutex_cpu_seconds_by_stage={k:v/1e6 for k,v in managed_mutex_stages.items()},
            managed_allocation_mutex_stacks=[dict(stack=list(k),cpu_seconds=v/1e6) for k,v in mutex_stacks.most_common(8)],
            resolved_fitting_leaf_percent=100*resolved_fit/fitting,
            inline_native_fitting_percent=100*inline_fit/native_fit if native_fit else 0,
            flat=rows(flat, encoder), fitting_flat=rows(fit_flat, fitting),
            fitting_inclusive=rows(fit_inclusive, fitting), fitting_leaf_lines=rows(lines, fitting), threads=threads)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(profile=str(args.profile), threads=result['captured_threads'],
                         **{k:v for k,v in result['attribution']['current'].items() if not isinstance(v, (list,dict))})))


if __name__ == '__main__':
    main()
