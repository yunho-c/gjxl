#!/usr/bin/env python3
"""Summarize raw paired precision screens without adding overlapping stage groups."""
import argparse
import json
from pathlib import Path
import statistics
from precision import ROOT, OUT, save, sha

def stage_total(path, group):
    raw = json.loads(path.read_text())
    def include(name):
        if group == "butteraugli":
            return name.startswith(("butteraugli.", "frontend.prepare_aq.reference."))
        if group == "ac": return name.startswith("frontend.ac_strategy.")
        if group == "epf": return name.startswith(("aq.epf.","aq.epf_linear.")) or name=="aq.opsin_to_linear"
        return True
    totals = []
    for sample in raw["workloads"][0]["samples"]:
        totals.append(sum(stage["gpu_nanoseconds"] for submission in sample["submissions"]
            for stage in submission["stages"] if include(stage["stage_id"])) / 1e6)
    return statistics.median(totals)

def report():
    results = {}
    for path in sorted((OUT / "quality").glob("*/screen-summary.json")):
        raw = json.loads(path.read_text())
        results[path.parent.name] = dict(quality_cases=raw["cases"], exact=raw["exact"],
            quality_failures=raw["quality_failures"])
    for mode in ("stage", "wall"):
        for path in sorted((OUT / ("screen-" + mode)).glob("*/summary.json")):
            variant = path.parent.name
            pair_count = json.loads((path.parent / "identity.json").read_text())["pairs"]
            rows = {}
            for row in json.loads(path.read_text()):
                workload = row["name"]
                metrics = {}
                if mode == "stage":
                    for group in ("all", "butteraugli", "ac", "epf"):
                        paired, baseline, candidate = [], [], []
                        for pair in range(pair_count):
                            b = stage_total(path.parent / f"{workload}-pair{pair}-baseline.raw.json", group)
                            c = stage_total(path.parent / f"{workload}-pair{pair}-candidate.raw.json", group)
                            baseline.append(b); candidate.append(c)
                            paired.append(100*(c/b-1) if b else 0)
                        metrics[group] = dict(baseline_ms=statistics.median(baseline),
                            candidate_ms=statistics.median(candidate), paired_changes=paired,
                            median_change_percent=statistics.median(paired), wins=sum(x<0 for x in paired))
                else:
                    metrics["total"] = row["metrics"]["total"]
                rows[workload] = metrics
            results.setdefault(variant, {})[mode] = rows
    save(OUT / "screen-analysis.json", results)
    for name, result in results.items():
        values = [name, "Q=" + str(result.get("quality_failures", "pending"))]
        for mode in ("stage", "wall"):
            if mode not in result: continue
            key = "all" if mode=="stage" else "total"
            values.append(mode + "=" + str({n:round(v[key]["median_change_percent"],2)
                                            for n,v in result[mode].items()}))
        print(" ".join(values))

def publish():
    """Retain compact results in the repository; raw evidence stays in build/."""
    def read(path):
        return json.loads(path.read_text())
    def brief(path):
        return {k:v for k,v in read(path).items() if k not in ('rows','samples')}
    decision=read(OUT/'decision.json')
    if decision['status'] != 'complete':
        raise RuntimeError('A completed decision is required before publication')
    result=dict(schema_version=1, decision=decision,
        baseline={k:v for k,v in read(OUT/'identity.json').items() if k in ('head','protocol','submodules')},
        raw_evidence='build/precision-study', experiments=read(OUT/'screen-analysis.json'),
        baseline_tests=read(OUT/'baseline-tests.json'), confirmations={}, qualification={})
    for path in sorted((OUT/'variants').glob('*/identity.json')):
        variant=path.parent.name
        identity=read(path)
        if sha(path.parent/'metal/gjxl.metallib') != identity['library']:
            raise RuntimeError('Variant artifact changed: '+variant)
        entry=result['experiments'].setdefault(variant,{})
        entry.update(shader=identity['shader'],flags=identity['flags'],
                     library_sha256=identity['library'],compiler=identity['compiler'],
                     patch_sha256=sha(path.parent/'shader.patch'))
        contract=OUT/'contracts'/(variant+'.json')
        if contract.exists():
            entry['contracts']={row['test']:row['passed'] for row in read(contract)['rows']}
    for variant, base in [('ba_relaxed_precise',OUT),
                          ('ba_malta_fastdivide',OUT/'integrations/ba_malta_fastdivide')]:
        qual={}
        for group in ('quality','quality-corpus'):
            path=OUT/group/variant/'summary.json'
            if path.exists():qual[group]=brief(path)
        path=OUT/'matched-quality'/(variant+'.json')
        if path.exists():qual['matched_quality']=brief(path)
        for mode in ('full-normal','screen-validation'):
            path=OUT/'metric-bias'/variant/mode/'summary.json'
            if path.exists():
                qual['metric_'+mode]=brief(path)
                qual['metric_'+mode]['max_map_error']=max(r['map_max_abs'] for r in read(path)['rows'])
        for filename in ('candidate-tests.json','tests-normal.json','tests-validation.json',
                         'candidate-artifacts.json','candidate-shader-equivalence.json'):
            path=base/filename
            if path.exists():qual[filename.removesuffix('.json')]=read(path)
        path=base/'confirmation-wall/summary.json'
        if path.exists():
            timing=[dict(name=r['name'],**r['metrics']['total']) for r in read(path)]
            for row in timing:
                row['favorable_pairs']=sum(v<0 for v in row['paired_changes_percent'])
            result['confirmations'][variant]=timing
        result['qualification'][variant]=qual
    result['live_shader_equivalence']=read(OUT/'live-shader-equivalence.json')
    result['live_artifacts']=read(OUT/'live-artifacts.json')
    result['device']=[{k:v for k,v in gpu.items() if k in
        ('sppci_model','sppci_cores','spdisplays_vendor')} for gpu in read(OUT/'device.json')['SPDisplaysDataType']]
    for variant,reason in decision['dispositions'].items():
        result['experiments'][variant]['disposition']=reason
    result['harness_sha256']={str(p.relative_to(ROOT)):sha(p) for p in
        sorted((ROOT/'tools/metal_dataflow').glob('precision*.py'))}
    save(ROOT/'docs/metal-precision-results.json',result)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    report()
    if args.publish:publish()
