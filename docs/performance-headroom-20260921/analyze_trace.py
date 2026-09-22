"""Correlate exported Instruments counters with two measured ordinary calls.

Device counters are correlated with time windows, not attributed by process.
Other processes' visible GPU intervals and 100 us shader edges are excluded.
"""
from array import array
import argparse
import collections
import csv
import json
import gzip
import hashlib
import platform
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

if sys.version_info >= (3, 14) and tuple(int(x) for x in np.__version__.split('.')[:2]) < (2, 3):
    raise RuntimeError('Use the pinned uv command: this Python/NumPy combination failed numerical validation.')

HERE = Path(__file__).resolve().parent
TRACE = HERE / 'trace-counters-24mp-e7'

def table(path):
    tree = ET.parse(path)
    ids = {x.get('id'): x for x in tree.iter() if x.get('id')}
    cols = [x.findtext('mnemonic') for x in tree.findall('.//schema/col')]
    def resolve(x):
        return ids[x.get('ref')] if x.get('ref') else x
    return [{k: {'text': resolve(x).text, 'fmt': resolve(x).get('fmt')}
             for k, x in zip(cols, row)} for row in tree.findall('.//row')]

def text(row, key):
    return row[key]['text']

def fmt(row, key):
    return row[key]['fmt'] or row[key]['text'] or ''

def union(xs):
    result = []
    for a,b in sorted(xs):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a,b])
    return result

def intersection(xs, ys):
    xs,ys = union(xs),union(ys)
    result=[];i=j=0
    while i<len(xs) and j<len(ys):
        a,b=max(xs[i][0],ys[j][0]),min(xs[i][1],ys[j][1])
        if b>a:result.append([a,b])
        if xs[i][1]<=ys[j][1]:i+=1
        else:j+=1
    return result

def subtract(xs,ys):
    xs,ys=union(xs),union(ys); result=[]
    for a,b in xs:
        for c,d in ys:
            if d<=a:continue
            if c>=b:break
            if c>a:result.append([a,min(c,b)])
            a=max(a,d)
            if a>=b:break
        if b>a:result.append([a,b])
    return result

def interval(row):
    a=int(text(row,'start'));return [a,a+int(text(row,'duration'))]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--edge-trim-us',type=int,default=100)
    args=parser.parse_args()
    edge=args.edge_trim_us*1000
    suffix='' if args.edge_trim_us==100 else f'-edge{args.edge_trim_us}'
    submissions=table(TRACE/'metal-application-command-buffer-submissions.xml')
    submissions.sort(key=lambda r:int(text(r,'start')))
    paired=json.loads((TRACE/'paired.json').read_text())['rows']
    assert len(submissions)==4*(1+len(paired))==28
    mapping={}
    for i,call in enumerate([{'mode':'reference','sample_index':-2}]+paired):
        for j,row in enumerate(submissions[i*4:i*4+4]):
            mapping[text(row,'cmdbuffer-id')]=(i,j,call)
    selected={k for k,(_,_,r) in mapping.items() if r['mode']=='ordinary' and r['sample_index']>=0}
    gpu=table(TRACE/'metal-gpu-intervals.xml')
    active=union(interval(r) for r in gpu if text(r,'cmdbuffer-id') in selected and fmt(r,'process').startswith('capture'))
    competing=union(interval(r) for r in gpu if not fmt(r,'process').startswith('capture') and fmt(r,'state')=='Active')
    clean=subtract(active,competing)
    shader=table(TRACE/'metal-shader-profiler-intervals.xml')
    names=collections.defaultdict(list)
    for r in shader:
        if fmt(r,'process').startswith('capture'):
            name=fmt(r,'name').split(' (')[0]
            a,b=interval(r)
            if b-a>2*edge:names[name].append([a+edge,b-edge])
    groups={'all ordinary GPU active':clean}
    for name,windows in names.items():
        windows=intersection(windows,clean)
        if sum(b-a for a,b in windows)>1e6:groups[name]=windows
    metadata=table(TRACE/'gpu-counter-info.xml')
    counter_names={int(text(r,'counter-id')):fmt(r,'name') for r in metadata}
    desired={3,4,5,6,7,8,11,12,13,14,17,18,21,22,23,24,25,29,30,31,47,50,51,54,61,62,63}
    # Keep only scalar reference values; discard formatted label subtrees.
    scalar_tags={'start-time','duration','uint32','fixed-decimal'}
    refs={}; data={k:(array('d'),array('d'),array('d')) for k in desired}
    count=0
    cache=HERE/'trace-counter-arrays.npz'
    raw_path=TRACE/'metal-gpu-counter-intervals.xml'
    raw_stream=None
    if not cache.exists():
        raw_stream=raw_path.open('rb') if raw_path.exists() else gzip.open(str(raw_path)+'.gz','rb')
    for event,elem in (() if cache.exists() else ET.iterparse(raw_stream,events=('end',))):
        if elem.tag!='row':continue
        for x in elem.iter():
            if x.tag in scalar_tags and x.get('id'):
                refs[int(x.get('id'))]=float(x.text)
        def scalar(x):
            return refs[int(x.get('ref'))] if x.get('ref') else float(x.text)
        cid=int(scalar(elem[2]))
        if cid in desired:
            a,d,v=scalar(elem[0]),scalar(elem[1]),scalar(elem[5])
            for target,val in zip(data[cid],(a,d,v)):target.append(val)
        count+=1
        elem.clear()
        if count%1000000==0:print('Parsed',count,'counter intervals',flush=True)
    if raw_stream is not None:raw_stream.close()
    if cache.exists():
        loaded=np.load(cache)
        data={k:(loaded[f'{k}_starts'],loaded[f'{k}_durations'],loaded[f'{k}_values']) for k in desired}
        count=int(loaded['count'])
    else:
        np.savez_compressed(cache,count=count,**{f'{k}_{name}':np.asarray(a) for k,arrays in data.items()
                           for name,a in zip(('starts','durations','values'),arrays)})
    results=[]
    for cid,(starts,durations,values) in data.items():
        starts=np.array(starts);durations=np.array(durations);values=np.array(values)
        order=np.argsort(starts);starts,durations,values=starts[order],durations[order],values[order]
        assert len(starts)>0
        assert np.all(starts[1:]>=starts[:-1]+durations[:-1])
        before_values=hashlib.sha256(values.tobytes()).hexdigest()
        if cid in (3,13,14,23,61):print('Raw counter range',counter_names[cid],np.quantile(values,[0,.5,.95,.99,1]),flush=True)
        weighted=np.r_[0,np.cumsum(durations*values)]
        covered=np.r_[0,np.cumsum(durations)]
        def integral(points,prefix,coefficient):
            index=np.searchsorted(starts,points,side='right')-1
            safe=np.maximum(index,0)
            return np.where(index<0,0,prefix[safe]+np.clip(points-starts[safe],0,durations[safe])*coefficient[safe])
        for group,windows in groups.items():
            a,b=np.array(windows).T
            denominator=np.sum(integral(b,covered,np.ones(len(starts)))-integral(a,covered,np.ones(len(starts))))
            numerator=np.sum(integral(b,weighted,values)-integral(a,weighted,values))
            # Independent direct intersections catch numerical/export anomalies.
            direct_weight=direct_sum=0.0
            for aa,bb in windows:
                lo=max(0,np.searchsorted(starts,aa,side='right')-1)
                hi=np.searchsorted(starts,bb,side='left')
                weights=np.maximum(0,np.minimum(starts[lo:hi]+durations[lo:hi],bb)-np.maximum(starts[lo:hi],aa))
                direct_weight+=weights.sum();direct_sum+=weights@values[lo:hi]
            assert np.isclose(denominator,direct_weight,rtol=1e-8,atol=1e-3),(cid,group,denominator,direct_weight)
            assert np.isclose(numerator,direct_sum,rtol=1e-6,atol=1e-2),(cid,group,numerator,direct_sum)
            if denominator>0:
                assert values.min()-1e-5 <= numerator/denominator <= values.max()+1e-5
                results.append({'window':group,'counter_id':cid,'counter':counter_names[cid],
                                'covered_ms':denominator/1e6,'mean':numerator/denominator})
        assert hashlib.sha256(values.tobytes()).hexdigest()==before_values,'Numerical library mutated the input values'
    with (HERE/f'trace-counters{suffix}.csv').open('w') as f:
        w=csv.DictWriter(f,list(results[0]));w.writeheader();w.writerows(results)
    manifest={'trace':str(TRACE),'ordinary_samples':2,'command_buffer_groups':mapping,
              'counter_intervals':count,'visible_gpu_active_ms':sum(b-a for a,b in active)/1e6,
              'after_competing_gpu_exclusion_ms':sum(b-a for a,b in clean)/1e6,
              'shader_edge_trim_us':args.edge_trim_us,'windows':groups,
              'python':platform.python_version(),'numpy':np.__version__,
              'validation':'Prefix integrals match independent direct interval-weighted sums; means within observed ranges; value arrays unchanged.',
              'warning':'Single diagnostic run with device-level counters and shader sampling; not a throughput qualification or peak measurement.'}
    (HERE/f'trace-analysis{suffix}.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('GPU active ms',manifest['visible_gpu_active_ms'],'clean ms',manifest['after_competing_gpu_exclusion_ms'])
    for name in groups:
        if name=='all ordinary GPU active' or any(s in name for s in ['low_medium','malta_fixed','resident_l2','dct32_candidate','dct32_forward','dct16_candidate']):
            print(name,{r['counter']:round(r['mean'],2) for r in results if r['window']==name and r['counter_id'] in (3,7,8,11,12,13,14,17,21,23,24,47,61)})

if __name__=='__main__':main()
