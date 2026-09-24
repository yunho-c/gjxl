import argparse, collections, json, math, random, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODES=('ordinary','host','graph','split','record','full')
FIELDS=('wall_ns','all_command_gpu_ns','submission_host_ns','resolution_host_ns','quantization_ns','serializer_ns')
CONTRASTS={'total_profile':('ordinary','full'),'gpu_profile':('host','full'),'stage_graph':('host','graph'),'encoder_split':('graph','split'),'recording':('split','record'),'timestamps':('record','full'),'post_split':('split','full')}
def interval(values):
    rng=random.Random(95121)
    boots=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(4000))
    return [boots[int(.025*len(boots))],boots[int(.975*len(boots))]]
def analyze(directory):
    output=[]
    for ok in sorted(directory.glob('*.ok.json')):
        stem=ok.name.removesuffix('.ok.json')
        rows=[json.loads(l) for l in (directory/(stem+'.jsonl')).read_text().splitlines()]
        rows=[r for r in rows if r['rep']>=0]
        modes=sorted({r['mode'] for r in rows});pairs=sorted({r['pair'] for r in rows})
        fields=[f for f in FIELDS if all(f in r for r in rows)]
        med={i:{m:{f:statistics.median(r[f]/1e6 for r in rows if r['pair']==i and r['mode']==m) for f in fields} for m in modes} for i in pairs}
        item={'case':stem,'effort':rows[0]['effort'],'width':rows[0]['width'],'height':rows[0]['height'],'samples':len(rows),'pairs':len(pairs),
          'submissions':sorted({r['submissions'] for r in rows}),
          'mode_median_ms':{m:{f:statistics.median(med[i][m][f] for i in pairs) for f in fields} for m in modes},'contrasts':{}}
        for name,(a,b) in CONTRASTS.items():
            if a not in modes or b not in modes:continue
            contrast={}
            for f in fields:
                deltas=[med[i][b][f]-med[i][a][f] for i in pairs]
                contrast[f]={'median_delta_ms':statistics.median(deltas),'range_delta_ms':[min(deltas),max(deltas)],'bootstrap_95ci_ms':interval(deltas),'paired_deltas_ms':deltas}
                if all(med[i][a][f]>0 for i in pairs):
                    ratios=[100*(med[i][b][f]/med[i][a][f]-1) for i in pairs]
                    contrast[f].update(median_percent=statistics.median(ratios),bootstrap_95ci_percent=interval(ratios),paired_percent=ratios)
            item['contrasts'][name]=contrast
        if 'encoders' in rows[0]:item['encoders']={m:sorted({r['encoders'] for r in rows if r['mode']==m}) for m in modes}
        output.append(item)
    return output
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');x=p.parse_args();directory=ROOT/x.directory
    result=analyze(directory);(directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for j in result:
        c=j['contrasts'];f=c['gpu_profile']['wall_ns'];split=c.get('encoder_split',{}).get('submission_host_ns',{});rec=c.get('recording',{}).get('submission_host_ns',{});full=j['mode_median_ms']['full']
        print(j['case'], 'GPU profile wall',round(f['median_delta_ms'],3), 'CI', [round(v,3) for v in f['bootstrap_95ci_ms']],
              'split encode',round(split.get('median_delta_ms',0),3),'record encode',round(rec.get('median_delta_ms',0),3),
              'resolve',round(full.get('resolution_host_ns',0),3))
