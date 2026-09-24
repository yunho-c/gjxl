import argparse, hashlib, json, os, platform, statistics, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def busy():
    rows=[]
    for line in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines():
        pid,state,command=line.strip().split(None,2)
        if any(x in state for x in 'TZ'):continue
        name=Path(command).name
        if name.startswith(('gjxl_','cjxl')) or name in ('ctest','ninja','make','clang','clang++','cc1','metal','xctrace'):
            rows.append(line.strip())
    return rows

def main():
    a=argparse.ArgumentParser();a.add_argument('--name',required=True);a.add_argument('--probe',type=Path,required=True)
    a.add_argument('--jobs',type=Path,required=True);a.add_argument('--modes',default='host,full');a.add_argument('--pairs',type=int,default=3)
    a.add_argument('--samples',type=int,default=3);a.add_argument('--warmups',type=int,default=1)
    x=a.parse_args();out=ROOT/x.name;out.mkdir(exist_ok=True);jobs=json.loads(x.jobs.read_text())
    identity={'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
      'probe':str(x.probe.resolve()),'probe_sha256':sha(x.probe),'runner_sha256':sha(__file__),
      'source_sha256':sha(ROOT/'probe.cpp'),'platform':platform.platform(),
      'modes':x.modes.split(','),'pairs':x.pairs,'samples':x.samples,'warmups':x.warmups,
      'distance':1.2,'cpu_threads':8,'final_score':False,'jobs':jobs,
      'inputs':{j['input']:sha(j['input']) for j in jobs if j['input']!='synthetic128'}}
    if (out/'identity.json').exists():assert json.loads((out/'identity.json').read_text())==identity,'Frozen identity changed'
    else:save(out/'identity.json',identity)
    for n,j in enumerate(jobs):
        stem=f'{n:03d}-{j["label"]}-e{j["effort"]}'; raw=out/(stem+'.jsonl');ok=out/(stem+'.ok.json')
        if ok.exists():
            assert sha(raw)==json.loads(ok.read_text())['raw_sha256'];continue
        waiting=False
        while b:=busy():
            if not waiting:print('WAIT competing work: '+str(b),flush=True);waiting=True
            time.sleep(10)
        print('RUN '+stem,flush=True)
        cmd=[str(x.probe.resolve()),j['input'],str(j['effort']),str(x.pairs),str(x.samples),str(x.warmups),x.modes,str(n),str(out/(stem+'.jxl'))]
        start=time.time();interference=[]
        with raw.open('w') as stdout,(out/(stem+'.stderr')).open('w') as stderr:
            p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr)
            while p.poll() is None:
                time.sleep(.5)
                interference.extend(busy())
        save(out/(stem+'.command.json'),{'command':cmd,'seconds':time.time()-start,'returncode':p.returncode,'interference':interference})
        if p.returncode or interference:raise RuntimeError(f'Invalid capture {stem}: status={p.returncode} interference={interference}')
        rows=[json.loads(s) for s in raw.read_text().splitlines()]
        assert len(rows)==x.pairs*(x.samples+x.warmups)*len(identity['modes'])
        assert all(r['byte_equal'] for r in rows)
        assert len({r['submissions'] for r in rows})==1,'Submission count changed'
        assert len({r['encoded_bytes'] for r in rows})==1
        med={m:statistics.median(r['wall_ns']/1e6 for r in rows if r['mode']==m and r['rep']>=0) for m in identity['modes']}
        print('DONE '+stem+' '+json.dumps(med),flush=True)
        save(ok,{'raw_sha256':sha(raw),'output_sha256':sha(out/(stem+'.jxl')),'rows':len(rows),'wall_median_ms':med})
    assert sha(x.probe)==identity['probe_sha256']
    for p,h in identity['inputs'].items():assert sha(p)==h
    save(out/'complete.json',{'cases':len(jobs),'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
if __name__=='__main__':main()
