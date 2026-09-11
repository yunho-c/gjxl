"""Compare production policy against same-build forced-32/64 oracles."""
from pathlib import Path
import hashlib,json,os,shlex,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
BUILD=ROOT/'build/release';OUT=ROOT/'build/ac-cap64-qualification'
PRIOR=ROOT.parent/'gjxl-writer-rate-attribution/build/ac-cap/study-v1'
DECODER=ROOT.parent/'gjxl-metal-expanded-resident-qualify/build/release/pinned-libjxl/tools/djxl'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def build():
    OUT.mkdir(parents=True,exist_ok=True)
    commands=subprocess.check_output(['ninja','-C',str(BUILD),'-t','commands','gjxl_encoding_benchmark'],text=True).splitlines()
    records=[]
    def compile_like(original,source,obj):
        command=shlex.split(next(c for c in commands if ' -c '+str(original) in c))
        old=command[command.index('-o')+1]
        args=[str(source) if a==str(original) else str(obj) if a==old else str(obj)+'.d' if a==old+'.d' else a for a in command]
        subprocess.run(args,cwd=BUILD,check=True);records.append(args)
    probe=OUT/'probe.cpp';probe.write_text((HERE/'probe.inc').read_text().replace('BENCHMARK_SOURCE',str(ROOT/'benchmarks/encoding_benchmark.cpp')))
    compile_like(ROOT/'benchmarks/encoding_benchmark.cpp',probe,OUT/'probe.o')
    current=(ROOT/'src/codestream/encoder.cpp').read_text()
    baseline=subprocess.check_output(['git','show','HEAD:src/codestream/encoder.cpp'],cwd=ROOT,text=True)
    key='codestream_internal::AcAnsClusterLimit(\n              frame.geometry().frame())';assert current.count(key)==1
    forced=current.replace(key,'gjxl::kMaximumAnsClusters')
    links=[];binaries={}
    for mode,source in [('baseline',baseline),('forced64',forced),('policy',None)]:
        objects=[str(OUT/'probe.o')]
        if source is not None:
            p=OUT/(mode+'.cpp');p.write_text(source)
            compile_like(ROOT/'src/codestream/encoder.cpp',p,OUT/(mode+'.o'));objects.append(str(OUT/(mode+'.o')))
        args=shlex.split(commands[-1])[2:-2]
        i=args.index('CMakeFiles/gjxl_encoding_benchmark.dir/benchmarks/encoding_benchmark.cpp.o');args[i:i+1]=objects
        args=[str(OUT/mode) if a=='gjxl_encoding_benchmark' else a for a in args]
        subprocess.run(args,cwd=BUILD,check=True);links.append(args);binaries[mode]=sha(OUT/mode)
    files={p for p in ROOT.joinpath('src').rglob('*') if p.suffix in ['.h','.cpp']}
    files|={p for p in HERE.iterdir() if p.is_file()};files|={ROOT/'benchmarks/encoding_benchmark.cpp'}
    for link in links:
        files|={Path(a) if Path(a).is_absolute() else BUILD/a for a in link if a.endswith('.a') or a.endswith('.o')}
    files|={p for p in OUT.glob('*.cpp')}
    save(OUT/'build.json',{'base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'commands':records,'links':links,'binaries':binaries,'files':{str(p):sha(p) for p in sorted(files)},'decoder_sha256':sha(DECODER)})
def collect():
    b=load(OUT/'build.json')
    for p,d in b['files'].items():assert sha(p)==d,p
    for mode,d in b['binaries'].items():assert sha(OUT/mode)==d
    assert sha(DECODER)==b['decoder_sha256']
    cases=load(PRIOR/'manifest.json')['cases'];results=[]
    identity={'build':b,'cases':cases}
    p=OUT/'identity.json'
    if p.exists():assert load(p)==identity
    else:save(p,identity)
    for c in cases:
        item=c['input'];assert sha(item['path'])==item['sha256']
        name=f"{item['name']}-{c['frontend']}-{c['distance']:g}";folder=OUT/'runs'/name;done=folder/'complete.json'
        if done.exists():
            r=load(done)
            for p,d in r['files'].items():assert sha(p)==d
            results.append(r);continue
        folder.mkdir(parents=True,exist_ok=True);attempt=folder/f'attempt-{len(list(folder.glob("attempt-*"))):03d}';attempt.mkdir()
        rows={};commands=[]
        for mode in b['binaries']:
            args=[str(OUT/mode),item['path'],str(attempt/(mode+'.jxl')),str(c['distance']),c['frontend']]
            with (attempt/(mode+'.json')).open('w') as out,(attempt/(mode+'.log')).open('w') as err:
                subprocess.run(args,stdout=out,stderr=err,check=True,timeout=180)
            commands.append(args);rows[mode]=load(attempt/(mode+'.json'));rows[mode]['sha256']=sha(attempt/(mode+'.jxl'))
        large=item['width']*item['height']>=3840*2160
        assert rows['policy']['sha256']==rows['forced64' if large else 'baseline']['sha256']
        assert rows['policy']['ac_clusters']<=(64 if large else 32)
        assert rows['policy']['dc_clusters']==rows['baseline']['dc_clusters']
        decoded=[]
        for mode in ['baseline','policy']:
            target=attempt/(mode+'.pfm');args=[str(DECODER),str(attempt/(mode+'.jxl')),str(target),'--num_threads=8']
            with (attempt/(mode+'-decode.log')).open('w') as out:subprocess.run(args,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=180)
            commands.append(args);decoded.append(sha(target));target.unlink()
        assert decoded[0]==decoded[1]
        r={'case':c,'rows':rows,'decoded_sha256':decoded[0],'commands':commands,'files':{str(p):sha(p) for p in attempt.iterdir() if p.is_file()},'complete':True}
        save(done,r);results.append(r);print(name,rows['baseline']['bytes'],rows['policy']['bytes'],rows['policy']['ac_clusters'],flush=True)
    assert len(results)==112
    save(OUT/'summary.json',{'complete':True,'cases':112,'codestreams':336,'encodes':672,'decoder_comparisons':112,'results':results})
if __name__=='__main__':
    if sys.argv[1]=='build':build()
    elif sys.argv[1]=='collect':collect()
    else:raise SystemExit('build|collect')
