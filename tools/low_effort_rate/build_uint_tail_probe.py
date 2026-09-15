#!/usr/bin/env python3
"""Isolate libjxl's extra-DC-precision-triggered modular HybridUint search."""
from pathlib import Path
import subprocess
import study


def main():
    root=Path('build/e3-e4-tail-uint-probe-20260914').resolve();root.mkdir(exist_ok=True)
    assert not (root/'build.json').exists()
    original=Path('build/e3-e4-tail-probe-20260914').resolve()
    base=study.read(original/'build.json');files=dict(base['files'])
    for p,digest in files.items():assert study.sha(p)==digest,p
    source=Path('/Users/yunhocho/GitHub/gjxl-libjxl-tail/third_party/libjxl/lib/jxl/enc_ans.cc')
    text=source.read_text();marker='  if (cparams.decoding_speed_tier >= 2) {'
    assert text.count(marker)==1
    text='#include <cstdlib>\n'+text.replace(marker,'  if (std::getenv("GJXL_RCA_MODULAR_UINT_NONE"))\n    params.uint_method = HistogramParams::HybridUintMethod::kNone;\n'+marker)
    overlay=root/'enc_ans.cpp';overlay.write_text(text)
    commands=study.rows(original/'build-commands.jsonl')
    compile=commands[0]['argv'][:]
    for flag,suffix in [('-MT','.o'),('-MF','.o.d'),('-o','.o'),('-c','.cpp')]:
        compile[compile.index(flag)+1]=str(root/('enc_ans'+suffix))
    link=commands[-1]['argv'][:];link[link.index('-o')+1]=str(root/'gjxl_tail_uint_probe')
    index=next(i for i,v in enumerate(link) if v.endswith('libjxl-internal.a'))
    link.insert(index,str(root/'enc_ans.o'))
    for argv in [compile,link]:
        result=subprocess.run(argv,cwd=commands[0]['cwd'],capture_output=True,text=True)
        study.append(root/'build-commands.jsonl',{'argv':argv,'cwd':commands[0]['cwd'],'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        result.check_returncode()
    for p in [source,overlay,root/'enc_ans.o',root/'gjxl_tail_uint_probe',Path(__file__).resolve()]:files[str(p)]=study.sha(p)
    study.save(root/'build.json',{'files':files,'intervention':'GJXL_RCA_MODULAR_UINT_NONE selects default HybridUint instead of the extra-precision-triggered modular candidate search. AC writer settings and the frozen frame are unchanged.',
        'qualification':'Pending neutral tail-byte controls and exact decoded-float checks for every variant.'})
    print(root/'gjxl_tail_uint_probe')


if __name__=='__main__':main()
