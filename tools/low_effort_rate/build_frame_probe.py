#!/usr/bin/env python3
"""Add checked native section accounting to the independently controlled DC probe."""
from pathlib import Path
import shlex
import subprocess
import study


def main():
    source=Path('/Users/yunhocho/GitHub/gjxl-runtime-main-e1-4-20260914')
    build=source/'build/quality-study'
    precision=Path('build/e3-e4-rate-20260914/precision-probe').resolve()
    output=Path('build/e3-e4-frame-probe-20260914').resolve();output.mkdir(exist_ok=True)
    old=study.read(precision/'build.json')
    for p,h in old['files'].items():assert study.sha(p)==h,p
    text=(source/'src/codestream/encoder.cpp').read_text()
    text=text.replace('#include <cstddef>','#include <cstddef>\n#include <cstdlib>\n#include <cmath>\n#include <fstream>\n#include <iomanip>\n#include <map>')
    hook='    *output = std::move(candidate_output);'
    assert text.count(hook)==1
    capture=output/'native_capture.inc'
    capture.write_bytes(Path('tools/low_effort_rate/native_capture.inc').read_bytes())
    text=text.replace(hook,f'#include "{capture}"\n'+hook)
    overlay=output/'encoder.cpp';overlay.write_text(text)
    obj=output/'encoder.cpp.o'
    commands=[['/usr/bin/c++','-O3','-DNDEBUG','-std=gnu++20','-arch','arm64',
               '-I'+str(source/'src'),'-c',str(overlay),'-o',str(obj)]]
    link=shlex.split((build/'CMakeFiles/gjxl_quality_benchmark.dir/link.txt').read_text())
    for i,arg in enumerate(link):
        if arg.endswith(('.o','.a')):link[i]=str(build/arg)
    link[link.index('-o')+1]=str(output/'gjxl_frame_probe')
    at=link.index(str(build/'libgjxl_codestream.a'))
    link[at:at]=[str(precision/'workflow.cpp.o'),str(obj)]
    commands.append(link)
    for argv in commands:
        result=subprocess.run(argv,capture_output=True,text=True)
        study.append(output/'build-commands.jsonl',{'argv':argv,'returncode':result.returncode,
                    'stdout':result.stdout,'stderr':result.stderr})
        result.check_returncode()
    files=dict(old['files'])
    for p in (output/'gjxl_frame_probe',overlay,obj,capture,Path(__file__).resolve(),
              Path('tools/low_effort_rate/native_capture.inc').resolve()):
        files[str(p)]=study.sha(p)
    study.save(output/'build.json',{'kind':'diagnostic native frame and section accounting',
        'base_revision':old['base_revision'],'files':files,
        'precision_control':'GJXL_RCA_DC_PRECISION','capture':'GJXL_FINAL_FIELD_RATE_DUMP',
        'accounting':'DC-group re-emission must match actual emitted bytes and bit lengths; model costs remain shared',
        'fingerprints':'Canonical integer planes/used coefficient values; FNV-1a-64, diagnostic identity checks',
        'qualification':'pending output byte controls before attribution'})
    print(output/'gjxl_frame_probe')


if __name__=='__main__':main()
