#!/usr/bin/env python3
"""Apply, build, validate, and freeze a generated candidate in our isolated worktree."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,subprocess
P=Path(__file__).resolve().parent
WT=Path(json.loads((P/'study.json').read_text())['worktree'])
a=argparse.ArgumentParser();a.add_argument('source');a.add_argument('label');args=a.parse_args()
out=P/'integrated'/args.label;out.mkdir(parents=True,exist_ok=False)
source=P/args.source
paths={'butteraugli.metal':'src/gpu/metal/kernels/butteraugli.metal',
 'metal_butteraugli.cpp':'src/gpu/metal/metal_butteraugli.cpp',
 'metal_backend_internal.h':'src/gpu/metal/metal_backend_internal.h'}
for name,rel in paths.items():(WT/rel).write_bytes((source/name).read_bytes())
cmd=['cmake','--build',str(WT/'build/traffic'),'--target','gjxl_encoding_benchmark','gjxl_encode',
 'gjxl_metal_butteraugli_test','gjxl_metal_aq_evaluation_test',
 'gjxl_adaptive_quantization_gpu_policy_test','-j','8']
(out/'build-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (out/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
cmd=json.loads((P/'baseline/build-probe-command.json').read_text());cmd[-1]=str(out/'encode_probe')
(out/'build-probe-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (out/'build-probe.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
for name,rel in paths.items():shutil.copy2(WT/rel,out/name)
shutil.copy2(WT/'build/traffic/metal/gjxl.metallib',out/'gjxl.metallib')
shutil.copy2(P/'encode_probe.cpp',out/'encode_probe.cpp')
shutil.copy2(WT/'build/traffic/gjxl_encode',out/'gjxl_encode')
shutil.copy2(Path(__file__),out/'build-protocol.py')
(out/'source.patch').write_text(subprocess.check_output(['git','diff','--','src/gpu/metal'],cwd=WT,text=True))
cmd=['ctest','--test-dir',str(WT/'build/traffic'),'-R',
 '^(metal_butteraugli|metal_aq_evaluation|adaptive_quantization_gpu_policy)$','--output-on-failure']
(out/'test-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (out/'tests.log').open('w') as f:
 subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,
 env={**os.environ,'MTL_DEBUG_LAYER':'1','MTL_SHADER_VALIDATION':'1'})
def sha(f):
 with f.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
identity={'source_base':json.loads((P/'study.json').read_text())['base_revision'],
 'source':str(source),'scope':'M4 Pro experiment',
 'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'files':{f.name:sha(f) for f in out.iterdir() if f.is_file()},'test_returncode':0}
(out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
print('Built and validated',args.label,flush=True)
