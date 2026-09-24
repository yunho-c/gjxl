"""Create a separate diagnostic build without modifying the frozen first run."""
from pathlib import Path
import json, shutil, subprocess, hashlib

root = Path(__file__).resolve().parent
out = root / 'followup-code'
out.mkdir(exist_ok=True)
include = out / 'include/gpu/metal'
include.mkdir(parents=True, exist_ok=True)
header = (root/'source/src/gpu/metal/overhead_control.h').read_text()
header = header.replace('encoders=0;', 'encoders=0, counter_resolve_ns=0, counter_buffer_ns=0;')
(include/'overhead_control.h').write_text(header)
source = (root/'source/src/gpu/metal/metal_submission.cpp').read_text()
old = '''    NS::Data* resolved = g_control == OverheadControl::Record ? nullptr :
      counter_sample_buffer_->resolveCounterRange(
        NS::Range::Make(0, static_cast<NS::UInteger>(sample_count)));'''
new = '''    NS::Data* resolved = nullptr;
    if (g_control != OverheadControl::Record) {
      OverheadTimer counter_timer{g_overhead.counter_resolve_ns};
      resolved = counter_sample_buffer_->resolveCounterRange(
        NS::Range::Make(0, static_cast<NS::UInteger>(sample_count)));
    }'''
assert source.count(old)==1
source=source.replace(old,new)
old = '''  MTL::CounterSampleBuffer* raw_sample_buffer = g_control == OverheadControl::Record ? nullptr :
    device_->newCounterSampleBuffer(sample_descriptor.get(), &sample_error);'''
new = '''  MTL::CounterSampleBuffer* raw_sample_buffer = nullptr;
  if (g_control != OverheadControl::Record) {
    OverheadTimer counter_timer{g_overhead.counter_buffer_ns};
    raw_sample_buffer = device_->newCounterSampleBuffer(sample_descriptor.get(), &sample_error);
  }'''
assert source.count(old)==1
source=source.replace(old,new)
(out/'metal_submission.cpp').write_text(source)

probe=(root/'probe.cpp').read_text().replace('#include <memory>', '#include <memory>\n#include <numeric>\n#include <random>')
old='''  for (int pair=0; pair<pairs; ++pair) {
    auto order = modes;
    std::rotate(order.begin(), order.begin() + (pair+seed)%order.size(), order.end());
    if (seed%2) std::reverse(order.begin(), order.end());
    for (const auto& mode : order) {
      for (int rep=-warmups; rep<samples; ++rep) {'''
new='''  // Williams balanced schedules: positions and first-order carryover balance
  // over each six rounds. Shuffle the rows separately in each replicate.
  if (modes.size()!=6 || samples!=1 || pairs%6!=0)
    throw std::runtime_error("Follow-up requires six modes, one sample, rounds multiple of six");
  const std::vector<size_t> base{0,1,5,2,4,3};
  std::vector<int> schedule;
  std::mt19937 rng(20260921 + seed);
  for (int block=0; block<pairs/6; ++block) {
    std::vector<int> rows{0,1,2,3,4,5};
    std::shuffle(rows.begin(),rows.end(),rng);
    schedule.insert(schedule.end(),rows.begin(),rows.end());
  }
  for (int pair=-1; pair<pairs; ++pair) {
    auto order = modes;
    if (pair>=0) for (size_t i=0;i<base.size();++i)
      order[i]=modes[(base[i]+schedule[pair])%modes.size()];
    for (const auto& mode : order) {
      for (int rep=(pair<0 ? -warmups : 0); rep<(pair<0 ? 0 : samples); ++rep) {'''
assert probe.count(old)==1
probe=probe.replace(old,new)
old='''          << ",\\\"encoders\\\":" << stats.encoders;'''
new='''          << ",\\\"encoders\\\":" << stats.encoders
          << ",\\\"counter_resolve_ns\\\":" << stats.counter_resolve_ns
          << ",\\\"counter_buffer_ns\\\":" << stats.counter_buffer_ns;'''
assert probe.count(old)==1, old
probe=probe.replace(old,new)
(out/'probe.cpp').write_text(probe)

runner=(root/'run.py').read_text()
runner=runner.replace("ROOT=Path(__file__).resolve().parent", "ROOT=Path(__file__).resolve().parent.parent")
runner=runner.replace("sha(ROOT/'probe.cpp')", "sha(ROOT/'followup-code/probe.cpp')")
runner=runner.replace("x.pairs*(x.samples+x.warmups)*len(identity['modes'])", "(x.pairs*x.samples+x.warmups)*len(identity['modes'])")
(out/'run.py').write_text(runner)

jobs=json.loads((root/'ablation-jobs.json').read_text())
# Fixed-DCT8 control, mixed-transform AQ, and four-iteration AQ on two sizes.
jobs=[j for j in jobs if j['label'] in ('kodak','12mp') and j['effort'] in (4,7,9)]
(out/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')

build=root/'ablation-build'
commands=[
 ['/usr/bin/c++','-O3','-DNDEBUG','-std=gnu++20','-arch','arm64',
  '-I'+str(out/'include'),'-I'+str(build/'metal'),'-I'+str(root/'source/third_party/metal-cpp'),
  '-I'+str(root/'source/src'),'-c',str(out/'metal_submission.cpp'),'-o',str(out/'metal_submission.cpp.o')],
 ['ar','rcs',str(out/'libgjxl_metal.a'),str(out/'metal_submission.cpp.o')],
 ['/usr/bin/c++','-O3','-DNDEBUG','-std=gnu++20','-arch','arm64','-DGJXL_OVERHEAD_ABLATION',
  '-I'+str(out/'include'),'-I'+str(root/'source/src'),'-I'+str(root/'source/benchmarks'),
  str(out/'probe.cpp'), str(build/'libgjxl_codestream.a'),str(build/'libgjxl_pfm_io.a'),str(out/'libgjxl_metal.a'),
  *[str(build/('libgjxl_'+name+'.a')) for name in ('gpu_butteraugli','gpu_ops','codec','gpu')],
  '-framework','Metal','-framework','Foundation','-framework','CoreGraphics','-o',str(out/'probe')]
]
(out/'build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
shutil.copy2(build/'libgjxl_metal.a',out/'libgjxl_metal.a')
for command in commands: subprocess.run(command,check=True)
files=[out/'probe',out/'probe.cpp',out/'run.py',out/'metal_submission.cpp',include/'overhead_control.h',out/'libgjxl_metal.a']
files+=list(build.glob('libgjxl_*.a'))
(out/'sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print('Built',out/'probe')
