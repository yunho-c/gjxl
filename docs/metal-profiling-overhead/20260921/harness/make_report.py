from pathlib import Path
import json, statistics as st, hashlib, datetime, subprocess

ROOT=Path(__file__).resolve().parent
follow=json.loads((ROOT/'followup/summary.json').read_text())
broad=json.loads((ROOT/'ablation/summary.json').read_text())
assert len(follow)==6 and len(broad)==20

def ci(v, unit='ms'):
    key='median_percent' if unit=='%' else 'median_delta_ms'
    bounds='bootstrap_95ci_percent' if unit=='%' else 'bootstrap_95ci_ms'
    return f"{v[key]:+.2f} [{v[bounds][0]:+.2f}, {v[bounds][1]:+.2f}]"
def label(j):
    return f"{j['width']*j['height']/1e6:.2f} MP / e{j['effort']}"
def raw(j):
    return [r for l in (ROOT/'followup'/(j['case']+'.jsonl')).read_text().splitlines()
            if (r:=json.loads(l))['rep']>=0]

wall=['| Image / effort | Ordinary (ms) | Full (ms) | Full minus ordinary (%) | Full minus host-profile (%) |',
      '|---|---:|---:|---:|---:|']
split=['| Image / effort | Encoders: graph → split | Complete encode (ms) | GPU command spans (ms) | Submission host time (ms) |',
       '|---|---:|---:|---:|---:|']
host=['| Image / effort | Metadata recording Δ | Metadata postprocess | Counter-buffer creation | Other timestamp setup Δ | Counter resolution | Full postprocess |',
      '|---|---:|---:|---:|---:|---:|---:|']
for j in follow:
    m=j['mode_median_ms']; c=j['contrasts']
    wall.append(f"| {label(j)} | {m['ordinary']['wall_ns']:.2f} | {m['full']['wall_ns']:.2f} | {ci(c['total_profile']['wall_ns'],'%')} | {ci(c['gpu_profile']['wall_ns'],'%')} |")
    enc=j['encoders']; s=c['encoder_split']
    split.append(f"| {label(j)} | {enc['graph'][0]} → {enc['split'][0]} | {ci(s['wall_ns'])} | {ci(s['all_command_gpu_ns'])} | {ci(s['submission_host_ns'])} |")
    r=raw(j); d={(v['pair'],v['mode']):v for v in r}; n=j['pairs']
    other=st.median((d[i,'full']['submission_host_ns']-d[i,'full']['counter_buffer_ns']-d[i,'record']['submission_host_ns'])/1e6 for i in range(n))
    host.append(f"| {label(j)} | {c['recording']['submission_host_ns']['median_delta_ms']:.3f} | {m['record']['resolution_host_ns']:.3f} | {m['full']['counter_buffer_ns']:.3f} | {other:.3f} | {m['full']['counter_resolve_ns']:.3f} | {m['full']['resolution_host_ns']:.3f} |")

checked=[]
for cohort, count in [('ablation',20),('followup',6)]:
    p=ROOT/cohort
    assert json.loads((p/'complete.json').read_text())['cases']==count
    for f in sorted(p.glob('*.ok.json')):
        ok=json.loads(f.read_text());stem=f.name.removesuffix('.ok.json');data=p/(stem+'.jsonl')
        assert hashlib.sha256(data.read_bytes()).hexdigest()==ok['raw_sha256']
        rows=[json.loads(l) for l in data.read_text().splitlines()]
        assert all(r['byte_equal'] for r in rows)
        assert len({r['submissions'] for r in rows})==1
        assert len({r['encoded_bytes'] for r in rows})==1
        assert all(r['encoders']==r['submissions'] for r in rows if r['mode'] in ('ordinary','host','graph'))
        assert len({r['encoders'] for r in rows if r['mode'] in ('split','record','full')})==1
        command=json.loads((p/(stem+'.command.json')).read_text())
        assert not command['interference'] and command['returncode']==0
        checked.append({'collection':cohort,'case':stem,'rows':len(rows),'measured':sum(r['rep']>=0 for r in rows),'output_sha256':ok['output_sha256']})
for f in [v for v in checked if v['collection']=='followup']:
    matches=[v for v in checked if v['collection']=='ablation' and v['case'].split('-',1)[1]==f['case'].split('-',1)[1]]
    assert len(matches)==1 and matches[0]['output_sha256']==f['output_sha256']
(ROOT/'verification.json').write_text(json.dumps({'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':checked,'measured_calls':sum(v['measured'] for v in checked),'all_logged_calls':sum(v['rows'] for v in checked),'cross_build_output_hashes_match':True},indent=2)+'\n')

text='''# Production-aligned GPU profiling overhead

Commit: `3e1ef9e` (`perf/profile-production-alignment`), local and not merged to main.
Date: 2026-09-21. Device: Apple M4 Pro, 20 GPU cores, 48 GiB; macOS 15.6.

## Decision

See `DECISION.md` for the interpretation of these measurements.

## Follow-up: complete encode

Each comparison is a within-round difference or ratio; cells in brackets are
95% paired bootstrap intervals for the median (4,000 resamples, fixed seed).
Ordinary and Full columns are separate medians, so their ratio need not equal
the median paired ratio. Negative differences do not establish a production
speedup from profiling. The baseline is an ordinary complete resident encode,
with the same small diagnostic observation scopes used by every control.

'''+ '\n'.join(wall)+'''

## #2: splitting compute encoders

`split − graph` holds the stage callback graph and command-buffer count fixed.
Both controls disable dispatch recording and timestamps. This isolates the
encoder boundaries much more directly than ordinary-versus-profiled timing.

'''+ '\n'.join(split)+'''

GPU command spans are `GPUEndTime − GPUStartTime`, summed across completed
command buffers, including input preparation. They are elapsed GPU spans,
not isolated active-kernel occupancy, and can still reflect scheduling noise.
The submission host interval includes encoding and commit, but not the
completion wait. It is elapsed host time, not OS-accounted CPU time.

## #3: recording and counter processing

All values below are milliseconds per complete encode. Metadata recording is
the paired `record − split` submission-time delta. Metadata postprocessing is
the direct duration of the no-counter `GpuProfile` path (profile copy, indirect
grid reads, and publication). Counter-buffer creation and counter resolution
are separately timed around the Metal calls. Other timestamp setup is the
paired submission-time delta `full − record`, less direct counter-buffer
creation; it includes timestamp attachments, different pass descriptors, and
associated driver work. Full postprocess contains counter resolution, profile
copy/validation, and publication. These columns overlap and must not be summed.

'''+ '\n'.join(host)+'''

## Protocol and controls

- Metal fully resident, distance 1.2, eight CPU threads, final score disabled.
  Release / AppleClang 17; production shaders, frontier experiments disabled.
- 20-case broad experiment: four inputs (128×96 synthetic, Kodak 768×512,
  CLIC 1507×2048, Unsplash 4249×2824), efforts 2/4/7/9/10. Six balanced
  blocks, two per-mode warmups and three measured calls per block.
- Six-case follow-up: Kodak and 12 MP, efforts 4/7/9; 36 rounds, one call
  per mode per round. Six Williams schedules balance position and first-order
  carryover; schedule rows are shuffled within each six-round replicate.
  Two initial warmups per mode. Both experiments also have eight initial
  untimed ordinary burn-in calls after the reference encode.
- Every call is checked against the ordinary reference codestream and complete
  encoding summary. Submission counts are equal across all six controls.
  Encoder counts agree for ordinary/host/graph, and for split/record/full.
- `ordinary`: ordinary workflow; no host or GPU profile.
- `host`: host workflow timers only; ordinary GPU encoders.
- `graph`: GPU-profiled stage callbacks, all in one encoder per original
  submission; no dispatch graph or timestamps; dummy profile for orchestration.
- `split`: the same callbacks, with one encoder per stage; no recording or
  timestamps; dummy profile for orchestration.
- `record`: normal split path, dispatch graph, retained indirect arguments,
  profile copy and grid resolution; no counter buffer or timestamp attachment.
  It retains small counter-capability/descriptor scaffolding from the full path.
- `full`: the committed stage profiler, plus measurement scopes.
- Image loading, backend/pipeline construction, output file I/O, output
  equality checking, JSON output, and destruction of returned outputs are
  outside the timed complete API call. Host profile/GPU profile construction
  and publication inside the API remain timed. This is not CLI startup timing.
- `record − split` approximates recording work; `full − record` includes both
  timestamp sampling and processing. They should not be described as pure
  GPU costs or as exactly additive end-to-end penalties.

## Interference and scope

The initial sweep overlapped background indexing and active desktop/remote
display work. The user then left the Mac idle for the follow-up. Background
system services and WindowServer remained active; no attempt was made to
terminate them. No competing codec/build/test job was detected during either
accepted collection, and the saved thermal snapshots reported no thermal or
performance warning. The follow-up is more informative but is not a controlled
headless/device-isolated performance qualification. Bootstrap intervals capture
observed run variability, not every source of systematic bias.

The experiment has one natural image in each size stratum, one target distance,
one device, and selected representative efforts. Do not extrapolate a universal
overhead percentage, claim production gains from negative deltas, or treat
profiled timestamps as an unperturbed production trace. Direct elapsed host
scopes can include preemption. No shader arithmetic or command-buffer
boundaries were changed by the controls, and the diagnostic and production
metallib hashes match.

## Artifacts and reproduction

- `ablation/identity.json`, `ablation/summary.json`: broad-run identity and all
  paired comparisons, including GPU/serializer/quantization and host metrics.
- `followup/identity.json`, `followup/summary.json`: final alternating follow-up.
- Each case retains raw JSONL (including warmups), command, exit status,
  interference log, stderr, reference codestream, and SHA256 completion record.
- `verification.json`: output equality, submission/encoder checks, raw hashes,
  and cross-build codestream equality for the six shared cases.
- `probe.cpp`, `run.py`, `patch_ablation.py`, `ablation.patch`, and the retained
  `source/src/gpu/metal/overhead_control.h` reproduce the first controls.
  `source/` is the commit archive plus that diagnostic-only patch/header.
- `followup-code/` retains the exact second probe, replacement Metal source,
  header, archive, binary, driver, job list, build commands, and SHA256 manifest.
  `prepare_followup.py` produces these from the frozen first diagnostic build.
- `analyze.py`, `analyze_followup.py`, `make_report.py` reproduce analysis.
- `environment.txt`, `background-activity.json`, and follow-up environment
  snapshots preserve machine/thermal/activity context. `pilot/` is exploratory
  original-library evidence and is not the basis for the decision.
- All diagnostic switches and measurements live under this ignored build
  directory. Production tracked files remain unchanged after the commit.

From the worktree, rerun the frozen follow-up with a fresh collection name:

```sh
python3 build/profiling-overhead-20260921/followup-code/run.py \\
  --name followup-repeat \\
  --probe build/profiling-overhead-20260921/followup-code/probe \\
  --jobs build/profiling-overhead-20260921/followup-code/jobs.json \\
  --modes ordinary,host,graph,split,record,full --pairs 36 --samples 1 --warmups 2
python3 build/profiling-overhead-20260921/analyze_followup.py followup-repeat
```
'''
(ROOT/'REPORT.md').write_text(text)
print('Wrote REPORT.md and verification.json;',sum(v['measured'] for v in checked),'measured calls verified')
