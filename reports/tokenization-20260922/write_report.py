#!/usr/bin/env python3
"""Build the decision report from completed, validated saved measurements."""
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def main():
    data = json.loads((ROOT / 'summary.json').read_text())
    decoded = json.loads((ROOT / 'decode/validation.json').read_text())
    assert len(decoded['rows']) == 18
    rows = data['profile']
    names = dict(zip(dict.fromkeys(r['image'] for r in rows),
                     ['Kodak 01 (0.39 MP)', 'CLIC A (3.09 MP)', 'CLIC B (3.36 MP)',
                      'Campus (12 MP)', 'Alpine (24 MP)', 'Forest (48 MP)']))
    ptable = table(['Image', 'Effort', 'Main ms', 'DC8 ms', 'CPU+ ms', 'GPU+overlap ms', 'Saving vs main', 'Saving vs CPU+'],
        [[names[r['image']], r['effort'], *(f"{r[m+'_ordinary_ms']:.2f}" for m in ['base','cpu','release','joined1']),
          f"{r['combined_saving_percent_median']:.1f}%", f"{r['gpu_over_cpu_saving_percent_median']:.1f}%"] for r in rows])
    btable = table(['Image', 'Effort', 'Batch', 'Main fps', 'CPU+ fps', 'GPU fps', 'GPU/CPU+ paired ratio (range)', 'Admitted in flight'],
        [[names[r['image']], r['effort'], r['batch'], *(f"{r[m+'_fps']:.2f}" for m in ['base','release','joined']),
          f"{r['gpu_over_cpu_throughput_ratio_median']:.3f} ({r['gpu_over_cpu_throughput_ratio_minimum']:.3f}–{r['gpu_over_cpu_throughput_ratio_maximum']:.3f})",
          f"{r['base_in_flight']}/{r['release_in_flight']}/{r['joined_in_flight']}"] for r in data['batch']])
    coldtable = table(['Image', 'Effort', 'Main ms', 'CPU+ ms', 'GPU ms'],
        [[names[r['image']], r['effort'], *(f"{r[m+'_first_encode_ms']:.1f}" for m in ['base','release','joined'])]
         for r in data['batch'] if r['batch'] == 1])
    large = [r for r in rows if r['image'].startswith('unsplash/')]
    ranges = {}
    for effort in [1,7,8]:
        for label in ['combined','gpu_over_cpu']:
            values = [r[label+'_saving_percent_median'] for r in large if r['effort']==effort]
            ranges[effort,label] = f'{min(values):.1f}–{max(values):.1f}%'
    v = data['validation']
    peak = max(r[m+'_peak_backing_gib'] for r in data['batch'] for m in ['base','release','joined'])
    target = next(r for r in rows if 'alpine_lake' in r['image'] and r['effort']==7)
    group = json.loads((ROOT / 'group-summary.json').read_text())
    gtable = table(['Image, e1', 'CPU+ ms', 'V7 GPU ms', 'Fused GPU ms', 'Saving vs V7 (range)'],
        [[names[r['image']], *(f"{r[m+'_ordinary_ms']:.2f}" for m in ['release','joined1','group256']),
          f"{r['joined1_saving_median']:.1f}% ({r['joined1_saving_min']:.1f}–{r['joined1_saving_max']:.1f}%)"] for r in group['profile']])
    gbtable = table(['Image, e1 B4', 'Fused/V7 throughput (range)', 'Fused/CPU+ throughput'],
        [[names[r['image']], f"{r['joined_ratio_median']:.3f} ({r['joined_ratio_min']:.3f}–{r['joined_ratio_max']:.3f})",
          f"{r['release_ratio_median']:.3f}"] for r in group['batch'] if r['batch']==4])
    phase = table(['Profiled 24 MP/e7 boundary', 'Main ms', 'CPU+ ms', 'GPU+overlap ms'],
        [[label, *(f"{target[m+'_'+key]:.2f}" for m in ['base','release','joined1'])] for label,key in
         [('Complete profiled call','profiled_ms'),('Quantization pipeline','quantization_pipeline_ms'),
          ('Complete serializer','codestream_encoding_ms'),('DC phase','dc_ms'),('Exposed AC phase','ac_ms'),
          ('Entropy optimization','codestream_entropy_optimization_ms'),('Section writing','codestream_section_writing_ms')]])
    text = f'''# GPU-resident tokenization and CPU DC scheduling on M4 Pro

Decision report, 22 September 2026. Measurements were collected in `perf/gpu-tokenization-20260922`, based on `7659cac8d933a9f6d41f1155503c7ca99fc46e6c`. The qualified implementation is committed as `59d976152937a5a2e171c87be83254569273a3c9`. The original dirty main checkout was preserved. The main tables describe the fully qualified V7 path; a subsequent pure-DCT8 group-fusion experiment has its own repeated comparison below.

**Finding.** GJXL had substantial achievable headroom on this machine. Exact GPU AC tokenization is feasible, and the combined implementation reduces ordinary complete-call latency on the three 12–48 MP images by **{ranges[1,'combined']} at e1, {ranges[7,'combined']} at e7, and {ranges[8,'combined']} at e8**. The incremental reduction against parallel DC plus parallel CPU token cleanup is **{ranges[1,'gpu_over_cpu']} / {ranges[7,'gpu_over_cpu']} / {ranges[8,'gpu_over_cpu']}**, respectively. These are ranges of the three image-specific paired medians, not a universal speedup claim.

**Checkpoint policy.** This commit preserves the measured experimental controls: GPU tokenization remains opt-in, while parallel DC groups are enabled in the candidate. Parallel CPU token cleanup remains a control and a possible separate CPU optimization. The observed gains support a subsequent reversible default-on rollout with an explicit CPU override, followed by broader measurements to refine automatic selection. That is a deployment choice under incomplete workload coverage; this study does not establish universal superiority or an absolute hardware optimum. The known 24 MP/e1 batch regression remains part of the evidence.

## What changed

Tracked source changes include [DC-group scheduling](../../src/codestream/dc_group.cpp), the [serializer producer/overlap seam](../../src/codestream/encoder.cpp), the [resident handoff](../../src/codestream/workflow.cpp), and concurrent scratch/idle-pool bounds. The implementation also includes the [Metal host producer](../../src/gpu/metal/metal_ac_tokenization.cpp), [kernels](../../src/gpu/metal/kernels/ac_tokenization.metal), [provider interface](../../src/codestream/ac_tokenization_provider_internal.h), and independent oracle/failure tests. This report, plots, aggregate measurements, validation logs, and provenance are tracked. Individual captures, frozen executables, and screening archives remain local, ignored artifacts; [the artifact inventory](ARTIFACTS.md) distinguishes the two.

- CPU workers tokenize independent DC groups into fixed output slots. Each group's weighted predictor remains sequential and unchanged; publication order and coding decisions stay exact. CPU participation and concurrent scratch are bounded.
- The GPU borrows the existing final quantized coefficient buffer. It computes transform nonzeros and last positions, completes the top/left prediction map, scans exact token offsets, then emits ordered values/contexts and integer fixed-HybridUint populations. The CPU still selects orders, contexts, entropy models, and writes the final bitstream.
- A compact shared-buffer lease supplies split token spans directly to the CPU. A capacity hint permits one warm GPU submission. Writes are bounded; overflow grows the buffer and retries before token spans are published. Buffers use the existing budget-aware cache and remain owned until readers finish.
- CPU DC work runs between GPU Begin and Finish. The selected emitter uses one thread per DCT8 channel only for pure-DCT8 frames; mixed frames retain the SIMD emitter. One histogram shard and 128 threads per threadgroup were selected after screening.
- The CPU control parallelizes destruction of large direct-token buffers after their readers finish. This exposed meaningful complete-call cost outside the narrow AC phase timer. Its contribution must not be attributed to GPU arithmetic.

```mermaid
flowchart LR
    A[Resident final AC coefficients] --> B[CPU orders and context map]
    B --> C[GPU metadata, scans, tokens, populations]
    B --> D[Parallel CPU DC groups]
    C --> E[CPU entropy models and section writers]
    D --> E
    E --> F[Final codestream]
```

## Complete-call results

Apple M4 Pro, 14 CPU / 20 GPU cores, 48 GB; macOS 15.6; Apple clang 17 / Xcode 26.3. Fully resident Metal, nominal Q80/distance 1.9, eight CPU participants. Separate Release builds use the same parent revision and the same capture harness. Single-image captures use the production-aligned backend-injected workflow helper with a persistent backend; the batch harness uses ordinary public APIs. Six frozen inputs, efforts 1/7/8, five process rounds, four ordinary/profiled pairs per process: **360 processes, 1,440 ordinary and 1,440 profiled measured calls**, plus references/warmups. All outputs match across methods and rounds.

`Main` is the untouched parent encoder. `DC8` adds bounded DC-group parallelism. `CPU+` adds parallel CPU token cleanup. `GPU+overlap` adds the selected GPU producer and CPU/GPU overlap to DC parallelism. Timing columns are medians of five process medians. Percentage columns are medians of the five same-round relative reductions; they can differ slightly from the ratio of displayed timing medians. Round order reverses/rotates. Input loading, equality checks, and destruction of returned public results are outside the complete-call timer; internal serializer cleanup is inside.

{ptable}

![Paired complete-call reductions](complete-call.png)

Whiskers are the observed range across five process rounds, not confidence intervals. Small images and small percentage differences require particular care. [All per-round ranges and phase summaries](profile-summary.csv), [raw process results](final-profile/results.json), and [plot PDF](complete-call.pdf) are retained.

## Batch throughput and first-call costs

The independent public batch harness repeats the same input B1/B4, with eight aggregate CPU participants and a **36 GiB managed-memory budget** shared by every variant. Three process rounds and five timed batch calls per process yield **108 processes / 540 timed batches**. The persistent batch driver is warmed first. Output validation and public-result destruction are outside the batch timer. B4 denotes four queued images; admission can reduce simultaneous work. The final column lists main/CPU+/GPU admitted counts.

{btable}

The 24 MP/e1 B4 result is a repeatable regression: GPU throughput is 5–13% below CPU+ across the three rounds, with all four images admitted in every variant. This prevents a blanket batch recommendation. At 48 MP/e1, main/CPU+ admit two images while the GPU plan admits one; this row measures the common-budget policy outcome, not equal-concurrency kernel efficiency. At 24 MP/e7 all variants admit two images, and the incremental GPU throughput gain is about 3%.

The unrestricted 48 MP/e7 B4 pilot reached approximately 44 GiB of managed backing and paged heavily; its unstable timings are excluded from the qualified comparison. A 24 GiB pilot could not admit the 48 MP plan. At 36 GiB, 48 MP/e7 admits one image and trims caches after each image. Its B4 rows therefore do not demonstrate four-way CPU/GPU overlap. The budget limits reserved/backed managed capacity, not RSS; caller inputs, immutable setup, and driver allocations are excluded. Maximum observed backing across the final batch matrix was {peak:.2f} GiB. All observed CPU peaks were at most eight, and all committed-capacity checks passed.

System-wide VM counters increased in {v['pageout_processes']} processes for pageouts and {v['swapout_processes']} for swapouts. These include background activity and are retained per process in [batch details](batch-process-details.csv); they should be considered when interpreting small effects. [Batch summary](batch-summary.csv) includes backing peaks, admission counts, and cache trimming.

The following is the first ordinary encode in each new B1 process, before warmup (median of three). It includes first-use backend/pipeline preparation. OS/driver disk caches were not reset, so these are not cold-hardware measurements.

{coldtable}

## Further pure-DCT8 group fusion

One additional experiment assigns a whole AC group to one GPU threadgroup. It keeps nonzero/last-position metadata in about 6 KiB of threadgroup memory, computes contiguous thread-slice prefixes, reserves a physical group interval atomically, and emits scalar per-channel streams. Group-indexed CPU views preserve the exact logical group and token order. Guarded overflow retries clear partial populations and recompute. The generic path remains available for mixed-transform frames.

At 24 MP/e1, screening reduced GPU token command time from about 4.4 ms to 2.1 ms with 256 threads. This fit behind DC work. Independent follow-up qualification used all six inputs at e1, five process rounds and four pairs per process (**90 processes / 360 ordinary samples**), plus three-image B1/B4 tests (**54 processes / 270 timed batches**) under the same 36 GiB / eight-CPU budget. Normal mixed-frame threadgroup width remains 128; the fused width is controlled separately.

{gtable}

{gbtable}

**Fusion decision:** retain it as an optional large-image latency specialization. It saved another 1.7% at 24 MP and 3.4% at 48 MP in the paired single-image medians, with positive reductions in all five rounds for those two images. The 12 MP result was inconclusive and small-image variation was substantial. It did not cure the 24 MP/e1 B4 regression; in the follow-up it was about 2.3% below V7 GPU throughput there. Halving device time therefore did not establish a batch gain. The experiment does not isolate the remaining cache, allocation, scheduling, or CPU-consumption costs.

All follow-up outputs match the independently decoded V7 codestreams. There were no unnamed dispatches, pageouts, or swapouts in the retained follow-up runs. [Follow-up latency data](group-profile-summary.csv), [batch data](group-batch-summary.csv), and [validation](group-summary.json) are separate from the main matrix; do not combine their absolute medians across runs as a new paired comparison against main.

## Why the gains exceed an AC-phase-only estimate

The preceding saved-profile feasibility analysis bounded an isolated AC-phase replacement. This implementation also parallelizes DC, changes token allocation/destruction and ownership, and overlaps GPU work with CPU DC. Some savings were outside the original fine AC timer. The stronger CPU control is necessary to identify the additional GPU-path benefit.

{phase}

With overlap, AC wall time excludes the interval attributed to CPU DC and represents exposed preparation/wait/reduction. It is not GPU execution duration. Fine worker-work totals can overlap and are not additive wall latency. Profiled timings are explanatory; ordinary complete calls are the primary performance boundary. GPU AC kernels are separately instrumented in the tuning screens; the regular AQ GPU-stage JSON does not include serializer token kernels.

The optimized 24 MP/e7 quantization pipeline still occupies about {100*target['joined1_quantization_pipeline_ms']/target['joined1_profiled_ms']:.0f}% of profiled latency. Its AC strategy evaluation, perceptual evaluation, and reconstruction remain the larger budget for further substantial high-effort gains. [GPU group summaries](gpu-stage-summary.csv) provide current-revision stage evidence. GPU utilization or bandwidth alone would not establish a percentage of theoretical encoder peak; this experiment establishes headroom empirically by preserving output and reducing complete-call time. No new Instruments capture was used for these final comparisons.

For a fixed-rest sensitivity at this optimized 24 MP/e7 point, eliminating the entire exposed AC phase would remove only about {100*target['joined1_ac_ms']/target['joined1_profiled_ms']:.1f}% of profiled latency. Eliminating all section writing would remove about {100*target['joined1_codestream_section_writing_ms']/target['joined1_profiled_ms']:.1f}%. These are optimistic phase-removal bounds, not expected GPU-rANS gains. Entropy writing is therefore a separate experiment with a limited single-image budget; the current result does not imply that moving every remaining step to the GPU is advantageous.

## Tuning and correctness evidence

The [work log](WORKLOG.md) records fresh allocation, cached arenas, two-submit exact compaction, guarded one-submit compaction, histogram sharding, threadgroup sizes, host reduction, scalar DCT8 emission, CPU cleanup, and overlap screens. Screens are distinct from the final repeated cohort. The first fresh-buffer version spent roughly 11 ms waiting for about 2.5 ms of GPU execution at 24 MP/e7; cached arenas reduced the exposed handoff cost. At 48 MP/e7, compact storage reduced the token working arenas from about 1 GB maximum capacity to about 148 MB in that screen. Scalar DCT8 helped pure-DCT8 low-effort frames; mixed-frame scalar emission did not justify selection.

- Independent CPU-tokenizer oracle: 2,560 submissions / 10,240 group comparisons per full run, all seven supported transforms plus mixed frames, natural/custom orders, context maps/thresholds, populations on/off, integer extremes, sparse/zero/last-position data, offsets and poisoned gaps. Dense, compact, guarded retry, scalar variants, and cache reuse were checked.
- Thirteen allocation/submission/completion failure and recovery scenarios cover retry paths, multiple live leases, repeated Finish rejection, truncated inputs, and cache trimming. Twenty DC cases cover exact predictors, worker counts, CPU caps, finite planned memory, partial launch failures, and recovery.
- The final build, including group fusion, passed **159/159 tests**, plus **9/9 explicit combined/fused-path tests**, including real workflow admission. Group fusion also passed six full oracle runs (64/128/256 threads, ordinary and forced-one-token-capacity retries) and the failure suite. The earlier V7 full run passed 158/159, with an intermittent AQ profiler-name assertion; the same assertion reproduced on the untouched baseline. The earlier AC-strategy profile failure passed 13 candidate and ten baseline isolated repetitions; both final isolated tests passed. Temporary diagnostics found registry-name lookup misses after eviction from the existing 512-slot pipeline registry. Instrumentation was reverted. The inherited intermittent profiling limitation remains recorded despite the passing final suite.
- Final captures contain {v['unnamed_dispatches']} unnamed dispatch records. [Attribution audit](unnamed-dispatches.json), full/isolated logs, stress results, and temporary diagnostic patch are under [verification](verification/).
- All **18 distinct final codestreams** decoded successfully with independent libjxl `djxl`, with expected dimensions and recorded decoded-PFM SHA-256. Every implementation/round/batch produced the identical corresponding encoded bytes. [Decoder validation](decode/validation.json) records the decoder version/hash and commands. This establishes unchanged rate and decoded output for the cohort; it is not new quality calibration.

## Scope and reproduction

The performance cohort uses repeated photographic inputs at one distance. Higher quality/token density, mixed-image batches, other devices, other memory limits, and exhaustive maximum-compression policies need separate qualification. Synthetic sparse/dense/extreme fixtures here establish correctness, not their complete-call performance. The GPU path adds kernels, scratch policy, and lifetime complexity. A default-on follow-up should retain an explicit CPU override and measure these unqualified workload dimensions while developing its selection policy. The inherited profiler registry issue should be fixed separately before relying on arbitrarily long-lived multi-backend attribution.

The qualified source is committed as `59d976152937a5a2e171c87be83254569273a3c9` on the experiment branch; it has not been merged. GPU and parallel-release controls require `GJXL_BUILD_TOKENIZATION_EXPERIMENT=ON`; GPU selection additionally requires the environment below. DC-group parallelism is part of the candidate implementation. Main remains untouched.

```sh
GJXL_EXPERIMENT_GPU_TOKENS=1 \\
GJXL_EXPERIMENT_DC_WORKERS=8 \\
GJXL_EXPERIMENT_TOKEN_COMPACT=2 \\
GJXL_EXPERIMENT_TOKEN_OVERLAP=1 \\
GJXL_EXPERIMENT_TOKEN_SHARDS=1 \\
GJXL_EXPERIMENT_TOKEN_SCALAR_DCT8=2 <encoder command>
```

For the optional group-fused specialization, additionally set `GJXL_EXPERIMENT_TOKEN_GROUP_DCT8=1` and `GJXL_EXPERIMENT_TOKEN_GROUP_THREADS=256`. It activates only for pure-DCT8 frames using the guarded compact layout. Keep it workload-selectable; the batch results do not justify enabling GPU tokenization for every workload.

[Qualification plan](qualification-plan.json), [frozen binary hashes](final-binaries.json), [hardware/compiler provenance](provenance.json), [pipeline commands](run-final.py), and the two run-config files bind the measurements. Each final run directory contains frozen executables, source diff/new files, input hashes, exact argv/environment, raw results, and per-case completion records. Use a fresh output directory for a new collection; existing manifests reject changed runner hashes, and original runner copies are retained with the frozen sources. [Analysis](analyze.py), [decoder validation script](decode.py), and [figure script](plot.py) consume retained artifacts. Measurements predate the checkpoint commits. No push, merge, or change to the original checkout was made.

The group-fusion follow-up has [its own pipeline](run-group-final.py), [binary hashes](group-final-binaries.json), [analysis](analyze_group.py), and frozen sources in `group-final-profile` / `group-final-batch`.
'''
    (ROOT / 'REPORT.md').write_text(text)


if __name__ == '__main__':
    main()
