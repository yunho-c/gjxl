# GPU tokenization default

Resident Metal encoding uses GPU AC tokenization by default. No experimental CMake option or environment variable is needed. The default includes guarded compact token allocation, one histogram shard, a 128-thread emitter, scalar emission for pure-DCT8 frames, and overlap with parallel CPU DC groups. This is the V7 configuration measured in the [qualification report](../reports/tokenization-20260922/REPORT.md). Optional whole-group DCT8 fusion remains experimental and disabled by default.

To use CPU AC tokenization while keeping the rest of the Metal encoder:

```sh
GJXL_GPU_TOKENIZATION=0 <encoder command>
```

`GJXL_GPU_TOKENIZATION=1` explicitly enables the default GPU policy. An unset or unrecognized value uses the default policy. The setting is process-wide: configure it before starting encodes and keep it fixed while jobs, batches, or their admission plans are active. It takes precedence over the legacy experimental GPU selector.

GPU tokenization applies to resident Metal fully-resident and throughput workflows with a completed resident coefficient buffer, including eligible target-size searches. CPU backends, compatibility routes, maximum-error workflows, and exhaustive maximum-compression serialization keep CPU tokenization. CUDA has the separate single-image policy below. The CPU still chooses coefficient orders, contexts and entropy models and writes the bitstream. The switch preserves encoded bytes; it changes execution and resource use.

The resident Metal admission plan includes token metadata/output arenas and their idle pools. CPU-only, compatibility, and exhaustive serializer plans do not reserve these arenas. Selection does not currently adapt to image size, effort, batch concurrency, or memory budget. Resource errors retain the existing error/publication behavior; they do not silently trigger an unplanned CPU retry.

Eligible single-image calls also use [earlier entropy readiness](entropy-readiness-capacity-policy.md)
with a complete DC-plus-AC CPU reservation. Batch calls retain the original
readiness schedule. `GJXL_EARLY_ENTROPY=0` disables only this scheduling policy;
it leaves GPU tokenization enabled and is independent of the tokenizer switch.

## Performance policy

### CUDA single-image policy

Normal CUDA builds enable GPU AC tokenization for eligible single-image calls.
`VarDctBatchEncoder` defaults to CPU AC tokenization, including a one-worker or
one-image batch. Fully-resident and throughput modes, including target-size
attempts and profiled calls, are eligible. Maximum error, exact coefficients,
maximum throughput, and maximum compression keep CPU tokenization.

`GJXL_GPU_TOKENIZATION=0` disables GPU tokenization. Setting it to `1` explicitly
enables eligible CUDA batch calls too. The legacy
`GJXL_EXPERIMENT_CUDA_GPU_TOKENS=0/1` remains supported: `0` disables CUDA tokens
even with the stable enable switch, and `1` enables eligible calls including
batches unless the stable CPU override is set. Unrecognized values use the
ordinary default. Configure selectors before calls and keep them fixed during
planning and encoding. The legacy build option
`GJXL_BUILD_CUDA_RESIDENT_TOKEN_EXPERIMENT` now defaults to ON; explicitly setting
it to OFF builds the CPU-token path only.

CUDA keeps an independent device copy of final coefficients while retaining its
native host representation for CPU order/context selection. GPU AC work overlaps
CPU DC tokenization; entropy models and bitstream writing remain on CPU. The newer
Metal early-entropy/earlier-DC schedule is not enabled for the CUDA provider by
this change. The separate device-only frame/ownership prototype is not included.

Batch admission resolves the same policy as worker execution, so default batches
do not reserve the optional GPU-token buffers. Tokenized calls account for the
retained coefficient copy, provider storage and bounded capacity retries.
Physical allocation OOM reclaims completed private-cache buffers and retries once;
live owners and resource-domain limits remain intact.

This is an API policy, not automatic detection of concurrent single-image callers.
Concurrent ordinary calls remain supported and share execution-domain limits; use
the batch API or the CPU override when conservative concurrent behavior is wanted.
No image-size threshold is inferred. Historical qualification found warm serial
gains and mixed batch behavior, including high-effort large concurrent regressions.
That historical study predates the current E1–E5 policies.

The 2026-10-08 validation used an RTX 3060 Laptop GPU (6 GiB), CUDA 11.8,
MSVC 14.37 and driver 577.00. Initial validation on `96cce2d` plus this change
passed all 168 native CUDA-build tests across the full run and targeted
rechecks, 114 CPU-only tests, and ten CUDA memcheck/initcheck checks. After
rebasing onto the accepted CUDA catch-up merge `657b9e7`, all 18 targeted
integration tests, 45 direct-short fixtures, 24 prepared legacy/fused pairs
and ten fresh sanitizer checks passed. Ninety fresh sequential large-image
encodes, including three distinct 12 MP photos and 12-to-24 MP transitions at
E1/E5/E8, passed without manual cache trims. Each route matched canonical CPU
bytes; repeated complete summaries remained stable. Batch tests cover default
and explicit selection, one-image batches, finite admission, failure isolation
and recovery.

A small warm screen on the final `657b9e7`-based branch compared the same binary's
CPU override with unset defaults at distance 1.9 and eight CPU participants.
Four balanced AB/BA rounds used two
warmups and five timed complete public calls after an initial validation encode:
144 processes and 720 timed encodes. All corresponding output hashes matched;
18 screen outputs and 12 large-image control outputs decoded independently.
The table reports elapsed-time change relative to CPU tokens, using the median
of four per-process sample medians; negative values are faster.

| Image | E1 | E2 | E3 | E4 | E5 | E8 |
|---|---:|---:|---:|---:|---:|---:|
| Kodak 01, 0.39 MP | -1.9% | -3.5% | -0.3% | -5.5% | -3.7% | -3.1% |
| CLIC test 097cb426, 3.09 MP | +4.0% | +0.5% | -3.6% | +2.6% | -2.9% | +1.8% |
| Alpine lake, 12 MP | -5.9% | -11.3% | -9.3% | -7.4% | -8.5% | -3.2% |

All four paired rounds improved for the 12 MP image at every measured effort;
all four regressed for the 3.09 MP/E1 case. Smaller-image results also varied
between the initial and final screens, so individual percentages should not
be treated as stable predictions. This supports the large-image latency
benefit while preserving the CPU override for content/effort regressions. It
does not establish a size threshold, universal speedup, or new paper-table
result. The final local reproducibility bundle is
`build/token-policy-validation/r2/` (reports, commands, raw samples, hashes,
decoder checks and GPU telemetry); the parent directory preserves the initial
screen and `r1-binaries/` preserves its executables and libraries.

### Metal performance policy

Default-on is an explicit rollout choice based on the measured benefits, with the CPU override available for regressions. On the qualification M4 Pro, large-image single-call latency improved at efforts 1, 7, and 8. The 24 MP/e1 four-image batch had a repeatable throughput regression against the improved CPU tokenizer. Other measured e1 batch cases improved, so effort and requested batch size alone do not identify the regression.

Broader measurement of image content/size, quality, actual concurrency, cold/warm state, memory budgets, and devices will refine automatic selection. The historical report describes the opt-in checkpoint at `59d9761`; this default change does not turn that limited cohort into a universal performance guarantee or provide new timing results.

## Development controls and validation

`GJXL_BUILD_TOKENIZATION_EXPERIMENT=OFF` is the normal build configuration. The option now gates diagnostic ablations, including alternate layouts, overlap, shards, emitters, forced capacity retries, whole-group DCT8 fusion, and parallel CPU token cleanup. It is not required for production GPU tokenization. Experimental builds accept `GJXL_EXPERIMENT_GPU_TOKENS=0` as a legacy CPU selector unless the stable override is set.

The ordinary build includes the independent GPU-token oracle and `gpu_tokenization_default` integration test. The latter compares default GPU, explicit GPU, and CPU-override bytes/summaries, verifies actual GPU submissions across efforts 1/4/7/8/9/10, and checks CPU/compatibility/exhaustive admission isolation. The existing resident and public-batch admission tests cover bounded resources and repeated calls. Failure-injection and tuning-specific tests remain available in experimental builds.

The screen runners explicitly select CPU or GPU tokenization for each named mode, record the stable override, and retain the source commit beside their diff. Use a fresh directory when collecting with the updated runners; historical runs remain bound to their frozen runner hashes. Rebuild standalone capture/batch binaries before collecting from changed source.
