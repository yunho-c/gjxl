# CUDA catch-up: reviewable checkpoint

This branch adds **production-default CUDA direct short filters**, an **opt-in
CUDA GPU AC tokenizer**, and a separately qualified backend-selection profiling
correction to upstream `715f033`. GPU tokenization remains opt-in. The short-filter
decision and its separate performance qualification are recorded in
[the production adoption notes](cuda-direct-short-production.md).

| Commit | Change |
| --- | --- |
| `5129f5a` | Independently owned resident coefficients, CUDA tokenization overlapping CPU DC work, bounded output/retry storage, profiling and finite concurrent-batch coverage. |
| `f698cdc` | Include admission-time backend selection in the existing workflow profile; preserve the accumulator on failure and read no extra clocks when it is null. |

The first commit's Git tree is `1102ee7387f6e2f263738d17705f8d46ff938043`.
The combined code checkpoint's tree is `a0f381a3b654d7c72d4c8e9bce09f54fb6dc7adc`.
Both unchanged upstream submodule references are retained. These tree identities
describe the tokenizer/profiling checkpoint before short-filter adoption.

## Selection and behavior

Normal CUDA Butteraugli preparation uses the fixed 32-row direct short filters
without an environment selector. CPU-order/exact-coefficient preparation retains
the legacy filters. Alternate 16/64-row schedules are test-only and disabled by
default. The following selectors apply only to the optional AC tokenizer.

Add `-DGJXL_BUILD_CUDA_RESIDENT_TOKEN_EXPERIMENT=ON` to an existing supported
CUDA configuration (`GJXL_ENABLE_CUDA=ON`). The experiment's CMake default is OFF.
For a process using the experimental build, select:

```text
GJXL_EXPERIMENT_CUDA_GPU_TOKENS=1
GJXL_GPU_TOKENIZATION=1
```

`GJXL_GPU_TOKENIZATION=0` takes precedence and retains CPU tokenization. Leaving
the experiment selector unset or setting it to zero also leaves the provider
unused. No global environment setting is required.

Eligible routes use fully resident or throughput CUDA AQ, an independently
completed frame and non-exhaustive entropy. Maximum-error, exact-coefficient,
maximum-throughput and maximum-compression paths retain CPU tokenization.
Eligible target-size attempts, host/GPU workflow profiles and finite concurrent
batches preserve the same selection rules.

The provider copies packed int32 coefficients into an independent device owner
before sparse/compact assembly can overwrite its source. Native host coefficients
remain available. GPU AC work begins before CPU DC serialization; finish waits,
validates counts and performs a bounded output-capacity retry when needed.
Values, contexts and integer populations are returned to the shared serializer.
There is no CPU token-count prepass. Device allocations, metadata, retained
owners and replacement overlap are included in resource admission. Failures
preserve public outputs, and outstanding work drains before its storage dies.

Workflow profiling retains the provider and accounts for the coefficient-copy
stage. It does not claim a complete trace of token-kernel execution. The second
commit corrects attribution of backend selection during admission; it is not
a speed optimization and does not explain occasional long first invocations.

## Ongoing qualification

The manually dispatched [CUDA qualification workflow](../.github/workflows/cuda.yml)
retains the default build and adds a separate experimental build. That build runs
all six tokenizer CTests: kernel token oracles, coefficient ownership, provider
retries, workflow selection, profiling and concurrent batches. Provider and batch
smoke cases also run under Compute Sanitizer memcheck and initcheck. Their logs
and the CTest results are retained with the workflow artifacts. These checks use
the workflow's Windows SM86/CUDA 11.8 runner; they are correctness checks, not a
performance gate or automatic PR coverage.

The default build also runs the direct-short and tall-shape CTests. Its guarded
short-filter fixtures run under all four Compute Sanitizer modes, with completion
markers and retained logs. Tokenizer tests exercise composition with the default
short filters; they do not establish a combined performance benefit.

## Qualification and measured benefit

Evidence uses the RTX 3060 Laptop (6 GiB, SM86), CUDA 11.8, MSVC 14.37 and Release
builds. Native correctness covers compact AC OFF and ON. Existing reviewed
fixtures cover independent token oracles, custom coefficient orders/contexts,
capacity retries, retained ownership, exact bytes and summaries, failure
injection, finite admission, profile selection and mixed-size concurrent batches.
The profiling and batch checkpoints have separate sanitizer and full concurrent
memory-check records. The admission-profile correction passed 16 tests and
216 diagnostic encodes. These are existing qualifications of these exact code
checkpoints, not newly rerun tests or combined latency measurements.

The tokenizer's ordinary natural-image confirmation contains 864 pairs and
1,728 processes, with main/main controls and retained first calls. Thirty new
natural sources span 2.36–4.19 MP. At distance 1 and eight CPU threads, their
geometric warm complete-encode ratios were:

| Effort | Provider/main ratio | Lower warm time | 95% source-bootstrap interval |
| --- | ---: | ---: | --- |
| 1 | 0.92362 | 7.64% | [0.90688, 0.94070] |
| 5 | 0.93249 | 6.75% | [0.92372, 0.94084] |
| 8 | 0.97860 | 2.14% | [0.97283, 0.98436] |

All successful outputs and summaries matched. Ninety new decoder cases were
independently regenerated; another 24 output cells reused exact prior decoder
evidence. Static extraction of the actual baseline/provider executables confirms
identical raw executable bytes in all 245 functions of their 13 common GPU
modules; the provider adds only its tokenizer module.

First-call, small-input, disabled-path and policy observations prevent a global
tokenizer default. Controls are not subtracted, long first calls are not trimmed,
and no image-size threshold has been fitted. The intervals describe source
variation in this study, not other machines or sessions. The numbers above
precede the separate admission-profile correction and short-filter adoption;
they do not establish the latency of the combined changes.

Qualified tokenizer archive SHA-256:
`199c692da5718fb9f7a4411f1b3dae75a1710858469c9f6bdb4befe8d61ef4a3`.
Combined profiling checkpoint archive SHA-256:
`dc7d34d0b1236e56d7efa0aa5ad53c8e4bbbf81cd9c968a2847453c0f3d5c056`.
The accompanying [evidence index](cuda-catchup-evidence.json) records review
identities and their scope. Full local artifacts remain in the adjacent
`gjxl-cuda-metal-catchup` evidence workspace and qualified source worktrees.

## Metal parity dispositions

Metal's production merge preserves ordered greedy selection: device candidate
costs feed per-tile selection and stable strategy/anchor metadata, which resident
AQ consumes without a host round trip. Its frontier/rectangle optimizers remain
experiments. The CUDA equivalent was implemented and qualified for exact output,
but whole-encode large-image/fallback/policy regressions exclude it from this
integration. Kernel-only savings are insufficient for adoption.

| Mechanism | Disposition |
| --- | --- |
| Fused AC candidate evaluation, reduced AC scratch, low-effort work elimination, DC/high-effort/CfL policy | Already in upstream CUDA/shared code. |
| Deferred host masks, final EPF conversion, coefficient-order populations, shared serializer | Already present. |
| Final-use release before serialization | Shared workflow already destroys AQ preparation and resident inputs after independent completion; target-size retries retain required state. |
| GPU AC tokens and CPU DC overlap | Added here, opt-in; broad warm gains with unresolved default boundaries. |
| Metal token execution policy | The CUDA checkpoint follows one histogram shard, 128-thread emission, scalar pure-DCT8 and warp mixed transforms. Extra shards/group fusion are Metal experiments. |
| Small-field quantizer and resident AC/AQ initialization/handoff | Tested; native/synchronization gains did not meet complete-encode acceptance. Excluded. |
| Butteraugli direct short filters | Adopted as the fixed 32-row production default after further startup, warm-encode and batch qualification; legacy CPU-order fallback retained. |
| Butteraugli scratch borrowing and traffic fusion | Tested with qualified lifetimes/numerics; mixed or insufficient whole-encode results. Excluded. |
| Device-only frames and direct final coefficient output | Qualified prototypes, kept separate. Startup/policy concerns and interrupted memory-pressure screens prevent performance acceptance. |
| Private-cache OOM reclamation | Correctness and changing-size tests pass; controlled ordinary performance remains mixed. Kept separate. |
| Metal shared mapped output and Apple SIMD/threadgroup choices | Hardware-specific; CUDA retains explicit discrete-GPU transport and measured NVIDIA layouts. |

The direct-output screen most recently stopped after 22 complete pairs when the
unchanged parent exhausted GPU memory at 24 MP effort 8. All 45 successful
processes / 269 calls matched prior outputs, but incomplete rounds establish
neither a speed nor a memory advantage. Its failed study remains intact and is
not represented by this branch's performance results.

This checkpoint supplies a new CUDA capability and adopts the qualified short
filters. It does **not** establish complete production optimization parity or a
platform-wide tokenizer default. Other GPU architectures/native Linux performance
are unmeasured by this laptop evidence. The older development-plan documents are
historical protocols; this document describes the current integrated behavior.
