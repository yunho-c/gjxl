# Native compact coefficient ownership (S107)

Starting revision: `71a130a`. This follows the consumer, composition and tile
prototypes in [S106](cuda-representation-fusion-scheduling-s106.md). Work ran
2026-09-07 evening local time / 2026-09-08 UTC with CUDA 11.8 and the RTX 3060
Laptop. Unlike S106's isolated consumer, this candidate carries narrow CUDA
readback through a complete encode. It remains opt-in:

```text
-DGJXL_ENABLE_CUDA=ON -DGJXL_CUDA_COMPACT_AC=ON
```

The default is OFF. The forty retained production runtime files are unchanged.
No composition or tile-scheduling change is combined with this experiment.

## Representation and consumers

The frame owns exactly one signed int8, int16 or int32 coefficient array, with
the existing fixed-capacity group/channel rows and zero edge tails. Typed
assembly validates layout, metadata, populations and tails before transferring
the caller's owner. Borrowed inputs are copied at their native width. Frame
copies deep-copy the coefficient owner; moves transfer it. Const reads require
no allocation, synchronization or mutable cache.

Following the request to skip compatibility layers, there is no lazy dense
cache or automatic expansion. `GetNativeAcGroup` exposes a variant of typed
spans. The int32-specific `GetAcGroup` query explicitly rejects narrow groups.
General consumers must use the native query. Token templates, direct tokens,
coefficient-order recounting and dequantization dispatch outside coefficient
loops. CPU reconstruction reads the native storage directly. CUDA exact-AQ
staging is updated likewise; the corresponding Metal consumer is updated in
source but cannot be compiled or executed on this Windows host.

The optional resident-policy producer first retains the ordinary packed int32
GPU result. A packing kernel writes byte and word representations into the
now-dead quantized-batch scratch allocation and sets exact signed-overflow
flags. Host code selects 1, 2 or 4 bytes without clamping; the untouched packed
int32 result is the fallback. Alignment, capacity and launch limits gate narrow
packing. Non-policy evaluation retains its dense path.

Only the selected host owner is allocated for normal compact policy encodes.
Pitched readback writes active values directly into its final rows; only tails
are cleared. Assembly moves that owner into the frame. There is no S84-style
second host payload and no post-readback expansion. GPU scratch capacity is
unchanged; the additional packing work is included in complete-call timing.

The internal profile reports the largest authoritative AC owner at a serializer
handoff across attempts, plus its element width. This excludes frame metadata,
device scratch and allocator overhead; it is not a total process-memory counter.

## Qualification

The compact-frame test covers 96 cases: seven transform shapes plus mixed
strategies, small and four-group odd-edged images, zero/sparse/boundary inputs,
and signed 8/16-bit limits. It verifies exact owner transfer, copied/moved frames,
borrowed-input isolation, population isolation, concurrent readers and rejected
typed queries. Five rejected assembly inputs per case leave owner/output intact.
Across each run there are 384 exact codestream comparisons, 96 bitwise CPU
reconstruction comparisons and 480 atomic-failure checks. Cached and recounted
orders cover full and effort-7 sampled policies.

The default CUDA build passes all 74 tests. The compact build initially passes
73/74: the population-test helper still requests int32 spans. Updating that
independent scalar recount and its dense test oracle to read native spans fixes
`cuda_aq`; all four affected tests pass afterward. No result comparison is
removed or relaxed. The default build's four affected tests also pass. The
fully instrumented host build passes 50 selected tests, and its two tests
affected by the fixture update pass again. The install-consumer test is covered
by the release suites, not the ASAN selection.

The CUDA packing test covers 102 cases, including zero, tiny, odd and CTA-edge
lengths, signed limits, isolated overflow, output guards/padding and preservation
of the original dense source. Compute Sanitizer memcheck, racecheck, initcheck
and synccheck each confirm all 102 cases with zero reported errors/hazards;
memcheck reports zero leaked bytes. An intentional uninitialized-source canary
in the same compiled packing kernel is detected as four errors with exit 71.
A complete compact sample encode also passes memcheck and three exact checks.

Two toolchain issues are preserved in the evidence. The host CLI's ASAN report
depends on linker identical-code folding: relinking the same objects with ICF
reproduces it, while NOICF passes. NOICF is used only in the diagnostic ASAN
build. Separately, Compute Sanitizer loses buffered final success lines on these
test exits. Explicitly flushing completion markers resolves the missing-output
gate. Runtime-link and initialization probes did not resolve it. The deprecated
CUDA-MEMCHECK launcher misses the deliberate canary, so its apparent clean runs
are discarded. Zero-error summaries alone are never accepted as test completion.
Build errors, the initial fixture failure and rejected/partial reporting runs
remain recorded, rather than being overwritten or counted as qualification.

## Complete encodes and memory

Twelve preflight jobs use sample, Flower 500, Keong 500, a Keong 2000 derivative,
1919x1079 and 3839x2159 inputs, with both dense and compact builds. They pass
36 frozen-byte checks. The main campaign runs five non-tiny inputs through six
adjacent process pairs, alternating dense/compact order and reversing input
order each repetition. Each process has one checked reference, two warm calls
and eight measured calls. All 660 main encodes match frozen bytes and fresh
summaries; summary fields are also compared across processes and repetitions.
Thus preflight plus main contain 696 exact encode checks and 480 measured rows.

Main timing is 03:25:51.143182–03:27:33.019173 UTC (101.88 seconds). Distance 1.2,
effort 7, fully resident, automatic CPU policy and no final score request remain
fixed. No compilation, sanitizer or telemetry sampler runs during timing.
Light host editing and natural machine-state drift remain limitations.
Outer time includes Encode and internal owner cleanup, but excludes backend
lifetime, file I/O, returned-codestream destruction and comparisons.

Each process contributes its median of eight measured calls. Below, dense and
compact columns are medians across six processes; paired percentages compare
adjacent process medians and need not equal ratios of the first two columns.

| Input | Dense ms | Compact ms | Median paired change | Six-pair range |
| --- | ---: | ---: | ---: | ---: |
| Flower 500 | 20.27 | 18.43 | -9.30% | -34.22% to -1.60% |
| Keong 500 | 19.44 | 18.63 | -5.10% | -12.31% to +10.28% |
| Keong 2000 derivative | 141.41 | 133.65 | -5.93% | -14.92% to -2.31% |
| 1919x1079 | 72.87 | 68.50 | -6.87% | -8.66% to -2.90% |
| 3839x2159 | 297.05 | 282.68 | -5.70% | -13.16% to -1.62% |

The three larger inputs improve in every pair, with quantization-phase paired
median reductions of 6.79, 3.85 and 13.82 ms respectively. Serializer-only
comparisons remain mixed in each cohort; do not attribute the whole-call gain
solely to typed tokenization. Flower's unusually broad range and Keong 500's
mixed results warn against treating these descriptive medians as universal or
statistically established gains. No size cutoff is promoted from this cohort.

Flower selects int16 and halves its 3 MiB AC owner. Other measured inputs select
int8 and quarter their owners. At 4K this is 106,168,320 to 26,542,080 bytes,
saving 79,626,240 bytes (75.94 MiB). The paired median process peak-working-set
reduction is 75.37 MiB. Working-set measurements include process initialization
and prior warm/reference calls; they are not isolated frame peaks. Process
private commit does not fall in this campaign, despite the smaller touched
owner, so no private-commit saving is claimed.

## Disposition and remaining work

Keep the implementation available for opt-in qualification. The compact owner
now produces exact complete encodes and a measured memory reduction, with
promising large-image whole-call gains. This is not yet a default rollout or
proof that optimization is exhausted. Independent concurrent batches, wider
image/quality coverage, allocation-failure injection and complete-encode
int32-overflow fallback still need focused qualification. Natural int16 and
int8 complete encodes are covered here; the kernel's int32 overflow behavior is
covered by unit tests, not a naturally overflowing full-image campaign.

The composition/reduction and scheduling dispositions from S106 are unchanged.
Do not add their isolated gains to these complete-call results.

Ignored scripts are `build-cuda-ninja/profiles/s107_*`; binaries, logs, inputs'
hash references and measurements are in `U:/gjxl-cuda-diagnostics/s107`.
A recomputing validator and manifest anchor evidence and unchanged retained
runtime files. No power, clock, priority, security, firewall or driver setting
is changed, and no permission prompt is observed. The user's three untracked
Markdown files remain untouched.

```powershell
python build-cuda-ninja/profiles/s107_validate.py --frozen
```
