# Direct narrow group packing (S109)

Starting revision: `2dc1191`, following [S108](cuda-compact-qualification-s108.md).
Work ran 2026-09-08 UTC/local time on Windows, CUDA 11.8, sm86, RTX 3060 Laptop.

The direct-packing kernel is retained as a tested internal primitive. **Encoder
routing is unchanged**: the integrated candidate was qualified and measured,
then its routing was restored to S107's two-pass compact implementation because
a general incremental complete-encode gain was not established. Compact CUDA
storage remains opt-in. No compatibility layer or new runtime option is added.

## Dataflow and implementation

S107 first scatters transform batches into packed int32 group/channel rows,
then rereads those rows to produce byte and word payloads plus overflow flags.
The new kernel writes both narrow payloads directly from transform batches.
One CTA handles an anchor; each thread handles four coefficients per channel.
Group-edge channel strides and precomputed anchor offsets retain the existing
layout. CTA overflow votes accumulate into the two signed-width flags. Neither
source coefficients nor metadata are modified. Output low bits are defined on
overflow, but only a fitting signed width may be consumed.

Ignoring metadata/flags, the normal packing sequence's logical traffic drops
from 15N to 7N bytes for N active coefficients: 4N source reads plus 3N narrow
writes, instead of an additional 4N write/read of packed int32. This is a
dataflow count, not a measured DRAM transaction counter. The standalone sm86
compile uses 40 registers and reports no stack or register spills.

The integrated candidate reused dead inverse-transform scratch for byte[N],
word[2N], flags[4]. It kept the original quantized source intact. After the width
flags were read, int16 overflow triggered a second, ordinary dense packing
submission into that scratch. No extra GPU allocation capacity or host owner
was introduced. Typed readback and native frame consumers were unchanged.
The source and linked CUDA library for this candidate are preserved as
`cuda_aq_resident_direct.cpp` and `gjxl_cuda_direct.lib` in the evidence folder;
they are not the final encoder routing.

## Correctness and safety

The new `cuda_direct_ac_pack` test passes 384 cases: eight block-grid geometries,
seven transform families plus mixed layouts, and six value patterns. Geometries
include 1x1, odd edges, exact/crossed group boundaries, 1080p and 4K block grids.
Patterns cover zeros, signed 8/16-bit boundaries, int32 extremes and isolated
overflow. Tests compare both narrow payloads against an independent scalar
layout, guard all outputs, check source/metadata preservation, reuse buffers,
and run the real dense packer afterward to validate fallback data.

Compute Sanitizer memcheck, racecheck, initcheck and synccheck each pass a
240-case subset through 33x33 blocks, with zero errors/hazards and an explicitly
flushed completion marker. The first test compilation lacked `<span>`; the
header was added and the corrected build passed. The failed build log remains.

Integrated qualification replays S108's quality, rate-control, synthetic
overflow and independent/public-driver batch preflights. All 480 successful
top-level encode checks match frozen dense bytes and every deterministic
summary field. Seven previously rejected extreme inputs reject with the same
errors; they are not counted as successful checks. Three synthetic inputs
exercise actual int32 selection and the additional dense fallback submission.

Six integrated memchecks cover those three fallback inputs, four independent
mixed-quality contexts, the four-worker public batch driver, and sample rate
control. All 45 additional exact checks complete with zero errors and zero
leaked bytes. This is not a qualification of other operating systems, GPU
architectures, broader concurrency, or allocation failures in the new GPU
submission path.

After restoring encoder routing, the rebuilt compact CUDA configuration passes
all 76 CTest tests, including the new packing test and install-consumer test.

## Isolated packing timing

The event-timed probe compares two-pass and direct packing in one executable,
with duplicate labels for each implementation. It tests 33x33, 240x135 and
480x270 block grids, DCT8/DCT32/mixed layout preferences, and signed 8/16-bit
inputs. Each process has four warm rounds and 24 measured rounds, four shuffled
labels per round, and eight packing sequences per event interval. Two process
repetitions reverse case order. There are 18 processes, 3,456 measured rows,
and 72 pre-timing scalar-output comparisons.

Every one of the 144 descriptive candidate/control paired medians is negative
(-10.28% to -57.00%). These correlated comparisons are not 144 independent
experiments or a significance test. At 4K, ranges across both coefficient
widths, repetitions and duplicate-label medians are:

| Layout preference | Two-pass ms | Direct ms |
| --- | ---: | ---: |
| DCT8 | 1.686–1.717 | 1.106–1.236 |
| DCT32 | 1.558–1.753 | 0.730–0.742 |
| Mixed | 1.658–1.673 | 0.890–0.917 |

This establishes a scoped packing improvement: the 4K paired median savings
span 0.484–1.003 ms. Event intervals include submission gaps. The probe keeps three
device arrays so both implementations can reuse immutable inputs; it is not
the production two-array scratch layout or a complete encode. No packing gain
is added arithmetically to whole-call timing.

## Complete encodes and the tighter control

An initial campaign compares dense, S107 two-pass compact and direct compact
executables. Five inputs have three-mode preflights, then six process
repetitions using all six mode orders, with reversed input order on alternating
repetitions. Every process has a fresh checked reference, two warm encodes and
eight measured encodes. The campaign adds 1,035 exact checks (45 preflight and
990 main), with 720 measured calls. Distance 1.2, effort 7, automatic CPU policy,
fully resident and no final score remain fixed.

Below, columns are medians of six process medians. The percentage is the median
of six within-repetition direct/two-pass ratios, not their column ratio.

| Input | Dense ms | Two-pass compact ms | Direct compact ms | Direct vs two-pass paired median |
| --- | ---: | ---: | ---: | ---: |
| Flower 500 | 19.58 | 19.74 | 19.08 | -2.76% |
| Keong 500 | 20.04 | 19.32 | 19.84 | +0.78% |
| Keong 2000 derivative | 144.92 | 137.30 | 137.80 | +0.42% |
| 1919x1079 | 77.66 | 70.77 | 71.08 | +1.59% |
| 3839x2159 | 319.16 | 300.98 | 296.96 | -0.53% |

Every direct/two-pass cohort contains positive and negative comparisons.
Keong 500 includes a +38.69% pair; it is retained, not silently filtered.
The larger direct-versus-dense medians are negative, but most of that difference
already exists in S107. These data do not isolate an incremental direct-packing
benefit.

To reduce process/code-layout variation, a separate diagnostic resident object
captures an atomic mode at each policy call. Both routes coexist in one binary;
labels 0/2 select two-pass and 1/3 direct. Host counters assert that exactly the
intended branch runs once per encode. Mode changes occur only between completed
encodes. This selector and its counters are diagnostic-only, not compatibility
or production machinery.

Ten processes run the five inputs twice in reverse order with different seeds.
Each has a checked reference, four warm Williams rounds and 24 measured rounds
of four labels: 1,130 further exact checks and 960 measured calls. Complete
result summaries and native owner width/size match the process reference.
The following ranges cover all four candidate/control paired medians in both
repetitions. Duplicate ranges compare labels of the same implementation.

| Input | Direct vs two-pass | Same-implementation duplicate comparisons |
| --- | ---: | ---: |
| Flower 500 | -3.32% to -0.43% | -2.79% to +0.96% |
| Keong 500 | -3.12% to +0.27% | -0.35% to +1.46% |
| Keong 2000 derivative | -1.59% to +3.56% | -3.04% to +2.12% |
| 1919x1079 | -1.45% to +0.81% | -1.91% to +2.76% |
| 3839x2159 | -2.25% to +1.07% | -0.27% to +1.02% |

Only Flower has all-negative candidate medians, and its duplicate variation
is still substantial. The larger complete-call effects are mixed and comparable
to controls. Quantization-phase comparisons are also mixed there. This is not
evidence that the isolated kernel saving is false; it is insufficient evidence
of a general incremental encode-time gain.

Across integrated qualification, initial timing and within-process control,
there are 2,645 successful exact top-level checks, plus the 45 memcheck checks.
Packing timing ran 04:13:13–04:13:43 UTC; main three-mode timing 04:15:02–04:17:39;
within-process timing 04:22:24–04:24:41. GPU jobs are serial. No compilation,
sanitizers, regression tests or telemetry sampling overlap timed jobs. Light
editing and natural machine-state drift remain limitations. Outer timings
include internal frame cleanup, but exclude backend lifetime, file I/O,
returned-result destruction and comparisons.

## Disposition and next work

Retain the kernel and focused test for further optimization; do not route
encodes through it yet. The final resident source is restored exactly to the
starting revision, and retained production runtime files are unchanged. No
power, clock, priority, driver, security or firewall setting changes; no admin
or firewall prompt is observed. The user's three untracked Markdown files
remain untouched.

Possible next packing work includes vectorized source loads and conditional
single-width emission, but neither has a measured benefit here. More importantly,
the packing opportunity is sub-millisecond at 4K, so the larger resident
perceptual kernels and the separately investigated composition/reduction and
tile-scheduling paths remain in scope. This is not optimization exhaustion.

Ignored scripts and diagnostic sources are `build-cuda-ninja/profiles/s109_*`;
frozen evidence is under `U:/gjxl-cuda-diagnostics/s109`. The validator recomputes
all comparisons, checks exact outputs and rejected runs, audits job non-overlap,
and verifies the forty retained runtime hashes.

```powershell
python build-cuda-ninja/profiles/s109_validate.py --frozen
```
