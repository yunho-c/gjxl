# CUDA optimization study S1

- Status: S1.1-S1.5, packed/register-tiled DCT, tiled Malta,
  specialized/tiled blurs, packed AC-search residuals, tiled EPF, and fused
  AC gather/DCT plus residual/inverse/loss fusion, compact AC-search scratch,
  factorized resident DCT, cooperative quantization adjustment, fused Malta,
  bounded-retention stream-ordered allocation, on-demand reconstruction
  host staging, overwrite-only coefficient staging, and direct resident
  transform image I/O, reused resident coefficient bases, direct local
  value access, fused resident coefficient passes, and encoding-only
  coefficient materialization, shape-specialized coefficient blocks, and
  fused blur/frequency splitting, and branch-free host coefficient-order
  counting, lightweight ANS token emission, direct AC token accumulation,
  contiguous AC nonzero reduction, fused L2/final masking, and fused
  vertical blur/low-medium construction, compact prepared Butteraugli scratch,
  fused mirrored RGB blur/Opsin conversion, and packed active-coefficient
  readback into ownership-backed final frame storage, in-place ANS clustering,
  hoisted histogram log-table access, borrowed prepared ANS populations,
  bounded narrow coefficient-order counters with portable SIMD counting,
  multi-row Malta halo reuse, joint-channel horizontal33 convolution,
  direct DC context lookup, success-path DC residual emission,
  inline validated ANS token scanning, fused erosion/L2/final masking,
  cooperative final color correlation, adjacent-row low-medium reuse,
  paired horizontal-convolution input reuse, and prefetched initial
  color-correlation samples, and exact zero-tile Malta responses
  implemented;
  optimization ongoing
- Profile revision: `a474937`
- Profile date: 2026-09-04
- Build: Release, CUDA 11.8, `CMAKE_CUDA_ARCHITECTURES=86`
- Device: NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6,
  6 GiB device memory
- Driver: 577.00
- Related analysis: [CUDA backend support analysis](cuda-support.md)

## Executive finding

The opening measurements describe revision `a474937`; the completion snapshots
below supersede them. The latest implemented checkpoint, S65 against S64
(`73179fd`), skips Malta response arithmetic only when every scaled value
in the tile and halo is exactly zero. It preserves scaling, the output
write/add, exceptional values, and the original tile policy. All twelve new
GPU bodies match the guarded, replay, sanitizer, and full-workflow controls;
all 176 previous bodies remain unchanged. Five of six selected-mode trace
pairs improve targeted Malta GPU duration by 8-21%; one 4K pair regresses
0.30%. All 73 CUDA / 50 CPU tests, five host ASan targets, seven release GPU
sanitizer checks, and 58 freshly decoded byte-identical image pairs pass.

Public whole-encode changes are warm -1.49% / +0.40% / -1.36% and cold
-1.09% / +0.14% / -0.57% at 4K / 1080p / Flower, with broad paired ranges.
The unchanged host codestream phase also moves, and earlier Flower wall
controls regress. Both final performance endpoints report thermal/power
limiting; 4K batch two loses to serial in all three pairs. This is an exact,
content-dependent GPU improvement, not a universal whole-encoder speedup.
See [S65](#exact-zero-tile-malta-responses-s65) for competing layouts,
25,800 timing windows, causal controls, diagnostic failures, and slower
observations. Optimization remains ongoing, not maxed out.

The subsequent [S66 division-sharing investigation](#shared-malta-reciprocal-investigation-s66-not-retained)
does not change production. Both guarded recipes pass arithmetic and captured
stage comparisons, but the correctly rounded reciprocal variant is neutral
at 4K and the other variant has only a small local gain without a general
rounding proof. A native-identical duplicate control exposes timing scatter.
The next candidate is immutable reference-mask work still repeated during
comparison, with preparation cost and additional storage treated explicitly.

The preceding implemented checkpoint, S60 against S58
production (`bf4968b`; S59 is investigation-only), fuses reference-mask erosion
with L2/final masking. It removes four launches per profiled encode and one
intermediate plane write/read per difference evaluation. Isolated combined
work improves 14.9-19.7%, winning all 432 pairs, and the release kernel matches
the measured body exactly. Warm public whole-encode changes are -2.9% / +1.5% /
+5.5% at 4K / 1080p / Flower; true cold changes are +2.7% / +1.6% / -1.9%.
Slower cohorts remain recorded. All 71 CUDA / 50 CPU tests, five host ASan
targets, seven scoped GPU sanitizer checks, and 58 byte-identical decoded
image pairs pass. The 27-plane allocation and public/private frame ABI remain
unchanged. Two initially cold-labeled cohorts actually used warmups; they are
preserved as warm repeats, and corrected cold runs are separate. This is a
targeted GPU improvement, not a universal wall-time speedup or a maxed-out
implementation. See [S60](#fused-reference-erosion-and-l2final-masking-s60).

S58 isolates the validated ANS token scan so conversion actually inlines;
warm 4K histogram work improves 14.0%, but whole-workflow and cold results
remain mixed. S59's per-channel Malta fusion is not retained because its
4K local gain is small and workflow evidence is inconsistent. See
[S58](#small-token-scanned-ans-histogram-routine-s58) and
[S59](#per-channel-malta-fusion-investigation-s59-not-retained).

The preceding implemented checkpoint, S56 against S55
(`81266d7`), replaces a 34-run DC context search with an exact compile-time
lookup and removes per-value success-status construction. Final warm release
DC tokenization improves 34.4% / 31.5% / 42.1% at 4K / 1080p / Flower,
winning all 21 pairs; whole encode improves 4.0% / 0.4% / 4.0%.
Cold Flower regresses in both release and control cohorts, and controls
do not give uniform whole-path gains. All 71 CUDA / 50 CPU tests, five host
ASan targets, 6,148 guarded public context cases, and 58 byte-identical
decoded-image pairs pass. The CUDA library remains byte-identical.
Diagnostic host bodies differ in native layout from release, so their
timings are separate evidence, not an exact-native match. See
[S56](#direct-dc-context-lookup-and-success-path-residual-emission-s56).

The preceding implemented checkpoint, S55 against S54
(`058b2df`), computes horizontal33 for all three channels in one kernel,
removing 12 launches per profiled encode without changing image-plane
traffic or arithmetic. Controlled horizontal-stage paired gains are
8.8% / 15.2% / 24.3% at 4K / 1080p / Flower, although one 4K target
observation and two 4K total-GPU observations regress. Warm release whole
encode improves only 0.45% / 0.93% / 0.37%; cold 4K regresses in both
release and control cohorts. All 71 CUDA / 50 CPU tests, four host ASan
targets, seven scoped CUDA sanitizer runs, and 58 byte-identical decoded
image pairs pass. All 170 existing GPU bodies remain unchanged; the new
production body matches the measured control. See
[S55](#joint-channel-horizontal33-convolution-s55) for evidence and limits.

The preceding implemented checkpoint, S54 against S53
(`7e5be16`), retains 256-thread Malta blocks while reusing each loaded halo
across multiple output rows per lane. Controlled GPU traces show median
paired Malta improvements of 18.7% / 7.6% / 7.9% at 4K / 1080p / Flower.
Final warm release whole-encode changes are -3.1% / -2.0% / -0.2%, but cold
4K and 1080p regress; unchanged host work and all slower observations remain
documented. All 71 CUDA / 50 CPU tests, four host ASan targets, seven scoped
CUDA sanitizer runs, and 58 byte-identical decoded-image pairs pass. The
original math and tiny-image instructions are retained. See
[S54](#multi-row-malta-halo-reuse-s54) for the policy, controls, and limits.
This is a targeted GPU improvement, not a universal speedup or a maxed-out
implementation.

The preceding checkpoint, S53 against S52
(`5ea11e6`), uses proven-safe 32-bit coefficient zero populations on ordinary
frames and retains the 64-bit fallback. A small portable helper enables
packed counting in the production MSVC object, without intrinsics, aliasing
promises, or changed compiler flags. Warm release coefficient-order work
improves by median paired 54.3% / 48.1% / 39.9% at 4K / 1080p / Flower;
whole encode improves 3.5% / 3.9% / 5.1%. Same-executable controls confirm
the targeted reduction but give smaller overall gains; cold Flower still
regresses in the control. All 71 CUDA / 50 CPU tests, four host ASan targets,
58 decoded-image pairs, and independent/forced-width differentials pass.
GPU code and ABI are unchanged. See
[S53](#bounded-narrow-coefficient-order-counting-s53) for controls and limits.
This is not a universal speedup or a maxed-out implementation.

The preceding checkpoint, S52 against S51
(`355180e`), borrows validated prepared ANS count arrays instead of initializing
and copying a full private histogram array. At 6,930 contexts it eliminates
14.14 MB of source storage. Complete large-partition replay improves 24-32%,
with identical output and unchanged GPU code. Whole-encode timing is mixed:
warm release cohorts regress slightly, and cold 1080p regressions remain in
the record alongside a favorable counterbalanced follow-up. All 71 CUDA /
50 CPU tests, three host ASan targets, 58 decoded-image pairs, and broad
partition differentials pass. See
[S52](#borrowed-prepared-ans-populations-s52) for the controls and limitations.
This is a targeted work/storage reduction, not a stable universal speedup.

The preceding checkpoint, S51 against S50
(`ad18132`), removes a full private histogram copy and per-symbol log-table
initialization checks from host ANS clustering. All 162 GPU bodies and the CUDA
library are unchanged. Warm entropy-optimization wall time improves by median
paired 20.8% / 21.7% / 20.8% at 4K / 1080p / Flower; whole encode improves
1.9% / 3.5% / 5.0%. The same-executable 1080p control nevertheless regresses
4.1% overall despite faster entropy work, and cold 4K is essentially flat.
All 71 CUDA / 50 CPU tests, three host ASan targets, broad clustering
differentials, 58 byte-identical decoded-image pairs, and batch checks pass.
See [S51](#in-place-ans-clustering-and-hoisted-log-table-access-s51) for controls,
retained outliers, and limits. This remains a targeted optimization, not proof
of a universal encoder speedup or a maxed-out implementation.

The preceding implemented checkpoint, S50 against the S48
production baseline `f22c2f8` (S49 is investigation-only), packs active integer
coefficients within the existing final submission and reads them directly into
overwrite-only final frame storage. It clears only host edge tails and transfers
ownership after validation. The extra host coefficient staging owner, full
final-buffer clear, and host layout copy disappear; D2H bytes and submission
counts are unchanged. Same-executable inclusive host handoff improves by median
paired 29.9% / 26.1% / 10.6% at 4K / 1080p / Flower, excluding the added GPU
packing cost. Warm whole-encode changes are -2.3% / -3.8% / -4.4%; cold Flower
regresses 3.2%, so this is not a universal speedup claim. All 71 CUDA / 50 CPU
tests, four host ASan targets, seven scoped CUDA sanitizer runs, 46 byte-identical
decoded-image pairs, and batch checks pass. See
[S50](#owned-active-coefficient-readback-s50) for costs, controls, and limits;
the rejected [S49](#coefficient-handoff-measurement-and-packing-prototype-s49)
prototype remains separately documented.

The preceding S48 checkpoint against `bf458d6` fuses
the three vertical five-tap RGB blurs with Opsin conversion. It removes
18 launches per normal encode and three intermediate plane writes/reads
per psycho pass without increasing scratch or transfers. The targeted
GPU subset improves by median paired 49.3% / 47.4% at 4K / 1080p;
including unchanged horizontal blurs, the bundle improves 35.3% / 33.3%.
Warm whole-encode changes remain mixed at +1.6% / -1.4% / +0.4%
for 4K / 1080p / Flower, with slower unchanged CPU work in the 4K cohort.
All 69 CUDA / 49 CPU tests, seven scoped sanitizers, 240 guarded cases
plus a tall case, 261 byte-exact prepared-map pairs, 46 byte-identical
decoded-image pairs, and batch checks pass. All 35 preexisting native
Butteraugli bodies are unchanged. No stable universal encoder speedup
is claimed, and no system settings are changed. Optimization is ongoing,
not maxed out.

The preceding checkpoint against `e7f0bcb` reduces
prepared Butteraugli storage from 33 to 27 full working planes through
lifetime reuse. Captured arena requests drop by 198,921,984 bytes at
odd 4K and 49,694,592 bytes at odd 1080p, with unchanged kernels,
launches, transfers, and cached-reference storage. All 67 CUDA / 49 CPU
tests, seven scoped sanitizers, 261 byte-exact full prepared-map pairs,
46 byte-identical decoded-image pairs, and batch checks pass. Timings
remain mixed, including +4.9% / -1.9% total GPU changes at 4K / 1080p
and +0.2% / +0.8% / -4.0% warm whole-encode changes at 4K / 1080p /
Flower. This establishes lower requested memory, not a stable encoder
speedup or an equal reduction in retained VRAM. No system setting changes.
Optimization remains ongoing, not maxed out.

The preceding checkpoint against `089fce9` fuses
three vertical Butteraugli blurs with low/medium construction, removing
18 launches per encode and three intermediate plane writes/reads per
psycho pass.
The target GPU bundle improves by median paired 16.5% / 24.3% at
4K / 1080p; total GPU changes are -0.2% / -4.1%. Primary warm
whole-encode changes are +4.0% / +11.6% / -7.3% at 4K / 1080p /
Flower. A same-executable control favors the fusion, but substantial
changes in unchanged CPU work and operating state prevent a stable
whole-encoder speedup claim. All 67 CUDA / 49 CPU tests, 220 guarded
cases plus a tall-image case, seven scoped sanitizers, batch checks,
and 46 byte-identical decoded image pairs pass. All 34 existing
Butteraugli native bodies are unchanged; allocations, transfers, and
system settings are unchanged. Optimization remains ongoing, not maxed out.

The preceding checkpoint against `51790b8` fuses
Butteraugli's L2 difference and final masking, avoiding six intermediate
plane writes/reads and four launches. The target GPU work improves by
median paired 34.6% / 34.7% at 4K / 1080p; total GPU kernel time improves
2.0% / 3.5% in the three-pair trace cohorts. Primary warm whole-encode
changes are -0.8% / -1.8% / -3.3% at 4K / 1080p / Flower, with individual
regressions and mixed cold results retained. All 65 CUDA / 49 CPU tests,
360 guarded differential cases, seven scoped sanitizer checks, batch
checks, and 46 byte-identical decoded image pairs pass. The original
33 Butteraugli native bodies remain unchanged; the new fused body has
zero stack/local/shared allocation. Device allocation requests and
host/device copies are unchanged, as are all system settings. Optimization
remains ongoing, not maxed out.

The preceding checkpoint against `efda0a0` makes AC
nonzero counting a contiguous integer reduction followed by LLF subtraction.
The qualified MSVC build vectorizes the plane scan using baseline x64
operations. Primary warm coefficient-tokenization work improves by median
paired 14.7% / 13.9% / 5.5% at 4K / 1080p / Flower; whole-encode changes
are -2.3% / -1.6% / +9.9%. The Flower regression remains adverse in
same-path and single-executable controls, although smaller; its cause is
unresolved and no stable small-image or cold whole-encode gain is claimed.
All 64 CUDA / 49 CPU tests, 4,096-case independent count/token/population
checks, fully instrumented host ASan, batch checks, and 46 byte-identical
decoded image pairs pass. All-zero and LLF-only isolated cases consistently
improve, while some other cases regress. CUDA and system settings remain
unchanged. Optimization is ongoing, not maxed out.

The preceding checkpoint against `0f5a7a7` replaces
per-token string-owning success results in direct AC accumulation with a
private enum, preserving the original checks and error codes. Coefficient
tokenization worker time improves by median paired 25.4% / 27.8% / 21.5%
at 4K / 1080p / Flower; warm whole-encode changes are
-5.4% / -3.6% / -3.8%. Cold results are mixed and near-neutral on the
smaller inputs. All 64 CUDA / 49 CPU tests, the expanded fully instrumented
host ASan fixtures, batch checks, and 46 byte-identical decoded image pairs
pass. The new 3,072-case differential test checks exact tokens and sparse
populations across transform layouts, patterns, orders, context maps, and
collection modes. No CUDA source or system setting changes. Optimization
remains ongoing, not maxed out.

The preceding checkpoint against `2260047` inlines
validated HybridUint conversion and removes string-owning success results
from the private ANS recurrence. Section-writing wall time improves by
median paired 42.0% / 34.3% / 16.0% at 4K / 1080p / Flower. Warm
whole-encode changes are -3.3% / -4.4% / -2.6%; cold paired medians also
favor the change, but individual runs and some unchanged phases regress.
All 63 CUDA / 48 CPU tests, fully instrumented host ASan checks, batch
checks, and 46 byte-identical decoded image pairs pass. Entropy policy,
arithmetic, failure contracts, and bytes are preserved. No CUDA source or
system setting changes; thermal/power limiting remains a measurement
caveat. Optimization remains ongoing, not maxed out.

The preceding checkpoint against `8f832d1` removes
per-coefficient branches and repeated pointer loads from host scan-order
zero counting without changing sampling, entropy policy, or bytes. Order
work improves by median paired 50.3% / 47.8% / 31.4% at 4K / 1080p / Flower;
whole-workflow warm medians improve 3.9% / 2.3% / 5.3%, while cold results
remain mixed. All 96 isolated order-case medians improve. All 63 CUDA / 48
CPU tests, the fully instrumented host ASan fixture, serial/batch checks,
and 46 byte-identical decoded image pairs pass. No CUDA kernel, allocation,
transfer, or system-setting change is made. Optimization remains ongoing,
not maxed out.

The preceding checkpoint against `fe13a54` fuses vertical
Butteraugli blur with in-place frequency splitting, removing 24 launches and
the intermediate blurred-plane write/read. Three production profile pairs
reduce the targeted vertical/split subset by median paired 26.6% / 22.6%
at 4K / 1080p; including unchanged horizontal blur, the bundles improve
15.3% / 13.7%. Total GPU changes are -0.9% / +2.4%, and wall timings remain
mixed; no stable whole-encoder gain is claimed. All 62 CUDA / 47 CPU tests,
seven scoped sanitizer checks, serial/batch checks, and 46 byte-identical
decoded image pairs pass. Allocations and transfers are unchanged. Every
preexisting Butteraugli native body is preserved, and the measured fused
entries have no additional register/shared or stack/local allocation.
Operating-state limitations remain recorded, with no system-setting changes.
Optimization remains ongoing, not maxed out.

The preceding checkpoint against `6fa9132` specializes
the seven physical resident coefficient shapes for both scored and
encoding-only passes, using smaller blocks for 64/128-coefficient transforms.
All fourteen native entries have zero stack/local allocation and retain the
existing arithmetic and error contracts. Three production profile pairs
reduce the coefficient subset by median paired 42.8% / 36.5% at 4K / 1080p.
Total GPU changes are -0.5% / -1.4%, but whole-encoder wall timing remains
mixed; no stable end-to-end gain is established. All 61 CUDA / 47 CPU tests,
seven scoped sanitizer checks, serial/batch checks, and 46 byte-identical
decoded image pairs pass. Explicit allocations and transfers are unchanged.
A sustained isolated probe directly observes software thermal/power limiting;
this limits performance interpretation, not correctness qualification. No
system settings are changed. Optimization remains ongoing, not maxed out.

The preceding checkpoint against `e968baa` omits unused
float reconstruction and LLF restoration in the encoding-only final
coefficient pass, preserving all validation and diagnostic paths. Three
production profile pairs reduce that final pass by median paired 33.0% /
35.3% at 4K / 1080p. All-coefficient time improves 7.2% / 15.8%; total GPU
timing is mixed, and no stable whole-encoder gain is established. Explicit
allocations and transfers do not change. All 61 CUDA / 47 CPU tests, seven
scoped sanitizer checks, serial/batch checks, and 46 byte-identical image
pairs with matching decoded scores pass. Both encoding-only and scored
materialization policies are qualified. Optimization remains ongoing,
not maxed out.

The preceding coefficient-kernel checkpoint against
`fbbe265` fuses Y quantization, X/B prediction, and color restoration,
removing three block barriers and intermediate reconstruction accesses.
A separately bounded entry uses 64 registers to preserve four feasible
256-thread blocks per SM on the qualified SM86 device. All fourteen
isolated shape/extent median pairs improve. Four production 4K profile pairs
reduce the coefficient subset by a median paired 17.8%; total GPU time
changes -0.8% with one regression, and whole-encoder wall results are mixed.
No stable end-to-end speedup is claimed. All 61 CUDA / 47 CPU tests, seven
scoped sanitizer runs, serial/batch checks, and 23 byte-identical codestream
pairs with matching decoded scores pass. Optimization remains ongoing,
not maxed out.

The preceding coefficient-kernel checkpoint against
`5ecb5f8` reuses tiny DC bases and eliminates mandatory thread-local parameter
and lookup-array copies. Its stack frame shrinks from 112 to 32 bytes, with
unchanged register count and 256 bytes of shared storage per block. Final
isolated large-shape probes improve 38.9-47.6%; the production coefficient
subset improves 45.0% / 40.9% / 43.7% at 4K / 1080p / Flower, with unchanged
launch counts, arena allocation sizes, and transfers. Whole-encoder timing
is highly variable: both warm cohorts are nearly flat at 4K, while the first
cohort's smaller-input regressions reverse in the second cohort.
No stable total-encode speedup is claimed. All 61 CUDA / 47 CPU tests,
seven scoped sanitizer runs, serial/batch checks, and 23 byte-identical
decoded image pairs pass. Optimization remains ongoing, not maxed out.

The preceding resident-transform checkpoint against
`5aefb93` removes packed gather/inverse pixel buffers, saving 199.07 MB of
live device allocation requests at padded 4K. It eliminates 21 launches and
reduces the targeted production transform/copy subset by a median paired
68.6% in four 4K profile pairs. Required host/device transfers and qualified
outputs are unchanged. The final warm cohort improves total time by paired
medians of 4.4% / 2.5% / 1.0% at 4K / 1080p / Flower, but the initial
cohort is flat and timings remain variable; no uniform end-to-end gain is
claimed. Both candidates and all observations remain documented.
All 60 CUDA / 47 CPU tests, seven scoped sanitizer runs, serial/batch checks,
and 23 byte-identical decoded image pairs pass. Optimization is ongoing;
the resident path is not demonstrated maxed out.

The preceding coefficient-staging checkpoint against
`b84ae35` removes a redundant 99.53 MB host clear at padded 4K without
changing allocation capacity, transfers, kernels, or frame layout. A
same-executable probe reduces combined staging/readback/assembly time from
62.1 to 50.5 ms at 4K, with all nine paired host-stage observations favorable.
Production warmed quantization improves 2.2% / 5.0% at 4K / 1080p, but total
changes are small and mixed, and Flower regresses. No stable whole-encode
speedup is claimed. A separate group-ordered assembly prototype is rejected
after broader 4K regressions. All 59 CUDA / 47 CPU tests, three scoped
sanitizer runs, serial/batch checks, and 23 byte-identical decoded image
pairs pass. Optimization remains ongoing, not maxed out.

The preceding host-staging checkpoint against `83200fb`
removes 99.46 MB of unused host RGB staging from a 3839x2159 encode, with no
change in device allocations, transfers, or kernels. A same-executable probe
reduces the host staging allocation/initialization phase from 31.0 to 13.2 ms
at 4K. Warm paired total time improves 4.5% at 4K, while smaller-input total
results are mixed; quantization improves 4.7% / 6.5% / 2.9% at 4K / 1080p /
Flower. All 59 CUDA and 47 CPU tests, three scoped sanitizer runs, batch
checks, and 23 byte-identical decoded image pairs pass. Diagnostic
reconstruction remains supported and allocates its host staging on first use.
Optimization remains ongoing, not demonstrated maxed out.

The preceding private-memory-pool checkpoint against
`f1f9fe6` reduces warmed fully-resident total time by paired medians of 16.1%
at odd 4K, 16.5% at odd 1080p, and 9.8% on Flower. Kernel counts, transfers,
and live requested allocation sizes are unchanged; warmed 4K allocation/free
API time falls from 61.0 to 0.31 ms. This trades retained GPU memory for reuse:
the default release target is half the device's memory, capped at 4 GiB,
shared across matching backend lanes, with an explicit cache-trim API.
Cold first-encode time is mixed or worse, so no cold-start gain is claimed.
All 59 CUDA and 47 CPU tests, ten explicitly scoped sanitizer runs, batch
checks, 36 same-executable policy outputs, and 23 byte-identical decoded
image pairs pass. The resident path is not demonstrated maxed out.

The preceding fused-Malta checkpoint against `eb1b624`
removes 24 launches per encode without changing transfers or scratch capacity.
An extended-warmup production-kernel probe improves 6.9-11.1%; whole-encoder
Malta profiles improve 22.4% at 1080p and 13.4% on Flower. Four 4K profile
pairs give a 5.6% median Malta improvement but mixed total-GPU results.
Wall-time changes are small and noisy, so no stable end-to-end gain is claimed.
All 58 CUDA and 47 CPU tests, ten explicitly scoped sanitizer runs, batch
checks, and 23 byte-identical decoded image pairs pass qualification.
Optimization remains ongoing, not demonstrated maxed out.

The preceding cooperative-adjustment checkpoint against
`f26e2ef` reduces its targeted kernel time by 84.4% at 4K, 92.4% at 1080p,
and 96.7% on Flower, with unchanged launches/transfers. All 56 CUDA and 47
CPU tests pass, as do eight sanitizer runs (full-AQ race instrumentation is
kernel-filtered). All 23 decoded image pairs remain byte-identical. Separate
parent/candidate cohorts, an identical-binary control, and a same-executable
policy probe show substantial wall-time variability: no stable end-to-end
gain is claimed. The same-executable probe also verifies exact output on the
three actual benchmark inputs. This is a kernel-level improvement, not a
demonstrated overall speedup or performance ceiling.

The preceding factorized-DCT checkpoint against `c1a75bb`
reduces 4K DCT kernel time from 107.9 to 23.1 ms (78.6%), without adding launches
or transfers. Paired fully-resident total time improves 5.1% at 4K, 7.2% at
1080p, and 7.8% on Flower; quantization improves 10.6%, 8.9%, and 5.9%.
All 55 CUDA and 47 CPU tests, 13 sanitizer runs, batch checks, and 23 decoded
before/after image pairs pass qualification. Rounding changes: only ten image
pairs are byte-identical, while independently decoded Butteraugli scores are
unchanged in 22 pairs and improve slightly in one. The largest size increase
is 0.028%. Laptop timing variance and corpus/device limits remain explicit.

The preceding compact-scratch checkpoint against `5ba4d86`
reduces the padded-4K AC-search arena from 626.8 to 323.8 MB (48.3%), saving
288.9 MiB per search. Nsight confirms that peak tracked device allocations fall
from 3.110 to 2.808 GB, with identical launches and transfer volumes. Paired total
time improves 3.2% at 4K, is effectively unchanged at 1080p (-0.2%), and improves
5.0% on Flower; substantial laptop timing variance remains. Batch checks pass,
but do not establish a batch-throughput gain. Qualified codestreams remain
byte-identical.

The preceding residual/inverse/loss fusion reduced targeted AC stages by 23.3%
at 4K, removing seven launches and a 2.55 GB coefficient-buffer round trip.
These are incremental gains, not a demonstrated performance ceiling.
Filtering, resident coefficient encoding, remaining resident allocations, host work, and
transfers remain material targets; the resident path has not reached its
performance limit.

The initial measurements showed two different performance profiles.

- Fully-resident encoding is compute-bound. Forward and inverse DCT kernels
  consume `133.7 ms`, or about 67% of measured GPU kernel time, in the warmed
  padded-1080p trace.
- Maximum-throughput encoding is not GPU-compute-bound. It executes only
  `10.1 ms` of kernels at 1080p. Its remaining quantization time is dominated
  by host preparation and validation, 27 device allocations and frees,
  synchronous transfer boundaries, readback, and host frame assembly.

The highest-confidence next optimization is to consolidate per-image device
allocations into a small number of arenas. It applies to both encoding modes,
has a measured CUDA API ceiling of `23.9 ms` in maximum-throughput and
`73.9 ms` in fully-resident encoding, reduces fragmentation and batch memory
pressure, and establishes stable pointers for later CUDA Graph capture.

After allocation consolidation, the next work should remove encoding-only
readbacks and host materialization. The main compute optimization should then
specialize the 32x32 and 16x32/32x16 DCT kernels. Moving linear-RGB-to-Opsin
preparation to CUDA has a larger architectural scope but becomes especially
valuable at 4K.

Adding more production streams is not currently justified. The two-lane pool
improves throughput, while the fully-resident path already shows contention
for shared GPU compute resources.

## Scope and method

The primary boundary was the public in-memory workflow from a caller-owned
linear RGB image through a completed codestream. Input generation, file I/O,
backend construction, and optional final-score diagnostics were excluded.

The measurements used:

- `gjxl_cuda_encoding_benchmark` for public-workflow wall time by stage;
- `gjxl_image_batch_benchmark` for paired serial and batch throughput;
- Nsight Systems 2023.2.3 for CUDA API, kernel, launch-geometry, and memory
  operation traces; and
- Nsight Compute 2022.3 as an attempted source of occupancy and hardware
  counter data.

Each latency workload was warmed before measurement. The principal 1080p
comparison used two warmups and seven samples. The 4K comparison used one
warmup and three samples. Nsight traces captured one warmed 1080p sample per
policy inside the benchmark's `cudaProfilerStart`/`cudaProfilerStop` range.
Profiler start time is excluded from every CUDA API total below.

The workload is synthetic and has odd source dimensions (`1919x1079` and
`3839x2159`) so that the normal padding path remains covered. Kernel geometry
and fixed per-pixel work are representative; entropy size and host
serialization behavior can vary for natural images.

This laptop changes clocks and power state aggressively. Absolute batch times
varied even after warmup, so the alternating paired speedup is more useful
than comparing isolated serial and batch medians. One abandoned batch-8 sweep
continued running after the command runner yielded and thermally contaminated
intermediate observations. That process was stopped, the GPU was returned to
an idle state, and none of the contaminated batch numbers are reported here.

## Public-workflow wall profile

### Padded 1080p

Warmed median times in milliseconds:

| AQ mode | Total | Input preparation | Quantization pipeline | Codestream encoding |
|---|---:|---:|---:|---:|
| Exact coefficients | 2756.6 | 1035.9 | 1661.3 | 69.1 |
| Fully resident | 430.9 | 36.8 | 317.9 | 71.2 |
| Throughput | 441.3 | 41.1 | 325.1 | 78.4 |
| Maximum throughput | 260.3 | 42.5 | 132.5 | 69.9 |

The resident modes remove most of the CPU-compatible preparation performed by
the exact-coefficient path. `throughput` is intentionally equivalent to
fully-resident inside the encoding workflow, so its small difference is run
variance rather than a separate optimization opportunity.

Quantization accounts for about 74% of fully-resident wall time and 51% of
maximum-throughput wall time. In maximum-throughput mode, input preparation
and CPU codestream generation are already material Amdahl limits.

### Padded 4K

Warmed median times in milliseconds:

| AQ mode | Total | Input preparation | Quantization pipeline | Codestream encoding |
|---|---:|---:|---:|---:|
| Fully resident | 2405.3 | 237.8 | 1718.3 | 442.9 |
| Maximum throughput | 1044.6 | 226.6 | 496.0 | 252.4 |

Fully-resident quantization remains dominant at 4K, at about 71% of total
time. Maximum-throughput quantization falls to about 47%; input preparation
and serialization together account for nearly another half of total wall
time. Optimizing only CUDA kernels therefore cannot produce a large multiple
of end-to-end maximum-throughput performance.

## CUDA timeline profile

### Fully resident

The warmed padded-1080p trace contains:

| Item | Measured value |
|---|---:|
| Kernel launches | 497 |
| GPU kernel execution | 200.0 ms |
| CUDA event synchronizations | 5 |
| Device allocations / frees | 27 / 27 |
| Allocation and free API time | 73.9 ms |
| Kernel launch API time | 12.7 ms |
| Memory operations | 60 copies, 3 memsets |
| Host-to-device payload | 54.6 MB |
| Device-to-host payload | 34.6 MB |
| Device copy execution | 13.7 ms |
| Blocking copy API time | 41.8 ms |

CUDA API time and GPU execution time are not additive. In particular,
`cudaEventSynchronize` is the host waiting for already-counted GPU work, and a
synchronous free can expose earlier outstanding work. The evaluator waits for
its submissions before destruction, however, so the repeated post-completion
allocation/free cost remains a meaningful optimization target.

The kernel-time distribution is:

| Kernel group | Time | Share of kernel time |
|---|---:|---:|
| Forward DCT | 68.1 ms | 34.1% |
| Inverse DCT | 65.6 ms | 32.8% |
| Malta response | 9.0 ms | 4.5% |
| Residual | 7.2 ms | 3.6% |
| Adjusted-quant selection | 5.7 ms | 2.9% |
| Four convolution variants | 12.1 ms | 6.1% |
| EPF | 4.9 ms | 2.4% |
| Initial CfL | 4.5 ms | 2.3% |
| All other kernels | 22.9 ms | 11.4% |

The generic DCT kernels use one 256-thread CUDA block per transform, 40
registers per thread, and one shared-memory intermediate plane. The largest
shape groups were:

| Transform group | Combined forward/inverse time |
|---|---:|
| 32x32 | approximately 66.1 ms |
| 16x32 and 32x16 | approximately 38.8 ms |
| Remaining shapes | approximately 28.8 ms |

The first two groups therefore account for approximately `105 ms`, 79% of DCT
time and 52% of all measured kernel time. They are the correct first targets
for shape-specific DCT work.

### Maximum throughput

The corresponding maximum-throughput trace contains:

| Item | Measured value |
|---|---:|
| Kernel launches | 32 |
| GPU kernel execution | 10.1 ms |
| CUDA event synchronizations | 2 |
| Device allocations / frees | 27 / 27 |
| Allocation and free API time | 23.9 ms |
| Kernel launch API time | 0.8 ms |
| Memory operations | 16 copies, 1 device copy, 2 memsets |
| Host-to-device payload | 24.9 MB |
| Device-to-host payload | 34.0 MB |
| Device copy execution | 10.5 ms |
| Blocking copy API time | 16.4 ms |

Only one kernel is individually large:

| Kernel | Time | Share of kernel time |
|---|---:|---:|
| Initial CfL | 6.2 ms | 61.1% |
| Forward DCT8 | 1.5 ms | 15.1% |
| Inverse Gaborish convolution | 0.7 ms | 6.9% |
| Quant adjustment | 0.7 ms | 6.4% |
| All other kernels | 1.0 ms | 10.5% |

`InitialCflKernel` launches only two 256-thread blocks for 510 color tiles.
Each active thread traverses one complete 64x64 tile twice and accumulates its
statistics serially. This explains why it dominates an otherwise short GPU
sequence.

Maximum-throughput also performs substantial host work around the CUDA calls:

- finite-value scans over full input images;
- copies of coding Opsin planes into contiguous host vectors;
- construction of one transform-layout entry per DCT8 block;
- allocation and initialization of compatibility output maps;
- validation and copying of downloaded quantization maps; and
- conversion of downloaded raw coefficient arrays into the final encoder
  frame.

The required final AC readback is 24.9 MB at 1080p. It cannot be eliminated
without moving codestream construction to the GPU, but it can write directly
into final frame storage and share one packed transfer with the other frame
metadata.

## Batch throughput

The clean 1080p paired measurements were:

| AQ mode | Batch | Batch throughput | Median paired speedup |
|---|---:|---:|---:|
| Maximum throughput | 1 | 2.58 images/s | 1.02x |
| Maximum throughput | 2 | 2.40 images/s | 1.19x |
| Maximum throughput | 4 | 4.44 images/s | 1.70x |
| Fully resident | 1 | 1.68 images/s | 0.98x |
| Fully resident | 2 | 1.85 images/s | 1.20x |
| Fully resident | 4 | 1.77 images/s | 1.45x |

The absolute values retain laptop power-state variance. The important signal
is that fully-resident scaling saturates earlier than maximum-throughput
scaling because its two streams compete for SM execution. The current two-lane
production pool should remain bounded at two. Allocation consolidation and
lower per-image memory pressure should be measured before reconsidering that
cap.

## Ranked optimization plan

The ceilings below are measured portions of the current trace, not promised
speedups. They overlap and cannot be summed. Every retained change needs a new
wall profile because removing one synchronization point can move time into a
later wait.

Each completed checkpoint should append a performance snapshot to its section.
The snapshot should identify the commit and workload, compare the same wall
profile before and after the change, report relevant CUDA timeline counters,
and call out noise or regressions as explicitly as improvements. Build and
test qualification belongs in the same snapshot when it materially defines
the result.

### S1.1: consolidate and reuse device allocations

**Priority:** highest; applies to both resident policies.

The prepared AC-strategy search owns three device buffers for each of seven
strategy stages, plus three shared scratch buffers. The maximum-throughput
prepared evaluator similarly allocates each image plane and working array as a
separate `DeviceBuffer`. At the public workflow boundary these objects are
created and destroyed for every image. The relevant allocation fan-out is in
`src/gpu/ops/ac_strategy_search.cpp` and `src/gpu/cuda/cuda_aq.cpp`.

Recommended implementation:

1. Plan candidates, matrices, costs, and common AC-search scratch into one or
   two `DeviceScratchArena` allocations.
2. Convert the maximum-throughput evaluator to persistent and staging arenas,
   matching the resident evaluator's ownership model.
3. Preserve arena capacity across compatible target-size attempts.
4. Consider a backend size-bucketed cache or `cudaMallocAsync` pool only after
   object-local arenas have removed the known allocation fan-out.

Object-local arenas are the preferable first step because they work with the
current CUDA 11.8 baseline, keep ownership explicit, preserve deterministic
failure behavior, and create stable pointers for later graph capture.

Measured reducible API ceiling:

- maximum throughput: `23.9 ms`, about 9% of the 1080p public wall time;
- fully resident: `73.9 ms`, about 17% of the 1080p public wall time.

The acceptance gate should require identical frames and codestreams for every
mode that promises identity, unchanged injected-failure behavior, zero
steady-state allocation for reused prepared operations, and a materially lower
allocation count for fresh public encodes.

### S1.2: remove encoding-only materialization and readbacks

**Priority:** high; lower risk than a kernel rewrite.

The encoding path currently downloads initial results that are not serialized:

- an 8.29 MB pixel mask;
- a 129.6 KB quant field; and
- a 129.6 KB strategy mask.

Maximum-throughput uses those values only to satisfy the complete public
pipeline adapter before continuing with resident frame encoding.
Fully-resident AC search already has resident Opsin, quant-field, and pixel-mask
views, but its common provider interface still requires valid host maps. The
readback implementations are in `src/gpu/cuda/cuda_aq.cpp` and
`src/gpu/cuda/cuda_aq_resident.cpp`.

Recommended implementation:

1. Add an internal encoding-only initial-quantization contract. Keep the
   complete public AQ and pipeline contracts unchanged.
2. Let resident AC search validate and consume device views without requiring
   populated host quant and pixel maps.
3. Keep numeric error checking on the device and download only the scalar
   error/quantizer values required by host control flow.
4. Allocate the final host frame arrays before readback and copy coefficients
   directly into their final destinations.
5. Pack final raw-quant, CfL, DC, AC, and error data where practical so that
   one ordered transfer replaces several synchronous copies.
6. Batch immutable metadata and small constant uploads into one staging
   payload during preparation.

The current copy API ceiling is `16.4 ms` for maximum-throughput and `41.8 ms`
for fully-resident encoding. Actual device transfer occupies `10.5 ms` and
`13.7 ms`, respectively, so eliminating host synchronization and copying is
more important than attempting to tune raw PCIe bandwidth alone.

Pinned reusable staging can improve the remaining large transfers and allow
them to overlap across the two production lanes. It should not weaken the
backend's synchronous host-buffer lifetime contract.

#### S1.2 completion snapshot (2026-09-04)

Commit `f2732d3` keeps initial quantization, masking, and initial CfL data on
the device for encoding-only operation. Resident AC search consumes those
device views directly, and fully-resident Butteraugli control reduces and
adjusts the initial field without a host field handoff. Small immutable
uploads and related readbacks share one stream synchronization per batch.
The complete public diagnostic APIs continue to materialize their maps.

The wall comparison uses the same synthetic `1919x1079` padded-1080p workload,
distance `1.2`, effort `7`, two warmups, seven GPU-only samples, and an RTX
3060 Laptop GPU. Times are warmed medians in milliseconds:

| AQ mode and stage | Before S1.2 | After S1.2 | Change |
|---|---:|---:|---:|
| Maximum throughput, total | 260.3 | 191.0 | -26.6% |
| Maximum throughput, quantization | 132.5 | 86.5 | -34.7% |
| Fully resident, total | 430.9 | 435.2 | +1.0% |
| Fully resident, quantization | 317.9 | 303.1 | -4.7% |

The post-change sample ranges were `178.8-270.3 ms` total and
`80.2-118.2 ms` quantization for maximum throughput, and `394.9-461.0 ms`
total and `297.2-317.7 ms` quantization for fully resident. The laptop's clock
and thermal variance make the fully-resident total effectively unchanged;
the smaller quantization-stage median is directionally consistent with the
copy reduction but should not be treated as a stable 4.7% end-to-end gain.

Warmed Nsight Systems captures confirm the intended traffic reduction:

| AQ mode and transfer | Before S1.2 | After S1.2 | Change |
|---|---:|---:|---:|
| Maximum throughput, DtoH | 34.0 MB / 11 copies | 25.4 MB / 8 copies | -25.2% bytes |
| Fully resident, DtoH | 34.6 MB / 23 copies | 25.9 MB / 18 copies | -25.1% bytes |
| Fully resident, HtoD | 54.6 MB / 37 copies | 54.2 MB / 34 copies | -0.7% bytes |

Maximum throughput benefits directly because the removed 8.55 MB initial-map
readback was a large share of its short GPU path. Fully resident removes the
same maps plus three host quant-field handoffs, but generic DCT execution still
dominates its wall time; this reinforces S1.3 as its next high-impact target.
The checkpoint passed all 53 tests in the CUDA build and all 47 tests in the
CPU-only build. The ignored trace artifacts are
`s12_maximum_throughput_1080p.nsys-rep` and
`s12_fully_resident_1080p.nsys-rep` under `build-cuda-ninja/profiles`.

### S1.3: specialize the dominant DCT shapes

**Priority:** highest compute optimization for fully-resident encoding.

The current transform kernel performs a generic separable matrix multiply for
all supported shapes. A shape-specific path should start with 32x32, then
16x32/32x16. The current implementation is in
`src/gpu/cuda/cuda_kernels.cu`. Useful candidates include:

- tiled shared-memory row and column passes;
- warp-specialized fixed-size transforms;
- factored radix-2 transforms when they preserve the accepted result; and
- shape-specific block sizes instead of reserving 256 threads for every
  transform.

A 2x improvement over the two dominant shape groups would save approximately
`52 ms` at 1080p. A 2x improvement over all DCT work would save approximately
`67 ms`, about 16% of quantization time and 15% of total fully-resident wall
time in this profile.

Fast math and tensor-core arithmetic should remain disabled until a separate
decision-sensitivity and output-quality contract explicitly permits them.

Before selecting a kernel design, enable NVIDIA GPU performance counters and
capture achieved occupancy, eligible warps, issue stalls, shared-memory bank
conflicts, L1/L2 traffic, and arithmetic throughput separately for the 32x32
and rectangular groups. Nsight Compute currently reports `ERR_NVGPUCTRPERM`,
so those counters were unavailable for this study.

#### S1.3 completion snapshot (2026-09-04)

This checkpoint adds compile-time CUDA paths for 32x32, 16x32, and 32x16
transforms. The arithmetic, basis values, and accumulation order remain the
same as the generic separable transform. The specialized kernels preload the
horizontal basis from a coalesced global-memory copy into a shared-memory tile
with one padding column. This replaces the generic horizontal pass's
warp-divergent constant-memory accesses and prevents shared-memory bank
conflicts. The vertical basis continues to use constant-memory broadcasts.
Fast math, factored transforms, and tensor-core arithmetic remain disabled.

Launch metadata also corrected an ambiguity in the initial profile. A 2,048
byte intermediate identifies only a 512-coefficient transform; it does not
identify its dimensions. The production call-site metadata shows that those
dominant launches are 16x32 and 32x16, not 8x64 and 64x8. The earlier labels
in this document have been corrected accordingly.

Nsight Compute 2022.3 was retried on the 32x32 forward transform, but the
driver again returned `ERR_NVGPUCTRPERM`. Kernel selection therefore used
Nsight Systems launch metadata and matched before/after timing rather than
hardware counters. In one warmed padded-1080p trace, combined forward and
inverse times were:

| Transform scope | After S1.2 | After S1.3 | Change |
|---|---:|---:|---:|
| Primary 32x32 batch | 53.0 ms | 4.0 ms | -92.5% |
| Primary 16x32 batch | 12.4 ms | 1.9 ms | -85.1% |
| Primary 32x16 batch | 40.3 ms | 2.0 ms | -94.9% |
| All occurrences of the three shapes | 138.9 ms | 10.4 ms | -92.5% |
| All DCT kernels | 176.4 ms | 39.1 ms | -77.8% |

The public wall comparison uses the same synthetic `1919x1079` workload,
distance `1.2`, effort `7`, two warmups, seven GPU-only samples, and RTX 3060
Laptop GPU as the S1.2 snapshot. Times are medians in milliseconds:

| AQ mode and stage | After S1.2 | After S1.3 | Change |
|---|---:|---:|---:|
| Fully resident, total | 435.2 | 317.7 | -27.0% |
| Fully resident, quantization | 303.1 | 208.1 | -31.3% |
| Maximum throughput, total | 191.0 | 186.5 | -2.4% |
| Maximum throughput, quantization | 86.5 | 78.8 | -8.9% |

The post-S1.3 fully-resident ranges were `296.9-367.0 ms` total and
`200.9-215.4 ms` quantization. Maximum-throughput ranges were
`173.7-256.6 ms` total and `77.3-99.1 ms` quantization. Maximum-throughput is
a control here: its DCT8-only encoding path does not dispatch any of the new
kernels, so its difference should be treated as clock and host variance.

The odd `3839x2159` 4K checkpoint measured `1161.3 ms` total and `766.5 ms`
quantization for fully resident, and `616.1 ms` total and `252.1 ms`
quantization for maximum throughput. Separate `1919x1079` and `3839x2159`
fully-resident CLI encodes were decoded successfully by the pinned `djxl`,
which reported the original dimensions. Paired 1080p batch qualification
produced these medians:

| AQ mode | Batch | Batch ms | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 354.9 | 2.82 | 0.95x |
| Fully resident | 2 | 557.7 | 3.59 | 1.19x |
| Fully resident | 4 | 974.6 | 4.10 | 1.51x |
| Maximum throughput | 1 | 230.5 | 4.34 | 1.00x |
| Maximum throughput | 2 | 307.9 | 6.50 | 1.42x |
| Maximum throughput | 4 | 737.8 | 5.42 | 1.29x |

The batch run began at 63 C, P8, and a 210 MHz SM clock and ended at 68 C,
P5, and 892 MHz, reinforcing that paired speedups are more reliable than its
absolute times. The checkpoint passed all 53 tests in the CUDA build and all
47 tests in the CPU-only build. CUDA coverage includes per-shape comparison
with the double-precision DCT reference, exact-mode CPU/CUDA codestream
identity, and iteration-zero fully-resident codestream identity. The ignored
trace artifacts are `s13_fully_resident_1080p.nsys-rep` and its SQLite export
under `build-cuda-ninja/profiles`.

#### S1.3 follow-up: pack the remaining AC-search DCT shapes (2026-09-04)

The current padded-4K trace exposed a different transform mix after the first
S1.3 specializations removed the formerly dominant 32-wide cost. The generic
8x8 kernel assigned one 256-thread block to only 64 coefficients, while 16x8
and 8x16 used only 128 lanes. The still-generic 16x16 path occupied every lane
but retained the horizontal pass's divergent constant-memory basis access.
Together those four shapes consumed `119.601 ms` in AC search, 77.4% of its
DCT time and 28.4% of all GPU kernel execution.

This follow-up adds compile-time kernels that execute four independent 8x8
transforms or two independent 128-coefficient transforms in each 256-thread
block. The 16x16 kernel uses one transform per block. All four variants load a
single padded horizontal-basis tile into shared memory and give every thread
one coefficient. A partial final block is explicitly masked. The dense basis,
coefficient layout, scaling, per-output accumulation order, and public
standalone transform contract remain unchanged.

Seven alternating independent-process parent/candidate pairs used the
synthetic odd `3839x2159` workload, distance `1.2`, effort `7`, fully-resident
AQ, one internal warmup, and one retained GPU-only sample. Times are cohort
medians in milliseconds; the paired ratio is also reported because host work
and laptop boost state remained noisy:

| Stage | Parent `39397b7` | Packed DCT | Cohort change | Median paired change |
|---|---:|---:|---:|---:|
| Complete workflow | 929.948 | 776.815 | -16.5% | -12.8% |
| Quantization pipeline | 635.794 | 530.151 | -16.6% | -14.4% |

All seven quantization pairs favored the packed kernels. Parent quantization
ranged from `613.765-664.463 ms`; the candidate ranged from
`523.044-590.200 ms`. Complete-workflow ranges were `830.120-1043.372 ms`
and `763.330-987.404 ms`, respectively. Two complete-workflow pairs regressed
despite faster quantization because the CPU codestream stage varied
independently; the quantization and GPU-timeline results are the reliable
signals for this checkpoint.

Matched warmed padded-4K Nsight Systems captures isolate the intended change:

| GPU scope | Parent | Packed DCT | Change |
|---|---:|---:|---:|
| All kernel execution | 420.943 ms | 332.657 ms | -21.0% |
| AC-strategy search | 210.652 ms | 123.390 ms | -41.4% |
| AC-search DCT | 154.592 ms | 64.583 ms | -58.2% |
| Targeted 8x8/16x8/8x16/16x16 DCT | 119.601 ms | 26.521 ms | -77.8% |
| Resident AQ and reconstruction | 180.324 ms | 179.315 ms | -0.6% |

Kernel launch count remains 516 because packing reduces the grid size of each
small-shape dispatch rather than its launch count. AC search is no longer the
largest phase in the candidate trace: resident AQ and reconstruction now lead
at `179.315 ms`. The unchanged 32-wide DCT families were collectively about
3 ms slower in the candidate trace, consistent with the session's clock and
thermal variation; the packed families still saved about 93 ms on their own.

All 53 tests passed in the CUDA build. The CUDA transform test covers every
supported shape against the double-precision reference, and the CUDA
AC-strategy test covers CPU cost parity, resident quant-norm evaluation,
invalid descriptors, and submission failure behavior. Compute Sanitizer
memcheck reported zero errors for the CUDA AC-strategy suite. Parent and
candidate effort-7 and effort-9 fully-resident encodes of
`testdata/codestream_sample.pfm` were byte-identical, with SHA-256 values
`61be086bab4db87984699245cc0fe2eef107050d667b1070c5d93e3a25a37f5d`
and `14dfb18d19fd11e590513b3a2188fc97bd2c8402ceaad61a46f3a65cac32eab3`,
respectively. Ignored trace artifacts use the `s17_ac_packed_4k` prefix under
`build-cuda-ninja/profiles`.

### S1.4: move input preparation to CUDA

**Priority:** high end-to-end potential, broader architectural change.

`PrepareWorkflow` currently performs `LinearRgbToPaddedOpsin` before backend
selection. This costs about `43 ms` at 1080p and `227-238 ms` at 4K in the
measured GPU workflows. Fully-resident preparation then uploads both the
original linear image and CPU-produced Opsin image, accounting for six large
plane uploads. The current workflow boundary is in
`src/codestream/workflow.cpp`.

A CUDA-native frontend should:

1. select or resolve the forced CUDA backend before materializing Opsin;
2. upload the three linear RGB planes once;
3. perform padding and linear-RGB-to-Opsin conversion on the assigned lane;
4. expose the resident Opsin views to initial AQ and AC strategy search; and
5. compute any required quantization-matrix scale statistics without forcing a
   complete Opsin image back to the host.

For maximum-throughput this moves a large CPU stage to an otherwise lightly
loaded GPU. For fully-resident it also removes roughly half of the initial
full-image H2D payload. The change touches workflow preparation, provenance,
validation, and matrix-scale statistics, so it should follow the more local
arena and readback work.

#### S1.4 completion snapshot (2026-09-04)

Forced CUDA workflows now resolve the backend before host Opsin preparation,
upload the three source RGB planes once, and perform edge padding plus the
linear-RGB-to-Opsin transform on the selected CUDA lane. The prepared source
and Opsin views remain resident through initial quantization, AC strategy
search, and AQ evaluation. Quantization-matrix scale selection downloads only
three scalar maxima and the device error word. CPU, Metal, and CUDA exact-
coefficient workflows retain the established host preparation path.

The wall comparison is paired against revision `7d846bb` on the same RTX 3060
Laptop GPU. It uses the same synthetic odd-sized workloads, distance `1.2`,
effort `7`, and GPU-only measurement boundary as the earlier snapshots. The
1080p runs used two warmups and seven samples; 4K used one warmup and three
samples. Times are warmed medians in milliseconds:

| Workload and stage | Before S1.4 | After S1.4 | Change |
|---|---:|---:|---:|
| 1080p fully resident, total | 377.7 | 294.3 | -22.1% |
| 1080p fully resident, input preparation | 36.8 | 7.7 | -79.1% |
| 1080p fully resident, quantization | 254.4 | 215.7 | -15.2% |
| 1080p maximum throughput, total | 183.6 | 106.9 | -41.8% |
| 1080p maximum throughput, input preparation | 40.0 | 8.2 | -79.6% |
| 1080p maximum throughput, quantization | 79.8 | 33.3 | -58.3% |
| 4K fully resident, total | 1469.9 | 1177.2 | -19.9% |
| 4K fully resident, input preparation | 157.8 | 30.3 | -80.8% |
| 4K fully resident, quantization | 1110.0 | 950.9 | -14.3% |
| 4K maximum throughput, total | 629.9 | 303.6 | -51.8% |
| 4K maximum throughput, input preparation | 175.4 | 30.8 | -82.4% |
| 4K maximum throughput, quantization | 282.2 | 99.2 | -64.8% |

The post-change 1080p total ranges were `283.8-312.8 ms` for fully resident
and `96.3-123.9 ms` for maximum throughput. The 4K ranges were
`1150.2-1208.1 ms` and `293.6-314.9 ms`, respectively. These paired runs are
the appropriate S1.4 comparison; their absolute values should not be compared
directly with the earlier S1.3 snapshot because laptop clocks and thermals
varied between sessions.

Warmed 1080p Nsight Systems captures show the intended transfer change:

| AQ mode and counter | Before S1.4 | After S1.4 |
|---|---:|---:|
| Fully resident, HtoD | 54.220 MB / 34 copies | 29.336 MB / 31 copies |
| Fully resident, DtoH | 25.924 MB / 18 copies | 25.924 MB / 19 copies |
| Fully resident, kernel launches | 499 | 501 |
| Fully resident, allocations / frees | 4 / 4 | 5 / 5 |
| Maximum throughput, HtoD | 24.917 MB / 5 copies | 24.881 MB / 5 copies |
| Maximum throughput, DtoH | 25.403 MB / 8 copies | 25.403 MB / 9 copies |
| Maximum throughput, kernel launches | 32 | 34 |
| Maximum throughput, allocations / frees | 2 / 2 | 3 / 3 |

The extra allocation is the single input arena, replacing evaluator-owned
source/coding planes. The extra DtoH operation is a 16-byte statistics/error
result. At 1080p the new conversion and statistics kernels took `0.198 ms` and
`0.171 ms` in fully-resident mode, and `0.199 ms` and `0.193 ms` in maximum-
throughput mode. Fully resident therefore removes 45.9% of initial HtoD bytes;
maximum throughput replaces the former Opsin upload with a same-sized RGB
upload, but still removes the host transform and duplicate evaluator work.

Post-change paired 1080p batch qualification produced:

| AQ mode | Batch | Batch ms | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 305.9 | 3.27 | 1.03x |
| Fully resident | 2 | 571.7 | 3.50 | 1.30x |
| Fully resident | 4 | 1080.0 | 3.70 | 1.24x |
| Maximum throughput | 1 | 120.3 | 8.31 | 1.07x |
| Maximum throughput | 2 | 172.1 | 11.62 | 1.53x |
| Maximum throughput | 4 | 287.6 | 13.91 | 2.14x |

The fully-resident batch-4 absolute result is about 10% slower than the older
S1.3 session despite the paired single-image improvement, consistent with
power-state variance and increased contention when preparation moves onto the
GPU. It should be watched in the next checkpoint rather than attributed to a
stable regression from these non-paired sessions.

The CUDA input test compares the uploaded source, padded Opsin result, and all
three matrix statistics exactly with the CPU oracle on a `257x17` source
padded to `264x24`. Baseline/current codestreams were byte-identical for the
sample image in both resident policies and for the existing odd 1080p/4K
qualification outputs. The checkpoint passed all 53 CUDA tests and all 47
CPU-only tests. The ignored traces are `s14_fully_resident_1080p.nsys-rep`,
`s14_maximum_throughput_1080p.nsys-rep`, and their paired `s14_baseline_*`
captures under `build-cuda-ninja/profiles`.

### S1.5: parallelize initial CfL

**Priority:** useful but bounded.

Assign a cooperative CUDA block or warp group to each 64x64 color tile and
reduce the means, quadratic term, and two linear terms in parallel. A retained
implementation must define its floating reduction order and demonstrate that
the resulting quantized CfL maps remain within the policy's output contract.
The serial-per-thread implementation is in
`src/gpu/cuda/cuda_aq_kernels.cu`.

The entire current ceiling is only `4.5-6.2 ms` at 1080p, approximately 1% of
fully-resident wall time and 2% of maximum-throughput wall time. This should
not displace allocation, transfer, or DCT work.

#### S1.5 completion snapshot (2026-09-04)

`InitialCflKernel` now assigns a four-thread cooperative group to each 64x64
color tile and packs 32 groups into each 128-thread block. Each thread owns one
of the CPU-compatible four accumulator lanes and visits that lane's samples in
the original order. Fixed-width subgroup shuffles then reproduce the original
`(lane 0 + lane 1) + (lane 2 + lane 3)` horizontal sum for the three means and
three regression terms. This defines the parallel floating-point order and
preserves the existing quantized-map contract rather than introducing a new
reduction approximation.

Paired Nsight Systems captures compare revision `4a29b1d` with the completed
kernel on the same RTX 3060 Laptop GPU. Each trace contains one warmed sample:

| Workload and AQ mode | Before S1.5 | After S1.5 | Change |
|---|---:|---:|---:|
| 1080p fully resident | 4.555 ms | 0.470 ms | -89.7% |
| 1080p maximum throughput | 6.175 ms | 0.539 ms | -91.3% |
| 4K maximum throughput | 5.904 ms | 1.190 ms | -79.8% |

The 1080p launch changes from two 256-thread blocks with one serial tile per
thread to sixteen 128-thread blocks with four cooperating threads per tile.
At 4K it changes from eight 256-thread blocks to sixty-four 128-thread blocks.
Register use rises from 40 to 42 per thread, no shared memory is introduced,
and the total workflow launch counts remain unchanged at 501 for fully
resident and 34 for maximum throughput.

The paired padded-1080p public wall profile used two warmups, nine samples for
maximum throughput, and seven samples for fully resident. Times are warmed
medians in milliseconds:

| AQ mode and stage | Before S1.5 | After S1.5 | Change |
|---|---:|---:|---:|
| Fully resident, total | 261.6 | 246.3 | -5.8% |
| Fully resident, quantization | 171.0 | 161.6 | -5.5% |
| Maximum throughput, total | 120.2 | 103.2 | -14.1% |
| Maximum throughput, quantization | 37.2 | 29.8 | -19.8% |

The post-change total ranges were `232.3-291.0 ms` for fully resident and
`99.9-131.0 ms` for maximum throughput; the corresponding baseline ranges
were `248.5-342.0 ms` and `109.8-141.0 ms`. The wall-stage changes are larger
than the isolated 4.1-5.6 ms kernel savings and therefore include clock and
host-timing variance. The kernel trace is the causal measurement; the wall
profile demonstrates that the gain survives the public workflow boundary.

The final post-change paired batch qualification was:

| AQ mode | Batch | Batch ms | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 251.1 | 3.98 | 1.06x |
| Fully resident | 2 | 430.9 | 4.64 | 1.24x |
| Fully resident | 4 | 790.1 | 5.06 | 1.67x |
| Maximum throughput | 1 | 110.5 | 9.05 | 0.99x |
| Maximum throughput | 2 | 170.1 | 11.76 | 1.42x |
| Maximum throughput | 4 | 281.9 | 14.19 | 1.67x |

The GPU warmed from 65 C/P8 to 74 C/P0 during this sweep. Separate baseline
and post-change batch sessions moved inconsistently by roughly 0-10% across
batch sizes, so they do not establish an S1.5 batch-throughput change. They do
show that batch sizes 1, 2, and 4 remain functional and that the two-lane pool
continues to overlap work; the isolated kernel traces remain the reliable
performance comparison for this bounded optimization.

CUDA tests now compare the resulting initial CfL maps directly against the CPU
four-lane oracle for both structured and deterministic noisy images. The test
geometry includes a partial right tile and partial bottom rows. Both maps and
complete maximum-throughput codestreams remain byte-identical, while repeated
four-worker runs cover fully-resident and maximum-throughput workflows. Odd
1080p fully-resident and maximum-throughput codestreams, plus the odd 4K
fully-resident codestream, are SHA-256 identical to their S1.4 outputs. The
pinned `djxl` decoded all three and reported the original `1919x1079` and
`3839x2159` dimensions. The checkpoint passed all 53 CUDA tests and all 47
CPU-only tests; Compute Sanitizer memcheck also reported zero errors for the
CUDA AQ suite. Ignored trace artifacts use the `s15_*` prefix under
`build-cuda-ninja/profiles`.

### S1.6: capture stable resident submissions with CUDA Graphs

**Priority:** follow-on optimization after arenas.

Fully-resident encoding launches 497 kernels and spends `12.7 ms` in launch
APIs. Once arena packing guarantees stable pointers and the execution plans
have fixed geometry, graph capture can reduce host launch work and repeated
submission setup. Dynamic strategy batches and control-mode differences may
require a small graph cache keyed by geometry and policy shape.

Maximum-throughput has only 32 launches and `0.8 ms` of launch API time, so it
does not justify a graph-specific implementation by itself.

### Follow-up: tile Butteraugli Malta response (2026-09-05)

After packed DCT, resident AQ and reconstruction became the largest GPU phase.
A fresh warmed odd-4K trace at revision `2b91775` attributes `46.621 ms` to
24 `MaltaResponseKernel` launches. Each output reads an overlapping radius-4
neighborhood from global memory and repeats bounds handling for its samples.
The compiled kernel uses 105 registers per thread.

The retained implementation cooperatively loads a 32x8 output tile and its
four-pixel halo into a 40x16 shared-memory array. Out-of-image halo entries
are zero, and partial-tile threads participate in loading and synchronization
before exiting. Each warp evaluates one row using fixed shared-memory offsets.
The response formulas, addition trees, explicit unfused square accumulation,
and stage ordering are unchanged. The kernel uses 40 registers per thread and
2,560 bytes of shared memory. It adds no device allocations or launches.

Matched warmed Nsight Systems captures on the RTX 3060 Laptop GPU show:

| GPU scope | Parent `2b91775` | Tiled Malta | Change |
|---|---:|---:|---:|
| Malta response, main scale | 37.332 ms | 12.181 ms | -67.4% |
| Malta response, subscale | 9.289 ms | 3.186 ms | -65.7% |
| All Malta response | 46.621 ms | 15.367 ms | -67.0% |
| All kernel execution | 336.691 ms | 307.198 ms | -8.8% |

Both traces contain 516 launches, 117,079,320 HtoD bytes, and 103,699,012
DtoH bytes. A 32x16 alternative was also measured: it uses 512 threads and
3,840 shared bytes per block but takes `16.365 ms` for Malta response, 6.5%
longer than 32x8 in these captures. The 32x8 shape was retained. These timings
and compiler resource counts establish the improvement without attributing
it to unmeasured occupancy or cache-hit counters.

Public workflow measurements use seven alternating parent/candidate process
pairs per workload, each with one warmup and one retained sample, distance
`1.2`, effort `7`, fully-resident AQ, automatic CPU threads, and no final-score
diagnostic. Cohort medians and median paired changes are:

| Workload and stage | Parent | Tiled Malta | Median paired change |
|---|---:|---:|---:|
| Odd 4K, total | 752.910 ms | 716.324 ms | -4.9% |
| Odd 4K, quantization | 526.240 ms | 498.863 ms | -7.0% |
| Odd 1080p, total | 224.350 ms | 240.598 ms | +1.9% |
| Odd 1080p, quantization | 140.054 ms | 137.667 ms | -5.3% |

Five of seven 4K pairs improved both total and quantization time. Parent 4K
totals ranged from `718.114-805.252 ms`, versus `686.161-765.478 ms` for the
candidate. At 1080p, six of seven quantization pairs improved, but only three
total-time pairs improved. CPU codestream serialization ranged from
`64.589-82.083 ms` in the parent and `63.538-135.582 ms` in the candidate,
despite unchanged output bytes. This session does not establish an end-to-end
1080p improvement; the 4K paired result and isolated kernel savings support
retaining the change. GPU state moved from 61 C/P8/210 MHz before the paired
runs to 71 C/P0/1762 MHz afterward.

Post-change 1080p batch checks used one warmup and three alternating
serial/batch samples:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 219.364 ms | 4.559 | 1.005x |
| Fully resident | 2 | 348.133 ms | 5.745 | 1.331x |
| Fully resident | 4 | 649.428 ms | 6.159 | 1.412x |
| Maximum throughput | 1 | 108.430 ms | 9.223 | 0.995x |
| Maximum throughput | 2 | 161.797 ms | 12.361 | 1.280x |
| Maximum throughput | 4 | 312.849 ms | 12.786 | 1.482x |

These qualify batch behavior, not a before/after batch-performance claim.
Maximum-throughput does not run Malta response. GPU state at the end of batch
qualification was 66 C/P0/1282 MHz.

All 53 CUDA-build tests and all 47 CPU-only tests pass. Butteraugli coverage
now includes `31x8`, `32x9`, `65x33`, and `127x65` inputs to exercise tile
boundaries, multiple interior tiles, and odd multiscale geometry with poisoned
host/device padding. The worst CPU-reference distance-map error is
`2.0504e-5`, below the existing `1.5e-3` tolerance. Compute Sanitizer memcheck,
racecheck, synccheck, and initcheck report zero errors or hazards on this
suite. Existing CUDA AQ tests cover exact CPU codestream identity, failure
atomicity, and repeated four-worker resident/maximum-throughput encoding.

Parent and candidate fully-resident codestreams are byte-identical for the
sample image at efforts 7 and 9 and the odd 1080p/4K qualification images at
effort 7, all with final-score collection enabled. Reported final scores also
match. The pinned `djxl` independently decodes all four at the original
dimensions. The odd 1080p SHA-256 is
`454cd84cc7ee6e915075ab741e7eec4bcc04a250c5f82d238c8ec4979f9f45f3`;
the odd 4K SHA-256 is
`86be75ba11d76a780edc6dea4b80daec848ddddfcf9452fe60f29f77afac0bc9`.

Ignored artifacts under `build-cuda-ninja/profiles` use the `s18_` prefix:
`s18_parent_4k`, `s18_malta_4k`, and `s18_tile32x16_4k` traces/SQLite exports,
`s18_paired_4k.json`, `s18_paired_1080p.json`, and qualification codestreams.

The next measured targets in the candidate 4K trace are Butteraugli
convolutions (`55.071 ms`), AC-search residual evaluation (`34.214 ms`), and
EPF (`20.597 ms`). CUDA launch API time is only `6.525 ms` for 516 launches,
so graph work should be compared against these larger compute opportunities.
CPU serialization and the remaining synchronous transfers also need attention
as GPU execution shrinks. This checkpoint does not establish a performance
ceiling or qualify other GPU generations and a natural-image corpus.

### Follow-up: specialize and tile Butteraugli blurs (2026-09-05)

At parent revision `736dbd5`, Butteraugli blur is the largest named kernel
family in a fresh odd-4K capture: `115.886 ms` over 146 launches. The 7-, 13-,
15-, and 33-tap filters use runtime loop bounds and repeatedly address
overlapping global-memory samples. The same weight sum is recomputed for
every interior pixel.

The retained implementation makes filter sizes compile-time parameters and
uses 256-thread blocks to evaluate 256x4 horizontal or 32x64 vertical output
tiles. Each block cooperatively loads its directional halo and weights into
shared memory. Four horizontal or eight vertical outputs per thread amortize
loading and synchronization. The complete weight sum is calculated once per
block in the original addition order; edge pixels still accumulate only
included weights in their original order. Padding does not contribute to
edge normalization. The separate mirrored 5-tap filter is unrolled without
tiling and preserves its reflection behavior. No approximate arithmetic or
new allocations are introduced.

The experiment compared fixed-size direct reads with two tiled layouts before
selecting the final kernels. All captures below use two warmups and one
profiled odd-4K sample, distance `1.2`, effort `7`, fully-resident AQ, and no
final-score diagnostic:

| Implementation | 7/13/15/33-tap blurs | All Butteraugli blurs |
|---|---:|---:|
| Parent, runtime bounds/direct reads | 100.142 ms | 115.886 ms |
| Fixed sizes/unrolled direct reads | 78.057 ms | 93.285 ms |
| Tiled 128x8 horizontal / 32x32 vertical | 49.854 ms | 65.447 ms |
| Tiled 256x4 horizontal / 32x64 vertical | 46.928 ms | 61.652 ms |
| Retained tiles plus unrolled mirrored 5-tap | 46.702 ms | 58.149 ms |

Specialization alone removes 22.1% of targeted time; tiling gives a further
substantial improvement. This supports reducing repeated sample access,
address calculation, and normalization work rather than attributing the
entire result to loop unrolling. It does not establish a DRAM-bandwidth or
occupancy bottleneck without hardware counters. The retained tiled kernels
use 31-55 registers per thread and at most 12,424 shared bytes per block.

Total blur time falls 49.8%. All GPU kernel execution falls from `565.944 ms`
to `487.564 ms` (-13.8%), but non-blur kernels also change from `450.058 ms`
to `429.415 ms` (-4.6%), so the entire total-kernel reduction should not be
attributed to this edit. Laptop execution state differs substantially from
the preceding Malta session; absolute times across those sessions are not
comparable. Both current parent/candidate traces still contain 516 launches,
117,079,320 HtoD bytes, and 103,699,012 DtoH bytes.

Public workflow measurements use seven alternating independent-process pairs
per workload, one warmup and one retained sample per process, and the same
distance/effort/AQ settings. Cohort medians and median paired changes are:

| Workload and stage | Parent | Retained blurs | Median paired change |
|---|---:|---:|---:|
| Odd 4K, total | 1045.051 ms | 976.917 ms | -5.4% |
| Odd 4K, quantization | 839.785 ms | 771.840 ms | -7.5% |
| Odd 1080p, total | 254.085 ms | 254.615 ms | +0.3% |
| Odd 1080p, quantization | 178.039 ms | 161.209 ms | -10.9% |
| Flower 510x532, total | 49.864 ms | 49.461 ms | -1.6% |
| Flower 510x532, quantization | 33.739 ms | 33.641 ms | -1.0% |

All seven 4K pairs improve both total and quantization time. Parent totals
range from `1010.224-1077.771 ms`, versus `948.344-1062.143 ms` for the
candidate. Five of seven 1080p quantization pairs improve, while total time
remains effectively unchanged. CPU serialization ranges from
`59.513-95.637 ms` in the parent and `60.977-105.995 ms` in the candidate.
The small Flower differences are not a reliable performance gain. GPU state
moves from 65 C/P8/210 MHz before these runs to 69 C/P3/825 MHz after the
4K/1080p pairs.

Flower uses the pinned `flower_small.rgb.depth8.ppm`, converted once outside
measurement to linear sRGB float32 PFM with the standard sRGB transfer
function. That PFM's SHA-256 is
`2ad3bf99e39d8b2d5e18130e8ab51dfb9b1ff360627a414a49646904ca3ee9cd`.
Its fully-resident frame selects all seven production AC strategies. Parent
and candidate codestreams are byte-identical for Flower at efforts 7 and 9,
the small sample at efforts 7 and 9, and the odd 1080p/4K fixtures at effort 7,
all with final-score collection enabled. Reported final scores match, and
the pinned `djxl` decodes all six at their original dimensions. Flower
codestream SHA-256 values are
`41c30c28169e09ff763cc242cce9e9b5b50db8841f2e44185bc91acc864c3b57` (effort 7)
and `e20404ed5eda52afc59cf1ab75d54d48433abcd31243e84266b04e28afd65978`
(effort 9). Synthetic fixture hashes remain those of the Malta checkpoint.

All 53 CUDA-build tests and all 47 CPU-only tests pass. Added `255x63`,
`257x67`, and `33x129` Butteraugli cases exercise complete and partial tiles,
vertical tile boundaries, wide filter clipping on small images, multiscale
evaluation, and poisoned strides. The worst CPU-reference map error is
`2.35289e-5`, within the unchanged `1.5e-3` tolerance. Compute Sanitizer
memcheck, racecheck, synccheck, and initcheck report zero errors or hazards.
Existing tests also verify exact CPU/CUDA codestream identity, failure
atomicity, allocation invariants, and repeated four-worker resident and
maximum-throughput workflows.

Post-change 1080p batch qualification uses one warmup and three paired
serial/batch samples:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 268.952 ms | 3.718 | 0.898x |
| Fully resident | 2 | 498.002 ms | 4.016 | 1.078x |
| Fully resident | 4 | 866.495 ms | 4.616 | 1.245x |
| Maximum throughput | 1 | 119.275 ms | 8.384 | 1.008x |
| Maximum throughput | 2 | 158.959 ms | 12.582 | 1.540x |
| Maximum throughput | 4 | 266.801 ms | 14.992 | 1.957x |

These are functional/overlap checks, not a before/after batch-performance
claim. In particular, resident batch-1 is slower than serial in this session.
The final GPU state is 65 C/P3/1282 MHz. Maximum-throughput without final-score
collection does not execute the changed blurs.

Ignored artifacts under `build-cuda-ninja/profiles` use `s19_` prefixes. The
paired trace baseline is `s19_parent_warm2_4k`; the earlier one-warmup
`s19_parent_4k` capture is not the comparison used above. Alternative traces
are `s19_specialized_4k`, `s19_tiled_4k`, and `s19_large_tiles_4k`; the retained
trace is `s19_final_4k`. Paired wall results are in
`s19_paired_{4k,1080p,flower}.json`.

The next trace target is AC-search `ResidualKernel` at `60.879 ms`. Source
inspection shows that every coefficient thread repeats the same candidate
validation, quant-norm field reduction, and CfL lookup. Moving this uniform
work out of individual coefficient lanes, and reducing the 1024-thread
launches, merits a measured experiment. Remaining DCT, EPF, host preparation,
serialization, and synchronous transfer costs also prevent a performance-
ceiling claim. Launch API time is only `6.572 ms` in the retained trace.

### Follow-up: pack AC-search residual evaluation (2026-09-05)

At parent revision `5513704`, a fresh odd-4K trace spends `61.815 ms` in
`ResidualKernel`. Each coefficient thread independently validates the same
candidate, computes the same strategy-aware quant norm, and loads the same
CfL factor. The norm includes a field reduction and logarithm/power
approximation for larger transforms. A 32x32 candidate repeats this work
3,072 times across its three channels, then computes the norm again in the
cost kernel. The residual reduction also uses a block-wide barrier at every
halving step, with one 64-1,024-thread block per channel transform.

The retained implementation:

- prepares one quant norm per candidate in a small separate kernel;
- temporarily stores those norms in the existing cost output, reads them
  during residual evaluation, then replaces each norm with its final cost;
- evaluates candidate validity and CfL once per channel transform;
- packs independent transforms into 256-thread blocks; and
- preserves the original halving addition tree across registers, shared
  memory where needed, and full-warp shuffles.

Cost storage is already disjoint from inputs and other scratch. Preparation,
residual evaluation, and final cost writes execute on the same ordered
stream; batch outputs are only valid after their submission completes.
This needs no new allocation, transfer, or public API. Invalid
descriptors still produce non-finite costs. Inactive tail transforms join
all required barriers but do not read candidates or touch outputs.

The selected geometry bounds register-held work to eight coefficients per
lane:

| Coefficients | Threads/transform | Transforms/block | Coefficients/thread |
|---|---:|---:|---:|
| 64 | 32 | 8 | 2 |
| 128 | 32 | 8 | 4 |
| 256 | 32 | 8 | 8 |
| 512 | 64 | 4 | 8 |
| 1,024 | 128 | 2 | 8 |

All experiments use two warmups and one profiled odd-4K sample, distance
`1.2`, effort `7`, fully-resident AQ, and no final-score diagnostic. The
AC total below includes gather, norm preparation when present, residual,
and cost kernels, but not DCT:

| Experiment | Residual | AC total |
|---|---:|---:|
| Parent: uniform work per coefficient | 61.815 ms | 102.775 ms |
| Uniform work once/block; original geometry | 53.141 ms | 92.601 ms |
| Packed residuals; at most 256 threads/transform | 30.990 ms | 73.591 ms |
| Cache norms in gather; same packed layout | 25.550 ms | 65.171 ms |
| Separate norm preparation; same packed layout | 26.286 ms | 64.558 ms |
| Separate preparation; 64 threads/transform | 16.693 ms | 58.538 ms |
| Separate preparation; 32 threads/transform | 19.503 ms | 61.280 ms |
| Retained shape-dependent layout | 13.705 ms | 54.443 ms |

Moving norm preparation out of gather keeps gather at 16 registers/thread
instead of 40. Seven preparation launches cost `0.104 ms` in the retained
trace. The all-32-thread experiment improves smaller transforms, but its
32x32 kernel takes `7.196 ms`; `cuobjdump` reports a 256-byte stack frame.
The retained 128-thread 32x32 group takes `2.317 ms` with no stack frame.
All retained residual variants use 26-33 registers/thread, 96-2,096 shared
bytes/block, and zero stack bytes. These observations support the work
sharing and register-footprint choices without claiming hardware-counter
proof of an occupancy or bandwidth bottleneck.

Residual execution falls 77.8%, and the four-stage AC total falls 47.0%.
All GPU kernels fall from `502.994 ms` to `471.199 ms` (-6.3%); unrelated
kernel time rises from `400.219 ms` to `416.756 ms` (+4.1%), so the complete
kernel total is affected by execution-state variation. Kernel launches
increase from 516 to 523. Transfers remain 117,079,320 HtoD bytes,
103,699,012 DtoH bytes, and 518,400 device-to-device bytes in both traces.

Public workflow measurements use seven alternating independent-process
pairs per workload, one warmup and one retained sample per process, with
the same distance/effort/AQ settings:

| Workload and stage | Parent median | Candidate median | Median paired change |
|---|---:|---:|---:|
| Odd 4K, total | 946.004 ms | 909.196 ms | -3.8% |
| Odd 4K, quantization | 738.959 ms | 715.452 ms | -3.7% |
| Odd 1080p, total | 232.420 ms | 228.542 ms | +0.2% |
| Odd 1080p, quantization | 159.536 ms | 156.485 ms | -3.4% |
| Flower 510x532, total | 50.418 ms | 49.974 ms | -3.5% |
| Flower 510x532, quantization | 33.167 ms | 32.382 ms | -1.3% |

All seven 4K quantization pairs and six total-time pairs improve. Parent
4K totals range from `910.241-964.777 ms`, versus `875.354-945.962 ms` for
the candidate. Four of seven 1080p quantization pairs improve, but there is
no reliable total-time improvement at 1080p. Flower's small differences
are also noisy: total-time ranges are `43.910-59.409 ms` and
`43.288-65.436 ms`. GPU state moves from 61 C/P8/210 MHz before the pairs
to 66 C/P3/1282 MHz afterward. Absolute measurements from the preceding
convolution session should not be used as this checkpoint's baseline.

All 53 CUDA-build tests and 47 CPU-only tests pass. New candidate-prefix
tests cover partial packed blocks, exact result independence from batch
length, and untouched output guards. Invalid-descriptor tests now also
check that neighboring valid candidates remain correct. Existing host-norm
and device-norm CPU-reference cost checks retain their tolerances; the
largest absolute/relative errors remain `0.00585938` / `4.03204e-7` for
32x32. Compute Sanitizer memcheck, racecheck, synccheck, and initcheck on
the AC-strategy test report zero errors or hazards.

Parent/candidate codestreams and reported final perceptual scores are
identical for the small sample at efforts 7 and 9, odd 1080p/4K at effort 7,
and Flower at efforts 7 and 9, with final-score collection enabled. The
pinned `djxl` decodes all six at their original dimensions. The six hashes
remain those recorded by the Malta/convolution checkpoints. Flower still
selects all seven production AC strategies. Full suites retain exact-mode
CPU/CUDA identity, failure-atomicity, allocation, and concurrent workflow
coverage.

Post-change 1080p batch qualification uses one warmup and three paired
serial/batch samples:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 243.159 ms | 4.113 | 0.981x |
| Fully resident | 2 | 500.675 ms | 3.995 | 1.098x |
| Fully resident | 4 | 869.448 ms | 4.601 | 1.172x |
| Maximum throughput | 1 | 102.415 ms | 9.764 | 1.000x |
| Maximum throughput | 2 | 175.958 ms | 11.366 | 1.448x |
| Maximum throughput | 4 | 259.002 ms | 15.444 | 1.781x |

These validate output stability and overlap, not a before/after batch
performance improvement. Resident batch-1 is slightly slower than serial,
and batch-2 does not improve images/s over batch-1 in this run. The final
GPU state is 67 C/P3/1282 MHz.

Ignored artifacts under `build-cuda-ninja/profiles` use `s20_` prefixes.
Trace names are `s20_{parent,hoisted,packed,cached_norm,separate_norm,
packed64,packed32,final}_4k`; each has an `.nsys-rep` and exported `.sqlite`.
Paired wall results are `s20_paired_{4k,1080p,flower}.json`. The saved parent
benchmark/encoder and `s20_verify.ps1` reproduce the measured comparison
and six-file identity/decode checks against the retained executables.

The retained profile still spends `161.739 ms` in forward/inverse DCT,
`31.603 ms` in EPF, `21.193 ms` in AC gather, and `19.441 ms` in AC cost.
Gather repeats runtime index division and candidate validation per pixel;
cost still has one coefficient-sized block per candidate and shared-memory
reductions. These are concrete follow-up targets alongside host assembly,
serialization, and transfer boundaries. This checkpoint does not establish
that fully-resident encoding is maxed out.

### Follow-up: stage and register-tile large DCTs (2026-09-05)

At parent revision `3c08276`, DCT again leads the fresh odd-4K profile:
`158.311 ms` across all shapes, including `111.060 ms` in 16x32, 32x16,
and 32x32 transforms. Their 256-thread blocks compute two or four outputs
per thread in separate dot-product loops. That reloads the same horizontal
basis for each output, then reloads the same intermediate samples for each
vertical output. Horizontal passes repeatedly read global input, and forward
column-major coefficient stores are strided across lanes.

The retained kernels cooperatively load input into a padded shared tile,
then accumulate independent outputs simultaneously in two or four registers
per thread. Each horizontal basis value and each vertical intermediate
sample is shared by those accumulators. The forward kernel reuses its input
tile for the final layout conversion and coalesced global stores, with one
additional barrier after the vertical pass. The inverse kernel loads native
coefficient order contiguously before broadcasting horizontal-pass samples.
Neither direction changes the dense basis, scaling, coefficient layout, or
any output's multiply-add order. No fast-math or factored-transform policy
is introduced.

For 16x32, two vertical basis addresses occur per warp, so that basis is also
staged in padded shared memory. The 32-wide transforms keep their vertical
constant-memory broadcasts. The final kernels retain 256 threads/block,
use 39-40 registers/thread and zero stack bytes, and require 9,536 shared
bytes for 16x32, 8,384 for 32x16, and 12,544 for 32x32. There are no new
device allocations, transfers, or kernel launches.

Experiments use two warmups and one profiled odd-4K sample, distance `1.2`,
effort `7`, fully-resident AQ, and no final-score diagnostic:

| Experiment | Large-shape DCT | All GPU kernels |
|---|---:|---:|
| Parent | 111.060 ms | 469.739 ms |
| Staged input/output, separate accumulators | 102.629 ms | 472.340 ms |
| Staging plus simultaneous accumulators, 256 threads | 70.447 ms | 417.325 ms |
| Same approach, 128 threads | 74.341 ms | 441.793 ms |
| 256 threads plus shared vertical basis for 16x32 | 69.993 ms | 420.766 ms |
| Also share vertical bases in smaller packed shapes | 71.169 ms | 429.120 ms |
| Retained large-only change, post-validation capture | 75.345 ms | 438.921 ms |

The small-shape extension raises packed-DCT time from `47.888 ms` to
`57.628 ms` in those trial captures and is not retained. Reducing the large
block size to 128 also fails to establish a win over 256. Staging alone has
limited benefit; sharing loads across independent accumulators supplies the
larger gain. These timings do not establish a hardware-counter diagnosis of
occupancy, bandwidth, or issue stalls.

A separate CUDA-event probe controls for noisy individual workflow calls.
It runs 65,536 transforms per dispatch, three warmup dispatches, then seven
samples of three dispatches each. Parent/candidate processes run in
parent-candidate-candidate-parent order. Ranges below span the two process
medians, not individual samples:

| Shape/direction | Parent median range | Retained median range |
|---|---:|---:|
| 16x32 forward | 6.551-6.745 ms | 5.956-6.024 ms |
| 16x32 inverse | 7.311-7.522 ms | 5.781-5.841 ms |
| 32x16 forward | 8.227-8.394 ms | 5.370-5.452 ms |
| 32x16 inverse | 8.401-8.488 ms | 5.440-5.491 ms |
| 32x32 forward | 23.507-24.166 ms | 12.321-13.190 ms |
| 32x32 inverse | 18.580-18.813 ms | 11.469-12.196 ms |

The post-validation workflow capture is less favorable for 16x32: its
combined forward/inverse time rises from `23.136 ms` to `24.109 ms`.
The event probe supports retaining that shape, but its workflow gain is
not established by a single trace. Combined 32x16 time falls from
`24.674 ms` to `17.538 ms`, and 32x32 from `63.250 ms` to `33.699 ms`.
Large-shape DCT falls 32.2%; all DCT falls from `158.311 ms` to
`126.923 ms` (-19.8%). Unchanged packed shapes rise from `47.251 ms` to
`51.577 ms`, while non-DCT execution is nearly unchanged at
`311.428 ms` versus `311.999 ms`. All kernel time falls 6.6%.
Both captures contain 523 launches, 117,079,320 HtoD bytes,
103,699,012 DtoH bytes, and 518,400 device-to-device bytes.

Public workflow measurements use seven alternating independent-process
pairs per workload, one warmup and one retained sample per process, with
the same distance/effort/AQ settings:

| Workload and stage | Parent median | Candidate median | Median paired change |
|---|---:|---:|---:|
| Odd 4K, total | 915.952 ms | 874.546 ms | -4.5% |
| Odd 4K, quantization | 709.443 ms | 670.049 ms | -6.1% |
| Odd 1080p, total | 245.894 ms | 214.149 ms | -11.2% |
| Odd 1080p, quantization | 162.712 ms | 142.060 ms | -11.5% |
| Flower 510x532, total | 46.744 ms | 42.446 ms | -7.3% |
| Flower 510x532, quantization | 32.041 ms | 27.749 ms | -4.7% |

All seven 4K quantization pairs improve; five total-time pairs improve.
The 4K total ranges are `895.220-1053.213 ms` and `839.043-948.671 ms`.
All seven 1080p pairs improve both stages, with total ranges of
`220.312-272.967 ms` and `211.792-232.445 ms`. Six of seven Flower pairs
improve both stages. GPU state moves from 63 C/P0/1282 MHz before the
wall pairs to 66 C/P3/1282 MHz afterward. Host serialization and execution
state still contribute to the wall-time changes; do not equate those
percentages with isolated kernel savings or compare absolute times across
earlier sessions.

All 53 CUDA-build tests and 47 CPU-only tests pass. The standalone CUDA DCT
test now uses 19 transforms per supported shape: impulses at distinct tile
positions, a constant, a checkerboard, unequal horizontal/vertical ramps,
and deterministic noise. It checks forward output and round trips, then
independent inverse output against the double-precision reference. Existing
tolerances remain `3e-5 + 3e-4 * abs(reference)` for forward output and
`5e-4 + 5e-4 * abs(reference)` for inverse/round-trip output. All nine
supported DCT shapes are covered, including the unchanged generic shapes.
AC-strategy cost errors remain unchanged.

Compute Sanitizer memcheck, racecheck, synccheck, and initcheck on the CUDA
backend test report zero errors or hazards. Initial initcheck found that
the pre-existing completion-failure fixture submitted uninitialized DCT8
input; the fixture now initializes that input without changing its failure
assertions. Runs use `--report-api-errors no` because a separate existing
test deliberately issues an invalid zero-grid launch and verifies error
consumption; memory and synchronization checking remain enabled.

Parent/candidate codestreams and reported final perceptual scores are
identical for the small sample at efforts 7 and 9, odd 1080p/4K at effort 7,
and Flower at efforts 7 and 9, with final-score collection enabled. The
pinned `djxl` decodes all six at their original dimensions. Hashes remain
those recorded by the preceding checkpoints; Flower still selects all seven
production strategies. Existing full-suite checks retain exact-mode
CPU/CUDA identity, allocation, failure-atomicity, and concurrent workflow
coverage.

Post-change 1080p batch qualification uses one warmup and three paired
serial/batch samples:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 233.947 ms | 4.274 | 0.944x |
| Fully resident | 2 | 425.952 ms | 4.695 | 1.183x |
| Fully resident | 4 | 802.579 ms | 4.984 | 1.193x |
| Maximum throughput | 1 | 97.867 ms | 10.218 | 0.998x |
| Maximum throughput | 2 | 146.258 ms | 13.674 | 1.417x |
| Maximum throughput | 4 | 284.660 ms | 14.052 | 1.757x |

These are output-stability and overlap checks, not a before/after batch
performance claim. Fully-resident batch-1 remains slower than serial.
The final GPU state is 66 C/P3/1282 MHz.

Ignored artifacts under `build-cuda-ninja/profiles` use `s21_` prefixes.
Trace names are `s21_{parent,staged_large,register_large,register128,
shared_vertical,small_shared,retained}_4k`, with `.nsys-rep` and `.sqlite`
files. The `small_shared` trace was initially captured as `s21_final_4k`
before that experiment was rejected and renamed; `s21_retained_4k` is the
final comparison. Paired wall data are `s21_paired_{4k,1080p,flower}.json`.
The saved parent executables and `s21_verify.ps1` reproduce identity/decode
qualification. `s21_dct_probe.cu`, `s21_parent_kernels.cu`, and the
`s21_probe_{parent,shared}` executables retain the independent event probe.

The retained profile still spends `126.923 ms` in DCT, `48.853 ms` in tiled
Butteraugli blurs, `42.133 ms` in Malta response, `32.227 ms` in EPF,
`20.024 ms` in AC gather, and `18.927 ms` in AC cost. Remaining DCT packing
and load-sharing opportunities, AC gather/cost reductions, neighborhood
filtering, host serialization, and transfer boundaries remain open. The
resident path is not yet at a demonstrated performance ceiling.

### Follow-up: specialize and tile EPF neighborhoods (2026-09-05)

The fresh parent is `6e4925f`, including the preceding large-DCT changes.
At odd 4K, four edge-preserving-filter (EPF) calls cost `33.330 ms`, behind
tiled Butteraugli convolution, Malta response, and the largest DCT families.
The old kernel chooses the pass at runtime and repeatedly samples mirrored
global coordinates inside the candidate, channel, and plus-patch loops.
This duplicates both neighborhood loads and coordinate work across pixels.

The retained implementation specializes all three passes and cooperatively
loads three channel planes into a shared-memory tile. A 256-thread block
produces a 32x32 output tile, with each warp processing one row at a time
and each thread processing four pixels in a non-unrolled outer loop. This
amortizes the halo without keeping four pixels' accumulators live. Halo
radii are 3, 2, and 1 for passes 0, 1, and 2. A flattened block grid avoids
introducing the CUDA grid-Y limit for tall images. No new device allocation,
transfer, or synchronization between kernels is needed.

The complete tile is loaded before any thread takes an out-of-image or
sigma-bypass branch. Bypass continues the per-thread pixel loop rather than
discarding later rows. Halo coordinates are mirrored from the original
patch coordinates: mirroring a candidate center first and then its patch
would differ at boundaries. The existing general modulo reflection remains
unchanged, including repeated reflection for one- and two-pixel dimensions.
Candidate/channel/patch accumulation order, explicit FMAs, division, sigma
threshold, border weighting, and non-finite error handling remain unchanged.
There is no fast-math or quality-policy change. Exact mode uses this shared
kernel too, but fully-resident performance is the optimization target.

Exploratory 4K traces use two warmups and one captured effort-7/distance-1.2
encode. They contain two calls each of passes 1 and 2, not pass 0:

| Variant | EPF total | Gaborish total |
|---|---:|---:|
| Parent, runtime pass | 33.330 ms | 3.032 ms |
| Pass specialization only | 10.880 ms | 3.531 ms |
| Specialized, branch-based mirror shortcut (rejected) | 39.809 ms | 5.565 ms |
| Shared 32x8 output tile | 6.611 ms | 3.036 ms |
| Shared 32x16 output tile | 5.792 ms | 3.488 ms |
| Shared 32x32 output tile | 5.673 ms | 3.519 ms |
| Retained 32x32, fresh capture after cleanup | 6.253 ms | 3.479 ms |

The mirror shortcut tried an interior fast return and one reflection before
falling back to modulo. It increased register use and slowed EPF as well
as Gaborish, which shares that helper; it is not retained. Gaborish is
otherwise unchanged and is an execution-state control. The small difference
between 16- and 32-row tiles is not a robust standalone speedup claim;
32 rows is the lowest exploratory total and avoids more duplicate halo loads.

The final EPF reduction is `27.077 ms` (81.2%). All-kernel time moves from
`435.721` to `418.439 ms` (4.0%), while unchanged DCT work moves from
`123.636` to `128.497 ms` and tiled blurs from `49.899` to `47.046 ms`.
These controls demonstrate the continuing clock/scheduling noise: do not
attribute the entire GPU-time difference to EPF. The retained compiler
reports 56/40/38 registers per thread and 17,328/15,552/13,872 shared bytes
for passes 0/1/2, with zero local bytes or stack. The parent's runtime kernel
uses 78 registers. Pass 0 has functional and sanitizer coverage but is not
timed by this default-profile comparison.

Initial public-workflow measurements use seven alternating independent
parent/candidate process pairs, one warmup and one measured encode per
process, GPU-only fully-resident mode, no final-score collection. Cohort
medians and median within-pair changes are separate statistics:

| Workload/stage | Parent median | Candidate median | Median paired change |
|---|---:|---:|---:|
| Odd 4K total | 894.389 ms | 849.697 ms | -5.2% |
| Odd 4K quantization pipeline | 686.358 ms | 644.961 ms | -6.3% |
| Odd 1080p total, initial | 217.774 ms | 228.658 ms | +6.2% |
| Odd 1080p quantization pipeline, initial | 143.347 ms | 149.054 ms | +4.8% |
| Flower total | 52.578 ms | 47.331 ms | -1.6% |
| Flower quantization pipeline | 33.572 ms | 31.560 ms | -2.3% |

Six of seven 4K total-time pairs and all seven quantization pairs improve;
total ranges are `866.884-972.024 ms` and `821.979-910.112 ms`. Flower's
small paired result is not strong evidence of a consistent wall-time gain.
The initial 1080p regression is retained in the record, not discarded:
candidate total times span `206.979-297.130 ms` versus `211.716-240.062 ms`
for the parent, and one candidate's unchanged CPU serialization takes
`125.373 ms` versus its paired parent's `66.116 ms`. Both host and
quantization-stage variation require further qualification.

A fresh three-warmup 1080p trace reduces EPF from `7.600` to `1.323 ms`
(82.6%) and all kernels from `88.353` to `80.271 ms`, while unchanged DCT
totals are `16.777` and `16.684 ms`. The follow-up wall comparison keeps
seven alternating process pairs but uses three warmups and five measured
encodes per process, comparing each process's median. All seven pairs
improve both stages. Total cohort medians are `230.794` and `222.892 ms`,
with a median paired reduction of 3.6%; quantization medians are `151.872`
and `142.812 ms`, with a paired reduction of 5.3%. The respective total
process-median ranges are `224.082-260.724` and `214.478-236.515 ms`.
This supports retaining the kernel change, not treating the initial wall
regression as a reproducible EPF regression. GPU state moves from
63 C/P8/210 MHz before the initial pairs to 67 C/P3/1282 MHz afterward.
Absolute timings should not be compared across checkpoints.

All 54 CUDA-build tests and 47 CPU-only tests pass. The new standalone
`cuda_epf` test compares 114 sequences with the independent CPU EPF
implementation: 19 image shapes, one/two/three iterations, and default/custom
weights. Shapes cover 1x1, single rows/columns, repeated mirror reflection,
8-pixel sigma boundaries, 32-pixel tile boundaries, and partial tiles.
Every observed reference error is zero, within the test tolerance
`2e-6 + 2e-6 * abs(reference)`. Input, both scratch images, and the sigma
field have distinct padded strides and guarded offsets; guards and read-only
inputs must remain intact. Sigma values include mixed active/bypass blocks
and the exact bypass threshold.

Additional cases cover NaN payloads, positive/negative infinity, all-bypass
and mixed-bypass rows, preservation of existing error bits, and invalid
pass rejection without output writes. Compute Sanitizer memcheck, racecheck,
synccheck, and initcheck each complete this test with zero errors/hazards.
The full suite retains exact-mode differential, allocation, failure-atomicity,
and concurrent-workflow checks.

Parent/candidate codestreams and reported final perceptual scores remain
identical for the small sample at efforts 7/9, odd 1080p/4K at effort 7,
and Flower at efforts 7/9, with final-score collection enabled. The pinned
`djxl` decodes all six at their source dimensions. Hashes remain those
recorded at the preceding checkpoints, and Flower selects all seven
production strategies.

Ignored artifacts under `build-cuda-ninja/profiles` use the `s22_` prefix.
They include parent and exploratory benchmark executables, parent encoder,
4K traces named `parent`, `specialized`, `fast_mirror`, `tiled`, `medium_tile`,
`large_tile`, and `retained`, plus parent/retained 1080p traces. Each trace
has `.nsys-rep` and `.sqlite` files. Paired results are
`s22_paired_{4k,1080p,flower}.json` and `s22_paired_1080p_warmed.json`;
`s22_compare_warmed.py` retains the longer-warmed protocol. Verification
uses `s22_verify.ps1`; sanitizer logs are `s22_epf_*.txt`.

Post-change 1080p batch qualification uses one warmup and three paired
serial/batch samples, with identical outputs at batch sizes 1, 2, and 4:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 226.141 ms | 4.422 | 0.980x |
| Fully resident | 2 | 426.733 ms | 4.687 | 1.131x |
| Fully resident | 4 | 831.689 ms | 4.809 | 1.275x |
| Maximum throughput | 1 | 101.691 ms | 9.834 | 1.038x |
| Maximum throughput | 2 | 147.719 ms | 13.539 | 1.472x |
| Maximum throughput | 4 | 287.812 ms | 13.898 | 1.689x |

These are output-stability and overlap checks, not before/after batch
performance claims. Fully-resident batch-1 is still slower than serial.
Logs are `s22_batch_{resident,maximum}.txt`; the final GPU state is
69 C/P3/1282 MHz.

The retained 4K profile still has `128.497 ms` of DCT, `47.046 ms` of tiled
Butteraugli convolution, `42.634 ms` of Malta response, `26.186 ms` of resident
coefficient encoding, `20.788 ms` of AC gather, and `19.476 ms` of AC cost.
EPF is now `6.253 ms`, rather than a leading hotspot. DCT, AC gather/cost,
coefficient work, other neighborhood filters, host serialization, and transfer
boundaries remain open targets. The resident path is not yet maxed out.

### Follow-up: register-tile packed small DCTs (2026-09-05)

The fresh parent is `49d6707`, including the EPF optimization. Small packed
DCT shapes still consume `48.082 ms` in the odd-4K trace; 16x16 alone costs
`29.386 ms` across forward and inverse calls. Those kernels compute one
output per thread, repeating basis and sample loads across threads and
launching one block per 16x16 transform. The large-DCT register tiling from
the earlier checkpoint does not cover these shapes.

The retained small kernels accumulate eight independent outputs per thread
without changing any output's sequence of multiply-adds. A 256-thread block
now packs 32 DCT8, 16 DCT16x8, 16 DCT8x16, or 8 DCT16x16 transforms, versus
4/2/2/1 previously. Each horizontal basis load feeds eight accumulators;
each vertical intermediate sample feeds eight accumulators. Input and
coefficient-layout conversion use a padded shared tile with coalesced global
I/O. The forward kernel reuses the input tile for output conversion.

Sub-warp transforms receive padding between their intermediate arrays so
neighboring transforms do not map their same-index samples to the same
shared-memory banks. The 16-high shapes stage their vertical basis because
each warp addresses multiple basis entries. The 8-high shapes retain
constant-memory broadcasts: with eight accumulators, each warp now uses one
vertical basis address at a time. Inactive tail transforms initialize shared
cells and participate in the required barriers without accessing global
input/output. The inverse kernel can return inactive lanes after its second
barrier; forward lanes remain through the output-conversion barrier.

The dense basis, coefficient layout, scaling, and arithmetic order remain
unchanged. No factored transform, fast-math option, quality-policy change,
device allocation, transfer, or extra kernel launch is introduced. Large
specialized and generic DCT kernels are unchanged. Shared-memory use is
18,720 bytes for DCT8, 19,008 for DCT16x8, 19,808 for DCT8x16, and 19,072
for DCT16x16. Compiler-reported registers/thread are respectively 48/48,
48/56, 40/40, and 39/40 for forward/inverse, with zero local or stack bytes.
The changed block/resource balance is not a hardware-counter occupancy claim.

Exploratory odd-4K profiles use two warmups, one captured fully-resident
encode, distance 1.2, effort 7, and no final-score collection:

| Variant | Packed DCT | Unchanged large DCT | All kernels |
|---|---:|---:|---:|
| Parent, one accumulator | 48.082 ms | 75.021 ms | 401.359 ms |
| Two accumulators, original global I/O | 36.341 ms | 80.551 ms | 398.331 ms |
| Two accumulators, shared I/O | 35.094 ms | 73.872 ms | 395.022 ms |
| Four accumulators, shared I/O | 30.114 ms | 75.784 ms | 386.447 ms |
| Four accumulators, all vertical bases shared | 34.623 ms | 72.798 ms | 382.522 ms |
| Eight accumulators, sub-warp padding, all vertical bases shared | 26.226 ms | 72.279 ms | 367.628 ms |
| Eight accumulators, selective vertical bases, forced dot-loop unrolling | 30.414 ms | 73.957 ms | 379.043 ms |
| Retained, selective vertical bases, compiler-chosen dot-loop unrolling | 28.220 ms | 76.317 ms | 391.223 ms |

Sharing every vertical basis with four accumulators helps 8x16 inverse but
regresses the aggregate result, so it is not selected. Forced unrolling of
the dot-product loops also loses; only the independent-accumulator loops
are explicitly unrolled in the retained kernel. Individual captures are
noisy, so these trials select candidates rather than prove additive gains
for every sub-change.

The final packed-DCT reduction is `19.862 ms` (41.3%). All DCT falls from
`123.103` to `104.538 ms` (15.1%); unchanged large DCT rises 1.7%, while
non-DCT execution rises from `278.256` to `286.685 ms`. All-kernel time falls
2.5%, not 41.3%. Both traces contain 523 kernel launches, 117,079,320 HtoD
bytes, 103,699,012 DtoH bytes, and 518,400 device-to-device bytes. These
controls limit attribution of total-workflow variation to the changed code.

An initial 65,536-transform event probe shows large relative timing swings
for short kernels. The final probe instead uses 67,108,864 elements for every
shape: 1,048,576 DCT8 transforms, 524,288 8x16/16x8 transforms, and 262,144
16x16 transforms. It performs three warmup dispatches, then seven samples
of three dispatches each. Parent/candidate processes run in
parent-candidate-candidate-parent order. The following ranges span the two
process medians, not individual sample ranges or confidence intervals:

| Shape/direction | Parent median range | Retained median range |
|---|---:|---:|
| 8x8 forward | 7.438-7.767 ms | 2.160-2.165 ms |
| 8x8 inverse | 8.669-8.941 ms | 5.388-5.539 ms |
| 16x8 forward | 9.789-10.209 ms | 5.854-5.919 ms |
| 16x8 inverse | 10.129-10.620 ms | 5.508-5.575 ms |
| 8x16 forward | 8.970-9.142 ms | 6.152-6.260 ms |
| 8x16 inverse | 10.367-10.488 ms | 6.348-6.415 ms |
| 16x16 forward | 10.993-11.242 ms | 7.929-8.264 ms |
| 16x16 inverse | 13.094-16.615 ms | 7.468-7.542 ms |

The unchanged large-shape controls still vary; 32x32 inverse, for example,
has parent process medians of `11.715-12.889 ms` and candidate medians of
`13.485-14.348 ms`. The packed-shape gains survive that unfavorable control,
but absolute probe and workflow times should not be interchanged. The
retained workflow's 16x16 inverse improvement is also smaller than the
isolated-probe improvement (`16.480` to `12.723 ms`).

Public wall measurements use seven alternating independent-process pairs
per workload, three warmups and five measured encodes per process, comparing
each process's median. Fully-resident GPU-only mode, distance 1.2, effort 7,
and skipped final-score collection match the profile boundary. Cohort
medians and median within-pair changes are distinct statistics:

| Workload/stage | Parent median | Candidate median | Median paired change |
|---|---:|---:|---:|
| Odd 4K total | 990.519 ms | 943.740 ms | -2.6% |
| Odd 4K quantization pipeline | 694.657 ms | 653.460 ms | -4.4% |
| Odd 1080p total | 230.694 ms | 248.354 ms | +2.3% |
| Odd 1080p quantization pipeline | 147.469 ms | 144.065 ms | -1.2% |
| Flower total | 46.465 ms | 47.426 ms | +3.2% |
| Flower quantization pipeline | 28.823 ms | 28.824 ms | +0.1% |

All seven 4K quantization pairs and six total-time pairs improve. Parent
and candidate total process-median ranges are `917.113-1045.410 ms` and
`907.796-970.551 ms`. At 1080p, five quantization pairs improve but only
three total-time pairs do; total ranges are `225.565-256.775 ms` and
`221.650-260.331 ms`. Flower total ranges are `45.601-55.813 ms` and
`47.016-53.850 ms`; only two total-time pairs improve. Thus this checkpoint
establishes a 4K wall-time gain, not a universal encoding-time improvement.
The smaller-workload wall regressions are retained in the record.

Additional three-warmup, one-sample traces investigate those smaller
workloads. At 1080p, packed DCT falls from `8.135` to `4.710 ms`; unchanged
large DCT moves from `10.017` to `9.507 ms` and all kernels from `81.087`
to `71.388 ms`. For Flower, packed DCT falls from `1.204` to `0.752 ms`,
unchanged large DCT stays near `1.21 ms`, and all kernels fall from `17.880`
to `17.416 ms`. These fresh traces support retaining the small-shape kernel
change, but do not explain away total-time regressions measured in different
runs. Host/driver work and scheduling variance remain material, especially
when the absolute kernel saving is below one millisecond. GPU state moves
from 65 C/P8/210 MHz before the wall pairs to 75 C/P3/1575 MHz afterward.

All 54 CUDA-build tests and 47 CPU-only tests pass. The standalone CUDA DCT
test now covers 14 batch counts per supported shape: 1/2/3/4, 7/8/9,
15/16/17/19, and 31/32/33. That exercises full and partial blocks around all
packing sizes, including the 32-transform DCT8 block. Across nine shapes,
126 configurations check forward output, round trips, independent inverse
output, and concurrent/repeated submission waits. Random inputs and the
existing impulse, constant, checkerboard, and unequal-ramp fixtures remain.
The double-precision reference tolerances are unchanged: forward
`3e-5 + 3e-4 * abs(reference)` and inverse/round-trip
`5e-4 + 5e-4 * abs(reference)`.

Compute Sanitizer memcheck, racecheck, synccheck, and initcheck each complete
the expanded backend test with zero errors/hazards. As before,
`--report-api-errors no` suppresses reporting of the separate, deliberately
invalid zero-grid launch used to test stale-error consumption; memory and
synchronization checking remain enabled. Existing suite checks retain
exact-mode differentials, allocation, failure-atomicity, and concurrent
fully-resident/maximum-throughput workflow coverage.

Six parent/candidate codestreams, selected strategies, and reported final
perceptual scores remain identical: the small sample at efforts 7/9,
odd 1080p/4K at effort 7, and Flower at efforts 7/9. All collect the final
score, and the pinned `djxl` decodes each candidate at its source dimensions.
Hashes remain those of the preceding checkpoints; Flower still exercises
all seven production strategies.

Post-change 1080p batch qualification uses one warmup and three paired
serial/batch samples and preserves identical output at sizes 1, 2, and 4:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 251.674 ms | 3.973 | 0.958x |
| Fully resident | 2 | 430.302 ms | 4.648 | 1.066x |
| Fully resident | 4 | 746.158 ms | 5.361 | 1.404x |
| Maximum throughput | 1 | 121.540 ms | 8.228 | 1.068x |
| Maximum throughput | 2 | 157.160 ms | 12.726 | 1.524x |
| Maximum throughput | 4 | 285.467 ms | 14.012 | 1.749x |

These are correctness and overlap checks, not before/after batch speedup
claims. Fully-resident batch-1 remains slower than serial. GPU state after
batch qualification is 70 C/P3/1282 MHz.

Ignored study artifacts use `build-cuda-ninja/profiles/s23_` prefixes.
The eight 4K trace names are `parent`, `register2`, `staged2`, `staged4`,
`vertical4`, `vertical8`, `unrolled8`, and `retained`; parent/retained traces
also cover `1080p` and `flower`. Each has `.nsys-rep` and `.sqlite` files.
Saved kernel sources and benchmark executables retain the explored variants.
`s23_dct_probe.cu` is the final equal-element probe, with final comparison
executables `s23_probe_parent_full.exe` and `s23_probe_retained.exe`, and
four `s23_probe_{1,2,3,4}_*.txt` logs. Earlier probe executables predate the
equal-element protocol and should not be used for that comparison.
`s23_compare_warmed.py` produces `s23_paired_{4k,1080p,flower}.json`;
`s23_verify.ps1` and `s23_identity.txt` record codestream/score/decode checks.
Sanitizer logs are `s23_dct_*.txt`; batch logs are
`s23_batch_{resident,maximum}.txt`.

The retained 4K profile still spends `76.317 ms` in large DCT,
`49.196 ms` in tiled Butteraugli convolution, `40.399 ms` in Malta response,
`25.818 ms` in resident coefficient encoding, `20.563 ms` in adjusted
quantization, `19.575 ms` in AC gather, and `19.553 ms` in AC cost. Larger
transforms, coefficient work, remaining filters, host work, and transfer
boundaries remain material targets. This is not a demonstrated performance
ceiling for the resident path.

### Follow-up: fuse AC candidate gathering into forward DCT (2026-09-05)

The parent is `1d72c2b`. AC search materializes each candidate's three
image rectangles in `scratch_a`, then immediately reads that buffer into
forward-DCT shared memory. The subsequent residual kernel overwrites all
active `scratch_a` elements, and cost evaluation consumes reconstructed
residual pixels, not the original packed pixels. The gathered intermediate
is therefore unnecessary. The seven 4K gather launches represent roughly
1.274 GB of packed pixels, or 2.55 GB of logical writes plus rereads. This
is an address-volume estimate, not a measured DRAM-counter result.

The retained forward kernels accept either a contiguous pointer or an
AC-candidate image source. The latter resolves the descriptor and channel
once per participating thread, then reads its strided rectangle directly
into the existing padded shared input tile. Packed/specialized geometry,
basis staging, output layout, and each output's multiply-add sequence are
unchanged. No lower-precision arithmetic or factored transform is introduced.
Candidate layout and validation now live in a shared CUDA-only header.
Invalid descriptors yield NaN without reading pixels; inactive packed
transforms do not fetch descriptors and still participate in barriers.

Both scratch allocations remain necessary for residual coefficients and
inverse reconstruction. Allocation sizes and host/device transfers do not
change. The optimization removes a buffer pass, not the scratch allocation.

One intermediate loader precomputed an offset contiguous pointer. NVCC
raised the ordinary forward 16x8 register count from 48 to 54 and 16x16
from 39 to 40. Retaining the original `input[base + index]` expression
restores all ordinary forward register counts. Fused forward counts are
48/48/40/40 for 8x8/16x8/8x16/16x16, and 40/39/40 for
32x16/16x32/32x32. All have zero local memory and stack; shared-memory
sizes are unchanged. Only fused 16x16 adds one register versus its parent.

#### GPU evidence and controls

Clock variation is large enough to invalidate isolated absolute comparisons.
The first saved parent trace totals 374.932 ms, whereas the first fused trace
totals 198.485 ms; that apparent near-halving is **not** attributed to fusion.
A fresh parent captured adjacent to the fused run provides a much closer
unchanged-kernel control. A later final-source pair provides a second check:

| 4K capture pair | Metric | Parent | Fused |
|---|---|---:|---:|
| Adjacent initial pair | Gather | 12.685 ms | 0 |
| Adjacent initial pair | All forward DCT | 24.365 ms | 25.964 ms |
| Adjacent initial pair | All inverse DCT | 25.181 ms | 25.117 ms |
| Adjacent initial pair | Gather + all DCT | 62.230 ms | 51.082 ms |
| Adjacent initial pair | Other kernels | 146.988 ms | 147.404 ms |
| Adjacent initial pair | All kernels | 209.218 ms | 198.485 ms |
| Final-source pair | Gather + all DCT | 130.173 ms | 103.036 ms |
| Final-source pair | Other kernels | 267.850 ms | 275.755 ms |
| Final-source pair | All kernels | 398.023 ms | 378.791 ms |

The initial pair reduces gather plus DCT by 17.9% and all kernels by 5.1%,
with other kernels within 0.3%. The final pair reduces those totals by
20.8% and 4.8%, respectively, but its other kernels increase 3.0%.
The initial fused trace predates the ordinary-pointer expression cleanup;
the final-source pair includes it. Both pairs eliminate exactly seven
launches, from 523 to 516, and retain 117,079,320 HtoD bytes,
103,699,012 DtoH bytes, and 518,400 D2D bytes.

Final-source diagnostic pairs also reduce gather plus DCT from 18.580 to
13.444 ms at 1080p and 2.431 to 1.879 ms on Flower. However, their other
kernels decrease 23.8% and 5.5%, respectively. Those are not controlled
estimates of an end-to-end gain, nor evidence that fusion explains every
observed DCT reduction.

A separate same-process CUDA-event probe compares the parent's gather body
plus the ordinary forward DCT against fused forward DCT. It uses deterministic
image data, legal synthetic candidate positions, seven alternating-order
pairs after three warmups, and three dispatch repetitions per measurement
(64 for 33-candidate batches). Events measure the device timeline, including
launch gaps, rather than host workflow latency. Counts are 33, 4096, and the
large per-shape counts inferred from the 4K trace. Every coefficient is checked
bitwise before timing. Two independent processes each pass all 21 checks and
favor fusion in every configuration's median paired ratio.

| Shape | Large candidate count | Paired gather + forward time reduction, two runs |
|---|---:|---:|
| 8x8 | 129600 | 51.6-52.8% |
| 16x8 | 113280 | 57.8-58.4% |
| 8x16 | 113400 | 31.9-32.1% |
| 16x16 | 99120 | 31.1-32.1% |
| 32x16 | 24240 | 18.2-19.0% |
| 16x32 | 24300 | 6.2-11.9% |
| 32x32 | 18180 | 29.9-30.2% |

The 33-candidate reductions range from 14.8% to 50.1%, and the 4096-candidate
reductions from 21.8% to 48.9%. Individual event samples still vary substantially;
these synthetic timings isolate the two implementations, not production
search locality. They support retaining fusion for every shape without a
small-batch fallback, but do not establish a Flower workflow latency win.

#### Warmed public workflow

Seven alternating parent/candidate process pairs use three warmups and five
samples per process, effort 7, distance 1.2, fully resident, no final score.
Each process contributes its median. All pairs, including outliers, are kept.
Negative paired changes mean faster:

| Workload | Total median, parent / fused | Median paired total change | Quantization median, parent / fused | Median paired quantization change |
|---|---:|---:|---:|---:|
| Odd 4K | 639.041 / 653.210 ms | -3.7% | 407.479 / 396.414 ms | -3.4% |
| Odd 1080p | 203.917 / 198.267 ms | -1.1% | 115.748 / 111.874 ms | -3.9% |
| Flower | 44.179 / 44.806 ms | +2.6% | 27.770 / 27.605 ms | +0.5% |

Medians of cohorts and medians of paired ratios are different statistics;
the 4K total medians move in the opposite direction to the paired ratio.
4K improves in five of seven total-time pairs and six quantization pairs;
1080p improves in six pairs for both metrics. Flower improves in only one
total-time pair and three quantization pairs. Its measured total-time
regression remains part of the result, not a discarded inconvenient sample.
4K parent/fused total ranges are 618.465-714.695 / 593.595-821.915 ms;
quantization ranges are 405.481-423.054 / 389.839-449.741 ms. GPU state during
the 4K sweep reached 78 C/P0/1395 MHz graphics/6000 MHz memory. Clocks were
not locked; absolute times must not be compared to preceding checkpoints.

#### Qualification and artifacts

The final build passes all 54 CUDA tests and the CPU-only build passes all
47 tests. AC coverage now checks 231 CPU-referenced candidate costs, including
both image corners, twelve prefix lengths crossing packed-DCT boundaries,
and exact per-candidate/output-guard identity. Padded rows, plane gaps,
nonzero offsets, reordered resident planes, and NaN input guards are tested
against contiguous costs. Ten invalid descriptor cases cover coordinate
overflow/footprint bounds, quant norm, entropy multiplier, and CfL factors;
unrelated candidates must remain valid. Resident CfL maps must supersede
non-finite descriptor factors. Resident quant norms, scratch alias rejection,
and independent submission waits remain covered.

Both AC-strategy and general CUDA-backend tests pass memcheck, racecheck,
synccheck, and initcheck with zero errors/hazards. Only the general backend
test uses `--report-api-errors no`, for its intentional invalid zero-grid
launch; memory/synchronization checking remains active. Its 126 ordinary
DCT configurations still check CPU-reference forward, roundtrip, and
independent inverse results.

Six parent/fused codestreams, strategy summaries, and final perceptual scores
remain identical: sample efforts 7/9, odd 1080p/4K effort 7, and Flower efforts
7/9. The pinned decoder reads every candidate at its original dimensions;
hashes remain those of prior checkpoints. Flower covers all seven strategies.

Post-change 1080p batch qualification uses one warmup and three paired
serial/batch samples, preserving identical output at sizes 1/2/4:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 222.608 ms | 4.492 | 0.980x |
| Fully resident | 2 | 382.894 ms | 5.223 | 1.265x |
| Fully resident | 4 | 777.363 ms | 5.146 | 1.333x |
| Maximum throughput | 1 | 112.129 ms | 8.918 | 1.028x |
| Maximum throughput | 2 | 165.620 ms | 12.076 | 1.468x |
| Maximum throughput | 4 | 300.619 ms | 13.306 | 1.719x |

These qualify correctness and overlap, not before/after batch performance.
Fully-resident batch-1 remains slower than serial. Ending GPU state is
70 C/P3/1282 MHz graphics/5500 MHz memory.

Ignored artifacts use `build-cuda-ninja/profiles/s24_` prefixes. Initial
traces are `parent_4k` (unmatched control), `parent_now_4k`, and `fused_4k`;
final traces are `final_{parent,retained}_{4k,1080p,flower}`, each with
`.nsys-rep` and `.sqlite` files. `profile_final.ps1` and
`profile_summary.py` reproduce capture and analysis. `compare_warmed.py`
produces `warmed_{4k,1080p,flower}.json` with raw process output.
`gather_dct_probe.cu` and `gather_probe_{1,2}.txt` retain the event comparison.
Compile the probe against `gjxl_cuda.lib` with NVCC CUDA 11.8,
`-std=c++17 -O3 -arch=sm_86 -Xcompiler=/MD --cudart=shared -I src`,
under the MSVC 14.37 developer environment. `verify.ps1`/`identity.txt`,
`{ac,dct}_{memcheck,racecheck,synccheck,initcheck}.txt`, and
`batch_{fully-resident,maximum-throughput}.txt` retain qualification results.
All artifact names in this paragraph include the `s24_` prefix.

The final 4K trace still spends 78.452 ms in large DCTs, 49.604 ms in tiled
convolution, 43.977 ms in Malta, 27.463 ms in resident coefficient encoding,
22.037 ms in adjusted quantization, and 19.121 ms in AC cost. Cost reduction
and remaining coefficient/filter work, plus host work and transfer boundaries,
remain substantial targets. This checkpoint does not demonstrate that the
fully-resident path is maxed out.

### Follow-up: pack and register-reduce AC costs (2026-09-05)

The parent is `255a706`. After gather fusion, AC cost still reads every
inverse-transformed residual pixel. Its original kernel launches one block
per candidate with 64/128/256/512/1024 threads, stages all three channels in
shared memory, and executes a block-wide barrier after every halving level.
Increasingly few threads participate as the reduction shrinks. At 32x32,
the kernel uses 1024 threads, 12 KiB of shared memory, and eleven barriers
for one cost. Runtime transform dimensions also require general integer
division/remainder for mask addressing.

The retained kernel specializes width/height and packs candidates into
256-thread blocks. Candidate groups have 32 threads through 16x16, 64 for
the 512-pixel shapes, and 128 for 32x32; each lane holds two, four, or eight
pixels for all three channels. The original halving order first combines
register-held values, then uses shared memory between warps where needed,
and finishes with five shuffle levels. Small shapes need no shared memory
or block barrier; 512/1024-pixel shapes use 3 KiB and two/three barriers.
Inactive tail candidates and invalid footprints participate safely without
reading pixels or writing outside the cost range. The allocation, transfer,
launch-count, and arithmetic-precision contracts do not change.

An initial eight-value tile reduces 4K cost execution from 17.074 to
6.089 ms, with other kernels moving from 344.337 to 339.071 ms. A sixteen-value
tile takes 6.601 ms despite lower other-kernel time (330.242 ms). Its large
shapes consume 84-86 registers instead of 40. A same-process comparison also
tests four-value tiles: they do not offer a consistent advantage over eight,
while sixteen is consistently worse on the large shapes. The retained final
register counts are 24/30/30/40/40/40/40 for physical widths/heights
8x8/16x8/8x16/16x16/32x16/16x32/32x32, with no stack or local-memory use.

#### Numerical pitfall caught by the direct comparison

The first packed version passes CPU-tolerance and batch-consistency tests,
but a direct parent-kernel comparison finds occasional one-ULP cost differences
(first observed at 16x8 with 4096 candidates). Reverting constant-count
normalization to the parent's runtime-count expression does not fix them.
PTX identifies the difference in channel combination, not the halving tree:
the parent contracts X weighting with the following Y addition into one
FMA, whereas register-resident channel totals cause NVCC to emit a multiply
and a separately rounded addition across the branch. The same source-level
summation order alone is insufficient to preserve the parent's result.

The final channel combination explicitly uses the parent's FMA sequence,
including weighted X entropy plus Y magnitude. A fresh direct probe compares
the original cost body with four/eight/sixteen-value variants across all seven
shapes and counts 33, 4096, and the production-size counts from the 4K trace.
All 63 variant/configuration comparisons pass bitwise in each of two
independent processes, including output tail guards. The probe uses strided
mask input, deterministic residuals and rates, and varying positive quant
norms/entropy multipliers. Its timing phase gives every implementation the
same warmup and repetition counts; each has independent in/out cost storage.
During timing, each cost becomes the following dispatch's norm. That feedback
is identical across implementations and keeps the timing loop focused on
the cost kernel; it does not model an AQ iteration.

Seven within-process forward/reverse-order sweeps follow three warmups,
using five repetitions per measurement (64 for 33-candidate batches).
Events measure the device timeline, including dispatch gaps. The retained
eight-value variant wins every configuration's median paired comparison in
both processes. Large-count reductions range from 62.9% to 83.4%, while
4096-candidate reductions range from 34.6% to 75.1%. The small 33-candidate
results range from 2.1% to 40.5% and are more sensitive to dispatch/clock
variation. Absolute timings and variant-order differences remain noisy;
these are isolated cost comparisons, not complete workflow speedups.

#### Warmed public workflow

Seven alternating parent/candidate process pairs use three warmups and five
samples per process, effort 7, distance 1.2, fully resident, no final score.
Each process contributes its median; all pairs and outliers are retained.
Negative paired changes mean faster:

| Workload | Total median, parent / packed | Median paired total change | Quantization median, parent / packed | Median paired quantization change |
|---|---:|---:|---:|---:|
| Odd 4K | 876.867 / 856.186 ms | +1.5% | 611.071 / 596.081 ms | -0.9% |
| Odd 1080p | 210.995 / 213.637 ms | +0.6% | 131.949 / 131.110 ms | -2.0% |
| Flower | 54.761 / 46.620 ms | -5.9% | 32.898 / 28.287 ms | -5.1% |

There is no demonstrated total-latency win on the large synthetic workloads:
only two of seven 4K total-time pairs and three 1080p pairs improve. Their
quantization stages improve in six and five pairs, respectively. Flower
improves in five total-time pairs and all seven quantization pairs, but its
parent total range is particularly broad (43.597-78.558 ms versus
44.508-55.674 ms). Its larger percentage must not be attributed entirely to
the cost kernel without an unchanged-work control.

The 4K total cohorts range from 826.371-916.619 / 839.592-948.853 ms;
quantization ranges are 579.212-625.331 / 586.411-614.051 ms. The cohort
medians and median paired ratios are different statistics: the former move
in the opposite direction to the 4K total paired ratio. During the 4K sweep
a device-state sample reads 71 C/P3/667 MHz graphics/5500 MHz memory.
Clocks are not locked, and preceding checkpoints' absolute times are not a
valid control. The isolated kernel improvement is retained, but this is not
claimed as an established 4K end-to-end speedup.

#### Final-source profiles and qualification

Adjacent parent/final-source traces use three warmups and one profiled sample:

| Workload | Cost, parent / packed | Other kernels, parent / packed | All kernels, parent / packed |
|---|---:|---:|---:|
| Odd 4K | 19.609 / 6.220 ms | 391.858 / 384.748 ms | 411.467 / 390.968 ms |
| Odd 1080p | 2.955 / 1.571 ms | 67.746 / 71.878 ms | 70.700 / 73.449 ms |
| Flower | 0.440 / 0.268 ms | 16.542 / 16.545 ms | 16.982 / 16.813 ms |

Cost execution decreases 68.3%, 46.8%, and 39.1%, respectively. At 4K, all
kernels decrease 5.0%, but other kernels also decrease 1.8%; the full change
is not attributed solely to cost packing. At 1080p, other kernels increase
6.1% and total kernel time increases 3.9% despite the cheaper cost stage.
Flower's other kernels are nearly level, with total kernel time decreasing
1.0%, much less than its paired workflow change. These controls reinforce
the distinction between a kernel gain and a universal latency gain.

The seven 4K cost launches now issue 78,150 blocks instead of 522,120.
Both complete traces still have 516 launches and transfer 117,079,320 bytes
HtoD, 103,699,012 bytes DtoH, and 518,400 bytes D2D. 1080p remains at 501
launches and Flower at 516. No scratch allocation or residency policy changes.

The final CUDA build passes all 54 tests; the CPU-only build passes all 47.
The AC fixture now checks independent padded mask rows and nonzero mask
offsets in legacy and resident modes, exact cost equality with contiguous
inputs, and untouched input/guard storage. Eighty-four bad-mask batches
cover zero, negative, NaN, and infinity values at the first, last, and an
interior pixel across all seven strategies. Each candidate must return NaN
iff it covers that pixel, otherwise retain its original cost. Existing
231 CPU-referenced costs, twelve prefix lengths, invalid descriptors,
resident quant norms/CfL, and independent-submission checks remain active.

After the FMA correction, the expanded AC test passes memcheck, racecheck,
synccheck, and initcheck with zero errors or hazards, without suppressing
API errors. Six parent/final codestreams, selected-strategy summaries, and
final perceptual scores remain identical: sample efforts 7/9, odd 1080p/4K
effort 7, and Flower efforts 7/9. The pinned decoder reads every output at
its original dimensions. Hashes remain those of the preceding checkpoints.

Post-change batch qualification at 1080p uses one warmup and three paired
serial/batch samples and preserves identical output at sizes 1/2/4:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 254.816 ms | 3.924 | 0.993x |
| Fully resident | 2 | 403.184 ms | 4.961 | 1.204x |
| Fully resident | 4 | 818.508 ms | 4.887 | 1.163x |
| Maximum throughput | 1 | 110.644 ms | 9.038 | 1.025x |
| Maximum throughput | 2 | 155.350 ms | 12.874 | 1.459x |
| Maximum throughput | 4 | 343.143 ms | 11.657 | 1.673x |

These qualify correctness and overlap, not before/after batch performance.
Ending GPU state is 74 C/P3/1282 MHz graphics/5500 MHz memory.

Ignored artifacts under `build-cuda-ninja/profiles/` use the `s25_` prefix:
`parent_4k`, `packed8_4k`, and `packed16_4k` retain the initial traces;
`final_{parent,retained}_{4k,1080p,flower}` retain the final traces, all as
`.nsys-rep`/`.sqlite`. `packed{8,16}_source.cu` are **pre-FMA-correction**
experiments, not the retained source. `cost_probe.cu` is the corrected
four/eight/sixteen-value probe, with `cost_probe_fma_{1,2}.txt` results.
`cost_probe_constant_count.txt`, `cost_probe_runtime_count.txt`, and
`cost_probe_before_fma.ptx` retain the failed numerical investigation.
The probe compiles with NVCC 11.8 under MSVC 14.37 using
`-std=c++17 -O3 -arch=sm_86 -Xcompiler=/MD --cudart=shared -I src`.
`compare_warmed.py` produces `warmed_{4k,1080p,flower}.json` with raw output;
`verify.ps1`/`identity.txt`, `ac_final_{memcheck,racecheck,synccheck,initcheck}.txt`,
and `batch_{fully-resident,maximum-throughput}.txt` retain final qualification.
`qualify.ps1` runs the benchmarks and GPU checks sequentially;
`profile_final.ps1`/`profile_summary.py` reproduce capture and extraction.
Every artifact name in this paragraph includes the `s25_` prefix.

The cost stage still logically reads 1.274 GB of residual pixels plus
0.425 GB of masks per 4K encode; overlapping mask reads can hit cache, so
this is not a DRAM-counter measurement or proof of a bandwidth ceiling.
Further packing has diminishing returns. A next dataflow experiment should
reduce weighted residual loss directly from inverse-DCT outputs and retain
only per-channel sums, removing the full residual-pixel buffer round trip.
It must measure any added inverse-kernel register/barrier cost and preserve
the now-explicit final FMA behavior. The final trace also retains 82.046 ms
of large DCT, 26.048 ms of packed DCT, 55.177 ms of convolution, 48.878 ms
of Malta, 29.036 ms of coefficient encoding, and 23.318 ms of adjusted
quantization. Host work and transfer boundaries remain important given the
mixed wall-time results. The fully-resident path is not demonstrated maxed out.

### Follow-up: fuse AC inverse transforms with residual loss (2026-09-05)

Parent: `85d7f42` (packed AC costs). AC search only needs a weighted
eighth-power loss sum for each candidate/channel, but the parent writes
every inverse residual pixel to global scratch and then reads it back in
the cost kernel. Packing the cost kernel did not eliminate that dataflow.

`AcStrategyDctLossOutput` now consumes the inverse transform's output
registers. The same templated inverse kernels retain their ordinary pointer
output for reconstruction and transform callers. The AC sink preserves the
inverse scaling's FP32 rounding explicitly, applies the mask/channel offset,
squares three times, and reduces with the original halving tree. It writes
only three floats per candidate. `FinalizeCostKernel` combines those sums
with the existing channel rates and cached quant norm, retaining the explicit
final FMAs introduced in the parent.

Small shapes reduce eight values per lane in registers and then use exact
8/16/32-lane shuffle masks. Inactive tail transforms do not participate in
those groups. Large shapes reuse the dead horizontal-basis shared storage:
after one barrier, the first warp loads eight warp-partials per lane and
evaluates the 128/64/32 halving tree in registers, followed by warp shuffles.
The horizontal-basis reads have ended at an earlier block barrier; the
remaining inverse work reads different shared regions. No extra shared
allocation is required. Invalid mask values or footprints still yield NaN.

Both large scratch buffers remain necessary for the forward/residual stages;
this is a traffic reduction, not an allocation-size reduction. At padded 4K,
522,120 candidates formerly wrote and reread 2,547,671,040 bytes of residual
pixels. The new loss-sum round trip is 12,530,880 bytes. Unlike the old
three-channel cost loop, separate inverse channels logically reread the
mask, adding 849,223,680 bytes of mask loads. The net logical reduction is
therefore 1,685,916,480 bytes, not the entire removed residual-buffer traffic.
Overlapping/coalesced mask accesses can hit cache; these are algorithmic byte
counts, not measured DRAM traffic or a bandwidth-ceiling claim.

Register counts remain 48/56/40/40 for inverse shapes 8x8/16x8/8x16/16x16 and
40/40/40 for 32x16/16x32/32x32, identical to the ordinary inverse kernels.
There are no stack/local spills, and dynamic shared-memory sizes are unchanged.
The finalizer uses 22 registers and no shared memory. Kernel launches and
host/device transfers are not removed: seven small finalizer launches replace
the seven old cost launches.

#### Experiments and numerical checks

The first fused implementation used four additional large-transform barriers
for shared-memory halving. Its fresh 4K trace reduced the AC inverse/cost
stages from 48.882 to 41.081 ms; other kernels also improved 3.0%, so the
full 4.8% kernel-total reduction was not wholly attributable to fusion.
A later trace of that version measured 52.631 to 44.458 ms for the targeted
stages (-15.5%), with other kernels nearly unchanged (+0.5%). Its paired
wall-time results were mixed: quantization improved 1.7%/1.8%/2.4% at
4K/1080p/Flower, while total time changed +2.0%/-1.5%/-1.0%.

The second implementation moves the cross-warp halving into first-warp
registers, removing three barriers. Disassembly of the normal builds confirms
identical instructions for all four small inverse-loss shapes; large shapes
have three total barriers instead of six, with unchanged instruction counts
and register footprints. This is an arithmetic-order-preserving reduction,
not a reassociated warp-first sum.

Two isolated processes compare the parent inverse plus packed cost, the first
fusion, and the one-barrier reduction at seven shapes and candidate counts
33, 4,096, and approximately the production 4K counts. All final costs and
output guards match bitwise, both initially and after the timed iterations.
The two-variant probe also passes all 21 configurations in both processes.
Timing uses seven alternating event-timed pairs, three warmups, and 64 repeats
for count 33 or five for larger counts. As in the preceding cost probe, each
timed result becomes the next quant norm equally for each implementation;
these iterations measure the kernel sequence, not an AQ iteration.

The three-way probe favors the one-barrier variant on large production-sized
batches, but even unchanged small-shape controls move substantially. At 4,096
candidates the small controls are much steadier and the large-shape benefit
is modest. Do not interpret the largest isolated ratios as causal speedups;
normal-build profiles and public-workflow measurements are the primary check.

`cuda_backend_test` now directly compares fused channel sums with a materialized
ordinary CUDA inverse followed by an independent host FP32 halving tree.
Seven shapes, 12 candidate counts (1/2/3/5/7/8/9/10/11/16/17/33), and five mask
variants exercise 420 fused batches and 12,810 channel-sum comparisons. The
inputs include noise, zeros, and impulses; mask rows have padding and a
nonzero offset; output ranges have prefix/suffix guards. Image corners,
overlapping tiles, out-of-bounds/maximum coordinates, and zero/negative/NaN/
infinite masks are covered. Finite results require bitwise equality, while
invalid results require NaN. The existing 126 ordinary DCT configurations
and expanded AC-cost invalid-input/resident-view tests remain active.

#### Retained-build measurements

Final one-barrier build versus `85d7f42`, effort 7, distance 1.2,
fully resident, no optional final-score pass. Each Nsight range captures one
encode after three warmups. Times are GPU execution milliseconds; AC inverse
and cost are measured together so that moving work into the inverse is not
misreported as a standalone cost-kernel speedup.

| Workload | AC inverse/cost parent | Final | Change | Other kernels parent/final | Total kernels parent/final |
|---|---:|---:|---:|---:|---:|
| Odd padded 4K | 49.465 | 44.159 | -10.7% | 331.312 / 332.473 | 380.777 / 376.631 |
| Odd padded 1080p | 7.177 | 5.991 | -16.5% | 63.145 / 64.869 | 70.322 / 70.860 |
| Flower | 1.086 | 0.860 | -20.8% | 15.705 / 15.720 | 16.792 / 16.580 |

The 4K total improves 1.1% with other kernels within 0.4%; Flower improves
1.3% with its other kernels within 0.1%. At 1080p, a 2.7% increase in other
kernel time exceeds the targeted saving. Launch counts remain 516/501/516
for 4K/1080p/Flower. The 4K trace retains 117,079,320 host-to-device bytes
in 31 copies, 103,699,012 device-to-host bytes in 19 copies, and 518,400
device-to-device bytes in one copy. None of that transfer traffic is removed.
The final trace still contains 79.528 ms of large DCT (including fused loss),
21.222 ms of packed DCT, 68.752 ms of convolution, 62.571 ms of Malta,
30.148 ms of resident coefficient encoding, and 23.699 ms of adjusted
quantization.

Six of seven shape-specific AC inverse/cost pairs improve in the final 4K
trace; 16x32 increases from 13.628 to 14.131 ms (+3.7%). Thus the aggregate
gain is not a demonstrated universal per-shape win. The earlier fused trace
and isolated comparisons do not show a consistent 16x32 regression.

The original `s26_v2_warmed_4k` run overlapped an unexpectedly long CPU
disassembly check. Its complete log is retained, but it is not used for
performance conclusions; `s26_v2_clean_warmed_4k` repeats the entire experiment
without concurrent profiling, disassembly, builds, or tests. The uncontended
1080p/Flower runs remain in `s26_v2_warmed_{1080p,flower}`. Every reported
wall comparison uses seven alternating independent-process pairs, three
warmups and five samples per process, and the median of each process's samples.
All outliers within these uncontended experiments are retained.

| Workload | Total median parent/final | Paired total change | Quantization median parent/final | Paired quantization change | Faster total/quantization pairs |
|---|---:|---:|---:|---:|---:|
| Odd padded 4K | 1065.965 / 1027.257 ms | -2.6% | 669.518 / 646.048 ms | -3.5% | 4/7, 7/7 |
| Odd padded 1080p | 230.436 / 223.762 ms | -3.2% | 138.728 / 133.548 ms | -3.1% | 6/7, 5/7 |
| Flower | 50.419 / 48.277 ms | -0.7% | 29.961 / 29.195 ms | -2.6% | 4/7, 5/7 |

The paired changes are medians of paired ratios, not ratios of the cohort
medians. Total-time ranges are 966.404-1153.517 / 940.211-1185.483 ms at 4K,
224.418-241.990 / 217.084-246.075 ms at 1080p, and 46.605-63.119 /
47.310-62.713 ms for Flower (parent/final). The 4K quantization improvement
is consistent across all pairs, but total-time wins are not universal and
the observed benefit is modest. Ending clean-run GPU state is
77 C/P3/1282 MHz graphics/5500 MHz memory. The uncontended 4K repeat is
reproduced by `s26_v2_compare_warmed.py` with the saved parent benchmark,
the final benchmark, workload `padded_4k`, and output
`s26_v2_clean_warmed_4k.json`.

#### Validation and remaining work

The final build passes all 54 CUDA tests; the CPU-only build passes all 47.
Both the AC-strategy test and expanded CUDA backend test pass memcheck,
racecheck, synccheck, and initcheck with zero errors/hazards. Only the backend
test uses `--report-api-errors no`, because its pre-existing launch-error test
deliberately submits a zero-grid launch. No AC-test API errors are suppressed.
Six final codestreams remain identical to the parent, as do selected strategies
and final perceptual scores: sample efforts 7/9, odd 1080p/4K effort 7, and
Flower efforts 7/9. All decode with the pinned decoder at the original
dimensions; hashes remain those recorded in earlier checkpoints.

Final 1080p serial/batch qualification uses one warmup and three alternating
samples per size. All batch outputs remain identical to serial outputs.

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 268.690 ms | 3.722 | 0.980x |
| Fully resident | 2 | 532.634 ms | 3.755 | 0.983x |
| Fully resident | 4 | 975.005 ms | 4.103 | 1.525x |
| Maximum throughput | 1 | 243.101 ms | 4.114 | 1.023x |
| Maximum throughput | 2 | 369.940 ms | 5.406 | 1.379x |
| Maximum throughput | 4 | 447.753 ms | 8.933 | 2.048x |

These qualify correctness and overlap, not before/after batch performance.
Timing variance remains substantial, including no demonstrated two-image
fully-resident throughput benefit in this run. Ending batch GPU state is
75 C/P5/765 MHz graphics/810 MHz memory.

The preceding residual-coefficient buffer is still materialized between AC
quantization/rate evaluation and the inverse transform. A next dataflow
experiment can quantize residual coefficients while loading inverse-DCT shared
memory, reduce channel rates before the transform, and write loss summaries
to the opposite scratch buffer. That could remove another seven launches and
the residual-coefficient round trip, but requires explicit checks of reduction
order, buffer aliasing, invalid descriptors, and register/shared-memory costs.
Large direct-matrix DCTs, filtering, host work, and transfers also remain
material targets. This checkpoint does not demonstrate a maxed-out encoder.

Ignored artifacts use `build-cuda-ninja/profiles/s26_`:
`parent_4k`/`fused_4k` are the initial traces; `final_{parent,retained}_*`,
`warmed_*`, `ac_final_*`, `backend_final_*`, `identity.txt`, and `batch_*`
describe the first four-barrier fusion, not the final reduction.
`inverse_loss_v1_source.cu` and `fused_v1_{benchmark,encode}.exe` preserve that
version. `inverse_loss_probe.cu`/`inverse_loss_probe_{1,2}.txt` compare it with
the parent. `inverse_loss_variant.cu`/`inverse_loss_variants_probe.cu` and
`inverse_loss_variants_probe_{1,2}.txt` preserve the three-way experiment.
The probes compile with NVCC 11.8/MSVC 14.37 using
`-std=c++17 -O3 -arch=sm_86 -Xcompiler=/MD --cudart=shared -I src` and link
`gjxl_cuda.lib`; the three-way probe also compiles the variant source.
Final one-barrier artifacts use `s26_v2_`: `qualify.ps1` runs warmed pairs,
eight sanitizer passes, encode/decode identity, final profiles, and both batch
modes sequentially. `s26_profile_summary.py` extracts the first seven AC
inverse/cost pairs separately from later ordinary reconstruction inverses.

### Follow-up: fuse AC residual evaluation into inverse loading (2026-09-05)

Parent: `b46bea9` (inverse/loss fusion). AC search still materialized the
quantized residual coefficients between its standalone residual/rate kernel
and inverse transform. The fresh parent 4K trace spends 12.070 ms in that
residual kernel and 53.862 ms across residual, inverse, and final cost stages.
Removing only the residual launch would miscount any work moved into inverse
loading, so the comparisons below measure the complete affected sequence.

`AcStrategyResidualDctSource` now computes color decorrelation, quantization,
round-away-from-zero, and residual dequantization while loading the inverse
shared tile. It reduces magnitude and nonzero rates before the DCT starts.
The inverse transform and the preceding checkpoint's weighted loss sink are
shared with the ordinary pointer-input path; that path retains its original
load loop and transform arithmetic. Device candidate validation and the
host/device CfL rules are unchanged. The device channel-rate layout and CfL
helper move to the shared CUDA AC header.

Small transforms keep their rate arrays in registers and use exact subwarp
shuffle masks. Large transforms borrow the not-yet-populated intermediate
tile for float magnitude and integer count partials. One extra barrier makes
those writes visible; the first warp evaluates the original high-stride
halving tree and shuffles the final levels. The existing input-tile barrier
ensures rate reads have ended before horizontal DCT output can overwrite that
storage. No extra dynamic shared-memory allocation is required.

The batch now runs quant-norm preparation, forward DCT, fused residual/inverse/
loss, and final cost. Forward coefficients stay in scratch B; compact loss
sums go to scratch A. Writing them to B would alias coefficients that other
inverse blocks might still read. Existing public range/overlap validation is
unchanged. Seven standalone residual launches disappear, along with
2,547,671,040 bytes of full residual-coefficient writes/reads at padded 4K.
Scalar metadata loads are now per inverse lane instead of one leader per
old residual group, with same-address coalescing/cache reuse; the removed
buffer-byte count is not a net DRAM-traffic or bandwidth-ceiling measurement.

Resource usage for fused inverse input/output, in width-by-height order:

| Shape | Parent registers | Fused registers | Stack/local spill bytes |
|---|---:|---:|---:|
| 8x8 | 48 | 48 | 0 / 0 |
| 16x8 | 56 | 52 | 0 / 0 |
| 8x16 | 40 | 47 | 0 / 0 |
| 16x16 | 40 | 47 | 0 / 0 |
| 32x16 | 40 | 40 | 0 / 0 |
| 16x32 | 40 | 39 | 0 / 0 |
| 32x32 | 40 | 39 | 0 / 0 |

The first fused 4K trace measures 53.862 to 40.989 ms for the affected stages
(-23.9%), with other kernels essentially unchanged (283.662 to 283.832 ms).
The kernel total falls from 337.524 to 324.822 ms (-3.8%), and launches fall
from 516 to 509. This is an initial trace, not the final wall-time result.

#### Direct numerical checks

`CheckResidualInverseLoss` extends the CUDA backend test with an independent
host FP32 residual/rate oracle. The host forms residual coefficients, then
the existing pointer-input inverse/loss path supplies reference loss sums;
the new fused path receives the original coefficients. Magnitude sums and
nonzero counts are checked independently of the final scalar cost.

Seven shapes, candidate counts 1/8/11/16/32/33, and both host/device CfL modes
cover 84 fused batches and 4,242 channel comparisons each for loss, magnitude,
and nonzero count. Inputs include random coefficients, exact positive/negative
half-way values, signed zero, varying matrices/norms, invalid footprints,
strided/offset masks and device CfL planes, and prefix/suffix output guards.
Device CfL must supersede deliberately NaN host factors. Finite float results
must match bitwise, invalid results must be NaN, and counts must match exactly.
The earlier 420 direct inverse/loss batches, 126 ordinary DCT configurations,
and expanded AC-cost validation tests also remain active.

The initial oracle used separately rounded multiply/add for the device B
factor and failed an exact loss comparison. The existing CUDA factor expression
contracts that operation; using `std::fma` in the host oracle fixes the test
without changing the kernel formula. All direct comparisons then pass.

#### Paired public-workflow timing

Seven alternating parent/final process pairs, three warmups and five measured
samples per process, fully resident, effort 7/distance 1.2, no final-score
diagnostic. Each process contributes its sample median; all outliers are
retained. No GPU tests, builds, or disassembly checks overlap these runs.

| Workload | Total median parent/final | Paired total change | Quantization median parent/final | Paired quantization change | Faster total/quantization pairs |
|---|---:|---:|---:|---:|---:|
| Odd padded 4K | 863.886 / 810.982 ms | -6.1% | 582.479 / 557.841 ms | -4.2% | 7/7, 7/7 |
| Odd padded 1080p | 212.853 / 207.994 ms | -0.9% | 130.792 / 127.351 ms | -3.9% | 6/7, 7/7 |
| Flower | 44.449 / 44.299 ms | +2.0% | 27.030 / 26.719 ms | -0.3% | 2/7, 4/7 |

Paired percentages are medians of per-pair ratios, not ratios of cohort medians.
Total-time ranges are 818.522-893.568 / 793.445-833.494 ms at 4K,
206.170-227.782 / 202.230-225.643 ms at 1080p, and 43.024-45.073 /
43.890-54.217 ms for Flower (parent/final). The large-image quantization
improvement is consistent across every pair, but the natural-image total
regression remains in the report. This is not a universal latency win.

#### Final profiles and qualification

Each final profile captures one effort-7, distance-1.2 fully-resident encode
after three warmups, without the optional final-score pass. GPU times are
milliseconds and include the entire affected residual/inverse/cost sequence.

| Workload | AC stages parent/final | Change | Other kernels parent/final | Kernel total parent/final | Launches parent/final |
|---|---:|---:|---:|---:|---:|
| Odd padded 4K | 56.439 / 43.304 | -23.3% | 319.454 / 297.916 | 375.893 / 341.220 | 516 / 509 |
| Odd padded 1080p | 8.829 / 7.054 | -20.1% | 59.129 / 56.244 | 67.958 / 63.298 | 501 / 494 |
| Flower | 1.276 / 1.025 | -19.6% | 15.232 / 15.285 | 16.508 / 16.310 | 516 / 509 |

All seven 4K shape pairs improve in the final trace. Its 9.2% kernel-total
reduction is not solely caused by this patch: other kernels also run 6.7%
faster. The initial 4K trace with nearly stable other kernels is a better
controlled observation of the aggregate saving. Other-kernel time also
improves 4.9% in the final 1080p trace, but is nearly stable for Flower (+0.3%).
Host/device transfer payloads and copy counts do not change: 4K still has
117,079,320 HtoD bytes/31 copies, 103,699,012 DtoH bytes/19 copies, and
518,400 D2D bytes/one copy.

The final build passes all 54 CUDA tests and the CPU-only build passes all 47.
Both the AC-strategy and expanded CUDA backend tests pass memcheck, racecheck,
synccheck, and initcheck with zero errors/hazards. Only the backend test
suppresses API-error reporting for its deliberately invalid zero-grid launch;
the AC test uses no API-error suppression. Six parent/final codestreams,
selected-strategy summaries, and final perceptual scores match exactly:
sample efforts 7/9, odd 1080p/4K effort 7, and Flower efforts 7/9. The pinned
decoder reads all six outputs at their original dimensions. Their hashes
remain those of the preceding checkpoints.

Final 1080p batch qualification uses one warmup and three alternating paired
serial/batch samples per size, with identical output required throughout:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 223.724 ms | 4.470 | 1.155x |
| Fully resident | 2 | 416.248 ms | 4.805 | 1.177x |
| Fully resident | 4 | 890.944 ms | 4.490 | 1.382x |
| Maximum throughput | 1 | 158.049 ms | 6.327 | 1.005x |
| Maximum throughput | 2 | 303.406 ms | 6.592 | 1.399x |
| Maximum throughput | 4 | 400.195 ms | 9.995 | 1.667x |

These are correctness/overlap qualifications, not before/after batch claims.
Ending GPU state is 74 C/P5/697 MHz graphics/810 MHz memory. Power-state and
host timing variation remain important; no clock or power policy was changed.

Ignored artifacts under `build-cuda-ninja/profiles/` use the `s27_` prefix:
`parent_4k`/`fused_4k` hold the initial `.nsys-rep`/`.sqlite` traces;
`final_{parent,retained}_{4k,1080p,flower}` hold final traces.
`compare_warmed.py` and `warmed_{4k,1080p,flower}.{json,txt}` retain paired
results and raw benchmark stdout. `verify.ps1`/`identity.txt`,
`{ac,backend}_final_{memcheck,racecheck,synccheck,initcheck}.txt`, and
`batch_{fully-resident,maximum-throughput}.txt` retain qualification results.
`qualify.ps1` runs GPU benchmarks and checks sequentially;
`profile_final.ps1`/`profile_summary.py` capture and extract measurements.
The profile extractor counts the first seven complete AC groups, recognizing
that parent groups have a residual launch and fused groups do not.

#### Remaining opportunities

The shared search planner still reserves two `maximum_packed_bytes` ranges,
and public CUDA batch validation still requires those full ranges. After this
fusion, A only needs three floats per candidate. At padded 4K, its maximum
could fall from 304,496,640 to 1,555,200 bytes, saving about 303 MB per search
instance before alignment. That reduction is not implemented here: it needs
an explicit scratch-size contract/query, corresponding arena planning and
range/alias validation, and coverage of the other backends and batch lifetimes.

The final 4K trace still contains 70.292 ms of large DCT (now including residual
and loss work), 23.203 ms of packed DCT, 64.880 ms of convolution, 54.398 ms of
Malta, 29.158 ms of coefficient encoding, and 23.708 ms of adjusted quantization.
The transforms still use direct basis-matrix dot products; a factorized-DCT
experiment would need numerical and perceptual qualification. Host work and
transfer boundaries also remain material. The fully-resident path is not
demonstrated maxed out.

### Follow-up: compact CUDA AC-search scratch allocation (2026-09-05)

Parent: `5ba4d86` (residual/inverse/loss fusion). This checkpoint completes the
allocation follow-up described above; it changes no device arithmetic or kernel
launches.

#### Allocation contract and implementation

The fused inverse evaluator only writes one loss sum per candidate channel to
scratch A. The shared frontend nevertheless reserved two complete packed
coefficient ranges, and CUDA validation enforced both obsolete ranges, including
in its alias checks. Reducing only the frontend allocation would therefore be
rejected (or leave misleading overlap checks when suballocating a larger arena).

`GpuAcStrategyEvaluation::GetAcStrategyScratchRequirements` now supplies the
per-strategy/per-count contract without allocating or submitting work. Its
checked conservative default retains the original A/B/rate sizes for Metal and
other implementations. CUDA overrides A to `candidate_count * 3 * sizeof(float)`;
B and rate scratch are unchanged. Unknown strategies, null outputs, overflow,
and unsupported CUDA strategies are rejected without changing the output.
Sizing alone does not certify device indexing or launch limits; submission
validation still checks those limits and buffer ownership.

The shared planner takes three independent maxima across stages and uses them
for both capacity planning and suballocation. CUDA uses the same query for
minimum buffer ranges and overlap validation. Old full-size allocations remain
valid, but adjacent compact suballocations no longer appear to overlap. The
existing ordered submission, scratch reuse, completion waits, and ownership
lifetimes are unchanged. Search statistics expose the three logical scratch
ranges and the actual retained owning arena capacity, excluding external
resident inputs and legacy input staging. Prepared searches retain capacity
when their logical geometry shrinks.

At padded 4K (3840x2160), the maximum coefficient range comes from the 99,120
16x16 candidates, while maximum loss storage comes from 129,600 DCT8 candidates:

| Range | Parent bytes | Compact bytes |
|---|---:|---:|
| Scratch A | 304,496,640 | 1,555,200 |
| Scratch B | 304,496,640 | 304,496,640 |
| Rate scratch | 3,110,400 | 3,110,400 |
| Complete search arena, including tables/costs/alignment | 626,787,328 | 323,845,888 |

The complete arena saves 302,941,440 bytes (288.9 MiB), or 48.3%. This is an
allocation-capacity reduction, not another reduction in device memory traffic:
the preceding fusion had already eliminated the unused intermediate writes.
The rest of the encoder's resident state is outside these arena totals.

#### Contract and regression coverage

CPU-only tests exercise the conservative default across all strategy metadata,
zero and ordinary counts, the largest non-overflowing count and its successor,
unknown strategies, null outputs, and missing capabilities. CUDA tests check
compact query results for all seven supported shapes and allocation/submission-
free error handling. Candidate tests now allocate exact-size A buffers, retain
the existing strided/resident/tail coverage, and also compare conservative A
buffers against compact results exactly.

For each supported shape, counts 1/8/11/16/32/33 exercise adjacent A/B/rate/cost
ranges with nonzero offsets, outer guards, and repeated ordered submissions.
Every output range is tested one float short and misaligned; all six output
pairs are tested for partial overlap, with input alias and foreign-backend A
rejected as well. Invalid range requests must not commit a submission.

The existing CPU/GPU search-parity test is now also built for CUDA. It covers
small and non-full-tile geometries, dependency-minimal candidate counts, atomic
failure, repeated prepared searches, and shrink/restore reuse. It reconstructs
the arena capacity independently from the candidate counts and checks it against
the actual retained capacity. Its optional `--memory-4k` mode runs the same
allocation and reuse checks at padded 4K. All 55 CUDA-enabled and 47 CPU-only
CTest tests pass, including install consumers. Metal runtime testing is not
available on this Windows host; its conservative sizing default is exercised
by the portable contract test.

#### Warmed public-workflow timing

Seven alternating parent/candidate process pairs each ran three warmups and
five measured fully-resident samples, without optional final-score diagnostics.
No builds, other GPU work, or heavy disassembly overlapped these runs. All
samples and outliers are retained. Medians below are medians of the seven
process medians; paired changes are independently computed from each pair's
ratio (not from the ratio of the two cohort medians).

| Input | Parent total ms | Compact total ms | Median paired total change | Parent quantization ms | Compact quantization ms | Median paired quantization change |
|---|---:|---:|---:|---:|---:|---:|
| Padded 4K | 837.477 | 811.058 | -3.2% | 570.773 | 562.826 | -1.7% |
| Padded 1080p | 209.757 | 211.064 | -0.2% | 129.130 | 126.483 | -1.7% |
| Flower, 510x532 | 47.045 | 46.153 | -5.0% | 29.025 | 27.848 | -3.0% |

Compact allocation wins 7/7 total and quantization pairs at 4K, 4/7 total and
6/7 quantization pairs at 1080p, and 6/7 total and quantization pairs for Flower.
Total-time ranges are 807.969–906.389 versus 778.103–842.434 ms at 4K,
204.266–234.241 versus 200.953–226.513 ms at 1080p, and 45.482–75.145 versus
42.603–49.127 ms for Flower. In particular, the long Flower parent process is
not discarded. These are observed wall-time changes on a clock-variable laptop,
not faster-transform claims; 1080p total time is effectively unchanged. The
allocation saving is the stronger deterministic result.

#### Allocation traces and final qualification

Separate Nsight captures enable `--cuda-memory-usage=true`. These are used for
allocation and launch accounting, not latency claims: the profiler documents
potentially significant overhead for memory tracking. The export also includes
warmup allocation/free events at timestamp zero, so the single-encode extractor
only counts positive-timestamp device-memory events. It verifies matched
allocation/free addresses and sizes, and an empty live-allocation set at the end.

| Input | Parent search arena bytes | Compact search arena bytes | Parent peak tracked device bytes | Compact peak tracked device bytes |
|---|---:|---:|---:|---:|
| Padded 4K | 626,787,328 | 323,845,888 | 3,110,455,104 | 2,807,513,664 |
| Padded 1080p | 156,741,248 | 81,005,952 | 777,459,334 | 701,724,038 |
| Flower | 20,602,112 | 10,675,712 | 102,332,176 | 92,405,776 |

Each encode still makes five device allocations and frees. Only the search
arena size changes; the other four allocations match exactly. The 4K peak of
tracked device allocations falls 9.7%. This is requested CUDA allocation memory,
not total board/driver memory or a direct measurement of physical VRAM residency.
Kernel counts, ordered launch geometry, registers, dynamic shared memory, and
transfer counts/bytes match exactly for each input. At 4K this remains 509 kernels,
31 HtoD copies (117,079,320 bytes), 19 DtoH copies (103,699,012 bytes), and one
518,400-byte D2D copy. Memory-tracked kernel totals are 356.652/364.086 ms for
parent/compact 4K, 61.595/60.618 ms at 1080p, and 16.333/16.308 ms for Flower;
no compute-speed improvement is inferred from these traces.

Both candidate and prepared-search executables pass memcheck, racecheck,
synccheck, and initcheck, with zero errors or race warnings. The complete 4K
prepared-search reuse check also passes memcheck. Six scored parent/compact
encodes (sample and Flower at efforts 7/9; odd padded 1080p/4K at effort 7)
retain identical SHA-256, strategy summaries, and final scores. All six compact
outputs decode with the pinned independent decoder at the expected dimensions.

Batch sizes 1/2/4 pass at 1080p in both fully-resident and maximum-throughput
modes. Compact fully-resident batch medians are 281.150/493.102/923.458 ms
(3.557/4.056/4.332 images/s); maximum-throughput medians are
118.138/289.011/583.037 ms (8.465/6.920/6.861 images/s). Fully-resident 4K
batch sizes 1/2 pass for both builds: parent medians 1128.849/2046.184 ms,
compact 1139.528/2134.260 ms. Those sequential cohorts are qualification runs,
not a controlled before/after batch speedup result. They follow instrumented
checks and show substantial clock/host variance. All batch outputs match their
serial references. End-of-sweep telemetry is 77 C, P3, 1282 MHz graphics,
5500 MHz memory.

A full-process memory trace of batch size 2 (one warmup and one paired serial/
batch sample) confirms both lanes' search arenas are live together. Peak tracked
device allocations fall from 6,222,823,552 to 5,616,940,672 bytes, saving
605,882,880 bytes (577.8 MiB). Both peaks contain ten allocations: two copies
of each of the five per-image allocations. All nine encodes in each captured
process have matching allocation-size counts, except for the smaller search
arena. These aligned 3840x2160 batch inputs have slightly different non-search
allocation sizes from the odd-source single-image workload above. Memory
tracking confirms the capacity benefit under concurrency without establishing
a batch latency improvement.

#### Artifacts and remaining work

Ignored artifacts are under `build-cuda-ninja/profiles/s28_*`. Saved
`parent_{benchmark,encode,batch}.exe` binaries precede the change. The
`warmed_{4k,1080p,flower}.{json,txt}` files retain every process/sample;
`final_{parent,retained}_{4k,1080p,flower}.{nsys-rep,sqlite}` and
`memory_summary.{py,json}` retain the launch/allocation accounting. Full-process
batch captures are `memory_batch2_{parent,retained}.{nsys-rep,sqlite}`, with
`batch_memory_summary.{py,json}`. The `ac_strategy_cuda{,_search}_*check.txt`,
`memory_4k.txt`, `identity.txt`, and `batch_*.txt` files retain sanitizer,
codestream, and batch qualification. `qualify.ps1` sequences the checks, and
`profile_batch_memory.ps1` records the separate concurrent-allocation traces.

No further full-sized dead AC-search intermediate remains in this allocation
layout. The retained forward coefficients in B are still required by the fused
inverse, including cross-channel CfL. Reducing B needs a different execution
schedule or more fusion, not merely a smaller range. Other resident arenas,
direct basis-matrix DCT arithmetic, reconstruction filters, host work, and
transfers remain material. A bounded factorized-DCT experiment can start from
the existing Metal radix-2 implementation, first checking CUDA register/spill
behavior and standalone transform accuracy, then integrating with fused AC
input/output paths and qualifying perceptual/size behavior. Byte identity is
not assumed to be the acceptance ceiling for that arithmetic experiment.
The fully-resident backend is not demonstrated maxed out.

### Follow-up: factorize resident CUDA DCT arithmetic (2026-09-05)

Parent: `c1a75bb` (compact AC-search scratch). This experiment replaces direct
basis-matrix arithmetic for the seven resident transform shapes through 32x32
with the radix-2 DCT-II/III factorization already used by the Metal backend.
It applies to ordinary forward/inverse transforms, descriptor-gathered AC
forward transforms, and fused residual/inverse/loss evaluation. The 64x32 and
32x64 ordinary transforms retain their matrix implementations. The internal
`LaunchCudaDctMatrix` entry point preserves the old arithmetic as a numerical
and performance oracle, including all smaller shapes.

#### Arithmetic and scheduling

The direct separable transform performs work proportional to
`W*H*(W+H)`. Factorization reduces that to `W*H*(log2(W)+log2(H))`, using
compile-time butterflies and constants instead of basis loads and long dot
products. Each lane owns a full one-dimensional register vector; recursion
and loops are forcibly inlined/unrolled. Forward output is normalized by
`1/(W*H)`, and inverse output is already in pixel units. The fused loss adapter
therefore omits the matrix inverse's orthonormal-to-pixel rescaling, while
retaining its eighth-power arithmetic and FP32 halving reduction.

Blocks contain 64 threads, packing `64/max(W,H)` transforms, each with a
`H*(W+1)` padded shared tile. Tall/square forward transforms process columns
first so input and coefficient transactions coalesce. Wide forward transforms
stage coalesced input reads, then process rows first to retain coalesced native
row-major coefficient stores. Inverse transforms use the same native
coefficient layout and coalesced pixel stores. Tall fused inverse outputs
are redistributed through the existing tile before loss reduction to preserve
the established row-major mask addressing and summation order. Inactive tail
groups participate in block barriers without reading or writing global data.
No new global intermediate, allocation, or submission boundary is introduced.

Standalone prototypes tested 64/128/256-thread blocks over 16,777,216 floats,
with three warmups and seven alternating rounds of three launches. The first
wide-forward variant had scattered stores; merely moving the scattered side
to input reads was insufficient. Coalesced staging fixed that issue. Selected
64-thread prototype medians (matrix to factorized) were 4.298 to 0.549 ms for
32x16 forward, 5.150 to 0.552 ms for 16x32 inverse, and 4.265 to 0.553 ms for
32x32 inverse. These are raw-kernel observations, not whole-encoder speedups;
clock variance was especially large in early small-transform rounds.

The initial integrated 32x32 residual/inverse/loss kernel compiled to a
384-byte stack frame, despite the standalone transform being spill-free.
SASS showed 32 square-root helper call sites and local array traffic.
Forcing the input/output adapters inline, spelling the root `__fsqrt_rn`, and
using a narrowly scoped FTZ square-root instruction did not remove that frame;
none of those changes is retained. Instead, coefficient rates now perform the
first FP32 halving step as each lower/upper pair arrives, retaining only half
the magnitude array. Exact integer nonzero counts accumulate immediately.
This preserves the magnitude tree and count semantics, uses ordinary `sqrtf`,
and reduces the 32x32 kernel to zero stack/local bytes (89 registers per
thread, 8,448 shared bytes per block). The helper call sites remain, but the
local loads/stores disappear. The same change reduces 16x16 residual/inverse
registers from 76 to 43 and both 512-coefficient shapes from 80 to 64.
No global fast-math option is enabled.

#### Numerical and quality qualification

All 55 CUDA CTests and all 47 CPU-only CTests pass. Existing transform
tolerances are unchanged: forward absolute/relative `3e-5/3e-4`, inverse and
round-trip `5e-4/5e-4`. The backend test now additionally runs both raw launch
paths against the independent double reference with seven leading and eleven
trailing sentinels, offset pointers, and an unchanged-input check. Coverage is
nine shapes times fourteen batch counts (1/2/3/4/7/8/9/15/16/17/19/31/32/33),
with independent forward/inverse inputs, impulses, constants, checkerboards,
unequal horizontal/vertical structure, and noise. Both raw paths and both
directions are checked in each configuration (504 guarded launches).

The existing fused oracles also pass: 420 inverse/loss batches (12,810 channel
loss comparisons), and 84 residual/inverse batches (4,242 channel comparisons)
with exact FP32 loss/rate reduction checks, exact nonzero counts, non-finite
handling, mask and descriptor boundaries, host/device CfL, output guards,
and partial blocks. Full AC-cost CPU-reference and ordered-search tests pass
without loosening their tolerances. The standalone double-reference probe's
largest 32x32 inverse absolute error was `1.2445e-4` (matrix `4.7849e-5`), with
factorized RMS error `5.25e-6`; factorization changes rounding and is not
claimed byte-identical to the matrix path.

Compute Sanitizer memcheck/racecheck/synccheck/initcheck all pass for the
expanded CUDA backend test, AC-candidate test, and AC-search test (12 runs).
The large `--memory-4k` search also passes memcheck. No errors or race hazards
are reported. Only the backend test suppresses CUDA API-error reporting,
because it intentionally launches a zero-sized grid to verify stale-error
consumption; memory/race/synchronization/uninitialized-access detection remains
enabled. Runs are serialized, with no overlapping build or benchmark.

The decoded-image comparison uses the pinned libjxl `djxl` and
`butteraugli_main` tools. Decoder output and metric input are explicitly
linear sRGB (`RGB_D65_SRG_Rel_Lin`), with default SDR metric intensity 80 nits.
Seven inputs comprise the 17x13 sample, odd-source synthetic 1080p/4K,
510x532 Flower, and three 500x500 color photographs from libjxl testdata's
[Wesaturate corpus](https://github.com/libjxl/testdata/tree/73695d303670c90e4d506ea89d9901b081385089/external/wesaturate/500px).
The latter's pinned README/license identify the originals as CC0. The exact
files are `cvo9xd_keong_macan_srgb8.png`,
`tmshre_riaphotographs_srgb8.png`, and `u76c0g_bliznaca_srgb8.png`;
source hashes, linear-PFM hashes, and license copies accompany the artifacts.
Every input is encoded at effort 7 and distances 0.5/1.2/3.0; sample and Flower
also run effort 9 at distance 1.2 (23 before/after pairs). A greater-than-0.5%
increase in either bytes or independently decoded Butteraugli score was
declared an investigation threshold before collecting results, not an assumed
acceptance result. All cases, including any outliers, are retained.

All 46 codestreams decode with the expected dimensions, and no pair exceeds
either investigation threshold. The independently decoded Butteraugli score
is identical to the tool's printed precision in 22 pairs; Bliznaca at distance
3 improves from 3.2424831390 to 3.2351341248 (-0.227%). File-size ratios range
from 0.9998273 to 1.0002802. The only size changes are:

| Input / distance / effort | Parent bytes | Factorized bytes | Change |
|---|---:|---:|---:|
| Synthetic 1080p / 0.5 / 7 | 207,139 | 207,173 | +0.0164% |
| Synthetic 1080p / 1.2 / 7 | 46,952 | 46,959 | +0.0149% |
| Synthetic 4K / 0.5 / 7 | 798,858 | 798,720 | -0.0173% |
| Bliznaca / 3.0 / 7 | 17,846 | 17,851 | +0.0280% |

Strategy histograms match in every pair. Ten codestream pairs are byte-identical;
the other thirteen are not, including some with identical length and metric.
Equal metric scores are not a claim of pixel identity. These results qualify
this SDR corpus and the tested device/toolchain, not arbitrary image content,
HDR inputs, or other GPU architectures. The decoder/metric libjxl revision is
`e8ff09762481785938d8e4e01333ed3917571161` (Clang 22.1.8).

#### Final warmed wall measurements

Seven alternating parent/candidate process pairs per workload, each with three
warmups and five measured samples. Each process contributes its sample median;
the reported percentage is the median of the seven paired candidate/parent
ratios, not the ratio of independently pooled medians. No build, sanitizer,
second GPU workload, or profiling overlaps these runs. Every sample is kept.

| Workload / stage | Parent median [range], ms | Factorized median [range], ms | Paired change | Winning pairs |
|---|---:|---:|---:|---:|
| Padded 4K / total | 838.434 [799.568-875.159] | 782.615 [736.428-822.077] | -5.1% | 7/7 |
| Padded 4K / quantization | 577.619 [555.050-584.848] | 511.189 [495.734-522.603] | -10.6% | 7/7 |
| Padded 1080p / total | 214.800 [210.419-220.980] | 199.259 [195.440-238.996] | -7.2% | 5/7 |
| Padded 1080p / quantization | 129.420 [126.951-134.078] | 116.943 [115.522-132.189] | -8.9% | 6/7 |
| Flower / total | 49.989 [45.986-68.615] | 44.917 [42.227-53.744] | -7.8% | 5/7 |
| Flower / quantization | 29.749 [27.704-35.669] | 27.309 [25.423-31.164] | -5.9% | 6/7 |

The large 1080p candidate and Flower parent/candidate outliers are not removed.
An earlier 4K cohort of the initial integrated factorization, before the
rate-storage fix, measured -5.8% total and -9.6% quantization. It is retained
as an experiment record, not pooled with the final implementation. Laptop
clock/host variability prevents interpreting these sequential cohorts as a
precise isolated end-to-end benefit of the stack fix.

#### Final kernel traces and batch checks

Nsight captures one fully-resident encode after three warmups, without memory
tracking. The unchanged-kernel column excludes all ordinary and AC DCT kernels,
so it does not accidentally count another improved transform as a control.
Static and dynamic shared memory are both accounted for.

| Workload | All DCT, parent -> factorized | Other kernels, parent -> factorized | Total kernel time, parent -> factorized |
|---|---:|---:|---:|
| Padded 4K | 107.922 -> 23.051 ms (-78.6%) | 258.020 -> 262.462 ms | 365.941 -> 285.514 ms |
| Padded 1080p | 16.145 -> 5.345 ms (-66.9%) | 47.084 -> 43.816 ms | 63.229 -> 49.160 ms |
| Flower | 2.185 -> 0.771 ms (-64.7%) | 14.125 -> 14.153 ms | 16.310 -> 14.924 ms |

The 4K DCT split is AC forward 46.182 -> 8.793 ms, fused residual/inverse/loss
48.255 -> 11.737 ms, and ordinary transforms 13.485 -> 2.522 ms. Other kernels
are 1.7% slower in that trace, effectively unchanged on Flower (+0.2%), and
6.9% faster at 1080p. The latter control movement means that not all of the
1080p total-kernel reduction can be attributed to this change. Kernel duration
is not end-to-end latency; the separately warmed wall results above govern
that claim.

The seven 4K fused inverse kernels show:

| W x H | Matrix, ms | Factorized, ms | Factorized registers/thread | Factorized static shared bytes/block |
|---|---:|---:|---:|---:|
| 8x8 | 1.228 | 1.257 | 40 | 2,304 |
| 8x16 | 3.855 | 1.632 | 46 | 2,304 |
| 16x8 | 2.034 | 1.512 | 40 | 2,176 |
| 16x16 | 7.484 | 2.307 | 43 | 4,352 |
| 16x32 | 12.893 | 1.289 | 64 | 4,352 |
| 32x16 | 9.494 | 1.236 | 64 | 4,224 |
| 32x32 | 11.266 | 2.505 | 89 | 8,448 |

All factorized instantiations have zero stack/local bytes in `cuobjdump`.
The 8x8 fused inverse is essentially flat, including a small regression in
this particular 4K trace; the retained improvement is primarily in larger
shapes. The initial integrated 32x32 trace was 8.356 ms with stack traffic,
versus 2.505 ms after the pairwise rate-storage change. These are separate
captures, but the resource and SASS changes independently explain the removed
local traffic.

Launch counts are unchanged: 509 at 4K and Flower, 494 at 1080p. All transfer
counts and byte totals match each parent. The 4K trace retains 31 HtoD copies
(117,079,320 bytes), 19 DtoH copies (103,699,012 bytes), and one DtoD copy
(518,400 bytes). This checkpoint changes arithmetic and on-chip scheduling,
not the compact search arena or global transfer boundaries.

Batch sizes 1/2/4 pass at 1080p in both modes, and fully-resident sizes 1/2
pass at aligned 3840x2160. Every batch codestream matches its serial reference.
Fully-resident 1080p batch medians are 220.137/362.894/859.030 ms;
maximum-throughput medians are 117.586/246.722/332.225 ms. Fully-resident 4K
medians are 815.233/1622.771 ms. These are qualification runs with substantial
host/clock variance, not an isolated before/after batch-throughput measurement.

#### Artifacts and next bottlenecks

Ignored artifacts are under `build-cuda-ninja/profiles/s29_*`:
`parent_{benchmark,encode}.exe` preserve the parent; `factored_probe*` retain
the standalone layout/block-size experiments; `initial_*` retain the first
integration; `final_{parent,retained}_{4k,1080p,flower}.{nsys-rep,sqlite}` and
`profile_summary.{py,json}` retain the final launch/shape accounting.
`warmed_{4k,1080p,flower}.{json,txt}` preserve every timing sample.
`final_resources.txt` and `final_32_spills.txt` retain resource/call-site
inspection (the latter contains calls but no local loads/stores).
`quality.py`, `quality.json`, `quality_*.{jxl,pfm}`, and `corpus/` retain all
decoded-image comparisons and provenance. `sanitize.ps1`, `*check.txt`
(filenames end in the individual `memcheck`/`racecheck`/`synccheck`/`initcheck`
mode), `memory_4k.txt`, `measure.ps1`, and `batch_*.txt` retain qualification
scripts and results.

DCT is now only 8.1% of kernel time in the retained 4K trace. Convolutions
consume 73.862 ms and Malta 58.477 ms, with coefficient encoding at 29.477 ms
and quantization adjustment at 23.010 ms. The adjustment stage also consumes
8.480 ms at 1080p and 7.280 ms (48.8% of kernels) on Flower. Its current
`SelectAdjustedQuantizationKernel` assigns one thread to an entire transform
and scans all three channels serially through `AdjustQuantForChannel`;
cooperative coefficient processing is a concrete next experiment, subject to
its floating-point reduction/threshold decisions and decoded-quality gates.
Filtering locality/fusion, remaining resident allocations, host work, and
transfers also remain material. Further AC fusion could remove the retained
forward-coefficient buffer, but requires a new cross-channel schedule, not
another scratch-size adjustment. This is a verified checkpoint, not evidence
that fully-resident encoding is maxed out.

### Follow-up: cooperate across quantization-adjustment coefficients (2026-09-05)

Parent: `f26e2ef` (factorized resident DCT). The next profile identified
`SelectAdjustedQuantizationKernel` at 23.010 ms for 4K, 8.480 ms for 1080p,
and 7.280 ms on Flower. The latter was 48.8% of all GPU kernel duration.
The old kernel assigns one thread to each transform anchor, serially scans
all coefficients in each of three channels, and dynamically indexes quadrant
arrays. For Flower, six of seven shape batches launched just one 256-thread
block. The 32x32 batch took about 0.96 ms per invocation. Resource/SASS
inspection found a 144-byte thread stack and repeated local loads/stores.
This combines underfilled grids, scattered coefficient reads across threads,
long serial scans, and local-array traffic.

#### Cooperative implementation and experiments

Each anchor now launches one 96-thread block: one complete warp per channel,
with eight lanes assigned to each canonical coefficient quadrant. Lanes read
and quantize separate coefficients, accumulate scalar statistics, then reduce
within quadrants and across quadrant leaders. The warp leader applies the
existing raw-quant and Y-threshold policy. Three warp decisions pass through
12 bytes of shared memory and one block barrier before publishing their
maximum. This barrier also ensures that every channel has read the initial
raw quant before it is overwritten. No global partial buffer, new allocation,
extra launch, or host synchronization is introduced.

Width/height and channel are compile-time parameters. Seven physical strategy
orientations share five canonical coefficient shapes: 8x8, 16x8, 16x16,
32x16, and 32x32. Fixing the strategy class also resolves the policy tables to
constants. An initial cooperative prototype still used dynamic class indexing
and allocated 96 local bytes for the two multiplier tables on larger shapes;
specialization removed them. Final registers per thread for those five shapes
are 39/41/38/40/40, with zero stack/local bytes and 12 shared bytes per block.
The serial implementation remains available through the internal
`LaunchCudaAqSelectAdjustedQuantizationScalar` oracle, and is the fallback for
internal batches outside the specialized shape/strategy combinations.

The prototype compared one/two/four anchors per block (96/192/384 threads),
with three warmups and seven alternating rounds of three event-timed launches.
Initial raw-quant arrays were reset outside the timed interval. Small batches
had 129 anchors; large cases used 8,388,608 coefficients per channel. The
96-thread schedule consistently helped small grids and was competitive on
large grids, so it is the retained launch policy. For 129 32x32 anchors,
the final prototype measured 1.082 ms scalar versus about 0.022 ms cooperative;
for 8,192 such anchors, 2.781 versus 0.744 ms. These are raw-kernel observations
with broad clock/event variance, not public-workflow speedups.

Each prototype version checked 63 batches spanning all seven strategies,
counts 1/2/3/7/17/129/259 plus shape-dependent large counts, offset buffers,
guards, and AC non-finites. Across 338,917 anchor decisions per version,
all three cooperative launch geometries matched scalar raw quants and Y
threshold bits. This is measured fixture parity, not a universal bit-identity
claim: error, border, and magnitude sums use a fixed parallel FP32 tree instead
of serial row-major addition. No global fast math or reduced precision is used.

#### Direct and integrated correctness checks

All 56 CUDA CTests and 47 CPU-only CTests pass. The new
`cuda_quantization_adjustment` test checks 182 batches (9,667 anchors) against
the retained scalar CUDA implementation. Raw quant values and threshold bits
match exactly in these fixtures. Its 9,429 finite-input anchors also compare
against the independent CPU policy: exact raw quant, Y thresholds within
`2e-6` absolute. The latter permits the pre-existing CUDA float versus CPU
double policy constants; the existing test tolerances are not relaxed.

Coverage includes seven strategies, counts 1/2/3/7/17/33/129/257, global scales
1/3541/32768, unequal channel multipliers, raw values through 255/256, flat,
sparse, active, high-frequency-border, threshold-tie, half-integer, signed-zero,
and quantizer-limit patterns. Inputs use the real default quantization matrices;
unused table entries are NaNs. Anchor/coefficient/raw pointers are offset,
raw fields are strided and guarded, threshold output has prefix/suffix guards,
and coefficients must remain bitwise unchanged. AC NaN/infinity inputs set
error bit 16 without clearing a pre-existing bit; low-frequency non-finites
are skipped as in the scalar kernel. The public AQ tests retain their
bounded/full, maximum-error, repeatability, arena-reuse, and frame checks.

Eight sanitizer runs complete with zero errors/hazards: all four modes on
the focused adjustment test, plus full-AQ memcheck, synccheck, initcheck,
and kernel-filtered racecheck. The latter executes the entire existing AQ
test, including four-worker public encodes in both policies, while
`--kernel-regex kns=SelectAdjustedQuantization` instruments the changed
cooperative instantiations and retained scalar kernel. All other listed
runs are unfiltered; no CUDA API errors are suppressed.

The initial unfiltered full-AQ racecheck was deliberately aborted after
approximately 52 minutes without a final summary. It is incomplete, not a
pass or a detected-race result. Its original log is preserved. A Windows
firewall/elevation prompt was subsequently reported, but its role in the
delay was not established. The kernel-filtered full-workflow racecheck
completed in approximately 27 seconds. This qualification does not claim
an unfiltered full-AQ racecheck pass.

All 23 before/after image pairs are byte-identical to parent `f26e2ef`;
encoded sizes, independently decoded Butteraugli scores, selected-strategy
summaries, and reported final scores match. The corpus and explicit linear
color protocol are the same as the preceding DCT checkpoint: seven inputs
at distances 0.5/1.2/3.0 and effort 7, plus sample/Flower effort-9 checks at
distance 1.2. Pinned libjxl revision
`e8ff09762481785938d8e4e01333ed3917571161` decodes all outputs at their source
dimensions and measures them with `RGB_D65_SRG_Rel_Lin`, SDR 80 nits.
No case crosses the predeclared 0.5% size/score investigation threshold;
every measured change is zero. This corpus result is not a universal
bit-identity guarantee for the reordered floating-point reductions.

Final parent/retained Nsight captures use three warmups and one captured
fully-resident encode, with no overlapping GPU test or benchmark work:

| Workload | Adjustment parent | Cooperative | Change | Other kernels parent / cooperative | All kernels parent / cooperative |
|---|---:|---:|---:|---:|---:|
| Odd 4K | 25.507 ms | 3.969 ms | -84.4% | 275.729 / 297.507 ms | 301.236 / 301.476 ms |
| Odd 1080p | 8.818 ms | 0.668 ms | -92.4% | 41.897 / 41.716 ms | 50.715 / 42.384 ms |
| Flower | 7.272 ms | 0.242 ms | -96.7% | 7.634 / 7.644 ms | 14.905 / 7.886 ms |

The 4K unchanged-kernel control grows 7.9%, offsetting the 21.538 ms
adjustment reduction; no total-GPU-time gain is claimed from that capture.
At 1080p and Flower, unchanged work is within 0.5% and total kernel time
falls 16.4% and 47.1%. Launch counts remain 509/494/509 respectively.
Each before/after pair has identical HtoD, DtoH, and D2D transfer volumes
and counts (31/19/1 copies); the optimization adds no transfers or launches.

All batch runs complete with codestream and full-summary identity against
their serial references. Current-build batch results below qualify behavior,
not an improvement over the parent:

| Workload / policy | Batch size | Batch median | Images/s | Median serial/batch speedup |
|---|---:|---:|---:|---:|
| 1080p / fully resident | 1 | 249.021 ms | 4.016 | 0.917x |
| 1080p / fully resident | 2 | 449.865 ms | 4.446 | 1.223x |
| 1080p / fully resident | 4 | 979.541 ms | 4.084 | 1.515x |
| 1080p / maximum throughput | 1 | 196.711 ms | 5.084 | 1.185x |
| 1080p / maximum throughput | 2 | 390.839 ms | 5.117 | 1.542x |
| 1080p / maximum throughput | 4 | 481.612 ms | 8.305 | 2.053x |
| 4K / fully resident | 1 | 1071.586 ms | 0.933 | 0.877x |
| 4K / fully resident | 2 | 1956.245 ms | 1.022 | 1.164x |

These batch checks use one warmup and three alternating serial/batch samples.
The GPU-state snapshots around the first timing cohort span 73-80 C,
P8/P3, and 210-1282 MHz; clocks were not locked. A post-measurement snapshot
reports AC power, the Windows Balanced scheme, and concurrent CPU activity
from Windows security, Office, and editor processes. This is evidence of
background activity, not proof that it caused every timing difference.

Wall-time qualification uses seven alternating independent-process pairs,
three warmups and five retained samples per process, distance 1.2, effort 7,
automatic CPU threads, and no final-score diagnostic. A second complete
cohort was run because the initial parent itself varied substantially.
All samples and both cohorts are retained; none are dropped as outliers.
Percentages below are medians of paired candidate/parent ratios, not ratios
of cohort medians. The pooled column contains all fourteen pairs:

| Workload / stage | Initial paired change | Repeat paired change | All 14 pairs |
|---|---:|---:|---:|
| Odd 4K / total | +20.8% | +3.7% | +7.6% |
| Odd 4K / quantization | -7.4% | +0.5% | -0.4% |
| Odd 1080p / total | +8.8% | -1.4% | -1.3% |
| Odd 1080p / quantization | -1.3% | -11.4% | -10.5% |
| Flower / total | -5.3% | -10.1% | -7.7% |
| Flower / quantization | -19.8% | -19.5% | -19.6% |

For context, initial 4K process medians span 755.503-2304.972 ms in the
parent and 1039.518-2202.232 ms in the candidate. A seven-pair control using
the identical parent executable in both positions reports a +2.2% median
paired total difference, with individual pairs from -4.4% to +28.4%.
Its quantization difference is +0.9%, ranging from -2.8% to +20.0%.
The unchanged CPU codestream stage is 8.7% slower in the pooled 4K pairs;
input preparation is +0.2%. These controls demonstrate measurement
variability, not that the measured 4K regression can be ignored or that a
particular background process caused it.

A further diagnostic compiles the retained scalar and cooperative policies
into one executable and chooses between them at process initialization.
Both modes use the same host code, libraries, binary layout, executable
path, and timing boundary. This ignored-build probe adds no production
environment switch. It confirms exact codestream identity on the actual
benchmark inputs: 1,048,983 bytes for odd 4K, 265,570 for odd 1080p, and
37,018 for Flower. The first-sample output dump is enabled only during
identity qualification and is outside the timed encode; timing runs do
not write codestream files. The same seven-pair protocol produces:

| Same-executable workload | Median paired total change | Median paired quantization change |
|---|---:|---:|
| Odd 4K | -0.2% | -0.5% |
| Odd 1080p | -0.1% | -5.5% |
| Flower | +2.6% | -15.8% |

This removes binary/path differences from that diagnostic but not host
contention, timing variation, or automatic GPU power-state changes. The
retention claim is therefore limited to the large, directly measured
adjustment-kernel reduction and qualified correctness. Stable whole-encode
latency or throughput gains are not established by this checkpoint.

The fresh retained 4K trace spends 54.798 ms in Malta response, 13.637 ms
in Malta scaling, and 33.347 ms in resident coefficient encoding. Tiled
convolutions and host serialization/launch boundaries remain substantial.
Read-only inspection confirms Malta has no local/stack storage and already
reuses repeated response sums; scale/response fusion would duplicate scale
arithmetic in tile halos and needs measurement, not an assumed win. The
fully-resident path is not demonstrated maxed out.

Ignored artifacts under `build-cuda-ninja/profiles` use the `s30_` prefix.
`parent_{encode,benchmark,batch}.exe` are hash-verified copies of the preceding
retained executables; `retained_{encode,benchmark,batch}.exe` preserve the
qualified current build. `quant_probe{,_v1}.exe`, `quant_probe{,_v2}.txt`,
`cooperative_quant{,_v1}.cuh`, and `parent_resident_kernels.cu` preserve the
event-probe variants and frozen baseline; `final_resources.txt` records
production resource usage. `quality.py`/`quality.json` and `quality_*`
retain the decoded comparisons, reusing the provenance-pinned `s29_corpus`.
`sanitize{,_remaining}.ps1`, individual `*check.txt` files, and
`sanitizer_abort.txt` preserve the exact sanitizer scopes and aborted run.
`measure.ps1`, `warmed_*`, `recheck.ps1`, `repeat_*`, `control_aa_4k.*`,
`wall_summary.*`, and GPU-state logs retain all wall-time observations.
`profile_final.ps1`, `profile_summary.*`, `final_{parent,retained}_*`,
`kernel_totals.*`, and `batch_*.txt` retain the profiles and batch checks.
`switch_probe.cu`, `switch_benchmark.cpp`, `switch_measure.py`, and
`switch_*` preserve the same-executable diagnostic and exact outputs.

### Follow-up: fuse Malta scaling into response tiles (2026-09-05)

Parent: `eb1b624` (cooperative quantization adjustment). Its retained 4K
profile spent 54.798 ms in Malta response and 13.637 ms in Malta scaling,
versus 33.347 ms in resident coefficient encoding. Scaling wrote a full
temporary plane; the response kernel then reloaded overlapping 40x16 halos
for 32x8 output tiles. The response already had no local/stack storage and
reused repeated directional sums. Merely removing source-level duplicate
sums would not remove additional executed arithmetic.

#### Experiments and retained implementation

Nine ignored-build variants compare the original separate passes, frequency
specialization with a 2D grid, larger tiles, direct scale/response fusion,
and warp-shuffle sharing. They all pass 120 guarded bitwise cases against
the frozen parent source. Three input patterns cover signed zeros,
near-identical values, and large independent values; ten shapes include
partial tiles, independent padded strides, and offset pointers. Both
frequency modes and initialization/addition policies are exercised.

The initial CUDA-event screen uses three warmups, seven alternating-order
rounds, and three repeats per event interval. Four image sizes range from
512x536 to 3840x2160. The 4K median pair/stage times are:

| Variant | Full response (ms) | Low-frequency response (ms) |
|---|---:|---:|
| Original separate passes, 32x8 | 2.483 | 2.553 |
| Specialized separate passes, 32x8 | 2.574 | 2.626 |
| Specialized separate passes, 32x16 | 2.623 | 2.457 |
| Fused, 32x8 | 2.102 | 2.220 |
| Fused, 32x16 | 2.501 | 2.932 |
| Fused, 32x32 | 2.884 | 3.558 |
| Fused, 64x8 | 2.574 | 2.997 |
| Separate passes with warp sharing, 32x8 | 6.459 | 5.278 |
| Fused with warp sharing, 32x8 | 5.346 | 4.677 |

The retained 32x8 fusion improves this isolated screen by approximately
13-20%, depending on size and frequency mode. Larger tiles and warp sharing
are rejected. None of the variants spills; the slow shuffle variants do
not support an occupancy/spill explanation. Hardware-counter evidence for
their precise bottleneck remains unavailable.

The production kernel scales reference/distorted values directly into the
shared tile, then evaluates the unchanged `MaltaLf` or `MaltaFull` expression.
Frequency selection is specialized at launch. The 2D grid avoids repeated
dynamic tile-column division. A flattened-grid specialization preserves the
previous geometry when the tile-row count exceeds CUDA's 65535 grid.y limit.
All threads load and synchronize before partial-edge threads return.

Halo scaling is deliberately repeated: an interior tile has 640 scaled
values for 256 outputs. The change removes the temporary-plane write/read
and one launch, not all duplicate input traffic. It does not remove the
working-plane allocation, which later mask/blur stages still use. Stage
weights, normalization, ordering, response sum trees, zero boundary policy,
and output accumulation order are unchanged. The original two-pass kernels
remain available through an internal test-oracle entry point.

The production binary uses 40 registers/thread for full response and 34
for low-frequency response, 2560 shared bytes/block, and zero stack/local
storage for both normal and flattened grids. These are production counts;
the generic-reader prototype used 38/35 registers. The parent response used
40 registers and 2560 shared bytes, plus a separate 21-register scale kernel.

#### Whole-encoder profiles and timing limits

All fresh profiles use three warmups and one captured public encode with
the same linear-RGB-to-codestream boundary as the preceding checkpoint.
The three ordinary traces remove exactly 24 launches: 509 to 485 at 4K
and on Flower, and 494 to 470 at 1080p. Malta changes from 24 scale plus
24 response launches to 24 fused launches. Copies remain 31 H2D, 19 D2H,
and one D2D, with identical byte totals:

| Workload | H2D bytes | D2H bytes | D2D bytes |
|---|---:|---:|---:|
| Odd 4K | 117,079,320 | 103,699,012 | 518,400 |
| Odd 1080p | 29,336,392 | 25,924,192 | 129,600 |
| Flower | 3,980,492 | 3,430,532 | 17,152 |

At 1080p, Malta falls from 5.820 to 4.518 ms (-22.4%) and all kernels
from 42.139 to 40.007 ms (-5.1%); non-Malta kernels change -2.3%.
On Flower, Malta falls from 0.767 to 0.664 ms (-13.4%), while other kernels
are essentially unchanged (+0.04%); all kernels change -1.3%.

The first 4K capture is unfavorable, so three additional pairs reverse and
alternate execution order. All four pairs, including the first, are retained:

| 4K pair (order) | Parent Malta (ms) | Fused Malta (ms) | Malta change | Other kernels change | All kernels change |
|---|---:|---:|---:|---:|---:|
| Original (parent first) | 44.799 | 49.949 | +11.5% | +8.0% | +8.7% |
| Repeat 0 (fused first) | 54.188 | 42.352 | -21.8% | -4.3% | -8.0% |
| Repeat 1 (parent first) | 52.359 | 46.763 | -10.7% | +0.03% | -2.2% |
| Repeat 2 (fused first) | 49.725 | 49.439 | -0.6% | +4.6% | +3.6% |

Across all four pairs, median paired changes are -5.6% for Malta, +2.3%
for other kernels, and +0.7% for all kernels. These traces support fewer
launches but not a stable 4K total-GPU-time improvement. The first capture's
regression is concentrated in early full-resolution Malta calls; it is not
silently discarded or explained away by an unmeasured clock assumption.

A follow-up event probe calls the actual production and reference wrappers
from one executable at 3840x2160. It covers all six production weight/norm
sets and three input patterns: identical nonzero values, all zeros, and
near-identical nonzero values. Short-warmup raw times drift sharply within
bursts. Extending warmup to 64 old/new pairs before seven alternating timed
rounds yields median paired improvements of 6.9-11.1% across all 18 cases.
Some individual intervals still vary. This supports an isolated production
kernel improvement without establishing the cause of every whole-encode
profile fluctuation. No clock, power, firewall, or background-service setting
was changed for these measurements.

The ordinary public-workflow wall measurement has seven alternating pairs,
three warmups and five timed samples per process. Each process contributes
its median, and the reported percentage is the median paired change:

| Workload | Parent total median (ms) | Fused total median (ms) | Paired total change | Paired quantization change |
|---|---:|---:|---:|---:|
| Odd 4K | 736.370 | 737.455 | -0.3% | -0.9% |
| Odd 1080p | 183.238 | 181.374 | -0.8% | -0.8% |
| Flower | 36.724 | 36.483 | -1.4% | -2.4% |

The median of paired ratios need not equal the ratio of marginal medians.
Observed total-time ranges are 676.5-813.3/674.7-779.4 ms for parent/fused
4K, 179.8-197.8/176.4-194.9 ms for 1080p, and 35.7-52.2/35.6-50.6 ms
for Flower. Given that variation and the mixed 4K profiles, this checkpoint
does not claim a stable end-to-end gain. Its retention case is the isolated
kernel improvement, reduced launch count, and correctness qualification.

#### Qualification and remaining scope

The full Release suites pass 58 CUDA and 47 CPU-only tests. New focused
tests check 160 ordinary cases and eight tall-grid cases, each with three
consecutive stages and bitwise comparisons to the retained separate passes.
They cover both frequency and initialization modes, signed zeros, near and
large differences, values at/adjacent to asymmetric thresholds, poisoned
input padding, independent row strides, guard prefixes/suffixes, and input
immutability. Heights 524280 and 524281 exercise the last 2D tile row and
the first flattened-grid fallback. Existing CPU-reference Butteraugli tests
cover the complete multiscale pipeline and non-default perceptual options.

All 23 before/after image pairs are byte-identical, with identical strategy
counts, final reported scores, and independently decoded Butteraugli metrics.
The seven-input corpus covers distances 0.5/1.2/3 at effort 7, with two
additional effort-9 cases. It reuses the preceding checkpoint's pinned
libjxl decoder/metric (`e8ff09762481785938d8e4e01333ed3917571161`,
Clang 22.1.8), explicit linear-sRGB interpretation, and provenance-pinned
natural images.

Ten sanitizer invocations complete successfully: memcheck, racecheck,
synccheck, and initcheck on each of the ordinary Malta differential test
and complete Butteraugli test, plus memcheck and synccheck on the tall-grid
test. The first eight runs are unfiltered. All report zero errors; both
racechecks report zero hazards/warnings. Tall-grid racecheck/initcheck are
not claimed. The ordinary and full-pipeline racechecks take 32.3 and 67.2
seconds respectively; tall-grid memory and synchronization checks take
15.6 and 7.1 seconds. No sanitizer is left running or aborted in this cycle.

Batch qualification passes at 1080p sizes 1/2/4 in fully-resident and
maximum-throughput modes, and at 4K sizes 1/2 in fully-resident mode. The
benchmark checks exact serial/batch identity. These are current-policy
concurrency checks, not a before/after batch-throughput improvement claim.

Ignored `build-cuda-ninja/profiles/s31_*` artifacts retain the frozen parent
source, nine-variant event probe, raw timings, production resource report,
ordinary and tall input tests, extended-warmup production probe, four-pair
4K profile diagnosis, all wall observations, batch logs, image qualification,
and exact sanitizer scopes. `s30_retained_{encode,benchmark,batch}.exe` are
the preserved parent binaries. Working-plane storage remains allocated and
resident coefficient encoding, convolutions, host work, and remaining
allocation/transfer boundaries remain material targets. The fully-resident
path is not demonstrated maxed out.

The fresh retained 4K API trace also records five `cudaMalloc` calls taking
16.2 ms, five `cudaFree` calls taking 51.4 ms, 48 `cudaMemcpyAsync` calls
taking 76.3 ms, and three `cudaMemcpy2DAsync` calls taking 22.8 ms. These
host API durations can include waits for device work; they are not additive
to kernel time or a proven removable-overhead budget. Allocation lifetime,
host staging, and transfer/synchronization boundaries deserve investigation
alongside the remaining kernels. `cudaProfilerStart` is excluded from this
analysis; `s31_api_totals.*` retain the raw API breakdown.

## Stream-ordered allocation follow-up (S32)

### Cause, experiment, and retained policy

The parent is `f1f9fe6`, the retained fused-Malta implementation. Its five
per-encode arenas still use `cudaMalloc` and `cudaFree`; consolidating arenas
did not remove the driver's repeated allocation and global synchronization
costs. The S31 4K trace records 16.2 ms in allocation and 51.4 ms in frees.
These host API spans may include GPU waits and are not additive to kernel
time. A new paired trace below isolates the allocation API change while
confirming unchanged kernels and transfers.

A same-executable 4K prototype compares the legacy allocator with
stream-ordered allocation using release thresholds of zero, 1 GiB, and
3 GiB. Three alternating rounds use three warmups and three measured
encodes per process. Median paired total-time changes versus legacy are
+2.9%, +0.01%, and -7.4%; quantization changes are +1.0%, -6.4%, and
-18.4%. The allocator change alone with no retained working set is not
enough. The 3 GiB policy keeps 2,818,572,288 reserved bytes for a
2,807,513,664-byte live requested working set. This prototype changes the
default pool only inside its isolated experiment process; production does
not change the application's default or current CUDA pool.

Production uses a gjxl-private `cudaMemPool_t`, `cudaMallocFromPoolAsync`,
and `cudaFreeAsync` ordered on each backend's existing non-blocking stream.
Matching device ordinals and release thresholds share one pool, including
the two production lanes. A mutex protects the weak-reference registry;
backend/buffer/submission state owns the pool lifetime. The registry cannot
keep unused pools alive. Internal dependency insertion for memory reuse is
disabled so the shared allocator does not introduce cross-lane dependencies
solely to reuse storage whose free has not completed.

`CudaBackendOptions` defaults the release threshold to
`min(totalGlobalMem / 2, 4 GiB)`. This is a cache-retention target, not a
hard allocation cap, and live working sets can exceed it. Callers can
override the threshold, use zero to release unused storage at synchronization
points, or disable stream-ordered allocation entirely. CUDA builds older
than 11.2 and devices reporting no pool support retain `cudaMalloc`/`cudaFree`.
Other pool-creation/configuration failures are reported, not hidden by an
unqualified fallback. The current toolkit/device and forced legacy path
are tested; older-toolkit and unsupported-device branches are not claimed
as real-hardware qualification.

`TrimCudaDeviceMemory(ordinal)` synchronizes work on the selected device in
the current context and trims all gjxl-private pools for that device. It
preserves live buffers and does not change other libraries' pools, although
the synchronization can wait for their CUDA work. Applications should
quiesce encoding first for full cache release. Concurrent encodes can grow
the pools again. Different custom thresholds create separate pools and can
increase retained memory; there is no automatic cross-pool OOM recovery.
See the [configuration and trimming example](cuda-support.md#device-allocation-policy).

Pool/free ordering and release semantics follow the
[CUDA 11.8 runtime pool API](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html)
and [stream-ordered allocator guide](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator).
The private-pool choice also follows NVIDIA's
[library integration guidance](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2/).
In particular, synchronous `cudaFree` does not itself wait for a pooled
allocation's users. The exceptional free-enqueue cleanup path first drains
the owning stream. Final device-state destruction similarly completes queued
frees before destroying the stream. Factory stream ownership is transferred
explicitly so a host allocation failure cannot destroy the stream twice.

### Warm and first-encode wall time

The ordinary public in-memory boundary is unchanged: caller-owned linear
RGB through the finished codestream, at distance 1.2 and effort 7. Seven
alternating independent-process parent/candidate pairs each use three
warmups and five samples; each process contributes its median. The paired
percentage is the median of candidate/parent ratios, not the ratio of the
two displayed marginal medians. No observation is discarded.

| Fully-resident input | Warm total ms, parent / retained | Paired total change | Warm quantization ms, parent / retained | Paired quantization change |
| --- | ---: | ---: | ---: | ---: |
| 3839x2159 | 719.916 / 602.002 | -16.1% | 482.433 / 384.860 | -19.2% |
| 1919x1079 | 175.378 / 146.393 | -16.5% | 97.894 / 69.481 | -26.7% |
| Flower 510x532 | 37.692 / 34.410 | -9.8% | 20.461 / 16.508 | -21.8% |

Total-time process-median ranges are 698.069-1019.053 / 480.952-646.886 ms
at 4K, 165.343-188.849 / 140.875-159.823 ms at 1080p, and
36.585-60.228 / 31.570-50.514 ms on Flower. The seventh 4K pair is a large
outlier (1019.053 / 480.952 ms); it remains included. The preceding six
4K pairs all favor the candidate by approximately 10-16% in total time.

A separate seven-pair first-encode cohort uses zero warmups and one sample
per process. Backend construction is outside this boundary, so these are
not complete CLI startup measurements. Parent/candidate total medians are
629.453 / 713.460 ms at 4K, 209.022 / 209.460 ms at 1080p, and
50.044 / 59.941 ms on Flower. Paired total changes are respectively
+16.2%, +0.4%, and +9.4%; quantization changes are +1.8%, -2.1%, and -4.4%.
First-encode total ranges are 583.439-896.518 / 593.170-968.451 ms at 4K,
201.391-396.236 / 198.849-409.822 ms at 1080p, and
48.754-70.611 / 54.035-116.855 ms on Flower. The cold observations are
unfavorable or mixed and highly variable: no cold-start improvement is
claimed. The retained policy targets repeated use of the persistent backend.

GPU samples span 64-79 C. The initial state is P3/1282 MHz graphics;
post-warm cohorts include P0/1762 MHz at 4K and P0/1282 MHz at 1080p.
Clocks, power policy, OS services, and firewall settings are not changed.
Laptop variation remains a limitation; no cause is inferred from clock
samples alone. No build, sanitizer, profiler, or other GPU probe overlaps
these ordinary wall-time runs.

### API, transfer, and memory evidence

Separate Nsight captures use three warmups and one profiled encode with
CUDA memory tracking enabled. Instrumented wall times are not used as
ordinary performance measurements. Allocation/free API totals are:

| Input | Parent `cudaMalloc` + `cudaFree`, ms | Retained pool allocation + async free, ms | GPU kernels, parent / retained, ms |
| --- | ---: | ---: | ---: |
| 3839x2159 | 13.115 + 47.893 | 0.157 + 0.148 | 135.378 / 135.844 |
| 1919x1079 | 3.398 + 19.636 | 0.113 + 0.118 | 37.967 / 37.955 |
| Flower | 3.173 + 1.709 | 0.187 + 0.147 | 7.566 / 7.596 |

Both versions make five allocations and five frees per encode. Kernel
counts are unchanged at 485 / 470 / 485. Every trace retains 31 H2D,
19 D2H, and one D2D copy. Their exact byte totals are unchanged:
117,079,320 / 103,699,012 / 518,400 at 4K;
29,336,392 / 25,924,192 / 129,600 at 1080p; and
3,980,492 / 3,430,532 / 17,152 on Flower. Kernel arithmetic is untouched.
The 99.5% reduction in 4K allocation/free API time is not a 99.5%
whole-encode speedup, nor may its old synchronization spans be added to
GPU work as an independent cost.

The exact five allocation sizes and peak logical live requests match the
parent: 2,807,513,664 bytes at odd 4K, 701,724,038 at odd 1080p, and
92,405,776 on Flower. Pool reservation after warming is respectively
2,818,572,288, 704,643,072, and 100,663,296 bytes. The actual default
threshold on this device is 3,220,963,328 bytes. This is retained pool
storage, not a reduction in live requested memory or a measurement of
total board/driver usage. The legacy allocator releases its arenas instead
of keeping this idle cache.

A full-process even-4K batch-size-two memory capture (one warmup and one
paired serial/batch sample, nine encodes per version) records the same 45
allocation requests and a 5,616,940,672-byte peak live requested set for
both versions. All candidate allocations use one private pool, not one
cache per lane. Its reserved high-water mark is 5,637,144,576 bytes during
concurrent work, falling to 3,187,671,040 bytes with zero utilized bytes
before destruction. Thus the release target is visibly not a live-memory
cap. No allocation remains unmatched at the end of either trace. The
explicit-trim test separately verifies zero reserved and utilized bytes
when all buffers have been released.

### Qualification and reproduction artifacts

The CUDA build passes all 59 tests; the CPU-only build passes all 47.
The new memory-pool test covers shared versus distinct policies, the forced
legacy path, unchanged default-pool settings, foreign-backend rejection,
failed allocation preserving the caller's existing buffer and statistics,
offset round trips, replacement of existing owners, queued release/reuse
without intermediate synchronization, four host threads sharing two
backends, pending work surviving backend destruction, explicit trimming
with live buffers, full cache reclamation, and weak registry lifetime.

Ten Compute Sanitizer invocations pass: memory-pool memcheck/racecheck/
synccheck/initcheck, plus backend, full AQ, and Butteraugli memcheck and
initcheck. Every memcheck uses `--track-stream-ordered-races all` and
`--leak-check full`; all report zero errors and zero leaked bytes. The
focused racecheck reports zero hazards. There are no kernel filters.
`--report-api-errors no` is used only for the pool test's deliberate
impossible allocation and the backend test's deliberate invalid launch;
their tests still validate runtime status, and memory instrumentation stays
enabled. Full-AQ shared-memory racecheck is not claimed in this cycle.
All ten runs complete in roughly 90 seconds combined; none is aborted,
left running, or observed blocked by a permission/admin prompt.

A same-executable policy probe switches only `CudaBackendOptions` between
legacy and the production pool. It writes three outputs outside the timing
boundary at both zero and three warmups. All 36 codestreams are exact
across cold allocation, repeated reuse, and allocator policy on the actual
odd-4K, odd-1080p, and Flower benchmark inputs (1,048,983, 265,570, and
37,018 bytes respectively).

All 23 parent/candidate qualification pairs have identical SHA-256,
codestream size, strategy counts, final encoder score, and independently
decoded Butteraugli score. The seven-input corpus is the 17x13 sample,
odd padded 1080p/4K, Flower, and the three provenance-pinned CC0 Wesaturate
photographs from S29, at distances 0.5/1.2/3 and effort 7, plus sample and
Flower at distance 1.2/effort 9. Decoder and metric remain pinned to libjxl
`e8ff09762481785938d8e4e01333ed3917571161`, Clang 22.1.8, with explicit
linear-sRGB decoding (`RGB_D65_SRG_Rel_Lin`) and 80-nit SDR metric input.

Exact serial/batch checks pass for 1080p batch sizes 1/2/4 in fully-resident
and maximum-throughput modes and 4K batch sizes 1/2 in fully-resident mode.
These current-policy checks establish concurrency/output behavior, not a
before/after batch-throughput gain.

Ignored `build-cuda-ninja/profiles/s32_*` artifacts retain the support
probe, isolated allocator prototype, threshold sweep, all warm/cold raw
pairs, GPU state, six single-encode Nsight captures and their API/transfer/
allocation summaries, both whole-process batch-memory captures, policy
identity probe, decoded qualification, batch logs, and exact sanitizer
commands/results. `s31_retained_{encode,benchmark,batch}.exe` preserve the
parent; `s32_retained_{encode,benchmark,batch}.exe` preserve the qualified
candidate. `s32_batch_memory_summary.py` also asserts byte-exactness and
matching strategy/score reports for all 23 qualification pairs.

The next investigation should account for remaining transfer/staging and
host work alongside resident coefficient encoding and filtering. Allocation
reuse is a warmed-latency improvement with an explicit memory tradeoff,
not proof that the resident path has reached its ceiling.

## On-demand reconstruction host staging follow-up (S33)

### Cause and retained implementation

The parent is `83200fb`, the private stream-ordered memory-pool checkpoint.
Accounting for its 4K transfers shows approximately 5.1-5.3 ms of GPU copy
time for each 33,153,604-byte input RGB plane and 16.9 ms for the final
99,532,800-byte AC coefficient readback. These bulk transfers run at
roughly 6 GB/s in that trace. The input copy is followed by a substantial
host-preparation gap; transfer API counts alone do not explain it.
`s33_transfers.{py,json}` retain the ordered GPU/API copy spans from S32.

Code inspection finds that `CudaPreparedResidentAqEvaluation::Prepare`
unconditionally allocates and value-initializes three full-resolution host
RGB vectors. Ordinary fully-resident encoding requests no reconstructed
host image, so it never reads or writes those vectors again. The required
reconstruction remains on the GPU for Butteraugli evaluation. Skipping the
already-optional readback was not enough: host allocation and zero-filling
still occurred on every encode.

The retained change leaves these vectors empty at preparation. Both ordinary
evaluation and the fused resident policy call a common helper on the first
diagnostic reconstruction request, while holding the prepared-object
mutex and before submitting the requested work. The helper constructs all
three vectors transactionally, then retains them for that object's later
evaluations. Allocation failure is reported before submitting new work;
existing output validation, finite-value checks, and atomic publication stay
in place. No caller-visible output or numerical contract is weakened, and
no production environment variable or mode switch is added.

For an encoding-only prepared object, omitted host payload is exactly
`3 * source_width * source_height * sizeof(float)`:
99,460,812 bytes at 3839x2159, 24,847,212 bytes at 1919x1079, and
3,255,840 bytes on 510x532 Flower. This is host staging, not VRAM or a
measurement of process-wide peak working set. Two simultaneous 4K encodes
omit two such host payloads. Diagnostic requests still incur and retain the
staging, and mandatory quantized-frame readback storage is unchanged.

An isolated same-executable probe restores eager RGB allocation with an
experiment-only switch and records host preparation stages. Three
alternating eager/lazy process pairs per input each use three warmups and
three samples; warmups are excluded from the stage summary. Median
per-process staging allocation/initialization time falls from 30.970 to
13.228 ms at 4K, 11.440 to 5.573 ms at 1080p, and 1.338 to 0.646 ms on
Flower. Every pair favors lazy allocation in this phase. The measured
RGB-vector capacities change from exactly the three payloads above to zero.
Metadata/setup medians are 11.065 / 11.023 ms, 3.609 / 3.088 ms, and
0.451 / 0.465 ms respectively. Staging timers end before logging and before
device arena preparation; they include the other unchanged host readback
vectors, and do not include vector destruction. This probe corroborates
the allocation/initialization cause, not a second independent whole-encode
speedup. It is run separately from qualification and ordinary timing.

### Public wall time

Seven alternating independent-process pairs use the unchanged public
in-memory fully-resident boundary at distance 1.2/effort 7. Warm measurements
use three warmups and five samples per process; each process contributes its
median. Percentages are medians of paired candidate/parent ratios, not ratios
of the displayed marginal medians. All observations are retained.

| Input | Warm total ms, parent / retained | Paired total change | Warm quantization ms, parent / retained | Paired quantization change |
| --- | ---: | ---: | ---: | ---: |
| 3839x2159 | 657.991 / 621.763 | -4.5% | 397.302 / 383.002 | -4.7% |
| 1919x1079 | 158.299 / 162.892 | -2.8% | 78.781 / 74.910 | -6.5% |
| Flower | 34.026 / 32.709 | +4.0% | 16.076 / 15.163 | -2.9% |

Every 4K and 1080p pair favors the candidate in quantization time. Total
time is noisier: its process-median ranges are 609.038-705.596 /
590.157-662.277 ms at 4K, 154.574-173.014 / 151.010-176.448 ms at 1080p,
and 31.381-37.938 / 31.083-60.404 ms on Flower. The 1080p marginal
total median worsens despite its favorable median paired ratio. Flower's
paired total result regresses despite its lower marginal median. Neither
inconsistency is hidden by selectively reporting only the favorable statistic.
The evidence supports reduced host preparation, not a uniform whole-encode
speedup on every input.

Separate first-encode measurements use seven pairs, zero warmups, and one
sample per process; backend construction remains outside the boundary.
Parent/candidate total medians are 688.119 / 674.670 ms at 4K,
192.578 / 187.323 ms at 1080p, and 50.709 / 49.617 ms on Flower.
Paired total changes are -2.1%, -2.9%, and -3.5%; quantization changes are
-1.4%, -3.4%, and -1.6%. First-encode total ranges are
676.794-771.083 / 659.683-881.879 ms, 188.831-200.277 /
182.574-229.108 ms, and 48.248-71.489 / 48.588-51.868 ms respectively.
Outliers and laptop variance remain material; these are not full CLI startup
measurements. Warm-cohort GPU samples span 66-73 C, P3, and 1282-1297 MHz
SM clocks. Power/clock settings and OS services are unchanged. No build,
sanitizer, profiler, or other GPU probe overlaps ordinary wall measurements.

### GPU accounting and qualification

Separate Nsight captures use three warmups and one captured encode with
memory tracking enabled. Both versions still issue five allocations/frees,
485 / 470 / 485 kernel launches, 31 H2D copies, 19 D2H copies, and one
D2D copy for 4K / 1080p / Flower. Exact transfer totals remain
117,079,320 / 103,699,012 / 518,400 bytes at 4K;
29,336,392 / 25,924,192 / 129,600 at 1080p; and
3,980,492 / 3,430,532 / 17,152 on Flower. Device allocation sizes,
peak logical live requests, and retained pool reservations match S32.

GPU kernel totals in the new parent/candidate captures are 265.977 /
276.857 ms at 4K, 41.430 / 43.044 ms at 1080p, and 7.810 / 7.813 ms on
Flower. No kernel-level improvement is claimed: device code is untouched,
and the instrumented timings themselves are variable. These captures
establish unchanged GPU work and transfer/memory accounting, not ordinary
wall-time speedups.

The expanded AQ test uses an internal, quiescent-object staging-capacity
query to prove zero host RGB staging after preparation and frame-only
evaluation. It then requests a strided diagnostic reconstruction, returns
to encoding-only use, and requests reconstruction again after poisoning the
caller buffer. Staging capacity is retained, pixels and padding match,
scores/block maps remain exact, and final codestreams remain byte-identical.
A submission failure after reuse leaves reconstruction, frame, quantizer,
score, and block map untouched. Existing full/bounded resident-policy and
maximum-error tests cover first-use materialization through both evaluation
entry points. Host allocation failure itself is not injected by these tests.

Both complete builds pass: 59 CUDA tests and 47 CPU-only tests. All 23
parent/candidate image pairs match SHA-256, encoded size, strategy counts,
encoder score, and independently decoded Butteraugli score. The corpus,
distances, efforts, pinned libjxl revision, explicit linear-sRGB decoding,
and 80-nit metric setup are unchanged from S32. These are output-identity
checks, not quality tolerance exceptions.

Serial/batch output identity passes at 1080p batch sizes 1/2/4 for both
fully-resident and maximum-throughput modes and at 4K sizes 1/2 for
fully-resident mode. These current-policy checks do not establish a
before/after batch-throughput improvement. `s33_validate_evidence.py`
additionally compares exact ordered kernel names, grid/block dimensions,
register counts, static/dynamic shared memory, local-memory use, transfers,
allocation sizes, and pool reservations in the paired captures.

The expanded full AQ test passes Compute Sanitizer memcheck, initcheck,
and synccheck with zero errors. Memcheck also enables
`--track-stream-ordered-races all --leak-check full` and reports zero leaked
bytes. There are no kernel filters or suppressed API errors. These three
runs take approximately 39, 27, and 24 seconds. Full shared-memory racecheck
is not claimed in this host-only change. No sanitizer is aborted or left
running, and no permission/admin prompt or unexplained stall is observed.

Ignored `build-cuda-ninja/profiles/s33_*` artifacts retain transfer accounting,
all warm/cold pairs, GPU samples, paired Nsight captures and summaries,
decoded qualification, serial/batch results, exact sanitizer commands/logs,
the same-executable host-stage probe and its raw observations, and the
evidence assertions. The isolated source/build/measurement files are
`s33_host_probe.cpp`, `s33_build_host_probe.ps1`, and
`s33_host_probe_measure.py`. `s32_retained_{encode,benchmark,batch}.exe`
preserve the parent and `s33_retained_{encode,benchmark,batch}.exe` preserve
the qualified candidate.

Remaining host readback staging still takes about 13 ms of allocation/
initialization in the isolated 4K probe, alongside roughly 11 ms of
metadata/setup. These are investigation targets, not proven removable costs:
deferring initialization may simply move page-fault costs into readback.
Coefficient handoff/assembly, metadata construction, remaining filters, and
codestream host work remain material. The resident path is not maxed out.

## Overwrite-only coefficient host staging follow-up (S34)

### Cause and retained implementation

The parent is `b84ae35`. After eliminating unused reconstructed-RGB host
staging in S33, the required coefficient staging still initializes a full
`3 * padded_width * padded_height` array of 32-bit coefficients to zero.
Frame materialization then overwrites the entire array with a synchronous
device-to-host batch before any host consumer can read it. This clears
99,532,800 bytes at padded 4K, 24,883,200 bytes at padded 1080p, and
3,293,184 bytes for Flower's 512x536 coding image, despite all those values
being replaced by readback.

The retained implementation uses an owning
`std::make_unique_for_overwrite<int32_t[]>` array and an explicit span at
frame assembly. The allocation still occurs during preparation and remains
available for repeated materialization. Only value initialization is omitted:
buffer size, ownership, readback byte count, synchronous completion checking,
unwritten-coefficient validation, and final frame assembly remain unchanged.
Failure returns before reading the staging or publishing a new frame.
This removes a host write pass, not the staging allocation, required PCIe
transfer, or final frame-layout copy. It is not a 99.5 MB memory saving.

An internal test hook poisons every host coefficient with the unwritten
sentinel before first materialization and repeated frame-only/diagnostic
evaluations. Successful assembly and exact output comparisons require that
readback replace the poison. Existing failure-atomicity, resident policy,
maximum-error, and concurrent workflow tests remain applicable. The poison
hook requires a quiescent prepared object and is not a public encoder mode.

### Isolated experiments and rejected assembly change

A same-executable prototype changes only initialized-vector versus
overwrite-only coefficient staging. Three alternating process pairs per
input use three warmups and three samples, with host staging, synchronous
readback, and frame assembly measured separately. The combined metric sums
those three stages per encode before taking a process median, so shifting
cost between stages cannot masquerade as a saving.

| Input | Staging ms, initialized / overwrite | Readback ms | Assembly ms | Combined ms |
| --- | ---: | ---: | ---: | ---: |
| 3839x2159 | 13.219 / 0.936 | 16.145 / 19.401 | 32.716 / 30.309 | 62.090 / 50.544 |
| 1919x1079 | 5.037 / 0.319 | 4.145 / 5.433 | 10.328 / 10.513 | 19.432 / 16.393 |
| Flower | 0.654 / 0.071 | 0.684 / 0.911 | 1.350 / 1.326 | 2.782 / 2.493 |

These are marginal medians of the per-process observations; displayed stage
medians need not sum to the displayed combined median. Readback becomes
slower, so the staging-only reduction is not the net gain. Nevertheless,
every one of the nine pairs improves the combined host-stage metric. Probe
timers exclude logging; its broader wall observations remain in the raw
artifact but are not substituted for ordinary production measurements.

A separate prototype removes the final frame's initial full-array zeroing
without changing its fixed-capacity group/channel format. It buckets
validated transform references by AC group, appends coefficients in final
group/channel order, and zeros only unused edge-group tails. This retains
standard vector ownership and avoids changing the frame ABI. The existing
frame unit tests pass in a separately linked probe, and natural-image
pipeline observations initially look favorable at 1080p. The 4K host-stage
result is mixed, prompting a wider assembly-only check before retention.

The assembly-only benchmark compares three alternating process pairs,
three warmups and five samples, for all-DCT8, DCT32-tiled with DCT8 edge
remainder, and mixed layouts. Mixed tiles cycle through seven supported
strategy shapes and fill the remaining blocks with DCT8. It verifies frame
validity and matching whole-coefficient 64-bit checksums, including zero
tails. Median paired assembly-time changes for the proposed append policy:

| Source extent | All DCT8 | DCT32-tiled | Mixed |
| --- | ---: | ---: | ---: |
| 510x532 | +15.0% | -12.5% | -10.1% |
| 1919x1079 | -4.8% | -20.6% | -4.2% |
| 3839x2159 | +14.8% | +1.5% | +18.9% |

The new grouping/append path is not retained. Its wider 4K regressions
outweigh the narrow favorable observations. No common frame-assembly or
frame-layout source changes are included in this checkpoint. A future
direct readback into the final layout may avoid the intermediate copy, but
must account for group tails, transfer count, GPU packing work, and atomic
failure handling; the rejected append experiment does not establish that
such a design wins.

### Production wall measurements

The retained binary and the frozen S33 parent run seven alternating process
pairs per input, with three warmups and five samples in each process. The
public fully-resident boundary remains linear RGB through an in-memory
codestream at distance 1.2 / effort 7, excluding backend construction,
input/file I/O, and optional final-score diagnostics. Negative paired change
is faster. Paired changes are medians of per-pair ratios, not ratios of the
displayed marginal medians.

| Input | Total ms, parent / retained | Paired total change | Quantization ms | Paired quantization change |
| --- | ---: | ---: | ---: | ---: |
| 3839x2159 | 645.112 / 632.266 | -0.8% | 381.775 / 372.292 | -2.2% |
| 1919x1079 | 156.539 / 152.713 | -0.5% | 74.476 / 70.396 | -5.0% |
| Flower | 34.723 / 36.041 | +4.8% | 16.005 / 16.221 | +2.2% |

Six of seven 4K quantization pairs improve, as do all seven 1080p pairs.
Whole-encode results are less consistent: parent/retained process-median
total ranges are 601.423-682.121 / 603.355-654.731 ms at 4K,
153.180-170.467 / 149.260-159.676 ms at 1080p, and 30.767-38.426 /
30.264-46.341 ms on Flower. The Flower regression is not omitted or used
to select an input-size threshold. The same-executable host-stage probe
supports removing the redundant clear; these ordinary production cohorts
do not establish a stable whole-encode speedup.

Separate first-encode measurements use seven pairs with zero warmups and
one sample per process; backend construction is still excluded. Total
parent/retained medians are 699.431 / 680.891 ms at 4K, 191.775 / 188.822 ms
at 1080p, and 48.633 / 52.557 ms on Flower. Paired total changes are -2.7%,
-1.5%, and +6.8%; paired quantization changes are -1.9%, -4.0%, and +4.6%.
Total ranges are 691.137-761.165 / 646.591-784.791 ms,
187.170-218.447 / 181.668-197.474 ms, and 46.205-52.225 /
49.137-60.822 ms respectively. These are not full CLI startup measurements.

Warm-cohort GPU samples span 66-73 C, P3, and 1282 MHz SM clocks. No clock,
power, firewall, or OS-service settings are changed. No builds, tests,
profilers, sanitizers, or other GPU experiments overlap ordinary timing.

### GPU accounting and qualification

Separate parent/retained Nsight captures use three warmups, one captured
encode, and memory tracking. Exact ordered kernel names, launch geometry,
register/shared/local-memory use, allocation sizes, transfer counts/bytes,
peak logical requested memory, and retained pool reservations match. There
are still five allocations/frees and 485 / 470 / 485 launches for 4K /
1080p / Flower. Every encode has 31 H2D, 19 D2H, and one D2D copy. Byte
totals remain 117,079,320 / 103,699,012 / 518,400 at 4K;
29,336,392 / 25,924,192 / 129,600 at 1080p; and
3,980,492 / 3,430,532 / 17,152 on Flower. Live requests and pool
reservations remain the values documented in S32.

Captured GPU kernel totals are 262.270 / 277.941 ms at 4K,
42.965 / 41.645 ms at 1080p, and 7.808 / 7.818 ms on Flower. Device code
is unchanged, and these variable instrumented observations are not evidence
of a kernel-level gain. They verify unchanged GPU work and memory/transfer
accounting.

The CUDA build and expanded poisoned-readback AQ test pass. All 59 CUDA
tests and all 47 CPU-only tests pass; common CPU sources are unchanged.
The poison test covers first materialization and reuse, both frame-only and
diagnostic reconstruction, with exact codestream/score/map comparisons.
Existing reconstruction padding, transactional failure, resident-policy,
maximum-error, and concurrent public-workflow checks also pass. Host
allocation failure itself is not injected.

All 23 parent/retained image pairs match SHA-256, encoded bytes, strategy
counts, final encoder score, and independently decoded Butteraugli score.
The seven-image corpus, distances 0.5 / 1.2 / 3 at effort 7, additional
sample/Flower distance-1.2 effort-9 cases, pinned libjxl revision, explicit
linear-sRGB decoding, and 80-nit metric setup remain unchanged from S32.
The actual odd synthetic benchmark inputs differ slightly from the quality
PFMs; the quality corpus is not presented as a byte comparison of those
benchmark inputs.

Serial/batch exact-output checks pass for 1080p sizes 1/2/4 in both
fully-resident and maximum-throughput modes, and for 4K sizes 1/2 in
fully-resident mode. Their median paired speedups are 0.958x / 1.268x /
1.437x, 0.976x / 1.501x / 1.672x, and 1.047x / 1.091x respectively.
These compare serial and batch operation of the current implementation,
not before/after batch throughput. `s34_validate_evidence.py` verifies the
exact paired profile structure and all 23 output/score identities.

The complete expanded AQ test passes Compute Sanitizer memcheck, initcheck,
and synccheck with zero errors. Memcheck enables
`--track-stream-ordered-races all --leak-check full` and reports zero leaked
bytes. No kernel filter or API-error suppression is used. The three runs
take approximately 36, 24, and 19 seconds; full shared-memory racecheck is
not claimed for this host-only change. All qualification jobs exit normally,
with no permission/admin prompt, unexplained stall, or aborted sanitizer.

Ignored `build-cuda-ninja/profiles/s34_*` artifacts retain the paired
warm/cold measurements, GPU samples, Nsight captures and accounting,
decoded qualification, serial/batch results, CTest archives, exact
sanitizer commands/logs, and evidence assertions. The isolated experiments
retain `s34_readback_probe.cpp`, `s34_frame_probe.cpp`,
`s34_assembly_benchmark.cpp`, `s34_build_probes.ps1`, and the corresponding
readback/frame/assembly measurement scripts and JSON results.
`s33_retained_{encode,benchmark,batch}.exe` preserve the parent;
`s34_retained_{encode,benchmark,batch}.exe` preserve the qualified candidate.

The required coefficient readback, CPU frame-layout copy/validation,
metadata setup, remaining filtering, and codestream host work remain
material. Neither a uniform whole-encode gain nor a performance ceiling is
demonstrated by this checkpoint.

## Direct resident transform image I/O follow-up (S35)

### Cause and implementation

The parent is `5aefb93`. Although AC-search DCTs already consume image
rectangles directly, final mixed-strategy resident reconstruction still
gathers coding pixels into a packed array before its forward DCT. Each
inverse DCT also writes a packed pixel array, which a separate scatter
kernel copies to the reconstructed image. These are real consumers, not
unused allocations left after the earlier AC-search fusion.

The parent 4K trace has seven gathers and fourteen scatters. Those copies
alone take 1.487 and 4.046 ms respectively. The two arrays each hold
`3 * padded_width * padded_height` floats: together 199,065,600 bytes at
padded 4K, 49,766,400 at padded 1080p, and 6,586,368 for Flower's 512x536
coding image. A one-forward/two-inverse encode writes and rereads a packed
full-image array three times, or 597,196,800 bytes of avoidable device-memory
traffic at 4K. This is device traffic, not a PCIe transfer saving.

The candidate adds resident image input/output accessors to the existing
factorized DCT kernels. The forward accessor maps a channel-major batch's
transform index to its validated anchor rectangle. Inactive packed lanes do
not fetch anchors. The inverse writes its register-held pixel columns
directly to distinct image rectangles; tall transforms do not need the
extra redistribution used by AC-search loss reduction. Arithmetic, scaling,
coefficient order, reconstruction pixels, and stream ordering are preserved.

The prepared resident arena no longer plans or allocates gathered or inverse
pixel arrays. Forward coefficients are still cached across evaluations;
the coefficient encoder, filters, required host readbacks, and CPU frame
assembly remain unchanged. Host metadata validation still establishes full,
nonoverlapping coverage and valid supported shapes. GPU submission failure
still returns before any caller output is published. Exact-coefficient and
maximum-throughput modes retain their previous paths.

The first candidate's compiler resource report exposed a 72-byte stack
frame in every new kernel. Its image accessor used `planes[channel]` on a
by-value argument containing the three plane pointers. A 32x32 forward
SASS inspection showed nine 64-bit local stores and a dynamic local load.
Selecting `planes[0]`, `[1]`, or `[2]` through explicit channel branches
removes the frame in all fourteen new variants. This is fixed-field access,
not a change in DCT arithmetic or transform geometry. The final binary has
zero compiler-reported stack and local storage for these variants; register
counts span 28-64, and shared memory remains 2,176-8,448 bytes per block.
For example, the 32x32 inverse uses 64 registers instead of the old
contiguous kernel's 40. Reduced copies do not imply unchanged register
pressure on other GPUs.

The first candidate is preserved as `s35_initial_{encode,benchmark,batch}.exe`
with its fully qualified `s35_*` measurements. It is not confused with the
fixed-field final candidate. The following initial-candidate subsections
preserve those observations; the final fixed-field results below supersede
them.

### Differential tests and initial performance experiments

`cuda_resident_dct` compares direct image transforms with the old gather /
contiguous DCT / scatter composition in the same executable. It covers all
seven resident shapes, anchor counts 1/2/3/5/9/17, and six input patterns:
signed zero, per-channel constants, impulses, small random values, large
random values, and ordinary random values. Shuffled anchors, gaps between
rectangles, nonzero metadata/coefficient/pixel offsets, different input and
output strides, partial packed blocks, and poisoned/guarded storage exercise
layout boundaries. Inverse inputs also contain arbitrary quantized-like
coefficients rather than only a forward/inverse round trip. All 252 cases
match bit-for-bit, including repeated inverse use, exact CPU reconstruction
of the scatter layout, unchanged inputs, and untouched guards. Supported
empty batches are no-ops and an unsupported 64x64 shape is rejected.

An isolated event-timed probe compares the two compositions with identical
buffers and arithmetic, including gather/scatter in the old stage. It uses
five alternating policy pairs, twenty warmups and twenty-one samples per
policy, three repetitions of the full composition per timed sample. Each of
the seven shapes tiles 512x536 and 3840x2160 source extents; partial bottom
tiles are omitted, and image row stride is source width plus thirteen.
Consequently these are homogeneous transform workloads, not public encodes
or exact replicas of the natural-image strategy mix. Whole-output bitwise
checks precede timing. Median paired stage changes are:

| Shape | 512x536 forward / inverse | 3840x2160 forward / inverse |
| --- | ---: | ---: |
| 8x8 | -8.4% / -36.7% | -39.6% / -40.2% |
| 16x8 | -2.1% / -40.4% | -35.2% / -38.5% |
| 8x16 | -11.7% / -29.7% | -30.0% / -33.1% |
| 16x16 | -27.4% / -45.6% | -47.0% / -44.6% |
| 32x16 | -20.4% / -50.9% | -37.3% / -48.8% |
| 16x32 | -27.2% / -42.8% | -44.2% / -39.3% |
| 32x32 | -37.5% / -51.4% | -51.0% / -58.0% |

Every shape/direction median favors fusion. For example, at 4K the marginal
32x32 forward medians are 5.393 / 2.642 ms and inverse medians are
5.970 / 2.507 ms. Those times include the old copy kernels, and are not
substituted for public-workflow measurements.

### Initial production wall measurements

Seven alternating parent/retained process pairs per input use three warmups
and five samples, distance 1.2 / effort 7, fully-resident mode, and the usual
linear-RGB-to-in-memory-codestream boundary. Backend construction, input
generation/file I/O, and optional final-score diagnostics are excluded.
Paired changes are medians of per-pair ratios, not ratios of the displayed
marginal medians. Negative is faster.

| Input | Total ms, parent / retained | Paired total change | Quantization ms | Paired quantization change |
| --- | ---: | ---: | ---: | ---: |
| 3839x2159 | 672.479 / 665.595 | +0.3% | 399.306 / 385.861 | -3.3% |
| 1919x1079 | 158.395 / 161.426 | -0.4% | 72.730 / 72.338 | -0.5% |
| Flower | 31.782 / 31.815 | +0.1% | 14.736 / 14.549 | -1.2% |

All seven 4K quantization pairs favor fusion. Whole-encode changes are small
and inconsistent: no stable end-to-end gain is claimed. Parent/retained
total process-median ranges are 643.803-683.838 / 645.438-690.971 ms,
150.792-231.029 / 154.609-175.970 ms, and 30.942-43.683 /
31.234-43.598 ms respectively. The 1080p marginal median worsens despite a
slightly favorable paired median; both are retained in the report.

Warm-cohort GPU samples span 73-75 C, P3, and 1282-1770 MHz SM clocks. No
clock/power/OS-service settings are changed. The isolated probe, ordinary
wall measurements, builds/tests, profiling, and sanitizers run sequentially.
These laptop/device/corpus observations do not establish a universal gain.

First-encode measurements use seven alternating pairs, zero warmups and
one sample per process, still excluding backend construction. Total
parent/retained medians are 708.611 / 725.765 ms at 4K,
193.567 / 190.612 ms at 1080p, and 52.710 / 55.162 ms on Flower. Paired
total changes are +2.8%, -2.7%, and +10.8%; quantization changes are
-3.3%, -4.6%, and +5.6%. Total ranges are 675.091-764.822 /
658.091-809.370 ms, 183.401-272.940 / 184.602-204.074 ms, and
47.621-59.695 / 51.996-76.813 ms respectively. No cold-start improvement
is claimed, and the slower Flower observations are not discarded.

### Initial production GPU and memory accounting

Parent/retained Nsight captures use three warmups and one captured encode
with memory tracking. Total launches fall 485 to 464 at 4K, 470 to 452 at
1080p, and 485 to 464 on Flower. The gather/scatter copies disappear, while
each resident transform keeps its existing grid and block dimensions. Other
kernel names/order, geometry, register/shared-memory counts and per-thread
local-memory use match exactly. The direct transform subset measures:

| Input | Old gather/DCT/inverse/scatter ms | Direct image DCT/inverse ms | Change |
| --- | ---: | ---: | ---: |
| 3839x2159 | 7.866 | 3.670 | -53.3% |
| 1919x1079 | 1.356 | 0.845 | -37.7% |
| Flower | 0.212 | 0.156 | -26.4% |

Total captured GPU kernel time is 266.663 / 281.203 ms at 4K,
41.335 / 41.456 ms at 1080p, and 7.815 / 7.749 ms on Flower. Thus the
targeted subset improves, but total GPU results are mixed; the unfavorable
4K total is not replaced by the targeted result. These instrumented
observations and ordinary wall measurements have different boundaries.

Five allocations/frees remain. Only the resident staging arena shrinks;
the other four sizes match. Peak tracked live allocation requests are:

| Input | Parent bytes | Retained bytes | Reduction |
| --- | ---: | ---: | ---: |
| 3839x2159 | 2,807,513,664 | 2,608,448,064 | 199,065,600 |
| 1919x1079 | 701,724,038 | 651,957,638 | 49,766,400 |
| Flower | 92,405,776 | 85,819,408 | 6,586,368 |

Pool reservations are distinct from live requests: 2,818,572,288 /
2,617,245,696 bytes at 4K, 704,643,072 / 671,088,640 at 1080p, and
100,663,296 / 100,663,296 on Flower. The shared default retention threshold
remains 3,220,963,328 bytes on this device; smaller requests need not change
the retained allocation granularity.

Transfers remain exactly 31 H2D, 19 D2H, and one D2D copy, with byte totals
117,079,320 / 103,699,012 / 518,400 at 4K;
29,336,392 / 25,924,192 / 129,600 at 1080p; and
3,980,492 / 3,430,532 / 17,152 on Flower. Removing the gather/scatter
kernels does not remove a host readback or a CUDA memcpy.

### Initial correctness qualification

Both complete builds pass: 60 CUDA tests and 47 CPU-only tests, including
the new 252-case transform differential test and the existing resident
policy, maximum-error, diagnostic reconstruction, failure-atomicity and
concurrent public-workflow checks. No CPU implementation is changed.

All 23 parent/retained image pairs match SHA-256, encoded size, strategy
counts, final encoder score, and independently decoded Butteraugli score.
The seven-image corpus, distances 0.5 / 1.2 / 3 at effort 7, additional
sample/Flower distance-1.2 effort-9 cases, pinned libjxl revision, explicit
linear-sRGB decoding and 80-nit metric setup remain unchanged from S32.
The synthetic quality PFMs differ slightly from the actual odd benchmark
inputs; their byte comparisons are not presented as comparisons of those
benchmark inputs.

Serial/batch exact-output checks pass at 1080p sizes 1/2/4 for both
fully-resident and maximum-throughput modes and at 4K sizes 1/2 for
fully-resident mode. Median paired serial/batch speedups are 1.021x /
1.122x / 1.198x, 1.011x / 1.936x / 2.048x, and 0.930x / 1.070x
respectively. These current-policy serial/batch comparisons do not establish
a before/after batch-throughput improvement.

The complete AQ test passes Compute Sanitizer memcheck, initcheck, and
synccheck; memcheck also uses `--track-stream-ordered-races all --leak-check
full`. All report zero errors and memcheck reports zero leaked bytes. The
focused 252-case resident-DCT test passes all four tools: memcheck,
initcheck, synccheck, and racecheck, with zero errors or race hazards.
There are no kernel filters or API-error suppressions. Full-AQ
shared-memory racecheck is not claimed; the focused test covers the new
shared-memory kernel instantiations. Full-AQ sanitizer durations are about
40 / 26 / 22 seconds and focused durations are 4 / 4 / 5 / 15 seconds.

### Final fixed-field measurements

The final binary uses explicit selection of the three plane-pointer fields;
the initial stack-generating accessor is not retained. The same-executable
event probe is rebuilt against this binary's kernels, with identical
workloads, warmups, pairing and sample counts. Its paired stage changes are:

| Shape | 512x536 forward / inverse | 3840x2160 forward / inverse |
| --- | ---: | ---: |
| 8x8 | -51.5% / -67.2% | -58.7% / -71.8% |
| 16x8 | -56.7% / -63.6% | -56.3% / -61.9% |
| 8x16 | -52.0% / -60.4% | -62.0% / -64.8% |
| 16x16 | -50.5% / -64.0% | -58.9% / -67.9% |
| 32x16 | -42.9% / -61.6% | -55.2% / -66.0% |
| 16x32 | -47.1% / -58.6% | -64.7% / -62.8% |
| 32x32 | -48.0% / -61.2% | -61.4% / -67.7% |

The final 4K 32x32 marginal stage medians are 5.480 / 2.113 ms forward
and 5.971 / 1.923 ms inverse. Small event-timed cases remain variable: for
512x536 8x8 forward, marginal medians are 0.068 / 0.090 ms despite a
favorable paired ratio of 0.485. One of its five pairs regresses; the two
later old-policy observations are approximately 0.257 ms. These raw paired
observations are preserved, not reduced to an assertion that every sample
improves. The event probe is not a public-workflow speedup measurement.

The final public-workflow cohort repeats seven alternating pairs, three
warmups and five samples with the same S34 parent and encoding boundary:

| Input | Total ms, parent / final | Paired total change | Quantization ms | Paired quantization change |
| --- | ---: | ---: | ---: | ---: |
| 3839x2159 | 733.879 / 698.971 | -4.4% | 416.412 / 400.087 | -2.7% |
| 1919x1079 | 176.403 / 167.755 | -2.5% | 76.832 / 72.660 | -5.2% |
| Flower | 32.691 / 32.581 | -1.0% | 15.161 / 14.927 | -1.4% |

All seven 4K quantization pairs improve, as do six of seven at 1080p.
Total parent/final ranges are 696.206-800.037 / 670.041-747.610 ms,
156.832-181.320 / 156.244-228.559 ms, and 31.587-53.883 /
32.175-55.373 ms respectively. Warm-cohort GPU samples span 74-77 C, P3,
and 1282-1440 MHz SM clocks. The initial cohort's flat total times and
these final favorable but noisy pairs are both reported; there is no
same-executable public-workflow comparison isolating the accessor revision.
Thus the study establishes a targeted transform/data-movement improvement,
not a uniform whole-encode speedup independent of system state.

The final first-encode cohort repeats seven pairs, zero warmups and one
sample. Parent/final total medians are 720.511 / 704.582 ms at 4K,
193.418 / 194.533 ms at 1080p, and 55.008 / 53.803 ms on Flower. Paired
total changes are -0.2%, -1.2%, and -2.6%; quantization changes are -2.6%,
-3.3%, and -3.0%. Total ranges are 686.186-801.927 / 673.100-827.292 ms,
187.961-215.678 / 183.152-225.718 ms, and 52.952-68.430 /
51.742-62.959 ms. Backend construction is excluded; neither these small
changes nor the initial cold regressions establish a stable startup gain.

Final single-encode profiles confirm the same allocation/request reductions,
pool reservations, launch reductions and exact transfer counts/bytes listed
above. The targeted old/direct-image subset is 8.047 / 2.665 ms at 4K
(-66.9%), 1.376 / 0.631 ms at 1080p (-54.2%), and 0.212 / 0.114 ms on
Flower (-46.1%). Total GPU kernel time is 272.723 / 282.531 ms,
41.102 / 40.242 ms, and 7.813 / 7.713 ms respectively. The larger 4K
total despite the smaller targeted subset motivates additional reversed-order
profile pairs; the targeted result alone is not used to claim total-GPU
improvement.

Three additional 4K profile pairs alternate final-first, parent-first, and
final-first. All per-version ordered kernel names, geometry/resources and
transfer totals match the first pair. Across the four pairs, targeted old /
final times are 8.047 / 2.665, 10.725 / 2.620, 9.106 / 2.695, and
7.691 / 2.625 ms. All favor fusion; the median paired reduction is 68.6%.
Total GPU times are 272.723 / 282.531, 318.844 / 270.923,
294.207 / 287.250, and 276.239 / 277.540 ms. The median paired total
change is -0.9%, with two pairs slower and two faster. These repeats do
not establish a consistent total-GPU regression or a uniform speedup. The
large parent variation and mixed untargeted-kernel totals are retained in
`s35_final_profile_repeat.json`.

The final 60-test CUDA and 47-test CPU suites pass, including all 252
guarded bitwise transform cases. All 23 decoded image pairs again match
SHA-256, byte count, strategy counts and both encoder/decoded scores.
Repeated serial/batch output-identity checks pass in both modes at 1080p
sizes 1/2/4 and in fully-resident mode at 4K sizes 1/2. Final median paired
serial/batch speedups are 0.975x / 1.157x / 1.316x, 0.785x / 1.562x /
1.724x, and 0.805x / 1.120x respectively. These retain the same limited
meaning as the initial serial/batch checks, not before/after throughput.

Final two-image 3840x2160 memory captures reduce peak tracked live requests
from 5,616,940,672 to 5,218,809,472 bytes: 398,131,200 bytes, exactly twice
the per-image reduction. Peak pool reservation falls from 5,637,144,576 to
5,234,491,392 bytes. Each capture uses one shared private pool and retains
the same 3,220,963,328-byte release threshold. These full-process captures
include warmups and serial/batch work; they are allocation evidence, not
ordinary throughput measurements. Requested allocations exclude other CUDA
driver/module reservations.

The final complete AQ test again passes memcheck with stream-ordered race
tracking and full leak checking, initcheck, and synccheck. The final focused
resident-DCT test again passes memcheck, initcheck, synccheck, and racecheck.
All report zero errors, zero leaks where checked, and zero race hazards.
No filters or API-error suppression are used. Final full-AQ runs take
approximately 38 / 25 / 19 seconds, and focused runs 4 / 4 / 5 / 15 seconds.
The fourteen new kernel variants all report `STACK:0` and `LOCAL:0` in
`cuobjdump`; a zero Nsight per-thread local field alone was not accepted as
proof, as the initial 72-byte stack-frame discovery demonstrates.

`s35_final_validate_evidence.py` checks exact remaining-kernel order,
geometry/resources, transform launch geometry, transfer counts/bytes,
allocation deltas, all 23 image/score identities, compiler stack/local
resources, and the two-image allocation reduction. Ignored
`build-cuda-ninja/profiles/s35_*` artifacts preserve the initial candidate;
`s35_final_*` preserve the final probes, raw paired observations, GPU samples,
captures, image qualification, serial/batch checks, CTest archives, sanitizer
commands/logs, resource reports, and evidence assertions. The probe source
includes the guarded test helpers and invokes production kernels; its
separate build script records the exact compiler/link command. The frozen
S34 binaries remain the parent; `s35_retained_{encode,benchmark,batch}.exe`
preserve the qualified final implementation.

All qualification and profiling jobs exit normally. No sanitizer is aborted
or left running, and no permission/admin prompt, permission error, or
unexplained stall is observed. No firewall, OS-service, or power/clock
settings are changed.

The remaining resident coefficient-encoding kernel takes about 32 ms in the
initial S35 4K trace. It repeatedly evaluates small DC conversion bases with
runtime cosine calls for only 1/2/4-element dimensions, which warrants an
isolated caching/specialization experiment rather than assuming the whole
cost is removable. Host coefficient handoff/assembly, perceptual filtering,
metadata and codestream work also remain substantial. The resident path is
not demonstrated maxed out.

## Resident coefficient basis/local-access follow-up (S36)

Baseline: `5ecb5f8` (S35). This experiment targets
`EncodeResidentCoefficientsKernel`, not the exact-coefficient evaluator or
the DCT8-only maximum-throughput path. The initial S36 parent 4K capture
spends 28.94 ms in its 21 launches. Each anchor block performs DC extraction,
DC quantization, Y and X/B AC quantization/reconstruction, color-correlation
restoration, and low-low-frequency reconstruction.

### Cause, candidates, and retained mechanism

DC conversion only uses dimensions 1, 2, and 4, but the original kernel
re-evaluates the same forward/inverse cosine bases for different channels
and samples. The retained implementation cooperatively constructs four
padded 4x4 shared tables (256 bytes per block), using the **original** basis
functions. The existing pre-DC barrier publishes the tables; no barrier or
launch is added. The original scale multiplications and nested reduction
orders are unchanged. In particular, no idealized identity/rational basis,
host-computed cosine table, fast cosine intrinsic, or global fast-math flag
replaces the original FP32 arithmetic.

Caching alone is insufficient. The first candidate passes all 280 guarded
cases and reduces the production coefficient subset by 22.8% / 20.9% /
15.8% in the initial 4K / 1080p / Flower captures, but regresses every large
homogeneous probe by paired medians of 3.9-26.9%. These observations are
retained, not discarded in favor of the smaller tests that improve.

Native code inspection identifies a second cost: dynamic access to local
scale/bias arrays, the four copied Y thresholds, and the by-value parameter
object's eight-entry sharpness array. The original kernel reports 50
registers, a 112-byte stack frame, and no static shared memory. Caching alone
uses 52 registers and the same 112-byte frame. Direct scale/bias selection
and reading the chosen threshold from its original device slot reduce the
frame to 96 bytes, but still leave large rectangular probe regressions.
Selecting the original sharpness parameter slots directly prevents a
per-thread copy of the by-value parameter object and reduces the frame to
32 bytes. This is the retained combined candidate: 50 registers and 256
bytes of static shared memory.

The final disassembly removes the unconditional parameter/threshold array
stores. Remaining local loads/stores occur in cosine range-reduction paths
guarded by `abs(angle) >= 105615`; valid 1/2/4-element basis arguments are
below 9. This supports attributing the residual 32-byte frame to the
unreachable-for-valid-bases large-argument path, not claiming that the
compiler reports zero stack. Both original and retained kernels report
`LOCAL:0`, so that field alone would have missed the original local-array
problem.

The internal reference launcher preserves the original uncached bases and
dynamic indexing. A temporary direct-access-only variant, without shared
basis caching, is also measured: it improves all large probe medians by
15.6-58.2%, confirming that removal of local copies is independently useful.
The experimental launcher is not retained in production source; its probe
binary and source snapshot remain under the ignored profiles directory.

### Isolated kernel evidence

The same-executable probe compares the original and optimized coefficient
kernels using default quantization matrices, deterministic nonzero inputs,
valid nonoverlapping anchors with gaps, and all six output/error arrays.
Allocation, initialization, and readback verification are outside CUDA event
timing. Each shape/extent uses five alternating-order pairs, ten warmups,
eleven timed samples, and three launches per sample. Requested extents are
512x536 and 3840x2160; only complete homogeneous transform tiles are used,
so these are kernel probes, not whole-image encoder measurements.

Large-probe median paired candidate/reference ratios across successive
experiments are:

| Transform | Basis cache only | Cache + partial direct access | Direct access only | Final combined |
|---|---:|---:|---:|---:|
| 8x8 | 1.0540 | 0.8518 | 0.4178 | 0.5442 |
| 16x8 | 1.0636 | 0.9032 | 0.4727 | 0.5483 |
| 8x16 | 1.0388 | 0.9284 | 0.4749 | 0.5535 |
| 16x16 | 1.0564 | 0.8674 | 0.5828 | 0.5669 |
| 32x16 | 1.2687 | 1.3136 | 0.8431 | 0.5510 |
| 16x32 | 1.2018 | 1.2505 | 0.8339 | 0.5236 |
| 32x32 | 1.1654 | 1.0291 | 0.7782 | 0.6115 |

Each column has its **own paired reference cohort**; ratios from different
columns are not a direct comparison of candidate variants at equal clocks.
The earlier fully direct combined cohort also improves every large-shape
median, by 17.2-55.9%. The final relinked probe improves large-shape medians
38.9-47.6% and small-shape medians 36.1-48.4%. Raw observations and marginal
medians are preserved because laptop operating state varies substantially.

### Qualification and whole-encoder measurements

The first warm cohort uses seven alternating-order pairs, three warmups,
five samples, and the complete in-memory fully-resident encode boundary.
The cold cohort uses seven pairs, zero warmups, and one sample. Parent and
candidate are separate executable runs; percentages below are medians of
paired ratios, not ratios of the displayed marginal medians.

| Cohort/input | Total parent/new (ms) | Paired total change | Quantization parent/new (ms) | Paired quantization change |
|---|---:|---:|---:|---:|
| First warm 4K | 733.373 / 726.928 | -0.9% | 264.791 / 244.718 | -3.0% |
| First warm 1080p | 346.758 / 395.579 | +14.8% | 106.883 / 115.069 | -0.3% |
| First warm Flower | 40.910 / 60.583 | +39.5% | 17.556 / 20.527 | +15.9% |
| Repeat warm 4K | 712.525 / 749.250 | -0.6% | 397.053 / 387.154 | -1.7% |
| Repeat warm 1080p | 213.790 / 166.238 | -17.6% | 78.355 / 72.352 | -7.2% |
| Repeat warm Flower | 35.124 / 38.622 | -5.3% | 15.524 / 16.541 | -6.0% |
| Cold 4K | 819.283 / 780.561 | -1.1% | 479.950 / 483.147 | -0.03% |
| Cold 1080p | 382.702 / 400.067 | +4.5% | 144.770 / 160.981 | -8.2% |
| Cold Flower | 73.334 / 64.626 | -9.5% | 38.126 / 31.974 | -11.0% |

The first warm cohort contains substantial regressions at the smaller
inputs. An additional seven-pair cohort with reversed starting order uses
the same binaries, settings, three warmups, and five samples. It reverses
the smaller-input regressions while 4K total time stays nearly flat. Both
cohorts are retained; neither the first regressions nor the later gains
alone establish stable end-to-end behavior. Marginal and paired medians
even disagree in sign for repeat 4K and Flower, as shown above.
Timing spread is unusually large in **both** versions: first-warm 4K
total ranges are 432.036-1370.424 / 430.164-1078.381 ms, and 1080p ranges
227.662-522.655 / 194.621-551.277 ms. GPU snapshots span 79-82 C, P0, and
1282-1755 MHz. A read-only host snapshot finds about 21 GiB of free physical
memory; it does not establish the cause of the timing variation. No clocks,
power policy, firewall, process priorities, or other OS settings are changed.
Repeat snapshots span 71-77 C, P3, and 1282-1477 MHz; repeat 4K total ranges
are 688.425-814.923 / 689.047-783.388 ms. No stable end-to-end or cold-start
gain is claimed.

Stage reports help localize, but do not establish the cause of, the wall
variation. First-warm 1080p CPU codestream generation has paired median
regression 21.7% (marginal medians 224.545 / 259.826 ms), while quantization
is nearly flat by paired median. Flower CPU codestream generation regresses
44.6% in that cohort. For example, 1080p pair 4 changes total time from
286.703 to 395.579 ms, CPU codestream generation from 175.619 to 259.826 ms,
and quantization from 97.739 to 102.143 ms. CPU codestream code, output bytes,
and work are unchanged. This is evidence of a large non-target timing
component, not permission to subtract that time or hide the total result.

Final production profiles use three warmups and one captured encode:

| Input | Coefficient launches, unchanged | Coefficient parent/new (ms) | Change | All-kernel parent/new (ms) |
|---|---:|---:|---:|---:|
| Odd 4K | 21 | 31.116535 / 17.106612 | -45.0% | 277.767619 / 253.533016 |
| Odd 1080p | 18 | 3.738464 / 2.208346 | -40.9% | 40.320190 / 40.164420 |
| Flower | 21 | 1.033908 / 0.582127 | -43.7% | 7.697213 / 7.250231 |

Total launch counts stay 464 / 452 / 464. The five allocation sizes per
encode, peak requested device bytes (2,608,448,064 / 651,957,638 /
85,819,408), pool release thresholds, and all 31 H2D / 19 D2H / one D2D
transfers remain unchanged. Byte totals are identical to S35. Every
non-coefficient kernel retains its ordered name, geometry, register/shared
resources, and per-thread local field. Coefficient launch order and geometry
also match. Module code and per-block shared storage are not claimed to be
unchanged merely because explicit arena allocations are unchanged.

The final builds pass all 61 CUDA and 47 CPU tests. The new coefficient test
covers 280 guarded cases: all seven shapes, counts 1/3/17, six input
patterns, both quantization modes, scales 1/3541/32768, raw quantization
extremes, varied X/B multipliers and color correlation, and nonzero
anchor/coefficient/block offsets with padded, permuted spatial layouts.
It checks bitwise AC, reconstructed coefficients, DC, quantized DC,
inverse sigma, and error flags, plus output guards and input immutability.
The same device allocations are reused with changed coefficient inputs.
NaN/infinite AC and DC inputs, DC overflow, and invalid EPF sharpness
exercise the original error behavior. The full AQ integration suite remains
an independent workflow gate.

All 23 parent/candidate image pairs match SHA-256, byte count, decoded
Butteraugli, chosen strategies, and encoder final score. The matrix uses
the existing sample, odd padded 1080p/4K PFMs, Flower, and the three CC0
photographs from S29 at distances 0.5/1.2/3, effort 7, plus sample/Flower
at distance 1.2, effort 9. Decode/metric use the pinned libjxl
`e8ff09762481785938d8e4e01333ed3917571161`, Clang 22.1.8,
`RGB_D65_SRG_Rel_Lin`, and the metric's 80-nit default. The benchmark's
synthetic odd inputs and quality PFMs are not asserted to have identical
floating-point pixels.

Serial/batch checks pass at even 1920x1080 with fully-resident and
maximum-throughput batches 1/2/4, and even 3840x2160 with fully-resident
batches 1/2. Median serial/batch speedups are 1.018/1.120/1.286,
1.001/1.524/1.748, and 1.061/1.019 respectively. These are current-build
batch-vs-serial comparisons, not S35-vs-S36 throughput claims.

All seven scoped sanitizer checks pass: full AQ memcheck/initcheck/synccheck
and focused coefficient memcheck/initcheck/synccheck/racecheck. Both memory
checks report zero leaked bytes; all error/hazard summaries are zero.
Full-AQ runs take approximately 39/26/20 seconds; focused runs 6/5/5/11
seconds. Full-AQ memcheck includes stream-ordered race tracking and full
leak checking, and focused memcheck includes full leak checking. There are
no kernel filters or API-error suppressions. No full-AQ racecheck is started.

The evidence script initially rejects CTest's successful `100% tests passed
out of N` wording because it expects the alternate `0 tests failed` wording.
The reporting parser is corrected to accept either form and additionally
count every individual passing test. Revalidation of the unchanged logs
passes; no completed test or sanitizer is repeated or reclassified.

`s36_final_validate.py` checks ordered non-coefficient kernel identities and
resources, all launch geometries, transfer counts/bytes, explicit allocation
sizes, all 23 image/score identities, 61/47 individual test results, all seven
sanitizer summaries, leak checks, and the native resource report. Ignored
`build-cuda-ninja/profiles/s36_*` files preserve the initial cache-only,
partial-direct, direct-only, and combined experiments; `s36_final_*` holds
the qualified final build's probes, warm/cold observations, captures, image
qualification, batch results, CTest archives, sanitizer logs, native
disassembly/resources, and evidence assertions. The probe build scripts
record the exact MSVC compiler/link commands and include the guarded test
fixture. Baseline executables remain `s35_retained_*`; final encode,
benchmark, batch, and coefficient-test executables are frozen as
`s36_retained_*`, with SHA-256 values in `s36_final_binary_hashes.json`.
`s36_repeat_*` preserves the reversed-start warm cohort, and
`s36_wall_summary.py` derives the per-stage summaries from both cohorts'
original benchmark stdout.

No permission error, admin prompt, or unexplained execution stall is
encountered. The previous long full-AQ racecheck is not evidence of a
confirmed firewall cause. No firewall, OS-service, or power/clock setting
is changed.

All execution jobs have completed. The coefficient kernel still has
separate Y, X/B, and color-restoration loops with repeated same-coefficient
global reads and barriers; dependency-preserving fusion is a concrete next
experiment. CPU coefficient handoff/assembly, perceptual filtering, and
codestream work also remain substantial. This checkpoint is not evidence
that the resident encoder is maxed out.

## Resident coefficient pass fusion follow-up (S37)

Baseline: `fbbe265` (S36), on the same CUDA 11.8 / MSVC 19.37 / SM86
configuration. This follow-up keeps the resident coefficient kernel's
input/output layout and arithmetic policy while fusing its Y AC pass,
X/B prediction/quantization pass, and color-restoration pass.

### Dependencies, rounding, and resource control

A thread owns the same coefficient index in all three channels. X/B
prediction depends on that thread's reconstructed Y value, not another
thread's output. Keeping Y in a register and completing X/B restoration
before their stores removes two Y reconstruction reads and the X/B
intermediate store/read round trips: 24 bytes of global accesses per
three-channel coefficient index. For one fully covered padded 4K pass this
is 199,065,600 bytes, or 597,196,800 across three passes. This is source-level
global-access accounting, **not** a measured DRAM or PCIe byte reduction;
the former accesses may have hit GPU caches. Explicit arena sizes and
required host/device transfers do not change.

Three of six block-wide barriers are removed. The fused AC work never reads
DC, so DC quantization publication can wait until the existing pre-LLF
barrier. Shared-basis publication, DC extraction before cross-channel DC
quantization, and the pre-LLF barrier remain. The last barrier is essential:
LLF reconstruction can overwrite coefficients owned by other threads during
the AC pass. Native code confirms six barriers in both reference kernels
and three in the fused kernel.

The first fusion expression fails a guarded bitwise test despite passing
the AQ integration test. For DCT8, count 1, pattern 3, unadjusted quantization,
the first reconstructed X mismatch is at index 21: expected `0xc181e8e7`,
actual `0xc181e8e8`. Inlining allows the compiler to contract a different
dequantization multiplication into the color-restoration addition. Explicit
`fmaf(factor, reconstructed_y, reconstructed)` preserves the unfused
restoration's final FMA with the previously rounded dequantized addend.
All 280 guarded cases then agree bitwise with both the original pre-S36
kernel and the S36 unfused kernel, including changed-input reuse and error
paths. The initial failure and correction are retained in the local study
artifacts; approximate agreement is not substituted for the failed gate.

Unrestricted fusion increases register use from 50 to 66. On the qualified
device, that reduces feasible simultaneous 256-thread blocks from four to
three. Large homogeneous 8x8/16x8/8x16 probes regress about 13-14%, although
larger transforms improve. A four-block launch bound reduces the fused
kernel to 64 registers, but initially applying it to all template variants
also changes the unfused control to 64 registers. That confounded setup is
not used for timing comparisons.

Instead, the common body is forced inline into separate bounded-production
and unbounded-reference entries. The S36 control remains at 50 registers,
32-byte stack, and 256-byte shared basis table; the fused entry uses 64
registers with the same stack/shared sizes and `LOCAL:0`. The original
pre-S36 reference still uses 50 registers and a 112-byte stack. The remaining
32-byte production stack is the existing cosine large-argument path, not a
new spill allocation. Native body comparison, ignoring only the kernel-name
line, proves that the unfused control matches the frozen S36 implementation.
The measured probe and final build also have identical native coefficient
kernel code.

### Isolated measurements

The probe uses the S36 unfused control and S37 fused candidate in one
executable, all seven shapes, default quantization matrices, deterministic
inputs, valid disjoint anchors, and the same 512x536 / 3840x2160 requested
tile coverage as S36. Allocation, initialization, and six-output verification
are outside CUDA event timing. Each shape/extent has five alternating-order
pairs, ten warmups, eleven samples, and three launches per sample.

| Transform | Unrestricted large paired ratio | Bounded small paired ratio | Bounded large paired ratio |
|---|---:|---:|---:|
| 8x8 | 1.1252 | 0.9118 | 0.9362 |
| 16x8 | 1.1417 | 0.8871 | 0.9803 |
| 8x16 | 1.1444 | 0.8955 | 0.9800 |
| 16x16 | 0.8885 | 0.9076 | 0.9044 |
| 32x16 | 0.9537 | 0.9044 | 0.9758 |
| 16x32 | 0.9538 | 0.9087 | 0.9735 |
| 32x32 | 0.8446 | 0.8821 | 0.9025 |

The bounded candidate improves all fourteen shape/extent median paired
observations: 8.8-11.8% for the smaller requested extent and 2.0-9.8% for
the larger one. These are per-column paired cohorts, not comparisons of
candidate absolute times across changing laptop operating states. The
unrestricted candidate is not retained.

### Whole-encoder measurements and qualification

Seven alternating-order warm pairs use three warmups and five measured
in-memory fully-resident encodes. Seven cold pairs use zero warmups and one
sample. Percentages are medians of paired ratios, not ratios of the
displayed marginal medians.

| Cohort/input | Total parent/new (ms) | Paired total change | Quantization parent/new (ms) | Paired quantization change |
|---|---:|---:|---:|---:|
| Warm 4K | 592.029 / 591.471 | +1.2% | 350.725 / 349.152 | -0.01% |
| Warm 1080p | 149.220 / 144.794 | -5.8% | 67.029 / 67.305 | +0.4% |
| Warm Flower | 30.699 / 30.274 | +1.8% | 13.770 / 13.775 | +0.2% |
| Cold 4K | 692.730 / 672.636 | -2.7% | 435.839 / 435.660 | -0.5% |
| Cold 1080p | 194.671 / 185.635 | -0.06% | 102.333 / 98.734 | -0.3% |
| Cold Flower | 49.804 / 49.525 | -3.0% | 26.434 / 26.435 | -2.1% |

Whole-encode and quantization changes are small or mixed; no stable
end-to-end speedup is claimed. Warm 4K total ranges are 555.188-625.816 /
558.662-610.461 ms; 1080p ranges are 142.036-175.434 / 140.512-163.803 ms.
Between-workload GPU snapshots span 66-74 C, P3, and 292-1282 MHz; these
snapshots are not asserted to be the clocks during every timed kernel.
No power, clock, process-priority, or OS setting is changed.

Initial final-build profiles use three warmups and one captured encode:

| Input | Coefficient parent/new (ms) | Change | All-kernel parent/new (ms) |
|---|---:|---:|---:|
| Odd 4K | 16.743435 / 16.627786 | -0.7% | 241.631942 / 261.957807 |
| Odd 1080p | 2.271932 / 1.825868 | -19.6% | 39.988563 / 37.826668 |
| Flower | 0.584431 / 0.524942 | -10.2% | 7.245813 / 7.188145 |

The initial 4K total GPU result regresses 8.4% even though its coefficient
subset is nearly flat. Three additional 4K pairs run after sanitizer
qualification, with the candidate first in repeats 1/3 and the parent first
in repeat 2, again using three warmups and one captured encode:

| 4K pair | Coefficient parent/new (ms) | All-kernel parent/new (ms) |
|---|---:|---:|
| Initial | 16.743435 / 16.627786 | 241.631942 / 261.957807 |
| Repeat 1 | 7.187412 / 5.593385 | 130.321417 / 127.239212 |
| Repeat 2 | 6.875377 / 5.617874 | 128.379705 / 126.743728 |
| Repeat 3 | 6.977140 / 5.775153 | 128.648472 / 128.192776 |

All four coefficient subsets improve, with a median paired reduction of
17.8%. All-kernel time improves in three pairs and regresses in the initial
pair: its median paired change is -0.8%, while the non-coefficient subset
changes +0.2%. Absolute GPU times vary substantially between the initial
and repeat captures; no cause is established and all observations are
retained. The repeats preserve each version's ordered kernel identities,
geometry, resources, and transfer accounting. The 1080p and Flower
coefficient subsets also improve, but a single capture per smaller input
is not treated as a stable whole-workflow measurement. This supports a
targeted kernel improvement, not a stable total-encode gain.

Total launches stay 464/452/464, including 21/18/21 coefficient launches.
All non-coefficient kernels retain their ordered names, geometry, register/
shared resources, and per-thread local fields. The five explicit allocation
sizes per encode, peak requested bytes (2,608,448,064 / 651,957,638 /
85,819,408), and pool thresholds are unchanged. All 31 H2D, 19 D2H, and one
D2D transfer keep their byte totals. The extra internal reference entry and
different kernel code are not claimed to have zero module-code footprint.

The final build passes 61 CUDA and 47 CPU tests. The existing 280-case
coefficient test now compares both original and S36-unfused controls with
the fused candidate, for all six outputs/error arrays, output guards,
input immutability, and changed-input reuse. It additionally reports the
first differing float/integer bit pattern and its complete case context.
All 23 image pairs match bytes, SHA-256, chosen strategies, final encoder
score, and decoded Butteraugli using the same named S36 quality matrix,
pinned libjxl revision, linear-sRGB interpretation, and 80-nit metric default.

Serial/batch checks pass at even 1080p in fully-resident and
maximum-throughput modes, batches 1/2/4, and even 4K fully-resident batches
1/2. Paired serial/batch median speedups are 1.102/1.159/1.197,
1.005/1.514/2.033, and 0.980/1.057 respectively. These qualify batch behavior
in the current build, not S36-vs-S37 throughput.

All seven scoped sanitizer checks pass: full-AQ memcheck/initcheck/
synccheck and focused coefficient memcheck/initcheck/synccheck/racecheck.
Both memory checks report zero leaked bytes, and all error/hazard summaries
are zero. Full-AQ runs take approximately 39/28/20 seconds; focused runs
8/7/5/20 seconds. Full-AQ memcheck includes stream-ordered race tracking
and full leak checking; focused memcheck includes full leak checking.
No kernel filter or API-error suppression is used. No full-AQ racecheck
is started. Builds, tests, sanitizers, wall timing, isolated probes, and
production profiling run sequentially without competing GPU experiments.

`s37_final_validate.py` checks profile accounting, all 23 image/score
identities, individual CTest passes, all sanitizer/leak summaries, native
resource tuples, the measured-vs-final kernel identity, the S36 control's
native identity, and the six-to-three barrier reduction.
`s37_final_profile_repeat.py` checks the four 4K captures' structure and
derives their paired timing summaries. Ignored
`build-cuda-ninja/profiles/s37_*` artifacts retain the failed rounding
experiment, unbounded and bounded probes, compiler resources, native
disassembly, and exact probe build commands. The `s37_final_*` files hold
warm/cold observations, profiles, repeated profiles, decoded qualification,
batch checks, CTest archives, sanitizer commands/logs, and evidence
assertions. Baselines remain `s36_retained_*`; final encode, benchmark,
batch, and coefficient-test executables are frozen as `s37_retained_*`,
with SHA-256 values in `s37_final_binary_hashes.json`.

All execution jobs have completed normally; no sanitizer is aborted or
left running. No permission error, admin prompt, or unexplained execution
stall is encountered. The earlier long full-AQ racecheck does not establish
a firewall cause. No firewall, OS-service, power, or clock setting is changed.

The resident path is not demonstrated maxed out. Per-anchor quantization
arithmetic still warrants an invariant-hoisting experiment with exact
rounding and register-pressure gates. A separate dataflow audit can test
whether final coefficient-only materialization needs every reconstruction
write; diagnostic and error-reporting contracts must be traced first.
CPU coefficient handoff/assembly, perceptual filtering, and codestream
work also remain substantial targets.

## Encoding-only resident coefficient materialization follow-up (S38)

Baseline: `e968baa` (S37), with the same CUDA 11.8 / MSVC 19.37 / SM86
configuration. The resident policy already distinguishes final evaluation
from encoding-only finalization. The latter applies every requested field
update and then quantizes the final field without an inverse transform,
filtering, or metric evaluation. It nevertheless used the full coefficient
kernel, including float reconstruction output and LLF restoration.

### Dataflow and retained contracts

The host validator rejects diagnostic reconstruction and block-map outputs
when `evaluate_final_field` is false. `AssembleFrame` reads raw quantization,
integer AC/DC coefficients, and color-correlation maps, not reconstructed
float coefficients. A subsequent evaluation runs a complete coefficient
pass before any inverse transform. The specialized final entry is used
only inside that encoding-only branch, with a null reconstruction pointer.
Scored iterations, explicitly requested final scoring, diagnostic images,
and the maximum-throughput path keep their existing behavior.

The specialized pass omits Y/X/B reconstruction stores, final X/B color
restoration, the forward DC-basis table, and LLF restoration. The pre-LLF
barrier also disappears. It still computes reconstructed Y for X/B
prediction, performs X/B dequantization for its finite-value error checks,
and preserves DC extraction, DC quantization, inverse sigma, and all
existing integer/error outputs. No validation is dropped merely because
its numerical result is not materialized.

At padded 3840x2160, one final pass avoids 99,532,800 bytes of AC float
stores plus 1,555,200 bytes of LLF overwrite stores, or 101,088,000 bytes
of source-level global stores. This is not measured DRAM or PCIe traffic:
caching and write behavior are not inferred from source counts. Explicit
allocation capacity and required transfers are unchanged; earlier scored
iterations still need the reconstruction buffer.

Native code reports 47 registers, 128 bytes of shared storage, two block
barriers, a 32-byte stack, and `LOCAL:0`, versus 64 registers, 256 shared
bytes, and three barriers in the S37 full pass. The stack remains the
existing cosine large-argument path, not a newly introduced spill. The
measured probe and production coefficient kernels have identical native
code. All three S37 full/reference native bodies are unchanged after
ignoring the full kernel's extra template-argument name.

### Focused verification and isolated timing

The guarded coefficient test expands from 280 to 301 cases. Every case
compares the full kernel with both prior arithmetic oracles and compares
the materialization kernel's integer AC/DC, float DC, inverse sigma, and
error arrays with the full kernel. A guarded reconstruction array remains
untouched; changed-input reuse also passes a null reconstruction pointer.
A following full launch must completely reconstruct the changed input.
Seven strategies, both adjustment modes, the existing signed-zero,
impulse, tiny, large, threshold, random, offset, and invalid-input cases
remain covered. New nonfinite dequantization-table cases independently
exercise X, B, and Y errors, including checks whose float output is omitted.

The AQ integration test additionally reuses one prepared object for
encoding-only policies with one/two updates and distinct initial fields,
then reconstructs their final fields and reruns each policy with final
evaluation enabled. Codestreams, fields, prior score histories, final
scores, block maps, and diagnostic RGB agree exactly between the relevant
paths. Existing transactional failure and optional host-staging checks
remain in place.

The same-executable event probe compares the S37 full pass with the new
materialization pass using five alternating-order pairs, ten warmups, and
eleven samples of three launches per sample. Allocation, initialization,
and output verification are excluded. Shapes use default quantization
matrices, deterministic inputs, valid disjoint anchors, and only complete
homogeneous tiles within requested 512x536 and 3840x2160 extents.

| Shape | Smaller extent paired ratio | Larger extent paired ratio |
|---|---:|---:|
| 8x8 | 0.660839 | 0.703583 |
| 16x8 | 0.707031 | 0.756563 |
| 8x16 | 0.721461 | 0.758661 |
| 16x16 | 0.751553 | 0.782574 |
| 32x16 | 0.684211 | 0.823896 |
| 16x32 | 0.693965 | 0.804515 |
| 32x32 | 0.594378 | 0.796807 |

All fourteen median paired ratios favor materialization: 24.8-40.6% at
the smaller extent and 17.6-29.6% at the larger extent. This improvement
applies to the final coefficient-only pass, not every scored pass or the
whole encoder. Public-workflow observations and qualification follow below.

Unlike prior coefficient checkpoints, the image matrix explicitly runs
both ordinary encoding and `--collect-final-score` for all 23 named
quality cases. That yields 46 before/after pairs: final-score collection
uses full reconstruction and alone would not exercise this specialization.
Cross-policy codestream/decoded-score identity is checked as well as
parent/candidate identity within each policy.

### Public workflow timing and qualification

Seven alternating-order warm pairs use three warmups and five samples;
seven cold pairs use zero warmups and one sample. Changes below are medians
of paired ratios, not ratios of the displayed marginal medians.

| Cohort/input | Total parent/new (ms) | Paired total change | Quantization parent/new (ms) | Paired quantization change |
|---|---:|---:|---:|---:|
| Warm 4K | 586.068 / 587.572 | -0.6% | 353.162 / 355.187 | -0.6% |
| Warm 1080p | 144.542 / 144.833 | +0.5% | 67.371 / 67.025 | -2.3% |
| Warm Flower | 28.805 / 29.132 | -1.4% | 13.470 / 13.396 | -0.7% |
| Cold 4K | 642.904 / 667.954 | +0.2% | 430.773 / 429.951 | -0.2% |
| Cold 1080p | 188.334 / 178.921 | -5.4% | 100.819 / 98.552 | -2.4% |
| Cold Flower | 49.750 / 48.869 | -2.6% | 27.004 / 26.175 | -3.5% |

No stable whole-encoder or cold-start speedup is established. Warm 4K
parent/new ranges are 565.528-652.923 / 565.709-619.010 ms; Flower ranges
are 28.352-51.376 / 28.185-52.284 ms. Between-workload GPU snapshots span
66-73 C, P3, and 1020-1282 MHz; these are not asserted to be clocks during
every timed kernel. No power, clock, process-priority, or OS setting changes.

Initial profiles capture one encode after three warmups:

| Input | Final coefficient pass parent/new (ms) | All coefficient passes parent/new (ms) | All kernels parent/new (ms) |
|---|---:|---:|---:|
| Odd 4K | 5.868661 / 4.887549 | 15.525772 / 15.017310 | 252.776916 / 267.074884 |
| Odd 1080p | 0.688722 / 0.445706 | 1.818028 / 1.770220 | 36.948355 / 40.342223 |
| Flower | 0.176004 / 0.122691 | 0.530125 / 0.476974 | 7.192666 / 7.136849 |

The final coefficient pass improves 16.7% / 35.3% / 30.3%, but total GPU
time regresses 5.7% / 9.2% in the initial 4K / 1080p captures, including
regressions in unchanged kernels. Two additional pairs for each larger
input run after sanitizer qualification with identical capture flags,
including memory-usage tracing, and the same three warmups/one capture.
Repeat 1 starts with the candidate; repeat 2 starts with the parent.

| Input/pair | Final coefficient pass parent/new (ms) | All coefficient passes parent/new (ms) | All kernels parent/new (ms) |
|---|---:|---:|---:|
| 4K repeat 1 | 6.726891 / 4.501745 | 14.747064 / 13.681114 | 248.976928 / 247.012014 |
| 4K repeat 2 | 6.644424 / 4.449874 | 14.927482 / 12.542113 | 259.444374 / 248.513897 |
| 1080p repeat 1 | 0.663119 / 0.429068 | 1.915341 / 1.612268 | 39.877021 / 38.176584 |
| 1080p repeat 2 | 0.778038 / 0.394794 | 2.011156 / 1.566568 | 41.090422 / 37.776327 |

All three final-pass observations per larger input improve. Median paired
changes across initial and repeat pairs are -33.0% / -35.3% for that final
pass, -7.2% / -15.8% for all coefficient passes, and -0.8% / -4.3% for all
kernels at 4K / 1080p. Each total-GPU cohort includes one regression; the
non-coefficient subsets change -0.4% / -3.7%, showing that untargeted timing
also varies. All observations are retained, and these captures do not
establish a stable end-to-end gain. Ordered per-version kernel structure,
resources, explicit allocations, and transfers match across the repeats.

Launches remain 464/452/464, including 21/18/21 coefficient launches.
Only the last 7/6/7 use the new specialization; all geometry and non-target
kernel identities/resources are unchanged. Five explicit arena sizes per
encode, peak requested device bytes (2,608,448,064 / 651,957,638 /
85,819,408), pool thresholds, and all 31 H2D / 19 D2H / one D2D transfer
counts and byte totals are unchanged. The added kernel is not claimed to
have zero module-code footprint.

All 61 CUDA and 47 CPU tests pass. All 46 image pairs match bytes, SHA-256,
chosen strategies, decoded Butteraugli, and requested final encoder scores.
Encoding-only and scored outputs also agree within both builds. The same
seven-input, three-distance, effort-7 matrix and two effort-9 cases use the
S36 quality sources, pinned libjxl revision, linear-sRGB interpretation,
and default 80-nit metric. Synthetic benchmark inputs are generated
separately from the qualification PFMs.

Serial/batch checks pass at even 1080p in fully-resident and
maximum-throughput modes, batches 1/2/4, and even 4K fully-resident batches
1/2. Median paired batch-vs-serial speedups are 1.019/1.057/1.501,
0.983/1.543/1.473, and 0.898/1.044 respectively. These compare concurrency
within the current build; they are not S37-vs-S38 throughput gains.

All seven scoped sanitizer checks pass with zero errors/hazards: full-AQ
memcheck/initcheck/synccheck and focused coefficient memcheck/initcheck/
synccheck/racecheck. Both memchecks report zero leaked bytes. Full-AQ runs
take about 36/25/19 seconds, and focused runs 10/8/7/31 seconds. Full-AQ
memcheck includes stream-ordered race tracking and full leak checking;
focused memcheck includes full leak checking. No kernel filter or API-error
suppression is used. No full-AQ racecheck is started.

The qualified native kernels are preserved in `s38_final_sass.txt` and
match those in `s38_probe.exe`; all S37 full/reference bodies remain
identical. `s38_final_validate.py` checks these native/resource/barrier
identities, ordered profile geometry/resources, explicit allocation and
transfer accounting, all 46 image identities and cross-policy identities,
individual CTest passes, and sanitizer/leak summaries. Ignored
`build-cuda-ninja/profiles/s38_*` files preserve the guarded probe and exact
build command; `s38_final_*` retains warm/cold observations, profiles,
decoded qualification, batch checks, CTest archives, sanitizer logs and
commands, and evidence assertions.

`s38_final_profile_repeat.py` derives the repeated timing summaries and
checks each version's ordered kernel structure, resources, allocations,
and transfer accounting. Baselines remain `s37_retained_*`; the qualified
encode, benchmark, batch, coefficient-test, and AQ-test executables are
frozen as `s38_retained_*`, with SHA-256 values in
`s38_final_binary_hashes.json`.

All jobs finish normally; no sanitizer is aborted or left running.
Benchmarks, builds/tests, disassembly, sanitizers, and profiles do not
overlap GPU work. No permission error, admin prompt, or unexplained
execution stall is encountered, and no firewall, OS-service, power, or
clock setting is changed. The previous long full-AQ racecheck does not
establish a firewall cause.

This checkpoint is not evidence that the encoder is maxed out. The full
and materialization entries still use one generic 256-thread block per
anchor, even for 64/128-coefficient transforms, and retain runtime shape
arithmetic. Shape specialization and block-size experiments are concrete
next candidates, alongside quantization-invariant hoisting. CPU coefficient
handoff/assembly, perceptual filtering, and codestream work remain material.

## Shape-specialized resident coefficients (S39)

### Cause and retained implementation

S38 (`6fa9132`) still launches a generic 256-thread block per anchor for
every physical shape. Small 64/128-coefficient transforms cannot use every
thread for their main coefficient work, and runtime dimensions survive into
basis construction and indexing. The follow-up specializes seven physical
width/height pairs for both full reconstruction and final materialization.
It reuses the unchanged coefficient body, preserving FP32 operation order,
the explicit rounding-preserving FMA, quantization decisions, and error
checks. Anchors, offsets, channel stride, pitches, and matrix/scalar values
remain dynamic.

Dispatch checks pixel dimensions, coefficient count, and covered dimensions
before selecting a specialization. Unknown strategies or noncanonical
internal batches retain the S38 generic behavior. Format strategy names use
rows x columns, whereas kernel tags use physical width x height: strategy 6
(`DCT16x8`) selects physical 8x16, and strategy 7 selects 16x8; strategies
10/11 similarly select 16x32/32x16. Both rectangular orientations use the
same corresponding quantization-table offset. An initial prototype reversed
those tags and therefore fell back to generic entries for rectangles; it
passed numerical tests but was corrected before timing. Production profile
validation now asserts that every coefficient launch actually specializes.

Blocks use `min(width * height, 256)` threads, with a launch bound targeting
1024 threads across resident blocks. This is a compiler resource constraint,
not a claim of measured occupancy. On the qualified CUDA 11.8/SM86 build:

| Physical width x height | Threads | Registers full/materialize | Stack/local bytes | Shared bytes full/materialize |
|---|---:|---:|---:|---:|
| 8x8 | 64 | 46 / 40 | 0 / 0 | 256 / 128 |
| 8x16 | 128 | 47 / 42 | 0 / 0 | 256 / 128 |
| 16x8 | 128 | 44 / 40 | 0 / 0 | 256 / 128 |
| 16x16 | 256 | 47 / 42 | 0 / 0 | 256 / 128 |
| 16x32 | 256 | 47 / 42 | 0 / 0 | 256 / 128 |
| 32x16 | 256 | 47 / 42 | 0 / 0 | 256 / 128 |
| 32x32 | 256 | 48 / 45 | 0 / 0 | 256 / 128 |

All fourteen specialized kernels have zero stack/local allocation, versus
32-byte stack frames in S38's bounded full/materialization entries. Full
and materialization barrier counts remain three and two. Four retained
generic/reference native bodies match S38 exactly. Final specialized native
bodies match the measured sized-block executable exactly; temporary
fixed-block comparison entry points are removed from production source.
Added specializations are not claimed to have zero module-code footprint.

The guarded fixture still runs 301 cases across all seven strategies,
both quantization modes, changed-input reuse, finite and invalid inputs,
dequantization overflow in each channel, offsets, gaps, and immutable
inputs. It now compares the new full entry with S38 generic, S36 unfused,
and original controls, and materialization with both generic materialization
and the full result. Guarded/no-write then null reconstruction pointers and
later reconstruction rebuilding remain covered. Four additional bounded
noncanonical cases exercise unknown strategy, aliased opposite orientation,
coefficient-count mismatch, and covered-dimension mismatch under both
policies. Existing prepared-AQ scored/unscored policy-reuse tests also pass.

### Experiments and operating-state limitation

Three same-executable studies retain all observations. Each uses default
matrices, deterministic pattern-5 inputs, valid gapped anchors, separate
buffers per variant, both full/materialization policies, all seven shapes,
and complete homogeneous tiles within requested 512x536 and 3840x2160
extents. Allocation, initialization, and output verification are outside
CUDA-event timing. These are not end-to-end encodes.

The first study specializes shapes but keeps 256 threads for every shape.
Five alternating-order pairs, ten warmups, and eleven samples of three
launches per sample favor specialization in all 28 shape/extent/policy
median pairs: ratios range 0.602-0.852 for the smaller extent and
0.633-0.844 for the larger extent. The second study adds 64/128-thread
blocks and uses the same protocol. It favors the candidate in 27 of 28
median pairs, but large `DCT32x16` materialization regresses 8.6%, despite
retaining the same 256-thread native body as the first study. Absolute
timings also vary substantially; that unfavorable observation is retained.

A third study compares generic, specialized/fixed-256, and specialized/sized
entries in all six order permutations. Each variant receives at least
200 ms of warmup before eleven three-launch event samples. It verifies all
six outputs before timing and records GPU state around each shape case.
Native comparison proves that the changed 64/128-thread entries have the
same instruction bodies as their fixed-256 controls; launch configuration
is the difference. Larger shapes call the same 256-thread kernel through
both specialized wrappers, but still use separate buffer allocations.

Across this sustained study, sized/generic median ratios favor specialization
in 24 of 28 cases. Regressions are small-extent full `DCT16x32` (+0.1%)
and `DCT32x32` (+15.8%), and large-extent full `DCT32x16` (+2.2%) and
`DCT16x32` (+2.4%). For the twelve comparisons where launch size actually
changes (three small shapes, two extents, two policies), sized/fixed ratios
range 0.819-1.000, all favorable or nearly flat. This supports retaining
the smaller blocks; it does not establish a uniform generic-to-specialized
speedup. Even unchanged larger-shape kernels show substantial timing
variation, with buffer placement and operating state not fully controlled.

At 2026-09-05 21:46:00 UTC during the sustained probe, a read-only GPU
query reports 100% utilization, 79 C, 262 MHz, and active software thermal
slowdown and software power-cap flags. Hardware thermal slowdown and
hardware power-brake flags are inactive in that sample. A subsequent
read-only detailed report shows a current/requested 40 W power limit,
versus a reported 60 W default. The GPU later cools to 68 C before final
qualification, but flags can still appear in between-workload snapshots;
the final measurements are not claimed to be uniformly unthrottled.

This directly establishes a performance constraint during this probe, not
the cause of every timing difference or of the earlier long full-AQ
racecheck. No firewall cause is established. The user is notified of the
observed limitation; no power, cooling, clock, priority, service, or firewall
setting is changed. The instantaneous sample and post-probe detailed report
are preserved as `s39_observed_gpu_limit_sample.txt` and
`s39_post_probe_gpu_limits.txt` under the ignored profiles directory.

### Public workflow timing and qualification

Seven alternating-order warm pairs use three warmups and five samples;
seven cold pairs use zero warmups and one sample. Changes are medians of
paired ratios, not ratios of displayed marginal medians.

| Cohort/input | Total parent/new (ms) | Paired total change | Quantization parent/new (ms) | Paired quantization change |
|---|---:|---:|---:|---:|
| Warm 4K | 631.129 / 621.108 | +0.4% | 362.980 / 353.615 | -0.3% |
| Warm 1080p | 150.316 / 162.614 | -0.1% | 67.650 / 70.315 | +3.6% |
| Warm Flower | 31.520 / 31.302 | +0.7% | 14.159 / 14.142 | +1.6% |
| Cold 4K | 683.534 / 679.137 | +2.2% | 435.174 / 444.968 | +1.6% |
| Cold 1080p | 188.661 / 188.250 | -2.7% | 100.569 / 99.952 | -0.1% |
| Cold Flower | 52.198 / 52.017 | +1.8% | 27.660 / 28.482 | +1.7% |

No stable whole-encoder or cold-start speedup is established. Warm 4K
parent/new ranges are 597.718-707.979 / 587.826-668.096 ms; warm 1080p
ranges are 147.121-168.046 / 146.683-170.428 ms. Public profile captures
use three warmups and one captured encode with CUDA memory tracing:

| Input | Final coefficient pass parent/new (ms) | All coefficient passes parent/new (ms) | All kernels parent/new (ms) |
|---|---:|---:|---:|
| Odd 4K | 4.721433 / 2.983469 | 12.997230 / 7.449501 | 248.445778 / 247.207837 |
| Odd 1080p | 0.594318 / 0.284102 | 1.819311 / 1.036377 | 40.648403 / 37.177459 |
| Flower | 0.122627 / 0.070307 | 0.473709 / 0.233099 | 7.145048 / 6.901906 |

Two further pairs per larger input run after sanitizer qualification with
identical capture flags and three warmups/one capture. Repeat 1 runs the
candidate first; repeat 2 runs the parent first. All observations are kept:

| Input/pair | Final coefficient pass parent/new (ms) | All coefficient passes parent/new (ms) | All kernels parent/new (ms) |
|---|---:|---:|---:|
| 4K repeat 1 | 5.207173 / 3.254228 | 14.749144 / 8.437081 | 265.002833 / 263.643993 |
| 4K repeat 2 | 5.143620 / 3.019150 | 14.695448 / 7.624898 | 282.835632 / 254.545008 |
| 1080p repeat 1 | 0.463916 / 0.394601 | 1.627179 / 1.159774 | 39.078626 / 38.835617 |
| 1080p repeat 2 | 0.440716 / 0.342634 | 1.777804 / 1.129213 | 40.522473 / 39.968048 |

Across three pairs per input, coefficient time improves in every observation,
with median paired reductions of 42.8% / 36.5% at 4K / 1080p. The final-only
subset improves 37.5% / 22.3%, and total GPU time improves 0.5% / 1.4%.
Median non-coefficient timing changes are +1.8% / +0.2%; the large total
improvement in 4K repeat 2 also includes faster unchanged kernels. These
captures support a targeted improvement, not a stable whole-encoder gain.
Flower has one capture pair, not a repeated cohort. Read-only snapshots
around the repeats span 73-74 C with thermal/power flags sometimes active
and sometimes inactive; they do not measure every timed kernel's clock.

Launch counts remain 464/452/464, including 21/18/21 coefficient launches.
All production coefficient launches select the expected specialized
physical shape, block size, resource signature, and full/final policy.
Grid dimensions, ordered non-coefficient identities/geometry/resources,
five arena allocation sizes, peak requested bytes (2,608,448,064 /
651,957,638 / 85,819,408), pool thresholds, and all transfer counts and
byte totals match S38. This optimization changes no explicit allocation
or host/device transfer requirements.

All 61 CUDA and 47 CPU tests pass. All 46 image pairs match codestream
bytes, SHA-256, selected strategies, decoded Butteraugli, and requested
encoder final scores. Encoding-only and scored outputs also match within
each build. The same seven inputs at distances 0.5/1.2/3, effort 7, plus
sample/Flower effort-9 cases use the previously pinned libjxl revision
`e8ff09762481785938d8e4e01333ed3917571161`, linear-sRGB interpretation,
and default 80-nit metric. Synthetic benchmark inputs are generated
separately from the qualification PFMs.

Serial/batch checks pass at even 1080p for fully-resident and
maximum-throughput batches 1/2/4, and even 4K fully-resident batches 1/2.
Median paired batch-vs-serial speedups are 1.022/1.371/1.457,
0.972/1.449/2.040, and 0.967/1.104 respectively. These measure concurrency
within this build, not S38-to-S39 throughput gains. They do not justify
adding execution lanes.

All seven scoped sanitizer runs finish normally with zero errors/hazards:
full-AQ memcheck/initcheck/synccheck and focused coefficient
memcheck/initcheck/synccheck/racecheck. Both memchecks report zero leaked
bytes. Full-AQ memcheck includes stream-ordered race tracking and full leak
checking; focused memcheck includes full leak checking. No kernel filter,
API-error suppression, or full-AQ racecheck is used. The checks take about
37/25/19 seconds for full AQ and 16/11/10/50 seconds for the focused fixture.
Builds, tests, sanitizers, disassembly, and GPU measurements are serialized.

Repeated-profile validation confirms each version's ordered kernel
geometry/resources, allocations, and transfers match its initial capture.
No sanitizer is aborted or left running, and every job finishes normally.
No permission error, admin prompt, or unexplained execution stall is
encountered. The earlier long full-AQ racecheck still does not establish
a firewall cause.

Ignored `build-cuda-ninja/profiles/s39_shape_*`, `s39_sized_*`, and
`s39_triple_*` preserve probe sources, build/measurement scripts, binaries,
native code/resources, and all per-case event observations. The three-way
study additionally retains its temporary kernel/header/fixture sources.
`s39_final_*` preserves the final native code, warm/cold cohorts, production
profiles, CTest logs, quality/decode outputs, batch observations, sanitizer
commands/logs, and evidence validation. `s39_native_check.py` and
`s39_final_validate.py` assert native identities, resource/barrier counts,
actual specialized dispatch, ordered production structure, allocation and
transfer accounting, per-image and cross-policy identity, and individual
test/sanitizer results. `s39_final_profile_repeat.py` checks repeated
per-version accounting and derives paired timing summaries;
`s39_final_repeat_gpu_state.txt` records the added telemetry. The baseline
remains the frozen `s38_retained_*` executables. Qualified encode, benchmark,
batch, coefficient-test, and AQ-test binaries are frozen as `s39_retained_*`,
with verified source/copy SHA-256 values in `s39_final_binary_hashes.json`.

This checkpoint does not establish that the resident encoder is maxed out.
Potential next investigations include narrower synchronization scopes for
the small DC work, compile-time coefficient-loop strides, and quantization
invariant hoisting, each requiring separate numerical and sanitizer gates.
CPU coefficient handoff/assembly, perceptual filtering, and codestream work
also remain material. The observed operating-state limitation should be
recorded in further experiments without silently changing system settings.

## Fused blur and frequency split (S40)

### Bottleneck and implementation

Baseline: `fe13a54` (S39). Re-reading its three retained production profiles
puts Malta at 49.8-58.1 ms of 247.2-263.6 ms GPU time at 4K. Prior larger-tile
and warp-sharing Malta experiments are already unfavorable. Source inspection
identifies another concrete dataflow cost: 24 frequency-split kernels read a
blurred plane immediately after a vertical convolution writes it. The split
alone costs about 9.6-9.7 ms at 4K, in addition to its producer. This follow-up
targets that redundant intermediate boundary, not another coefficient tweak.

The tiled convolution body now accepts a compile-time output writer. Existing
plain convolution entries retain their signatures and native instructions.
Four new vertical entries feed their rounded blur value directly to the
channel-specific low/high split: channels 0/1 use 15 taps and channels 3/4
use 7 taps. Horizontal convolution is unchanged. The unmodified separate
frequency kernel and plain blur entries remain the differential oracle.

The vertical convolution reads only the completed, tightly packed horizontal
intermediate. Each output owner reads and updates its own original input
pixel and writes its corresponding high-frequency output. No neighboring
in-place input value is consumed by convolution. Halo loading, synchronization,
partial-edge handling, tap accumulation order, included-weight normalization,
division, and range/clamp formulas are preserved. No global fast math or
precision change is introduced. Unsupported internal channels are rejected
without work; empty extents are no-ops for supported channels.

Each fused call removes one launch and one float store plus one float load
per active pixel of the intermediate blurred image. That is eight bytes of
source-level global accesses, not measured DRAM or PCIe traffic. The working
plane is still used by later mask/blur stages, so this does not remove an
arena or reduce allocation capacity. Low/medium decomposition, the B-channel
blur, X suppression, Malta, masking, and multiscale composition are unchanged.

Native comparison against the frozen S39 encoder covers all 29 preexisting
Butteraugli kernels, including both directions of all four tiled blur sizes.
Their complete instruction dumps are identical after normalizing only the
compiler-generated private-namespace hash. The four new entries match the
isolated measured binary after production integration:

| Channels | Taps | Threads | Registers | Shared bytes | Stack/local bytes |
|---|---:|---:|---:|---:|---:|
| 0 / 1 | 15 | 256 | 39 | 10,048 | 0 / 0 |
| 3 / 4 | 7 | 256 | 31 | 8,992 | 0 / 0 |

Register/shared usage matches the corresponding plain vertical kernel.
Added entry points have a module-code footprint; no zero-footprint claim is
made. The initial native-check script mistakenly compared against S39's
resident-only dump and then used new private-symbol hashes to select old
functions. Those evidence-selection errors were corrected by extracting all
29 names from the frozen encoder's own symbol table. They were not numerical
test failures and are not used as passing evidence.

### Guarded test and isolated experiment

A new CUDA-only CTest fixture runs 320 cases: sixteen geometries, four
channels, and five patterns. It includes one-pixel dimensions, partial
horizontal/vertical tiles, boundaries around 32/64 and 256, independently
padded strides, offset pointers, signed zeros, small/wide random values,
impulse weights at range/clamp thresholds and adjacent floats, and very
small/large finite inputs. Each case has three stages: initial evaluation,
in-place reuse, and changed-input reuse. Low/high outputs and the horizontal
intermediate must be bit-identical; guards and weights must be unchanged.
The candidate does not receive or write a blurred-plane pointer. Separate
empty-extent and unsupported-channel cases exercise early exits.

The isolated executable compares old and fused complete blur/split bundles,
including the unchanged horizontal pass. Both variants use the same input,
intermediate, output, and weight allocations. A device-to-device restore
from a seed buffer precedes each event interval but is outside timing; three
successive transformations are timed per interval. Initial low/high outputs
are compared bitwise before measurement. Five alternating-order pairs use
ten warmups and eleven event samples per variant. Initialization, resets,
allocation, verification, and file I/O are excluded.

| Channel | 512x536 paired ratio | 1919x1079 paired ratio | 3839x2159 paired ratio |
|---|---:|---:|---:|
| 0 | 0.812121 | 0.816092 | 0.854834 |
| 1 | 0.761905 | 0.816117 | 0.856834 |
| 3 | 0.698529 | 0.874568 | 0.822601 |
| 4 | 0.617834 | 0.837494 | 0.830205 |

All twelve median paired ratios favor fusion, including 14.3-17.7% at odd
4K. These are isolated bundles with patterned inputs and padded strides,
not whole encodes. Read-only before/after snapshots span 63-67 C and
210-1305 MHz, with thermal/power flags mostly active. The post-process
snapshots are not claimed to be clocks during timed kernels. No system,
power, cooling, firewall, priority, or clock setting is changed.

### Public workflow qualification

The integrated focused test, prepared Butteraugli test, and full-AQ test
pass, followed by all 62 CUDA and 47 CPU tests. Seven alternating-order
warm pairs use three warmups/five samples; seven cold pairs use zero
warmups/one sample. Changes are medians of paired ratios, not ratios of
the displayed marginal medians:

| Cohort/input | Total parent/new (ms) | Paired total change | Quantization parent/new (ms) | Paired quantization change |
|---|---:|---:|---:|---:|
| Warm 4K | 538.428 / 547.524 | +1.2% | 338.442 / 327.024 | -2.3% |
| Warm 1080p | 137.638 / 141.079 | -0.5% | 64.519 / 63.581 | -2.7% |
| Warm Flower | 27.073 / 27.283 | +0.8% | 12.812 / 12.706 | -1.0% |
| Cold 4K | 629.828 / 614.183 | -2.2% | 428.686 / 413.066 | -3.6% |
| Cold 1080p | 178.205 / 180.217 | +1.1% | 98.107 / 97.928 | +0.2% |
| Cold Flower | 49.886 / 49.156 | -3.3% | 27.521 / 27.199 | -1.7% |

Warm quantization improves on all three inputs, but total wall and cold-start
results are mixed. No stable whole-encoder gain is established. Warm 4K
parent/new total ranges are 505.784-566.208 / 514.918-572.790 ms; cold 1080p
ranges are 171.509-197.023 / 171.422-229.262 ms. Between-workload snapshots
remain observations of a variable operating state, not locked timed clocks.

The first production profile pair uses three warmups and one captured encode
with CUDA memory tracing. The targeted subset is the old vertical blur plus
separate split versus the new fused vertical entry. The complete bundle also
includes the unchanged horizontal blur:

| Input | Target parent/new (ms) | Complete bundle parent/new (ms) | All kernels parent/new (ms) |
|---|---:|---:|---:|
| Odd 4K | 18.096654 / 13.317013 | 27.588990 / 23.848608 | 244.326355 / 267.357309 |
| Odd 1080p | 3.357204 / 2.548418 | 4.812986 / 4.049287 | 38.294058 / 37.405818 |
| Flower | 0.431688 / 0.301576 | 0.695342 / 0.561361 | 6.891470 / 6.757611 |

The target improves 26.4% / 24.1% / 30.1%, but initial 4K total GPU time
regresses 9.4%, including slower unchanged kernels. That observation is
retained and triggers alternating-order repeat profiles after sanitizers.

All 24 separate frequency-split launches disappear, reducing total launch
counts from 464/452/464 to 440/428/440. The 24 replacement entries use the
expected channel/tap sequence and preserve their predecessor's grid, block,
register, shared, and local-memory geometry. All ordered non-target kernel
identities/geometry/resources match. Five arena sizes, peak requested device
bytes (2,608,448,064 / 651,957,638 / 85,819,408), pool thresholds, and all
31 H2D / 19 D2H / one D2D transfer counts and byte totals are unchanged.

All 46 before/after image pairs match codestream bytes, SHA-256, selected
strategies, decoded Butteraugli, and requested final encoder scores.
Encoding-only and scored policies agree within each build as well. The
seven-input, distances 0.5/1.2/3, effort-7 matrix plus two effort-9 cases
retains the pinned libjxl revision, linear-sRGB interpretation, and default
80-nit metric from S39. Benchmark-generated synthetic images are separate
from the quality PFMs.

Serial/batch checks pass at even 1080p in fully-resident and
maximum-throughput modes, batches 1/2/4, and even 4K fully-resident batches
1/2. Paired batch-vs-serial speedups are 1.165/1.146/1.328,
1.006/1.524/1.947, and 0.910/0.984 respectively. These are current-build
concurrency comparisons, not before/after optimization gains; the 4K batch
results do not support adding execution lanes.

All seven scoped sanitizer checks finish normally with zero errors/hazards:
full-AQ memcheck/initcheck/synccheck, and focused blur/frequency
memcheck/initcheck/synccheck/racecheck. Both memchecks report zero leaked
bytes. Full-AQ memcheck enables stream-ordered race tracking and full leak
checking; focused memcheck enables full leak checking. All four focused
runs cover the entire 320-case, three-stage fixture; none uses a kernel
filter or API-error suppression. Full-AQ checks take about 34/23/17 seconds;
focused checks take about 5/5/7/118 seconds. No full-AQ racecheck is started,
and no sanitizer is aborted or left running. Progress output confirms that
the longer focused racecheck continues advancing through the fixture.

Two more pairs per larger input use the same three warmups/one capture,
CUDA memory tracing, and frozen parent. Repeat 1 starts with the candidate;
repeat 2 starts with the parent:

| Input/pair | Target parent/new (ms) | Complete bundle parent/new (ms) | All kernels parent/new (ms) |
|---|---:|---:|---:|
| 4K repeat 1 | 18.136366 / 13.319575 | 27.741733 / 23.502079 | 259.012609 / 256.805127 |
| 4K repeat 2 | 18.288593 / 12.750881 | 27.818821 / 22.023949 | 249.354622 / 240.892142 |
| 1080p repeat 1 | 3.284693 / 2.594884 | 4.661272 / 4.156555 | 36.407171 / 39.050532 |
| 1080p repeat 2 | 3.310132 / 2.561314 | 4.717557 / 4.069802 | 37.178868 / 38.070696 |

Every target and complete-bundle observation improves. Across all three
pairs per input, median paired target reductions are 26.6% / 22.6% at
4K / 1080p; complete-bundle reductions are 15.3% / 13.7%. All-kernel changes
are -0.9% / +2.4%, with median non-target changes +1.1% / +4.8%.
The initial unfavorable 4K capture and both unfavorable 1080p total repeats
remain included. The evidence supports the targeted reduction, not a stable
whole-encoder improvement. Flower has one capture pair, not a repeated cohort.
Per-version ordered kernel geometry/resources, explicit allocations, and
transfers match the initial capture throughout the repeats. Before/after
snapshots span 72-74 C, with thermal/power flags alternating between active
and inactive; they are not measurements of every kernel's operating clock.

All jobs finish normally. Builds, tests, sanitizers, disassembly, and GPU
measurements do not overlap GPU work. No permission error, admin prompt,
or unexplained stall is encountered. No firewall or other system setting
is changed; the earlier long full-AQ racecheck does not establish a firewall
cause.

Ignored `build-cuda-ninja/profiles/s40_*` artifacts preserve the baseline
source, bottleneck scan, native dumps/resources, same-buffer probe and its
build/measurement scripts, every paired observation, and GPU snapshots.
`s40_final_*` preserves warm/cold cohorts, profiles, CTest logs, decoded image
outputs, batch observations, sanitizer commands/logs, and evidence assertions.
The native checker normalizes only private symbol hashes; the profile
validator verifies every removed/replaced launch and every retained ordered
kernel, resource, allocation, and transfer record. The baseline is the frozen
S39 build, not a rebuilt or retimed substitute for recorded S39 observations.

`s40_probe_verify.ps1` extracts native code from the actual timed executable;
`s40_probe_identity.py` proves all 33 Butteraugli bodies match the integrated
encoder. `s40_final_validate.py` also requires that identity, all 320 cases
in every focused sanitizer log, and all image/test/accounting assertions.
`s40_final_profile_repeat.py` validates repeated structure and derives the
paired summaries. Qualified encode, benchmark, batch, frequency-test,
AQ-test, and Butteraugli-test binaries are frozen as `s40_retained_*`, with
source/copy SHA-256 checks in `s40_final_binary_hashes.json`.

The encoder is not demonstrated maxed out. Malta, large-radius convolution,
remaining perceptual passes, CPU coefficient handoff, and CPU codestream
generation still matter. S39's first warm candidate report alone spends
about 212 ms / 72 ms in codestream encoding at 4K / 1080p, respectively;
further work should measure that host stage's internal costs as well as
remaining GPU work. No expanded math/quality contract or system-setting
change is implied by this next investigation.

## Branch-free coefficient-order zero counting (S41)

### Host bottleneck and bounded change

Baseline: `8f832d1` (S40). The resident workflow already collects detailed
`VarDctCodestreamProfile` counters, although the CUDA benchmark prints only
its outer stages. An ignored diagnostic copy prints the existing counters;
it adds no instrumentation inside the encoder. Three warmups/five samples
on the S40 library identify the following medians:

| Stage (ms) | Odd 4K | Odd 1080p | Flower |
|---|---:|---:|---:|
| Entire codestream wall | 179.389 | 72.049 | 14.201 |
| AC tokenization wall | 76.186 | 21.875 | 4.461 |
| Coefficient-order work | 34.307 | 9.223 | 1.477 |
| Entropy optimization wall | 29.794 | 25.262 | 5.419 |
| Section-writing wall | 51.373 | 16.851 | 3.151 |

These nested stages are not additive. In particular, aggregate coefficient
tokenization work (302.735 / 60.411 / 4.621 ms) and section-token writing
work (321.154 / 80.330 / 5.054 ms) sum time across workers, not elapsed wall
time. Scan-order preparation is serial within its task and gates subsequent
AC tokenization. The existing balanced/direct-ANS policy is retained; S41
does not repeat the earlier removal of exhaustive entropy candidate searches.

`CountGroupZeros` previously branches on every coefficient, then checks a
64-bit counter for overflow before incrementing it. All counters start at
zero. A validated frame has a size_t-representable block area; each visited
anchor occupies at least one distinct block and adds at most one to any
counter. Thus no count can exceed that area or overflow uint64_t on the
supported 32/64-bit hosts. A static assertion makes the integer-width premise
explicit. The update becomes `counts[i] += coefficients[i] == 0`, with the
vector's stable data pointer acquired before scanning each channel.

Frame validation, family/shape checks, group consumption checks, 64-bit
counter storage, selected transforms, RNG sequence, float-scaled sorting
keys, stable tie ordering, LLF prefixes, and output atomicity are unchanged.
There is no entropy-policy, floating-point, CUDA-kernel, allocation-capacity,
or transfer change. This shared host-stage optimization also applies to
non-CUDA encodes; fully-resident CUDA remains the measured public workflow.

MSVC 14.37 native inspection finds scalar compare/set/add loops, not SIMD.
The first prototype removes branches but still reloads the vector data
pointer within each iteration. Acquiring that pointer explicitly removes
the repeated load and address calculation. A separate helper-function
experiment also remains scalar and is not retained. No vectorization claim
is made, and no architecture flag or compiler-wide option is changed.

### Differential fixture and measurement design

The new backend-independent `coefficient_order` CTest contains 96 cases:
seven supported transform shapes plus a mixed layout, six coefficient
patterns, and both full and effort-7 sampled order policies. Its 36x36-block
frames have odd pixel dimensions and four groups, including narrow right and
bottom groups. Mixed tiles exercise both orientations of rectangular shapes
sharing an order family. Patterns include all zeros, no zeros, sparse/dense
zeros, deterministic tied populations, and signed int32 extremes. A separate
coefficient-major scalar reference counts each coefficient across selected
transform spans and sorts precomputed integer keys. It compares complete
orders and their Lehmer token streams. Small-frame cutoff and invalid-input
atomicity are checked separately. The frozen original implementation passes
the same fixture, so the reference is not validated only against the change.

The final workflow cohort uses seven alternating-order pairs per input,
three warmups and five samples, with the detailed diagnostic probe linked to
the respective production library. Its profiling counters already existed
in the original benchmark. Seven zero-warmup/one-sample cold pairs use the
unmodified production benchmark and the frozen S40 binary. The image matrix
retains seven inputs at distances 0.5/1.2/3, effort 7, plus sample/Flower at
effort 9 and distance 1.2; both scored and encoding-only policies are checked.
The independent decoder/metric remain pinned libjxl
`e8ff09762481785938d8e4e01333ed3917571161`, linear sRGB, default 80 nits.
Benchmark-generated odd synthetic images are distinct from the quality PFMs.

### Public workflow observations

The initial branch-removal prototype (before explicit data-pointer caching)
reduces coefficient-order work in all nine preliminary pairs, including
36.081/43.406/46.425 to 22.495/22.697/24.877 ms at 4K. It is preserved as
`s41_scalar_phase_probe.exe`; the subsequent pointer-cached implementation
is the qualified candidate. The preliminary 1080p pair 0 has slower total
and codestream time and is retained in `s41_prototype_phases.json`.

Final warm changes are medians of per-pair ratios, not ratios of the
displayed marginal medians:

| Input | Order parent/new (ms) | Paired order change | Codestream parent/new (ms) | Paired codestream change | Total parent/new (ms) | Paired total change |
|---|---:|---:|---:|---:|---:|---:|
| Odd 4K | 38.447 / 18.670 | -50.3% | 221.893 / 195.921 | -7.2% | 606.673 / 556.484 | -3.9% |
| Odd 1080p | 9.723 / 5.077 | -47.8% | 73.277 / 66.515 | -7.2% | 147.112 / 139.435 | -2.3% |
| Flower | 1.592 / 1.077 | -31.4% | 15.405 / 14.118 | -8.8% | 30.438 / 28.804 | -5.3% |

All 21 order-work pairs improve; AC-tokenization wall medians improve
20.9% / 18.6% / 15.7%. Codestream wall improves in 6/7, 5/7, and 6/7 pairs;
total improves in 5/7, 5/7, and 6/7. All observations remain included, notably
Flower pair 0's total regression from 29.722 to 42.188 ms even though its
order work improves. Unchanged quantization-pipeline median paired changes
are +1.1% / -0.3% / -2.8%, illustrating variability outside the changed loop.

| Cold input | Total parent/new (ms) | Paired total change | Quantization parent/new (ms) | Paired quantization change |
|---|---:|---:|---:|---:|
| Odd 4K | 686.312 / 678.586 | -0.4% | 437.592 / 442.423 | +1.6% |
| Odd 1080p | 187.120 / 181.303 | -4.4% | 100.791 / 100.138 | +0.1% |
| Flower | 51.268 / 52.490 | +1.8% | 28.159 / 28.872 | -0.2% |

Cold results are mixed, with large unfavorable observations retained:
candidate 4K reaches 889.364 ms and candidate Flower 75.594 ms. The result
supports reduced order work and this cohort's favorable warm medians, not a
uniform or platform-independent whole-encoder speedup. Read-only snapshots
before/after the timing cohorts show 64/72 C, 210/1282 MHz, and software
thermal/power flags inactive/active. These are boundary snapshots, not timed
CPU/GPU clocks and not a causal diagnosis of each outlier. No power, cooling,
clock, priority, firewall, or service setting changes are made.

### Correctness and safety qualification

All 63 CUDA-build and 48 CPU-build tests pass, including the new 96-case
fixture under MSVC and GNU, the existing pinned order/token goldens, and
public workflow/conformance tests. All 46 image pairs are byte-identical
and have matching SHA-256, decoded Butteraugli, selected strategies, and
requested encoder scores. Scored and encoding-only outputs agree within
each build. Serial/batch identity checks pass for even 1080p in resident and
maximum-throughput modes at batches 1/2/4 and even 4K resident at batches 1/2.
Current-build batch-vs-serial paired speedups are 1.003/1.362/1.261,
0.988/1.530/1.910, and 0.969/1.137 respectively; these are concurrency
observations, not before/after gains from this host-loop change.

A separate Clang 22.1.8 Release CPU build instruments all linked GJXL host
code with AddressSanitizer. The complete 96-case fixture, small-frame cutoff,
and invalid-input checks pass with allocation/deallocation mismatch checking
enabled and no sanitizer suppression. Initial harness attempts fail before
execution: a combined compile/link output argument, mixed instrumented/plain
standard-library annotations, a debug-runtime compiler probe, and missing
explicit runtime libraries in CMake's direct-link command. These setup
failures are retained, not counted as sanitizer passes. A complete standalone
instrumented build, Release compiler probes, and the compiler driver's
observed runtime-link arguments resolve them without changing production
flags or disabling annotations. No memory-error failure is observed.

### CPU-only pattern sweep and retained evidence

A separate MSVC probe invokes only coefficient-order derivation on completed
132x132-block frames (1053x1049 active pixels, 1056x1056 padded). It excludes
frame construction/validation, hashing, and destruction from the timed call.
The same eight layouts, six patterns, and two policies produce 96 cases;
three alternating-order process pairs use three warmups/nine timed samples
per case. The original source is linked ahead of the production library for
the parent, while the candidate links the production library directly.
All 288 corresponding output hashes match, and all 96 median paired ratios
favor the candidate:

| Coefficient pattern | Median ratio across 16 cases | Minimum / maximum case ratio |
|---|---:|---:|
| All zero | 0.744 | 0.674 / 0.910 |
| No zeros | 0.812 | 0.589 / 0.912 |
| Sparse nonzeros | 0.378 | 0.325 / 0.459 |
| Dense nonzeros | 0.383 | 0.317 / 0.551 |
| Deterministic tied populations | 0.791 | 0.684 / 0.909 |
| Signed extremes with zeros | 0.275 | 0.205 / 0.334 |

These are synthetic, warm, CPU-only order calls, not complete encodes or
GPU timings. The all-nonzero cases explicitly exercise the new add-zero
stores that the original loop skipped; they do not regress in this cohort.
This does not guarantee every coefficient distribution or host improves.
MSVC production-object disassembly matches the pointer-cached prototype's
complete 526-instruction `CountGroupZeros` body. All three channel counting
loops contain seven scalar instructions, including only the loop-control
branch; there is no per-iteration vector-data pointer reload or SIMD claim.

No CUDA source, device-memory contract, or launch dispatch changes in S41.
The existing GPU fixtures run in the full suite; GPU sanitizer/profile traces
are not recaptured for this host-only change, and prior traces are not relabeled
as new measurements. All measured workflows and build/test/sanitizer jobs run
sequentially, with no competing benchmark or build. No permission error or
admin/firewall prompt is observed, and the earlier long sanitizer run still
does not establish a firewall cause.

Ignored `build-cuda-ninja/profiles/s41_*` artifacts preserve the original
source, phase-reporting probe, initial/scalar and final candidates, all
paired observations, independent image outputs, batch results, CTest logs,
compiler/native evidence, and standalone ASan build instructions. The
corrected isolated-probe include-path error is also a setup issue, not a
numerical failure. `s41_final_validate.py` checks the complete image/policy,
test, sanitizer, timing-count, and order-hash evidence;
`s41_native_validate.py` checks production-object identity and the three
scalar loops. Qualified executables are frozen as `s41_retained_*`, with
source/copy SHA-256 checks in `s41_final_binary_hashes.json`; the S40 baseline
hashes are rechecked against their original manifest.

The encoder is not demonstrated maxed out. CPU AC tokenization, ANS
histogram/model construction, and token emission remain substantial, along
with GPU perceptual work. The section writer's per-chunk bit preparation and
bytewise append are concrete next inspection targets; they require their
own measurements and atomicity/byte-equivalence tests before any change.

## Lightweight ANS token emission (S42)

Date: 2026-09-05. Baseline: `2260047` (S41).

### Bottleneck and scope

After the scan-order improvement, section writing remains a substantial
shared-host cost in the fully resident workflow. A diagnostic copy of the
S41 serializer times model validation, reverse-chunk reservation, ANS token
processing, bit packing, and append separately. Timers surround each stream
stage, not each token; atomic totals are accumulated once per stage and
printed after encoding. These are aggregate worker durations over three
warmups plus five samples, **not** stage wall medians:

| Input / CPU workers | Model validation ms | Reserve ms | ANS processing ms | Bit packing ms | Append ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Odd padded 4K / auto | 234.620 | 7.765 | 2,096.835 | 149.969 | 44.376 |
| Odd padded 1080p / auto | 42.280 | 5.577 | 389.438 | 24.338 | 8.378 |
| Flower / auto | 4.311 | 0.292 | 28.850 | 4.387 | 0.855 |
| Odd padded 4K / one | 129.546 | 3.563 | 1,495.301 | 87.821 | 21.515 |

The 4K run processes 61,906,776 ANS tokens and 4,353,144 reverse chunks
over eight encodes. ANS processing accounts for about 83% of the measured
auto-worker components and 86% with one worker. This points first to the
per-token conversion/recurrence, not model validation or byte packing.
The diagnostic 4K section-wall median is 51.305 ms; aggregate token-write
work is 319.808 ms and must not be added to wall time. Instrumented runs
are diagnosis, not optimization qualification.

Two success-path costs are removed:

- The original HybridUint conversion formula moves to a private constexpr
  helper with the explicit precondition `config.valid()`. The public API
  retains null-output/configuration validation and unchanged failure
  behavior. ANS stream processing validates immutable configurations once,
  then inlines that conversion instead of making one checked out-of-line
  call returning a string-owning `Status` per token.
- The private `AdvanceAnsState` recurrence returns a null pointer on success
  or a static error message on failure. Its three callers materialize a
  public `Status` only on failure. This removes per-token success-object
  construction without removing context, symbol, frequency, reciprocal,
  reverse-map, or allocation checks. It does not change the public API or
  claim that an empty success string previously allocated heap storage.

ANS state arithmetic, reciprocal precision, extra-bit and renormalization
ordering, model selection, partitioning, output layout, and entropy policy
remain unchanged. The two multi-candidate measurement loops retain their
checked HybridUint calls; only their shared recurrence return is changed.
No CUDA source, device allocation, transfer, or kernel dispatch is changed.

### Preliminary comparisons and correctness

The first prototype changes only HybridUint conversion/config validation.
Three alternating warm pairs reduce section-writing time by median paired
17.0% / 7.3% / 4.2% at 4K / 1080p / Flower, but whole-encode ratios are
0.964 / 1.042 / 1.002. All three preliminary 1080p whole-encode pairs
regress; these observations are retained, not discarded.

Adding the lightweight recurrence return gives preliminary section-writing
ratios of 0.602 / 0.692 / 0.851 and aggregate token-write-work ratios of
0.562 / 0.618 / 0.634 against S41. Whole-encode ratios are
0.978 / 0.998 / 1.028. These small cohorts establish a target-stage lead,
not a uniform end-to-end improvement.

The backend-independent entropy fixture now checks all 816 valid HybridUint
configurations over 443,904 value encodings: small values, power-of-two
boundaries, deterministic random values, and `UINT32_MAX`. An independent
inverse reconstructs the original integer from symbol and extra-bit fields;
three constexpr goldens pin key results. Another 4,097 invalid configuration
combinations must leave the public output unchanged. The ANS fixture also
compares payloads to a separate ordinary-division/modulo recurrence, covers
empty and nonempty split/interleaved streams and count/write parity, and
verifies destination atomicity for invalid contexts, absent symbols, and
invalid configurations. Existing malformed lookup/reciprocal, model golden,
direct-policy, and deferred-width tests remain active.

### Native-code check

MSVC 14.37 Release COFF inspection confirms the two stream-processing
instantiations (write and count) have one HybridUint call relocation and
one state-advance call relocation in S41. The conversion-only prototype
removes the former but retains the latter. The final build has neither:
both conversion and recurrence inline in each token loop. The standalone
instantiations can still be emitted in the object; their presence alone
does not imply a loop call.

Inlining grows the enclosing function from 176 instructions in S41 to
278/299 for count/write, while eliminating the repeated calls and
success-result handling. This is not described as a reduction in that
function's static instruction count. The unchanged `CountGroupZeros`
native body remains identical to the S41 snapshot at 526 instructions,
which rules out a change to that body but does not explain its wall-time
variation. The new entropy fixtures also pass when linked against the
original S41 ANS and entropy sources.

### Isolated CPU token comparison

A separate MSVC probe compares S41, the conversion-only prototype, and the
final build using the same ordered tokens and prepared models. Four
deterministic distributions (single value, mostly zero with full-width
outliers, dense 8-bit, and full-width random integers), two storage layouts,
and write/count operations give 16 cases. Each invocation processes 262,144
tokens; each sample contains four invocations. There are three warmups,
nine measured samples, and three triplets rotating executable order so
each version occupies each position once. Model construction and output
hashing are outside the measured interval. No GPU workflow is invoked.

All 16 final median paired ratios favor S42; 47/48 individual comparisons
favor it. Model/payload FNV-1a hashes and exact bit counts agree across all
three versions in every triplet. Ranges below span interleaved and split
storage, and are final/S41 median paired ratios:

| Distribution | Write ratio | Count ratio |
| --- | ---: | ---: |
| Single value | 0.479-0.489 | 0.460-0.476 |
| Mostly zero with outliers | 0.517-0.582 | 0.461-0.524 |
| Dense 8-bit | 0.738-0.742 | 0.480-0.488 |
| Full-width integers | 0.845-0.902 | 0.445-0.483 |

The one adverse observation, split full-width writing in triplet 1,
is 20.121275 to 21.935125 ms and remains included. All 16 conversion-only
medians also improve versus S41; adding the lightweight recurrence return
improves each further. This isolates a host token-processing gain without
claiming that these ratios apply to complete image encoding.

### Complete-workflow timing

Seven alternating before/after pairs per input use three warmups and five
samples per process, explicit CUDA, fully resident AQ, and no requested
final score. The reporting probe reuses the same compiled S41 benchmark
object and exposes the existing 41 serializer/workflow fields. Each pair
compares against the frozen S41 executable. The final cold cohort uses the
unmodified production benchmark, seven fresh-process pairs, zero warmups,
and one sample. No build, test, sanitizer, or other benchmark overlaps these
runs. Percentages below are **median paired changes**, not ratios of the
two separately listed medians:

| Warm input / stage | S41 median ms | S42 median ms | Paired change |
| --- | ---: | ---: | ---: |
| 4K section writing | 54.256 | 30.893 | -42.0% |
| 4K token-write aggregate work | 341.417 | 182.436 | -46.6% |
| 4K codestream encoding | 184.616 | 166.727 | -9.7% |
| 4K complete encode | 548.102 | 530.225 | -3.3% |
| 1080p section writing | 15.798 | 10.516 | -34.3% |
| 1080p token-write aggregate work | 72.753 | 40.624 | -44.4% |
| 1080p codestream encoding | 66.068 | 63.233 | -5.3% |
| 1080p complete encode | 140.623 | 135.446 | -4.4% |
| Flower section writing | 3.636 | 2.732 | -16.0% |
| Flower token-write aggregate work | 5.544 | 3.434 | -33.7% |
| Flower codestream encoding | 15.028 | 14.091 | -3.6% |
| Flower complete encode | 30.605 | 28.762 | -2.6% |

All 21 section-writing and token-work pairs improve. Complete-encode pairs
improve in 6/7, 5/7, and 5/7 cases. The unrelated coefficient-order phase
gets slower in all seven 4K and 1080p pairs (paired +25.3% and +15.3%) despite
unchanged source; Flower is +7.2%. Those adverse observations are retained.
This study does not establish their cause or assign them to firewall,
clock, or compiler behavior.

| Cold input / complete encode | S41 median ms | S42 median ms | Paired change |
| --- | ---: | ---: | ---: |
| 4K | 670.090 | 657.900 | -3.1% |
| 1080p | 179.639 | 180.758 | -4.6% |
| Flower | 55.392 | 50.494 | -3.4% |

Cold complete-encode pairs improve in 5/7, 4/7, and 6/7 cases. The 1080p
ratio of medians is slightly adverse even though the paired median favors
S42. Cold outliers, including 4K candidate 803.622 ms, 1080p candidate
212.744 ms, and Flower parent 84.977 ms, remain in the data. The unchanged
quantization stage has cold paired changes of +0.8% / -0.3% / -2.1%.
The target-stage gain is consistent; a uniform whole-encoder gain across
runs, devices, or operating states is not claimed.

Telemetry sampled before the warm cohort reports 65 C, 210 MHz, and inactive
software thermal/power limit flags; after the cold cohort it reports 73 C,
1,282 MHz, and both flags active. These are boundary observations, not
per-kernel clocks. No firewall, power, cooling, clock, priority, or service
settings were changed. The older long sanitizer run remains an aborted run,
not a pass or a confirmed firewall diagnosis.

### Qualification and retained evidence

Both complete suites pass: 63 CUDA-enabled MSVC tests and 48 CPU-only GNU
tests. A fully instrumented clang-cl host AddressSanitizer build also passes
the expanded entropy and codestream-encoder fixtures with
`halt_on_error=1:alloc_dealloc_mismatch=1`; no annotations or allocator checks
are suppressed. No new GPU sanitizer or Nsight run is claimed for this
host-only change, and older device checks are not relabeled as S42 evidence.

The image matrix contains seven inputs (17x13 repository sample, the two
odd padded qualification PFMs, Flower, and three CC0 corpus photographs),
three distances (0.5, 1.2, 3.0) at effort 7, plus sample/Flower effort 9 at
distance 1.2. Both encoding-only and requested-final-score policies give
46 before/after pairs. Every pair has identical codestream SHA-256, byte
count, and independent decoded Butteraugli score. Decoding and scoring use
the pinned libjxl revision `e8ff09762481785938d8e4e01333ed3917571161`,
linear-sRGB interpretation, and the default 80-nit metric setting. The
qualification PFMs differ from the synthetic benchmark images.

Even-size batch checks cover 1080p fully resident and maximum throughput
at batch sizes 1/2/4, and 4K fully resident at sizes 1/2. Median paired
serial/batch speedups are 0.979/1.333/1.341, 1.001/1.613/2.302, and
1.033/1.037 respectively. These compare current-build concurrency modes,
not S41 versus S42; each batch output must match its serial reference.

Ignored local artifacts include `s42_emit_final_*.txt` and the associated
instrumented source copies, `s42_generic_phases.json`,
`s42_success_phases.json`, `s42_final_phases.json`,
`s42_final_cold_*.json`, `s42_final_quality.json`,
`s42_final_{cpu,cuda}_ctest.log`, `s42_final_batch_*.txt`, and
`s42_asan.txt`, `s42_token_compare.json`, the token probe sources/build
scripts, `s42_native_summary.json`, and the associated COFF disassemblies.
The first diagnostic-print attempt produced no counters;
only the corrected explicit-print runs support the component table.

`s42_final_validate.py` verifies the baseline source identity, original S41
binary hashes, final source/copy hashes, both eliminated loop calls,
46 image pairs including strategy and requested encoder-score equality,
cross-policy output identity, full-suite counts, host ASan completion,
warm/cold cohort sizes, isolated hashes, and batch coverage. Qualified
executables are frozen as `s42_retained_*` with
`s42_final_binary_hashes.json`. All builds, tests, benchmarks, and sanitizer
jobs complete normally; no elevation/firewall blocker is reported.

The encoder is not demonstrated maxed out. CPU AC tokenization and ANS
histogram/model construction remain substantial, as does GPU perceptual
work. The remaining byte-packing/append costs are smaller measured leads;
the unexplained change in unchanged order-phase timing also warrants
separate attribution before drawing broader CPU conclusions.

## Lightweight direct AC token accumulation (S43)

Date: 2026-09-05. Baseline: `0f5a7a7` (S42).

### Bottleneck and implementation

After ANS emission improves, the retained S42 4K cohort reports 70.259 ms
of AC tokenization wall time, including order/context preparation, and
325.961 ms of coefficient-tokenization aggregate worker time. The latter
is work across parallel groups, not additional wall time. The equivalent
1080p figures are 17.252 / 54.456 ms and Flower 4.237 / 4.968 ms.

The direct AC path appends token values and contexts and, in balanced mode,
accumulates fixed-HybridUint symbol populations in the same pass. Its
`AppendDirectAcToken` helper returns a string-owning `Status` for every
token, which the enclosing loop assigns and checks, even on success.
This is the next concrete success-path cost, not a reason to change
coefficient decisions or entropy policy.

The helper now returns a private `AcTokenError` enum. The two callers
construct public `Status` objects only after a non-success result. A
separate mapping preserves all four original error messages and their
codes: three internal consistency errors and one invalid-argument count
overflow. Value/context append order, disabled-collection behavior,
HybridUint arithmetic, sparse-slot allocation, overflow guards, histogram
increments, maximum symbol, exception handling, and caller-visible output
atomicity are unchanged. An empty success string is not alleged to have
allocated heap storage; the target is repeated object/call/assignment
handling. No CUDA code, kernel dispatch, device-memory contract, or
system setting changes.

### Differential coverage

A new backend-independent `ac_direct_tokenization` fixture covers 3,072
group cases: seven physical transform shapes plus mixed layouts, six
coefficient patterns, natural/derived custom orders, four block-context
maps, both population-collection modes, and four groups per frame. The
36x36-block frames have odd 285x281 active extents and narrow right/bottom
groups. Patterns include all zero, all nonzero, sparse, dense, tied, and
signed extrema. Context maps include the compact, JPEG XL default,
two-channel, and quantization-threshold variants.

The checked public token-template path provides independent token values
and contexts. Separately accumulated 64-bit reference populations verify
every sparse symbol/count, ordering and offset, token total, extra-bit
total, and maximum symbol. One scratch object is reused across groups,
layouts, maps, and enabled/disabled collection; disabled output must not
retain old populations. Invalid group indexes preserve all destination
vectors. The deterministic completed-frame builder is extracted unchanged
from the S41 order fixture into `tests/quantized_frame_fixture.h`; the
original 96-case order differential test remains active.

The preliminary three-pair cohort uses the same 41-field reporting object
and frozen S42 baseline, with three warmups and five samples per process.
Median paired coefficient-tokenization-work ratios are
0.605 / 0.575 / 0.730 at 4K / 1080p / Flower. AC wall ratios are
0.710 / 0.762 / 0.843 and whole-encode ratios 0.922 / 0.927 / 0.945.
The 1080p pair 1 whole encode nevertheless regresses from 138.075 to
147.249 ms; Flower parent 39.746 ms is an outlier. These are retained,
and a larger cohort and isolated check are required for interpretation.

### Native and isolated checks

MSVC 14.37 Release COFF inspection shows that the append helper remains
out of line: both versions of the enclosing tokenizer have two append-call
sites. Its static
body shrinks from 272 to 180 instructions and returns the enum in a
register. The enclosing tokenizer changes from 1,703 to 1,664 instructions;
the 113-instruction error conversion is separate. At both sites, a zero
enum result branches over that conversion. This is not claimed as call
elimination or a dynamic instruction-count measurement. The unrelated
`CountGroupZeros` body remains identical to S42 at 526 instructions.

The new 3,072-case fixture also passes when linked against the original
S42 `ac_group.cpp`. A separate single-threaded CPU probe uses completed
frames with the JPEG XL default block-context map, not GPU/image-analysis
work. Eight layouts, six patterns, two order
policies, and two population modes give 192 cases. Three alternating
process pairs each use three warmups and nine samples. Timing includes
all four groups' direct tokenization and output allocation, but excludes
frame construction, order derivation, hashing, and final output destruction.
The same scratch object is reused within a case.

Complete token/population FNV-1a hashes match in all 576 before/after case
pairs. All 160 nonzero-pattern case medians improve; 476/480 individual
nonzero comparisons improve. All-zero cases are mixed: only 20/32 medians
improve. Overall, 180/192 medians and 534/576 individual comparisons
improve. Ratios below are final/S42 median paired ratios; each range spans
the sixteen layout/order combinations:

| Pattern | No populations: median [range] | Collect populations: median [range] |
| --- | ---: | ---: |
| All zero | 0.958 [0.834, 1.161] | 0.978 [0.814, 1.302] |
| All nonzero | 0.401 [0.334, 0.474] | 0.576 [0.522, 0.705] |
| Sparse | 0.401 [0.332, 0.493] | 0.660 [0.514, 0.759] |
| Dense | 0.406 [0.300, 0.509] | 0.652 [0.462, 0.731] |
| Tied | 0.410 [0.327, 0.483] | 0.586 [0.382, 0.673] |
| Signed extrema | 0.397 [0.344, 0.451] | 0.685 [0.618, 0.809] |

All-zero frames append only the per-transform/channel nonzero-count token,
so this edit addresses little of their work. Their adverse observations
are nevertheless retained: the worst median-paired case with populations
has separately aggregated medians 0.3690 to 0.4673 ms. The benefit on
token-heavy inputs is supported independently of the GPU timing noise;
no all-input improvement is asserted.

### Warm and cold workflow results

Final warm comparisons use seven alternating process pairs per input,
three warmups, five samples, explicit CUDA/fully-resident encoding, and
the same 41-field reporting object. The frozen S42 baseline is compared
with the final library. Independent cold comparisons use the unmodified
production benchmark with seven fresh-process pairs, zero warmups, and
one sample. No builds, tests, sanitizers, or other benchmarks overlap.
The paired change is the median of within-pair ratios, not the ratio of
the two independently listed medians:

| Warm input / stage | S42 median ms | S43 median ms | Paired change |
| --- | ---: | ---: | ---: |
| 4K coefficient-tokenization work | 308.661 | 230.236 | -25.4% |
| 4K AC-tokenization wall | 70.921 | 58.286 | -17.8% |
| 4K codestream encoding | 165.329 | 146.595 | -12.2% |
| 4K complete encode | 536.807 | 508.005 | -5.4% |
| 1080p coefficient-tokenization work | 55.487 | 38.447 | -27.8% |
| 1080p AC-tokenization wall | 17.657 | 14.341 | -17.4% |
| 1080p codestream encoding | 62.048 | 57.600 | -5.9% |
| 1080p complete encode | 133.797 | 129.105 | -3.6% |
| Flower coefficient-tokenization work | 4.878 | 3.809 | -21.5% |
| Flower AC-tokenization wall | 4.050 | 3.690 | -10.5% |
| Flower codestream encoding | 13.692 | 12.985 | -2.8% |
| Flower complete encode | 28.334 | 27.185 | -3.8% |

Coefficient work and AC wall improve in 7/7, 7/7, and 6/7 pairs; complete
encoding improves in 5/7 for each input. The unchanged order phase also
gets faster by paired 19.6% / 14.4% / 12.7%, unlike its S42 regression.
That observation is retained but not attributed to the append-helper edit.
Warm adverse cases include 4K 534.606 to 557.012 ms and 1080p 141.486 to
158.921 ms; Flower parent 52.366 ms is an outlier. None is removed.

| Cold input / complete encode | S42 median ms | S43 median ms | Paired change |
| --- | ---: | ---: | ---: |
| 4K | 632.287 | 614.581 | -4.2% |
| 1080p | 185.611 | 182.217 | -0.2% |
| Flower | 52.184 | 52.466 | -0.3% |

Cold complete encodes improve in 5/7, 4/7, and 4/7 pairs. Flower's ratio
of medians is slightly adverse despite its near-neutral paired median.
The unchanged quantization stage has cold paired changes of -2.0% /
+0.8% / +0.2%. Large observations include 4K parent 756.964 ms,
1080p parent/candidate 291.805/258.106 ms (different pairs), and Flower
candidate 78.997 ms. Thus the isolated target needs separate evidence;
the warm cohort is not a claim of a uniform whole-encoder speedup.

Boundary telemetry is 64 C / 210 MHz with inactive software thermal/power
limit flags before the warm cohort, and 73 C / 1,282 MHz with both flags
active after the cold cohort. These are not per-kernel clock measurements.
No elevation/firewall block is reported, and no system setting is changed.

### Workflow and sanitizer qualification

All 64 CUDA-enabled MSVC and 49 CPU-only GNU tests pass. A fully instrumented
clang-cl host AddressSanitizer build passes the new 3,072-case fixture,
existing AC-group tests, the shared-builder 96-case order test, and
codestream-encoder tests, with `halt_on_error=1:alloc_dealloc_mismatch=1`.
No annotations or allocator checks are suppressed. This host-only edit
does not claim new GPU sanitizer or Nsight captures.

The same seven-input, three-distance, two-score-policy matrix as S42,
including the two effort-9 cases, passes all 46 before/after comparisons
with identical byte counts, codestream SHA-256, and independently decoded
Butteraugli scores. The pinned libjxl revision remains
`e8ff09762481785938d8e4e01333ed3917571161`, with linear-sRGB decoding/scoring
and the default 80-nit metric setting. The synthetic timing images remain
distinct from the qualification PFMs.

Even-size batch checks cover 1080p fully resident and maximum throughput
at sizes 1/2/4, and 4K fully resident at sizes 1/2. Median paired
serial/batch speedups are 1.038/1.332/1.369, 1.054/1.453/2.140, and
1.066/1.053 respectively. These compare current-build concurrency modes,
not S42 versus S43; every batch result must match its serial reference.

Ignored artifacts include `s43_enum_phases.json`, `s43_final_phases.json`,
`s43_final_cold_*.json`, `s43_final_quality.json`, full-suite logs,
`s43_final_batch_*.txt`, `s43_asan.txt`, `s43_ac_compare.json`,
`s43_native_summary.json`, and their source/build scripts and native dumps.
`s43_final_validate.py` checks parent source identity, unchanged extracted
fixture content, native return/call facts, all 46 image/strategy/score pairs,
cross-score-policy identity, suite counts, host ASan completion, paired
cohort coverage, and isolated hashes. Qualified binaries are frozen as
`s43_retained_*`; source/copy and original S42 hashes are checked through
`s43_final_binary_hashes.json`. All qualification jobs finish normally.

Optimization remains ongoing. The append call, per-token population
metadata updates, and coefficient nonzero-count scan remain concrete host
leads, while GPU perceptual work still dominates much of the workflow.
None of these is assumed improved without a separate experiment; the
unexplained variation of unchanged order work also remains an attribution
limit rather than a claimed optimization.

## Contiguous AC nonzero reduction (S44)

Date: 2026-09-05. Baseline: `efda0a0` (S43).

### Motivation and arithmetic

S43 improves token-heavy inputs, but its all-zero isolated cases remain
mixed: almost no coefficient tokens are appended, while every coefficient
is still examined to count nonzeros. Both the public/template and direct
paths use `CountNonzerosExceptLlf`, whose nested coordinate loops test the
low-frequency rectangle for every coefficient.

The replacement counts the contiguous coefficient plane, then subtracts
nonzeros in the small LLF rectangle. The validated production strategies
contain at most 1,024 coefficients and at most 16 LLF entries, so the
integer sum and every subtraction fit `int32_t`. LLF subtraction cannot
make the result negative because it removes a subset already counted.
Only zero/nonzero predicates are used, including for signed extrema.
Completed encoder frames initialize their owned coefficient storage and
copy full transform planes, including LLF entries; this is not a read of
unmaterialized float reconstruction scratch.
The plane length remains `info.coefficient_count()` rather than an
arbitrary longer input span. Existing strategy/span validation and both
callers remain intact. This exposes a contiguous reduction to the compiler
without enabling a new CPU ISA, changing scan order, or altering tokens,
contexts, population policy, or CUDA code.

### Independent count validation

The prior direct/template comparison is insufficient by itself here:
both routes share the changed counter. The expanded fixture therefore
uses a separate coordinate-wise scalar oracle that skips LLF positions,
checks every emitted nonzero-count token, and walks the original scan to
check coefficient values and exact token consumption. Direct/template
context equality and independent 64-bit sparse-population checks remain.

Two new patterns isolate the boundary conditions: LLF-only nonzeros must
produce zero AC counts; a single last coefficient tests scan termination.
Together with the original six patterns, eight layouts, two order policies,
four context maps, two population modes, and four groups, the fixture now
covers 4,096 group cases. Existing poisoned-LLF/signed-extreme token goldens,
malformed-span rejection, the 96-case order test, and encoder tests remain
active. The original six completed-frame patterns are unchanged.

### Native mechanism

The same MSVC 14.37 Release options produce a scalar, 74-instruction
counter in S43 and a 171-instruction counter here. The new plane loop
processes eight coefficients per iteration through two unaligned 128-bit
loads, integer equality comparisons, predicate masking, and independent
packed 32-bit accumulators, followed by a horizontal integer reduction.
These are baseline x64 SSE2 operations, not a new AVX requirement. The
coefficient loads use `movdqu`; no stronger input alignment is assumed.
The compiled plane length is a multiple of 64, so its eight-entry loop
does not read beyond the validated coefficient plane.

The larger static body also contains a generic vector LLF-correction path
for widths of at least eight; production LLF widths are at most four and
take the scalar correction. Static instruction count is not dynamic work.
The separate direct-token accumulator remains 180 instructions and out of
line; its caller remains 1,664 instructions with the same two append and
two error-conversion relocations. Error conversion remains 113 instructions.
The unchanged coefficient-order counting body still matches S43's 526
instructions. No claim is made that the whole linked executable has
identical code addresses or cache behavior.

### Complete-workflow measurements

The preliminary three alternating pairs (three warmups, five samples)
give coefficient-tokenization-work ratios of 0.884 / 0.836 / 0.785 at
4K / 1080p / Flower. AC-wall ratios are 0.907 / 0.942 / 0.916, and
whole-encode ratios 0.962 / 0.979 / 0.965. Individual regressions include
4K 510.703 to 516.634 ms and 1080p 125.264 to 135.278 ms; Flower parent
37.989 ms is an outlier. These remain in the data and are not replaced
by the later cohort.

The primary qualification uses seven alternating process pairs per input,
three warmups and five samples, with the same 41-field phase-probe object
as S43. Worker time is accumulated work, not serial wall latency. Percent
changes below are medians of paired ratios, not ratios of column medians.

| Warm input / stage | S43 median ms | S44 median ms | Paired change |
| --- | ---: | ---: | ---: |
| 4K coefficient-tokenization work | 211.499 | 179.996 | -14.7% |
| 4K AC-group wall | 53.615 | 47.165 | -9.5% |
| 4K codestream wall | 148.259 | 137.076 | -8.3% |
| 4K complete encode | 518.527 | 507.904 | -2.3% |
| 1080p coefficient-tokenization work | 39.569 | 34.085 | -13.9% |
| 1080p AC-group wall | 14.510 | 13.653 | -6.4% |
| 1080p codestream wall | 59.240 | 56.441 | -3.1% |
| 1080p complete encode | 131.046 | 128.517 | -1.6% |
| Flower coefficient-tokenization work | 4.028 | 3.577 | -5.5% |
| Flower AC-group wall | 3.830 | 3.963 | +3.5% |
| Flower codestream wall | 14.006 | 15.569 | +10.8% |
| Flower complete encode | 28.961 | 31.959 | +9.9% |

Target worker time improves in 7/7, 6/7, and 5/7 pairs. Whole encode
improves in only 5/7, 4/7, and 1/7. Flower's regression also spans
unchanged quantization (+8.8%), entropy optimization (+13.1%), and section
writing (+10.1%). Its only whole-encode win includes a 42.380 ms parent
outlier. The adverse result is retained, not explained away by the faster
counter or removed as noise.

Seven zero-warmup, one-sample production-benchmark pairs give:

| Cold input / complete encode | S43 median ms | S44 median ms | Paired change |
| --- | ---: | ---: | ---: |
| 4K | 631.444 | 607.753 | -0.8% |
| 1080p | 172.971 | 173.367 | -1.0% |
| Flower | 49.959 | 50.433 | +0.9% |

Whole-encode wins are 4/7, 4/7, and 3/7. Quantization paired changes
are +1.7%, -2.1%, and +1.5%, respectively. Parent/candidate total ranges
are 602.336-725.354 / 597.584-678.448 ms at 4K,
164.967-230.012 / 166.803-207.819 ms at 1080p, and
46.108-52.439 / 47.896-74.466 ms for Flower. These mixed cold results
do not establish a stable cold whole-encoder gain.

### Small-image replication

The primary Flower regression triggers an additional seven-pair comparison
in each of two regimes. Before each process, its source binary is copied
to one dedicated executable path and verified by SHA-256; the preceding
process has exited before replacement. The retained binaries are untouched.
This controls the executable path, but not internal code layout or operating
state. All 41 timing fields and raw output remain recorded.

With three warmups/five samples, coefficient-tokenization work changes
-18.2% and complete encode +0.2% (26.289 / 26.479 ms column medians;
3/7 whole-encode wins). With 40 warmups/15 samples, target work changes
-18.3% and complete encode +1.7% (26.912 / 28.055 ms; 2/7 wins).
Quantization changes +0.4% / +2.0%, and codestream wall +3.4% / +3.2%.
The original +9.9% cohort is not replaced by these smaller regressions.
No stable small-image whole-encode speedup is claimed.

A further diagnostic links both original counter bodies into one
executable. Only a process-local environment flag, read once at startup,
selects the counter; it does not change any system setting. Both bodies
remain out of line and match their 74/171 native instruction counts, and
both branches pass the 4,096-case fixture. The same executable hash is
verified before/after the experiment. This controls executable layout
between the two selections through a shared dispatcher; it is not
the unmodified production binary.

Seven Flower pairs with 40 warmups/15 samples give target-work -18.8%
(3.749 / 2.917 ms column medians), quantization +1.6%, codestream +0.4%,
and complete encode +2.1% (26.934 / 26.978 ms). All seven target-work
pairs improve, but only two whole-encode pairs do. Three additional pairs
per large input, with three warmups/five samples, give target-work
-15.5% / -16.4% and whole-encode -0.4% / -4.3% at 4K / 1080p;
each has two of three target and whole-encode wins. 4K codestream wall
regresses +4.2%; 1080p changes -1.1%. These diagnostic results do not
erase the production regressions. Executable path/layout alone does not
explain the Flower observation; operating-state interactions remain a
hypothesis, not a demonstrated cause. The change is retained for its
verified counting mechanism and larger-input target gains, with small-image
whole-workflow behavior still an open measurement/performance issue.

### Isolated completed-frame tokenization

The same completed-frame probe as S43 now covers eight layouts, eight
patterns, two order policies, and two population modes: 256 cases. Each
case uses three warmups and nine samples; three alternating process pairs
yield 768 paired cases. The timed region includes direct tokenization of
all four groups and its output allocations, but excludes frame/order
construction, hashing, and output destruction. The 15-context default map
and one reusable scratch per case are fixed. No CUDA workflow is invoked.

Every paired FNV-1a digest matches across token values, contexts, population
metadata, and sparse symbols/counts. There are 230/256 improving case
medians and 641/768 improving individual pairs. All 64 all-zero or LLF-only
case medians improve, as do all 192 of their individual pairs. Dense and
sparse cases are less uniform; the full ranges below retain regressions.

| Pattern | No populations: median ratio (range) | With populations: median ratio (range) |
| --- | ---: | ---: |
| All zero | 0.355 (0.265-0.432) | 0.379 (0.233-0.461) |
| Dense alternating signs | 0.868 (0.684-1.151) | 0.919 (0.694-1.083) |
| Sparse random | 0.904 (0.810-0.995) | 0.921 (0.806-1.083) |
| Dense random | 0.861 (0.810-1.032) | 0.945 (0.763-1.193) |
| Periodic negatives/zeros | 0.876 (0.667-1.152) | 0.891 (0.736-1.105) |
| Signed extrema | 0.873 (0.724-1.024) | 0.894 (0.689-1.123) |
| LLF only | 0.311 (0.260-0.570) | 0.350 (0.267-0.603) |
| Single last coefficient | 0.744 (0.273-1.073) | 0.773 (0.291-1.099) |

Each table cell summarizes 16 layout/order case ratios. The largest case
regression is dense-random DCT16x32/custom-order/population collection:
3.0018 / 3.5135 ms column medians, paired ratio 1.193. This is not hidden
by the particularly large all-zero gains.

### Qualification and boundaries

All 64 CUDA and 49 CPU tests pass. The expanded 4,096-case fixture also
passes when linked against the frozen S43 AC implementation. Fully
instrumented host AddressSanitizer builds pass direct tokenization, AC
groups, coefficient ordering, and codestream encoder fixtures; no annotation
suppression or partial-library instrumentation is used. This host-only
change does not trigger another full CUDA AQ racecheck.

The 46 independently decoded image pairs retain identical codestream
bytes and decoded Butteraugli scores. They cover seven inputs at distances
0.5/1.2/3, effort 7, plus sample/Flower at effort 9/distance 1.2, each in
encoding-only and final-score collection modes. Strategy selection,
requested encoder scores, and cross-policy byte/score identity are checked
separately. The pinned decoder and linear-sRGB metric policy are unchanged.

Serial/batch identity checks pass. Current-build serial/batch speedups,
not S43/S44 ratios, are 1.003/1.170/1.307 for fully-resident 1080p batch
sizes 1/2/4, 1.004/1.555/1.756 for maximum-throughput 1080p, and
1.048/1.056 for fully-resident 4K batch sizes 1/2.

Boundary telemetry moves from 64 C, 210 MHz, P8, with both software
thermal/power flags inactive, to 72 C, 1282 MHz, P3, with both active.
These are boundary observations, not per-kernel clocks or proof of the
cause of every timing change. No power, cooling, clocks, services, priority,
firewall, or security setting is changed. No privilege/firewall error is
observed in these completed jobs; the earlier reported prompt remains an
unconfirmed explanation for the historical long run. CUDA kernels,
allocation requests, and transfer paths are unchanged; their traces are
not recaptured for this host-only edit.

Evidence is retained under ignored `build-cuda-ninja/profiles/`:
`s44_contiguous_phases.json`, `s44_final_phases.json`, the three
`s44_final_cold_*.json` files, `s44_flower_same_path.json`,
`s44_dispatch_compare.json`, `s44_ac_compare.json`,
`s44_native_summary.json` and native listings, `s44_final_quality.json`,
both CTest logs, `s44_asan.txt`, the parent/dispatch fixture logs,
three batch logs, and boundary telemetry. The source/copy SHA-256 manifest
`s44_final_binary_hashes.json` freezes eleven qualified executables;
`s44_final_validate.py` checks the baseline, artifacts, and result matrices.
Probe source, build/run scripts, and raw per-run output are retained beside
the summaries. None of the diagnostic dispatch code is linked into the
retained production binaries.

## Fused L2 difference and final masking (S45)

Date: 2026-09-05. Baseline: `51790b8` (S44).

### Bottleneck and dataflow

Fresh fully-resident traces identify Butteraugli's pointwise L2 and final
masking kernels as a remaining target. In the exploratory 4K capture,
L2 accounts for 7.166 ms and final masking 3.014 ms of 122.563 ms total
GPU kernel time. This single capture is diagnostic context, not the later
paired performance result; operating state varies substantially.

L2 reads psycho planes and two Malta accumulations, then writes three AC
and three DC planes. The only consumer of those six values is final
masking. Intervening reference-mask erosion and distorted-mask blur touch
`kWork..kWork+4`, not the psycho or Malta inputs. L2 can therefore execute
inside the final kernel, after mask preparation, with its six values kept
in registers. Both stages are pointwise; no cross-thread dependency or
additional synchronization is introduced. Reference/distorted/work/output
strides remain independent, and 64-bit addressing and launch geometry
are preserved. The two original kernels remain intact as a separate-pass
oracle; no fast-math option or numerical policy changes.

The next Malta comparison initializes its two accumulations before adding
responses, so leaving the prior L2 results unmaterialized does not alter
reuse. The former AC/DC storage also serves earlier psycho-image stages;
this change does not remove its allocation. Crop, multiscale composition,
score reduction, and reference-mask caching are unchanged.

The fused operation eliminates six float stores and six float loads per
active pixel, or 48 logical bytes. Two full-resolution and two half-scale
comparisons for the odd 3839x2159 benchmark input (1920x1080 half-scale)
remove 994,752,096 logical intermediate bytes per encode. The odd
1919x1079 input (960x540 half-scale) removes 248,544,096 bytes.
These are source-level traffic counts, not measured DRAM transactions.
Four launches disappear. Allocation requests and host/device copies do
not change.

### Independent qualification and native mechanism

The new CUDA-only fixture checks 12 geometries, packed and independently
padded strides, five value patterns, and three asymmetries: 360 cases,
each with three-stage reuse. It compares final output bit-for-bit against
the original kernels and verifies all guards, immutable inputs, and the
fused path's untouched AC/DC storage. Patterns include signed zeros,
ordinary finite values, subnormals/overflow, non-finite inputs, and values
at or adjacent to asymmetric thresholds. The last reuse changes distorted
inputs and sets unused fused `ac[2]` and DC pointers to null. Empty extents
and all four insufficient-stride cases are covered separately. The scoped
sanitizer subset retains 90 cases, including odd/padded geometry and every
value/asymmetry combination.

All 33 prior Butteraugli native bodies remain identical, including both
oracle kernels. The added fused body uses 40 registers with zero stack,
local, or shared memory; the original L2/final bodies use 32/21 registers.
On the qualified SM86 device, 40 registers still permit six feasible
256-thread blocks per SM, before other scheduling constraints; this is
not a measurement of achieved occupancy or a cross-device claim.
Native instruction-line counts fall from 328 + 352 to 528. Global-load
instructions fall from 18 + 9 to 21, and stores from 6 + 1 to 1. Static
counts include all emitted paths and do not represent dynamic execution.
The final build is checked against all 34 profiled prototype bodies.

### Paired production GPU traces

Three alternating pairs at each large input use the retained S44 benchmark
and the candidate, three warmups and one captured sample, without another
benchmark, build, or sanitizer running concurrently. Every pair has the
same allocation sizes and host/device copy totals. The target changes from
eight L2/final launches to four fused launches. Overall counts are
440 to 436 at 4K and 428 to 424 at 1080p. Reported percent changes are
medians of paired ratios, not ratios of column medians.

| Input / GPU scope | S44 median ms | S45 median ms | Paired change |
| --- | ---: | ---: | ---: |
| 4K L2/final target | 11.153 | 7.298 | -34.6% |
| 4K all kernels | 242.239 | 227.863 | -2.0% |
| 1080p L2/final target | 2.805 | 1.831 | -34.7% |
| 1080p all kernels | 37.291 | 36.138 | -3.5% |

Every target and all-GPU pair improves. Unchanged-kernel paired medians
change -0.3% / -1.0%, with individual regressions retained. Boundary
temperature rises from 65 to 68 C; software thermal/power flags alternate
between inactive and active, and sampled SM clocks include 270/315/390
and 1282 MHz. These are not per-kernel clocks. No operating-system,
security, power, cooling, or clock settings are changed, and no privilege
or firewall failure is observed during these captures.

The preliminary three-pair warm phase cohort remains separate: whole
encode changes -1.7% / -0.1% / +0.4% at 4K / 1080p / Flower, while
quantization changes +2.1% / -1.9% / -0.7%. A Flower candidate outlier
reaches 32.199 ms versus 25.900 ms in its pair. These mixed wall results
are not replaced or explained away by the GPU target improvement.

### Complete-workflow measurements

The primary warm cohort uses seven alternating process pairs per input,
three warmups and five samples, with the same 41-field phase probe as S44.
All timing changes below are median paired ratios.

| Warm input / stage | S44 median ms | S45 median ms | Paired change |
| --- | ---: | ---: | ---: |
| 4K quantization pipeline | 340.412 | 333.321 | -1.8% |
| 4K codestream wall | 143.856 | 135.687 | +3.2% |
| 4K complete encode | 507.074 | 501.897 | -0.8% |
| 1080p quantization pipeline | 65.041 | 64.913 | -1.8% |
| 1080p codestream wall | 60.028 | 61.120 | -5.0% |
| 1080p complete encode | 132.436 | 131.644 | -1.8% |
| Flower quantization pipeline | 13.123 | 12.674 | -1.8% |
| Flower codestream wall | 13.120 | 12.512 | -2.3% |
| Flower complete encode | 27.763 | 26.797 | -3.3% |

Whole encode improves in 4/7, 5/7, and 7/7 pairs; quantization improves
in 6/7, 5/7, and 7/7. The unchanged CPU serializer remains variable:
4K codestream paired timing regresses despite its lower column median,
and AC-tokenization wall changes +6.1%. A 1080p candidate whole-encode
outlier reaches 164.392 ms versus 126.121 ms; the 4K parent includes
561.786 ms versus 501.897 ms. Neither is discarded.

Seven zero-warmup, one-sample production-benchmark pairs give:

| Cold input / stage | S44 median ms | S45 median ms | Paired change |
| --- | ---: | ---: | ---: |
| 4K quantization pipeline | 436.814 | 426.188 | -3.0% |
| 4K complete encode | 612.936 | 598.807 | -1.9% |
| 1080p quantization pipeline | 100.988 | 99.075 | +0.3% |
| 1080p complete encode | 183.346 | 171.161 | +0.4% |
| Flower quantization pipeline | 27.711 | 28.598 | +0.1% |
| Flower complete encode | 49.592 | 50.446 | -0.2% |

Cold whole-encode wins are 5/7, 3/7, and 4/7. Parent/candidate total
ranges are 600.247-720.245 / 592.038-619.856 ms at 4K,
170.462-222.770 / 165.697-309.110 ms at 1080p, and
48.295-77.910 / 48.704-64.670 ms for Flower. The 1080p candidate
outlier also reaches 170.024 ms in quantization versus 106.215 ms in
its parent pair. No universal warm/cold whole-encoder gain is claimed.

Warm/cold boundary telemetry moves from 65 C, 210 MHz, P8, with both
software limiting flags inactive, to 73 C, 1282 MHz, P3, with both active.
As with the GPU traces, this limits timing interpretation without proving
the cause of every outlier. No settings are changed.

### Completed qualification

All 65 CUDA and 49 CPU tests pass, including the 360-case guarded fixture.
The new kernel's scoped memcheck, initcheck, synccheck, and racecheck pass
all 90 subset cases with three-stage reuse. Full AQ memcheck (including
stream-ordered race tracking and leak checks), initcheck, and synccheck
also pass. All seven checks finish normally with zero reported errors or
hazards, and the AQ leak summary is zero. The historical full-AQ racecheck
is not repeated or presented as passed.

All 46 independently decoded image pairs have identical codestream bytes
and Butteraugli scores. Coverage remains seven inputs at distances
0.5/1.2/3 and effort 7, plus sample/Flower at effort 9/distance 1.2, each
with encoding-only and final-score collection. Strategy decisions,
requested encoder scores, and cross-policy byte/score identity are
validated separately. The pinned decoder and linear-sRGB metric policy
are unchanged.

Current-build serial/batch identity checks pass. Paired speedups for
fully-resident 1080p batch sizes 1/2/4 are 0.987/1.317/1.338; for
maximum-throughput 1080p they are 1.032/1.495/1.828. Fully-resident 4K
batch sizes 1/2 give 1.056/1.009, with individual regressions down to
0.964/0.924. These compare serial and batch execution of this build,
not S44 against S45, and the marginal/negative results are retained.

### Isolated launch-window measurements and artifacts

One executable contains both the original separate-pass and fused entry
points. Eight geometry/layout cases each run three alternating pairs with
three warmups and nine CUDA-event samples per version. Identical device
copies restore the two Malta accumulations before every sample, outside
the timed window; allocation, input construction, readback, and hashing
are also excluded. The event interval includes the one- or two-kernel
launch window, including inter-launch gaps, not just a sum of kernel time.
Each case uses the same finite input and asymmetry 1.7 for both versions.

| Geometry / layout | Separate median ms | Fused median ms | Paired change |
| --- | ---: | ---: | ---: |
| 31x63 packed | 0.016384 | 0.010240 | -33.3% |
| 31x63 padded | 0.017408 | 0.011264 | -35.3% |
| 512x512 packed | 0.160768 | 0.107520 | -32.3% |
| 512x512 padded | 0.158720 | 0.105472 | -33.5% |
| 1920x1080 packed | 1.135616 | 1.115136 | -1.1% |
| 1920x1080 padded | 1.127424 | 0.734208 | -34.9% |
| 3840x2160 packed | 4.454400 | 2.890752 | -35.1% |
| 3840x2160 padded | 4.463616 | 2.896896 | -35.1% |

All eight case medians and 23/24 individual pairs improve. The packed
1080p case is explicitly mixed: one pair regresses 1.135616 to 1.284096 ms,
while another has a 1.627136 ms parent versus 0.741376 ms candidate.
Every output FNV-1a hash, including padding, matches. These hashes supplement
the bitwise guarded fixture; they do not replace it or prove stable speedups
across devices or operating states.

Evidence remains under ignored `build-cuda-ninja/profiles/`: the fresh
`s45_baseline_*.sqlite` captures, `s45_profile_*` paired traces and summary,
`s45_native_*` and `s45_final_native_*` listings, preliminary/final phase
and cold JSON, `s45_final_quality.json`, both CTest logs, the seven sanitizer
logs, three batch logs, `s45_l2_probe.txt` / `s45_l2_summary.json`, and their
source/build/run scripts. The profiled executable and its SHA-256 are
retained separately. `s45_final_binary_hashes.json` freezes twelve qualified
executables and matches each source/copy. `s45_final_validate.py` checks
those artifacts and result matrices against the S44 baseline.

An initial artifact-check assertion expected the fixture's final unflushed
summary line, which the sanitizer logs did not capture. The corrected gate
requires all three explicitly flushed geometry-completion records ending
at exactly 90 cases, plus the zero-error/hazard summary; each sanitizer
command also returned success. This was a log-validation mismatch, not a
test failure, and no production code or test result was changed to pass it.

## Fused vertical blur and low/medium construction (S46)

Baseline: `089fce9` (the S45 implementation plus its logical-traffic
documentation correction). The retained S45 executables are the unchanged
before-version. This checkpoint targets fully-resident Butteraugli's
33-tap low-frequency preparation, not exact-coefficient mode.

### Dependency and storage change

Previously each of three XYB channels ran a horizontal and vertical blur,
then `LowMediumKernel` read all three blurred planes and the original XYB
planes to construct six outputs. The low-B output depends on both blurred
Y and blurred B, so blindly fusing independent channel kernels would not
preserve that dependency.

The new `ConvolutionLowMediumKernel<48>` jointly loads three vertical
halos into shared storage, accumulates each channel in the original
33-tap order, performs the original separately rounded divisions, and
evaluates the unchanged six output expressions. Edge normalization keeps
the original valid-tap order. All partial-tile lanes participate in the
cooperative loads and barrier before inactive output lanes skip work.
A flattened one-dimensional tile grid avoids a 65,535-row grid.y limit.

Three horizontal kernels remain unchanged. Their packed intermediates
reuse planes 24-26, whose blurred RGB values have already been consumed by
Opsin; XYB occupies separate planes 21-23. This is safe for both full and
subsampled extents and does not enlarge an arena. The original vertical
and low/medium kernels remain available through a CUDA-internal reference
entry, independently exercising the old seven-launch computation.

Each psycho-image pass drops from seven launches to four and avoids
three blurred-plane writes plus three reads: 24 logical bytes per active
pixel. The profiled encode has three full-resolution and three half-scale
psycho passes, removing 18 launches and 72*(full pixels + half pixels)
logical bytes. That is 746,064,072 bytes for the actual 3839x2159 benchmark
with 1920x1080 half scale, and 186,408,072 bytes for 1919x1079 with 960x540
half scale. These are eliminated program-level accesses, not measured
DRAM traffic. No allocation-capacity reduction is claimed.

### Tile investigation and native code

Four fixed heights were evaluated at width 32 with 256 threads. Every
variant passed the original 160 guarded differential cases and the
1x2,097,153 image. The latter exceeds 65,535 tile rows for heights 16/32,
but not 48/64; the final fixture therefore raises its height to 4,194,305
and adds the 47/48/49 and 95/96/97 boundaries, for 220 normal cases.
The isolated probe uses one executable containing both reference and
candidate, three alternating-order pairs per packed/padded geometry,
ten warmups and fifteen CUDA-event samples per version. Launch-window
times include inter-kernel gaps; host setup and nine-plane output hashing
are outside the windows. All compared intermediate/output hashes match.

| Tile height | Shared bytes | Packed 1080p paired ratio | Packed 4K paired ratio | Padded 4K paired ratio |
| --- | ---: | ---: | ---: | ---: |
| 16 | 18,568 | 0.744 | 1.025 | 1.037 |
| 32 | 24,712 | 0.765 | 0.961 | 0.956 |
| 48, retained | 30,856 | 0.743 | 0.942 | 0.932 |
| 64 | 37,000 | 0.720 | 0.915 | 0.928 |

These are separate cohorts, each normalized to its own original-path
measurements; they are not a direct inter-tile paired tournament. At
1080p the candidate medians themselves are 1.020/1.060/1.041/1.103 ms for
16/32/48/64 rows. Reference outliers make some ratios look better than
that ordering. The 16-row candidate regresses in all six 4K observations;
64 rows helps 4K but is slower on the smaller images. Height 48 is retained
as a balanced fixed policy, not a demonstrated universal optimum. Halo
traffic versus shared-memory residency explains the structural tradeoff;
achieved occupancy and cache transactions have not been measured.

All four variants use 46 registers and zero stack/local allocation.
The retained 48-row shared footprint is 30,856 bytes. Original vertical33
uses 55 registers / 12,424 shared bytes; horizontal33 uses 55 / 4,744;
the separate low/medium body uses 22 registers and no shared storage.
All 34 preexisting Butteraugli native bodies are identical to S45 after
normalizing only the translation-unit symbol hash. There is one new body.
The final comment/fixture rebuild also retains the profiled native code.

### Production GPU traces

Three alternating-order pairs per workload use three warmups and one
captured fully-resident sample. The bundle includes all horizontal33,
vertical33, and low/medium work: 42 original launches versus 24 fused
launches. All six target-bundle observations improve.

| Workload | Parent target ms, three runs | Candidate target ms | Median paired target change | Total GPU change |
| --- | --- | --- | ---: | ---: |
| Odd padded 4K | 35.458 / 34.023 / 37.236 | 28.706 / 28.404 / 32.286 | -16.5% | -0.2% |
| Odd padded 1080p | 5.389 / 5.332 / 5.479 | 4.079 / 4.119 / 3.980 | -24.3% | -4.1% |

Total launches are 436->418 at 4K and 424->406 at 1080p. Five allocation
requests and all copies match pairwise, including counts and byte totals.
Other-kernel time changes +2.4% / -0.6%. One 4K total-GPU observation
regresses 2.3%; the unchanged work absorbs much of the target gain.
No uniform 4K whole-encoder improvement follows from these traces.
Boundary samples span 65-67 C and 262-1282 MHz, with software thermal/power
flags often active. These are boundary observations, not per-kernel state.

All twelve captures/export operations completed normally. The first
summary attempt rejected Nsight's explicit template-argument formatting;
correcting the name predicate made the saved captures pass the expected
42->24 target and 18-launch total-delta checks. No capture was replaced or
discarded for this reporting-script error.

### Final qualification and wall timings

The final Release builds pass all 67 CUDA and 49 CPU CTests. The expanded
220-case fixture passes bit-for-bit across 22 geometries, packed/padded
strides, misaligned offsets, Gaussian/impulse/irregular weights, signed
zeros, random values, extreme magnitudes, cancellation boundaries, and
NaN/Inf inputs. Three reuse iterations check guards and immutable inputs;
the last changes the input values and leaves unused blurred pointers null
with zero blurred stride. Empty extents and invalid required strides are
also covered. The separate 1x4,194,305 test passes.

The repeated final isolated probe also has eight favorable case medians
and 24/24 favorable individual pairs, with all nine-plane hashes matching.
Packed/padded median paired ratios are 0.793/0.516 at 31x63,
0.733/0.736 at 512x512, 0.770/0.771 at 1920x1080, and
0.792/0.794 at 3840x2160. The final 4K candidate medians are
3.944/3.891 ms versus 5.003/4.900 ms for the original path, much faster
absolute timings than the earlier tile cohorts for both paths. Native
code is unchanged. A boundary sample now records 1770 MHz with software
limits inactive, rather than the earlier lower clocks/active flags;
this illustrates operating-state variability without proving its share
of the timing difference.

Primary warm timing uses seven alternating-order pairs per input, three
warmups and five samples per executable, retaining all 41 phase fields.
Entries below report marginal medians and the median of within-pair
ratios, which need not equal their quotient.

| Workload | Phase | Parent / candidate median ms | Median paired change | Candidate wins |
| --- | --- | ---: | ---: | ---: |
| Odd 4K | Quantization | 205.972 / 204.486 | +0.3% | 3/7 |
| Odd 4K | Codestream | 142.198 / 151.641 | +10.1% | 2/7 |
| Odd 4K | Total | 383.214 / 385.129 | +4.0% | 2/7 |
| Odd 1080p | Quantization | 62.658 / 68.830 | +9.9% | 3/7 |
| Odd 1080p | Codestream | 65.587 / 78.166 | +19.2% | 2/7 |
| Odd 1080p | Total | 135.982 / 151.814 | +11.6% | 3/7 |
| Flower | Quantization | 14.059 / 13.244 | -6.5% | 6/7 |
| Flower | Codestream | 16.255 / 14.647 | -9.9% | 4/7 |
| Flower | Total | 31.716 / 29.415 | -7.3% | 5/7 |

These adverse large-image results are not discarded. For example, a 4K
pair is 386.741->494.689 ms overall with codestream time
139.646->237.526 ms; a 1080p pair is 142.965->252.505 ms.
Flower also has a 30.065->47.125 ms regression. Large changes in
unchanged CPU work motivate the same-executable control below, rather
than an unsupported attribution to CUDA, file layout, or thermal state.

Seven cold pairs per input use zero warmups and one measured sample.

| Workload | Parent / candidate total median ms | Total paired change | Quantization paired change | Total wins |
| --- | ---: | ---: | ---: | ---: |
| Odd 4K | 517.852 / 576.718 | +6.6% | -1.0% | 2/7 |
| Odd 1080p | 200.597 / 202.857 | -7.1% | -5.1% | 4/7 |
| Flower | 57.057 / 56.873 | -2.9% | -0.8% | 4/7 |

Cold total ranges are 480.854-762.469 / 492.643-683.227 ms at 4K,
176.247-388.938 / 180.407-323.682 ms at 1080p, and
53.737-110.335 / 52.335-66.084 ms for Flower. Warm/cold boundary samples
span 70-76 C, P0, 1770->1282 MHz, with both software flags inactive at
the sampled boundaries. No clocks, cooling, power, priority, services,
security settings, or firewall rules were changed.

The 46 decoded-image pairs all pass with identical codestream bytes,
decoded Butteraugli scores, strategy reports, and requested final scores.
Encoding-only and score-collecting policies also match within each version.
The matrix retains the seven S45 inputs, distances 0.5/1.2/3 at effort 7,
plus sample/Flower effort-9 cases, using the same pinned libjxl decoder and
linear-RGB metric procedure.

All seven scoped Compute Sanitizer runs terminate successfully: the new
40-case / three-reuse fixture under memcheck, initcheck, synccheck, and
racecheck, then the complete AQ fixture under memcheck, initcheck, and
synccheck. Every error/hazard summary is zero; AQ memcheck additionally
checks stream-ordered races and leaks and reports zero leaked bytes.
The new scoped racecheck takes about 62 seconds; AQ memory/init/sync
checks take about 49/31/30 seconds. Full-AQ racecheck was not repeated.
The earlier S30 roughly 52-minute aborted run remains an abort, not a pass
and not a proven firewall failure.

Current-version batch checks retain serial/batch byte identity. Paired
batch-versus-serial median speedups for batch sizes 1/2/4 are
0.988/1.481/2.035 at fully-resident 1080p and 0.950/1.841/2.108 at
maximum-throughput 1080p; fully-resident 4K is 1.012/1.601 for sizes 1/2.
These compare batch scheduling against serial work in the same version,
not S46 against S45. Individual size-1 observations regress as far as
0.822x at 4K.

The final artifact validator verifies the frozen S45 baseline and thirteen
S46 executables, unchanged oracle sources and 34 preexisting native bodies,
profiled/final native identity, complete test/sanitizer logs, all image
policy checks, 21 warm + 21 cold pairs, six GPU profile pairs, 24 isolated
pairs, and the three batch cohorts.

### Same-executable control

The diagnostic was built separately under the ignored profile directory
without modifying production libraries or introducing a supported runtime
option. It embeds the exact original per-channel horizontal/vertical/
low-medium sequence, including its reused horizontal scratch, alongside
the new sequence. A process-local environment flag selects the path once
and emits a checked marker. Both modes pass the full AQ fixture, and
all 35 native Butteraugli bodies match production after normalizing the
translation-unit symbol prefix. No second archived Butteraugli object is
linked. Both choices therefore run from the same executable, with the
same CPU serialization code and file layout.

Seven alternating-order warm pairs per input retain the same three
warmups, five samples, 41 fields, and output-size checks as the primary
cohort. Every process contains exactly one expected path marker.

| Workload | Parent / candidate total median ms | Total paired change | Quantization paired change | Codestream paired change | Total wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Odd 4K | 595.905 / 538.349 | -2.1% | -7.3% | -12.8% | 5/7 |
| Odd 1080p | 236.270 / 182.968 | -20.3% | -7.9% | -20.5% | 6/7 |
| Flower | 33.914 / 31.759 | -2.7% | -5.0% | -2.2% | 5/7 |

This control does not reproduce the primary large-image median
regressions, but the large gains in unchanged CPU work must not be
attributed to this GPU fusion. It is a later cohort with changing
operating state, not proof that executable layout caused the initial
regressions. At 4K, total ranges are 424.352-955.907 /
396.248-993.326 ms, including a 678.995->916.501 ms adverse pair.
Flower includes 87.681->31.759 ms and 32.479->40.357 ms pairs.
Boundary state moves from 75 C / P0 / 1762 MHz with software flags
inactive to 78 C / P3 / 1282 MHz with both active. The sources of the
wall-time variation remain unresolved. A short read-only post-run CPU
activity sample also cannot establish what happened during those runs.

### Retention and evidence

Retain the 48-row fusion for its proven dependency-preserving elimination
of three intermediate writes/reads and 18 launches, consistent isolated
and production target-kernel gains, and complete correctness qualification.
Do not advertise the same-executable cohort's large wall improvements as
a stable encoder speedup or hide the adverse primary warm/cold results.
The fully-resident backend is still not demonstrated maxed out.

Thirteen qualified production executables remain under
`build-cuda-ninja/profiles/s46_retained_*.exe`, with exact source/copy
SHA-256 checks in `s46_final_binary_hashes.json`. The two control executables
have their own `s46_control_hashes.json`. The final validator additionally
checks both control AQ paths, all 21 marked control pairs, output sizes,
and native identity; production source contains no diagnostic switch.

Reproduction/evidence scripts are `s46_native.py`,
`s46_low_medium_probe{.cpp,_run.ps1}`, `s46_low_medium_summary.py`,
`s46_profile_pairs.ps1`, `s46_profile_summary.py`,
`s46_phase_build.ps1`, `s46_phase_compare.py`,
`s46_final_run.ps1`, `s46_final_quality.py`, `s46_sanitizers.ps1`,
`s46_control_{build.ps1,native.py,aq.py,compare.py,run.ps1}`,
`s46_freeze.ps1`, and `s46_final_validate.py` in the ignored profile
directory. Individual tile source/binary/native/timing snapshots,
all traces and SQLite exports, untrimmed wall observations, sanitizer
logs, and decoded-image reports are retained. There was no observed
administrator/firewall block in this checkpoint and no security-setting
change.

A next bounded lead is to re-audit scratch lifetimes now that the
pointwise L2 intermediates and low/medium blurred outputs are gone.
For example, image temporaries are dead during difference/mask work,
while only two Malta accumulations remain live; the full-scale final
map must still survive subsequent half-scale psycho construction.
This is an opportunity to investigate, not an allocation saving already
implemented by S46.

## Compact prepared Butteraugli scratch (S47)

Baseline: retained S46 executables at `e7f0bcb`. This checkpoint reuses
storage whose values are already dead; it does not change device arithmetic,
kernel bodies, launch count, or transfers. The prepared allocation changes
from 33 full working planes to 27: twenty psycho-image planes, one cached
reference mask, and six reusable work planes. The optional ten-plane
half-scale reference cache and two reduction buffers are unchanged.

### Lifetime audit and implementation

S45 removed the materialized L2 values, and S46 removed three intermediate
vertical-blur outputs, but their old scratch layout still reserved twelve
work planes. The remaining live sets fit into six. In particular, the
full-scale comparison result survives half-scale psycho construction in the
caller's distance map, not in the internal pre-crop/half-scale staging plane.
That internal staging plane is written only after its other work is dead.

| Working planes | Psycho construction | Difference and mask work |
| --- | --- | --- |
| 0-9 | Persistent main reference psycho image | Read-only reference |
| 10-19 | Distorted psycho image | Read-only distortion until the next scale |
| 20 | Cached main reference mask | Read-only cached mask |
| 21-22 | Blurred RGB, then pointwise XYB | Two Malta AC accumulations |
| 23 | Third blurred RGB / XYB plane | Mask input, then blurred distorted mask |
| 24 | Five-tap horizontal scratch; then first 33-tap horizontal plane; then frequency scratch | Mask horizontal scratch; then pre-crop/half-scale output |
| 25 | Second 33-tap horizontal plane | Uncached half-scale reference mask |
| 26 | Third 33-tap horizontal plane | Fuzzy reference mask |

Expanded or subsampled original RGB is staged in the first three
not-yet-produced low-frequency output planes. Expanded reference uses
main-reference outputs; expanded/subsampled distortion uses distorted
outputs. Subsampled reference uses its own packed half-scale outputs,
including their smaller row stride. These input values are dead after
Opsin, before low/medium construction overwrites those output planes.

The five-tap RGB blur writes planes 21-23. Opsin loads all six original
and blurred RGB values for one pixel before writing its three XYB values,
so blurred RGB and XYB may coincide pointwise. No cross-pixel dependency
or `restrict` contract is introduced. Three distinct horizontal 33-tap
results then occupy 24-26 until the joint vertical/low-medium pass finishes.
Subsequent frequency passes reuse plane 24 serially.

During mask work only the two Malta accumulations remain live in 21-23.
The distorted mask's separable blur uses 23 -> 24 -> 23, with ordered
horizontal and vertical launches, not an in-place neighborhood kernel.
The horizontal intermediate is dead before the fused L2/final kernel
writes plane 24. Production plans leave the unused third AC and three
DC pointers null; the fused final kernel does not read them. The retained
separate L2 oracle still receives all its original pointers.

### Exact resource result

Each removed plane saves `align_up(working_width * working_height * 4, 64)`
requested arena bytes. Six are removed without another allocation or copy.
Three alternating-order production profile pairs per workload directly
confirm five allocation requests, with only the Butteraugli request changing:

| Workload | Parent / candidate Butteraugli arena bytes | Saving bytes | Parent / candidate sum of five requests |
| --- | ---: | ---: | ---: |
| Odd 4K, 3839x2159 | 1,177,274,420 / 978,352,436 | 198,921,984 | 2,608,448,064 / 2,409,526,080 |
| Odd 1080p, 1919x1079 | 294,121,460 / 244,426,868 | 49,694,592 | 651,957,638 / 602,263,046 |

The Flower 510x532 layout formula gives 38,537,908 -> 32,026,036 arena
bytes, saving 6,511,872 bytes; it is not a third captured allocation trace.
Cached-reference and Gaussian-weight bytes are unchanged. The separately
reported comparison-scratch statistic drops by six unaligned logical
plane sizes. A permanent independent layout oracle checks all four public
memory statistics, including every 64-byte aligned allocation boundary.

These are exact requested-allocation reductions, not a promise of the same
drop in driver-reported dedicated VRAM. Pool allocation granularity and
retention still apply. The existing 3,220,963,328-byte pool retention
threshold is unchanged; no memory-pool or system setting is adjusted.

All 35 native Butteraugli bodies match S46 after normalizing only the
translation-unit hash in symbol names; none is added or removed. The
profile pairs preserve the full launch structure, including grids, blocks,
registers, shared/local allocation, and all transfers: 418 launches at
4K and 406 at 1080p. This is a storage-lifetime optimization, not a
kernel-fusion or logical-traffic reduction claim.

### Prepared-map differential and permanent regression coverage

A retained S46 CUDA archive and the current archive are linked separately
against the same probe object and unchanged supporting libraries. The
probe covers 29 geometries from 1x1 through odd 4K, including skinny,
expanded, multiscale-boundary, tile-edge, and padded-stride cases. Three
option policies and three reuse passes produce 261 before/after pairs.
Each case prepares its reference once, compares a distortion, compares
identity, then restores the original distortion. Device inputs and guards
remain intact; comparison adds no allocation; restored maps and score
bits match their first result.

The probe writes the complete padded map and double score for every pass.
The two 399,602,628-byte files match byte for byte, not merely by a sampled
map, aggregate score, or hash. Every row also checks unchanged cached bytes,
the exact aligned arena saving, and the logical comparison-scratch saving.

The permanent prepared-Butteraugli fixture grows to 26 geometry/option
cases plus one identity case. It checks the independent 27-plane allocation
formula, guarded inputs/outputs, bitwise determinism, changed-distortion
reuse and restoration, and no extra comparison allocations. CPU map/score
tolerances remain 1.5e-3 and identity tolerance 1e-7; observed worst errors
are 0.000219455 / 0.000020504. Existing failure invalidation checks remain.
The sanitizer subset explicitly exercises 3x7 expansion/crop, 8x8
single-scale, and 17x29 multiscale, each with four comparison passes.

### Timing observations, including regressions

Three alternating-order GPU profile pairs use three warmups and one
captured encode each. Total GPU kernel times in pair order are:

| Workload | Parent GPU ms | Candidate GPU ms | Median paired change | Wins |
| --- | --- | --- | ---: | ---: |
| Odd 4K | 210.845887 / 220.239876 / 236.616386 | 221.165184 / 234.523247 / 225.129117 | +4.9% | 1/3 |
| Odd 1080p | 34.327186 / 35.398416 / 33.953660 | 33.670398 / 34.509891 / 36.175839 | -1.9% | 2/3 |

The 4K regression is retained, not discarded because the code is smaller.
Boundary observations span 64-68 C and 262-1282 MHz, with software thermal
and power flags sometimes active. That state limits interpretation but
does not establish the cause of any individual timing difference.

Seven alternating-order warm pairs per input use three warmups, five
samples, all 41 phase fields, and matching output sizes. Changes below
are medians of candidate/parent ratios within pairs, not ratios of the
two marginal medians. Negative means faster.

| Workload | Parent / candidate total median ms | Total paired change | Quantization paired change | Codestream paired change | Total wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Odd 4K | 486.394 / 491.609 | +0.2% | +0.3% | -6.7% | 2/7 |
| Odd 1080p | 122.444 / 123.419 | +0.8% | +1.0% | +1.5% | 2/7 |
| Flower | 27.842 / 28.775 | -4.0% | -1.6% | -3.7% | 4/7 |

All observations remain in the evidence. Warm Flower includes an adverse
27.839 -> 43.878 ms pair; the unchanged codestream phase changes
13.155 -> 23.352 ms in that pair. The code change cannot by itself
justify attributing unchanged host-work variation to memory reuse.

Seven fresh-process cold pairs per input give:

| Workload | Parent / candidate total median ms | Total paired change | Quantization paired change |
| --- | ---: | ---: | ---: |
| Odd 4K | 588.955 / 576.347 | +0.1% | -1.4% |
| Odd 1080p | 169.872 / 167.766 | -1.5% | -2.9% |
| Flower | 50.254 / 51.577 | +5.9% | +6.1% |

Cold total ranges are 574.015-625.469 / 570.962-613.883 ms at 4K,
163.441-204.461 / 159.084-226.110 ms at 1080p, and
46.234-51.715 / 46.663-70.959 ms for Flower. The adverse cold Flower
result remains a limitation. Warm/cold boundary state is 66 -> 72 C,
P3 / 1282 MHz, with both software flags active. No stable whole-encoder
speedup is demonstrated by this checkpoint.

### Isolated prepared-comparison follow-up

To investigate the adverse 4K trace, a short additional probe links the
same host object separately against the frozen old/new CUDA archives.
It prepares once, performs five warmups and eleven measured comparisons,
and times the synchronous public `Compare` call, including its submission
wait but excluding allocation, preparation, transfers, and readback.
It uses bounded synthetic RGB at each extent; the 510x532 case is not
the Flower photograph. Inputs/output guards, unchanged input values,
no comparison allocations, full-map checksums, score bits, and the
exact arena saving are checked for every process.

Seven alternating-order pairs per extent give:

| Extent | Parent / candidate median ms | Median paired change | Wins |
| --- | ---: | ---: | ---: |
| 3839x2159 | 88.0575 / 87.4383 | +0.25% | 3/7 |
| 1919x1079 | 20.6830 / 20.6517 | -0.36% | 5/7 |
| 510x532 | 1.5706 / 1.5582 | -2.00% | 5/7 |

The compact layout does not reproduce the trace cohort's +4.9% 4K
regression in this comparison-only cohort. This is not proof of the
trace regression's cause, a same-executable control, or an end-to-end
speedup. All observations remain, including small-case pairs
3.3341 -> 1.5967 ms and 1.4955 -> 1.5681 ms. Boundary state is
67 -> 72 C, P3 / 1282 MHz, with both software flags active.
The production sources, libraries, and thirteen frozen executables are
not rebuilt or modified for this follow-up.

The source and scripts are `s47_prepared_timing.cpp`,
`s47_prepared_timing.py`, and `s47_prepared_timing_run.ps1`; the
two executables have `s47_prepared_timing_hashes.json`. All 21 pairs,
eleven raw samples per process, outputs, summaries, and boundary
observations are retained and checked by the final validator.

### Final qualification and retention

Both complete suites pass: 67 CUDA and 49 CPU tests. The 46-pair decoded
image matrix covers the established seven inputs at distances 0.5/1.2/3.0
and effort 7, plus sample/Flower at effort 9, under encoding-only and
requested-final-score policies. Encoded bytes, decoded Butteraugli scores,
strategy reports, and requested score reports match S46, and collecting
the final score does not change encoded output. The pinned decoder/metric,
linear-RGB interpretation, and quality tolerances are unchanged.

Current-version batch checks preserve serial/batch bytes. Median paired
batch-versus-serial speedups for sizes 1/2/4 are 0.952/1.214/1.420 at
fully-resident 1080p and 0.955/1.538/1.994 at maximum-throughput 1080p.
Fully-resident 4K gives 1.116/1.071 for sizes 1/2. These are within-version
scheduling comparisons, not S47-versus-S46 improvements. Individual size-1
observations regress to 0.693x at maximum-throughput 1080p.

All seven scoped CUDA sanitizer checks pass: prepared Butteraugli
memcheck/initcheck/synccheck/racecheck, and full AQ
memcheck/initcheck/synccheck. Prepared racecheck completes in about
14 seconds, with zero errors or hazards. AQ memcheck additionally checks
stream-ordered races and full leaks, reporting zero leaked bytes; AQ
memory/init/sync checks take about 35/23/17 seconds. Full-AQ racecheck
is not repeated. The earlier S30 approximately 52-minute aborted run
remains an abort, not a pass or a confirmed firewall failure. The entire
serial final build/test/benchmark/quality/batch/sanitizer workflow takes
about seven minutes and encounters no observed administrator/firewall block.

Thirteen qualified production executables are frozen as
`build-cuda-ninja/profiles/s47_retained_*.exe`, with source/copy SHA-256
checks in `s47_final_binary_hashes.json`. The prepared differential's
two executables, common object/source, old/new CUDA archives, and original
CUDA source/header snapshots have a separate `s47_prepared_hashes.json`.
The fixture helpers included by the probe are verified unchanged from
S46; the CUDA library is unchanged between the original map probe and
final qualification (the final build only rebuilds the strengthened test).

The final validator checks both production binary manifests, probe
artifacts/archives, unchanged oracle sources and all 35 native bodies,
67/49 test logs, seven sanitizer summaries and scoped completion markers,
the permanent 27-case completion marker, all 46 image/policy checks,
21 warm + 21 cold pairs, six identical-structure profile pairs, complete
261-pair map bytes and memory deltas, and all three batch cohorts.

Retain the compact layout for its proven six-plane storage reduction,
explicit non-overlapping lifetime argument, and complete correctness
qualification. Keep the adverse timing observations visible; neither
GPU execution nor whole-encoder speed has a stable universal improvement.
This resource improvement does not establish that CUDA is maxed out.

Reproduction/evidence scripts under the ignored profile directory include
`s47_prepared_probe{.cpp,_run.ps1}`, `s47_prepared_compare.py`,
`s47_native.py`, `s47_profile_pairs.ps1`, `s47_profile_summary.py`,
`s47_phase_build.ps1`, `s47_phase_compare.py`, `s47_report.py`,
`s47_final_run.ps1`, `s47_final_quality.py`, `s47_sanitizers.ps1`,
`s47_freeze.ps1`, and `s47_final_validate.py`. Raw maps, all traces and
SQLite exports, untrimmed timing observations, decoded-image reports,
and terminal sanitizer logs are retained. No security, cooling, power,
clock, service, or priority settings are changed.

A next bounded lead is the three five-tap vertical RGB blurs immediately
followed by pointwise Opsin. In the current candidate traces that bundle
has 24 launches and median 13.229 / 2.195 ms at odd 4K / 1080p.
Keeping three horizontal RGB intermediates in the six-plane work set
could permit one joint vertical/pointwise kernel, subject to exact
mirroring, tap order, non-finite handling, register pressure, and guarded
differential checks. This is an investigation lead, not an implemented
fusion or a measured speedup.

## Fused mirrored RGB blur and Opsin conversion (S48)

Baseline: retained S47 executables at `bf458d6`. The three five-tap
vertical RGB blurs write full intermediate planes that the immediately
following pointwise Opsin kernel reads. S47's three-pair candidate traces
put this 24-launch bundle at median 13.229 / 2.195 ms at odd 4K / 1080p,
before including the unchanged horizontal blurs.

### Dependency-preserving fusion

Three unchanged horizontal five-tap kernels now retain separate packed
RGB intermediates. One joint vertical/Opsin kernel cooperatively loads
three directional halos, computes each channel's five taps in the original
order, divides by the original ordered weight sum, and evaluates the same
Opsin expressions directly from those blurred values and original RGB.
The vertical halo uses repeated reflection, including one-pixel extents;
it does not use the truncated/renormalized edge rule of the longer blurs.
Partial-tile threads load data and reach the barrier before output bounds
checks. A flattened tile grid avoids a 65,535-row grid-y limit.

The original `MirroredConvolution5Kernel` and `OpsinKernel` remain unchanged
as independent separate-pass oracles. The new pointwise helper preserves
the existing non-finite check, clamps, fast-log approximation, and explicit
unfused multiply/add operations. No global fast-math or reduced precision
is introduced. The internal test entry permits the oracle's original
shared horizontal scratch and blurred-RGB/XYB in-place layout; production
uses three disjoint horizontal intermediates and separate original RGB.

| Phase | Planes 21-23 | Planes 24-26 | Original RGB |
| --- | --- | --- | --- |
| Three horizontal five-tap blurs | Not yet produced | Three packed RGB intermediates | Read-only external or staged low-output planes |
| Joint vertical/Opsin | XYB output | Read-only RGB intermediates | Read-only pointwise input |
| Horizontal 33-tap blurs | Read-only XYB | Reused for packed XYB intermediates | Dead |
| Joint vertical/low-medium | Read-only XYB | Read-only XYB intermediates | Staged storage may become low-frequency output |

The six-plane work set still fits S47's 27-plane allocation. Horizontal
RGB intermediates are dead before horizontal XYB construction, and staged
original RGB is dead before low-frequency construction. Reference caches,
mask work, cropped/half-scale map staging, and external result lifetimes
are unchanged.

The fusion removes three launches and three blurred-plane writes/reads
per psycho pass. At the normal six psycho passes per encode, that is
18 launches and `24 * 3 * (full_pixels + half_pixels)` logical bytes of
intermediate materialization. For odd 4K / 1080p, the byte counts are
746,064,072 / 186,408,072. These are source-level intermediate accesses,
not measured DRAM traffic. Cooperative halo loading may also reduce
repeated input loads, but no additional DRAM-saving number is claimed.

### Tile exploration and native resources

Tiles of 32x8, 32x16, 32x32, and 32x64 all pass 240 guarded differential
cases and a 1x4,194,305 tall-image case, each with three-stage reuse.
All four variants preserve the 35 existing native Butteraugli bodies
and add one new body. Each new body uses 34 registers, zero stack/local
storage, and respectively 4,632 / 7,704 / 13,848 / 26,136 shared bytes.

One executable per variant compares the original seven-launch sequence
with the fused four-launch sequence, using the original shared horizontal
scratch and in-place blurred/XYB layout for the reference. Eight
geometry/stride cases use ten warmups, fifteen CUDA-event samples, and
three alternating-order pairs. Full output checksums match throughout.

| Tile height | Packed 1080p candidate median ms | Packed 4K candidate median ms | Padded 4K candidate median ms |
| --- | ---: | ---: | ---: |
| 8 | 0.547840 | 6.984704 | 6.738944 |
| 16 | 0.539648 | 6.411264 | 5.856256 |
| 32 | 0.549888 | 6.430720 | 5.447680 |
| 64 | 0.615424 | 6.239232 | 6.222848 |

These are different cohorts, not tightly controlled direct tile-to-tile
comparisons. The 4K observations vary substantially, including individual
candidate medians of 2.224128 ms for height 16 and 2.476032 ms for height
64. They are retained rather than trimmed. Height 64 is noticeably worse
at 1080p, while heights 8/16/32 are closer; retain 16 as a balanced choice
with smaller shared storage than 32/64, not a universal tile optimum.
Boundary observations include software thermal/power limiting; no
operating-state cause is assigned to an individual timing change.

After production integration, the final same-executable isolated cohort
again improves all 24 pairs, with exact output checksums:

| Geometry | Packed paired ratio | Padded paired ratio |
| --- | ---: | ---: |
| 31x63 | 0.431 | 0.405 |
| 512x512 | 0.699 | 0.702 |
| 1920x1080 | 0.678 | 0.687 |
| 3840x2160 | 0.641 | 0.635 |

Final packed/padded 4K marginal medians are 10.469376 / 10.474496 ms
for the reference and 6.632448 / 6.650880 ms for the candidate. This
isolated result includes the unchanged horizontal blurs and is not an
end-to-end encoding speedup.

### Exactness and regression coverage

The new permanent fixture exercises twenty geometries, packed and
channel-distinct padded input strides, six patterns, and three reuse
passes at intensity targets 80/255/1000. It checks bitwise XYB equality,
input and weight immutability, horizontal results where layouts agree,
prefix/suffix/row guards, the reference's shared/in-place storage policy,
and ignored/null fused blurred pointers. Patterns include signed zero,
random negative/positive values, tiny/huge magnitudes, identity weights,
NaN/infinities, and asymmetric weights. Zero extents and all four invalid
stride positions are checked. The separate tall case crosses 65,535
tile rows. The scoped sanitizer mode covers 60 cases, including repeated
reflection and partial tiles, with the same three reuse policies.

The integrated prepared-map differential links a common probe object
against the retained S47 CUDA archive and the candidate archive. All
261 complete padded maps and double scores across 29 geometries, three
option policies, and distortion/identity/restored-distortion reuse match
byte for byte: both files contain 399,602,628 bytes. Prepared allocation,
cached-reference, and comparison-scratch statistics are unchanged in every
case. The integrated AQ and prepared Butteraugli fixtures also pass.

### Production GPU profiles

Three alternating-order pairs per workload use three warmups and one
captured encode. The joint subset is the old three vertical blurs plus
Opsin versus the new joint kernel; the full bundle additionally includes
the unchanged horizontal blurs.

| Workload | Parent joint ms, in pair order | Candidate joint ms, in pair order | Joint paired change | Full-bundle paired change | Total GPU paired change |
| --- | --- | --- | ---: | ---: | ---: |
| Odd 4K | 14.255022 / 13.382868 / 13.185780 | 6.925361 / 6.781836 / 6.933200 | -49.3% | -35.3% | -9.2% |
| Odd 1080p | 2.207897 / 2.265049 / 2.200187 | 1.161664 / 1.158525 / 1.173919 | -47.4% | -33.3% | -3.0% |

The target improves in all six profile pairs. The larger 4K total-GPU
gain must not all be assigned to this fusion: non-target work also
improves by median paired 6.4%, including 213.458950 -> 183.641390 ms
in one pair, while another pair regresses 2.9%. The 1080p non-target
paired median is +0.1%. All observations remain in the evidence.
Boundary state spans 67-68 C and 285-1290 MHz, with software thermal/power
flags mostly active; no system settings are changed.

Launch counts change from 418 to 400 at odd 4K and from 406 to 388
at odd 1080p. The full target bundle changes from 42 to 24 launches,
and the joint subset from 24 to six. Every non-target kernel's name,
grid/block dimensions, and native resource fields remain identical,
as do all transfers and all five allocation requests. The Butteraugli
arenas remain 978,352,436 / 244,426,868 bytes at odd 4K / 1080p;
S47's memory saving is preserved.

### Whole-encode observations

Seven alternating-order warm pairs per input retain three warmups,
five samples, all 41 phase fields, and matching output sizes. Changes are
medians of within-pair candidate/parent ratios, not ratios of the displayed
marginal medians. Negative means faster.

| Workload | Parent / candidate total median ms | Total paired change | Quantization paired change | Codestream paired change | Total wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Odd 4K | 492.051 / 500.294 | +1.6% | -2.8% | +7.4% | 3/7 |
| Odd 1080p | 123.374 / 123.277 | -1.4% | -1.1% | -1.7% | 5/7 |
| Flower | 26.784 / 27.090 | +0.4% | -1.2% | +2.4% | 3/7 |

Quantization improves in 6/7, 4/7, and 6/7 pairs respectively, but that
does not produce a consistent whole-encode gain. In particular, the 4K
codestream work is unchanged by this GPU patch and is slower in five
pairs. Its variation cannot simply be attributed to the fusion. Warm
total ranges are 469.618-508.811 / 474.333-506.710 ms at 4K,
122.635-144.159 / 120.461-137.823 ms at 1080p, and
26.232-34.608 / 26.161-32.261 ms for Flower. The 1080p observations
include adverse 123.488 -> 137.823 ms and favorable
144.159 -> 128.276 ms pairs; neither is excluded.

Seven fresh-process cold pairs use zero warmups and one sample, still
excluding backend construction rather than measuring complete CLI startup:

| Workload | Parent / candidate total median ms | Total paired change | Quantization paired change |
| --- | ---: | ---: | ---: |
| Odd 4K | 585.366 / 583.319 | +0.3% | -2.9% |
| Odd 1080p | 170.265 / 167.168 | -0.5% | +0.1% |
| Flower | 51.196 / 52.977 | +3.3% | +1.3% |

Cold total ranges are 568.585-668.273 / 558.898-684.795 ms,
159.604-233.628 / 159.372-191.419 ms, and
49.041-58.531 / 47.819-79.101 ms respectively. The cold Flower
regression remains visible. Warm/cold boundary state moves from
68 C / 1770 MHz to 72 C / 1282 MHz, P3 with both software flags
active. No stable universal encoder speedup is established, and no
clock, power, cooling, security, service, or priority setting is changed.

### Final qualification

Both complete suites pass: 69 CUDA and 49 CPU tests. The additional
permanent fixtures report all 240 guarded Opsin cases and the tall-image
case complete; the 27-case prepared Butteraugli fixture and its exact
27-plane accounting still pass. All 35 existing native bodies remain
unchanged, and the final fused body matches the qualified 16-row prototype.

All 46 decoded-image pairs pass across seven inputs, distances
0.5/1.2/3.0 at effort 7, and sample/Flower at effort 9, under both
encoding-only and requested-final-score policies. Encoded SHA-256/size,
decoded Butteraugli score, strategy reports, and requested score reports
match S47. Collecting the final score does not change the output bytes.
The pinned decoder/metric, linear-RGB interpretation, and tolerances
are unchanged.

All three current-version batch cohorts preserve serial/batch bytes.
Median paired batch-versus-serial speedups for sizes 1/2/4 are
0.993/1.139/1.217 at fully-resident 1080p and 0.957/1.556/2.043 at
maximum-throughput 1080p; fully-resident 4K is 0.949/1.094 for sizes
1/2. These measure batch scheduling within S48, not S48 against S47.
Individual size-1 results regress to 0.837x at 4K.

Seven scoped CUDA sanitizer checks pass: Opsin
memcheck/initcheck/synccheck/racecheck and full AQ
memcheck/initcheck/synccheck. All summaries report zero errors or hazards;
AQ memcheck additionally tracks stream-ordered races and full leaks,
reporting zero leaked bytes. The 60-case Opsin racecheck takes about
four seconds; AQ memory/init/sync checks take about 33/23/16 seconds.
Full-AQ racecheck is not repeated. The earlier approximately 52-minute
S30 abort remains an abort, not a pass or a confirmed firewall failure.
The serial final qualification completes normally in about seven minutes,
with no observed administrator/firewall prompt, permission error, or stall.

### Retention, evidence, and remaining work

Retain the 16-row fusion for the proven intermediate-access and launch
elimination, consistent isolated and production target-kernel gains, and
full correctness qualification. The adverse whole-encode and cold results
remain part of the record; the larger 4K total-GPU gain includes unrelated
work variation and is not a stable encoder speedup claim.

Fourteen qualified production executables are frozen as
`build-cuda-ninja/profiles/s48_retained_*.exe` and checked by
`s48_final_binary_hashes.json`. The prepared-map probe and old/new CUDA
archives have `s48_prepared_hashes.json`. Four tile source/probe/test
snapshots, the final isolated probe, and its source/object/header/fixture
snapshots have `s48_experiment_hashes.json`. The final validator verifies
these identities, all existing native bodies and final/prototype identity,
69/49 test logs, seven sanitizer summaries and completion markers,
all 46 image/policy checks, 21 warm + 21 cold pairs, six GPU profile pairs
with unchanged non-target structure and allocation/copy totals, complete
261-pair map bytes, 96 exploratory + 24 final isolated pairs, and all
three batch cohorts.

Reproduction scripts under the ignored profile directory include
`s48_variant.ps1`, `s48_native.py`, `s48_opsin_probe{.cpp,_run.ps1}`,
`s48_opsin_summary.py`, `s48_prepared_probe{.cpp,_run.ps1}`,
`s48_prepared_compare.py`, `s48_integrated_run.ps1`,
`s48_profile_pairs.ps1`, `s48_profile_summary.py`,
`s48_phase_build.ps1`, `s48_phase_compare.py`, `s48_report.py`,
`s48_final_run.ps1`, `s48_final_quality.py`, `s48_sanitizers.ps1`,
`s48_freeze.ps1`, and `s48_final_validate.py`. Raw maps, native reports,
traces/SQLite exports, all paired observations, decoded-image reports,
and terminal sanitizer logs are retained. No system settings are changed.

The resident path is not demonstrated maxed out. A remaining architectural
lead is coefficient materialization: `AssembleFrame` still downloads
batch-ordered quantized coefficients into a host staging allocation,
then frame assembly validates and copies them into group/channel storage.
Re-measuring those stages and investigating direct final-layout readback
could address work beyond the remaining individual GPU kernels. The S34
group-ordered host-append experiment was rejected and must not be treated
as an already successful layout change. Likewise, S31 established that
the compiler already reuses repeated Malta directional sums; removing
those source duplicates is not a new arithmetic-saving opportunity.

## Coefficient handoff measurement and packing prototype (S49)

Baseline: `f22c2f8`, after S48. This is a measurement/design checkpoint, not a
production optimization. No tracked codec, CUDA, test, or build source is changed.
Diagnostic clones of the current resident evaluator and frame assembler are
linked before the unchanged production archives. This is distinct from S34's
rejected group-ordered host-append experiment.

### Baseline attribution

Seven independent processes per workload, each with three warmups and five
measured encodes, time the current preparation and finalization stages. Entries
below are medians of per-process sample medians in milliseconds. The odd-sized
benchmark images are 3839x2159 and 1919x1079; Flower is 510x532.

| Stage | 4K | 1080p | Flower |
| --- | ---: | ---: | ---: |
| Host metadata preparation | 9.570 | 2.468 | 0.330 |
| Host staging allocation | 0.845 | 0.243 | 0.043 |
| Final five-copy readback | 16.870 | 4.513 | 0.822 |
| Whole frame assembly | 27.166 | 9.298 | 1.184 |
| AC allocation/zeroing, within assembly | 10.856 | 4.969 | 0.691 |
| DC/raw-quant/EPF setup, within assembly | 1.150 | 0.283 | 0.041 |
| Transform validation/copy and group completion | 14.222 | 3.733 | 0.366 |
| Staging + readback + intervening setup + assembly | 44.953 | 14.076 | 2.052 |
| Whole public encode | 454.906 | 118.653 | 25.719 |

Component medians are not additive. The AC copy already uses an OR-reduction
for the unwritten-value check, not a per-element early-return branch. The new
opportunity is the full intermediate host allocation and layout conversion,
not simply removing a branch that was eliminated in an earlier checkpoint.

### Isolated packing prototype

An environment switch in one diagnostic executable selects either the current
path or the following experiment. Neither the switch nor its thread-local
ownership hook is proposed as a production API.

1. During existing metadata construction, build one 24-byte source/destination
   record per transform and final group/channel offsets for frame assembly.
2. Extend the reconstruction-coefficient scratch allocation to fixed AC-group
   capacity. Its float contents are dead after reconstruction/inverse-transform
   consumers finish, and a subsequent evaluation rewrites them before reuse.
3. At finalization, allocate the ordinary zero-initialized final host vector,
   submit a device clear and packing kernel into the reused scratch, then wait.
4. Download that final-layout array and the four existing small readbacks.
   Validate metadata, coefficient sentinels, and zero edge tails, then move the
   vector into the private candidate frame. Publish only on successful assembly.

All existing coefficient/quantization kernels come from the unchanged S48 CUDA
archive. The prototype adds a data-movement kernel, not a new arithmetic path.
It avoids the original full host coefficient staging array and host AC copy,
but **does not remove host clearing**: that cost moves before readback.

Fixed group capacity also transfers unused zero-filled edge tails. The following
are layout/request arithmetic, not captured transfer or retained-VRAM claims:

| Quantity, bytes | 4K | 1080p | Flower |
| --- | ---: | ---: | ---: |
| Original coefficient readback/staging | 99,532,800 | 24,883,200 | 3,293,184 |
| Fixed-capacity packed coefficient readback | 106,168,320 | 31,457,280 | 4,718,592 |
| Extra scratch capacity for edge padding | 6,635,520 | 6,574,080 | 1,425,408 |
| Additional record-arena capacity, 24 bytes per block | 3,110,400 | 777,600 | 102,912 |

The actual record upload contains only anchors, not all blocks. Padding alone
increases coefficient transfer size by about 6.7%, 26.4%, and 43.3%, respectively.
The separate packing submission is also a correctness-contract concern below.

### Same-executable paired observations

Seven alternating pairs per workload use three warmups and five samples per
process. Both branches are linked into the same executable. All 21 pairs and
their 336 warmup/measured finalizations are retained; no outliers are dropped.
The pairs are a different cohort from the baseline-attribution table above.

| Timed region | 4K baseline / packed | 1080p baseline / packed | Flower baseline / packed |
| --- | ---: | ---: | ---: |
| Metadata preparation | 10.067 / 11.524 | 3.629 / 3.391 | 0.428 / 0.471 |
| Host clear moved before readback | 0.000 / 11.440 | 0.000 / 5.762 | 0.000 / 0.893 |
| Added packing submission/wait | 0.000 / 1.657 | 0.000 / 0.455 | 0.000 / 0.112 |
| Readback | 17.500 / 17.069 | 6.141 / 5.229 | 0.887 / 0.957 |
| Frame assembly after readback | 28.271 / 8.289 | 11.458 / 2.596 | 1.473 / 0.506 |
| Staging + clear + packing + readback + setup + assembly | 47.309 / 39.902 | 17.844 / 14.483 | 2.381 / 2.533 |
| Above total plus metadata preparation | 58.537 / 51.693 | 21.621 / 17.792 | 2.795 / 3.098 |
| Whole public encode | 469.126 / 462.698 | 148.725 / 130.573 | 28.918 / 30.118 |

The median **paired** change in the handoff total including metadata is -9.4% /
-2.6% / +7.5%; without metadata it is -15.0% / -6.2% / +2.5%. Totals are summed
within each sample before process medians and paired ratios are calculated.
Reporting only post-readback assembly would overstate the benefit by hiding the relocated
host clear, packing submission, and additional preparation. Whole-encode paired
changes are -1.7% / +0.3% / +3.0%, with 5/7, 3/7, and 1/7 wins. Quantization-
pipeline paired changes are -0.1% / -0.4% / +5.2%; unchanged codestream work
changes by -5.2% / +0.6% / +4.2%. Ratios of aggregate medians are not paired ratios.

The 1080p cohort varies particularly sharply: baseline process medians span
119.736-291.342 ms and packed medians 121.589-222.885 ms. Retained pair 3 is
291.342 -> 130.573 ms, while pair 5 is 149.146 -> 207.364 ms. In pair 3,
unchanged codestream work alone changes from 179.070 to 60.914 ms. These changes
cannot be attributed to packing. 4K ranges are 436.705-484.087 /
442.769-484.368 ms; Flower ranges are 27.981-43.828 / 28.559-40.608 ms.
No operating-state trace was captured for these timing cohorts, so their
variation is not assigned a thermal, power, firewall, or other specific cause.
This experiment does not establish a stable complete-encoder speedup.

### Qualification limits and disposition

The unmodified CUDA AQ fixture fails with the packed branch enabled at its
combined bounded/full check. A separate diagnostic preserves the failing
assertion and reports: three allocations, **four submissions instead of three**,
equal quant fields, equal block maps, equal score histories containing four
scores, and a valid final frame. Thus the observed assertion failure is the
additional submission, not an output mismatch. Full-suite or sanitizer success
is **not claimed** for this prototype. The production S48 qualification remains
separate; no test assertion is weakened in tracked code.

Separately, all 46 production-versus-prototype image pairs are byte-identical
and have identical independently decoded Butteraugli scores. The matrix retains
the seven inputs at distances 0.5/1.2/3.0 and effort 7, plus sample and Flower at
distance 1.2/effort 9, each with encoding-only and final-score collection.
The pinned decoder/metric revision is `e8ff09762481785938d8e4e01333ed3917571161`,
with `RGB_D65_SRG_Rel_Lin` decode/metric settings. Encoding-only/scored byte
identity is checked as well. These image checks do not make the failed
conformance fixture a pass or qualify unchecked failure/reuse paths.

Source inspection also finds that the existing coefficient-poison test hook
expects the old staging allocation. The prototype skips that allocation, so
the hook must be redesigned to poison the actual future readback destination
before extending conformance coverage. Do not run it against the null pointer
or treat image-byte agreement as a substitute for this overwrite test.

The fixed-capacity packing prototype is not adopted. The useful next design
directions are ownership-backed overwrite-only final storage, clearing only
unused tails, packing into an existing final GPU submission, and avoiding
transfers of padded tails. Packed active-group rows with pitched readback, or
direct integer output offsets independent of the forward-float batch layout,
deserve separate measurements; neither is implemented here. Preserve output
atomicity, failure invalidation, prepared reuse, complete overwrite checks,
exact/scored behavior, and the existing immutable frame contract. Do not infer
that a new container or shared-ownership refactor is already justified.
Final color correlation consumes the forward **float** coefficients, not this
integer output, so an integer-layout experiment need not change its float
source layout or metadata.

Reproduction artifacts under the ignored `build-cuda-ninja/profiles` directory:
`s49_{readback,frame}_probe.cpp`, `s49_build_probes.ps1`,
`s49_readback_measure.py`, `s49_readback_baseline.json`,
`s49_packed_{resident,frame}.cpp`, `s49_pack.{h,cu}`,
`s49_build_packed.ps1`, `s49_pack_measure.py`, `s49_pack_comparison.json`,
`s49_link_checks.ps1`, `s49_check_packed.ps1`, `s49_packed_quality.py`,
and `s49_contract_diagnostic.{cpp,ps1}`. `s49_validate.py` checks the retained
experiment identities, the unchanged S48 production binaries, timing records,
46 image/policy pairs, and the explicitly failed conformance diagnostic.
`s49_artifact_hashes.json` records diagnostic source/object/executable identity.
Per-process stdout/stderr, diagnostic executables/objects, image outputs/reports,
and the failed conformance result
are retained. No system settings were changed, and no admin/firewall failure
was reported by these runs. Optimization remains ongoing, not maxed out.

## Owned active-coefficient readback (S50)

S50 implements a revised handoff after the rejected S49 prototype. Production
comparison is against `f22c2f8`; `8d8dfad` only records S49's investigation.
Qualification ran on 2026-09-05 America/New_York (2026-09-06 UTC), Release
MSVC 14.37 / CUDA 11.8, architecture 86, RTX 3060 Laptop / driver 577.00.
Global math, power, clock, cooling, priority, security, and service settings
were not changed. Optimization remains ongoing, not maxed out.

### Storage, layout, and ordering

The former resident handoff reads batch-ordered integers into a full active
host staging allocation, allocates and clears final fixed-capacity AC-group
storage, then validates and copies transforms into that layout. S50 separates
the integer readback layout from the existing forward-float batch layout:

1. Host metadata records an eight-byte packed destination offset per anchor.
   Active group/channel rows are contiguous on the device; edge rows use their
   actual coefficient counts rather than the fixed 65,536-value host capacity.
2. One integer-copy kernel per nonempty strategy batch runs after the final
   inverse consumer, inside the existing final reconstruction/policy submission.
   It reuses the existing reconstruction-coefficient scratch allocation. Scored
   policy iterations do not pack; only requested final frames do. A subsequent
   evaluation rewrites reconstruction floats before any inverse consumer.
3. Consecutive rows with equal active widths form readback runs. Contiguous or
   pitched copies write active values directly into final group/channel rows;
   only unused host tails are cleared. There is no device padding clear or
   padded-tail transfer, and no extra submission/wait pair.
4. Frame assembly validates geometry, transforms, offsets, group coverage,
   DC/raw quantization, EPF, optional active-value sentinels, and zero tails,
   then transfers ownership. On failure both the input owner and published
   output remain unchanged. The poison hook allocates and poisons the actual
   next readback destination, including after ownership has been consumed.

`OverwriteArray<int32_t>` provides fixed-size overwrite-only allocation,
deep copying, and noexcept ownership transfer; it does not introduce shared
ownership. Public frame views, capacity, and copy semantics remain unchanged.
The private C++ object representation changes, so consumers and diagnostic
objects must be rebuilt; old S41/S49 objects must not be linked to these
libraries. CPU reconstruction and borrowed assembly retain initialized storage
and the original copy path. Installed-consumer tests pass in both builds.

`CudaDeviceToHostCopy` gains row count and optional source/destination pitches;
existing four-field descriptors still mean a single contiguous copy. All
descriptors are validated before enqueueing, including source ownership, byte
ranges, pitch widths, and row-span overflow. A common synchronization completes
the batch. This is an internal API, not a new public readback contract.

### Deterministic costs and GPU evidence

Three alternating CUDA-only Nsight pairs per odd-padded workload retain all
original kernel names, ordering, grids, blocks, and native resource usage.
All 161 preexisting native bodies are byte-identical; the added packing body
uses 28 registers, zero stack/local/shared bytes, and no barriers. Its median
summed GPU time is 0.935 ms at 4K (seven launches) and 0.224 ms at 1080p
(six launches). These are added costs, not removed GPU work.

| Captured quantity | 4K parent -> S50 | 1080p parent -> S50 |
| --- | ---: | ---: |
| Kernel launches | 400 -> 407 | 388 -> 394 |
| Total D2H bytes | 103,699,012 -> same | 25,924,192 -> same |
| D2H operations | 19 -> 20 | 19 -> 28 |
| Total H2D bytes | 117,079,320 -> 118,254,624 | 29,336,392 -> 29,631,464 |
| H2D operations | 31 -> 33 | 31 -> 33 |
| Persistent arena request | 402,396,664 -> 403,433,464 | 100,617,726 -> 100,877,054 |

There are still five arena requests; only the persistent metadata arena grows,
by 1,036,800 / 259,328 bytes including alignment. The coefficient scratch arena
does not grow. D2D remains one 518,400 / 129,600-byte transfer, and device
memsets remain four operations totaling 28 bytes. Additional H2D copies upload
initial and reconfigured packing offsets. AC readback uses two / ten runs.
The active coefficient payload remains 99,532,800 / 24,883,200 bytes. Removing
its separate host staging owner saves that allocation during final assembly;
this is an ownership/accounting result, not a measured RSS claim.

The original-kernel subset nevertheless varies by median paired -8.4% / -2.2%
in these short profiled cohorts despite identical code and launch structure.
Do not attribute those changes to packing. Boundary readings show 62-67 C,
1282 MHz, P3 where sampled, with software thermal/power-limit flags active.
These are boundary observations, not per-kernel clock normalization. All
outliers are retained; no hardware counter privileges were requested.

### Same-executable handoff control

`s50_handoff_probe.exe` selects the old staging/copy path with diagnostic-only
`S50_COPY_AC`; the default branch is the packed handoff. Both branches retain
S50's metadata allocation/uploads and new frame representation, isolating the
handoff from those changes. Neither switch nor timing instrumentation is in
production. Both branches match qualified production image bytes in all 46
cases. This is a mechanism control, not an independent old-release baseline.

Seven alternating process pairs per workload use three warmups and five samples.
The host sum includes preparation's staging allocation, tail preparation,
readback, assembly setup, and full assembly. The inclusive sum also includes
metadata construction. Sums are formed per sample before taking medians;
medians below are medians of process medians, while changes are medians of
paired ratios. GPU packing occurs earlier and is excluded from these host sums.

| Controlled host quantity (ms) | 4K baseline -> packed | 1080p baseline -> packed | Flower baseline -> packed |
| --- | ---: | ---: | ---: |
| Host tail/descriptor preparation | 0.004 -> 1.995 | 0.005 -> 1.695 | 0.001 -> 0.315 |
| Readback | 16.937 -> 17.493 | 4.737 -> 5.933 | 0.821 -> 0.844 |
| Frame assembly | 26.719 -> 7.845 | 9.369 -> 2.483 | 1.172 -> 0.453 |
| Assembly full clear | 10.857 -> approximately 0 | 5.236 -> 0 | 0.684 -> 0 |
| Transform copy/validation phase | 13.989 -> 5.782 | 3.561 -> 1.837 | 0.359 -> 0.347 |
| Combined host sum | 44.452 -> 28.259 | 14.532 -> 10.488 | 2.037 -> 1.692 |
| Inclusive host sum | 54.604 -> 38.959 | 17.754 -> 13.363 | 2.351 -> 2.011 |

The copy/validation phase becomes metadata and coefficient validation, not
zero work. Readback itself is slower by paired median 3.8% / 18.7% / 5.7%;
additional copy descriptors and the different page-touch pattern are plausible
contributors, not separately established causes. Inclusive host changes are
-29.9% / -26.1% / -10.6%, improving 7/7, 7/7, and 6/7 pairs respectively;
paired ranges are [-32.6%, -23.7%], [-32.4%, -19.5%], and [-31.8%, +3.9%].
The same-executable quantization pipeline improves -4.2% / -7.8% / -2.6%,
while whole encode changes -1.0% / -3.9% / -2.4% with only 5/7 wins each.
Unchanged codestream work varies substantially; the control does not establish
a stable small-image or universal end-to-end gain.

### Public workflow timing and qualification

The independent production-baseline comparison also uses seven alternating
process pairs, three warmups and five samples, with 41 retained phase fields.
Benchmark inputs are odd 3839x2159 / 1919x1079 and linear Flower 510x532,
distance 1.2, effort 7, fully resident, final-score collection disabled.
The candidate phase object is rebuilt against current headers; the parent is
the retained S48 executable.

| Warm phase | 4K parent -> S50 ms; paired change | 1080p parent -> S50 ms; paired change | Flower parent -> S50 ms; paired change |
| --- | ---: | ---: | ---: |
| Quantization pipeline | 297.409 -> 279.327; -5.6% | 57.947 -> 54.469; -6.8% | 11.843 -> 11.542; -3.5% |
| Whole encode | 437.635 -> 420.257; -2.3% | 115.248 -> 111.599; -3.8% | 24.865 -> 24.148; -4.4% |
| Unchanged codestream encoding | 115.762 -> 116.174; -0.7% | 50.416 -> 49.789; -1.4% | 11.528 -> 11.101; -5.0% |

All seven quantization pairs improve for each workload. Whole encode improves
6/7, 7/7, and 6/7; paired ranges are [-9.3%, +5.8%], [-8.0%, -1.4%], and
[-30.8%, +2.7%]. Flower's 32.451/34.784-ms parent outliers and all other slow
observations remain in the report. The earlier three-pair exploratory cohort
is retained separately, not pooled with this final cohort.

Seven cold process pairs use zero warmups and one sample. Whole-encode medians
are 546.312 -> 524.998 ms, 159.307 -> 153.680 ms, and 47.373 -> 50.601 ms;
paired changes are -1.9% / -4.3% / **+3.2%**, with 5/7, 6/7, and 3/7 wins.
Cold quantization changes are -2.9% / -5.6% / +3.9%. Cold startup and the small
photo regression remain limits of the adopted optimization, not discarded data.

Qualification includes:

- All **71 CUDA-enabled tests**, repeated after final profiling, and all
  **50 CPU-only GCC tests**, including installed consumers. The final CUDA run
  completes in 54.35 s. Packing's unavailable-device exit is registered as skip.
- Four fully host-instrumented Clang ASan targets: overwrite-array ownership,
  VarDCT frame, reconstruction, and codestream encoder.
- New packing fixture: 64 geometry/strategy cases x three changed-data reuses,
  through 480x270 blocks, all seven strategy shapes plus mixed patterns, integer
  extremes, prefix/suffix guards, unchanged inputs/metadata, pitched-copy
  active rows, and untouched tails. Ten malformed descriptor cases verify
  pre-enqueue atomic rejection, alongside empty/zero-row launch cases.
- Owned frame tests cover single, full, and partial groups, mixed transforms,
  malformed views/layouts, active sentinels, nonzero tails, failure atomicity,
  optional sentinel policy, deep copies, and repeated exact-pointer transfers.
  Existing AQ poison, prepared reuse, scored/encoding-only equality, failure,
  and allocation/submission assertions pass without relaxed expectations.
- Four scoped packing sanitizer runs (memcheck/initcheck/synccheck/racecheck)
  each complete 48 cases x three reuses with zero errors/hazards. Three final
  AQ runs (memcheck with stream-ordered race/leak checks, initcheck, synccheck)
  report explicit completion and zero errors; memcheck reports zero leaks.
  Earlier unflushed AQ logs are retained but final claims use the reruns with
  explicit flushed markers. The old aborted full-AQ race run is not repeated
  or counted as passing.
- All 46 parent/candidate image pairs are byte-identical and independently
  decode to identical Butteraugli scores with pinned libjxl
  `e8ff09762481785938d8e4e01333ed3917571161`, linear RGB
  `RGB_D65_SRG_Rel_Lin`. Seven inputs, distances 0.5/1.2/3 at effort 7 plus
  sample/Flower effort 9, each encoding-only and scored, retain the established
  matrix. Quality 1080p/4K inputs are distinct from odd timing inputs. All 46
  same-executable control triples also match those qualified bytes.
- Batch byte checks pass with one warmup / three samples: even 1080p fully
  resident batch 1/2/4 gives paired speedups 1.019x / 1.079x / 1.315x against
  same-version serial execution; maximum-throughput gives 0.971x / 1.488x /
  1.937x. Even 4K fully resident batch 1/2 gives 1.027x / 1.058x. These are
  concurrency measurements, not parent-versus-S50 speedups.

Artifacts under ignored `build-cuda-ninja/profiles` include
`s50_{phase,handoff}_build.ps1`, fresh probe source/objects/executables,
`s50_{run,more}_checks.ps1`, `s50_performance.ps1`,
`s50_final_phases.json`, `s50_control_comparison.json`, `s50_cold_*.json`,
`s50_{quality,control_quality}.json`, native dumps and summary,
twelve Nsight reports/SQLite exports, sanitizer/ASan/CTest logs, batch outputs,
GPU-state boundaries, and all image outputs/reports. `s50_report.py` derives
paired summaries; `s50_validate.py` checks identities and evidence.
`s50_final_binary_hashes.json` and `s50_artifact_hashes.json` freeze current
binaries, production source snapshots, diagnostic objects, and evidence for
future comparisons. The historical S49 validator expects unchanged S48 live
binaries and is therefore not a validator of S50's current build.

No permission error, admin/firewall failure, or stalled process was reported
by these checks. The performance workflow completed in about five minutes;
the user's suspected cause of the older long runtime remains unconfirmed.
Future leads include writing final integer offsets directly from the producer
to remove packing traffic/launches, reducing validated host assembly costs,
and measuring deferred readback allocation for bounded/no-frame use. Current
preparation still allocates final-capacity host storage even in that mode.
Neither a new bypass nor an unmeasured size threshold is introduced here.

## In-place ANS clustering and hoisted log-table access (S51)

S51 compares against `ad18132` on the same RTX 3060 Laptop, MSVC 14.37 / CUDA
11.8 Release build. Qualification date is 2026-09-05 America/New_York
(2026-09-06 UTC). This is host work on the fully-resident encode path, not GPU
entropy coding. No CUDA source, GPU math, transfer layout, public API, or
system configuration changes are involved.

### Locate the work before changing it

`s51_rank.py` ranks all kernels in the three retained S50 traces per workload.
Malta responses remain the largest GPU component, followed by wide blurs;
the already rejected larger Malta tiles and shuffle variants are not repeated.
The S50 host profile also reports roughly 30 / 19 ms of ANS histogram work
at 4K / 1080p. These `_work` fields accumulate worker time and must not be
added to wall phases: DC and AC entropy tasks can overlap.

An isolated source diagnostic separates histogram population construction,
clustering, and normalization search. On the large AC model, both image sizes
have 6,930 contexts, of which 2,849 / 2,539 are populated. Farthest-first seed
selection and subsequent ordered assignment perform 180,784 / 160,944 distance
evaluations; Flower has 1,485 contexts, 528 populated, and 18,513 evaluations.
Warmed detailed samples put the redundant source copy near 2.25 / 1.98 /
0.43 ms and seed plus assignment work near 14.4 / 9.0 / 1.4 ms. The broad
histogram-work field is not predominantly normalization search.

Native MSVC inspection establishes a second cost: the old distance function
calls `ExactCountLog2` at its total-count and per-populated-bin sites. That
helper performs a thread-safe lazy-static guard check on each call, including
TLS access. The table lookup itself is cheap, but the surrounding function and
guard work repeats across the many distance evaluations. This is executed-code
evidence, unlike the previously rejected source-duplicate Malta hypothesis.

### Retained implementation and isolated controls

`FastClusterDirectAnsHistograms` now caches Shannon costs directly in the
partition builder's private histogram vector. Counts, context order, and totals
are unchanged; no caller-owned prepared population is borrowed or mutated.
This removes the extra 256-bin source copy, including empty contexts. At 6,930
contexts the bins alone occupied 14,192,640 bytes, plus histogram metadata.
This is eliminated allocation/copy accounting, not a measured RSS claim.

The existing exact log table is factored into a reference-returning accessor.
Clustering obtains the initialized table once and passes it to distance
evaluation; individual Shannon calculations also acquire it outside their
symbol loops. Entries, the 65,536 cutoff, the `std::log2` fallback above it,
ordered double accumulation, overflow checks, seed ties, the 32-cluster cap,
assignment order, and canonicalization remain unchanged. There is still one
thread-safe lazy table, with no approximation or global fast-math change.

The candidate's native distance body has no calls to the guarded log helper
and no lazy-initialization guard references. Its two direct `log2` call sites
are the unchanged out-of-table fallback, not new approximate math. All **162
GPU bodies** compare identically, and `gjxl_cuda.lib` hashes identically to
the retained S50 library. GPU performance gains are not claimed.

Nine captured real population sets include the small/DC and AC partitions for
each workload, with their exact original clusters and symbol maps. A single
replay executable tests baseline, no-copy only, hoisted lookup only, and both.
Three warmups precede nine alternating-order rounds, with two repetitions for
large partitions and twenty for small ones. Timed calls include the same
output verification on every branch. The large AC partitions give:

| Replay variant, median ms | 4K | 1080p | Flower |
| --- | ---: | ---: | ---: |
| Frozen baseline | 12.553 | 12.388 | 1.785 |
| No source copy | 10.548 | 10.960 | 1.280 |
| Hoisted lookup | 10.085 | 9.803 | 1.358 |
| Both | 7.114 | 6.831 | 0.792 |
| Both, median paired change | -44.8% | -43.4% | -55.8% |

All nine sets match captured clusters, integer populations, symbol maps, and
Shannon-cost bits on every replay. The production implementation also hoists
table access from individual Shannon loops and passes the table once per
clustering call, so it is qualified separately rather than assumed identical
to the experimental wrapper.

### Whole-workflow measurements and limits

Release and same-executable comparisons each retain seven alternating process
pairs per workload, three warmups, five samples, and all 41 phase fields.
Inputs are odd 3839x2159 / 1919x1079 and linear Flower 510x532, distance 1.2,
effort 7, fully resident, encoding-only. Entries below are medians of process
medians; percentages are medians of paired ratios, not ratios of those medians.

| Warm release phase | 4K parent -> S51 ms; paired change | 1080p parent -> S51 ms; paired change | Flower parent -> S51 ms; paired change |
| --- | ---: | ---: | ---: |
| ANS histogram worker time | 31.174 -> 24.131; -17.6% | 19.023 -> 14.254; -24.5% | 4.454 -> 2.952; -26.8% |
| Entropy optimization wall time | 26.731 -> 20.536; -20.8% | 21.838 -> 16.959; -21.7% | 5.514 -> 3.854; -20.8% |
| Codestream encoding wall time | 112.083 -> 107.942; -9.1% | 49.235 -> 44.365; -9.1% | 13.395 -> 10.575; -9.9% |
| Whole encode | 417.857 -> 412.923; -1.9% | 110.008 -> 105.956; -3.5% | 27.254 -> 23.713; -5.0% |

Entropy optimization improves in 6/7, 7/7, and 7/7 release pairs. Whole encode
improves in 4/7, 6/7, and 5/7, with ranges [-4.9%, +4.2%], [-16.1%, +1.9%],
and [-33.7%, +1.7%]. The unchanged quantization pipeline changes by paired
+0.6% / +0.7% / +0.8%; it is not credited to this host optimization. Neither
the 34.403-ms Flower parent outlier nor any slower candidate is discarded.

The diagnostic `s51_control.exe` selects the retained clustering algorithm with
`S51_REFERENCE_CLUSTER`; its default uses the production implementation. Both
branches share the current surrounding encoder and canonicalization helpers.
There is no switch or instrumentation in production. All 46 control image
triples match qualified production bytes. The control's entropy-optimization
wall time improves -27.0% / -18.4% / -22.5% (7/7, 7/7, 6/7 wins), while whole
encode changes **-3.7% / +4.1% / -3.9%** (6/7, 3/7, 5/7 wins).

In particular, the 1080p control improves every entropy pair but regresses
overall: unchanged quantization and AC tokenization change +4.0% and +9.4%.
Flower's optimized 40.164-ms outlier has slower work across several stages.
These observations are retained without assigning an unproven cause. They
prevent a stable universal end-to-end gain claim despite the isolated mechanism
and targeted phase improvements. Median paired ratios can differ in direction
from ratios of cohort medians in these noisy paired records.

Seven cold pairs (zero warmups, one sample) give whole-encode medians
524.736 -> 514.138 ms, 158.227 -> 155.706 ms, and 48.124 -> 46.078 ms, with
paired changes **+0.2% / -0.8% / -4.3%** and 3/7, 4/7, 7/7 wins. The first
two cold cohorts remain inconclusive. Boundary GPU readings are 60-67 C,
P3 / 1282 MHz, with thermal/power-limit flags active; no clock normalization
or power/security changes were attempted. Exploratory diagnostic timings are
not pooled with these contemporaneous paired comparisons.

### Qualification and reproduction

- All **71 CUDA-enabled tests** pass in 55.25 s and all **50 CPU-only GCC
  tests** pass in 16.91 s, including installed consumers. Twelve new sparse
  population cases cover 1/33/257/6,930 contexts, empty and identical
  distributions, sparse distinct contexts, counts above the log-table cutoff,
  scanned-versus-prepared model/cost equality, and unchanged caller populations.
- Three fully host-instrumented Clang ASan targets pass: entropy primitives,
  codestream encoder, and public codestream workflow. Existing multithreaded
  workflow/batch tests exercise the shared, still thread-safe lazy table.
  No new CUDA sanitizer claim is made for this host-only change.
- A production-source oracle compares retained clustering control flow against
  the actual candidate on nine captured sets, **321 synthetic clustering
  cases**, **1,024 distance pairs**, and **65,536 exact table values**. It checks
  cluster populations/maps, bitwise Shannon costs and distances, unchanged
  source counts, and matching errors, including large counts and overflow.
- All **46 primary image pairs** are byte-identical and independently decode
  to equal Butteraugli scores using pinned libjxl
  `e8ff09762481785938d8e4e01333ed3917571161`, linear RGB
  `RGB_D65_SRG_Rel_Lin`. The established seven-input, three-distance, effort
  7 plus sample/Flower effort-9, encoding-only/scored matrix is unchanged.
  Twelve additional pairs cover sample, Flower, and Keong Macan at distance
  1.2, each encoding-only/scored, under legacy `--high-density` (no explicit
  effort) and `--maximum-compression --effort 7`. All twelve match bytes and
  decoded scores, for **58 qualified pairs** total. The initial invalid
  high-density-plus-effort invocation was corrected; final labels and metadata
  explicitly identify the legacy mode rather than claiming effort 7.
- Batch byte checks pass with one warmup and three samples. Even 1080p fully
  resident batch 1/2/4 gives paired speedups 1.037x / 1.215x / 1.433x against
  same-version serial execution; maximum-throughput gives 1.002x / 1.212x /
  1.960x. Even 4K fully resident batch 1/2 gives 0.996x / 1.064x. These are
  concurrency results, not cross-version gains.

Ignored `build-cuda-ninja/profiles` artifacts include `s51_rank.{py,json}`,
`s51_ans_{probe,detail}.cpp` and per-workload diagnostics,
`s51_capture.{cpp,ps1,asm}` and nine `s51_population_*.bin` files,
`s51_replay.{cpp,ps1,txt}` and paired replay summary,
`s51_oracle.{cpp,ps1,txt}`, `s51_checks.ps1`, `s51_performance.ps1`,
native GPU/host dumps and summary, the same-executable control source/objects,
`s51_{final_phases,control_comparison}.json`, `s51_cold_*.json`,
primary/control/extra-mode image reports and outputs, CTest/ASan logs, batch
results, and GPU-state boundaries. `s51_report.py` derives the summaries;
`s51_validate.py` checks the recorded identities and evidence.
`s51_final_binary_hashes.json` and `s51_artifact_hashes.json` freeze binaries,
libraries, production snapshots, captured populations, and diagnostic artifacts.
Older S50 validators that compare against live source/binary hashes are not
validators of this new checkpoint; the retained S50 files remain intact.

The change is adopted for verified removal of redundant host work with
unchanged output, not as proof that every encode is faster. Remaining leads
include the prepared-population allocation/validation path, unresolved
whole-workflow variability, and the dominant Malta/wide-blur GPU work. No
admin/firewall or permission failure was reported, and the performance workflow
completed normally. Optimization remains active, not maxed out.

## Borrowed prepared ANS populations (S52)

S52 compares against S51 `355180e` on the same RTX 3060 Laptop, MSVC 14.37 /
CUDA 11.8 Release configuration, on 2026-09-05 America/New_York (September 6
UTC). This is host preparation on the fully-resident path. GPU code, floating
point operations, transfer layout, public headers, and ABI are unchanged.

### Locate and remove the remaining copy

A current-source diagnostic separates initial histogram allocation from
validation/import. For the 6,930-context AC partition, warmed allocation
medians are about 3.21 / 2.24 ms at 4K / 1080p; validation plus import takes
3.89 / 3.27 ms. Flower's 1,485 contexts give 0.49 / 0.57 ms. These are scoped
instrumented observations, not timings to pool with the final cohorts. DC
and AC entropy workers overlap, so their work must not be added as wall time.

An initial nine-set replay tests the unchanged builder, reserve/push instead
of zero-initialization, cached symbol extra-bit counts with a proven overflow
bound, and their combination. On large AC populations the no-initialization
variant improves preparation by paired 24.1% / 27.9% / 20.7% at 4K / 1080p /
Flower. The combination is worse than that variant and regresses Flower 4.1%
against the baseline. It is not adopted. In particular, **the original
per-bin division and every validation check remain in production**.

Instead, an unmapped prepared partition now constructs a small private vector
of `DirectAnsHistogramView` records. Each record references the caller's
read-only 256-bin array and owns its metadata and mutable Shannon cache.
Clustering reads that view directly. Selected seeds, assigned/merged clusters,
and canonicalized output retain owning arrays. An initial context map still
requires merged owned histograms; token-scanned and high-density paths also
keep their owning working sets. The same algorithm is instantiated for owned
and borrowed sources, without a separate approximate clustering path.

The span's lifetime covers this synchronous partition build; no view escapes
into prepared output or asynchronous work. View-vector reallocation does not
move the referenced population arrays. Every count bin, including tails of
allegedly empty contexts, is validated before clustering. Counts, totals,
extra-bit overflow checks, symbol metadata checks, ordered double sums, seed
ties, assignment order, and output-on-failure behavior remain unchanged.
Only the partition's private Shannon caches are mutable.

Native `sizeof` checks report 2,080 bytes per owning source histogram versus
40 per view. Thus the source element storage changes as follows, excluding
allocator metadata, vector headers, the unchanged caller populations, and
the small owning cluster vectors:

| Context count | S51 source storage | S52 source storage | Eliminated bytes |
| ---: | ---: | ---: | ---: |
| 6,930 | 14,414,400 | 277,200 | 14,137,200 |
| 1,485 | 3,088,800 | 59,400 | 3,029,400 |

This removes both initialization and copying of the large source arrays,
not just their initialization. It is storage/traffic accounting, not an RSS
measurement. Native host output confirms the view instantiation and retains
the validator's division instruction. All **162 GPU bodies** and the entire
`gjxl_cuda.lib` are identical to the retained S51 versions.

### Production-source isolated qualification

The oracle embeds the actual candidate source and S51's frozen owning
histogram, clustering, and partition implementation. Its identity check strips
only explicit diagnostic/reference additions and compares the candidate with
production and the reference with the retained S51 source snapshot. It passes
**17,961 partition checks**: nine captured real partitions plus eleven
synthetic patterns in mapped/unmapped forms across all **816 HybridUint
configurations**. Checks cover exact partition/prepared outputs, equal error
codes and messages, unchanged caller populations, and atomic outputs on
failure. Nine additional owned-versus-borrowed clustering pairs compare
integer populations, context maps, and bitwise Shannon costs.

That executable also times complete partition preparation and clustering on
all nine real captures. There are three warmups and nine alternating-order
rounds, with three repetitions for large partitions and twenty for small
ones. Output equality checks and destruction are outside each timed call.
The large AC partitions give:

| Complete partition replay | 4K | 1080p | Flower |
| --- | ---: | ---: | ---: |
| S51 median ms | 15.277 | 16.793 | 2.161 |
| S52 median ms | 10.806 | 11.926 | 1.467 |
| Median paired change | -25.9% | -24.0% | -32.1% |
| Faster rounds | 9/9 | 8/9 | 9/9 |

The small/DC captures are replayed as prepared populations only for isolated
coverage; the actual encoder still builds those partitions from token streams.
Their isolated improvements are therefore not claimed as workflow savings.
The first replay invocation incorrectly used the default HybridUint config for
the eight-context permutation/order partition and was correctly rejected.
Encoder source establishes `{0,0,0}` for that partition; the corrected replay
passes all nine sets. This was diagnostic setup, not a production failure.

### Whole-workflow qualification

Release and same-executable warm comparisons each use seven alternating
process pairs per workload, three warmups, five samples, and all 41 phase
fields. Inputs are odd 3839x2159 / 1919x1079 and linear Flower 510x532,
distance 1.2, effort 7, fully resident, encoding-only. Times below are medians
of process medians; changes are medians of paired ratios, not ratios of those
medians. No slower samples are removed.

| Warm release phase | 4K S51 -> S52 ms; paired change | 1080p S51 -> S52 ms; paired change | Flower S51 -> S52 ms; paired change |
| --- | ---: | ---: | ---: |
| ANS histogram worker time | 25.493 -> 24.677; -4.0% | 14.456 -> 13.106; -13.3% | 3.073 -> 2.605; -9.6% |
| Entropy optimization wall time | 20.991 -> 20.568; -0.7% | 17.198 -> 15.261; -15.8% | 4.120 -> 3.365; -7.8% |
| Codestream encoding wall time | 110.103 -> 113.910; +4.9% | 45.800 -> 44.992; -3.6% | 10.979 -> 10.306; +1.6% |
| Whole encode | 309.647 -> 318.859; +1.7% | 103.114 -> 102.581; +0.4% | 24.035 -> 23.110; +0.7% |

Entropy wall time improves in 4/7, 4/7, and 5/7 release pairs; whole encode
improves in only 3/7 each. Whole-encode ranges are [-2.9%, +6.8%],
[-18.3%, +18.2%], and [-32.5%, +28.9%]. Notably, the unchanged coefficient
order stage is slower in **all seven** 4K and 1080p release pairs, by paired
25.3% / 22.1%. This stage precedes the changed ANS preparation. That
cross-executable observation is unresolved, not credited to the optimization
and not dismissed as established random noise. Flower's 34.255-ms parent and
31.277-ms candidate outliers both remain in the report.

The single diagnostic `s52_control.exe` selects S51's frozen preparation and
clustering with `S52_REFERENCE_PREPARATION`; absence selects the actual
candidate. It has no production switch. All 46 baseline/optimized/production
image triples match bytes. Warm control histogram work changes -26.1% /
-19.1% / -10.4%, entropy wall time -20.5% / -16.7% / -12.6%, and whole encode
**-4.5% / -5.3% / +0.4%**, with 5/7, 4/7, 3/7 whole-encode wins. The
36.582-ms optimized Flower outlier and all slower control pairs are retained.

Seven cold release pairs (zero warmups, one sample) give whole-encode medians
496.704 -> 417.168, 156.118 -> 173.837, and 47.410 -> 45.955 ms, with paired
changes **-15.0% / +11.3% / -3.5%** and 6/7, 1/7, 5/7 wins. Unchanged
quantization changes -11.3% / +3.4% / -1.4%, so the large 4K difference is not
attributed to this host change. A separately retained same-executable cold
follow-up gives **+0.7% / +9.9% / -3.6%** overall, with 3/7, 1/7, 4/7 wins.
In that 1080p control, entropy wall time also regresses 2.0%; the repeated
cold regression cannot be hidden by the favorable isolated or warm results.

To examine that contradiction, a further 1080p cold A/A/B cohort executes all
six orders twice: A0 and A1 are identical baseline commands/environments, and
B selects borrowing in the same executable. Across twelve triples, identical
A1/A0 whole-encode ratios range from -8.8% to +10.2% (median +0.9%). B versus
the mean of its two contemporaneous baselines gives -2.9% whole encode (9/12
wins), -17.3% entropy wall time, and -13.9% histogram work (12/12 each).
The entropy-wall range is [-31.6%, -5.3%]; unchanged quantization is -0.3%
and coefficient-order work -0.7%. This confirms substantial timing variation
and supports the targeted reduction in that cohort, but does **not** establish
the cause of the earlier regressions or replace those results.

The change is retained for eliminating redundant source storage/traffic and
its independently checked targeted benefit. Stable cross-workload full-encode
and cold speedups remain unproven. Boundary GPU readings are 60 -> 77 C,
P0, 1282 -> 1267 MHz, with software power-cap active at the final boundary;
software thermal-slowdown is inactive at both boundaries. The unchanged 4K
quantization pipeline is near 175 ms in this final warm cohort versus roughly
275 ms in earlier cohorts. These cohorts are not pooled, and no GPU speedup,
clock normalization, or causal thermal diagnosis is claimed.

### Correctness and retained evidence

- All **71 CUDA-enabled tests** pass in 53.48 s; all **50 CPU-only GCC tests**
  pass in 16.82 s, including installed consumers. Twelve sparse-population
  fixtures now also exercise initial context maps. New cases reject each of
  256 hidden malformed bins and cover symbol bounds, total/extra-bit overflow,
  invalid maximum metadata, and mapped/unmapped merge overflow while checking
  caller immutability and atomic code/cost outputs.
- Three fully host-instrumented Clang ASan targets pass: entropy primitives,
  codestream encoder, and public codestream workflow. Large sparse borrowed
  inputs are exercised by the entropy target. No new CUDA sanitizer claim is
  made for unchanged GPU code.
- All **58 image pairs** match bytes and independently decoded Butteraugli
  scores with pinned libjxl `e8ff09762481785938d8e4e01333ed3917571161` and
  linear RGB `RGB_D65_SRG_Rel_Lin`: the 46-case primary matrix plus six legacy
  high-density and six maximum-compression pairs. Encoding-only and scored
  outputs agree; high-density uses no explicit effort, while the additional
  maximum-compression cases use effort 7.
- Batch byte checks pass with one warmup and three samples. Even 1080p fully
  resident batch 1/2/4 gives same-version paired speedups 0.934x / 1.403x /
  1.655x; maximum-throughput gives 1.058x / 1.303x / 1.901x. Even 4K fully
  resident batch 1/2 gives 1.001x / 1.087x, including a retained 479.185-ms
  batch-one outlier. These are not cross-version gains.

Ignored `build-cuda-ninja/profiles` evidence includes
`s52_prepare_probe.{cpp,ps1}` and diagnostics, the four-way preparation replay
and native assembly, `s52_oracle.{cpp,ps1,txt}` and partition replay summary,
native GPU/host comparisons, `s52_checks.ps1`, `s52_performance.ps1`,
`s52_followup.ps1`, primary/control/extra-mode image reports, CTest/ASan logs,
all warm/cold/control records, `s52_cold_aab.{py,json,txt}`, batch results,
and GPU boundaries. `s52_report.py` derives paired summaries;
`s52_validate.py` checks source identities, comparisons, outputs, and hashes.
`s52_final_binary_hashes.json` and `s52_artifact_hashes.json` freeze retained
binaries, libraries, source snapshots, and experiment evidence. S51's retained
files remain unchanged; its validator of live S51 sources is not a validator
of this later checkpoint.

No admin/firewall or permission error was reported; all runs completed and
no security, power, cooling, clock, priority, or service settings were changed.
Remaining work includes the cross-executable coefficient-order regression,
the cold-run variability, token-scanned DC population construction, and the
dominant Malta/wide-blur GPU work. Optimization remains active, not maxed out.

## Bounded narrow coefficient-order counting (S53)

S53 compares against S52 `5ea11e6` on the RTX 3060 Laptop, MSVC 14.37 /
CUDA 11.8 Release configuration. Qualification completes on 2026-09-06
America/New_York. This is host coefficient-order preparation in the
fully-resident workflow; CUDA source, transfers, GPU allocations, public
headers, API/ABI, and production compiler flags are unchanged.

### Establish the native bottleneck and bound

S52's coefficient-order slowdown preceded its changed ANS work. The S51
library's extracted coefficient-order object is byte-identical to the
pre-change S52 production object (SHA-256
`9efc34e02140d1fe7c5d2f0b3939479a48481a8ef3e0ee6ade2727c38c6f4227`).
Thus a coefficient-order source/object change does not explain that earlier
cross-executable anomaly. Link/layout effects and external variability remain
unresolved; the new optimization is not presented as its causal diagnosis.

The original native `CountGroupZeros` has 526 instructions and a scalar
64-bit count update, consistent with the S41 investigation. Simply narrowing
the inlined counts remains scalar in the first prototype. Isolating the
contiguous update in a small helper enables MSVC's packed 32-bit counting.
A 64-bit helper stays scalar, including the exploratory restricted-pointer
variant. The selected 32-bit helper needs neither `__restrict` nor intrinsics:
it is ordinary `counts[i] += coefficients[i] == 0` C++.

The private implementation selects `uint32_t` when the validated frame's
block area fits `UINT32_MAX`, and `uint64_t` otherwise. Each zero-initialized
counter is incremented at most once per anchor; anchors partition the block
area. Consequently neither selected type can overflow. The bound uses checked
area multiplication, with compile-time checks at `UINT32_MAX`, at 2^32, and
for `size_t` multiplication overflow. The existing size-width assertion and
all validation remain. No public frame representation changes.

Group/anchor traversal, sampling RNG state and decisions, selected transforms,
stable ties, LLF prefixes, the float-scaled sorting formula, and exact order
and Lehmer-token output are retained. The 64-bit fallback uses the same
algorithm, not a reduced-support path. For all five supported order families
and three channels, counter element storage is **23,808 instead of 47,616
bytes**. This excludes vector/allocator overhead and is not an RSS claim.

Disassembly of the actual CMake production object, not just a diagnostic
build, contains packed `pcmpeqd` / `psubd` updates inside the 32-bit
`CountGroupZeros` specialization. The emitted narrow helper has 54
instructions; the wide helper has 12 scalar instructions. The enclosing
narrow/wide group functions have 525/490 instructions; instruction count
alone is not used as a performance estimate. All **162 GPU bodies** and the
entire CUDA library remain identical to S52. `/Qvec-report:2` is used only
in a separate diagnostic compile, not added to production flags.

### Exact differential tests and isolated replay

The tracked coefficient-order fixture expands from 96 to **218 cases**,
using an independent coefficient-major 64-bit scalar reference. It covers
all seven transform shapes plus mixed families, full and sampled policies,
all-zero/nonzero, sparse/dense, tied, signed-int32-extreme, LLF-only, and
last-coefficient populations. Added 8/32/68-block-side cases cover single
groups, exact boundaries, and multiple narrow edge groups. Full order arrays
and Lehmer token streams match; small-frame cutoff and invalid-input
atomicity checks remain.

The production-source oracle also compares **218 frame cases** across frozen
S52, forced narrow, forced wide, and the adaptive entry point. It passes
**1,800 counter/guard cases**, including empty/tail lengths, offsets,
near-limit initial counts, unchanged coefficient inputs, and wide counts
beyond `UINT32_MAX`; **12,800 scaled-key cases** around FP32 precision and
integer boundaries; and three width-selection boundaries. Forced wide is
tested on ordinary frames: no multi-billion-anchor frame is allocated.
The oracle is native MSVC; the tracked fixture also runs under GCC and ASan.

A separate same-executable replay measures frozen S52, current forced-wide,
and adaptive implementations on the same completed frame. It uses a
132x132-block grid (active 1053x1049), sparse nonzero pattern, seven shapes
plus mixed, and both policies: **16 cases**. Three warmups precede twelve
rounds covering all six mode orders twice, with three calls per mode/round.
Frame construction, equality checks, output destruction, GPU work, and
Lehmer tokenization are outside the timed calls.

Adaptive counting wins **12/12 rounds in all 16 cases**, with median paired
changes from **-31.0% to -62.5%**. Representative complete order-computation
medians are:

| Replay case | S52 ms | S53 adaptive ms | Median paired change |
| --- | ---: | ---: | ---: |
| DCT8, full | 1.964 | 0.883 | -55.5% |
| DCT8, sampled | 1.631 | 1.127 | -31.0% |
| DCT16x16, full | 1.872 | 0.727 | -61.1% |
| Mixed, full | 3.042 | 1.483 | -53.7% |
| Mixed, sampled policy | 3.501 | 1.582 | -52.1% |

The forced-wide implementation stays much closer to S52: paired changes
range from -4.8% to +5.8%. These controlled results support narrow packed
counting, not a claim that any helper extraction is faster. Exploratory
unpaired runs and all replay outliers remain available but are not pooled
with this final cohort.

### Whole-workflow results and limitations

Warm release and same-executable control cohorts each use seven alternating
process pairs, three warmups, five samples, and all 41 phase fields. Inputs
are odd 3839x2159 / 1919x1079 and linear Flower 510x532, distance 1.2,
effort 7, fully resident, encoding-only. Times are medians of process medians;
percentages are medians of paired ratios, not ratios of those medians.
The coefficient-order phase includes order computation and tokenization.

| Warm release phase | 4K S52 -> S53 ms; paired change | 1080p S52 -> S53 ms; paired change | Flower S52 -> S53 ms; paired change |
| --- | ---: | ---: | ---: |
| Coefficient-order work | 19.530 -> 8.777; -54.3% | 5.185 -> 2.644; -48.1% | 1.155 -> 0.667; -39.9% |
| AC tokenization | 43.249 -> 30.608; -27.2% | 13.978 -> 11.465; -17.4% | 3.436 -> 2.863; -12.8% |
| Codestream encoding wall time | 111.778 -> 100.787; -9.8% | 46.748 -> 41.698; -10.3% | 10.906 -> 9.701; -8.7% |
| Whole encode | 423.997 -> 408.950; -3.5% | 108.249 -> 105.592; -3.9% | 23.977 -> 23.131; -5.1% |

Coefficient-order work improves in **7/7 pairs per workload**, with ranges
[-64.3%, -51.8%], [-58.4%, -41.3%], and [-49.8%, -35.1%]. Whole encode
wins 5/7, 6/7, and 5/7, with ranges [-11.6%, +5.0%], [-16.0%, +9.0%],
and [-27.9%, +20.3%]. Work counters can overlap and are not summed as wall
time. Unchanged quantization shifts -1.3% / -1.1% / -0.3% in this cohort;
those differences are not credited to the host optimization.

The single diagnostic `s53_control.exe` selects frozen S52 order computation
with `S53_REFERENCE_ORDER`; absence selects the actual adaptive code. There
is no production switch. All **46 baseline/optimized/production image
triples** match bytes. Its independent warm cohort gives:

| Same-executable warm control | 4K | 1080p | Flower |
| --- | ---: | ---: | ---: |
| Coefficient-order ms, baseline -> optimized | 16.073 -> 9.565 | 4.497 -> 2.713 | 1.157 -> 0.680 |
| Coefficient-order paired change; wins | -40.2%; 7/7 | -39.4%; 7/7 | -35.9%; 7/7 |
| Codestream wall-time paired change | +1.5% | -4.6% | -9.4% |
| Whole-encode ms, baseline -> optimized | 431.576 -> 427.037 | 108.564 -> 107.040 | 27.267 -> 24.760 |
| Whole-encode paired change; wins | -1.1%; 4/7 | -2.0%; 6/7 | -4.3%; 6/7 |

The 4K control's unchanged histogram, entropy, and section work is slower
by 10.9%, 6.4%, and 4.0%; its codestream wall-time regression is retained.
Whole-encode control ranges are [-6.2%, +4.0%], [-8.1%, +0.8%], and
[-30.9%, +29.5%]. Flower's 34.288-ms baseline and 34.333-ms optimized
outliers occur in different pairs and are both retained.

Seven cold release pairs, zero warmups and one sample, give whole-encode
medians 514.141 -> 512.727, 151.463 -> 143.386, and 44.667 -> 44.053 ms.
Paired changes are **-0.02% / -3.2% / -1.9%**, with 4/7, 6/7, and 5/7
wins. The 587.410-ms optimized 4K and 68.510-ms baseline Flower outliers
remain. A separate seven-pair same-executable cold follow-up gives whole
changes **-3.0% / -0.3% / +1.3%**, with 7/7, 4/7, and only 2/7 wins.
Its coefficient-order work improves 43.8% / 35.3% / 18.9%, with 7/7,
7/7, and 6/7 wins, but Flower codestream wall time regresses 4.5%.
The optimized Flower 65.289-ms outlier includes a 67.5% increase in unchanged
quantization in that pair. This does not establish the cause or erase the
cold regression. No new A/A/B cohort was run for S53.

GPU boundary readings are 60 -> 66 C, P3, 1282 -> 847 MHz, with software
thermal-slowdown and power-cap flags active at both boundaries. Unchanged
warm 4K quantization is around 275 ms here versus around 175 ms in S52's
final cohort. Cohorts are not pooled or clock-normalized; no GPU gain or
causal thermal diagnosis is claimed. The consistent targeted reduction is
retained, with smaller and non-universal whole-workflow benefits.

### Qualification and retained evidence

- All **71 CUDA-enabled CTests** pass in 59.76 s and all **50 CPU-only GCC
  CTests** in 18.83 s, including installed consumers.
- Four fully host-instrumented Clang 22 ASan targets pass: entropy,
  coefficient order, codestream encoder, and public codestream workflow.
  No new CUDA sanitizer claim is made for unchanged GPU code.
- All **58 image pairs** match bytes and independently decoded Butteraugli
  scores using pinned libjxl `e8ff09762481785938d8e4e01333ed3917571161`
  and linear RGB `RGB_D65_SRG_Rel_Lin`. The 46-case primary matrix includes
  seven images at distances 0.5/1.2/3, effort 7, plus sample/Flower effort 9.
  Twelve extra cases cover legacy high-density without explicit effort and
  maximum-compression at effort 7. Encoding-only and scored outputs agree.
- Batch byte checks pass with one warmup and three samples. Even 1080p
  fully-resident batch 1/2/4 gives same-version paired speedups
  1.016x / 1.169x / 1.372x; maximum-throughput gives
  0.979x / 1.503x / 2.082x. Even 4K fully-resident batch 1/2 gives
  1.089x / 1.067x. These compare concurrency within S53, not versions.

Ignored `build-cuda-ninja/profiles` evidence includes `s53_order_original.cpp`,
the narrow/restricted/portable exploratory variants and logs,
`s53_oracle.{cpp,ps1,txt}`, `s53_order_replay.{cpp,ps1,txt}` and its summary,
native host/GPU comparisons, the control source and build, correctness and
performance scripts, CTest/ASan logs, primary/control/extra-mode image reports,
all warm/cold/control records, batch results, and GPU boundaries.
`s53_report.py` derives paired summaries; `s53_validate.py` checks source
identity against production and frozen S52, ordered replay records, native
evidence, image hashes, timing arithmetic, and checkpoint identities.
`s53_final_binary_hashes.json` and `s53_artifact_hashes.json` freeze retained
binaries/libraries, source snapshots, and experimental evidence. All 33
retained S52 binary/library identities remain unchanged.

Two setup issues were corrected before performance qualification: a prototype
retained a wide pointer declaration and failed compilation, and the first
native verifier also matched lambda/cleanup symbols instead of only the
actual function. That verification attempt stopped before benchmarking;
exact symbol-prefix matching fixes the verifier. Neither was a production
failure. No admin/firewall or permission error was reported, and no security,
power, clock, cooling, priority, or service configuration was changed.

Remaining leads include token-scanned DC population construction and dominant
Malta/wide-blur GPU work. Accumulating coefficient-order populations on the
GPU is a possible later experiment, not a measured or implemented gain.
Earlier cross-executable and cold-run variability remain unresolved.
Optimization stays active; this checkpoint does not establish "maxed out."

## Multi-row Malta halo reuse (S54)

S54 compares against S53 `7e5be16` on the RTX 3060 Laptop, MSVC 14.37 /
CUDA 11.8 Release configuration, on 2026-09-06 America/New_York. It targets
the largest remaining Butteraugli GPU component in fully-resident encoding.

### Refresh the profile and reject conditional division

Fresh baseline traces, three warmups and one captured encode, spend
**42.865 of 200.831 ms** in Malta at odd 4K and **4.673 of 33.266 ms** at
odd 1080p. Horizontal33 plus joint vertical33/low-medium work is next at
25.722 / 3.978 ms. These are individual traces, not pooled timing estimates.
The capture has 407 / 394 total launches, including S50's packing kernels.

Native Malta scaling computes two correctly rounded divisions per loaded
halo value, even when the asymmetric correction is unused. A guarded
secondary-division prototype passes all 160 existing guarded cases and
36 timed exact-output cases. It helps identical/zero inputs roughly 2-5%
at 1080p/4K, but mixed inputs regress roughly 1-2%. That branch-based change
is **not adopted**; production retains both original divisions and the
original correction expression. Earlier S31 larger-thread-block and
warp-shuffle experiments also remain rejected.

### Reuse more halo values without enlarging the thread block

The retained approach keeps a 32-column tile and **256 threads**, while
allowing each thread to compute multiple output rows after one shared load
and barrier. It tests heights 16/24/32/64, with 2/3/4/8 outputs per thread.
This is different from S31's larger blocks with one output per thread.
The response loop stays rolled to avoid making multiple responses live
in registers at once. Each output keeps the original response tree and
initialization/addition expression; output pixels remain independent.

The interior logical work per tile is:

| Tile height | Outputs | Loaded/scaled halo values | Values per output | Reduction versus 8 rows |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 256 | 640 | 2.500 | baseline |
| 24 | 768 | 1,280 | 1.667 | 33.3% |
| 64 | 2,048 | 2,880 | 1.406 | 43.75% |

These are program-level load/scale counts for interior tiles, not measured
DRAM traffic. There is no approximate reciprocal, fast-math change, new
intermediate, extra submission, allocation, transfer, or image-plane capacity.
All lanes load and synchronize before skipping invalid outputs; partial
rows cannot bypass the shared-memory barrier. No barrier or shared overwrite
occurs between a lane's independent output rows.

The first five-mode screen uses 18 cases (three sizes, three input patterns,
and full/low-frequency response), sixteen warmups per mode and ten cyclic
mode-order rotations, with three launches per CUDA-event interval. All
outputs match the original fused kernel exactly; all four candidate heights
also pass the 160 guarded cases. The 64-row variant improves the six 4K case
medians by paired 19.3-22.0%, but regresses the 512x536 cases by 3.6-9.3%.
Height 24 improves all eighteen case medians, with smaller large-image gains.
All intervals and burst-wide drift remain in the report.

A second, 16-case crossover screen uses five forward and five reverse
rotations, intermediate square sizes, and wide/narrow aspect ratios. At
768x768, 64 rows regresses 1.6-2.6%; at 1024x1024 it improves only 1.5-4.1%,
versus about 12.6% for 24 rows. At 1536x1536, 64 rows improves 15.2-17.7%.
The narrow 31x32767 and wide 16384x64 cases also favor 24/32 over 64 rows.
This supports a size-dependent policy, not a single universally best tile.

The production policy uses 64-bit arithmetic for its tile-count comparisons:
retain the original 8-row computation below 128 original tiles; otherwise
use 24 rows until at least 768 64-row tiles are available, then use 64.
The cutoffs are empirically screened on this GPU, not an achieved-occupancy
measurement or a universal device optimum. Checked boundary examples are
32x1016/1017 and 1024x1472/1473. The flattened-grid fallback uses the
selected height, preserving images beyond 65,535 tile rows. Its current
64-row boundary is height 4,194,240/4,194,241.

### Native identity and controlled whole-workflow evidence

All **158 non-target GPU bodies** match S53. The four 8-row native bodies
match the original after removing only the renamed function header. All
eight 24/64-row bodies match the same-executable diagnostic kernels that
were profiled. There are 170 GPU bodies in total versus 162 before; no
stack/local allocation is introduced. The standard-grid 24/64-row kernels
use 37 registers for full response and 38 for low-frequency; flattened-grid
versions use 39/37. Shared storage is 5,120 / 11,520 bytes, versus the
original 2,560. The 8-row path retains 40/34 registers and its original code.

The first production template revision unnecessarily retained a loop even
for eight rows. Native comparison exposed that difference. A compile-time
single-output branch restores the original instructions before final
qualification; the first revision's native evidence is kept separately.
The scaling function, sum helpers, response expressions, and separate-pass
oracle are also checked against frozen S53 source.

A diagnostic-only `S54_MALTA_MODE` selects baseline, fixed 24, fixed 64,
or adaptive inside one executable. A balanced four-order whole-encode screen
uses three warmups and five samples per process for 4K, 1080p, and Flower.
Adaptive quantization changes -3.3% / -3.4% / +2.0%; whole encode changes
-2.6% / -4.3% / +1.6%. Flower quantization is slower in all four adaptive
observations, and that result is retained. Matching encoded sizes in this
screen alone are not a byte-identity claim. Separately, all **46 diagnostic
baseline/adaptive/S53-production image triples** match bytes.

Three alternating-order profile pairs per workload then isolate Malta using
that same executable. Each capture has three warmups and one measured encode.
All 24 Malta launches remain; other-kernel structure, allocation requests,
and copy counts/bytes match pairwise. Full/half-scale 4K uses 64 rows; 1080p
uses 64 for full scale and 24 for half scale; Flower uses 24 for both.

| Controlled GPU traces | 4K | 1080p | Flower |
| --- | ---: | ---: | ---: |
| Baseline Malta ms, three runs | 44.912 / 48.634 / 48.236 | 4.579 / 4.122 / 4.448 | 0.661 / 0.662 / 0.664 |
| Adaptive Malta ms | 37.389 / 38.171 / 39.229 | 3.934 / 4.062 / 4.109 | 0.612 / 0.610 / 0.610 |
| Median paired Malta change | -18.7% | -7.6% | -7.9% |
| Other GPU work change | -4.1% | +2.7% | -0.08% |
| Total GPU change | -7.9% | +1.3% | -0.9% |

Malta improves in all nine pairs, but two 1080p total-GPU observations
regress, including +6.8%. The Flower target improves while its earlier wall
screen was adverse; neither cohort replaces the other or establishes the
cause of that discrepancy. No universal full-encode gain follows from the
targeted reduction, and unchanged GPU/CPU timing changes are not credited
to it. No clock, cooling, power, security, priority, or service settings change.

### Final qualification

Final warm release and same-executable comparisons each use seven alternating
process pairs per input, three warmups, five samples, and all 41 timing
fields. Inputs are odd 3839x2159 / 1919x1079 and linear Flower 510x532,
distance 1.2, effort 7, fully resident, encoding-only. Times are medians of
process medians; changes are medians of paired ratios, not ratios of those
medians. No samples are discarded.

| Final warm release | 4K S53 -> S54 ms; paired change | 1080p S53 -> S54 ms; paired change | Flower S53 -> S54 ms; paired change |
| --- | ---: | ---: | ---: |
| Quantization pipeline | 279.704 -> 273.689; -1.1% | 54.605 -> 53.700; -2.1% | 11.768 -> 11.966; -1.3% |
| Codestream wall time | 99.944 -> 100.503; +8.2% | 39.893 -> 40.262; -2.5% | 9.877 -> 9.892; -3.2% |
| Whole encode | 409.357 -> 398.021; -3.1% | 102.123 -> 101.624; -2.0% | 23.082 -> 23.485; -0.2% |

Quantization improves in 7/7, 5/7, and 4/7 pairs; whole encode in 4/7, 5/7,
and 4/7. Whole-encode ranges are [-5.0%, +3.4%], [-10.8%, +4.4%], and
[-27.2%, +30.2%]. Unchanged 4K host codestream work is slower in 5/7 pairs,
including paired +6.9% in coefficient-order work. That slowdown is retained
and is not attributed to changed host instructions: `gjxl_codestream.lib`
is byte-identical to S53.

The final same-executable cohort gives quantization changes **-3.5% / -2.9%
/ -1.0%**, with 6/7, 6/7, and 4/7 wins. Whole-encode medians are
420.123 -> 408.606, 109.734 -> 106.592, and 23.777 -> 23.812 ms; paired
changes are **-3.7% / -1.9% / -2.9%**, with 4/7, 5/7, and 4/7 wins.
Ranges are [-6.5%, +5.4%], [-6.0%, +21.6%], and [-10.4%, +27.5%].
The 1080p 103.302 -> 125.621-ms pair and Flower 23.522 -> 30.001-ms pair
remain, alongside the earlier adverse Flower screen. Codestream wall-time
changes are +0.5% / -2.4% / -2.7%, not assumed to be GPU savings.

Seven cold pairs per input use zero warmups and one measured sample:

| Cold whole encode | 4K ms; paired change | 1080p ms; paired change | Flower ms; paired change |
| --- | ---: | ---: | ---: |
| Release S53 -> S54 | 504.066 -> 531.290; +2.8% | 141.210 -> 145.820; +3.3% | 47.974 -> 46.582; -3.1% |
| Same-executable baseline -> adaptive | 514.395 -> 517.763; -1.2% | 146.674 -> 144.508; -2.2% | 45.953 -> 45.413; -2.2% |

Cold release wins are only 1/7, 3/7, and 5/7; quantization changes
+1.0% / +2.9% / -3.3%. The separate cold control wins 4/7, 5/7, and 4/7,
with quantization changes -2.0% / -1.3% / -1.9%. Release ranges are
[-8.3%, +8.5%], [-9.0%, +12.3%], and [-16.0%, +16.9%]; control ranges
are [-12.6%, +9.3%], [-2.9%, +5.5%], and [-7.9%, +8.0%]. The favorable
controls do not erase the cold release regressions or establish their cause.
No new A/A/B cohort is claimed.

Main timing-sequence boundary readings are 59 -> 68 C, P8 -> P3, and
210 -> 337 MHz; software thermal/power flags change from inactive to active.
These are boundary readings, not per-kernel measurements. Cohorts are not
pooled or clock-normalized. No causal thermal diagnosis is claimed.

- All **71 CUDA-enabled CTests** pass in 67.35 s and all **50 CPU-only GCC
  CTests** in 17.32 s, including installed consumers. The expanded Malta
  fixture passes **656 guarded cases** and **16 tall cases**, three stages
  each, with bitwise output/guard comparisons and input immutability.
  Coverage includes partial rows, both policy cutoffs, all three tile
  heights, both grid layouts, initialization/addition, threshold neighbors,
  signed zeros, subnormals, extremes, NaN/Inf, and identical inputs.
- An internal CUDA-only testing entry launches the actual production
  specializations on small inputs independently of size policy. Its
  **48-case** fixture passes memcheck, initcheck, synccheck, and racecheck;
  all error/hazard/warning counts are zero. The racecheck completes in
  about 23 s. The complete AQ fixture also passes memory, initialization,
  and synchronization checks; memory checking includes stream-ordered
  races and leak checking, with zero leaked bytes. No full-AQ racecheck
  or tall-image sanitizer run is claimed for S54.
- Four host-instrumented Clang 22 ASan targets pass: entropy, coefficient
  order, codestream encoder, and public workflow. Their CPU-only build has
  no changed implementation and requires no rebuild work.
- All **58 image pairs** match encoded bytes and independently decoded
  Butteraugli scores using pinned libjxl
  `e8ff09762481785938d8e4e01333ed3917571161` and linear RGB
  `RGB_D65_SRG_Rel_Lin`. The primary 46 cases cover seven inputs at distances
  0.5/1.2/3, effort 7, plus sample/Flower effort 9. Twelve additional cases
  cover legacy high-density without explicit effort and maximum-compression
  at effort 7. Encoding-only and scored outputs agree.
- Batch byte checks pass with one warmup and three samples. Even 1080p
  fully-resident batch 1/2/4 gives same-version paired speedups
  1.007x / 1.331x / 1.386x; maximum-throughput gives
  0.882x / 1.454x / 1.804x. Even 4K fully-resident batch 1/2 gives
  1.026x / 0.978x. The slower throughput batch-one and 4K batch-two
  observations remain; these compare scheduling within S54, not versions.

Ignored `build-cuda-ninja/profiles/s54_*` evidence retains fresh baseline
and controlled Nsight captures/SQLite exports, original CUDA/fixture
snapshots, division and multi-row/crossover probes, resource/native output,
the four-mode diagnostic control, all image reports, CTest/ASan/sanitizer
logs, all final timing pairs, batch results, and GPU boundaries.
Historical probe sources retain their original fixture include; the
`*_repro.cu` companions change only that include to the frozen S53 fixture
for rebuilding after the tracked fixture expanded. These companions were
not substituted for the executables that produced the recorded timings.
`s54_report.py` derives paired summaries; `s54_validate.py` checks source
provenance, native identities, raw timing arithmetic/order, captured profile
structures, image hashes, sanitizer completion, and checkpoint identities.
The final binary/artifact manifests freeze binaries, libraries, all five
changed source/document files, experiment evidence, and profile exports.
All 34 retained S53 binary/library identities remain unchanged.

Two diagnostic setup issues preceded testing: a nested initializer list
needed an explicit array type, and the first standalone probe reported a
static/dynamic CUDA runtime link warning. Subsequent standalone probes use
the same shared runtime as production. A native-source verifier endpoint
was corrected to extract complete functions; saved native output was reused,
not recaptured to replace an adverse result. No admin/firewall or permission
error was reported, and every run reached a terminal state.

Public API/ABI and arithmetic remain unchanged; the new entry is confined
to internal CUDA testing. Wide blurs, remaining Malta work, token-scanned DC
population construction, and full-workflow variability remain live leads.
The optimization goal remains active, not demonstrated maxed out.

## Joint-channel horizontal33 convolution (S55)

S55 compares against S54 `058b2df` on the same RTX 3060 Laptop,
MSVC 14.37 / CUDA 11.8 configuration. Fresh three-warmup, one-capture
fully-resident profiles identify horizontal33 at 12.506 / 2.021 ms
for odd 4K / 1080p, across 18 launches. Total GPU time is
186.556 / 32.954 ms with 407 / 394 launches. The unchanged joint
vertical33/low-medium pass takes another 12.901 / 2.072 ms. This is
a targeted blur optimization, not a new arithmetic or quality mode.

### What is shared, and what is not

The previous path launches the same horizontal33 convolution separately
for three XYB channels. The joint kernel loads the three independent
halos into shared storage, shares weight loading, edge tests, normalization,
and coordinate calculations, and advances three separate sums in the
original tap order. Each channel still performs its original rounded
division. No symmetry regrouping, reciprocal approximation, reduced
precision, global fast math, or change to vertical filtering is introduced.

The retained tile remains 256 columns by four rows with 256 threads.
Every thread participates in the cooperative load and barrier; only
then can out-of-image output lanes skip work. The flattened grid retains
support for more than 65,535 tile rows. A fixed tile is retained from
the screen below; this is not a claim of universal optimality.

Low/medium preparation changes from three horizontal launches plus one
joint vertical/output launch to two launches. Six psycho-image passes
therefore remove 12 launches per profiled encode. All three packed
horizontal intermediates are still written and consumed, and all image
halo loads remain. This does **not** eliminate an image-sized temporary,
image-plane traffic, arena capacity, allocation, transfer, or submission.
Weight/address reuse is a program-level structural change; no DRAM
traffic reduction or achieved-occupancy measurement is claimed.

The original horizontal and vertical kernels remain available through
the CUDA-internal low/medium reference path. No header, public API/ABI,
host coefficient layout, encoder policy, or diagnostic runtime switch
changes in production.

### Same-executable tile screen

The diagnostic binary contains frozen S54 source, the frozen 220-case
low/medium fixture, and four independently selectable joint candidates.
All five modes, including S54, pass those guarded cases with three-stage
reuse. The fixture checks full allocations, misaligned offsets, independent
padding, Gaussian/impulse/irregular weights, signed zero, extreme magnitudes,
cancellation boundaries, NaN/Inf inputs, and immutable inputs/weights.

The timing screen has 24 cases: packed/padded 510x532, 960x540, 1920x1080,
3840x2160, 31x32767, and 16384x64, each measuring horizontal-only and complete
low/medium preparation. All 16 device allocations compare bit-for-bit
outside the timing windows. Sixteen warmups per mode precede ten balanced
orders: five cyclic forward orders and five reversed orders. Each CUDA
event window contains three repetitions; launch gaps are included, while
setup and host checking are excluded. All 1,200 timings and drift remain.

| Joint tile | Registers | Shared bytes | Packed 1080p horizontal / complete change | Packed 4K horizontal / complete change |
| --- | ---: | ---: | ---: | ---: |
| 256x4, retained | 46 | 13,960 | -13.6% / -6.7% | -11.0% / -4.7% |
| 128x8 | 46 | 15,496 | -12.9% / -6.6% | -10.9% / -3.5% |
| 128x4 | 46 | 7,816 | -9.7% / -4.8% | -6.6% / -2.2% |
| 64x8 | 46 | 9,352 | -8.9% / -4.3% | -5.6% / -3.1% |

All candidates use zero stack/spill storage. Every one of the 24 case
medians favors each candidate over its paired baseline; individual slower
samples are retained. The retained tile's padded 1080p and 4K complete
changes are -6.9% and -4.8%. Narrow 31x32767 favors 64x8 more strongly
(-52.5% / -52.4% complete, packed/padded) than the retained tile
(-43.5% / -42.8%). That does not establish a broad size-dispatch policy.

Final native qualification finds 171 GPU bodies versus S54's 170.
Every preexisting body is instruction-identical after translation-unit
hash normalization. The new production 256x4 body matches the measured
same-executable control body after removing only its renamed function
header: 46 registers, 13,960 shared bytes, no stack/local allocation.
The host codestream library is byte-identical to S54.

### Controlled full-workflow traces

A diagnostic-only `S55_HORIZONTAL_MODE` selects baseline or joint inside
one linked executable; its frozen S54 baseline is unchanged outside the
selector/prototype insertion. Three alternating pairs per workload use
three warmups and one captured encode. Every pair has 18 -> 6 horizontal
launches, identical non-target kernel structure, identical copies and byte
counts, and identical allocation requests. Total launches fall by 12.
All 46 control image triples match retained production bytes.

| Workload | Baseline horizontal ms, three runs | Joint horizontal ms | Median paired target change | Other GPU change | Total GPU change |
| --- | --- | --- | ---: | ---: | ---: |
| Odd 4K | 14.239 / 12.974 / 13.879 | 15.850 / 11.838 / 12.456 | -8.8% | +4.4% | +3.3% |
| Odd 1080p | 2.082 / 1.917 / 1.965 | 1.677 / 1.626 / 1.701 | -15.2% | +0.1% | -0.7% |
| Flower | 0.365 / 0.368 / 0.366 | 0.276 / 0.276 / 0.277 | -24.3% | -0.01% | -1.4% |

The 4K target wins two of three pairs, not all three. Its first target
regresses 11.3%, with other GPU work +15.2% and total +14.9%. Two of
three 4K total-GPU observations regress; one 1080p total also regresses.
No samples are discarded and unchanged GPU work is not attributed to the
new kernel. These are control-executable traces, not final release
captures; the exact native match ties the candidate body to production.

An earlier same-executable wall cohort has seven alternating pairs,
three warmups and five samples per process. Quantization-pipeline changes
are +0.06% / -1.32% / +0.76% for 4K / 1080p / Flower; whole-encode changes
are -1.29% / +0.27% / -0.83%, with 6/7, 3/7, and 4/7 wins. The 1080p
whole-path regression and Flower's +41.5% individual whole-encode outlier
remain. This cohort does not justify a universal end-to-end speedup.

### Qualification and final timing

The final Release build passes all 71 CUDA CTests (67.35 seconds), all
50 CPU-only GCC CTests (16.25 seconds), and installed consumers. Four
Clang 22 CPU-only host ASan targets pass: entropy, coefficient order,
codestream encoder, and public workflow. Host sources require no rebuild.

The tracked low/medium fixture expands to 320 cases over 32 geometries,
packed/padded layouts and five value/weight patterns, with three-stage
reuse and full guard checking. Added cases cover both sides of the
256-column and four-row boundaries. Its separate 1x4,194,305 case passes,
exercising the flattened grid well beyond 65,535 tile rows.

All four scoped CUDA sanitizers pass on 30 guarded cases: 33x33, 1x5,
and 257x5, both padding modes, all five patterns, three reuse stages.
Memcheck/initcheck/synccheck take about 2.6 / 2.7 / 2.8 seconds;
racecheck completes in 40.7 seconds with zero hazards, errors, or warnings.
The complete AQ fixture also passes memcheck, initcheck, and synccheck
(28.4 / 20.1 / 13.2 seconds). Memcheck reports zero leaked bytes with
stream-ordered race tracking and full leak checking enabled. The AQ
fixture's legacy console wording omits fully resident, but its main
function explicitly runs `CheckFullyResident`. No full-AQ racecheck or
tall-image sanitizer qualification is claimed.

All 58 encode/decode comparisons against S54 have identical codestream
bytes and identical independently decoded Butteraugli scores. The primary
46 span seven inputs, distances 0.5/1.2/3.0 at effort 7, sample/Flower
effort 9, and scoring on/off. Twelve additional pairs cover legacy
high-density and effort-7 maximum-compression modes. The pinned libjxl
`e8ff09762481785938d8e4e01333ed3917571161` Clang 22 decoder/metric uses
linear sRGB/D65; no size/score threshold is flagged. All 46 diagnostic
baseline/joint/production triples also agree exactly.

Final warm release measurements use seven alternating S54/S55 pairs per
workload, three warmups and five samples per process, with 41 phase fields.
Cold comparisons use seven pairs with zero warmups and one sample.
Percentages below are medians of paired ratios, not ratios of the separately
reported marginal median times; those two statistics can have opposite signs.

| Warm release phase | 4K S54 -> S55 ms; paired change | 1080p S54 -> S55 ms; paired change | Flower S54 -> S55 ms; paired change |
| --- | --- | --- | --- |
| Quantization pipeline | 266.463 -> 266.965; +0.19%, 3/7 wins | 53.905 -> 52.671; -1.44%, 5/7 | 11.742 -> 11.585; -1.75%, 4/7 |
| Codestream encoding | 93.590 -> 92.113; +2.85%, 3/7 | 38.754 -> 39.283; +1.16%, 3/7 | 9.494 -> 9.601; +0.76%, 3/7 |
| Whole encode | 396.315 -> 389.955; -0.45%, 4/7 | 99.451 -> 99.017; -0.93%, 5/7 | 22.610 -> 22.727; -0.37%, 4/7 |

The 4K quantization median is effectively flat, and all three unchanged
codestream-phase paired medians regress. Whole-encode pair ranges are
-5.69..+5.50%, -22.24..+9.55%, and -14.89..+17.82%. The retained
401.133 -> 423.207 ms 4K and 21.345 -> 25.149 ms Flower pairs illustrate
why the targeted GPU result is not a uniform wall-speedup claim.
The separate same-executable warm cohort above remains independent.

| Cold whole encode | 4K baseline -> joint ms; paired change | 1080p baseline -> joint ms; paired change | Flower baseline -> joint ms; paired change |
| --- | --- | --- | --- |
| Release S54 -> S55 | 497.822 -> 499.690; +0.87%, 3/7 wins | 144.039 -> 151.206; -1.58%, 4/7 | 45.411 -> 44.367; -6.06%, 6/7 |
| Same-executable control | 494.393 -> 503.944; +0.98%, 3/7 | 140.071 -> 139.908; -0.72%, 4/7 | 47.276 -> 47.145; +0.06%, 3/7 |

Both cold 4K whole-encode cohorts regress. Release quantization changes
are -0.29% / -0.69% / -3.13%; cold control quantization changes are
+1.49% / -0.20% / -0.77%. Control 4K codestream work is +6.09%, despite
the byte-identical host library. Cold whole-path outliers include +14.64%
at release 1080p and +22.62% at control Flower. No S55 A/A/B cohort is
claimed, and the controls do not explain away the slower observations.

Batch qualification uses one warmup and three alternating serial/batch
samples. Every result is checked against the single-image codestream and
summary. At even 1080p, fully-resident batch sizes 1/2/4 have paired
speedups 0.981/1.135/1.300x and batch medians 101.373/190.497/358.675 ms;
size 1 is slower in all three observations. Maximum-throughput gives
0.987/1.520/1.849x and 70.554/91.205/163.077 ms, with size 1 slower
in two observations. Even 4K fully-resident sizes 1/2 give 1.005/0.996x
and 410.622/874.178 ms; size 2 is slower in two of three observations
(range 0.982..1.052x). These are same-version scheduling comparisons,
not S55-versus-S54 improvements.

Control-workflow boundary readings move from 56 C / 1282 MHz / 22.48 W
to 63 C / 1282 MHz / 25.11 W. Final release/batch boundaries move from
59 C / P0 / 1282 MHz / 22.97 W to 66 C / P3 / 1282 MHz / 24.61 W.
Software thermal/power flags change from inactive to active. These are
boundary samples, not per-kernel readings, performance normalization, or
causal thermal diagnoses; the latter reading precedes the short cold-control
follow-up. No clock, power, cooling, priority, security, or service settings
were changed.

### Retention and limits

The ignored `build-cuda-ninja/profiles/s55_*` bundle retains frozen parent
source and fixture, the four-candidate probe, all 1,200 raw timings, control
source/binaries, 20 Nsight reports and SQLite exports, complete test and
sanitizer logs, native dumps, 58 quality pairs, 46 control triples, warm/cold
phase records, batch records, and summary/validation scripts. The production
and diagnostic sources used for each result are retained separately.
`s55_freeze.ps1` retains 35 binaries/libraries and four tracked source/doc
snapshots without overwriting an existing checkpoint. Use a fresh evidence
prefix for reruns, not the frozen output names.

`s55_validate.py --frozen` verifies native identity, source scope, exact
screen rows/orders/arithmetic, saved SQLite kernel/copy/allocation records,
test markers, quality hashes/scores, raw phase fields and paired summary
formulas, batch observations, unchanged parent identities, and both frozen
manifests. Final native matching connects the controlled kernel timings
to the released body without substituting a standalone prototype.

The first 4K capture completed, but Windows PowerShell stopped its export
wrapper on profiler stderr. Exporting that same saved capture completed;
a single-element array mistake then rejected the 1080p workload before a
valid capture, and the corrected resume script succeeded. No valid
capture or slower observation was replaced. The first screen summarizer
also required explicit UTF-16 decoding of PowerShell's saved log; its
corrected rerun used the same 1,200 timings. These were script errors,
not observed firewall/admin failures. All final runs are terminal.
The cause of the earlier S30 wait remains unconfirmed.

This checkpoint reduces a measured horizontal blur cost and launch count
with unchanged results. It does not establish a uniform whole-encode gain
or demonstrate that the fully-resident path is maxed out. Remaining
vertical blur, Malta, other horizontal filters, token-scanned DC population
construction, and workflow variability remain concrete investigation leads.

## Direct DC context lookup and success-path residual emission (S56)

S56 compares against S55 `81266d7`. This is host serialization work on
the fully-resident path, not an exact-coefficient GPU optimization.
Inspection of DC preparation finds a per-value linear search over 34
gradient-context runs. The search input always lies in a fixed
1,024-entry domain after the existing signed 64-bit clamp. DC tokenization
also constructs and inspects a successful `Status` for each residual.

### Remove repeated work without changing the tokens

The pinned run representation now expands into a 1,024-byte
`constexpr std::array<uint8_t, 1024>`. Context selection uses the original
`int64_t` clamp followed by one indexed byte load. A compile-time extent
assertion ties the table to the final run boundary. There is no dynamic
initialization or heap allocation for the lookup.

The DC loop also computes the original 64-bit residual directly, applies
the same signed-32-bit range checks, returns the same error message on
failure, and appends the original `PackSigned` value. Only the failure path
constructs a status inside that per-value loop. Metadata's existing
`AppendResidual` helper is unchanged. Predictor equations, Y/X/B token
order, row/group boundary behavior, vector reserve policy, output-on-failure
atomicity, and exception handling remain unchanged.

No API, ABI, public header, GPU code, image quality policy, or
coefficient/transfer layout changes. `gjxl_cuda.lib` is byte-identical
to S55, establishing preservation of its 171 GPU bodies by library identity;
this is not a fresh S56 GPU trace or native dump. Repeated HybridUint
validation in token-scanned ANS histogram construction was also identified,
but remains unchanged and is a separate next lead.

### Captured-input and four-way investigation

A diagnostic-only `S56_DC_CAPTURE` records the actual quantized DC inputs
from one fully-resident encode each of odd 4K, odd 1080p, and Flower.
There are six group-local captures: four 4K rectangles, one 240x135
1080p rectangle, and one Flower rectangle. Each stores two little-endian
32-bit dimensions followed by three packed signed-32-bit planes.
These capture-run timings include file I/O and are **not** performance
evidence.

One diagnostic executable retains the complete S55 implementation and
three independently selectable variants: lookup only, direct residual
emission only, and both. All four run the frozen original DC/metadata
fixtures successfully. A direct context oracle compares 16,387 properties
(including signed-64-bit endpoints) to the original run search. Another
72 synthetic cases span six geometries, packed/padded and offset planes,
zero/random/extreme values, and both overflow directions. All candidates
match tokens, status codes/messages, immutable inputs, and atomic failures.

The replay measures 20 cases: the six captures in packed/padded layouts
plus four 256x256 synthetic patterns in both layouts. Twelve warmups per
mode precede every one of the 24 permutations of four modes. Each recorded
time averages three calls, including allocation/token construction but
excluding output comparison and destruction. All 1,920 timings and
absolute-time drift remain. Every case median favors every candidate,
although individual slower samples remain.

| Replay case | Lookup-only paired change | Residual-only change | Combined change |
| --- | ---: | ---: | ---: |
| Captured 1080p, packed / padded | -55.7% / -56.0% | -9.1% / -6.0% | -60.8% / -61.3% |
| Captured 4K first group, packed / padded | -49.4% / -50.3% | -8.3% / -9.5% | -55.1% / -56.3% |
| Captured Flower, packed / padded | -61.0% / -59.2% | -8.6% / -5.9% | -66.9% / -67.0% |
| Near INT32_MIN, packed / padded | -2.7% / -2.0% | -16.0% / -14.8% | -20.7% / -17.7% |

The last pattern mostly selects the first run, so removing the run search
has little benefit there; the residual change remains useful. This is an
inference consistent with the source and measurements, not a hardware
branch-counter result.

### Workflow controls and native-code limitation

The first workflow screen uses four balanced orders of all four modes,
three warmups and five samples per process, on 4K / 1080p / Flower. Lookup
alone improves the DC stage by paired 35.9% / 31.3% / 33.3%, while direct
residual emission alone gives 7.9% / 12.1% / 5.4%. The combination gives
35.1% / 38.2% / 32.6%, but whole encode changes +0.63% / +1.59% / +9.41%.
In particular, **all four combined Flower whole-encode observations are
slower**, and its codestream phase regresses 7.62%. Those results remain.

A follow-up keeps the same executable and tests baseline, lookup, and
combined modes in all six permutations, with three warmups and seven
samples. Combined DC work improves 40.2% / 37.3% / 39.2%, winning all six
rounds per workload; whole encode changes -2.68% / -1.81% / -0.23%, with
4/6, 5/6, and 3/6 wins. Lookup alone gives whole changes
+0.48% / -1.38% / -0.79%. The follow-up does not reproduce the first
cohort's all-slowness Flower result, but it does not erase or explain it.
Both cohorts retain every process output, encoded-size check, and phase.

The combined change is retained on the strength of the isolated result,
the consistent DC-stage reductions, and final production qualification,
not by discarding the unfavorable first cohort. Diagnostic selectors and
capture I/O are absent from production.

Native inspection finds 430 instructions in the retained production DC
tokenizer versus 325 in the new production body. The two run-table
relocations disappear and one lookup-table relocation remains. These are
static instruction counts, not runtime retired-instruction measurements.
The diagnostic bodies are **not instruction-identical** to the corresponding
release bodies: its baseline has 362 instructions and its combined body 293,
including different validation inlining, register allocation, and layout.
Therefore no exact-native bridge is claimed from the diagnostic timings to
release. Source-equivalent controls and final release timing are independent
evidence; native inspection separately confirms the intended production
search removal. The native report preserves both failed identity comparisons.

### Final qualification and wall measurements

The final Release configuration passes all 71 CUDA CTests (69.81 seconds)
and 50 CPU-only GCC CTests (17.77 seconds), including installed consumers.
Five Clang 22 CPU-only host ASan targets pass: DC group, entropy,
coefficient order, codestream encoder, and public workflow. The tracked
DC fixture checks 6,148 public-entry context cases across all 1,024 lookup
positions, below/above-range properties, signed-32-bit endpoints, misaligned
offsets, and two guarded layouts. Both overflow directions retain the
exact status message and leave output unchanged. Existing pinned tokens,
metadata, predictor resets, invalid-input, and modular-header tests remain.

All 58 release comparisons against S55 produce identical codestream bytes
and independently decoded Butteraugli scores. The primary 46 pairs cover
seven inputs, distances 0.5/1.2/3.0 at effort 7, sample/Flower effort 9, and
scoring on/off. Six legacy high-density and six effort-7 maximum-compression
pairs also agree. The pinned libjxl
`e8ff09762481785938d8e4e01333ed3917571161` Clang 22 decoder/metric uses
linear sRGB/D65. All 46 baseline/combined/production control triples match.
No fresh CUDA-sanitizer run is claimed for this host-only change; CUDA
functional tests are rerun and the CUDA archive is byte-identical.

Final warm release and same-executable controls each have seven alternating
pairs per workload, three warmups and five samples per process, with all
41 phase fields retained. Cold runs use seven pairs, zero warmups, and one
sample. Reported times are marginal medians of process medians; percentages
are medians of paired ratios, which can differ in sign from their ratio.

| Warm release phase | 4K S55 -> S56 ms; paired change | 1080p S55 -> S56 ms; paired change | Flower S55 -> S56 ms; paired change |
| --- | --- | --- | --- |
| DC tokenization | 13.437 -> 8.822; -34.35%, 7/7 wins | 3.263 -> 2.239; -31.54%, 7/7 | 0.560 -> 0.334; -42.06%, 7/7 |
| Codestream encoding | 103.041 -> 90.494; -14.96%, 5/7 | 39.021 -> 38.067; -1.86%, 5/7 | 9.735 -> 9.259; -6.80%, 6/7 |
| Whole encode | 400.375 -> 383.334; -3.99%, 5/7 | 98.797 -> 98.086; -0.40%, 5/7 | 22.706 -> 22.186; -3.98%, 5/7 |

All 21 final release DC-stage pairs improve. Their ranges are
-42.04..-27.51%, -36.03..-23.55%, and -59.22..-33.86%. Whole-encode
ranges are -8.07..+2.96%, -11.74..+3.85%, and -25.70..+3.55%.
The larger codestream/whole changes include variation in unchanged work:
4K AC tokenization is -12.48% and histogram work -3.58%; 1080p section
writing regresses 2.61%, and Flower coefficient-order work regresses 1.30%.
The DC change alone is not credited with those non-target movements.

Final same-executable warm DC reductions are 38.00% / 38.27% / 42.15%,
again 7/7 wins each. Whole changes are -0.17% / -2.71% / -1.20%, with
4/7, 6/7, and 6/7 wins. The 4K codestream phase regresses **7.18%**,
alongside +9.41% histogram work, +11.50% AC tokenization, and +11.78%
section writing. One 1080p whole-path pair regresses 14.65% and one
Flower pair 6.13%. These observations are not dropped or explained away
by the consistent DC-stage improvement.

| Cold whole encode | 4K baseline -> candidate ms; paired change | 1080p baseline -> candidate ms; paired change | Flower baseline -> candidate ms; paired change |
| --- | --- | --- | --- |
| Release S55 -> S56 | 500.512 -> 498.749; +0.11%, 3/7 wins | 142.922 -> 139.570; -1.98%, 5/7 | 45.355 -> 45.842; +5.28%, 3/7 |
| Same-executable control | 512.713 -> 509.951; +0.22%, 3/7 | 157.667 -> 150.743; -4.39%, 5/7 | 48.429 -> 49.994; +3.73%, 1/7 |

Cold 4K is effectively flat, and both cold Flower cohorts regress.
The control DC phase still improves 35.99% / 44.70% / 32.13%, winning
all 21 pairs. Flower's control quantization and codestream phases regress
4.50% and 6.39%, each faster only once; its worst whole-path observation
is +30.95%, and the release cohort reaches +35.56%. These are retained
limits, not evidence that every workload is faster. No S56 A/A/B cohort
or causal explanation for non-target variability is claimed.

Batch checks use one warmup and three alternating serial/batch samples,
with every output codestream and summary checked against the single-image
reference. At even 1080p, fully-resident sizes 1/2/4 give paired speedups
1.019/1.275/1.271x and batch medians 116.764/179.448/367.900 ms.
Maximum-throughput gives 1.002/1.717/2.208x and
80.959/100.480/175.407 ms; one size-1 sample is slower (0.978x).
Even 4K fully-resident sizes 1/2 give 1.059/1.060x and
408.530/889.276 ms. These measure same-version scheduling, not cross-version
S56 gains; especially, no 2.208x optimizer speedup is implied.

Performance-sequence boundary readings span 58 C / P8 / 210 MHz / 11.12 W
to 67 C / P3 / 1282 MHz / 30.50 W, with software thermal/power flags
inactive initially and active at the latter reading. The latter precedes
the short cold-control follow-up. These are boundary observations, not
per-operation state, normalization, or causal thermal diagnosis. No power,
clock, cooling, priority, security, or service settings were changed.

### Evidence retention and remaining work

The ignored `build-cuda-ninja/profiles/s56_*` bundle retains frozen S55
DC/ANS source and DC fixtures, six input captures, the four-mode diagnostic,
all 1,920 replay records, both exploratory workflow cohorts, final release
and same-executable warm/cold records, 58 quality pairs, 46 control triples,
test/ASan logs, native object dumps, source-scope and arithmetic validators,
and batch observations. The freeze retains 37 executables/libraries and
four tracked source/doc snapshots. Reruns should use fresh output prefixes,
not overwrite the frozen evidence.

`s56_validate.py --frozen` rechecks source scope against S55, exact capture
dimensions, raw replay rows/permutations and summary arithmetic, test
markers, image hashes and metric reports, workflow fields and paired
statistics, native findings including the **non-identical** control layout,
unchanged CUDA archive identity, and both artifact manifests.
The source and native distinction is explicit; no failing identity test
is presented as an exact-native match.

All runs completed without an observed admin/firewall failure. The cause
of the earlier long S30 wait remains unconfirmed. The optimization goal
remains active: DC lookup work is removed, but token-scanned ANS histogram
construction, metadata validation/tokenization, major perceptual GPU
passes, and whole-workflow variability remain concrete leads.

## Token-scanned ANS validation-hoist experiment (S57, rejected)

S57 investigated the repeated public `EncodeHybridUint` call in
`PrepareDirectAnsPartition`. The partition entry already rejects invalid
HybridUint configurations, and every uint32 value has an encoding once the
configuration is valid. The candidate replaced only that per-token public
call with `EncodeHybridUintValidated`; context checks, ANS symbol/count/extra-bit
overflow checks, initial maps, prepared populations, clustering, and failure
atomicity stayed unchanged. No public API/ABI, GPU, allocation policy, or
quality-policy changes were proposed.

**The production change was rejected and restored to S56.** Captured-partition
replay improved, but the release workflow did not demonstrate an improvement.
The retained tracked change is regression coverage, not an encoder speedup.

### Captures, differential checks, and generated code

The diagnostic executable contains separate original and candidate partition
functions, selected once per partition by `S57_ANS_MODE`. Its six captures
come from single fully-resident encodes of padded 4K, padded 1080p, and Flower.
Each has an eight-context coefficient-order stream (1,756 / 1,629 / 441
tokens) and a 45-context DC stream (557,106 / 139,588 / 20,804 tokens).
Capture I/O is enabled separately and those encode times are not performance
evidence. Prepared AC population construction is not captured or changed.

The frozen original entropy fixture passes with both diagnostic modes.
An additional 19,632 comparisons cover all 816 valid configurations,
bounded/random/UINT32_MAX values, both entropy policies, initial maps,
interleaved and guarded offset-split layouts, invalid configurations,
malformed views, out-of-range contexts, and map errors. Partition/prepared
outputs, exact status codes/messages, input immutability, and unchanged
sentinel outputs on failure agree. Some valid HybridUint configurations
produce symbols outside the 256-bin ANS alphabet; those failures must not
be mistaken for invalid configurations or silently accepted.

Replay measures complete partition construction, including allocation and
clustering, not just token conversion. Each of 12 capture/layout cases uses
six warmups per mode and 12 alternating AB/BA pairs, with three calls per
timing: all 288 raw timings are retained. Paired median reductions for the
larger DC partitions are:

| Capture | Interleaved | Offset split |
| --- | ---: | ---: |
| 4K | 5.15%, 11/12 wins | 13.56%, 12/12 |
| 1080p | 4.68%, 12/12 | 5.50%, 12/12 |
| Flower | 17.34%, 12/12 | 11.70%, 11/12 |

Small coefficient-order captures are mixed: 4K interleaved regresses 4.23%
and both 1080p layouts regress 1.31% / 0.20%; Flower improves 15.77% / 12.66%.
All slower individual observations remain in the replay records.

MSVC release disassembly shows 2,113 -> 2,071 static instructions in the
whole partition function. The public conversion relocation disappears,
but **one private conversion call remains**: the compiler does not inline
the helper here. The saved work is repeated configuration validation and
successful `Status` construction, not elimination of the call itself.
Diagnostic bodies contain 1,999 / 1,956 instructions and neither is
instruction-identical to its release counterpart. These are source-equivalent
controls with a measured native-layout limitation, not exact-native controls.
Static instruction counts are not retired-instruction measurements.

The CUDA archive remains byte-identical to S56, preserving all 171 GPU
bodies by archive identity. No fresh CUDA native dump, GPU profile,
hardware-counter result, or CUDA sanitizer result is claimed for S57.

### Complete-workflow results and rejection

All percentages below are medians of within-pair ratios, not ratios of
marginal median times. Cohorts are separate; their absolute timings are
not pooled. Every warm process uses three warmups and five samples.
The histogram profile field also includes clustering and other partition
work; it is not an isolated DC token-scan timer.

The first five-pair same-executable screen gives whole-encode changes
-1.06% / +5.87% / -3.18% at 4K / 1080p / Flower (4/5, 1/5, 5/5 wins).
The separate seven-pair follow-up gives -2.59% / +1.05% / -0.72%
(4/7, 3/7, 5/7). Histogram changes in that follow-up are
-4.91% / +2.48% / +0.37%. The repeated 1080p regression is retained.
Those controls check encoded sizes during timing, not full output bytes.

A freshly compiled release phase probe then compares the candidate with
the frozen S56 release, using seven alternating pairs per workload:

| Workload | Histogram work | Codestream encoding | Whole encode |
| --- | ---: | ---: | ---: |
| 4K | +8.90%, 1/7 wins | +1.93%, 1/7 | +0.58%, 3/7 |
| 1080p | +9.85%, 1/7 | +2.95%, 1/7 | +1.78%, 1/7 |
| Flower | +1.09%, 2/7 | -5.77%, 4/7 | -2.32%, 4/7 |

Whole-encode pair ranges are -4.68..+3.71%, -13.50..+20.41%, and
-32.79..+6.28%. Unchanged stages
also vary: release quantization changes -1.09% / +1.48% / -0.47%,
and 4K coefficient-order work rises 7.74%. These observations do not prove
that the source edit caused every regression, nor establish compiler layout
or thermal state as the cause. They do fail to justify shipping this variant.

Seven cold release pairs (zero warmups, one sample per fresh process) give
whole-encode changes +1.42% / +0.67% / -3.89% (2/7, 2/7, 6/7 wins).
Separate cold same-executable pairs give +0.16% / +6.45% / -4.07%
(3/7, 2/7, 4/7). The 1080p cold control has histogram +11.87% and
codestream +15.48%, while unchanged quantization rises 3.22%.
The 4K cold-control codestream range reaches +74.39%; it is not discarded.

The rejected candidate also completes byte/summary-checked same-version
batch scheduling tests (one warmup, three samples). Even 1080p batch sizes
1/2/4 give fully-resident speedups 1.004/1.104/1.221x and maximum-throughput
0.974/1.471/2.005x. Even 4K fully-resident sizes 1/2 give 0.993/1.053x.
These are not candidate-versus-S56 optimizer gains and include slower
size-one results. Raw batch medians and all sample rows remain retained.

Performance boundary readings are 58 C / P0 / 1282 MHz / 22.97 W with
software thermal/power flags inactive, then 66 C / P3 / 937 MHz / 26.72 W
with both flags active. Boundaries do not provide per-operation state or
causal normalization. No power, clock, cooling, priority, security, or
service settings were changed. All runs completed without an observed
admin/firewall error; the earlier S30 long-run cause remains unconfirmed.

### Qualification, restoration, and retained coverage

The candidate passes all 71 CUDA CTests (73.48 s), all 50 CPU GCC CTests
(18.68 s), installed consumers, and five CPU-only Clang ASan targets:
DC groups, entropy, coefficient order, codestream encoder, and workflow.
All 58 encoded-image pairs are byte-identical to S56 and independently
decoded/scored with the pinned libjxl decoder and Butteraugli metric:
46 primary cases plus six legacy high-density and six maximum-compression.
These establish correctness of the rejected candidate, not a reason to
ignore the performance result.

Tracked entropy tests now compare scanned construction with populations
built through the checked public conversion for every valid configuration,
mapped/unmapped contexts, interleaved/offset-split layouts, and empty
sections. Exact models, costs, failure statuses, and input/output atomicity
must agree. Additional tests preserve invalid-configuration behavior even
on empty input, out-of-range contexts, and valid encodings whose symbols
exceed the ANS alphabet. The pre-existing 443,904 independently inverted
HybridUint encoding cases remain in place.

After restoration, all 71 CUDA CTests (69.04 s), all 50 CPU CTests
(17.57 s), and all five host ASan targets pass again. Production ANS
source is identical to S56; the added tests pass on that implementation.
The restored ANS object differs only in three bytes of the COFF build-time
field, and its complete instruction/relocation dump matches the parent.

The ignored `build-cuda-ninja/profiles/s57_*` bundle distinguishes the
rejected candidate sources/binaries from the restored checkpoint. It keeps
the six captures, source controls, native dumps, 288 replay rows, all
workflow cohorts, 58 image comparisons, and both qualification logs.
The timing scripts describe the rejected candidate; reproductions after
restoration must use its retained binaries, not silently substitute the
restored encoder. `s57_validate.py --frozen` checks this distinction,
restoration, differential evidence, paired arithmetic, and artifact hashes.

The optimization goal remains active. A concrete next experiment is a
small separately compiled token-scan routine that permits actual conversion
inlining without enlarging the already substantial partition function.
It still requires generated-code verification and whole-workflow acceptance;
S57 does not demonstrate that this work, or fully-resident encoding, is maxed out.

## Small token-scanned ANS histogram routine (S58)

S58 follows the rejected S57 experiment without treating its replay gains
as release acceptance. The failed version still called the private integer
converter per token. This version extracts the token-scanned loop into a
small private `AddDirectAnsTokenHistograms` routine, where the compiler
can inline `EncodeHybridUintValidated`.

The caller still validates the configuration, initial context map, and
histogram extent before calling the helper. The helper preserves section-view
validation, context bounds, ANS symbol/count/extra-bit overflow checks, and
the original token order and histogram updates. It returns one successful
status per partition, not per token. Exceptions are still caught by the
caller's existing allocation-failure boundary, and caller-visible outputs
are assigned only after the entire partition succeeds. Prepared AC
populations, clustering, model selection, and all other integer-conversion
call sites remain unchanged. There are no public API/ABI, GPU, memory-layout,
quality-policy, or scheduling changes.

### Three-way experiment and exactness

The diagnostic compares the original partition, an extracted scan retaining
the checked public conversion, and an extracted scan using the validated
conversion. Selection happens outside the scan, once per partition. This
separates extraction effects from actual conversion inlining; it is not a
per-token runtime toggle in production.

S58 reuses S57's six frozen input captures because production ANS behavior
was restored before this experiment. The three eight-context order streams
contain 1,756 / 1,629 / 441 tokens; the corresponding 45-context DC streams
contain 557,106 / 139,588 / 20,804 tokens at 4K / 1080p / Flower. Capture
hashes remain checked against S57's frozen manifest; no fresh capture-I/O
timings are presented as S58 performance.

All three diagnostic modes pass the frozen S57 entropy fixture. The replay
then checks 39,264 exact candidate/reference comparisons across 816 valid
configurations, both entropy policies, mapped/unmapped contexts, interleaved
and guarded offset-split layouts, bounded/random/UINT32_MAX values, invalid
views/configurations/maps/contexts, and real captures. Status codes/messages,
partitions, prepared populations, input immutability, and failure-atomic
outputs agree. The original alphabet-limit errors remain authoritative even
when the HybridUint configuration itself is valid.

Each of 12 capture/layout cases uses six warmups per mode, then all six
three-mode permutations twice, with three calls per timing. All 432 raw
timings are retained. They include allocation and complete partition
construction/clustering; output comparison and destruction are outside the
timed interval. Paired median DC-partition changes relative to the original:

| Capture | Extracted checked, interleaved / split | Extracted validated, interleaved / split |
| --- | ---: | ---: |
| 4K | +3.93% / +0.20% | -34.91% / -36.44% |
| 1080p | +6.94% / +2.33% | -30.07% / -31.99% |
| Flower | -5.26% / +3.72% | -29.24% / -24.02% |

The validated variant wins all 12 pairs in every capture/layout case,
including the smaller order streams, which improve 29.17–40.45%.
Extraction alone is mixed and has slower individual observations; its
slower DC medians are not discarded. Absolute timings are not pooled with
S57's separately collected replay cohort.

### Release machine code

The new release scan contains 248 static instructions, no public conversion
call, and no private conversion call. Its full instruction/relocation body
matches the whole-workflow control's validated scan exactly after anonymous-namespace
symbol normalization. The checked extracted diagnostic scan contains 222
instructions and retains its public conversion call. The larger static body
of the validated scan includes the previously out-of-line conversion; it
does not imply more dynamic work than calling that conversion per token.

The release partition changes from 2,113 to 1,888 static instructions and
calls the scan once. Diagnostic original/validated partition bodies contain
1,999 / 1,782 instructions and are **not** native-identical to those release
partition bodies. The exact match applies to the small scan, not the whole
partition or executable. No retired-instruction or hardware-counter claim
is inferred from static instruction counts.

The CUDA archive is byte-identical to S57, preserving all 171 GPU bodies
by archive identity. No fresh CUDA native dump, GPU profile, or CUDA
sanitizer result is claimed for this host-only change.

### Correctness qualification

All 71 CUDA CTests pass (68.51 s), all 50 CPU GCC CTests pass (17.90 s),
installed consumers pass, and five CPU-only Clang ASan targets pass:
DC groups, entropy, coefficient order, codestream encoder, and workflow.
The S57 all-816-configuration scanned/population oracle remains enabled,
alongside 443,904 independently inverted HybridUint encoding cases.

New tracked coverage exercises 40 failures in a later section after valid
tokens have populated local histograms and an empty section has been read.
Both entropy policies, initial-map choices, and first-section layouts are
covered. Mismatched split lengths, mixed backing storage, out-of-range
contexts, and symbols outside the ANS alphabet preserve exact errors and
unchanged code/cost outputs, including section-cost sentinels.

All 58 candidate/S57 encoded-image pairs are byte-identical and independently
decoded/scored with the pinned libjxl decoder and Butteraugli metric:
46 primary cases plus six legacy high-density and six maximum-compression.
Every paired decoded-quality score agrees exactly. GPU/host quality policy
and encoded-size contracts remain unchanged.

### Workflow screening and final measurement

The first whole-encode control screen uses all six three-mode permutations
per workload, three warmups and five samples per process. It retains 41 raw
profile fields and matching encoded-size checks, not byte comparisons
inside the timing run. The validated variant's histogram changes are
-16.67% / +2.31% / -1.73% at 4K / 1080p / Flower, with 6/6, 2/6, and 4/6
wins. Whole-encode changes are -1.43% / +3.34% / -0.53% (4/6, 1/6, 3/6).
The 1080p screen regression is retained. Extraction alone gives whole-encode
-1.16% / +1.77% / +3.03%; it is not presented as a universal improvement.

The subsequent seven-pair same-executable warm comparison selects original
versus extracted validated scans. Histogram changes are -11.40% / -8.34% /
-3.46% (5/7, 6/7, 4/7 wins); whole-encode changes are -1.78% / -1.99% /
+2.14% (4/7, 5/7, 2/7). Importantly, 4K codestream time rises 7.49% and
Flower codestream time rises 3.27% in this cohort. Those regressions remain
reported alongside the histogram gains, not credited to the optimization.

A freshly compiled release phase probe compares S58 with frozen S57 over
seven alternating pairs per workload, using the same warmup/sample policy:

| Workload | Histogram work | Codestream encoding | Whole encode |
| --- | ---: | ---: | ---: |
| 4K | -13.97%, 7/7 wins | -8.06%, 4/7 | -1.21%, 4/7 |
| 1080p | -5.15%, 4/7 | +1.97%, 3/7 | +0.08%, 2/7 |
| Flower | +1.49%, 3/7 | +3.61%, 3/7 | +2.00%, 3/7 |

The 4K histogram marginal medians are 24.375 -> 20.631 ms; all seven
paired changes lie between -32.35% and -2.28%. Percentages throughout this
section are medians of within-pair ratios, not ratios of marginal medians.
The histogram field includes clustering and other partition work, not just
the extracted token scan. Unchanged stages also vary: 4K AC tokenization
falls 13.44%, while release quantization changes +0.45% / +0.18% / +1.90%.
The code change does not establish a cause for those unrelated-stage changes.

All original cold results are retained. Seven fresh-process release pairs
(zero warmups, one sample) give whole-encode +4.09% / +5.54% / +0.82%
(2/7, 1/7, 3/7 wins). Quantization changes +0.58% / +0.08% / -1.08%.
Separate cold same-executable pairs give whole-encode -0.95% / +2.07% /
+5.26% (4/7, 2/7, 2/7); their histogram changes are -4.15% / -0.87% /
+2.82%. Cold 1080p and Flower therefore do not demonstrate an end-to-end
improvement, and Flower regresses in both principal warm cohorts as well.

Because the initial cold release totals were worse, a separately recorded
cold release **phase** cohort collects all 41 fields, again with seven
alternating pairs and zero warmups/one sample. Histogram changes are
-14.58% / -6.93% / -12.06% (7/7, 6/7, 4/7 wins), and whole-encode changes
are -4.40% / -3.80% / -3.39% (5/7, 6/7, 4/7). Codestream changes are
-2.55% / -4.25% / -8.13%, while unchanged quantization improves
1.62% / 3.24% / 5.24%. The initial 4K parent run takes 627.468 ms versus
473.280 ms for the candidate, and the 1080p parent has a 170.721 ms
outlier; both remain in the record. This follow-up supports a histogram
benefit but neither erases the first cold regression nor proves its cause.
The extra profiling boundary and separately collected cohort are explicit;
their absolute times are not pooled with the uninstrumented cold benchmark.

Same-version batch scheduling (one warmup, three samples) passes exact
byte/summary checks. Even 1080p batch sizes 1/2/4 give fully-resident
speedups 1.036/1.223/1.356x and batch medians 117.167/183.444/341.053 ms.
Maximum-throughput gives 0.965/1.517/2.067x and
69.472/91.793/158.768 ms; all three size-one samples are slower.
Even 4K fully-resident sizes 1/2 give 1.024/1.038x and
418.029/850.550 ms. These are scheduling comparisons within S58, not
S58-versus-S57 optimizer speedups.

The main performance sequence's boundary observations span 58 C / P8 /
210 MHz / 15.80 W with software thermal/power flags inactive, to
66 C / P3 / 945 MHz / 32.19 W with both flags active. The later cold-phase
sequence starts at 59 C / P0 / 1282 MHz / 22.90 W and ends at 61 C / P3 /
1282 MHz / 22.63 W. These are boundary readings, not per-operation state,
normalization, or a causal thermal diagnosis. No power, clock, cooling,
priority, security, or service settings were changed. No admin/firewall
failure or stalled job was observed.

### Retention and remaining work

S58 retains the small scan as an exact, portable CPU-path improvement:
the conversion is demonstrably inlined in release, the measured scan body
matches production, all captured partitions improve, and the 4K release
histogram phase improves in all warm and detailed cold pairs. This is a
narrower claim than a universal whole-encode speedup. The cold baseline
regressions and small-image warm regressions remain unresolved; later
work must continue measuring complete encodes rather than assume that a
local improvement predicts every workload's wall time.

The ignored `build-cuda-ninja/profiles/s58_*` bundle retains the parent
source/fixture, three-mode control, 432 replay rows, initial and final
workflow cohorts, native object dumps, 58 image comparisons, test/ASan
logs, batch observations, and the extra cold-phase cohort. The six input
captures remain immutable in S57's bundle. The freeze retains 37 tested
executables/libraries and four source/doc snapshots. Reproductions must
use fresh output prefixes rather than overwrite frozen evidence.

`s58_validate.py --frozen` checks exact extraction scope against S57,
capture identity, all differential/timing records and paired arithmetic,
the small-scan native match and whole-partition non-match, unchanged CUDA
archive identity, tests/quality results, both manifests, and predecessor
binary identities. The goal remains active: perceptual GPU work and
entropy-task scheduling/whole-workflow variability remain substantial
investigation targets. Fully-resident encoding is not demonstrated maxed out.

## Per-channel Malta fusion investigation (S59, not retained)

S59 returns to GPU perceptual work after S58. No production source, test,
API/ABI, allocation layout, arithmetic policy, or executable is changed by
this checkpoint. The fusion candidates remain in the ignored diagnostic
bundle; the retained implementation is S58. This is a completed experiment,
not a claim that fully-resident encoding is maxed out.

### Fresh profile and hypothesis

Fresh Nsight Systems captures use the current release benchmark, three
warmups, one captured sample, fully-resident AQ, and the existing profiler
range. These are individual instrumented traces, not paired speed estimates:

| Workload | Total GPU kernel ms | Kernel launches | Malta kernel ms |
| --- | ---: | ---: | ---: |
| odd 4K | 179.195 | 395 | 32.874 |
| odd 1080p | 32.266 | 382 | 3.976 |
| Flower | 6.173 | 395 | 0.611 |

The 4K trace contains 16 low-frequency Malta calls totaling 20.608 ms and
eight full-response calls totaling 12.266 ms. Other large individual
components include joint low/medium horizontal convolution (12.375 ms),
joint horizontal33 convolution (10.448 ms), and L2/final masking (7.251 ms).
The approximately 173–200 ms profiler-start API duration is instrumentation
overhead, not an uninstrumented encoder optimization target.

Each difference evaluation currently launches six Malta kernels. Per
channel, a full response initializes the accumulation and two low-frequency
responses add to it. The prototype preserves `(full + low1) + low2`, both
rounded additions, all directional-sum trees, both divisions in scaling,
threshold branches, zero halo, and the original host weight calculations.
It gathers the original stages into two independent channel plans and
launches channel 1 then channel 0. Live psycho inputs occupy separate planes
from AC accumulations; the production working-plane reuse does not require
interleaving the two channels.

Fusing each channel could reduce six launches to two per difference
evaluation and five global accumulation accesses to one per channel/pixel.
That is 32 bytes/pixel of removed program-level global accesses across both
channels, not measured DRAM traffic. Fusion does not eliminate the six
input-plane reads per channel or the scale/response math. Using 24-row tiles
also loads more halo values than the existing 64-row policy on large images: an
interior tile loads 40/32 × 32/24 values per output instead of
40/32 × 72/64. That increases theoretical input loads by 12.5 bytes/pixel
across both channels, before caches and partial-edge effects. Shared-memory
traffic, barriers, instruction size, and occupancy remain tradeoffs.

### Independent prototypes and fixture failure

All designs use 256 threads, 32 output columns, 8/24/64 output rows, and
both ordinary and flattened grids:

- **Staged:** reuse one scaled shared tile and keep the partial response in
  shared memory. Five barriers protect tile reuse; no lane exits before a
  barrier, including partial image tiles.
- **Three tiles:** hold all three scaled tiles simultaneously, synchronize
  once, and compute the three responses at each output position.
- **Thread-local array (v2):** replace shared partials with a per-thread
  array. Despite zero reported spill loads/stores, the 24/64-row variants
  have 16/32-byte stack frames. A targeted native 64-row dump confirms
  `LDL`/`STL`; this is not register-only accumulation.
- **Explicitly indexed registers (v3):** use compile-time row indices so the
  compiler can scalarize the partial array. All six grid/height variants
  have zero stack/local storage, but unrolling enlarges the response body.

The first fixture omitted a dependency between pageable host-to-device
initialization on the default stream and an oracle on a nonblocking stream.
The initial timing attempt stopped after seven cases. A diagnostic rerun
failed on the **unchanged baseline**, with zero input producing correct zero
output versus NaNs in the expected result: 138,363 differing values at
510×532. The 20-case focused reproduction had passed, so it was not a
deterministic image-arithmetic failure. CUDA permits a pageable synchronous
copy to return after staging but before its device transfer finishes; see
[NVIDIA's synchronization contract](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).
The corrected fixture synchronizes initialization before launching the
nonblocking-stream oracle. No kernel or equality rule changed to resolve
this failure. Both incomplete timing logs, the original binary/source, and
the diagnostic NaN failure remain retained and excluded from performance
summaries.

The synchronized first design passes 1,152 fused comparisons. V2 and v3
each pass 1,728: eight edge/odd shapes, six input patterns, two channels,
nine variants, and both grid forms. Patterns cover signed zero, near-equal
and mixed random values, branch thresholds and adjacent floats, identical
inputs, subnormals, finite extremes, infinities, and NaNs. Full allocations
are compared bitwise, including output guards; input buffers and oracle
scratch guards are checked. These are not tall-grid-limit tests: flattening
is forced on small cases, and no production tall-grid qualification is
claimed for the new kernels.

V3 also passes the original 656 guarded single-stage fixtures and scoped
Compute Sanitizer memcheck, initcheck, synccheck, and racecheck. Each tool
runs 72 fused cases on 65×65 inputs on a nondefault stream. There are zero
reported errors and zero race hazards. The expensive full-AQ racecheck is
not repeated. No fresh full CPU/CUDA CTest, ASan, batch, or decoder run is
claimed for this diagnostic-only checkpoint.

The v3 native audit finds 189 GPU bodies: all 171 existing bodies are
identical to the retained implementation, with 18 added experimental
kernels. All 18 have zero stack/local storage. At 24 rows, staged fusion
uses 8,192 bytes shared memory and 39/40 registers (2D/flat); three-tile
fusion uses 15,360 bytes shared memory; explicit-register fusion uses 5,120
bytes and 42 registers. At 64 rows, explicit-register fusion uses 11,520
bytes shared memory and 49/44 registers, while three-tile fusion needs
34,560 shared bytes. Zero spills alone did not establish the intended
storage behavior in v2, and fewer global accesses did not establish speed.

### Isolated timings

Every complete cohort contains 18 cases: 3840×2160, 1920×1080, and 510×532,
three patterns (zero, near-equal, mixed), and both channels. The first
cohort has seven modes, 14 balanced rounds, and 1,764 event-timing rows.
V2/v3 each have ten modes, 20 balanced rounds, and 3,600 rows. Orders rotate
and reverse so each mode appears in every position twice. Each interval
contains three channel evaluations; a full bitwise output check follows
every interval outside its event timing. Warmup counts are recorded and all
timing rows and raw outliers are retained. No incomplete cohort is summarized,
and absolute times from separate runs are not pooled.

The staged 24-row design is the consistent isolated candidate. Ranges below
are across the six per-workload case medians of paired changes, not pooled
samples or ratios of unrelated medians:

| Cohort / design | 4K | 1080p | 510×532 |
| --- | ---: | ---: | ---: |
| synchronized v1, staged24 | -3.17% to -0.57% | -13.81% to -10.02% | -18.39% to -15.90% |
| v2, staged24 | -2.37% to -0.93% | -14.06% to -6.88% | -17.22% to -14.02% |
| v3, staged24 | -2.08% to -0.52% | -14.14% to -8.15% | -17.06% to -14.71% |
| v3, explicit registers24 | -0.36% to +2.04% | -14.15% to -6.13% | -16.51% to -15.04% |

The v2 local-array 64-row variant improves 4K by 2.55–3.15%, but explicit
register scalarization at that height instead regresses 4K by 3.35–5.62%.
Three-tile 64-row fusion is approximately 43–47% slower at 4K across these
cohorts. The unfavorable variants remain in the raw records; they are not
replaced by only the winning tile height. These synthetic-input results do
not measure real captured psycho-plane populations or whole encodes.

### Whole-workflow checks and decision

A same-binary control selects the original six launches, staged24, or
explicit-register24 outside the GPU kernels. All 46 established image/
distance/effort/final-score cases produce byte-identical outputs in all
three modes, matching the frozen S58 outputs. This is 138 generated files;
S58's pinned-decoder and metric evidence is reused by exact byte identity,
not represented as fresh decode/metric executions. The control links the
S58-compatible phase object; no pre-S50 private frame ABI is mixed in.

The first whole-workflow cohort uses all six permutations of the three
modes, three warmups and five samples per process. Separate focused warm
and cold cohorts compare only staged24 against baseline in seven alternating
pairs. Warm uses three warmups/seven samples, cold zero warmups/one sample.
Every run retains all 41 phase fields. The following percentages are
medians of within-cohort paired ratios; parentheses show faster whole-encode
pairs. They must not be pooled across cohorts:

| Cohort / candidate | Workload | Quantization | Codestream | Whole encode |
| --- | --- | ---: | ---: | ---: |
| initial staged24 | 4K | +0.48% | +5.26% | +2.29% (1/6) |
| initial staged24 | 1080p | -1.45% | -1.12% | -1.53% (4/6) |
| initial staged24 | Flower | -3.37% | -3.91% | -1.82% (4/6) |
| initial registers24 | 4K | -0.82% | +0.14% | -0.43% (4/6) |
| initial registers24 | 1080p | -0.89% | -1.56% | -1.82% (4/6) |
| initial registers24 | Flower | -2.97% | -2.27% | -2.32% (4/6) |
| focused warm staged24 | 4K | -2.73% | -1.97% | -2.45% (5/7) |
| focused warm staged24 | 1080p | -1.49% | -0.22% | -0.53% (5/7) |
| focused warm staged24 | Flower | -3.17% | -2.72% | -3.28% (4/7) |
| cold staged24 | 4K | +0.52% | +7.44% | -0.04% (4/7) |
| cold staged24 | 1080p | -2.54% | -0.54% | -1.89% (4/7) |
| cold staged24 | Flower | +2.59% | +1.96% | +3.62% (2/7) |

The focused warm result does not erase the initial 4K regression, nor does
the cold 1080p improvement erase the Flower regression. Initial Flower
whole-encode pair ranges span -20.63% to +30.60% for staged24; the focused
warm range is -19.68% to +29.04%. Cold 4K codestream changes span -5.48% to
+56.37% although no CPU codestream code changed. Those observations limit
causal claims about small whole-workflow differences. The full raw phase
distributions and all outliers remain available.

The focused cohorts start at 59 C / P3 / 1282 MHz / 22.17 W and end at
65 C / P3 / 1282 MHz / 25.68 W, with thermal-slowdown and power-cap flags
active at both boundaries. These are boundary observations, not a causal
diagnosis or normalization. Builds, sanitizers, native dumps, profiling,
and performance cohorts run serially. No power, clock, cooling, priority,
security, or service settings change. No admin/firewall prompt, permission
failure, or stalled process is observed; the earlier long-running job's
suspected firewall cause remains unconfirmed.

S59 does **not retain** these kernels. There is a repeatable isolated
medium/small-image gain and a favorable focused warm cohort, but the 4K
local gain is small, initial and cold workflow results remain mixed, and
fusion adds considerable code/storage tradeoffs. This is a conservative
acceptance decision, not proof that all Malta fusion is ineffective.
A useful next step is real psycho-plane capture/replay or a more focused
scale/response experiment; other large convolution passes remain open.
No runtime fallback, geometry heuristic, or numerical relaxation is added
to conceal a failed case.

The ignored `build-cuda-ninja/profiles/s59_*` bundle retains all three
prototype generations, both invalid timing attempts, corrected checks,
8,964 valid timing rows, three workflow cohorts, 46 quality triples,
native/resource evidence, and scoped sanitizer logs. The S58 production
binary and artifact manifests remain immutable and revalidate. S59's
manifest freezes its diagnostic evidence and source/doc snapshots;
reproductions must choose new output prefixes. `s59_validate.py --frozen`
checks source/binary identity, exact-output/timing coverage, quality hashes,
native findings, paired arithmetic, and the evidence manifest. The broader
optimization goal remains active.

The v1 diagnostic probe requires `S59_SYNC_FIXTURE=1` for valid reruns;
its preserved original build script describes the failed unsynchronized
attempt. V2/v3 synchronize fixture initialization unconditionally. Do not
rerun any of these scripts in place over frozen outputs.

## Fused reference erosion and L2/final masking (S60)

Date: 2026-09-06. Parent tree: `76c962b`, with production still at S58
(`bf4968b`). Qualification uses the same RTX 3060 Laptop / CUDA 11.8 / sm86
Release configuration. This checkpoint retains the original erosion
selection logic inside a new fused kernel; the separately tested
conditional-selection rewrite is not adopted.

### Remove the sole intermediate consumer

`LaunchDifference` previously computed reference-mask erosion into plane 26,
then launched `L2FinalKernel`, whose masking was its sole consumer. The new
`ErosionL2FinalKernel` computes the same erosion value locally before
`MaskY`/`MaskDcY`. Original L2 arithmetic, asymmetry, raw-mask activity
difference, final square root, and invalid-result behavior are unchanged.
This removes one launch and one float-plane write/read per difference
evaluation: eight nominal bytes per output pixel, not a measured DRAM-byte
claim. Default profiled encodes evaluate four differences, removing four
launches overall. No transfers, scheduling policy, or quality decisions change.

The old `StoreMin3`, `FuzzyErosionKernel`, and `L2FinalKernel` bodies remain
unchanged as independent conformance oracles. The new helper keeps strict
comparisons and neighborhood traversal, including negative, unsorted,
non-finite, tied, and signed-zero inputs. It does not replace the operation
with generic min/max. The private redundant cached-mask-stride argument is
removed: every cached call already uses the working stride, also required
by the raw-mask activity calculation. Main, expanded, and subsampled paths
retain their existing geometry and output strides.

Plane 26 is no longer an erosion output, but it remains the third horizontal
33-tap psycho-work plane. The 27-plane allocation must therefore remain;
there is no capacity/VRAM-reduction claim. Public API, frame ABI, and prepared
plan layout are unchanged. The private test entry accepts a disjoint erosion
scratch plane for the reference implementation; the fused branch ignores
that scratch and the old eroded-mask pointer. Output must not alias inputs
or neighboring reference-mask values.

### Rounding failure, corrected candidates, and native code

The initial exactness gate fails before timing at 1x19, tight stride,
positive-random mask, asymmetry 0.6. Three erosion values differ by one ULP;
for example offset 9 is `0x40954561` versus `0x40954562`. Native inspection
shows a contraction-tree change, not a selection-logic error: the retained
kernel first rounds `0.3 * minimum1`, while the new source expression first
rounds `0.45 * minimum0`. Equivalent-looking FP32 expressions are not enough.
The failed sources, objects, executable, guard output, and disassembly remain
available; no timing from that version is used.

V2 explicitly matches the retained sm86 contraction tree:

```cpp
const float weighted1 = __fmul_rn(0.3f, minimum1);
const float weighted01 = __fmaf_rn(0.45f, minimum0, weighted1);
return __fmaf_rn(0.25f, minimum2, weighted01);
```

There is no tolerance relaxation or global fast-math change. Other toolkits
and architectures still require qualification; this native comparison is
for the measured sm86 build. The diagnostic executable offers four fixed
modes: original separate passes, conditional-selection erosion with separate
L2/final, fused original selection, and fused conditional selection. The
environment switch exists only in the ignored diagnostic sources.

Corrected guards pass 2,592 candidate comparisons across 12 tiny/edge shapes,
tight/padded strides, six mask patterns, three asymmetries, three candidate
modes, and two reuse rounds. Input guards and output bits are checked; the
separate erosion result is also checked where materialized. Default-stream
fixture copies are explicitly completed before nonblocking-stream work.
Four scoped diagnostic sanitizer tools pass 48 comparisons apiece, with
zero errors or race hazards.

All 171 preexisting GPU bodies are unchanged in both diagnostic and release
binaries. The diagnostic has three new bodies: standalone selection uses
24 registers, fused original uses 40, and fused selection uses 42. All have
zero stack, local, and shared storage. Release has 172 bodies, and its sole
new kernel is instruction-for-instruction identical to the measured fused
original-selection body. The first native-audit script had an overescaped
regular expression; its failed output is retained. Correcting the parser
and reusing the original dumps passes the audit without changing kernels.

### Isolated and instrumented measurements

The isolated screen contains 18 cases: 3840x2160, 1920x1080, and 510x532;
zero, positive-random, and smooth masks; tight and padded strides. Combined
timings use every permutation of four modes, with three calls per event;
standalone erosion uses 12 alternating pairs. All 2,160 timing rows retain
exact-output checks outside the event interval. Every observation is kept.
The following ranges span per-case median paired changes, not pooled times;
negative means faster.

| Size | Fused original erosion + L2/final | Fused selection + L2/final | Standalone selection erosion |
| --- | ---: | ---: | ---: |
| 4K | -14.87% to -17.62% | -14.97% to -17.83% | +11.23% to +36.68% |
| 1080p | -15.33% to -19.71% | -15.38% to -20.05% | +14.50% to +34.86% |
| 510x532 | -17.07% to -19.64% | -18.29% to -20.74% | -2.76% to +25.55% |

Chosen fused-original mode wins all 24 paired observations in all 18 cases:
432/432. Separate selection plus unchanged L2/final regresses 0.77-4.26%
across the screen. Fused selection is similar, but adds a second numerical
implementation and two registers without establishing a compelling advantage.
No input-dependent selection or geometry heuristic is added.

Sixteen same-executable Nsight traces provide two AB/BA pairs for each warm
workload and two cold Flower pairs. The targeted eight launches become four.
Total kernel counts fall 395 to 391 at 4K/Flower and 382 to 378 at 1080p.

| Instrumented cohort | Targeted GPU time, pair 1 / pair 2 | Total GPU time, pair 1 / pair 2 |
| --- | ---: | ---: |
| Warm 4K | -26.29% / -28.61% | +7.38% / -5.04% |
| Warm 1080p | -20.64% / -20.62% | +3.93% / +0.19% |
| Warm Flower | -19.99% / -20.26% | -1.11% / -1.21% |
| Cold Flower | -20.26% / -19.50% | -1.08% / -1.02% |

The cold Flower follow-up investigates its wall regression: targeted launch
API time also falls 64.21% / 62.55%, including first-use calls. Thus these
instrumented traces do not explain the wall regression as extra target GPU
or launch-API work. They also do not diagnose the remaining cause. Slower
total-GPU traces remain recorded, and profiler timing is not substituted for
release wall timing.

### Release qualification

All 71 CUDA tests and 50 CPU-only tests pass, as do five host ASan targets:
DC group, entropy, coefficient order, codestream encoder, and public workflow.
The existing 360 guarded L2/final cases with three reuse stages remain intact.
The same test adds 432 erosion/final fixtures, each with three nondefault-stream
reuse stages (1,296 output comparisons). Twelve shapes, two stride modes, six
raw-mask patterns, and three asymmetries cover signed zero, signed random,
gradients, arbitrary float bit patterns, infinities, NaNs, and subnormals.
Reuse changes the raw mask, and every input and padding guard is checked.
Zero dimensions, four invalid stride fields in both branches, null unused
inputs, and missing reference scratch are covered.

Four release sanitizer tools exercise the eight-case erosion subset, each
with three reuse stages; memcheck, initcheck, and synccheck also run full AQ.
All report zero errors; racecheck reports zero hazards, and full-AQ memcheck
reports zero leaks with stream-ordered race tracking enabled. Full-AQ
racecheck is not repeated. The sanitizer logs retain the flushed eight-case
geometry marker but omit the final buffered summary line; a separate direct
run and the full CTest log retain that summary. No missing line is treated
as an additional sanitizer result.

The diagnostic full-encode gate produces 46 baseline/fused byte-identical
pairs matching frozen S58 hashes. Release qualification freshly encodes,
decodes with pinned libjxl `e8ff09762481785938d8e4e01333ed3917571161`, and
measures 58 parent/candidate image pairs: 46 primary, six high-density,
and six maximum-compression. All codestream bytes, dimensions, and decoded
Butteraugli scores match; size/score ratios are exactly 1.0. These checks
include encoding-only and final-score paths. Timing runs check encoded size
only; their separate byte/decoded-quality gate is not inferred from size.

### Whole workflow, mislabeled cohorts, and retention

Every cohort uses seven alternating parent/candidate pairs at each size.
Same-executable controls use three warmups/seven samples, or zero/one for
cold. Release phase/public cohorts use three/five when warm and zero/one
when cold. Public benchmarks expose seven timing fields; the freshly rebuilt
S60 phase probe exposes 41. No pre-S50 frame-ABI probe object is reused.
Medians below are of paired percentage changes; the whole column includes
the number of faster candidate pairs. They are not ratios of separate medians.

| Cohort | Size | Quantization pipeline | Codestream | Whole encode (wins/7) |
| --- | --- | ---: | ---: | ---: |
| Control warm | 4K | -1.21% | -5.31% | -1.55% (4) |
| Control warm | 1080p | -1.53% | -8.14% | -4.45% (4) |
| Control warm | Flower | -2.36% | +0.21% | -0.92% (4) |
| Control cold | 4K | -0.38% | +2.06% | -0.49% (4) |
| Control cold | 1080p | -2.94% | -2.14% | +0.09% (3) |
| Control cold | Flower | +3.89% | +10.94% | +5.02% (1) |
| Release phase warm | 4K | -1.69% | +1.12% | -1.88% (4) |
| Release phase warm | 1080p | -2.34% | +0.88% | -1.77% (4) |
| Release phase warm | Flower | +2.44% | +4.73% | +2.40% (2) |
| Release public warm | 4K | -1.95% | -7.02% | -2.94% (5) |
| Release public warm | 1080p | +3.90% | -0.51% | +1.48% (3) |
| Release public warm | Flower | -1.60% | +4.44% | +5.46% (1) |
| Release phase true cold | 4K | -3.50% | -0.72% | -1.93% (4) |
| Release phase true cold | 1080p | -1.51% | +0.62% | -1.39% (5) |
| Release phase true cold | Flower | +2.08% | +5.89% | +6.32% (1) |
| Release public true cold | 4K | +2.79% | +5.03% | +2.71% (3) |
| Release public true cold | 1080p | -0.86% | +2.62% | +1.61% (3) |
| Release public true cold | Flower | -0.81% | -2.98% | -1.85% (4) |

The evidence audit catches a PowerShell harness error before freezing:
assigning a one-item array through an `if` expression produces a scalar
string, and splatting it passes `c`, `o`, `l`, `d` as four arguments. The
original Python driver silently treats that as warm. Consequently the first
`s60_release_{phase,public}_cold` files are **warm repeats, not cold evidence**.
Their recorded commands, metadata (`cold: false`), samples, and results remain
unchanged. A V2 driver validates arguments strictly, and explicit `cold`
invocations produce separate `*_cold_v2` artifacts. Diagnostic control and
profiler cold runs already had correct arguments and are unaffected.

The mislabeled warm-repeat results are retained separately, including their
largest regressions; correcting the labels does not discard observations:

| Actual warm repeat | Size | Quantization pipeline | Codestream | Whole encode (wins/7) |
| --- | --- | ---: | ---: | ---: |
| Phase, old cold label | 4K | -1.29% | -8.52% | -2.31% (6) |
| Phase, old cold label | 1080p | +1.20% | -0.88% | +0.65% (3) |
| Phase, old cold label | Flower | +0.47% | -0.19% | +0.85% (2) |
| Public, old cold label | 4K | -0.40% | +5.54% | +0.37% (3) |
| Public, old cold label | 1080p | -3.47% | -6.28% | -4.84% (5) |
| Public, old cold label | Flower | +6.70% | +8.69% | +10.62% (2) |

No favorable cohort erases a slower one. Cold-control Flower quantization
loses all seven pairs. Warm-public Flower whole changes range -6.01% to
+26.15%; true-cold public 4K spans -5.42% to +11.82%. Unchanged CPU codestream
work also varies substantially. The repeatable isolated reduction and exact
native match support retention as a targeted GPU optimization, not a stable
whole-encoder speedup. Unlike S59, its isolated 4K gain is substantial, with
no shared storage or new heuristic. Workflow variability remains unresolved.

Batch checks compare this release against its own sequential path, not S58.
Median paired speedups at batch sizes 1/2/4 are 0.932x / 1.190x / 1.389x for
1080p fully-resident and 0.990x / 1.507x / 1.941x for maximum-throughput.
4K fully-resident at sizes 1/2 gives 0.948x / 1.007x. All three-sample runs
complete and retain their byte checks; these are concurrency qualification,
not cross-version speedup estimates.

The initial release measurements start at 57 C / P8 / 210 MHz / 15.30 W,
with limiting flags inactive, and end at 68 C / P3 / 1282 MHz / 33.81 W,
with thermal-slowdown and power-cap flags active. Corrected cold runs start
at 55 C / P8 / 210 MHz / 14.61 W, inactive, and end at 60 C / P3 / 1282 MHz /
24.47 W, active. These boundary observations are not a causal diagnosis or
a normalization. Builds, sanitizers, profiling, native dumps, and performance
run serially. No power, clock, cooling, priority, security, or service settings
change. No admin/firewall prompt or permission failure is reported by these
runs; the user's suspected explanation for the earlier long runtime remains
unconfirmed. The argument-labeling failure is unrelated to firewall access.

The ignored `build-cuda-ninja/profiles/s60_*` bundle retains both numerical
generations, the failed native parser, all eight workflow cohorts with raw
fields, 16 traces, correctness/native/sanitizer evidence, and batch results.
`s60_validate.py --frozen` rechecks scoped source changes, paired arithmetic,
profile SQL correlations, decoded-image hashes, coverage, and manifests.
Thirty-seven release binaries/libraries and five source/doc snapshots are
frozen. S58/S59 evidence remains immutable and revalidates. Reproduction must
use new output prefixes, strict argument parsing, and the actual commands
inside JSON; do not rerun frozen producers in place. Larger convolution work
and complete-workflow variability remain open. The optimization goal is active.

## Cooperative final color correlation (S61)

Date: 2026-09-06. Parent: S60 `ea8ce7a`, on the same RTX 3060 Laptop /
CUDA 11.8 / sm86 Release configuration. This checkpoint changes final
color correlation in the fully-resident path, not initial CfL or exact-mode
coefficient policy. The preceding turn made verified implementation progress;
the broader optimization goal remains active.

### Keep the accumulator order, parallelize its inputs

The latest S60 trace spends 5.46 / 0.77 / 0.40 ms in final color correlation
at 4K / 1080p / Flower. Its original kernel launches one four-thread block
per color tile. Each thread follows one of four CPU-compatible accumulation
chains through every transform and coefficient. Most warp lanes are idle,
and a short serial coefficient loop repeatedly performs address, validity,
scaling, and regression work. Simply widening the final reduction would
change decision-sensitive floating-point order.

The retained kernel instead gives each tile a full 32-thread warp. Lanes
load and scale 32 contiguous coefficient positions together. Each lane keeps
a duplicate of one of the original four accumulation chains; shuffles feed
that chain eight groups of four values in the original order. The four FMA
chains, low-frequency exclusions, transform traversal, quantization scaling,
sample count, final pairwise sums, regularization, and signed-byte decisions
are unchanged. Low-frequency values may be loaded/scaled but never enter an
accumulator. Validated coefficient counts are multiples of 32. Invalid
records retain the error-bit and output-write behavior; only disjoint input
and output buffers are supported, as before.

Two full tile warps share each 64-thread block. Different tile loops never
share a warp, and there is no block-wide barrier or shared-memory exchange.
An out-of-range tail warp exits as a whole; every participating tile warp
executes its shuffles with all 32 lanes present. This is one fixed launch
configuration, not an image-size, content, or strategy-dependent fallback.

The original `FinalColorCorrelationKernel` remains unchanged behind a private
reference entry. No diagnostic environment switch enters production.
Transform-record layout, prepared-frame ABI, device allocations, transfers,
launch count, scheduler, public API, and quality policy remain unchanged.
The new kernel uses 40 registers and no stack, local, or shared storage.
All 172 prior GPU bodies remain instruction-identical; the release's sole
new body matches the measured `<32,2>` cooperative prototype exactly.

### Real inputs and competing explanations

Twelve captures come from complete baseline encodes, each matching frozen
S60 codestream bytes. They cover the tiny sample, 1080p, 4K, Flower, and
three CC0 photographs at distance 1.2/effort 7; 4K and Flower additionally
use distances 0.5 and 3.0, and Flower uses effort 9. Captures retain all
transform records, tile offsets, tables, forward coefficients, raw quantizers,
and global scale. Tile counts are 1, 510, 2,040, 72, and 64 for the respective
image groups. No random surrogate replaces these inputs in the timing gate.

Four diagnostic generations test 27 candidates:

- Seven original-arithmetic packing variants use 1/2/4/8/16/32/64 tiles
  per block, replacing shared reduction with four-lane shuffles.
- Eight variants split independent X/B or quadratic/linear terms across
  more lanes, with different numbers of tiles per block.
- Ten variants cooperatively load 8/16/32 coefficients while preserving
  the four accumulation chains, with different tile/block layouts.
- Two variants replace division/remainder by valid power-of-two widths
  with exact shifts/masks: one original kernel and one cooperative kernel.

Packing helps the large synthetic images but can badly hurt photographs:
the initial screen has 46-356% photographic regressions across packing
choices. It combines differing transform loops within warps and reduces
independently schedulable tiles on smaller frames. Splitting regression
terms alone is approximately flat when a block still owns one tile;
packing those variants again trades large-image gains for photographic
regressions. These results reject idle-lane count alone as an adequate
performance model.

Cooperative loading improves both regimes. Two separate screens of the
selected 32-lane/two-tile configuration give the following median paired
target-stage changes. Ranges below span image cases within a group, not
pooled absolute times; negative means faster.

| Captured input group | V3 screen | Independent V4 screen |
| --- | ---: | ---: |
| 1080p | -77.78% | -76.81% |
| 4K, three distances | -66.64% to -67.67% | -65.47% to -65.86% |
| Flower, three distances and effort 9 | -63.34% to -67.99% | -63.86% to -68.43% |
| Three CC0 photographs | -63.68% to -65.88% | -62.28% to -65.33% |

V3 uses 26 balanced orders over 13 selected modes; V4 uses all 24
permutations of four modes. The selected kernel wins every paired
observation for all 11 non-tiny captures in both screens. The tiny sample
also improves by approximately half, but not every tiny observation wins.
V1/V2/V3/V4 retain 1,536 / 6,144 / 4,056 / 1,152 exact-output timing rows:
12,888 total, each with output/error checks outside the event interval.
Each event contains three kernel calls. Absolute 4K baseline times vary
materially between cohorts, so they are not pooled or used to normalize one
another. All outliers and slower candidates remain in the record.

Native inspection confirms a general integer-division sequence despite
widths being limited to 8/16/32. The indexing-only variant improves the
non-tiny original kernel by 12-20%, but does not consistently improve the
cooperative kernel. It is not retained. Native reciprocal counts fall from
four to three; total static instruction counts are 560 / 896 for original /
cooperative versus 848 / 904 for their indexed variants. Static instruction
count is not an executed-work estimate. All 197 V3 GPU bodies remain
unchanged in V4, which adds only those two variants. An overescaped audit
regex initially records zero instruction counts; the failed count metadata
and script are preserved. A corrected parser reuses the same dumps and
validates positive counts without rebuilding or changing kernels.

### Arithmetic and workflow qualification

The four prototype generations pass 4,032 / 8,640 / 14,400 / 15,552 guarded
map/error comparisons. These are repeated comparisons across overlapping
fixtures, not that many unique images. Cases exercise every supported
strategy, nontrivial channel spacing, all four tile-value-offset residues,
partial groups, empty tiles, invalid strategies/counts/quantizers, signed
zero, overflow-scale inputs, NaNs, infinities, and subnormals. Inputs and
both output guards are checked after nonblocking-stream execution.

A separate instrumented oracle exposes all 16 floating-point partial sums
per tile before final reduction. It passes 3,240 synthetic and 360 real-capture
comparisons, including three reuse stages and changed quantizer scales.
Every partial sum, map byte, error word, and guard matches bit-for-bit.
These checks prevent matching rounded map bytes from concealing a changed
accumulation order. The instrumented audit is not the performance binary.

All 72 CUDA tests and 50 CPU-only tests pass. The new permanent final-CfL
test adds 192 fixtures with three reuse stages (576 map/error comparisons),
including valid global-scale boundaries 1 and 32,768. Pageable fixture
copies are explicitly completed before the nonblocking work stream.
Five host ASan targets pass: DC group, entropy, coefficient order, codestream
encoder, and public codestream workflow.

Four diagnostic GPU sanitizer tools pass 600 comparisons each. Four release
tools exercise the 24-comparison small CfL subset, and memcheck, initcheck,
and synccheck exercise full AQ. The first release wrapper accidentally
retains an undefined old variable in its optional-argument splat: basic
memcheck passes, but the leak/stream-order options do not run. The final
validator catches the absent leak summary. An explicit-argument rerun
passes with zero leaks and errors and stream-ordered race tracking enabled.
Both runs, the faulty wrapper, and the failed validation log are retained;
the basic run is not mislabeled as extended coverage. All scoped racecheck
reports have zero hazards. Full-AQ racecheck is not repeated.

Forty-six control codestream pairs match S60 hashes. Release qualification
then freshly encodes, independently decodes, and measures 58 image pairs:
46 primary, six high-density, and six maximum-compression, including
encoding-only and final-score paths. Pinned libjxl
`e8ff09762481785938d8e4e01333ed3917571161` accepts every output; bytes,
dimensions, and decoded Butteraugli scores match exactly. All size/score
ratios are 1.0. Timing runs check encoded size only; their byte/quality
evidence comes from these separate gates.

### Complete-workflow traces and limits

Sixteen same-executable traces contain two AB/BA pairs per warm workload
and two cold Flower pairs. The target remains one launch; total launch
counts remain 391 at 4K/Flower and 378 at 1080p.

| Instrumented cohort | Target GPU, pair 1 / pair 2 | Total GPU, pair 1 / pair 2 |
| --- | ---: | ---: |
| Warm 4K | -61.16% / -55.42% | +1.31% / +3.79% |
| Warm 1080p | -76.96% / -77.14% | +1.67% / -2.50% |
| Warm Flower | -65.62% / -65.68% | -4.15% / -4.15% |
| Cold Flower | -65.95% / -65.81% | -4.24% / -4.42% |

The targeted gain survives instrumented complete-workflow execution, but
slower total-GPU traces remain. The first 1080p pair also has a +1,124.61%
target launch-API outlier. No profiler time is substituted for public wall
time, and no unrelated phase improvement is attributed to this kernel.

Same-executable wall controls use seven alternating pairs, three warmups /
seven samples when warm and zero warmups / one sample when cold. These are
fresh-process zero-warmup measurements, not a guarantee of a physically cold
GPU. Every raw phase and observation is retained.

| Control cohort | Size | Quantization | Codestream | Whole encode (wins/7) |
| --- | --- | ---: | ---: | ---: |
| Warm | 4K | +0.98% | -1.37% | +1.08% (2) |
| Warm | 1080p | -1.15% | +1.57% | -0.69% (4) |
| Warm | Flower | -3.73% | -0.33% | -2.66% (6) |
| Cold | 4K | -2.09% | +4.57% | -1.30% (5) |
| Cold | 1080p | -2.98% | -5.66% | -2.48% (5) |
| Cold | Flower | -2.09% | +11.82% | -2.93% (4) |

Warm-control 4K quantization loses all seven pairs despite the targeted
GPU reduction. Cold gains do not erase that observation. Unchanged CPU
codestream work also varies; the full-workflow cause is not established.
The change is retained for its substantial, exact, independently repeated
target-stage gain, not as evidence of a universal encoder speedup.

### Final release measurements and evidence retention

Release comparisons use frozen S60 binaries against the final S61 build.
The phase probe is freshly rebuilt with current headers and exposes all
41 fields; the public benchmark exposes seven. Each cohort has seven
alternating pairs, with three warmups/five samples for warm measurements
and zero/one for cold. Argument parsing is strict, and the driver builds
a multi-item argument array before adding `cold`; the S60 scalar-splat
labeling problem does not recur. JSON metadata and actual commands verify
every warm/cold label.

All changes below are medians of paired percentages, not ratios of separate
medians. Pair distributions and unchanged-phase observations remain available.

| Release cohort | Size | Quantization | Codestream | Whole encode (wins/7) |
| --- | --- | ---: | ---: | ---: |
| Phase warm | 4K | -0.35% | +5.29% | +1.17% (2) |
| Phase warm | 1080p | -1.68% | +1.07% | -0.53% (5) |
| Phase warm | Flower | -4.19% | -4.52% | -5.01% (7) |
| Public warm | 4K | -2.34% | -3.71% | -2.12% (5) |
| Public warm | 1080p | -2.55% | -1.44% | -1.42% (5) |
| Public warm | Flower | -3.89% | -4.83% | -2.77% (6) |
| Phase cold | 4K | -1.94% | +0.28% | -0.56% (4) |
| Phase cold | 1080p | -5.22% | -7.24% | -6.51% (7) |
| Phase cold | Flower | -1.57% | +7.11% | +2.99% (3) |
| Public cold | 4K | -4.65% | -1.95% | -3.27% (5) |
| Public cold | 1080p | +3.28% | +5.87% | +2.64% (1) |
| Public cold | Flower | +0.67% | +9.61% | +5.28% (2) |

The warm public gains do not erase warm phase/control 4K regressions or
cold public 1080p/Flower regressions. Public warm Flower whole-encode
pairs range -26.95% to +19.46%; public cold 1080p ranges -0.49% to +41.56%.
The favorable separate cold phase cohort does not replace the slower cold
public cohort. Kernel gains are much larger percentages than complete
encode gains because the target is only one part of the pipeline; changing
timing in unchanged work also prevents assigning the entire wall change
to this kernel.

Batch qualification compares the new release against its own sequential
path, not against S60. At batch sizes 1/2/4, median paired speedups are
0.970x / 1.232x / 1.285x for 1080p fully-resident and
0.942x / 1.524x / 1.864x for maximum-throughput. At 4K, fully-resident
sizes 1/2 give 1.068x / 0.961x. Each has one warmup and three samples and
retains its byte checks. This validates the existing concurrency path;
it is not a cross-version throughput claim.

The first replay screen starts at 55 C / P8 / 210 MHz / 15.59 W with
limiting flags inactive and ends at 60 C / P3 / 1777 MHz / 28.60 W with
thermal-slowdown and power-cap flags active. Same-executable wall controls
start at 60 C / P3 / 1282 MHz / 24.17 W and end at 64 C / P3 / 1282 MHz /
25.17 W, both flags active. Final release measurements start at
61 C / P3 / 1777 MHz / 28.41 W and end at 68 C / P3 / 322 MHz / 24.46 W,
both flags active. These boundary snapshots are not continuous clock
measurements, normalization factors, or a causal explanation for regressions.
Builds, profiling, sanitizers, native dumps, and performance runs are serial.
No clock, power, cooling, priority, security, or service settings change.
No admin/firewall prompt or permission failure is reported by these runs;
the earlier suspected firewall cause remains unconfirmed.

The ignored `build-cuda-ninja/profiles/s61_*` bundle retains every generation,
12 real-input captures and hashes, all 12,888 timing rows, floating-partial
audits, 16 traces, six whole-workflow cohorts, quality/native/sanitizer logs,
and the two validation-script issues. Thirty-eight release binaries/libraries
and six source/doc snapshots are frozen. `s61_validate.py --frozen` checks
source scope and ABI, native findings, capture sizes/hashes, exact coverage,
paired arithmetic, SQL correlations, test/quality evidence, and manifests.
S59/S60 evidence remains immutable and revalidates. Reproduction requires
fresh output prefixes; do not rerun frozen producers in place, and use the
explicit extended-memcheck script for the omitted optional flags. Larger
convolutions and remaining complete-workflow variability are still open.

## Vertical low-medium row reuse (S62)

Date: 2026-09-06. Parent: S61 `fbf2249`, on the same RTX 3060 Laptop /
CUDA 11.8 / sm86 Release configuration. This checkpoint targets the
fully-resident Butteraugli vertical33/low-medium pass. S61 made verified
implementation progress; this work continues the original optimization goal.

### Share tap inputs without changing a sum

The latest S61 cooperative-control 4K trace attributes 13.65 ms to six
vertical33/low-medium launches, alongside 11.76 ms of horizontal33 work.
The existing vertical kernel gives each thread six output rows in sequence,
with three independent channel sums for each row. Adjacent output rows
repeatedly read overlapping shared-memory neighborhoods.

The retained kernel groups three adjacent rows per thread. A loaded input
value feeds up to three output sums, in increasing input-row order. Each
individual output still visits taps 0-32 in its original FMA order. Nine
independent accumulation chains replace three-at-a-time sequential work.
For an interior output triple, source-level input loads fall from 99 to 35
per channel. The compiler's shared-load instruction count also decreases
against a multiple-output counterfactual described below. This is not a
claim of reduced image-plane or measured DRAM traffic.

The fixed tile remains 32 columns by 48 rows, with 256 threads and the
same three shared halos. All lanes reach the cooperative-load barrier;
only then may invalid columns return or partial output rows skip stores.
The flattened tile grid retains tall-image support. A triple touching a
vertical edge uses the original included-weight order separately for each
row, including rows in that triple which individually have full coverage.
Rounded channel divisions and all six low/medium expressions remain fixed.

The new body uses 54 registers versus 46 previously, and the same 30,856
shared bytes, with zero stack/local storage. The runtime occupancy API
reports three 256-thread blocks per SM for both: a theoretical 768 of 1,536
threads, not measured achieved occupancy. No size/content dispatch, precision
change, diagnostic environment switch, or graph enters production. Launch
count, horizontal intermediates, arena capacity, allocations, transfers,
metadata/frame ABI, public API, and quality policy remain unchanged.

The previous resident vertical kernel and the older separate-pass path both
remain available as private conformance oracles. All 173 prior GPU bodies
are instruction-identical; the release adds one body, exactly matching the
measured first screen, second screen, and complete-workflow control.

### Competing layouts and explanations

The first diagnostic binary contains the unchanged S61 Butteraugli source
and eight candidates: 48-row tiles with 2/3/4/6/8 outputs per thread at 256
threads, 32 rows/four outputs/128 threads, and 64 rows/four or eight outputs/
256 threads. All nine modes pass 320 guarded fixtures with three reuse
stages: 8,640 comparisons. Every mode also passes the 1x4,194,305 tall case,
whose flattened launch exceeds 65,535 tile rows.

The second generation adds 24-row tiles with 2/3/4 outputs at 256 threads,
48 rows/three outputs/512 threads, and two tap-major controls. The latter
interleave three or six output sums but reload each tap/output input in
source, rather than explicitly forwarding an input to multiple outputs.
All six new modes plus baseline pass the same fixtures: 6,720 comparisons.
These counts include overlapping cases, not that many distinct images.

V1 retains 3,888 event observations across 24 cases: horizontal-plus-vertical
preparation and vertical-only, packed/padded 510x532, 960x540, 1920x1080,
3840x2160, 31x32767, and 16384x64. Nine modes use 18 balanced forward/reverse
cyclic orders. V2 retains 2,400 vertical-only observations across the same
12 geometries/paddings, ten selected modes, and 20 balanced orders. V2 uses
three-kernel captured submissions on a nonblocking stream; V1 uses three
ordinary launches. Six warmup submissions per mode precede each screen.
All 16 allocations compare bit-for-bit after every event, outside timing.
These are Gaussian-weight synthetic fixtures, not captured psycho images;
real complete-encoder evidence is separate. Different cohorts are not pooled.

The 24-row/three-output variant raises theoretical residency to four blocks
per SM and improves smaller images more, but is nearly flat at 4K. Its ideal
interior image-plane accesses rise from 56 to 64 bytes/output pixel because
of the larger halo-to-output ratio, excluding weight loads and partial tiles.
Those are logical access counts, not measured memory transactions. The
512-thread variant offers another 1,024-thread theoretical residency point,
but is not consistently better. More theoretical occupancy alone does not
identify the best tile.

The retained three-output row-reuse kernel has 219 static shared-load
instructions versus 411 for its tap-major counterpart; register counts are
54 versus 53. These counts include both interior and edge paths, not executed
instructions per pixel. Yet their V2 packed median differences are only
0.2-2.0% on five tested shapes; border-heavy 16384x64 differs by 13.9%.
Substantial improvement therefore survives without explicit input forwarding:
shared-load count alone does not explain the gain. Multiple independent sums,
weight reuse, loop structure, and residency all belong in the performance model.

### Repeated stage measurements and a timing audit

The selected 48-row/three-output configuration gives these median paired
changes against S61 within each cohort. Negative means faster. The two-pass
column includes horizontal filtering, not the whole encoder.

| Geometry, packed | V1 vertical | V1 two-pass preparation | V2 graph vertical |
| --- | ---: | ---: | ---: |
| 510x532 | -10.24% | -5.59% | -11.97% |
| 960x540 | -9.80% | -2.25% | -14.27% |
| 1920x1080 | -14.65% | -6.34% | -15.05% |
| 3840x2160 | -8.83% | -12.34% | -7.95% |
| 31x32767 | -15.00% | -6.93% | -16.28% |
| 16384x64 | -11.38% | -6.07% | -15.80% |

Individual regressions and large outliers remain. A larger two-pass change
than a vertical-only change does not imply that unchanged horizontal work
became faster: those are separate event cohorts. The selected V2 mode wins
20/20 packed 4K pairs; V1 wins 15/18. No outlier filtering, clock normalization,
or cross-cohort pooling is used.

A separate diagnostic uses all 24 permutations of four modes: baseline,
selected row reuse, the smaller 24-row candidate, and the three-output
tap-major control. It records 96 three-kernel event windows per capture and
checks every allocation after each window. Two plain-launch and two
node-traced graph captures link all 384 windows to 1,152 correctly named GPU
kernels in chronological order. These are instrumented diagnostics, not
release wall-time measurements.

| Short trace | Selected event-window change | Selected kernel-duration change |
| --- | ---: | ---: |
| 4K, plain | -6.41% | -6.55% |
| 4K, graph/node tracing | -6.45% | -6.47% |
| 510x532, plain | -6.31% | -12.51% |
| 510x532, graph/node tracing | -10.57% | -12.64% |

In the plain 4K baseline, the maximum event average is 15.089 ms and the
maximum summed-kernel average is 15.061 ms, versus medians 2.170 / 2.138 ms.
Median internal inter-kernel gaps are only 0.0029 ms per call, and median
event excess over summed kernel time is 0.0299 ms. The large spike appears
inside recorded kernel intervals, not primarily between launches. This
does not distinguish lower GPU clocks, preemption, or other device-side
interference; no causal power/OS diagnosis is claimed.

### Measurement-tool corrections

The first source snapshot was truncated by a command-output limit and failed
compilation. The complete snapshot was reconstructed in bounded chunks and
verified against the current source before any successful diagnostic build.
The failed snapshot/build log remain. A nested entry-point macro also caused
a timing-audit compile failure; the corrected audit includes the fixture
directly. Neither issue changes production code or invalidates a passing run.

The first two short captures do not relay the application report through the
profiling wrapper's stdout. Moving the report before profiler stop alone
does not resolve this. The successful audit writes a fresh report file,
flushes it before stopping capture, and explicitly allows normal process exit.
Those incomplete captures remain, without mapped event-window claims.
Default graph-granularity captures also lack the kernel table required by
the parser on this installation. Two fresh graph captures explicitly request
node granularity; the original graph-only captures and failed parser remain.
Only the node-granularity replacements supply the graph kernel-duration
comparisons above. Their extra instrumentation is not treated as free.

A pre-performance source validator initially mismatches comma whitespace in
its expected reference entry. The expected text is corrected; production
source and binaries are unchanged, and the corrected validation passes
before release performance starts.

### Complete-workflow controls and release measurements

A diagnostic executable retains every S61 GPU body and selects baseline or
row reuse before encoding through a cached, diagnostic-only environment
choice. Its 46 image pairs exactly match the frozen S61 codestream hashes.
Two unprofiled cohorts use seven alternating parent/candidate pairs per
workload: three warmups/seven samples for warm runs, zero warmups/one sample
for cold runs. All 41 phase fields and actual arguments remain in the reports.
The following are median paired changes, not ratios of aggregate medians.

| Same-executable control | Quantization pipeline | Codestream encoding | Whole encode | Whole wins / 7 |
| --- | ---: | ---: | ---: | ---: |
| Warm 4K | -0.90% | -3.89% | -0.98% | 4 |
| Warm 1080p | -0.97% | +1.57% | +0.63% | 3 |
| Warm Flower | +0.03% | +0.13% | -0.65% | 5 |
| Cold 4K | -6.85% | +1.64% | -4.83% | 5 |
| Cold 1080p | -1.51% | -3.25% | -1.45% | 5 |
| Cold Flower | +4.17% | -0.04% | +4.77% | 2 |

The cold Flower regression prompted separate complete-workflow traces, not
removal of that observation. Sixteen captures cover two alternating pairs
for each warm workload and two cold Flower pairs. Each encode still launches
the targeted vertical kernel six times, with 391 total launches at 4K/Flower
and 378 at 1080p. Both target and total GPU durations are retained separately.

| Complete trace | Target GPU pair 1 / 2 | Total GPU pair 1 / 2 |
| --- | ---: | ---: |
| Warm 4K | -36.37% / -25.16% | -11.82% / +0.16% |
| Warm 1080p | -14.53% / -12.03% | -7.07% / -2.12% |
| Warm Flower | -15.03% / -15.70% | -0.84% / -0.76% |
| Cold Flower | -15.55% / -14.86% | -0.84% / -0.77% |

The targeted GPU work improves in all eight trace pairs, including cold
Flower. Host API intervals do not uniformly improve: target launch API time
increases 342.8% in the second cold Flower pair and 138.6% in the second warm
Flower pair. These intervals, total-GPU regressions, and unprofiled whole
times remain distinct evidence. Profiling does not explain all host/device
variability or establish a release wall-time gain.

The final release comparison uses retained S61 phase/public executables and
fresh S62 executables, with seven alternating pairs per workload and cohort.
Warm runs use three warmups/five samples; cold runs use zero warmups/one
sample. Phase and public reports retain all 41 and seven fields respectively,
including commands, binary hashes, raw samples, paired ranges, and win counts.
Timing checks encoded sizes; byte identity comes from separate quality runs.

| Release cohort | Quantization pipeline | Codestream encoding | Whole encode | Whole wins / 7 |
| --- | ---: | ---: | ---: | ---: |
| Phase warm 4K | -3.14% | -4.13% | -3.37% | 7 |
| Phase warm 1080p | -0.94% | +0.51% | -1.12% | 4 |
| Phase warm Flower | -3.31% | -1.24% | -1.57% | 4 |
| Phase cold 4K | -1.43% | -7.59% | -4.98% | 5 |
| Phase cold 1080p | +3.35% | +3.29% | +3.16% | 2 |
| Phase cold Flower | -1.95% | -4.25% | -2.84% | 4 |
| Public warm 4K | -3.01% | -5.00% | -1.60% | 4 |
| Public warm 1080p | +0.67% | +1.42% | +1.00% | 3 |
| Public warm Flower | +0.76% | +3.30% | +5.03% | 2 |
| Public cold 4K | -1.39% | -2.97% | -2.18% | 5 |
| Public cold 1080p | -0.95% | +6.10% | +1.19% | 2 |
| Public cold Flower | +2.68% | +5.02% | +1.89% | 2 |

All slower samples remain, including +31.25% whole-encode change in one cold
1080p phase pair and +36.09% in one warm Flower phase pair. No clocks are
normalized and no cohort is substituted for another. The V1 screen's endpoint
snapshots span 54 C/P8/210 MHz to 62 C/P3/1,185 MHz; V2 graph spans 58 C/P8/
210 MHz to 61 C/P5/862 MHz. The unprofiled same-executable control spans
59 C/P3/1,282 MHz to 64 C/P3/1,282 MHz. Final release performance starts at
59 C/P0/1,282 MHz/23.02 W with thermal/power flags inactive and ends after the
batch checks at 69 C/P3/322 MHz/24.22 W with both flags active. These are
endpoint snapshots, not continuous attribution to individual pairs. The
throttled endpoint further limits causal interpretation of small wall-time
differences. No clock, power, priority, security, or service setting is changed.

Current-build batch checks use one warmup/three samples and require exact
serial/batch codestream identity. Median paired speedups below compare batch
operation with serial operation of S62, not S61-versus-S62 throughput.

| Current batch policy | Batch 1 | Batch 2 | Batch 4 |
| --- | ---: | ---: | ---: |
| 1080p fully-resident | 1.067x | 1.162x | 1.311x |
| 1080p maximum-throughput | 0.978x | 1.423x | 1.882x |
| 4K fully-resident | 1.042x | 1.040x | not run |

### Conformance, sanitizers, and retained evidence

The permanent low/medium test compares candidate, previous resident kernel,
and separate-pass oracle across all 16 allocations. Reuse changes input
values, resets poisoned output/padding, and finally nulls the unused blur
planes for both resident paths. Six added geometries cover short triples and
the vertical boundary: 1x2, 1x6, 1x7, 33x16, 33x17, and 33x18. The resulting
380 fixtures, three reuse stages, and two oracle comparisons give 2,280
comparisons, not 2,280 distinct images. Read-only inputs/weights, all padding,
and ignored planes remain intact. Empty/invalid plans and the tall flattened
grid remain covered. All 72 CUDA and 50 CPU tests pass, as do the five host
ASan targets for DC grouping, entropy, coefficient order, codestream encoding,
and public workflow.

Fresh release quality runs encode and decode 58 parent/candidate pairs: 46
fully-resident cases plus six high-density and six maximum-compression cases.
Every codestream is byte-identical; decoded dimensions and Butteraugli scores
match, with size/score ratios of 1.0. The pinned reference decoder is revision
`e8ff09762481785938d8e4e01333ed3917571161`. These are fresh release decodes,
separate from the earlier same-executable hash-only control.

All four diagnostic GPU sanitizer tools pass across nine prototype modes,
30 fixtures, and three reuse stages per tool. Release memcheck, initcheck,
synccheck, and racecheck pass on the new low/medium test's 60 sanitizer
fixtures; release full-AQ memcheck/initcheck/synccheck also pass. Both release
memchecks and diagnostic memcheck explicitly enable stream-order race and
full leak checks: zero errors/leaks. Both low/medium racechecks report zero
hazards. No full-AQ racecheck is repeated.

The diagnostic racecheck runs for 323.755 seconds with observed CPU/GPU work.
After five minutes a possible stop/narrowing was announced, but a read-only
process check found it had already exited successfully before any stop was
issued. It was not interrupted. Release low/medium racecheck takes 98.385
seconds and passes. No administrator prompt, firewall block, or permission
failure is encountered in S62; the older suspected firewall explanation is
not retrospectively confirmed.

The ignored `build-cuda-ninja/profiles/s62_*` bundle retains scripts, failed
and successful diagnostic sources/logs, all raw screens and short-capture
CSV reports, disassembly/resources, 16 workflow traces, four release timing
cohorts, quality outputs, tests, sanitizers, and batch/state logs. The checkpoint
retains 38 binaries/libraries and snapshots the two changed CUDA files, test,
and two documentation files. `s62_validate.py --frozen` checks source scope,
native identity, raw observations and trace mapping, paired statistics and
documented tables, quality hashes, test/sanitizer outcomes, frozen artifacts,
and unchanged S59/S60/S61 evidence. No generated build artifact is committed.

S62 is retained for the repeated bit-exact targeted improvement across
competing layouts and complete-workflow traces. It is not a universal
whole-encoder speedup. Stable-condition workflow repeats, other CUDA devices,
and the remaining horizontal preparation and dominant resident GPU stages
remain open; this checkpoint does not establish that optimization is maxed out.

## Horizontal input reuse (S63)

S63 compares against S62 `3cdcc0e` on the same RTX 3060 Laptop, CUDA 11.8,
sm86/MSVC 14.37 configuration. The retained S62 same-executable candidate
captures put horizontal33 at 11.532 / 13.915 ms in two 4K traces, 1.493 /
1.616 ms at 1080p, and 0.277 ms at Flower. Each encode has six horizontal
launches. These are identified retained captures, not newly acquired S63
baselines; S63 subsequently collects its own full-workflow comparisons.

### Paired outputs, fixed arithmetic, and the narrow boundary

The selected kernel gives each thread two adjacent columns and interleaves
their six channel accumulation chains. Each loaded horizontal input feeds
up to two outputs. Each output independently visits taps 0-32 in the original
FMA order and retains its original rounded division. An interior pair needs
34 source-level input loads per channel instead of 66. Clipped pairs keep
separate included-weight sums, in the original order, for both outputs.

The tile remains 256 columns by four rows, with 256 threads. All lanes load
the halo and reach the barrier before any out-of-image output is skipped.
The three shared halos still include zero-filled invalid entries. The
horizontal intermediates, vertical S62 kernel, allocations, transfers,
launch count, arena capacity, metadata/frame ABI, public API, and quality
policy are unchanged. This is not an image-plane traffic reduction.

The new body uses 48 registers versus 46 previously, the same 13,960 shared
bytes, and zero stack/local storage. The occupancy API reports five blocks
and 1,280 threads per SM for both bodies, out of a 1,536-thread maximum;
this is theoretical residency, not measured achieved occupancy. All 174
previous release GPU bodies remain instruction-identical. The sole new body
matches all four stage-screen binaries and the full-workflow control exactly.

Unconditional pairing regresses the 31-pixel-wide, 32,767-row fixture by
about 23-25% horizontally and 13-15% for two-pass preparation. A boundary
sweep exposes a sharp transition: in one packed 1,024-row cohort, widths
31 and 32 regress 28.43% and 23.78%, while width 33 improves 24.75%.
This is consistent with useful-lane coverage at the warp boundary: up to
32 columns fit in one unpaired warp, whereas pairing leaves roughly half
its lanes unused; the unpaired mapping needs another warp above 32.
That explanation is an inference from the mapping and measurements, not
a hardware-counter measurement of lane utilization.

The production rule therefore keeps the original horizontal body for a
logical plane width of at most 32 and uses pairs above 32. This is an
explicit, tested geometry dispatch, not a claim that one layout wins every
shape. The sequential conformance entry also retains the original horizontal
body. Both choices preserve one launch and the same tile grid. At narrow
widths the final diagnostic selects the identical original GPU body; its
small timing fluctuations are not attributed to an optimization. No graph,
diagnostic environment switch, zero-fill omission, or volatile memory access
enters production.

### Competing schedules and measurement controls

V1 compares baseline with twelve candidates: adjacent groups of two, three,
four, or five outputs; 96/192/256/320/384-column tiles; 2/4/8-row heights;
128/256/512 threads; and tap-major three/four-output controls that interleave
the sums without explicit input forwarding. All thirteen modes pass 380
fixtures with three reuse stages and two oracles: 29,640 comparisons.
The original separate-pass and previous resident paths compare all sixteen
allocations, including poisoned padding and read-only inputs/weights.

The first screen retains 8,112 three-launch event windows across packed and
padded 510x532, 960x540, 1920x1080, 3840x2160, 31x32767, and 16384x64, for
horizontal-only and complete two-pass preparation. Thirteen modes use 26
balanced forward/reverse cyclic orders after six warmup calls per mode.
Every allocation is compared bit-for-bit after every window, outside timing.
These Gaussian-weight synthetic fixtures are not captured psycho images.

V2 adds eight row-interleaved alternatives, keeping neighboring threads on
neighboring columns and sharing weight/address work across independent rows.
Rows per thread are 2/3/4, with varied tile heights and thread counts. Baseline
and the eight new modes pass 380 fixtures each: 20,520 reuse/oracle comparisons.
Its graph-based screen has 3,456 windows across twelve geometries/paddings,
twelve selected modes, and 24 balanced orders. Each submission has three
horizontal nodes. The row-interleaved variants use 53-64 registers and lower
theoretical residency; none supplies a broad improvement on ordinary shapes.
Some smaller tiles help the narrow fixture, but regress ordinary images.
More independent arithmetic alone is not sufficient here.

The unconditional two-column candidate gives these median paired changes.
The two-pass column includes horizontal and vertical preparation, not the
whole encoder. Negative means faster; different cohorts are not pooled.

| Geometry, packed | V1 horizontal | V1 two-pass preparation | V2 graph horizontal |
| --- | ---: | ---: | ---: |
| 510x532 | -30.52% | -18.05% | -40.25% |
| 960x540 | -26.26% | -21.23% | -43.50% |
| 1920x1080 | -48.27% | -23.34% | -50.32% |
| 3840x2160 | -50.17% | -20.67% | -50.34% |
| 31x32767 | +24.65% | +14.64% | +24.18% |
| 16384x64 | -41.72% | -18.49% | -42.33% |

V3 tests omitting shared-memory writes for invalid coordinates, in both the
paired and original bodies. Their read conditions exclude those entries.
Five modes each pass the original 380 fixtures plus 320 additional boundary
fixtures: 21,000 reuse/oracle comparisons. A 3,000-window graph screen adds
eighteen widths from 1 through 127 at height 1,024, packed/padded, to the
original horizontal/two-pass cases. Omitting writes improves paired narrow
31x32767 by only 4.22% relative to pairing, leaving an 18.73% regression
against baseline. Ordinary shapes do not consistently benefit. This
alternative is not selected. One 4K horizontal cohort has enough large
outliers to move the paired median sharply; those observations remain.

V4 adds a two-output tap-major control, a volatile-shared-memory version of
input forwarding, and the final narrow fallback. Five modes pass the same
700 fixtures: another 21,000 reuse/oracle comparisons. Its 3,000-window graph
screen repeats all sixty V3 cases. Graphs are diagnostic only; both complete
preparation kernels remain ordinary production launches.

| Final screen, packed horizontal | Unconditional pairs | Tap-major pairs | Volatile shared pairs | Narrow-fallback policy |
| --- | ---: | ---: | ---: | ---: |
| 510x532 | -39.10% | -31.50% | -23.76% | -38.65% |
| 960x540 | -43.10% | -40.20% | -29.18% | -44.43% |
| 1920x1080 | -46.99% | -45.22% | -33.04% | -47.59% |
| 3840x2160 | -50.93% | -49.68% | -35.14% | -52.91% |
| 31x32767 | +23.94% | +49.39% | +27.28% | -1.04% |
| 16384x64 | -44.67% | -45.28% | -29.79% | -43.13% |

The selected pair body has 162 static shared-load instructions, versus 258
for tap-major pairs and 213 for volatile shared pairs; all three have 435
static FFMA instructions. Counts include interior/edge paths, not executed
instructions per pixel. Pairing combines shared reads into 64-bit loads:
the selected body has 52 static LDS.64 instructions versus one in baseline.
The volatile control prevents that combining, but also restricts compiler
ordering; it is not a pure hardware bank-conflict experiment. Tap-major pairs
retain most of the ordinary-image gain despite more static loads. Adjacent
address layout, load combining, shared weight/address work, and multiple
accumulation chains therefore matter together; explicit forwarding or static
instruction count alone does not explain the result.

All 17,568 event windows and 92,160 guarded reuse/oracle comparisons remain,
including overlapping fixtures, outliers, and regressions. The guard count
is not a count of distinct images. The V3 probe initially fails compilation
because its optional mode parser lacks the string header; the failed source
and build log are retained, and the corrected build passes before measuring.

### Same-executable complete-workflow evidence

The full-workflow control retains all 174 S62 GPU bodies and adds the 24
candidate bodies; its diagnostic-only mode 25 implements the final dispatch.
All 46 baseline/candidate codestream pairs match frozen S62 hashes. These
are hash-only control runs, separate from the fresh release decode/metric
qualification. The selector preserves one launch-error check for either
horizontal body, avoiding an extra candidate-only CUDA API call.

Seven alternating pairs per workload use three warmups/seven samples for
warm runs and zero warmups/one sample for cold runs. All 41 phase fields,
arguments, binary hashes, and raw observations remain. The table gives
median paired percentage changes, not ratios of aggregate medians.

| Same-executable control | Quantization pipeline | Codestream encoding | Whole encode | Whole wins / 7 |
| --- | ---: | ---: | ---: | ---: |
| Warm 4K | -2.29% | -3.30% | -3.88% | 7 |
| Warm 1080p | -2.57% | -0.66% | -0.69% | 4 |
| Warm Flower | +0.98% | +3.10% | +3.22% | 3 |
| Cold 4K | -4.30% | -5.13% | -3.29% | 5 |
| Cold 1080p | +2.32% | +7.27% | +3.68% | 1 |
| Cold Flower | -0.45% | -1.99% | -0.82% | 4 |

Twelve fresh warm traces provide two alternating pairs per workload. Target
and total GPU durations are kept separate from unprofiled wall times. Every
trace retains six horizontal launches, with total counts 391 at 4K/Flower
and 378 at 1080p. The non-target kernel sequence is unchanged.

| Complete trace | Target GPU pair 1 / 2 | Total GPU pair 1 / 2 |
| --- | ---: | ---: |
| Warm 4K | -54.88% / -51.88% | -6.61% / -0.48% |
| Warm 1080p | -50.17% / -47.57% | -4.21% / -2.93% |
| Warm Flower | -39.90% / -39.81% | -1.89% / -1.88% |

Target GPU time improves in all six pairs. Target launch API time does not:
the first 4K pair rises 51.75%, and the second Flower pair rises 7.08%.
Warm Flower and cold 1080p whole-encode regressions remain in the unprofiled
record. These traces do not establish a universal encoder wall-time gain.

### Fresh release measurements and batch checks

The final cohorts compare retained S62 binaries against freshly linked S63,
with seven alternating pairs per workload. Warm runs use three warmups and
five samples; cold runs use zero warmups and one sample. The instrumented
phase probe retains all 41 fields and the unmodified public benchmark all
seven. Commands, hashes, samples, pair order, ranges, and size checks remain
in each report. Quality qualification is separate from these size checks.

| Fresh release cohort | Quantization pipeline | Codestream encoding | Whole encode | Whole wins / 7 |
| --- | ---: | ---: | ---: | ---: |
| Phase warm 4K | -0.22% | -3.39% | -1.48% | 4 |
| Phase warm 1080p | -2.59% | -3.90% | -2.75% | 5 |
| Phase warm Flower | -0.19% | -0.83% | +0.41% | 3 |
| Phase cold 4K | -1.17% | -3.29% | -0.56% | 6 |
| Phase cold 1080p | +0.06% | -0.40% | +0.14% | 3 |
| Phase cold Flower | +6.00% | +3.53% | +6.00% | 2 |
| Public warm 4K | -1.28% | +13.56% | +0.34% | 3 |
| Public warm 1080p | -2.49% | +1.12% | -0.32% | 4 |
| Public warm Flower | -1.61% | -3.67% | -3.13% | 4 |
| Public cold 4K | -0.49% | +0.41% | -0.39% | 4 |
| Public cold 1080p | +2.89% | +3.44% | +0.46% | 3 |
| Public cold Flower | -5.62% | -9.89% | -7.34% | 6 |

Public warm 4K codestream time regresses despite lower quantization time.
Phase cold Flower regresses while public cold Flower improves; neither
cohort replaces the other. Whole-encode pair ranges include +32.95% in
phase warm Flower and +29.77% in public cold 1080p. The targeted body has
strong local evidence, but these wall-time results do not establish a
uniform gain or isolate unrelated CPU-time changes as effects of the kernel.

Current-build batch checks use one warmup and three alternating samples,
requiring exact serial/batch codestream identity. These ratios compare S63
batch operation with S63 serial operation, not S62-versus-S63 throughput.

| Current S63 batch policy | Batch 1 | Batch 2 | Batch 4 |
| --- | ---: | ---: | ---: |
| 1080p fully-resident | 0.993x | 1.216x | 1.303x |
| 1080p maximum-throughput | 1.002x | 1.521x | 2.159x |
| 4K fully-resident | 1.018x | 1.006x | not run |

GPU-state snapshots bracket V1 at 57 C/P0/1,282 MHz/22.64 W to
65 C/P3/1,140 MHz/24.63 W; V2 at 59 C/P0/1,282 MHz/23.03 W to
63 C/P3/1,042 MHz/22.53 W; V3 at 57 C/P0/1,282 MHz/22.61 W to
61 C/P3/1,042 MHz/21.90 W; and V4 at 57 C/P0/1,282 MHz/22.64 W to
61 C/P3/1,162 MHz/23.67 W. Thermal/power flags are inactive at each
screen's start and active at its end. The complete-workflow control spans
61 C/P3/1,282 MHz/24.91 W to 65 C/P3/1,282 MHz/25.04 W with both flags
active. Final release performance begins at 60 C/P0/1,282 MHz/22.74 W
with both inactive and ends after batches at 67 C/P3/1,282 MHz/24.39 W
with both active. These are endpoint observations, not continuous evidence
attributing particular outliers to clocks or the OS. No clocks are normalized;
no power, security, service, priority, or clock setting is changed.

### Conformance, sanitizers, and retained evidence

The permanent low/medium test adds eight short/boundary geometries:
2x3, 3x4, 4x5, 16x3, 17x4, 18x5, 34x3, and 35x5. Its 460 fixtures,
three reuse stages, and both sequential-resident and separate-pass oracles
give 2,760 comparisons across all sixteen allocations. Inputs, weights,
poisoned padding, ignored planes, and the final null-unused-plane reuse
remain checked. The tall test retains 1x4194305 and adds padded 33x262145
so the selected paired horizontal body itself exceeds 65,535 tile rows.
Both tall cases pass. No test target or CMake entry is added.

All 72 CUDA and 50 CPU CTest tests pass, as do five host ASan targets:
DC grouping, entropy, coefficient order, codestream encoding, and public
workflow. Fresh release quality runs encode/decode/measure 58 pairs:
46 fully-resident, six high-density, and six maximum-compression cases.
Every parent/candidate codestream is byte-identical, with matching decoded
dimensions and Butteraugli scores; all size/score ratios are 1.0. The
independent decoder is pinned to `e8ff09762481785938d8e4e01333ed3917571161`.

Seven release GPU sanitizer runs pass. Low/medium memcheck, initcheck,
synccheck, and racecheck use 80 fixtures, now including widths 34 and 35,
with three reuse stages and both oracles. Full-AQ memcheck, initcheck,
and synccheck also pass. Both memchecks explicitly enable stream-order
race tracking and full leak checks, reporting zero errors and zero leaks.
Low/medium racecheck takes 108.661 seconds and reports zero hazards.
No full-AQ racecheck or prototype sanitizer campaign is claimed for S63.
No administrator prompt, firewall block, permission failure, or stalled
GPU process is encountered; the old suspected firewall cause remains
unconfirmed. The initial V1 stage screen takes about eleven minutes and
finishes normally with its repeated allocation checks.

The ignored `build-cuda-ninja/profiles/s63_*` bundle retains all four
prototype generations, the failed/corrected V3 build, raw screens, native
code/resources, occupancy reports, twelve workflow traces, control and
release timing/quality evidence, tests, sanitizers, and batch/state logs.
It retains 38 binaries/libraries and snapshots the changed CUDA source,
low/medium test, and two documentation files. `s63_validate.py --frozen`
checks exact source scope, unchanged native bodies, measured/release native
identity, raw windows and pair statistics, documented tables, trace mapping,
quality hashes, test/sanitizer outcomes, frozen artifacts, and unchanged
S59-S62 archives. Generated build artifacts are not committed.

S63 is retained for the repeated bit-exact horizontal-stage improvement
across competing schedules and complete-workflow traces, with a measured
narrow-width fallback. Stable-condition whole-workflow repeats, other GPU
architectures, initial color correlation, and the remaining dominant
resident stages are still open. Optimization is not maxed out.

## Prefetched initial color correlation (S64)

S64 compares against S63 `6f968df` on the same RTX 3060 Laptop, CUDA 11.8,
sm86/MSVC 14.37 configuration. Retained S63 candidate traces put initial CfL
at 0.486 ms in Flower, its largest single GPU kernel, and 0.565 ms at 1080p.
At 4K, Malta remains much larger: its two 64-row variants total about 38.9 ms.
These are identified retained captures, not fresh S64 baselines. Initial CfL
is a bounded target, not a claim to have removed the dominant 4K cost.

### Preserve the four chains while preloading samples

Each color tile still uses the original four accumulation lanes. A lane now
preloads eight samples of each channel into registers before accumulating
them in their original order. Full chunks span 32 columns; the remaining
samples use the original scalar schedule. Both the mean pass and the centered
regression pass preserve their four per-lane chains, the `(0+1)+(2+3)` sum,
division, thresholding, and signed-map quantization. Nonfinite samples retain
the original error-bit and untouched-output behavior. This is not a new
reduction tree or a reduced-precision approximation.

The selected block holds eight four-lane tiles in 32 threads, instead of
32 tiles in 128 threads. Register use rises from 42 to 48 per thread; there
is no shared, stack, or local storage. For that launch size, the occupancy
API reports sixteen blocks and 512 threads per SM, versus ten blocks and
1,280 threads for the original launch. These are theoretical limits, not
measured occupancy. The smaller blocks distribute the finite tile population
across more SMs; a higher theoretical thread ceiling alone does not predict
the measured result.

Logical image widths below 32 contain no complete eight-sample chunk. Their
measured small regressions are avoided with an explicit width dispatch to
the original kernel. Width 32 already uses the new body. Partial right-hand
color tiles in wider images use its scalar remainder. This is a geometry
dispatch, not a content or quality heuristic. Both the resident initial-CfL
entry and the DCT8 frame-encoding entry use the same dispatch; the original
body remains available through a private conformance entry.

The release adds one GPU body. All 175 previous bodies remain instruction-
identical, and the new body exactly matches V5, V6, and the complete-workflow
control. The number of launches, allocations, transfers, map layout, frame
ABI, public API, and quality policy do not change. Both passes still read
the same logical input samples; there is no image-plane traffic reduction.

### Competing schedules and the causal controls

V1 compares baseline with six four-lane block packings, from one through 64
tiles per block, and ten cooperative-load variants using 8/16/32 lanes per
tile. Cooperative variants load adjacent samples and broadcast them into
replicas of the original four chains. The 4,624 ordinary-launch event windows
cover eight synthetic geometries, seventeen modes, and 34 balanced forward/
reverse orders. Each window contains three launches, after six warmup calls
per mode. Full-warp loading helps small images but strongly regresses 4K;
smaller blocks alone are also not a universal improvement.

Three diagnostic captures retain the actual post-preprocessing CfL planes
from fully-resident 4K, 1080p, and Flower encodes. All 175 GPU bodies in the
capture executable match S63. Synchronized capture copies make these input
captures unsuitable for wall-time comparisons. They are replay data, not
performance observations.

V2 adds two-, four-, and eight-sample vector loads within the four-lane group,
with subgroup register transposes, at three block packings each. Its 3,168
graph windows cover eight synthetic and three captured inputs, twelve modes,
and 24 balanced orders. Graph submissions contain three kernel nodes and
are diagnostic only. Representative captured-input median paired changes
are below; negative means faster.

| V2 captured input | Full warp, two tiles/block | Vector 2, eight tiles/block | Vector 4, eight tiles/block | Vector 8, eight tiles/block |
| --- | ---: | ---: | ---: | ---: |
| Flower | -56.92% | +185.82% | +61.60% | -4.72% |
| 1080p | -48.84% | +167.61% | +49.73% | -13.92% |
| 4K | +56.36% | +86.60% | +19.83% | -21.87% |

V3 adds sixteen-sample vector loads and scalar preload/accumulate controls
with 4/8/16 samples per lane and 8/16/32 tiles per block. The scalar controls
load a chunk before updating the sums but do not use vector loads or input
shuffles. The 7,128 graph windows cover eleven input cases, eighteen modes,
and 36 balanced orders. Scalar preloading wins broadly, showing that vector
transactions or transpose-based coalescing are not necessary for the gain.

| V3 captured input, eight tiles/block | Vector 16 | Scalar preload 4 | Scalar preload 8 | Scalar preload 16 |
| --- | ---: | ---: | ---: | ---: |
| Flower | -28.36% | -22.48% | -36.00% | -19.63% |
| 1080p | -31.93% | -21.90% | -32.81% | -20.53% |
| 4K | -11.61% | -23.91% | -27.12% | -18.72% |

V4 compares that explicit preloading with plain compiler unroll factors
4/8/16. The latter all regress: roughly 100-114% on captured 1080p/Flower
and 33-39% at 4K. A 60-geometry small-width/height sweep also exposes waste
in incomplete preload chunks. V4 retains 6,762 graph windows across 69
synthetic/captured cases, seven modes, and fourteen balanced orders.

V5 separates complete chunks from scalar remainders. Four new variants use
4/8 samples and 8/16 tiles per block, in another 6,762-window graph screen.
Eight samples/eight tiles is selected. V6 repeats the 69-case screen with
ordinary launches and the final below-32 fallback: another 6,762 windows.
It keeps the full-warp alternative and earlier controls for comparison.

| Captured input | V5 chunk/remainder, graph | V6 final dispatch, ordinary launches | V6 wins / 14 |
| --- | ---: | ---: | ---: |
| Flower | -42.97% | -41.56% | 14 |
| 1080p | -41.07% | -39.69% | 14 |
| 4K | -24.14% | -24.59% | 14 |

The full-chunk body uses ordinary scalar global loads. Its 75 static load
instructions and 83 static FFMAs include both fast and remainder paths;
they are not executed counts per sample. Its 48 registers are fewer than
the earlier guarded eight-sample preloader's 56. Plain unrolling uses only
34 registers but is much slower. The evidence supports batching independent
loads before dependent arithmetic, reducing chunk-control work, and changing
block packing together. It does not establish a hardware-counter claim about
cache misses, memory throughput, achieved occupancy, or a single instruction
as the cause. All 35,206 timing windows, including regressions, remain.

### Diagnostic failures and stronger arithmetic checks

The first cooperative prototype stalls at the partial-width case after
64x56. Its shuffle mask names the whole active warp inside loops whose trip
counts differ between tiles. A live process check finds GPU utilization at
100%; the user is notified, and only the identified probe process is stopped.
Restricting the mask to each tile's lanes lets the unchanged guard sequence
finish. The stalled source, binary, build log, and partial guard log remain;
that run is not counted as a completed experiment. No administrator prompt,
firewall block, or permission failure is observed.

The first floating audit also exposes a harness ordering error: an output
reset can finish after a kernel on the non-blocking stream, leaving unchanged
poison rather than a computed sum. Both failed audit generations remain.
An explicit default-stream wait before launching fixes the audit. CUDA 11.8
documents that a pageable host-to-device copy may return before its device
transfer finishes. The V3 and later fixture resets explicitly wait for that
copy. [CUDA synchronization behavior](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-runtime-api/api-sync-behavior.html)

V1/V2 map-only guard runs retain that setup-ordering limitation; they are not
the final conformance evidence. The corrected V3 floating audit rechecks all
37 candidates, not just its screen subset, over 870 synthetic reuse cases
and nine captured-input reuse cases: 32,523 comparisons. All 24 per-tile
floating sums, maps, error bits, and guards match bit-for-bit, and actual
non-audit kernel outputs are checked separately. V4 and V5 audits each add
5,274 comparisons for their six selected/control modes. These counts include
overlapping fixtures, not distinct images. V6 additionally checks output
guards independently and checks error bits against the host's input-finiteness
expectation. Read-only inputs and their guards remain intact.

A mechanically malformed V2 report parser initially fails with SyntaxError;
its source is preserved, and the correction reads the same raw measurements.
No screen is restarted to replace unfavorable observations. The earlier S61
fixture/audit source is checked read-only and already orders its resets;
the old archives are not changed by this S64 harness correction.

### Complete-workflow controls

The same-executable control retains all 175 S63 GPU bodies and adds the 44
experimental bodies. Diagnostic mode 45 selects the final dispatch. All 46
baseline/candidate codestream pairs match frozen S63 hashes. These are
hash-only control checks, separate from fresh release decodes and metrics.

Seven alternating pairs per workload use three warmups/seven samples when
warm, and zero warmups/one sample when cold. All 41 phase fields and raw
commands remain. These are median paired percentage changes, not ratios of
aggregate medians.

| Same-executable S64 control | Quantization pipeline | Codestream encoding | Whole encode | Whole wins / 7 |
| --- | ---: | ---: | ---: | ---: |
| Warm 4K | -1.74% | -5.85% | -1.02% | 6 |
| Warm 1080p | -2.36% | -5.42% | -3.33% | 4 |
| Warm Flower | -2.80% | -2.15% | -1.87% | 5 |
| Cold 4K | -0.40% | +3.42% | +0.88% | 3 |
| Cold 1080p | +1.56% | -2.35% | -0.32% | 4 |
| Cold Flower | -6.34% | +4.94% | -2.45% | 5 |

Twelve warm traces provide two alternating pairs per workload. Every trace
has one initial-CfL launch and the same non-target kernel sequence. Total
launch counts remain 391 at 4K/Flower and 378 at 1080p.

| S64 complete trace | Target GPU pair 1 / 2 | Total GPU pair 1 / 2 |
| --- | ---: | ---: |
| Warm 4K | -45.30% / -44.18% | -8.03% / -2.08% |
| Warm 1080p | -43.13% / -44.53% | -1.30% / -4.68% |
| Warm Flower | -44.07% / -43.53% | -3.77% / -3.44% |

All six target-duration pairs improve, but target launch API duration rises
14.50% and 89.93% in the two 1080p pairs. Total-GPU changes can exceed the
saved CfL duration, so they cannot all be attributed to this one kernel.
Instrumented durations are not substituted for unprofiled wall times; cold
4K whole-encode and other slower observations remain in the record.

### Release correctness and sanitizer observability

All 73 CUDA and 50 CPU CTest targets pass, as do five host ASan targets
(DC group, entropy, coefficient order, codestream encoder, and public
codestream workflow). The new permanent initial-CfL test compares the original
and dispatched kernels over 310 fixtures with three-stage reuse: 930 map,
error, and guard comparisons. Its 31 geometries cover short widths around
the dispatch boundary, partial color tiles, tile counts around block/warp
boundaries, and tall/narrow and wide/short images. Ten patterns include
structured/noisy data, constant values, signed zero, denormals, cancellation,
large finite values, NaNs, and infinities. Reset copies explicitly finish
before work on the non-blocking stream; error bits are checked against the
host's input-finiteness expectation, not only against the GPU reference.

The scoped sanitizer test uses eight boundary geometries and four patterns,
with 32 fixtures and 96 comparisons per tool. Memcheck (including stream-
ordered races and full leak checking), initcheck, synccheck, and racecheck
all pass. Full AQ separately passes memcheck/initcheck/synccheck, with zero
errors and zero leaked bytes where checked. Full-AQ racecheck is not rerun
or claimed; the earlier aborted long run remains unqualified.

The first new-test sanitizer invocation prints a zero-error summary but no
application-completion marker, so it is not accepted as a pass. Direct runs
do print the marker, and a session-details retry identifies the CUDA target.
Existing sanitizer tests explicitly flush their final messages. Adding that
flush to this test restores its 96-comparison marker under instrumentation;
a fresh seven-check campaign then asserts both tool success and application
completion. The unflushed source/binary and unsuccessful diagnostic wrapper
attempts remain. This is evidence of buffered output in this invocation, not
of a firewall block. No administrator, firewall, or permission change is made.
Only the host test is rebuilt for the flush; its CTest is rerun and passes.

The fresh release quality campaign encodes, decodes, and measures 58 parent/
candidate pairs: 46 core cases and six each in high-density and maximum-
compression modes. All codestream bytes, decoded dimensions, and measured
Butteraugli scores match; size and score ratios are 1. The pinned decoder
revision is `e8ff09762481785938d8e4e01333ed3917571161`. These fresh checks are
separate from the 46 hash-only same-executable controls.

### Fresh release timings and remaining limits

Fresh release executables are compared with retained S63 executables in four
separate seven-pair cohorts. Warm uses three warmups/five samples; cold uses
zero warmups/one sample. Phase builds retain all 41 fields and public builds
all seven. Executable hashes, exact commands, raw outputs, alternating order,
pair ranges, and win counts remain. Encoded-size checks in timing runs are
not substituted for the fresh quality campaign above.

| Fresh release cohort | Quantization pipeline | Codestream encoding | Whole encode | Whole wins / 7 |
| --- | ---: | ---: | ---: | ---: |
| Phase warm 4K | -1.93% | -3.77% | -2.80% | 5 |
| Phase warm 1080p | -0.54% | -7.66% | -2.14% | 4 |
| Phase warm Flower | +0.91% | +5.53% | +2.81% | 3 |
| Phase cold 4K | -0.11% | +0.12% | +0.05% | 3 |
| Phase cold 1080p | -2.88% | -3.06% | -2.26% | 5 |
| Phase cold Flower | +1.47% | -0.44% | -2.15% | 4 |
| Public warm 4K | -4.81% | -19.02% | -10.00% | 5 |
| Public warm 1080p | -1.60% | +2.01% | -0.11% | 4 |
| Public warm Flower | -3.46% | +4.80% | +3.14% | 3 |
| Public cold 4K | -7.22% | -27.26% | -14.70% | 5 |
| Public cold 1080p | +1.10% | +16.67% | +7.18% | 3 |
| Public cold Flower | -2.07% | -12.60% | -5.60% | 4 |

The public warm 4K whole-encode range is -18.85% to +3.35%; warm 1080p is
-11.92% to +68.51%, and warm Flower is -33.11% to +70.92%. Cold public ranges
are -21.73% to +20.17%, -5.76% to +21.53%, and -25.16% to +33.19%.
The host codestream phase is unchanged by this GPU implementation yet also
moves substantially. The large 4K public median gains cannot reasonably be
assigned entirely to a sub-millisecond CfL saving. These observations are
retained without filtering or rerunning a cohort to obtain a favorable
result. The much tighter target-kernel controls support the local change;
whole-encoder effects remain uncertain and workload/state dependent.

The current batch policy is checked with one warmup and three alternating
serial/batch pairs. Values below are median paired serial/batch ratios, not
S63-to-S64 speedups. Each output is checked against its serial reference.

| Current S64 batch policy | Batch 1 | Batch 2 | Batch 4 |
| --- | ---: | ---: | ---: |
| 1080p fully-resident | 0.988x | 1.168x | 1.733x |
| 1080p maximum-throughput | 1.169x | 1.503x | 1.805x |
| 4K fully-resident | 1.046x | 1.190x | not run |

Batch ratios also vary widely: 1080p resident batch four ranges from 1.117x
to 1.998x. Its median milliseconds per image is worse than batch two in this
cohort; the paired ratio does not establish improved absolute throughput.
4K batch four is not run. No lane-count or concurrency policy changes.

The final performance endpoints are 60 C/P0/1,282 MHz/25.19 W and
74 C/P0/1,762 MHz/44.79 W, both reporting neither thermal nor power limiting.
Earlier stage-screen endpoints differ: V1 reports both limits at each end;
V2-V6 end with both active, as does the complete-workflow control. V6 moves
from 58 C/P0/1,282 MHz/22.68 W to 60 C/P3/1,605 MHz/33.61 W; the control moves
from 57 C/P8/210 MHz/15.77 W to 65 C/P3/1,282 MHz/24.99 W. All raw snapshots
remain. Endpoint samples neither characterize every timed interval nor
identify the cause of host timing scatter. No clock, power, process-priority,
security, or service settings are changed, and no clock normalization is
applied. Different cohorts are not pooled.

Reproduction is under ignored `build-cuda-ninja/profiles/s64_*`: candidate
sources/binaries, all six raw screens, arithmetic audits, native bodies and
resources, input captures, twelve workflow traces/SQLite exports, quality
artifacts, sanitizer reports, release timings, and validators. The checkpoint
retains 39 release binaries/libraries and six source/document snapshots.
The validator recomputes timing statistics, native-body equality, trace
launch/runtime totals, decoded dimensions and hashes, documentation tables,
and the frozen manifests; S59-S63 evidence remains unchanged.

S64 is a precise initial-CfL improvement, not a maxed-out backend. The much
larger 4K Malta work, remaining resident reconstruction/quantization costs,
and host serialization remain relevant. Further changes still need explicit
causal controls and whole-workflow qualification; more streams or reduced
precision are not justified by this experiment.


## Exact zero-tile Malta responses (S65)

S65 compares against S64 `73179fd` on the same RTX 3060 Laptop,
CUDA 11.8/sm86/MSVC 14.37 configuration. Identified retained S64 candidate
traces put 4K Malta at about 41.75 ms across its full/LF variants, against
about 190 ms total GPU work. These are not freshly measured S65 baselines.

### Skip a provably zero response, not its output operation

Each block still scales its original 32-column tile and four-pixel halo.
While doing so, each thread records whether any scaled value compares
nonzero. A block-wide OR replaces the existing barrier. If every scaled
halo value is signed zero, all sixteen directional sums of squares produce
positive zero, so the response calculation can be omitted. Initialization
still writes that positive zero; accumulation still executes the original
addition. Skipping the addition would incorrectly preserve negative zero or
change exceptional accumulator behavior.

NaNs and infinities compare nonzero and retain the original response path.
The decision is made after the original scaling arithmetic, not from a
threshold, channel number, approximate difference, or unscaled input test.
Both correctly rounded divisions and all nonzero-response arithmetic remain.
Every thread participates in the cooperative load and collective before any
partial-tile exit. The existing adaptive heights 8/24/64, 256 threads,
2D/flat-grid boundary, halo, strides, and initialization policy are unchanged.

Native instructions confirm a uniform branch before the first response
shared-memory load; the compiler really skips response arithmetic on the
zero path. In the 64-row LF body the branch is at 0x7f0, preceding the first
response LDS at 0x800. The collective is BAR.RED.OR, not an extra launch.
Shared storage is unchanged; the 64-row LF register count falls from 38 to
36. All twelve new specializations have zero stack/local storage, exactly
match the guard/replay/sanitizer/workflow prototypes, and leave all 176
previous GPU bodies instruction-identical. The old fused response and
separate-pass oracle remain accessible to private conformance tests.
Allocations, transfers, launches, public API, frame ABI, and quality policy
do not change.

### Wider tiles and raw-zero detection were competing experiments

The initial sweep tests 27 layouts plus the original dispatch: widths
16/32/64/128, heights 8 through 128, and 128/256/512-thread variants.
It covers 18,816 balanced timing windows over twelve synthetic cases, each
with exact three-stage output and guards. All twelve width-32 controls
are instruction-identical to the original height/grid/frequency variants.
The independent guard sweep passes 5,184 fixtures/15,552 comparisons.

Eighteen real stage captures then replay twelve selected layouts in 5,184
balanced windows. Width 64/height 24 improves the six 4K stages by
0.40-2.08% and HD by 0.89-4.97%, but regresses Flower by 0.41-3.31%.
Larger tiles are not consistently better: width 128/height 64 regresses 4K
by 21.35-32.93%. The occupancy API reports six resident blocks for the
original 256-thread layouts, four for width 64/height 64, and two for width
128/height 64, limited by shared storage. These are theoretical ceilings,
not achieved occupancy measurements.

The captures expose substantial exact-zero populations after psycho-plane
range removal. MF-x is entirely zero in the captured first evaluation of
4K, HD, Flower, and three 500x500 corpus photographs. UHF-x is entirely zero
in 4K/HD, 95.91% zero in Flower, and 97.36-99.95% zero in those photographs.
Those are observations of content, not structural guarantees about channels;
the production collective checks every actual scaled tile and halo.

Four zero-aware alternatives are tested: scaled-zero detection with the
original policy, raw-zero detection with two shared tiles and a second
barrier on nonzero blocks, and scaled-zero detection with fixed width
64/height 24 or 32. The raw variant preserves exceptional scaling parameters
but substantially penalizes dense stages. It and both wider policies are
rejected. The selected mode keeps the original geometry.

| Selected mode: range over six captured stage medians | Paired time change |
| --- | ---: |
| 4K | -9.62% to +0.45% |
| 1080p | -12.61% to +0.53% |
| Flower | -38.49% to +1.06% |
| Keong macan | -54.35% to +2.39% |
| Riaphotographs | -54.02% to +2.54% |
| Bliznaca | -50.94% to +5.98% |

The selected corpus MF-x/UHF-x stages win all ten paired rotations each,
but dense-stage regressions remain, including +5.98% on one Bliznaca stage.
These two zero-replay cohorts each contain 900 exact-output timing windows;
all four alternatives also pass 3,200 guarded fixtures/9,600 comparisons.
Across wide and zero replays, 25,800 timing windows are retained. Replay
packs the first full-resolution six-stage evaluation and repeats three
launches; it does not reproduce every iteration, subscale, workflow stride,
or launch dependency. Independent stage medians are not summed into a claimed
whole-encode speedup. Host FP32 population counts are not a CPU output oracle.

### Full-workflow controls retain the adverse observations

The same-binary control changes only the Malta dispatch. All 46 encoded
pairs for each of modes 1 and 3 match frozen S64 hashes. This control gate
reuses S64 decoded evidence; it is distinct from fresh release decoding.
The first six-permutation wall cohort records selected-mode whole changes
of -0.76% / -0.39% / +4.49% at 4K / HD / Flower. Flower loses all six pairs,
with a +0.23% to +45.66% range. That cohort is retained, not replaced.

The subsequent alternating seven-pair controls use three warmups/seven
samples, or zero warmups/one cold sample:

| S65 same-binary control | Quantization pipeline | Codestream encoding | Whole encode | Whole wins / 7 |
| --- | ---: | ---: | ---: | ---: |
| Warm 4K | -2.54% | -4.36% | -2.12% | 7 |
| Warm 1080p | -0.67% | -2.31% | -1.26% | 6 |
| Warm Flower | +1.30% | +2.04% | +1.61% | 2 |
| Cold 4K | -1.20% | -5.79% | -2.26% | 5 |
| Cold 1080p | +4.05% | -1.74% | +3.06% | 2 |
| Cold Flower | -0.44% | +1.95% | -0.03% | 4 |
| Warm Keong macan | +1.96% | -0.44% | +0.02% | 3 |
| Warm Riaphotographs | -1.61% | -1.52% | -0.07% | 4 |
| Warm Bliznaca | -0.22% | -0.98% | -1.37% | 4 |

Two alternating trace pairs per workload cover both selected and wider-zero
controls: 24 traces total. Each retains 24 Malta launches and the complete
untargeted kernel sequence; total launch counts are 391 at 4K/Flower and
378 at HD. Selected-mode target/total GPU time changes are:

| S65 complete trace | Target GPU pair 1 / 2 | Total GPU pair 1 / 2 |
| --- | ---: | ---: |
| 4K | -20.93% / +0.30% | -7.12% / +13.57% |
| 1080p | -15.95% / -11.19% | -2.51% / -1.28% |
| Flower | -8.42% / -8.71% | -0.95% / -1.08% |

The slower second 4K pair is not excluded. Target launch API time also
scatters, including +14.39% in one HD pair. The width-64/height-24 zero
control improves 4K target time 18.87-20.79% and HD 12.28-18.40%, but
regresses Flower target time 2.34-2.47% in both pairs. This supports retaining
the original geometry, not a claim that the selected mode wins every stage
or workflow. Total-GPU and host-serialization changes cannot all be assigned
to skipped Malta arithmetic. Instrumented traces are not release wall times.

### Qualification and retained evidence

All 73 CUDA tests, 50 CPU tests, and five host ASan targets pass. The permanent
Malta suite now checks 1,856 guarded fixtures/5,568 three-stage comparisons,
including zero/denormal tiles, sparse halo-edge values, isolated NaNs,
negative-zero and exceptional accumulators, and zero normalization parameters.
It retains the old fused oracle checks. Thirty-two tall fixtures cover both
sides of the old and current flat-grid boundaries, including all-zero input.

Four release Malta sanitizer tools each pass 168 fixtures/504 comparisons;
full AQ passes memcheck/initcheck/synccheck. The independent four-tool
prototype campaign covers 320 fixtures/960 comparisons per tool. Memcheck
explicitly enables stream-ordered race and leak checking; racechecks report
zero hazards. No full-AQ racecheck is run. All 58 newly encoded parent/candidate
pairs are byte-identical, freshly decoded at the correct dimensions, and have
identical independent Butteraugli scores, including the extra entropy modes.

The first wide diagnostic failed to compile because of a missing string
include and ambiguous initializer lists; corrected sources and all failed
logs remain. An unused renamed synthetic entry in the wide replay retains a
missing-return warning but is never called. A release-runner label replacement
mistakenly requested nonexistent vcvars65.bat after the CUDA/CPU suites had
passed. Its failed script/log remain; the corrected runner resumed at ASan
without overwriting those results. These are harness issues, not GPU faults.
Native extraction and racecheck delays completed successfully. The user was
notified during unexpectedly slow diagnostics; no firewall or permission block
was confirmed and no OS security settings were changed.

An initial evidence-parser assertion expected the older CTest success
wording. The retained log actually says "100% tests passed out of 73".
The corrected parser accepts both success formats and independently counts
all 73 passed rows; no functional test was rerun or relabeled. Its failed
validator snapshot and first pre-freeze report remain.

### Release wall and batch measurements

These are separate parent/release executables, not the dispatch-control
binary. Each cohort has seven alternating pairs per workload, three warmups
and five samples for warm runs, zero warmups and one sample for cold runs.
All raw fields remain: 41 in the diagnostic phase probe, seven in the public
benchmark. Negative percentages mean less elapsed time. Rows are medians of
paired ratios, not ratios of independent medians.

| S65 release comparison | Quantization pipeline | Codestream encoding | Whole encode | Whole wins / 7 |
| --- | ---: | ---: | ---: | ---: |
| Phase warm 4K | -3.02% | -2.45% | -2.27% | 5 |
| Phase warm 1080p | -1.02% | +3.40% | +0.56% | 2 |
| Phase warm Flower | -3.91% | -3.34% | -5.48% | 5 |
| Phase cold 4K | -2.26% | -2.02% | -2.37% | 4 |
| Phase cold 1080p | -3.44% | -4.35% | -1.81% | 5 |
| Phase cold Flower | -2.09% | +1.18% | -0.65% | 4 |
| Public warm 4K | -1.85% | -6.53% | -1.49% | 5 |
| Public warm 1080p | -1.75% | +0.88% | +0.40% | 3 |
| Public warm Flower | -3.40% | +1.37% | -1.36% | 4 |
| Public cold 4K | -2.97% | -0.16% | -1.09% | 4 |
| Public cold 1080p | +0.23% | +0.48% | +0.14% | 2 |
| Public cold Flower | -1.06% | +1.33% | -0.57% | 4 |

Whole-time ranges remain broad. Warm public 4K spans -7.22% to +4.79%,
HD -10.84% to +2.98%, and Flower -30.21% to +7.78%. Cold Flower spans
-57.94% to +20.54%. The unchanged host codestream phase moves by -6.53%
in warm public 4K and +3.40% in warm phase HD. The local instruction/replay
evidence supports skipped work; it does not explain every host timing change
or establish significant end-to-end improvements in these noisy cohorts.

The current batch policy uses one warmup and three alternating serial/batch
pairs, checking each output against its serial reference. These ratios are
within S65, not S64-to-S65 speedups:

| Current S65 batch policy | Batch 1 | Batch 2 | Batch 4 |
| --- | ---: | ---: | ---: |
| 1080p fully-resident | 1.048x | 1.192x | 1.284x |
| 1080p maximum-throughput | 0.823x | 1.495x | 1.886x |
| 4K fully-resident | 1.010x | 0.961x | not run |

4K batch two loses all three pairs (0.934-0.999x), with 425.03 ms per image
versus 387.89 ms at batch one. Maximum-throughput batch one also loses all
three pairs. 4K batch four is not run, and no lane-count policy changes.

Release timing endpoints are 68 C/P3/1,732 MHz/30.41 W and
70 C/P3/240 MHz/22.68 W, both reporting thermal and power limiting. Earlier
wide replay ends at 67 C/P3/1,777 MHz/35.93 W with both active. The selected
wall control moves from 55 C/P0/1,282 MHz/6,000 MHz memory/22.39 W with
neither limit to 64 C/P3/1,282 MHz/5,500 MHz memory/28.15 W with both active.
These are endpoint snapshots, not continuous samples or explanations of
individual outliers. No clock normalization or system-setting changes are
made, and separate cohorts are not pooled.

Reproduction and retained failures are under ignored
`build-cuda-ninja/profiles/s65_*`: candidate sources/binaries, raw screens,
36 stage captures, native instructions/resources, all 24 trace/SQLite pairs,
quality artifacts, sanitizer reports, release timings, and validators.
The checkpoint retains 39 release binaries/libraries and five source/document
snapshots. The validator recomputes source/native equivalence, every timing
rotation/statistic, trace sequences and GPU/API totals, decoded dimensions
and hashes, documentation tables, and frozen manifests. S59-S64 archived
artifacts remain unchanged.

The backend is not maxed out. Nonzero Malta still scales the full halo and
executes the original divisions and directional response; reconstruction,
quantization, other psycho stages, and host serialization remain material.
The rejected wider/raw-zero schedules constrain the next investigation.
Neither more streams nor approximate zero thresholds follow from this result.


## Shared Malta reciprocal investigation (S66, not retained)

S66 investigates S65 `7160cc4` on the same RTX 3060 Laptop, CUDA 11.8,
sm_86, and MSVC 14.37. Production source, tests, and all 39 retained release
binaries/libraries remain unchanged. This is a closed experiment, not a
released optimization or evidence that the backend is maxed out.

### Confirm the repeated division work

The identified S65 selected-mode traces still put Malta first at 4K:
the medians across two captures total 21.52 ms for low-frequency and
15.53 ms for full response. Vertical low-medium construction follows at
11.55 ms and erosion/L2/final masking at 10.08 ms. These are retained
instrumented S65 captures, not newly acquired S66 baseline traces.

Native instructions confirm that the two divisions in `MaltaScaleValue`
independently compute and refine a reciprocal of the same denominator.
The compiler has not already shared that work. The candidate changes only
this two-quotient calculation inside the S65 zero-aware kernel; halo loads,
the exact-zero vote, response arithmetic, tile policy, and accumulation stay
the same. The original helper and production dispatch are untouched.

Mode 1 shares an approximate reciprocal followed by Newton refinement,
mirroring the observed sm_86 fast path. Mode 2 shares `__frcp_rn`, then
uses one rounded multiplication and two explicit fused multiply-adds per
quotient. Mode 3 is a separately named copy of the unchanged kernel.
Mode 0 calls current production. Native inspection proves all twelve mode-3
tile/frequency/grid bodies instruction-identical to production. Both
guarded and replay executables contain the same 36 added bodies and all
60 original Butteraugli bodies unchanged. These diagnostic executables do
not contain the entire 188-body release program.

Both candidate recipes require positive numerators in [2^-30, 2^30] and
a positive denominator in [2^-60, 2^60], inclusive. All other cases execute
the two original rounded divisions, including zero, negative, subnormal,
nonfinite, and out-of-range operands. A range audit of all 36 retained
captures proves that all 68,281,932 in-bounds input pixels take the candidate
range, so this screen is not merely timing the fallback.

Correctly rounding the reciprocal alone does not make multiplication an
exact replacement for division. The mode-2 multiply/residual/correction
sequence is Algorithm 1 of Brisebarre, Muller, and Raina, whose Theorem 1
covers round-to-nearest when overflow and underflow do not occur. The broad
normal-range guard is intended to keep those intermediate results in range;
it is not a substitute for the theorem. That result is not asserted for
mode 1's approximate reciprocal. Billions of passing samples are useful
evidence, not a general proof for that recipe or a second GPU architecture.
See the authors' [division paper](https://perso.ens-lyon.fr/jean-michel.muller/DivIEEETC-aug04.pdf)
and NVIDIA's [CUDA 11.8 arithmetic intrinsic contracts](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html).

### Arithmetic and captured-input evidence

The independent arithmetic probe compares both quotients from both recipes
bit-for-bit against `__fdiv_rn`. The quick run completes 21,757,952 comparisons;
the full run completes 2,785,017,856, with zero mismatches in either recipe.
Full coverage includes eight seeds each of bounded positive triples,
arbitrary FP32 patterns, and binade-adjacent values, plus every 23-bit divisor
mantissa at seven exponents for five fixed numerator pairs. These counts are
comparisons, not unique triples or exhaustive coverage of the entire guarded
domain. Arbitrary patterns also exercise the fallback.

The Malta probe passes 2,880 guarded fixtures and 8,640 three-stage output
comparisons across the three diagnostic modes, ten geometries, both grid
forms, frequency modes, initialization modes, and twelve input patterns.
The source-level reference is the original separate scaling/response path;
guards, padded strides, input immutability, signed zeros, exceptional values,
sparse tiles, and zero normalization remain checked bit-for-bit.

The replay uses the unchanged S65 first full-resolution six-stage captures
for 4K, HD, Flower, Keong macan, Riaphotographs, and Bliznaca. It does not
recapture S66 inputs or cover every iteration and subscale. Each of 36 stages
has eight balanced forward/reverse four-mode orders, three launches per event
window, explicit output resets and stream ordering, and exact output checks
after every window. All 1,152 event windows pass. There are no omitted rounds.

The following values compare sums of six independently measured stage
medians, not paired whole-workflow durations. Negative means less time.
These descriptive aggregates must not be interpreted as encode speedups.

| S66 captured stage aggregate | Mode 1, refined approximate | Mode 2, rounded reciprocal | Mode 3, identical control |
| --- | ---: | ---: | ---: |
| 4K | -0.42% | +0.01% | -0.09% |
| 1080p | -0.98% | -0.21% | +0.03% |
| Flower | -1.53% | -1.05% | +0.85% |
| Keong macan | -2.19% | -0.31% | +0.21% |
| Riaphotographs | -3.36% | +0.57% | -1.72% |
| Bliznaca | +0.33% | +0.17% | -0.33% |

Against production, mode 1's six 4K stage paired medians improve only
0.32-0.76%; the identical control itself ranges from -0.50% to +0.05%.
Against that duplicate, mode 2 is slower in all six 4K stage paired medians,
by 0.19-0.67%. At smaller sizes the scatter grows: Flower's identical-control
stage medians range from -0.53% to +7.65%, and the Riaphotographs mode-2
last stage regresses 17.35%. These adverse results are retained.

For height 64, all three compiled forms use 39 registers for full response
and 36 for low frequency, with 11,520 shared bytes and no local/stack storage.
Full-response static instruction counts are 424 / 536 / 600 for unchanged /
mode 1 / mode 2; low-frequency counts are 400 / 520 / 584. These include
fallbacks, helper code and padding, not dynamically executed instruction
counts. The candidates reduce reciprocal setup on the guarded path but add
range checks and compiled exceptional paths; no occupancy reduction or
specific timing attribution follows from static counts alone.

### Disposition and next work

Neither variant clears the local cost/benefit gate. Mode 2 has no useful 4K
gain; mode 1 adds numerical proof obligations for a small and inconsistent
benefit. Neither enters a whole-workflow control, production dispatch, or
release build. No fresh release tests, decoded-quality campaign, sanitizer
campaign, batch cohort, or whole-encode speedup is claimed for S66. The
unchanged S65 implementation retains its previously recorded qualification.

Replay endpoints are 57 C and 59 C, both P3, 1,282 MHz SM, 5,500 MHz memory,
21.73 W and 23.06 W, with thermal and power limiting reported at both ends.
These are snapshots, not continuous clocks or an explanation for individual
outliers. No OS, security, power, clock, or lane-count settings change.
Native extraction completes; no firewall or permission block is confirmed.
An initial orchestration string interpolation failed before compilation;
the corrected harness subsequently built and ran successfully. No GPU test
failure is suppressed or relabeled.

The first evidence validator used Python's compensated `sum` for baseline
totals originally accumulated with ordered `+=`, producing a 3.6e-15 ms
last-bit difference. Its failed source and log remain; the corrected check
reproduces the original accumulation order without changing raw timings or
relaxing equality. No GPU test is rerun or reclassified for this parser fix.

Ignored `build-cuda-ninja/profiles/s66_*` retains the arithmetic and Malta
sources/binaries, build logs, native dumps/resources, all replay records,
range audit, summary scripts, and a recomputing validator. A frozen manifest
covers these artifacts and five source/document snapshots. The validator
also checks all 39 S65 release binaries and the S59-S65 archived manifests.
The division build runner is a clean-directory reproduction recipe added
after the measured builds; it is not a second recorded build or rerun.

The next investigation is reference-only work: `LaunchDifference` reuses the
main blurred reference mask but recomputes its erosion and both nonlinear
mask values for every comparison. Half-scale comparisons also rebuild their
reference mask. Caching could remove repeated work, but must count extra
device storage, first-encode preparation, reuse count, multiscale layout, and
failure-atomic resource planning before retention. This is a measured next
target, not an implemented or qualified cache.


## Reuse dead prepared Butteraugli planes (S67)

S67 starts from S66 `1e3944b`, whose production implementation and retained
binaries are S65 `7160cc4`. This is the same RTX 3060 Laptop, CUDA 11.8,
MSVC 14.37 configuration. The retained improvement is a smaller prepared
allocation, not new arithmetic or a general wall-time speedup.

### Reject immutable reference-mask caches

The first experiment caches either one eroded reference-mask float per pixel
or two nonlinear reference-mask floats. Mode 0 is the existing fused
erosion/L2/final pass; mode 3 is a native-identical duplicate control. Two
preparation kernels reuse the existing ordered erosion and mask functions.
All 432 guarded fixtures pass 5,184 exact comparisons across three modes
and four reuse scenarios: initial use, unchanged reuse, changed distorted
input without rebuilding the cache, and changed reference mask with rebuild.
Padding, input immutability, exceptional values, and an independent old
erosion-kernel oracle are checked. This is not a sanitizer qualification of
the rejected caches.

The screen records 768 exact-output synthetic event windows across odd 4K,
odd 1080p, and Flower dimensions, two input patterns, four preparation/reuse
policies, and eight balanced forward/reverse four-mode orders. Pattern 1 is
random finite data; pattern 2 combines tiny/large psycho values and a ramp
reference mask. These are not captured encoder inputs. Cache allocations and
original reference-mask construction are excluded throughout. The steady
window runs three final passes without preparation; other windows include
one cache preparation plus the stated number of final passes.

| 4K random finite window | One-float cache | Two-float cache | Identical control |
| --- | ---: | ---: | ---: |
| Steady state, preparation excluded | +4.77% | +9.49% | -0.02% |
| Preparation + one comparison | +23.77% | +30.59% | -0.24% |
| Preparation + two comparisons | +14.54% | +20.04% | -0.06% |
| Preparation + four comparisons | +10.21% | +15.18% | -0.07% |

Both caches regress at every size/pattern/reuse paired median. The existing
final pass streams eight reference planes, eight distorted planes, two AC
planes, two raw masks, and an output: a nominal 21 plane-equivalents. Neighbor
erosion reads can hit cache; adding one or two streaming input planes raises
that nominal footprint by 4.76% or 9.52%, close to the measured 4K losses.
This is evidence consistent with bandwidth limitation, not a measurement of
DRAM traffic. Reusing free storage would not remove the extra reads.

All 60 parent Butteraugli native bodies remain identical in the 65-body
probe. The five additions are two final kernels, two preparation kernels,
and the identical duplicate; final kernels use 40 registers, preparations
23, with no local/stack/shared allocation. Endpoint state changed from
53 C/P0/1,282 MHz SM/6,000 MHz memory/22.12 W with neither limit active to
58 C/P3/1,335 MHz/5,500 MHz/32.29 W with thermal/power limits active.

One unelevated Nsight Compute 2022.3 probe, with clock/cache control disabled,
returned `ERR_NVGPUCTRPERM`. No counters or profiled kernels were obtained;
its ordinary eight-fixture test passed 96 comparisons. The user was promptly
notified. No elevation, firewall change, driver setting, or clock change was
attempted. This optional counter-access limit does not block ordinary CUDA
tests or Nsight Systems tracing and does not establish a firewall failure.

### Retained lifetime changes

The horizontal RGB intermediates and later horizontal XYB intermediates are
dead before psycho outputs 7-9 are produced. Both groups can use those future
outputs. RGB inputs are external or staged in psycho outputs 0-2, so they
remain disjoint; packed temporary indexing also fits the packed reference
subscale planes. XYB outputs remain in working planes 21-23, and frequency
scratch remains in 24.

After Malta and distorted-mask precompute, distorted psycho output 9 is
dead. Finish its mask blur first, then reuse that plane for the uncached
half-scale reference mask and its in-place vertical blur. The distorted mask
remains in plane 23; horizontal scratch is 24. Final L2 reads psycho outputs
0-7 only. The cached full-resolution reference mask in plane 20 is unchanged.
All operations remain ordered on the same stream; no synchronization or
kernel arithmetic changes are introduced.

| Working planes | Psycho construction | Difference phase |
| --- | --- | --- |
| 0-9 | Reference outputs; future 7-9 temporarily hold horizontal blurs | Cached reference psycho |
| 10-19 | Distorted outputs; future 17-19 temporarily hold horizontal blurs | Distorted psycho; 19 later holds uncached half-scale reference mask |
| 20 | Cached full-resolution reference mask | Same cached mask |
| 21-22 | XYB outputs | Malta AC accumulations |
| 23 | Third XYB output | Distorted-mask input and blurred result |
| 24 | Frequency scratch | Horizontal mask scratch, then crop/subscale result |

The private plan's working-plane array shrinks from 27 to 25. The allocator
and memory statistics already derive their counts from that constant; the
independent test oracle is updated separately. Cached-reference bytes remain
11 full planes plus ten optional subscale planes. Logical comparison scratch
drops from 16 to 14 full planes plus two reductions. One prepared allocation
and zero allocations on comparison reuse are preserved. The public ABI and
prepared-operation ownership/failure contracts do not change.

For full-plane bytes `P = 4 * max(8,width) * max(8,height)`, the physical arena
saves exactly `2 * align_up(P,64)` bytes. The two reductions, Gaussian kernels,
and optional ten-plane reference subscale cache are unchanged.

| Prepared arena | Parent bytes | S67 bytes | Saved bytes |
| --- | ---: | ---: | ---: |
| 3839 x 2159 | 978,352,436 | 912,045,108 | 66,307,328 |
| 1919 x 1079 | 244,426,868 | 227,862,004 | 16,564,864 |
| 510 x 532 | 32,026,036 | 29,855,412 | 2,170,624 |

### Isolate layout before shrinking the allocation

A four-mode diagnostic keeps the physical 27-plane arena: original layout,
psycho-temporary reuse only, half-scale-mask reuse only, and both changes.
The newly unused plane 26 pointer is null for psycho reuse; both 25 and 26
are null for the combined mode. All four modes pass the original 27-case
prepared suite. Across 29 extents, three policies, and three reuse cases,
all 261 full maps and double scores are bit-identical, including host-row
padding. All 46 encode cases in all four modes match frozen S65 codestream
hashes (184 encodes); this control gate does not claim fresh decoding.
All 188 GPU bodies in the combined diagnostic are instruction-identical.

The first link runner failed after successful compilation because a
PowerShell array misparsed concatenated object paths. Its script/log are
retained; a corrected runner linked the existing objects. This was a harness
failure, not a GPU failure. These diagnostic objects/executables were built
against the retained parent 27-plane header and libraries before production
edits. Rebuilding them against the new private 25-plane plan ABI without
restoring those dependencies is invalid.

Seven alternating same-binary pairs compare combined layout against mode 0.
Warm processes use three warmups/seven samples; cold processes zero/one.
All 41 phase fields and size checks are retained. Separate hash qualification
establishes bytes; timing output alone only checks size. These controls still
allocate 27 planes and are not pooled with release measurements below.

| Layout-only control | Quantization pipeline | Codestream encoding | Whole encode | Whole wins / 7 |
| --- | ---: | ---: | ---: | ---: |
| Warm 4K | +1.18% | +1.78% | +1.00% | 3 |
| Warm 1080p | +0.56% | -0.21% | +0.70% | 3 |
| Warm Flower | +1.34% | -1.51% | +0.42% | 3 |
| Cold 4K | +0.23% | -0.29% | +2.96% | 3 |
| Cold 1080p | -0.82% | -3.87% | -2.18% | 5 |
| Cold Flower | -1.61% | -4.84% | -3.91% | 4 |

These small mixed effects do not establish an intrinsic layout speedup.
Whole-encode ranges are retained, including warm Flower -29.72% to +7.11%
and cold 4K -5.39% to +19.49%. Control endpoints were 55 C/P3/1,282 MHz SM/
5,500 MHz memory/21.86 W and 63 C/P3/1,282 MHz/5,500 MHz/24.90 W, both limited.

### Fresh release qualification and timing

All 73 CUDA tests, all 50 CPU-only tests, and five host ASan targets pass.
The prepared test now has 31 cases, adding 15x1025, 1025x15, 513x257, and
257x513 to exercise packed subscale aliasing and partial tiles. Its sanitizer
subset grows to seven cases. Worst observed prepared map/score errors versus
CPU are 0.000231382 / 0.000020504, within the existing tolerances.
The release passes prepared memcheck/initcheck/synccheck/racecheck and AQ
memcheck/initcheck/synccheck; no full-AQ racecheck is claimed. The prepared
racecheck completed in about 67 seconds, not an hour-long blocked run.

The smaller release independently matches all 261 parent maps/scores and
all 29 physical memory-statistic fixtures. Flower's resource value in the
table is additionally confirmed by the complete-workflow allocation trace,
not one of those 29 map-probe extents. All 58 fresh parent/candidate pairs
(46 main, six high-density, six maximum-compression) are byte-identical,
decode to the expected dimensions with pinned libjxl, and have identical
independent Butteraugli scores. All 188 release native GPU bodies match S65.

Four separate seven-pair release cohorts compare retained S65 with S67:
phase-probe/public benchmark, each warm/cold. Warm uses three warmups/five
samples per process; cold uses zero/one. All 41 phase or seven public raw
fields are retained. Percentages are medians of paired candidate/parent
ratios, not ratios of independently selected medians.

| Release comparison | Quantization pipeline | Codestream encoding | Whole encode | Whole wins / 7 |
| --- | ---: | ---: | ---: | ---: |
| Phase warm 4K | -1.37% | -1.69% | -0.40% | 5 |
| Phase warm 1080p | -0.73% | -1.09% | -0.78% | 5 |
| Phase warm Flower | -2.23% | -1.78% | -1.20% | 5 |
| Phase cold 4K | +0.17% | -2.55% | -0.05% | 4 |
| Phase cold 1080p | -1.75% | -9.01% | -2.61% | 4 |
| Phase cold Flower | +3.17% | -4.62% | +5.88% | 3 |
| Public warm 4K | +0.50% | -0.20% | +2.59% | 1 |
| Public warm 1080p | +1.31% | +1.25% | +1.12% | 3 |
| Public warm Flower | -0.96% | -0.97% | +0.41% | 3 |
| Public cold 4K | -1.78% | +1.77% | -1.93% | 4 |
| Public cold 1080p | -3.07% | -6.98% | -4.05% | 7 |
| Public cold Flower | -2.76% | +1.27% | -0.52% | 4 |

Public warm 4K loses six of seven pairs; it must not be presented as a win.
Whole ranges include -6.26% to +9.10% there, -8.01% to +32.44% for phase cold
Flower, and -41.11% to +7.52% for public cold Flower. Endpoints are
67 C/P3/1,702 MHz SM/36.48 W and 69 C/P3/915 MHz/32.97 W with thermal and power
limits active at both. These snapshots do not explain each sample, but the
scatter and disagreement rule out a universal speed claim. The retained
benefit is exactly two fewer allocated full planes.

Twelve fresh Nsight Systems traces cover two alternating parent/candidate
pairs per workload, three warmups and one captured encode each. All launch
names/order/grid/block/register/shared/local metadata and copy counts/bytes
match. There are 391 / 378 / 391 launches at 4K / 1080p / Flower. Each trace
records five device allocation requests; only request four changes, by the
exact arena savings above. Copies remain H2D 118,254,624 / 29,631,464 /
4,028,828 bytes, D2H 103,699,012 / 25,924,192 / 3,430,532 bytes, and D2D
518,400 / 129,600 / 17,152 bytes, respectively.

Instrumented total-GPU pair changes are +2.57% / -13.13% at 4K,
-0.61% / +0.67% at 1080p, and +0.10% / +0.09% at Flower. They are diagnostics,
not release wall times. Fresh 4K candidate medians still put the two 64-row
Malta response specializations at 18.90 and 13.53 ms, joint low-medium rows
at 10.60 ms, and fused erosion/L2/final at 9.45 ms. The next speed-focused
investigation should target remaining input traffic/reuse in these measured
stages, without assuming that an added immutable cache is free.

Current-policy batch checks use one warmup and three alternating pairs and
verify serial/batch codestream identity. These are not S65-to-S67 speedups:

| S67 batch policy | Batch 1 | Batch 2 | Batch 4 |
| --- | ---: | ---: | ---: |
| 1080p fully resident | 0.966x | 1.156x | 1.132x |
| 1080p maximum throughput | 0.969x | 1.588x | 2.139x |
| 4K fully resident | 0.990x | 1.025x | Not run |

4K batch two spans 0.919x-1.085x; no reliable batching gain follows. Batch
four at 4K is intentionally untested on this memory-constrained device.

### Evidence and disposition

The ignored `build-cuda-ninja/profiles/s67_*` artifacts retain cache sources,
guards and all event windows; four-mode diagnostic sources, objects, full-map
dumps and encodes; failed and corrected link scripts; release build/test,
quality, sanitizer, phase/public/batch and trace records; native instructions;
and the denied optional counter probe. `s67_validate.py` recomputes evidence
and paired statistics. The freeze retains 39 release binaries/libraries,
five source/document snapshots, and a hash manifest while checking earlier
S59-S66 archives. No evidence from a rejected mode replaces production.
The first evidence validator expected the older CTest summary text including
`0 tests failed`; these logs say `100% tests passed out of N`. Its source/log
are preserved. The corrected parser accepts either form and still verifies
every individual passed record (73 CUDA and 50 CPU); no test result changed.

Keep the 25-plane allocation for its verified resource reduction, with the
mixed/adverse timings visible. Do not retain either reference-mask cache,
claim changed numerical behavior, or treat this checkpoint as a maxed-out
backend. Further speed-focused work remains open.

## Fuse mirrored RGB blur and Opsin (S68)

S68 starts from S67 `b69fde4` and targets fully-resident Butteraugli input
traffic and launch overhead. S67's retained candidate traces contain eighteen
horizontal five-tap RGB convolutions and six vertical/Opsin kernels per
encode. Their independent 4K medians are 6.97 and 6.94 ms; 1080p is 0.964 and
1.162 ms, and Flower is 0.151 and 0.161 ms. `s68_baseline.json` identifies and
hashes those historical captures; these are not fresh S68 baseline timings.

The old resident path already fuses vertical blur with pointwise Opsin but
still writes and rereads three full horizontal RGB planes. S68 replaces that
four-launch bundle with a shared-memory two-dimensional tile on sufficiently
wide images. A cooperatively loaded, reflected RGB halo feeds the ordered
horizontal five-tap sums; their rounded results remain in shared memory for
the ordered vertical sums and unchanged Opsin conversion. The original RGB
sample is also reused from the raw tile. Both divisions and FMA chains keep
their previous order. No approximate reciprocal or global fast math is added.

For three full-resolution and three half-resolution psycho preparations,
eliminating the intermediate image roundtrip removes
`24 * 3 * (full_pixels + half_pixels)` logical image-access bytes: 746,064,072
at odd 4K and 186,408,072 at odd 1080p. These are not measured DRAM bytes;
shared tiles repeat halo input loads and the cache can serve some accesses.
There is no additional arena reduction: S67's 25 full working planes remain,
because narrow Opsin and subsequent 33-tap XYB work still need horizontal
storage. The verified arenas remain 912,045,108 / 227,862,004 / 29,855,412 bytes
for odd 4K / odd 1080p / Flower, respectively.

### Geometry experiments and selected implementation

An eleven-mode probe compares the original four-launch operation (0), joint
RGB horizontal plus original vertical/Opsin (1), direct-input fused tiles
32x8/16/32 (2/3/4), shared-input fused tiles 32x8/16/32 (5/6/7), shared 64x16
(8), an instruction-identical duplicate of the old horizontal kernel (9),
and direct-input 64x16 (10). Full fusion permits null horizontal pointers.
All modes pass 5,148 guarded fixtures across 26 geometries, packed/padded
layouts and nine patterns, with three reuse passes: 15,444 bitwise output
comparisons. Patterns include zero-sum, NaN and infinite weights. All eleven
modes also pass the forced 1 x 4,194,305 tall case beyond 65,535 tile rows.

Four separate synthetic timing cohorts retain 2,904 / 2,904 / 8,228 / 4,840
event windows, totaling 18,876 windows and 780 paired summaries. Each window
contains three graph-captured complete Opsin operations; its reported time
is divided by three. Eight warmup graphs per mode precede twenty-two
balanced forward/reverse orders across all eleven modes. Every window checks
all three guarded XYB arrays against the seven-launch oracle outside timing;
inputs, weights, ignored buffers and remaining guards are checked separately.
These are Gaussian-weight synthetic RGB fixtures, not captured encoder-stage
inputs. No adverse samples are trimmed.

Shared 32x16 halves odd-4K operation time in the initial packed/padded screen
(-50.03% / -50.28%, 22/22 wins each); direct 32x16 gives -46.71% / -46.28%.
However, unconditional shared 32x16 is wrong for thin images: the
1 x 1,048,577 fixture regresses +164.46% / +124.59%, 3 x 32,767 regresses
+73.33% / +40.06%, and 7 x 4,097 regresses +16.29% / +13.71%. Inactive columns
and repeated halo work outweigh saved global materialization there. Joint
horizontal RGB avoids that penalty. Widths 24-26 show mixed packed/padded
crossover; widths 27-31 favor fusion in the measured tall fixtures. Short
images and small tile grids favor eight rows, while larger grids amortize
the halo better with sixteen rows.

The retained policy is a measured compromise, not a universal optimum:

| Geometry | Resident operation | Launches per psycho preparation |
| --- | --- | ---: |
| Width <= 24 | Joint RGB horizontal, then original vertical/Opsin | 2 |
| Wider, height <= 8 or fewer than 256 32x16 tiles | Shared fused 32x8 | 1 |
| Remaining wider images | Shared fused 32x16 | 1 |

The tile-count decision uses 64-bit arithmetic. Tiles use flattened block
indexing, repeated mirror reflection and no early exit before either shared
barrier. The internal plan layout and public ABI do not change. Wider inputs
ignore horizontal pointers; the narrow path still requires three disjoint
packed intermediates. Production contains neither the experimental mode
environment variable nor rejected direct/large-tile specializations.

Native extraction finds 70 probe bodies (60 unchanged parent bodies plus
ten additions), 198 in each full-encoder diagnostic control (188 unchanged
plus ten), and 191 in release (188 unchanged plus three). The duplicate
horizontal control is instruction-identical to its parent. All three added
release bodies are instruction-identical to the tested/timed prototypes.
Joint horizontal uses 40 registers and no shared storage. Shared fused 32x8
and 32x16 use 37 registers and 9,792 / 16,320 shared bytes. None uses stack
or local storage. These resources are not a measurement of achieved occupancy.

### Correctness and release qualification

The first full-encoder control keeps modes 0/1/3/6/9 in one executable;
each passes 31 prepared cases, and modes 0/3/6 produce identical sets of 261
full maps/scores across 29 extents, three policies and three reuse scenarios.
The selected-policy control adds mode 11, dispatching to modes 1/5/6; modes
0/11 pass the same functional and full-map gates. All memory statistics match
the independent 25-plane accounting, with one prepare allocation and no
comparison allocation. The first control's 230 encodes and the policy
control's 92 encodes match the 46 frozen S67 codestreams. Those diagnostic
hash gates do not claim fresh decoding or metric computation.

Fresh release qualification passes:

- The release build, all 73 CUDA tests (106.01 seconds), all 50 CPU tests
  (15.74 seconds), and five existing host AddressSanitizer targets.
- The expanded permanent Opsin fixture: 29 shapes, nine patterns and both
  layouts, totaling 522 guarded cases with three-stage reuse. Both the old
  seven-launch separate oracle and the old four-launch materialized resident
  oracle are checked. Width 24/25, height 8/9, 255/256-tile boundaries,
  multiscale shapes, exceptional weights, input/weight immutability, ignored
  buffers and prefix/suffix/row guards are covered. The existing production
  tall case now exercises narrow dispatch; forced tall fused coverage belongs
  to the eleven-mode prototype, not that permanent tall test.
- All four GPU sanitizer tools on 264 prototype fixtures (792 reuse
  comparisons per tool), plus eleven release checks: all four tools on the
  85-case scoped Opsin fixture and seven-case prepared fixture, and memcheck,
  initcheck and synccheck on AQ. No full-AQ racecheck is claimed. Memcheck
  enables full leak checks and stream-ordered race tracking; all errors,
  leak counts and race warnings are zero.
- A fresh release set of 261 full maps/scores is bit-identical to S67 and
  all resource rows match the independent unchanged 25-plane model.
- Fifty-eight fresh parent/candidate decoded pairs: 46 main cases, six
  high-density and six maximum-compression. Both codestream and decoded PFM
  bytes match, with identical dimensions and Butteraugli scores. None is
  flagged by size or quality thresholds.

### Full-workflow measurements

Six same-binary diagnostic cohorts retain seven alternating baseline/candidate
pairs per workload. Warm processes use three warmups and seven samples;
cold processes use zero warmups and one sample. All 41 raw phase fields,
process time bounds and size checks remain. Codestream identity comes from
the separate hash gates above, not these size checks. Percentages below are
the median of paired candidate/parent ratios minus one; negative is faster.
Independent medians need not yield the paired percentage. Whole wins are
out of seven.

| Diagnostic control | Quantization pipeline | Codestream encoding | Whole encode | Whole wins |
| --- | ---: | ---: | ---: | ---: |
| Mode 6 warm 4K | -3.90% | -7.87% | -3.98% | 6 |
| Mode 6 warm 1080p | -2.61% | -0.09% | -1.12% | 4 |
| Mode 6 warm Flower | -1.62% | -2.08% | -1.86% | 5 |
| Mode 6 cold 4K | +0.22% | -7.25% | -0.88% | 5 |
| Mode 6 cold 1080p | -2.53% | +3.22% | +0.01% | 3 |
| Mode 6 cold Flower | -5.53% | +1.23% | -3.53% | 4 |
| Mode 3 warm 4K | -4.66% | +2.69% | -2.53% | 5 |
| Mode 3 warm 1080p | -1.19% | +0.97% | -0.95% | 4 |
| Mode 3 warm Flower | +0.71% | -0.55% | -0.49% | 4 |
| Mode 9 warm 4K | -0.21% | +4.95% | -0.64% | 4 |
| Mode 9 warm 1080p | +0.62% | +0.53% | +0.06% | 3 |
| Mode 9 warm Flower | -0.52% | +3.51% | +3.49% | 3 |
| Mode 11 warm 4K | -4.10% | -3.37% | -4.69% | 7 |
| Mode 11 warm 1080p | -0.84% | -1.62% | -1.88% | 4 |
| Mode 11 warm Flower | +0.01% | +1.52% | +1.22% | 3 |
| Mode 11 cold 4K | -5.83% | -16.05% | -8.66% | 7 |
| Mode 11 cold 1080p | -1.18% | -2.54% | -1.94% | 5 |
| Mode 11 cold Flower | -3.95% | -9.40% | -5.20% | 5 |

The native-identical mode-9 control's Flower whole median is +3.49% and
range -2.72% to +39.22%; unchanged code also exhibits material scatter.
Selected-policy mode 11 wins all seven 4K whole pairs in both warm and cold
cohorts, but warm Flower regresses +1.22% (range -7.85% to +23.84%).
Its endpoints are 57 C/P0/1,282 MHz SM/6,000 MHz memory/22.50 W without
reported limits, then 64 C/P3/1,282 MHz SM/5,500 MHz memory/24.63 W with
thermal and power limits active. These snapshots do not explain individual
samples. Large codestream-phase changes are not caused by new entropy code.

Each full-encoder control also has twelve Nsight Systems traces: two
alternating pairs at 4K, 1080p and Flower, three warmups and one captured
encode. Fixed mode 6 reduces the target bundle by -45.94% / -32.72% at 4K,
-45.41% / -39.96% at 1080p, and -37.68% / -37.50% at Flower. Selected policy
reduces it by -32.68% / -40.70%, -44.23% / -47.11%, and -37.84% / -38.02%,
respectively. Both sets remove exactly eighteen launches, with unchanged
non-target kernel structure, allocation requests and copy counts/bytes.
These are diagnostic control binaries, not release-binary qualification.

Four separate release cohorts compare retained S67 with the production S68
build. They use seven alternating pairs each, three warmups/five samples for
warm processes and zero/one for cold. The phase probe retains 41 raw fields;
the public benchmark retains seven. No GPU jobs overlap these measurements.

| Release cohort | Quantization pipeline | Codestream encoding | Whole encode | Whole wins |
| --- | ---: | ---: | ---: | ---: |
| Phase warm 4K | -5.22% | -3.93% | -4.78% | 7 |
| Phase warm 1080p | -1.72% | -1.71% | -1.19% | 5 |
| Phase warm Flower | -3.46% | -1.28% | -2.03% | 6 |
| Phase cold 4K | +1.41% | -1.49% | -1.53% | 4 |
| Phase cold 1080p | -1.86% | -0.98% | -1.49% | 4 |
| Phase cold Flower | +0.45% | +1.53% | +1.17% | 3 |
| Public warm 4K | -3.59% | +1.17% | -3.80% | 6 |
| Public warm 1080p | -4.03% | -2.01% | -0.77% | 5 |
| Public warm Flower | +1.55% | +7.50% | +4.92% | 2 |
| Public cold 4K | -3.72% | -4.02% | -1.88% | 5 |
| Public cold 1080p | -2.27% | +3.83% | +1.36% | 3 |
| Public cold Flower | +6.51% | +10.54% | +6.74% | 1 |

Warm 4K agrees directionally between the phase probe and public benchmark:
phase whole medians are 382.108 -> 363.853 ms and public medians are
395.942 -> 387.916 ms, while their paired changes are -4.78% and -3.80%.
The whole paired ranges are -8.35% to -2.57% and -8.16% to +3.61%.
This does not establish an across-workload speedup. Public warm Flower loses
five of seven pairs (+4.92%, range -5.21% to +38.59%); public cold Flower
loses six (+6.74%, range -5.23% to +34.29%). Public cold 1080p is +1.36%,
and phase cold Flower is +1.17%. All adverse results remain. Release/batch
endpoints are 63 C/P3/1,282 MHz SM/22.77 W and 70 C/P3/547 MHz SM/25.31 W;
thermal and power limits are active at both. No clock, power or cooling
settings were changed.

Twelve fresh release traces independently verify the selected dispatch.
Each pair preserves all non-target launch order/grid/block/register/shared/
local metadata, five allocation requests, and copy counts/bytes. The target
changes from eighteen horizontal plus six vertical/Opsin launches to six
fused launches. 4K and 1080p use six launches of the 32x16 specialization;
Flower uses three 32x16 and three 32x8 kernel launches.

| Release trace workload | Target bundle pair changes | Total GPU pair changes | Launches |
| --- | ---: | ---: | ---: |
| 4K | -39.08% / -11.81% | -10.68% / +15.27% | 391 -> 373 |
| 1080p | -43.96% / -40.51% | -2.99% / -2.46% | 378 -> 360 |
| Flower | -37.60% / -37.27% | -2.05% / -2.08% | 391 -> 373 |

The second 4K parent target bundle happens to take 10.223 ms versus 14.100 ms
in the first parent capture; candidate times are 9.015 and 8.590 ms.
The +15.27% total-GPU result remains visible even though the target bundle
improves in that pair. Instrumented total-GPU changes are not release wall
times and cannot all be attributed to Opsin. Copies remain H2D 118,254,624 /
29,631,464 / 4,028,828 bytes, D2H 103,699,012 / 25,924,192 / 3,430,532 bytes,
and D2D 518,400 / 129,600 / 17,152 bytes for 4K / 1080p / Flower.

Current-policy serial/batch identity checks use one warmup and three
alternating pairs. These compare batching against serial execution under
S68, not S67-to-S68 improvements:

| S68 batch policy | Batch 1 | Batch 2 | Batch 4 |
| --- | ---: | ---: | ---: |
| 1080p fully resident | 0.991x | 1.238x | 1.208x |
| 1080p maximum throughput | 1.086x | 1.634x | 2.065x |
| 4K fully resident | 0.946x | 1.039x | Not run |

4K batch two spans 1.017x-1.117x. Batch four at 4K remains intentionally
untested on this memory-constrained device.

### Evidence and disposition

Retain the geometry-qualified fusion for its proven local stage reduction
and repeatable warm 4K benefit, while explicitly retaining the mixed/adverse
small-image whole timings. It adds no memory saving beyond S67. Fresh 4K
candidate kernel medians still put the two 64-row Malta responses at 17.00
and 12.68 ms, low-medium rows at 9.96 ms, fused Opsin at 8.80 ms, and
erosion/L2/final at 8.60 ms. These remaining stages offer the next bounded
input-reuse investigations; this checkpoint does not establish a maxed-out
backend.

Ignored `build-cuda-ninja/profiles/s68_*` evidence retains all prototype,
control and release sources/objects, complete guards/windows/maps/encodes,
native instructions/resources, tests, quality metrics, sanitizers, wall
cohorts, batches and 36 traces. `s68_validate.py` reconstructs raw evidence,
paired statistics and document tables without rerunning GPU work.
The checkpoint freezes 39 release binaries/libraries, five source/document
snapshots and the artifact manifest, and verifies the S59-S67 archives by
their saved-file hashes.

The first prototype build lacked an explicit `<string>` include for
`std::stoul`; its source/log and corrected build remain. An initial release
profiler script assembly split at the wrong import, leaving a no-op capture
script and a summary script that hit the pre-existing-report guard before
any GPU launch. Both failed scripts and the error log remain; the corrected
scripts produced the twelve release traces above without overwriting older
evidence. No admin/firewall prompt or new permission block was observed.
S67's optional `ERR_NVGPUCTRPERM` hardware-counter restriction was not
retried; no security settings or privileges changed. Ordinary CUDA tests,
sanitizers and Nsight Systems completed normally.

## Finer zero-region Malta investigation (S69, not retained)

S69 investigates S68 `6b91212` on the same RTX 3060 Laptop, CUDA 11.8,
sm_86 and MSVC 14.37. It retains no runtime change: production source,
tests, all 39 S68 binaries/libraries, allocation layout, numerical policy
and ABI are unchanged. This is a completed negative experiment, not a new
release optimization or evidence that the backend is maxed out.

### Hypothesis and exact band classification

Identified retained S68 release traces still put the two 64-row Malta
specializations at independent 4K candidate medians of 17.00 ms for LF and
12.68 ms for full response. `s69_baseline.json` records the source hash and
individual samples; these are not newly captured S69 baseline timings.

The S65 shortcut classifies a whole scaled tile and halo. A single nonzero
region therefore makes every output row execute all directional responses.
The S69 candidates instead classify 4-, 8-, 16- or 32-row output bands while
keeping the original 32-column tiles, adaptive heights 8/24/64, 256 threads,
and 2D/flattened grid policy. They reuse the values already produced by the
unchanged `MaltaScaleValue`, including both rounded divisions.

An input at shared row `r`, including the four-row halo, can affect output
rows `r-8..r`. Intersect that interval with the tile and mark every touched
band. Each thread ORs those bits across its scaled loads; the block combines
eight warp masks through 32 additional shared bytes and its barrier. All
lanes participate before any partial-tile exit. The validator exhaustively
compares this bit formula with direct interval membership for all input rows,
three tile heights and four band sizes, including the partial final bands
of a 24-row tile.

Only an entirely signed-zero rectangle skips the original directional sums.
NaNs and infinities mark their bands nonzero. A skipped response is still
positive zero, and initialization or the original accumulation addition
still executes. No approximate threshold or input-channel assumption
substitutes for the actual scaled-zero test.

| Mode | Operation |
| ---: | --- |
| 0 | Unchanged production |
| 1 | Eight-row bands, five warp-shuffle OR steps |
| 2 | Sixteen-row bands, five warp-shuffle OR steps |
| 3 | Thirty-two-row bands, five warp-shuffle OR steps |
| 4 | Separately named unchanged-kernel control |
| 5 | Four-row bands, five warp-shuffle OR steps |
| 6 | Eight-row bands, sm80+ hardware warp OR, shuffle fallback |

### Correctness, instructions and replay scope

All six diagnostic modes pass 7,680 guarded fixtures / 23,040 three-stage
bitwise comparisons: ten geometries, both grid forms, full/LF and
initialize/add flags, and sixteen patterns. The extended patterns place
nonzero values, infinities and NaNs around band and halo boundaries.
Existing signed-zero, denormal, threshold, exceptional-accumulator and
zero-normalization patterns remain. Input immutability, padding and output/
scratch guards are checked against the original separate scale/response
oracle. Flattening is forced on small cases; this is not a new tall-grid
limit qualification.

Each of memcheck, initcheck, synccheck and racecheck passes 576 scoped
fixtures / 1,728 three-stage comparisons, forcing all three heights on
65x65 inputs with both grid/frequency/initialization forms and four patterns.
Memcheck explicitly checks leaks and stream-ordered races. Errors, leaked
bytes, and race warnings are zero. Racecheck finishes in approximately
234 seconds on its original process, with progress markers preserved.
No full-AQ or release sanitizer campaign is claimed.

Both guarded and replay executables contain 135 GPU bodies: all 63 original
Butteraugli bodies unchanged and 72 experimental additions. All twelve
mode-4 specializations are instruction-identical to production, and the two
executables contain identical GPU bodies. Every addition has zero stack/
local storage. The additional shared requirement is exactly 32 bytes for
the mask modes: 2,592 / 5,152 / 11,552 bytes at heights 8/24/64, while the
duplicate retains 2,560 / 5,120 / 11,520. Native mode 6 uses `REDUX.OR` in
each specialization. These facts do not establish achieved occupancy or
dynamic instruction counts. Harmless unused-variable warnings in the
duplicate template are retained; native extraction confirms the unused
mask storage is eliminated.

Replay uses all 36 retained S65 first full-resolution stage captures for
4K, 1080p, Flower and three 500x500 corpus photographs. It does not recapture
S69 inputs or cover all iterations/subscales. Fourteen balanced forward/
reverse orders of seven modes, three launches per window, explicit resets
and stream synchronization produce 3,528 exact-output windows. Every window
checks the full guarded output after the original three-stage accumulation;
input arrays remain unchanged. No round or outlier is removed.

An untimed separate GPU scaling pass, followed by host integer/zero tests,
records 144 population rows across the four band sizes. These classify
active valid output pixels under the conservative rectangle test, not actual
nonzero responses, dynamic instructions or measured time saved. They are
not a CPU FP32 output oracle.

### Savings in one stage do not pay for classification everywhere

In 4K stage 1, 3,941,338 of 8,288,401 scaled values are nonzero. Whole-tile
classification makes 7,170,144 output pixels active; four-/eight-/sixteen-/
thirty-two-row bands reduce that to 6,622,824 / 6,661,600 / 6,779,600 /
6,965,408. Mode 6 improves this stage's paired median by 3.63%, winning all
fourteen pairs (3.39% faster against the duplicate control).

However, stages 0, 2 and 4 retain all 8,288,401 active pixels under every
band size, and stages 3 and 5 are already entirely zero under production.
For mode 6 the other five stage paired medians regress by 0.77%, 1.18%,
0.27%, 1.25% and 0.05%, respectively. Fine classification buys nothing there.
This coverage/overhead tradeoff explains why a sparse-stage win does not
establish a useful complete Malta replacement.

The following descriptive aggregates compare sums of six independently
measured stage medians. They are not paired whole-workflow durations or
encode speedups; negative means less stage time.

| Captured stage aggregate | Mode 1 | Mode 2 | Mode 3 | Identical mode 4 | Mode 5 | Mode 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | +0.40% | +0.53% | +0.80% | -0.02% | +0.36% | +0.10% |
| 1080p | -0.53% | -0.15% | +0.11% | +0.24% | -0.52% | -0.45% |
| Flower | +0.94% | +1.02% | +3.11% | +0.26% | +0.27% | +1.49% |
| Keong macan | +2.03% | +3.04% | +3.78% | +0.00% | +2.95% | +0.18% |
| Riaphotographs | +3.46% | +5.07% | +3.89% | -0.01% | +3.20% | +1.01% |
| Bliznaca | +3.63% | +3.70% | +2.19% | +1.74% | +2.87% | +2.64% |

At 4K, every candidate aggregate is neutral or slower, from +0.10% for the
hardware eight-row variant to +0.80% for thirty-two rows; the identical
control is -0.03%. All candidates also have adverse stage medians.
Small-case scatter remains visible: Bliznaca's identical control is +1.74%
in aggregate and ranges from -3.83% to +7.03% across its stage paired medians.
Some corpus sparse stages improve, but none of the candidate aggregates
establishes a general benefit. Selecting only those stage wins would discard
the measured classification costs.

Replay endpoints are 56 C/P0/1,282 MHz SM/6,000 MHz memory/22.41 W without
reported limits and 61 C/P3/1,282 MHz SM/5,500 MHz memory/26.59 W with thermal
and power limits active. These snapshots do not explain each sample.
No clock, cooling, security, power or privilege settings change. No new
firewall or permission block occurs; the previous optional hardware-counter
restriction is not retried.

### Disposition and evidence

Reject all five finer-band candidates. Their correctness passes, but their
local cost/benefit does not justify a production policy or full-workflow
control. No fresh release build, CPU/CUDA suite, ASan, decoded-quality,
batch or whole-encode improvement is claimed. The unchanged S68 release
retains its prior qualification and all 39 binary/library hashes.

The ignored `build-cuda-ninja/profiles/s69_*` bundle retains sources,
executables, complete build/guard/sanitizer logs, native instructions,
original capture attribution, all windows and classification counts.
`s69_validate.py` reconstructs the recorded evidence and paired tables,
checks the mask intervals, verifies unchanged S68 production and hashes
the S59-S68 archives. The freeze includes five source/document snapshots
and the diagnostic artifact manifest.

Dense response work still dominates several stages, while the coarse
shortcut already eliminates the all-zero ones. The next investigation
should address shared-load reuse or the directional response calculation
itself, preserving the established sum trees; more zero-classification
machinery is not the leading option. The broader optimization goal remains
open.


## Reuse neighboring Malta response inputs (S70)

S70 starts at S69 `cbeab54`, whose runtime is S68 `6b91212`, on the same
RTX 3060 Laptop / CUDA 11.8 / MSVC 14.37 / `sm_86` configuration. S69 left
dense directional response work as the next Malta target. S70 retains a
two-row response schedule: adjacent outputs share overlapping shared-memory
loads within the existing scaled tile. It does not relax arithmetic, change
the resident contract, or alter CPU serialization.

### Response reuse and geometry policy

`MaltaResponsePair` interleaves each full-response direction across two
adjacent output rows. LF instead calls the original response helper once
per row, keeping register liveness lower. Every output preserves the
original `Sum5`/`Sum7`/`Sum9` tree and directional `AddSquare` order. The
original scaling, both divisions, signed-zero handling and
`__syncthreads_or` shortcut remain. NaN/Inf do not qualify for a zero shortcut;
zero responses still initialize or add to the destination.

Blocks still contain 256 threads, use a 32-wide tile plus a four-pixel halo,
and select heights 8/24/64 with the S65 policy. The group loop is rolled;
the two output rows are unrolled. Invalid columns exit only after the
collective. An unused final paired row stays inside the shared halo and
never writes outside the image. Shared storage, launches and allocation
layout are unchanged; the arena still has 25 physical planes.

The original zero-aware schedule remains the production fallback when
`height < 4` or `ceil(width/32) * ceil(height/8) < 32`. This conservative
measured guard avoids tiny-grid and very-short-image setup costs without
discarding gains at the 81/127-block examples that still use eight-row
tiles. It is not a claim of an exact crossover for every GPU or input.
Compile-time assertions and permanent tests cover both sides of the new
31/32-tile and height-3/4 boundaries. The existing 2D/flat-grid policy and
tall-image limits are retained.

The experiment compared seven additions against production: direction-major
pairs at row spacing 1/2; four rows at spacing 1/2; row-major pairs at
spacing 1/2; and a separately named unchanged kernel. Full mode 1 and LF
mode 6 are selected. All twelve unchanged-control bodies are instruction
identical to production. The two standalone probes preserve 63 existing
device bodies and add 84; each whole-encoder control preserves 191 and
adds those same 84. Release keeps just twelve new specializations, for 203
bodies, with all 191 previous bodies unchanged. Every release addition is
instruction-identical to its tested/timed selected prototype.

At height 64 with ordinary grids, the native static shared-load counts
and register allocation are:

| Response schedule | Full registers | LF registers | Full shared floats/output | LF shared floats/output |
| --- | ---: | ---: | ---: | ---: |
| Original | 39 | 36 | 61 | 49 |
| Selected adjacent pair | 48 | 40 | 35 | 35 |
| Direction-major adjacent four rows | 96 | 78 | 22 | 24 |

These are complete-body static instruction counts divided by group outputs,
not dynamically executed loads or DRAM traffic. All added bodies have zero
stack/local storage. The occupancy API reports ordinary-grid resident-block
limits of six to five for full responses and six to six for LF; this is
theoretical occupancy, not a hardware-counter measurement. Flat height-24/64
pairs use 56/47 registers and permit four/five blocks respectively. Four-row
candidates reduce loads further but lose on large inputs: their independent
six-stage 4K sums regress 15.74% (adjacent) and 17.93% (spacing two).

### Prototype and release qualification

The standalone probe passes 8,960 guarded fixtures / 26,880 three-stage
comparisons, including forced tile/grid forms, exceptional values, sparse
halos and signed accumulation. Four prototype sanitizer campaigns each pass
672 fixtures / 2,016 comparisons with zero diagnostics. Racecheck takes
214.8 seconds; this is instrumented validation, not performance timing.

The replay uses the 36 retained S65 first full-resolution stage captures
from 4K, 1080p, Flower, Keong, Ria and Bliznaca. These are identified older
captures, not newly recorded S70 inputs or all iterations/subscales. Sixteen
balanced forward/reverse orders cover eight modes in 4,608 exact-output
windows, each timing a three-launch accumulation burst. Full mode 1 improves
the two 4K full-response paired medians 25.75% / 23.73%, with 16/16 wins
each. LF mode 6 improves the dense stages 4.30% / 4.41%; the zero stages
are effectively neutral. The selected independent six-stage median sums
improve 10.01% 4K, 11.93% 1080p, 13.57% Flower, 21.60% Keong, 16.43% Ria
and 12.66% Bliznaca. These sums are not complete-encode measurements.

Two synthetic geometry cohorts each pass 48 cases / 6,144 exact-output
windows. A final boundary cohort adds 32 cases / 4,096 windows. They cover
narrow/tall inputs, eight/24/64-row dispatch cutoffs, varying aspect ratios
at 32/64 eight-row tiles, and heights 1/3/4/5/6/7/8/9. The final height-4/5/6/7
full-response dense comparisons improve about 5-11%; LF is mostly neutral
or favorable relative to the duplicate control. The 31/32-tile boundary
remains noisy, consistent with a conservative guard, not a universal
threshold. Generated packed scalar planes are not real encoder captures.
Across all replay/geometry cohorts, 20,992 guarded timing windows and 1,148
paired summaries are retained, including unfavorable results.

The same-binary policy control passes both modes' 31 prepared fixtures,
261 complete map/score comparisons with unchanged memory statistics, and
92 encodes matching 46 retained S68 codestream hashes. That control gate
does not claim new decoded-quality measurements.

Release qualification is separate:

- 73/73 CUDA tests pass in 125.82 seconds; 50/50 CPU tests in 17.90 seconds,
  plus five host AddressSanitizer checks.
- The permanent Malta test passes 2,944 guarded cases with three-stage reuse
  against two oracles (17,664 stage comparisons). The retained tall suite
  adds 32 cases at the old and current 64-row 2D/flat boundaries, including
  height 4,194,241. The scoped sanitizer fixture has 264 cases.
- All four release Malta and four prepared-Butteraugli sanitizer checks
  pass, plus AQ memcheck/initcheck/synccheck: eleven checks, zero errors,
  leaks or race hazards. There is no full-AQ racecheck claim.
- All 261 release maps/scores match S68 bit-for-bit. Independent accounting
  checks 29 extents and the unchanged 25-plane arena; odd 4K allocation
  remains 912,045,108 bytes.
- All 58 freshly encoded, decoded and metric-compared parent/candidate
  pairs have identical codestreams, decoded pixels, size and Butteraugli
  score, including final-score collection and both extra entropy modes.

### Whole-workflow controls and release timing

The same-binary mode-10 policy comparison uses seven alternating process
pairs per case, three warmups/seven measured samples when warm and zero
warmups/one sample when cold. Core warm/cold, corpus warm and mode-4
duplicate-control cohorts retain 84 pairs / 168 processes / 924 measured
encodes. All 41 phase fields and raw samples are retained; timing checks
encoded size, while the separate quality gates establish byte identity.

These controls are deliberately not presented as a clean wall-time win.
Mode-10 warm totals change +0.30% 4K, +1.77% 1080p and +13.47% Flower;
cold totals -0.53%, +1.77%, -3.63%. Corpus warm totals are -2.31% Keong,
-4.50% Ria, -2.89% Bliznaca. The unchanged-kernel control itself reports
-3.67% 4K total with 7/7 wins, +0.62% 1080p and -0.32% Flower, with wide
serializer variation. Its different-time cohort is not a correction factor
to subtract from the candidate. Twelve control traces independently confirm
Malta reductions in every pair with unchanged non-target launch structure.

The release comparison instead uses retained S68 and newly linked S70
binaries: seven alternating pairs, three warmups/five samples when warm,
zero warmups/one sample when cold. The phase probe retains 41 fields and
the ordinary public benchmark seven. These four cohorts contain 84 pairs,
168 processes and 504 measured encodes. Percentages below are medians of
paired candidate/parent ratios, not ratios of independent medians; negative
means faster. Whole-encode wins are out of seven.

| Release cohort | Quantization pipeline | Serialization | Whole encode | Wins |
| --- | ---: | ---: | ---: | ---: |
| Phase warm 4K | -2.32% | -2.81% | -2.85% | 7 |
| Phase warm 1080p | -1.75% | -0.59% | -4.40% | 5 |
| Phase warm Flower | -1.62% | +0.76% | +0.70% | 3 |
| Phase cold 4K | -5.20% | -2.16% | -4.53% | 5 |
| Phase cold 1080p | +3.58% | +8.78% | +3.94% | 0 |
| Phase cold Flower | -7.22% | -10.23% | -7.23% | 6 |
| Public warm 4K | -3.55% | -3.55% | -2.01% | 5 |
| Public warm 1080p | -0.83% | -4.03% | -1.93% | 4 |
| Public warm Flower | -2.01% | -2.45% | -4.81% | 5 |
| Public cold 4K | -2.37% | -2.55% | -2.66% | 6 |
| Public cold 1080p | -1.01% | -6.57% | -4.77% | 4 |
| Public cold Flower | +1.42% | +3.93% | +1.98% | 2 |

The 4K warm phase whole-ratio range is -7.94% to -0.53%; public warm is
-5.58% to +6.76%. Phase cold 1080p loses every pair (+3.28% to +35.27%),
and public cold Flower loses five of seven. Their contradictory companion
cohorts and sizeable unchanged-code/control variation limit causal claims;
they do not erase the regressions. CPU serialization is unchanged, so its
observed changes must not be attributed to the Malta implementation.

The release GPU-state samples move from 62C/P3/1282 MHz to 69C/P3/802 MHz,
with thermal and power-limit flags active. Control samples also retain
active limits. No clock, power, cooling, firewall or privilege setting was
changed. No new permission/firewall prompt blocked the work; the earlier
optional hardware-counter restriction was not retried. These laptop
measurements do not establish a universal speedup or sustained server rate.

### Release traces, batches and remaining work

Two alternating CUDA-profiler-API capture pairs per core workload produce
twelve release traces. Raw SQLite checks prove identical launch counts,
grids/blocks, shared/local memory, allocations and copies. Only the Malta
kernel identities/register allocation change. These are sums of instrumented
kernel durations, not release wall times:

| Workload | Malta time, pairs 0 / 1 | All kernel time, pairs 0 / 1 | Launches |
| --- | ---: | ---: | ---: |
| 4K | -18.84% / -23.11% | +3.79% / -2.88% | 373 -> 373 |
| 1080p | -17.92% / -14.35% | -1.91% / +0.48% | 360 -> 360 |
| Flower | -13.12% / -13.17% | -1.46% / -1.34% | 373 -> 373 |

Full and LF groups improve in every pair; 4K full responses improve
27.64% / 30.73%, LF 12.06% / 17.02%. Other-kernel timing variation again
precludes treating the target reduction as an equal total-GPU speedup.

Current-S70 batch checks use one warmup and three alternating serial/batch
samples. Their paired speedup medians compare serial and batch execution
within S70, **not** S68 with S70:

| Current S70 batch workload | Batch 1 | Batch 2 | Batch 4 |
| --- | ---: | ---: | ---: |
| 1080p fully-resident | 0.987x | 1.232x | 1.319x |
| 1080p maximum-throughput | 1.099x | 1.497x | 1.880x |
| 4K fully-resident | 1.063x | 1.076x | not run |

The ignored `build-cuda-ninja/profiles/s70_*` evidence retains prototypes,
controls, sources, all timing windows, complete logs, native instructions,
traces, quality outputs and read-only reconstruction scripts. The checkpoint
retains 39 binaries/libraries and five source/document snapshots; the
artifact manifest includes this whole S70 bundle. `s70_validate.py --frozen`
reconstructs native, replay, quality, timing and trace evidence, checks the
document tables and frozen files, and hashes every S59-S69 archive.

This is a retained kernel improvement with a modest measured 4K whole-encode
benefit, not a claim that fully-resident encoding is maxed out. The two
release traces' independent 4K per-kernel medians still put LF Malta at
15.54 ms, low/medium vertical convolution at 10.80 ms, full Malta at
10.12 ms, Opsin at 9.76 ms and erosion/L2/final at 9.64 ms. Those medians
are not a single common sample. Further work should investigate the LF
response/scaling split or another measured convolution bottleneck; larger
row groups and finer zero classification are not justified by this evidence.

## Conditional second Malta division (S71, not retained)

S71 starts from S70 `5eb8751` on the same RTX 3060 Laptop, CUDA 11.8,
MSVC 14.37 and `sm_86` setup. LF Malta remains the largest independent
4K kernel-time median. In retained-input replays, the two entirely zero
LF stages still take about 0.64 ms despite the existing response shortcut.
S66 established two independently refined divisions in `MaltaScaleValue`;
S71 tests skipping the second when no asymmetric correction consumes it.
This is an arithmetic/control-flow experiment, not another reciprocal
approximation, relaxed FP mode or raw-input-zero shortcut.

### Candidates, guards and emitted instructions

The diagnostic copies both the prior single-row and S70 paired-row kernels,
changing only the scaling helper. Halo loading, response sum trees, zero
classification, accumulation and the existing geometry policy stay fixed.

| Mode | Change | Disposition |
| --- | --- | --- |
| 0 | Actual S70 dispatch | Timing baseline |
| 1 | Factor correction term/sign; divide only when needed | Reject: finite bitwise mismatch |
| 2 | Put the original division inside each correction branch | Reject: slower |
| 3 | Original helper in separately named kernels | Native-identical control |
| 4 | One outer correction-needed guard, then original branch tree | Reject: no general benefit |

Mode 1 fails at stage 0 of the 31x9, full-response, additive, finite
near-equal pattern, with the single-row ordinary-grid schedule. Factoring
the expression is not bitwise equivalent in this compiled implementation;
no timing or sanitizer qualification is claimed for that mode. The build
driver completes all four independent guard processes before reporting
this expected rejection. Both original compilations succeed.

Modes 2, 3 and 4 each pass 2,560 fixtures / 15,360 two-oracle stage
comparisons against the separate scale/response and prior zero-aware
implementations. Ten geometries, both row schedules, both grid forms,
full/LF, initialize/add and sixteen patterns exercise finite thresholds,
signed zeros, denormals, infinities, NaNs and exceptional accumulators.
Guards, padding and input immutability remain checked. These runs force
flattening on bounded shapes; they are not new tall-grid limit tests.

Modes 2 and 4 also pass memcheck, initcheck, synccheck and racecheck:
eight scoped campaigns, each with 192 fixtures / 1,152 two-oracle stage
comparisons, forcing heights 8/24/64 and both schedules/grid forms on
65x65 inputs. Errors, leaked bytes and race warnings are zero. Memcheck
includes explicit leak and stream-ordered-race checks. The original
racecheck processes finish in approximately 96 seconds each, with progress
markers retained. These are prototype checks, not full-AQ/release campaigns.

All three diagnostic executables contain the same 171 GPU bodies:
75 unchanged parent bodies and 96 additions. All 24 mode-3 bodies match
their parent counterparts exactly; modes 1/2/4 differ in all 24. Registers
and shared storage match the corresponding control in every specialization,
with zero stack/local storage. Paired ordinary height-64 full/LF remain
48/40 registers and 11,520 shared bytes.

The native scale-loop audit confirms the outer guard ahead of the second
reciprocal/refinement/exceptional path in all 24 mode-4 bodies. For example,
paired height-64 LF branches at `0x420` to `0x600`, skipping the second
`MUFU.RCP` at `0x430`. Mode 2 instead emits five division sequences: the
first quotient plus four separate correction paths. The static input-load
to shared-store region has 106 instructions in mode 2 versus 59 in the LF
control and 60 in mode 4. These are static code counts, not dynamic executed
instructions or achieved occupancy; divergent lanes can traverse different
correction paths.

### Population and retained-input timing

A source-predicate audit covers all 36 retained S65 first-full-resolution
captures, totaling 68,281,932 raw pixels. It models the actual 32-lane
cooperative shared-halo loading cohorts, including padding, rather than
counting only individual pixels that avoid a correction. At 4K the
whole-load-warp skip fractions for stages 0 through 5 are approximately
0%, 23.775%, 0%, 100%, 0.000275% and 100%. Dense stages therefore provide
almost no warp-wide division elimination, even when many individual pixels
need no correction. This is a static population model, not a hardware
counter measurement.

The selected replay excludes invalid mode 1 and retains the original
five-mode binary unexecuted. Modes 0/2/3/4 use 24 balanced forward/reverse
rounds, three launches per window, explicit resets and stream synchronization.
All 3,456 windows pass exact guarded three-launch accumulation checks;
no round or outlier is removed. The six workloads are 4K, 1080p, Flower
and three 500x500 corpus photographs. These are retained S65 inputs, not
fresh S71 captures, all iterations or complete encoder workflows.

The following sums of six independently measured stage medians are
descriptive aggregates, not paired whole-encode measurements. Negative
means less stage time.

| Captured stage aggregate | Four branches (2) | Identical control (3) | Outer guard (4) |
| --- | ---: | ---: | ---: |
| 4K | +3.71% | -0.05% | +0.44% |
| 1080p | +2.67% | -0.26% | +0.19% |
| Flower | +5.26% | -0.24% | -0.13% |
| Keong macan | +3.55% | +0.22% | -1.18% |
| Riaphotographs | +4.46% | +4.81% | +5.95% |
| Bliznaca | +3.36% | -1.15% | +0.00% |

Mode 2's 4K stage-0 paired median regresses 12.04%, losing all 24 pairs.
Mode 4 regresses 4K dense stages 0/1/2/4 by 1.14% / 1.65% / 0.52% /
0.25%; its entirely zero stages 3/5 improve only 0.54% / 0.43%, or
0.32% / 0.48% versus the identical control. The compiler did retain the
intended division skip, but it barely changes these large zero-stage times.
Small-case scatter remains visible, including the Riaphotographs control's
+4.81% aggregate. No selective sparse-stage policy is justified here.

### Disposition and evidence

Reject all three changed layouts. Production source, tests, allocation
policy and the 39 S70 release binaries/libraries remain unchanged. No fresh
release build, full CPU/CUDA suite, ASan, decoded-quality campaign, batch
qualification or whole-encode speedup is claimed.

The ignored `build-cuda-ninja/profiles/s71_*` bundle preserves the failed
guard, both original binaries, the selected replay binary, all build and
native logs, source-predicate populations and every timing window. Native
identity ties the guarded and timed bodies together. `s71_validate.py`
read-only reconstruction checks the native reports, guards, sanitizer logs,
all population rows, every paired timing summary, capture attribution and
the document table. The freeze includes five source/document snapshots;
validation also verifies all S59-S70 artifact hashes. No new firewall,
admin or permission block occurs, and the previous optional-counter
restriction is not retried. No security, clock, cooling or power setting
changes.

The next investigation should separate unavoidable input/output and tile
staging costs from scale/response arithmetic, or address another measured
convolution bottleneck. The small benefit from removing a real division
does not itself prove a bandwidth ceiling. S71 narrows one hypothesis;
it does not establish that the backend is maxed out.

## Align Malta accumulator rows (S72, not retained)

S72 starts from S71 `7f4f024`, whose runtime is S70 `5eb8751`, on the same
RTX 3060 Laptop / CUDA 11.8 / MSVC 14.37 configuration. S71 established
that suppressing an unused division barely changes the large zero-LF
stage time. S72 separates component costs, rejects a materialized split,
then isolates accumulator row layout as a useful optimization target.
The production candidate changes storage/addressing, not Malta arithmetic.

### Component experiment and rejected split

The diagnostic copies both single-row and paired-row response kernels.
Nine modes compare actual S70, a separately named native-identical control,
linear scale/accumulation, tiled scale/center accumulation, linear
subtraction/accumulation, tiled subtraction/center accumulation, response
from an untimed precomputed scaled plane, timed materialization plus that
response, and positive-zero accumulation. The five component-only modes
omit the directional response and are not valid encoder replacements.
The response-only timing excludes its setup. Component costs are not
assumed additive, and no measured DRAM ceiling is inferred by subtraction.

The timed materialized split is exact but rejected: the six-stage sum of
independent medians regresses 46.71% at 4K and 45.98% at 1080p. Every 4K
stage loses all 18 pairs. The named control and split each pass 2,560
fixtures / 15,360 comparisons against two exact oracles; the split also
passes all four scoped GPU sanitizer tools. Native inspection preserves
all 75 production bodies and adds 63 bodies. All three diagnostic binaries
contain the same 138 bodies, including 24 instruction-identical controls.
The intended loads, shared staging and barriers remain emitted; all 63
additions have zero stack/local storage.

At the zero-LF 4K stage 3, full Malta takes 0.639488 ms, linear scaling
0.526336, tiled scaling 0.637440, linear subtraction 0.525653, tiled
subtraction 0.632320, response-only 0.518656, timed split 0.918187 and
zero accumulation 0.271019. Replacing scale arithmetic with subtraction
barely changes these large component times. Linear components being faster
than tiled components motivates checking memory layout; it does not prove
that either is a complete replacement or establish a bandwidth limit.

All replays use the same 36 attributed first-full-resolution S65 captures:
six stages each for odd 4K, odd 1080p, Flower and three corpus images. They
are not fresh S72 captures or all encoder iterations/subscales. The
nine-mode run retains 5,832 output-checked windows, 288 paired summaries
and 48 descriptive aggregates. Component expected outputs intentionally
differ from full Malta; inputs, row padding and temporary guards are checked.
The additional four-tool component sanitizer scope is only a 65x65 crop of
Flower stage 0, ordinary height-8 full-frequency dispatch, all nine modes.
Instrumented timings are not performance evidence. The first replay's
host-reference-lambda warning is removed in a distinct v2 source/binary;
the original remains unexecuted, and all GPU bodies remain identical.

### Isolating row layout

The next probe uses only the 75 unchanged production GPU bodies. Four
layouts compare packed rows, rows rounded to eight floats, rows rounded
to 32 floats, and a duplicate packed allocation. All allocations have
the same 128-byte pointer offset, with logical output equality and input/
padding checks. It retains 3,456 windows across 24 balanced rounds per
captured stage. Both aligned layouts improve every 4K LF stage in all
24 pairs. The 32-float six-stage aggregate improves 11.16% at 4K and
8.80% at 1080p; these are stage aggregates, not workflow times.

A separate five-layout isolation keeps packed input/output, aligns only
inputs, aligns only output, aligns both, or uses another packed duplicate.
It again contains exactly the 75 unchanged bodies and retains all 3,600
windows across 20 balanced rounds per stage. Output-only alignment
reproduces the large LF gain; input-only does not:

| Six-stage independent-median sum | Input only | Output only | Both | Packed duplicate |
|---|---:|---:|---:|---:|
| 4K | +1.36% | -11.18% | -11.14% | -0.14% |
| 1080p | +0.33% | -8.66% | -8.84% | -0.73% |
| Flower | +3.43% | -2.62% | -2.73% | -3.22% |
| Keong macan | -0.10% | +0.00% | -2.05% | -1.27% |
| Riaphotographs | -1.70% | -0.63% | -2.51% | -1.52% |
| Bliznaca | -0.43% | -4.96% | -3.67% | -7.10% |

Output-only 4K LF stages 2/3/4/5 improve 14.84% / 16.60% / 15.62% /
15.44%, each winning all 20 pairs; versus the duplicate packed allocation
they improve 14.89% / 16.32% / 15.63% / 15.69%. Input-only alignment
regresses those stages. Full-frequency initialization stages are near
neutral. Small-case scatter, including the strong Bliznaca duplicate,
prevents a general small-case benefit claim. This is an empirical layout
effect, not a DRAM-transaction or achieved-occupancy measurement.

### Candidate storage and correctness

Only work slots 21/22 grow to a row stride rounded to 32 floats. They hold
packed RGB temporaries during psycho construction, then AC accumulators;
those lifetimes do not overlap, and the earlier packed data fits inside
the larger storage. The other 23 working planes, masks and cached reference
layout remain packed; input strides are unchanged. Plane-base alignment remains 64
bytes. The physical range is `(height - 1) * stride + width`, without
padding after the final row. Overflow checks retain the full representable
geometry domain, using packed fallback when uint32 round-up cannot fit.
The private plan's zero stride preserves diagnostic packed compatibility.

Malta receives the aligned output stride. The two final kernels address
AC with that stride while continuing to address masks with packed width.
All arithmetic, launch dispatch and the 25-plane count remain unchanged.
The candidate release contains the same 203 bodies: 201 instruction-identical to
S70; only `L2FinalKernel` and `ErosionL2FinalKernel` change. Both stay at
40 registers, zero shared/stack/local storage. Cached reference bytes are
unchanged; peak scratch is 12 packed planes plus two physical AC planes
and the two reductions. Prepared allocation grows by 17,152 bytes at
3839x2159, 8,576 at 1919x1079 and 8,448 at 510x532, including inter-plane
alignment. Padding costs at most 31 floats per preceding row of each AC
plane; tiny widths can have proportionally larger overhead.

The release passes 73 CUDA and 50 CPU tests, five host ASan targets,
31 prepared cases with independently updated memory accounting, 2,944
guarded Malta cases, and 15 release GPU sanitizer checks. The latter are
four tools each for Malta, prepared Butteraugli and L2/final, plus three
AQ tools; full-AQ racecheck is not rerun. L2 scope covers 90 L2/final and
eight erosion/final cases; prepared scope includes seven expanded, odd,
tall, wide and partial shapes. No errors, leaks or race warnings occur.

All 261 complete maps/scores across 29 extents match retained S70 bitwise.
A diagnostic host-only environment selector switches packed/aligned rows
inside one binary with all 203 release GPU bodies unchanged. Each layout
also passes 261 full-map comparisons and the corresponding independently
checked memory statistics. The selector exists only in ignored source,
not in production. Two extra mode values duplicate the same two host
branches for subsequent timing controls.

The 58-case quality matrix comprises 46 balanced, six high-density and
six maximum-compression cases. Release and both layouts produce 174 exact
codestream matches to hash-verified retained S70 outputs. All 58 fresh
release decodes match retained parent decoded pixels and Butteraugli
metrics exactly with the pinned independent tools. The parent outputs
are verified/reused, not freshly encoded or decoded this cycle; the two
controls use byte identity rather than additional fresh decoder runs.

### Whole-workflow evidence and disposition

The same-binary layout comparison retains eight balanced rotating/reverse
rounds, three warmups and seven samples per warm process; cold processes
have no warmup and one sample. Modes 0/2 duplicate the packed host branch,
1/3 the aligned branch. All 288 control process windows and 41 raw timing
fields per process remain recorded. The selected paired medians are:

| Control cohort / workload | Aligned quantization | Aligned total | Packed duplicate total | Aligned duplicate total |
|---|---:|---:|---:|---:|
| warm 4K | +0.69% | +0.13% | -1.21% | +2.50% |
| warm 1080p | +0.27% | +0.42% | -0.45% | +0.90% |
| warm Flower | -2.21% | -3.00% | -3.65% | -3.09% |
| cold 4K | +1.42% | -0.58% | -2.51% | -1.66% |
| cold 1080p | -0.28% | -0.39% | -0.82% | +3.51% |
| cold Flower | +0.72% | +1.41% | -0.94% | -1.15% |
| corpus warm Keong macan | -0.07% | +0.48% | +5.77% | +0.07% |
| corpus warm Riaphotographs | +0.97% | +0.93% | +4.83% | +0.50% |
| corpus warm Bliznaca | +0.28% | -0.23% | +6.47% | +1.31% |

All values compare with the first packed policy; duplicate scatter is not
discarded or treated as a correction factor. The descriptive comparison
of per-round means of the two aligned versus two packed policies is also
retained in `s72_timing_tables.md`: warm 4K total +1.41%, 1080p +0.66%;
cold +0.93% / +2.47%. These aggregates are additional descriptions, not
substitutes for the individual results.

The release comparison uses retained S70 and candidate S72 binaries:
seven alternating pairs, three warmups/five samples for warm processes,
zero warmups/one sample for cold. The phase probe retains 41 fields and
public benchmark seven. All 168 process windows remain recorded:

| Release cohort / workload | Quantization paired median | Total paired median | Total wins / 7 |
|---|---:|---:|---:|
| Phase warm 4K | +0.10% | +1.13% | 3 |
| Phase warm 1080p | -0.50% | -0.33% | 4 |
| Phase warm Flower | -0.90% | -3.05% | 4 |
| Phase cold 4K | -0.50% | -0.38% | 4 |
| Phase cold 1080p | -1.56% | -4.30% | 5 |
| Phase cold Flower | +2.09% | +2.30% | 3 |
| Public warm 4K | -1.83% | -1.22% | 5 |
| Public warm 1080p | -1.46% | +0.08% | 3 |
| Public warm Flower | +2.18% | +2.64% | 2 |
| Public cold 4K | -1.62% | +1.68% | 3 |
| Public cold 1080p | +2.82% | +5.89% | 1 |
| Public cold Flower | -7.23% | -8.61% | 4 |

Public warm 4K quantization wins all seven pairs, with total wins in five,
but phase warm 4K and the same-binary controls do not corroborate a clear
net benefit. Cold public 1080p and warm public Flower regress. The broad
cold/small-case ranges, CPU codestream variance and duplicate-control
scatter prevent interpreting favorable individual cohorts as a general
speedup. Timing output-size checks are separate from the exact-byte
qualification above; no samples are discarded.

Twelve alternating Nsight Systems traces retain identical launch counts,
grids, blocks, registers, shared/local memory and copies. Exactly one arena
allocation grows by the independently expected amount. Each trace covers
one explicitly marked warm encode. Instrumented kernel sums change:

| Workload | LF Malta, two pairs | Full Malta, two pairs | All GPU kernel time, two pairs | Launches |
|---|---:|---:|---:|---:|
| 4K | -6.62% / -4.29% | -15.00% / +0.87% | -6.04% / -0.47% | 373 |
| 1080p | -1.28% / +3.76% | -6.66% / +2.20% | -3.74% / +0.84% | 360 |
| Flower | -4.68% / -4.47% | +2.25% / +2.13% | -0.20% / -0.26% | 373 |

The first 4K pair also moves unchanged non-target kernels substantially;
its whole reduction cannot be attributed to AC row layout. The complete
encoder LF effect is smaller than isolated replay, and 1080p reverses
between pairs. These are not public wall-clock results. Candidate 4K
independent median hot spots remain LF Malta 14.96 ms, low/medium vertical
convolution 10.87, fused Opsin 9.89, erosion/final 9.47 and full Malta 9.32.

A focused same-binary prepared diagnostic removes CPU codestream work.
Four prepared arenas remain resident, using the permanent synthetic
reference/distortion fixture at 3839x2159, 1919x1079 and 510x532; the last
is not the Flower photograph. Three warmups per arena precede 24 balanced
rounds, one synchronous comparison per event/wall window, 288 windows
total. Full maps/scores/guards match before and after every shape, with
unchanged inputs and zero timed allocations. All 203 GPU bodies equal the
candidate release. Events include host launch gaps, not only kernel time.

| Synthetic prepared extent | Aligned event median | Packed duplicate | Aligned duplicate |
|---|---:|---:|---:|
| 3839x2159 | +2.01% | +2.55% | +3.62% |
| 1919x1079 | +0.17% | +0.15% | -0.21% |
| 510x532 | -0.64% | +0.61% | +0.08% |

Wall results track the event results. Large outliers remain, including a
4K aligned-duplicate maximum +160.72%; simultaneous arenas and host gaps
limit attribution. This is a synthetic prepared calculation, not a fresh
captured encoder distortion. It does not establish a net resident gain.

Candidate-only serial/batch checks use one warmup and three alternating
samples, with exact internal codestream equality. At 1080p, sizes 1/2/4
give paired median speedups 1.027x / 1.614x / 1.345x fully resident and
1.026x / 1.830x / 1.992x maximum throughput. At 4K, fully resident sizes
1/2 give 0.931x / 1.012x. These compare serial with batch inside the
candidate, not S70 with S72. Batch-end thermal/power flags are active at
69 C / P3 / 240 MHz / 24.60 W. Endpoint states neither describe every
sample nor justify clock normalization.

Reject both the materialized split and production AC-row alignment for
this checkpoint. Alignment demonstrably helps narrow replay, but the
integrated evidence is insufficient to retain the extra layout/addressing
complexity and storage. No dimension-specific policy is inferred from
these mixed results. Future layout work needs a new hypothesis that
accounts for integrated staging, memory placement or launch gaps; a
different measured convolution bottleneck is also still open. The
backend is not considered maxed out.

### Preservation and restoration

The ignored `s72_*` bundle preserves all diagnostic sources/binaries,
candidate build/test/native logs, raw replay/quality/timing/trace data,
and the 39 candidate binaries/libraries with hashes. The four edited
sources matched the saved candidate snapshot before their known changes
were reversed. Production and tests again match S71/S70. Nineteen changed
build paths were restored from hash-verified retained S70 artifacts after
checking their candidate backups; all 39 retained artifacts now match
byte-for-byte. The candidate's successful qualification is not a claim
that its runtime was retained.

`s72_validate.py` reconstructs the native/guard/sanitizer evidence, all
replay windows and paired summaries, exact maps/memory, decoded quality,
serial performance order and medians, trace structure/allocation/copies,
and batch summaries. Historical candidate build paths resolve through
the preserved candidate manifest after restoration. The initial ASan-log
parser expected “passed” from a test that correctly prints “Verified 218”;
the failed parser log is retained and the corrected reconstruction passes.
No production test failure is hidden by that correction.

No new firewall, admin or permission block occurs. Slow racecheck runs
report progress and finish; optional restricted hardware counters are not
retried. No security, clock, cooling or power setting changes. The restored
runtime passes all 73 CUDA tests again in 117.66 seconds. Final evidence
hashes and seven current source/document snapshots accompany the checkpoint.

## Phase convolution channel halos (S73, not retained)

S73 starts from S72 documentation commit `b73ac42`, with the S70 runtime
unchanged. It investigates the retained 33-tap vertical low/medium pass,
not another Malta layout. The RTX 3060 Laptop / CUDA 11.8 / sm86 / MSVC
14.37 configuration is unchanged. No production source or library is
modified during this investigation.

### Channel lifetime and resource counterfactuals

The retained kernel stages three 32-column, 80-row halos for a 48-output-row
tile. It uses 30,856 shared bytes, 54 registers and three theoretical
256-thread blocks per SM. Three adjacent output rows share inputs while
retaining nine independent FMA chains. S62 already tested row counts,
tile sizes, thread counts and tap-major controls; shared-load count alone
did not explain its results.

X's low/medium outputs are independent of the joint Y/B outputs. The new
diagnostic computes them in two phases and reuses two shared halos. Both
channel orders are tested, at 48 and 96 output rows. Every output retains
the original tap order, included-weight sum, rounded division and final
expression. Cooperative-load and inter-phase barriers remain reachable by
threads in partial columns; normalization is read only after initialization.

| Mode | Vertical body | Registers | Shared bytes | Stack bytes | Theoretical blocks/SM |
|---|---|---:|---:|---:|---:|
| 0 | Retained 48-row joint body | 54 | 30,856 | 0 | 3 |
| 1 | 48 rows, X then Y/B | 73 | 20,616 | 0 | 3 |
| 2 | 48 rows, Y/B then X | 73 | 20,616 | 0 | 3 |
| 3 | 96 rows, X then Y/B | 73 | 32,904 | 0 | 3 |
| 4 | 96 rows, Y/B then X | 73 | 32,904 | 0 | 3 |
| 5 | Renamed, native-identical retained control | 54 | 30,856 | 0 | 3 |
| 6 | Mode 1 with four-block launch bounds | 63 | 20,616 | 0 | 4 |
| 7 | Mode 2 with four-block launch bounds | 64 | 20,616 | 16 | 4 |

The uncapped 48-row versions reduce shared storage but increase register
pressure enough to leave theoretical residency unchanged. This motivates
modes 6/7 rather than assuming the shared-memory reduction is sufficient.
Mode 7 has four static local loads and stores; the CUDA attribute API reports
16 local bytes while cuobjdump reports a 16-byte stack and zero LOCAL bytes.
Mode 6 has no stack/local traffic. These are compiler/API resource results,
not measured achieved occupancy.

The original/control body has 1,736 static instructions, 219 LDS instructions
and one barrier. Mode 4 has 2,312 / 228 / three, and mode 6 has 2,352 / 228 /
three. All contain 651 static FFMA instructions. Counts include interior and
edge paths, not executed instructions per pixel. Ideal interior logical
plane traffic, excluding weights and partial tiles, changes from 56 bytes
per output pixel at 48 rows to 52 at 96 rows. This is a reuse calculation,
not measured transactions or a bandwidth roof.

### Qualification and isolated measurements

The six-mode V1 contains 80 GPU bodies: 75 unchanged production bodies,
four new phase variants and the native-identical control. The eight-mode
V2 contains 82, with all 80 V1 bodies instruction-identical. Each mode
passes the permanent 460-fixture suite with three-stage reuse and two
bitwise oracles. That is 2,760 fixture executions / 16,560 two-oracle stage
comparisons in V1 and 3,680 / 22,080 in V2, not thousands of distinct images.
All 16 allocations, guards, inputs, weights and unused blurred planes are
checked. Every mode also passes both flattened-grid tall cases.

Memcheck, initcheck, synccheck and racecheck pass for both builds. Each V1
tool covers 96 fixtures / 576 two-oracle comparisons; each V2 tool covers
128 / 768. Memcheck explicitly enables stream-ordered race tracking and
full leak checking. The longest racecheck is 209.13 seconds with per-mode
progress, not a permission stall.

The focused V2 screen uses packed/padded 510x532, 1919x1079 and 3839x2159
synthetic inputs. Thirty-two balanced rotating/reverse rounds measure all
eight modes in one executable, three vertical kernels per graph window.
Every window is followed by exact comparison of all 16 allocations outside
the event interval: 1,536 windows and 42 paired summaries. The original
larger V1 screen scripts were prepared but never run.

| Mode | Packed 510x532 | Packed HD | Packed 4K | Padded 4K |
|---|---:|---:|---:|---:|
| 1 | -1.01% | +0.17% | -0.88% | -1.06% |
| 2 | 0.00% | +0.03% | -1.47% | -1.51% |
| 3 | +12.56% | -5.46% | -7.12% | -7.38% |
| 4 | +17.09% | -5.23% | -7.91% | -7.87% |
| 5, unchanged control | -0.25% | +0.14% | -0.14% | +0.04% |
| 6 | -2.51% | -5.53% | -6.04% | -5.85% |
| 7 | +3.02% | +0.14% | +0.60% | +0.46% |

Negative is faster. All individual samples remain, including paired
outliers above +700%; none are filtered or clock-normalized. The uncapped
48-row result is largely flat, while a larger tile or changed register
limit improves large-image stage measurements. Neither is yet an encoder
speedup claim.

### Captured inputs and integrated correctness

A diagnostic copy changes only the public low/medium host forwarding
function and appends the candidates plus an environment-controlled selector.
The phase probe, encoder and prepared test each contain 210 GPU bodies:
all 203 retained bodies are unchanged, and all 82 standalone probe bodies
match. This preserves the unrelated GPU work in the same executable.

Mode 0 captures the first six actual low/medium input sets from each of
the 4K, HD and Flower workflows: three packed XYB channels and 33 weights,
with original input/output strides recorded. These calls alternate full
and half resolution; they are not six Malta stages or three pyramid levels.
The half-size reference output is packed, while subsequent half-size outputs
use the full working-width stride. Synchronized capture runtimes are not
performance evidence.

A separate replay restores those 18 input sets and original strides, while
retaining diagnostic guard offsets and separate allocations rather than the
integrated arena's exact base addresses. Modes 0/4/6/5 use 24 balanced rounds
and three horizontal-plus-vertical repetitions per event window. All 1,728
windows match all 16 allocations exactly; 54 paired summaries retain every
outlier. All 82 GPU bodies equal V2. Only its `--replay` path is used: copied,
unused `--scope`/`--timing` labels still describe the earlier eight-mode
probe and are not qualification claims for this binary.

Mode 4 improves the three full-resolution 4K two-pass captures by
5.41% / 5.44% / 5.23%; mode 6 improves them by 3.64% / 3.73% / 3.49%.
Both improve all six HD captures. On Flower, mode 4 regresses full-size
work 4.05-6.70% and half-size work 17.13-18.72%. Mode 6 modestly improves
full-size work but is mixed/slower at half size. Those small-image results
preclude treating either fixed dispatch as a universal improvement.

The first integrated prepared test fails its memory-accounting assertion
even in mode 0. The diagnostic link reused a build-tree test object from
S72's aligned-layout candidate, written before source restoration. The
retained libraries/binaries and current source were correct; the stale host
fixture expected different storage. That object and failed test remain.
Rebuilding a current-source snapshot initially misses the relative test
helper include directory; the failed build and script also remain. The
corrected build supplies `-I tests` and passes. Its 210 GPU bodies and
resource dump are byte-identical to the first diagnostic test executable.

All four selected modes then pass 31 prepared-workflow cases each. At
distance 1.2 / effort 7, with and without final-score collection, 24 fresh
encodes across 4K/HD/Flower exactly match the retained S70 bitstream hashes,
strategy text and final-score text. These are not fresh decoder/metric
runs or the full release qualification matrix. No permanent fixture or
production assertion is weakened to obtain a pass.

### Complete-workflow timings and traces

Warm runs use three warmups/seven samples; cold runs use zero/one. Eight
balanced rotating/reverse rounds compare modes 0/4/6/5 serially per workload
and cohort. All 41 phase fields and raw reports remain: 192 process windows
and 54 paired summaries. Size equality is checked in timing runs; bitstream
equality comes from the separate preflight. The table reports median paired
percent changes relative to mode 0, not differences of aggregate medians.

| Workload / mode | Warm quantization | Warm whole encode | Cold quantization | Cold whole encode |
|---|---:|---:|---:|---:|
| 4K / 4 | +0.02% | +0.48% | -2.50% | -3.19% |
| 4K / 6 | -0.12% | -2.38% | +0.43% | -2.44% |
| 4K / unchanged control | -0.25% | -0.90% | +0.42% | +1.93% |
| HD / 4 | -1.15% | -1.16% | -2.87% | -2.09% |
| HD / 6 | -0.11% | +0.60% | -0.84% | -0.67% |
| HD / unchanged control | +0.05% | +0.62% | -3.20% | -3.15% |
| Flower / 4 | +0.33% | -2.67% | +0.86% | -0.14% |
| Flower / 6 | +1.60% | +1.89% | +12.32% | +10.63% |
| Flower / unchanged control | -1.67% | -2.59% | +1.05% | +0.89% |

Mode 6's warm 4K whole reduction is not attributable entirely to the target:
quantization is essentially flat and unchanged codestream work also moves.
Mode 4's cold HD whole reduction is smaller than the unchanged control's.
Raw ranges, win counts and direct comparisons with mode 5 are preserved;
no slow samples or adverse workloads are removed. The cohorts are not
pooled or normalized using endpoint device clocks.

Eight integrated warm 4K traces repeat the four modes in forward/reverse
order. All contain 373 launches, including six target vertical calls.
Allocation sizes, copy totals/counts and every non-target launch's ordered
grid/block/register/shared/local structure match. Only the expected target
name, grid and resources differ. The 96-row target has grids 2,760/720
instead of 5,400/1,380; mode 6 keeps the original grid. These are instrumented
diagnostics, not unprofiled wall-clock measurements.

| Mode | Target GPU change, pairs 1 / 2 | Total GPU change, pairs 1 / 2 |
|---|---:|---:|
| 4 | -6.28% / -2.93% | +2.55% / +5.14% |
| 6 | +8.37% / -3.00% | +11.71% / +6.78% |
| 5, unchanged control | +2.59% / +3.24% | +4.57% / +1.36% |

The taller variant's target gain survives integration in both pairs, but
non-target GPU work also moves substantially. Mode 6 reverses at the target
and has a visible cold small-image regression. This evidence does not
establish a consistent net benefit for either tested fixed dispatch.

### Disposition and preservation

Neither fixed-dispatch candidate is retained. Production sources, tests and
all 39 retained S70 binaries/libraries remain unchanged; no restoration is
needed for this checkpoint. The shared-storage reduction alone is not a
speedup: the useful counterfactuals also change tile reuse or register-limited
residency. The taller version has a repeatable large-image stage gain, but
small-image regressions and mixed complete-workflow results remain.

A size-dependent policy combining the retained small-image body with a
large-image phase body is an open experiment, not an inferred production
policy. It would need its own same-binary controls, integrated measurements
and release qualification. Exact arena placement and other integrated
convolution costs remain potential hypotheses. The backend is not considered
maxed out; this checkpoint closes the tested fixed-dispatch variants only.

The ignored `s73_*` bundle preserves both prototype versions, resource and
native audits, eight sanitizer runs, all guarded/tall results, synthetic and
captured-input timing windows, 24 encoded outputs, 192 complete-workflow
windows, eight traces, and all failed diagnostic build/test evidence.
Five retained source/header snapshots and two current document snapshots
accompany final artifact, binary and dependency hashes. The unused proposed
V1 screens are marked unexecuted, not included in completed timing counts.

`s73_qualification_audit.py` and `s73_integrated_audit.py` validate fixture,
sanitizer, native/source identity, capture, bitstream, raw timing-order/median
and trace evidence. `s73_validate.py --frozen` verifies the preserved bundle;
adding `--current` also rechecks the source snapshots, retained runtime and
reconstruction against the current worktree. This checkpoint changes docs
only and does not claim a fresh full CUDA/CPU/ASan release suite.

No firewall, admin or permission block is observed. The baseline memory
assertion and missing test include are reported and corrected in diagnostic
builds, not hidden behind a passing summary. Slow sanitizer runs make
progress and terminate. Restricted hardware counters are not retried; no
security, clock, power or cooling settings are changed.

## Provisional resident metadata cost (S74, investigation)

S74 investigates a host preparation gap before changing more convolution
tiles. The retained runtime is still S70; this checkpoint changes no
production source, test, library or executable.

### Critical-path attribution

A read-only analysis of all eight frozen S73 warm 4K traces includes kernels,
copies and memsets in the GPU interval union. Each trace has 373 kernels and
54 copies. Gaps total 48.67-60.23 ms; 46.39-58.04 ms occurs where the next
correlated API starts after the previous GPU operation ends. The largest gap,
18.73-27.18 ms, lies between a 16-byte input-preparation result readback and a
1,036,800-byte upload of provisional DCT8 anchors. Kernel-to-kernel gaps total
only 1.50-1.70 ms. Thus simply capturing consecutive kernel launches in a
CUDA Graph would not address most of the observed idle span. The intervening
host-result dependencies still matter. Event synchronization API durations
overlap GPU execution and are not additive savings; profiler-start duration
is capture overhead outside the useful GPU span.

An ignored CPU-only resident-evaluator override adds nested steady-clock
scopes. Both diagnostic executables preserve all 203 retained GPU bodies.
Removing the scope include and declarations exactly reconstructs the current
production source. Three public encodes match retained bitstreams; each of
three phase cohorts has one warmup and three subsequent encodes. Splitting
metadata scopes by lifecycle, excluding that warmup, gives:

| Input | Initial provisional metadata, median ms | Replacement metadata, median ms |
| --- | ---: | ---: |
| Padded 4K | 8.1157 | 1.8728 |
| Padded 1080p | 2.3815 | 0.5346 |
| Flower | 0.2667 | 0.1470 |

The original grouped scope summary mixes initial and replacement invocations;
`s74_host_roles.py` provides the explicit split. Nested scope durations
include child work/logging and GPU waits. They are neither independent costs
nor uninstrumented speedup measurements.

The fully resident frontend initially prepares an all-DCT8 grid. Its initial
quantization and AC-strategy input views use image, mask, quant-field and CfL
buffers, not transform metadata. After selecting the actual strategies, the
encoder calls `Reconfigure` before adjustment or encoding policy evaluation.
The initial metadata is discarded. Arena capacities depend on image geometry
and maximum block counts, not the provisional anchor list.

### Encoder-only counterfactual

`s74_defer_aq_resident.cpp` is a diagnostic override, not a generic lazy
prepared-evaluator implementation. Mode 1 omits initial metadata construction,
upload and commit for complete resident frontends, but still uploads quant
tables and allocates the same arenas. Guards reject metadata-dependent public
operations until a successful `Reconfigure` commits the replacement. Modes 0
and 2 take the same eager path; mode 2 is a duplicate baseline for process
noise. Both executables again preserve all 203 retained GPU bodies.

All 18 public d1.2/e7 encodes (4K, 1080p and flower; scored and encoding-only;
three modes) match retained S70 bitstreams, strategy reports and score text.
No premature metadata-use guard fires. This does not qualify generic direct
prepared use, invalid-input semantics, a fresh decoder/metric comparison or
the complete release suite.

The first 54-window warm timing cohort accidentally overlaps the end of a
CPU native-code audit. It is preserved as shakedown evidence and excluded
in full, not selectively trimmed. Separate clean warm and cold cohorts run
only after both producers terminate: 54 serial process windows each, all six
orders of the three modes, three inputs. Warm windows use three warmups and
seven samples; cold windows use zero warmups and one sample. All 41 timing
fields and encoded-size checks are preserved. These are same-binary
counterfactual measurements, not retained production speedups.

| Input | Warm quantization change | Warm total change | Cold quantization change | Cold total change |
| --- | ---: | ---: | ---: | ---: |
| Padded 4K | -3.62% | -2.39% | -0.37% | -0.33% |
| Padded 1080p | -6.64% | -3.54% | -2.54% | -2.88% |
| Flower | -4.51% | -1.47% | -7.05% | -5.15% |

Entries are medians of six candidate/control ratios. Warm quantization saves
8.96 ms at 4K and 3.29 ms at 1080p, winning all six pairs in both cases.
Against duplicate mode 2, those phase changes are -5.60% and -4.85%.
Whole-workflow effects are less decisive: the identical warm 4K duplicate
itself moves +2.41% in total time, and candidate warm 4K total wins five of
six pairs. Cold 4K is inconclusive. Small-image and serialization variation
remain visible; no median is a promise for other workloads or machines.

### Qualification limits and next implementation

Two 1080p memcheck attempts produce exact retained bitstreams and zero-error,
zero-leak tool summaries, but omit the application's mode/completion text.
The first combined file redirection and the second separate tool-log/piped
application-output approach both fail the wrapper's marker assertion. Neither
is counted as completed sanitizer qualification, and the intended six-check
matrix does not run. Both failed wrappers, reports, logs and outputs remain.
The cause of the missing application output is unresolved. No active admin
prompt or matching application-crash event is found in the scoped read-only
check; there is no evidence establishing a firewall cause. No permissions or
security settings are changed, and restricted counters are not retried.

The measured redundant work warrants an API-preserving lazy implementation
experiment. It must retain owned initial state, reject invalid metadata at
preparation as before, realize metadata under the existing operation lock
before any dependent use, and let `Reconfigure` replace pending state without
constructing the discarded plan. A narrow all-DCT8 frontend fast path could
avoid general deferred validation; an explicit opt-in hint is another design
to evaluate. Neither is implemented or inferred safe by this counterfactual.
Required tests include direct use without reconfiguration, caller input
lifetime, invalid EPF/strategy grids, failed reconfiguration, repeated use,
submission failure and unchanged allocation/output contracts. The backend
is not considered maxed out.

The ignored `s74_*` bundle preserves host attribution, original and clean
timing cohorts, all diagnostic binaries/sources, exact-output checks and
failed sanitizer evidence. Frozen hashes cover dependencies, retained runtime
and source/document snapshots; `s74_validate.py --frozen` checks the bundle,
with `--current` additionally checking the current source/runtime snapshots.

## Defer provisional DCT8 metadata (S75)

S75 retains an API-preserving implementation of the S74 finding. Only a
complete resident frontend with an all-DCT8 grid can defer metadata. The
backend checks active EPF values and conservatively bounds batch launches and
record offsets before device allocation. Mixed grids and large geometries use
the original eager builder and its validation. No public option or signature
changes, and exact/frame-only preparation is unaffected.

The prepared object owns copies of the initial strategy grid and active EPF
values. It reserves the same device arenas and uploads quantization tables,
but postpones construction and upload of the transform plan. Initial field
generation and strategy-input views do not require that plan. `Reconfigure`
can replace pending state directly; if a caller instead uses the initial grid,
`EnsureMetadata` builds/uploads/commits it under the existing operation mutex
before adjustment, resident policy setup, evaluation or host-input policy.
Host build failure leaves pending state retryable; transfer failure invalidates
the object, as in existing reconfiguration. There is no steady-state device
allocation. Direct first use still performs the necessary plan construction;
this change saves discarded frontend work, not all metadata work.

### Qualification and the S74 logging issue

The new CUDA AQ fixture checks invalid EPF, incomplete grids and color-tile
crossing before any allocation/submission. It exercises seven first-use cases
against explicitly reconfigured frontend objects: direct evaluation,
adjustment, resident policy setup, host-input policy, replacement while
pending, submission failure and completion failure. Caller strategy/EPF
storage is overwritten and destroyed before use, and invalid reconfiguration
must preserve pending state. Successful cases compare fields, masks, block
maps, scores, quantizers and encoded frames across eager/lazy and repeated
calls. Failure cases check unchanged caller outputs and invalidation. Internal
state queries are quiescent test hooks, not new public capabilities.

The rebuilt release passes all 73 CUDA-enabled tests and all 50 CPU-only
tests. The initial CPU command selected a benchmark-only build with zero
tests; that log is preserved and is not counted as a pass. The corrected
50-test run is separate. Full AQ memcheck, initcheck and synccheck include the
new lifecycle fixtures and pass with zero reported errors/leaks. No new host
ASan or GPU racecheck run is claimed for this host-only change.

All 58 fresh candidate encodes, independent decodes and perceptual-metric
runs match the frozen S70 bitstreams, decoded pixels, strategy/score reports
and metric values. The matrix retains the three distances, effort-9 cases,
high-density and maximum-compression variants of that corpus. Another 18
same-binary control encodes match retained outputs. These checks supplement,
rather than replace, the invalid-input and lifecycle tests.

The missing S74 sanitizer application output is resolved by explicit flushing
in ignored copies of the diagnostic evaluator and encoder. Those are the only
source differences: flush the mode marker and the final console report. All
six planned full scored memcheck/initcheck encodes then capture the expected
markers, report zero errors/leaks and produce exact retained bitstreams.
The two failed S74 wrappers remain unchanged; they are not retrospectively
counted. Production console behavior is not changed by this investigation.

Five executable native audits preserve all 203 retained S70 GPU bodies,
including the flush-only diagnostic, same-binary controls and production
encoder. Removing the diagnostic include/helper call exactly reconstructs
the production resident source. The full release rebuild also refreshes the
previously restored kernel/test objects; stale S72 objects are not relied on.
No permission/firewall block occurs and no security, counter-access, clock,
power or cooling setting is changed.

### Complete-workflow measurements

All builds, native audits and correctness jobs finish before timing starts.
The same-binary comparison selects eager modes 0/2 or production lazy mode 1;
mode 2 is a duplicate eager control. Warm/cold cohorts have 54 process windows
each, with every ordering of three modes for each input. Warm windows use
three warmups/seven samples; cold windows use zero/one. Separately built
public benchmarks compare retained S70 and production with eight alternating
pairs per input, three warmups/five samples or zero/one: another 96 windows.
All 204 windows, raw fields, encoded-size checks, adverse ranges and duplicate
results are retained. Encoded sizes alone are not the correctness gate.

| Input | Same-binary warm quantization / total | Public warm quantization / total | Same-binary cold total | Public cold total |
| --- | ---: | ---: | ---: | ---: |
| Padded 4K | -4.22% / -3.12% | -3.66% / -1.77% | +5.63% | -6.03% |
| Padded 1080p | -5.93% / -2.73% | -5.07% / -3.46% | -1.09% | -0.81% |
| Flower | -2.78% / -1.79% | -1.19% / +3.57% | +5.87% | +2.57% |

Entries are medians of paired ratios. Same-binary warm 4K quantization saves
10.30 ms and total saves 11.04 ms, both winning all six pairs; the phase gain
against duplicate mode 2 is 3.55%. Warm 1080p phase and total also win all six
pairs. Public warm 4K total wins all eight pairs; 1080p wins six. This supports
retaining the large-image warm improvement, not a universal speedup claim.
Cold total results are mixed, and the small-image public totals regress while
same-binary warm total improves. CPU serialization variation remains large.
No cold or small-image net benefit is asserted, and those cases remain open
for investigation rather than excluded from the evidence.

### Transfer and critical-path evidence

Four reverse-order warm 4K captures preserve all 373 kernel launches and
their dimensions/register/shared-memory fingerprints. Allocation API counts
and four memsets are unchanged. Copy count drops from 54 to 49, removing only
the provisional anchors (1,036,800 bytes), EPF (129,600), color records
(3,110,400), color-tile offsets (8,164) and packing offsets (1,036,800):
5,321,764 host-to-device bytes per encode. The entire remaining ordered copy
sequence is identical in both paired comparisons.

The frontend GPU gap changes from 26.01 to 10.20 ms in one pair and from
30.20 to 18.53 ms in the other. These instrumented intervals support the host
attribution; they are not added to unprofiled encoding savings. GPU arithmetic
has not been accelerated or changed. Remaining host strategy preparation,
readback/frame assembly, serialization and kernel costs are still relevant;
the backend is not considered maxed out.

The ignored `s75_*` bundle preserves sources, binaries, all passing and failed
diagnostics, raw measurements, decoded outputs and traces. Reconstruction and
frozen artifact/dependency/runtime/source hashes are checked by
`s75_validate.py --frozen`; `--current` additionally checks the current
retained source/runtime snapshots. S70 remains preserved as the parent
runtime, while the S75 candidate is now retained.

## Interleaved small-image metadata controls (S76, investigation)

S76 investigates S75's adverse small-image public results without changing
production code or introducing an image-size gate. The earlier warm public
Flower total/serializer regressions of 3.57%/9.07%, and cold total regression
of 2.57%, remain preserved. They are not relabeled as passing measurements.
S75's same-binary warm result had instead improved, motivating a test that
switches metadata behavior between individual encodes inside one process.

### Probe and qualification

The new ignored `s76_interleave.exe` links the frozen S75 control object.
Modes 0 and 2 both use eager metadata; mode 1 uses the retained, validated,
owned-state deferred implementation. The probe reuses the frozen 41-field
phase mapper and checked profiled encoding boundary. All 203 native GPU
function bodies match S70, and the reused control source/object/header hashes
match the frozen S75 bundle. No device arithmetic or production policy changes.

Each process reads one input, obtains an eager reference, runs six warmup
triples, then runs measured triples. Every block of six triples visits all
six mode orders once, in deterministically shuffled order. Each encode must
match its process's reference bytes and summary exactly. The first bitstream
is saved and must match the frozen S70 distance-1.2, effort-7 encoding-only
reference hash and size. The three inputs are Flower, padded 1080p and padded
4K. No scored, other-effort, or other-image performance claim is added here.

Six preflight processes pass 222 encodes. Twelve subsequent measurement
processes pass 3,168 measured encodes plus 228 warmup/reference encodes, for
3,618 exact encodes altogether. All 18 saved reference files match S70.
All 3,600 logged timing rows, all 41 fields, mode markers, orders and paired
summaries reconstruct from raw output. Warmups and adverse measurements are
retained; none of the measured samples is excluded. Percentage comparisons
are undefined when a tiny baseline field rounds to zero; these cases are
counted explicitly and retain their absolute timings.

Flower has two 120-triple replicates for each of four configurations:
persistent versus newly created backend, and automatic versus one CPU
participant. The second replicate reverses configuration order. Four large
controls between the Flower replicates each use 24 triples, automatic CPU
threading, and persistent or fresh backends. Native auditing and preflight
finish before measurement; all measurement children execute serially from
23:02:57 to 23:05:56 UTC on September 6. Every child completes within its
120-second limit. No permission failure, elevation prompt, or firewall block
is observed, and no security or system scheduling setting changes.

Fresh backend creation/destruction occurs on every encode **within an already
warm process**. It is not a cold-process experiment. The profiled total excludes
backend construction/destruction; a separate outer timer includes them. The
control's flushed mode marker is inside preparation timing, equally for all
three modes. Row formatting and byte/summary comparison are outside the
encoding call, although buffered row output can flush with the next marker.
These are diagnostic, instrumented same-binary comparisons, not replacement
public-release benchmarks. S75's full tests, decoded quality and sanitizers
remain the production qualification; S76 adds no fresh sanitizer or decoder
run and revalidates the frozen S75 evidence.

### Results and decision

Each percentage below is the median of adjacent, paired per-triple changes,
not a ratio of separate timing medians. Negative values are faster. Flower
rows combine both replicates, with 240 pairs per comparison.

| Flower backend / CPU budget | Quantization | Serializer | Total | Duplicate eager total |
| --- | ---: | ---: | ---: | ---: |
| Persistent / automatic | -2.97% | -0.06% | -2.18% | -0.04% |
| Fresh / automatic | -2.54% | -0.19% | -1.69% | -0.08% |
| Persistent / one | -2.84% | +0.78% | -0.77% | -0.20% |
| Fresh / one | -3.04% | -0.09% | -1.56% | -0.13% |

Automatic-thread Flower saves a median paired 0.326/0.361 ms in quantization
and 0.452/0.468 ms total for persistent/fresh backends. Total improves in
152/240 and 145/240 pairs; quantization improves in 177/240 and 172/240.
Comparison to duplicate eager mode 2 also improves total by 2.25%/1.65%.
All six order-specific quantization medians improve in both automatic-thread
configurations. Descriptive 2,000-resample, six-triple-block bootstrap
intervals for total change are [-2.88%, -1.16%] and [-2.84%, -0.40%]; serializer
intervals cross zero. These within-cohort intervals do not establish behavior
across cold starts, other machines or workloads.

Each of the eight Flower process-level paired total medians is negative,
but persistent single-thread replicate 1 is essentially flat (-0.04%).
Single-thread serialization is not generally faster: eager serializer
medians are 11.03/10.68 ms versus automatic 9.11/9.14 ms for persistent/fresh
configurations. Those absolute cross-process observations are not an
interleaved CPU-policy experiment. The source still creates and joins
`std::thread` workers in `RunParallelSections`; this is a possible future
profiling target, not evidence for forcing serial execution or adding a pool.

The smaller large-image controls also favor deferred metadata. Persistent /
fresh 1080p paired quantization changes are -4.41% / -3.49%, and total changes
-4.66% / -1.38%. At 4K they are -3.98% / -2.35% for quantization and -5.42% /
-2.16% total. Duplicate eager total changes range from -2.43% to +1.49%, so
these four-block controls remain descriptive; they do not improve the
precision of S75's public large-image results.

The warm-process data do not reproduce a metadata-induced small-image
slowdown. They support retaining S75 without an image-size gate. They do not
identify the cause of the older public regressions or settle cold-process
behavior. No global CPU-thread policy or runtime implementation is changed.
Further work should directly measure remaining host preparation, serializer
work and worker-lifecycle costs, rather than infer them from cross-process
noise. The overall CUDA optimization goal remains open.

The ignored `s76_*` bundle preserves the diagnostic source/object/executable,
producers, native dump, all raw runs and reconstructed results. Its seven
source/document snapshots, diagnostic files and dependencies are hashed;
all 39 retained runtime artifacts remain byte-identical to S75 and reuse
its immutable snapshots. `s76_validate.py --frozen` verifies the bundle and
reconstructs every recorded comparison; `--current` also checks current
source/runtime identity.

## Serializer worker lifecycle (S77, not retained)

S77 follows S76 with direct scheduling evidence. `RunParallelSections`
creates and joins new `std::thread` workers at multiple serializer stages.
Automatic mode spawns every participant and leaves the caller waiting;
explicit CPU budgets already use the caller as one participant. No
production implementation or thread policy changes in this checkpoint.

An ignored serializer override records each call's source location, task and
participant counts, thread-creation/join spans, and per-worker lifetime and
task count. Records are collected in memory and printed after the encode.
The analysis uses unions of overlapping worker intervals, not the sum of
launch and join times. Top-level call intervals are disjoint; nested intervals
would be reported separately. All 42 instrumented encodes pass exact bytes
against S70 and repeated summary equality, across Flower/1080p/4K and
automatic/single-CPU configurations, with two warmups and five samples each.

| Automatic CPU scheduling | Flower | Padded 1080p | Padded 4K |
| --- | ---: | ---: | ---: |
| Serializer calls per encode | 5 | 5 | 5 |
| Threads spawned per encode | 17 | 21 | 25 |
| Median summed call time with no worker active | 0.807 ms | 1.122 ms | 1.624 ms |
| Median serializer time | 8.427 ms | 22.488 ms | 50.333 ms |

No zero-task or late-start-after-all-task-workers-finish worker appears in
these probes. Single-CPU serialization takes 10.855/33.474/101.367 ms, despite
much smaller no-worker intervals. The gaps include call setup and cleanup,
not just OS thread startup; work can also be descheduled inside a recorded
worker lifetime. Thus these are instrumented attribution observations, not
an additive prediction of removable overhead or public speedups.

### Two caller-participation counterfactuals

The first override uses the caller for every automatic parallel stage,
spawning N-1 instead of N workers with the same participant limit and queue.
The second enables this only for the two-task block-context/coefficient-order
preparation call. Explicit budgets remain unchanged. Each executable has
mode 0 original, mode 1 candidate and mode 2 identical-original control.
There is no image-size gate. Mode markers and flushed timing rows are outside
the timed encode, and the scheduling instrumentation is absent.

Each candidate passes 222 preflight encodes, then 1,296 measured encodes plus
114 warmup/reference encodes. Together with the instrumented probe, all
3,306 encodes preserve bytes and per-process summaries; all 30 saved reference
bitstreams match the corresponding S70 hash and size. Three native/source
audits preserve all 203 GPU function bodies and reconstruct exactly the
instrumentation or scheduling-only source differences. No new decoder,
sanitizer or production test-suite run is claimed. All 39 retained runtime
artifacts remain byte-identical to S75; the frozen S76 evidence revalidates.

Each candidate has two 120-triple Flower replicates and two 48-triple
replicates per large input. Every six-triple block visits all six mode orders;
the second replicate reverses input order. Native audits, preflight and
measurements are serial. Broad-caller measurement runs from 23:20:35 to
23:23:11 UTC on September 6; targeted measurement runs from 23:28:05 to
23:30:43. Every child completes within its 120-second limit. All raw warmups,
adverse samples, duplicate controls and 41 phase fields are retained.

The table combines both replicates and reports medians of per-triple paired
changes, not ratios of separate medians. Negative values are faster.

| Candidate / input | AC tokenization | Serializer | Total | Duplicate-original total |
| --- | ---: | ---: | ---: | ---: |
| All stages / Flower | -11.17% | -4.25% | -1.99% | -0.50% |
| All stages / 1080p | -2.47% | -0.83% | -0.04% | +0.31% |
| All stages / 4K | -2.79% | -2.16% | -0.40% | -1.58% |
| Preparation only / Flower | -10.32% | -2.79% | -1.55% | -0.02% |
| Preparation only / 1080p | -0.53% | +0.17% | -0.18% | +0.52% |
| Preparation only / 4K | -0.97% | +0.37% | +1.22% | +0.36% |

Broad-caller Flower serializer improvement repeats at -4.25% in both
replicates, with combined paired savings of 0.395 ms; total saves 0.401 ms.
Targeted Flower serializer saves 0.262 ms and total 0.313 ms. Both retain
Flower benefits relative to duplicate mode 2. Six-triple-block descriptive
bootstrap intervals for Flower total are [-2.58%, -1.41%] and
[-2.18%, -0.43%], respectively; quantization intervals cross zero.
These do not establish cross-machine or cold-process behavior.

The larger-input results do not establish a net win. Broad-caller 4K is
0.80% slower in total and 1.32% slower in serialization relative to duplicate
mode 2, despite looking faster relative to mode 0. Targeted 4K total regresses
in both replicates (+1.06%/+1.49%); relative to duplicate mode 2 its combined
total is almost flat (+0.12%). Unchanged quantization timings also move,
showing residual within-process variation. Neither candidate is retained,
and no small-image gate is inferred from this limited input set.

Before a general caller policy could be retained, higher-effort nested work,
concurrency and participant accounting would need explicit qualification:
entering another `ParallelScope` on an already-participating caller can count
the same physical thread twice. This was a source-inspection concern, not a
failure observed in these five-call effort-7 runs.

The evidence bounds a real but modest scheduling opportunity. Coefficient-
order work remains about 8.45 ms in the broad-control 4K baseline. GPU zero-
population accumulation was already identified after S53 but not implemented
or measured there; it is a larger remaining host-scan lead to investigate
with exact sampling/order semantics and a public-boundary gate. No new GPU
histogram or worker pool has been implemented here. Optimization remains open.

The ignored `s77_*` bundle contains eight diagnostic binary/object files,
seven source/document snapshots, all producers and raw evidence. Its
`s77_validate.py --frozen` checks hashes, all worker intervals, all paired
rows/fields/orders, three native/source audits and the serial job timeline;
`--current` additionally verifies current source/runtime identities. No
firewall, elevation or permission block is observed, and no security,
power, clock, cooling, priority or service setting changes.

## GPU coefficient-order population replay (S78, investigation)

S78 starts from S77 `9a7381f` on the same RTX 3060 Laptop / CUDA 11.8 /
MSVC 14.37 / `sm_86` system. It qualifies diagnostic GPU counters, not a
resident integration. Production source and all 39 retained S75 runtime
artifacts remain unchanged. No public encoding speedup is established here.

### Isolate the host scan

An instrumented copy of `coefficient_order.cpp` separates strategy-presence
detection, zero counting, stable order sorting and validation. Four real
encoding-only distance-1.2 workflows each run two warmups and five measured
encodes. All 28 preserve S70 bytes and repeated summaries. The first warmup
exports the completed quantized frame and CPU populations; export I/O is
outside the five-sample statistics. All 203 GPU bodies remain identical to
S70, and removing only the explicit instrumentation reconstructs the current
production source exactly.

Median host subphases, milliseconds:

| Input / effort | Presence | Count | Sort | Validation |
| --- | ---: | ---: | ---: | ---: |
| Padded 1080p / 7 | 0.2988 | 1.6928 | 0.0754 | 0.0220 |
| Padded 4K / 7 | 1.0560 | 6.1737 | 0.0866 | 0.0185 |
| Flower / 7 | 0.0536 | 0.2086 | 0.0809 | 0.0285 |
| Flower / 9 | 0.0003 | 0.2283 | 0.0977 | 0.0321 |

All four real frames contain mixed transform families, so even effort 7
counts every anchor. Only an entirely DCT8 frame uses the existing sampled
policy. The 4K presence/count medians sum to 7.23 ms, versus about 0.105 ms
for sorting/validation. S53 already narrowed and vectorized the CPU count;
this evidence supports moving the scan, without changing float-scaled keys,
stable ties, LLF prefixes or sampling semantics.

### Exact replay and competing kernels

Four real exports and 104 synthetic exports form a 108-frame replay corpus.
The synthetic cases cover all seven supported transform shapes and mixed
grids, both full/sampled policies, 8/36/68-block sides, group boundaries,
all-zero/nonzero, sparse/dense/tied populations, extreme signed coefficients,
LLF-only and last-coefficient cases. Each replay independently recounts the
exported coefficients with a scalar CPU implementation and validates the
PRNG decisions in AC-group-first anchor order before sorting descriptors
into GPU family batches. Output is 5,952 exact unsigned 32-bit counts,
23,808 bytes; oversized counts require the existing wide CPU fallback.

Three diagnostic recipes use ordinary integer arithmetic:

- One 256-thread block per anchor, globally atomically adding each zero.
- One launch per family, with 32 coefficient lanes by eight anchor rows;
  shared reduction combines up to 64 anchors before global atomic adds.
- The same reduction with all family batches dispatched in one launch;
  power-of-two coefficient tile sizes permit mask/shift indexing.

These respectively use 20/29/32 registers and 0/3,072/3,072 shared bytes,
with no stack, local storage or spills. Original atomic/tiled native bodies
are identical across all executables; the final comparison binary contains
the same three GPU bodies as the preceding unified-launch binary.

Each of three replay executables passes all 108 cases and full-corpus
memcheck, initcheck, synccheck and racecheck: 12 sanitizer jobs, zero reported
errors/leaks/hazards. Every output bin matches. Guards and unchanged device
coefficient/anchor sources are checked after each frame case. Empty/null
launcher smoke checks also pass, but are not exhaustive API qualification.
This is not a fresh full production test-suite or decoder qualification.

### Same-binary comparison and limits

Earlier separate-executable cohorts are preserved, including an unchanged
atomic 4K clear/count/readback shift from 1.152544 to 0.497296 ms. That
variation prevents using those cohorts to rank the two reduction layouts.
The final comparison therefore puts atomic, separate-family tiled, unified
and duplicate atomic into modes 0/1/2/3 of one binary. Each input gets two
process replicates, with reversed input order in the second. Each process
runs 24 warmup and 48 measured quadruples, covering all 24 permutations once
during warmup and twice during measurement: 2,304 total replay rows, 1,536
measured rows, 96 paired measured quadruples per combined input.

CUDA-event clear/count/readback medians, milliseconds:

| Input / effort | Atomic 0 | Tiled 1 | Unified 2 | Atomic 3 |
| --- | ---: | ---: | ---: | ---: |
| Padded 1080p / 7 | 0.181920 | 0.213248 | 0.181168 | 0.181712 |
| Padded 4K / 7 | 0.522432 | 0.540032 | 0.497392 | 0.523472 |
| Flower / 7 | 0.091952 | 0.117568 | 0.076608 | 0.093744 |
| Flower / 9 | 0.082208 | 0.116576 | 0.073056 | 0.092656 |

Unified paired changes versus atomic 0 are +0.67%, -4.49%, -16.06%, -18.56%,
and versus duplicate atomic 3 are +0.54%, -5.43%, -17.87%, -20.28% in table
order. These are medians of within-quadruple ratios, not ratios of the
independent medians above. Duplicate 3 versus 0 moves +0.07%, +0.12%,
+2.69%, +3.36%. Separate-family tiled is slower than atomic 0 on all four
combined inputs. Unified 4K replicates are 0.496000/0.498624 ms; unified
host-wall medians are 0.19345/0.51095/0.08785/0.08235 ms in table order.

The unified recipe is the preferred integration candidate, roughly tied at
1080p and better on 4K/Flower in this comparison, not a universal kernel win.
Replay excludes coefficient upload, descriptor preparation/sorting, CPU
recount, startup and actual resident-pipeline integration. CUDA-event ranges
include clear, kernel launch gaps and the 23,808-byte readback, not just GPU
arithmetic. No dedicated thermal telemetry accompanies these short cohorts;
duplicate controls do not eliminate machine-state uncertainty. Do not
subtract replay duration from a host phase and claim a public encode gain.

The initial tiny 3x2-block case correctly takes the natural-order cutoff and
does not export a frame. Seven internal encodes pass, but the export wrapper
fails with `FileNotFoundError`; its incomplete report and traceback remain.
A later read-only check confirms its saved 237-byte JXL matches S70 without
reclassifying the failed export. V2 uses separate paths and excludes this
cutoff case. The first scratch GPU build also fails before compilation due
to a malformed escaped Windows CUDA path; its script/log remain alongside
the successful forward-slash-path build. Scratch bounds/signedness fixes
were rebuilt before qualification, without replacing measured evidence.
Neither setup failure establishes a firewall or administrator-prompt cause.

The ignored `s78_*` bundle includes all producers, 108 frame exports, raw
logs, analysis, 13 diagnostic binary/object files and 12 source/document
snapshots. `s78_validate.py --frozen` reconstructs timing rows/orders/pairs,
native identities, source instrumentation, the two failed setup records and
141 nonoverlapping recorded native/encode/export/GPU job intervals;
`--current` additionally checks current source/runtime identities. Build
logs and standalone replay native dumps lack structured timestamps, so the
timeline check is not a blanket machine-idleness claim. No observed
firewall, elevation or permission block and no system-setting change occur.

Next, compute populations from already resident quantized AC only when a
frame is materialized, attach owned immutable counts to that frame, and feed
the unchanged CPU sorting logic. Pure-DCT8 full/sampled policy compatibility,
AC-group-first PRNG order, generic/wide fallback, arena/readback accounting,
copy/move ownership, reconfiguration and failure atomicity must be qualified.
Public byte/decoded-quality and paired total-encode gates remain required.
No frame cache or production launcher is implemented in S78. Optimization
remains ongoing, not maxed out.

## Resident coefficient-order populations (S79)

S79 implements the S78 lead from `f71335c`, using frozen S75 runtime as the
performance baseline. Final resident quantized AC now feeds an exact GPU
zero-population reduction when a frame is materialized. An owned immutable
cache follows that frame into the unchanged CPU stable sort and order-token
writer. This removes a measured host scan; it is not GPU entropy coding or a
claim of universal end-to-end improvement.

### Ownership, sampling and resource contract

The internal frame-assembly input optionally supplies 5,952 full count bins
and 192 pure-DCT8 sampled bins: 24 KiB of unsigned 32-bit counters. Assembly
checks family presence, anchor-count bounds, absent-family zeros and sampled
subset bounds, then copies the result into frame-owned immutable storage.
It does not recount AC; exact equality to the producer's final coefficients
is an explicit internal-producer invariant. No caller-owned mutable cache
alias survives assembly. Copies/moves retain the appropriate cache, and
allocation/validation failures do not consume owned AC or change output.

Full and effort-7 sampled policies remain selectable on the same frame.
Only an entirely DCT8 frame samples, using the existing AC-group-first PRNG
order; host flags are stored at the resident batch's global-raster indexes.
Mixed frames need no flag upload. Stable float-scaled keys, ties, LLF prefixes,
public frame validation and encoded order tokens are unchanged. Generic CPU,
exact-coefficient and separate maximum-throughput frame-only producers retain
the existing uncached CPU route. The existing small-frame natural-order cutoff
and wide-counter fallback remain; there is no empirical image-size gate.

One launch dispatches all resident batches. A 32-by-8-thread block combines
64 anchors before global atomic adds. The full-only specialization uses
30 registers and 3,072 shared bytes; pure-DCT8 full-plus-sampled uses 38 and
6,144. Both have zero stack/local storage. All 203 earlier GPU bodies are
identical to S70; the two new bodies are identical in the production encoder,
phase probe, direct replay and interleaved control. No fast-math change occurs.

Existing arenas account for 24 KiB additional staging plus one byte per block
of persistent sampling capacity, with normal alignment rounding. Counts are
cleared/recomputed for each materialized frame and read back in the existing
batch. No extra device arena or steady-state device allocation is introduced;
frame assembly does allocate the small immutable host cache.

### Qualification

- The 73-test CUDA suite and 50-test CPU-only suite pass, including installed
  consumers. The subsequently expanded CUDA packing/population test is rebuilt
  and rerun separately: 64 geometry/strategy cases by five reuse passes,
  including all seven shapes, mixed grids, edge groups through 480-by-270
  blocks, original random/extreme-value cases and added dense-zero cases.
- The 218 coefficient-order cases now also compare cached/uncached full and
  sampled policies, frame copies/moves and mutable-input isolation. Cutoff and
  invalid-input/cache-assembly atomicity tests pass. Actual resident AQ tests
  independently recount cached bins and compare both policies and order tokens
  after stripping the cache through checked assembly; repeated evaluation,
  reconfiguration, pending metadata and injected failure contracts still pass.
- All 108 S78 frame exports pass the actual production-layout kernel replay,
  three repetitions each, with 6,144 exact bins and unchanged guarded sources,
  flags and outputs. Eleven GPU sanitizer jobs pass: four full replay tools,
  three full AQ tools, and four tools on 48 permanent small/medium packing
  cases by five reuse passes. No errors, leaks or hazards are reported.
- Eight host AddressSanitizer targets pass: DC, entropy, coefficient order,
  codestream encoder/workflow, overwrite storage, frame and reconstruction.
- All 58 freshly encoded/independently decoded cases match S70 bitstreams,
  decoded pixels, strategy/score reports and perceptual scores exactly,
  including scored/unscored, effort 9, high-density and maximum-compression.

An initial AQ assertion incorrectly expected a cache on the separate frame-only
maximum-throughput frontend. The failed log remains; the assertion now checks
that this path has no cache, without relaxing a numerical tolerance. No
production defect was found in that failure.

### Public measurements and same-process controls

Warm phase and public cohorts each alternate eight parent/candidate process
pairs per input, three warmups and five measured encodes per process. A cold
public cohort uses zero warmups and one measured encode. These are 144 serial
windows and 816 encodes, with encoded-size checks in the benchmark itself;
the separate 58-case qualification supplies byte/decoder evidence. Inputs
are the saved padded 1080p/4K and Flower PFMs, not the benchmark's generated
`--workload` images. All raw fields, adverse pairs and endpoint telemetry
remain in the evidence bundle.

Median within-pair changes, percent (negative is faster):

| Endpoint | Padded 4K | Padded 1080p | Flower |
| --- | ---: | ---: | ---: |
| Warm phase-probe total | +1.13 | -2.70 | -2.71 |
| Warm public total | -0.84 | -1.70 | -2.31 |
| Warm public serialization | -12.73 | -9.41 | -4.87 |
| Cold public total | -5.95 | +0.89 | -2.47 |

These small cohorts have broad ranges. The first cold 4K parent has a large
startup outlier; it is not discarded. Cold 1080p remains a limitation. Warm
phase/public endpoints range from 59 to 66 C and report differing sampled
memory clocks, without changing power, clock, cooling, priority or services.
Independent medians must not be substituted for the paired percentages.

A diagnostic switch then interleaves disabled/enabled/duplicate-disabled
population pipelines within one executable, preserving its frame ABI and
other code. It disables planning, flags, counting, readback and cache attachment
together; the production source contains no switch. All six permutations are
balanced in six-triple blocks, with deterministic reshuffling and reverse
configuration order in a second replicate. There are two backend lifetimes:
persistent and freshly created inside the already-warm process. Neither is
a cold-process substitute. Each replicate/lifetime uses 120 Flower triples
and 24 triples for each larger input.

All 222 preflight, 2,016 measured and 228 warmup/reference encodes preserve
bytes and summaries; 18 saved outputs match S70. Combined paired results:

| Backend lifetime / endpoint | Padded 4K | Padded 1080p | Flower |
| --- | ---: | ---: | ---: |
| Persistent total vs disabled 0 | -1.74% | -2.68% | -1.41% |
| Persistent total vs duplicate 2 | -1.98% | -2.76% | -1.66% |
| Fresh total vs disabled 0 | -2.25% | -1.21% | -0.87% |
| Fresh total vs duplicate 2 | -1.97% | +0.11% | -0.85% |

Persistent coefficient-order medians fall from 8.5184 to 0.30235 ms at 4K,
2.4545 to 0.3391 ms at 1080p, and 0.58165 to 0.3067 ms on Flower. Their
paired savings are 8.22275/2.0888/0.2697 ms. All replicate totals improve
against mode 0, but fresh-backend 1080p is essentially flat against duplicate
mode 2 and its first replicate regresses 1.15%. Flower descriptive balanced-
block bootstrap intervals for total are [-2.02%, -0.79%] persistent and
[-2.18%, -0.07%] fresh; these do not establish cross-machine guarantees.

### Integrated trace accounting and remaining limits

Two opposite-order Nsight trace pairs on the saved 4K PFM preserve all 308
previous kernel names, grids and resources. One population kernel adds
0.513069/0.507341 ms, using 4,064 blocks. Exactly one 24 KiB zero clear and
one 24 KiB D2H transfer are added; all other copies and clears are identical.
Allocation calls remain five `cudaMallocFromPoolAsync` and five `cudaFreeAsync`
per captured encode; synchronization-call counts also match. Arena byte
increments above are source-derived, not captured allocation-event payloads.
The replay's extra descriptor/upload setup is not part of this integration.

The trace analyzer initially used the wrong memset schema column and then
inherited S75's 373-launch assumption for its generated workload. Both failed
analyzers/logs remain; the corrected analyzer checks the actual saved-PFM
308-to-309 sequence and ordered copy/clear fingerprints. No firewall,
elevation or permission block is observed, and no restricted counters are used.

S79 is retained as exact resident host-scan removal with qualified warm
benefits, not universal warm/cold or batch-throughput improvement. Additional
architectures/toolkits, cold-run variation, batch throughput, remaining GPU
work and CPU token/entropy work remain open. The backend is not maxed out.
The ignored `s79_*` bundle preserves producers, source/runtime snapshots,
qualification, raw profiles, paired controls, traces and failed diagnostics;
`s79_validate.py --frozen` reconstructs the evidence, with `--current` also
checking current source/runtime identities.

## Lossless sparse AC transfer investigation (S80)

S80 starts from retained S79 `914b42c`. It tests whether a lossless sparse
transfer can remove a substantial resident-to-CPU handoff cost. Production
code, tests and all 40 retained runtime artifacts remain unchanged. These
are transfer/reconstruction experiments, not public encoder speedups.

### Attribution and format

Read-only reconstruction of the two S79 candidate 4K traces finds 38.04 and
48.43 ms of copies and 34.09 and 34.87 ms of gaps in the device-work union.
The final active AC payload is 99,532,800 bytes: 94,371,840 contiguous and
5,160,960 pitched. In trace pair 1 these two copies take 20.032253 ms.
The three original-image uploads each carry 33,153,604 bytes. Profiler-start
time is capture setup, not an additive encode saving, and copy-adjacent host
gaps are not evidence of a permission block or a measured bandwidth ceiling.

The S78 4K export has only 278,533 nonzeros among 24,883,200 coefficients;
1080p has 70,907 of 6,220,800, and Flower has 42,804 of 823,296. A fixed
256-value block header containing eight masks and one payload offset reduces
the 4K representation to 4,613,332 bytes, excluding a four-byte total count.
Values remain signed int32; no numerical-range or quantization assumption is
introduced. A dense input can instead expand to 103.515625% of its old size.

The two CUDA prototypes use 256 or 1,024 threads. Warp ballots and a shared
prefix sum compact nonzeros; one integer atomic reserves each nonempty
block's payload, and its header records the resulting offset. Global block
ordering need not be deterministic. Resources are respectively 21/40 registers
and 36/132 shared bytes, with zero stack/local storage. Both native bodies
are identical in all five diagnostic executables, including host-ASan builds.

Six modes compare dense transfer, sparse-256 with global CPU zero/scatter,
the identical dense branch, sparse-1024 with global zero/scatter, and the two
sparse kernels with block-local CPU zero/scatter. The input is already packed
active AC. Count clear, kernel, count readback/wait, payload readback/wait and
CPU reconstruction are timed. Exact comparison runs after every observation.

### Qualification and lifetime controls

Three additional corpus exports perform 21 exact S70/S79-identical encodes.
Together with S78 they provide 111 frames. Sixty synthetic arrays cover empty
work, warp/block/tail boundaries, random signed values, INT32_MIN/MAX and
zero/dense populations. V1 checks all six modes twice on every input, with
complete host/device guards, unchanged source and unused header/value tails.
All four GPU sanitizer tools pass on the same scope. Race checking takes
225 seconds with advancing case output and terminates with zero hazards.

V2 adds fresh ordinary output and sparse host-buffer allocations, plus four
roughly 100 MB repeated-vector stress inputs: zero, all nonzero, approximately
75% extreme nonzero, and approximately 12.5% nonzero. These are explicitly
synthetic coefficient vectors, not additional natural images. Both reused and
fresh lifetimes pass plain, host-ASan and stream-ordered memcheck qualification.

The V1 screen has 2,880 measured and 720 warm observations, but a V2 host
build overlaps a few small-image reverse-order windows. All are preserved
as exploratory. The separate V2 study runs after qualification and builds:
40 serial cohorts, 5,760 measured and 1,440 warm observations, ten inputs,
two lifetimes and reverse-order replicates. Each six-round block balances
the six modes across all six positions. Adverse pairs are not discarded.

For sparse-256/global-zero mode 1 versus dense mode 0, V2 median paired
changes show why reused buffers alone are insufficient (negative is faster):

| Input | Reused host buffers | Fresh ordinary buffers |
| --- | ---: | ---: |
| Padded 4K | -55.31% | -10.33% |
| Padded 1080p | -22.72% | +22.20% |
| Flower | -24.02% | +17.11% |
| Keong macan | -35.53% | +13.87% |
| Riaphotographs | -36.69% | +6.62% |
| Bliznaca | -27.76% | +16.52% |

Ordinary fresh dense-buffer reconstruction erases most small-image gains.
Dense and extreme-value stress inputs regress heavily; even the roughly
12.5%-nonzero stress can lose. Unconditional sparse readback is not justified.

### Zero-backed output counterfactual

V3 additionally allocates fresh Windows output with ordinary
`VirtualAlloc(MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)` pages, omitting the
explicit dense zero fill only for that owner. Microsoft documents initial
zero contents and the matching `VirtualFree(base, 0, MEM_RELEASE)` contract.
No large pages, executable pages, page locking, foreign processes or security
settings are involved. This is a diagnostic owner, not a portable production
allocator. [VirtualAlloc documentation](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc),
[VirtualFree documentation](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualfree).

Crucially, V3 times a full dense checksum read for **all** modes/lifetimes so
demand-zero pages cannot defer their first-read cost to the later oracle.
The checksum is additional to, not a replacement for, exact comparison.
Allocation, count/transfer, reconstruction and dense consumption are separate
components. Buffer destruction and the tiny gap between reconstruction and
consumption clock scopes remain excluded. Host ASan covers the ordinary
allocations/decoder; explicit guards also cover the VirtualAlloc output,
which is not a normal ASan heap allocation.

All three V3 lifetime policies pass 115 inputs plus 60 boundary arrays,
six modes twice each, both plain and under host ASan. New zero-backed ownership
also passes stream-ordered GPU memcheck. Across versions there are 37,560
guarded qualification comparisons, seven GPU sanitizer jobs and five host
ASan jobs. The additional timed dense read still is not full frame validation,
fixed-capacity group-tail assembly, serialization or complete lifecycle timing.

V3 runs 60 isolated cohorts: ten inputs, three lifetime policies, two
reverse-order replicates, 8,640 measured and 2,160 warm observations. Mode 1
versus mode 0, now including dense consumption, has these median paired
changes within each lifetime policy:

| Input | Reused | Fresh ordinary | Fresh zero-backed |
| --- | ---: | ---: | ---: |
| Padded 4K | -30.08% | -10.53% | -0.68% |
| Padded 1080p | -16.89% | +11.54% | +3.92% |
| Flower | -22.04% | +9.86% | -27.32% |
| Keong macan | -25.27% | +8.64% | +32.29% |
| Riaphotographs | -30.49% | +1.77% | +9.90% |
| Bliznaca | -25.35% | +9.87% | +22.08% |

The zero-backed 4K result is also nearly flat versus duplicate mode 2
(-0.83%). Its isolated Flower win is relative to that allocation policy's
own dense baseline, not proof that changing the production allocator wins.
The lifetime policies run in separate process windows; cross-policy ratios
of independent medians are not paired counterfactuals. All-zero 100 MB
zero-backed mode 1 is only 2.10% faster than its dense mode 0. Fully nonzero,
75%-extreme and 12.5%-nonzero zero-backed stress regress 77.38%, 89.60%
and 16.48%, respectively. A large byte reduction does not establish a gain.

### Disposition and next work

Reject unconditional sparse readback and the proposed general zero-backed
allocator fix. The qualified format remains a useful prototype, especially
with reused output storage, but neither experiment is retained in production.
Next work should measure bounded frame-storage reuse/ownership and cheaper
linear reconstruction, or compare a lossless fixed-width transport. A public
integration must preserve final group layout/tails and frame ownership, include
setup and destruction, provide a dense/oversized fallback, and pass complete
encode/decode, concurrency and batch gates. The observed 4K stage savings are
not an end-to-end speedup claim or justification for a fitted image-size gate.

The ignored `s80_*` bundle retains all versions, failed diagnostics and raw
rows. Initial NumPy analysis failed before execution and was replaced with
standard-library binary arrays; no dependency was installed. An initial Clang
ASan compile/link command was rejected and corrected in a separate script.
Neither was a firewall or privilege failure. V1's host-build overlap remains
disclosed; later V2/V3 cohorts run after their qualification/build/native work.
`s80_validate.py --frozen` reconstructs populations, checksums, native bodies,
all paired summaries, 37,560 guarded and 21,600 timed exact comparisons,
and 141 recorded non-overlapping GPU-job intervals. That timeline excludes
CPU-only build/native work and is not a blanket machine-idleness claim.
`--current` additionally checks source snapshots and the 40 unchanged S79
runtime files. No security, clock, power, cooling or priority setting changes,
and no observed firewall/admin/permission block. Optimization remains ongoing.

## Lossless narrow AC transfer investigation (S81)

S81 starts from S80 `ef5a18c`, with S79 production/runtime unchanged. It tests
linear lossless byte/int16 transport instead of sparse masks and scatter.
One CUDA kernel optionally writes both representations and returns two exact
signed-range overflow flags. The CPU selects byte, int16, or the original
int32 payload; no coefficient is clamped, rounded, or accepted out of range.

The six-mode replay has duplicate dense controls (0/2), dual packing with an
ordinary compiler-optimized expansion loop (1), identical dual transport with
SSE2 non-temporal expansion (3), and int16-only/byte-only packing with dense
fallback (4/5). Streaming stores require 16-byte-aligned output, otherwise
mode 3 uses ordinary stores; a store fence precedes consumption. GPU byte/word
payload capacities include the final padded pack word, with 4/8-byte alignment.

### Correctness and measurement boundary

The same 111 validated exports and four approximately 100 MB repeated-vector
stresses as S80 pass. Another 231 synthetic cases cover 21 lengths around
scalar, SIMD, warp and CTA tails, exact signed-width endpoints, last-element
first-overflow values, random int32 and sparse INT32_MIN/MAX. Every case uses
two repeats, two output offsets and all six modes. Checks include exact dense
output, checksum, both packed low-bit representations even on overflow,
padded final words, unused outputs, canaries, flags and unchanged source.

Plain and host-ASan qualification each run both reused and fresh ordinary
allocation policies. Four CUDA sanitizer jobs use fresh storage: memcheck,
initcheck, synccheck and racecheck all report zero errors/hazards; memcheck
reports zero leaks. In total, 66,432 guarded comparisons pass. The MSVC timing
and Clang-ASan executables have exactly the same three sm_86 kernel bodies.
Dual/int16/byte variants use 22/18/16 registers, with zero stack and spills.
Unused specialization accumulators produce harmless compiler warnings.

Timing includes ordinary dense allocation/first-touch, packing and flags
readback/synchronization, payload allocation/transfer, CPU expansion and a full
dense checksum read. Full equality and guard checks are outside timing. The
checksum is a consumption proxy, not actual frame assembly/entropy coding.
Input upload, device allocation, original group packing, final fixed-capacity
group tails, metadata/frame assembly and destruction are excluded. Reused
buffers are a separate policy; independent process medians across allocation
policies are not paired allocator counterfactuals.

### Paired results

Forty serial process windows cover ten inputs, two allocation policies and
two reversed input/policy-order replicates. Each window has six warm rounds
and 24 measured rounds; six-mode Latin rotations balance every position in
each six-round block. All 7,200 timed outputs are exact (5,760 measured and
1,440 warm). The following are median same-round total-time changes versus
dense mode 0, across 48 measured pairs per cell; negative is faster. Mode 2
is the same dense branch, not another implementation.

| Input | Reused, dual ordinary (1) | Reused, dual streaming (3) | Fresh, dual ordinary (1) | Fresh, dual streaming (3) | Fresh, duplicate dense (2) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Padded 4K | -12.13% | -20.84% | +1.93% | -10.73% | -0.39% |
| Padded 1080p | -8.52% | -19.11% | +14.35% | -0.56% | -1.16% |
| Flower | -9.90% | -1.40% | +22.39% | +20.18% | +0.24% |
| Keong macan | -19.19% | -14.46% | +4.11% | -0.08% | -0.22% |
| Riaphotographs | -21.65% | -16.43% | +2.01% | -1.82% | -0.40% |
| Bliznaca | -21.10% | -15.74% | +4.19% | -0.25% | -0.21% |

Fresh 4K streaming also improves 8.95% against duplicate dense mode 2,
with a 4.17 ms median same-round saving and 36/48 wins. Its separate replicate
medians versus mode 2 are -9.63% and -7.34%. Fresh 1080p is instead 1.63%
slower against mode 2, reinforcing the near-flat interpretation. Fresh Flower
requires int16, not byte transport, and is 20.11% slower against mode 2.
The ordinary dual loop does not win on any fresh real-image combined median.
Int16-only is 19.90%-26.56% slower on these fresh real-image cohorts. Byte-only
has a small 4K combined win (-2.81%), but its two replicate signs differ;
Flower's exact dense fallback loses 8.13%. Neither single-width variant is a
general fix. All modes and individual cohorts remain in the raw analysis.

The synthetic stress vectors distinguish density from value range. All-zero,
all-nonzero +/-1 and 12.5%-nonzero vectors all fit bytes; the 75%-extreme vector
must use the original int32 payload. Fresh-policy results are:

| Stress | Dual ordinary (1) | Dual streaming (3) | Int16 only (4) | Byte only (5) | Duplicate dense (2) |
| --- | ---: | ---: | ---: | ---: | ---: |
| All zero | +0.86% | -9.94% | +15.41% | -3.31% | -0.45% |
| All nonzero, +/-1 | +0.40% | -8.83% | +16.28% | -2.79% | -0.00% |
| 75% int32 extremes | +2.38% | +3.07% | +3.37% | +3.38% | -0.40% |
| 12.5% nonzero | +1.08% | -9.55% | +14.56% | +1.36% | -0.09% |

Individual timings are noisy even when combined duplicate-control medians
are close: fresh 4K streaming versus mode 2 ranges from -42.61% to +63.18%.
Packing time includes clear, kernel, flags transfer and synchronization, not
just GPU kernel duration. Ordinary device power management remains enabled;
boundary telemetry records 67/64 C and 210/1282 MHz SM clocks, not locked
or necessarily representative in-kernel clocks. No build/native audit or
evidence reconstruction overlaps measurement. CPU-only prior-evidence checking
does overlap race qualification, which is not used for performance claims.

### Disposition and next gates

Do not deploy unconditional narrow readback or select an image-size cutoff
from these samples. Streaming expansion is a qualified lossless prototype
with a repeated fresh 4K stage gain, not a general production improvement.
No production sources/binaries change and no new full encode/decode, CUDA/CPU
suite, quality or batch campaign is claimed. S79 remains retained.

Next work should investigate the extra flags synchronization, possible reuse
of existing mandatory readback, and bounded host-storage reuse/ownership. A
public integration must preserve exact signed values, final group layout and
zero tails, failure atomicity, immutable frame ownership and oversized/dense
fallbacks, then include setup, destruction, real serialization and concurrent
batch behavior. These stage results do not establish whole-encode speedup or
that fully resident optimization is maxed out.

The ignored `s81_*` bundle contains source, binaries, native dumps, serial raw
logs, paired summaries and independent reconstruction. `s81_validate.py
--frozen` verifies the frozen bundle, all 48 non-overlapping recorded GPU-job
intervals, 66,432 guarded comparisons and 7,200 timed comparisons, independently
recomputing source range flags/checksums and the document tables. `--current`
additionally checks source/docs snapshots and all 40 unchanged S79 runtime
files. No security, privilege, clock, power, cooling or priority changes were
made. No firewall/admin/permission block was observed; the user's suggested
cause for an earlier delay remains unconfirmed.

## Integrated narrow AC transport and metadata batching (S82)

S82 starts from S81 `d0ce0ea`, with retained S79 production/runtime unchanged.
It integrates the lossless transport prototype into a shadow resident host
implementation and measures complete encodes. Modes 0/2 are duplicate dense
controls; 1/4 only batch metadata readback; 3 adds dual byte/int16 transport
with streaming expansion; 5 uses the same transport with ordinary expansion.
These mode numbers are local to S82, not the S81 replay mode mapping.

### Integration and correctness boundary

The resident-policy error flag, scores, requested diagnostic maps and final
quantizer share one checked readback batch. Narrow flags join that batch, so
they introduce no extra synchronization. Device errors are checked before
processing copied metadata; caller-visible outputs still commit only after
successful validation and frame assembly. Generic evaluation stays dense.

After the unchanged GPU population count and AC group pack, the old quantized
scratch is dead. Its existing 4N-byte capacity holds N byte values, 2N int16
values and four flag bytes; the packed original int32 source remains intact
in reconstruction scratch for dense fallback. Narrow eligibility requires
N >= 8, N divisible by eight, uint32 count and supported grid bounds. Exact
range flags choose byte, int16 or original int32 without clamping or rounding.

CPU expansion writes the original fixed-capacity group layout and preserves
zero tails. Mode 3 uses aligned SSE2 non-temporal stores with a store fence,
falling back to ordinary stores on unaligned destinations. No device arena,
frame ownership, population cache or serializer change is needed. There is
additional ordinary host byte/int16 staging, and the final dense host owner
still exists: unchanged device allocation statistics do not mean unchanged
host memory use. No pinned storage, pool or custom deleter is introduced.

Fourteen preflight windows (seven inputs, persistent/fresh backend) perform
518 exact codestream/summary comparisons, including an extra dense readback
oracle for every active coefficient and fixed group tail. Three release
functional jobs cover AQ, public workflow and batch workflow. Five scoped
host-ASan jobs cover AQ, batch and full sample/Flower/4K encodes. Instrumentation
covers the changed resident host source and callers, not the retained linked
libraries; MSVC STL container annotations are disabled to match those libraries.

Six completed CUDA sanitizer jobs cover AQ memcheck/initcheck/synccheck/
racecheck, batch memcheck and all six integrated modes at full padded 4K.
All report zero errors/hazards; memcheck also reports zero leaks. The final
4K sanitizer-only driver performs seven full encodes, including its reference,
with fresh backends, dense-oracle checks and normal shutdown; it omits repeated
benchmark warmups, not input coverage or the full image. Twelve diagnostic
executables contain exactly the 205 retained and three S81 native GPU bodies.

Another 58 fresh mode-3 CLI encodes match the existing S70/S79 codestream
hashes, strategy summaries and optional final scores, including high-density
and maximum-compression cases. These exact matches reuse already decoded/
scored references; S82 does not claim fresh decoder or metric runs, a complete
CPU/CUDA suite, or a concurrent batch-throughput campaign.

### Trace and whole-encode measurements

Six unrestricted CUDA-only 4K captures run in orders 0/1/3 and 3/1/0. All 309
prior kernel launches/resources are identical. Batching removes two stream
synchronizations (34 to 32), with unchanged event synchronization and device
allocation API counts. Narrow adds one 0.818-0.850 ms kernel and a four-byte
clear. It replaces 99,532,800 active AC bytes with 24,883,200 payload bytes plus
four flag bytes; total captured D2H falls from 103,723,588 to 29,073,992 bytes.
Those transfer/kernel timings alone are not end-to-end speedups.

The timing protocol has 28 serial process windows: seven inputs, two backend
lifetime policies and two reversed input/policy-order replicates. Each window
has six warm and 24 measured rounds of all six modes, with shuffled Latin
rotations balancing each mode in every position. Same-round pairs provide
48 measured comparisons per combined cell. Duplicate controls expose timing
variation; no build, native audit, sanitizer or evidence reconstruction runs
alongside measurement.

Persistent backend does not mean reused host coefficient storage: each encode
still prepares and owns its final frame. Fresh backend is recreated within
the process, not a cold process/context experiment. The profiled encode total
includes actual serialization. A separate outer timer also includes optional
backend creation/destruction and call teardown. Input loading, equality checks,
file output and destruction of the returned codestream are outside timing.

All 5,068 benchmark encodes are exact: 4,032 measured, 1,008 warm and 28
references. The following are median same-round profiled-total percentage
changes; negative is faster. `Batch` is mode 1 versus dense 0; `Narrow` is
streaming mode 3 versus dense 0; `Added transport` is mode 3 versus batch 1.

| Input | Persistent, batch | Persistent, narrow | Persistent, added transport | Fresh, batch | Fresh, narrow | Fresh, added transport |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sample | -1.51% | +0.37% | +1.65% | +0.77% | -0.02% | -1.29% |
| Padded 1080p | -1.08% | -2.03% | -1.57% | +1.61% | +1.76% | -0.42% |
| Padded 4K | -0.52% | +0.68% | -0.80% | -0.01% | -1.84% | +0.18% |
| Flower | +0.47% | +0.54% | +1.77% | +2.58% | +1.42% | -0.93% |
| Keong macan | -0.28% | +0.45% | -0.35% | -0.87% | -1.41% | +0.10% |
| Riaphotographs | -1.16% | +0.31% | +1.30% | +1.45% | -0.95% | -2.78% |
| Bliznaca | -2.33% | -2.27% | -1.11% | +1.35% | -0.05% | +0.60% |

Persistent 1080p streaming improves 2.13% against duplicate dense mode 2,
with 31/48 wins and a 1.64 ms median same-round saving. Its separate replicate
medians are -2.50% and -1.72%; it also improves against both batch-only controls
in each replicate. The outer timer improves 1.59% against mode 2. This is a
qualified local result, not a universal transport win: fresh 1080p is +1.76%
versus mode 0 and -0.45% versus mode 2 (+0.11% on the outer timer versus 2).

At persistent 4K, streaming is +0.68% versus dense 0 but -2.03% versus dense 2;
the replicate signs against 0 differ (+1.88%, -1.41%). Fresh 4K is -1.84%
versus 0 but +0.28% versus 2, with only 24/48 wins against 2; its outer timer
is +0.66% versus 2. Added transport is near flat against fresh batch-only
mode 1 (+0.18%). Thus the S81 fresh 4K stage result does not establish a robust
integrated full-encode gain here. Flower requires int16; the other six inputs
fit bytes. None of these timing inputs exercises narrow mode's int32 fallback;
the signed-extreme fallback has S81 replay coverage, not a new S82 timing claim.

Ordinary expansion (5) is not a general alternative: versus dense 0 it is
+1.34%/+1.95% at persistent/fresh 4K and +0.57%/+1.55% at 1080p. Metadata-only
batching also lacks a general attributable whole-encode benefit in these
samples. All six modes, duplicate controls and separate replicates remain
reported. Combined duplicate dense changes span -1.03% to +3.11%; duplicate
batch-only changes span -1.20% to +1.61%. Individual paired timings are much
noisier still, so these small medians are not confidence bounds. Boundary
telemetry is 57/78 C and 210/1282 MHz SM clock on the RTX 3060 Laptop GPU;
ordinary power management remains enabled, and these are not in-kernel clocks.

### Investigation failures and evidence

The first input-selection assertion expected six rather than seven references
and failed before any GPU job. An initial scoped-ASan link failed on the STL
annotation mismatch; separately named V2 objects/executables correct it. A
piped native dump timed out; its child was terminated/waited, no dump JSON was
created, and its partial piped stdout was not preserved. A new file-backed dump
succeeds. The first trace-summary parser expected an unversioned API name;
it now accepts the observed `_v3020` suffix without changing count expectations.
Evidence preflight also corrected a tuple/list JSON round-trip mismatch in
that summary builder; the existing summary values and captures are unchanged.

The first experimental and unchanged-baseline batch memchecks return zero and
report zero errors/leaks but omit the application success marker. Neither is
counted as a completed pass. A wrapper with flushed output and explicit start/
exit markers passes twice without a CUDA-unavailable skip. The original
37-encode 4K memcheck was progressing at about 30-31 seconds per encode, which
would exceed its 900-second cap. Only its revalidated owned process pair was
intentionally stopped; partial logs/output remain and are not counted as a
pass. The seven-encode sanitizer-only driver then completes. Partial V1/V2
manifests remain unchanged; completed V3 records preserve their lineage.

Neither unconditional narrow transport nor metadata-only batching is promoted
to production. The prototype removes the extra flags-sync objection and
qualifies integration, but does not provide a general stable end-to-end win.
All 40 S79 runtime files and public production sources remain unchanged.
The next bounded lead is to overlap ordinary dense-output first-touch with
resident GPU work between successful submission and its mandatory wait. It
must preserve repeated prepared-object ownership and failure atomicity, compare
post-wait first-touch, and qualify concurrent memory-bandwidth effects. It is
not yet implemented or measured; optimization is not considered maxed out.

The ignored `s82_*` bundle preserves diagnostic source/binaries, raw logs,
native dumps, traces, partial-attempt lineage and paired summaries. Running
`python build-cuda-ninja/profiles/s82_validate.py --frozen` reconstructs the
tables and verifies 5,762 explicitly counted exact encodes, twelve native-
identical executables and 120 non-overlapping completed campaign intervals.
The count excludes additional functional-test/profile encodes and inconclusive/
stopped attempts. The interval check excludes CPU-only build/native work and
is not a blanket machine-idleness claim; some native work overlaps sanitizer
qualification, never measurement. `--current` additionally checks source/docs
snapshots and the retained runtime. No security, privilege, clock, power,
cooling or priority changes were made, and no firewall/admin/permission block
was observed. The suggested cause of the user's earlier delay is unconfirmed.

## Overlapped dense host first-touch (S83)

S83 starts from S82 `559028c` and tests ordinary host-buffer initialization
inside the asynchronous resident-policy interval. S82's narrow transport and
metadata batching are not included. All existing GPU kernels, dense transfers,
metadata synchronizations, tail zeroing and final frame ownership stay intact.

Modes 0/2 are duplicate retained dense controls; 1/4 fill the final coefficient
owner with zeros after successful submission and before its mandatory wait;
3 performs the same full fill after successful wait; 5 writes one int32 at
each 4-KiB stride and the last element before waiting. The machine reports
4096-byte system pages, but the stride mode has no page locking, allocator
replacement or platform-specific mapping. Full fill uses ordinary C++ stores.

The buffer is not referenced by the queued GPU computation. It is already
allocated during fresh preparation; repeated prepared use after owned frame
handoff ensures replacement storage under the existing prepared-object lock,
before submitting work. Initialization cannot allocate or return a failure.
Every successful submission still waits, including injected completion
failure, and caller outputs retain the existing commit-after-validation rule.
Generic evaluation does not take this diagnostic path. No new device arena,
host staging owner, helper thread or pool is introduced.

### Correctness and measurement boundary

Fourteen serial preflight windows cover seven images and persistent/fresh
backend lifetimes. All 518 encodes match retained codestream bytes and summary
fields. An extra dense readback oracle checks every active coefficient and
fixed-capacity tail. Recorded submission/fill/wait/readback intervals verify
each mode's exact placement and touch count; controls must report no touch.

Seven release functional jobs cover AQ and batch in full-pre-wait,
full-post-wait and stride-pre-wait modes, plus the public workflow test.
Seven scoped host-ASan jobs cover full/stride AQ and batch, and all six modes
on sample/Flower/4K encodes. Instrumentation includes the changed resident
source and callers, not the retained linked libraries; MSVC STL container
annotations are disabled consistently with those libraries. Three GPU
memcheck jobs cover AQ, batch and seven full 4K encodes (reference plus each
mode once). Explicit application completion and zero errors/leaks are required
and pass. No new initcheck/synccheck/racecheck campaign is claimed: the three
audited timing, ASan-timing and AQ executables have exactly the 205 retained
GPU bodies, and no device memory operation changes.

The full 4K memcheck initially emits no application progress for about a
minute because its config is buffered until after the reference and first
mode. A read-only live check finds the owned application/sanitizer pair
consuming CPU time, 100% GPU activity and no consent process; rows then arrive
at about 32 seconds per encode. It completes without stopping or restarting.
This is active instrumented work, not an observed firewall/admin block.

All 58 fresh CLI encodes in the default full-pre-wait mode match the retained
S70/S79 codestream hashes, strategy summaries and optional final scores,
including high-density and maximum-compression cases. The byte-identical
references were already independently decoded/scored; S83 does not claim a
fresh decoder/metric run or complete CPU/CUDA suite.

The timing protocol uses 28 serial process windows: seven inputs, persistent
or freshly created backend, and two reversed input/lifetime-order replicates.
Each has six warm and 24 measured six-mode rounds, with shuffled Latin
rotations balancing every mode's position. A persistent backend does not reuse
the final host coefficient owner; fresh backend is not a fresh process.
The profiled total includes actual serialization; the outer timer additionally
includes optional backend creation/destruction and call teardown. Input load,
equality/oracle checks, file output and returned codestream destruction are
outside timing. The dense oracle is enabled only for qualification. Separate
fill, wait, readback, post-wait and post-submission intervals help distinguish
moving work from removing it. Builds, native audits, sanitizer/quality/profile
work and evidence reconstruction finish before performance measurement.

### Trace and paired results

Eight unrestricted CUDA-only 4K captures run in orders 0/1/3/5 and 5/3/1/0.
All 309 kernel launches/resources, transfer and clear fingerprints, and runtime
API counts match. D2H remains 103,723,588 bytes. The final submission's host
gap between event recording and event wait is 16.90/17.21 ms for full pre-wait
fill, with 16.74/17.03 ms of GPU kernel activity inside it. Stride touching
has 12.70/12.28 ms host gaps and 12.57/12.17 ms overlapping GPU activity. Dense
and post-wait controls have only 0.002-0.014 ms in that gap. Source placement
and independent harness interval checks identify the added host work as the
initialization; kernel-active durations are interval unions, not summed
double-counted overlapping launches.

Captured D2H time falls from 21.61/22.68 ms dense to 16.59/16.05 ms full
pre-wait, 16.56/17.04 ms full post-wait and 16.05/16.40 ms stride pre-wait.
The post-wait control pays its initialization outside GPU overlap. These
traces establish the mechanism, not a public encode or batch speedup.

The initial position-balanced campaign completes 5,068 exact encodes: 4,032
measured, 1,008 warm and 28 references. Each cell below is the median
same-round profiled-total percentage change over 48 pairs; negative is faster.
`Full/dense` is mode 1 versus 0, `Overlap` is mode 1 versus the identical
post-wait fill (3), and `Stride/dense` is mode 5 versus 0.

| Input | Persistent, full/dense | Persistent, overlap | Persistent, stride/dense | Fresh, full/dense | Fresh, overlap | Fresh, stride/dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sample | -1.59% | -0.25% | +1.13% | +0.86% | -2.10% | +1.88% |
| Padded 1080p | -4.41% | -7.31% | -4.02% | -3.59% | -4.81% | -3.46% |
| Padded 4K | -2.07% | -2.32% | -0.89% | -1.19% | -2.20% | +1.53% |
| Flower | -3.35% | -3.49% | -2.30% | -0.14% | -0.75% | -2.34% |
| Keong macan | -3.24% | -1.70% | -3.41% | -0.85% | -0.52% | -2.17% |
| Riaphotographs | -1.99% | -3.53% | -1.13% | +0.50% | -0.39% | -2.36% |
| Bliznaca | -2.46% | -1.97% | -1.97% | -3.12% | -2.31% | -1.83% |

Full pre-wait fill at 1080p also improves 2.91%/2.02% against duplicate dense
mode 2 for persistent/fresh backend. Both full-fill implementations (1/4)
improve against both dense controls in the combined 1080p results. Persistent
4K differs: mode 1 is 0.81% slower against dense 2 despite the 2.07% improvement
against 0; both separate replicate medians against 2 are positive. Fresh 4K
mode 1 improves against both dense controls in aggregate, but its replicate
signs against 2 differ. The sparse-touch and small-image results are also mixed.
These observations motivate a schedule-control follow-up before retention.

### Preceding-mode-balanced follow-up (V2)

The initial cyclic Latin rotations balance positions but not preceding-mode
exposure. In persistent 4K replicate 0, dense control 0 follows mode 3 ten
times while dense control 2 follows mode 3 five times. This imbalance is not
proof that carryover caused the control discrepancy. V1 remains reported
separately and is neither overwritten nor pooled into V2.

V2 changes only the caller schedule, using the six-row Williams construction
based on `[0,1,5,2,4,3]`, with shuffled labels and row order in each block.
Every mode occupies every position once, and each of the 30 ordered pairs of
different modes occurs exactly once within rounds. A checked retained-dense
encode precedes every round and is destroyed before its timed modes. This
conditioning standardizes the start; it does not claim to erase all allocator,
cache, thermal or higher-order history. Conditioning encodes are counted but
excluded from paired timing statistics. The unchanged resident object and
libraries retain the prior qualification; the scheduling-only V2 caller is
not separately host-ASan-instrumented. Its separate native audit preserves all
205 kernels, and its preflight rechecks exact outputs, order and touch bounds.

V2 passes 602 preflight encodes (including 84 conditioning encodes), then
5,908 performance-campaign encodes: 4,032 measured, 1,008 warm, 840 conditioning
and 28 references. Its 28 serial windows and 48 measured pairs per combined
cell match V1's input/lifetime/replicate coverage. The following percentages
are profiled-total paired medians, not conditioning-inclusive throughput.

| Input | Persistent, full/dense 0 | Persistent, full/dense 2 | Persistent, stride/dense 0 | Fresh, full/dense 0 | Fresh, full/dense 2 | Fresh, stride/dense 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sample | -0.99% | -0.69% | -0.04% | -0.29% | -0.14% | +1.31% |
| Padded 1080p | -4.65% | -2.96% | -4.47% | -3.71% | -3.71% | -3.59% |
| Padded 4K | -3.36% | -2.75% | -1.19% | -0.90% | +0.67% | +0.74% |
| Flower | -0.96% | -3.06% | -1.08% | +0.79% | -0.47% | +0.50% |
| Keong macan | -2.82% | -3.94% | -3.54% | -2.50% | -2.07% | -3.40% |
| Riaphotographs | -0.93% | -0.41% | -3.24% | -0.68% | -1.71% | -1.73% |
| Bliznaca | -1.01% | -1.21% | -1.63% | -0.63% | -0.96% | -1.62% |

Full pre-wait fill's 1080p total improvement against dense 2 is 2.96%/3.71%
for persistent/fresh backend, with 28/48 and 33/48 wins and median same-round
savings of 2.27/3.53 ms. The outer timer improves 2.82%/3.54%. Each of its four
separate 1080p process-window medians improves against both dense controls.
Duplicate full-fill mode 4 improves in the combined 1080p cohorts too, though
its second fresh-backend replicate is near flat/slightly adverse. This
supports a real local benefit without claiming every encode is faster.

At persistent 4K, mode 1 improves against dense 2 by 2.75%, with 33/48 wins
and an 8.91 ms median saving; its separate replicate medians are -2.63% and
-4.94%. But duplicate full-fill mode 4 is 2.94% slower than mode 1, despite
identical implementation, and one of its replicate medians loses against both
dense controls. Fresh 4K mode 1 remains +0.67% versus dense 2 (only 19/48 wins,
outer timer +0.56%) while improving 0.90% versus dense 0. Stride touching is
also not a general replacement. V2 reduces neither all noise nor the need for
duplicate controls: combined dense duplicates span -1.18% to +1.38%, and
full-fill duplicates span -2.62% to +2.94%. Paired medians are not confidence
bounds; no causal explanation for all remaining variation is established.

The measured subinterval gives a clearer mechanism at 1080p. Full pre-wait
fill costs a median 5.21/5.31 ms for persistent/fresh backend, while wait time
falls from dense 2's 17.09/15.81 ms to 11.78/10.62 ms. Readback falls from
6.02/6.31 ms to 4.97/4.90 ms. The complete post-submission interval saves
2.83/2.97 ms against dense 2, winning 47/48 and 48/48 pairs. Against post-wait
fill, that interval wins every pair and saves 5.16/4.86 ms. Separate medians
need not add exactly, so these are not an additive latency budget.

Fresh 4K illustrates why a stage gain is insufficient: its post-submission
interval saves 6.54 ms against dense 2, yet the complete total is slower by a
median 2.67 ms. All 41 phase fields, interval timings and separate replicates
are retained, not filtered to favorable stages. Boundary telemetry is
63/71 C and 210/1282 MHz SM clock for V1, and 64/71 C and 1380/1282 MHz for V2.
Clocks/power management are not locked; boundary samples are not kernel clocks.

### Disposition and next gates

Overlapped ordinary first-touch is a qualified prototype with repeated 1080p
whole-encode gains and directly observed CPU/GPU overlap. It is not promoted
unconditionally: fresh 4K is not a stable gain, duplicate controls still vary,
and production/batch-throughput qualification is not complete. All public
production sources and the 40-file S79 runtime remain unchanged. No image-size
cutoff or host-storage reuse policy is inferred from these samples.

Two bounded follow-ups remain unimplemented: skip the redundant tail clear
only when this exact owner has already been fully initialized, and combine
pre-wait host first-touch with S82's narrow transfer while comparing against
both dense and narrow controls. The latter must include compact staging's
allocation, initialization and memory cost. Any retained variant still needs
failure/ownership, complete production and concurrent batch gates. This is
continued optimization, not evidence that fully resident encoding is maxed out.

The ignored `s83_*` bundle preserves V1/V2 callers, the unchanged diagnostic
resident implementation, binaries, traces, raw logs and independent analysis.
`python build-cuda-ninja/profiles/s83_validate.py --frozen` reconstructs both
tables and verifies 12,272 explicitly counted exact encodes, four native-
identical executables, eight traces and 167 non-overlapping completed campaign
intervals. The count excludes additional functional-test and profile encodes;
interval checks exclude CPU-only work and do not claim blanket machine idleness.
V2 conditioning is counted separately and not included in paired performance
statistics. `--current` additionally checks source/docs snapshots and the
unchanged runtime. No security, privilege, clock, power, cooling or priority
changes were made, and no firewall/admin/permission block was observed.

## Work that should not lead the next cycle

### More execution lanes

The current two-lane pool is sufficient. Four in-flight requests improve
overlap of CPU preparation, GPU work, and serialization, but fully-resident
GPU throughput no longer scales with the request count. More streams would
increase simultaneous resident memory without creating additional SM
capacity.

### Exact-coefficient optimization

Exact mode preserves the CPU-compatible coefficient decision path and spends
about 2.7 seconds at 1080p. It is valuable as a compatibility and differential
oracle, not as the first throughput target.

### GPU entropy coding

CPU codestream generation is already visible at about `70 ms` at 1080p and
`252-443 ms` at 4K. Moving entropy coding to CUDA could eventually raise the
end-to-end ceiling, but it is a much larger project than removing current
allocation and handoff overhead. Batch workers already overlap serialization
for one image with GPU work for another.

### Tensor cores or global fast math

The DCT, quantization, and selection paths are decision-sensitive. Reduced
precision or globally relaxed math needs an explicit mode and quality
contract; it should not silently alter fully-resident behavior. The retained
FP32 radix-2 factorization uses ordinary math, preserves existing numerical
tolerances, and separately qualifies rounding changes against decoded images.

## Suggested implementation checkpoints

1. **Arena checkpoint:** reduce fresh-encode allocation count and re-run the
   1080p Nsight API summary.
2. **Readback checkpoint:** account for every transfer by semantic payload and
   prove that encoding-only output omits diagnostic maps.
3. **DCT checkpoint:** enable counters, specialize 32x32, then 16x32/32x16, with
   per-shape differential tests and public-workflow timing.
4. **Frontend checkpoint:** add resident CUDA Opsin preparation and compare
   both 1080p and 4K wall profiles.
5. **Small-kernel checkpoint:** parallelize initial CfL and evaluate graph
   capture only after the preceding pointer and dataflow changes settle.

Each checkpoint should run:

- CPU-only configuration and tests;
- the complete CUDA functional suite;
- exact-coefficient CPU/CUDA codestream comparison;
- repeated fully-resident and maximum-throughput concurrency stress;
- odd padded 1080p and 4K encode/decode qualification; and
- paired batch sizes 1, 2, and 4, with GPU temperature and clock state noted.

## Reproduction commands

Public wall profiles:

```powershell
.\build-cuda-ninja\gjxl_cuda_encoding_benchmark.exe `
  --workload padded_1080p --gpu-aq fully-resident `
  --warmups 2 --samples 7 --gpu-only

.\build-cuda-ninja\gjxl_cuda_encoding_benchmark.exe `
  --workload padded_1080p --gpu-aq maximum-throughput `
  --warmups 2 --samples 7 --gpu-only

.\build-cuda-ninja\gjxl_cuda_encoding_benchmark.exe `
  --workload padded_4k --gpu-aq fully-resident `
  --warmups 1 --samples 3 --gpu-only

.\build-cuda-ninja\gjxl_cuda_encoding_benchmark.exe `
  --workload padded_4k --gpu-aq maximum-throughput `
  --warmups 1 --samples 3 --gpu-only
```

Batch profiles:

```powershell
.\build-cuda-ninja\gjxl_image_batch_benchmark.exe `
  --workload 1080p --batch-sizes 1,2,4 `
  --gpu-aq maximum-throughput --backend cuda `
  --warmups 1 --samples 5

.\build-cuda-ninja\gjxl_image_batch_benchmark.exe `
  --workload 1080p --batch-sizes 1,2,4 `
  --gpu-aq fully-resident --backend cuda `
  --warmups 1 --samples 3
```

Nsight Systems capture, using the installed 2023.2.3 executable path:

```powershell
$gjxlNsysRoot =
  'C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.2.3'
$gjxlNsys = Join-Path $gjxlNsysRoot 'target-windows-x64\nsys.exe'
& $gjxlNsys `
  profile --trace=cuda --sample=none --cpuctxsw=none `
  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown `
  --force-overwrite=true `
  --output=build-cuda-ninja\profiles\cuda_fully_resident_1080p `
  .\build-cuda-ninja\gjxl_cuda_encoding_benchmark.exe `
  --workload padded_1080p --gpu-aq fully-resident `
  --warmups 2 --samples 1 --gpu-only --profile-range
```

The local study artifacts are:

- `build-cuda-ninja/profiles/issue6_fully_resident_1080p.nsys-rep`;
- `build-cuda-ninja/profiles/issue6_fully_resident_1080p.sqlite`;
- `build-cuda-ninja/profiles/issue6_maximum_throughput_1080p.nsys-rep`; and
- `build-cuda-ninja/profiles/issue6_maximum_throughput_1080p.sqlite`.

They live under the ignored build directory and are intentionally not source
artifacts.
