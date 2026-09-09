# CUDA four-output-row rolling convolution (S150)

## Scope and mechanism

S149 found a repeatable four-million-pixel preference for the retained
48-row rolling tile, but no stable larger-area crossover. S150 isolates a
different mechanism: computing four adjacent output rows per lane instead
of three. Starting commit: `f8dfbd5`. This is a diagnostic prototype;
production source remains S148.

All variants keep 256 threads and three 64-row by 32-column shared planes
(24,576 bytes). Eight warps now produce a 32-row chunk, instead of 24 rows.
The first chunk loads 64 rows; subsequent chunks replace 32 retired rows
after the existing read-completion barrier. New tile heights are 64, 96,
and 128. A ring-liveness simulation checks all 288 output rows and 9,504
ordered tap associations across these three variants.

Four outputs consume all 64 ring rows, leaving no unused slot for the
normalization scalar. Each warp's lane zero performs the original ordered
33-tap sum in registers, followed by one full-warp broadcast. There is no
host normalization, reciprocal substitution, reordered FMA chain, shared
scalar allocation, or retained caller pointer. The extra per-warp
normalization work is included in the measurements. All output arithmetic
and the horizontal convolution are unchanged.

At a 96-row tile, old and new kernels have identical CTA counts and load
the same 128 global input rows per full CTA. The new kernel uses three
chunks and five block barriers, versus four chunks and seven barriers.
Each lane's input-row load feeds up to four outputs: 36 input rows per
four outputs, instead of 35 per three. This predicts less shared-load work
per output; it is not itself a throughput result. The 64- and 128-row
variants separately change scheduling and halo duplication.

## Native code

The prototype object contains all 80 original Butteraugli GPU bodies
instruction-exact, plus three new bodies. Release and host-ASAN probe,
boundary, replay, and counter executables have identical multisets of
three GPU modules.

| Vertical body | Registers | Shared bytes | Static instructions | Static LDS | Static SHFL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained plain 48 | 54 | 30,856 | 1,736 | 219 | 0 |
| Retained rolling three-row 48/96 | 46 | 24,576 | 1,984 | 211 | 0 |
| New four-row 64/96/128 | 47 | 24,576 | 2,352 | 216 | 1 |

All new bodies have zero stack/local storage and zero spills. Their
constant-0 footprint is 604 bytes. Static code grows because each chunk
computes more outputs; executed-work counters are needed to interpret it.
The driver confirms four 256-thread blocks (32 warps) per SM for every new
body, matching retained rolling tiles. Plain48 admits three blocks.

## Correctness and lifetime qualification

Release and host-ASAN each pass 2,760 original fixtures across six forced
families, preserving both permanent reference oracles, three reuse stages,
and exact checks of sixteen arrays including guards and padding. Each
build also passes 540 extra fixtures around heights 127/128/129,
191/192/193, and 255/256/257; 64 launch-contract checks; and twelve tall
cases exercising the flattened grid beyond 65,535 tile rows.

Ownership tests cover all three new variants, two independent streams,
temporary and caller-weight poisoning, two weight generations including
negative taps, and repeated captured launches. Each build passes 108
checked graph executions. Scoped guards (180 cases) and ownership (108
graph checks) each pass memcheck, initcheck, synccheck, and racecheck.
Memory checks report no leaks. Scoped race checking takes 594.902 seconds;
ownership race checking takes 83.577 seconds. Both report zero hazards.

## Measurement protocol

The replay uses hash-checked, fully resident S145 captures, not synthetic
pixels or older payloads. All 24 unique captures receive release and
host-ASAN preflights. Timing selects four main sizes (500-square flower,
1919x1079, 3839x2159, and 2000-square flower), plus both packed and padded
output layouts of the 4K half image.

The four families are retained three-row tile96 and new four-row
tile64/tile96/tile128. Each has two identical labels, in an eight-label
randomized Williams design with eight warmup and sixteen measured rounds.
Both 4-pair and 128-pair bursts are tested, with a second process order
reversed. Every graph is validated against sixteen arrays outside the
event interval. The timed unit includes unchanged horizontal convolution
plus the selected vertical kernel; it is not full-encoder throughput.

Primary comparisons average the two label medians per family. All four
candidate-versus-baseline label comparisons and within-family duplicate
deltas are retained. This screen alone cannot qualify a production
geometry selector, especially where the baseline selector chooses a
different body.

Counter runs use current full 4K, ten direct same-variant warmups, and one
explicitly delimited vertical launch. Plain48, retained rolling48/96, and
all three new variants are profiled in forward and reverse orders. Clock
and cache control are disabled. Profiling results diagnose work and
resources, not ordinary throughput.

## Ordinary timing results

All 48 capture preflights and 24 timing processes pass. The 4,608 event
windows contain 3,072 measured windows and 202,752 measured pairs. All
10,368 ordinary preflight/timing power-limit endpoints report 40 W. The
replay harness also passes memcheck and initcheck with no errors or leaks.
Every graph is checked, including warmup and capture-preflight graphs.

Negative changes below mean faster than retained three-row tile96.

| Burst | Four-row tile | Favorable primary | Favorable cross-label | Median change across captures/repeats |
| --- | ---: | ---: | ---: | ---: |
| 4 pairs | 64 | 10/12 | 39/48 | -2.07% |
| 4 pairs | 96 | 12/12 | 41/48 | -1.63% |
| 4 pairs | 128 | 11/12 | 42/48 | -1.49% |
| 128 pairs | 64 | 11/12 | 45/48 | -5.24% |
| 128 pairs | 96 | 12/12 | 47/48 | -4.52% |
| 128 pairs | 128 | 10/12 | 40/48 | -4.20% |

Full-resolution cases, retaining both process repetitions:

| Input | Burst | Four-row 64 change | Four-row 96 change | Four-row 128 change |
| --- | ---: | ---: | ---: | ---: |
| 1919x1079 | 4 | -1.91% / -1.95% | -0.97% / -0.91% | -1.72% / -1.71% |
| 1919x1079 | 128 | -7.54% / -5.78% | -5.46% / -5.25% | -6.16% / -4.10% |
| 3839x2159 | 4 | -1.95% / -0.92% | -1.69% / -1.54% | -2.86% / -1.10% |
| 3839x2159 | 128 | -2.73% / -3.12% | -4.07% / -4.61% | -4.81% / -4.79% |
| 2000x2000 flower | 4 | -5.38% / -4.78% | -1.82% / -2.10% | -1.27% / -2.35% |
| 2000x2000 flower | 128 | -5.07% / -5.49% | -4.36% / -3.09% | -3.51% / -4.40% |

On every larger capture (the three larger main planes and both 4K half
layouts), all three new variants win both sustained repetitions and all
four cross-label comparisons: 30/30 primary and 120/120 cross-label wins.
Short-burst gains are smaller, and several cross-label comparisons disagree.
The largest absolute duplicate-label gap is 0.132648 ms/pair, between the
two retained controls in the first sustained 4K run. Short 4K runs also
have duplicate gaps larger than some candidate/control improvements.

The 500-square control is mixed. Four-row tile64 regresses 6.53%/5.84%
at short bursts, while tile128 regresses 11.55%/5.61% under sustained load.
Although tile96 improves against the deliberately forced three-row tile96
control here, that control is not the production selector's small-image
choice. These measurements do not justify enabling rolling tiles on small
images or treating any new tile as a universal replacement.

The unchanged horizontal kernel remains inside each timed interval. The
stage result therefore includes possible interactions between consecutive
horizontal and vertical work. Power-limit endpoints are not measurements
of instantaneous kernel clock or power, and short and sustained timings
are not pooled. No whole-encoder gain is inferred from this replay.

## Hardware-counter evidence

Both counter orders complete, including release/host-ASAN preflight of
all six variants and memcheck/initcheck of the counter harness. The table
compares each new vertical kernel with retained three-row tile96 on the
same current 4K input. Percentages retain both profiler repetitions.

| Four-row tile | Executed warp instructions | Shared-load wavefronts | Global-load sectors | DRAM bytes read |
| --- | ---: | ---: | ---: | ---: |
| 64 | -3.323% / -3.323% | -22.67% / -23.13% | +7.08% / +7.09% | +7.32% / +7.45% |
| 96 | -5.624% / -5.624% | -23.39% / -23.26% | +0.02% / +0.08% | +0.63% / +0.76% |
| 128 | -7.176% / -7.176% | -23.28% / -23.30% | -3.81% / -3.76% | -2.72% / -2.59% |

Predicated-on thread FFMA counts are identical across all five rolling
variants, in both repetitions. The equal-96-row experiment therefore
reduces executed instructions and shared traffic without fewer computed
FMAs, a different global halo, or lower theoretical residency. Its L2
traffic is nearly flat (+0.001%/+0.063%). This supports the intended
shared-load/barrier mechanism; it does not isolate the independent
contribution of each change.
Measured active warps average 30.82-30.85 for new tile96 versus 31.14-31.16
for the retained tile96; the gain is not explained by increased occupancy.

The 64-row tile trades additional global traffic for smaller work units,
and the 128-row tile reduces both instruction and halo costs. Their
ordinary relative performance still depends on geometry. Compared with
plain48, the equal-96-row prototype also executes fewer warp instructions;
the retained three-row tile96 had still executed more. Counter timing and
traffic differences are not substituted for ordinary timing results.

## Decision and next step

Keep the four-output-row prototype as a qualified candidate for integrated
testing. The equal-tile comparison shows sustained pair improvements of
4.07%/4.61% at 4K, 3.09-4.36% at 2000-square, and 5.25-5.46% at HD,
alongside repeatable executed-work reductions. These are additional gains
over the retained three-row tile96 kernel, not over a CPU baseline.

Do not change production dispatch from this stage-only screen. In
particular, production currently selects plain48 for the small control
and rolling48 for HD and the 4K half planes; this experiment deliberately
forces rolling96 as the common mechanism control. The next step is a
within-executable integrated comparison against actual S148 dispatch,
including rolling48 controls, the new 64/96/128 schedules, both coefficient
storage modes, full codestream/summary oracles, and complete-encode timing.
S148 remains the retained runtime until that qualification succeeds.

All 143 recorded jobs pass, including 117 serial GPU-facing jobs and
twelve CUDA sanitizer jobs. No admin/firewall blocker is encountered and
no clock, power, thermal, priority, affinity, or security setting is changed.
No production source or permanent test changes are made in this study.

Evidence is under `build-cuda-ninja/profiles/s150-artifacts`: source/binary
pins, native bodies, ring-liveness checks, both reference oracles, every
timing/control sample, twelve profiler reports, and derived results.
Run `python -X utf8 build-cuda-ninja/profiles/verify_s150.py --frozen` to
recheck the frozen study, all S149 artifacts, unchanged production sources,
and forty retained-runtime hashes.
