# Validated unsigned Malta addresses (S123)

## Outcome and scope

Explicit unsigned widening of already-validated Malta tile coordinates
removes 1.01% of executed warp instructions on the current full 4K capture,
without changing output, register allocation or memory traffic. Unlike the
[rejected recurrences](cuda-malta-loader-recurrence-s122.md), this changes
only address construction inside the original bounds check.

Uninstrumented 4K encoder medians favor the both-coordinate candidate in all
four tested windows. However, duplicate-control variation is substantial,
short replay is essentially neutral, and instrumented Malta timing reverses
in one confirmation window. This is evidence to continue production-build
qualification, not a demonstrated stable end-to-end speedup. Production is
unchanged in this study; no compatibility layer or runtime dispatch is added.

The prototype and encoder substitution cover only the existing paired
32-by-64 output tile. Production also instantiates other tile heights; their
native resource and performance behavior is not established by this result.

## Change and native evidence

Mode 0 is production and mode 1 is an independent native-identical copy.
Mode 2 converts both validated coordinates to `uint32_t`, then widens the row
to `size_t` for multiplication by the input stride. Mode 3 converts only the
row. Since both coordinates have already passed the same nonnegative signed
bounds checks, the converted integer values are unchanged. Neither mode
changes index division, loop progression, lane ordering, shared locations,
halo predicates, floating-point operations or accumulation behavior.

All 78 original Butteraugli GPU bodies are unchanged. The standalone object
adds twelve prototype bodies. All four mode-1 bodies match production
exactly. Both unsigned variants use the same resources as the control:
48/40 registers for full/low-frequency 2D kernels, 56/47 for flat-grid
kernels, 11,520 bytes shared memory, and zero stack/spill storage. Each body
has two fewer static instructions than its control. The two unsigned variants
are not native-identical despite equal instruction counts.

Native checks cover all 102 bodies in each standalone release/ASAN executable
and all 223 bodies in each whole-encoder release/ASAN executable. The latter
contains 90 target bodies and 133 linked dependencies checked against retained
S117 full-encoder evidence. The encode-only source differs from current
production solely by renaming the public Malta launcher definition, leaving
internal call sites routed through the experimental wrapper. The wrapper
changes only eligible paired-64 launches; all other calls retain production.

Eight counter captures select one eligible launch apiece, in forward and
reverse treatment order, with clock/cache control disabled. Both unsigned
variants execute exactly 71,236,245 warp instructions, versus 71,963,205 for
production and its copy: 726,960 fewer, or 1.01%. All eight runs execute
139,303,116 predicated-on FFMA thread instructions and 367,200 shared-store
wavefronts. Shared-load wavefronts and DRAM/L2 traffic remain essentially
unchanged. Counter-profiled duration is not treated as encode performance.

## Short and sustained replay

Replay uses S117's current 3839-by-2159 full-response call-12 capture, with
original packed strides and idempotent initialization. A scalar scale/response
reference defines expected output. Every burst validates output, guards,
unused scaled storage and unchanged inputs. Four- and 128-launch bursts run
in both orders, using four qualification bursts, four warmup rounds and eight
measured randomized Williams rounds per job. Five CUDA events delimit the
whole burst and its quarters; power-limit queries and validation are outside.

Within each round the two controls are averaged. Positive percentages below
mean slower; rows are medians of eight paired comparisons, not differences
between independently pooled medians.

| Burst / order | Control ms per launch | Both coordinates | Row only | Copy vs production |
| --- | ---: | ---: | ---: | ---: |
| 4 / forward | 0.5566 | +0.33% | +0.33% | −0.12% |
| 4 / reverse | 0.5578 | +0.06% | −0.01% | −0.21% |
| 128 / forward | 2.0171 | +0.63% | +3.52% | +2.18% |
| 128 / reverse | 2.1042 | −1.60% | −0.59% | −2.39% |

Thus reduced integer work does not establish a replay speedup. Both unsigned
variants pass correctness, but only the both-coordinate variant advances to
the whole-encoder experiment below. No row-only encoder claim is made.

## Resident encoder and confirmation

The encoder uses the fully resident mode, target 1.2, effort 7, automatic CPU
thread count and the current pool allocator. Each encode is checked against
the retained codestream oracle and reference summary, including coefficient
storage width and size. Eligible-call counters are 0/12/24 for the 500-pixel,
padded HD and padded 4K cases respectively. Labels 0/2 retain production;
labels 1/3 use unsigned widening. Both treatments therefore have duplicate
controls inside the same executable and randomized Williams blocks.

The initial campaign uses four warmup rounds plus twelve measured rounds,
with two case orders, both without per-kernel events and with events around
each eligible Malta launch. The confirmation uses the same binaries, new
seeds, sixteen measured rounds, HD/4K only, and reverses the ordering of the
event and non-event cases. No compilation, profiling, native dump or second
recorded job overlaps a timed job. All raw results, including the adverse
confirmation window, are retained.

Each number below is the median of within-round differences between the
mean candidate labels and mean baseline labels. The two timing columns come
from separate jobs and are not additive or interchangeable boundaries.

| 4K window | Uninstrumented whole-encode change | Instrumented Malta change |
| --- | ---: | ---: |
| Initial order 0 | −2.588 ms | −1.011 ms |
| Initial order 1 | −7.501 ms | −1.642 ms |
| Confirmation order 0 | −4.264 ms | −1.710 ms |
| Confirmation order 1 | −2.833 ms | +0.669 ms |

Uninstrumented 4K baseline medians range from 302.8 to 330.9 ms; paired
relative changes are −0.84%, −2.41%, −1.37% and −0.86%. Duplicate whole-encode
differences are substantial, reaching about 10.2 ms in the initial campaign
and 7.1 ms in confirmation. The event-instrumented whole-encode changes are
also mixed in confirmation: +0.785 and −1.341 ms. The last instrumented Malta
result changes sign, alongside approximately 0.5–0.6 ms duplicate differences.
These controls limit causal and precision claims about the apparent gain.

Uninstrumented HD changes are +0.135/+0.179 ms initially and −1.006/−0.366 ms
in confirmation. HD Malta event differences are +0.034/+0.037 ms initially
and +0.012/−0.007 ms in confirmation. The small image has no candidate
launches and serves as a control. There is no stable HD speedup claim.

All 3,424 unprofiled enforced-power-limit endpoint records report 40 W:
448 in replay and 2,976 around the 1,488 encodes. No endpoint transition is
observed. This does not establish constant clock/thermal state within each
interval, nor diagnose a particular governor. No operating-state correction
or event-overhead subtraction is applied.

## Qualification and retained evidence

Release and host-ASAN each pass 7,168 three-stage differential fixtures.
Four CUDA sanitizers each pass 160 scoped fixtures with zero reported errors
or hazards, for 14,976 fixture invocations. Coverage includes full/LF,
initialize/add, 2D/flat, partial/tall shapes, signed zero, subnormals/extremes,
infinity, NaN and exceptional accumulation. Replay passes 224 checked bursts
and 14,784 logical Malta launches, including eight host-ASAN bursts. Counter
captures add eight checked four-launch bursts, excluding profiler-internal
replays from the logical count.

The initial encoder campaign checks 840 encodes; confirmation adds 648, for
1,488 exact encoded outputs, including thirty host-ASAN checks. All forty
retained runtime binary hashes pass. No new production build, permanent CTest
run or independent decoder run is claimed by this prototype study.

Evidence is retained under `U:/gjxl-cuda-diagnostics/s123`, including the
`confirm/` campaign, source snapshots, objects/executables, complete native
dumps, capture/library identities, raw logs, analyses and artifact hashes.
Validation covers eighty accepted job records, twenty-four timing isolation
windows against other recorded jobs, native/body identities, source/input
hashes and all stated qualification counts. Unknown desktop activity is not
claimed to be excluded.

The first encoder native gate incorrectly expected the standalone replay's
102 bodies; full encoding links 223. All dump jobs completed, and the
corrected full-set comparison checks all 133 dependency bodies against
retained evidence. This setup failure is preserved in `audit_notes.json`;
no GPU code changed and no encoder timing ran before the corrected gate passed.

All jobs are terminal. No admin/firewall/permission blocker was observed.
No clock, power, security, priority or affinity setting was changed. User
scratch files remain untouched, and no new compatibility surface is added.

## Next decision

The both-coordinate candidate remains worth a fresh production-build
qualification because it removes verified work and all four uninstrumented
4K comparisons favor it. That qualification must check the actual production
specializations, whole-encode correctness and broader workloads; it must not
assume that this forced 64-row prototype proves behavior of 8/24-row kernels.
The mixed event result and duplicate variation remain part of that decision,
not discarded outliers. Neither a guaranteed speedup nor a maxed-out backend
is established here.
