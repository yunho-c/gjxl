# Production CUDA short filters

Normal CUDA Butteraugli preparation now uses the qualified fixed 32-row direct
short filters by default. No environment setting or experimental build option is
required. The CPU-order/exact-coefficient route retains the legacy implementation.
`GJXL_CUDA_SHORT_FILTER_TEST_SCHEDULES` defaults to OFF and enables only the
alternate 16/64-row schedules used for differential testing.

The direct high-X/Y, B-medium and ultra-X/Y filters retain tap order and rounded
division. They reuse the existing 25-plane arena with disjoint transient inputs
and outputs, reducing this stage from 11 launches to 7 without adding persistent
device allocation. The separate-pass implementation remains the test oracle.

## Adoption evidence

The qualification compared upstream `715f033` with this fixed32 implementation
on an RTX 3060 Laptop (6 GiB, SM86), Windows, CUDA 11.8 and MSVC 14.37 Release.
Ordinary settings were distance 1 and eight CPU threads. GPU AC tokenization and
compact AC were disabled. Balanced randomized rounds, unchanged-binary controls,
retained first calls and independent decoding were used.

| Workload | Change in warm complete-encode time |
| --- | ---: |
| 30 natural sources, effort 5 | -1.76%; 95% source-bootstrap interval -2.56% to -1.00% |
| 30 natural sources, effort 8 | -1.85%; interval -2.35% to -1.35% |
| Two distinct 12 MP images, one batch worker, effort 5 / 8 | -5.85% / -5.49% |

Whole-process one-shot time was essentially flat overall (+0.015% equal-case
geometric mean). The second-scale delays motivating the startup investigation
occurred before filter execution in both variants; subsequent warm calls did not
repeat them. Most initial regressions did not reproduce in targeted confirmation.
A small Kodak effort-5 cost remains possible; gains are not universal.

The study completed 13,376 exact-output image encodes, independently decoded 96
distinct cases, and passed 19 CTests and all four Compute Sanitizer modes. The
candidate's 97 GPU function bodies matched the earlier qualified fixed32 build;
all 90 shared baseline function bodies were unchanged. Mixed 12+24 MP batches
exhausted memory in both variants (32 matched failed process arms); these failures
are retained and no memory-capacity improvement is claimed.

This laptop ran under a 40 W limit and showed substantial identical-binary noise.
The intervals cover source variation in this study, not other GPUs or sessions.
Other architectures, Linux, different power modes, cold caches and combined
tokenizer/short-filter performance remain unmeasured.

## Reproducibility and ongoing checks

The local evidence workspace is `gjxl-cuda-metal-catchup/build/short-production-side-r1`.
Its `DECISION.md`, `REPORT.md`, `FINAL.json`, raw measurements and failed runs are
preserved. The [evidence index](cuda-catchup-evidence.json) records their hashes.
The adoption keeps the qualified kernel implementation; only its status comment
and the prepared fixture's use of the default selector were adjusted.

Integration validation on the `0270512` catch-up checkpoint completed on
2026-09-23 using two fresh Release builds. All 23 selected default-build CTests
and six tokenizer composition CTests passed without skips. The direct-short
fixture passed all four sanitizer modes; the tokenizer provider and batch smoke
fixtures passed memcheck and initcheck (eight sanitizer runs total). All 97 short
filter module GPU function bodies matched the qualified build in both variants.
Sources were unchanged throughout validation. Logs, source identities and result
hashes are retained in `build/short-default-adoption-r1` in the delivery worktree
and referenced in the evidence index. This validates correctness of composition;
it does not add a combined performance claim.

The [CUDA workflow](../.github/workflows/cuda.yml) runs both direct-short CTests
and the guarded fixture under memcheck, racecheck, synccheck and initcheck. Its
separate opt-in tokenizer lane also covers composition with the new default.
These are correctness checks, not a performance gate.
