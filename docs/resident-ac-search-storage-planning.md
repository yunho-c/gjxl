# AC-search host-storage bounds

This additive milestone-4 checkpoint is based on `d70ab11`. It extends the
[shared AC device plan](resident-storage-planning.md) with the CPU placement and
backend-independent host owners needed for whole-workflow admission. No encoder
currently installs an admission reservation from these component plans.
**Whole-workflow planning, public execution domains and scheduling remain
unfinished.**

## Owners and lifetime boundary

[CPU search planning](../src/codec/ac_strategy_storage_plan.h) accepts padded
pixel geometry. For `B` base blocks, its output is a fresh `B`-byte strategy grid;
the complete working bound is `2B` because the temporary placement grid survives
until atomic export finishes. Direct CPU candidate evaluation and table-driven
placement use the same two owners. Tile state, coefficients, transform scratch
and candidate-cost arithmetic are stack-only; quantization matrices are immutable
tables. Caller images, fields, supplied cost tables and an old output grid are
separate allocations.

[GPU-search host planning](../src/gpu/ops/ac_strategy_storage_plan.h) composes
that merge bound with the existing exact per-family candidate counts. In the
current seven-family policy, let `N` be the sum of candidate counts, `P = 64B`
the padded pixel count, and `M = 6 * 2624 * sizeof(float) = 62976` matrix bytes.

| Host owner | Fresh backing | Prepared reuse rule |
| --- | ---: | --- |
| Per-family candidate descriptors | `24N` | Explicit reserve, no push beyond the planned count; retained at most `24N`, replacement peak at most `48N` |
| Per-family dequant/inverse matrices | `M` | Size is constant for each family; no growth after initialization |
| Per-family readback costs | `4N` | `resize` can grow geometrically; retained at most `8N`, replacement peak at most `12N` |
| Dense per-strategy block costs | `28B` | Seven `assign(B)` owners; retained at most `28B`, replacement peak at most `56B` |
| Packed input staging | `16P` nonresident, zero resident | Fresh Opsin vector of `3P` floats and one mask vector; both survive through merge |
| CPU placement and exported grid | `2B` | Both fresh on every search; an already tracked old output must be added separately |

Matrices and dense block-cost tables exist even for empty candidate families.
Their device buffers do not; applying the device plan's nonempty-family test to
these host owners would undercount small or thin images.

Thus fresh prepared host storage is exactly `28N + M + 28B`. Its reuse bound is
`32N + M + 28B` retained and `60N + M + 56B` peak. Complete working storage adds
staging and the `2B` merge bound. Summed replacement peaks are conservative;
they need not all happen simultaneously. A working bound's retained field is
an owner-inventory bound, not a prediction of bytes surviving function return.

The fresh plan requires initially empty state. For reused prepared state, the
planned width **and** height must each bound every earlier search since the
owner was empty. An area bound alone does not bound candidate counts for every
family. The bounds cover failed growth followed by recovery, shrinking, changed
images and backend changes (which clear the old prepared state before rebuilding).
They do not transfer charges from another reservation or account arbitrary
historically oversized state. The workflow must keep its original reservation
for such reuse or explicitly account/dispose of the old owner.

The implementation uses checked `HostStorageBound` operations under the reviewed
libc++ C++20 capacity contract. Planning reads no pixels, touches no backend and
allocates no managed backing on success. Rejected/null plans leave outputs
unchanged. CPU geometry is not artificially restricted to the GPU's 32-bit
index domain; GPU host planning uses its existing device geometry validation.

## Qualification and limits of this checkpoint

The new Release tree is `build/resident-ac-host-plans`; its ASan/UBSan counterpart
is `build/resident-ac-host-plans-asan`. Both enable tests, disable benchmarks,
libjxl-reference fixtures and compile-time Metal profiling. Frozen parent
`build/resident-perceptual-plans` is not rebuilt. Initial disk availability was
353 MiB, so qualification builds the relevant targets instead of duplicating a
full suite and raw corpus artifacts. Existing qualification evidence is neither
deleted nor rewritten.

The four targeted Release tests pass: CPU AC cost, CPU search, the existing
8,200-case device plan test, and the new host plan test. All four pass three
times each under ASan/UBSan, with leak detection disabled, both sanitizers
halting on error and no suppressions. Strict `-Wall -Wextra -Wpedantic -Werror`
syntax checks pass for both planner sources and the new test.

Permanent host-plan coverage includes:

- 16,384 geometry/residency/reuse formula cases; invalid/null/overflow atomicity;
  planning a near-32-bit-limit GPU image with a one-byte reservation and an armed
  physical-allocation hook, without consuming backing or escaping to the default
  domain.
- 234 real host execution/reservation cases: 13 padded geometries, three input
  and synthetic-cost variants, and six direct/table/GPU-wrapper/prepared modes.
  Fresh managed peaks match the plan exactly; only the exported grid survives
  local preparation destruction, remains charged after producer closure, and
  drains on output destruction.
- 12 persistent-state transitions spanning growth, shrinkage, changed costs,
  thin images, backend changes and an overlapping previously tracked output.
  Destroying prepared state after producer closure preserves the output charge.
- 98 fresh physical-allocation failures and 15 prepared-growth failures,
  each followed by recovery on the same state/reservation. Failed calls preserve
  old grid contents and, where provided, search statistics.
- Six zero-credit underplans and six one-byte-short late underplans. The latter
  reach export after preparation and placement, then fail without publishing a
  grid or statistics. A submission-failure fixture also recovers on the retained
  prepared owner. Physical failure and `ResourcePlanExceeded` remain distinct.

The new host test uses a synchronous test backend with unmanaged device bytes
and deterministic candidate costs, including ties. This deliberately isolates
the host envelope: it exercises production candidate generation, retained
vectors, readback, CPU merge and failure cleanup, **not Metal arithmetic or
backend metadata storage**. Direct CPU search separately exercises actual DCT
cost evaluation. The existing device-count test supplies an independent frozen
tile-by-tile enumeration oracle.

For real Metal coverage, the unchanged AC-search test is compiled against the
new codec/GPU-operation libraries and the verified frozen parent's Metal and
GPU-Butteraugli libraries/metallib. Parent and this mixed-library candidate
harness each pass three runs, covering scalar/SIMD/factored transforms,
CPU/GPU-grid parity, resident reuse and atomic invalid-input rejection. This is
an explicitly scoped integration check, not a fresh whole-encoder qualification.
All 28 freshly built common execution objects in codec/GPU/GPU-ops are byte-identical
to the frozen parent; only the added/extended planner objects are excluded.
No arithmetic, allocation recipe, kernel, search policy or runtime call site
changes in this checkpoint.

The initial compile failed because the new test omitted the GPU candidate API
header. That include was added; the initial failure log is retained. No numerical
golden or tolerance changed. The inherited CPU quantization golden mismatch is
not covered by this targeted run; it is neither requalified nor reported fixed.

Local commands, file/library hashes, object comparisons and logs are sealed in
`build/ac-host-plan-validation.json` by `build/ac-host-plan-qualification.py`.
Earlier build/test logs remain separately named. Full-suite, corpus/decode,
physical-footprint and paired performance gates are **not newly run here**.
The independent runtime-characterization study was still active during this
work; these functional tests make no quiet-machine timing claim.

## Remaining composition

The AC host and device bounds are not yet a complete AC/encode reservation.
Metal's validated-batch vector, profiled stage/context arrays, submission and
profile-result backing need their own checked metadata bounds. So do the
remaining frontend/evaluator wrappers and attempt/diagnostic/batch-result
envelopes. Whole-workflow composition must account old/new owners, cache capacity
and retry lifetimes before public admission can promise a hard managed limit.
The roadmap's resource, lifetime-disposition and scheduling gates remain intact.
