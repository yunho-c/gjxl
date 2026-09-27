# Earlier entropy readiness for single-image encoding

Eligible resident Metal single-image calls use earlier entropy readiness by
default in normal builds. No experimental build flag is required. Single-image
APIs favor individual-image latency; use `VarDctBatchEncoder` for multiple images.
Concurrent ordinary calls remain supported and share CPU/memory limits, but this
policy can reduce their aggregate throughput. It does not detect or serialize
concurrent callers.

## Scheduling and eligibility

GPU AC tokenization starts at the existing point. One branch prepares DC tokens
and their entropy model. The other prepares any coefficient-order model,
consumes completed GPU AC tokens/populations, and builds the AC entropy model.
Section writing waits for both branches. Encoded bytes and codec decisions are
unchanged.

Before launching either branch, admission requires the complete desired DC
participation plus one AC participant from both the per-image and shared-domain
CPU budgets. It releases any partial grant before falling back. The complete
grant is split without returning reserved DC workers to the shared pool. Nested
DC dispatch therefore retains its promised capacity. Reservation is nonblocking
and does not bypass queued callers.

The policy applies to the resident Metal fully-resident and throughput routes
that install an AC tokenization provider. Batch calls (including a batch driver
with one in-flight image), CPU-token routes, detailed serializer profiles,
rate-optimized dual-representation encoding, explicit nested work, missing CPU
participation and insufficient capacity retain the original schedule. Existing
exhaustive and maximum-error route exclusions remain in place. A single CPU
participant cannot enable the overlap. The section-worker ceiling is eight;
sufficient nominal domain capacity alone does not imply admission.

## Controls

`GJXL_EARLY_ENTROPY=0` retains the original readiness schedule without disabling
GPU tokenization. Unset, `1`, and unrecognized values use the capacity-aware
default. Configure this process-wide setting before encoding and keep it fixed
while calls or batches are active. The storage bound conservatively covers both
schedules, independently of this setting.

Diagnostic builds with `GJXL_BUILD_TOKENIZATION_EXPERIMENT=ON` additionally accept
`GJXL_EXPERIMENT_EAGER_ENTROPY=0/1/2/3`: original schedule, earlier completion,
static early-entropy split, and capacity-aware early entropy respectively. Unset
or invalid values select mode 3. The stable disable switch takes precedence;
normal builds ignore all four experimental values. Batch exclusion applies to
all modes.

## Profiling contract

The public `EncodeLinearRgbVarDctCodestreamProfiled` API returns workflow timing
and uses the same scheduling policy as ordinary calls. Surround the public call
to include teardown and result publication in complete latency.

Internal detailed serializer profiling intentionally uses the original schedule.
Its additive phase durations must not be presented as a decomposition of normal
early-entropy latency. This preserves the existing phase-accounting contract;
independent diagnostic traces are needed to explain overlapping execution.

## Memory and failure behavior

Normal-build serializer plans include the outer dispatcher's two `Status` entries
and one `std::thread` entry in addition to the summed token/model envelopes.
Plans cover fallback cases conservatively. Split reservations and their owners
are stack objects retaining existing budget states; there is no new task queue.
Provider-owned token storage survives every entropy reader, all workers join
before return, and failed calls leave public output unchanged. Managed admission
is a capacity bound, not a process RSS bound.

## Evidence and rollout decision

The frozen [wider AC study](../reports/entropy-capacity-wide-20260927/REPORT.md)
covered 105 inputs on one M4 Pro. Warm individual-image gains were strongest on
CLIC; Kodak and large-image gains were smaller, and cold-call results were mixed.
It also found repeated concurrent-call regressions. Those results are retained:
default enablement is an explicit choice to prioritize the intended serial use
of the single-image API, with the batch API recommended for multi-image work.
The historical report's keep-opt-in recommendation predates that accepted API
policy; it is not rewritten as a claim that the regressions disappeared.

The [production integration qualification](../reports/entropy-default-20260927/REPORT.md)
records normal-build tests, the before/after comparison, bounded-memory checks,
source/binary hashes and preservation of unrelated working-tree changes.
Qualification is AC-only at the user's request and does not establish a
cross-device, cold-start, or concurrent-throughput improvement guarantee.
