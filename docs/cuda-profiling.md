# CUDA profiling integration

CUDA implements stage and dispatch profiling for coherent image-primitive,
AC-candidate and prepared resident AQ submissions through the shared internal
profiling interfaces. The diagnostic public-workflow entry point accepts both
profiles for explicit CUDA fully-resident/throughput Butteraugli-target encodes.
The whole-workflow storage recipe includes the retained graph, nested capture,
submission recordings and snapshot/aggregation overlap. Normal finite-domain
admission and publication apply. A benchmark profile-export interface remains
unfinished; this is not full Metal diagnostics parity.

Timing-enabled CUDA events bracket the compute callback on the backend stream
under its submission lock. Uploads, readbacks and host preparation outside that
callback are not included in device elapsed time. The ordinary path retains
its timing-disabled completion event and creates no diagnostic graph. All 134
kernel launch sites use a thread-local scope hook. Stage mode records launch
identifiers, grid/block dimensions and per-identifier invocation numbers.
Dispatch mode additionally brackets each launch with timing-enabled events.
The C++17 launch units keep allocation and event ownership in the C++20 backend;
the launch scope avoids NVCC 11.8's nested templated-lambda issue.

Resident AQ scopes capture reference preparation, initial quantization, field
adjustment/policy setup and resident evaluation, including optional sparse AC
packing submissions. A scope is thread-local and tied to one backend. Each
captured submission is resolved and appended before its operation can publish
host outputs; this diagnostic path synchronizes accordingly. The current
stages identify whole submissions; their dispatch records identify individual
launches. Kernel IDs use normalized source launch expressions, including source
template parameter names, rather than compiler-demangled specializations.
They are diagnostic labels, not a stable external naming ABI. A compile-time
check enforces the 78-character bound used by the storage planner.
Encoding-only initial quantization and policy setup retain their device-owned
fields through an optional profiling interface, without diagnostic host-field
materialization.

CUDA event durations have a submission-local zero origin. The stage's
`begin_timestamp` is zero; dispatch begin/end values are offsets from that
stage origin. Durations are converted from milliseconds to nanoseconds. These
values must not be compared across submissions or treated as a correlation
with the host clock. Nanosecond units do not imply nanosecond precision.
Stage mode leaves dispatch timestamp fields zero. Dispatch mode checks that
each interval is finite, ordered and contained in its enclosing stage. Event
recording can perturb execution, so diagnostic timings must remain separate
from ordinary whole-call measurements.

Each submission retains its stage graph; resolution creates an independent
snapshot only after successful completion. Managed diagnostic vectors and
strings retain their allocation-domain tickets until destruction/publication.
The shared one-stage submission storage recipe bounds recorded and resolved
graphs, including long identifiers and overlapping snapshots. The CUDA recipe
also charges the growable event-handle array; opaque runtime event allocations
remain under the driver-internal exclusion. Whole-workflow launch bounds follow
the seven transform-family batches, radix quantizer, at most two perceptual
scales, bounded reduction passes, AQ iterations and packing calls. Both modes
reserve the larger dispatch-mode bound. A metadata/event failure suppresses
later launches, drains queued work and preserves unpublished caller outputs.

The real-device test checks dependent primitive results, repeated resolution,
foreign/unprofiled submissions, mismatched modes, insufficient reservations,
failed completion and retained-domain lifetime. AC-search coverage compares a
profiled search with the ordinary result while omitting its redundant host mask.
Concurrent and nested captures check thread/backend isolation, invocation
numbering, finite diagnostic reservations and scope restoration. Additional
allocation-failure sweeps cover failures between dependent kernels and during
snapshot resolution, then verify recovery on the same backend. One hundred sixty
fresh/reused resident pipeline comparisons cover two fixtures, efforts
1/4/7/8/10, fully-resident/throughput modes and final-score toggles; profiled
and ordinary results have identical codestream bytes and score histories.
Injected initial-quantization completion failure and invalid preparation keep
caller outputs unchanged.

An additional 160 fresh/reused public-workflow comparisons preserve codestream
bytes and full summaries under their calculated finite admission bounds. The
profile shapes fit the policy-derived counts, published profiles release domain
charges, and allocations remain in the selected domain. A sweep of all 678
diagnostic allocation boundaries for a representative effort-7 workflow keeps
every caller-visible output unchanged on failure. A reservation one byte below
the complete plan is rejected before GPU work. Planning itself allocates no
managed backing across 400 geometry/policy combinations, including odd 4K.

The ordinary-path overhead screen compares the pre-dispatch `6c90d83` build
with this implementation on the same RTX 3060 Laptop/CUDA 11.8/MSVC 19.37 host.
Each of four pairs alternates process order and measures nine complete public
calls; medians use samples 3--9. All output signatures match across builds.

| Workload | Median paired time ratio (instrumented/control) |
| --- | ---: |
| 129x133, effort 1 | 1.001 |
| 129x133, effort 7 | 0.945 |
| 1919x1079, effort 1 | 1.026 |
| 1919x1079, effort 7 | 1.007 |
| 3839x2159, effort 7 | 0.988 |
| 3839x2159, effort 10 | 0.987 |
| Native flower photo, effort 8 | 1.008 |
| Mixed batch, two callers, effort 7 | 0.966 |

The 1080p effort-1 result prompted twelve additional pairs with 31 samples
per process; their median ratio is 0.9875, with individual ratios 0.957--1.063.
The initial slowdown did not reproduce. These noisy single-device controls do
not establish a speedup or a general overhead bound. Scripts, binary hashes,
all cold/warm samples and output signatures are retained under the local
`build/integration-evidence/dispatch-performance*` artifacts.
