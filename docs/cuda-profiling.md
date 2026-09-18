# CUDA profiling integration

CUDA currently implements stage profiling for coherent image-primitive,
AC-candidate and prepared resident AQ submissions through the shared internal
profiling interfaces. The diagnostic public-workflow entry point accepts stage
profiles for explicit CUDA fully-resident/throughput Butteraugli-target encodes.
The whole-workflow storage recipe includes the retained graph, nested capture,
submission recordings and snapshot/aggregation overlap. Normal finite-domain
admission and publication apply. Dispatch records and a benchmark profile-export
interface remain unfinished; this is not full Metal diagnostics parity.

Timing-enabled CUDA events bracket the compute callback on the backend stream
under its submission lock. Uploads, readbacks and host preparation outside that
callback are not included in device elapsed time. The ordinary path retains
its timing-disabled completion event and creates no diagnostic graph.

Resident AQ scopes capture reference preparation, initial quantization, field
adjustment/policy setup and resident evaluation, including optional sparse AC
packing submissions. A scope is thread-local and tied to one backend. Each
captured submission is resolved and appended before its operation can publish
host outputs; this diagnostic path synchronizes accordingly. The current
stages identify whole submissions, not individual kernels or AQ iterations.
Encoding-only initial quantization and policy setup retain their device-owned
fields through an optional profiling interface, without diagnostic host-field
materialization.

CUDA event durations have a submission-local zero origin. `begin_timestamp`
is zero; `end_timestamp`, `gpu_nanoseconds` and the submission duration are
the device elapsed duration converted from milliseconds to nanoseconds. These
values must not be compared across submissions or treated as a correlation
with the host clock. Nanosecond units do not imply nanosecond precision.
There are no invented dispatch timestamps: dispatch mode is unavailable and
the capability bit is false.

Each submission retains its stage graph; resolution creates an independent
snapshot only after successful completion. Managed diagnostic vectors and
strings retain their allocation-domain tickets until destruction/publication.
The shared one-stage submission storage recipe bounds recorded and resolved
graphs, including long identifiers and overlapping snapshots. CUDA event
handles are opaque runtime resources, under the same driver-internal exclusion
as ordinary completion events.

The real-device test checks dependent primitive results, repeated resolution,
foreign/unprofiled submissions, unsupported modes, insufficient reservations,
failed completion and retained-domain lifetime. AC-search coverage compares a
profiled search with the ordinary result while omitting its redundant host mask.
Concurrent and nested captures check thread/backend isolation, invocation
numbering, finite diagnostic reservations and scope restoration. Eighty
fresh/reused resident pipeline comparisons cover two fixtures, efforts
1/4/7/8/10, fully-resident/throughput modes and final-score toggles; profiled
and ordinary results have identical codestream bytes and score histories.
Injected initial-quantization completion failure and invalid preparation keep
caller outputs unchanged.

An additional 80 fresh/reused public-workflow comparisons preserve codestream
bytes and full summaries under their calculated finite admission bounds. The
profile shapes fit the policy-derived counts, published profiles release domain
charges, and allocations remain in the selected domain. A sweep of all 55
diagnostic allocation boundaries for a representative effort-7 workflow keeps
every caller-visible output unchanged on failure. A reservation one byte below
the complete plan is rejected before GPU work. Planning itself allocates no
managed backing across 400 geometry/policy combinations, including odd 4K.
