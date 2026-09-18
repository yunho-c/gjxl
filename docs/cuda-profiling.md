# CUDA profiling integration

CUDA currently implements stage profiling for coherent image-primitive and
AC-candidate submissions through the shared internal profiling interfaces.
This is an integration foundation, not complete public-workflow profiling:
prepared AQ, dispatch records and whole-workflow diagnostic storage planning
remain unfinished. Public CUDA GPU-profile requests still fail explicitly.

Timing-enabled CUDA events bracket the compute callback on the backend stream
under its submission lock. Uploads, readbacks and host preparation outside that
callback are not included in device elapsed time. The ordinary path retains
its timing-disabled completion event and creates no diagnostic graph.

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
