# GPU-profile graph storage bounds

This milestone-4 checkpoint follows `5ebd24e`. It supplies the string and nested
profile-graph bounds required to include diagnostics in whole-workflow admission.
It does **not** finish backend metadata planning, infer workflow counts, install
a public execution domain or change scheduling. Ordinary execution still creates
no profiling session. Existing allocation/publication behavior is unchanged.

## String backing is not string length

[`HostStorageBound::AddString`](../src/core/host_storage_bound.h) accounts char
storage including its terminator under the reviewed libc++ C++20 contract. The
review used the active Xcode macOS SDK's `string` implementation: `__recommend`,
`__grow_by_and_replace`, `__grow_by`, and `__shrink_or_extend`. The local validation
manifest pins that header's hash. This is not an ISO-library capacity guarantee.

Let `L` bound every size and reserve request since a string owner was empty. If
all requests fit inline storage, there is no separate character allocation: the
containing record already accounts for those bytes. Otherwise:

| Construction history | Retained bound | Replacement peak bound |
| --- | ---: | ---: |
| Fresh count/pointer/forward-range/copy construction | `L + 32` | `L + 32` |
| Assignment, append, resize, reserve, insert and shrink-to-fit | `2(L + 32)` | `3(L + 32)` |

The 32-byte allowance conservatively covers terminator, eight-byte rounding and
the inline/long boundary adjustment. Growth occurs only when old capacity is
below the requested new length; it doubles that capacity and rounds. The
near-`max_size` saturation branch is also covered. Shrinking can retain the old
doubled allocation while making one fresh replacement. Bounds validate the
allocator/string maximum and all arithmetic before changing their output.

Cleared strings can retain long capacity. Moving from a historically larger
owner, nontrivial/input iterators and non-char strings require separate bounds.
Allocator headers and small control objects retain the established exclusion.

## Profile graph and submission overlap

[`ComputeProfileStorageBound`](../src/gpu/ops/profile_storage_plan.h) combines
growing-vector bounds for wall stages, submissions, stages and dispatches with
their labels. If their count bounds are `W`, `U`, `S`, `D`, there are
`W + U + 2S + D` string owners. Counts mean the sum of each vector owner's maximum
size/reserve history, not just a snapshot's live logical counts. Child graphs
still alive during session aggregation and old output graphs remain separate
owners. The same applies to labels moved into a graph from elsewhere.

`ComputeSubmissionProfileStoragePlan` describes the source-backed Metal recipe:

- Recording reserves the stage array exactly once, builds stage/group labels,
  and appends dispatch records and kernel labels. Stage-mode profiling also
  records dispatch metadata even though it does not sample each dispatch's
  timestamps. Thus its dispatch count cannot be assumed zero.
- The submission retains this original graph until destruction. Its submission
  ID is still empty; resolution names a separate snapshot.
- Resolving the completed submission makes a deep copy of stages, dispatches and
  their strings. Assigning its submission ID can take the string growth path.
  The copy then moves into a one-element execution-profile submission vector.
- `resolution` adds the original graph's retained bound to the complete snapshot
  and wrapper peak. It does not substitute the smaller final returned size for
  this simultaneous ownership.

The kernel-label bound must include the fallback expression
`stage_id + ".dispatch_" + decimal invocation`, not only the registered kernel
name limit. The first concatenation can grow when the numeric suffix is appended;
that replacement is included. The decimal temporary has at most 20 characters
on the reviewed 64-bit platform and fits inline storage. Stage/group IDs supplied
by callers also need correct maximum lengths.

These functions are pure size planning: no pixel access, backend initialization
or managed backing on success, and atomic plan failure. They are used by tests,
not yet by workflow admission. They exclude input stage/context arrays and
callback-owned temporaries. Metal counter buffers, resolved `NS::Data` and
command/driver internals retain the documented driver exclusion; host graph
arrays and strings are not excluded.

## Qualification

Small fresh Release and ASan/UBSan trees are `build/resident-profile-plans` and
`build/resident-profile-plans-asan`. Tests are enabled; benchmarks,
libjxl-reference fixtures and compile-time Metal profiling are disabled.

Three targeted Release tests pass: the new profile planner, existing diagnostic
ownership/publication test, and the preceding AC host-storage planner. All three
pass three repetitions each under ASan/UBSan, with leak detection disabled,
halt-on-error enabled and no suppressions. Strict warning checks cover both
CPU and Metal variants of the new test and the new planner source.

The new CPU test covers 574 string copy/growth/shrink/publication cases around
inline and allocation boundaries through length 4096; 48 submission shapes with
up to seven stages and 257 dispatches; 23 snapshot allocation failures with
recovery; zero-credit planning and a typed snapshot underplan with the original
graph still charged. It also exercises production session aggregation, repeated
submission IDs, cleared labels/dispatch arrays, producer closure and publication.
The existing diagnostic suite retains its 29 graph-failure positions, nested
ownership and cross-thread publication coverage. AC regression coverage remains
the preceding 234 execution cases and 113 failure/recovery positions.

The Metal variant of the new test is registered permanently. For this checkpoint
it is compiled against the fresh codec/GPU-operation libraries and the verified
frozen `d70ab11` Metal/GPU-Butteraugli libraries and metallib. Twelve real
stage-profile cases vary ID length (including the inline boundary) and one,
three or seven affine dispatches. Measured managed recording/resolution storage
fits the planned envelope; output pixels, profile shape and publication charges
are checked. Dispatch-boundary timestamps are unsupported on this M4 Pro and
are reported as skipped, not qualified. This mixed-library check is not a fresh
whole-encoder build or native Metal sanitizer qualification.

Initial builds hit `No space left on device` while generating shaders/archiving
Metal, because the existing diagnostic test unnecessarily linked the entire
codestream library. Standalone Release and sanitizer compilation confirmed that
this test only needs core headers. Its CMake dependency is now `gjxl_core`; the
three targeted builds subsequently succeed without rebuilding Metal. About
25 MiB of newly generated, unqualified shader/object files from these failed
attempts were removed. They are reproducible; failure logs, successful CPU
artifacts, all frozen builds and all earlier qualification evidence remain.

Local evidence is sealed by `build/profile-plan-qualification.py` in
`build/profile-plan-validation.json`: exact commands, source/test/library hashes,
active libc++ header hash, repeated tests and 30 byte-identical common Release
objects against `5ebd24e`'s frozen AC-host tree. No production execution source changes.
Full-suite/corpus/decode, physical-memory and paired performance gates are not
newly run here. The separate runtime study remained live, and disk availability
was critically low; no timing or whole-workflow-completion claim follows from
these functional tests. The inherited CPU quantization golden mismatch is not
covered by this targeted run and is not reported fixed.

## Remaining composition

The next required bounds are backend validated-batch and stage/context metadata,
remaining AQ/evaluator host owners, and policy-to-stage/dispatch/attempt counts.
Those counts must include failure paths and timestamp-capacity overflow paths;
the 4096-sample dispatch limit alone is not a graph-allocation bound because
recording continues within the callback before overflow is reported. Stage mode
does not use that dispatch timestamp limit at all. These graphs must then be
combined with existing host/device, retry, diagnostics and retained-batch-result
owners before public admission can promise a hard managed limit. Resource,
lifetime-disposition and scheduling milestones remain unfinished.
