# S165: isolate ANS validation, state processing, emission, and append

## Decision

This is a diagnostic checkpoint, not a production optimization. Production
remains S162 (`cd37bd4d4c1f1d1a4946bc12c5db258cdc6f2af9`); S163 and S164
did not promote either worker-pool policy. No compatibility layer, public API,
coefficient representation, GPU kernel, tile policy, or scheduler changes here.

The source lead from S164 is real, but its importance depends on the input:

- High-density `p2` streams spend about half of ANS writer time on forward bit
  emission. Appending is another material cost for the larger dense cases.
- Ordinary-density, sparse, and real-image streams spend most writer time in
  reverse token processing and chunk staging, not forward emission.
- The 4K case has a separate large cost: full entropy-model validation repeated
  before each token stream. Its measured share of validation plus ANS writing
  is 49.1–56.4% across whole/retained and automatic/eight-participant runs.

The next experiment should test validation reuse inside the private AC section
writer, followed by lower-overhead bit emission and aligned append. This does
not justify removing validation from public mutable `EntropyCode` consumers.
Neither optimization is implemented or claimed as a speedup by S165.

## Probe and boundaries

Two private diagnostic DLLs link CPU overlays before frozen S162 archives.
`s165_ans.cpp` is production `ans.cpp` with phase boundaries and counters only;
`s165_entropy.cpp` times validation immediately inside `WriteTokenStream` and
separately times its prefix path. The verifier reconstructs both overlays from
the unchanged production sources and requires exact equality.

The whole wrapper uses the fully resident CUDA workflow. The retained wrapper
deep-copies one GPU-produced frame before timing, serializes that unchanged
native owner, and checks each result against its captured whole-encode oracle.
It also checks unchanged backend allocation/submission counts. The harness
checks retained capture count and the explicit participant cap. Original
legacy scheduling is used throughout.

Diagnostic scope is serialized per DLL by a mutex. A coordinator-selected
enabled flag is published atomically; worker threads accumulate through atomic
counters, not coordinator-only TLS. Each ANS call collects local counters and
publishes once per field at exit, with no per-token atomic updates. Counters are
reset before an encode and read after all serializer workers have joined.
This is a diagnostic scope, not a concurrent-encode production facility.

The ANS boundaries are:

| Field | Included work |
| --- | --- |
| Reservation | Reverse-chunk vector construction and `reserve(2 * tokens)` |
| State | Config checks, reverse token traversal, HybridUint encoding, ANS recurrence, and chunk pushes |
| Emission | Temporary writer construction, initial 32-bit state, and every forward `WriteBits` call |
| Append | Appending the temporary to the destination, including destination growth |
| Total | All of the above plus successful-return bookkeeping and temporary/chunk destruction |
| Model validation | Separate full `ValidateEntropyCode` call preceding the ANS writer |

State processing and chunk staging are not separated further. Emission does
not separate allocation/growth, width checking, status handling, and per-byte
bit stores. Validation of models during header writing or entropy search is
outside the new validation counter. Prefix writing has separate counters.

Times are accumulated worker elapsed time and may overlap or include
preemption. They are not CPU-cycle measurements, section wall time, or additive
end-to-end time. ANS total excludes the audit's final atomic publication;
existing outer/profile timers still include instrumentation overhead. The
residual includes cleanup and bookkeeping, not just deallocation.

## Measurements

The diagnostic matrix has 28 jobs: whole and retained, automatic and eight
participants, seven cases. Synthetic heights are width plus six; `p0` is the
low-amplitude sparse case, `p1` ordinary density, and `p2` the tighter-target
high-density case. Real cases are frozen `flower_500` and padded 3839×2159 4K.

Every job uses four warmup rounds and eight measured rotating duplicate-ABBA
rounds, with two samples of each implementation per round. The baseline is
the frozen S162 DLL and the candidate is the instrumented DLL. All 48 calls
per job compare exact bytes and encoding summaries; real baselines additionally
match frozen codestreams. Job order is predeclared and shuffled. These are
diagnostic comparisons, not a production performance trial.

Representative retained/eight results follow. Milliseconds are separate
16-sample medians; percentages are medians of each sample's phase/ANS-total
ratio, not ratios of independently computed medians. They need not sum to 100%.

| Case | ANS calls | ANS total ms | Validation ms | State % | Emission % | Append % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 `p2` | 3 | 0.881 | 0.062 | 42.1 | 51.2 | 6.0 |
| 513 `p0` | 11 | 0.255 | 0.078 | 85.6 | 5.3 | 2.0 |
| 513 `p1` | 12 | 12.125 | 1.259 | 77.5 | 16.4 | 3.9 |
| 513 `p2` | 12 | 48.789 | 1.447 | 39.3 | 44.7 | 15.2 |
| 1025 `p2` | 28 | 290.740 | 6.338 | 37.1 | 51.8 | 9.9 |
| flower 500 | 7 | 2.205 | 0.311 | 71.7 | 22.3 | 4.6 |
| 4K | 144 | 17.763 | 16.272 | 77.9 | 13.0 | 4.1 |

The ranking is consistent across the four mode/budget combinations. Emission
is 43.4–45.4% for 513 `p2` and 50.5–51.8% for 1025 `p2`. In contrast, state
processing is 80.4–86.2% for sparse 513, 74.6–77.5% for ordinary 513,
70.8–71.7% for flower 500, and 76.3–77.9% for 4K.

### Chunk storage and alignment

Counts are identical across measured and warmup samples, budgets, and modes.
The observed reservation is exactly 16 bytes per token: two reserved chunks
per token and an eight-byte `ReverseBitChunk` on this build. Reservation sums
are cumulative requested capacity across streams, not simultaneous live memory,
resident pages, or whole-process peak usage.

| Case | Tokens | Emitted chunks | Reserved MiB | Reserved / used chunk bytes | Bits appended at aligned destination |
| --- | ---: | ---: | ---: | ---: | ---: |
| 65 `p2` | 15,992 | 19,506 | 0.244 | 1.64× | 98.44% |
| 513 `p0` | 18,961 | 327 | 0.289 | 115.97× | 11.00% |
| 513 `p1` | 735,801 | 111,727 | 11.227 | 13.17× | 96.14% |
| 513 `p2` | 836,402 | 1,040,234 | 12.762 | 1.61× | 98.51% |
| 1025 `p2` | 3,294,564 | 4,103,079 | 50.271 | 1.61× | 98.53% |
| flower 500 | 116,747 | 22,462 | 1.781 | 10.40× | 74.81% |
| 4K | 1,131,618 | 90,326 | 17.267 | 25.06× | 68.10% |

An aligned bulk-copy append would address most emitted bits in the dense
cases, but not most sparse-stream bits. Reservation waste is substantial in
the low-emission cases without being the largest measured writer cost.
For retained/eight, reservation is 5.1% of the small sparse writer total,
2.0% for 4K, and only 0.1–0.2% for the larger high-density cases.

## Audit-on/off controls

Forty additional jobs compare audit disabled/enabled in one loaded diagnostic
DLL, with independent states, both creation orders, both budgets and modes,
and five cases. They use the same 4/8 warmup/measured protocol. The disabled
branch still contains overlay code, scope locking, resets, and snapshot reads;
it is not production code with every diagnostic operation removed.

Outer-time differences below are audit-on minus audit-off, median within-round
duplicate-label means. Each cell lists creation orders 0 and 1, in percent.
These are neither confidence bounds nor a correction to subtract from phases.

| Mode / case | Automatic | Eight participants |
| --- | ---: | ---: |
| Whole / 65 `p2` | +1.734, +5.123 | −0.282, +2.222 |
| Whole / 513 `p0` | −0.965, −0.522 | −2.138, +2.638 |
| Whole / 1025 `p2` | +0.959, +0.117 | +0.130, +1.667 |
| Whole / flower 500 | +0.979, −1.645 | −0.313, +0.832 |
| Whole / 4K | +0.168, −0.069 | +1.595, +1.821 |
| Retained / 65 `p2` | +0.093, −5.682 | +2.833, +0.749 |
| Retained / 513 `p0` | −0.540, +0.979 | −6.815, −0.437 |
| Retained / 1025 `p2` | +4.614, +1.347 | −1.901, +1.016 |
| Retained / flower 500 | −0.873, +0.499 | +0.960, +2.567 |
| Retained / 4K | +2.041, +1.012 | +1.543, +0.926 |

The controls are nonzero and sometimes reverse sign. They support treating
the decomposition as a coarse ranking, not exact uninstrumented cost or a
small-effect speedup estimate. Phase ranking is much more separated than
most control shifts; prospective changes still need uninstrumented paired
whole and retained measurements, same-policy controls, and repeated orders.

## Source-backed next experiments

`WriteAcSections` first calls `WriteSimpleAcGlobal`. That path calls
`WriteEntropyCode(ac_code)`, which validates the entire entropy model before
writing it. Only after success does the private section writer launch its
per-group token writes, each of which invokes `WriteTokenStream` and validates
the same model again. The model belongs to the local encoding candidate and
is read through const references during this batch.

For ANS, full validation checks dimensions, context mapping, configs,
frequencies, reciprocal values, and every reverse-map entry, including a
4096-entry uniqueness bitmap per histogram. The 4K encode has 144 ANS writer
calls in total, including its 135 AC groups. This explains a plausible source
of repeated work; the probe does not separately attribute AC versus DC/order
validation time, and does not establish an achievable wall-time saving.

A bounded private validation-reuse experiment should preserve the successful
global-header validation boundary, read-only model lifetime, token/config/state
checks, error propagation, and output atomicity. Public `EntropyCode` remains
mutable, so a general address-based validation cache or unconditional public
validation bypass is not appropriate. No cache or trusted API is introduced
in this checkpoint.

For high-density cases, separately test bulk aligned append and reduced
per-chunk bit-emission overhead. Preserve exact LSB-first order, partial-byte
padding, width/overflow checks, logical vector bounds, and allotment rollback;
do not use unchecked wide stores outside resized vector storage. Eliminating
reverse-chunk storage or fusing composition should be tested separately from
those smaller changes to keep attribution clear.

## Qualification and evidence

Normal MSVC and clang-cl AddressSanitizer entropy suites pass with auditing
enabled: each reports 816 HybridUint configurations, 443,904 value encodings,
scanned ANS coverage for mapped/unmapped and interleaved/offset-split layouts,
and 40 late-section failure cases. The audit wrapper observes 33 ANS and 23
prefix writer calls per executable. This is the existing entropy suite with
an audit scope, not a new exhaustive codec or full CTest campaign.

CUDA memcheck with full leak checking and initcheck each pass a whole flower
500/eight comparison: four exact-byte comparisons and one frozen-oracle call,
zero errors, and zero memcheck leaks. All 11 raw CUDA modules in each new DLL
match frozen S162 modules byte-for-byte. No GPU code is rebuilt.

There are 77 job journals: 76 accepted and one preserved rejection. The first
combined checks build successfully produced both entropy executables, then
failed compiling its control harness because `/TP` applied to `nvml.lib`
placed before `/link`. A separate compile/link recovery produces
`control_v2.exe` without overwriting the failed command's artifacts; the
already-built normal and ASAN entropy executables then pass. This was a
diagnostic command error, not a codec failure or privilege/firewall issue.

Artifacts are frozen under `build-cuda-ninja/profiles/s165-artifacts`.
Preparation verifies all 2,131 S164 artifact files and archives 340 unchanged
production sources. Build, checks, and recovery inputs are separately pinned
and archived. The verifier checks predecessor hashes, exact overlays, all
journals/log hashes, CUDA module equality, and recomputes both analyses.

The initial matrix ran 20:33:40–20:35:06 UTC on 2026-09-09; controls ran
20:39:18–20:41:33 UTC. No build or sanitizer job overlaps either campaign.
Together they contain 2,176 measured calls, 1,088 warmups, 68 oracle calls,
and 68 untimed whole captures for retained states. The two sanitizer jobs
add eight comparison calls and two oracle calls. All 6,544 NVML observations
report a 40,000 mW enforced limit; that is not a claim of constant actual
power, clocks, temperature, or exclusive machine activity.

The protected untracked Markdown files remain unread and excluded. No system
power, clock, thermal, affinity, priority, or security setting is changed.
No admin/firewall blocker was encountered. Production remains unchanged and
the optimization goal remains open.
