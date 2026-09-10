# S178: device-counter observation of selection cadence

The diagnostic supports a cadence-sensitive elapsed-counter-rate effect in
4K encoding: GPU-only selection changes the aggregate sampled cycle count by
less than 1%, but increases sampled elapsed time by 20–28%. Post-selection
pauses reverse the rate/time pattern. The observer is not neutral, the rate
varies strongly across launches, and no physical hardware cause or production
speedup is established. Nothing is promoted.

## Question and scope

S175's compact device-side strategy selection removes host cost staging but
does not consistently improve whole encoding. S176 locates its large-image
offset in later, unchanged GPU kernels. S177 changes only the post-selection
host pause and changes those kernels' execution durations. This stage asks
whether sampled device elapsed-cycle/global-timer rates also change with that
cadence. It does not add a production pause or normalize benchmarks to an
assumed GPU frequency.

The parent is `328dee26ebd3d737d710a3fdecf6e71b8fa9dac0` on `feat/cuda`.
The promoted S168 implementation and all 343 pinned production files remain
unchanged. Work is confined to diagnostic overlays and this report. No
compatibility layer is added.

Local evidence is in `build-cuda-ninja/profiles/s178-artifacts` (C drive).
Versioned helpers are `build-cuda-ninja/profiles/s178_*`. Prior frozen helpers
and artifacts are neither edited nor rebuilt.

## Observer and controls

The existing paired Malta kernel remains present. A clone has exactly the
same arithmetic body, with counter reads around sampled thread 0's execution.
At most 128 evenly spaced linear blocks per launch are sampled, with capacity
for 64 launches and 8,192 records. Each 64-byte record contains two triples
`globaltimer-before / clock64 / globaltimer-after`, both SM IDs, the linear
block index, and a per-encode epoch. Launch metadata records geometry, low-
frequency specialization, grid shape, sample stride, count, and offset.

The first recorder holds beginning values across the arithmetic body. The
second writes them immediately and commits the epoch after its ending values.
The 64-row, non-flat high/low-frequency register counts are:

| Entry | High-frequency | Low-frequency |
| --- | ---: | ---: |
| Original | 48 | 40 |
| First observer | 63 | 54 |
| Second observer used for whole encoding | 55 | 48 |

Shared storage remains 11,520 bytes for these entries. All 24 paired entries
in each compilation report zero stack and spill bytes. The observer still
changes resource usage; lower register usage is not a claim of negligible
measurement overhead.

Each whole process fixes one observer mode:

| Mode | Paired Malta entry | Record stores |
| --- | --- | --- |
| 0 | Original | None |
| 1 | Observer clone with null record pointer | None |
| 2 | Same observer clone | Enabled in sampled blocks |

All modes plan identical launch metadata and allocate identical guarded
device buffers and first-touched maximum host buffers before warm-up. After
each timed Encode and its final NVML endpoint, the harness synchronizes,
copies the full 524,800-byte guarded device buffer, validates records, and
copies active metadata/records into a preallocated host buffer. Binary output
is created exclusively and written after every encode in that process has
finished. Device readback and host copying are outside Encode but can change
cadence. The controls are therefore part of the experiment, not optional
bookkeeping.

## Meaning of the counters

CUDA documents `clock64` as a per-SM cycle counter; elapsed counts include
time-slicing rather than counting only a thread's useful instructions.
See [CUDA's clock documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#clock-and-clock64).
The [CUDA 11.8 PTX reference](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi)
describes `globaltimer` as target-specific tooling state whose behavior may
change on other targets. This diagnostic targets the existing `sm_86` build.

For a same-SM record with positive inner time span, the reported rate is
elapsed cycles divided by the midpoint-bracket elapsed time, multiplied by
1,000 to express an MHz-like counter rate. Outer and inner timer brackets
also bound that estimate. Zero-inner-span and cross-SM records remain in the
raw data and explicit counts, but not rate distributions. This is a sampled
thread's elapsed span, not a whole-kernel duration or proof of a physical
clock, thermal, power-policy, or driver cause.

## Qualification and protocol

Both observer versions pass 7,056 cases and 21,168 three-stage accumulation
checks in each of normal and host-ASAN builds: 28,224 cases / 84,672 stages
combined. They use the existing guarded Malta fixture, exact separate-pass
and zero-aware differential oracles, original input bytes, output/work guards,
signed zeros, NaN payloads, infinities, partial tiles, 8/24/64-row and flat/non-
flat variants, initialization and accumulation, large images, and thin images.
Observer checks cover record bounds, epochs, block indices, geometry, timer
ordering, SM identity, and rejected duplicate initialization/begin/null-read
calls. This is not an exhaustive allocation-failure or concurrent lifecycle
qualification of the diagnostic controller.

For the second observer, memcheck, racecheck, synccheck, and initcheck each
pass 648 cases / 1,944 stages. Reports contain zero errors/hazards; memcheck
also reports zero leaked bytes. Racecheck runs about 6 minutes 35 seconds
with regular process heartbeats and advancing mode markers, not a firewall
or admin-prompt stall.

Normal and ASAN whole DLLs contain the same 12 native CUDA modules. Eleven
match S177 byte-for-byte. In the changed module, all 80 original entries have
identical disassembled instructions and encoded words; only the 12 observer
entries are added. The two fixtures contain the same observer module plus two
unchanged support modules. Source auditing independently strips the include,
clone, and host dispatch addition to recover the complete original CUDA file.
All existing S177 whole-output, summary, coefficient-owner, phase, selection,
actual-pause, and Williams-order checks are preserved.

Whole preflight passes all 11 inputs, auto/eight threads, normal/ASAN builds,
and three observer modes: 132 processes and 1,188 encodes. Each process uses
one normal S168 reference encode and eight measured-build calls. Thus the
66 ASAN processes exercise 528 ASAN Encode calls plus 66 normal references.

The timing protocol is fixed before preflight: 1080p and 4K, eight threads,
three observer controls, two reversed process-order passes. Each process
performs one S168 reference, eight warm-up and sixteen measured eight-label
Williams rounds. Labels 0/1 are original CPU selection, 2/3 GPU selection with
no pause, 4/5 GPU selection requesting 8 ms, and 6/7 requesting 24 ms. Actual
pause time remains inside frontend and whole-encode time; it is never
subtracted from either reported total.

Primary within-process comparisons use the median of 16 within-round
differences between duplicate-label means. Per-encode quantization minus
frontend time is a host-phase proxy, not direct GPU timing. Counter summaries
first take a median across sampled threads per launch, then an equal-weight
median across launches per encode; all per-launch comparisons remain in the
analysis. Observer-mode comparisons cross process boundaries and are reported
as descriptive group-median differences, not within-process paired estimates.

## Results

All 12 timing processes pass: 2,316 encodes including references/warm-up,
1,536 measured encodes, and 2,299,392 raw records including warm-up. There are
no zero-inner-span or cross-SM records in this timing cohort. All 4,608 timing
NVML endpoints report the existing 40,000 mW limit. Preflight and timing
together exercise 3,504 whole encodes.

For recording-enabled 4K, the table gives paired changes for pass 0 / pass 1.
These are separately computed hierarchical statistics, not ratios obtained
by dividing already aggregated cycle and time medians.

| Comparison | Sampled cycles | Sampled elapsed time | Elapsed-counter rate |
| --- | ---: | ---: | ---: |
| GPU zero-pause vs CPU selection | -0.69% / -0.23% | +28.16% / +20.34% | -27.68% / -18.50% |
| GPU 8 ms request vs GPU zero-pause | +2.06% / +1.18% | -29.77% / -29.72% | +61.53% / +32.35% |
| GPU 24 ms request vs GPU zero-pause | +1.78% / +1.36% | -30.97% / -40.36% | +75.14% / +76.05% |

The 8 ms requests actually take median 15.948 / 17.000 ms; 24 ms requests
take 32.690 / 31.167 ms. The quantization-minus-frontend proxy increases
17.944 / 16.669 ms for GPU zero-pause vs CPU. Relative to GPU zero-pause,
the 8 ms request decreases that proxy by 25.651 / 27.046 ms, and the 24 ms
request by 29.773 / 32.339 ms. Whole-encode changes remain distinct: GPU
zero-pause vs CPU is -2.123 / +4.566 ms; 8 ms vs GPU zero-pause is
-5.029 / -13.025 ms; 24 ms vs GPU zero-pause is +2.460 / -0.482 ms.

The direction is not an artifact of only one launch: with GPU zero-pause vs
CPU, elapsed time increases for 22/24 and 24/24 launch ordinals, and the rate
decreases for 21/24 and 24/24. Both pause requests reduce elapsed time and
increase rate at all 24 ordinals in both passes. Individual cycle-count
changes are heterogeneous; the nearly constant cycle count above describes
the specified aggregate, not every launch or useful instruction count.

For illustration, selected 4K launch group-median rates are below. Each cell
is the range across the two passes, in MHz-like elapsed-counter units, not
a claimed physical SM clock. Ordinals are zero-based.

| Launch ordinal | CPU selection | GPU zero-pause | GPU 8 ms request | GPU 24 ms request |
| --- | ---: | ---: | ---: | ---: |
| 0: first full-size high-frequency entry | 797–893 | 572–598 | 984–986 | 991–992 |
| 12: later full-size high-frequency entry | 272–298 | 259–279 | 294–330 | 303–349 |
| 23: final half-size low-frequency entry | 243–256 | 233–253 | 246–261 | 250–274 |

Full-size observed geometry is 3839×2159; half-size is 1920×1080. All 24
launches use non-flat 64-row tiles and 128 samples. Raw counter brackets and
per-launch distributions are retained. This launch dependence is another
reason not to apply one frequency factor to the encoder's whole duration.

At 1080p, GPU zero-pause vs CPU changes the downstream proxy only -0.599 /
-0.122 ms in recording mode. Its rate contrast changes sign between passes
(+53.672 / -21.542 MHz-like units). Pauses add roughly 16 / 31–32 ms to whole
encoding rather than delivering a useful total-time gain.

### Observer effects prevent a neutral-overhead claim

Cross-process 4K null-recorder minus original-kernel whole group medians
range from -7.185 to +9.845 ms. Recording minus null-recorder ranges from
-16.158 to +9.878 ms. Recording minus original ranges from -7.261 to
+13.286 ms. These are not paired estimates of a fixed recorder cost.
Within-process duplicate-label whole deltas also span -12.160 to +8.391 ms
across the six 4K processes. No neutral overhead or universal speedup follows.

Nevertheless, the downstream cadence direction survives all observer modes:
GPU zero-pause vs CPU adds 13.871–22.182 ms to the 4K proxy; 8 ms vs GPU
zero-pause removes 17.575–27.046 ms; 24 ms removes 26.330–32.339 ms. Readback
itself takes group medians about 0.271–0.312 ms outside the timed Encode.
This is evidence for the measured cadence/rate relationship, not proof that
temperature, physical frequency, time-slicing, or a particular driver policy
alone explains the original regression.

## Recoveries and limitations

The initial preparation stops at its 700 MB free-space gate, before creating
the stage root or starting a build. A later read-only check finds 610,725,888
bytes free on C. Six completed, non-inventoried Nsight ETL scratch files from
S176/S177 are losslessly archived, then the exact originals are removed only
after entry size/SHA-256 verification and original-file rehashing. The
1,174,405,120 original bytes remain recoverable in two ZIPs totaling 1,310,888
bytes, copied into this stage's `recovered_scratch` with their inventories.
Both predecessors' complete frozen inventories remain unchanged.

The first archival helper fails before creating a ZIP or deleting files
because Windows PowerShell needs the explicit compression assembly. A new
version adds that assembly. The first CUDA compilation succeeds, but the
fixture link lacks two existing AQ reduction entrypoints; a corrected build
links the pinned support library and reuses the compiled GPU object. The
first observer's passing qualification is retained when the lower-register
second observer is built in a new directory. No pinned helper is edited.

The first native audit expects two fixture modules but finds three. Its
completed disassemblies and normal-fixture extraction are preserved. A new
audit validates one observer plus two unchanged support modules, reuses those
outputs, and performs only the missing ASAN extraction. All 80 original
entries already matched. This correction requires no rebuild or fixture
rerun. These helper failures are distinct from encoder correctness failures;
none is hidden by overwriting output.

All tool observation timeouts resume their existing process handles. No
firewall/admin prompt blocks this stage. Nothing changes clocks, power limits,
cooling, priorities, affinity, the driver, security, or firewall settings.
No other benchmark, build, sanitizer, or profiler is intentionally run during
the timing campaign. Light report editing and ordinary shared-machine
activity remain limitations; this is not an isolated hardware experiment.

No fresh Nsight trace, CTest, install-consumer, independent decoder, concurrent
encoder, batch encoder, Linux, or Metal qualification is claimed here. The
counter observer and experimental pauses are not production features.

## Evidence and decision

| Phase (UTC, September 10, 2026) | Start | Finish |
| --- | --- | --- |
| First observer full normal/ASAN fixtures | 06:56:46.034 | 06:58:05.222 |
| Second observer normal/ASAN build | 07:05:08.785 | 07:05:43.832 |
| Second observer full normal/ASAN fixtures | 07:06:06.644 | 07:07:21.313 |
| Four CUDA sanitizer tools | 07:08:36.646 | 07:15:31.506 |
| Whole-probe normal/ASAN build | 07:15:38.224 | 07:16:02.761 |
| All-input whole preflight | 07:18:48.379 | 07:23:11.678 |
| Twelve-process timing campaign | 07:23:30.225 | 07:30:29.245 |

There are 156 journaled jobs: 155 accepted and the retained initial fixture
link failure. The two startup-helper failures and native-inventory assertion
are separately preserved/described, not counted as successful journaled
encoder runs. All journaled processes are terminal and nonoverlapping.
About 1.374 GB remains free on C at the post-verification observation; no new
data is placed on the nearly full U drive.

`before.json`, build/protocol input inventories, all compile/test logs,
`source_audit.json`, `native_v2.json`, raw `.bin` records, `preflight.json`,
`measurements.json`, `analysis.json`, and `decision.json` retain the evidence.
`final_sources.json`, `final_summary.json`, and `artifact_hashes.json` record
the final source snapshot, verification summary, and immutable inventory.
The verifier rehashes predecessor and current artifacts, checks both recovered
ZIPs against their original entries, recomputes native identity, parses every
whole log and raw counter stream, checks complete schedules and all outcomes,
and recomputes the reported statistics:

```powershell
python build-cuda-ninja/profiles/s178_verify.py --frozen
```

Do not promote a pause, the recorder, or a whole-time frequency correction.
The new evidence constrains further compact-consumption and scheduling work:
local time/traffic reductions are not safely additive, and changes still need
reproducing end-to-end gains against original and duplicate controls. This
stage narrows the timing question; it does not establish that the encoder is
maxed out. The optimization goal remains active.
