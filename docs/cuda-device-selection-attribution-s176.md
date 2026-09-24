# Device strategy selection: downstream attribution (S176)

September 10, 2026. Starting revision: `c6bee76`, branch `feat/cuda`.
Windows, RTX 3060 Laptop (`sm_86`), CUDA 11.8, Nsight Systems 2023.2.3.

## Outcome

S175's saved frontend time is offset by slower execution of unchanged
downstream kernels, not longer device gaps. In two balanced 4K/eight-thread
traces, the new selection kernel costs 0.819/0.822 ms; the 230 standard
kernels after it take an additional 11.732/10.463 ms. Whole-device gaps
shrink by 11.189/11.153 ms. These are within-round paired medians, not
additive independent components or unprofiled throughput results.

No runtime change is promoted. The trace locates the offset but does not
establish a power, clock, thermal, cache, allocation, preemption or RDP cause.
The [S175 unprofiled campaign](cuda-device-strategy-selection-s175.md)
remains the performance decision: all four primary 4K cells were slower
despite substantial frontend savings. A favorable second traced whole-call
result does not override that campaign.

## Controlled capture and qualification

The original S175 corrected normal/ASAN DLLs and S168 reference DLL are
reused without rebuilding. Only a new normal/ASAN harness is compiled. Its
source is exactly the S175 harness plus a trace flag, NVTX header and one
named range around each complete `Encode` call. A source check reconstructs
those edits. Input/setup, reference generation, result clearing/checking,
NVML calls and logging remain outside the timed call. No new per-stage
instrumentation, CUDA event or synchronization is added to the encoder.

Labels 0/1 use identical original CPU selection; 2/3 use identical GPU
selection. Both captures use 3840 × 2160, eight requested CPU threads, four
warm and eight measured rounds of the four even Williams orders from S175.
The second process reverses order. Each process checks one reference and
48 labeled encodes: 98 whole encodes, 64 measured, 96 captured windows
including warmups. Four preflight processes exercise normal/ASAN harnesses
with markers disabled/enabled, checking 20 more encodes. Total: 118.

Every call matches reference bytes, full summary, coefficient width and
native owner size, and passes S175's frontend invocation, launch, readback
and zero-host-cost-capacity checks. Two ASAN preflight processes contain
eight instrumented DLL calls and two normal reference calls. No new CUDA
kernel is built here; S175's sanitizer evidence is retained, not recounted
as new S176 sanitizer testing. No CTest/install/batch/concurrent-encoder,
independent-decoder or cross-platform qualification is claimed.

Nsight records CUDA/NVTX only; CPU sampling and context-switch capture are
disabled. The capture runs to process completion, including terminal stdout.
The analyzer extracts embedded target stdout from SQLite and checks all
48 windows against exact round/position/mode and host timing records.
Warmups and the unlabelled reference are excluded from paired statistics.

All 33,648 GPU activities across the 96 windows correlate with successful
CUDA runtime calls inside their own complete-call window. Activities finish
within that window and belong to one measured device/context/stream/process
per trace, excluding the separate reference context. The two encoder families
have exact ordered standard launch identity: kernel names, grids, blocks,
registers and shared/local memory. Their only kernel-sequence difference is
one `S175SelectTilesKernel` inserted after 65 standard launches; the remaining
230 launches are identical. This is machine-work identity, not an assertion
of equal cache state, scheduling or execution duration.

The complete copy sequence is identical except seven D2H cost transfers
become one grid/error transfer. The unchanged prefix contains 25 copies and
suffix 20. Removed sizes are 518,400, 453,120, 453,600, 396,480, 96,960,
97,200 and 72,720 bytes, totaling 2,088,480. The replacement is 137,760 bytes.
Original/candidate copy counts are 52/46, and memset sizes/counts match.

## Paired attribution

For each round, average candidate labels 2/3 and original labels 0/1, take
their difference, then the median across eight rounds. Negative is faster.
Trace columns below are separate processes, not confidence intervals.

| Metric, candidate minus original (ms) | First trace | Reversed trace |
| --- | ---: | ---: |
| Complete call | +2.706 | −7.327 |
| Complete strategy frontend | −10.328 | −10.740 |
| Quantization outside strategy frontend | +12.360 | +8.209 |
| All kernel execution, union | +12.034 | +6.762 |
| Standard kernels before new selection | −0.348 | −1.336 |
| Existing fused candidate-cost kernels, subset of prefix | −0.456 | −1.149 |
| New selection kernel | +0.819 | +0.822 |
| Standard kernels after new selection | +11.732 | +10.463 |
| All copy execution, union | −0.039 | +0.082 |
| All device activity, union | +11.396 | +6.411 |
| Gaps within complete device span | −11.189 | −11.153 |
| Complete device span | −0.141 | −6.193 |
| Time after last device activity | +0.870 | −2.513 |

The prefix ends at `FinalizeCostKernel`; the suffix begins at
`AdjustQuantFieldKernel`. The original split uses the corresponding position
in its exactly matched launch sequence. Prefix/suffix aggregates are
calculated per encode before pairing. Quantization-minus-frontend likewise
uses each encode's own enclosing and nested timers. Separate table medians
must not be summed or subtracted to reconstruct another row.

The downstream increase is distributed. Paired kernel-name aggregates
include increases of 1.632/2.317 ms for one paired-Malta specialization,
1.095/1.490 ms for the other, and 1.148/0.888 ms for mirrored opsin. Erosion,
convolution, EPF and other unchanged kernels also vary. The new selection
kernel itself is much too small to account for the entire observed offset.

Duplicate controls remain material. Original 1-versus-0 kernel-union deltas
are −5.721/−2.821 ms, while candidate 3-versus-2 deltas are +5.095/−6.895 ms.
Whole-call duplicate deltas are −1.497/−3.372 ms for originals and
+4.325/+3.097 ms for candidates. Whole trace times also differ from the
unprofiled campaign. This is attribution under instrumentation, not a new
universal speedup/slowdown claim, sample-normalization method or promotion
gate fitted after seeing the results.

## Limits and next hypothesis

All 192 captured warm/measured NVML endpoints report a 40,000 mW enforced
limit. This does not establish constant clocks or thermals. No build,
sanitizer, source archive, heavy hash sweep or other task GPU workload runs
during these captures; light analysis-code editing and ordinary shared-machine
activity remain limitations. No power, clock, thermal, priority, affinity,
firewall, driver or security setting changes. No admin/firewall blocker occurs.

S113 and S136 observed related unchanged-kernel offsets after shortening host
work. S113's nominal 10 ms NVML polling received changed SM clock readings
only about every 500 ms, too coarse to attribute individual policy calls.
Repeating that polling would not turn it into per-kernel evidence. Neither
those older results nor the current traces establish a hardware/driver cause.

A distinct next diagnostic is a controlled host-cadence intervention at the
post-selection boundary, with fixed precommitted pauses and zero-pause
controls, while preserving the same candidate outputs and standard GPU work.
Measure actual added wall time and subsequent policy execution; do not
subtract an intended pause from complete-call time or silently normalize
results with coarse clocks. Such a test can examine whether timing the
downstream submission differently changes its observed cost. A result would
still not identify a particular clock/thermal mechanism. Diagnostic pauses
are not a proposed production optimization, compatibility layer or setting
change. The encoder is not established to be maxed out.

## Evidence

New root: `build-cuda-ninja/profiles/s176-artifacts` on C. The U drive is nearly
full, so it is not used for new captures. Preparation requires 1.8 GB free on
C and each capture requires 900 MB free. About 1.65 GB remains after both
captures at the observed check. No material file is deleted. Durable native
reports, SQLite exports, embedded stdout, job logs, source/protocol pins and
all analyses remain; transient Nsight/compiler `temp` scratch is excluded
from immutable inventory. No predecessor root or pinned helper is modified.

Seven journaled jobs all succeed: one normal/ASAN build job, four preflights,
two captures. There is no failed or restarted encode/capture. The initial
unpinned verifier compared integer-keyed Python dictionaries directly with
JSON's string keys; normalizing the recomputed value through JSON fixes that
checker comparison. No executable, sample or analysis value changes.

| Phase (UTC, September 10) | Start | Finish |
| --- | --- | --- |
| Normal/ASAN harness build | 06:00:59.180 | 06:01:16.216 |
| Four preflights | 06:02:12.413 | 06:02:23.112 |
| First trace, including export | 06:02:53.530 | 06:03:16.058 |
| Reversed trace, including export | 06:03:16.065 | 06:03:38.092 |

The frozen S175 inventory's 869 artifacts is reverified. Verification checks
all input/source/build hashes, job outcomes/nonoverlap, schedules, counters,
raw-log/SQLite reconstruction, correlated activities, exact launch/copy
identities and the final inventory:

```powershell
python build-cuda-ninja/profiles/s176_verify.py --frozen
```

No production code is changed or new runtime qualification claimed. The
optimization goal remains active.
