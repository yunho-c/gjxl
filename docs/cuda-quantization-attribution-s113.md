# Quantization time outside the strategy merge (S113)

Date: 2026-09-08. Starting revision: `8b729ac`, branch `feat/cuda`.
Windows 11, RTX 3060 Laptop (sm86), CUDA 11.8, MSVC 14.37, Release.

**Outcome:** the S112 outside-merge offset is concentrated in subsequent AQ
policy execution, not metadata or serial grid export. The trace attributes
policy-time variation to GPU kernel execution, not launch starvation or host
wake-up delays. The hardware/driver cause is not established. No scheduler,
clock adjustment, compatibility layer or other runtime change is promoted.

## Question and method

[S112](cuda-tile-scheduling-s112.md) found that parallel CPU candidate-cost
merging saves scoped time but does not deliver a reliable complete-encode
gain. Its odd-4K quantization-minus-merge residual was slower across all
eight automatic-thread paired medians. This study asks where that residual
occurs. It is an attribution study, not another proposed runtime selector.
The retained runtime remains serial at the merge; S111 composition/AQ fusion
stays enabled and compact coefficient storage stays OFF in these probes.

The diagnostic build overlays four translation units on the qualified S112
libraries: GPU quantization providers, GPU AC-strategy search, resident CUDA
AQ, and the archived S112 clean strategy candidate. Serial and scheduled
labels share one executable and the same GPU kernel library. Only diagnostic
timers and the already-qualified candidate selection differ. The tracked
runtime is not rebuilt or modified.

Twenty-four fixed, thread-local wall scopes measure the pipeline. No GPU
event, device synchronization, per-cell check or per-tile timer is added.
The important nesting is:

```text
quantization
  frontend: evaluator preparation + initial quantization
  search: preparation + GPU wait + cost readback/scatter + CPU merge
    CPU merge: grid export
  adaptive quantization
    metadata reconfiguration: build + upload
    resident policy setup
    resident policy
      preparation + submission + wait + readback
        frame: allocation/descriptors + copy + validation/assembly
```

The validator checks one invocation of every scope and these nesting bounds
for every instrumented encode. The original enclosing workflow timer remains
available. Differences and residuals are computed **per encode before
within-round pairing and medians**. Separate medians are not additive.

Labels 0/1 are duplicate serial routes and 2/3 duplicate bounded scheduled
routes. The plain timer probe uses four warm and 12 measured balanced rounds
per job, plus one reference: 65 checked encodes, 48 measured. Every label
occupies each position three times in the measured rounds. HD and odd 4K run
twice in reversed input order. A one-thread 4K control and a timers-disabled
4K control follow. These controls were repeated separately as described below.
Input I/O, backend construction and exact comparisons are outside the outer
timing boundary; returned results live until that boundary ends.

## Attribution results

The table shows ranges across four candidate-versus-serial paired medians
per repetition. Negative means faster; ranges are not confidence intervals.

| Input / repetition | CPU merge (ms) | Quantization minus merge (ms) | Before merge (ms) | AQ after merge (ms) | Policy submit + wait (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Odd 4K / 0 | −4.45 to −3.57 | +1.51 to +11.11 | −1.12 to +1.36 | +4.75 to +9.13 | +2.13 to +11.34 |
| Odd 4K / 1 | −3.84 to −3.63 | +3.16 to +14.47 | −2.34 to +1.15 | +3.69 to +8.80 | +4.36 to +6.59 |
| HD / 0 | −0.70 to −0.47 | −0.41 to +0.60 | −0.10 to +0.24 | −0.33 to +0.84 | −0.61 to +0.23 |
| HD / 1 | −0.85 to −0.77 | −0.53 to +0.17 | −0.63 to +0.14 | −0.51 to +0.31 | −0.06 to +0.31 |

The eight automatic-thread 4K residual medians are positive again. In these
plain-timer runs, reconfiguration changes only −0.17 to +0.32 ms and frame
assembly/readback changes −0.70 to +0.87 ms. These do not account for the
larger consistent AQ policy shift. Before-merge changes are mixed, providing
another indication of ordinary run-to-run variation. Complete-call deltas
remain mixed; local merge savings are not a general encoder speedup.

The clean timers-disabled control still has four slower quantization-minus-
merge paired medians, +9.36 to +14.43 ms. Thus the new detailed wall scopes are
not required for the offset to appear. The clean one-thread control has a
mixed residual (−3.66 to +10.97 ms) despite no tile workers, reinforcing the
need for duplicate controls and rejecting a simple scheduler-only causal
interpretation. All control values, including the excluded initial controls,
remain in the machine-readable analyses.

## GPU trace

The separate trace build adds nested NVTX ranges and captures four balanced
measured rounds after four warm rounds. Each trace therefore checks 33 encodes
but captures 16. Nsight Systems 2023.2.3 records CUDA and NVTX only; CPU sampling
and context-switch tracing are disabled. Trace timings are diagnostic, not
unprofiled throughput results.

Kernel attribution uses runtime API correlation IDs inside each policy
submission range, not just timestamp containment. The analyzer checks that
all selected activity belongs to one device/context/stream, ends before the
policy wait returns, and has an identical ordered kernel signature across
all 32 captured encodes (names, grids, blocks, registers and dynamic shared
memory). GPU busy time is the union of kernel/memset intervals; gaps are
subtracted from the GPU execution span, and time outside that span is measured
against the enclosing submission-through-wait host interval.

Each captured policy executes **229 kernels and two memsets**, with no copy
activity attributed to that submission. Across the 16 captured serial-label
encodes, the kernel sum has a median of 121.04 ms (range 108.88–139.69 ms),
the GPU activity gaps a median of 1.05 ms, and time outside the GPU span a
median of 0.074 ms. These scoped trace measurements are not full-encode time.

| Trace repetition | Paired kernel-time delta (ms) | Paired GPU-gap delta (ms) | Paired time-outside-GPU-span delta (ms) |
| --- | ---: | ---: | ---: |
| 0 | +4.14 to +8.00 | −0.001 to +0.029 | −0.011 to +0.000 |
| 1 | −2.99 to +7.66 | −0.005 to +0.038 | −0.007 to +0.027 |

The policy execution-time changes track kernel-time changes closely. Six of
eight candidate paired kernel medians are slower, but two are faster; the
second repetition's duplicate candidate comparison differs by −14.88 ms.
That counterexample prevents claiming a deterministic GPU penalty caused by
tile scheduling. It does not erase the repeated outside-merge offset in the
larger untraced probes. The data locates the variation but does not establish
whether its cause is power management, thermal state, cache behavior,
preemption or another hardware/driver effect.

The largest short-name kernel aggregates, taking the median per-encode sum
over serial labels in both traces, are useful priorities for reducing device
work (not independent components whose separate medians should be summed):

| Kernel aggregate | Median GPU time per policy (ms) |
| --- | ---: |
| `PairedMaltaScaleResponseKernel` | 23.49 |
| `ConvolutionTiledKernel` | 17.34 |
| `ErosionL2FinalKernel` | 9.60 |
| `ConvolutionFrequencyKernel` | 9.49 |
| `FusedMirroredOpsinKernel` | 8.23 |
| `ConvolutionLowMediumRowsKernel` | 8.15 |
| `EpfTiledKernel` | 8.15 |

In contrast, plain-timer serial-label 4K grid-export medians are 0.47–0.57 ms.
Its extra owner/reconstruction remains removable work, but it is not the
source of the observed multi-millisecond AQ shift.

## Read-only telemetry

The separate telemetry build samples NVML at a nominal 10 ms interval. It
reads SM/memory clocks, temperature, power and clock-throttling reasons, with
status codes retained. Samples are buffered before being printed after the
worker joins. No clock, power, priority, affinity, driver or security setting
is changed. Policy submission/wait wall timestamps delimit eligible samples;
a sample must begin and end inside the interval. This is observational
sampling, not per-kernel clock measurement or a causal intervention.

Two automatic-thread jobs and one one-thread control each have 48 measured
encodes, adding 195 checked encodes including warmups/references. The new
telemetry yields 3,927 samples in those jobs, of which 1,177 lie fully inside
measured policy intervals. Median sampling intervals are 15.56–15.57 ms,
despite the requested 10 ms wait; median query duration is 0.219–0.225 ms,
with a maximum of 16.04 ms. No high-resolution timer or scheduling setting
was changed to force the requested interval.

Every eligible sample reports memory clock 5,500 MHz and reason mask `0x24`.
The installed CUDA 11.8 NVML header and [NVIDIA's clock-event reason
definitions](https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksEventReasons.html)
decode that mask as software power-cap (`0x4`) plus software thermal-limit
(`0x20`) reporting. These are reported operating-state flags, not proof that
the scheduler caused either condition or that a particular temperature
sensor exceeded its threshold. GPU temperature readings span 59–65 C over
eligible samples; memory temperature was not measured.

Reported SM clocks vary widely (262–1,560 MHz across the three jobs), but
changed values arrive about every **500 ms**, much more slowly than sampling
or an individual policy interval. In all three jobs each measured policy's
eligible samples retain one reported SM-clock value. Pearson correlations
between that value and policy submission/wait time are −0.053 and +0.054 in
the two automatic-thread repetitions, and −0.254 in the one-thread control.
These observations do not support correcting policy times by the sampled
clock or treating it as a per-kernel exposure measurement. The data establishes
an operating-state caveat, not a causal power/thermal diagnosis.

The telemetry probes again have eight positive automatic-thread policy
submit/wait paired medians (+2.55 to +12.65 ms), while the one-thread control
is mixed (−4.09 to +0.46 ms). Whole-call results remain mixed. Telemetry and
trace probes are kept separate from the clean untraced timing population.

## Qualification and provenance

Every accepted encode matches frozen codestream bytes; its complete summary
matches the freshly computed serial reference in that process. Scoped ASAN
instruments all four overlaid translation units and the harness, not all
linked libraries or the NVML driver. No new runtime code or test is committed,
so this study does not claim a new full CTest qualification of changed code.

Totals are **861 checked encodes**, including **40 under scoped ASAN**.
There are 288 eligible clean untraced measured encodes, 144 measured encodes
with telemetry, and 32 captured encodes in accepted GPU traces. The 96
measured encodes in the excluded initial controls remain part of the exact
check count, not the eligible performance population. There are 29 accepted
GPU jobs, two rejected trace-capture attempts, and seven successful builds.

Two initial traces produced usable reports but failed the strict success
marker check: this Nsight Windows capture retained target stdout in the
report, not the outer console log, and the original marker was printed after
capture stopped. Enabling console forwarding alone did not repair that
boundary. The corrected probe prints its checked-encode success marker before
capture stop; the runner extracts and verifies embedded process stdout from
the SQLite export. Both rejected attempts and their sources are retained and
are not counted as accepted traces.

The initial timer-probe build also had two fields named `merge_ms`. This was
caught before any encode job, renamed to distinguish entry and wrapper scopes,
and rebuilt with exclusive artifact names. The unused initial binary/source
are retained. A parser now rejects duplicate field names.

A repeated hash-analysis command overlapped the final two jobs of the first
plain-timer campaign. Those one-thread and timers-disabled controls are
excluded from timing conclusions, with logs and results retained. Their
replacement jobs run in a separate clean window after the analysis process
was confirmed absent. The earlier HD/automatic-thread 4K pairs had already
finished before that command started. No task build, sanitizer or GPU trace
overlaps an accepted plain-timer campaign window. Light editing and ordinary
shared-machine activity remain limitations.

Campaign windows on 2026-09-08 (UTC):

| Campaign | Start | Finish |
| --- | --- | --- |
| Plain timer, HD/auto-4K pairs eligible | 07:14:46.029 | 07:15:37.871 |
| Initial controls, timing excluded | 07:15:38.004 | 07:16:23.733 |
| Replacement controls | 07:19:26.537 | 07:20:10.727 |
| Accepted traces, including export | 07:24:03.179 | 07:24:42.358 |
| Telemetry campaign | 07:31:25.569 | 07:32:30.486 |

Evidence root: `U:/gjxl-cuda-diagnostics/s113`; diagnostic sources, runners and
analyzers: `build-cuda-ninja/profiles/s113_*`. Outputs use exclusive names.
The S112 archive and retained runtime files are not overwritten. No firewall,
admin prompt or permission blocker was observed.

`s113_validate.py` checks job logs and executable/input/oracle hashes,
recomputes all timing/trace/telemetry analyses, validates scope nesting,
invocation counts, schedule balance and trace signatures, and verifies job
nonoverlap and build exclusion. The manual hash-analysis exclusion is explicit
in `timing_exclusions.json`; its exact process timestamps were not retained,
so the validator does not pretend to reconstruct that overlap automatically.
It also verifies the source against `8b729ac`, the S112 baseline binaries and
all 40 retained runtime files. `s113_freeze.py` hashes the evidence and report.

```powershell
python build-cuda-ninja/profiles/s113_validate.py --frozen
```

## Next action

Keep CPU tile scheduling disabled. Prioritize reducing work in Malta and
tiled-convolution GPU kernels, using actual encoder-stage scopes plus clean
duplicate-control whole-call measurements. Do not pursue launch-gap tuning
as an explanation for this offset: the measured gaps are much too small.
Do not silently normalize results with coarse clock telemetry or change
power/cooling settings to obtain a favorable comparison. Serial export is
a secondary, smaller opportunity. The backend is not established to be
maxed out.
