# Resident frame handoff: milestone 1

This refactor removes the owned-frame requirement from the internal codestream
consumer interface. It does not yet eliminate Metal's final frame assembly or
change the GPU coefficient layout. AQ iterations, AC selection, quantization,
coefficient-order policy, and entropy policy are unchanged.

Base revision: `4ea12ab` (the Metal fusion merge). Development branch:
`refactor/resident-frame-handoff`; worktree: `../gjxl-resident-frame-handoff`.

## Interface and ownership

`codec/vardct_frame_view_internal.h` defines `VarDctFrameViewData` and
`VarDctFrameView`. The view copies small value metadata and borrows strategy,
quantizer, color-correlation, raw-quantization, sharpness, DC, and AC storage.
It neither allocates storage nor materializes coefficients. Block planes may
have independent row strides. AC storage retains the existing fixed-capacity,
group/channel-major layout, row-major transform anchors, and zero edge tails.
The decoder-equivalent floating DC cache is still required and validated; its
removal or lazy construction is not part of this milestone.

`BorrowFrame(const VarDctEncoderFrame&)` adapts existing owned frames without
copying their planes. It checks owned allocation sizes before exposing core
plane views; value validation is shared with independently backed views.
Borrowing temporary owners is explicitly rejected at compile time. The public
owned-frame APIs and class storage layout remain intact.

`codestream_internal::EncodeVarDctCodestreamFromView` is the synchronous internal
entry point, with optional profiling. Serializer helpers, coefficient-order
derivation, block-context derivation, and DC/AC tokenizers consume the common
view. Public owned-frame entry points adapt to these implementations. The
shared validator preserves structural and initial-profile gates, and failed
serialization leaves both codestream output and optional profile unchanged.
Trusted `ForEncoder` helpers continue to require previously validated inputs.

The zero-tail invariant uses an exact unsigned integer-OR reduction. This avoids
a scalar early-exit scan introduced by the first view-based validator; the
final compiler output vectorizes the check. Negative coefficients and both ends
of every partial group's channel tail are covered by rejection tests. No
floating-point encoding calculation or policy is changed by this check.

The borrowing contract is explicit:

- Every backing allocation and referenced metadata object stays alive,
  address-stable, and immutable until the last consumer returns.
- Device-backed producers must complete before the view is consumed. A future
  output lease must prevent writes, reuse, and purging until serialization and
  every serialization worker have finished.
- The view does not perform synchronization, retain a device lease, or extend
  lifetimes when copied. As with core plane views, callers supply sufficient
  live backing for every addressed row; validation cannot detect dangling or
  undersized raw-pointer allocations.
- Completed output lifetime is distinct from transient AQ scratch lifetime.
  This interface does not require keeping an entire evaluator arena alive.

## Correctness qualification

Both builds use AppleClang 17, Ninja, Release, tests and benchmarks enabled, and
optional libjxl-reference fixtures disabled. `build/baseline` was built from the
unmodified parent before source edits and was not rebuilt afterward.

- Parent full CTest: **61/62**. Final candidate full CTest: **62/63**, including
  the new `vardct_frame_view` test and installed C/C++ consumer checks.
- Both fail only the inherited CPU `quantization_pipeline` golden:
  `0.24919039011001587` actual versus `0.24914586544036865` expected.
  This was reproduced on the fresh parent, not inferred from older records.
- The new test covers tiny/padded images, mixed transforms, multiple AC/DC
  groups, quant-field context splitting, independent strided backing,
  borrowed-pointer identity, all three entropy modes, full/sampled coefficient
  orders, 1/8 CPU-thread budgets, and concurrent readers. External backing is
  serialized after the original owned frame is destroyed.
- Invalid/missing metadata, malformed planes and group storage, nonzero tails,
  incorrect DC reconstruction, unsupported profiles, cross-group strategies,
  and output preservation are checked. AddressSanitizer and UBSan passed this
  test using a separate RelWithDebInfo build.
- All **38 canonical images** (Kodak, natural 1080p/4K photographs, and padded
  stress images) produced byte-identical parent/candidate codestreams at Metal
  fully-resident, distance 1.2, effort 7.
- **18 additional policy cases** matched bytes: efforts 1–10, high density,
  maximum compression, final-score diagnostics, exact coefficients, both
  throughput modes, target bytes, and CPU maximum error. Target bytes uses the
  small repository sample; the other additional cases use Kodak17.
- Both builds passed all **22 full pinned conformance fixtures**, including
  stored codestream hashes, decoded-reconstruction comparisons, ordinary
  entropy variants, workflow cases, and corruption checks. Decoder:
  `djxl` 0.13.0, revision `e8ff09762481785938d8e4e01333ed3917571161`.
- Independently decoding parent and candidate Kodak17, planter 4K, and padded
  stress 4K also produced identical PFM bytes. The Metal libraries themselves
  are byte-identical; no shaders changed.

Local qualification artifacts are under `build/qualification/`: parent and
candidate CTest/conformance logs, `parity.json`, retained codestreams and decoded
PFMs, and the qualification runner. These are local build artifacts, not durable
repository fixtures. `parity.json` records input and executable hashes and exact
commands; the permanent regression coverage is in `tests/vardct_frame_view_test.cpp`.

## Performance qualification

Apple M4 Pro, 48 GiB, macOS 15.6. For each workload, seven alternating
parent/candidate process pairs used two warmups and five retained samples per
process. Both builds used fully-resident Metal, tuned AC kernels, SIMD CPU
implementation, distance 1.2, effort 7, automatic CPU threads, and Metal-only
validation. Timings are profiled public-workflow wall time, excluding input
generation/loading and process startup; they are not GPU duration or aggregate
worker time. Corpus byte/decoder comparisons ran separately from timing.

| Workload | Parent total | Candidate total | Change | Median paired change |
| --- | ---: | ---: | ---: | ---: |
| Padded 1080p, 1919×1079 | 83.799 ms | 82.789 ms | -1.20% | -0.72% |
| Padded 4K, 3839×2159 | 282.683 ms | 282.309 ms | -0.13% | +0.06% |
| Kodak17, 512×768 | 24.619 ms | 24.795 ms | +0.71% | +0.01% |
| Planter 4K, 3840×2160 | 290.945 ms | 291.636 ms | +0.24% | +0.13% |

The total columns are medians of the seven process medians. Paired change is
the median of the seven candidate/parent ratios, not a ratio of aggregate
worker counters. These results support **approximately neutral end-to-end
performance**, not a meaningful overall speedup. Codestream wall time changed
by -0.68%, -1.54%, -0.79%, and +0.52%, respectively, using process medians.

The initial view validator made the large-image validation phase roughly
0.2 ms slower by losing the parent's unrolled zero-tail scan. The final exact
integer reduction removes that regression: validation changed from
0.356→0.240 ms (padded 1080p), 0.721→0.622 ms (padded 4K), and
0.733→0.659 ms (planter 4K). Kodak17 validation is tiny and changed from
0.023→0.027 ms. These are substage observations, not additional overall gains.

All timing samples retained identical codestream sizes within and across
builds for each workload. Recorded process-boundary snapshots showed no other
GJXL encode/benchmark/test jobs. Raw samples, executable hashes, exact commands,
and process-pair summaries are retained in `build/qualification/performance/`,
`performance.json`, and `performance-summary.json`.

Example command (substitute `baseline` for the parent):

```sh
build/release/gjxl_encoding_benchmark --workload padded_4k \
  --scope metal-public-workflow --implementation simd \
  --gpu-aq fully-resident --validation metal-only \
  --distance 1.2 --effort 7 --cpu-threads auto \
  --warmups 2 --samples 5 --raw-samples /tmp/frame-handoff-samples.json
```

For natural inputs, replace `--workload padded_4k` with `--input INPUT.pfm`.

## Next milestone

Connect completed Metal output through a bounded owner/lease, initially at the
existing synchronous completion boundary. Then measure final coefficient writes
into the consumer layout, removing the compulsory repack/owned copy across the
whole handoff. The current milestone intentionally retains that assembly;
merely routing it through a view is not a zero-copy GPU handoff or a claimed
copy-elimination speedup. Shared resource admission and new scheduling remain
separate later work.
