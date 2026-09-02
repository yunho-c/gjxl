# GJXL/libjxl codestream performance comparison plan

Status: Phase 1 and Phase 2 complete as of 2026-09-02. The no-source-change
pilot established the broad attribution, and the isolated opt-in libjxl
profiler now supplies direct matched-quality serializer wall-stage timing.
Profiled timings remain diagnostic; the unprofiled Phase 1 measurements remain
the end-to-end performance gate.

## Purpose

This plan defines a reproducible comparison of the CPU work and elapsed time
spent by GJXL and libjxl after coefficient decisions, with particular emphasis
on tokenization, entropy-model construction, token emission, and final
codestream assembly.

The work is deliberately split into two phases:

1. Run a no-libjxl-source-change pilot using unprofiled benchmarks and sampled
   CPU attribution. This is sufficient to rank broad opportunities and test
   whether the apparent serializer gap survives matched inputs.
2. Add opt-in, benchmark-only timing to libjxl for exact stage latency and a
   durable cross-encoder diagnostic ratio. This phase was subsequently
   requested and completed without changing ordinary libjxl builds.

The pilot must not be described as exact stage wall-clock timing. Conversely,
the instrumented phase must not replace unprofiled end-to-end measurements as
the final performance gate.

## Questions to answer

The comparison should answer these questions separately:

- How much aggregate CPU work does each encoder spend preparing and writing
  the codestream?
- How much critical-path wall time does each encoder spend in those stages?
- Which substage explains any gap: tokenization, histogram/model search,
  entropy-token emission, or framing and copying?
- Does the result hold for real photographic 1080p and 4K inputs, or only for
  duplicate-heavy padded stress inputs?
- How much output-size improvement is purchased by additional entropy search?
- Does the ranking change between a serial diagnostic configuration and the
  normal bounded-parallel configuration?

These are implementation comparisons, not a claim that both encoders make the
same coefficient, context, or entropy-model decisions.

## Comparison definitions

### Primary comparison: real encoder stages

The primary result compares each encoder's actual policy on the same decoded
linear-RGB input and a matched decoded-quality target. It preserves each
encoder's native token and model representation. This is the result relevant
to complete encoding latency.

Two quality views must be retained:

- **Nominal-distance view:** run both encoders with the same requested distance.
- **Matched-quality view:** calibrate libjxl's distance so that its independently
  decoded Butteraugli score matches GJXL within the declared tolerance.

The nominal-distance view is easy to reproduce but is not a matched-quality
performance claim. The matched-quality view is the primary cross-encoder
result. Encoded bytes and bits per pixel must be reported in both views.

GJXL has no direct equivalent of libjxl's effort scale. Libjxl effort 7 is the
primary policy comparison because it is the normal cjxl default; efforts 5 and
9 may be reported as explicitly labeled speed and compression sensitivity
points. They must not be described as matched GJXL effort levels.

### Secondary comparison: entropy-engine microbenchmark

Feeding an identical token corpus to both entropy engines would isolate coder
implementation cost, but the token types, context maps, clustering policy,
HybridUint choices, and model serialization differ. Building adapters would
therefore be a separate project and could remove precisely the policy work that
dominates real encoding.

Do not block the primary comparison on a same-token microbenchmark. Consider it
only if the real-stage results cannot distinguish model-search cost from token
emission cost.

## Current measurement capabilities

### GJXL

The public workflow benchmark emits raw workflow schema 10. Its relevant
wall-clock phases are:

- `codestream_dc_tokenization`
- `codestream_ac_tokenization`
- `codestream_entropy_optimization`
- `codestream_section_writing`
- `codestream_assembly`

The schema also records aggregate worker-time substages for coefficient-token
template construction, context materialization, prefix and ANS model search,
selected-token writing, candidate measurement, and final assembly. Values
ending in `_work` may exceed their enclosing wall phase because overlapping
worker durations are summed. They must never be added to complete-encode
latency. See [metal-encoding-performance.md](metal-encoding-performance.md) for
the complete schema contract.

GJXL's serializer normally chooses up to eight workers internally. The
benchmark-only `--serializer-workers` option now records and enforces a
process-wide limit: zero preserves the normal policy and one makes serializer
section work serial. The production comparison should use the normal worker
policy and configure libjxl with the same worker limit.

### libjxl without source changes

The pinned libjxl source already exposes stable profiler symbols for the major
operations:

- `TokenizeAllCoefficients` prepares VarDCT coefficient tokens.
- `BuildAndEncodeHistograms` and
  `EntropyEncodingData::BuildAndStoreEntropyCodes` perform histogram building,
  clustering, HybridUint selection, and entropy-model construction.
- `EncodeGroupTokenizedCoefficients` and `WriteTokens` emit selected token
  streams.
- `EncodeGroups` writes global, DC, AC, and Modular group data.
- `WriteFrameHeader`, `WriteTocPermutation`, `WriteTocSizes`,
  `AppendByteAligned`, and `AppendData` perform final framing and assembly.

`AuxOut` and `JxlEncoderStats` report bits, histogram counts, and transform
statistics, but not elapsed stage time. `cjxl` measures only the complete
`EncodeImageJXL` call. Consequently, a sampling profiler can recover aggregate
CPU attribution without changing libjxl, but cannot directly recover exact
parallel stage wall time.

Libjxl's `EstimateEntropy` under AC-strategy search is rate-distortion candidate
scoring. It is frontend heuristic work and must not be counted as final
codestream entropy optimization merely because of its name.

## Canonical stage mapping

The comparison report should use neutral stage names and retain the native
substage names underneath them.

| Neutral stage | GJXL evidence | libjxl evidence | Included work |
| --- | --- | --- | --- |
| Coefficient/token preparation | DC/AC tokenization wall phases and coefficient-token/context `_work` counters | `TokenizeAllCoefficients`; relevant Modular `ComputeTokens` work | Convert decided coefficients and side data into ordered tokens and contexts |
| Entropy model construction | `codestream_entropy_optimization` and its prefix/ANS `_work` counters | `BuildAndEncodeHistograms`, `BuildAndStoreEntropyCodes`, clustering, HybridUint selection, ANS/Huffman construction | Build and select the model used by the written stream |
| Token and model emission | `codestream_section_writing`, model/header work, and token-write work | `EncodeHistograms`, `WriteTokens`, `EncodeGroupTokenizedCoefficients`, Modular `EncodeStream` | Serialize the selected model and entropy-coded payload |
| Framing and assembly | `codestream_assembly` and its candidate/header/TOC/output-copy counters | frame header, permutation, TOC, byte-aligned append, and output append functions | Produce the final in-memory codestream bytes |
| Complete serializer | `codestream_encoding` | Union of the four neutral stages | All post-decision token, entropy, and framing work |

The libjxl union must be defined as mutually exclusive stack classifications in
the sampling parser. Inclusive function totals overlap and must not be summed.

## Workload and input contract

### Corpus

The retained comparison corpus should contain:

- at least six real photographic 1920x1080 inputs;
- at least six real photographic 3840x2160 inputs;
- a mix of smooth, textured, noisy, high-frequency, and saturated content;
- Kodak PhotoCD images as a continuity set for the existing cjxl Samply
  profile, but not as the high-resolution headline result; and
- GJXL's padded 1080p and padded 4K workloads as explicitly labeled stress
  tests, never folded into the photographic aggregate.

Record the source, license, original dimensions, conversion command, and
SHA-256 of every retained input. Do not construct the headline 4K corpus solely
by tiling small images; periodic duplicates can materially change entropy-model
search behavior.

### Common decoded input

Convert each source once, before any timed work, to a canonical planar or PFM
linear-sRGB representation. Both encoders must receive pixels derived from the
same canonical file. When cjxl reads a metadata-free PFM, pass the explicit
linear-sRGB color hint `RGB_D65_SRG_Rel_Lin`.

The benchmark manifest must record:

- canonical input SHA-256 and dimensions;
- source and canonical color descriptions;
- row stride and channel order used by each harness;
- requested distance and libjxl effort;
- actual decoded Butteraugli score and encoded bytes; and
- whether dimensions required edge extension inside either encoder.

Input decoding, color conversion into the canonical representation, process
startup, backend construction, and output-file I/O are outside the primary
timed boundary. The final codestream must still be produced in memory.

## Build and host contract

Use isolated build directories and record exact revisions. The first retained
comparison should use the repository's pinned libjxl submodule revision
`e8ff09762481785938d8e4e01333ed3917571161`. A comparison against a newer
libjxl revision may be added, but it must be pinned separately and must not
silently replace the reference result.

Both builds must use:

- the same Apple Clang toolchain and architecture;
- Release optimization with assertions disabled;
- symbols or line tables that do not change optimization;
- the same CPU dispatch policy recorded in the manifest;
- one-shot, in-memory output rather than streaming output; and
- no profiler in the untraced timing runs.

Record the host model, CPU/GPU, logical core count, memory, macOS and Xcode
versions, power source, thermal state when available, and relevant background
load. If the host is visibly busy, preserve diagnostic captures but do not use
their absolute timings for retained claims.

## Phase 1: no-libjxl-source-change pilot

### Progress

| Milestone | Status | Evidence or remaining work |
| --- | --- | --- |
| Pin and prepare the comparison corpus | Complete | 6 photographic scenes at both 1080p and 4K, 24 Kodak continuity images, and padded 1080p/4K stress inputs; 38 canonical PFM inputs total |
| Build the isolated libjxl harness and comparison driver | Complete | Pinned libjxl revision, binary hashes, balanced independent processes, serial and production policies, and never-overwritten artifacts are recorded |
| Run the nominal-distance serial and production pilot | Complete, diagnostic | All 38 inputs ran at requested distance 1.0 and libjxl effort 7; host load makes absolute production timing unsuitable for a publication-grade claim |
| Validate outputs and retained artifacts | Complete | 760 subprocesses succeeded; every output decoded and received an independent Butteraugli score; retained codestream and decoded-file hashes were rechecked |
| Run the matched-quality view | Complete, diagnostic | All 38 inputs were calibrated and rerun under both thread policies; high host load and a battery-to-AC transition limit absolute-time claims |
| Capture neutral-stage sampled CPU attribution | Complete | 20 profiles cover both encoders, both policies, and every reported workload group; all sidecars are retained and weighted symbol resolution is 99.61-99.83% |
| Decide whether direct libjxl stage timing is necessary | Deferred | Sampled serializer CPU differs by at least 19.7x in every representative group, so profiler uncertainty cannot change the optimization priority; instrumentation remains necessary for an exact wall-time multiplier |

The completed run used GJXL's factored DCT implementation and the fully
resident Metal public-workflow path. "Fully resident" describes the Metal AQ
and coefficient-decision path: entropy coding and final codestream assembly
still execute on the CPU. The matching libjxl policy used effort 7. The serial
configuration limited both serializers to one worker; the production
configuration used the normal eight-worker limit.

### Implemented tooling

Phase 1 is implemented without modifying the pinned libjxl source. The pieces
are deliberately separated:

- `tools/libjxl_comparison.py fetch-corpus` downloads each unique HTTPS source
  or invokes a declared GJXL built-in-source generator, retains the result
  atomically, and requires its declared SHA-256;
- `tools/libjxl_comparison.py prepare-corpus` converts or validates sources,
  requires provenance and license fields, applies only explicit manifest crop
  and resize operations, and writes a never-overwritten canonical-PFM corpus
  with source and canonical SHA-256 hashes;
- `tools/libjxl_comparison.py build-libjxl` verifies the pinned revision,
  builds shared `libjxl`, `djxl`, and `butteraugli_main`, then builds the
  GJXL-owned public-C-API harness under an isolated build root;
- `tools/libjxl_comparison.py calibrate-quality` takes the independently
  decoded GJXL scores from a retained nominal summary as immutable per-input
  targets, searches bounded libjxl distances outside the timed boundary, and
  emits a corpus-, effort-, revision-, and tolerance-bound calibration map;
- `tools/libjxl_comparison.py run` alternates independent process pairs,
  retains serial and production policies, writes native raw schemas and
  binary/host/command manifests, independently decodes both outputs, records
  Butteraugli scores, and emits normalized summaries. With `--quality-map`, it
  applies the calibrated distance independently to every libjxl input and
  fails closed if the newly decoded encoder scores exceed the declared match
  tolerance. Repeatable `--input` filters preserve corpus order for diagnostic
  profiling subsets while keeping the full corpus manifest and calibration
  binding;
- `tools/samply_neutral_stages.py` consumes presymbolicated Samply captures,
  applies an ordered mutually exclusive neutral-stage mapping, rejects weighted
  leaf-symbol resolution below 95%, and labels every result as sampled thread
  CPU rather than wall time;
- `tools/libjxl_comparison.py bundle-phase1` verifies the corpus/revision
  bindings, exit counts, quality gates, stage coverage, symbol resolution, and
  all recorded output/profile hashes before emitting one indexed Phase 1
  artifact; and
- `gjxl_encoding_benchmark --serializer-workers` supplies the internal
  diagnostic worker cap, while `--codestream-output` writes the final measured
  in-memory result after the timed samples for independent validation. Raw
  workflow schema 10 records the worker policy.

The libjxl harness parses the canonical PFM once, constructs the parallel
runner outside the timer, and encloses encoder configuration, image submission,
and complete in-memory output generation. It performs one untimed validation
encode before requested warmups, matching the GJXL public-workflow benchmark.
Optional codestream file output happens after all measured samples.

A source manifest has this minimal form:

```json
{
  "schema_version": 1,
  "inputs": [
    {
      "name": "photo-1080p-01",
      "path": "sources/photo-1080p-01.png",
      "source": "original URL or corpus identifier",
      "license": "license and attribution",
      "source_color": "embedded sRGB",
      "category": "photographic-1080p"
    }
  ]
}
```

An optional `sha256` pins the fetched source. Optional `crop` and `resize`
objects make preprocessing reviewable rather than hiding it in a wrapper. Crop
coordinates apply after EXIF orientation. Resizing occurs after conversion to
linear RGB and uses a fixed Lanczos filter. A PFM identity input cannot request
either transform.

The complete Phase 1 corpus recipe is pinned at
`benchmarks/libjxl_comparison/corpora/phase1-pilot.json`. It contains 38
canonical inputs in four separately reported groups:

- six public-domain photographs from the canonical imazen-26 test split at
  revision `187fbf338ce08e8e6654db7f04ddae58d5263da2`, each represented by a
  3840x2160 center crop and a 1920x1080 linear-light Lanczos reduction;
- all 24 Kodak PhotoCD PCD0992 images at their native 768x512 or 512x768
  dimensions for continuity with the existing cjxl Samply analysis; and
- exact exports of GJXL's built-in `padded_1080p` and `padded_4k` synthetic
  sources at 1919x1079 and 3839x2159 for edge-extension stress testing.

The padded sources use a manifest generator rather than a reimplementation of
the synthetic formula. `gjxl_encoding_benchmark --source-output` writes the
same `ImageStorage` produced for the timed workload as an atomic linear-sRGB
PFM; `fetch-corpus` then checks the pinned hash. This keeps both encoders on
identical bytes and makes future changes to the built-in source fail closed
until the corpus recipe is reviewed. Source URLs or generator identity,
licenses, and SHA-256 hashes are recorded per entry. Kodak and padded results
must not be folded into the high-resolution photographic aggregate.

Run the pilot from this worktree as follows:

```sh
python3 tools/libjxl_comparison.py fetch-corpus \
  --source-manifest \
    benchmarks/libjxl_comparison/corpora/phase1-pilot.json \
  --output build/libjxl-comparison/corpus-sources \
  --gjxl-benchmark build/release/gjxl_encoding_benchmark

python3 tools/libjxl_comparison.py prepare-corpus \
  --source-manifest \
    benchmarks/libjxl_comparison/corpora/phase1-pilot.json \
  --source-root build/libjxl-comparison/corpus-sources \
  --output build/libjxl-comparison/corpus-phase1-pilot-pinned

python3 tools/libjxl_comparison.py build-libjxl

python3 tools/libjxl_comparison.py run \
  --corpus-manifest \
    build/libjxl-comparison/corpus-phase1-pilot-pinned/manifest.json \
  --configuration both
```

After the nominal run, calibrate and run the matched-quality view with:

```sh
python3 tools/libjxl_comparison.py calibrate-quality \
  --corpus-manifest \
    build/libjxl-comparison/corpus-phase1-pilot-pinned/manifest.json \
  --nominal-summary \
    logs/libjxl-comparison/<nominal-run>/summary.json

python3 tools/libjxl_comparison.py run \
  --corpus-manifest \
    build/libjxl-comparison/corpus-phase1-pilot-pinned/manifest.json \
  --configuration both \
  --quality-map \
    logs/libjxl-calibration/<calibration-run>/calibration.json
```

The normal match tolerance is an absolute Butteraugli difference of 0.015.
This accommodates discrete quantization-policy transitions while keeping the
maximum score mismatch close to one percent at the usual Phase 1 quality
targets. Calibration starts at distance 1.0, verifies the target against the
relevant 0.05 or 2.0 bound, and uses at most 12 evaluations per input. The 0.05
lower bound is libjxl's minimum lossy VarDCT distance at the pinned revision.
If the expected endpoint does not bracket the score, the driver samples a
bounded coarse grid across the interval and refines any adjacent crossing it
finds. This is required because libjxl's decoded score can be non-monotonic
across encoder policy transitions even though requested distance is ordinarily
directional.

If that bound cannot reach GJXL's score, or a discrete policy transition skips
over the absolute window, calibration may retain the closest candidate only
when its relative score difference is at most 2.5%. Such inputs are explicitly
labeled `boundary-limited-relative-tolerance` or
`quantized-relative-tolerance`; they must be disclosed separately from normal
absolute-tolerance matches in the final report. The comparison run rechecks
both decoded scores and the applicable tolerance rather than trusting the map.

Every search evaluation, command, raw encoder record, score, and hash remains
in the calibration manifest; only the selected candidate's codestream and
decoded PFM are retained to bound disk usage.

Add `--capture-samply` only to a diagnostic profiling run. The unprofiled
process-pair summary remains the performance source of record. The repository
does not vendor the corpus itself. `fetch-corpus` reconstructs the hash-pinned
source set, and retained timing still requires a quiet host.

The retained sampled-attribution pass uses one representative from each
separately reported workload group under both thread policies:

```sh
python3 tools/libjxl_comparison.py run \
  --corpus-manifest \
    build/libjxl-comparison/corpus-phase1-pilot-pinned/manifest.json \
  --configuration both \
  --quality-map \
    logs/libjxl-calibration/<calibration-run>/calibration.json \
  --input imazen26-1409-rainbow-4k \
  --input imazen26-1207-bedroom-noise-1080p \
  --input kodak-kodim17 \
  --input padded-stress-1080p \
  --input padded-stress-4k \
  --pairs 1 --warmups 0 --samples 1 \
  --capture-samply --profile-samples 20
```

This diagnostic run is not a replacement timing source. Its manifest hashes
each capture, presymbolication sidecar, analyzer, and Samply executable; the
summary retains the neutral-stage rows and weighted symbol-resolution result.

### Nominal-distance pilot: 2026-09-01

The first complete unprofiled pilot is retained locally at
`logs/libjxl-comparison/20260901T212206.521314Z-94b1716cab17`. It ran all 38
canonical inputs from clean GJXL commit `94b1716cab17`, with the pinned libjxl
revision, requested distance 1.0, libjxl effort 7, three alternating process
pairs, two warmups, and five measured samples per process. GJXL used the
factored, fully resident Metal public-workflow path; its entropy coding and
codestream assembly remained CPU work. Both the one-worker serial diagnostic
configuration and the normal eight-worker production configuration were run
without Samply.

The table reports the median of the three per-process medians, averaged per
image within multi-image categories. `GJXL relative` is complete GJXL time
divided by complete libjxl time. `Codestream share` is GJXL's profiled
codestream phase as a percentage of its own complete encode.

| Configuration | Input group | GJXL ms/image | libjxl ms/image | GJXL relative | Codestream share |
| --- | --- | ---: | ---: | ---: | ---: |
| Serial | 4K photographs | 2427.7 | 2016.2 | 1.20x slower | 81.2% |
| Serial | 1080p photographs | 1136.8 | 505.7 | 2.25x slower | 88.7% |
| Serial | Kodak continuity | 315.6 | 102.3 | 3.09x slower | 87.5% |
| Serial | Padded 1080p stress | 1444.9 | 464.9 | 3.11x slower | 90.5% |
| Serial | Padded 4K stress | 4341.1 | 1840.0 | 2.36x slower | 89.1% |
| Production | 4K photographs | 935.0 | 467.1 | 2.00x slower | 47.2% |
| Production | 1080p photographs | 347.2 | 123.8 | 2.80x slower | 63.4% |
| Production | Kodak continuity | 91.9 | 30.4 | 3.02x slower | 67.8% |
| Production | Padded 1080p stress | 420.3 | 125.9 | 3.34x slower | 68.5% |
| Production | Padded 4K stress | 1309.1 | 457.7 | 2.86x slower | 63.2% |

The same requested distance did not produce matched decoded quality. Lower
Butteraugli is better in this table; `Size delta` is aggregate GJXL bytes
relative to aggregate libjxl bytes.

| Input group | GJXL Butteraugli | libjxl Butteraugli | Size delta |
| --- | ---: | ---: | ---: |
| 4K photographs | 1.3224 | 1.3901 | -2.26% |
| 1080p photographs | 2.2126 | 2.2555 | +1.96% |
| Kodak continuity | 1.2695 | 1.3480 | -2.29% |
| Padded 1080p stress | 1.2756 | 1.4552 | +25.73% |
| Padded 4K stress | 1.4207 | 1.4726 | +28.71% |

This nominal-distance result is therefore policy sensitivity, not the primary
matched-quality encoder comparison. GJXL produced the lower mean Butteraugli
score in every category, so the matched-quality calibration reported below is
required before attributing the complete speed or size gap to implementation
alone.

The run completed 456 benchmark processes plus 304 independent decode and
Butteraugli commands: all 760 subprocesses exited successfully. It retained
152 encoder/configuration validations. All 456 raw schemas parsed, encoded
size was deterministic across process pairs, serial and production streams
were byte-identical for each encoder/input pair, and an independent rehash of
all 304 retained codestream and decoded files matched the recorded SHA-256.

Host load is a material caveat. The manifest records load averages of
26.44/29.09/35.30 at start and 17.34/20.03/21.63 at completion, with AC power
and no thermal or performance warning. Serial photographic process medians
were stable to at most 1.33%, but production 1080p ranges reached 34.48% for
GJXL and 40.44% for libjxl. Every production input still ranked libjxl faster,
and the gap was larger than the observed range, so the pilot is sufficient for
directional prioritization. Its production absolute times are not suitable for
a publication-grade retained claim without a quieter rerun.

GJXL's production entropy-optimization phase consumed 38.0% of complete time
on 4K photographs, 54.8% on 1080p photographs, 60.7% on Kodak, and 53.1-59.2%
on the padded stress inputs. Entropy optimization alone took 0.76x libjxl's
complete 4K-photo time and 1.52-1.98x libjxl's complete time in the other
categories. The complete GJXL codestream phase took 0.94x libjxl's complete
4K-photo time and 1.78-2.29x libjxl's complete time elsewhere. These compare a
direct GJXL phase wall time with complete libjxl latency; they do not measure
libjxl's entropy stage. They are enough to prioritize GJXL entropy work before
adding direct libjxl stage instrumentation.

### Matched-quality pilot: 2026-09-01

The retained calibration is
`logs/libjxl-calibration/20260901T223312.215074Z-80e7c9e9d90d`; the resulting
unprofiled comparison is
`logs/libjxl-comparison/20260901T223737.519919Z-80e7c9e9d90d`. GJXL remained at
requested distance 1.0. Libjxl effort remained 7, with per-input distances from
0.5375 to 1.2611328125. The timing protocol remained three alternating process
pairs, two warmups, and five measured samples under both serial and production
policies.

Thirty-four inputs matched within the normal 0.015 absolute Butteraugli window.
Four inputs encountered discrete, non-monotonic policy transitions and used the
declared relative fallback:

| Input | Absolute difference | Relative difference | libjxl distance |
| --- | ---: | ---: | ---: |
| `imazen26-1029-planter-1080p` | 0.06394 | 1.68% | 0.5375 |
| `imazen26-1049-river-city-1080p` | 0.01967 | 1.01% | 1.2611328125 |
| `imazen26-1207-bedroom-noise-4k` | 0.01551 | 1.07% | 0.88125 |
| `kodak-kodim17` | 0.02485 | 1.95% | 0.940625 |

All four are labeled `quantized-relative-tolerance` in the calibration. None
reached the 2.5% ceiling. They remain in the aggregate because the exception is
bounded and explicit, but any publication-grade result should retain the label
or report a sensitivity view without them.

The matched complete-encode timings are:

| Configuration | Input group | GJXL ms/image | libjxl ms/image | GJXL relative | Codestream share |
| --- | --- | ---: | ---: | ---: | ---: |
| Serial | 4K photographs | 2439.0 | 2041.7 | 1.19x slower | 80.1% |
| Serial | 1080p photographs | 1139.2 | 500.4 | 2.28x slower | 88.0% |
| Serial | Kodak continuity | 316.8 | 103.2 | 3.07x slower | 87.4% |
| Serial | Padded 1080p stress | 1461.4 | 474.5 | 3.08x slower | 91.3% |
| Serial | Padded 4K stress | 4378.8 | 1882.5 | 2.33x slower | 89.5% |
| Production | 4K photographs | 877.8 | 448.4 | 1.96x slower | 46.8% |
| Production | 1080p photographs | 327.6 | 107.0 | 3.06x slower | 62.5% |
| Production | Kodak continuity | 85.1 | 27.8 | 3.07x slower | 67.5% |
| Production | Padded 1080p stress | 380.5 | 105.2 | 3.62x slower | 68.4% |
| Production | Padded 4K stress | 1244.0 | 425.3 | 2.93x slower | 61.8% |

The decoded-quality and size view is identical across thread policies:

| Input group | GJXL Butteraugli | libjxl Butteraugli | Size delta |
| --- | ---: | ---: | ---: |
| 4K photographs | 1.3224 | 1.3234 | -8.88% |
| 1080p photographs | 2.2126 | 2.2207 | -8.33% |
| Kodak continuity | 1.2695 | 1.2680 | -7.47% |
| Padded 1080p stress | 1.2756 | 1.2866 | +0.30% |
| Padded 4K stress | 1.4207 | 1.4137 | +19.25% |

The run completed 456 benchmark processes and 304 independent decode and
Butteraugli commands. All 760 subprocesses succeeded. All 456 raw schemas, 152
aggregate records, and 76 quality-match records parsed; every match passed its
declared gate, encoded sizes were stable across process pairs, and
serial/production codestreams were byte-identical for every encoder/input pair.
An independent rehash of all 304 retained codestream and decoded files matched
the manifest.

Absolute timing remains diagnostic. Load averages were 11.35/14.23/19.69 at
start and 14.56/17.94/17.82 at completion; power changed from battery to AC
during the run. No thermal or performance warning was recorded. The maximum
range across three process medians was 29.50% for serial GJXL and 35.84% for
production libjxl, although the category-level direction remained consistent
with the nominal pilot. The source revision was `80e7c9e9d90d`; the worktree's
only dirty path was the separately owned `.gitignore` addition for `logs/`, and
all benchmark binaries were hash-recorded.

Matched quality strengthens the optimization priority rather than reversing
it. GJXL remains about 2.0x slower on production 4K photographs and 3.1x slower
on production 1080p photographs and Kodak. Its codestream phase still consumes
46.8% of 4K and 62.5-67.5% of 1080p/Kodak complete time. Direct libjxl stage
instrumentation was therefore deferred pending the sampled CPU attribution
below.

### Sampled CPU attribution: 2026-09-01

The source-of-record diagnostic capture is retained at
`logs/libjxl-comparison/20260901T232209.162595Z-cefe85f9bc20`. It profiles one
representative from each separately reported group under both serial and
production policies, for 20 captures total. Each capture contains one untimed
validation encode plus 20 repeated encodes sampled at 1 kHz. The table reports
sampled thread CPU per encode; it is not stage wall time.

`Serializer CPU` is the mutually exclusive union of coefficient/token
preparation, entropy-model construction, token/model emission, and framing and
assembly. `Model CPU` is entropy-model construction as a percentage of all
sampled encoder CPU, including frontend and runtime work. The ratio compares
serializer CPU, not complete-encoder latency.

| Configuration | Input group | GJXL serializer CPU ms/encode | libjxl serializer CPU ms/encode | GJXL/libjxl serializer CPU | GJXL model CPU | libjxl model CPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Serial | 4K photograph | 1871.82 | 94.98 | 19.71x | 77.01% | 1.09% |
| Serial | 1080p photograph | 1074.79 | 28.05 | 38.32x | 88.35% | 1.75% |
| Serial | Kodak continuity | 286.18 | 6.94 | 41.24x | 91.61% | 3.00% |
| Serial | Padded 1080p stress | 1323.39 | 49.30 | 26.84x | 88.73% | 2.69% |
| Serial | Padded 4K stress | 3903.45 | 197.61 | 19.75x | 85.61% | 2.42% |
| Production | 4K photograph | 1926.38 | 77.50 | 24.86x | 78.48% | 0.94% |
| Production | 1080p photograph | 1111.60 | 22.96 | 48.41x | 89.30% | 1.55% |
| Production | Kodak continuity | 282.29 | 3.47 | 81.31x | 93.67% | 1.93% |
| Production | Padded 1080p stress | 1373.98 | 45.39 | 30.27x | 89.62% | 2.40% |
| Production | Padded 4K stress | 4100.87 | 186.28 | 22.01x | 86.05% | 2.09% |

The corresponding GJXL/libjxl serializer costs in sampled CPU nanoseconds per
source pixel range from 225.67/11.45 for the serial 4K photograph to
727.78/17.65 for serial Kodak. All normalized per-encode and per-source-pixel
values are retained per stage in `summary.json`. A common native token count is
not available across the two encoders, so the report does not invent a
nanoseconds-per-token normalization.

Every positive sample is assigned exactly once by the ordered classifier.
Weighted leaf-symbol resolution ranges from 99.6107% to 99.8342%, comfortably
above the 95% gate, and no capture assigns CPU to
`unresolved_or_missing_stack`. `other_encoder_and_runtime` remains large for
libjxl (89.5-97.0%) because its CPU frontend—particularly AC-strategy search,
DCT, quantization, and filtering—is intentionally outside the final serializer
union. `EstimateEntropy` under AC-strategy search is therefore not mislabeled
as final entropy-model construction.

The artifact contains 20 compressed captures, 20 presymbolication sidecars,
20 JSON analyses, and 20 Markdown analyses. All 120 subprocesses and 10 fresh
quality matches succeeded. An independent audit rehashed the 20 captures, 20
sidecars, 20 analysis files, 20 codestreams, and 20 decoded files with zero
mismatches, and recomputed every stage sum and normalization with zero failures.
The run hashes GJXL, libjxl, `djxl`, Butteraugli, Samply 0.13.1, and the analyzer.

Host load rose from 29.83/26.43/21.80 to 26.49/26.96/24.07. That does not make
the profiler suitable for latency claims, but sampled thread-CPU attribution
is used only to identify the responsible operation class. The consistent
19.7-81.3x sampled serializer-CPU gap and GJXL's 77.0-93.7% sampled-CPU share
in entropy-model construction were already decisive for prioritization. Phase
2 subsequently confirmed the ranking with direct wall-stage timing: GJXL's
entropy-model construction remains the first serializer optimization target.

### 1. Add a comparison driver — complete

Add a GJXL-owned driver that:

- builds or locates the exact GJXL and pinned cjxl binaries;
- verifies revision and binary hashes before running;
- prepares or validates the canonical input corpus;
- invokes one encoder per independent process;
- alternates order as GJXL/libjxl then libjxl/GJXL;
- separates warmups from measured samples;
- disables file output while retaining complete bitstream generation; and
- writes raw samples and a manifest under a never-overwritten
  `logs/libjxl-comparison/<timestamp>-<revision>/` directory.

Do not parse human-readable timing summaries when a raw numeric output can be
added to the comparison harness. Keep the driver independent of the Metal
profile driver because the comparison does not require an Instruments GPU
trace.

### 2. Establish unprofiled end-to-end timings — complete

For every workload and policy point, run at least three independent balanced
process pairs. Each process should perform at least two warmups and enough
measured samples to produce a stable per-process median. Increase process and
sample counts when the median pair-to-pair range is wider than the effect under
discussion.

Retain two thread configurations:

- **Production:** normal GJXL serializer concurrency and the same libjxl worker
  limit.
- **Serial diagnostic:** one serializer worker in both encoders. This requires a
  benchmark-only GJXL worker override; libjxl already accepts
  `--num_threads=0` for no worker threads.

Report distributions, not a single timing. The final table should show the
median of per-process medians and the complete range of those medians.

The 2026-09-01 nominal and matched-quality pilots completed this step. Their raw
distributions are retained, but the busy-host caveats above limit the absolute
production timings to diagnostic use.

### 3. Capture sampled CPU attribution — complete

Capture both encoders directly with Samply at 1 kHz. Keep the symbol sidecar
beside every capture. Use enough repeated encodes to accumulate a useful sample
count inside the serializer rather than relying on one short encode.

Extend or reuse the retained cjxl Samply parser to emit mutually exclusive
neutral stages. Apply the same attribution rule and symbol-resolution checks to
both encoders. Report:

- total sampled thread CPU;
- CPU milliseconds and percentage for each neutral stage;
- resolved and unresolved CPU percentage;
- flat and inclusive hotspot tables as supporting evidence; and
- capture count, sample count, and profiler configuration.

Use the serial diagnostic capture as the cleanest CPU-work comparison. Use the
parallel capture to understand distribution across workers, not to infer stage
wall latency from wait frames. Sampling results are attribution evidence and
must remain separate from the unprofiled latency table.

### 4. Normalize and report — complete for Phase 1

For every stage, report the native time plus:

- milliseconds per megapixel;
- CPU nanoseconds per source pixel;
- nanoseconds per token when a reliable native token count exists;
- milliseconds per encoded megabyte as secondary context;
- encoded bits per pixel; and
- percentage of complete encoder time or sampled encoder CPU.

No single normalization is sufficient. Time per output byte can reward a
larger output, while time per pixel can hide different token densities. Keep
raw time, size, quality, and normalization values adjacent.

The nominal and matched-quality reports currently include absolute
complete-encode time, relative time, GJXL codestream share, encoded-size delta,
and decoded Butteraugli. The machine-readable raw and aggregate artifacts also
retain milliseconds per megapixel, CPU nanoseconds per pixel, bits per pixel,
and milliseconds per encoded megabyte. The sampled-attribution report adds CPU
milliseconds per encode, CPU nanoseconds per source pixel, and percentage of
sampled encoder CPU for every neutral stage. Nanoseconds per token are omitted
because the encoders do not expose a reliable common native token count.

### 5. Pilot exit criteria — satisfied

The no-touch pilot is complete when:

- both encoders consume identical canonical pixels for every retained input;
- every output decodes independently and has a recorded Butteraugli score;
- all raw samples, manifests, binary hashes, and symbol sidecars are retained;
- the operation classifier is mutually exclusive and resolves at least 95% of
  sampled serializer CPU;
- photographic and padded stress results are reported separately;
- serial and production thread policies are explicitly labeled; and
- the report states that libjxl stage figures are sampled CPU attribution, not
  direct stage wall time.

The pilot may guide optimization priority. It is not sufficient for an exact
claim such as "GJXL's entropy stage is N times slower in wall-clock latency."

The 2026-09-01 nominal, matched-quality, and Samply runs collectively satisfy
all of these criteria. The full timing artifacts retain identical canonical
inputs, independent output decodes and scores, raw schemas, binary hashes, and
separate photographic/stress and serial/production results. The diagnostic
profile artifact retains every capture and symbol sidecar, uses a mutually
exclusive classifier, exceeds 99.61% weighted symbol resolution, and labels
all stage values as sampled thread CPU rather than wall time. The high-load
caveat still prevents publication-grade absolute latency claims; it does not
leave a Phase 1 exit criterion open.

## Phase 2: libjxl stage instrumentation — complete

Phase 2 adds exact serializer wall-stage boundaries and aggregate worker-time
substage counters to an isolated libjxl build. It was run after Phase 1 at
matched decoded quality on representative photographic, Kodak, and padded
stress inputs under both serial and production policies.

### Architecture

The profiler is implemented in the isolated
`/Users/yunhocho/GitHub/libjxl-gjxl-stage-profile` worktree on branch
`perf/gjxl-stage-profile`. Instrumented revision
`b1596acff4bd93775e8cf7b42b1fdffa87c555bd` is based on pinned libjxl revision
`e8ff09762481785938d8e4e01333ed3917571161`. The two focused commits are:

- `0f42cd5f Add opt-in encoder stage profiling`; and
- `b1596acf Cover streaming frame assembly in stage profiles`.

`JPEGXL_ENABLE_STAGE_PROFILER` defaults to `OFF`. When enabled, it exposes a
private benchmark hook rather than adding timing fields to libjxl's public C
API or `JxlEncoderStats`; symbol inspection confirmed that this hook is absent
from an ordinary build.

Add an internal, opt-in profile sink with these properties:

- a null sink is the default and performs no profile clock reads or profile
  allocations;
- stable stage IDs are independent of function names;
- top-level timers surround synchronization/barrier boundaries and report wall
  time;
- worker-local substage accumulators are merged after group work, avoiding a
  shared atomic update for every token or group;
- aggregate worker time is labeled separately and may exceed enclosing wall
  time;
- the profile session is active only inside a measured encoder call, so
  frontend heuristic `BuildAndEncodeHistograms` calls are not misattributed to
  the final serializer;
- invocation counts, token counts, histogram counts, model bits, token bits,
  and output bytes use checked, saturating integer accumulation with an
  explicit overflow flag; and
- the benchmark harness, not the encoder library, serializes the result as
  JSON.

Worker-local accumulators are merged only after `RunOnPool`; the hot token and
group paths do not update shared atomics. Both buffered and streaming frame
output modes are covered. The comparison tool builds the profiler, records the
pinned base and instrumentation revisions, hashes the source patch and
binaries, and rejects dirty or unrelated source ancestry.

### Required stage boundaries

The implementation records wall phases for:

1. coefficient and Modular side-data tokenization;
2. entropy histogram/model construction;
3. model/header and token-stream emission;
4. frame header, TOC, section concatenation, and final output copy; and
5. their complete serializer union.

It records aggregate work counters for:

- histogram population;
- histogram clustering;
- HybridUint configuration selection;
- entropy-model construction;
- histogram/context-map serialization;
- token encoding and bit writing;
- Modular/DC side-data encoding; and
- output assembly and copying.

The complete serializer phase is the exact sum of the four non-overlapping wall
phases. This invariant is enforced per measured sample. Leaf timers contribute
only aggregate work and invocation accounting; overlapping worker durations
are never presented as wall-stage latency.

### Raw schema

Unprofiled harness output retains schema 1. `--stage-profile` requires an
instrumented build and emits schema 2:

```json
{
  "schema_version": 2,
  "timing_semantics": {
    "elapsed_nanoseconds": "complete-encode-wall-time",
    "phase_nanoseconds": "wall-clock-barrier-time",
    "work_nanoseconds": "aggregate-worker-time"
  },
  "stage_profile_enabled": true,
  "encoder": "libjxl",
  "revision": "<full commit>",
  "thread_count": 8,
  "requested_distance": 1.0,
  "effort": 7,
  "samples": [
    {
      "sample_index": 0,
      "elapsed_nanoseconds": 0,
      "encoded_bytes": 0,
      "phase_nanoseconds": {},
      "work_nanoseconds": {},
      "invocation_counts": {
        "phase": {},
        "work": {}
      },
      "counts": {
        "token_count": 0,
        "histogram_count": 0,
        "model_bits": 0,
        "token_bits": 0,
        "output_bytes": 0
      }
    }
  ]
}
```

The comparison tool translates both native schemas into a neutral stage table
without changing the retained raw records. For GJXL, DC plus AC tokenization
maps to coefficient tokenization, entropy optimization maps to entropy-model
construction, section writing maps to model/token emission, assembly maps to
framing, and `codestream_encoding` maps to the complete serializer. GJXL's
complete serializer timer is not asserted to equal the sum of its translated
components because the native schema has different timing boundaries.

### Instrumentation validation

The profile passed the planned validation:

- Ordinary, instrumented-with-sink-off, and instrumented-with-sink-on libjxl
  builds emitted byte-identical codestreams for all 38 corpus inputs.
- All retained outputs decoded successfully, and the harness enforced stable
  counts and exact serializer unions across measured samples.
- Focused libjxl ANS, TOC, frame-encoding, output-mode, parallel-runner,
  roundtrip, and decoder tests passed.
- Balanced alternating perturbation runs used three independent process pairs,
  one warmup, and three measured samples on a 1080p photograph, a 4K
  photograph, Kodak, and padded 4K stress input. Profiled/unprofiled median
  ratios ranged from 0.9881 to 0.9997 (-1.19% to -0.03%), so no material
  profiling overhead was detected.

The complete validation artifact is retained at
`logs/libjxl-stage-profile-validation/20260902T003251.087745Z-b1596acff4bd`.

### Direct matched-quality result

The direct timing run uses three balanced independent process pairs, two
warmups, and five measured samples per process. It covers representative 1080p
and 4K photographs, Kodak continuity, and padded 1080p/4K stress inputs under
serial and production policies. Every one of the ten matched-quality records
passed and every retained output decoded.

| Input | Policy | GJXL serializer (ms) | libjxl serializer (ms) | Ratio | GJXL entropy construction (ms) | libjxl entropy construction (ms) | Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1080p photograph | Serial | 1076.711 | 25.982 | 41.44x | 1015.696 | 6.938 | 146.39x |
| 1080p photograph | Production | 218.328 | 14.371 | 15.19x | 191.069 | 6.735 | 28.37x |
| 4K photograph | Serial | 1883.485 | 85.668 | 21.99x | 1702.309 | 18.465 | 92.19x |
| 4K photograph | Production | 376.616 | 43.611 | 8.64x | 299.983 | 18.398 | 16.30x |
| Kodak continuity | Serial | 282.974 | 6.398 | 44.23x | 271.517 | 2.417 | 112.34x |
| Kodak continuity | Production | 57.608 | 4.595 | 12.54x | 52.011 | 2.336 | 22.27x |
| Padded 1080p stress | Serial | 1324.505 | 47.127 | 28.11x | 1246.674 | 10.822 | 115.20x |
| Padded 1080p stress | Production | 259.875 | 21.073 | 12.33x | 224.978 | 10.778 | 20.87x |
| Padded 4K stress | Serial | 3877.282 | 189.448 | 20.47x | 3593.083 | 40.937 | 87.77x |
| Padded 4K stress | Production | 793.495 | 77.397 | 10.25x | 656.368 | 40.715 | 16.12x |

For production settings, GJXL's complete serializer is 8.64-15.19x slower and
its entropy-model-construction wall phase is 16.12-28.37x slower than libjxl's
on these representative matched-quality inputs. In serial mode, those ranges
are 20.47-44.23x and 87.77-146.39x respectively. The direct result confirms
Phase 1's prioritization: GJXL entropy-model construction is the dominant
serializer optimization target. These profiled ratios are diagnostic; the
unprofiled Phase 1 runs remain the end-to-end performance evidence.

## Result artifact

The completed Phase 1 bundle is retained at
`logs/libjxl-phase1/20260901`. `tools/libjxl_comparison.py bundle-phase1`
validated and rehashed the four never-overwritten source artifacts, then wrote
`README.md`, `phase1-index.json`, and `neutral-comparison.json` beside relative
links to the nominal, calibration, matched-quality, and Samply directories. The
index records 784 successful rehashes, all 76 matched-quality records, all 20
profiles, and the 99.6107% minimum weighted symbol resolution.

The completed Phase 2 bundle is retained at
`logs/libjxl-phase2/20260902`. It contains `README.md`,
`phase2-index.json`, and `direct-stage-comparison.json` beside relative links
to the never-overwritten timing and validation artifacts. The index verifies
38 identity inputs, four perturbation groups, 60 timing-process rows, 20
aggregate rows, 50 neutral direct-stage rows, and ten quality matches. The
matched-quality source timing artifact is
`logs/libjxl-comparison/20260902T003438.333660Z-5776d06438c0`.

Recreate the bundle with:

```sh
python3 tools/libjxl_comparison.py bundle-phase1 \
  --nominal-result \
    logs/libjxl-comparison/20260901T212206.521314Z-94b1716cab17 \
  --calibration-result \
    logs/libjxl-calibration/20260901T223312.215074Z-80e7c9e9d90d \
  --matched-result \
    logs/libjxl-comparison/20260901T223737.519919Z-80e7c9e9d90d \
  --profile-result \
    logs/libjxl-comparison/20260901T232209.162595Z-cefe85f9bc20 \
  --output logs/libjxl-phase1/20260901
```

Recreate the Phase 2 bundle with:

```sh
python3 tools/libjxl_comparison.py bundle-phase2 \
  --timing-result \
    logs/libjxl-comparison/20260902T003438.333660Z-5776d06438c0 \
  --validation-result \
    logs/libjxl-stage-profile-validation/20260902T003251.087745Z-b1596acff4bd \
  --output logs/libjxl-phase2/20260902
```

The final artifact directory should contain:

- `README.md` with scope, conclusions, caveats, and reproduction commands;
- `manifest.json` with revisions, hashes, build commands, host state, and input
  provenance;
- raw unprofiled samples for both encoders;
- native GJXL schema-10 samples;
- parsed libjxl and GJXL Samply operation tables;
- native libjxl stage-profile samples;
- decoded-quality and encoded-size results;
- all Samply captures and symbol sidecars; and
- a machine-readable neutral comparison table.

The headline table should separate 1080p photographs, 4K photographs, Kodak
continuity images, and padded stress workloads. It should show both absolute
time and normalized cost, together with output size and actual decoded quality.

## Decision rules

- If the no-touch CPU attribution shows a large, consistent GJXL gap across the
  photographic corpus, it is sufficient to prioritize the responsible GJXL
  substage, but not to publish an exact wall-time multiplier.
- If the result is small, content-dependent, or dominated by unresolved stacks,
  add libjxl instrumentation before drawing a latency conclusion.
- If padded inputs differ sharply from photographs, treat them as a robustness
  target for model-search complexity rather than a representative average.
- If matching quality reverses the same-distance ranking, use the matched-
  quality result for encoder conclusions and retain the same-distance result
  only as parameter sensitivity.
- If an optimization changes encoded bytes, report the speed/size tradeoff and
  rerun independent decode and quality validation; do not describe it as an
  exact-output optimization.
- Re-profile after every retained optimization because the original sampled
  hotspot may no longer dominate.

## Implementation order

1. **Complete.** Pin the six-image imazen-26 headline set as paired 1080p/4K
   views, all 24 native Kodak continuity images, and exact generated padded
   1080p/4K stress sources in one reproducible corpus recipe.
2. **Complete.** Add the benchmark-only GJXL serializer worker override.
3. **Complete.** Add the comparison driver and raw unprofiled result schema.
4. **Complete.** Extend the Samply classifier with the neutral stage mapping
   and tests.
5. **Complete with host-load caveat.** Run the no-libjxl-source-change nominal-
   distance serial and production pilot across all 38 inputs. Retain the
   directional result, but rerun production timing on a quieter host before a
   publication-grade absolute-time claim.
6. **Complete.** Calibrate and run the matched-quality view across all 38
   inputs under both policies.
7. **Complete.** The matched-quality and sampled-attribution views consistently
   prioritized GJXL entropy-model construction; Phase 2 was then requested to
   replace attribution uncertainty with direct stage boundaries.
8. **Complete.** Implement and validate the isolated opt-in libjxl profiler.
   All 38 inputs remain byte-identical across ordinary and profiled builds,
   focused tests pass, and no material profiling perturbation was detected.
9. **Complete with host-load caveat.** Retain the full unprofiled runs, sampled
   profiles, symbol sidecars, neutral table, and indexed Phase 1 bundle. Rerun
   absolute production timing on a quieter host before publication.
10. **Complete.** Run representative matched-quality Phase 2 timing under both
    policies and retain the indexed Phase 2 bundle. Direct wall timing confirms
    entropy-model construction as GJXL's first serializer optimization target.
