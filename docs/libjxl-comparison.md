# GJXL/libjxl codestream performance comparison plan

Status: proposed, 2026-09-01.

## Purpose

This plan defines a reproducible comparison of the CPU work and elapsed time
spent by GJXL and libjxl after coefficient decisions, with particular emphasis
on tokenization, entropy-model construction, token emission, and final
codestream assembly.

The work is deliberately split into two phases:

1. Run a no-libjxl-source-change pilot using unprofiled benchmarks and sampled
   CPU attribution. This is sufficient to rank broad opportunities and test
   whether the apparent serializer gap survives matched inputs.
2. Add opt-in, benchmark-only timing to libjxl only if exact stage latency or a
   durable cross-encoder ratio is required.

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

### Implemented tooling

Phase 1 is implemented without modifying the pinned libjxl source. The pieces
are deliberately separated:

- `tools/libjxl_comparison.py prepare-corpus` converts or validates sources,
  requires provenance and license fields, and writes a never-overwritten
  canonical-PFM corpus with source and canonical SHA-256 hashes;
- `tools/libjxl_comparison.py build-libjxl` verifies the pinned revision,
  builds shared `libjxl`, `djxl`, and `butteraugli_main`, then builds the
  GJXL-owned public-C-API harness under an isolated build root;
- `tools/libjxl_comparison.py run` alternates independent process pairs,
  retains serial and production policies, writes native raw schemas and
  binary/host/command manifests, independently decodes both outputs, records
  Butteraugli scores, and emits normalized summaries;
- `tools/samply_neutral_stages.py` consumes presymbolicated Samply captures,
  applies an ordered mutually exclusive neutral-stage mapping, rejects weighted
  leaf-symbol resolution below 95%, and labels every result as sampled thread
  CPU rather than wall time; and
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

Run the pilot from this worktree as follows:

```sh
python3 tools/libjxl_comparison.py prepare-corpus \
  --source-manifest path/to/sources.json \
  --output build/libjxl-comparison/corpus

python3 tools/libjxl_comparison.py build-libjxl

python3 tools/libjxl_comparison.py run \
  --corpus-manifest build/libjxl-comparison/corpus/manifest.json \
  --configuration both
```

Add `--capture-samply` only to a diagnostic profiling run. The unprofiled
process-pair summary remains the performance source of record. The repository
does not vendor the photographic corpus itself; selecting and licensing the
retained 1080p/4K sources and running them on a quiet host remain execution
steps, not implementation steps.

### 1. Add a comparison driver

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

### 2. Establish unprofiled end-to-end timings

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

### 3. Capture sampled CPU attribution

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

### 4. Normalize and report

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

### 5. Pilot exit criteria

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

## Phase 2: optional libjxl stage instrumentation

Proceed with this phase when exact stage latency is required, when the sampled
result is close enough that profiler uncertainty could change the conclusion,
or when repeated optimization work needs a stable libjxl baseline.

### Architecture

Implement the profiler in an isolated worktree at the pinned libjxl revision.
Do not modify the checked-out submodule in place and do not add timing fields to
libjxl's public C API or `JxlEncoderStats`.

Add an internal, opt-in profile sink with these properties:

- a null sink is the default and performs no clock reads or allocations;
- stable stage IDs are independent of function names;
- top-level timers surround synchronization/barrier boundaries and report wall
  time;
- worker-local substage accumulators are merged after group work, avoiding a
  shared atomic update for every token or group;
- aggregate worker time is labeled separately and may exceed enclosing wall
  time;
- invocation counts, token counts, histogram counts, model bits, token bits,
  and output bytes use checked integer accumulation; and
- the benchmark harness, not the encoder library, serializes the result as
  JSON.

Keep the patch small enough to rebase onto another pinned libjxl revision. A
compile-time option may expose the internal hooks to the comparison harness,
but profiling must remain disabled in ordinary cjxl and library builds.

### Required stage boundaries

Record wall phases for:

1. coefficient and Modular side-data tokenization;
2. entropy histogram/model construction;
3. model/header and token-stream emission;
4. frame header, TOC, section concatenation, and final output copy; and
5. their complete serializer union.

Record aggregate work counters for at least:

- histogram population;
- histogram clustering;
- HybridUint configuration selection;
- ANS or prefix/Huffman model construction;
- histogram/context-map serialization;
- token encoding and bit writing;
- Modular/DC side-data encoding; and
- output assembly and copying.

Instrumentation around a leaf such as every `WriteTokens` call is acceptable
for invocation and aggregate-work accounting, but the direct wall comparison
must use the enclosing group barrier. Summing overlapping leaf durations is not
a stage-latency measurement.

### Proposed raw schema

The libjxl record should contain:

```json
{
  "schema_version": 1,
  "timing_semantics": {
    "phase_nanoseconds": "wall-clock-barrier-time",
    "work_nanoseconds": "aggregate-worker-time"
  },
  "encoder": "libjxl",
  "revision": "<full commit>",
  "thread_count": 8,
  "requested_distance": 1.0,
  "effort": 7,
  "samples": [
    {
      "encoded_bytes": 0,
      "token_count": 0,
      "histogram_count": 0,
      "phase_nanoseconds": {},
      "work_nanoseconds": {},
      "invocation_counts": {}
    }
  ]
}
```

The comparison tool should translate both native schemas into the neutral
stage table. Do not rename or discard native fields in the retained raw files.

### Instrumentation validation

Before using the profile:

- verify profiled and unprofiled builds emit byte-identical codestreams for the
  complete corpus;
- run focused ANS, histogram, frame-encoding, and decoder tests from the pinned
  libjxl revision;
- confirm all outputs decode and match the unprofiled decoded pixels;
- verify stage and substage counts are stable across repeated encodes;
- check that wall phases do not overlap unless explicitly documented;
- confirm aggregate worker time is never interpreted as a wall-time sum; and
- measure profiling perturbation against the unprofiled build in balanced
  alternating process pairs.

Use the profiler to explain the result. Retain performance claims only from the
unprofiled build or explicitly label direct stage timings as diagnostic when
their perturbation is material.

## Result artifact

The final artifact directory should contain:

- `README.md` with scope, conclusions, caveats, and reproduction commands;
- `manifest.json` with revisions, hashes, build commands, host state, and input
  provenance;
- raw unprofiled samples for both encoders;
- native GJXL schema-9 samples;
- parsed libjxl and GJXL Samply operation tables;
- optional native libjxl stage-profile samples;
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

1. **Infrastructure complete; corpus selection remains.** Pin the comparison
   corpus, canonical conversion, revisions, and host/build manifest.
2. **Complete.** Add the benchmark-only GJXL serializer worker override.
3. **Complete.** Add the comparison driver and raw unprofiled result schema.
4. **Complete.** Extend the Samply classifier with the neutral stage mapping
   and tests.
5. Run the no-libjxl-source-change serial and production pilot.
6. Calibrate and run the matched-quality view.
7. Decide from the pilot whether direct libjxl stage timing is necessary.
8. If required, implement and validate the isolated opt-in libjxl profiler.
9. Capture the final unprofiled paired runs and publish the retained artifact.
