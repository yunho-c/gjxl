# Selected histogram sample-row prototype

Implemented and qualified on `feat/modular-sample-rows`, based on `feat/modular`
at `9d8a011e803476cce13e012b0ffeac07ad0645ec`, including the newly received ANS allocation
and training-row hash reuse. This uses the public native
histogram learner and the existing completed weighted-feature maps. The legacy
GPU search learner is not part of the experiment.

**Decision: keep the experiment opt-in.** Neither selected CPU preparation nor
Metal sample-row generation established a useful complete-encode speedup on this
three-image run. Metal preparation still spends about 90% of its gathering
interval loading rows into native columns and executing ordered CPU deduplication.
Reducing the remaining CPU hashing/lookup work and row materialization is the
next priority before promoting the new producer.

## Implemented boundary

| Component | Implementation |
| --- | --- |
| Weighted prediction and property 15 | Reuse the existing completed feature owner and its Metal plane/feature buffers. No second plane upload or new WP implementation. |
| Sampling decisions and quantizer cuts | Original native CPU algorithms, seeds, channel resets, group holes, tables and admission counts. |
| Raster CPU control | Existing arithmetic and RNG order; the experimental mode dispatch is outside the pixel loop. |
| Selected CPU rows | Skip unselected feature/property work when completed WP maps are available. Reconstruct property 8 from the preceding pixel's local neighborhood. Scalar WP fallback still visits every pixel. |
| Metal rows | One thread per selected position: local neighbors, properties, configured predictor residuals, native residual tokens, property-to-bin lookups and compact SoA output. |
| Deduplication and weights | Original ordered two-hash lookup/insertion, resize behavior and saturated weights, replayed on CPU. |
| Fitting, tokens and entropy | The same current native learner, existing GPU tokenization and CPU estimated-ANS path for all three variants. |

This includes the proposed property-to-bin and compact-row task. Cut construction,
sampling policy and deduplication stay on CPU.

Once the WP map is complete, row computations have no inter-row dependency.
Property 8's previous gradient can be reconstructed from source neighbors; it
requires no scan. All 16 properties and all 14 predictors use the existing wide
integer arithmetic shared with adaptive Metal tokenization. Residual overflow is
checked before publication. The shader performs integer arithmetic, table lookups
and stores, with no floating-point quantizer or approximate predictor.

The GPU dispatch unit is one independent learning scope, batching its included
channels. The existing local-learning scheduler can have multiple scopes in
flight. The three large images each dispatch 192 nonempty scopes, without
materializing all frame rows at once. The shared/global workflow deliberately
uses selected CPU rows in this prototype; reported GPU counts expose that fallback.

Default rows contain two uint32 static bins, fourteen byte dynamic bins and two
pairs of residual-token bytes: **26 bytes per selected row**, before CPU weights.
The GPU receives a uint32 selected-position list and native mapping tables. The
extra host position list, device dispatch storage and compact output are included
in managed admission. Source buffers remain borrowed through the joined batch.

## Measurement

Apple M4 Pro, 14 CPU participants, 48 GiB, AC power; macOS 27.0.1 and Apple Clang
21.0.0. Release build: `-O3 -g -DNDEBUG`. Feature CPU fallback remains at four
participants. Automatic buffering, best predictors (weighted and gradient),
estimated ANS, and the same automatic GPU feature/token route are held fixed.

Three fresh-process repetitions per input and variant, with variant order rotated
between repetitions: **27 timed encodes**. Backend creation, file I/O and
independent decoding are outside the timer. The timer wraps the complete native
`EncodeReferenceModularImage` call, including admission, selection, weighted
features, sample preparation, fitting, tokenization, entropy and output publication.
This is the modern native workflow, not a separate public-API-wrapper timing.

Other codec qualifications were allowed to finish before builds/measurements.
The runner monitors overlapping codec tests, builds and qualification scripts,
retaining and retrying an affected run. This capture waited for ThreadSanitizer
tests; one attempt overlapped a later TSAN ANS probe and was discarded and retried.
No overlap was detected in the retained measurements. Spotlight indexing was
active in this session, so these are **session measurements, not isolated peak throughput**.
The raw repeat spread matters; percent-level differences are not a robust speedup
claim.

Complete-call medians in seconds; positive delta means Metal took longer:

| Input | Raster CPU rows | Selected CPU rows | Metal rows | Metal time delta |
| --- | ---: | ---: | ---: | ---: |
| photo-large-1029 | 4.404 | 4.443 | 4.382 | -0.5% |
| photo-large-1039 | 3.814 | 3.776 | 3.851 | +1.0% |
| photo-large-1049 | 3.156 | 3.160 | 3.232 | +2.4% |

The sum of per-input medians is 11.374 s for raster, 11.379 s for selected CPU
(+0.0%), and 11.464 s for Metal (+0.8%). All three variants produce identical
encoded bytes within each input/predictor policy. These observations do not
justify changing the default.

Preparation and its Metal subintervals, in **summed scope-wall seconds**:

| Input | Raster prep | Selected CPU prep | Metal prep | Metal generation | CPU replay/dedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| photo-large-1029 | 4.666 | 4.752 | 5.042 | 0.399 | 4.504 |
| photo-large-1039 | 3.882 | 3.857 | 4.306 | 0.417 | 3.735 |
| photo-large-1049 | 3.732 | 3.687 | 4.278 | 0.368 | 3.756 |

These work intervals overlap across participants; they are not additive
caller-wall stages. Preparation includes within-job planning, allocation and
quantizer work. Generation includes output/dispatch allocation, map/index copying,
submission and completion wait. No device timestamp was taken, so generation is
not a pure kernel duration. Replay includes reading compact GPU rows, appending
native columns, hashing, lookup/insertion and weighting; it is not a hash-only
measurement.

Metal removes CPU property/residual/bin calculations but adds a materialized
26-byte row and a CPU pass over it. Replay alone remains close to the original
entire gathering cost. Parallel arithmetic therefore does not translate into a
useful net gain here. The learner's columns and ordered deduplication are the
important integration boundary.

Allocation/copy takes roughly 0.014–0.015 summed seconds per large image;
submit/wait takes roughly 0.35–0.40 seconds. Bigger dispatch batches could reduce
submission cost, but leave most replay work intact. A useful follow-up is to
retain the received CPU row-hash reuse and consider eliminating the extra column
copy or generating the two remaining native row hashes alongside bins. GPU hash generation adds
output bytes and must preserve native hash/order semantics; generic sorting and
deduplication is not byte-parity evidence. Tree-fitting acceleration remains a
separate effort.

## Correctness and coverage

Six focused CTest targets pass: histogram learning, histogram preparation, hybrid
workflow, hybrid palette workflow, Metal reference workflow, and the new Metal
sample-row test. The new test runs **16 cases** and compares all prepared columns,
residual tokens, weights, quantizer cuts, tree options and complete learned trees
against the raster control. It exercises:

- Width-one, height-one, padded strides, slice boundaries, metadata prefixes,
  excluded channels and holes in learning-image IDs.
- Nondefault WP parameters; distinct feature-owner versus native learning IDs;
  group IDs above the mapping clamp range.
- Sampling fractions 0.01, 0.65 and 1; 2, 8 and 256 requested quantizer bins;
  zero, one and two static columns; reordered property subsets.
- Default predictors and all 14 predictors in reversed order; signed
  near-int32-limit ramps, ordinary 16-bit samples and constant inputs.
- Duplicate weights reaching 65535, allocation-free storage planning, actual
  backing within admission, released resources, and injected GPU allocation,
  submission and completion failures preserving the published output.

The final binary passes **45 accepted complete encode/decode calls**, plus the
byte-exact attempt excluded for competing work: the 27 timed calls,
three variants on four small photo/alpha/palette/16-bit inputs plus a 769x513
RGBA16 local-learning fixture (15 calls), and all 14 predictors on that local
fixture (3 calls). Every call passes independent exact decoding using the pinned
libjxl oracle; variants match encoded bytes within each input/predictor policy.
The local RGBA16 fixture uses 12 GPU row dispatches and 1,030,284 selected rows.
Small shared/global inputs have zero GPU row dispatches: they qualify the explicit
selected-CPU fallback, not GPU global preparation.

Receiving `9d8a011` was followed by a fresh six-test pass and a fresh 45-call
accepted matrix. All output hashes also match the pre-receive captures. Earlier
pilot captures are retained separately and are not mixed into the tables above.

## Reproduction and artifacts

Configure with the repository's Metal headers and existing pinned-libjxl oracle
build. Compiler/options, input hashes, binary hash, dirty source patch and newly
added source files are retained with the captures.

```sh
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -g -DNDEBUG' \
  -DGJXL_BUILD_TESTS=ON -DGJXL_ENABLE_METAL=ON \
  -DGJXL_ENABLE_LIBJXL_REFERENCE=OFF \
  -DGJXL_MODULAR_ORACLE_BUILD=/Users/yunhocho/GitHub/gjxl/build/pinned-libjxl
cmake --build build/release --target gjxl_modular_hybrid_profile \
  gjxl_metal_histogram_sample_rows_test -j 12
ctest --test-dir build/release --output-on-failure -R '^metal_histogram_sample_rows$'

# INPUT is a PAM/PNM file; OUT must not already exist.
build/release/gjxl_modular_hybrid_profile INPUT OUT auto auto best \
  fit-off 14 14 14 14 14 metal
# Last argument: raster, selected, or metal. Use all instead of best for all
# 14 predictors. The public encoder default remains unchanged.
python3 tools/modular_reference_learning/run_sample_rows_qualification.py \
  --binary build/release/gjxl_modular_hybrid_profile --out NEW_CAPTURE \
  --input INPUT_1 --input INPUT_2 --input INPUT_3 --repeats 3
```

- [Machine-readable results](modular-sample-rows-prototype-results.json).
- [Final timed runs and provenance](../build/sample-rows-9d8a011-qualification/runs.json).
- [Final coverage](../build/sample-rows-9d8a011-coverage/runs.json) and
  [all-predictor coverage](../build/sample-rows-9d8a011-all-predictors/runs.json).
- [Final test log](../build/sample-rows-9d8a011-tests.log).
- [Native preparer](../src/codec/modular/histogram_sample_preparation.cpp),
  [row interface](../src/codec/modular/histogram_sample_rows.h),
  [Metal operation](../src/gpu/metal/metal_modular_adaptive_tokens.cpp),
  [shader](../src/gpu/metal/kernels/modular_adaptive_tokens.metal),
  [exactness test](../tests/metal_histogram_sample_rows_test.cpp), and
  [measurement runner](../tools/modular_reference_learning/run_sample_rows_qualification.py).

Build-directory captures are local artifacts. Report, results and implementation
are maintained on the isolated `feat/modular-sample-rows` branch. Public policy
remains unchanged; integration into `feat/modular` is a separate step.
