# Reusing training-row lookup hashes

This change starts from `2f9c4de657312e94ec57eeebbebcebf4e3a474e6` on
`perf/training-row-hash-reuse`, including the constant-property and ANS-storage
optimizations. It changes only `histogram_sample_preparation.cpp` in production.

`HistogramSampleBuilder::Add()` retains the two probe positions computed for
duplicate lookup and passes them to `Insert()`. The row and table size are
unchanged between these operations. A retained new row therefore evaluates
each hash once instead of twice: two full row hash traversals instead of four.
Rows merged into an existing row already computed each hash once.

Both hash functions, first/second-slot preference, collision handling, row
order, weight saturation and overflow rejection are unchanged. `BeginChannel()`
still computes fresh positions when rebuilding the table. No new allocations,
retained buffers, shared state or public options are introduced.

## Correctness

Seven focused Release suites passed: `modular_histogram_preparation`,
`modular_histogram_learning`, `modular_parallel_learning`,
`modular_reference_workflow`, `modular_reference_hybrid`,
`modular_reference_hybrid_palette` and `modular_api`.

The standalone reference harness also passed `native_preparation_contract`
and `native_preparation_differential`: 119 generated comparisons retained every
sample field and raw tree node, and the explicit pinned-reference weight
overflow case was rejected. This includes duplicate-heavy inputs, table growth,
saturating weights, different predictors/property layouts, group holes and
weighted-predictor presets. The contract suite checks admission, storage bounds,
allocation failures and failure-atomic publication.

The reference build uses newly generated hooks for pinned libjxl
`e8ff09762481785938d8e4e01333ed3917571161`. Its native sampler/fitter compile with
ordinary Release flags. The local bridge configuration additionally links the
matching Metal and GPU Butteraugli archives/frameworks because this production
build enables Metal; the reference adapter's existing CMake list assumes CPU
archives. No reference algorithm or production build configuration was changed
to obtain the comparison.

Logs are retained in `build/hash-reuse-release-tests.log`,
`build/hash-reuse-reference-tests.log` and
`build/reference/bin/Testing/Temporary/LastTest.log`.

## Public encode capture

The paired runner uses the unchanged public `gjxl_ans_allocation_profile`
driver. It freezes both executables, the candidate sources/patch, dependency
hashes and input identities. The baseline is the successful ANS-reuse capture's
candidate executable, whose production source differs only in the sampler.
Both builds use Release `-O3 -g -DNDEBUG`, default learned/estimated-ANS policy,
and 14 CPU participants on AC power.

Each of the three 4000x3000 RGB8 images runs through explicit CPU and Metal
routes, with one discarded call and three measured calls per build/condition.
Baseline/candidate order alternates. All calls check the previously qualified
compressed SHA-256 and independently decode to the original integer samples.
Backend initialization, file I/O and independent decoding are outside timing.
The training interval includes selection, preparation and fitting; it is not
an isolated hash or sample-preparation benchmark.

The other fitting-schedule qualification completed before this work began.
The runner checks for other codec/test processes before each call and monitors
them during encoding. If another job appears, it discards that attempt and
waits before retrying. Desktop activity is still recorded and uncontrolled.

All **48 encodes** (12 discarded calls and 36 measured calls) matched the
qualified bytes and decoded exactly. Every call used 14 entropy participants
and 192 populated streams. No competing-work interruption occurred. Source,
executable, input and dependency hashes remained stable throughout capture;
all 985 frozen source/tool files also matched the final working copy.

Measured medians on Apple M4 Pro, in seconds:

| Image / route | Complete encode, baseline -> reuse | Training, baseline -> reuse |
| --- | ---: | ---: |
| 1029 CPU | 4.049 -> 3.995 | 2.968 -> 2.938 |
| 1029 Metal | 4.492 -> 4.466 | 3.521 -> 3.480 |
| 1039 CPU | 3.062 -> 2.974 | 2.120 -> 2.047 |
| 1039 Metal | 3.901 -> 3.830 | 3.048 -> 2.978 |
| 1049 CPU | 3.009 -> 3.034 | 2.114 -> 2.122 |
| 1049 Metal | 3.261 -> 3.169 | 2.442 -> 2.358 |

Complete-call medians ranged from **2.9% faster to 0.8% slower**. These are
modest, mixed changes on an active desktop, not a general throughput guarantee.
For example, CPU 1049 ranged from 3.005–3.329 seconds before and 2.917–3.687
seconds after. Process-CPU medians were 0.2–2.6% lower across these conditions,
but this capture does not isolate hash cost or sample-preparation time.

[Structured results](modular-training-row-hash-reuse-results.json) retain all
48 observations, timing medians/ranges, input/executable/source identities and
test-log hashes, bound to the raw capture manifest. The raw before/after
environment snapshots remain in that manifest. This change is enabled
automatically; there is no opt-in flag.

## Reproduction

Configure a Release Metal build with tests enabled and
`GJXL_MODULAR_ORACLE_BUILD=/path/to/pinned-libjxl`, then build
`gjxl_ans_allocation_profile` and the focused test targets above. The existing
ANS-reuse capture supplies the frozen baseline and its provenance manifest.

```sh
python3 tools/modular_reference_learning/run_sample_hash_qualification.py \
  --baseline-capture /path/to/ans-allocation/qualification \
  --binary build/release/gjxl_ans_allocation_profile \
  --input /path/to/photo-large-1029.pam \
  --input /path/to/photo-large-1039.pam \
  --input /path/to/photo-large-1049.pam \
  --out build/sample-hash-qualification
```

The output directory must be new. Its `manifest.json`, frozen executables and
source archive, per-call logs/results, and byte-verified APFS-cloned outputs
retain the raw evidence. No timing threshold is a correctness gate.
