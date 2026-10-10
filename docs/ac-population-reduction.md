# AC population reduction

The Metal tokenizer uses the specialized CPU population reducer by default in
normal builds. This is proposal #4 in the tokenization host-handoff work, layered
on the existing early-entropy, earlier-DC, and host-metadata defaults. It applies
whenever the existing Metal provider produces populations, including effort 4;
it adds no effort, single-image, or concurrency gate and requires no experiment
flag. Existing route eligibility is unchanged. `GJXL_GPU_TOKENIZATION=0` still
selects CPU tokenization and therefore bypasses this Metal population handoff.

## Representation and contracts

Production uses one histogram shard. The reducer primarily widens the GPU's 128
uint32 bins into CPU uint64 counts and computes token count, extra-bit count,
and maximum symbol. It specializes the supported 1/2/4/8 shard counts, keeps
summary accumulators local, and writes the summary fields once per context.
Each input count is widened before shard addition.

Output clusters must be value-initialized. Their unused upper 128 bins remain
zero. The caller retains the population-total check before publication. The
change introduces no allocations, workers, GPU kernels/submissions, or changes
to storage admission, buffer lifetime, failure recovery, or encoded bytes.
This remains CPU work; it does not move entropy modeling to the GPU.

The historical `GJXL_EXPERIMENT_TOKEN_SPECIALIZED_REDUCE` selection is removed;
the normal and experimental builds use this reducer. The experimental shard
control remains available in experiment-enabled builds.

## Qualification and adoption

The AC-only M4 Pro qualification covered 105 inputs, 967 image/effort/distance
settings, and nine additional resource/route controls. Across the sweep and
longer repeats, 12,496 public calls preserved complete codestream and public
summary equality. All 966 distinct bitstreams have independent decoder evidence
(637 fresh decodes and 329 exact-hash reuses). The normal Release suite passed
170/170 tests; two additional experiment-enabled GPU oracle/failure tests passed.
The independent CPU arithmetic oracle passed 2,136 cases under ASan/UBSan,
including every symbol/shard, sparse/dense data, wide sums, and unused-bin checks.

For the same 18 images at distance 1.9, geometric mean complete-call time savings
were +1.30% at effort 1, -0.86% at effort 4, and +0.54% at effort 7. These are
incremental results over #1–#3, on one device. Longer selected effort-4 repeats
had small median losses up to 0.13 ms / 0.72%; an isolated 31.8% round slowdown
is retained with no proven cause. No universal, tail-latency, batch-throughput,
or other-device improvement is claimed.

Default adoption is an explicit decision to accept that uncertain effort-4
tradeoff for the broader measured benefits. The historical investigation's
recommendation to defer unconditional adoption is preserved; it predates this
decision and is not rewritten as a claim that every slowdown disappeared.

The original checkout retains the investigation, broader qualification, and
production integration evidence under `reports/population-reduction-20260927`,
`reports/population-reduction-qualified-20260927`, and
`reports/population-reduction-default-20260927`, respectively. The production
runtime matches the qualified candidate apart from a comment identifying it as
production code. The integration uses a separate build tree and preserves the
frozen qualification source and artifacts.
