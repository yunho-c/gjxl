# Resident stage-profiling overhead checkpoint

Measured source: `3e1ef9e8e31bc809a6c3f767af3eed23f4ce69bb`.
Apple M4 Pro (20 GPU cores), 48 GiB, macOS 15.6, production Release shaders.

This bundle preserves the 2026-09-21 overhead study before collecting the new
runtime-breakdown dataset. It contains no new production implementation or
new per-stage notebook captures. The subsequent decision is to proceed with
measurement and plotting, deferring further profiler optimization. Use matched
ordinary/profiled captures to report overhead; any rescaling must be explicitly
labeled as an estimated attribution and use matching timing boundaries.

## Findings and scope

- The 20-case sweep covered four image sizes and efforts 2/4/7/9/10. The
  follow-up covered 0.39 MP and 12 MP at efforts 4/7/9, with 36 alternating
  rounds and balanced mode order. There was one image per size.
- Full profiling added a median paired 1.47 ms at e7 and 2.73 ms at e9 on
  the 0.39 MP image. Large-image differences remained too variable for a
  reliable general overhead estimate. No universal correction is established.
- Splitting encoders alone had no consistently resolved complete-call penalty.
  Timestamp setup was a larger measured host cost than metadata handling.
- All 3,456 measured encodes and 1,512 logged warmups passed byte/summary
  equality checks against ordinary references. Submission counts matched
  across controls. Cross-build reference hashes matched in the six shared cases.

[REPORT.md](REPORT.md) has the full tables, definitions, protocol, and limits.
[DECISION.md](DECISION.md) retains the interpretation made immediately after the
experiment. Its optimization priorities are conditional; optimization was
subsequently deferred in favor of collecting the figure data.

## Contents and provenance

- `measurements.tar.gz`: 110 exact original files from `ablation/` and
  `followup/`: raw JSONL, completion records, commands, stderr, identities,
  and full paired summaries. The archive uses deterministic paths/timestamps.
- `harness/`: exact original probes, analysis/collection scripts, diagnostic
  patch and header, generated follow-up source, build commands, job lists,
  and binary/library hash records. These are experimental controls, not
  supported production modes. Original absolute paths are retained as provenance.
- `verification.json`: verification recorded at collection time.
- `original-manifest.json`: the original local artifact manifest, including
  hashes of files deliberately omitted from this compact bundle.
- `environment.txt` and `environment-context.json`: machine/thermal context.
  The latter omits full process inventories and retains their original hashes.
- `bundle-sha256.json`: integrity hashes for this committed bundle.
- `verify.py`: offline integrity, coverage, equality-record, and analysis check.

The full original directory remains at
`build/profiling-overhead-20260921/` in the profiling-alignment worktree. Large
build trees, source archives, binaries, inputs, reference codestreams, the
exploratory pilot, and full process inventories are not duplicated here.
Their omission does not remove any accepted timing rows used by the report.
The original reports describe that full local layout; archived cohort files
can be inspected with `tar -tzf measurements.tar.gz` or extracted elsewhere.

## Verify without encoding or GPU activity

From the repository root, with Python 3.11 or newer:

```sh
python3 docs/metal-profiling-overhead/20260921/verify.py
```

This checks bundle and original raw hashes, the two probe/driver identities,
all recorded equality flags and submission/encoder counts, cross-build output
hashes, and recomputes both paired summaries from the raw rows. It uses only
the Python standard library and temporary files. It does not build, encode,
access the corpus, or validate decoded pixels anew; it verifies the preserved
results of the original C++ byte/summary checks.

## Rebuild the diagnostic follow-up (macOS / Metal)

The original scripts expect the measured revision and their original relative
build layout. Use a fresh detached checkout to preserve those assumptions and
keep new measurements separate from this bundle. The corpus paths/hashes in
`harness/followup-code/jobs.json` and the archived identity must be available;
using different inputs requires a separately declared new study.

These commands build a new diagnostic binary but do not start collection:

```sh
EVIDENCE="$PWD/docs/metal-profiling-overhead/20260921"
REPRO=/tmp/gjxl-profile-overhead-repro-20260921
git worktree add --detach "$REPRO" 3e1ef9e8e31bc809a6c3f767af3eed23f4ce69bb
RUN="$REPRO/build/profiling-overhead-20260921"
mkdir -p "$RUN/source"
cp -R "$EVIDENCE/harness/." "$RUN/"
git archive 3e1ef9e8e31bc809a6c3f767af3eed23f4ce69bb | tar -x -C "$RUN/source"
python3 "$RUN/patch_ablation.py"
cmake -S "$RUN/source" -B "$RUN/ablation-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=OFF \
  -DGJXL_BUILD_BENCHMARKS=ON -DGJXL_ENABLE_LIBJXL_REFERENCE=OFF \
  -DGJXL_ENABLE_METAL_PROFILING=OFF -DGJXL_BUILD_FRONTIER_EXPERIMENT=OFF
cmake --build "$RUN/ablation-build" --target gjxl_encode -j 8
python3 "$RUN/prepare_followup.py"
```

Build paths/toolchain changes may change binary hashes. Preserve the new
`followup-code/sha256.json` and build commands rather than claiming binary
identity with the frozen measurement. The retained jobs and probe fix distance
1.2, eight CPU threads, Metal fully resident, and final-score collection off.

To deliberately start a new follow-up collection, then analyze its saved data:

```sh
python3 "$RUN/followup-code/run.py" --name followup-repeat \
  --probe "$RUN/followup-code/probe" --jobs "$RUN/followup-code/jobs.json" \
  --modes ordinary,host,graph,split,record,full --pairs 36 --samples 1 --warmups 2
python3 "$RUN/analyze_followup.py" followup-repeat
```
