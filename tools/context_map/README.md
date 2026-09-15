# Native context-map qualification

This explicitly invoked study compares the native context-map encoder with the
completed DC integer-mapping implementation (`e8abc34`). It uses the retained
65-image corpus and seven original distances; it does not download inputs.

Keep the baseline executable and its Metal library immutable. Build the new
implementation separately:

```sh
cmake -S . -B build/context-map-native -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON
cmake --build build/context-map-native -j 8
python3 tools/context_map/qualify.py rate
python3 tools/context_map/qualify.py timing
python3 tools/context_map/qualify.py report
```

The default baseline is `build/dc-uint-fast-candidate/gjxl_quality_benchmark`.
Its hash must match the completed `build/dc-uint-fast-qualification-20260914`
manifest. The default candidate is `build/context-map-native/gjxl_quality_benchmark`.
Both executable paths, the previous qualification directory and the new output
directory can be overridden with command-line options.

`rate --limit 14` is a resumable smoke run. Run `rate` without the limit for
all 455 points. Every fresh baseline must reproduce its retained codestream;
every candidate must match the retained linear-sRGB float PFM hash before its
measured score can be reused. Temporary decoded PFMs are removed after that
check; codestreams, raw samples, source snapshots, hashes and logs remain.

The first call freezes sources, including new source files, and helper versions.
Use a new `--root` when changing the implementation; never modify a frozen run.
The report rejects incomplete coverage and integrates log(bytes) over
SSIMULACRA2 75–85 with PCHIP and Akima, without extrapolation.

Timing is a separate action after correctness collection and with other heavy
work idle. It uses 12 declared images, four alternating process pairs, two
warmups and five complete-encode samples per process. Process snapshots and
power state are retained; successful collection alone does not prove that the
machine was uncontended.

For the independent context-map decoder oracle:

```sh
cmake -S . -B build/context-map-reference -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_ENABLE_LIBJXL_REFERENCE=ON
cmake --build build/context-map-reference -j 8 --target gjxl_context_map_test
ctest --test-dir build/context-map-reference -R '^context_map$' --output-on-failure
```

The reference uses only test-linked libjxl decoder sources. The native encoder
has no libjxl runtime dependency.

## Fresh latency qualification

After a completed rate audit, reuse its immutable binaries and measured image
points without collecting the rate corpus again:

```sh
python3 tools/context_map/latency.py collect
python3 tools/context_map/latency.py report
```

The default output is a separate `build/context-map-latency-20260915` directory;
`--root` selects a new run and `--reference` selects the completed rate audit.
The earlier rate and provisional timing artifacts are preserved. Executables,
production sources, helper versions, inputs and reference codestream hashes
are checked before collection and again during reporting.

The protocol uses the same 12 images at their original Q80 distances, effort 4,
fully resident Metal and eight participating CPU threads. It collects two
blocks of four process pairs, alternating arm order and rotating image order,
with two warmups and five measured complete calls per process. Each output must
match the already decoded rate-run codestream hash. No builds, decoder runs
or quality scoring occur during measurement.

The runner samples process CPU time, competing codec/build processes, AC power
and thermal/performance status every second and at pair boundaries. It waits
for 20 quiet seconds initially and after contention. A pair is excluded when
a competing job appears, another process exceeds 80% of one CPU, aggregate
other CPU exceeds 200%, AC power is lost, or monitoring/thermal checks fail.
CPU thresholds and retry rules are frozen before measurement. Every attempt
and its environment evidence remain on disk; both arms are retried together,
with a maximum of three attempts per pair. Latencies never control exclusion.
An idle wait stops after ten minutes and can be resumed later.

The report audits every accepted and excluded attempt, raw sample and output
hash. It reports per-image paired ratios and separate block medians. Passing
the sampled checks establishes a local warm-call comparison under this
protocol; it does not prove that all background activity was absent.
