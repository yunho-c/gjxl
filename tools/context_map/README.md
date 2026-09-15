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
