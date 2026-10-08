# Phase 1 extraction qualification

This tool compiles against an installed GJXL package and the matching source
checkout's internal headers/fixtures. It must be built separately for the
baseline and candidate. See the [protocol](../../docs/modular-phase1/PROTOCOL.md)
and [qualification report](../../docs/modular-phase1/README.md).

Use an audited compiler from `docs/storage-toolchain.md`. The Windows run used
Release x64 MSVC 19.37.32826 with STL update 202305 and Windows SDK 10.0.26100.0.
Do not change the storage compatibility guard to make another compiler pass.

## Build and capture

Example commands from the repository root, in an initialized compiler shell:

```powershell
git worktree add --detach build/modular-phase1/baseline-src 9ca5da9
# Initialize the pinned libjxl submodule in that checkout, or point a read-only
# reference build at the existing clean third_party/libjxl checkout.

cmake -S build/modular-phase1/baseline-src -B build/modular-phase1/baseline -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 -DGJXL_ENABLE_METAL=OFF -DGJXL_ENABLE_CUDA=OFF -DGJXL_ENABLE_LIBJXL_REFERENCE=ON
cmake --build build/modular-phase1/baseline --parallel 6 -- -k 0
cmake --install build/modular-phase1/baseline --prefix build/modular-phase1/baseline-install

cmake -S tools/modular_extraction -B build/modular-phase1/baseline-capture -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 "-DCMAKE_PREFIX_PATH=$pwd/build/modular-phase1/baseline-install" "-DGJXL_SOURCE_DIR=$pwd/build/modular-phase1/baseline-src"
cmake --build build/modular-phase1/baseline-capture --parallel 4
```

Repeat for `candidate`, using `-S .` and `GJXL_SOURCE_DIR=$pwd`. The recorded
Windows all-target build has pre-existing test failures; the native libraries,
CLI, capture and applicable test binaries still build. Retain those failures.

The multi-group photo is an exact horizontal tiling of a pinned libjxl test
image, converted to linear float32. Pillow is needed only for this preparation:

```powershell
python tools/modular_extraction/prepare_photo.py third_party/libjxl/testdata/external/wesaturate/500px/tmshre_riaphotographs_srgb8.png build/modular-phase1/photograph-tiled.pfm
build/modular-phase1/baseline-capture/modular_extraction_capture.exe build/modular-phase1/baseline-stages 0 build/modular-phase1/photograph-tiled.pfm
build/modular-phase1/candidate-capture/modular_extraction_capture.exe build/modular-phase1/candidate-stages 0 build/modular-phase1/photograph-tiled.pfm
python tools/modular_extraction/evidence.py build/modular-phase1/baseline-stages --candidate build/modular-phase1/candidate-stages --output build/modular-phase1/stage-comparison.json
```

`REPEATS=0` writes all 204 deterministic artifacts (54 files, 54 serializer or
workflow snapshots, 96 numerical DC-stage snapshots) without timed repetitions.
Positive repeats write the original 108 artifacts and `timings.csv`. A final
optional comma-separated list selects exact case names for timing confirmation.
Snapshots encode each integer or float bit pattern as a little-endian uint64;
they never serialize structure padding. DC-stage snapshots include row padding.

`--cases photo_frame_p0,photo_frame_p1` in `measure.py` opts into photographic
stage measurements (and four additional correctness artifacts). The capture
prepares one fixed-DCT8, uniform-quantization CPU frame with zero AQ refinement
iterations and Gaborish disabled, then measures gradient and weighted residual
coding on that same frame. Padding, color conversion and CPU preparation are
outside the DC/serialization timers. The complete photographic workflow remains
the separate `workflow_photo` case.

Frame captures record geometry, selected tree tokens and context maps,
DC/metadata tokens, serializer and tokenization bounds, and entropy/profile
decisions. Each ordinary timed encode must match that case's profiled output.
Workflow captures retain bytes and non-timing summary decisions; repeated calls
must match both. All timed calls use one CPU participant. Fixture construction,
file writing, comparison, and the initial warm encode are outside timed regions.

## Independent tests

Build the reference tools from the clean pinned source using
`cmake/BuildPinnedDjxl.cmake`. On Windows pass the **x64** `clang-cl.exe` explicitly
and quote `"-DGJXL_EXECUTABLE_SUFFIX=.exe"` in PowerShell. The pinned build uses
its own compiler; GJXL continues to use the audited MSVC compiler.

```powershell
cmake "-DGJXL_LIBJXL_SOURCE=$pwd/third_party/libjxl" "-DGJXL_LIBJXL_BUILD=$pwd/build/modular-phase1/pinned-libjxl-x64" "-DGJXL_LIBJXL_REVISION=e8ff09762481785938d8e4e01333ed3917571161" "-DGJXL_GENERATOR=Ninja" "-DGJXL_EXECUTABLE_SUFFIX=.exe" "-DGJXL_C_COMPILER=$clang_x64" "-DGJXL_CXX_COMPILER=$clang_x64" -P cmake/BuildPinnedDjxl.cmake
cmake -S . -B build/modular-phase1/candidate "-DGJXL_DC_ORACLE_BUILD=$pwd/build/modular-phase1/pinned-libjxl-x64"
cmake --build build/modular-phase1/candidate --parallel 6 -- -k 0
ctest --test-dir build/modular-phase1/candidate --output-on-failure --parallel 4 --timeout 180 --output-junit "$pwd/build/modular-phase1/candidate-tests.xml"
```

Run the frozen baseline CTest suite too. Invoke each build's
`gjxl_codestream_conformance_test` with the same `--decoder .../tools/djxl.exe`,
`--info .../tools/jxlinfo.exe`, its own `--encoder .../gjxl_encode.exe`,
`--sample testdata/codestream_sample.pfm`, and a separate `--artifacts` directory.
The optional direct oracle dependencies are test-only. Production libraries and
installed targets do not link libjxl.

For the recorded baseline quantization-oracle reproduction, the unchanged oracle
test object was linked to `baseline/gjxl_codec.lib` instead of the candidate
library, with the identical pinned reference libraries. The exact link arguments
are in the qualification manifest. This avoids modifying the frozen checkout.

## Paired timing

Stop other builds/tests before running. The script warms both binaries, then
alternates their execution order for seven pairs and retains every sample:

```powershell
python tools/modular_extraction/measure.py build/modular-phase1/baseline-capture/modular_extraction_capture.exe build/modular-phase1/candidate-capture/modular_extraction_capture.exe build/modular-phase1/photograph-tiled.pfm build/modular-phase1/paired-timings
```

The output directory must be new. Exit code 1 indicates a gate was flagged;
confirm flagged cases with a new seven-pair run and `--cases name1,name2`.
Keep both results. `--repeats 31` retains more per-call samples within each pair
when a noisy case needs further resolution. Hash correctness artifacts independently with `evidence.py`;
performance results never replace byte comparisons.

`record.py build/modular-phase1 docs/modular-phase1/manifest.json` records this
Windows qualification's source, tool, test and log identities. It checks that
the frozen source and pinned reference remain clean and that no baseline test
passes were lost.
