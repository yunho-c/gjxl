# Resident execution qualification

This package reconstructs two committed GJXL revisions and qualifies them with
one standalone driver compiled separately against each revision's headers and
libraries. It does not consume earlier build directories, validation manifests,
or sanitizer logs. Runtime source is never taken from the working-tree diff.

The default baseline is the integrated preparation/handoff merge
`ec4d4c5317d983b5f6df29d2474923443c9cfe63`; the candidate defaults to the repository's
committed `HEAD`. `--historical` selects candidate `07dd92e`, the runtime measured
in the [historical scheduling record](../../docs/resident-scheduling-qualification.md).
Fresh results are separate experiments, even with those same commits.

## Dependencies

Use Apple Silicon macOS, Xcode with Metal compiler tools, CMake 3.24+, Ninja,
Git, and Python 3.12+. Python optimization (`-O`/`PYTHONOPTIMIZE`) is rejected
because protocol assertions are qualification gates. Builds use Release,
C++20, the selected Xcode compiler/SDK, tests and benchmarks enabled, and optional
libjxl reference/Metal profiling disabled. The archived baseline predates the
current toolchain checks; consult the [storage contract](../../docs/storage-toolchain.md)
for supported current toolchains. Record different machines/toolchains as fresh
measurements, not reproductions of historical timings or executable hashes.

The recorded SDK is also passed to test subprocesses through `SDKROOT`, including
tests that invoke Clang directly.

Supply a canonical PFM directory and pinned `djxl`/`jxlinfo` paths explicitly.
No user-specific directory is a default. Paths containing spaces are supported.

### Canonical corpus

[corpus.json](corpus.json) specifies 38 inputs: names, dimensions, canonical and
source SHA-256, source URLs/licenses, color interpretation, and exact conversion
recipes. Supply the files directly beneath `--corpus`; additional files are
ignored. Configuration checks every required hash before starting builds.

The natural-image source PNGs can be retrieved from the manifest's `source`
URLs and must match `source_sha256`. Run each entry's `conversion` command with
`INPUT` and `OUTPUT` replaced by the source file and `canonical_path`. The
historical conversion used ImageMagick **7.1.2-24 Q16-HDRI**, linear-sRGB D65,
white alpha background, and the declared crop/resize operations. Verify the
canonical hash after conversion; another ImageMagick version is not assumed to
produce identical floating-point PFMs. Existing canonical files can be reused
on any machine only when their hashes match.

The two padded stress inputs are exact PFM exports of `FillSynthetic` in
[driver.cpp](driver.cpp), at 1919x1079 and 3839x2159. They use the same float
operations as the original encoding benchmark. Store RGB float32 pixels in
bottom-up PFM order with little-endian scale `-1.0`; preserve the canonical file
header as well as pixels when matching the manifest. These generated inputs
are distinct from the driver's in-memory `--synthetic` measurements. The
canonical corpus is an explicit external input, not downloaded during a run.
The [synthetic export recipe](corpus.md) regenerates both files directly from
the checked-in function and verifies their canonical hashes.

### Pinned decoder reconstruction

Build both decoder tools from libjxl
`e8ff09762481785938d8e4e01333ed3917571161`. For example, from the GJXL checkout:

```sh
qual_decoder_source="$PWD/build/qualification-libjxl-source"
qual_decoder_build="$PWD/build/qualification-libjxl-build"
git clone https://github.com/libjxl/libjxl.git "$qual_decoder_source"
git -C "$qual_decoder_source" checkout --detach e8ff09762481785938d8e4e01333ed3917571161
git -C "$qual_decoder_source" submodule update --init --recursive
cmake -DGJXL_LIBJXL_SOURCE="$qual_decoder_source" \
  -DGJXL_LIBJXL_BUILD="$qual_decoder_build" \
  -DGJXL_LIBJXL_REVISION=e8ff09762481785938d8e4e01333ed3917571161 \
  -DGJXL_GENERATOR=Ninja -P cmake/BuildPinnedDjxl.cmake
```

Pass the resulting `tools/djxl` and `tools/jxlinfo` together. Configuration checks
the decoder's reported revision and freezes hashes of both executables.
Rebuilt executable hashes may differ with compiler/SDK; the tools used for a
particular run cannot change during resume or verification.

## Configure and run

```sh
python3 tools/resident_qualification/run.py --out build/resident-qualification configure \
  --repo . --candidate-ref HEAD --profile smoke \
  --corpus /path/to/canonical \
  --decoder /path/to/pinned/tools/djxl --info /path/to/pinned/tools/jxlinfo
python3 tools/resident_qualification/run.py --out build/resident-qualification all
```

Use a dedicated output directory. Put it outside the source tree or under
`build/`. Reconfiguration must match its existing configuration exactly; use a
new directory when changing revisions, inputs, tools, or profile. No commit,
checkout, or reset is performed in the supplied repository. Required commits
must be available there. Only the required pinned metal-cpp submodule is
exported; if its object is unavailable locally, reconstruction fetches that
commit from Apple's repository. Optional libjxl sources are not part of these
GJXL builds.

`build` archives each commit into its own source tree and builds both with
Ninja. It finishes `gjxl_metal_shaders` before embedding/linking, avoiding the
historical parallel-Make embedding race without patching baseline source.
A separately linked probe compares `EmbeddedMetalLibrary()` byte-for-byte with
the generated metallib. This protocol requires matching shader payloads across
the comparison; kernel changes require a separately reviewed protocol.

`--baseline-ref` and `--baseline-capability current` also support comparisons
against a newer baseline. The default `legacy` capability matches `ec4d4c5`;
configuration/build checks the presence of the admission API. Legacy managed
counters and image scheduling fields are unavailable (`null`), not zero.

Individual resumable phases are:

```text
build → smoke → tests → sanitizers → parity → conformance → retained
      → performance → pressure → seal → verify
```

`status` reports recorded progress, `identities` checks frozen dependencies,
`report` regenerates the selected-profile report, and `audit` checks phase
completeness and raw-record semantics. `verify` validates sealed hashes and
recomputes report arithmetic **without rewriting evidence**.

Completed sample jobs and tests resume only with matching arguments, identities
and logs. Interrupted command logs and partial outputs are retained; retries use
new attempt names. An interrupted source export requires a new output directory.
Timeout/interruption terminates the launched process group. Do not run two
controllers against the same output directory simultaneously. Keep redirected
controller console logs outside that directory so they cannot change after
sealing. The append-only execution event log is deliberately not sealed.
Resume and verification require the recorded source/build trees, package, corpus,
and decoder tools at their configured paths. The output directory is an evidence
workspace, not a relocatable archive of every external dependency.

## Profiles and measurement boundaries

Both profiles run full discovered Release suites, repeated integration tests,
scoped sanitizers, 56 corpus/policy byte comparisons, six pinned decodes forming
three comparisons, 22 conformance fixtures per revision, and 24 retained
changed-image mixed-batch/two-caller comparisons.

- **Smoke** additionally exercises five representative timing workloads with
  two alternating pairs each, and six pressure processes covering a 4K single
  and simultaneous callers at CPU one/three and tight/full limits as applicable.
  It validates the measurement machinery; it is not full performance qualification.
- **Full** runs all 27 workloads, seven alternating independent-process pairs
  each, and 42 pressure processes. Select it at configuration with `--profile full`.

Performance processes make nine calls/cohorts, discarding two warmups; pressure
processes make five, discarding two. Reports compare process medians within each
pair and retain every pair. Before every measured process the controller refuses
launch while another encoder, test, compiler, or build is active. It does not
stop unrelated processes or promise that macOS is idle.

Backend setup is separate. Single-call timing covers the complete public API;
concurrent makespan covers dispatch through joining both callers. Queue, service,
and ready measurements are per-image, not aggregate worker time. Physical peak
is the process-lifetime high-water mark; backend-alive idle follows result
release; post-trim follows synchronous driver stop/destruction and cache trim.
The current driver objects remain allocated after shutdown, while legacy driver
objects are destroyed before trim. Inputs and backend remain alive. Managed
committed capacity includes unbacked reservation and is not an RSS limit.

## Correctness exceptions and sanitizer scope

A completely passing suite is accepted. The only permitted Release failure is
`quantization_pipeline` with the documented actual `0.24919039011001587` and
expected `0.24914586544036865`, independently reproduced on the fresh baseline.
Test inventories and counts are discovered rather than copied from history.
Skipped required tests, changed diagnostics, and additional failures fail the run.

TSan instruments C++ and Objective-C++, running all seven CPU launch groups and
ten repetitions of CPU execution/lifecycle/scheduling. ASan/UBSan instruments
C++ with Objective-C++ flags unchanged: seven unsuppressed CPU launch groups,
three repetitions of CPU execution/lifecycle/scheduling, four Metal launch
groups and three repetitions of Metal lifecycle/scheduling. Metal groups use
only [metal-ubsan.supp](metal-ubsan.supp); the unsuppressed Metal diagnostic is
retained separately. That probe accepts success or the documented metal-cpp
null-call diagnostic, whether the sanitizer exits with code 1 or aborts on macOS.
Other failures remain terminal. Leak detection is disabled. These results are not claims
of leak-sanitizer or unsuppressed Metal cleanliness.

CMake registers the controller tests when Python 3.12+ is available. Run them
independently with:

```sh
python3 tools/resident_qualification/test_runner.py
```

## Implementation validation

On 2026-09-07, the tracked runner completed the **smoke** profile against freshly
reconstructed `ec4d4c5` and `2443ae5`. The machine was Mac16,7 with 14 logical CPUs
and 48 GiB memory, macOS 15.6, Apple Clang 17.0.0 (`clang-1700.6.4.2`), and SDK 26.2.
The runtime came from those commits; this package and documentation were the
working-tree changes under validation.

- All 10 controller tests passed, including SDK propagation, paths with spaces,
  interrupted-command retention, stale/tampered evidence, profile completeness,
  and the known Metal diagnostic's abort/exit classification. CTest registration
  also passed.
- Release suites passed 63/64 baseline and 120/121 candidate tests. Both failures
  were the exact inherited quantization mismatch above. All 13 selected
  integration tests passed three repetitions.
- All 37 TSan and 26 scoped ASan/UBSan test executions passed. The unsuppressed
  Metal probe reproduced the documented null-call diagnostic with `SIGABRT`;
  it is retained as a limitation, not counted as a passing sanitizer test.
- All 56 byte comparisons, three decoded-image pairs, 22 conformance fixtures
  per revision, and 24 retained changed-image comparisons passed.
- All 10 alternating timing pairs and six pressure processes passed their raw
  output, timing, and resource checks. The full 189-pair/42-process matrix was
  not rerun, so these samples establish no new performance conclusion.

Evidence is retained locally in `build/resident-qualification-reproducible/`.
The 637-file seal has SHA-256
`e44e53e9e17744658e04ae5f9a148edae929eec67c333b93823af03bbd5cd491`
for `validation.json`. Repeating `performance` launched no commands and left raw
logs unchanged. A subsequent `verify` changed no file modification times across
3,818 retained files. Separate extraction checks matched the original and
standalone drivers' source/output signatures for small, padded 4K, and mixed
cohort workloads on both revisions; the documented synthetic-export recipe also
reproduced both canonical hashes.
