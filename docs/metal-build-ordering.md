# Single-producer Metal build graph

This build-only checkpoint follows `eda2ee5`. It closes the shader embedding
race found while preparing the integrated resident-execution baseline; it does
not change encoder code, shader arithmetic or scheduling policy.

## Cause and fix

The raw shader target and embedded-library target both reached the custom
command producing `gjxl.metallib` through file dependencies. Recursive Make
could therefore run independent copies of that rule in parallel. The embedding
consumer could read a file that another producer was still writing. The
[original qualification](resident-cpu-coordination.md#integrated-baseline-preparation-and-embedding-validation)
retains a linked payload containing only 151,552 of the complete 1,489,228 bytes.

Both embedding and optional profiling-symbol generation now depend on the
`gjxl_metal_shaders` target **and** the library file. The target dependency
orders consumers after the sole producer; the file dependency preserves
incremental regeneration when shader bytes change. A target-only dependency
would fix ordering but could leave consumers stale after a later shader edit.

## Permanent regression tests

`metal_build_graph` configures the actual production CMake project with an
injected consumer target and a deliberately slow fake Metal toolchain. The
fixture rejects concurrent producers and publishes a partial library before
completing it. It checks Make and, when installed, Ninja, with profiling off/on:

- Initial parallel build: exactly one link and complete embedded bytes/symbols.
- Unchanged rebuild: no extra link.
- Changed IR input: exactly one new link and refreshed bytes/symbols.

The test never edits source shaders. Before the fix, its Make case fails with
`Concurrent producers for gjxl.metallib`; after the fix all four cases pass.
This is a build-graph test, not a claim to emulate a Metal compiler.

`metal_embedded_library` separately compares the real linked
`EmbeddedMetalLibrary()` span against the complete on-disk Metal library. It
detects a truncated embedding even if the generated include compiles cleanly.

## Reproduction and evidence

From this worktree:

```sh
python3 tests/metal_build_graph_test.py --source .
cmake -S . -B build/metal-build-ordering -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD_TESTS=ON \
  -DGJXL_BUILD_BENCHMARKS=ON -DGJXL_ENABLE_LIBJXL_REFERENCE=OFF \
  -DGJXL_ENABLE_METAL_PROFILING=OFF
cmake --build build/metal-build-ordering --parallel 8
ctest --test-dir build/metal-build-ordering --output-on-failure
```

Retained graph logs are under `build/metal-build-graph-before-fixed-fixture/`
and `build/metal-build-graph-after/`. Real Release build/test logs use the
`build/metal-build-ordering-*` and `build/metal-build-ordering-ninja-*` prefixes.
The fresh Make build logs one library link. The fresh Ninja build also passes
the real linked-payload test. Both use Apple Clang 17 on the M4 Pro configuration
recorded in the CPU checkpoint, with the optional libjxl reference disabled.

The fresh Make Release suite passes 100/101 tests. Its sole failure is the
previously reproduced `quantization_pipeline` golden mismatch: actual
`0.24919039011001587`, expected `0.24914586544036865`. This is not an all-green
suite or a new regression attributed to the build graph.

All 60 codec, codestream and Metal runtime objects in the Make build are identical
to its qualified `build/cpu-coordination` counterpart. Archive container hashes
differ; they are not evidence of changed runtime objects. The shader SHA-256
remains `9dcbc4dff15fd81c0793a1df3d68fd06ac821d0f34929a0e397f1de4e9cafae4`.
No new performance claim is made for this dependency-only repair.
