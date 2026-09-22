# Investigation incidents

> Historical study checkpoint. The final disposition and commit status are in [REPORT.md](REPORT.md) and [README.md](README.md). Any pending work below refers to the time of that checkpoint.

- The initial ordinary/stage harness compilation lacked the internal profiling namespace import. Compilation failed; the import was added before any harness measurement. Both compiler logs remain available.
- The first rolling integration failed the Butteraugli test at backend creation. A diagnostic encode reported that the rolling function was absent from the embedded library. `shutil.copy2` had preserved an older generated source timestamp when replacing a newer geometry source, so Ninja did not rebuild the shader. This was a build-provenance failure, not a numerical failure. The failed attempt is retained. The integration tool now writes shader bytes with a fresh timestamp and uses a new attempt directory. No timing or correctness conclusion uses the failed binary.
- The first integrated stage invocation was rejected by the benchmark argument parser, which deliberately disallows combining raw workflow samples and GPU profile output. No encode occurred. The investigation harness now derives its own complete-call sample filename from the GPU profile directory in stage mode; the ordinary and stage command interfaces remain separate. Version-2 binaries/campaign directories preserve the failed attempt and earlier ordinary measurements.

- New short-blur prototype initially used Metal reserved word `thread` as a local name. Compilation failed before execution. Renamed it to `thread_index`; failed source/command/diagnostic retained in `probe-builds-v2/`.

- Expanded-contract build hit an inherited C API `-Werror,-Wunused-parameter` failure in `src/gpu/ops/gpu_execution_profile_internal.h` at lines 322-324. Header bytes match frozen baseline `4f3e414`. Preserve the failed build log/identity; `workflow_admission` is unavailable in this configuration. Continue the seven relevant Metal/resident tests separately without weakening warnings or numerical tolerances.

## Closure-screen attribute placement

The first closure-screen compile failed because the Metal thread-limit attribute followed the parameter list and was parsed as a type attribute. It was moved before `kernel void`, following the declaration form; the failed source/log and state remain under `phase8-closure/` and `campaign8-first-failed-state.json`. No timing ran from that failed build. Apple documents the shader-side thread limit at https://developer.apple.com/documentation/metal/mtlcomputepipelinedescriptor/maxtotalthreadsperthreadgroup .

Phase 13 generator initially stopped on a source-match assertion before writing any candidate files: the medium plane declaration also occurs in the original high-stage loop. Restricted the replacement to the low/medium producer scope and reran successfully. No build or timing used the failed generation.

Phase 17 generation initially stopped before writing sources because a naive closing-brace search ended inside the generated filter core. Corrected the boundary to the high kernel's final brace before the shared helper. Both finalist source sets then generated with explicit source/geometry assertions; no failed-generation code was compiled or measured.
