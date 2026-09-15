# Native DC integer-mapping qualification

`qualify.py` is an explicitly invoked local benchmark, using the retained
September 14 65-image corpus and original seven distances. Run it from the
repository root. It does not download inputs or silently create missing data.

Build the unmodified baseline before editing source, and keep its executable
and Metal library immutable. Build the candidate in a separate Release
directory. The default paths are `build/dc-uint-baseline/` and
`build/dc-uint-candidate/`; both build `gjxl_quality_benchmark` with
`GJXL_BUILD_BENCHMARKS=ON`. The baseline for this study is `b1fbfc1` plus the
research-only commit `6a8f8fd`.

```sh
python3 tools/dc_uint_search/qualify.py rate
python3 tools/dc_uint_search/qualify.py timing
python3 tools/dc_uint_search/qualify.py report
```

The final single-model implementation uses
`--root build/dc-uint-fast-qualification-20260914`
and `--candidate build/dc-uint-fast-candidate/gjxl_quality_benchmark` on all
three commands. The default directories retain the completed first
implementation. Its live source hashes now differ; its completed audit and
`source-snapshot.zip` preserve the original evidence.

Collection is resumable and locks its output directory. `rate --limit 14`
is a bounded smoke run; rerun `rate` without the limit to finish. The final
report rejects partial coverage. A frozen manifest hashes source files,
executables, Metal libraries, input pixels, decoder and collection code.
Change `--root` for a different implementation; never overwrite a frozen run.

Every fresh baseline must reproduce the original codestream hash. Every
candidate must reproduce the original decoded linear-RGB float hash before
reusing its measured quality score. Temporary decoded images are removed
only after verification; codestreams, raw samples and command logs remain.
The no-extrapolation report integrates log(bytes) over SSIMULACRA2 75–85
with PCHIP and Akima, then averages per-image percentages.

Timing is a separate action: 12 declared inputs, original Q80 distances,
four alternating baseline/candidate process pairs, two warmups and five
complete-encode samples per process. Run after correctness collection and
with other expensive workloads idle. A successful timing run does not by
itself establish an uncontended machine or qualify a latency claim.
