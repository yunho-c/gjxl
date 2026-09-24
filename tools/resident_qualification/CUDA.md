# CUDA driver

CUDA benchmark builds provide `gjxl_cuda_qualification` from the same standalone
driver used by the Metal qualification package. This target runs the current
build; it does not reconstruct or certify two committed revisions. The existing
Metal controller and its frozen-source protocol remain separate.

For example, after building with `GJXL_ENABLE_CUDA=ON` and
`GJXL_BUILD_BENCHMARKS=ON`:

```text
gjxl_cuda_qualification --synthetic 3839x2159 --count 7 --limit tight --cpu-limit 3
gjxl_cuda_qualification --synthetic 129x133 --synthetic 257x263 --batch 4 --in-flight 2 --callers 2 --count 7 --limit full
```

Use `--input image.pfm` for photographic inputs. The driver alternates original
and changed pixels, records output sizes/hashes, and verifies domain invariants
after each call and after shutdown/trim. `--effort N` selects effort (default 7).
`--dc-policy legacy` selects ordinary rounding, gradient prediction and disabled
smoothing in the integrated CUDA build, for comparisons against pre-integration
CUDA. It does not undo other shared writer or effort policies.

Timing surrounds complete public encode or batch calls. Backend setup is
reported separately. Sample zero is the first encode after setup; report it
separately from warm samples. Multi-caller cohort timing includes thread release
through joining both callers. For performance comparisons, alternate independent
processes and keep compiler/test/encoder activity quiescent. Preserve all samples,
source and executable hashes, selected policies, and decoded quality evidence.

Memory fields have different scopes:

- Windows `footprint` fields are current/peak process working set. Linux uses
  current/peak RSS. macOS retains the existing physical-footprint measurements.
- `managed` fields describe the library's accounted backing and reservations;
  they are not a physical-memory limit.
- CUDA `device_used_bytes` is total minus free from `cudaMemGetInfo`, sampled
  outside timed calls. It includes other processes and driver allocations,
  and is neither a per-process measurement nor a peak-memory measurement.

Record the initial, output-retained and post-trim samples together. Do not infer
device peaks from endpoint snapshots or compare host-memory field meanings
across operating systems as if they were the same metric.
