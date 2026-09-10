# Image batch benchmark

Build and run the existing small/1080p/4K synthetic matrix:

```sh
just image-batch-benchmark all 1,2,4
```

Defaults are Metal, **fully-resident**, distance 1.2, effort 7, one warmup and
three paired samples. The full default batch list is 1,2,4,8. Other existing
AQ modes and synthetic workload names remain supported.

Use the built executable to measure real images and save raw samples:

```sh
build/release/gjxl_image_batch_benchmark \
  --input /path/to/pfms --batch-sizes 1,2,4 --raw-samples run.csv
```

`--input` accepts an RGB PFM file or a directory, and can be repeated. Directory
selection is non-recursive and includes `.pfm` files case-insensitively. Inputs
are sorted and deduplicated by canonical path. They replace synthetic workload
selection and retain their original dimensions. PFM pixels must be finite
linear RGB. No downloading, resizing, or new asset registry is involved.

For a normal still image, reuse the existing optional ImageMagick wrapper:

```sh
python3 tools/benchmark_encoding_image.py \
  --benchmark build/release/gjxl_image_batch_benchmark photo.png -- \
  --batch-sizes 1,2,4 --raw-samples photo.csv
```

The wrapper converts once before timing. Its temporary PFM path is recorded as
the source; retain the command/original filename alongside the CSV for later
identification. To forward flags through Just, supply its six positional options
first, for example:

```sh
just image-batch-benchmark all 1,2,4 3 1 metal fully-resident \
  --input /path/to/image.pfm --raw-samples run.csv
```

Each image is measured separately. For batch size N, both paths encode **N
copies of that image**: one persistent worker versus N persistent workers using
`VarDctBatchEncoder`. Automatic per-image CPU threading is identical in both
paths; increasing batch size does not divide a fixed host thread budget. Every
output and deterministic summary must match the single-image reference.

Timing starts with preloaded planar linear RGB and ends with in-memory
codestreams. It includes batch scheduling and the complete encoding workflow,
including CPU preparation and serialization. Image loading/conversion, worker
construction, warmups, and output validation are outside the timed intervals.
Sample order alternates serial-first and batch-first.

The terminal summary retains its existing CSV columns. `batch_median_ms` is for
N copies, `batch_ms_per_image` divides that median by N, and
`batch_images_per_second` is N times 1000 divided by that median. Speedups are
paired serial/batch ratios. Their minimum and maximum describe samples within
**one process**, not independent-process medians or confidence bounds.

`--raw-samples NEW.csv` writes a standalone CSV with one row per validated pair:
`codec`, `workload`, `source`, `width`, `height`, `batch_size`, `sample`, `order`,
`requested_backend`, `backend`, `aq_mode`, `distance`, `effort`, `thread_policy`,
`timing_boundary`, `serial_ns`, `batch_ns`, and `encoded_bytes_per_image`.
Sample indices start at zero; times are integer nanoseconds. The actual backend
is recorded separately from its requested selection. CPU rows use `n/a` for AQ
mode. Synthetic sources are empty; external workload/source fields contain the
canonical input path and are CSV-escaped. Byte counts are per image.

The raw destination must not exist and its parent must already exist. Rows are
flushed as pairs complete; a failed run returns nonzero and may leave partial
results. Use a new filename for each independent run. For downstream comparisons,
match workloads and timing boundaries, and report settings and thread policies
explicitly.
The separate libjxl-side `jxl_image_batch_benchmark` emits the same raw schema;
see `doc/image-batch-benchmark.md` in the libjxl checkout. Run both tools on the
same prepared PFMs with unit scale (`+1` or `-1`). Its fixed per-image thread
setting is recorded as `fixed_per_image:N`; report that alongside GJXL's
automatic policy. Equal requested effort and distance describe a configuration
comparison, not matched decoded quality. Neither command builds or invokes the
other encoder, and paired speedups describe batching within each encoder.
