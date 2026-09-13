# Image batch benchmark

Build and run the existing small/1080p/4K synthetic matrix:

```sh
just image-batch-benchmark all 1,2,4
```

Defaults are Metal, **fully-resident**, distance 1.2, effort 7, one warmup and
three paired samples. `--batch-sizes auto` (also the Just default) uses 1,2,4,8
below 12 million pixels and **1** at or above that size, including external
PFMs. An explicit numeric batch list overrides this choice. Other existing AQ
modes and synthetic workload names remain supported.

## Photographic corpus: `photo-large`

Use `photo-large` for photographic comparisons. It reuses the existing Unsplash
corpus: **three source photographs, each prepared at approximately 12, 24, and
48 MP** (nine inputs). These are Lanczos downscales in linear RGB from 50–58 MP
JPEG originals, preserving aspect ratio with rounded dimensions; none is
upscaled. They are not nine independent scenes or native captures at each size.
Keep their results separate from the synthetic stress cases below.

The small [corpus manifest](../tools/benchmark_corpora/photo-large.json) records
source URLs, photographers, preparation arguments, dimensions, and SHA-256
hashes of both originals and prepared PFMs. Image files stay outside Git.

Inspect and verify all nine inputs without loading an encoder or encoding:

```sh
python3 tools/benchmark_image_batch.py --workload photo-large --dry-run
```

When ready to measure:

```sh
just image-batch-benchmark photo-large
just image-batch-benchmark photo-large 1 3 1 metal fully-resident \
  --raw-samples photo-large.csv
```

The preset explicitly defaults to **batch size 1 for every photograph**, including
the three “12MP” inputs that round slightly below 12 million pixels. Pass an
explicit list such as `1,2` to test concurrency. All nine hashes are checked before
encoding; missing or modified files stop the run. Hashing is outside timing.
With `--raw-samples NEW.csv`, the wrapper also writes `NEW.csv.inputs.json`,
recording corpus IDs, size classes, input paths/hashes, the manifest hash, and the
benchmark command/binary hash. Both output paths must be new. The sidecar records
selected inputs, not run completion; consult the raw rows and exit status.

By default, the wrapper reuses the existing sibling directory
`../libjxl-runtime-study-2026-09-03/corpus` when present. Otherwise it uses the
ignored cache `build/benchmark-corpus/photo-large`. Set `GJXL_PHOTO_CORPUS` or
pass `--photo-root CORPUS` to select another root containing `pfm/unsplash/...`.
If the corpus is missing, explicitly prepare it once:

```sh
python3 tools/benchmark_image_batch.py --prepare-photos \
  --photo-root build/benchmark-corpus/photo-large
```

Preparation downloads only missing originals, resizes only missing PFMs, and
exits without encoding. It requires ImageMagick and network access when those
files are missing; reusing all nine verified PFMs requires neither. Existing
files are never replaced. The manifest pins the original ImageMagick version;
a conversion producing different bytes is rejected. The PFMs occupy about 3 GB.
Use the same `--photo-root` (or environment variable) when running the benchmark.

The same wrapper can select identical inputs for the libjxl-side batch benchmark:

```sh
python3 tools/benchmark_image_batch.py --workload photo-large \
  --benchmark ~/GitHub/libjxl/build-release-codex/tools/jxl_image_batch_benchmark \
  --raw-samples libjxl-photo-large.csv
```

The wrapper forwards other encoder options unchanged. Report each encoder's
settings and thread policy; equal requested distance is not matched decoded
quality. Neither command builds or invokes the other encoder.

## Synthetic stress corpus: `synthetic-large`

The large synthetic corpus is opt-in; `all` retains the original five workloads
through 4K. It uses the same deterministic gradient/texture pattern.

The pattern is defined by `FillBatchTexture` in
[`benchmarks/synthetic_images.h`](../benchmarks/synthetic_images.h). That header
also contains the distinct `FillEncodingStress` pattern used by the encoding
benchmark and resident qualification. Sharing the helper preserves each corpus's
existing pixels; the two patterns remain separate workloads.

| Workload | Dimensions | Pixels |
| --- | --- | ---: |
| `synthetic-12mp` | 4000x3000 | 12,000,000 |
| `synthetic-24mp` | 6000x4000 | 24,000,000 |
| `synthetic-48mp` | 8000x6000 | 48,000,000 |

Inspect the selected cases without allocating images, initializing Metal, or
encoding anything:

```sh
build/release/gjxl_image_batch_benchmark --workload synthetic-large --dry-run
```

When ready to measure, select one size or `synthetic-large` for all three:

```sh
just image-batch-benchmark synthetic-12mp
just image-batch-benchmark synthetic-large
```

Both commands default to batch size 1. To deliberately test concurrency, pass
the batch list, for example `just image-batch-benchmark synthetic-12mp 1,2`. Each concurrent
image requires its own encoder working storage. Single-image success does not
establish that a larger batch fits in memory, and the default execution domain
has no hard managed-memory cap. These fixtures remain useful as deterministic
size stress cases. The old names `large`, `12mp`, `24mp`, and `48mp` remain aliases;
new CSV workload names explicitly say `synthetic-`. Generated pixels and exported
filenames are unchanged. Historical results using the old built-in names describe
these synthetic fixtures, not the photographic corpus.

To prepare the **same inputs for both encoders**, export the corpus once:

```sh
build/release/gjxl_image_batch_benchmark --workload synthetic-large \
  --export-inputs /path/to/new-large-pfms
```

Export writes unit-scale, interleaved float32 linear-RGB PFMs and performs no
encoding. It requires a new output directory, generates one image at a time,
and may leave partial files on failure. Payloads are 144, 288, and 576 MB
(1,008 MB total, decimal). The native executable's `--dry-run` and `--export-inputs`
are synthetic-only modes and cannot be combined with each other, `--input`, or
`--raw-samples`. The Python wrapper additionally supports photographic previews.
Neither exporting nor adding the workload definitions runs a benchmark.

Later, use those PFMs for both tools (each raw CSV destination must be new):

```sh
build/release/gjxl_image_batch_benchmark \
  --input /path/to/new-large-pfms --batch-sizes 1 --raw-samples gjxl-large.csv
~/GitHub/libjxl/build-release-codex/tools/jxl_image_batch_benchmark \
  --input /path/to/new-large-pfms --batch-sizes 1 --raw-samples libjxl-large.csv
```

## External images and measurements

Use the built executable to measure real images and save raw samples:

```sh
build/release/gjxl_image_batch_benchmark \
  --input /path/to/pfms --batch-sizes 1,2,4 --raw-samples run.csv
```

`--input` accepts an RGB PFM file or a directory, and can be repeated. Directory
selection is non-recursive and includes `.pfm` files case-insensitively. Inputs
are sorted and deduplicated by canonical path. They replace synthetic workload
selection and retain their original dimensions. PFM pixels must be finite
linear RGB. Direct `--input` selection requires no corpus manifest and performs
no downloading or resizing.

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
`VarDctBatchEncoder`. Both paths request the same automatic per-image CPU policy
and share the default execution domain, which caps aggregate CPU participation
at hardware concurrency (between 1 and 256). Increasing batch concurrency can
reduce the CPU workers available to each image. See
[Resident execution](resident-execution.md) for the shared admission policy.
Every output and deterministic summary must match the single-image reference.

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

The `image_queue_*`, `image_service_*`, and `image_ready_*` columns summarize
per-image wall spans pooled across measured batched calls, excluding warmups.
Queue ends at initial CPU admission; service runs from admission through internal
result retention, and ready is their sum. Results become publicly available when
the whole batch returns. These spans are separate from the throughput-derived
`batch_ms_per_image` value.

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
