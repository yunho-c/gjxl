# DC behavior at effort 4 and above

This study qualifies the existing prediction-aware DC quantization and adaptive
DC smoothing controls as a candidate automatic policy from effort 4 upward.
It does not change encoder defaults.

- [REPORT.md](REPORT.md): interpretation and recommendation.
- [MEASUREMENTS.md](MEASUREMENTS.md): generated tables and explicit completion counts.
- [PROTOCOL.md](PROTOCOL.md): prospective scope and measurement boundaries.
- CSVs and `summary.json`: retained-data analysis.

Raw photographic and timing evidence: `build/dc-e4-qualification-20260913-v1`.
Corrected synthetic evidence: `build/dc-e4-synthetic-corrected-20260914-v1`.
The raw directory retains a clean main source export, fresh Release build,
source/tool/input fingerprints, seven focused test results, all encoded streams,
independent decoder/scorer logs, machine-load records, and visual review images.
External libjxl executables and libraries are copied into the run and their
actual library loading paths audited. Source revision: `8e8c49a`.

## Reproduce analysis

```sh
python3 tools/dc_coding/effort_report.py report \
  --output build/dc-e4-qualification-20260913-v1
```

Reporting reads saved measurements only. Open [review.html](review.html) for
the combined visual gallery. It offers same-distance mode switching, native-size viewing,
matched-size candidates where calibration succeeded, and amplified errors.
Full review images use 16-bit sRGB PNG; numeric analysis uses float decoder output.

## Explicit collection commands

These commands launch codecs. Use a newly initialized `OUTPUT` directory;
they resume completed cases without resetting budgets. Only one collector may
run at a time. Build, collection, scoring, and timing must not overlap. The
fixed same-distance timing phase is not a matched-quality speed comparison.

```sh
python3 tools/dc_coding/effort_qualification.py collect --output OUTPUT
python3 tools/dc_coding/effort_qualification.py calibrate --output OUTPUT
python3 tools/dc_coding/effort_qualification.py timing --output OUTPUT
python3 tools/dc_coding/effort_report.py review --output OUTPUT
```

For a new run, first export the desired Git revision and Metal C++ headers into
`OUTPUT/source`, record `source-identity.json`, and make a fresh Release build
with benchmarks enabled and Metal profiling disabled. Copy the external tools
and libraries with verified loader paths. Add a prospective `PROTOCOL.md` and
run `effort_qualification.py init --output OUTPUT --previous PHOTO_STUDY_DIR`.
The prior photographic manifest supplies pinned input and scorer identities;
its performance observations are not pooled into this run.

The original run is closed: its collector was archived after identity verification.
All four original synthetic fixtures are excluded because their ranges did not
match the intended formulas. `visual_fixtures.py` now generates them using explicit
NumPy output buffers. The supplement was initialized with `--synthetic-only` and
independent scalar-reference/range checks. Its 36 cases replace the original 36
synthetic cases; the report automatically follows `synthetic-supplement.json`.
The supplement has no timing cases. Completed original photographic data remain
unchanged. Gallery images stay in the two raw directories; the generated HTML
links to those local artifacts.

## Why this qualification exists

libjxl's [DC precision discussion](https://github.com/libjxl/libjxl/pull/1762)
emphasizes subtle gradient and chroma artifacts requiring manual inspection.
The JPEG XL authors explain adaptive LF smoothing's anti-banding purpose in
[section 6.2 of the design paper](https://arxiv.org/html/2506.05987v1).
That motivates checking visual behavior as well as rate at matched metric scores.
