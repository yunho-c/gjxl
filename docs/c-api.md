# Experimental C API and ABI

This document describes GJXL's implemented C interface, which is deliberately
small. GJXL owns a safe wrapper in `rust/gjxl`; Slimg is expected to be its first
downstream consumer, with OIMG using that wrapper indirectly. The interface is
experimental: its layouts are designed for future ABI evolution, but
compatibility is not promised until the API has been implemented, integrated,
and versioned as stable.

The guiding principle is:

> The CLI may expose convenience and diagnostic controls; the C API exposes
> only canonical encoding concepts.

The encoder options contain perceptual distance, effort, and an independent
automatic/maximum entropy-search control. Execution policy belongs to a
reusable context, and pixel memory belongs to a separate non-owning image view.

## Current implementation boundary

The API must accurately describe the encoder that exists today. The
current public C++ workflow in [`workflow.h`](../src/codestream/workflow.h):

- accepts three planar float32 channels in linear sRGB;
- encodes one regular, final, 4:4:4 VarDCT frame;
- returns a raw JPEG XL codestream, not an ISO BMFF container;
- has no alpha or other extra channels;
- has no lossless or Modular-only image mode;
- has no progressive or multi-pass mode; and
- performs tokenization, entropy coding, and codestream assembly on the CPU.

The canonical workflow now defaults to fully resident Metal when automatic
selection qualifies, while the exact-coefficient path remains an explicit
reference/compatibility implementation. Metal AQ implementation selection is
deliberately not exposed through the initial C interface; C and Rust callers
inherit the canonical workflow policy without growing their stable compression
contract. The full profile boundary is documented in
[`codestream.md`](codestream.md).

The existing workflow already caches the production Metal backend for the
process lifetime. The C context should reuse that cache rather than creating a
Metal device and loading pipelines for every image.

## Goals

The first interface should:

1. be usable from C, Rust, Dart FFI, Python, and other C-compatible runtimes;
2. accept Slimg's interleaved RGB/RGBA byte representation without requiring
   callers to construct planar float images;
3. make distance, effort, and explicit maximum compression the stable
   compression controls;
4. keep CPU/GPU execution policy separate from bitstream options;
5. return one library-owned contiguous output buffer;
6. report unsupported capabilities distinctly from malformed input;
7. prevent C++ exceptions and types from crossing the ABI; and
8. preserve atomic caller-visible output on failure.

## Non-goals for the first version

The first version does not expose:

- lossless or Modular encoding;
- alpha compression or `alpha_distance`;
- containers, Exif, XMP, ICC profiles, previews, or thumbnails;
- progressive or responsive coding;
- faster-decoding controls;
- target-byte, target-BPP, or maximum-error rate control;
- Metal AQ implementation selection;
- device indexes;
- streaming input or output;
- stage-specific profiling data; or
- internal codec heuristics such as AQ strength, AC-search depth, transform
  implementations, entropy clusters, or GPU dispatch dimensions.

These can be added later through appended sized structs or separate entry
points. The maximum-compression field demonstrates that versioning contract.

## Public header

The installed header lives at `include/gjxl/gjxl.h`. The following is its
abridged shape rather than a complete export-macro definition:

```c
#ifndef GJXL_GJXL_H_
#define GJXL_GJXL_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GJXLContext GJXLContext;

typedef int32_t GJXLResult;
enum {
    GJXL_OK = 0,
    GJXL_ERROR_INVALID_ARGUMENT = 1,
    GJXL_ERROR_UNSUPPORTED = 2,
    GJXL_ERROR_UNAVAILABLE = 3,
    GJXL_ERROR_OUT_OF_MEMORY = 4,
    GJXL_ERROR_BACKEND = 5,
    GJXL_ERROR_INTERNAL = 6,
};

typedef int32_t GJXLBackend;
enum {
    GJXL_BACKEND_AUTO = 0,
    GJXL_BACKEND_CPU = 1,
    GJXL_BACKEND_METAL = 2,
    GJXL_BACKEND_CUDA = 3,
};

enum {
    GJXL_MAX_CPU_THREADS = 256,
};

typedef struct {
    uint32_t struct_size;
    GJXLBackend backend;
    uint32_t num_cpu_threads;
} GJXLContextOptions;

typedef int32_t GJXLCompressionMode;
enum {
    GJXL_COMPRESSION_AUTOMATIC = 0,
    GJXL_COMPRESSION_MAXIMUM = 1,
};

typedef struct {
    uint32_t struct_size;
    float distance;
    int32_t effort;
    GJXLCompressionMode compression_mode;
} GJXLEncoderOptions;

typedef int32_t GJXLPixelFormat;
enum {
    GJXL_PIXEL_FORMAT_RGB8_SRGB = 1,
    GJXL_PIXEL_FORMAT_RGBA8_SRGB = 2,
};

typedef struct {
    uint32_t struct_size;
    uint32_t width;
    uint32_t height;
    GJXLPixelFormat pixel_format;
    const void* pixels;
    size_t pixels_size;
    size_t row_stride_bytes;
} GJXLImageView;

typedef struct {
    uint8_t* data;
    size_t size;
} GJXLBuffer;

GJXLResult gjxl_context_options_init(
    GJXLContextOptions* options, size_t caller_size);

GJXLResult gjxl_encoder_options_init(
    GJXLEncoderOptions* options, size_t caller_size);

GJXLResult gjxl_context_create(
    const GJXLContextOptions* options, GJXLContext** context);

void gjxl_context_destroy(GJXLContext* context);

GJXLResult gjxl_encode(
    GJXLContext* context,
    const GJXLImageView* image,
    const GJXLEncoderOptions* options,
    GJXLBuffer* output);

void gjxl_buffer_free(GJXLBuffer* buffer);

const char* gjxl_get_last_error(void);

float gjxl_distance_from_quality(float quality);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GJXL_GJXL_H_
```

The final header also needs a platform-specific `GJXL_API` export macro on
public functions. Visibility should be hidden by default so that a shared
library exports only the intended `gjxl_*` entry points.

## ABI rules

### Sized structs

`struct_size` is the first field of every extensible input struct. Initializer
functions receive the caller's allocation size explicitly:

```c
GJXLEncoderOptions options;
if (gjxl_encoder_options_init(&options, sizeof(options)) != GJXL_OK) {
    /* handle error */
}
```

A one-argument initializer is unsafe across library/header versions: a newer
library could write a larger struct into storage allocated using an older
header. The size-taking initializer may write only the bytes provided by the
caller. Future fields should reserve zero as their default whenever possible.

The encoder must:

- reject a struct smaller than the minimum version it can understand;
- read only fields covered by `struct_size`;
- ignore an unknown trailing portion supplied by a newer caller; and
- never require callers to reproduce compiler-specific padding manually.

`GJXLImageView` has no meaningful default initializer. Callers zero-initialize
it, set `struct_size = sizeof(image)`, and then fill every required field.

### Fixed ABI types

ABI-visible options and result values use fixed-width integers rather than C
`bool` or enum-typed struct fields. The named enum blocks above define integer
constants; functions and structs carry `int32_t` typedefs. Pointer-sized byte
counts use `size_t`.

The implementation is compiled as C++ but exports functions with C linkage.
Every entry point catches `std::bad_alloc`, other standard exceptions, and
unknown exceptions. No exception, `std::string`, `std::vector`, C++ class, or
C++ standard-library allocator crosses the boundary.

### Versioning

The experimental library should report an API/ABI version before it is shipped
as a shared dependency. Fields are appended rather than reordered or reused.
Result-code values and pixel-format values are never renumbered once released.

The initial Slimg/OIMG experiment may link a static `gjxl::c` target to avoid
application-bundle and code-signing complexity. A shared `libgjxl` should be
added from the same implementation after the interface has settled enough to
assign an `SOVERSION` and enforce its exported-symbol list.

## Encoder options

`gjxl_encoder_options_init` produces:

```text
distance = 1.0
effort   = 7
compression_mode = GJXL_COMPRESSION_AUTOMATIC
```

### Distance

`distance` is the sole canonical quality control. Lower values request higher
quality and generally produce larger output. The first API accepts finite
values in `(0, 25]`, with the practically useful range around `0.5` to `3.0`.

Distance zero is a meaningful JPEG XL request but is not implemented by GJXL's
VarDCT-only profile. It returns `GJXL_ERROR_UNSUPPORTED`, not
`GJXL_ERROR_INVALID_ARGUMENT`. A future lossless Modular encoder may give zero
its normal lossless meaning without adding another quality field.

Quality remains a caller or CLI convenience:

```c
options.distance = gjxl_distance_from_quality(90.0f);
```

The helper should reproduce libjxl's public quality-to-distance mapping and
document quality input in `[0, 100]`. The pinned implementation is
[`JxlEncoderDistanceFromQuality`](../third_party/libjxl/lib/jxl/encode.cc).
Quality 100 maps to distance zero, so a current caller must fall back to
another encoder after the encode call returns `GJXL_ERROR_UNSUPPORTED`. A
reverse helper is deferred until a concrete consumer requires it.

### Effort

Effort controls both the established AQ update policy and the resolved entropy
behavior:

| Effort | AQ updates | Entropy behavior |
| ---: | ---: | --- |
| 1-3 | 0 | Balanced |
| 4-6 | 1 | Balanced |
| 7 | 2 | Balanced default |
| 8 | 3 | Balanced |
| 9 | 3 | High density |
| 10 | 4 | High density |

The numbers communicate the same user intent as `cjxl`, not identical
algorithms or identical output. In particular, effort 10 does not select the
exhaustive former serializer; that requires the independent compression mode.
The CLI's `--high-density` compatibility override continues to request four AQ
updates while resolving to the same high-density entropy behavior as efforts
9-10.

### Compression mode

`GJXL_COMPRESSION_AUTOMATIC` resolves entropy search from effort. Balanced
efforts commit one block-context map and coefficient-order representation and
use effort-7-like direct ANS construction. Efforts 9-10 use the measured
effort-9-like high-density policy.

`GJXL_COMPRESSION_MAXIMUM` preserves the former exhaustive serializer policy
as an explicit opt-in. It changes only entropy/codestream search and does not
silently change distance, AQ updates, backend choice, or CPU thread limits.

`compression_mode` was appended after the original 12-byte encoder-options
layout. The implementation accepts that V1 size, defaults its absent field to
automatic, and reads the field only when `struct_size` covers all 16 current
bytes. Exact-size and larger caller allocations are tested in both C and C++;
an unknown mode returns `GJXL_ERROR_INVALID_ARGUMENT` without changing output.

## Context and backend selection

Execution choices are not bitstream properties. The initial context options
therefore contain one field:

```text
AUTO   use the existing qualified automatic policy
CPU    require the CPU workflow
METAL  require the default fully resident Metal workflow
CUDA   require the default fully resident CUDA workflow
```

`AUTO` retains the current source-backed Metal policy, including its device,
geometry, and distance gates. CUDA remains opt-in until it has its own measured
qualification envelope. `METAL` and `CUDA` are explicit unqualified overrides;
backend creation or capability failure returns an error rather than silently
falling back. GPU AQ implementation selection remains internal.

`gjxl_context_options_init` defaults to `AUTO`, and passing `NULL` options to
`gjxl_context_create` has the same meaning. Slimg's default quality 80 maps to
distance 1.9, outside the current automatic Metal interval of `[1.0, 1.2]`.
An experiment that specifically intends to exercise a GPU must therefore
request `METAL` or `CUDA` explicitly rather than treating `AUTO` as a general
GPU mode.

Process-cached production GPU backends should be shared by contexts. Forced
Metal or CUDA context creation resolves the requested backend eagerly so an
unavailable device or missing workflow capability is reported before the first
encode. Automatic contexts may resolve Metal lazily only when an image is
eligible. CPU contexts initialize neither GPU backend.

The wrapper stores immutable execution configuration so `gjxl_encode` can be
called concurrently on one context. Destroying a context must not overlap an
active call using it.

`num_cpu_threads` limits the participating host CPU threads in each encode.
Zero retains the existing automatic, stage-specific worker policy; one runs
CPU work serially; larger values allow the caller plus at most `N - 1`
background workers. The current maximum explicit value is 256. An explicit
budget also suppresses nested worker fan-out, so a parallel codestream task
does not create another pool beneath itself.

The limit is per encode, not a process-wide pool or a cap across concurrent
calls. It does not constrain Metal threadgroups, although host preparation and
codestream assembly remain within the CPU budget. The field is an appended
sized-struct tail: callers built against the original prefix retain automatic
behavior.

Device indexes remain omitted because the current factory uses the
system-default Metal device.

## Image-view contract

The first consumer path uses interleaved 8-bit pixels:

- `GJXL_PIXEL_FORMAT_RGB8_SRGB` has three bytes per pixel;
- `GJXL_PIXEL_FORMAT_RGBA8_SRGB` has four bytes per pixel; and
- rows may contain padding described by `row_stride_bytes`.

`pixels_size` allows the implementation to validate the last accessible byte
instead of trusting only a pointer and stride. Validation uses checked
arithmetic for row width, stride, the final-row offset, and total image area.
Zero dimensions, null pixels, short buffers, undersized strides, and
unrepresentable geometry return `GJXL_ERROR_INVALID_ARGUMENT` without reading
the buffer.

Byte samples are interpreted as nonlinear sRGB and converted to planar
float32 linear sRGB before entering `EncodeLinearRgbVarDctCodestream`. The
conversion belongs in private C-adapter code rather than the codec's canonical
linear-float workflow.

The initial codestream cannot carry alpha. RGBA input is accepted only if every
alpha byte is 255. Any non-opaque pixel returns `GJXL_ERROR_UNSUPPORTED`; alpha
must never be silently discarded or composited against an undocumented
background. Slimg may then fall back to libjxl for that image.

ICC profiles and non-sRGB color descriptions are not represented by this
view. A caller requesting profile preservation must use another encoder.

## Output ownership and atomicity

Callers pass an empty output:

```c
GJXLBuffer output = {0};
GJXLResult result = gjxl_encode(context, &image, &options, &output);
```

`gjxl_encode` rejects a nonempty output so it cannot overwrite and leak an
existing allocation. On success, `data` points to `size` contiguous bytes
owned by the GJXL library. On failure, the caller-visible buffer remains
unchanged.

`gjxl_buffer_free` releases memory using the same runtime that allocated it and
then resets both fields to zero. Passing `NULL` or an already-empty buffer is a
no-op. Callers must not use `free`, `delete[]`, or a language allocator on the
returned pointer.

Incremental `NEED_MORE_OUTPUT` processing is deliberately omitted. A separate
streaming API can be added later without complicating the high-level encode
call.

## Errors and diagnostics

The stable result code communicates the category. A thread-local diagnostic
string supplies detail:

```c
if (result != GJXL_OK) {
    fprintf(stderr, "gjxl: %s\n", gjxl_get_last_error());
}
```

The returned pointer remains valid until the next GJXL call on the same thread.
Thread-local storage avoids races between concurrent calls on one context.
Applications must branch on `GJXLResult`, not parse the diagnostic message.

Internal status translation should preserve these distinctions:

| C result | Meaning |
| --- | --- |
| `INVALID_ARGUMENT` | Malformed pointers, sizes, layouts, or option values |
| `UNSUPPORTED` | Valid concept outside the current profile, such as lossless or alpha |
| `UNAVAILABLE` | Requested execution backend cannot be created or selected |
| `OUT_OF_MEMORY` | Host or device allocation failure |
| `BACKEND` | Submission, completion, or device execution failure |
| `INTERNAL` | Invariant failure or unexpected exception |

## Completed implementation history

### 1. Add an internal effort policy

Files:

- `src/codestream/workflow.h`
- `src/codestream/workflow.cpp`
- `tools/gjxl_encode.cpp`
- `tests/codestream_workflow_test.cpp`

Add `effort = 7` to `VarDctEncodingOptions`, validate the range, and derive AQ
iterations in one helper. Keep the existing density mode temporarily for source
compatibility, with high density overriding to four updates. Add `--effort` to
the CLI and reject conflicting explicit `--effort`/`--high-density` requests.

Verify the effort resolver and expected score-history length on CPU and
supported Metal paths. The later entropy-alignment change deliberately replaced
the effort-7 default hash; maximum compression retains the former serializer
hash instead.

### 2. Add the private packed-pixel adapter

Files:

- `src/c_api/image_conversion.h`
- `src/c_api/image_conversion.cpp`
- `tests/c_api_image_conversion_test.cpp`

Implement checked RGB8/RGBA8 validation, opaque-alpha detection, the sRGB
transfer function, and conversion into `Image3FBuffer`. Test known transfer
points, padded rows, minimal images, overflow boundaries, short buffers, and
atomic failure.

### 3. Add the C implementation

Files:

- `include/gjxl/gjxl.h`
- `src/c_api/gjxl.cpp`
- `tests/c_api_test.c`
- `tests/c_api_cpp_test.cpp`

Implement the opaque context, sized initializers, quality helper, result-code
translation, exception barrier, thread-local diagnostics, encode adapter, and
buffer ownership. The C test must exercise the complete API without including
any C++ header. The C++ test verifies that the header compiles cleanly in both
languages and retains C linkage.

### 4. Add build and install support

Files:

- `CMakeLists.txt`
- `cmake/gjxlConfig.cmake.in`
- `cmake/test_installed_consumer.cmake`
- `tests/downstream/CMakeLists.txt`
- `tests/downstream/c_api_consumer.c`

Add a `gjxl::c` target linked privately to the existing codestream
implementation. Install only `include/gjxl/gjxl.h` as its public header; do not
make consumers depend on internal C++ include directories. Extend the installed
consumer test to configure, compile, link, and run a strict C client. Enable C
as a project language so that this test proves the header with a real C
compiler rather than compiling a nominal C example as C++.

Begin with a static target for the Slimg/OIMG experiment. Before shipping a
shared library, compile dependencies as position-independent code, apply hidden
visibility, assign an experimental ABI version, and inspect global symbols so
only the intended C surface is exported.

### 5. Add an experimental Slimg backend

Keep this as a separate Slimg change after the GJXL interface passes its own
tests. GJXL owns the reusable Rust boundary in `rust/gjxl-sys` and `rust/gjxl`:

- `gjxl-sys` generates the raw C bindings and owns native link metadata;
- `gjxl` owns validated packed image views, context and output lifetimes, and
  typed C error translation; and
- neither crate chooses another codec or defines application fallback policy.

Slimg should consume the safe crate and:

- keep libjxl for decoding;
- feature-gate the GJXL encoder on supported macOS builds;
- pass its RGBA8 `ImageData` directly with a checked length and stride;
- use `gjxl_distance_from_quality`;
- reuse its existing 0-100 to 1-10 effort mapping;
- retain one reusable context rather than creating it per image; and
- fall back to libjxl for quality 100, non-opaque alpha, unsupported profile
  preservation, or an unavailable experimental backend.

Backend choice and fallback should be observable in tests and diagnostics so a
benchmark cannot silently mix GJXL and libjxl samples.

### 6. Append maximum compression without breaking V1 callers

Append `GJXLCompressionMode compression_mode` at byte offset 12, retain 12 as
the V1 minimum size, and default a missing field to automatic behavior. Tests
cover old, exact 16-byte, and larger allocations, maximum-mode encoding, and
atomic rejection of unknown constants. This phase is complete.

## Validation gates

### ABI and boundary tests

- Compile the installed header as C11 and C++ with warnings treated as errors.
- Verify sized initialization with exact, smaller, and larger caller structs.
- Exercise null pointers, invalid constants, invalid dimensions, short buffers,
  undersized strides, overflow, and nonempty output buffers.
- Verify that no C++ exception crosses an entry point.
- Run AddressSanitizer and UndefinedBehaviorSanitizer over the packed-pixel and
  allocation boundary.
- Inspect the shared library's global symbols before assigning compatibility.

### Encoding correctness

- Pin the balanced effort-7 hash and preserve the former exhaustive hash under
  `GJXL_COMPRESSION_MAXIMUM`.
- Encode RGB8 and opaque RGBA8 with padded and tight row strides.
- Confirm that transparent RGBA and distance zero return `UNSUPPORTED` and do
  not modify output.
- Encode distances `0.5`, `1.0`, `2.0`, and `3.0` at efforts `1`, `7`, and `10`.
- Decode every result with the pinned independent `djxl` and verify dimensions,
  decoded pixels, and expected color metadata.
- Run the existing pinned conformance fixtures and the complete parent test
  baseline. Any inherited `quantization_pipeline` mismatch must be reproduced
  on the exact parent and reported separately from new failures.

### Effort qualification

Measure low/default/high tiers on representative natural images and
high-resolution inputs using fresh Release builds, warmups, alternating order,
and independent processes. Report:

- complete C-call wall time, including packed-pixel conversion;
- internal workflow time separately when available;
- output bytes and bits per pixel;
- independent decoded Butteraugli or another declared quality measure; and
- peak memory where practical.

The effort policy is acceptable only if low/default/high tiers demonstrate a
useful, reproducible tradeoff and maximum compression preserves the former
serializer behavior. The retained qualification is documented in
[`entropy-behavior-alignment.md`](entropy-behavior-alignment.md); a single
fixture or one timing sample remains directional evidence only.

## Deferred extensions

The following additions do not require changing the initial encode signature:

- add pixel formats to `GJXLPixelFormat`;
- append context options when device selection exists;
- append encoder options only for canonical bitstream intent;
- add `gjxl_encode_with_stats` with a separately sized stats struct;
- add a streaming encoder API;
- add alpha only after the frame and serializer support extra channels;
- add lossless semantics only after a complete Modular encoder exists; and
- add containers and metadata through a separate, explicit metadata contract.

Stage timings, GPU kernel choices, entropy settings, and experimental resident
policies should remain private or enter a separately named unstable diagnostic
API. They must not constrain future codec improvements through the stable C
ABI.
