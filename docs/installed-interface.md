# Installed interface

[`cmake/InstalledHeaders.cmake`](../cmake/InstalledHeaders.cmake) defines the
installed headers explicitly. It separates public C/C++ entry points from the
implementation dependencies needed to compile them. Adding a header under
`src/` does not add it to the package.

The public list preserves the C API, C++ domain bridge, core value/owner types,
codec algorithms, serializer primitives, single/batch workflows, GPU operation
contracts, and Metal backend factory. Existing include paths and exported target
names are unchanged.

| Consumer interface | Headers | CMake target |
| --- | --- | --- |
| C API | `gjxl/gjxl.h` | `gjxl::c` |
| C/C++ execution-domain bridge | `gjxl/execution_domain.hpp` | `gjxl::c` plus the relevant C++ target |
| Core values, images and execution domain | Public `core/` entries in the manifest | `gjxl::core` |
| CPU codec algorithms and frames | Public `codec/` entries | `gjxl::codec` |
| Serialization and complete workflows | Public `codestream/` entries | `gjxl::codestream` |
| GPU interfaces and Metal factory | Public `gpu/` entries | Existing `gjxl::gjxl_gpu`, `gjxl::gjxl_gpu_ops`, `gjxl::gjxl_gpu_butteraugli`, `gjxl::gjxl_metal` targets, also linked transitively by `gjxl::codestream` |

The package components remain `c`, `core`, `codec`, and `codestream`. For example:

```cmake
find_package(gjxl CONFIG REQUIRED COMPONENTS codestream)
target_link_libraries(my_encoder PRIVATE gjxl::codestream)
```

C ABI-only clients can request just `COMPONENTS c`. C++ consumers use the
audited C++20/C++23 [storage toolchain contract](storage-toolchain.md).

## Required implementation dependencies

Public headers contain templates, inline methods, and owning data members.
Their dependencies must be installed even when the implementation lives in an
internal namespace. The support list records these dependencies separately;
shipping them does not make their internal names supported consumer APIs.

| Public dependency | Required support headers |
| --- | --- |
| `ExecutionDomain` construction and snapshots | `core/cpu_budget.h`, `core/resource_budget.h`, `core/resource_context.h` |
| Managed storage in strategy grids, image buffers, frames and serializer records | `core/managed_allocator.h` and its resource dependencies |
| Adaptive quantization's `score_history` output | `core/publication_output.h`, `core/publication_vector.h`, `core/stdlib_storage_compat.h` |
| Serializer container types and ordinary-vector output overloads | `codestream/storage.h` and its allocator/publication dependencies |
| Installed C++ toolchain probe | `core/stdlib_storage_compat.h` |

These headers retain the same definitions as the source build, including the
managed allocator's existing fault-injection plumbing. Internal declarations
embedded in public headers remain implementation details. Installation does not
strip declarations or introduce consumer-specific macros that could change
class layouts or violate the one-definition rule.

Storage planners, workflow admission, standalone test hooks, worker management,
generated quantization tables, Metal implementation helpers, and profiling
internals are outside the installed list. Neither generated build headers nor
the private `metal-cpp` headers are needed to compile the installed interface.

## Maintaining the boundary

When adding an API, add its header to the public list and trace its dependencies.
Put any required implementation headers in the support list, with the reason
they are needed. When removing a dependency, remove its support entry if no
other public header or package probe needs it. Public members may require a
support header even if the filename sounds private; filename exclusions cannot
establish this boundary.

Run `ctest --test-dir <build> --output-on-failure -R '^codec_install_consumer$'`
after building the installable targets. The test:

- Installs to a fresh prefix, relocates it, and compares the complete installed
  header inventory with the manifest, catching missing and accidental exports.
- Checks that package files do not refer to the source tree, build tree, or
  original staging prefix.
- Compiles every installed header in its own translation unit in both C++20
  and C++23, with warnings as errors; the C header also compiles independently
  as C11. There are no source, generated-header, or `metal-cpp` include paths.
- Builds and runs the existing C, C++, batch/shutdown and shared-domain callers
  against the relocated archives, and retains the toolchain rejection checks.

An in-place install over an older package can retain obsolete headers: CMake
does not uninstall files removed from this manifest. Use a fresh prefix or
remove the old package through its packaging/uninstall mechanism when replacing
such an installation.
