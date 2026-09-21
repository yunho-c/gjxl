# Storage toolchain contract

GJXL's managed-memory bounds depend on reviewed standard-library allocation
behavior. The source build and installed C++ interface currently support this
contract:

| Platform | Compiler / headers | ABI and modes |
| --- | --- | --- |
| macOS | Apple Clang 17.0.0, build `17000604`; SDK libc++ `200100` | Stable ABI 1; C++20/C++23, including Objective-C++ |
| Windows x64 | MSVC 19.37 / STL 143 update `202305` (VS 2022 toolset 14.37.32822) | Release iterator level 0; C++20 or MSVC's C++23/latest mode |
| Linux x86-64 | Ubuntu 24.04 GCC `13.3.0-6ubuntu2~24.04.1`; libstdc++ 13 headers `20240904` | C++11 string ABI; release library; C++20/C++23 |

Windows and Linux support is being qualified on the CUDA integration branch.
Their allocation and installed-interface results must be recorded before final
acceptance. GNU debug/parallel modes, other library versions and compiler builds
remain rejected. GCC 13 reports its C++23 mode as `__cplusplus == 202100L`;
MSVC 19.37 reports `/std:c++latest` as `_MSVC_LANG == 202004L`.

The native and Rust CI matrices select Xcode 26.3 (build `17C529`) on
the `macos-26` runner through `DEVELOPER_DIR`, as does publishing. This provides the audited
compiler and libc++ headers above; the runner's default Xcode can change
independently and is not a supported-toolchain selection. The shared
`setup-native-toolchain` action pins the GNU compiler/header/runtime packages
listed above and selects MSVC 14.37 explicitly on Windows. It installs the
Microsoft `Microsoft.VisualStudio.Component.VC.14.37.17.7.x86.x64` component
when that toolset is absent. A missing pinned package or toolchain is a CI
failure; there is no fallback that disables storage validation.

The qualified machine is Apple M4 Pro on macOS 15.6, as recorded in the
[final scheduling qualification](resident-scheduling-qualification.md). The
version checks identify the reviewed implementation family, not the hash of
every compiler or SDK file. Modified standard-library headers or custom ABI
configuration are unsupported even if they retain those version macros.

In the audited libc++, C++23 uses a different `allocate_at_least` dispatch
path, but the allocator implementations preserve GJXL's backing counts:

- `std::allocator<T>::allocate_at_least(n)` returns exactly `{allocate(n), n}`
  in this SDK's `__memory/allocator.h`.
- `ManagedAllocator` has no `allocate_at_least` member, so
  `__memory/allocator_traits.h` falls back to `{allocator.allocate(n), n}`.
- Vector growth, string rounding/inline storage, and hash-node/bucket behavior
  for GJXL's existing operations follow the same bounds in both modes.

This is an implementation-specific conclusion, not a general C++23 guarantee
that `allocate_at_least` cannot return extra capacity. C++26 and later, other
standard-library versions, compiler builds, and changes to GJXL's allocator
dispatch require a new audit. Checking capacity only after an allocation is
insufficient to establish the pre-allocation guarantee.

The source build checks both the C++ and Objective-C++ compiler/SDK combinations
during CMake configuration, before building the encoder or shaders. The
installed package runs the same probe for enabled C++ languages when selecting
the C++ components at `find_package(gjxl)` time. Explicit unsupported
`CMAKE_CXX_STANDARD` or `CMAKE_OBJCXX_STANDARD` values are rejected. Header checks
also cover direct compiler invocations and target-specific overrides after
configuration.

The exported `cxx_std_20` feature selects the minimum for CMake consumers;
the storage contract accepts the audited C++20 and C++23 modes. Clients using
only `gjxl/gjxl.h` can request `find_package(gjxl CONFIG REQUIRED COMPONENTS c)`
and link `gjxl::c` without this C++ interface restriction; the installed test
also exercises such a client in C++23. Request `codec`, `codestream` or `core`
when using those C++ interfaces. Omitting `COMPONENTS` checks the full interface.
Building the underlying library still requires the supported C++ toolchain.
The [installed interface](installed-interface.md) distinguishes public headers
from the implementation dependencies needed by those C++ consumers.

## Compatibility boundary

[`src/core/stdlib_storage_compat.h`](../src/core/stdlib_storage_compat.h) owns the
implementation-specific checks, backing factors, construction helpers, string
pointer classification, and private hash-node type. Codec planners use that
boundary without naming private standard-library types.

| Audited source in the SDK's `usr/include/c++/v1` | Contract used by GJXL |
| --- | --- |
| `__vector/vector.h`, `__memory/allocate_at_least.h`, `__memory/allocator.h`, `__memory/allocator_traits.h` | Fresh count/forward-range construction and reserve allocate the requested count in both audited modes. Growth uses the larger of twice the old capacity and the requested size, with max-size saturation. Old backing remains alive during replacement. |
| `string` | Character backing includes the terminator and rounded capacity. Growth and replacement fit the declared factors/slack. Short-string data lies inside the object; long-string data is the original allocation pointer. |
| `__hash_table`, `unordered_map` | Node allocation uses the concrete rebound node type. Starting empty with default load factor 1 and unique-key `operator[]` insertions, bucket growth and replacement fit the declared bounds. |

The Windows audit covers `vector`, `xstring`, `list`, `xhash` and
`yvals_core.h` in toolset 14.37.32822. Fresh vectors/reserve allocate exact
counts; growth is 1.5x, inside the shared 2x envelope. String growth/rounding fits
the shared slack. Hash maps own an allocated list sentinel and 8 initial
buckets with two pointer-sized slots per bucket; subsequent 8x/power-of-two
bucket growth is covered by the MSVC-specific empty, retained and peak bounds.

The GNU audit covers `bits/stl_vector.h`, `bits/vector.tcc`,
`bits/basic_string.h`, `bits/basic_string.tcc`, `bits/hashtable.h`,
`bits/hashtable_policy.h` and `bits/unordered_map.h` in the listed package.
Vectors grow by `size + max(size, added)`; fresh counts and reserve are exact.
C++11 strings double growing capacity and allocate a terminator. The short
buffer remains inside the object. The hash node uses the actual default hash's
cache trait; empty buckets are embedded, with no allocated sentinel.

The GNU runtime on the qualification host is `libstdc++6
14.2.0-4ubuntu2~24.04.1`. The out-of-line prime rehash policy was audited in both
[GCC 13.3](https://github.com/gcc-mirror/gcc/blob/releases/gcc-13.3.0/libstdc%2B%2B-v3/src/c%2B%2B11/hashtable_c%2B%2B0x.cc)
and [GCC 14.2](https://github.com/gcc-mirror/gcc/blob/releases/gcc-14.2.0/libstdc%2B%2B-v3/src/c%2B%2B11/hashtable_c%2B%2B0x.cc).
The first insertion allocates 13 buckets; later growth requests at most twice
the new element count. The [prime table](https://github.com/gcc-mirror/gcc/blob/releases/gcc-14.2.0/libstdc%2B%2B-v3/src/shared/hashtable-aux.cc)
has adjacent ratios below two, including its 64-bit tail. For default load
factor 1 and the restricted insertion pattern, 13 pointers per entry bounds
retained buckets and 14 bounds simultaneous old/replacement buckets. Runtime
and header packages are part of this qualification; a different runtime needs
its own audit even if the header identity passes the compile-time check.

The boundary does not account for arbitrary container operations. Existing
planner restrictions still apply: for example, input-iterator vector growth,
`vector<bool>`, nested element allocations, or hash-table reserve/load-factor
changes need their own bounds. The block-context map's tripling resize also
checks that the audited vector growth factor permits its exact-capacity bound.

Managed backing remains distinct from process footprint. Allocator headers,
runtime metadata and other documented exclusions are unchanged. An admitted
allocation that exceeds its plan still returns `ResourcePlanExceeded` before
allocating its backing.

## Validation and extending support

Configure a separate Release build with the supported Xcode selected. The
default is C++20; select C++23 for the complete implementation with
`-DCMAKE_CXX_STANDARD=23`. Objective-C++ inherits that mode unless explicitly
set with `CMAKE_OBJCXX_STANDARD`.

```sh
cmake -S . -B build/storage-toolchain -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/storage-toolchain --parallel
ctest --test-dir build/storage-toolchain --output-on-failure \
  -R '^(stdlib_storage_compat(_cxx23)?|storage_toolchain|managed_allocator|publication_storage|diagnostic_storage|profile_storage_plan|entropy_storage_plan|codec_install_consumer)$'
```

`stdlib_storage_compat` and `stdlib_storage_compat_cxx23` compile the same
allocation tests in C++20 and C++23. They check actual vector capacities,
managed replacement peaks, string allocation/publication boundaries, and
independently observed hash-node/bucket allocation requests. The C++23 test
also exercises standard and managed `allocate_at_least` paths where provided.
`storage_toolchain` checks public-header compilation in both modes and rejection
of unsupported modes, library/compiler identities, and unstable ABI selection.
The installed-consumer test builds and runs the C/C++ callers in both modes
against the installed library, rejects C++26 at configuration, and exercises a
client requesting only the C ABI.

To add another toolchain, first audit the corresponding library source paths
and update the compatibility boundary's supported identities and proofs. Run
these tests, the component storage-plan and publication suites, and the
existing exactness/concurrency qualification relevant to any changed allocation
behavior. Tests sample implementation behavior; they do not replace the source
argument for a hard bound over all supported input sizes. Record the compiler,
SDK, ABI, language mode, and qualification results before widening support.
