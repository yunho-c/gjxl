# Storage toolchain contract

GJXL's managed-memory bounds depend on reviewed standard-library allocation
behavior. The source build and installed C++ interface currently support this
contract:

| Property | Supported value |
| --- | --- |
| Platform | macOS |
| Compiler | Apple Clang 17.0.0, audited as `clang-1700.6.4.2`, compiler build macro `17000604` |
| Standard library headers | Apple SDK libc++, `_LIBCPP_VERSION == 200100` |
| Standard library ABI | Stable ABI version 1; stock SDK configuration |
| C++ and Objective-C++ language mode | C++20 or C++23, including GNU modes; default C++20 |

The qualified machine is Apple M4 Pro on macOS 15.6, as recorded in the
[final scheduling qualification](resident-scheduling-qualification.md). The
version checks identify the reviewed implementation family, not the hash of
every compiler or SDK file. Modified standard-library headers or custom ABI
configuration are unsupported even if they retain those version macros.

C++23 uses a different `allocate_at_least` dispatch path, but the audited
allocator implementations preserve the backing counts used by GJXL:

- `std::allocator<T>::allocate_at_least(n)` returns exactly `{allocate(n), n}`
  in this SDK's `__memory/allocator.h`.
- `ManagedAllocator` has no `allocate_at_least` member, so
  `__memory/allocator_traits.h` falls back to `{allocator.allocate(n), n}`.
- Vector growth, string rounding/inline storage, and hash-node/bucket behavior
  for GJXL's existing operations follow the same bounds in both modes.

This is an implementation-specific conclusion, not a general C++23 guarantee
that `allocate_at_least` cannot return extra capacity. C++26 and later, other
libc++ versions, libstdc++, other compiler builds, and changes to GJXL's allocator
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

## Compatibility boundary

[`src/core/stdlib_storage_compat.h`](../src/core/stdlib_storage_compat.h) owns the
implementation-specific checks, backing factors, construction helpers, string
pointer classification, and private hash-node type. Codec planners use that
boundary without naming private libc++ types.

| Audited source in the SDK's `usr/include/c++/v1` | Contract used by GJXL |
| --- | --- |
| `__vector/vector.h`, `__memory/allocate_at_least.h`, `__memory/allocator.h`, `__memory/allocator_traits.h` | Fresh count/forward-range construction and reserve allocate the requested count in both audited modes. Growth uses the larger of twice the old capacity and the requested size, with max-size saturation. Old backing remains alive during replacement. |
| `string` | Character backing includes the terminator and rounded capacity. Growth and replacement fit the declared factors/slack. Short-string data lies inside the object; long-string data is the original allocation pointer. |
| `__hash_table`, `unordered_map` | Node allocation uses the concrete rebound node type. Starting empty with default load factor 1 and unique-key `operator[]` insertions, bucket growth and replacement fit the declared bounds. |

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
also directly exercises both standard and managed `allocate_at_least` paths.
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
