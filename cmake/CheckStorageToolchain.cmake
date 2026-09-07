# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

include(CheckSourceCompiles)
include(CMakePushCheckState)

# Used by the source build and the installed package. The header is the single
# version/ABI policy; compile it with each enabled C++ compiler and selected SDK.
function(gjxl_check_storage_toolchain include_dir)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES "${include_dir}")
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  foreach(language IN ITEMS CXX OBJCXX)
    if(NOT CMAKE_${language}_COMPILER_LOADED)
      continue()
    endif()
    if(DEFINED CMAKE_${language}_STANDARD AND
       NOT CMAKE_${language}_STANDARD STREQUAL "20" AND
       NOT CMAKE_${language}_STANDARD STREQUAL "23")
      message(FATAL_ERROR
        "GJXL storage contract requires C++20 or C++23 (${language}); "
        "CMAKE_${language}_STANDARD=${CMAKE_${language}_STANDARD}. "
        "See docs/storage-toolchain.md.")
    endif()
    if(NOT DEFINED CMAKE_${language}_STANDARD)
      set(CMAKE_${language}_STANDARD 20)
    endif()
    set(CMAKE_${language}_STANDARD_REQUIRED ON)
    # Recheck after SDK/compiler/flags or the audited contract changes.
    unset(GJXL_${language}_STORAGE_TOOLCHAIN_SUPPORTED CACHE)
    check_source_compiles(${language}
      "#include <core/stdlib_storage_compat.h>\nint main() { return 0; }"
      GJXL_${language}_STORAGE_TOOLCHAIN_SUPPORTED)
    if(NOT GJXL_${language}_STORAGE_TOOLCHAIN_SUPPORTED)
      message(FATAL_ERROR
        "GJXL ${language} compiler/SDK does not satisfy the audited storage "
        "contract. See the compile diagnostic in CMakeFiles/CMakeConfigureLog.yaml "
        "(CMakeFiles/CMakeError.log on older CMake) and docs/storage-toolchain.md.")
    endif()
  endforeach()
  cmake_pop_check_state()
endfunction()
