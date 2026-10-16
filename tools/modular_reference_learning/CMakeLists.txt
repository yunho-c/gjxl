cmake_minimum_required(VERSION 3.25)
project(modular_reference_learning LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(GJXL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
foreach(var GJXL_BUILD REFERENCE_SOURCE REFERENCE_BUILD GENERATED_SOURCE)
  if(NOT DEFINED ${var} OR NOT IS_ABSOLUTE "${${var}}")
    message(FATAL_ERROR "Set ${var} to an absolute path")
  endif()
endforeach()
include("${GJXL_SOURCE}/cmake/ModularSelectionOracle.cmake")
gjxl_generate_wp_cost_oracle("${REFERENCE_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}/modular_wp_cost_reference.cpp")
# Native fitting uses ordinary production compiler settings. Reference-specific
# vectorization/RTTI flags below apply only to the adapter/reference target.
add_library(native_histogram_fit OBJECT
  ${GJXL_SOURCE}/src/codec/modular/histogram_tree_learning.cpp
  ${GJXL_SOURCE}/src/codec/modular/histogram_sample_preparation.cpp)
target_include_directories(native_histogram_fit PRIVATE
  ${GJXL_SOURCE}/include ${GJXL_SOURCE}/src)
add_executable(reference_learning runner.cpp reference_capture.cpp native_fit.cpp native_fit_tests.cpp native_preparation.cpp paired_learning.cpp policy_trace.cpp stream_replay.cpp coding_replay.cpp entropy_replay.cpp selection_cost_replay.cpp
  $<TARGET_OBJECTS:native_histogram_fit>
  ${GJXL_SOURCE}/src/io/pnm.cpp
  ${GENERATED_SOURCE}/reference_codec.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/modular_wp_cost_reference.cpp
  ${GENERATED_SOURCE}/lib/jxl/modular/encoding/enc_ma.cc
  ${GENERATED_SOURCE}/lib/jxl/modular/encoding/enc_encoding.cc)
target_include_directories(reference_learning PRIVATE
  ${GENERATED_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}
  ${GJXL_SOURCE}/include ${GJXL_SOURCE}/src ${GJXL_SOURCE}/tests
  ${REFERENCE_SOURCE} ${REFERENCE_SOURCE}/lib/include
  ${REFERENCE_BUILD}/lib/include ${REFERENCE_SOURCE}/third_party/highway)
target_compile_definitions(reference_learning PRIVATE JXL_INTERNAL_LIBRARY_BUILD
  JPEGXL_ENABLE_SKCMS=1 JPEGXL_ENABLE_TRANSCODE_JPEG=0 JPEGXL_ENABLE_BOXES=1
  "HWY_DISABLED_TARGETS=(HWY_AVX3|HWY_AVX3_SPR|HWY_AVX3_ZEN4|HWY_RVV|HWY_SSSE3|HWY_SVE_256)")
# Match the pinned reference's floating-point/vectorization configuration.
target_compile_options(reference_learning PRIVATE -fno-slp-vectorize -fno-vectorize)
set_source_files_properties(
  ${GENERATED_SOURCE}/lib/jxl/modular/encoding/enc_ma.cc
  ${GENERATED_SOURCE}/lib/jxl/modular/encoding/enc_encoding.cc
  # Pair these flags as in the pinned build. On Apple libc++, no RTTI with
  # exceptions enabled emits private std::exception typeinfo that breaks
  # catches in the RTTI-enabled harness.
  PROPERTIES COMPILE_OPTIONS "-fno-rtti;-fno-exceptions")
target_link_libraries(reference_learning PRIVATE
  ${GJXL_BUILD}/libgjxl_codestream.a ${GJXL_BUILD}/libgjxl_gpu_ops.a
  ${GJXL_BUILD}/libgjxl_codec.a ${GJXL_BUILD}/libgjxl_gpu.a
  ${REFERENCE_BUILD}/lib/libjxl-internal.a
  ${REFERENCE_BUILD}/third_party/highway/libhwy.a
  ${REFERENCE_BUILD}/lib/libjxl_cms.dylib
  ${REFERENCE_BUILD}/third_party/brotli/libbrotlienc.dylib
  ${REFERENCE_BUILD}/third_party/brotli/libbrotlidec.dylib
  ${REFERENCE_BUILD}/third_party/brotli/libbrotlicommon.dylib)

# The bridge can consume either CPU-only or Metal-enabled production archives.
# The latter's workflow object also references Metal admission/backend helpers.
file(STRINGS "${GJXL_BUILD}/CMakeCache.txt" gjxl_metal_enabled
  REGEX "^GJXL_ENABLE_METAL:BOOL=ON$")
if(gjxl_metal_enabled)
  foreach(framework Metal Foundation CoreGraphics)
    find_library(gjxl_${framework}_framework ${framework} REQUIRED)
    target_link_libraries(reference_learning PRIVATE "${gjxl_${framework}_framework}")
  endforeach()
  target_link_libraries(reference_learning PRIVATE
    ${GJXL_BUILD}/libgjxl_metal.a ${GJXL_BUILD}/libgjxl_gpu_butteraugli.a)
endif()

# A native-only target also runs in this small research build. It intentionally
# has no reference headers/libraries and uses the normal compiler math settings.
add_executable(native_fit_test
  ${GJXL_SOURCE}/tests/modular_histogram_learning_test.cpp
  $<TARGET_OBJECTS:native_histogram_fit>)
target_include_directories(native_fit_test PRIVATE
  ${GJXL_SOURCE}/include ${GJXL_SOURCE}/src)
target_link_libraries(native_fit_test PRIVATE ${GJXL_BUILD}/libgjxl_codec.a)
enable_testing()
add_test(NAME native_fit_contract COMMAND native_fit_test)
add_executable(native_preparation_test ${GJXL_SOURCE}/tests/modular_histogram_preparation_test.cpp
  $<TARGET_OBJECTS:native_histogram_fit>)
target_include_directories(native_preparation_test PRIVATE ${GJXL_SOURCE}/include ${GJXL_SOURCE}/src)
target_link_libraries(native_preparation_test PRIVATE ${GJXL_BUILD}/libgjxl_codec.a)
add_test(NAME native_preparation_contract COMMAND native_preparation_test)
add_test(NAME native_fit_differential COMMAND reference_learning native-self-test)
add_test(NAME native_preparation_differential COMMAND reference_learning preparation-self-test)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME native_replay_ledger
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_native_replay.py)
add_test(NAME native_preparation_ledger
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_native_preparation.py)
add_test(NAME paired_learning_ledger
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_paired_learning.py)
add_test(NAME stream_replay_contract
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_stream_replay.py
    --binary $<TARGET_FILE:reference_learning>)
add_test(NAME coding_replay_contract
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_coding_replay.py
    --binary $<TARGET_FILE:reference_learning>)
add_test(NAME entropy_replay_contract
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_entropy_replay.py
    --binary $<TARGET_FILE:reference_learning>)
