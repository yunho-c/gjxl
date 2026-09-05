# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
# Included only with GJXL_BUILD_TESTS. These targets are deliberately opt-in.
option(GJXL_ENABLE_METAL_QUALIFICATION_TEST
  "Register the prepared compact Metal qualification with CTest" OFF)
set(GJXL_METAL_QUALIFICATION_COMPARISON "same-distance" CACHE STRING
  "Reference comparison: same-distance (no search) or matched-quality")
set_property(CACHE GJXL_METAL_QUALIFICATION_COMPARISON PROPERTY STRINGS
  same-distance matched-quality)
if(NOT GJXL_METAL_QUALIFICATION_COMPARISON STREQUAL "same-distance" AND
   NOT GJXL_METAL_QUALIFICATION_COMPARISON STREQUAL "matched-quality")
  message(FATAL_ERROR "Invalid Metal qualification comparison mode")
endif()
set(GJXL_METAL_QUALIFICATION_BASELINE "" CACHE FILEPATH
  "Accepted compact baseline (empty performs initial qualification)")
set(GJXL_METAL_QUALIFICATION_FULL_BASELINE "" CACHE FILEPATH
  "Accepted full baseline (empty performs initial qualification)")
set(GJXL_METAL_QUALIFICATION_PILOT_MANIFEST "" CACHE FILEPATH
  "Existing hash-verified canonical 38-image pilot corpus to reuse")
set(GJXL_METAL_QUALIFICATION_MAX_LIBJXL_SIZE_RATIO "" CACHE STRING
  "Matched-quality only: reviewed maximum Metal/libjxl byte ratio")
if(GJXL_METAL_QUALIFICATION_COMPARISON STREQUAL "same-distance" AND
   NOT GJXL_METAL_QUALIFICATION_MAX_LIBJXL_SIZE_RATIO STREQUAL "")
  message(FATAL_ERROR "A libjxl size allowance requires matched-quality comparison")
endif()
if(GJXL_ENABLE_METAL_QUALIFICATION_TEST AND
   NOT GJXL_METAL_QUALIFICATION_BASELINE AND
   (GJXL_METAL_QUALIFICATION_COMPARISON STREQUAL "same-distance" OR
    NOT GJXL_METAL_QUALIFICATION_MAX_LIBJXL_SIZE_RATIO))
  message(FATAL_ERROR
    "Metal CTest needs an accepted baseline (or a matched-quality size allowance)")
endif()

set(qualification_root "${CMAKE_CURRENT_BINARY_DIR}/metal-qualification")
set(qualification_runner "${CMAKE_CURRENT_SOURCE_DIR}/tools/metal_resident_qualification.py")

add_test(NAME metal_resident_qualification_driver
  COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/metal_resident_qualification_test.py")
set_tests_properties(metal_resident_qualification_driver PROPERTIES LABELS "cli;qualification-unit")

add_custom_target(gjxl_pinned_quality_tools
  COMMAND "${CMAKE_COMMAND}"
    "-DGJXL_LIBJXL_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjxl"
    "-DGJXL_LIBJXL_BUILD=${GJXL_PINNED_LIBJXL_BUILD}"
    "-DGJXL_LIBJXL_REVISION=${GJXL_PINNED_LIBJXL_REVISION}"
    "-DGJXL_GENERATOR=${CMAKE_GENERATOR}"
    -DGJXL_LIBJXL_QUALITY_TOOLS=ON
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildPinnedDjxl.cmake"
  USES_TERMINAL VERBATIM)

set(qualification_common
  --comparison "${GJXL_METAL_QUALIFICATION_COMPARISON}"
  --encoder $<TARGET_FILE:gjxl_encode>
  --cjxl "${GJXL_PINNED_LIBJXL_BUILD}/tools/cjxl"
  --djxl "${GJXL_PINNED_LIBJXL_BUILD}/tools/djxl"
  --butteraugli "${GJXL_PINNED_LIBJXL_BUILD}/tools/butteraugli_main"
  --reference-manifest "${GJXL_PINNED_LIBJXL_BUILD}/quality-reference.json"
  --cache "${qualification_root}/cache")
if(NOT GJXL_METAL_QUALIFICATION_MAX_LIBJXL_SIZE_RATIO STREQUAL "")
  list(APPEND qualification_common --max-libjxl-size-ratio
    "${GJXL_METAL_QUALIFICATION_MAX_LIBJXL_SIZE_RATIO}")
endif()

foreach(suite IN ITEMS compact full)
  if(suite STREQUAL "full" AND NOT GJXL_BUILD_BENCHMARKS)
    continue()
  endif()
  set(corpus "${qualification_root}/${suite}-corpus/manifest.json")
  set(prepare_args --suite "${suite}" --output "${qualification_root}/${suite}-corpus")
  set(prepare_dependencies)
  set(baseline "${GJXL_METAL_QUALIFICATION_BASELINE}")
  if(suite STREQUAL "full")
    set(baseline "${GJXL_METAL_QUALIFICATION_FULL_BASELINE}")
    if(GJXL_METAL_QUALIFICATION_PILOT_MANIFEST)
      list(APPEND prepare_args --pilot-manifest "${GJXL_METAL_QUALIFICATION_PILOT_MANIFEST}")
    else()
      list(APPEND prepare_args --gjxl-benchmark $<TARGET_FILE:gjxl_encoding_benchmark>)
      list(APPEND prepare_dependencies gjxl_encoding_benchmark)
    endif()
  endif()
  add_custom_command(OUTPUT "${corpus}"
    COMMAND "${Python3_EXECUTABLE}" "${qualification_runner}" prepare ${prepare_args}
    DEPENDS ${prepare_dependencies}
    COMMENT "Preparing ${suite} resident qualification corpus"
    VERBATIM)
  add_custom_target(metal-resident-prepare-${suite} DEPENDS "${corpus}")
  set(run_args run --suite "${suite}" --corpus "${corpus}"
    --output "${qualification_root}/${suite}-${GJXL_METAL_QUALIFICATION_COMPARISON}-run"
    ${qualification_common})
  if(baseline)
    list(APPEND run_args --baseline "${baseline}")
  endif()
  add_custom_target(metal-resident-qualify-${suite}
    COMMAND "${Python3_EXECUTABLE}" "${qualification_runner}" ${run_args}
    DEPENDS gjxl_encode gjxl_pinned_quality_tools metal-resident-prepare-${suite}
    USES_TERMINAL VERBATIM)
  if(suite STREQUAL "compact" AND GJXL_ENABLE_METAL_QUALIFICATION_TEST)
    add_test(NAME metal_resident_qualification
      COMMAND "${Python3_EXECUTABLE}" "${qualification_runner}" ${run_args})
    set_tests_properties(metal_resident_qualification PROPERTIES LABELS "metal;qualification"
      RUN_SERIAL TRUE)
  endif()
endforeach()
