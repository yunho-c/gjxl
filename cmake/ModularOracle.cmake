# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
# Included only under GJXL_BUILD_TESTS. No target here is installed/exported.
set(GJXL_MODULAR_ORACLE_BUILD "" CACHE PATH "Pinned libjxl build for Modular oracle tests")
if(NOT GJXL_MODULAR_ORACLE_BUILD)
  return()
endif()
file(STRINGS "${GJXL_MODULAR_ORACLE_BUILD}/CMakeCache.txt" modular_oracle_home
  REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
string(REPLACE "CMAKE_HOME_DIRECTORY:INTERNAL=" "" modular_oracle_source "${modular_oracle_home}")
execute_process(COMMAND git -C "${modular_oracle_source}" rev-parse HEAD
  OUTPUT_VARIABLE modular_oracle_revision OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE modular_oracle_revision_result)
execute_process(COMMAND git -C "${modular_oracle_source}" diff --quiet HEAD --
  RESULT_VARIABLE modular_oracle_dirty)
if(NOT modular_oracle_revision_result EQUAL 0 OR NOT modular_oracle_dirty EQUAL 0 OR
   NOT modular_oracle_revision STREQUAL GJXL_PINNED_LIBJXL_REVISION)
  message(FATAL_ERROR "Modular oracle requires a clean pinned libjxl source/build")
endif()
if(WIN32)
  set(modular_oracle_prefix "")
  set(modular_oracle_archive ".lib")
  set(modular_oracle_link ".lib")
else()
  set(modular_oracle_prefix "lib")
  set(modular_oracle_archive ".a")
  set(modular_oracle_link "${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()
add_library(gjxl_modular_reference STATIC tests/modular_reference.cpp)
target_include_directories(gjxl_modular_reference PUBLIC
  "${modular_oracle_source}" "${modular_oracle_source}/lib/include"
  "${GJXL_MODULAR_ORACLE_BUILD}/lib/include"
  "${modular_oracle_source}/third_party/highway")
target_compile_features(gjxl_modular_reference PUBLIC cxx_std_20)
target_link_libraries(gjxl_modular_reference PUBLIC
  "${GJXL_MODULAR_ORACLE_BUILD}/lib/${modular_oracle_prefix}jxl-internal${modular_oracle_archive}"
  "${GJXL_MODULAR_ORACLE_BUILD}/lib/${modular_oracle_prefix}jxl${modular_oracle_link}"
  "${GJXL_MODULAR_ORACLE_BUILD}/third_party/highway/${modular_oracle_prefix}hwy${modular_oracle_archive}"
  "${GJXL_MODULAR_ORACLE_BUILD}/lib/${modular_oracle_prefix}jxl_cms${modular_oracle_link}"
  "${GJXL_MODULAR_ORACLE_BUILD}/third_party/brotli/${modular_oracle_prefix}brotlienc${modular_oracle_link}"
  "${GJXL_MODULAR_ORACLE_BUILD}/third_party/brotli/${modular_oracle_prefix}brotlidec${modular_oracle_link}"
  "${GJXL_MODULAR_ORACLE_BUILD}/third_party/brotli/${modular_oracle_prefix}brotlicommon${modular_oracle_link}")
add_executable(gjxl_modular_reference_workflow_check
  tools/modular_reference_learning/reference_workflow_check.cpp src/io/pnm.cpp)
target_include_directories(gjxl_modular_reference_workflow_check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_link_libraries(gjxl_modular_reference_workflow_check PRIVATE gjxl_codestream gjxl_modular_reference)
add_executable(gjxl_modular_hybrid_profile
  tools/modular_reference_learning/hybrid_profile.cpp src/io/pnm.cpp)
add_executable(gjxl_ans_allocation_profile
  tools/modular_reference_learning/ans_allocation_profile.cpp src/io/pnm.cpp)
target_include_directories(gjxl_ans_allocation_profile PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_link_libraries(gjxl_ans_allocation_profile PRIVATE gjxl_codestream gjxl_modular_reference)
target_include_directories(gjxl_modular_hybrid_profile PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_link_libraries(gjxl_modular_hybrid_profile PRIVATE gjxl_codestream gjxl_modular_reference)
if(TARGET gjxl_metal)
  target_link_libraries(gjxl_modular_hybrid_profile PRIVATE gjxl_metal)
  target_compile_definitions(gjxl_modular_hybrid_profile PRIVATE GJXL_HYBRID_PROFILE_METAL=1)
endif()
add_executable(gjxl_modular_reference_wp_sweep
  tools/modular_reference_learning/reference_wp_sweep.cpp src/io/pnm.cpp)
target_include_directories(gjxl_modular_reference_wp_sweep PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_link_libraries(gjxl_modular_reference_wp_sweep PRIVATE gjxl_codestream gjxl_modular_reference)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME modular_reference_wp_sweep
  COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/modular_reference_learning/test_reference_wp_sweep.py"
    --binary $<TARGET_FILE:gjxl_modular_reference_wp_sweep>
    --out "${CMAKE_CURRENT_BINARY_DIR}/reference-wp-sweep-test")
set_tests_properties(modular_reference_wp_sweep PROPERTIES TIMEOUT 600)
target_link_libraries(gjxl_modular_reference_workflow_test PRIVATE gjxl_modular_reference)
target_compile_definitions(gjxl_modular_reference_workflow_test PRIVATE GJXL_REFERENCE_WORKFLOW_ORACLE=1)
target_link_libraries(gjxl_modular_public_allocation_test PRIVATE gjxl_modular_reference)
target_compile_definitions(gjxl_modular_public_allocation_test PRIVATE GJXL_PUBLIC_ALLOCATION_ORACLE=1)
target_link_libraries(gjxl_modular_parallel_learning_test PRIVATE gjxl_modular_reference)
target_compile_definitions(gjxl_modular_parallel_learning_test PRIVATE GJXL_REFERENCE_WORKFLOW_ORACLE=1)
target_link_libraries(gjxl_modular_parallel_entropy_test PRIVATE gjxl_modular_reference)
target_compile_definitions(gjxl_modular_parallel_entropy_test PRIVATE GJXL_REFERENCE_WORKFLOW_ORACLE=1)
target_link_libraries(gjxl_modular_parallel_selection_test PRIVATE gjxl_modular_reference)
target_compile_definitions(gjxl_modular_parallel_selection_test PRIVATE GJXL_REFERENCE_WORKFLOW_ORACLE=1)
target_link_libraries(gjxl_modular_parallel_prepared_tokenization_test PRIVATE gjxl_modular_reference)
target_compile_definitions(gjxl_modular_parallel_prepared_tokenization_test PRIVATE GJXL_REFERENCE_WORKFLOW_ORACLE=1)
if(TARGET gjxl_modular_reference_hybrid_test)
  target_link_libraries(gjxl_modular_reference_hybrid_test PRIVATE gjxl_modular_reference)
  target_compile_definitions(gjxl_modular_reference_hybrid_test PRIVATE GJXL_REFERENCE_WORKFLOW_ORACLE=1)
endif()
if(WIN32)
  set_tests_properties(modular_reference_workflow modular_reference_wp_sweep
    modular_parallel_learning modular_parallel_entropy modular_parallel_selection
    modular_parallel_prepared_tokenization modular_public_allocations PROPERTIES
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${GJXL_MODULAR_ORACLE_BUILD}/tools")
endif()
target_link_libraries(gjxl_ans_selection_test PRIVATE gjxl_modular_reference)
target_compile_definitions(gjxl_ans_selection_test PRIVATE GJXL_ANS_SELECTION_REFERENCE=1)
target_link_libraries(gjxl_modular_candidate_cost_test PRIVATE gjxl_modular_reference)
target_compile_definitions(gjxl_modular_candidate_cost_test PRIVATE GJXL_CANDIDATE_COST_REFERENCE=1)
target_link_libraries(gjxl_modular_candidate_search_test PRIVATE gjxl_modular_reference)
target_compile_definitions(gjxl_modular_candidate_search_test PRIVATE GJXL_CANDIDATE_COST_REFERENCE=1)
target_link_libraries(gjxl_modular_training_budget_test PRIVATE gjxl_modular_reference)
target_compile_definitions(gjxl_modular_training_budget_test PRIVATE GJXL_TRAINING_BUDGET_REFERENCE=1)
if(WIN32)
  set_tests_properties(ans_selection ans_selection_faults
    modular_candidate_cost modular_candidate_search
    modular_training_budget modular_training_budget_faults
    modular_training_checkpoints modular_training_checkpoint_faults
    modular_training_coverage modular_training_coverage_faults PROPERTIES
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${GJXL_MODULAR_ORACLE_BUILD}/tools")
endif()
if(GJXL_BUILD_BENCHMARKS)
  if(TARGET gjxl_metal)
    add_executable(gjxl_modular_metal_workflow_profile
      tools/modular_reference_learning/metal_workflow_profile.cpp src/io/pnm.cpp)
    target_include_directories(gjxl_modular_metal_workflow_profile PRIVATE tests)
    target_link_libraries(gjxl_modular_metal_workflow_profile PRIVATE
      gjxl_codestream gjxl_modular_reference gjxl_metal)
  endif()
  add_executable(gjxl_modular_search_profile tools/modular_training_study/profile.cpp src/io/pnm.cpp)
  target_include_directories(gjxl_modular_search_profile PRIVATE tests)
  target_link_libraries(gjxl_modular_search_profile PRIVATE gjxl_codestream gjxl_modular_reference)
  if(TARGET gjxl_metal)
    target_link_libraries(gjxl_modular_search_profile PRIVATE gjxl_metal)
    target_compile_definitions(gjxl_modular_search_profile PRIVATE GJXL_MODULAR_PROFILE_METAL=1)
  endif()
endif()
add_executable(gjxl_modular_oracle_test tests/modular_oracle_test.cpp)
target_link_libraries(gjxl_modular_oracle_test PRIVATE gjxl_codestream gjxl_modular_reference)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  # As for the DC oracle: pinned internal archives omit RTTI.
  target_compile_options(gjxl_modular_reference PRIVATE -fno-sanitize=vptr)
  target_compile_options(gjxl_modular_oracle_test PRIVATE -fno-sanitize=vptr)
endif()
add_test(NAME modular_oracle COMMAND gjxl_modular_oracle_test)
add_executable(gjxl_modular_conformance_test tests/modular_conformance_test.cpp)
target_link_libraries(gjxl_modular_conformance_test PRIVATE gjxl_codestream gjxl_modular_reference)
add_test(NAME modular_conformance COMMAND gjxl_modular_conformance_test
  "${CMAKE_CURRENT_BINARY_DIR}/modular-conformance")
if(WIN32)
  set_tests_properties(modular_oracle modular_conformance PROPERTIES
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${GJXL_MODULAR_ORACLE_BUILD}/tools")
endif()

add_executable(gjxl_modular_policy_test tests/modular_policy_test.cpp)
target_link_libraries(gjxl_modular_policy_test PRIVATE gjxl_codestream gjxl_modular_reference)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(gjxl_modular_policy_test PRIVATE -fno-sanitize=vptr)
endif()
add_test(NAME modular_policy COMMAND gjxl_modular_policy_test
  "${CMAKE_CURRENT_BINARY_DIR}/modular-policy")
if(WIN32)
  set_tests_properties(modular_policy PROPERTIES
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${GJXL_MODULAR_ORACLE_BUILD}/tools")
endif()

add_executable(gjxl_modular_transform_test tests/modular_transform_test.cpp)
target_link_libraries(gjxl_modular_transform_test PRIVATE gjxl_codestream gjxl_modular_reference)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(gjxl_modular_transform_test PRIVATE -fno-sanitize=vptr)
endif()
add_test(NAME modular_transform COMMAND gjxl_modular_transform_test
  "${CMAKE_CURRENT_BINARY_DIR}/modular-transform")
add_test(NAME modular_ordered_transform COMMAND gjxl_modular_transform_test
  "${CMAKE_CURRENT_BINARY_DIR}/modular-ordered-transform" ordered)

add_executable(gjxl_modular_reference_palette_oracle_test tests/modular_reference_palette_test.cpp)
target_link_libraries(gjxl_modular_reference_palette_oracle_test PRIVATE gjxl_codestream gjxl_modular_reference)
target_compile_definitions(gjxl_modular_reference_palette_oracle_test PRIVATE GJXL_PALETTE_REFERENCE=1)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(gjxl_modular_reference_palette_oracle_test PRIVATE -fno-sanitize=vptr)
endif()
add_test(NAME modular_reference_palette_oracle COMMAND gjxl_modular_reference_palette_oracle_test
  "${CMAKE_CURRENT_BINARY_DIR}/modular-reference-palette")
add_test(NAME modular_reference_channel_palette_oracle COMMAND gjxl_modular_reference_palette_oracle_test
  "${CMAKE_CURRENT_BINARY_DIR}/modular-reference-channel-palette" wide)

add_executable(gjxl_modular_prepared_stream_reference_test tests/modular_prepared_stream_reference_test.cpp)
target_link_libraries(gjxl_modular_prepared_stream_reference_test PRIVATE gjxl_codestream gjxl_modular_reference)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(gjxl_modular_prepared_stream_reference_test PRIVATE -fno-sanitize=vptr)
endif()
add_test(NAME modular_prepared_stream_reference COMMAND gjxl_modular_prepared_stream_reference_test
  "${CMAKE_CURRENT_BINARY_DIR}/modular-prepared-streams")

# These exact-cost controls use the qualified four-lane Apple AArch64 oracle.
# A different reference SIMD width can change floating-point reduction/rounding.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  include("${CMAKE_CURRENT_LIST_DIR}/ModularSelectionOracle.cmake")
  gjxl_generate_wp_cost_oracle("${modular_oracle_source}"
    "${CMAKE_CURRENT_BINARY_DIR}/modular_wp_cost_reference.cpp")
  gjxl_generate_selection_oracle("${modular_oracle_source}"
    "${CMAKE_CURRENT_BINARY_DIR}/modular_selection_reference.cpp")
  add_executable(gjxl_modular_reference_selection_oracle_test
    tests/modular_reference_selection_test.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/modular_selection_reference.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/modular_wp_cost_reference.cpp")
  target_include_directories(gjxl_modular_reference_selection_oracle_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  target_link_libraries(gjxl_modular_reference_selection_oracle_test PRIVATE gjxl_codestream gjxl_modular_reference)
  target_compile_definitions(gjxl_modular_reference_selection_oracle_test PRIVATE GJXL_SELECTION_REFERENCE=1)
  target_compile_options(gjxl_modular_reference_selection_oracle_test PRIVATE -fno-sanitize=vptr)
  add_test(NAME modular_reference_selection_oracle COMMAND gjxl_modular_reference_selection_oracle_test
    "${CMAKE_CURRENT_BINARY_DIR}/modular-reference-selection")
  foreach(selection_variant reference scalar_reference)
    add_executable(gjxl_modular_selection_cost_${selection_variant}_test
      tests/modular_selection_cost_reference_test.cpp
      "${CMAKE_CURRENT_BINARY_DIR}/modular_wp_cost_reference.cpp")
    target_link_libraries(gjxl_modular_selection_cost_${selection_variant}_test PRIVATE gjxl_codec gjxl_modular_reference)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(gjxl_modular_selection_cost_${selection_variant}_test PRIVATE -fno-sanitize=vptr)
    endif()
    if(selection_variant STREQUAL "scalar_reference")
      target_sources(gjxl_modular_selection_cost_${selection_variant}_test PRIVATE src/codec/modular/selection_cost.cpp)
      target_compile_definitions(gjxl_modular_selection_cost_${selection_variant}_test PRIVATE GJXL_SELECTION_COST_SCALAR=1)
    endif()
    add_test(NAME modular_selection_cost_${selection_variant} COMMAND gjxl_modular_selection_cost_${selection_variant}_test)
  endforeach()
endif()
if(WIN32)
  set_tests_properties(modular_transform modular_ordered_transform modular_prepared_stream_reference modular_reference_palette_oracle modular_reference_channel_palette_oracle PROPERTIES
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${GJXL_MODULAR_ORACLE_BUILD}/tools")
endif()

add_executable(gjxl_modular_api_test tests/modular_api_test.cpp)
target_link_libraries(gjxl_modular_api_test PRIVATE gjxl_c gjxl_codestream gjxl_modular_reference)
target_compile_definitions(gjxl_modular_api_test PRIVATE GJXL_TEST_HAS_METAL=$<BOOL:${GJXL_ENABLE_METAL}>)
add_test(NAME modular_api COMMAND gjxl_modular_api_test)
if(WIN32)
  set_tests_properties(modular_api PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${GJXL_MODULAR_ORACLE_BUILD}/tools")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(Python3_Interpreter_FOUND)
  add_test(NAME modular_cli COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/modular_cli_test.py"
    "$<TARGET_FILE:gjxl_encode>" "$<TARGET_FILE:gjxl_modular_api_test>" "${CMAKE_CURRENT_BINARY_DIR}/modular-cli" "$<BOOL:${GJXL_ENABLE_METAL}>")
  if(WIN32)
    set_tests_properties(modular_cli PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${GJXL_MODULAR_ORACLE_BUILD}/tools")
  endif()
endif()

add_executable(gjxl_modular_reference_capture tools/modular_oracle/capture.cpp src/io/pnm.cpp)
target_compile_definitions(gjxl_modular_reference_capture PRIVATE GJXL_ORACLE_REFERENCE=1)
target_include_directories(gjxl_modular_reference_capture PRIVATE tests)
target_link_libraries(gjxl_modular_reference_capture PRIVATE gjxl_codestream gjxl_modular_reference)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(gjxl_modular_reference_capture PRIVATE -fno-sanitize=vptr)
endif()
if(Python3_Interpreter_FOUND)
  add_test(NAME modular_frozen_reference COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/modular_oracle/verify.py"
    "$<TARGET_FILE:gjxl_modular_reference_capture>" "${CMAKE_CURRENT_SOURCE_DIR}/testdata/modular"
    "${CMAKE_CURRENT_BINARY_DIR}/modular-frozen-reference")
  if(WIN32)
    set_tests_properties(modular_frozen_reference PROPERTIES
      ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${GJXL_MODULAR_ORACLE_BUILD}/tools")
  endif()
endif()
