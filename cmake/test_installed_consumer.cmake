# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

if(NOT DEFINED GJXL_BUILD_DIR OR NOT DEFINED GJXL_SOURCE_DIR)
  message(FATAL_ERROR "GJXL_BUILD_DIR and GJXL_SOURCE_DIR are required")
endif()

if(NOT DEFINED GJXL_TEST_CONFIG OR GJXL_TEST_CONFIG STREQUAL "")
  set(GJXL_TEST_CONFIG Release)
endif()

set(test_root "${GJXL_BUILD_DIR}/installed-consumer-test")
set(install_prefix "${test_root}/prefix")
set(consumer_build "${test_root}/build")
file(REMOVE_RECURSE "${test_root}")

execute_process(
  COMMAND
    "${CMAKE_COMMAND}" --install "${GJXL_BUILD_DIR}"
    --prefix "${install_prefix}"
    --config "${GJXL_TEST_CONFIG}"
  RESULT_VARIABLE install_result
  OUTPUT_VARIABLE install_output
  ERROR_VARIABLE install_error
)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR
    "gjxl installation failed:\n${install_output}\n${install_error}")
endif()

if(NOT EXISTS "${install_prefix}/include/gjxl/gjxl.h")
  message(FATAL_ERROR "installed C API header is missing")
endif()
if(EXISTS "${install_prefix}/include/c_api")
  message(FATAL_ERROR "private C adapter headers were installed")
endif()

if(GJXL_TEST_COMPILER_ID STREQUAL "MSVC")
  # A VS generator reconstructs the Windows SDK environment even when CTest
  # itself was launched outside a Developer Command Prompt. The installed
  # libraries remain ABI-compatible with the parent Ninja+MSVC build.
  set(consumer_generator_arguments -G "Visual Studio 17 2022" -A x64)
  set(consumer_compiler_arguments)
else()
  set(consumer_generator_arguments -G "${GJXL_TEST_GENERATOR}")
  set(consumer_compiler_arguments
    "-DCMAKE_C_COMPILER=${GJXL_TEST_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${GJXL_TEST_CXX_COMPILER}"
    "-DCMAKE_RC_COMPILER=${GJXL_TEST_RC_COMPILER}"
    "-DCMAKE_MT=${GJXL_TEST_MT}")
endif()

execute_process(
  COMMAND
    "${CMAKE_COMMAND}"
    ${consumer_generator_arguments}
    -S "${GJXL_SOURCE_DIR}/tests/downstream"
    -B "${consumer_build}"
    "-DCMAKE_PREFIX_PATH=${install_prefix}"
    "-DCMAKE_BUILD_TYPE=${GJXL_TEST_CONFIG}"
    ${consumer_compiler_arguments}
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_output
  ERROR_VARIABLE configure_error
)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR
    "downstream configure failed:\n${configure_output}\n${configure_error}")
endif()

execute_process(
  COMMAND
    "${CMAKE_COMMAND}" --build "${consumer_build}"
    --config "${GJXL_TEST_CONFIG}"
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_output
  ERROR_VARIABLE build_error
)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR
    "downstream build failed:\n${build_output}\n${build_error}")
endif()

foreach(
  consumer
  gjxl_codec_consumer
  gjxl_codestream_consumer
  gjxl_c_consumer)
  if(WIN32)
    set(consumer_filename "${consumer}.exe")
  else()
    set(consumer_filename "${consumer}")
  endif()
  set(consumer_executable "${consumer_build}/${consumer_filename}")
  if(NOT EXISTS "${consumer_executable}")
    set(consumer_executable
      "${consumer_build}/${GJXL_TEST_CONFIG}/${consumer_filename}")
  endif()

  execute_process(
    COMMAND "${consumer_executable}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error
  )
  if(NOT run_result EQUAL 0)
    message(FATAL_ERROR
      "downstream ${consumer} failed:\n${run_output}\n${run_error}")
  endif()
endforeach()
