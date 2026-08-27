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

execute_process(
  COMMAND
    "${CMAKE_COMMAND}"
    -S "${GJXL_SOURCE_DIR}/tests/downstream"
    -B "${consumer_build}"
    "-DCMAKE_PREFIX_PATH=${install_prefix}"
    "-DCMAKE_BUILD_TYPE=${GJXL_TEST_CONFIG}"
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

foreach(consumer gjxl_codec_consumer gjxl_codestream_consumer)
  set(consumer_executable "${consumer_build}/${consumer}")
  if(NOT EXISTS "${consumer_executable}")
    set(consumer_executable
      "${consumer_build}/${GJXL_TEST_CONFIG}/${consumer}")
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
