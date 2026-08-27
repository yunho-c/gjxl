# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

foreach(required GJXL_ENCODER GJXL_SAMPLE GJXL_TEST_DIR)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${GJXL_TEST_DIR}")
file(MAKE_DIRECTORY "${GJXL_TEST_DIR}")

set(sentinel "${GJXL_TEST_DIR}/sentinel.jxl")
file(WRITE "${sentinel}" "unchanged")
execute_process(
  COMMAND
    "${GJXL_ENCODER}" --distance 0 "${GJXL_SAMPLE}" "${sentinel}"
  RESULT_VARIABLE invalid_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(invalid_result EQUAL 0)
  message(FATAL_ERROR "CLI accepted an invalid perceptual target")
endif()
file(READ "${sentinel}" sentinel_contents)
if(NOT sentinel_contents STREQUAL "unchanged")
  message(FATAL_ERROR "CLI failure changed an existing output")
endif()

execute_process(
  COMMAND
    "${GJXL_ENCODER}" --distance 1.0 --backend invalid
    "${GJXL_SAMPLE}" "${sentinel}"
  RESULT_VARIABLE invalid_backend_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(invalid_backend_result EQUAL 0)
  message(FATAL_ERROR "CLI accepted an invalid backend override")
endif()
file(READ "${sentinel}" sentinel_contents)
if(NOT sentinel_contents STREQUAL "unchanged")
  message(FATAL_ERROR "Invalid backend changed an existing output")
endif()

set(malformed "${GJXL_TEST_DIR}/malformed.pfm")
file(WRITE "${malformed}" "PF\n2 2\n-1.0\ntruncated")
execute_process(
  COMMAND
    "${GJXL_ENCODER}" --distance 1.0 "${malformed}" "${sentinel}"
  RESULT_VARIABLE malformed_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(malformed_result EQUAL 0)
  message(FATAL_ERROR "CLI accepted a truncated PFM")
endif()
file(READ "${sentinel}" sentinel_contents)
if(NOT sentinel_contents STREQUAL "unchanged")
  message(FATAL_ERROR "Input failure changed an existing output")
endif()

set(first "${GJXL_TEST_DIR}/first.jxl")
set(second "${GJXL_TEST_DIR}/second.jxl")
set(metal "${GJXL_TEST_DIR}/metal.jxl")
execute_process(
  COMMAND
    "${GJXL_ENCODER}" --distance 1.0 --backend cpu
    "${GJXL_SAMPLE}" "${first}"
  RESULT_VARIABLE first_result
  OUTPUT_VARIABLE first_output
  ERROR_VARIABLE first_error
)
if(NOT first_result EQUAL 0)
  message(FATAL_ERROR "First CLI encode failed: ${first_error}")
endif()
execute_process(
  COMMAND
    "${GJXL_ENCODER}" --distance 1.0 "${GJXL_SAMPLE}" "${second}"
  RESULT_VARIABLE second_result
  OUTPUT_VARIABLE second_output
  ERROR_VARIABLE second_error
)
if(NOT second_result EQUAL 0)
  message(FATAL_ERROR "Second CLI encode failed: ${second_error}")
endif()
execute_process(
  COMMAND
    "${GJXL_ENCODER}" --distance 1.0 --backend metal
    "${GJXL_SAMPLE}" "${metal}"
  RESULT_VARIABLE metal_result
  OUTPUT_VARIABLE metal_output
  ERROR_VARIABLE metal_error
)
if(NOT metal_result EQUAL 0)
  message(FATAL_ERROR "Forced Metal CLI encode failed: ${metal_error}")
endif()

file(SHA256 "${first}" first_hash)
file(SHA256 "${second}" second_hash)
file(SHA256 "${metal}" metal_hash)
if(NOT first_hash STREQUAL second_hash)
  message(FATAL_ERROR "CLI output is not deterministic")
endif()
if(NOT first_hash STREQUAL metal_hash)
  message(FATAL_ERROR "Forced Metal changed the CLI codestream")
endif()
set(expected_hash
  48abd331b4b4e37f0b158af86ef7c766c72ed760a51ce6903a415bf2544031c7)
if(NOT first_hash STREQUAL expected_hash)
  message(FATAL_ERROR
    "checked sample codestream hash changed: ${first_hash}")
endif()
foreach(expected "Encoded 17x13" "using CPU" "Strategies:"
                 "Final perceptual score:")
  string(FIND "${first_output}" "${expected}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR "CLI report is missing: ${expected}")
  endif()
endforeach()
string(FIND "${metal_output}" "using Metal" metal_found)
if(metal_found EQUAL -1)
  message(FATAL_ERROR "Forced Metal CLI report did not identify Metal")
endif()

file(GLOB temporary_outputs "${GJXL_TEST_DIR}/*.tmp.*")
if(temporary_outputs)
  message(FATAL_ERROR "CLI left temporary outputs: ${temporary_outputs}")
endif()
message(STATUS "CLI deterministic SHA-256: ${first_hash}")
