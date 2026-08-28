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
execute_process(
  COMMAND
    "${GJXL_ENCODER}" --distance 1.0 --metal-aq throughput
    "${GJXL_SAMPLE}" "${sentinel}"
  RESULT_VARIABLE implicit_throughput_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(implicit_throughput_result EQUAL 0)
  message(FATAL_ERROR "CLI accepted throughput AQ without forced Metal")
endif()
file(READ "${sentinel}" sentinel_contents)
if(NOT sentinel_contents STREQUAL "unchanged")
  message(FATAL_ERROR "Invalid throughput request changed an existing output")
endif()

execute_process(
  COMMAND
    "${GJXL_ENCODER}" --distance 1.0 --target-bytes 280
    "${GJXL_SAMPLE}" "${sentinel}"
  RESULT_VARIABLE conflicting_rate_control_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(conflicting_rate_control_result EQUAL 0)
  message(FATAL_ERROR "CLI accepted conflicting rate-control modes")
endif()
file(READ "${sentinel}" sentinel_contents)
if(NOT sentinel_contents STREQUAL "unchanged")
  message(FATAL_ERROR "Conflicting rate control changed an existing output")
endif()

execute_process(
  COMMAND
    "${GJXL_ENCODER}" --target-bytes 280 --size-tolerance 1.01
    "${GJXL_SAMPLE}" "${sentinel}"
  RESULT_VARIABLE invalid_tolerance_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(invalid_tolerance_result EQUAL 0)
  message(FATAL_ERROR "CLI accepted an invalid size tolerance")
endif()
file(READ "${sentinel}" sentinel_contents)
if(NOT sentinel_contents STREQUAL "unchanged")
  message(FATAL_ERROR "Invalid size tolerance changed an existing output")
endif()

execute_process(
  COMMAND
    "${GJXL_ENCODER}" --target-bytes 280 --size-selection invalid
    "${GJXL_SAMPLE}" "${sentinel}"
  RESULT_VARIABLE invalid_selection_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(invalid_selection_result EQUAL 0)
  message(FATAL_ERROR "CLI accepted an invalid size-selection policy")
endif()
file(READ "${sentinel}" sentinel_contents)
if(NOT sentinel_contents STREQUAL "unchanged")
  message(FATAL_ERROR "Invalid size selection changed an existing output")
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
set(throughput "${GJXL_TEST_DIR}/throughput.jxl")
set(throughput_repeat "${GJXL_TEST_DIR}/throughput-repeat.jxl")
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
execute_process(
  COMMAND
    "${GJXL_ENCODER}" --distance 1.0 --backend metal
    --metal-aq throughput "${GJXL_SAMPLE}" "${throughput}"
  RESULT_VARIABLE throughput_result
  OUTPUT_VARIABLE throughput_output
  ERROR_VARIABLE throughput_error
)
if(NOT throughput_result EQUAL 0)
  message(FATAL_ERROR "Throughput CLI encode failed: ${throughput_error}")
endif()
execute_process(
  COMMAND
    "${GJXL_ENCODER}" --distance 1.0 --backend metal
    --metal-aq throughput "${GJXL_SAMPLE}" "${throughput_repeat}"
  RESULT_VARIABLE throughput_repeat_result
  OUTPUT_QUIET
  ERROR_VARIABLE throughput_repeat_error
)
if(NOT throughput_repeat_result EQUAL 0)
  message(FATAL_ERROR
    "Repeated throughput CLI encode failed: ${throughput_repeat_error}")
endif()

file(SHA256 "${first}" first_hash)
file(SHA256 "${second}" second_hash)
file(SHA256 "${metal}" metal_hash)
file(SHA256 "${throughput}" throughput_hash)
file(SHA256 "${throughput_repeat}" throughput_repeat_hash)
if(NOT first_hash STREQUAL second_hash)
  message(FATAL_ERROR "CLI output is not deterministic")
endif()
if(NOT first_hash STREQUAL metal_hash)
  message(FATAL_ERROR "Forced Metal changed the CLI codestream")
endif()
if(NOT throughput_hash STREQUAL throughput_repeat_hash)
  message(FATAL_ERROR "Throughput CLI output is not deterministic")
endif()
set(expected_hash
  0cf93a5c330f19c5faad0cf20d42c446f820adfbc685aa9a29ae5251643da5a8)
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
string(FIND "${throughput_output}" "Metal throughput AQ" throughput_found)
if(throughput_found EQUAL -1)
  message(FATAL_ERROR "Throughput CLI report did not identify its policy")
endif()

set(target_bytes_first "${GJXL_TEST_DIR}/target-bytes-first.jxl")
set(target_bytes_second "${GJXL_TEST_DIR}/target-bytes-second.jxl")
set(target_bpp "${GJXL_TEST_DIR}/target-bpp.jxl")
set(target_closest "${GJXL_TEST_DIR}/target-closest.jxl")
foreach(target_output IN ITEMS
        "${target_bytes_first}" "${target_bytes_second}")
  execute_process(
    COMMAND
      "${GJXL_ENCODER}" --target-bytes 280 --size-tolerance 0.1
      --max-attempts 8 --backend cpu "${GJXL_SAMPLE}" "${target_output}"
    RESULT_VARIABLE target_result
    OUTPUT_VARIABLE target_report
    ERROR_VARIABLE target_error
  )
  if(NOT target_result EQUAL 0)
    message(FATAL_ERROR "Target-byte CLI encode failed: ${target_error}")
  endif()
  foreach(expected "for target 280 bytes (met"
                   "selected Butteraugli target" "in 7 attempts"
                   "using CPU")
    string(FIND "${target_report}" "${expected}" found)
    if(found EQUAL -1)
      message(FATAL_ERROR "Target-byte CLI report is missing: ${expected}")
    endif()
  endforeach()
endforeach()
file(SIZE "${target_bytes_first}" target_size)
if(target_size GREATER 280 OR target_size LESS 252)
  message(FATAL_ERROR
    "Target-byte output ${target_size} is outside [252, 280]")
endif()
file(SHA256 "${target_bytes_first}" target_bytes_first_hash)
file(SHA256 "${target_bytes_second}" target_bytes_second_hash)
if(NOT target_bytes_first_hash STREQUAL target_bytes_second_hash)
  message(FATAL_ERROR "Target-byte CLI output is not deterministic")
endif()

# 10.14 * (17 * 13) / 8 floors to the same 280-byte budget.
execute_process(
  COMMAND
    "${GJXL_ENCODER}" --target-bpp 10.14 --size-tolerance 0.1
    --max-attempts 8 --backend cpu "${GJXL_SAMPLE}" "${target_bpp}"
  RESULT_VARIABLE target_bpp_result
  OUTPUT_VARIABLE target_bpp_report
  ERROR_VARIABLE target_bpp_error
)
if(NOT target_bpp_result EQUAL 0)
  message(FATAL_ERROR "Target-BPP CLI encode failed: ${target_bpp_error}")
endif()
string(FIND "${target_bpp_report}" "10.14 bpp / 280 bytes (met" bpp_found)
if(bpp_found EQUAL -1)
  message(FATAL_ERROR "Target-BPP CLI report did not expose its byte budget")
endif()
file(SHA256 "${target_bpp}" target_bpp_hash)
if(NOT target_bpp_hash STREQUAL target_bytes_first_hash)
  message(FATAL_ERROR "Equivalent byte and BPP targets diverged")
endif()

execute_process(
  COMMAND
    "${GJXL_ENCODER}" --target-bytes 280 --size-tolerance 0.1
    --max-attempts 8 --size-selection closest --backend cpu
    "${GJXL_SAMPLE}" "${target_closest}"
  RESULT_VARIABLE target_closest_result
  OUTPUT_VARIABLE target_closest_report
  ERROR_VARIABLE target_closest_error
)
if(NOT target_closest_result EQUAL 0)
  message(FATAL_ERROR
    "Closest-absolute CLI encode failed: ${target_closest_error}")
endif()
foreach(expected "for target 280 bytes (met" "0 failed" "using CPU")
  string(FIND "${target_closest_report}" "${expected}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR
      "Closest-absolute CLI report is missing: ${expected}")
  endif()
endforeach()

set(maximum_error_first "${GJXL_TEST_DIR}/maximum-error-first.jxl")
set(maximum_error_second "${GJXL_TEST_DIR}/maximum-error-second.jxl")
foreach(maximum_error_output IN ITEMS
        "${maximum_error_first}" "${maximum_error_second}")
  execute_process(
    COMMAND
      "${GJXL_ENCODER}" --maximum-error 0.1 0.1 0.1 --backend cpu
      "${GJXL_SAMPLE}" "${maximum_error_output}"
    RESULT_VARIABLE maximum_error_result
    OUTPUT_VARIABLE maximum_error_report
    ERROR_VARIABLE maximum_error_error
  )
  if(NOT maximum_error_result EQUAL 0)
    message(FATAL_ERROR
      "Maximum-error CLI encode failed: ${maximum_error_error}")
  endif()
  foreach(expected "under maximum-error control (0.1,0.1,0.1"
                   "met in 6 evaluations" "using CPU")
    string(FIND "${maximum_error_report}" "${expected}" found)
    if(found EQUAL -1)
      message(FATAL_ERROR
        "Maximum-error CLI report is missing: ${expected}")
    endif()
  endforeach()
endforeach()
file(SHA256 "${maximum_error_first}" maximum_error_first_hash)
file(SHA256 "${maximum_error_second}" maximum_error_second_hash)
if(NOT maximum_error_first_hash STREQUAL maximum_error_second_hash)
  message(FATAL_ERROR "Maximum-error CLI output is not deterministic")
endif()
set(expected_maximum_error_hash
  4ffe300d01d3ae46a669cd3c06267442227de7f5e0474b39cdeab5cb22dbbcf0)
if(NOT maximum_error_first_hash STREQUAL expected_maximum_error_hash)
  message(FATAL_ERROR
    "Maximum-error sample hash changed: ${maximum_error_first_hash}")
endif()

file(GLOB temporary_outputs "${GJXL_TEST_DIR}/*.tmp.*")
if(temporary_outputs)
  message(FATAL_ERROR "CLI left temporary outputs: ${temporary_outputs}")
endif()
message(STATUS "CLI deterministic SHA-256: ${first_hash}")
