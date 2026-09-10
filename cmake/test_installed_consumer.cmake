# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

if(NOT DEFINED GJXL_BUILD_DIR OR NOT DEFINED GJXL_SOURCE_DIR)
  message(FATAL_ERROR "GJXL_BUILD_DIR and GJXL_SOURCE_DIR are required")
endif()

if(NOT DEFINED GJXL_TEST_CONFIG OR GJXL_TEST_CONFIG STREQUAL "")
  set(GJXL_TEST_CONFIG Release)
endif()

set(test_root "${GJXL_BUILD_DIR}/installed-consumer-test")
set(staging_prefix "${test_root}/staging-prefix")
set(install_prefix "${test_root}/prefix")
set(consumer_source "${test_root}/consumer-source")
if(NOT DEFINED GJXL_INSTALL_INCLUDEDIR)
  set(GJXL_INSTALL_INCLUDEDIR include)
endif()
file(REMOVE_RECURSE "${test_root}")

# Keep the same configure/build/run checks for each supported consumer mode.
function(run_checked description)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${description} failed:\n${output}\n${error}")
  endif()
endfunction()

run_checked("gjxl installation"
  "${CMAKE_COMMAND}" --install "${GJXL_BUILD_DIR}"
  --prefix "${staging_prefix}" --config "${GJXL_TEST_CONFIG}")
# Verify relocation and keep the consumer fixture separate from source headers.
file(RENAME "${staging_prefix}" "${install_prefix}")
file(COPY "${GJXL_SOURCE_DIR}/tests/downstream/" DESTINATION "${consumer_source}")
file(COPY "${GJXL_SOURCE_DIR}/cmake/InstalledHeaders.cmake" DESTINATION "${consumer_source}")

include("${consumer_source}/InstalledHeaders.cmake")
set(installed_include "${install_prefix}/${GJXL_INSTALL_INCLUDEDIR}")
# This glob inspects the artifact; production installation uses only the list.
file(GLOB_RECURSE actual_headers RELATIVE "${installed_include}" "${installed_include}/*")
set(expected_headers ${GJXL_INSTALLED_HEADERS})
list(SORT expected_headers)
list(SORT actual_headers)
if(NOT actual_headers STREQUAL expected_headers)
  set(missing_headers ${expected_headers})
  list(REMOVE_ITEM missing_headers ${actual_headers})
  set(extra_headers ${actual_headers})
  list(REMOVE_ITEM extra_headers ${expected_headers})
  message(FATAL_ERROR
    "installed header boundary differs: missing=[${missing_headers}], unexpected=[${extra_headers}]")
endif()

file(GLOB_RECURSE package_files "${install_prefix}/*.cmake")
foreach(package_file IN LISTS package_files)
  file(READ "${package_file}" package_contents)
  foreach(forbidden_path IN ITEMS "${GJXL_SOURCE_DIR}" "${GJXL_BUILD_DIR}" "${staging_prefix}")
    string(FIND "${package_contents}" "${forbidden_path}" found)
    if(NOT found EQUAL -1)
      message(FATAL_ERROR "${package_file} leaks a source/build/staging path: ${forbidden_path}")
    endif()
  endforeach()
endforeach()

foreach(standard IN ITEMS 20 23)
  set(consumer_build "${test_root}/cxx${standard}")
  run_checked("downstream C++${standard} configure"
    "${CMAKE_COMMAND}" -S "${consumer_source}" -B "${consumer_build}"
    "-DCMAKE_PREFIX_PATH=${install_prefix}" "-DCMAKE_BUILD_TYPE=${GJXL_TEST_CONFIG}"
    "-DCMAKE_CXX_STANDARD=${standard}" -DCMAKE_CXX_STANDARD_REQUIRED=ON
    -DGJXL_CHECK_INSTALLED_HEADERS=ON)
  run_checked("downstream C++${standard} build"
    "${CMAKE_COMMAND}" --build "${consumer_build}" --config "${GJXL_TEST_CONFIG}" --parallel 4)
  foreach(consumer IN ITEMS gjxl_codec_consumer gjxl_codestream_consumer gjxl_c_consumer gjxl_domain_consumer)
    set(consumer_executable "${consumer_build}/${consumer}")
    if(NOT EXISTS "${consumer_executable}")
      set(consumer_executable "${consumer_build}/${GJXL_TEST_CONFIG}/${consumer}")
    endif()
    run_checked("downstream C++${standard} ${consumer}" "${consumer_executable}")
  endforeach()
endforeach()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${consumer_source}"
    -B "${test_root}/unsupported-cxx26" "-DCMAKE_PREFIX_PATH=${install_prefix}"
    -DCMAKE_CXX_STANDARD=26
  RESULT_VARIABLE unsupported_result OUTPUT_VARIABLE unsupported_output ERROR_VARIABLE unsupported_error
)
if(unsupported_result EQUAL 0 OR
   NOT "${unsupported_output}${unsupported_error}" MATCHES
       "GJXL storage contract requires C[+][+]20 or C[+][+]23")
  message(FATAL_ERROR
    "installed package did not reject unsupported C++26 at configure time:\n"
    "${unsupported_output}\n${unsupported_error}")
endif()

# A client requesting only the C ABI does not need the C++ interface probe.
set(c_abi_build "${test_root}/c-abi-cxx23")
run_checked("C ABI-only C++23 configure"
  "${CMAKE_COMMAND}" -S "${consumer_source}" -B "${c_abi_build}"
  "-DCMAKE_PREFIX_PATH=${install_prefix}" "-DCMAKE_BUILD_TYPE=${GJXL_TEST_CONFIG}"
  -DCMAKE_CXX_STANDARD=23 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DGJXL_TEST_C_ABI_ONLY=ON)
run_checked("C ABI-only C++23 build"
  "${CMAKE_COMMAND}" --build "${c_abi_build}" --config "${GJXL_TEST_CONFIG}")
set(c_abi_executable "${c_abi_build}/gjxl_c_abi_consumer")
if(NOT EXISTS "${c_abi_executable}")
  set(c_abi_executable "${c_abi_build}/${GJXL_TEST_CONFIG}/gjxl_c_abi_consumer")
endif()
run_checked("C ABI-only C++23 encode" "${c_abi_executable}")
