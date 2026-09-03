# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

foreach(required IN ITEMS GJXL_LIBJXL_SOURCE GJXL_LIBJXL_BUILD
                          GJXL_LIBJXL_REVISION GJXL_GENERATOR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "Missing required ${required}")
  endif()
endforeach()

if(NOT DEFINED GJXL_EXECUTABLE_SUFFIX)
  set(GJXL_EXECUTABLE_SUFFIX "")
endif()

set(GJXL_COMPILER_ARGUMENTS)
if(DEFINED GJXL_C_COMPILER AND NOT "${GJXL_C_COMPILER}" STREQUAL "")
  list(APPEND GJXL_COMPILER_ARGUMENTS
    "-DCMAKE_C_COMPILER=${GJXL_C_COMPILER}")
endif()
if(DEFINED GJXL_CXX_COMPILER AND NOT "${GJXL_CXX_COMPILER}" STREQUAL "")
  list(APPEND GJXL_COMPILER_ARGUMENTS
    "-DCMAKE_CXX_COMPILER=${GJXL_CXX_COMPILER}")
endif()

find_program(GJXL_GIT_EXECUTABLE git REQUIRED)
execute_process(
  COMMAND "${GJXL_GIT_EXECUTABLE}" -C "${GJXL_LIBJXL_SOURCE}" rev-parse HEAD
  RESULT_VARIABLE revision_result
  OUTPUT_VARIABLE actual_revision
  ERROR_VARIABLE revision_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT revision_result EQUAL 0)
  message(FATAL_ERROR "Unable to inspect pinned libjxl: ${revision_error}")
endif()
if(NOT actual_revision STREQUAL GJXL_LIBJXL_REVISION)
  message(FATAL_ERROR
    "libjxl revision ${actual_revision} does not match ${GJXL_LIBJXL_REVISION}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -S "${GJXL_LIBJXL_SOURCE}"
    -B "${GJXL_LIBJXL_BUILD}"
    -G "${GJXL_GENERATOR}"
    ${GJXL_COMPILER_ARGUMENTS}
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_TESTING=OFF
    -DJPEGXL_ENABLE_TOOLS=ON
    -DJPEGXL_ENABLE_DEVTOOLS=OFF
    -DJPEGXL_ENABLE_BENCHMARK=OFF
    -DJPEGXL_ENABLE_EXAMPLES=OFF
    -DJPEGXL_ENABLE_JNI=OFF
    -DJPEGXL_ENABLE_MANPAGES=OFF
    -DJPEGXL_ENABLE_DOXYGEN=OFF
    -DJPEGXL_ENABLE_SJPEG=OFF
    -DJPEGXL_ENABLE_OPENEXR=OFF
    -DJPEGXL_ENABLE_VIEWERS=OFF
    -DJPEGXL_ENABLE_TRANSCODE_JPEG=OFF
    -DHWY_ENABLE_TESTS=OFF
    -DHWY_ENABLE_EXAMPLES=OFF
  RESULT_VARIABLE configure_result
)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR "Pinned libjxl configuration failed")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${GJXL_LIBJXL_BUILD}"
          --config Release --target djxl jxlinfo
  RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR "Pinned djxl/jxlinfo build failed")
endif()

if(WIN32)
  file(GLOB GJXL_PINNED_RUNTIME_LIBRARIES
    "${GJXL_LIBJXL_BUILD}/lib/*.dll"
    "${GJXL_LIBJXL_BUILD}/third_party/brotli/*.dll")
  if(NOT GJXL_PINNED_RUNTIME_LIBRARIES)
    message(FATAL_ERROR "Pinned libjxl runtime DLLs were not produced")
  endif()
  file(COPY ${GJXL_PINNED_RUNTIME_LIBRARIES}
    DESTINATION "${GJXL_LIBJXL_BUILD}/tools")
endif()

foreach(tool IN ITEMS djxl jxlinfo)
  if(NOT EXISTS
      "${GJXL_LIBJXL_BUILD}/tools/${tool}${GJXL_EXECUTABLE_SUFFIX}")
    message(FATAL_ERROR "Pinned ${tool} was not produced")
  endif()
endforeach()
