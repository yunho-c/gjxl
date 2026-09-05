# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

foreach(required IN ITEMS GJXL_LIBJXL_SOURCE GJXL_LIBJXL_BUILD
                          GJXL_LIBJXL_REVISION GJXL_GENERATOR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "Missing required ${required}")
  endif()
endforeach()

find_program(GJXL_GIT_EXECUTABLE git REQUIRED)
set(reference_tools djxl jxlinfo)
set(reference_devtools OFF)
set(reference_build_options)
if(GJXL_LIBJXL_QUALITY_TOOLS)
  list(APPEND reference_tools cjxl butteraugli_main)
  set(reference_devtools ON)
  list(APPEND reference_build_options -DBUILD_SHARED_LIBS=OFF)
endif()

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
if(GJXL_LIBJXL_QUALITY_TOOLS)
  execute_process(
    COMMAND "${GJXL_GIT_EXECUTABLE}" -C "${GJXL_LIBJXL_SOURCE}"
      diff --quiet HEAD --
    RESULT_VARIABLE reference_dirty
  )
  if(NOT reference_dirty EQUAL 0)
    message(FATAL_ERROR "Pinned libjxl reference tree has tracked modifications")
  endif()
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -S "${GJXL_LIBJXL_SOURCE}"
    -B "${GJXL_LIBJXL_BUILD}"
    -G "${GJXL_GENERATOR}"
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_TESTING=OFF
    -DJPEGXL_ENABLE_TOOLS=ON
    -DJPEGXL_ENABLE_DEVTOOLS=${reference_devtools}
    ${reference_build_options}
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
          --config Release --target ${reference_tools} --parallel 8
  RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR "Pinned djxl/jxlinfo build failed")
endif()

foreach(tool IN LISTS reference_tools)
  if(NOT EXISTS "${GJXL_LIBJXL_BUILD}/tools/${tool}")
    message(FATAL_ERROR "Pinned ${tool} was not produced")
  endif()
endforeach()

if(GJXL_LIBJXL_QUALITY_TOOLS)
  foreach(tool IN ITEMS cjxl djxl butteraugli_main)
    file(SHA256 "${GJXL_LIBJXL_BUILD}/tools/${tool}" ${tool}_sha256)
  endforeach()
  file(WRITE "${GJXL_LIBJXL_BUILD}/quality-reference.json"
    "{\n  \"revision\": \"${actual_revision}\",\n  \"tools\": {\n"
    "    \"cjxl\": \"${cjxl_sha256}\",\n"
    "    \"djxl\": \"${djxl_sha256}\",\n"
    "    \"butteraugli\": \"${butteraugli_main_sha256}\"\n  }\n}\n")
endif()
