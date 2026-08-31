# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

cmake_minimum_required(VERSION 3.25)

project(gjxl_downstream_consumer LANGUAGES C CXX)

find_package(gjxl CONFIG REQUIRED COMPONENTS c)

add_executable(
  gjxl_codec_consumer
  butteraugli_consumer.cpp
)

target_link_libraries(
  gjxl_codec_consumer
  PRIVATE
    gjxl::codec
)

add_executable(
  gjxl_codestream_consumer
  codestream_consumer.cpp
)

target_link_libraries(
  gjxl_codestream_consumer
  PRIVATE
    gjxl::codestream
)

add_executable(
  gjxl_c_consumer
  c_api_consumer.c
)

set_target_properties(
  gjxl_c_consumer
  PROPERTIES
    LINKER_LANGUAGE CXX
)

target_compile_features(
  gjxl_c_consumer
  PRIVATE
    c_std_11
)

target_compile_options(
  gjxl_c_consumer
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

target_link_libraries(
  gjxl_c_consumer
  PRIVATE
    gjxl::c
)
