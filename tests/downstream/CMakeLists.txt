# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

cmake_minimum_required(VERSION 3.25)

project(gjxl_downstream_consumer LANGUAGES CXX)

find_package(gjxl CONFIG REQUIRED)

add_executable(
  gjxl_codec_consumer
  butteraugli_consumer.cpp
)

target_link_libraries(
  gjxl_codec_consumer
  PRIVATE
    gjxl::codec
)
