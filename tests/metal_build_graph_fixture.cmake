# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
# Inject only a consumer target. All shader/embedding rules under test come
# from the real production CMakeLists.txt, not a copied miniature graph.
cmake_language(DEFER CALL add_custom_target gjxl_metal_graph_test_consumer)
cmake_language(DEFER CALL add_dependencies gjxl_metal_graph_test_consumer
  gjxl_metal_shaders gjxl_embedded_metal_shaders)
if(GJXL_ENABLE_METAL_PROFILING)
  cmake_language(DEFER CALL add_dependencies gjxl_metal_graph_test_consumer
    gjxl_metal_profile_symbols)
endif()
