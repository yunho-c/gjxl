// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

// Deliberately compiled without the current API header: these are the two
// published pre-domain layouts, not current structs with a reduced size field.
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct GJXLContext GJXLContext;
typedef struct {
  uint32_t struct_size;
  int32_t backend;
} ContextV1;
typedef struct {
  uint32_t struct_size;
  int32_t backend;
  uint32_t num_cpu_threads;
} ContextV2;
extern int32_t gjxl_context_options_init(void *options, size_t caller_size);
extern int32_t gjxl_context_create(const void *options, GJXLContext **context);
extern void gjxl_context_destroy(GJXLContext *context);

int check_legacy_contexts(void) {
  struct {
    ContextV1 options;
    unsigned char canary[32];
  } v1;
  struct {
    ContextV2 options;
    unsigned char canary[32];
  } v2;
  memset(&v1, 0xa5, sizeof(v1));
  memset(&v2, 0xa5, sizeof(v2));
  if (gjxl_context_options_init(&v1.options, sizeof(v1.options)) != 0 ||
      gjxl_context_options_init(&v2.options, sizeof(v2.options)) != 0)
    return 0;
  for (size_t i = 0; i < sizeof(v1.canary); ++i)
    if (v1.canary[i] != 0xa5 || v2.canary[i] != 0xa5)
      return 0;
  if (v1.options.struct_size != 8 || v2.options.struct_size != 12 ||
      v2.options.num_cpu_threads != 0)
    return 0;
  v1.options.backend = 1; // GJXL_BACKEND_CPU, unchanged ABI value.
  v2.options.backend = 1;
  v2.options.num_cpu_threads = 1;
  GJXLContext *first = NULL;
  GJXLContext *second = NULL;
  const int good = gjxl_context_create(&v1.options, &first) == 0 && first != NULL &&
                   gjxl_context_create(&v2.options, &second) == 0 && second != NULL;
  gjxl_context_destroy(first);
  gjxl_context_destroy(second);
  return good;
}
