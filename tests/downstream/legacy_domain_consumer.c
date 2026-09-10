// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

// Deliberately compiled without today's header. These are the shipped v1
// declarations; canaries check that the newer library respects their sizes.
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct GJXLExecutionDomain GJXLExecutionDomain;
typedef struct {
  uint32_t struct_size;
  uint64_t managed_memory_bytes;
} DomainOptionsV1;
typedef struct {
  uint32_t struct_size;
  uint64_t live_requested_bytes, live_capacity_bytes, idle_capacity_bytes;
  uint64_t reserved_unbacked_bytes, peak_backing_bytes, peak_committed_bytes;
  uint64_t active_reservations, waiting_requests;
} DomainSnapshotV1;
extern int32_t gjxl_execution_domain_options_init(DomainOptionsV1*, size_t);
extern int32_t gjxl_execution_domain_create(const DomainOptionsV1*, GJXLExecutionDomain**);
extern int32_t gjxl_execution_domain_snapshot(const GJXLExecutionDomain*, DomainSnapshotV1*, size_t);
extern void gjxl_execution_domain_destroy(GJXLExecutionDomain*);

int check_legacy_domains(void) {
  struct { DomainOptionsV1 value; unsigned char canary[16]; } options;
  struct { DomainSnapshotV1 value; unsigned char canary[16]; } snapshot;
  unsigned char canary[16];
  memset(&options, 0xa5, sizeof(options));
  memset(&snapshot, 0xa5, sizeof(snapshot));
  memset(canary, 0xa5, sizeof(canary));
  if (gjxl_execution_domain_options_init(&options.value, sizeof(options.value)) != 0 ||
      options.value.struct_size != sizeof(options.value) || options.value.managed_memory_bytes != 0 ||
      memcmp(options.canary, canary, sizeof(canary)) != 0) return 0;
  GJXLExecutionDomain* domain = NULL;
  if (gjxl_execution_domain_create(&options.value, &domain) != 0 || domain == NULL) return 0;
  const int good = gjxl_execution_domain_snapshot(domain, &snapshot.value, sizeof(snapshot.value)) == 0 &&
    snapshot.value.struct_size == sizeof(snapshot.value) && snapshot.value.active_reservations == 0 &&
    snapshot.value.waiting_requests == 0 && snapshot.value.peak_committed_bytes == 0 &&
    memcmp(snapshot.canary, canary, sizeof(canary)) == 0;
  gjxl_execution_domain_destroy(domain);
  return good;
}
