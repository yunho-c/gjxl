#pragma once
#include <cstdint>
#include <string_view>
namespace gjxl::metal_internal {
struct OverheadStats { uint64_t gpu_ns=0, encode_ns=0, resolve_ns=0, encoders=0; };
void SetOverheadControl(std::string_view mode);
void ResetOverheadStats();
OverheadStats GetOverheadStats();
}
