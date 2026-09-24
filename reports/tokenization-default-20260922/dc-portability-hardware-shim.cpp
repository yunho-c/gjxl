// Review-only link-time shim: simulate the scheduler's hardware thread query.
#include <cstdlib>
#include <thread>
unsigned std::thread::hardware_concurrency() noexcept {
  const char* value = std::getenv("GJXL_REVIEW_HARDWARE_THREADS");
  return value ? static_cast<unsigned>(std::strtoul(value, nullptr, 10)) : 2;
}
