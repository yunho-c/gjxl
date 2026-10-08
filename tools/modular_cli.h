// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "gjxl/modular.hpp"
#include <filesystem>
int RunModularCli(int argc, char **argv,
                  gjxl::Status (*write)(const std::filesystem::path &, std::span<const uint8_t>));
