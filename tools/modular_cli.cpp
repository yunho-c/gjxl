// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "modular_cli.h"
#include "io/pnm.h"
#include <iostream>
#include <charconv>
int RunModularCli(int argc, char **argv,
                  gjxl::Status (*write)(const std::filesystem::path &, std::span<const uint8_t>)) {
  gjxl::ModularEncodingOptions options;
  std::filesystem::path input, output;
  bool srgb = false, modular = false;
  auto usage = [] {
    std::cerr << "Usage: gjxl_encode --modular --input-color-space srgb [--search] [--entropy "
                 "prefix|ans] "
                 "[--threads 0..256] [--backend auto|cpu] INPUT.pgm|ppm|pam OUTPUT.jxl\n";
    return 1;
  };
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--modular" && !modular)
      modular = true;
    else if (arg == "--search")
      options.search = true;
    else if (arg == "--input-color-space" && i + 1 < argc) {
      if (std::string_view(argv[++i]) != "srgb")
        return usage();
      srgb = true;
    } else if (arg == "--entropy" && i + 1 < argc) {
      const std::string_view value = argv[++i];
      if (value == "prefix")
        options.entropy = gjxl::ModularEntropy::kPrefix;
      else if (value == "ans")
        options.entropy = gjxl::ModularEntropy::kAns;
      else
        return usage();
    } else if (arg == "--threads" && i + 1 < argc) {
      const std::string_view value = argv[++i];
      auto r = std::from_chars(value.data(), value.data() + value.size(), options.cpu_thread_count);
      if (r.ec != std::errc{} || r.ptr != value.data() + value.size() ||
          options.cpu_thread_count > 256)
        return usage();
    } else if (arg == "--backend" && i + 1 < argc) {
      const std::string_view value = argv[++i];
      if (value == "auto")
        options.backend = gjxl::ModularBackend::kAutomatic;
      else if (value == "cpu")
        options.backend = gjxl::ModularBackend::kCpu;
      else if (value == "metal")
        options.backend = gjxl::ModularBackend::kMetal;
      else if (value == "cuda")
        options.backend = gjxl::ModularBackend::kCuda;
      else
        return usage();
    } else if (arg.starts_with("--"))
      return usage();
    else if (input.empty())
      input = argv[i];
    else if (output.empty())
      output = argv[i];
    else
      return usage();
  }
  if (!modular || !srgb || input.empty() || output.empty())
    return usage();
  gjxl::io::IntegerImage image;
  auto status = gjxl::io::ReadPnm(input, &image);
  std::vector<uint8_t> encoded;
  gjxl::ModularEncodingSummary summary;
  if (status.ok())
    status = gjxl::EncodeModularImage(image.view(), options, &encoded, &summary);
  if (status.ok())
    status = write(output, encoded);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 1;
  }
  std::cout << "Encoded " << summary.extent.width << 'x' << summary.extent.height
            << " lossless Modular to " << summary.encoded_bytes << " bytes on CPU\n";
  return 0;
}
