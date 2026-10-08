// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "io/pnm.h"
#include <fstream>
#include <map>
#include <cctype>
#include <charconv>
namespace gjxl::io {
namespace {
Status Invalid() {
  return Status::InvalidArgument("Unsupported or malformed integer PGM/PPM/PAM input");
}
bool Line(std::istream &in, std::string *out) {
  out->clear();
  for (size_t i = 0; i < 4096; ++i) {
    const int c = in.get();
    if (c == EOF)
      return false;
    if (c == '\n')
      return true;
    out->push_back(static_cast<char>(c));
  }
  return false;
}
bool Token(std::istream &in, std::string *out) {
  out->clear();
  for (;;) {
    int c = in.peek();
    if (c == EOF)
      return false;
    if (std::isspace(static_cast<unsigned char>(c))) {
      in.get();
      continue;
    }
    if (c == '#') {
      std::string comment;
      if (!Line(in, &comment))
        return false;
      continue;
    }
    break;
  }
  while (out->size() < 32) {
    int c = in.get();
    if (c == EOF)
      return false;
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (c == '\r' && in.peek() == '\n')
        in.get();
      return !out->empty();
    }
    out->push_back(static_cast<char>(c));
  }
  return false;
}
bool Number(const std::string &s, size_t *out) {
  const auto r = std::from_chars(s.data(), s.data() + s.size(), *out);
  return r.ec == std::errc{} && r.ptr == s.data() + s.size() && *out != 0;
}
} // namespace
Status ReadPnm(const std::filesystem::path &path, IntegerImage *out) try {
  if (!out)
    return Invalid();
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return Status::InvalidArgument("Cannot open integer image");
  std::string magic, text;
  if (!Token(in, &magic))
    return Invalid();
  size_t width = 0, height = 0, depth = 0, maxval = 0;
  if (magic == "P5" || magic == "P6") {
    depth = magic == "P5" ? 1 : 3;
    if (!Token(in, &text) || !Number(text, &width) || !Token(in, &text) || !Number(text, &height) ||
        !Token(in, &text) || !Number(text, &maxval))
      return Invalid();
  } else if (magic == "P7") {
    std::map<std::string, std::string> fields;
    size_t header_bytes = 0;
    while (Line(in, &text)) {
      header_bytes += text.size();
      if (header_bytes > 4096)
        return Invalid();
      if (!text.empty() && text.back() == '\r')
        text.pop_back();
      if (text == "ENDHDR")
        break;
      if (text.empty() || text[0] == '#')
        continue;
      const auto sep = text.find_first_of(" \t");
      if (sep == std::string::npos)
        return Invalid();
      auto value = text.substr(sep + 1);
      const auto first = value.find_first_not_of(" \t");
      if (first == std::string::npos)
        return Invalid();
      value.erase(0, first);
      if (!fields.emplace(text.substr(0, sep), value).second)
        return Invalid();
    }
    if (text != "ENDHDR" || fields.size() != 5 || !Number(fields["WIDTH"], &width) ||
        !Number(fields["HEIGHT"], &height) || !Number(fields["DEPTH"], &depth) ||
        !Number(fields["MAXVAL"], &maxval))
      return Invalid();
    if ((depth == 1 && fields["TUPLTYPE"] != "GRAYSCALE") ||
        (depth == 3 && fields["TUPLTYPE"] != "RGB") ||
        (depth == 4 && fields["TUPLTYPE"] != "RGB_ALPHA"))
      return Invalid();
  } else
    return Invalid();
  if ((depth != 1 && depth != 3 && depth != 4) || (maxval != 255 && maxval != 65535) ||
      width > UINT32_MAX || height > UINT32_MAX)
    return Invalid();
  const size_t sample_bytes = maxval == 255 ? 1 : 2;
  if (width > SIZE_MAX / (depth * sample_bytes))
    return Invalid();
  const size_t stride = width * depth * sample_bytes;
  if (height > SIZE_MAX / stride)
    return Invalid();
  const size_t count = stride * height;
  const auto start = in.tellg();
  in.seekg(0, std::ios::end);
  const auto end = in.tellg();
  if (start < 0 || end < start || static_cast<uint64_t>(end - start) != count ||
      count > PTRDIFF_MAX)
    return Invalid(); // Also reject concatenated images and trailing data.
  in.seekg(start);
  IntegerImage image;
  image.extent = {width, height};
  image.row_stride = stride;
  image.format =
      depth == 1   ? (sample_bytes == 1 ? ModularPixelFormat::kGray8 : ModularPixelFormat::kGray16)
      : depth == 3 ? (sample_bytes == 1 ? ModularPixelFormat::kRgb8 : ModularPixelFormat::kRgb16)
                   : (sample_bytes == 1 ? ModularPixelFormat::kRgba8 : ModularPixelFormat::kRgba16);
  image.bytes.resize(count);
  if (!in.read(reinterpret_cast<char *>(image.bytes.data()), static_cast<std::streamsize>(count)))
    return Invalid();
  *out = std::move(image);
  return Status::Ok();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Integer image allocation failed");
} catch (const std::exception &) {
  return Invalid();
}
} // namespace gjxl::io
