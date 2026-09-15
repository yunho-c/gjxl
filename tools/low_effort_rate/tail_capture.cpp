// Diagnostic only: serialize one immutable completed frame through libjxl.
#include "codestream/libjxl_tail_internal.h"
#include "codestream/writer_diagnostics_internal.h"
#include "codec/vardct_frame_view_internal.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

namespace gjxl {
Status CaptureRcaTail(const vardct_frame_internal::VarDctFrameView& frame) {
  static bool captured=false;
  const char* path=std::getenv("GJXL_RCA_TAIL_OUTPUT");
  if (!path || captured) return Status::Ok();
  captured=true;
  const char* effort=std::getenv("GJXL_RCA_TAIL_EFFORT");
  const char* distance=std::getenv("GJXL_RCA_TAIL_DISTANCE");
  if (!effort || !distance) return Status::InvalidArgument("Missing diagnostic tail settings");
  writer_diagnostics::tail_gaborish=frame.profile().loop_filter.gaborish;
  writer_diagnostics::tail_epf=frame.profile().loop_filter.epf_options.iterations;
  writer_diagnostics::tail_dc_smoothing=frame.profile().adaptive_dc_smoothing;
  codestream_internal::LibjxlTailOptions options{
    .effort=std::atoi(effort),.butteraugli_distance=std::strtof(distance,nullptr),.thread_count=8};
  codestream_internal::LibjxlTailStateAudit before,after;
  auto status=codestream_internal::AuditVarDctStateWithLibjxl(frame,options,&before);
  if (!status.ok()) return status;
  if (before.source!=before.copied) return Status::Internal("Tail state-copy mismatch");
  std::unique_ptr<codestream_internal::LibjxlTailContext> context;
  status=codestream_internal::CreateLibjxlTailContext(8,&context);
  if (!status.ok()) return status;
  std::vector<uint8_t> bytes,repeat;
  codestream_internal::LibjxlTailProfile profile;
  status=codestream_internal::EncodeVarDctCodestreamWithLibjxlContextProfiled(frame,options,*context,&bytes,&profile);
  if (!status.ok()) return status;
  const auto stats=writer_diagnostics::libjxl_stats;
  status=codestream_internal::EncodeVarDctCodestreamWithLibjxlContextProfiled(frame,options,*context,&repeat,&profile);
  if (!status.ok()) return status;
  if (bytes!=repeat) return Status::Internal("Tail repeat bytes differ");
  status=codestream_internal::AuditVarDctStateWithLibjxl(frame,options,&after);
  if (!status.ok()) return status;
  if (before.source!=after.source || after.source!=after.copied)
    return Status::Internal("Tail frame changed during serialization");
  std::filesystem::path folder(path);std::filesystem::create_directories(folder);
  std::ofstream output(folder/"tail.jxl",std::ios::binary);
  output.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());output.close();
  if (!output) return Status::Internal("Could not write tail diagnostic output");
  std::ofstream record(folder/"tail.json");
  record << "{\"bytes\":" << bytes.size() << ",\"extra_dc_precision\":"
    << unsigned(frame.profile().extra_dc_precision)
    << ",\"source_copy_equal\":true,\"repeat_bytes_equal\":true,\"frame_immutable\":true,\"stats\":"
    << (stats.empty()?"null":stats) << "}\n";
  return record?Status::Ok():Status::Internal("Could not write tail diagnostic record");
}
}
