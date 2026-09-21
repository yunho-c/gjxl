from pathlib import Path
root=Path(__file__).parent/'source'
p=root/'src/gpu/metal/metal_submission.cpp'
s=p.read_text()
def replace(a,b,count=1):
    global s
    assert s.count(a)==count,(a[:100],s.count(a),count)
    s=s.replace(a,b)
replace('#include "gpu/metal/metal_backend_internal.h"', '#include "gpu/metal/metal_backend_internal.h"\n#include "gpu/metal/overhead_control.h"\n#include <chrono>')
replace('namespace gjxl::metal_internal {\nnamespace {', '''namespace gjxl::metal_internal {
namespace {
enum class OverheadControl { Full, Graph, Split, Record };
thread_local OverheadControl g_control = OverheadControl::Full;
thread_local OverheadStats g_overhead;
struct OverheadTimer {
  uint64_t& target;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  ~OverheadTimer() {
    target += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-begin).count();
  }
};
}
void SetOverheadControl(std::string_view mode) {
  g_control = mode == "graph" ? OverheadControl::Graph :
              mode == "split" ? OverheadControl::Split :
              mode == "record" ? OverheadControl::Record : OverheadControl::Full;
}
void ResetOverheadStats() { g_overhead = {}; }
OverheadStats GetOverheadStats() { return g_overhead; }
namespace {''')
replace('      command_buffer_->waitUntilCompleted();', '''      command_buffer_->waitUntilCompleted();
      uint64_t measured_gpu_ns = 0;
      if (GpuDuration(&measured_gpu_ns).ok()) g_overhead.gpu_ns += measured_gpu_ns;''')
replace('  Status GpuProfile(GpuSubmissionProfile* profile) {', '''  Status GpuProfile(GpuSubmissionProfile* profile) {
    OverheadTimer resolution_timer{g_overhead.resolve_ns};''')
replace('    if (counter_sample_buffer_.get() == nullptr || profile_.stages.empty()) {', '''    if (g_control == OverheadControl::Graph || g_control == OverheadControl::Split) {
      GpuSubmissionProfile dummy;
      dummy.stages.push_back({.stage_id="overhead.control"});
      Status status = GpuDuration(&dummy.command_buffer_gpu_nanoseconds);
      if (!status.ok()) return status;
      *profile = std::move(dummy);
      return Status::Ok();
    }
    if ((g_control != OverheadControl::Record && counter_sample_buffer_.get() == nullptr) || profile_.stages.empty()) {''')
replace('    NS::Data* resolved = counter_sample_buffer_->resolveCounterRange(\n      NS::Range::Make(0, static_cast<NS::UInteger>(sample_count)));', '''    NS::Data* resolved = g_control == OverheadControl::Record ? nullptr :
      counter_sample_buffer_->resolveCounterRange(
        NS::Range::Make(0, static_cast<NS::UInteger>(sample_count)));''')
replace('    if (resolved == nullptr || resolved->bytes() == nullptr ||\n        resolved->length() < required_bytes) {', '''    if (g_control != OverheadControl::Record &&
        (resolved == nullptr || resolved->bytes() == nullptr ||
         resolved->length() < required_bytes)) {''')
replace('      const auto empty_dispatch = [](const GpuDispatchProfile& dispatch) {', '''      if (g_control == OverheadControl::Record) {
        Status status = GpuDuration(&candidate.command_buffer_gpu_nanoseconds);
        if (!status.ok()) return status;
        *profile = std::move(candidate);
        return Status::Ok();
      }
      const auto empty_dispatch = [](const GpuDispatchProfile& dispatch) {''')
# Count normal encoders, preserving the normal callback and original command buffer.
needle='Status MetalBackend::SubmitCompute(\n'
idx=s.index(needle); idx=s.index('  if (submission == nullptr)',idx)
s=s[:idx]+'  OverheadTimer encoding_timer{g_overhead.encode_ns};\n  ++g_overhead.encoders;\n'+s[idx:]
idx=s.index('Status MetalBackend::SubmitComputeProfiled('); idx=s.index('  if (submission == nullptr)',idx)
s=s[:idx]+'  OverheadTimer encoding_timer{g_overhead.encode_ns};\n'+s[idx:]
# The two controls traverse exactly the same input stage graph. They do not
# collect a dispatch graph or timestamp samples. Graph uses one encoder; split
# uses the normal profiling encoder boundaries. Dummy output is never plotted.
needle='  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());\n  MTL::CounterSet* timestamp_set = FindTimestampCounterSet(device_.get());'
replace(needle, '''  if (g_control == OverheadControl::Graph || g_control == OverheadControl::Split) {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto* raw = command_queue_->commandBuffer();
    if (!raw) return Status::SubmissionFailed("Overhead control command buffer failed");
    auto command = NS::RetainPtr(raw);
    raw->setLabel(NS::String::string(label, NS::UTF8StringEncoding));
    MTL::ComputeCommandEncoder* encoder = nullptr;
    for (const auto& stage : stages) {
      if (!encoder) {
        encoder = raw->computeCommandEncoder(); ++g_overhead.encoders;
        if (!encoder) return Status::SubmissionFailed("Overhead control encoder failed");
      }
      encoder->setLabel(NS::String::string(stage.stage_id, NS::UTF8StringEncoding));
      stage.encode(*this, encoder, stage.context);
      if (g_control == OverheadControl::Split) { encoder->endEncoding(); encoder=nullptr; }
    }
    if (encoder) encoder->endEncoding();
    std::unique_ptr<GpuSubmission> pending(new MetalSubmission(
        command, command_queue_, device_, fail_completion));
    raw->commit(); RecordCommittedSubmission(); *submission=std::move(pending);
    return Status::Ok();
  }
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  MTL::CounterSet* timestamp_set = FindTimestampCounterSet(device_.get());''')
replace('  MTL::CounterSampleBuffer* raw_sample_buffer =\n    device_->newCounterSampleBuffer(sample_descriptor.get(), &sample_error);\n  if (raw_sample_buffer == nullptr) {', '''  MTL::CounterSampleBuffer* raw_sample_buffer = g_control == OverheadControl::Record ? nullptr :
    device_->newCounterSampleBuffer(sample_descriptor.get(), &sample_error);
  if (g_control != OverheadControl::Record && raw_sample_buffer == nullptr) {''')
# This condition appears only in SubmitComputeProfiled (resolution uses profiling_mode_).
replace('    if (mode == GpuProfilingMode::kStage) {', '    if (mode == GpuProfilingMode::kStage && g_control != OverheadControl::Record) {')
replace('    const ScopedComputeEncoding encoding_scope(encoder);', '    ++g_overhead.encoders;\n    const ScopedComputeEncoding encoding_scope(encoder);')
p.write_text(s)
