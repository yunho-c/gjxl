// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Complete-call comparison of the same histogram policy through explicit
// routes.
#include "codec/modular/profile.h"
#include "codestream/modular/reference_learning_profile.h"
#include "codestream/modular/reference_workflow.h"
#include "codestream/workflow_admission_scope.h"
#include "codestream/workflow_internal.h"
#include "core/thread_budget.h"
#include "histogram_profile_json.h"
#include "io/pnm.h"
#include "modular_reference.h"
#if GJXL_HYBRID_PROFILE_METAL
#include "gpu/metal/metal_backend.h"
#endif
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
namespace reference = gjxl::test::modular_reference;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
template <typename T> void Array(std::ostream &out, const T &a) {
  out << '[';
  for (size_t i = 0; i < a.size(); ++i) {
    if (i)
      out << ',';
    out << a[i];
  }
  out << ']';
}
} // namespace
int main(int argc, char **argv) try {
  // Named controls compose the independent experiments. Keep the three old
  // positional tails accepted so archived qualification runners still work.
  std::string row_mode = "raster", row_diagnostics = "rows-off";
  std::string learning_schedule = "cohorts", feature_window = "8";
  std::string gpu_profile_mode = "gpu-off", histogram_shards, active_properties;
  bool private_flags = false;
  int positional = 1;
  std::vector<std::string_view> named;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (!arg.starts_with("--")) {
      argv[positional++] = argv[i];
      continue;
    }
    const auto equal = arg.find('=');
    Check(equal != std::string_view::npos, "Named profile controls require =VALUE");
    const auto key = arg.substr(0, equal);
    Check(std::find(named.begin(), named.end(), key) == named.end(), "Duplicate profile control");
    named.push_back(key);
    const auto value = arg.substr(equal + 1);
    if (key == "--sample-rows") row_mode = value;
    else if (key == "--row-diagnostics") row_diagnostics = value;
    else if (key == "--learning-schedule") learning_schedule = value;
    else if (key == "--feature-window") feature_window = value;
    else if (key == "--gpu-profile") gpu_profile_mode = value;
    else if (key == "--histogram-shards") histogram_shards = value;
    else if (key == "--active-properties") active_properties = value;
    else Check(false, "Unknown named profile control");
    private_flags = true;
  }
  argc = positional;
  const auto named_control = [&](std::string_view key) {
    return std::find(named.begin(), named.end(), key) != named.end();
  };
  const auto is_row_mode = [](std::string_view value) {
    return value == "raster" || value == "selected" || value == "metal" ||
           value == "metal-deferred" || value == "raster-unique" ||
           value == "selected-unique" || value == "metal-unique";
  };
  if (argc >= 13 && is_row_mode(argv[12])) {
    Check(argc <= 14 && !named_control("--sample-rows") &&
              (argc != 14 || !named_control("--row-diagnostics")), "Duplicate sample row controls");
    row_mode = argv[12];
    if (argc == 14) row_diagnostics = argv[13];
    argc = 12;
    private_flags = true;
  }
  if (argc >= 16 && (std::string_view(argv[15]) == "cohorts" ||
                     std::string_view(argv[15]) == "pipeline")) {
    Check(argc <= 17 && !named_control("--learning-schedule") &&
              (argc != 17 || !named_control("--feature-window")), "Duplicate schedule controls");
    learning_schedule = argv[15];
    if (argc == 17) feature_window = argv[16];
    argc = 15;
  } else if (argc >= 16) {
    Check(!named_control("--gpu-profile") &&
              (argc < 17 || !named_control("--histogram-shards")) &&
              (argc < 18 || !named_control("--active-properties")), "Duplicate GPU fitting controls");
    gpu_profile_mode = argv[15];
    if (argc >= 17) histogram_shards = argv[16];
    if (argc >= 18) active_properties = argv[17];
  }
  Check(argc >= 6 && argc <= 18,
        "hybrid_profile INPUT OUT cpu|shared-cpu|hybrid|auto auto|buffered best|all "
        "[fit-off|schedule|fit-on [learning-participants [cpu-ceiling [entropy-participants "
        "[selection-participants [tokenization-participants [public|fitter-cpu|fitter-resumable|"
        "fitter-gpu|fitter-auto [minimum-rows [batch-fits]]]]]]]]] "
        "[--sample-rows=MODE] [--row-diagnostics=rows-off|rows-on] "
        "[--learning-schedule=cohorts|pipeline] [--feature-window=8|16] "
        "[--gpu-profile=gpu-off|gpu-wall|gpu-stage] [--histogram-shards=N] [--active-properties=N]");
  Check(gpu_profile_mode == "gpu-off" || gpu_profile_mode == "gpu-wall" ||
            gpu_profile_mode == "gpu-stage", "Unknown histogram GPU profiling mode");
  const bool gpu_profile_enabled = gpu_profile_mode != "gpu-off";
  const bool gpu_profile_output = argc >= 16 || named_control("--gpu-profile") ||
      named_control("--histogram-shards") || named_control("--active-properties");
  const std::string fitter_flag = argc >= 13 ? argv[12] : "fitter-cpu";
  const bool public_api = fitter_flag == "public";
  Check((!public_api || (argc == 13 && !private_flags)) &&
            (public_api || fitter_flag == "fitter-cpu" || fitter_flag == "fitter-resumable" ||
             fitter_flag == "fitter-gpu" || fitter_flag == "fitter-auto"), "Unknown fitting route flag");
  const bool fitter_diagnostics = argc >= 13 && !public_api;
  Check(is_row_mode(row_mode), "Invalid sample row mode");
  const bool gpu_rows = row_mode == "metal" || row_mode == "metal-deferred" || row_mode == "metal-unique";
  Check(row_diagnostics == "rows-off" || row_diagnostics == "rows-on", "Invalid sample row diagnostics mode");
  Check(learning_schedule == "cohorts" || learning_schedule == "pipeline", "Invalid learning schedule");
  Check(feature_window == "8" || feature_window == "16", "Feature window must be 8 or 16 Mi samples");
  const bool tokenization_diagnostics = argc >= 12;
  const size_t tokenization_participants = tokenization_diagnostics ? std::stoull(argv[11]) : 1;
  Check(tokenization_participants <= 256, "Invalid tokenization participant limit");
  const bool selection_diagnostics = argc >= 11;
  const size_t selection_participants = selection_diagnostics ? std::stoull(argv[10]) : 1;
  Check(selection_participants <= 256, "Invalid selection participant limit");
  const bool entropy_diagnostics = argc >= 10;
  const size_t entropy_participants = entropy_diagnostics ? std::stoull(argv[9]) : 1;
  Check(entropy_participants <= 256, "Invalid entropy participant limit");
  const std::string fitting_mode = argc >= 7 ? argv[6] : "fit-off";
  // Preserve the serial control and schema used by the older paired studies.
  // Supplying a participant argument opts into explicit wall/work schema 2.
  const bool parallel_diagnostics = argc >= 8;
  const size_t learning_participants = argc >= 8 ? std::stoull(argv[7]) : 1;
  const size_t cpu_ceiling = argc >= 9 ? std::stoull(argv[8]) : 4;
  Check(learning_participants <= 256 && cpu_ceiling > 0 && cpu_ceiling <= 256,
        "Invalid participant limit");
  Check(fitting_mode == "fit-off" || fitting_mode == "fit-on" || fitting_mode == "schedule",
        "Invalid fitting profile mode");
  const bool fitting_profile_enabled = fitting_mode == "fit-on";
  // Paused fits migrate between workers; the ordinary job-lane trace would
  // misrepresent their GPU waits. The new route exposes batch counters/times.
  const bool schedule_enabled = fitting_mode != "fit-off" &&
                                (public_api || fitter_flag == "fitter-cpu");
  const std::string route = argv[3], buffering = argv[4], predictors = argv[5];
  Check(route == "cpu" || route == "shared-cpu" || route == "hybrid" || route == "auto",
        "Invalid route");
  Check(buffering == "auto" || buffering == "buffered", "Invalid buffering policy");
  Check(predictors == "best" || predictors == "all", "Invalid predictor policy");
  Check(!public_api || ((route == "cpu" || route == "auto") && buffering == "auto" &&
                           predictors == "best" && learning_participants == cpu_ceiling &&
                           entropy_participants == cpu_ceiling && selection_participants == cpu_ceiling &&
                           tokenization_participants == cpu_ceiling),
        "Public profile requires default policy and matching stage/CPU limits");
  const std::filesystem::path directory = argv[2];
  Check(!std::filesystem::exists(directory), "Output directory already exists");
  io::IntegerImage input;
  Ok(io::ReadPnm(argv[1], &input));
  PackedModularImageView view{input.bytes, input.extent, input.row_stride,
                              static_cast<PackedModularFormat>(input.format),
                              SampleByteOrder::kBigEndian};
  ModularInputProfile format;
  Ok(ResolveModularInput(view.extent, view.format, &format));
  reference::IntegerImage expected{view.extent.width,
                                   view.extent.height,
                                   static_cast<uint32_t>(format.channel_count),
                                   format.metadata.bits,
                                   {}};
  expected.samples.reserve(view.extent.width * view.extent.height * format.channel_count);
  for (size_t y = 0; y < view.extent.height; ++y)
    for (size_t x = 0; x < view.extent.width * format.channel_count; ++x) {
      const size_t pos = y * view.row_stride + x * format.bytes_per_sample;
      expected.samples.push_back(format.bytes_per_sample == 1
                                     ? view.bytes[pos]
                                     : (uint16_t(view.bytes[pos]) << 8) | view.bytes[pos + 1]);
    }
  ReferenceWorkflowOptions options;
  options.fitting_backend = fitter_flag == "fitter-gpu" ? ReferenceFittingBackend::kGpu
                            : fitter_flag == "fitter-resumable" ? ReferenceFittingBackend::kResumableCpu
                            : fitter_flag == "fitter-auto" ? ReferenceFittingBackend::kAuto
                            : ReferenceFittingBackend::kCpu;
  if (argc >= 14) options.fitting_minimum_rows = std::stoull(argv[13]);
  if (argc >= 15) options.fitting_batch_fits = std::stoull(argv[14]);
  if (!histogram_shards.empty()) options.fitting_execution.histogram_shards = std::stoull(histogram_shards);
  if (!active_properties.empty()) options.fitting_execution.maximum_active_properties = std::stoull(active_properties);
  options.learning_schedule = learning_schedule == "pipeline" ? ReferenceLearningSchedule::kPipeline
                                                               : ReferenceLearningSchedule::kCohorts;
  options.feature_window_samples = std::stoull(feature_window) * 1024 * 1024;
  options.sample_rows = gpu_rows ? HistogramRowBackend::kGpu
                        : row_mode == "selected" || row_mode == "selected-unique"
                            ? HistogramRowBackend::kSelectedCpu
                            : HistogramRowBackend::kRasterCpu;
  options.use_unique_row_proof = row_mode.ends_with("-unique");
  options.defer_gpu_row_materialization = row_mode == "metal-deferred" || row_mode == "metal-unique";
  options.gpu_row_diagnostics = row_diagnostics == "rows-on";
  Check(!gpu_rows || route != "cpu",
        "Metal sample rows require a hybrid feature owner");
  options.selection_cpu_participants = selection_participants;
  options.learning_cpu_participants = learning_participants;
  options.buffering =
      buffering == "auto" ? ReferenceBuffering::kAutomatic : ReferenceBuffering::kBuffered;
  options.predictors =
      predictors == "best" ? ReferencePredictors::kBest : ReferencePredictors::kAll;
  std::unique_ptr<GpuBackend> gpu;
  GpuBackend *backend = nullptr;
  std::string device = "CPU";
  if (route != "cpu") {
#if GJXL_HYBRID_PROFILE_METAL
    if (public_api)
      Ok(codestream_internal::BorrowProductionMetalBackend(&backend));
    else {
      Ok(CreateEmbeddedMetalBackend({}, &gpu));
      backend = gpu.get();
    }
    options.gpu = backend;
    device = backend->name();
    options.feature_producer = route == "shared-cpu" ? WeightedFeatureBackend::kCpu
                               : route == "hybrid"   ? WeightedFeatureBackend::kGpu
                                                     : WeightedFeatureBackend::kAuto;
#else
    throw std::runtime_error("This build has no Metal backend");
#endif
  }
  options.entropy.cpu_participants = entropy_participants;
  options.tokenization.cpu_participants = tokenization_participants;
  // Keep feature production fixed at four while varying local learning.
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({size_t{8} << 30, cpu_ceiling}, &domain));
  const auto &budget = codestream_internal::WorkflowAdmission::Budget(*domain);
  auto context = CurrentResourceContext();
  context.domain = domain.get();
  ResourceContextScope resource_context(context);
  thread_budget_internal::CpuExecutionScope cpu;
  if (!public_api) Ok(cpu.Start(domain, cpu_ceiling));
  thread_budget_internal::EncodeScope participation(public_api ? 0 : cpu_ceiling);
  ReferenceWorkflowStats stats;
  ModularProfile profile;
  HistogramFittingProfile fitting_profile;
  histogram_gpu_internal::HistogramFittingProfile gpu_profile;
  std::vector<histogram_gpu_internal::HistogramFittingBatchProfile> gpu_records(
      gpu_profile_enabled ? 16384 : 0);
  gpu_profile.batches = gpu_records;
  gpu_profile.mode = gpu_profile_mode == "gpu-stage"
      ? histogram_gpu_internal::HistogramFittingProfileMode::kStage
      : histogram_gpu_internal::HistogramFittingProfileMode::kWall;
  if (gpu_profile_enabled) options.fitting_gpu_profile = &gpu_profile;
  // Diagnostic backing is allocated before timing and outside codec admission.
  // There is at most one fit per local group plus the global stream.
  std::vector<HistogramFitRecord> fitting_records;
  const size_t group_count = ((view.extent.width + 255) / 256) * ((view.extent.height + 255) / 256) + 1;
  // Canonical slots include empty DC groups in addition to image groups.
  // DC group count never exceeds image group count.
  std::vector<LearningJobTiming> job_records(schedule_enabled ? 2 * group_count : 0);
  std::vector<LearningWaveTiming> wave_records(schedule_enabled ? 2 * group_count : 0);
  std::vector<LearningBatchTiming> batch_records(schedule_enabled ? 2 * group_count : 0);
  LearningScheduleProfile schedule;
  schedule.jobs = job_records;
  schedule.waves = wave_records;
  schedule.batches = batch_records;
  if (fitting_profile_enabled) {
    fitting_records.resize(((view.extent.width + 255) / 256) * ((view.extent.height + 255) / 256) +
                           1);
    fitting_profile.records = fitting_records;
  }
  codestream_internal::CodestreamBuffer encoded;
  std::vector<uint8_t> public_encoded;
  const size_t submissions = backend ? backend->stats().committed_submissions : 0;
  double wall, process_cpu;
  {
    ProfileSession session(&profile);
    HistogramFittingProfileSession fitting_session(fitting_profile_enabled ? &fitting_profile
                                                                           : nullptr);
    ReferenceWorkflowProfileSession workflow_session(&stats, schedule_enabled ? &schedule : nullptr);
    const auto start = std::chrono::steady_clock::now();
    schedule.epoch = start;
    const auto cpu_start = std::clock();
    if (public_api) {
      gjxl::ModularEncodingOptions public_options;
      public_options.backend = route == "cpu" ? ModularBackend::kCpu : ModularBackend::kMetal;
      public_options.cpu_thread_count = cpu_ceiling;
      public_options.execution_domain = domain;
      Ok(gjxl::EncodeModularImage(input.view(), public_options, &public_encoded));
    } else {
      Ok(EncodeReferenceModularImage(view, options, budget, &encoded, &stats));
    }
    wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    process_cpu = double(std::clock() - cpu_start) / CLOCKS_PER_SEC;
  }
  const std::span<const uint8_t> bytes_view = public_api ? std::span<const uint8_t>(public_encoded) : encoded.view();
  Check(reference::DecodeLossless(bytes_view) == expected, "Exact independent decode differs");
  Check(!schedule.dropped, "Learning schedule trace exhausted caller storage");
  Check(!gpu_profile.dropped && (!gpu_profile_enabled || gpu_profile.written == stats.histogram_gpu_batches),
        "Histogram GPU attribution missed a batch");
  if (fitting_profile_enabled)
    Check(fitting_profile.calls == stats.fits && fitting_profile.successes == stats.fits &&
              !fitting_profile.records_dropped,
          "Histogram fitting attribution missed a fit");
  if (route == "cpu")
    Check(!stats.feature_batches && !stats.token_batches && !gpu,
          "CPU control used the hybrid route");
  else
    Check(stats.feature_samples == stats.token_samples &&
              stats.feature_samples == stats.scanned_pixels &&
              profile.training_work.reused_prediction_samples == stats.scanned_pixels &&
              profile.training_work.weighted_prediction_samples == 0,
          "Hybrid route repeated or omitted prediction features");
  if (gpu_rows && stats.local_trees)
    Check(stats.rows.gpu_rows == stats.selected_samples &&
              stats.rows.gpu_calls != 0,
          "Metal row experiment did not cover every selected local sample");
  if (gpu_rows && stats.local_trees && options.gpu_row_diagnostics)
    Check(stats.rows.replay_rows == stats.selected_samples &&
              stats.rows.retained_rows == stats.distinct_samples &&
              stats.rows.duplicate_hits == stats.selected_samples - stats.distinct_samples &&
              stats.rows.diagnostic_calls == stats.rows.gpu_calls,
          "GPU row diagnostic population mismatch");
  if (stats.rows.gpu_calls && options.defer_gpu_row_materialization)
    Check(stats.rows.deferred_calls == stats.rows.gpu_calls,
          "GPU deferred materialization coverage mismatch");
  Check(stats.rows.unique_calls <= stats.rows.uniqueness_checks &&
            stats.rows.unique_rows <= stats.distinct_samples &&
            (options.use_unique_row_proof || !stats.rows.uniqueness_checks),
        "Unique-row proof population mismatch");
  const size_t bytes = bytes_view.size();
  std::filesystem::create_directories(directory);
  std::ofstream output(directory / "output.jxl", std::ios::binary);
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output.write(reinterpret_cast<const char *>(bytes_view.data()), bytes);
  output.close();
  const auto memory = budget.snapshot();
  encoded = {};
  const auto released = budget.snapshot();
  Check(!released.committed_bytes() && !released.total.backing_count &&
            !released.total.pending_count && !released.open_reservations,
        "Completed encode leaked managed storage");
  std::ofstream out(directory / "result.json");
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out << std::setprecision(17) << std::boolalpha << "{\"schema\":" << (gpu_profile_output ? 8 : fitter_diagnostics ? 7 : public_api ? 6 : tokenization_diagnostics ? 5 : selection_diagnostics ? 4 : entropy_diagnostics ? 3 : parallel_diagnostics ? 2 : 1)
      << ",\"public_api\":" << public_api
      << ",\"gpu_submissions\":" << (backend ? backend->stats().committed_submissions - submissions : 0)
      << ",\"route\":" << std::quoted(route)
      << ",\"device\":" << std::quoted(device) << ",\"buffering\":" << std::quoted(buffering)
      << ",\"predictors\":" << std::quoted(predictors) << ",\"width\":" << view.extent.width
      << ",\"height\":" << view.extent.height << ",\"channels\":" << format.channel_count
      << ",\"bits\":" << unsigned(format.metadata.bits) << ",\"bytes\":" << bytes
      << ",\"exact_decode\":true,\"total\":{\"wall\":" << wall << ",\"cpu\":" << process_cpu
      << "},\"profile_wall\":";
  Array(out, std::span(profile.seconds).first(entropy_diagnostics ? profile.seconds.size() : 8));
  out << ",\"profile_cpu\":";
  Array(out, std::span(profile.cpu_seconds).first(entropy_diagnostics ? profile.cpu_seconds.size() : 8));
  out << ",\"work\":{\"local_trees\":" << stats.local_trees
      << ",\"selection_calls\":" << stats.selection_calls
      << ",\"learning_calls\":" << stats.learning_calls << ",\"fits\":" << stats.fits
      << ",\"empty_learning_calls\":" << stats.empty_learning_calls
      << ",\"scanned_pixels\":" << stats.scanned_pixels
      << ",\"selected_samples\":" << stats.selected_samples
      << ",\"distinct_samples\":" << stats.distinct_samples
      << ",\"tree_nodes\":" << stats.tree_nodes
      << "},\"routing\":{\"feature_batches\":" << stats.feature_batches
      << ",\"feature_samples\":" << stats.feature_samples
      << ",\"gpu_feature_batches\":" << stats.gpu_feature_batches
      << ",\"gpu_feature_samples\":" << stats.gpu_feature_samples
      << ",\"cpu_feature_batches\":" << stats.cpu_feature_batches
      << ",\"feature_cohorts\":" << stats.feature_cohorts
      << ",\"feature_memory_limited_cohorts\":" << stats.feature_memory_limited_cohorts
      << ",\"token_batches\":" << stats.token_batches
      << ",\"token_samples\":" << stats.token_samples << ",\"cpu_feature_peak_participants\":"
      << profile.adaptive_gpu.feature_preparation.peak_participants
      << ",\"learning_reused_features\":" << profile.training_work.reused_prediction_samples
      << ",\"learning_computed_features\":" << profile.training_work.weighted_prediction_samples
      << "},\"learning\":{\"requested_participants\":" << learning_participants
      << ",\"cpu_ceiling\":" << cpu_ceiling
      << ",\"peak_participants\":" << stats.learning_peak_participants
      << ",\"waves\":" << stats.learning_waves
      << ",\"memory_limited_waves\":" << stats.learning_memory_limited_waves
      << ",\"schedule\":" << std::quoted(learning_schedule)
      << ",\"feature_window_samples\":" << options.feature_window_samples
      << ",\"pipeline_seconds\":" << stats.pipeline_seconds
      << ",\"pipeline_peak_jobs\":" << stats.pipeline_peak_jobs
      << ",\"pipeline_peak_batches\":" << stats.pipeline_peak_batches
      << ",\"pipeline_admission_stalls\":" << stats.pipeline_admission_stalls
      << ",\"pipeline_serial_fallbacks\":" << stats.pipeline_serial_fallbacks
      << "},\"entropy\":{\"requested_participants\":" << entropy_participants
      << ",\"admitted_participants\":" << stats.entropy_cpu_participants
      << ",\"memory_reductions\":" << stats.entropy_memory_reductions
      << ",\"peak_stream_participants\":" << profile.parallel_entropy.peak_participants
      << ",\"stream_jobs\":" << profile.parallel_entropy.stream_jobs
      << ",\"parallel_calls\":" << profile.parallel_entropy.parallel_calls
      << ",\"model_work_seconds\":" << profile.parallel_entropy.model_work_seconds
      << ",\"emission_work_seconds\":" << profile.parallel_entropy.emission_work_seconds
      << "},\"selection\":{\"requested_participants\":" << selection_participants
      << ",\"peak_participants\":" << stats.selection_peak_participants
      << ",\"waves\":" << stats.selection_waves
      << ",\"memory_limited_waves\":" << stats.selection_memory_limited_waves
      << ",\"wall_seconds\":" << stats.selection_seconds
      << ",\"work_seconds\":" << stats.selection_work_seconds
      << "},\"tokenization\":{\"requested_participants\":" << tokenization_participants
      << ",\"admitted_participants\":" << stats.tokenization_cpu_participants
      << ",\"memory_reductions\":" << stats.tokenization_memory_reductions
      << ",\"peak_participants\":" << profile.parallel_tokenization.peak_participants
      << ",\"stream_jobs\":" << profile.parallel_tokenization.stream_jobs
      << ",\"parallel_calls\":" << profile.parallel_tokenization.parallel_calls
      << ",\"generated_tokens\":" << profile.parallel_tokenization.generated_tokens
      << ",\"borrowed_tokens\":" << profile.parallel_tokenization.borrowed_tokens
      << "},\"detail_work\":{\"sample_preparation\":" << stats.sample_preparation_seconds
      << ",\"fitting\":" << stats.fitting_seconds
      << "},\"detail_wall\":{\"learning\":" << stats.learning_seconds
      << ",\"features\":" << stats.feature_seconds << ",\"gpu_tokens\":" << stats.token_seconds;
  if (!parallel_diagnostics)
    out << ",\"sample_preparation\":" << stats.sample_preparation_seconds
        << ",\"fitting\":" << stats.fitting_seconds;
  out << "},\"sample_rows\":{\"requested\":" << std::quoted(row_mode)
      << ",\"gpu_calls\":" << stats.rows.gpu_calls
      << ",\"gpu_rows\":" << stats.rows.gpu_rows
      << ",\"gather_work\":" << stats.rows.gather_seconds
      << ",\"selection_work\":" << stats.rows.selection_seconds
      << ",\"generation_work\":" << stats.rows.generation_seconds
      << ",\"replay_work\":" << stats.rows.replay_seconds
      << ",\"allocation_copy_work\":" << stats.rows.allocation_copy_seconds
      << ",\"submit_wait_work\":" << stats.rows.submit_wait_seconds
      << ",\"output_bytes\":" << stats.rows.output_bytes
      << ",\"dispatch_bytes\":" << stats.rows.dispatch_bytes;
  out << ",\"diagnostics\":" << (options.gpu_row_diagnostics ? "true" : "false")
      << ",\"diagnostic_calls\":" << stats.rows.diagnostic_calls
      << ",\"table_growth_work\":" << stats.rows.table_growth_seconds
      << ",\"materialization_work\":" << stats.rows.materialization_seconds
      << ",\"replay_rows\":" << stats.rows.replay_rows
      << ",\"retained_rows\":" << stats.rows.retained_rows
      << ",\"lookup_probes\":" << stats.rows.lookup_probes
      << ",\"occupied_probes\":" << stats.rows.occupied_probes
      << ",\"duplicate_hits\":" << stats.rows.duplicate_hits
      << ",\"saturated_rows\":" << stats.rows.saturated_rows
      << ",\"table_growths\":" << stats.rows.table_growths
      << ",\"rehashed_rows\":" << stats.rows.rehashed_rows
      << ",\"deferred_calls\":" << stats.rows.deferred_calls
      << ",\"contiguous_materializations\":" << stats.rows.contiguous_materializations
      << ",\"unique_proof\":" << (options.use_unique_row_proof ? "true" : "false")
      << ",\"uniqueness_work\":" << stats.rows.uniqueness_seconds
      << ",\"uniqueness_checks\":" << stats.rows.uniqueness_checks
      << ",\"unique_calls\":" << stats.rows.unique_calls
      << ",\"unique_rows\":" << stats.rows.unique_rows;
  out << "},\"memory\":{\"peak_backing_bytes\":" << memory.peak_backing_bytes
      << ",\"peak_committed_bytes\":" << memory.peak_committed_bytes
      << ",\"maximum_stage_bytes\":" << stats.maximum_stage_bytes
      << ",\"peak_feature_samples\":" << stats.peak_feature_samples
      << ",\"peak_feature_bytes\":" << stats.peak_feature_bytes << "},\"histogram_fitting\":";
  WriteHistogramProfile(out, fitting_profile, fitting_profile_enabled);
  out << ",\"histogram_gpu\":{\"route\":" << std::quoted(fitter_flag)
      << ",\"minimum_rows\":" << options.fitting_minimum_rows
      << ",\"batch_fits\":" << options.fitting_batch_fits
      << ",\"preparations\":" << stats.histogram_gpu_preparations
      << ",\"batches\":" << stats.histogram_gpu_batches
      << ",\"nodes\":" << stats.histogram_gpu_nodes
      << ",\"rows\":" << stats.histogram_gpu_rows
      << ",\"properties\":" << stats.histogram_gpu_properties
      << ",\"cpu_nodes\":" << stats.histogram_cpu_nodes
      << ",\"admission_fallbacks\":" << stats.histogram_admission_fallbacks
      << ",\"prepared_bytes\":" << stats.histogram_prepared_bytes
      << ",\"dispatch_bytes\":" << stats.histogram_dispatch_bytes
      << ",\"peak_scratch_bytes\":" << stats.histogram_peak_scratch_bytes
      << ",\"preparation_seconds\":" << stats.histogram_preparation_seconds
      << ",\"evaluation_seconds\":" << stats.histogram_evaluation_seconds << '}';
  if (gpu_profile_output) {
    out << ",\"histogram_gpu_profile\":{\"mode\":" << std::quoted(gpu_profile_mode)
        << ",\"shards\":" << options.fitting_execution.histogram_shards
        << ",\"active_properties\":" << options.fitting_execution.maximum_active_properties
        << ",\"wall_phases\":[";
    for (size_t i = 0; i < histogram_gpu_internal::kHistogramFittingWallPhases.size(); ++i)
      out << (i ? "," : "") << std::quoted(histogram_gpu_internal::kHistogramFittingWallPhases[i]);
    out << "],\"gpu_stages\":[";
    for (size_t i = 0; i < histogram_gpu_internal::kHistogramFittingGpuStages.size(); ++i)
      out << (i ? "," : "") << std::quoted(histogram_gpu_internal::kHistogramFittingGpuStages[i]);
    out << "],\"batches\":[";
    for (size_t i = 0; i < gpu_profile.written; ++i) {
      const auto &b = gpu_records[i];
      out << (i ? "," : "") << "{\"nodes\":" << b.nodes << ",\"rows\":" << b.rows
          << ",\"tasks\":" << b.tasks << ",\"active_properties\":" << b.active_properties
          << ",\"waves\":" << b.waves << ",\"maximum_bins\":" << b.maximum_bins
          << ",\"maximum_symbols\":" << b.maximum_symbols << ",\"scratch_bytes\":" << b.scratch_bytes
          << ",\"wall_ns\":";
      Array(out, b.wall_nanoseconds);
      out << ",\"gpu_ns\":"; Array(out, b.gpu_nanoseconds);
      out << ",\"dispatches\":"; Array(out, b.dispatches);
      out << ",\"command_buffer_gpu_ns\":" << b.command_buffer_gpu_nanoseconds << '}';
    }
    out << "]}";
  }
  out << ",\"learning_schedule\":{\"enabled\":" << schedule_enabled << ",\"dropped\":" << schedule.dropped
      << ",\"waves\":[";
  for (size_t i = 0; i < schedule.waves_written; ++i) {
    const auto &w = wave_records[i];
    out << (i ? "," : "") << "{\"first_job\":" << w.first_job << ",\"jobs\":" << w.jobs
        << ",\"participants\":" << w.participants << ",\"start\":" << w.start
        << ",\"dispatch\":" << w.dispatch << ",\"joined\":" << w.joined << ",\"finish\":" << w.finish << '}';
  }
  out << "],\"jobs\":[";
  for (size_t i = 0; i < schedule.jobs_written; ++i) {
    const auto &j = job_records[i];
    out << (i ? "," : "") << "{\"stream\":" << j.stream << ",\"worker\":" << j.worker
        << ",\"start\":" << j.start << ",\"finish\":" << j.finish
        << ",\"preparation\":" << j.preparation << ",\"fitting\":" << j.fitting
        << ",\"calls\":" << j.calls << ",\"fits\":" << j.fits << '}';
  }
  out << "],\"batches\":[";
  for (size_t i = 0; i < schedule.batches_written; ++i) {
    const auto &b = batch_records[i];
    out << (i ? "," : "") << "{\"first_stream\":" << b.first_stream
        << ",\"stop_stream\":" << b.stop_stream << ",\"samples\":" << b.samples
        << ",\"prepare_start\":" << b.prepare_start << ",\"ready\":" << b.ready
        << ",\"fits_observed_done\":" << b.fits_observed_done
        << ",\"tokenize_start\":" << b.tokenize_start << ",\"released\":" << b.released << '}';
  }
  out << "]}";
  out << "}\n";
  out.close();
  std::cout << route << ": " << wall << " s, " << bytes << " bytes, " << stats.gpu_feature_batches
            << " GPU feature batches; exact decode\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
