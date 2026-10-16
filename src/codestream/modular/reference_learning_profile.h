// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/reference_workflow.h"
#include <chrono>

namespace gjxl::modular_internal {
// Private opt-in diagnostics. The caller allocates all records before encoding.
// Only the scheduler appends records; jobs write disjoint slots until the join.
// These intervals measure active job lifetimes, not CPU execution or utilization.
struct LearningJobTiming {
  size_t stream = 0, worker = 0;
  double start = 0, finish = 0, preparation = 0, fitting = 0;
  size_t calls = 0, fits = 0;
};
struct LearningWaveTiming {
  size_t first_job = 0, jobs = 0, participants = 0;
  double start = 0, dispatch = 0, joined = 0, finish = 0;
};
struct LearningBatchTiming {
  size_t first_stream = 0, stop_stream = 0, samples = 0;
  double prepare_start = 0, ready = 0, fits_observed_done = 0,
         tokenize_start = 0, released = 0;
};
struct LearningScheduleProfile {
  using Clock = std::chrono::steady_clock;
  Clock::time_point epoch = Clock::now();
  std::span<LearningJobTiming> jobs;
  std::span<LearningWaveTiming> waves;
  std::span<LearningBatchTiming> batches;
  size_t jobs_written = 0, waves_written = 0, dropped = 0;
  size_t batches_written = 0;
  double Now() const { return std::chrono::duration<double>(Clock::now() - epoch).count(); }
};
inline thread_local LearningScheduleProfile *active_learning_schedule_profile = nullptr;
inline thread_local ReferenceWorkflowStats *active_reference_workflow_stats = nullptr;
class ReferenceWorkflowProfileSession {
public:
  ReferenceWorkflowProfileSession(ReferenceWorkflowStats *stats, LearningScheduleProfile *schedule)
      : previous_stats_(active_reference_workflow_stats),
        previous_schedule_(active_learning_schedule_profile) {
    active_reference_workflow_stats = stats;
    active_learning_schedule_profile = schedule;
  }
  ~ReferenceWorkflowProfileSession() {
    active_reference_workflow_stats = previous_stats_;
    active_learning_schedule_profile = previous_schedule_;
  }
  ReferenceWorkflowProfileSession(const ReferenceWorkflowProfileSession &) = delete;
  ReferenceWorkflowProfileSession &operator=(const ReferenceWorkflowProfileSession &) = delete;
private:
  ReferenceWorkflowStats *previous_stats_;
  LearningScheduleProfile *previous_schedule_;
};
class LearningWaveProbe {
public:
  LearningWaveProbe() : profile_(active_learning_schedule_profile) {
    if (profile_) start_ = profile_->Now();
  }
  void Dispatch(size_t count, size_t participants) {
    if (!profile_) return;
    if (profile_->waves_written == profile_->waves.size() ||
        count > profile_->jobs.size() - profile_->jobs_written) {
      ++profile_->dropped;
      return;
    }
    wave_ = &profile_->waves[profile_->waves_written++];
    *wave_ = {profile_->jobs_written, count, participants, start_, profile_->Now(), 0, 0};
    profile_->jobs_written += count;
  }
  void Joined() { if (wave_) wave_->joined = profile_->Now(); }
  // A ready queue reserves trace slots for its remaining canonical range.
  // Memory pressure may stop admission early; trim only after all jobs join.
  void Joined(size_t count) {
    if (wave_) {
      profile_->jobs_written -= wave_->jobs - count;
      wave_->jobs = count;
      Joined();
    }
  }
  ~LearningWaveProbe() { if (wave_) wave_->finish = profile_->Now(); }
  class Job {
  public:
    Job(const LearningWaveProbe &wave, size_t index, size_t stream, size_t worker,
        const ReferenceWorkflowStats &stats) : profile_(wave.profile_), stats_(stats) {
      if (!wave.wave_) return;
      job_ = &profile_->jobs[wave.wave_->first_job + index];
      *job_ = {stream, worker, profile_->Now(), 0, 0, 0};
      preparation_ = stats.sample_preparation_seconds;
      fitting_ = stats.fitting_seconds;
      calls_ = stats.learning_calls;
      fits_ = stats.fits;
    }
    ~Job() {
      if (!job_) return;
      job_->finish = profile_->Now();
      job_->preparation = stats_.sample_preparation_seconds - preparation_;
      job_->fitting = stats_.fitting_seconds - fitting_;
      job_->calls = stats_.learning_calls - calls_;
      job_->fits = stats_.fits - fits_;
    }
    Job(const Job &) = delete;
    Job &operator=(const Job &) = delete;
  private:
    LearningScheduleProfile *profile_;
    const ReferenceWorkflowStats &stats_;
    LearningJobTiming *job_ = nullptr;
    double preparation_ = 0, fitting_ = 0;
    size_t calls_ = 0, fits_ = 0;
  };
  LearningWaveProbe(const LearningWaveProbe &) = delete;
  LearningWaveProbe &operator=(const LearningWaveProbe &) = delete;
private:
  LearningScheduleProfile *profile_;
  LearningWaveTiming *wave_ = nullptr;
  double start_ = 0;
};
} // namespace gjxl::modular_internal
