# Combined resident ACS and AQ

The ordinary Metal frontend now records candidate scoring, greedy selection,
strategy metadata construction and the resident AQ policy in one command
buffer. There is no host wait or strategy-map readback between ACS and AQ.
The selected host grid is published after AQ completes. Initial quantization
still has its existing completion boundary, and final frame assembly/serialization
still consume host metadata.

The selected layout preserves the established CPU greedy policy. Full frontier
DP and rectangle search remain experiment controls; their measured rate results
did not justify a production policy change. This integration targets residency
and latency while preserving compression and decoded pixels.

## Integration and lifetime

An explicit deferred provider hook replaces the old requirement that search
return a complete host grid before AQ. Search preparation creates seven scoring
descriptors without submitting work. AQ validates and encodes that producer
before its metadata kernels, then consumes indirect family counts. The public
synchronous search API remains available.

Search buffers and resident input views outlive the synchronous AQ call,
including all failures. The last-use search owner is released after AQ returns.
A failure before submission also drops the AQ borrower before search buffers
can be reset. Mutable search/selection storage cannot alias AQ-owned arenas or
borrowed original/coding images. Workflow admission includes prefix storage,
late host publication, and the increased overlap between scoring buffers and
completed-frame allocation.

The path requires ordinary placement, Butteraugli resident AQ, direct-image
SIMD-matrix transforms and disabled GPU profiling. Dense placement, exact
coefficient AQ, unsupported transform configurations, maximum-error mode and
profiling keep their established paths. Fixed DCT8 bypasses search. Experiment
capture/CPU-policy controls also retain synchronous search.

## Correctness evidence

The [combined-pipeline manifest](evidence/combined-acs-aq-20260918.json) retains
normal and Metal API/shader-validation results, source hashes and external
metallib identity. Frozen artifacts are in
`build/frontier/results/combined-acs-aq-20260918/`, based on `197a20b` plus the
retained patch.

- 26 combined attempts match the CPU-selection path's codestreams and scores
  exactly. They cover every AQ update count 0–4, scored/unscored final output,
  last-use buffer release, reuse and recovery. The CPU-selection reference uses
  the same backend kernels through the established profiling path.
- Six upload/submission/completion/readback/numeric/staging failures preserve
  grids, frames and scores. Each recovers with a new preparation. Aliased
  producer descriptors are rejected before submission.
- Fresh evaluated host-input pipelines use three submissions; zero-update
  unscored pipelines use two. Reused attempts use two: initial quantization,
  then combined ACS/AQ. No separate ACS submission is recorded.
- All [60 retained CPU-control encodes](evidence/combined-acs-aq-encode-parity-20260918.json)
  match byte for byte: six Kodak/CLIC images, efforts 5/8 and five distances
  0.7–7. Another [12 paired encodes](evidence/combined-acs-aq-mid-efforts-20260918.json)
  at efforts 6/7 and distance 1.2 also match the frozen CPU selector exactly.
- Twelve related codec/Metal/storage/admission regressions pass. The broader
  CPU-only `quantization_pipeline` pinned-score test fails identically in a
  fresh Release build of parent `197a20b`: 0.24919039011001587 versus expected
  0.24914586544036865. Its quant field and raw quant values also reproduce
  exactly. The parent source archive, build and log are retained; the fixture
  was not changed for this work.

## Complete-call timing

The [timing evidence](evidence/combined-acs-aq-timing-20260918.json) records a
54-process comparison on the 20-core M4 Pro: two 24MP photographs and a 3MP
downsample, efforts 5/8 at distance 1.2, eight CPU threads, three independent
processes per arm/case, three warmups and seven timed encodes per process.
Case and arm order rotate. The values below are medians of process medians in
milliseconds, measured without GPU profiling. All codestreams are byte-identical.

| Image | Effort | CPU selector | GPU with host handoff | Combined ACS/AQ | CPU / combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| Alpine 3MP | 5 | 59.200 | 57.597 | 57.893 | 1.023x |
| Alpine 3MP | 8 | 115.079 | 113.581 | 113.216 | 1.016x |
| Alpine 24MP | 5 | 382.796 | 374.278 | 370.915 | 1.032x |
| Alpine 24MP | 8 | 728.264 | 719.265 | 715.001 | 1.019x |
| Forest 24MP | 5 | 422.264 | 411.654 | 405.890 | 1.040x |
| Forest 24MP | 8 | 844.993 | 830.001 | 826.193 | 1.023x |

The combined implementation delivers 1.6–4.0% higher throughput than the frozen
CPU-selection control in this pilot. Compared with the preceding GPU selector
and fused AQ initialization, eliminating the remaining handoff adds 0.46–1.42%
on the 24MP cases. The 3MP changes (0.51% slower and 0.32% faster) are within the
observed process spread. This small pilot supports a modest latency benefit;
it does not establish a corpus-wide speedup or a compression improvement.

Artifacts, individual samples, hashes and commands are retained in
`build/frontier/results/combined-acs-aq-timing-20260918/`. The combined executable
was compiled from the implementation committed as `195464c`; the run's source
tree was clean. Its configure-time version banner still reports `407cd05`,
because CMake had not been reconfigured after the later source edits. The
retained executable hash, source revision and qualification artifacts identify
the actual measured implementation. The fresh paper study uses a new production
build with experiment controls disabled and a matching revision banner.
