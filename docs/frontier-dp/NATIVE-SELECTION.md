# Native GPU greedy selection checkpoint

The ordinary resident Metal search now scores and selects in one submission.
The selector reads the seven existing candidate cost buffers directly, computes
partial-tile offsets from the shared enumeration, and writes a compact strategy
map plus per-tile error flags into the now-dead rate scratch. It preserves the
CPU traversal, FP32 addition order and tie policy. Dense-bank and profiled search
still use the existing CPU selector. This checkpoint retains the final strategy
map readback and CPU AQ metadata construction; it does not yet eliminate the
ACS-to-AQ synchronization.

There are no extra device allocations. The existing DCT8 cost-readback owner
holds the much smaller strategy map. Candidate cost readback/dense-table
construction and CPU selection are skipped. Storage remains within the existing
conservative admission plan, including prepared reuse and policy switching.
Invalid candidate costs are flagged on device and rejected before publishing
the host grid or statistics. The device error path emits a safe DCT8 cover for
future dependent metadata kernels, while retaining its error flag.

The native kernel retains the prototype's subnormal multiplication/addition
handling and out-of-line mutating-helper workaround. Its separate strict-math
compilation leaves scoring kernels unchanged.

Validation at this checkpoint:

- All 60 CPU-control pilot codestreams matched byte for byte, covering six
  images, efforts 5 and 8, and distances 0.7, 1.2, 2, 4, 7.
- Search tests cover all 64 partial-tile shapes after full tile rows/columns,
  prepared reuse, sparse/dense switching, error atomicity and invalid resident
  input. They pass with and without Metal API/shader validation.
- Existing AC search and host/device storage-plan checks pass.
- Every CPU/GPU arm in the timing experiment produces the same codestream,
  including deterministic repeated encodes.

Retained artifacts:
`build/frontier/results/native-selection-encode-parity-20260918/` and
`build/frontier/results/native-selection-timing-20260918/`. Each freezes its
binary, changed sources, source diff, controls and input hashes. The parity run
precedes the removal of unused host cost-table allocation; the timing run
includes that allocation change and rechecks output identity.

Complete-call wall time, distance 1.2, 8 CPU threads, no stage profiler. Three
independent processes per arm, each with three warmups and seven retained samples;
the table uses medians of process medians with rotated arm/case order. CPU and
GPU arms use the same frozen binary, with the experiment override selecting the
CPU control. Measurements were sequential on the interactive M4 Pro host.

| Input | Effort | CPU selection encoder | GPU selection encoder | Speedup |
|---|---:|---:|---:|---:|
| Alpine 3MP | 5 | 61.703 ms | 60.144 ms | 1.026× |
| Alpine 3MP | 8 | 119.436 ms | 117.172 ms | 1.019× |
| Alpine 24MP | 5 | 392.346 ms | 380.937 ms | 1.030× |
| Alpine 24MP | 8 | 748.177 ms | 742.916 ms | 1.007× |
| Forest 24MP | 5 | 431.775 ms | 420.630 ms | 1.026× |
| Forest 24MP | 8 | 882.549 ms | 867.411 ms | 1.017× |

These are bounded engineering measurements, not final corpus/paper timings.
The smaller effort-8 gains reflect a much longer complete-encode workload;
selection kernel speedup alone would overstate the encoder improvement. The
selected maps are unchanged, so this implementation carries no rate-quality
policy improvement. The independent rectangle/frontier alternatives and their
small mixed rate-quality effects remain documented in SELECTORS.md.

The next checkpoint keeps quant-field adjustment and bound reduction in the AQ
submission; see [AQ-INITIALIZATION.md](AQ-INITIALIZATION.md). Device-generated
anchor/CfL/dispatch metadata and deferred map materialization remain open, scoped
in [HANDOFF.md](HANDOFF.md). Final qualification and new paper study sessions
remain pending.
