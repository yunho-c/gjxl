# Native context-map latency qualification — September 15

The native context-map implementation adds **0.59% median paired encode time**
on the retained 12-image effort-4 selection. The four Kodak inputs show
2.06–3.51% increases; CLIC and larger inputs range from −0.70% to +0.66%.
The implementation's previously qualified −0.403% BD-rate improvement and
exact reconstruction identities are unchanged.

This completes the local warm-call qualification under the declared environment
checks. It supports a small overall encode-time cost, concentrated on the small
images. Sub-percent differences on individual larger images should not be
treated as established speedups or hardware-wide guarantees.

## Protocol and controls

The baseline is DC integer-mapping search at `e8abc34`; the candidate is native
context-map compression at `4f4fd27`. The exact binaries and Metal libraries
from the completed rate qualification are reused. Production source, binary,
helper, input and reference-codestream hashes pass the final audit. No encoder
source or production policy changed for this measurement.

The test machine is an AC-powered Apple M4 Pro, 14 CPU cores and 48 GiB memory.
Both arms use fully resident Metal, effort 4, eight participating CPU threads,
and the same 12 images at their original Q80 distances. Those distances are
retained operating points, not a claim that every measured score equals 80.

Two blocks each contain four process pairs per image. Arm order alternates and
image order rotates by pair index. Each process performs one validation encode,
two warmups and five timed complete public encode calls. Input-file loading and
output-file writing are outside this call boundary; encoding through publication
of the final codestream buffer is included. Stage profiling and final quality
scoring are disabled. No builds, decoder runs or scoring jobs are launched as
part of timing collection.

All **192 accepted processes / 960 measured calls** pass the raw-sample and
codestream-hash audit. Hash identity connects every output to its independently
decoded rate-run counterpart. The ten excluded pairs also reproduce their
expected output hashes: 212 successful processes in total.

## Results

The headline is the median of the 12 per-image paired ratios. Each image's
ratio is the median of eight candidate/baseline process-median ratios. The
time columns below pool the 40 accepted samples per arm, so dividing those
columns need not reproduce the separately computed paired percentage.

| Image | Baseline median, ms | Candidate median, ms | Paired time change |
|---|---:|---:|---:|
| Kodak 01 | 13.494 | 13.739 | +2.06% |
| Kodak 17 | 13.170 | 13.721 | +2.37% |
| CLIC 097cb426 | 48.697 | 48.848 | +0.66% |
| CLIC d1a9be98 | 39.919 | 40.041 | +0.24% |
| CLIC b51d5fb5 | 36.144 | 35.975 | −0.70% |
| Kodak 08 | 13.671 | 13.954 | +2.82% |
| Kodak 13 | 13.630 | 13.959 | +3.51% |
| CLIC 28d24b9c | 41.646 | 42.168 | +0.60% |
| Alpine lake, 24 MP | 366.932 | 364.067 | −0.54% |
| Campus interior, 12 MP | 181.729 | 183.131 | +0.58% |
| Campus interior, 48 MP | 828.467 | 827.451 | −0.39% |
| Forest stream, 48 MP | 845.769 | 843.527 | +0.08% |

The two block-level medians are +0.138% and +0.412%. These nested medians are
computed separately from the eight-pair aggregate; they are not confidence
intervals. The cohort remains the preselected investigative set, rather than
a random or held-out sample of all images.

## Environment audit and temporary service pause

The runner samples process CPU time, codec/build jobs, AC power and
thermal/performance status at nominal one-second intervals and pair boundaries.
Before collection and after contention, it performs a nominal 20-second quiet
check. The actual maximum snapshot gap within accepted pairs is 1.097 seconds.

Thresholds and retry rules were fixed before collection: reject a whole pair
on a competing codec/build process, another process above 80% of one CPU,
aggregate other CPU above 200%, loss of AC power, a thermal/performance warning,
or monitoring failure. Both arms are retried together, at most three attempts
per pair. Latency values never determine exclusion.

Ten pairs were excluded because of desktop/system CPU spikes or another
Python job. All attempts remain available with their exact exclusion reasons.
The 531 snapshots covering accepted pairs contain no declared violations;
aggregate other CPU has a median of 68.4% and a maximum of 183.7% of one CPU.
The maximum individual background process is 78.0%. Thus the evidence describes
a monitored desktop comparison, not a claim that background activity was absent.

After the user made the machine available, `suggestd` continued saturating a
CPU. With explicit user approval, only that process was temporarily suspended.
A watchdog resumed it automatically when collection ended, after **108.43
seconds**, below its ten-minute deadline. Its resumed running state was verified.
There are 66 accepted pairs with ordinary services and 30 during the pause;
no pair straddles either intervention. This service condition is part of the
qualification record. No persistent service setting was changed.

## Reproduction and retained evidence

Use [`tools/context_map/latency.py`](../tools/context_map/latency.py) with the
[documented workflow](../tools/context_map/README.md#fresh-latency-qualification).
The collector never suspends services itself; the one authorized intervention
above used a separate retained watchdog script.

Local artifacts are in `build/context-map-latency-20260915/`: frozen manifest,
every attempt's codestream and raw samples, command ledger, 2,318 environment
snapshots, analysis, independent audit, service intervention log and watchdog.
The earlier rate and provisional timing artifacts remain unchanged.

[Committed results](context-map-latency-results.json) include the per-image and
per-block ratios, exclusions, service audit and provenance hashes. The
September 14 +0.85% timing observation remains labeled provisional; this
separate run supplies the qualified local comparison.
