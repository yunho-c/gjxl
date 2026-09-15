# Effort 3/4 diagnostic tools

Read [the investigation](../../docs/e3-e4-rate-investigation.md) for results,
scope and the implementation proposal. These tools target the retained
September 14 plotted binary and artifacts on this machine. They are not a
general benchmark suite or production policy implementation.

Completed studies retain manifests, command ledgers, codestreams and audits
under `build/`. Collection is explicitly invoked and expensive. Source files
named in frozen manifests must not be edited during a run. Analysis consumes
saved data; it does not silently repair incomplete curves or calibration.

| Phase | Entry points | Artifact directory |
|---|---|---|
| DC/AQ factorial | `study.py`, `report.py` | `e3-e4-rate-20260914` |
| Independent DC grid | `build_precision_probe.py`, `precision_split.py` | `e3-e4-rate-20260914/precision-split` |
| All 65 DC-grid curves | retained `collector.py`, `confirm_report.py`, `plotted_reference.py` | `e3-e4-dc-grid-confirm-20260914` |
| Matched score and Butteraugli | `cross_metric.py`, `cross_metric_audit.py`, `quality_panels.py` | `e3-e4-cross-metric-20260914` |
| Native component budgets | `build_frame_probe.py`, `frame_accounting.py`, `frame_report.py` | `e3-e4-frame-probe-20260914` |
| Same-frame writer | `build_tail_probe.py`, `tail_study.py`, `tail_report.py` | `e3-e4-tail-confirm-20260914` |
| Modular mapping ablation | `build_uint_tail_probe.py`, `uint_tail_study.py` | `e3-e4-tail-uint-confirm-20260914` |
| All 65 native e4 writer curves | retained `collector.py`, `tail_report.py --root ...` | `e4-native-tail-full-20260914` |

`precision_study.py` records the **terminal rejected** prediction-aware / zero
extra-bit experiment. Preserve that failure; use the supported
`precision_split.py` experiment instead. The earlier prediction-aware
full-corpus protocol was prepared but not collected.

The probes compile copied source overlays and link retained libraries;
production sources and defaults are unchanged. Timing includes diagnostic
work and concurrent studies, so it is not latency qualification. Temporary
decoded PFMs are removed only after their hashes have been recorded/checked.
