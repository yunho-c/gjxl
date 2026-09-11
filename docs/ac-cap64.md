# AC ANS cap 64 for high-resolution images

Ordinary balanced and high-density encoding now allow up to 64 AC ANS models
for frames with at least 8,294,400 pixels (3840 × 2160). The policy uses original
frame area, so it applies equally to portrait images and does not depend on block
padding or CPU/Metal selection. Smaller frames keep the 32-model budget.

This threshold follows the six photographic 4K inputs in the completed rate and
full-encode studies. On that evidence, cap 64 captured 68–85% of cap 128's byte
savings for 0.26–1.74% additional warm encode time after exact clustering
improvements. Those timings describe the frozen diagnostic workflow; they are
not new performance measurements of this integration branch.

The implementation is based on entropy commit `06857be`, including its existing
exact clustering improvements. `EntropyCodeOptions::maximum_ans_clusters`
passes the AC budget into both scanned and borrowed-population ANS paths.
The default remains 32 for other callers, including DC and coefficient orders.
Prefix retains its separate 32-model limit. Maximum-compression search still
derives its ANS partition from Prefix and therefore retains that partition's
limit. A cap is an upper bound; clustering can stop earlier.

ANS model validation and model-storage bounds support 64 models independently
of Prefix. Optimizer storage bounds follow the requested direct-ANS cap,
including refinement queues; serializer task and header bounds use the same
area policy as encoding. Small-frame and maximum-compression header bounds
retain their previous model budget. Invalid direct limits fail before publishing
an output.

Tests cover the area boundary, orientation, zero/overflow-sized extents, invalid
limits and unchanged error outputs. A deterministic 96-context fixture reaches
64 models in scanned and borrowed paths under managed allocation bounds. Model
and emission checks exercise all 64 models, while Prefix retains its 32-model
coverage and rejection of 33-model storage requests.

## Corpus qualification

`tools/ac_cap64/qualify.py` builds three encoders against the same Release
libraries: the production policy, the preceding 32-cap encoder, and a forced-64
AC oracle. It uses the 112 saved rate-study cases. Every encode is repeated;
large policy outputs must equal forced-64, and small outputs must equal the
32-cap baseline. The pinned external libjxl decoder compares baseline/policy
PFM outputs byte for byte. This is correctness qualification, not timing.

```sh
python3 tools/ac_cap64/qualify.py build
python3 tools/ac_cap64/qualify.py collect
python3 tools/ac_cap64/audit.py
```

The builder expects the policy as an uncommitted diff on `06857be`, which supplies
its `HEAD` baseline. After committing the policy, reproduce in a separate
worktree at that base with the policy diff applied without committing. The audit
checks the recorded baseline revision; do not treat a later `HEAD` as the old
32-cap encoder.

The harness requires the existing sibling rate-study artifacts and pinned
decoder. Build/source/library/binary identities, commands, codestreams, decoder
hashes and resumable case records are retained in `build/ac-cap64-qualification`.
The audit also compares byte savings with the previous weighted-DC experiment;
absolute file sizes need not match that diagnostic DC representation.

## Results

All 112 cases pass: 16 large-image cases match forced-64 output and reproduce
the prior cap-64 byte savings exactly; all 96 smaller cases remain byte-identical
to the 32-cap baseline. All 336 codestreams repeat exactly (672 encodes), and
all 112 baseline/policy comparisons decode to identical PFM bytes. See the
[qualification summary](ac-cap64-qualification.json).

Release passes 128/129 tests. The only failure is the inherited
`quantization_pipeline` golden mismatch at index 1: actual
`0.24919039011001587`, expected `0.24914586544036865`. Entropy, entropy storage,
serializer storage planning and serializer storage also pass ASan/UBSan, with
leak detection disabled. Their logs and source hashes are retained alongside
the corpus evidence. No performance claims are inferred from qualification
running alongside other tests.
