# Account for admission-time backend selection

This candidate starts from the exact qualified 1,276-file split-provider archive
(715f033, SHA256 199c692da5718fb9f7a4411f1b3dae75a1710858469c9f6bdb4befe8d61ef4a3).
The closed production-adapter diagnostic showed that backend_selection_nanoseconds
contains only cached selection: PlanWorkflowAdmission already initializes CUDA
before that timer starts. No startup cause or provider default is inferred.

Add an optional accumulator to the private admission planner. Time only its
SelectAttemptBackend call, publish the added duration only on successful planning,
and read no clocks when no accumulator is supplied. A profiled standalone encode
passes its local profile accumulator; later resident preselection adds to it
instead of overwriting it. Existing batch/C admission callers keep the default
null accumulator. Planning outside an encode's outer reservation stays outside
that encode's profile. No GPU kernels, routing, allocation recipe or public ABI
changes; no new timing fields or production defaults.

The shared regression checks nonzero accumulated CPU selection over repeated
plans, preservation of a prior contribution, exact storage-plan equivalence,
failure atomicity and the automatic CPU-only no-selection path. Existing complete
workflow profile-sum/output checks, admission/C/batch lifecycle tests and CUDA
host/GPU profiling, token route and finite-admission fixtures must remain passing.
Native validation and corrected startup observations are still required.

Freeze this source and preserve failures before native work. Keep the qualified
split provider and all closed diagnostic libraries unchanged. This is a
profiling-coverage fix; it is not a tokenization performance optimization.
