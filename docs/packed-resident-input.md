# Packed input in resident Metal encoding

`gjxl_encode` with an explicitly selected Metal backend converts validated
RGB8 or opaque RGBA8 sRGB directly into the resident input's shared linear-RGB
planes. The conversion uses the same lookup table and pixel traversal as the
owned host-image converter. It removes one full float-RGB host allocation and
the subsequent upload copy; conversion itself still runs on the CPU.

The C adapter validates the packed image and encoder options, then acquires
memory admission and CPU participation before preparing input. A synchronous
fill callback receives the resident allocation's writable source planes.
Resident preparation runs the existing GPU numeric validation, padded Opsin
conversion, and optional matrix-scale statistics after filling completes.
Failure returns without publishing a prepared owner or C output.

The completed allocation exposes an immutable host view for CPU readers.
`ResidentEncodingInput` transfers that view and its owner together into the
encoding workflow. The prepared workflow destroys borrowing AQ/quantization
state before releasing the input at its existing pre-serialization boundary.
Neither the fill callback nor its context is retained.

The production entry point requires forced Metal, fully resident AQ, and a
Butteraugli target. Automatic backend selection and CPU encoding keep their
existing owned-input path. The C ABI, alpha policy, caller pixel ownership,
and final-byte publication are unchanged. The existing storage admission
estimate remains conservative and includes the legacy host-input allowance;
this change reduces actual allocation, not the reservation bound.

For a 3839 x 2159 source, the omitted host planes contain 99,460,812 bytes
(94.85 MiB). This is an allocation-volume reduction, not a guaranteed reduction
in process peak footprint or a prediction of complete-encode latency.

Regression coverage includes all 256 sRGB byte values in strided RGB/RGBA
conversion, destination padding and guards, conflicting input sources,
callback failure and recovery, numeric validation, completed host-view and
Opsin parity, and resource-plan checks proving that forced Metal no longer
allocates the redundant host input. The CPU adapter retains its previous
allocation-failure expectations.
