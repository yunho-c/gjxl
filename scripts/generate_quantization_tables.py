#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

"""Generates built-in JPEG XL AC quantization matrices.

The parameters and construction follow libjxl commit
e8ff09762481785938d8e4e01333ed3917571161. Only the DCT table families used
by gjxl's currently supported Metal transforms are emitted.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "src/codec/quantization_tables_generated.h"
PINNED_LIBJXL_REVISION = "e8ff09762481785938d8e4e01333ed3917571161"


TABLES = (
    (
        "Dct8",
        8,
        8,
        1,
        1,
        (
            (3150.0, 0.0, -0.4, -0.4, -0.4, -2.0),
            (560.0, 0.0, -0.3, -0.3, -0.3, -0.3),
            (512.0, -2.0, -1.0, 0.0, -1.0, -2.0),
        ),
    ),
    (
        "Dct16",
        16,
        16,
        2,
        2,
        (
            (
                8996.8725711814115328,
                -1.3000777393353804,
                -0.49424529824571225,
                -0.439093774457103443,
                -0.6350101832695744,
                -0.90177264050827612,
                -1.6162099239887414,
            ),
            (
                3191.48366296844234752,
                -0.67424582104194355,
                -0.80745813428471001,
                -0.44925837484843441,
                -0.35865440981033403,
                -0.31322389111877305,
                -0.37615025315725483,
            ),
            (
                1157.50408145487200256,
                -2.0531423165804414,
                -1.4,
                -0.50687130033378396,
                -0.42708730624733904,
                -1.4856834539296244,
                -4.9209142884401604,
            ),
        ),
    ),
    (
        "Dct32",
        32,
        32,
        4,
        4,
        (
            (
                15718.40830982518931456,
                -1.025,
                -0.98,
                -0.9012,
                -0.4,
                -0.48819395464,
                -0.421064,
                -0.27,
            ),
            (
                7305.7636810695983104,
                -0.8041958212306401,
                -0.7633036457487539,
                -0.55660379990111464,
                -0.49785304658857626,
                -0.43699592683512467,
                -0.40180866526242109,
                -0.27321683125358037,
            ),
            (
                3803.53173721215041536,
                -3.060733579805728,
                -2.0413270132490346,
                -2.0235650159727417,
                -0.5495389509954993,
                -0.4,
                -0.4,
                -0.3,
            ),
        ),
    ),
    (
        "Dct8x16",
        8,
        16,
        1,
        2,
        (
            (
                7240.7734393502,
                -0.7,
                -0.7,
                -0.2,
                -0.2,
                -0.2,
                -0.5,
            ),
            (
                1448.15468787004,
                -0.5,
                -0.5,
                -0.5,
                -0.2,
                -0.2,
                -0.2,
            ),
            (
                506.854140754517,
                -1.4,
                -0.2,
                -0.5,
                -0.5,
                -1.5,
                -3.6,
            ),
        ),
    ),
    (
        "Dct16x32",
        16,
        32,
        2,
        4,
        (
            (
                13844.97076442300573,
                -0.97113799999999995,
                -0.658,
                -0.42026,
                -0.22712,
                -0.2206,
                -0.226,
                -0.6,
            ),
            (
                4798.964084220744293,
                -0.61125308982767057,
                -0.83770786552491361,
                -0.79014862079498627,
                -0.2692727459704829,
                -0.38272769465388551,
                -0.22924222653091453,
                -0.20719098826199578,
            ),
            (
                1807.236946760964614,
                -1.2,
                -1.2,
                -0.7,
                -0.7,
                -0.7,
                -0.4,
                -0.5,
            ),
        ),
    ),
)


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fused_multiply_add32(a: float, b: float, c: float) -> float:
    # math.fma was added after several still-common Python 3 releases. The
    # fallback remains exact for these float32 inputs because their product
    # and sum fit within Python's binary64 precision.
    fma = getattr(math, "fma", lambda x, y, z: x * y + z)
    return float32(fma(a, b, c))


def float32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fast_log2(value: float) -> float:
    # libjxl FastLog2f's degree-2 rational approximation.
    p = (
        float32(-1.8503833400518310e-06),
        float32(1.4287160470083755),
        float32(0.74245873327820566),
    )
    q = (
        float32(0.99032814277590719),
        float32(1.0096718572241148),
        float32(0.17409343003366853),
    )

    value_bits = float32_bits(value)
    exponent = (value_bits - 0x3F2AAAAB) >> 23
    mantissa = float32_from_bits(value_bits - (exponent << 23))
    x = float32(mantissa - float32(1.0))
    numerator = fused_multiply_add32(p[2], x, p[1])
    numerator = fused_multiply_add32(numerator, x, p[0])
    denominator = fused_multiply_add32(q[2], x, q[1])
    denominator = fused_multiply_add32(denominator, x, q[0])
    rational = float32(numerator / denominator)
    return float32(rational + float32(exponent))


def fast_pow2(value: float) -> float:
    # libjxl FastPow2f's rational approximation.
    floor_value = math.floor(value)
    exponent = float32_from_bits((floor_value + 127) << 23)
    fraction = float32(value - float32(floor_value))

    numerator = float32(fraction + float32(1.01749063e01))
    numerator = fused_multiply_add32(
        numerator,
        fraction,
        float32(4.88687798e01))
    numerator = fused_multiply_add32(
        numerator,
        fraction,
        float32(9.85506591e01))
    numerator = float32(numerator * exponent)

    denominator = fused_multiply_add32(
        fraction,
        float32(2.10242958e-01),
        float32(-2.22328856e-02))
    denominator = fused_multiply_add32(
        denominator,
        fraction,
        float32(-1.94414990e01))
    denominator = fused_multiply_add32(
        denominator,
        fraction,
        float32(9.85506633e01))
    return float32(numerator / denominator)


def fast_pow(base: float, exponent: float) -> float:
    return fast_pow2(float32(fast_log2(base) * exponent))


def multiplier(value: float) -> float:
    value = float32(value)
    if value > 0.0:
        return float32(float32(1.0) + value)
    return float32(float32(1.0) / float32(float32(1.0) - value))


def channel_weights(rows: int, columns: int, parameters: tuple[float, ...]) -> list[float]:
    bands = [float32(parameters[0])]
    for parameter in parameters[1:]:
        bands.append(float32(bands[-1] * multiplier(float32(parameter))))

    # Keep every scalar intermediate at float precision and use the same
    # rational power approximation as the pinned source construction.
    sqrt_two = float32(1.41421356237)
    distance_scale = float32(
        float32(len(bands) - 1) /
        float32(sqrt_two + float32(1.0e-6)))
    row_scale = float32(distance_scale / float32(rows - 1))
    column_scale = float32(distance_scale / float32(columns - 1))
    result: list[float] = []

    for y in range(rows):
        dy = float32(float32(y) * row_scale)
        dy_squared = float32(dy * dy)
        for x in range(columns):
            dx = float32(float32(x) * column_scale)
            position = float32(math.sqrt(fused_multiply_add32(dx, dx, dy_squared)))
            index = min(int(position), len(bands) - 2)
            fraction = float32(position - float32(index))
            a = bands[index]
            b = bands[index + 1]
            ratio = float32(b / a)
            power = fast_pow(ratio, fraction)
            result.append(float32(a * power))

    return result


def format_array(name: str, values: list[float]) -> list[str]:
    lines = [f"inline constexpr std::array<float, {len(values)}> {name} = {{"]
    for offset in range(0, len(values), 4):
        chunk = values[offset : offset + 4]
        literals = ", ".join(value.hex() + "f" for value in chunk)
        lines.append(f"  {literals},")
    lines.append("};")
    return lines


def generate() -> str:
    lines = [
        "// SPDX-License-Identifier: Apache-2.0",
        "// Copyright (c) 2026 Yunho Cho",
        "",
        "// Generated by scripts/generate_quantization_tables.py from the",
        f"// default DCT quantization parameters in libjxl {PINNED_LIBJXL_REVISION}.",
        "// Internal generated data; include through codec/quantization.cpp only.",
        "",
        "#pragma once",
        "",
        "#include <array>",
        "",
        "namespace gjxl::quantization_internal {",
        "",
    ]

    for table_name, rows, columns, lf_rows, lf_columns, channels in TABLES:
        dequant: list[float] = []
        inverse: list[float] = []

        for channel in channels:
            weights = channel_weights(rows, columns, channel)
            dequant.extend(float32(float32(1.0) / weight) for weight in weights)

            for y in range(rows):
                for x in range(columns):
                    index = y * columns + x
                    inverse.append(
                        0.0 if y < lf_rows and x < lf_columns else weights[index]
                    )

        lines.extend(format_array(f"k{table_name}Dequant", dequant))
        lines.append("")
        lines.extend(format_array(f"k{table_name}InverseDequant", inverse))
        lines.append("")

    lines.append("}  // namespace gjxl::quantization_internal")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    generated = generate()
    if args.check:
        if not args.output.exists() or args.output.read_text() != generated:
            raise SystemExit(f"{args.output} is not up to date")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)


if __name__ == "__main__":
    main()
