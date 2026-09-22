# Butteraugli traffic investigation — complete

The investigation is **complete at the user-approved practical stopping point**. Substantial exact-output gains are established; the final refinements are small and workload dependent. This is not a proof of a theoretical hardware optimum. All changes are isolated from the primary checkout.

Baseline: `4f3e414`, current main plus production-aligned profiling. Existing numerical tolerances are unchanged. See [opportunity inventory](OPPORTUNITIES.md), [traffic model](TRAFFIC-NOTES.md), and [incidents](INCIDENTS.md).

## Kernel screens

GPU command-buffer timing of repeated isolated filter dispatches. Each extent/variant uses five alternating pairs, three warmup submissions and five retained submissions per side, three dispatches per submission. Inputs, padding, output planes and guards are compared bitwise. These are not whole-encoder speedups.

| Family / variant | 4K change | 24 MP change | Status |
|---|---:|---:|---|
| geometry control | -0.05% (5/5) | -0.10% (4/5) | screened |
| geometry_w16_h64_t64 | -0.09% (5/5) | -0.06% (3/5) | screened |
| geometry_w16_h64_t32 | +0.09% (0/5) | -0.02% (3/5) | screened |
| geometry_w16_h96_t48 | -6.53% (5/5) | -6.69% (5/5) | screened |
| geometry_w16_h128_t64 | -9.05% (5/5) | -8.61% (5/5) | screened |
| geometry_w16_h128_t32 | +0.05% (2/5) | +0.39% (2/5) | screened |
| geometry_w8_h128_t128 | +9.57% (0/5) | +9.35% (0/5) | screened |
| geometry_w8_h128_t64 | +2.34% (0/5) | +3.22% (0/5) | screened |
| geometry_w8_h256_t128 | +4.30% (0/5) | +4.73% (0/5) | screened |
| geometry_w8_h256_t64 | +5.79% (0/5) | +5.67% (0/5) | screened |
| geometry_w4_h256_t256 | +55.55% (0/5) | +57.19% (0/5) | screened |
| geometry_w4_h512_t256 | +41.52% (0/5) | +42.09% (0/5) | screened |
| geometry_w32_h32_t32 | +17.29% (0/5) | +17.17% (0/5) | screened |
| geometry_w32_h48_t24 | +6.21% (0/5) | +6.85% (0/5) | screened |
| geometry_w32_h48_t16 | +12.84% (0/5) | +13.82% (0/5) | screened |
| rolling control | -0.03% (3/5) | -0.04% (3/5) | screened |
| rolling_s1_p1 | -0.11% (3/5) | -0.01% (4/5) | screened |
| rolling_s1_p2 | +0.12% (2/5) | +0.00% (2/5) | screened |
| rolling_s2_p1 | +3.73% (0/5) | +4.87% (0/5) | screened |
| rolling_s2_p2 | -6.02% (5/5) | -5.55% (5/5) | screened |
| rolling_s4_p1 | +6.58% (0/5) | +6.60% (0/5) | screened |
| rolling_s4_p2 | -9.40% (5/5) | -9.60% (5/5) | screened |
| rolling_s8_p1 | +11.22% (0/5) | +7.98% (0/5) | screened |
| rolling_s8_p2 | -9.29% (5/5) | -11.91% (5/5) | screened |
| adjacent control | +0.13% (1/5) | -0.10% (4/5) | screened |
| adjacent_o2_s1_p1 | +18.23% (0/5) | +18.42% (0/5) | screened |
| adjacent_o2_s1_p2 | +13.57% (0/5) | +13.67% (0/5) | screened |
| adjacent_o2_s4_p1 | +36.53% (0/5) | +37.66% (0/5) | screened |
| adjacent_o2_s4_p2 | +13.61% (0/5) | +14.07% (0/5) | screened |
| adjacent_o2_s8_p1 | +44.63% (0/5) | +41.70% (0/5) | screened |
| adjacent_o2_s8_p2 | +15.48% (0/5) | +13.77% (0/5) | screened |
| adjacent_o4_s1_p1 | +35.68% (0/5) | +35.65% (0/5) | screened |
| adjacent_o4_s1_p2 | +21.05% (0/5) | +18.81% (0/5) | screened |
| adjacent_o4_s4_p1 | +83.65% (0/5) | +83.47% (0/5) | screened |
| adjacent_o4_s4_p2 | +41.33% (0/5) | +37.44% (0/5) | screened |
| adjacent_o4_s8_p1 | +96.32% (0/5) | +91.29% (0/5) | screened |
| adjacent_o4_s8_p2 | +41.31% (0/5) | +37.30% (0/5) | screened |
| norm control | -0.02% (3/5) | -0.05% (4/5) | screened |
| norm_x_s1_p1 | +0.10% (2/5) | +0.49% (0/5) | screened |
| norm_x_s1_p2 | -0.03% (3/5) | +0.06% (2/5) | screened |
| norm_x_s8_p1 | +9.32% (0/5) | +6.84% (0/5) | screened |
| norm_x_s8_p2 | -9.31% (5/5) | -11.58% (5/5) | screened |
| norm_y_s1_p1 | +0.24% (0/5) | -0.11% (3/5) | screened |
| norm_y_s1_p2 | +1.26% (0/5) | +1.16% (0/5) | screened |
| norm_y_s8_p1 | +14.25% (0/5) | +12.18% (0/5) | screened |
| norm_y_s8_p2 | -8.47% (5/5) | -10.58% (5/5) | screened |
| norm_xy_s1_p1 | +0.99% (0/5) | +0.79% (0/5) | screened |
| norm_xy_s1_p2 | +1.22% (0/5) | +1.40% (0/5) | screened |
| norm_xy_s8_p1 | +13.72% (0/5) | +11.11% (0/5) | screened |
| norm_xy_s8_p2 | -9.40% (5/5) | -7.34% (4/5) | screened |
| interior control | -0.02% (3/5) | +4.98% (0/5) | screened |
| interior_o2_s1_p1 | -15.31% (5/5) | -16.97% (5/5) | screened |
| interior_o2_s1_p2 | -19.34% (5/5) | -20.57% (5/5) | screened |
| interior_o2_s8_p1 | +3.88% (0/5) | +3.65% (0/5) | screened |
| interior_o2_s8_p2 | -11.53% (5/5) | -11.30% (5/5) | screened |
| interior_o4_s1_p1 | -9.45% (5/5) | -11.12% (5/5) | screened |
| interior_o4_s1_p2 | -18.84% (5/5) | -20.10% (5/5) | screened |
| interior_o4_s8_p1 | +29.15% (0/5) | +23.48% (0/5) | screened |
| interior_o4_s8_p2 | -1.25% (4/5) | +3.18% (0/5) | screened |
| short control | -0.09% (3/5) | -0.13% (3/5) | screened |
| short_r3_w16_h32_p1 | -16.79% (5/5) | -27.57% (5/5) | screened |
| short_r3_w16_h32_p2 | -20.33% (5/5) | -30.41% (5/5) | screened |
| short_r3_w16_h64_p1 | -19.76% (5/5) | -30.82% (5/5) | screened |
| short_r3_w16_h64_p2 | -22.16% (5/5) | -33.01% (5/5) | screened |
| short_r3_w32_h32_p1 | -15.91% (5/5) | -26.92% (5/5) | screened |
| short_r3_w32_h32_p2 | -17.79% (5/5) | -29.20% (5/5) | screened |
| short_r3_w16_h128_p2 | -23.39% (5/5) | -34.16% (5/5) | screened |
| short_r3_w32_h64_p2 | -21.26% (5/5) | -32.26% (5/5) | screened |
| short_r6_w16_h32_p1 | +28.88% (0/5) | +20.06% (0/5) | screened |
| short_r6_w16_h32_p2 | +28.57% (0/5) | +16.88% (0/5) | screened |
| short_r6_w16_h64_p1 | +19.63% (0/5) | +10.45% (0/5) | screened |
| short_r6_w16_h64_p2 | +17.92% (0/5) | +9.57% (0/5) | screened |
| short_r6_w32_h32_p1 | +30.34% (0/5) | +21.68% (0/5) | screened |
| short_r6_w32_h32_p2 | +28.80% (0/5) | +18.90% (0/5) | screened |
| short_r6_w16_h128_p2 | +13.48% (0/5) | +5.69% (0/5) | screened |
| short_r6_w32_h64_p2 | +19.39% (0/5) | +11.27% (0/5) | screened |
| short_r7_w16_h32_p1 | +34.60% (0/5) | +29.59% (0/5) | screened |
| short_r7_w16_h32_p2 | +32.39% (0/5) | +27.54% (0/5) | screened |
| short_r7_w16_h64_p1 | +23.53% (0/5) | +19.20% (0/5) | screened |
| short_r7_w16_h64_p2 | +22.29% (0/5) | +16.85% (0/5) | screened |
| short_r7_w32_h32_p1 | +35.57% (0/5) | +29.35% (0/5) | screened |
| short_r7_w32_h32_p2 | +34.61% (0/5) | +29.34% (0/5) | screened |
| short_r7_w16_h128_p2 | +16.54% (0/5) | +12.65% (0/5) | screened |
| short_r7_w32_h64_p2 | +23.44% (0/5) | +18.35% (0/5) | screened |
| malta control | -0.22% (4/5) | -0.43% (3/5) | screened |
| malta_w16_h16_p1 | -0.25% (4/5) | -0.65% (5/5) | screened |
| malta_w16_h32_p1 | +3.35% (1/5) | +2.50% (0/5) | screened |
| malta_w16_h32_p2 | -0.25% (4/5) | -1.08% (5/5) | screened |
| malta_w32_h8_p1 | +0.77% (0/5) | +0.78% (0/5) | screened |
| malta_w32_h16_p1 | +2.01% (0/5) | +1.66% (1/5) | screened |
| malta_w32_h16_p2 | -3.02% (5/5) | -3.02% (5/5) | screened |
| malta_w32_h32_p1 | +3.86% (1/5) | +3.78% (0/5) | screened |
| malta_w32_h32_p2 | -1.41% (5/5) | -2.18% (5/5) | screened |
| malta_w64_h8_p1 | +5.36% (1/5) | +5.36% (0/5) | screened |
| malta_w64_h16_p1 | +5.23% (1/5) | +4.61% (0/5) | screened |
| malta_w64_h16_p2 | -2.18% (5/5) | -2.40% (5/5) | screened |
| malta_w32_h64_p2 | +7.18% (1/5) | +6.87% (0/5) | screened |
| vertical control | +0.27% (2/5) | -0.07% (5/5) | screened |
| vertical_o2_s1_p2 | -32.93% (5/5) | -34.76% (5/5) | screened |
| vertical_o2_s8_p2 | -20.38% (5/5) | -7.15% (5/5) | screened |
| vertical_o4_s1_p2 | -25.57% (5/5) | -27.29% (5/5) | screened |
| vertical_o4_s8_p2 | -4.15% (5/5) | +17.20% (0/5) | screened |
| malta-adj control | -0.04% (3/5) | -0.04% (3/5) | screened |
| malta_adj_w16_h32_p2 | -3.39% (5/5) | -4.14% (5/5) | screened |
| malta_adj_w16_h64_p4 | +3.02% (0/5) | +2.13% (0/5) | screened |
| malta_adj_w32_h16_p2 | -3.12% (5/5) | -3.45% (5/5) | screened |
| malta_adj_w32_h32_p2 | -2.09% (5/5) | -2.68% (5/5) | screened |
| malta_adj_w32_h32_p4 | -1.26% (5/5) | -2.10% (5/5) | screened |
| malta_adj_w32_h64_p4 | -1.20% (5/5) | -1.75% (5/5) | screened |
| malta_adj_w64_h16_p2 | -2.39% (5/5) | -1.93% (5/5) | screened |
| malta_adj_w64_h32_p4 | +0.52% (0/5) | -0.48% (5/5) | screened |
| vertical-wide control | -0.03% (3/5) | -0.06% (5/5) | screened |
| vertical_wide_o2_p4 | -37.06% (5/5) | -37.86% (5/5) | screened |
| vertical_wide_o2_p8 | -23.51% (5/5) | -23.25% (5/5) | screened |
| vertical_wide_o4_p4 | -33.70% (5/5) | -35.05% (5/5) | screened |
| vertical_wide_o4_p8 | -24.44% (5/5) | -26.73% (5/5) | screened |
| reuse-geometry control | -0.04% (3/5) | -0.09% (5/5) | screened |
| reuse_w8_h128_p2 | -18.87% (5/5) | -20.10% (5/5) | screened |
| reuse_w8_h128_p4 | -26.23% (5/5) | -27.23% (5/5) | screened |
| reuse_w8_h128_p8 | -24.05% (5/5) | -23.75% (5/5) | screened |
| reuse_w8_h256_p2 | -22.42% (5/5) | -23.53% (5/5) | screened |
| reuse_w8_h256_p4 | -20.91% (5/5) | -23.06% (5/5) | screened |
| reuse_w8_h256_p8 | -5.52% (5/5) | -8.75% (5/5) | screened |
| reuse_w16_h64_p2 | -33.55% (5/5) | -35.28% (5/5) | screened |
| reuse_w16_h64_p4 | -37.13% (5/5) | -38.03% (5/5) | screened |
| reuse_w16_h64_p8 | -23.14% (5/5) | -22.83% (5/5) | screened |
| reuse_w16_h96_p2 | -37.06% (5/5) | -37.88% (5/5) | screened |
| reuse_w16_h96_p4 | -38.93% (5/5) | -38.95% (5/5) | screened |
| reuse_w16_h96_p8 | -26.96% (5/5) | -27.72% (5/5) | screened |
| reuse_w16_h128_p2 | -38.04% (5/5) | -37.76% (5/5) | screened |
| reuse_w16_h128_p4 | -36.79% (5/5) | -36.63% (5/5) | screened |
| reuse_w16_h128_p8 | -20.78% (5/5) | -20.84% (5/5) | screened |
| reuse_w16_h136_p4 | -37.32% (5/5) | -37.21% (5/5) | screened |
| reuse_w16_h136_p8 | -22.35% (5/5) | -22.81% (5/5) | screened |
| reuse_w32_h32_p2 | -27.86% (5/5) | -29.19% (5/5) | screened |
| reuse_w32_h32_p4 | -25.15% (5/5) | -26.84% (5/5) | screened |
| reuse_w32_h32_p8 | +1.38% (0/5) | -2.25% (5/5) | screened |
| reuse_w32_h48_p2 | -35.20% (5/5) | -36.11% (5/5) | screened |
| reuse_w32_h48_p4 | -30.19% (5/5) | -31.40% (5/5) | screened |
| reuse_w32_h48_p8 | -4.50% (5/5) | -6.44% (5/5) | screened |
| reuse-closure control | -0.05% (5/5) | -0.06% (5/5) | screened |
| reuse_close_h64_s1_b0 | -37.15% (5/5) | -37.80% (5/5) | screened |
| reuse_close_h96_s1_b0 | -38.80% (5/5) | -38.77% (5/5) | screened |
| reuse_close_h64_s2_b0 | -39.88% (5/5) | -37.92% (5/5) | screened |
| reuse_close_h64_s4_b0 | -38.75% (5/5) | -35.04% (5/5) | screened |
| reuse_close_h96_s2_b0 | -38.48% (5/5) | -36.96% (5/5) | screened |
| reuse_close_h96_s4_b0 | -37.62% (5/5) | -26.35% (5/5) | screened |
| reuse_close_h64_s1_b256 | -37.14% (5/5) | -37.90% (5/5) | screened |
| reuse_close_h96_s1_b384 | -38.79% (5/5) | -38.65% (5/5) | screened |
| short-reuse control | -0.18% (4/5) | +0.51% (1/5) | screened |
| short_reuse_r3_w16_h64_p2 | -53.22% (5/5) | -55.64% (5/5) | screened |
| short_reuse_r3_w16_h64_p4 | -56.34% (5/5) | -57.03% (5/5) | screened |
| short_reuse_r6_w16_h64_p2 | -45.27% (5/5) | -48.86% (5/5) | screened |
| short_reuse_r6_w16_h64_p4 | -50.19% (5/5) | -53.57% (5/5) | screened |
| short_reuse_r6_w16_h128_p2 | -45.51% (5/5) | -47.30% (5/5) | screened |
| short_reuse_r6_w16_h128_p4 | -50.59% (5/5) | -53.03% (5/5) | screened |
| short_reuse_r7_w16_h64_p2 | -45.71% (5/5) | -47.41% (5/5) | screened |
| short_reuse_r7_w16_h64_p4 | -50.27% (5/5) | -52.68% (5/5) | screened |
| short_reuse_r7_w16_h128_p2 | -46.10% (5/5) | -45.91% (5/5) | screened |
| short_reuse_r7_w16_h128_p4 | -51.39% (5/5) | -52.45% (5/5) | screened |
| short-interactions control | +0.12% (2/5) | -1.04% (4/5) | screened |
| short_interaction_r3_w16_h64_p4 | -55.65% (5/5) | -57.67% (5/5) | screened |
| short_interaction_r3_w8_h64_p4 | -51.77% (5/5) | -52.57% (5/5) | screened |
| short_interaction_r3_w32_h64_p4 | -57.49% (5/5) | -60.75% (5/5) | screened |
| short_interaction_r3_w16_h32_p4 | -55.37% (5/5) | -57.79% (5/5) | screened |
| short_interaction_r3_w32_h32_p4 | -57.72% (5/5) | -61.32% (5/5) | screened |
| short_interaction_r3_w16_h64_p8 | -55.38% (5/5) | -57.95% (5/5) | screened |
| short_interaction_r3_w16_h128_p8 | -55.17% (5/5) | -57.28% (5/5) | screened |
| short_interaction_r6_w16_h64_p4 | -50.46% (5/5) | -55.37% (5/5) | screened |
| short_interaction_r6_w8_h64_p4 | -47.21% (5/5) | -50.84% (5/5) | screened |
| short_interaction_r6_w32_h64_p4 | -48.48% (5/5) | -53.35% (5/5) | screened |
| short_interaction_r6_w16_h32_p4 | -46.91% (5/5) | -52.13% (5/5) | screened |
| short_interaction_r6_w32_h32_p4 | -45.65% (5/5) | -50.81% (5/5) | screened |
| short_interaction_r6_w16_h64_p8 | -52.35% (5/5) | -56.87% (5/5) | screened |
| short_interaction_r6_w16_h128_p8 | -52.90% (5/5) | -56.01% (5/5) | screened |
| short_interaction_r7_w16_h64_p4 | -50.27% (5/5) | -53.07% (5/5) | screened |
| short_interaction_r7_w8_h64_p4 | -47.92% (5/5) | -51.23% (5/5) | screened |
| short_interaction_r7_w32_h64_p4 | -49.24% (5/5) | -52.33% (5/5) | screened |
| short_interaction_r7_w16_h32_p4 | -45.56% (5/5) | -48.92% (5/5) | screened |
| short_interaction_r7_w32_h32_p4 | -45.34% (5/5) | -48.91% (5/5) | screened |
| short_interaction_r7_w16_h64_p8 | -53.45% (5/5) | -55.53% (5/5) | screened |
| short_interaction_r7_w16_h128_p8 | -53.57% (5/5) | -55.97% (5/5) | screened |
| short-horizontal control | +0.09% (1/5) | -0.01% (3/5) | screened |
| short_horizontal_r3_w16_h64_p4_q2 | -56.39% (5/5) | -56.84% (5/5) | screened |
| short_horizontal_r3_w16_h64_p4_q4 | -57.90% (5/5) | -57.38% (5/5) | screened |
| short_horizontal_r3_w16_h64_p4_q8 | -57.02% (5/5) | -57.19% (5/5) | screened |
| short_horizontal_r3_w32_h32_p4_q2 | -58.50% (5/5) | -60.40% (5/5) | screened |
| short_horizontal_r3_w32_h32_p4_q4 | -59.91% (5/5) | -61.28% (5/5) | screened |
| short_horizontal_r3_w32_h32_p4_q8 | -58.73% (5/5) | -61.13% (5/5) | screened |
| short_horizontal_r6_w16_h64_p4_q2 | -50.10% (5/5) | -53.43% (5/5) | screened |
| short_horizontal_r6_w16_h64_p4_q4 | -50.07% (5/5) | -53.54% (5/5) | screened |
| short_horizontal_r6_w16_h64_p4_q8 | -46.49% (5/5) | -50.07% (5/5) | screened |
| short_horizontal_r6_w16_h64_p8_q2 | -52.07% (5/5) | -55.77% (5/5) | screened |
| short_horizontal_r6_w16_h64_p8_q4 | -52.91% (5/5) | -55.73% (5/5) | screened |
| short_horizontal_r6_w16_h64_p8_q8 | -49.86% (5/5) | -53.43% (5/5) | screened |
| short_horizontal_r7_w16_h64_p4_q2 | -50.19% (5/5) | -52.18% (5/5) | screened |
| short_horizontal_r7_w16_h64_p4_q4 | -51.55% (5/5) | -53.73% (5/5) | screened |
| short_horizontal_r7_w16_h64_p4_q8 | -49.03% (5/5) | -51.03% (5/5) | screened |
| short_horizontal_r7_w16_h64_p8_q2 | -52.14% (5/5) | -54.44% (5/5) | screened |
| short_horizontal_r7_w16_h64_p8_q4 | -54.62% (5/5) | -56.81% (5/5) | screened |
| short_horizontal_r7_w16_h64_p8_q8 | -51.94% (5/5) | -54.06% (5/5) | screened |

## Complete encoding and attribution

Initial exploratory cohorts: three alternating independent-process pairs, two warmups and three measured complete calls per process. Profiling and ordinary timing are separate. Inputs are Alpine Lake 24 MP/e7 and Forest Stream 48 MP/e10, distance 1.9, eight CPU participants. Successful paired records require identical baseline/candidate codestream hashes, deterministic per-call bytes/summaries, equal submission counts, and no detected competing build/benchmark. Three pairs do not qualify sub-percent improvements. Cohorts labeled confirm instead use seven pairs, three warmups and seven samples. The dc-sub-increment cohort compares against main-only DC fusion, and ac-increment compares against main+sub DC fusion; older cohorts otherwise use the original baseline; newer cohort baselines are given in the table and identity.json. Displayed percentages are medians of per-pair ratios, not ratios of separately aggregated medians.

| Cohort / input / baseline | Complete-call change | Low/medium stage change |
|---|---:|---:|
| integrated-control / alpine24-e7 / baseline (ordinary) | +0.086% (1/3) | not instrumented |
| geometry-integrated-wall / alpine24-e7 / baseline (ordinary) | -0.524% (3/3) | not instrumented |
| geometry-integrated-wall / forest48-e10 / baseline (ordinary) | +1.745% (1/3) | not instrumented |
| integrated-control-stage-v2 / alpine24-e7 / baseline-v2 (profiled) | +1.583% (1/3) | -0.154% (2/3) |
| geometry-integrated-stage-v2 / alpine24-e7 / baseline-v2 (profiled) | +0.192% (0/3) | -9.690% (3/3) |
| geometry-integrated-stage-v2 / forest48-e10 / baseline-v2 (profiled) | -3.918% (3/3) | -11.189% (3/3) |
| rolling-integrated-wall-v2 / alpine24-e7 / baseline-v2 (ordinary) | +0.066% (1/3) | not instrumented |
| rolling-integrated-wall-v2 / forest48-e10 / baseline-v2 (ordinary) | -0.487% (2/3) | not instrumented |
| rolling-integrated-stage-v2 / alpine24-e7 / baseline-v2 (profiled) | +1.412% (1/3) | -11.760% (2/3) |
| rolling-integrated-stage-v2 / forest48-e10 / baseline-v2 (profiled) | +0.177% (1/3) | -1.502% (3/3) |
| dc-integrated-stage / alpine24-e7 / baseline-v2 (profiled) | -1.104% (2/3) | +0.314% (1/3) |
| dc-integrated-stage / forest48-e10 / baseline-v2 (profiled) | -1.754% (3/3) | +2.078% (0/3) |
| dc-integrated-wall-confirm / alpine24-e7 / baseline-v2 (ordinary) | -0.508% (5/7) | not instrumented |
| dc-integrated-wall-confirm / forest48-e10 / baseline-v2 (ordinary) | -0.947% (6/7) | not instrumented |
| dc-sub-increment-stage / alpine24-e7 / dc-fusion (profiled) | -0.256% (2/3) | +0.235% (0/3) |
| dc-sub-increment-stage / forest48-e10 / dc-fusion (profiled) | +0.053% (1/3) | -0.219% (2/3) |
| interior-integrated-stage / alpine24-e7 / baseline-v2 (profiled) | -1.375% (3/3) | -21.287% (3/3) |
| interior-integrated-stage / forest48-e10 / baseline-v2 (profiled) | -1.679% (2/3) | -22.411% (3/3) |
| control-wall-confirm / alpine24-e7 / baseline-v2 (ordinary) | +0.401% (3/7) | not instrumented |
| control-wall-confirm / forest48-e10 / baseline-v2 (ordinary) | -0.430% (6/7) | not instrumented |
| ac-increment-stage / alpine24-e7 / dc-sub-fusion (profiled) | -0.829% (3/3) | -0.088% (2/3) |
| ac-increment-stage / forest48-e10 / dc-sub-fusion (profiled) | -2.319% (3/3) | -0.110% (2/3) |
| dc-interior-stage / alpine24-e7 / baseline-v2 (profiled) | +0.658% (1/3) | -10.093% (3/3) |
| dc-interior-stage / forest48-e10 / baseline-v2 (profiled) | -0.944% (3/3) | -17.660% (3/3) |
| interior-wall-confirm / alpine24-e7 / baseline-v2 (ordinary) | -2.939% (7/7) | not instrumented |
| interior-wall-confirm / forest48-e10 / baseline-v2 (ordinary) | -1.530% (6/7) | not instrumented |
| dc-interior-wall-confirm / alpine24-e7 / baseline-v2 (ordinary) | -2.538% (7/7) | not instrumented |
| dc-interior-wall-confirm / forest48-e10 / baseline-v2 (ordinary) | -2.504% (7/7) | not instrumented |
| ac-interior-stage / alpine24-e7 / baseline-v2 (profiled) | -4.261% (3/3) | -19.036% (3/3) |
| ac-interior-stage / forest48-e10 / baseline-v2 (profiled) | -3.679% (3/3) | -18.826% (3/3) |
| ac-interior-broad-wall / campus12-e7 / baseline-v2 (ordinary) | -4.845% (3/3) | not instrumented |
| ac-interior-broad-wall / alpine24-e7 / baseline-v2 (ordinary) | -4.987% (3/3) | not instrumented |
| ac-interior-broad-wall / forest48-e10 / baseline-v2 (ordinary) | -3.261% (3/3) | not instrumented |
| ac-interior-broad-wall / alpine12-e10-d08 / baseline-v2 (ordinary) | -3.362% (3/3) | not instrumented |
| ac-interior-broad-wall / forest24-e7-d08 / baseline-v2 (ordinary) | -4.260% (3/3) | not instrumented |
| ac-interior-broad-wall / campus48-e10-d08 / baseline-v2 (ordinary) | -4.208% (3/3) | not instrumented |
| ac-interior-broad-wall / kodak01-e7 / baseline-v2 (ordinary) | +1.640% (1/3) | not instrumented |
| ac-interior-broad-wall / kodak17-e10-d08 / baseline-v2 (ordinary) | -1.352% (2/3) | not instrumented |
| ultra-stage / alpine24-e7 / baseline-v2 (profiled) | -0.701% (3/3) | +5.961% (1/3) |
| ultra-stage / forest48-e10 / baseline-v2 (profiled) | -1.347% (3/3) | +0.036% (1/3) |
| malta-geometry-stage / alpine24-e7 / baseline-v2 (profiled) | -0.833% (3/3) | -0.614% (3/3) |
| malta-geometry-stage / forest48-e10 / baseline-v2 (profiled) | -0.421% (2/3) | -0.615% (2/3) |
| materialized-increment-stage / alpine24-e7 / ac-fusion (profiled) | +4.905% (1/3) | +5.134% (1/3) |
| materialized-increment-stage / forest48-e10 / ac-fusion (profiled) | +0.681% (1/3) | -0.814% (2/3) |
| vertical-increment-stage / alpine24-e7 / ac-interior (profiled) | -1.385% (3/3) | -17.306% (3/3) |
| vertical-increment-stage / forest48-e10 / ac-interior (profiled) | -1.192% (3/3) | -17.284% (3/3) |
| ultra-bundle-increment-stage / alpine24-e7 / ac-vertical (profiled) | -2.020% (3/3) | +0.452% (1/3) |
| ultra-bundle-increment-stage / forest48-e10 / ac-vertical (profiled) | -2.861% (3/3) | +0.002% (1/3) |
| malta-bundle-increment-stage / alpine24-e7 / fusion-bundle (profiled) | +0.797% (1/3) | -7.600% (3/3) |
| malta-bundle-increment-stage / forest48-e10 / fusion-bundle (profiled) | -0.514% (3/3) | +0.045% (0/3) |
| address-increment-stage / alpine24-e7 / ac-fusion (profiled) | -0.068% (2/3) | +0.087% (1/3) |
| address-increment-stage / forest48-e10 / ac-fusion (profiled) | +0.524% (1/3) | +0.016% (1/3) |
| fusion-bundle-broad-wall / campus12-e7 / baseline-v2 (ordinary) | -6.114% (3/3) | not instrumented |
| fusion-bundle-broad-wall / alpine24-e7 / baseline-v2 (ordinary) | -7.404% (3/3) | not instrumented |
| fusion-bundle-broad-wall / forest48-e10 / baseline-v2 (ordinary) | -7.389% (3/3) | not instrumented |
| fusion-bundle-broad-wall / alpine12-e10-d08 / baseline-v2 (ordinary) | -5.005% (3/3) | not instrumented |
| fusion-bundle-broad-wall / forest24-e7-d08 / baseline-v2 (ordinary) | -6.981% (3/3) | not instrumented |
| fusion-bundle-broad-wall / campus48-e10-d08 / baseline-v2 (ordinary) | -5.845% (3/3) | not instrumented |
| fusion-bundle-broad-wall / kodak01-e7 / baseline-v2 (ordinary) | -3.335% (3/3) | not instrumented |
| fusion-bundle-broad-wall / kodak17-e10-d08 / baseline-v2 (ordinary) | -3.080% (3/3) | not instrumented |
| mask-compose-increment-stage / alpine24-e7 / ac-fusion (profiled) | +0.619% (0/3) | -0.353% (2/3) |
| mask-compose-increment-stage / forest48-e10 / ac-fusion (profiled) | +0.648% (0/3) | +0.031% (0/3) |
| mask-compose-transpose-increment-stage / alpine24-e7 / ac-fusion (profiled) | +0.261% (1/3) | +0.031% (1/3) |
| mask-compose-transpose-increment-stage / forest48-e10 / ac-fusion (profiled) | +0.144% (1/3) | +0.021% (1/3) |
| reuse-h64-bundle-increment-stage / alpine24-e7 / fusion-bundle (profiled) | -0.962% (2/3) | -13.387% (3/3) |
| reuse-h64-bundle-increment-stage / forest48-e10 / fusion-bundle (profiled) | +2.062% (1/3) | -0.553% (3/3) |
| reuse-h96-bundle-increment-stage / alpine24-e7 / fusion-bundle (profiled) | -1.325% (2/3) | -8.833% (3/3) |
| reuse-h96-bundle-increment-stage / forest48-e10 / fusion-bundle (profiled) | +0.962% (0/3) | -2.646% (3/3) |
| fusion-bundle-original-stage / alpine24-e7 / baseline-v2 (profiled) | -8.639% (3/3) | -35.911% (3/3) |
| fusion-bundle-original-stage / forest48-e10 / baseline-v2 (profiled) | -6.885% (3/3) | -32.646% (3/3) |
| fusion-bundle-wall-confirm / alpine24-e7 / baseline-v2 (ordinary) | -7.659% (7/7) | not instrumented |
| fusion-bundle-wall-confirm / forest48-e10 / baseline-v2 (ordinary) | -6.883% (7/7) | not instrumented |
| fusion-bundle-wall-confirm / kodak01-e7 / baseline-v2 (ordinary) | -2.483% (6/7) | not instrumented |
| fusion-bundle-wall-confirm / kodak17-e10-d08 / baseline-v2 (ordinary) | -2.275% (7/7) | not instrumented |
| control-wall-final / alpine24-e7 / baseline-v2 (ordinary) | -0.161% (5/7) | not instrumented |
| control-wall-final / forest48-e10 / baseline-v2 (ordinary) | +0.322% (2/7) | not instrumented |
| reuse-h64-bundle-increment-wall / alpine24-e7 / fusion-bundle (ordinary) | +0.177% (1/3) | not instrumented |
| reuse-h64-bundle-increment-wall / forest48-e10 / fusion-bundle (ordinary) | +0.249% (1/3) | not instrumented |
| reuse-h96-bundle-increment-wall / alpine24-e7 / fusion-bundle (ordinary) | +0.309% (1/3) | not instrumented |
| reuse-h96-bundle-increment-wall / forest48-e10 / fusion-bundle (ordinary) | -1.187% (2/3) | not instrumented |
| malta-bundle-increment-wall / alpine24-e7 / fusion-bundle (ordinary) | +4.928% (0/3) | not instrumented |
| malta-bundle-increment-wall / forest48-e10 / fusion-bundle (ordinary) | -0.782% (2/3) | not instrumented |
| opsin32-bundle-increment-stage / alpine24-e7 / fusion-bundle (profiled) | +1.218% (0/3) | +3.487% (0/3) |
| opsin32-bundle-increment-stage / forest48-e10 / fusion-bundle (profiled) | -1.133% (3/3) | +0.165% (1/3) |
| opsin32-bundle-increment-wall / alpine24-e7 / fusion-bundle (ordinary) | -0.768% (3/3) | not instrumented |
| opsin32-bundle-increment-wall / forest48-e10 / fusion-bundle (ordinary) | +0.066% (1/3) | not instrumented |
| mask16-bundle-increment-stage / alpine24-e7 / fusion-bundle (profiled) | -0.329% (3/3) | -0.231% (2/3) |
| mask16-bundle-increment-stage / forest48-e10 / fusion-bundle (profiled) | -0.526% (3/3) | -0.111% (2/3) |
| mask16-bundle-increment-wall / alpine24-e7 / fusion-bundle (ordinary) | -0.025% (2/3) | not instrumented |
| mask16-bundle-increment-wall / forest48-e10 / fusion-bundle (ordinary) | -0.265% (2/3) | not instrumented |
| ultra-reuse-bundle-increment-stage / alpine24-e7 / fusion-bundle (profiled) | +0.615% (1/3) | -0.193% (2/3) |
| ultra-reuse-bundle-increment-stage / forest48-e10 / fusion-bundle (profiled) | -0.420% (2/3) | -0.243% (2/3) |
| ultra-reuse-bundle-increment-wall / alpine24-e7 / fusion-bundle (ordinary) | +0.949% (1/3) | not instrumented |
| ultra-reuse-bundle-increment-wall / forest48-e10 / fusion-bundle (ordinary) | -0.773% (3/3) | not instrumented |
| high-reuse-bundle-increment-stage / alpine24-e7 / fusion-bundle (profiled) | -3.057% (3/3) | +0.222% (1/3) |
| high-reuse-bundle-increment-stage / forest48-e10 / fusion-bundle (profiled) | -2.117% (3/3) | -0.025% (2/3) |
| high-reuse-bundle-increment-wall / alpine24-e7 / fusion-bundle (ordinary) | -2.907% (3/3) | not instrumented |
| high-reuse-bundle-increment-wall / forest48-e10 / fusion-bundle (ordinary) | -2.656% (3/3) | not instrumented |
| mask-reuse-bundle-increment-stage / alpine24-e7 / fusion-bundle (profiled) | -1.289% (2/3) | +3.282% (0/3) |
| mask-reuse-bundle-increment-stage / forest48-e10 / fusion-bundle (profiled) | -0.550% (2/3) | +0.209% (0/3) |
| mask-reuse-bundle-increment-wall / alpine24-e7 / fusion-bundle (ordinary) | -1.459% (3/3) | not instrumented |
| mask-reuse-bundle-increment-wall / forest48-e10 / fusion-bundle (ordinary) | -0.007% (2/3) | not instrumented |
| medium-reuse-bundle-increment-stage / alpine24-e7 / fusion-bundle (profiled) | -2.059% (3/3) | +0.469% (1/3) |
| medium-reuse-bundle-increment-stage / forest48-e10 / fusion-bundle (profiled) | -0.809% (3/3) | -0.046% (2/3) |
| medium-reuse-bundle-increment-wall / alpine24-e7 / fusion-bundle (ordinary) | -1.286% (3/3) | not instrumented |
| medium-reuse-bundle-increment-wall / forest48-e10 / fusion-bundle (ordinary) | +0.341% (1/3) | not instrumented |
| short-bundle-increment-stage / alpine24-e7 / fusion-bundle (profiled) | -6.048% (3/3) | +1.167% (1/3) |
| short-bundle-increment-stage / forest48-e10 / fusion-bundle (profiled) | -4.463% (3/3) | +0.035% (1/3) |
| short-bundle-increment-wall / alpine24-e7 / fusion-bundle (ordinary) | -4.795% (3/3) | not instrumented |
| short-bundle-increment-wall / forest48-e10 / fusion-bundle (ordinary) | -3.629% (2/3) | not instrumented |
| short-bundle-broad-wall / alpine24-e7 / baseline-v2 (ordinary) | -12.424% (3/3) | not instrumented |
| short-bundle-broad-wall / forest48-e10 / baseline-v2 (ordinary) | -10.948% (3/3) | not instrumented |
| short-bundle-broad-wall / campus12-e7 / baseline-v2 (ordinary) | -12.276% (3/3) | not instrumented |
| short-bundle-broad-wall / alpine12-e10-d08 / baseline-v2 (ordinary) | -9.521% (3/3) | not instrumented |
| short-bundle-broad-wall / forest24-e7-d08 / baseline-v2 (ordinary) | -11.527% (3/3) | not instrumented |
| short-bundle-broad-wall / campus48-e10-d08 / baseline-v2 (ordinary) | -9.253% (3/3) | not instrumented |
| short-bundle-broad-wall / kodak01-e7 / baseline-v2 (ordinary) | -2.214% (2/3) | not instrumented |
| short-bundle-broad-wall / kodak17-e10-d08 / baseline-v2 (ordinary) | -3.995% (3/3) | not instrumented |
| short-bundle-wall-confirm / alpine24-e7 / baseline-v2 (ordinary) | -12.697% (7/7) | not instrumented |
| short-bundle-wall-confirm / forest48-e10 / baseline-v2 (ordinary) | -10.715% (7/7) | not instrumented |
| short-bundle-wall-confirm / kodak01-e7 / baseline-v2 (ordinary) | -5.121% (6/7) | not instrumented |
| short-bundle-wall-confirm / kodak17-e10-d08 / baseline-v2 (ordinary) | -5.468% (7/7) | not instrumented |
| control-wall-short-final / alpine24-e7 / baseline-v2 (ordinary) | -0.471% (6/7) | not instrumented |
| control-wall-short-final / forest48-e10 / baseline-v2 (ordinary) | +0.364% (1/7) | not instrumented |
| short-bundle-original-stage / alpine24-e7 / baseline-v2 (profiled) | -12.388% (3/3) | -34.227% (3/3) |
| short-bundle-original-stage / forest48-e10 / baseline-v2 (profiled) | -11.326% (3/3) | -32.733% (3/3) |
| short-bundle-high-increment-wall-confirm / alpine24-e7 / high-reuse (ordinary) | -2.104% (7/7) | not instrumented |
| short-bundle-high-increment-wall-confirm / forest48-e10 / high-reuse (ordinary) | -2.299% (7/7) | not instrumented |
| short-bundle-high-increment-wall-confirm / kodak01-e7 / high-reuse (ordinary) | -0.008% (4/7) | not instrumented |
| short-bundle-high-increment-wall-confirm / kodak17-e10-d08 / high-reuse (ordinary) | -1.123% (6/7) | not instrumented |
| short-eight-increment-stage / alpine24-e7 / short-bundle (profiled) | -0.068% (2/3) | +0.060% (1/3) |
| short-eight-increment-stage / forest48-e10 / short-bundle (profiled) | -0.279% (2/3) | +0.514% (1/3) |
| short-eight-increment-wall / alpine24-e7 / short-bundle (ordinary) | +1.593% (1/3) | not instrumented |
| short-eight-increment-wall / forest48-e10 / short-bundle (ordinary) | -1.023% (2/3) | not instrumented |
| short-wide-ultra-increment-stage / alpine24-e7 / short-bundle (profiled) | -0.935% (2/3) | +0.441% (0/3) |
| short-wide-ultra-increment-stage / forest48-e10 / short-bundle (profiled) | +0.653% (1/3) | -1.303% (2/3) |
| short-wide-ultra-increment-wall / alpine24-e7 / short-bundle (ordinary) | -0.206% (3/3) | not instrumented |
| short-wide-ultra-increment-wall / forest48-e10 / short-bundle (ordinary) | -0.602% (2/3) | not instrumented |
| short-final-increment-stage / alpine24-e7 / short-bundle (profiled) | +0.167% (1/3) | -2.974% (2/3) |
| short-final-increment-stage / forest48-e10 / short-bundle (profiled) | -0.589% (3/3) | +0.166% (0/3) |
| short-final-increment-wall-confirm / alpine24-e7 / short-bundle (ordinary) | -1.220% (7/7) | not instrumented |
| short-final-increment-wall-confirm / forest48-e10 / short-bundle (ordinary) | -0.296% (5/7) | not instrumented |
| short-final-increment-wall-confirm / kodak01-e7 / short-bundle (ordinary) | -0.377% (4/7) | not instrumented |
| short-final-increment-wall-confirm / kodak17-e10-d08 / short-bundle (ordinary) | +0.057% (3/7) | not instrumented |
| control-wall-short-increment / alpine24-e7 / short-bundle (ordinary) | +0.284% (1/7) | not instrumented |
| control-wall-short-increment / forest48-e10 / short-bundle (ordinary) | +0.209% (3/7) | not instrumented |
| short-final-increment-broad-wall / alpine24-e7 / short-bundle (ordinary) | -0.905% (2/3) | not instrumented |
| short-final-increment-broad-wall / forest48-e10 / short-bundle (ordinary) | -0.260% (2/3) | not instrumented |
| short-final-increment-broad-wall / campus12-e7 / short-bundle (ordinary) | -0.245% (2/3) | not instrumented |
| short-final-increment-broad-wall / alpine12-e10-d08 / short-bundle (ordinary) | -0.920% (2/3) | not instrumented |
| short-final-increment-broad-wall / forest24-e7-d08 / short-bundle (ordinary) | -0.877% (3/3) | not instrumented |
| short-final-increment-broad-wall / campus48-e10-d08 / short-bundle (ordinary) | -1.347% (3/3) | not instrumented |
| short-final-increment-broad-wall / kodak01-e7 / short-bundle (ordinary) | -0.364% (2/3) | not instrumented |
| short-final-increment-broad-wall / kodak17-e10-d08 / short-bundle (ordinary) | +0.427% (1/3) | not instrumented |

Negative changes mean less time. The stage-mode complete-call number is instrumented and does not replace ordinary timing. Changes in untouched stages show meaningful run-to-run variability; preliminary whole-call ratios cannot be attributed entirely to the filter.

## Current interpretation

See [REPORT.md](REPORT.md), [PHASE10-11.md](PHASE10-11.md), and [HYPOTHESIS-AUDIT.md](HYPOTHESIS-AUDIT.md) for current interpretation and qualification limits. The tables above retain historical cohorts; their baseline and mode are explicit. Do not compare candidates by dividing medians from different cohorts, add their effects, or infer ordinary gains from profiled calls.

The earlier fusion bundle passes seven Metal/resident tests, 56 exact encoder comparisons and three identical finite decoded pairs. Seven-pair ordinary confirmation improves 24 MP/e7 by 7.659% and 48 MP/e10 by 6.883%, with 7/7 favorable pairs each. Independent broader ordinary coverage also improves every covered large-image setting. Fresh same-binary controls are -0.161% and +0.322%.

Four-output/Malta and geometry-only refinements do not show consistent ordinary increments. New shared-load short filters reveal further headroom: the combined short-bundle independently passes seven focused tests, 56 exact encoder comparisons and three identical finite decoded pairs. Its broad three-pair ordinary cohort improves all six large-image settings by 9.3-12.4% versus original baseline, with all 18 pairs favorable. Seven-pair confirmation improves 24 MP/e7 by 12.697% and 48 MP/e10 by 10.715%, with 7/7 favorable pairs each. High-only ablation confirms the joint medium-B/mask addition. All geometry/reuse integrations and controls are complete; see PHASE15-17.md. The independently validated short-final refinement saves another 1.220%/0.296% in seven-pair direct-parent confirmation, with smaller/mixed broad results and unchanged-binary controls of +0.284%/+0.209%. The study closes at the practical stopping point; possible future small or workload-specific gains are not ruled out.

Updated: 2026-09-22T02:27:29.640533+00:00
