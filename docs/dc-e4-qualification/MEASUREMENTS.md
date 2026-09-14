# e4 DC default qualification: measured results

Revision `8e8c49a42e69a24d469d4bcf0772a9e2c5012d6a`. Defaults unchanged.

Completion: 21/21 timing cases, 252/252 visual outputs, 17/28 matched-size targets.

## Complete-encode timing at unchanged distance 1.2

Three warmups; 20 balanced paired rounds; eight CPU participants; fully resident Metal. Input/output file I/O, validation, and final diagnostic scoring excluded. These are same-setting costs, not matched-quality timing.

| Photo / size | Effort | Default ms | Both ms | Paired time change | Size change | SSIMULACRA2 change |
|---|---:|---:|---:|---:|---:|---:|
| alpine_lake / 12mp | 3 | 101.86 | 110.33 | +7.78% | +2.50% | +0.0867 |
| campus_interior / 12mp | 3 | 111.02 | 118.89 | +6.16% | +1.92% | +0.3586 |
| forest_stream / 12mp | 3 | 123.30 | 134.22 | +7.78% | +1.48% | +0.0577 |
| alpine_lake / 12mp | 4 | 292.64 | 309.25 | +6.53% | +2.61% | +0.0223 |
| campus_interior / 12mp | 4 | 296.70 | 317.57 | +2.65% | +1.97% | +0.3366 |
| forest_stream / 12mp | 4 | 314.70 | 338.59 | +7.07% | +1.53% | +0.0382 |
| alpine_lake / 24mp | 4 | 593.40 | 600.55 | +2.13% | +2.84% | +0.1740 |
| campus_interior / 24mp | 4 | 617.05 | 639.33 | +3.44% | +2.11% | +0.2498 |
| forest_stream / 24mp | 4 | 642.79 | 659.00 | +2.33% | +1.83% | +0.1035 |
| alpine_lake / 48mp | 4 | 1396.48 | 1400.34 | +0.09% | +3.20% | +0.1226 |
| campus_interior / 48mp | 4 | 1374.39 | 1386.04 | +1.08% | +2.31% | +0.3043 |
| forest_stream / 48mp | 4 | 1475.26 | 1499.97 | +1.06% | +2.25% | +0.1090 |
| alpine_lake / 12mp | 7 | 476.04 | 503.96 | +7.19% | +2.56% | +0.2507 |
| campus_interior / 12mp | 7 | 464.08 | 489.98 | +4.35% | +2.07% | +0.3197 |
| forest_stream / 12mp | 7 | 497.01 | 532.39 | +7.46% | +1.49% | +0.0502 |
| alpine_lake / 24mp | 7 | 894.57 | 920.08 | +2.75% | +2.76% | +0.0887 |
| campus_interior / 24mp | 7 | 994.37 | 998.60 | +0.36% | +2.23% | +0.3015 |
| forest_stream / 24mp | 7 | 997.38 | 1024.80 | +1.76% | +1.87% | +0.1001 |
| alpine_lake / 48mp | 7 | 2066.33 | 2078.58 | +0.75% | +3.21% | +0.2684 |
| campus_interior / 48mp | 7 | 1961.98 | 1997.21 | +1.66% | +2.44% | +0.3385 |
| forest_stream / 48mp | 7 | 2140.75 | 2174.34 | +1.95% | +2.33% | +0.1353 |

Milliseconds are separate variant medians. Paired change is the median of the 20 within-round time ratios, so it need not equal the ratio of those two medians.

## Baseline versus baseline controls

| Invocation | Paired change | Minimum round | Maximum round |
|---|---:|---:|---:|
| 0 | -1.07% | -3.55% | +7.56% |
| 1 | +0.10% | -4.09% | +8.83% |
| 2 | +0.86% | -4.28% | +4.88% |

## Four-mode visual measurements

Each cell is actual bytes / SSIMULACRA2 at the same requested distance. Photographic derivatives are 2048 pixels wide; corrected synthetic fixtures are 2048×1024.

| Case | Default | Quantize | Smooth | Both |
|---|---:|---:|---:|---:|
| alpine_lake-visual-e3-d1.2 | 419,935 / 80.3581 | 429,207 / 80.5760 | 419,934 / 80.3186 | 429,206 / 80.4803 |
| alpine_lake-visual-e3-d3 | 188,283 / 58.3567 | 196,850 / 59.0710 | 188,282 / 58.2969 | 196,849 / 58.7643 |
| alpine_lake-visual-e3-d6 | 93,876 / 30.6505 | 101,362 / 32.5106 | 93,875 / 30.5796 | 101,361 / 31.6040 |
| alpine_lake-visual-e4-d1.2 | 410,427 / 79.9079 | 419,863 / 80.0794 | 410,374 / 79.8853 | 419,610 / 80.0895 |
| alpine_lake-visual-e4-d3 | 176,242 / 55.8579 | 184,848 / 56.6455 | 176,334 / 55.8008 | 184,856 / 56.2688 |
| alpine_lake-visual-e4-d6 | 85,473 / 25.4554 | 92,846 / 27.2431 | 85,532 / 25.3782 | 92,859 / 26.3440 |
| alpine_lake-visual-e7-d1.2 | 408,367 / 80.8952 | 418,137 / 81.2191 | 408,304 / 80.8616 | 417,538 / 80.9891 |
| alpine_lake-visual-e7-d3 | 159,302 / 58.8709 | 168,260 / 59.9971 | 159,973 / 59.0294 | 168,698 / 59.5917 |
| alpine_lake-visual-e7-d6 | 79,141 / 33.1061 | 86,579 / 35.9282 | 79,382 / 33.1971 | 86,860 / 34.6662 |
| campus_interior-visual-e3-d1.2 | 470,325 / 85.8669 | 479,226 / 86.2692 | 470,324 / 85.8309 | 479,225 / 86.2225 |
| campus_interior-visual-e3-d3 | 231,916 / 68.5693 | 236,753 / 70.8693 | 231,915 / 68.5420 | 236,752 / 70.5828 |
| campus_interior-visual-e3-d6 | 115,653 / 45.9464 | 122,396 / 48.5494 | 115,652 / 45.5594 | 122,395 / 47.7933 |
| campus_interior-visual-e4-d1.2 | 451,319 / 85.1187 | 460,121 / 85.5140 | 451,331 / 85.0755 | 460,173 / 85.5406 |
| campus_interior-visual-e4-d3 | 213,372 / 66.3635 | 217,012 / 68.3321 | 213,551 / 66.3698 | 217,259 / 68.1354 |
| campus_interior-visual-e4-d6 | 103,813 / 41.8589 | 110,109 / 44.1200 | 103,518 / 41.1272 | 110,325 / 43.4799 |
| campus_interior-visual-e7-d1.2 | 425,537 / 85.0787 | 433,852 / 85.5629 | 425,709 / 85.0822 | 433,920 / 85.5100 |
| campus_interior-visual-e7-d3 | 187,977 / 67.3822 | 192,615 / 69.9165 | 188,000 / 67.3657 | 192,286 / 69.4020 |
| campus_interior-visual-e7-d6 | 93,555 / 46.3523 | 99,753 / 49.3165 | 93,223 / 45.6364 | 100,077 / 48.5534 |
| forest_stream-visual-e3-d1.2 | 1,081,642 / 80.5581 | 1,094,554 / 80.6238 | 1,081,641 / 80.5531 | 1,094,553 / 80.6193 |
| forest_stream-visual-e3-d3 | 526,920 / 59.5830 | 538,008 / 59.7748 | 526,919 / 59.5765 | 538,007 / 59.7567 |
| forest_stream-visual-e3-d6 | 264,414 / 29.8058 | 275,087 / 30.4713 | 264,413 / 29.7212 | 275,086 / 30.3544 |
| forest_stream-visual-e4-d1.2 | 1,063,192 / 80.1731 | 1,076,153 / 80.2376 | 1,063,190 / 80.1732 | 1,076,166 / 80.2325 |
| forest_stream-visual-e4-d3 | 496,633 / 57.6353 | 507,631 / 57.8138 | 496,612 / 57.6175 | 507,626 / 57.7932 |
| forest_stream-visual-e4-d6 | 239,941 / 25.1019 | 250,629 / 25.7046 | 239,945 / 25.0251 | 250,643 / 25.5851 |
| forest_stream-visual-e7-d1.2 | 1,063,953 / 80.1861 | 1,076,745 / 80.2602 | 1,063,946 / 80.1777 | 1,076,737 / 80.2523 |
| forest_stream-visual-e7-d3 | 448,356 / 57.1297 | 459,348 / 57.3334 | 448,420 / 57.1134 | 459,446 / 57.3101 |
| forest_stream-visual-e7-d6 | 211,802 / 29.4852 | 222,408 / 30.1721 | 211,808 / 29.4059 | 222,317 / 29.9993 |
| gray-gradient-e3-d1.2 | 1,652 / 95.3142 | 3,039 / 96.6481 | 1,651 / 96.6454 | 3,038 / 97.0004 |
| gray-gradient-e3-d3 | 914 / 90.9866 | 1,555 / 95.0581 | 913 / 93.7832 | 1,554 / 96.5442 |
| gray-gradient-e3-d6 | 614 / 84.6640 | 971 / 91.7127 | 613 / 89.1780 | 970 / 94.3802 |
| gray-gradient-e4-d1.2 | 5,855 / 95.3184 | 3,904 / 96.6838 | 4,051 / 96.7411 | 4,140 / 97.0581 |
| gray-gradient-e4-d3 | 5,929 / 91.2069 | 5,299 / 95.0957 | 5,993 / 93.8057 | 3,531 / 96.5652 |
| gray-gradient-e4-d6 | 4,345 / 84.8289 | 6,039 / 91.8499 | 5,149 / 89.5030 | 5,318 / 94.4644 |
| gray-gradient-e7-d1.2 | 1,708 / 95.1044 | 3,176 / 96.5524 | 1,695 / 96.5802 | 3,223 / 97.0100 |
| gray-gradient-e7-d3 | 974 / 90.2464 | 1,561 / 94.6696 | 977 / 93.6697 | 1,598 / 96.4975 |
| gray-gradient-e7-d6 | 1,023 / 83.8905 | 1,058 / 91.3052 | 763 / 89.1902 | 1,053 / 94.2835 |
| dark-gradient-e3-d1.2 | 1,387 / 96.0978 | 2,552 / 98.1140 | 1,386 / 97.7879 | 2,551 / 98.9310 |
| dark-gradient-e3-d3 | 775 / 91.6660 | 1,288 / 95.6992 | 774 / 94.2957 | 1,287 / 97.4698 |
| dark-gradient-e3-d6 | 542 / 85.5637 | 817 / 92.3291 | 541 / 89.5916 | 816 / 94.7606 |
| dark-gradient-e4-d1.2 | 6,502 / 96.0393 | 6,377 / 98.1049 | 5,407 / 97.6284 | 4,746 / 98.9341 |
| dark-gradient-e4-d3 | 5,819 / 91.7599 | 6,247 / 95.6403 | 5,506 / 94.3100 | 5,654 / 97.4462 |
| dark-gradient-e4-d6 | 4,337 / 85.9463 | 5,920 / 92.5884 | 4,351 / 89.8766 | 5,738 / 94.9648 |
| dark-gradient-e7-d1.2 | 1,564 / 95.4618 | 2,623 / 97.7741 | 1,493 / 97.5274 | 2,604 / 98.9229 |
| dark-gradient-e7-d3 | 1,262 / 91.0697 | 1,536 / 95.1897 | 1,099 / 94.0171 | 1,518 / 97.3824 |
| dark-gradient-e7-d6 | 940 / 84.9297 | 1,227 / 91.8183 | 904 / 89.4721 | 1,195 / 94.7698 |
| sky-gradient-e3-d1.2 | 4,983 / 90.0615 | 6,032 / 92.0655 | 4,982 / 92.3288 | 6,031 / 92.8046 |
| sky-gradient-e3-d3 | 2,664 / 81.2657 | 3,345 / 89.7927 | 2,663 / 85.8513 | 3,344 / 91.7009 |
| sky-gradient-e3-d6 | 1,631 / 69.8568 | 2,234 / 84.9336 | 1,630 / 76.0979 | 2,233 / 88.5707 |
| sky-gradient-e4-d1.2 | 7,425 / 90.3285 | 7,971 / 92.0457 | 7,314 / 92.3423 | 8,849 / 92.7631 |
| sky-gradient-e4-d3 | 5,074 / 81.2273 | 5,339 / 90.0374 | 5,290 / 86.3110 | 5,403 / 91.6273 |
| sky-gradient-e4-d6 | 3,636 / 68.9199 | 4,390 / 84.5784 | 3,885 / 76.2340 | 4,363 / 88.4848 |
| sky-gradient-e7-d1.2 | 5,056 / 89.4863 | 6,251 / 91.8788 | 5,282 / 92.2829 | 6,287 / 92.8375 |
| sky-gradient-e7-d3 | 2,926 / 80.9925 | 3,498 / 89.3586 | 2,832 / 85.6868 | 3,533 / 91.6156 |
| sky-gradient-e7-d6 | 1,802 / 68.0818 | 2,327 / 84.1340 | 1,872 / 75.4301 | 2,403 / 88.4146 |
| chroma-gradient-e3-d1.2 | 1,026 / 92.0901 | 1,428 / 93.4757 | 1,025 / 92.7798 | 1,427 / 93.9468 |
| chroma-gradient-e3-d3 | 626 / 89.4183 | 848 / 92.5431 | 625 / 90.7467 | 847 / 93.2229 |
| chroma-gradient-e3-d6 | 488 / 87.4334 | 709 / 89.8152 | 487 / 89.0069 | 708 / 90.9723 |
| chroma-gradient-e4-d1.2 | 3,988 / 91.8430 | 4,616 / 93.4359 | 3,958 / 92.6274 | 4,517 / 93.8674 |
| chroma-gradient-e4-d3 | 2,611 / 89.4646 | 2,691 / 92.5192 | 2,621 / 89.4564 | 2,723 / 92.6897 |
| chroma-gradient-e4-d6 | 2,582 / 85.8572 | 2,475 / 90.2073 | 2,560 / 90.0936 | 2,548 / 91.2998 |
| chroma-gradient-e7-d1.2 | 1,500 / 91.6042 | 1,949 / 93.5553 | 1,475 / 92.4387 | 1,995 / 93.8365 |
| chroma-gradient-e7-d3 | 963 / 89.2004 | 1,258 / 91.8768 | 977 / 90.8731 | 1,220 / 93.1815 |
| chroma-gradient-e7-d6 | 729 / 87.2021 | 1,019 / 89.0371 | 851 / 86.5032 | 948 / 91.0575 |

## Matched-size visual comparisons

All attempts, including unresolved targets, appear below. Passing means within 1% of default bytes; it does not mean visual acceptance.

| Case | Size error | SSIMULACRA2 change | Probes | Status |
|---|---:|---:|---:|---|
| alpine_lake-visual-e4-d3 | 0.300% | -1.4180 | 3 | within-tolerance |
| alpine_lake-visual-e4-d6 | 0.541% | -3.5646 | 3 | within-tolerance |
| alpine_lake-visual-e7-d3 | 0.754% | -1.0077 | 3 | within-tolerance |
| alpine_lake-visual-e7-d6 | 0.442% | -3.1167 | 3 | within-tolerance |
| campus_interior-visual-e4-d3 | 0.000% | +1.1859 | 4 | within-tolerance |
| campus_interior-visual-e4-d6 | 0.345% | -0.9546 | 3 | within-tolerance |
| campus_interior-visual-e7-d3 | 0.069% | +1.6640 | 4 | within-tolerance |
| campus_interior-visual-e7-d6 | 0.171% | -0.1014 | 3 | within-tolerance |
| forest_stream-visual-e4-d3 | 0.097% | -0.6828 | 4 | within-tolerance |
| forest_stream-visual-e4-d6 | 0.409% | -2.0013 | 3 | within-tolerance |
| forest_stream-visual-e7-d3 | 0.012% | -0.6048 | 4 | within-tolerance |
| forest_stream-visual-e7-d6 | 0.513% | -1.9949 | 3 | within-tolerance |
| gray-gradient-e4-d3 | 21.859% | +5.8247 | 10 | probe-budget-exhausted |
| gray-gradient-e4-d6 | 22.371% | +6.6949 | 4 | distance-bound |
| gray-gradient-e7-d3 | 2.464% | +1.9787 | 10 | probe-budget-exhausted |
| gray-gradient-e7-d6 | 0.782% | +10.2271 | 8 | within-tolerance |
| dark-gradient-e4-d3 | 2.836% | +5.6863 | 10 | probe-budget-exhausted |
| dark-gradient-e4-d6 | 22.596% | +6.0601 | 4 | distance-bound |
| dark-gradient-e7-d3 | 1.506% | +4.6356 | 10 | probe-budget-exhausted |
| dark-gradient-e7-d6 | 9.149% | +6.8226 | 4 | distance-bound |
| sky-gradient-e4-d3 | 0.236% | +10.0527 | 2 | within-tolerance |
| sky-gradient-e4-d6 | 1.843% | +15.6252 | 4 | distance-bound |
| sky-gradient-e7-d3 | 0.273% | +9.2910 | 6 | within-tolerance |
| sky-gradient-e7-d6 | 0.888% | +14.0340 | 4 | within-tolerance |
| chroma-gradient-e4-d3 | 0.919% | +3.3176 | 2 | within-tolerance |
| chroma-gradient-e4-d6 | 1.317% | +5.4425 | 10 | probe-budget-exhausted |
| chroma-gradient-e7-d3 | 3.115% | +2.1471 | 10 | probe-budget-exhausted |
| chroma-gradient-e7-d6 | 3.018% | +4.3125 | 10 | probe-budget-exhausted |

## Proposed e3/e4 transition on photographs

Same requested distance; bytes can differ. The current e4 baseline separates existing effort behavior from the incremental DC change.

| Image | Distance | e3 default score | e4 default score | e4 both score | DC increment at e4 |
|---|---:|---:|---:|---:|---:|
| alpine_lake-visual | 1.2 | 80.3581 | 79.9079 | 80.0895 | +0.1816 |
| alpine_lake-visual | 3 | 58.3567 | 55.8579 | 56.2688 | +0.4109 |
| alpine_lake-visual | 6 | 30.6505 | 25.4554 | 26.3440 | +0.8886 |
| campus_interior-visual | 1.2 | 85.8669 | 85.1187 | 85.5406 | +0.4220 |
| campus_interior-visual | 3 | 68.5693 | 66.3635 | 68.1354 | +1.7719 |
| campus_interior-visual | 6 | 45.9464 | 41.8589 | 43.4799 | +1.6211 |
| forest_stream-visual | 1.2 | 80.5581 | 80.1731 | 80.2325 | +0.0594 |
| forest_stream-visual | 3 | 59.5830 | 57.6353 | 57.7932 | +0.1579 |
| forest_stream-visual | 6 | 29.8058 | 25.1019 | 25.5851 | +0.4832 |

## Boundaries

Three natural scenes are reused across sizes. High-resolution inputs retain the earlier Lanczos overshoot. Visual photographic derivatives are separately resized and bounded to [0,1]. Four analytic gradient diagnostics are synthetic and must not be aggregated as photographic compression results.

One process supplies each timing case; rounds are paired repetitions, not independent sessions. Background CPU snapshots and control spread remain part of interpretation. Diagnostic block-error roughness is not a validated perceptual metric. Agent visual inspection and browser previews do not establish blinded human preference on a calibrated display.

Raw artifacts and the 16-bit visual gallery: `/Users/yunhocho/GitHub/gjxl/build/dc-e4-qualification-20260913-v1`. The explicit `review` command decodes retained streams; `report` reads saved measurements only.

All original synthetic measurements are excluded. Corrected synthetic data come from `/Users/yunhocho/GitHub/gjxl/build/dc-e4-synthetic-corrected-20260914-v1`; see its independent fixture audit. The combined gallery is [review.html](review.html).
