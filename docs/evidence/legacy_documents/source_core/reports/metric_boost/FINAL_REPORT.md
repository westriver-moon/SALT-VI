# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/metric_boost/FINAL_REPORT.md`  
> Original SHA-256: `3622ec220d0908be4eee8de9e4261ae7c9916a26d354b4ff59a36f0ff442d5b5`  
> This is read-only experiment evidence, not an active runtime instruction.

# FINAL REPORT — SYSU-MM01 Metric Boost

## Execution status

- E4 reproduced in this run: **yes**.
- Experiments actually succeeded: **132**.
- Experiments configured but not run/terminal: **2**.
- Experiments failed: **0**.
- Reference only (not a new run): Rank-1 `0.81620`, mAP `0.78663`, mINP `0.67711`.
- No test identity label is authorized for training, tuning, re-ranking selection, or model selection.

## Unified metrics

| Experiment | Training | MER | TTA | Re-ranking | Ensemble | Rank-1 | mAP | mINP | ΔR1 | ΔmAP | ΔmINP | Validity | Status | Reproducibility |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| EVAL-0 | no | no | no | no | none | 0.81620 | 0.78663 | 0.67711 | -0.000 | +0.000 | +0.000 | standard baseline reproduction | succeeded | legacy-insufficient-evidence |
| EVAL-1 | no | no | no | no | none | 0.81152 | 0.78192 | 0.67058 | -0.468 | -0.471 | -0.653 | standard legacy equal-weight MER | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p00_text0p00 | no | no | no | no | none | 0.81620 | 0.78663 | 0.67711 | -0.000 | +0.000 | +0.000 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p00_text0p25 | no | no | no | no | none | 0.81175 | 0.78529 | 0.67716 | -0.445 | -0.134 | +0.005 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p00_text0p50 | no | no | no | no | none | 0.80226 | 0.77818 | 0.67009 | -1.394 | -0.845 | -0.702 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p00_text0p75 | no | no | no | no | none | 0.78798 | 0.76739 | 0.65909 | -2.822 | -1.924 | -1.802 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p00_text1p00 | no | no | no | no | none | 0.77357 | 0.75565 | 0.64666 | -4.263 | -3.098 | -3.045 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p25_text0p00 | no | no | no | no | none | 0.80539 | 0.77547 | 0.66240 | -1.081 | -1.116 | -1.471 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p25_text0p25 | no | no | no | no | none | 0.81507 | 0.78508 | 0.67469 | -0.113 | -0.155 | -0.242 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p25_text0p50 | no | no | no | no | none | 0.81536 | 0.78731 | 0.67858 | -0.084 | +0.068 | +0.147 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p25_text0p75 | no | no | no | no | none | 0.81152 | 0.78511 | 0.67697 | -0.468 | -0.152 | -0.014 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p25_text1p00 | no | no | no | no | none | 0.80458 | 0.78025 | 0.67226 | -1.162 | -0.638 | -0.485 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p50_text0p00 | no | no | no | no | none | 0.79216 | 0.76155 | 0.64506 | -2.404 | -2.508 | -3.205 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p50_text0p25 | no | no | no | no | none | 0.80563 | 0.77573 | 0.66264 | -1.057 | -1.090 | -1.447 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p50_text0p50 | no | no | no | no | none | 0.81281 | 0.78348 | 0.67279 | -0.339 | -0.315 | -0.432 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p50_text0p75 | no | no | no | no | none | 0.81615 | 0.78713 | 0.67784 | -0.005 | +0.050 | +0.073 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p50_text1p00 | no | no | no | no | none | 0.81441 | 0.78708 | 0.67847 | -0.179 | +0.045 | +0.136 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p75_text0p00 | no | no | no | no | none | 0.77946 | 0.74902 | 0.62981 | -3.674 | -3.761 | -4.730 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p75_text0p25 | no | no | no | no | none | 0.79587 | 0.76511 | 0.64925 | -2.033 | -2.152 | -2.786 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p75_text0p50 | no | no | no | no | none | 0.80584 | 0.77592 | 0.66288 | -1.036 | -1.071 | -1.423 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p75_text0p75 | no | no | no | no | none | 0.81178 | 0.78264 | 0.67168 | -0.442 | -0.399 | -0.543 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir0p75_text1p00 | no | no | no | no | none | 0.81599 | 0.78642 | 0.67651 | -0.021 | -0.021 | -0.060 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir1p00_text0p00 | no | no | no | no | none | 0.76797 | 0.73860 | 0.61755 | -4.823 | -4.803 | -5.956 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir1p00_text0p25 | no | no | no | no | none | 0.78612 | 0.75510 | 0.63682 | -3.008 | -3.153 | -4.029 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir1p00_text0p50 | no | no | no | no | none | 0.79805 | 0.76739 | 0.65205 | -1.815 | -1.924 | -2.506 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir1p00_text0p75 | no | no | no | no | none | 0.80594 | 0.77601 | 0.66298 | -1.026 | -1.062 | -1.413 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-2-ir1p00_text1p00 | no | no | no | no | none | 0.81152 | 0.78192 | 0.67058 | -0.468 | -0.471 | -0.653 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-3-fusion-flip | no | no | no | no | none | 0.82133 | 0.79229 | 0.68464 | +0.513 | +0.566 | +0.753 | standard | succeeded | legacy-insufficient-evidence |
| EVAL-3-mer-flip | no | no | no | no | none | 0.81620 | 0.78663 | 0.67711 | -0.000 | +0.000 | +0.000 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-4-fusion-288 | no | no | no | no | none | 0.81620 | 0.78663 | 0.67711 | -0.000 | +0.000 | +0.000 | standard | succeeded | legacy-insufficient-evidence |
| EVAL-4-mer-288 | no | no | no | no | none | 0.81620 | 0.78663 | 0.67711 | -0.000 | +0.000 | +0.000 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-4-fusion-256_288 | no | no | no | no | none | 0.82093 | 0.79425 | 0.68828 | +0.473 | +0.762 | +1.117 | standard | succeeded | legacy-insufficient-evidence |
| EVAL-4-mer-256_288 | no | no | no | no | none | 0.81620 | 0.78663 | 0.67711 | -0.000 | +0.000 | +0.000 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-4-fusion-288_320 | no | no | no | no | none | 0.82348 | 0.79725 | 0.69364 | +0.728 | +1.062 | +1.653 | standard | succeeded | legacy-insufficient-evidence |
| EVAL-4-mer-288_320 | no | no | no | no | none | 0.81620 | 0.78663 | 0.67711 | -0.000 | +0.000 | +0.000 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-4-fusion-256_288_320 | no | no | no | no | none | 0.82351 | 0.79931 | 0.69689 | +0.731 | +1.268 | +1.978 | standard | succeeded | legacy-insufficient-evidence |
| EVAL-4-mer-256_288_320 | no | no | no | no | none | 0.81620 | 0.78663 | 0.67711 | -0.000 | +0.000 | +0.000 | exploratory test-set-tuned | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k10-k3-l0p2 | no | no | no | no | none | 0.82422 | 0.79363 | 0.68471 | +0.802 | +0.700 | +0.760 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k10-k3-l0p3 | no | no | no | no | none | 0.82535 | 0.79399 | 0.68484 | +0.915 | +0.736 | +0.773 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k10-k3-l0p5 | no | no | no | no | none | 0.82640 | 0.79374 | 0.68414 | +1.020 | +0.711 | +0.703 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k10-k6-l0p2 | no | no | no | no | none | 0.87044 | 0.83773 | 0.74077 | +5.424 | +5.110 | +6.366 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k10-k6-l0p3 | no | no | no | no | none | 0.87123 | 0.83781 | 0.74037 | +5.503 | +5.118 | +6.326 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k10-k6-l0p5 | no | no | no | no | none | 0.86989 | 0.83440 | 0.73455 | +5.369 | +4.777 | +5.744 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k20-k3-l0p2 | no | no | no | no | none | 0.82243 | 0.79889 | 0.69690 | +0.623 | +1.226 | +1.979 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k20-k3-l0p3 | no | no | no | no | none | 0.82661 | 0.80006 | 0.69623 | +1.041 | +1.343 | +1.912 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k20-k3-l0p5 | no | no | no | no | none | 0.83121 | 0.80013 | 0.69367 | +1.501 | +1.350 | +1.656 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k20-k6-l0p2 | no | no | no | no | none | 0.87605 | 0.84515 | 0.75222 | +5.985 | +5.852 | +7.511 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k20-k6-l0p3 | no | no | no | no | none | 0.87870 | 0.84715 | 0.75411 | +6.250 | +6.052 | +7.700 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k20-k6-l0p5 | no | no | no | no | none | 0.87954 | 0.84669 | 0.75255 | +6.334 | +6.006 | +7.544 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k30-k3-l0p2 | no | no | no | no | none | 0.82472 | 0.81403 | 0.72867 | +0.852 | +2.740 | +5.156 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k30-k3-l0p3 | no | no | no | no | none | 0.83195 | 0.81596 | 0.72685 | +1.575 | +2.933 | +4.974 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k30-k3-l0p5 | no | no | no | no | none | 0.83873 | 0.81446 | 0.71898 | +2.253 | +2.783 | +4.187 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k30-k6-l0p2 | no | no | no | no | none | 0.87560 | 0.84743 | 0.76061 | +5.940 | +6.080 | +8.350 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k30-k6-l0p3 | no | no | no | no | none | 0.87973 | 0.85183 | 0.76511 | +6.353 | +6.520 | +8.800 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-fusion-k30-k6-l0p5 | no | no | no | no | none | 0.88094 | 0.85212 | 0.76398 | +6.474 | +6.549 | +8.687 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k10-k3-l0p2 | no | no | no | no | none | 0.82422 | 0.79363 | 0.68471 | +0.802 | +0.700 | +0.760 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k10-k3-l0p3 | no | no | no | no | none | 0.82535 | 0.79399 | 0.68484 | +0.915 | +0.736 | +0.773 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k10-k3-l0p5 | no | no | no | no | none | 0.82640 | 0.79374 | 0.68414 | +1.020 | +0.711 | +0.703 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k10-k6-l0p2 | no | no | no | no | none | 0.87044 | 0.83773 | 0.74077 | +5.424 | +5.110 | +6.366 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k10-k6-l0p3 | no | no | no | no | none | 0.87123 | 0.83781 | 0.74037 | +5.503 | +5.118 | +6.326 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k10-k6-l0p5 | no | no | no | no | none | 0.86989 | 0.83440 | 0.73455 | +5.369 | +4.777 | +5.744 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k20-k3-l0p2 | no | no | no | no | none | 0.82243 | 0.79889 | 0.69690 | +0.623 | +1.226 | +1.979 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k20-k3-l0p3 | no | no | no | no | none | 0.82661 | 0.80006 | 0.69623 | +1.041 | +1.343 | +1.912 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k20-k3-l0p5 | no | no | no | no | none | 0.83121 | 0.80013 | 0.69367 | +1.501 | +1.350 | +1.656 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k20-k6-l0p2 | no | no | no | no | none | 0.87605 | 0.84515 | 0.75222 | +5.985 | +5.852 | +7.511 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k20-k6-l0p3 | no | no | no | no | none | 0.87870 | 0.84715 | 0.75411 | +6.250 | +6.052 | +7.700 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k20-k6-l0p5 | no | no | no | no | none | 0.87954 | 0.84669 | 0.75255 | +6.334 | +6.006 | +7.544 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k30-k3-l0p2 | no | no | no | no | none | 0.82472 | 0.81403 | 0.72867 | +0.852 | +2.740 | +5.156 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k30-k3-l0p3 | no | no | no | no | none | 0.83195 | 0.81596 | 0.72685 | +1.575 | +2.933 | +4.974 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k30-k3-l0p5 | no | no | no | no | none | 0.83873 | 0.81446 | 0.71898 | +2.253 | +2.783 | +4.187 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k30-k6-l0p2 | no | no | no | no | none | 0.87560 | 0.84743 | 0.76061 | +5.940 | +6.080 | +8.350 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k30-k6-l0p3 | no | no | no | no | none | 0.87973 | 0.85183 | 0.76511 | +6.353 | +6.520 | +8.800 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer-k30-k6-l0p5 | no | no | no | no | none | 0.88094 | 0.85212 | 0.76398 | +6.474 | +6.549 | +8.687 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k10-k3-l0p2 | no | no | no | no | none | 0.82485 | 0.79740 | 0.69185 | +0.865 | +1.077 | +1.474 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k10-k3-l0p3 | no | no | no | no | none | 0.82537 | 0.79760 | 0.69198 | +0.917 | +1.097 | +1.487 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k10-k3-l0p5 | no | no | no | no | none | 0.82493 | 0.79744 | 0.69199 | +0.873 | +1.081 | +1.488 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k10-k6-l0p2 | no | no | no | no | none | 0.85396 | 0.81730 | 0.71307 | +3.776 | +3.067 | +3.596 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k10-k6-l0p3 | no | no | no | no | none | 0.85306 | 0.81695 | 0.71274 | +3.686 | +3.032 | +3.563 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k10-k6-l0p5 | no | no | no | no | none | 0.84954 | 0.81499 | 0.71086 | +3.334 | +2.836 | +3.375 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k20-k3-l0p2 | no | no | no | no | none | 0.82348 | 0.79753 | 0.69324 | +0.728 | +1.090 | +1.613 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k20-k3-l0p3 | no | no | no | no | none | 0.82540 | 0.79845 | 0.69373 | +0.920 | +1.182 | +1.662 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k20-k3-l0p5 | no | no | no | no | none | 0.82679 | 0.79884 | 0.69371 | +1.059 | +1.221 | +1.660 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k20-k6-l0p2 | no | no | no | no | none | 0.85335 | 0.81725 | 0.71427 | +3.715 | +3.062 | +3.716 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k20-k6-l0p3 | no | no | no | no | none | 0.85485 | 0.81793 | 0.71468 | +3.865 | +3.130 | +3.757 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k20-k6-l0p5 | no | no | no | no | none | 0.85569 | 0.81837 | 0.71435 | +3.949 | +3.174 | +3.724 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k30-k3-l0p2 | no | no | no | no | none | 0.82293 | 0.80172 | 0.70313 | +0.673 | +1.509 | +2.602 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k30-k3-l0p3 | no | no | no | no | none | 0.82706 | 0.80285 | 0.70229 | +1.086 | +1.622 | +2.518 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k30-k3-l0p5 | no | no | no | no | none | 0.82916 | 0.80227 | 0.69940 | +1.296 | +1.564 | +2.229 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k30-k6-l0p2 | no | no | no | no | none | 0.84757 | 0.81461 | 0.71393 | +3.137 | +2.798 | +3.682 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k30-k6-l0p3 | no | no | no | no | none | 0.85201 | 0.81750 | 0.71634 | +3.581 | +3.087 | +3.923 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-5-mer_tta-k30-k6-l0p5 | no | no | no | no | none | 0.85567 | 0.81896 | 0.71618 | +3.947 | +3.233 | +3.907 | exploratory test-set-tuned post-processing | succeeded | legacy-insufficient-evidence |
| EVAL-6-feature-ensemble | no | no | no | no | none | — | — | — | — | — | — | standard | blocked | legacy-insufficient-evidence |
| EVAL-6-score-ensemble | no | no | no | no | none | — | — | — | — | — | — | standard | blocked | legacy-insufficient-evidence |
| TRAIN-1-U1 | yes | no | no | no | none | 0.81835 | 0.78748 | 0.67605 | +0.215 | +0.085 | -0.106 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-1-U2 | yes | no | no | no | none | 0.81835 | 0.78748 | 0.67605 | +0.215 | +0.085 | -0.106 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-1-U3 | yes | no | no | no | none | 0.81835 | 0.78748 | 0.67605 | +0.215 | +0.085 | -0.106 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-1-U4 | yes | no | no | no | none | 0.81835 | 0.78748 | 0.67605 | +0.215 | +0.085 | -0.106 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-2-pa-0p3 | yes | no | no | no | none | 0.77691 | 0.75736 | 0.64714 | -3.929 | -2.927 | -2.997 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-2-pa-0p4 | yes | no | no | no | none | 0.81060 | 0.78319 | 0.67320 | -0.560 | -0.344 | -0.391 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-2-pa-0p5 | yes | no | no | no | none | 0.81835 | 0.78748 | 0.67605 | +0.215 | +0.085 | -0.106 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-2-pa-0p6 | yes | no | no | no | none | 0.80613 | 0.77577 | 0.66253 | -1.007 | -1.086 | -1.458 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-2-pa-0p7 | yes | no | no | no | none | 0.78330 | 0.75342 | 0.63464 | -3.290 | -3.321 | -4.247 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-2-learnable-pa | yes | no | no | no | none | 0.81804 | 0.78630 | 0.67442 | +0.184 | -0.033 | -0.269 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-3-H0 | yes | no | no | no | none | 0.81835 | 0.78748 | 0.67605 | +0.215 | +0.085 | -0.106 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-3-H1 | yes | no | no | no | none | 0.82240 | 0.79358 | 0.68584 | +0.620 | +0.695 | +0.873 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-3-H2 | yes | no | no | no | none | 0.81901 | 0.78911 | 0.67891 | +0.281 | +0.248 | +0.180 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-3-H3 | yes | no | no | no | none | 0.81975 | 0.79047 | 0.68120 | +0.355 | +0.384 | +0.409 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-4-seed-0 | yes | no | no | no | none | 0.82240 | 0.79358 | 0.68584 | +0.620 | +0.695 | +0.873 | exploratory replicate if selected without independent validation | succeeded | legacy-insufficient-evidence |
| TRAIN-4-seed-1 | yes | no | no | no | none | 0.81975 | 0.79223 | 0.68416 | +0.355 | +0.560 | +0.705 | exploratory replicate if selected without independent validation | succeeded | legacy-insufficient-evidence |
| TRAIN-4-seed-42 | yes | no | no | no | none | 0.82067 | 0.79405 | 0.68808 | +0.447 | +0.742 | +1.097 | exploratory replicate if selected without independent validation | succeeded | legacy-insufficient-evidence |
| TRAIN-4-feature-ensemble | yes | no | no | no | none | 0.83255 | 0.80405 | 0.69966 | +1.635 | +1.742 | +2.255 | exploratory seed ensemble if selected without independent validation | succeeded | legacy-insufficient-evidence |
| TRAIN-4-score-ensemble | yes | no | no | no | none | 0.83153 | 0.80347 | 0.69919 | +1.533 | +1.684 | +2.208 | exploratory seed ensemble if selected without independent validation | succeeded | legacy-insufficient-evidence |
| TRAIN-5-llm-0p25 | yes | no | no | no | none | 0.81912 | 0.78857 | 0.67764 | +0.292 | +0.194 | +0.053 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-5-llm-0p5 | yes | no | no | no | none | 0.82109 | 0.79128 | 0.68119 | +0.489 | +0.465 | +0.408 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-6-eps0 | yes | no | no | no | none | 0.81830 | 0.78746 | 0.67607 | +0.210 | +0.083 | -0.104 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-6-eps005 | yes | no | no | no | none | 0.80150 | 0.77366 | 0.65965 | -1.470 | -1.297 | -1.746 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-6-eps01 | yes | no | no | no | none | 0.80026 | 0.77016 | 0.65504 | -1.594 | -1.647 | -2.207 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-6-id05 | yes | no | no | no | none | 0.81675 | 0.78648 | 0.67523 | +0.055 | -0.015 | -0.188 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-6-id15 | yes | no | no | no | none | 0.81893 | 0.78952 | 0.67968 | +0.273 | +0.289 | +0.257 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-6-wrt05 | yes | no | no | no | none | 0.81801 | 0.78713 | 0.67591 | +0.181 | +0.050 | -0.120 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-7-320x160 | yes | no | no | no | none | 0.81357 | 0.78609 | 0.67638 | -0.263 | -0.054 | -0.073 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-7-384x192 | yes | no | no | no | none | 0.80715 | 0.77930 | 0.66689 | -0.905 | -0.733 | -1.022 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-8-cls | yes | no | no | no | none | 0.81830 | 0.78746 | 0.67607 | +0.210 | +0.083 | -0.104 | training experiment; manually stopped by user; partial run, best observed checkpoint | stopped_by_user | legacy-insufficient-evidence |
| TRAIN-8-mean | yes | no | no | no | none | 0.78735 | 0.74488 | 0.61371 | -2.885 | -4.175 | -6.340 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-8-gem | yes | no | no | no | none | 0.76035 | 0.71652 | 0.57602 | -5.585 | -7.011 | -10.109 | training experiment | succeeded | legacy-insufficient-evidence |
| TRAIN-8-cls-gem | yes | no | no | no | none | 0.81796 | 0.78649 | 0.67459 | +0.176 | -0.014 | -0.252 | training experiment | succeeded | legacy-insufficient-evidence |
| PAIRWISE-1-hard-llm05 | yes | no | no | no | none | 0.82214 | 0.79593 | 0.69252 | +0.594 | +0.930 | +1.541 | training experiment; pairwise interaction test | succeeded | retrospective-reconstructed-reachable-code-and-checkpoint |
| PAIRWISE-1-hard-id15 | yes | no | no | no | none | 0.82119 | 0.79190 | 0.68348 | +0.499 | +0.527 | +0.637 | training experiment; pairwise interaction test | succeeded | retrospective-reconstructed-reachable-code-and-checkpoint |
| PAIRWISE-1-llm05-id15 | yes | no | no | no | none | 0.82196 | 0.79161 | 0.68142 | +0.576 | +0.498 | +0.431 | training experiment; pairwise interaction test | succeeded | retrospective-reconstructed-reachable-code-and-checkpoint |
| IMTA-M1-prototype-retry1 | yes | no | no | no | none | 0.81052 | 0.78255 | 0.67139 | -0.568 | -0.408 | -0.572 | training experiment; identity-manifold text alignment | succeeded | retrospective-reconstructed-reachable-code-and-checkpoint |
| IMTA-M2-relation | yes | no | no | no | none | 0.81217 | 0.78396 | 0.67332 | -0.403 | -0.267 | -0.379 | training experiment; identity-manifold text alignment | succeeded | retrospective-reconstructed-reachable-code-and-checkpoint |
| IMTA-M2-relation-light | yes | no | no | no | none | 0.81404 | 0.78491 | 0.67401 | -0.216 | -0.172 | -0.310 | training experiment; identity-manifold text alignment | succeeded | retrospective-reconstructed-reachable-code-and-checkpoint |
| FGAP-P1-asym-hard | yes | no | no | no | none | 0.82303 | 0.79248 | 0.68408 | +0.683 | +0.585 | +0.697 | exploratory-test-set-tuned | succeeded | prelaunch-verified-exact-commit |
| FGAP-P2-asym-hard-u2 | yes | no | no | no | none | 0.82051 | 0.79209 | 0.68508 | +0.431 | +0.546 | +0.797 | exploratory-test-set-tuned | succeeded | prelaunch-verified-exact-commit |
| FGAP-P3-asym-hard-qbn | yes | no | no | no | none | 0.82377 | 0.79345 | 0.68492 | +0.757 | +0.682 | +0.781 | exploratory-test-set-tuned | succeeded | prelaunch-verified-exact-commit |

## Required best-result categories

- A. Best standard single model, no TTA/re-ranking: `TRAIN-4-feature-ensemble` — Rank-1 `0.83255`, mAP `0.80405`, mINP `0.69966`.
- B. Best standard ensemble, optional TTA, no re-ranking: not available (no qualifying experiment has run).
- C. Best metric-only result with optional MER/TTA/ensemble/re-ranking: `EVAL-5-fusion-k30-k6-l0p5` — Rank-1 `0.88094`, mAP `0.85212`, mINP `0.76398`.

## Training and technique audit

- Retraining: no prepared TRAIN experiment is counted as run until its status is `succeeded`.
- TTA, MER, re-ranking, and ensemble are separate columns and are never merged into an unlabeled result.
- Weighted MER and re-ranking searches are explicitly `exploratory test-set-tuned` without an independent validation set.
- Checkpoint ensemble uses only existing, loadable checkpoints and is capped at five.
- Best retainable training configuration: not available until TRAIN experiments run.
- TRAIN-4 seed statistics: `{"count": 3, "experiments": ["TRAIN-4-seed-0", "TRAIN-4-seed-1", "TRAIN-4-seed-42"], "Rank-1": {"mean": 0.8209395805994669, "std": 0.0011010957168609559, "best": 0.8224033117294312}, "mAP": {"mean": 0.7932853182031114, "std": 0.0007713715525610164, "best": 0.794050726528253}, "mINP": {"mean": 0.6860273409677976, "std": 0.0016062736651945378, "best": 0.688080808198223}}`.

## Failed directions

- None can be scientifically declared failed before the corresponding experiment runs.

## Next experiments (maximum three)

1. EVAL-0 — reproduce E4 under the official 10-trial all-search single-shot protocol.
2. EVAL-1 — measure legacy equal-weight MER only if EVAL-0 passes.
3. EVAL-2 — run the bounded 25-point weighted MER grid and label it exploratory.

Every path, command, checkpoint, log, Git SHA, time, GPU, and status is retained in `summary.csv` and `summary.json`.
