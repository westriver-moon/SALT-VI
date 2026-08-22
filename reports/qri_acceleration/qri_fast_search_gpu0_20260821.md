# QRI imagination super-resolution acceleration study (GPU0)

Run: `qri-fast-search-gpu0-20260821`
Host GPU: NVIDIA RTX 3090 24 GiB
Repository revision at pilot start: `27ff03a1`

## Decision

Use the following conservative two-stage flow for full-data preprocessing:

1. Batch-16 official SwinIR x2.
2. One pose/parsing pass and one cached SAM image embedding per image.
3. Rank regions with `u_blur` only and retain the top two masks.
4. Run one no-thinking, 1024-token Qwen plan per identity-modality group and
   cache it for the group's images.
5. Use SwinIR as the default image output. Keep PASD disabled by default.

An optional PASD 256/8 branch may be used only with automatic fallback and the
locked absolute quality gates. No further PASD quality search is recommended;
replace the diffusion backend if better imaginative realism is required.

## Evidence

### SwinIR and ROI

- Batch-16 SwinIR: RGB 0.1500 s/image, IR 0.1553 s/image, peak allocation
  about 5.2 GiB.
- Batch-32 RGB was only marginally faster (0.1477 s/image) and IR showed a
  56.7 s outlier, so batch 16 is the stable choice.
- On the 48-image camera-balanced validation, cached ROI took 0.452 s/image
  for RGB and 0.537 s/image for IR.
- SAM image embedding calls were exactly one for all 48 images.
- Pose fallback occurred on 4/48 images, all IR; the full-frame fallback kept
  every source processable.

### Removed 12-TTA

- Two measured images paid 3.16 and 3.75 seconds for the extra 12 SwinIR
  variants.
- All 25 measured regional `u_swin` values were exactly zero.
- This stage is removed because it consumes roughly the entire per-image
  budget without changing the ranking signal.

### Qwen grouping

- A no-thinking one-shot response took 38.76, 86.68, and 91.00 seconds;
  mean 72.15 seconds.
- Every sample preserved positive-candidate and unresolved coverage for both
  selected regions.
- Train/validation contains 34,167 images, 395 identities, and 790
  identity-modality groups. Group caching therefore cuts Qwen calls by about
  43.2x compared with per-image planning.

### Bounded PASD result

| Variant | Mean time | Inside change | Outside leakage | LR-cycle energy |
|---|---:|---:|---:|---:|
| 512 / 20 | 7.893 s | 6.287 | 0.0263 | 0.000998 |
| 256 / 12 | 2.397 s | 11.968 | 0.0523 | 0.001854 |
| 256 / 8 | 1.630 s | 9.408 | 0.0438 | 0.001511 |

For 256/8, PMT-ViT identity cosine to SwinIR was 0.9682-0.9933 (mean
0.9826). Absolute LR-cycle error was 0.00090-0.00229, or about 0.23-0.58 of
one 8-bit intensity level. These numbers pass the basic identity/locality
checks, but the visual sample is too small to enable PASD by default.

Locked optional gates:

- inside mean absolute change >= 3.0;
- outside mean absolute change <= 0.1 on the 0-255 scale;
- LR-cycle energy <= 0.003 on the 0-1 scale;
- identity cosine to SwinIR >= 0.96;
- otherwise emit the unchanged SwinIR result.

## Full-data projection

For SYSU train/validation (34,167 images):

- selected Phase A: about 6.0 GPU-hours;
- 790 group-level Qwen calls: about 15.8 GPU-hours;
- conservative flow total before file-I/O allowance: about 21.8 GPU-hours;
- optional PASD 256/8 on every image adds about 15.5 GPU-hours, yielding
  about 37.3 GPU-hours before identity-gate and I/O overhead.

For all 44,745 formal sources, the conservative projection is about 27.6
GPU-hours using an upper bound of 982 identity-modality groups. Running PASD
on every formal source would bring the compute-only projection to about 47.8
hours and leaves no safe I/O/QA margin, so it is not recommended.

## Artifact locations

- Code: `scripts/experiments/qri_acceleration/`
- Locked configuration: `configs/experiments/qri_acceleration/`
- This report: `reports/qri_acceleration/`
- Logs, metrics, masks, and images:
  `/home/lab929/ybj/experiments/qri_acceleration/qri-fast-search-gpu0-20260821/`

No production QRI module was overwritten, and no extra top-level directory was
created under `/home/lab929/ybj`.
