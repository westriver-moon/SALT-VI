# Person assets: canonical V1 + YOLO26 release

The only supported person-fit release is the audited V1 + YOLO26 result at:

`/home/lab929/ybj/experiments/person_preprocessing/person-fit-yolo26-full-20260827/final`

It covers 99,752 images. The final result is 96,257 passes and 3,495 fallbacks
(3.5037%), compared with 3,745 fallbacks (3.7543%) in V1. SYSU, RegDB, and LLCM
all improve over V1.

Quality evidence:

- 14,878 high-risk passes were reviewed in full.
- 3,448 machine fallbacks were reviewed in full.
- 103 partial-box structural risks were reviewed separately; 56 received a
  verified safe-union correction and 47 were moved to fallback.
- 1,800 ordinary passes were sampled by dataset and modality with zero observed
  mis-crops.
- No known partial-person structural failure remains in the pass set.

Sharpening is disabled because the 2,031-image probe reduced YOLO26 target
consensus from 1,617 on original images to 1,538 with mild unsharp masking and
1,234 with medium unsharp masking.

The canonical output root is:

`/home/lab929/ybj/datasets/person-assets-512x256/person_fit`
