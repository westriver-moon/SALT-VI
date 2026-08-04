# SALT-VI canonical document

This document consolidates related legacy material. All configuration, code, data and output references below have been rewritten to the SALT-VI layout.


---

## Migrated source: super resolution protocol

> Source document ID: `source_baseline:docs/super_resolution_protocol.md`  
> Original SHA-256: `22eebdc002442659057305d68ce79f9ea3dd54f3df70c1f7b40d7a6328a9bf10`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# SYSU-MM01 PMT-MBPatch x2 Super-Resolution Protocol

## Experiment identity

This ablation is a **pure-visual PMT experiment**. It does not inherit SALT-VI,
TEXTV1, captions, text losses, or a text warm-start. The common model is the
independent Stage-A PMT-MBPatch structure in `sysu_pmt_mbpatch.yaml`, trained
from the same ImageNet ViT initialization with the PMT 24-epoch recipe.

The Stage-A checkpoint was re-evaluated over the complete 10-trial SYSU-MM01
protocol before selecting this structure. Its audit result is Rank-1 67.24%,
mAP 65.82%, and mINP 53.51%. The historical 70.44% number is a single-trial
observation and is not used as the formal baseline.

## Controlled groups

All pixels first pass through the exact PMT low-resolution source contract:
the original SYSU input is resized to 256x128 with PIL bilinear interpolation.

- A0: model input is the 256x128 common source.
- A1: the common source is enlarged to 512x256 with bicubic interpolation.
- A2: RGB uses SwinIR-M classical SR x2; IR uses the A1 bicubic path.
- A3: RGB and grayscale IR use SwinIR-M classical SR x2.

The SwinIR assets must be schema v4 data generated from the same 256x128
bilinear source. The old 288x144 to 576x288 derived data is incompatible and
must never be reused by these experiments.

## Resource and validity controls

The per-modality batch remains 32 with four samples per identity (PK=8x4).
High-resolution jobs use AMP, gradient checkpointing, and backbone-only batch
chunking; features are concatenated before the shared BN, classifier, and
metric losses, so global batch/mining semantics are unchanged.

Each group must pass a provenance-bound 20-step forward/backward preflight and
a one-trial retrieval evaluation below 22 GiB before formal launch. Formal
training evaluates 10 SYSU trials every two epochs. The launcher only accepts
clean `origin/main`, exact-current preflights, empty output directories, and
dynamically idle GPUs.


---

## Migrated source: SYSU SR FAILURE ANALYSIS

> Source document ID: `source_core:SYSU_SR_FAILURE_ANALYSIS.md`  
> Original SHA-256: `e92fde231219e7a2c4524ace834d5d1a974b8e7c77e1787de570e36327282b4a`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# SYSU-MM01 SR failure analysis and rebuild contract

Date: 2026-07-18  
Affected branch: `codex/sysu-sr-ablation`  
Affected implementation: through commit `6d913057d`  
Affected derived dataset: `/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-v1`

## Decision

The first SR-A0/A1/A2/A3 run is not a valid four-way super-resolution
ablation. SR-A2 and SR-A3 consumed corrupted SwinIR assets. SR-A1 did not use
the same 288 x 144 low-resolution training source as the SwinIR groups. The
run must not be used to conclude that SwinIR is ineffective on SYSU-MM01.

## Confirmed failures

### 1. SwinIR CUDA FP16 produced non-finite output

`src/salt_vi/utils/super_resolution/build_sysu_swinir_x2.py` ran the official SwinIR-M
model under CUDA autocast. On the installed PyTorch 1.8.1 + CUDA 11.1 stack,
the tested FP16 forward contained no finite output values. The same image and
checkpoint produced a finite, non-degenerate result in FP32 (approximately
0.009 to 0.734).

The conversion path clamped and cast the tensor to `uint8` without first
checking `torch.isfinite`. NaNs consequently became zero-valued pixels without
raising an exception.

Full scans of the generated assets found:

| Asset | Image count | Non-zero values/images |
|---|---:|---:|
| RGB train memmap | 22,258 | 0 |
| IR train memmap | 11,909 | 0 |
| RGB evaluation PNGs | 6,775 | 0 |
| IR evaluation PNGs | 3,803 | 0 |

All 10,578 evaluation images and both training arrays were therefore black.

### 2. Validation proved integrity, not semantic correctness

The v1 validator checked count, shape, dtype, file coverage, hashes, and equal
IR channels. A reproducible all-zero array satisfies all of those checks.
There was no finite-value gate before conversion and no output range,
variance, all-zero-image, or source/output consistency check.

The preflight similarly accepted a successful process and acceptable peak
memory without recording or gating on actual retrieval metrics or feature
diversity. It therefore marked collapsed inputs as valid.

### 3. SR-A1 violated the shared-source contract during training

The original training arrays are 384 x 144, not 288 x 144:

| Array | Shape |
|---|---|
| `train_rgb_resized_img.npy` | `(22258, 384, 144, 3)` |
| `train_ir_resized_img.npy` | `(11909, 384, 144, 3)` |

The SwinIR builder explicitly standardized those arrays to 288 x 144 before
x2 inference. Evaluation also standardized non-SR inputs. Training for SR-A1,
however, resized the original arrays directly from 384 x 144 to 576 x 288.
Thus `A2 - A1` was not a controlled SwinIR-versus-bicubic comparison.

### 4. The protocol omitted the warm-start baseline and overfit rapidly

SR-A0 reached its best recorded Rank-1 at epoch 1 (72.61%) and fell to 45.30%
by epoch 31 while training accuracy approached 99.7%. No epoch-0 evaluation
was recorded after loading the warm start. This is a protocol weakness rather
than the cause of the black assets, but it prevents a clean measurement of the
starting checkpoint and makes a fixed 33-epoch interpretation unsafe.

### 5. Token-count acceptance criterion was off by one

For the current visual patch embedding (`kernel=16`, `stride=12`), a 576 x 288
image yields a 47 x 23 grid: 1,081 patch tokens and 1 class token, for 1,082
tokens total. The observed interpolated positional embedding `(1, 1082, 768)`
is correct; the previous 1,083-token requirement was not.

## Invalid result record

| Group | Best Rank-1 | mAP | mINP | Status |
|---|---:|---:|---:|---|
| SR-A0 | 72.61% | 69.92% | 57.11% | Numeric run valid; protocol overfit |
| SR-A1 | 63.66% | 61.26% | 47.44% | Invalid as controlled size baseline |
| SR-A2 | 1.13% | 4.12% | 4.84% | Invalid: RGB assets black |
| SR-A3 | 1.05% | 3.60% | 4.19% | Invalid: RGB and IR assets black |

## Cleanup boundary

The following generated state is safe and necessary to remove before rebuild:

- `/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-v1/` in full. It contains
  only the confirmed all-zero derived assets and their invalid manifest.
- model checkpoints under the old untracked `logs/raw/source_core/reports/super_resolution/` run
  directories. Small logs, event streams, configs, manifests, and summaries
  may be retained under an explicitly invalid archive for auditability.

The source dataset `/home/cgv841/datasets/SYSU-MM01/`, the official SwinIR
checkpoint, and the user's unrelated untracked metric-boost run are outside
the cleanup boundary and must not be modified.

## Rebuild contract

The replacement implementation is complete only when all of these conditions
hold:

1. SwinIR asset generation is explicitly FP32 end-to-end; it does not depend
   on ambient autocast state.
2. Every model output is checked for finite values and exact x2 shape before
   conversion or writing.
3. Generation begins with a real-checkpoint smoke forward before allocating
   the full output arrays.
4. Manifest schema v2 records numeric policy and streaming content statistics.
5. Full validation rejects all-zero images, degenerate dynamic range, unequal
   IR channels, source mutation, output mutation, missing paths, and poor
   downsampled-output/source consistency.
6. Partial-build resume state is versioned and bound to source/checkpoint
   hashes so an incompatible partial cannot be reused.
7. Every non-SwinIR training modality is explicitly resized to 288 x 144
   before its final model-size resize. The same contract applies at evaluation.
8. Preflight records finite retrieval metrics and rejects missing or collapsed
   fusion results in addition to enforcing the memory limit.
9. Tests cover NaN rejection, degenerate-content rejection, IR equality,
   source-size normalization, and the correct 1,082-token high-resolution
   geometry.
10. Rebuilt assets use a new root (`SYSU-MM01-swinir-x2-v2`) so stale v1 state
    can never be selected accidentally.

Until all gates pass and the four groups are rerun under the same rebuilt
protocol, any SR conclusion must be labelled invalid/preliminary.


---

## Migrated source: SYSU SR POST MERGE ERROR AUDIT

> Source document ID: `source_core:SYSU_SR_POST_MERGE_ERROR_AUDIT.md`  
> Original SHA-256: `bcf36d7f71a664c683efbaec3b288cdd576d018544aa931919b8fa9b94b4b526`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# SYSU-MM01 SR post-merge error audit

Date: 2026-07-18  
Audited branch: `main`  
Audited commit: `28d104f622d35ce88b92a5cb5456e85573542db1`  
Audit type: static inspection plus read-only runtime reproduction

## Decision

The supplied review is substantially correct. All seven reported implementation
problems exist in the audited commit. Four directly break execution or the
controlled-ablation/provenance contract; three are validation and artifact
management weaknesses that can make a completed run ambiguous or insufficiently
guarded.

This audit does **not** find evidence that a v2 derived dataset is already
contaminated. At audit time:

- `/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-v2` did not exist;
- `logs/raw/source_core/reports/super_resolution/` contained no active preflight files;
- old preflight files existed only under the explicitly invalid archive
  `reports/super_resolution-invalid-20260718/`.

Therefore the correct conclusion is: the code is not ready for a formal four-group
run, but no existing v2 assets need to be declared corrupt.

## Confirmed high-priority errors

### 1. The formal runner cannot start from `main`

Status: **confirmed and reproduced**.

`scripts/super_resolution/run_sysu_sr_ablation.py:70-71` requires the current
branch name to equal `codex/sysu-sr-ablation`. That branch was merged into and
deleted in favor of `main`.

Read-only reproduction on the audited server returned:

```text
RuntimeError: Formal SR runs require branch codex/sysu-sr-ablation
```

The same source-state check runs only `git diff --quiet`. That checks unstaged
tracked changes but does not prove that the index has no staged changes. Formal
provenance therefore has two defects: an obsolete branch-name requirement and an
incomplete clean-tree check.

Impact: formal execution on the requested sole branch is blocked immediately;
staged-but-uncommitted algorithm changes could also evade the intended gate.

### 2. Preflight results are reused without provenance binding

Status: **confirmed code defect; no active stale file currently present**.

`scripts/super_resolution/continue_sysu_sr_pipeline.py:72` treats the existence
of a same-named preflight JSON as sufficient reason not to regenerate it.
`scripts/super_resolution/run_sysu_sr_ablation.py` subsequently checks only that
the loaded result has `valid: true`.

The preflight artifact is not bound to the current:

- Git commit;
- serialized runtime configuration hash;
- SR manifest hash and data root;
- warm-start checkpoint hash;
- corrected-text manifest hash;
- preflight implementation version.

Because the experiment IDs are unchanged and the report directory is ignored by
Git, a stale same-named result can survive code changes and be accepted as current.
The prior invalid preflights have been moved outside the active directory, so this
failure mode has not happened to the new v2 run yet.

Impact: a future stale file can bypass the new data and model checks and authorize
a run it did not actually test.

### 3. `SR-A1-bicubic-x2` uses bilinear training interpolation

Status: **confirmed and reproduced in the actual `clipreid` environment**.

`src/salt_vi/data/loader.py:54-55` constructs both source-size and model-size
`torchvision.transforms.Resize` operations without specifying interpolation.
Runtime inspection produced:

```text
a1_resize 0 (288, 144) InterpolationMode.BILINEAR
a1_resize 1 (576, 288) InterpolationMode.BILINEAR
```

By contrast, the SwinIR builder and SYSU evaluation source normalization use PIL
bicubic. Consequently the training comparison is currently:

```text
A1: 384x144 --bilinear--> 288x144 --bilinear--> 576x288
A2/A3 source: 384x144 --PIL bicubic--> 288x144 --SwinIR--> 576x288
```

The current tests assert only Resize sizes, not interpolation mode or pixel-level
equivalence.

Impact: `A2 - A1` mixes the SR-method effect with a different low-resolution
source construction and is not a controlled SwinIR-versus-bicubic comparison.

### 4. Finalized assets can be reused without a manifest/source contract

Status: **confirmed code defect; no v2 assets currently exist**.

The versioned progress sidecar correctly protects an in-progress training array.
However, after a training array has been renamed to its final path, a restart with
no manifest enters `if output_path.exists()` and reuses it based on shape, dtype,
and non-degenerate statistics. It does not prove which checkpoint, source hash,
Git revision, or numeric policy created that finalized array.

Evaluation generation similarly skips every destination file that already exists.
Those files have no per-file or build-session provenance sidecar. A crash before
manifest creation followed by a checkpoint or code change can therefore produce a
single final dataset containing assets from different generators, while the final
manifest describes only the last invocation.

Impact: manifest provenance can make a mixed-history dataset appear homogeneous.
The safe production rule is to reject a non-empty output root without a completed
manifest, or to bind every resumable asset to one immutable build identity.

## Confirmed secondary errors

### 5. Preflight evaluates only after 20 training steps and uses a 2% Rank-1 gate

Status: **confirmed**.

`scripts/super_resolution/preflight_sysu_sr.py` performs the limited training
loop before its full evaluation. It therefore cannot distinguish a bad initial
load/input domain from a failure introduced during the 20-step update. The formal
training entrypoint has an epoch `-1` warm-start evaluation, but preflight does not.

The default acceptance floor is Rank-1 `0.02`. Metrics are checked for finiteness
and range, but only Rank-1 is compared with a quality floor; mAP and mINP have no
quality threshold. A severe regression well above random retrieval can pass.

Impact: the gate catches the known approximately 1% black-image collapse but is
too weak to protect the expected approximately 70-80% warm-start regime.

### 6. Source/output consistency gates only mean PSNR

Status: **confirmed**.

`src/salt_vi/utils/super_resolution/validate_sysu_swinir_x2.py` records both
`minimum_psnr` and `mean_psnr`, but lines 216-220 reject only when the mean is
below the threshold. A small number of badly ordered, mismapped, or corrupted
sample pairs can be hidden by otherwise high-PSNR samples.

Sampling is deterministic and spread across the ordered arrays, which is better
than a prefix-only sample, but it does not explicitly guarantee per-camera or
per-identity coverage for evaluation data.

Impact: localized mapping errors are not strongly gated even though their minimum
PSNR is already calculated.

### 7. An epoch `-1` winner has no run-local best checkpoint

Status: **confirmed**.

the canonical train entrypoint (`src/salt_vi/entrypoints/train.py`) records the warm-start evaluation as epoch `-1` and initializes
the best metrics from it. The event points to the external warm-start path, but the
model's `_metric_checkpoint_paths` are not initialized from that artifact and no
copy or hard link is created in the run directory.

`run_sysu_sr_ablation.py:208-219` selects the best event by Rank-1 but returns only
the epoch and metrics. If no trained epoch beats the warm start, status and summary
can correctly say that epoch `-1` won while exposing no run-local selected
checkpoint or selected-checkpoint SHA-256.

Impact: downstream consumers can fail to find the true winning model or select an
inferior trained checkpoint despite the metrics summary being numerically correct.

## Parts of the previous repair that remain valid

The new review does not overturn the following verified repairs:

- SwinIR inference is explicitly FP32 with autocast disabled;
- non-finite output is rejected before uint8 conversion;
- exact x2 output geometry and non-degenerate content are checked;
- IR output channels are forced equal and validated;
- RGB/IR training and evaluation modality routing is correct;
- PMT high-resolution position geometry is 47 x 23 patches plus CLS, or 1,082
  total tokens;
- current active storage contains neither stale preflight output nor a partially
  generated v2 dataset.

## Readiness conclusion

Do not launch the formal four-group experiment from `main@28d104f62`. The runner
is currently guaranteed to fail, and after bypassing that failure the A1/A2
comparison would still violate the intended interpolation control. Provenance and
artifact-selection weaknesses could additionally authorize stale preflight state
or leave a completed run without an unambiguous winning checkpoint.

This document records diagnosis only. No implementation was changed as part of
this audit.


---

## Migrated source: SYSU SR POST MERGE FIX VERIFICATION

> Source document ID: `source_core:SYSU_SR_POST_MERGE_FIX_VERIFICATION.md`  
> Original SHA-256: `193420e44adfd92f6bcf2f8ee932be0f77f7b5108c8336e9145400fa4d3ef519`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# SYSU-MM01 SR post-merge remediation verification

Date: 2026-07-18  
Target branch: `main`  
Source audit: `SYSU_SR_POST_MERGE_ERROR_AUDIT.md`

## Resolution summary

All seven confirmed findings from the post-merge audit have corresponding
implementation changes and regression coverage.

| Finding | Resolution | Verification |
|---|---|---|
| Runner hard-coded deleted branch; index not checked | Formal source gate compares `HEAD` with an explicit ref (`origin/main` by default), checks both worktree and index, and rechecks the frozen commit before every launch | Unit tests cover main-by-commit acceptance and staged-change rejection |
| Stale preflight reuse | Experiment IDs are `SRV2-*`; provenance v4 binds Git SHA, the complete algorithm tree, images and labels, test IDs/evaluation trees, verified text assets, warm start, SR manifest, config and optional A0 reference | Exact-provenance tests reject stale JSON, dirty algorithm code and modified text assets |
| A1 used bilinear | SYSU source and final Resize operations explicitly use `InterpolationMode.BICUBIC`; evaluation uses the same PIL/torchvision bicubic definition | Pixel-level regression test matches the two-stage training result to explicit PIL bicubic bytes |
| Finalized assets reused without provenance | An immutable build identity is established before any asset write and governs train partials, finalized arrays, and evaluation files; a non-empty root without that identity is rejected | Tests cover provenance-free rejection, exact resume, and mismatched-identity rejection |
| Preflight evaluated only after training and used a 2% gate | Preflight now evaluates the untouched warm start, runs 20 training steps, evaluates again, and gates Rank-1/mAP/mINP against A0 floors/reference and post-training drop limits | Real A0 dual-stage GPU run completed successfully |
| Only mean PSNR was gated | Validation now gates minimum, P05, and mean PSNR and reports P10; evaluation samples are stratified per camera while training samples combine endpoints, spread, and deterministic random indices | A bad-tail test fails even when mean PSNR remains high |
| Epoch -1 winner lacked an exact run-local checkpoint | The converted in-memory model state is saved after QBN expansion and positional interpolation; source and converted hashes plus conversion metadata are recorded | Tests verify exact state materialization, metric mapping and winner SHA reporting |
| Preflight/formal epoch-0 controls diverged | Preflight now constructs the same warmup scheduler and applies `configure_epoch_trainability(0)` plus `scheduler.step(0)` before its 20-step smoke | Regression coverage verifies the formal epoch-0 control sequence |
| SwinIR identity was recorded but not enforced | Builder and validator require the official revision, checkpoint hash and imported network hash; the build identity is rechecked before manifest publication | Identity mismatch tests reject non-official inputs |

## Runtime verification

### Full test suite

```text
125 passed
```

### Official SwinIR FP32 smoke

```text
RGB downsampled-source PSNR: 57.7464 dB
IR  downsampled-source PSNR: 59.5720 dB
all-zero images: 0
non-finite values: 0
```

### Superseded A0 dual-stage preflight on RTX 3090

The earlier output was intentionally written under `/tmp`, not the formal report
directory. Its untouched warm-start metrics remain informative, but its 20-step
result is superseded because that preflight omitted the formal warmup scheduler.

| Phase | Rank-1 | mAP | mINP |
|---|---:|---:|---:|
| Untouched warm start | 64.18% | 62.55% | 49.95% |
| After 20 steps (superseded; wrong LR) | 71.82% | 69.18% | 56.27% |

Peak allocated GPU memory was 3,357,317,632 bytes (about 3.13 GiB), below the
22 GiB limit. This remains useful only as a memory observation; metric gates
must be rerun under provenance v4 and the formal epoch-0 scheduler.

## Remaining external work

This remediation validates the implementation. It does not generate the full
v2 derived dataset or launch formal SRV2 training. Those are subsequent data
production and experiment operations and must begin only from the pushed clean
`main` commit so the new source/provenance gates can succeed.


---

## Migrated source: SR TAIL EXTENSION PROTOCOL

> Source document ID: `source_baseline:SR_TAIL_EXTENSION_PROTOCOL.md`  
> Original SHA-256: `42a4f3c5476c83ab3c03a96bdab4cb8cd6216c673c091a2706e75db7f891bf41`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# Equal-budget A1/A3 tail-extension protocol

This is an auxiliary performance-exploration protocol, not a replacement for the
completed 24-epoch A1/A3 super-resolution ablation. The original results remain
the only equal-protocol comparison with A0.

Both runs use the completed epoch-24 `latest.pth` as *model weights only*, then
train epochs 25--32 in new output directories. The optimizer and AMP scaler are
fresh. Do not pass `--resume`: it would restore the optimizer's exhausted original
cosine state and defeat the tail-restart schedule.

The two runs have matched tail budgets, image resolutions, AMP, PK=8x4, test batch
size, visual chunking, evaluation interval, and learning-rate schedule. At epoch
25 the head LR is 1.46e-5; it cosine-decays to 1e-6 at epoch 32.

Commands (do not launch until explicitly requested):

```bash
python -m salt_vi.baselines.vision_text.train \
  --config configs/vision_text/super_resolution/sr_a1_bicubic_x2_tail_e32.yaml \
  --device cuda:0 \
  --output /home/cgv841/ybj/SALT-VI/logs/raw/source_baseline/outputs/super_resolution/PMSR-A1-bicubic-x2-tail-e32 \
  --weights /home/cgv841/ybj/SALT-VI/logs/raw/source_baseline/outputs/super_resolution/PMSR-A1-bicubic-x2/checkpoints/latest.pth \
  --override model.backbone_chunk_size=8 \
  --override test.batch_size=8

python -m salt_vi.baselines.vision_text.train \
  --config configs/vision_text/super_resolution/sr_a3_swinir_both_x2_tail_e32.yaml \
  --device cuda:0 \
  --output /home/cgv841/ybj/SALT-VI/logs/raw/source_baseline/outputs/super_resolution/PMSR-A3-swinir-both-x2-tail-e32 \
  --weights /home/cgv841/ybj/SALT-VI/logs/raw/source_baseline/outputs/super_resolution/PMSR-A3-swinir-both-x2/checkpoints/latest.pth \
  --override model.backbone_chunk_size=8 \
  --override test.batch_size=8
```

Report maximum Rank-1 in epochs 25--32 with same-epoch mAP and mINP, separately
from the formal 24-epoch result. Do not claim an A1/A3-vs-A0 ablation difference
using the extended scores unless A0 receives the same extension.
