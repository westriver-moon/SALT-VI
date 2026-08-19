# Unified PASD Plugin

`pasd_plugin` is the single offline PASD generator for SYSU-MM01, RegDB, and LLCM.
It creates one 256x512 PNG per official RGB and IR/NIR source image, including
training and evaluation sources. The default `person_fit_blurred_background` mode preserves
source aspect ratio by uniformly fitting the complete frame over a blurred, same-image
background. QRI uses the separate `direct_rewrite` mode: its input is already a canonical
256×512 SwinIR canvas, so PASD must keep identity coordinates and is forbidden to detect,
letterbox, pad, crop, or restore a different background canvas.

Both modalities use PASD. Infrared/NIR PASD outputs are converted to three-channel
greyscale after generation, preserving their modality contract.

## Commands

```bash
PYTHONPATH=. python -m pasd_plugin build-records --config pasd_plugin/configs/llcm.yaml
PYTHONPATH=. python -m pasd_plugin generate --config pasd_plugin/configs/llcm.yaml --workers 1
PYTHONPATH=. python -m pasd_plugin validate --config pasd_plugin/configs/llcm.yaml
```

The configuration carries the dataset root, both caption dictionaries, PASD assets,
output root, and GPU policy. `build-records` rejects missing or unused captions.
`generate` is resumable and writes `build.json`, source metadata, `manifest.jsonl`,
and `manifest.json`. `validate` checks source/output checksums, protocol membership,
image integrity, and the selected geometry invariant. Validation distinguishes the legacy
aspect-preserving person-fit transform from QRI's identity-coordinate direct rewrite.

RegDB emits one output per unique source and records membership for all ten trials;
it never duplicates generated images by trial. SYSU records train/val/test identity
membership and LLCM records canonical train/test index membership and labels.

This plugin replaces the former `pasd_offline` module. It deliberately does not
modify SALT-VI training loaders; consuming these manifests is a separate training
integration task.
