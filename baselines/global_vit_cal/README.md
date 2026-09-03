# SALT-VI global CAL baseline

This is the standalone baseline implementation for the final method. It is
deliberately separate from SALT-VI and contains no experiment checkpoints or
training outputs.

The implementation combines five pieces:

1. a single global ViT backbone with CAL;
2. pose-posterior token confidence dropout with the final 15% contract;
3. the proven shallow superellipse-family attention loss (the validated
   exponent is p=2, i.e. the ellipse member of that family);
4. Qwen SwinIR ROI zoom annotation with repeated regional hypotheses,
   complete-link clustering and empirical joint-world sampling.
5. an optional Stage-B cross-modal hard-mining schedule, with an offline
   Qwen-world-to-text bridge.

The main ViT consumes the original 288x144 image. SwinIR is an auxiliary
512x256 observation used only by pose/Qwen ROI annotation; it is never fed to
the ViT backbone.

## Quick validation

From the repository root:

    python -m unittest discover -s tests -v
    python scripts/verify_contract.py
    python scripts/dry_run.py

No real model weights, Qwen server or training run are required by these
checks. See docs/BASELINE_CONTRACT.md for the fixed equations and
docs/IMPLEMENTATION_NOTES.md for the audited implementation sources.
See docs/STAGE_B_PROFILE.md for the Stage-B code profile and its separate
offline-Qwen derived configuration.

The source layout is intentionally flat: backbone.py owns the global ViT,
token_dropout.py owns the final gate, pose_posterior.py owns dense TTA support,
losses.py and recipe.py own objectives, and qwen/ owns the complete offline
regional annotation path. docs/ARCHITECTURE.md describes the two data flows.
