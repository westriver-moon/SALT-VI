# QRI glasses diffusion benchmark

This isolated benchmark uses the archived Qwen-v2 eyewear hypothesis for
`SYSU-MM01/cam1/0001/0001.jpg`. Repository code and configuration live under
`scripts/experiments/qri_glasses/` and `configs/experiments/qri_glasses/`.

Large outputs are written below:

`/home/lab929/ybj/experiments/qri_glasses/qri-glasses-gpu0-20260821/`

Downloaded diffusion models belong below:

`/home/lab929/ybj/models/qri-diffusion-bases/`

The benchmark first reproduces PASD with two seeds and three
guidance/conditioning settings. Alternative inpainting bases are evaluated only
after the PASD result fails visual inspection.

`inpaint_glasses_benchmark.py` evaluates the official Stable Diffusion v1.5
inpainting UNet. It reuses SALT's existing SD1.5 tokenizer, text encoder, VAE,
and scheduler, so the model store only needs the dedicated 9-channel UNet. All
generated pixels are composited through the archived eye mask, leaving the rest
of the person unchanged.

`identity_glasses_audit.py` reuses the established SALT-VI identity checkpoint
to measure cosine similarity between each localized candidate and the Swin
reference. It writes a separate JSON result next to the diffusion metrics.

`qwen_glasses_judge.py` sends only server-side source/reference/candidate crop
boards to the configured local Qwen-VL service and records a structured visual
judgment. It does not copy candidate images off the research server.

`blend_glasses_candidates.py` performs a zero-diffusion opacity sweep from one
Qwen-validated candidate. This cheap post-step searches for a point that keeps
the eyewear visible while restoring the ReID identity cosine gate.
