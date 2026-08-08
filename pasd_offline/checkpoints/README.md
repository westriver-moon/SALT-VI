# Runtime model assets

Store the independent offline runtime weights here:

- `stable-diffusion-v1-5/`: scheduler, tokenizer, text encoder, VAE, and feature extractor;
- `pasd/checkpoint-100000/`: PASD UNet and ControlNet.

The files are intentionally excluded from Git. No symlink may point to the
separate PASD reproduction workspace.
