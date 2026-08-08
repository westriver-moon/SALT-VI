# Vendored PASD source

- Upstream: `https://github.com/yangxy/PASD.git`
- Revision: `396f9ac24f9fa2b9787658bd9eea31729e51f264`
- Imported scope: the upstream `pasd` Python package
- Runtime purpose: offline PASD inference and future PASD model training

The snapshot is stored under `vendor/pasd` so the offline module has no runtime
dependency on a separate PASD checkout. Upstream code remains covered by the
license copied to `vendor/LICENSE-PASD`.

Local compatibility patch: tiled-VAE attention calls use the standard
`torch.nn.Linear` signature required by the upstream-pinned Diffusers 0.29.2.
