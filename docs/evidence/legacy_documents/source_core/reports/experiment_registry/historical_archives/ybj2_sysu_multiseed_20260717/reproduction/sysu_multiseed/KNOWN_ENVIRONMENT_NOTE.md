# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/historical_archives/ybj2_sysu_multiseed_20260717/reproduction/sysu_multiseed/KNOWN_ENVIRONMENT_NOTE.md`  
> Original SHA-256: `73f0b69c79be7cbf4fb574bc6def33b4993d682a22a68b9ccbe033140c89b5fd`  
> This is read-only experiment evidence, not an active runtime instruction.

# Environment compatibility note

The official `requirements.txt` pins both `torch==2.0.1` and `timm==0.3.2`.
Importing `timm==0.3.2` under PyTorch 2.0.1 fails because that old timm release
imports the removed `torch._six` module.

The pinned package is retained unchanged for provenance. A source scan of the
official SALT-VI commit found no timm import in any Python execution path, and
the real `scripts/train.py --help` entrypoint imports successfully. Therefore preflight
checks the actual project entrypoint rather than importing this unused package.
No dependency was upgraded and no source shim was introduced.
