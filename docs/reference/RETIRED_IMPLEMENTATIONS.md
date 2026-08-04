# SALT-VI canonical document

This document consolidates related legacy material. All configuration, code, data and output references below have been rewritten to the SALT-VI layout.


---

## Migrated source: multistage 5 8 retirement

> Source document ID: `source_core:docs/archive/multistage_5_8_retirement.md`  
> Original SHA-256: `3d9d8cab9411dbb042f85ff8d428953e19b30edef6711404f464d3d6fea947ac`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# Retired 5/8 multistage text-conditioning path

The block-5/block-8 multistage text-conditioning experiment is retired from the active source tree. It is not a compatible implementation of the bidirectional token-cycle contract: it only injected at selected depths, did not preserve the new per-block recurrent state contract, and used different checkpoint keys.

Removed active paths:

- `SALT-VI/src/salt_vi (retired path; not present in the active tree)`
- `SALT-VI/configs (retired path; no active configuration)`
- `SALT-VI/scripts (retired path; no active launcher)`
- the matching `test_multistage_*` regression files

The last pre-retirement repository state is commit `d27b017095e82387e0f708125b858e075e74ca82`. Git can recover any removed source or configuration, for example:

```bash
git show d27b017095e82387e0f708125b858e075e74ca82:SALT-VI/SALT-VI/src/salt_vi (retired path; not present in the active tree)
```

Old checkpoints containing `multistage_text_conditioner.*` keys are rejected explicitly. They must not be loaded as warm starts for `bidirectional_cycle_v1`, because silently dropping those keys would make provenance ambiguous.

Historical report artifacts under `SALT-VI/reports/multistage_text_conditioning/` were not deleted. They remain experimental evidence, not executable source.
