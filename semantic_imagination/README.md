# SALT Semantic Imagination

This offline plugin turns one ambiguous pedestrian image into a weighted set of
semantic hypotheses and exports those hypotheses as canonical PASD source
records. It does not import SALT training code or PASD model code.

The authoritative mathematical meaning is fixed in
[`MATHEMATICAL_SPEC.md`](MATHEMATICAL_SPEC.md). Code and downstream consumers
must preserve that document's invariants.

The plugin deliberately does not choose a VLM, perturbation family, or text
embedding model. A backend supplies four operations: factual observation,
semantic-preserving perturbation, stochastic imagination, and text embedding.
The core owns sampling, clustering, empirical mass, medoid selection, and PASD
record export.

```python
from pathlib import Path

from semantic_imagination import build_hypothesis_manifest, to_pasd_record

manifest = build_hypothesis_manifest(
    image=Path("person.jpg"),
    source_key="cam1/0001/person.jpg",
    backend=my_backend,
    instruction="Describe only plausible details that are not sufficiently observed.",
    sample_count=20,
    seed=20260811,
    similarity_threshold=0.85,
)
record = to_pasd_record(manifest, output_dir="images/cam1/0001/person")
```

For data-dependent hypothesis count, use `views_per_source: 0` in the PASD
generation config and `sysu_sr_views_per_image: 0` in the SALT config. Existing
one-view and five-view datasets remain valid. Missing hypothesis weights in
legacy manifests are interpreted as a uniform distribution.
