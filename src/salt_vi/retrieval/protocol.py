"""Structured metadata for one retrieval evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolSpec:
    dataset: str
    search_mode: str
    gallery_mode: str
    gallery_trials: int
    trial_ids: tuple
    query_modalities: tuple
    gallery_modalities: tuple
    caption_source: str | None
    caption_lookup: str | None
    eval_caption_seed: int | None
    retrieval_backend: str
    direction: str
    test_modality: str
    official_sampling_protocol: bool

    @property
    def identifier(self):
        """Return a human-readable identity covering every protocol dimension."""
        fields = (
            ("dataset", self.dataset),
            ("search", self.search_mode),
            ("gallery", self.gallery_mode),
            ("trial_ids", ",".join(str(value) for value in self.trial_ids)),
            ("direction", self.direction),
            ("backend", self.retrieval_backend),
            ("test_modality", self.test_modality),
            ("query", "+".join(self.query_modalities)),
            ("gallery_modalities", "+".join(self.gallery_modalities)),
            ("caption_lookup", self.caption_lookup or "none"),
            (
                "caption_seed",
                "none" if self.eval_caption_seed is None else str(self.eval_caption_seed),
            ),
            ("caption_source", self.caption_source or "none"),
        )
        return "|".join("{}={}".format(key, value) for key, value in fields)

    def as_dict(self):
        return {
            "identifier": self.identifier,
            "dataset": self.dataset,
            "search_mode": self.search_mode,
            "gallery_mode": self.gallery_mode,
            "gallery_trials": self.gallery_trials,
            "trial_ids": list(self.trial_ids),
            "query_modalities": list(self.query_modalities),
            "gallery_modalities": list(self.gallery_modalities),
            "caption_source": self.caption_source,
            "caption_lookup": self.caption_lookup,
            "eval_caption_seed": self.eval_caption_seed,
            "retrieval_backend": self.retrieval_backend,
            "direction": self.direction,
            "test_modality": self.test_modality,
            "official_sampling_protocol": self.official_sampling_protocol,
        }


def build_protocol_spec(config, backend):
    dataset = str(getattr(config, "dataset", "")).lower()
    backend_name = str(getattr(backend, "NAME", "identity_text"))
    if dataset == "regdb":
        trial_count = int(getattr(config, "eval_num_regdb", 1))
        first_trial = int(getattr(config, "trial", 1))
        trial_ids = tuple(range(first_trial, first_trial + trial_count))
    else:
        trial_count = int(getattr(config, "gallery_trials", 10))
        trial_ids = tuple(range(trial_count))
    if trial_count < 1:
        raise ValueError("evaluation trial count must be positive, got {}".format(trial_count))
    if dataset == "regdb" and (trial_ids[0] < 1 or trial_ids[-1] > 10):
        raise ValueError(
            "RegDB evaluation trials must be a consecutive range within [1, 10], "
            "got {}".format(list(trial_ids))
        )

    test_modality = str(getattr(config, "test_modality", ""))
    eval_caption_seed = int(getattr(config, "eval_caption_seed", 0))
    if not 0 <= eval_caption_seed <= 2**32 - 1:
        raise ValueError(
            "eval_caption_seed must be in [0, 2**32 - 1], got {}".format(
                eval_caption_seed
            )
        )
    if backend_name == "ir_to_rgb_text":
        query_modalities = ("infrared-image",)
        gallery_modalities = ("visible-image", "visible-image-caption")
        caption_source = str(getattr(config, "gallery_caption_manifest", "") or "")
        caption_lookup = "gallery:image"
    elif backend_name == "identity_text":
        query_modalities = ("infrared-image",)
        caption_source = None
        caption_lookup = None
        if "Fusion" in test_modality:
            query_modalities = ("infrared-image", "visible-identity-caption")
            caption_source = str(getattr(config, "text_data_root", "") or "")
            caption_lookup = "query:identity"
        elif "Text" in test_modality:
            query_modalities = ("visible-identity-caption",)
            caption_source = str(getattr(config, "text_data_root", "") or "")
            caption_lookup = "query:identity"
        gallery_modalities = ("visible-image",)
    else:
        raise ValueError("Unsupported retrieval protocol: {!r}".format(backend_name))

    if dataset == "sysu":
        direction = "infrared-to-visible"
        search_mode = str(getattr(config, "test_mode", "all")).lower()
        gallery_mode = str(getattr(config, "gall_mode", "single")).lower()
        official_sampling_protocol = (
            search_mode == "all" and gallery_mode == "single" and trial_count == 10
        )
    elif dataset == "regdb":
        regdb_mode = str(getattr(config, "regdb_test_mode", "t-v")).lower()
        if regdb_mode == "t-v":
            direction = "thermal-to-visible"
        else:
            direction = "visible-to-thermal"
            query_modalities, gallery_modalities = gallery_modalities, query_modalities
            if caption_lookup == "query:identity":
                caption_lookup = "gallery:identity"
        search_mode = "numbered-trial"
        gallery_mode = "single"
        official_sampling_protocol = False
    elif dataset == "llcm":
        direction = "near-infrared-to-visible"
        search_mode = "standard"
        gallery_mode = "single"
        official_sampling_protocol = trial_count == 10
    else:
        raise ValueError("Unsupported retrieval dataset: {!r}".format(dataset))

    caption_seed = eval_caption_seed if caption_lookup is not None else None
    return ProtocolSpec(
        dataset=dataset,
        search_mode=search_mode,
        gallery_mode=gallery_mode,
        gallery_trials=trial_count,
        trial_ids=trial_ids,
        query_modalities=query_modalities,
        gallery_modalities=gallery_modalities,
        caption_source=caption_source,
        caption_lookup=caption_lookup,
        eval_caption_seed=caption_seed,
        retrieval_backend=backend_name,
        direction=direction,
        test_modality=test_modality,
        official_sampling_protocol=official_sampling_protocol,
    )
