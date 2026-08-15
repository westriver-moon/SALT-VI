"""Structured metadata for one retrieval evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolSpec:
    dataset: str
    search_mode: str
    gallery_mode: str
    gallery_trials: int
    query_modalities: tuple
    gallery_modalities: tuple
    caption_source: str | None
    caption_lookup: str | None
    retrieval_backend: str
    direction: str
    official: bool

    @property
    def identifier(self):
        if self.dataset == "sysu":
            return "{}-search-{}-shot-{}-trial-{}".format(
                self.search_mode,
                self.gallery_mode,
                self.gallery_trials,
                self.retrieval_backend,
            )
        if self.dataset == "regdb":
            return "regdb-{}-{}-trial-mean-{}".format(
                self.direction, self.gallery_trials, self.retrieval_backend
            )
        return "{}-{}-shot-{}-trial-{}".format(
            self.dataset,
            self.gallery_mode,
            self.gallery_trials,
            self.retrieval_backend,
        )

    def as_dict(self):
        return {
            "dataset": self.dataset,
            "search_mode": self.search_mode,
            "gallery_mode": self.gallery_mode,
            "gallery_trials": self.gallery_trials,
            "query_modalities": list(self.query_modalities),
            "gallery_modalities": list(self.gallery_modalities),
            "caption_source": self.caption_source,
            "caption_lookup": self.caption_lookup,
            "retrieval_backend": self.retrieval_backend,
            "direction": self.direction,
            "official": self.official,
        }


def build_protocol_spec(config, backend):
    dataset = str(getattr(config, "dataset", "")).lower()
    backend_name = str(getattr(backend, "NAME", "legacy"))
    if dataset == "regdb":
        trial_count = int(getattr(config, "eval_num_regdb", 1))
    else:
        trial_count = int(getattr(config, "gallery_trials", 10))
    if trial_count < 1:
        raise ValueError("evaluation trial count must be positive, got {}".format(trial_count))

    test_modality = str(getattr(config, "test_modality", ""))
    if backend_name == "ir_to_rgb_text":
        query_modalities = ("infrared-image",)
        gallery_modalities = ("visible-image", "visible-image-caption")
        caption_source = str(getattr(config, "gallery_caption_manifest", "") or "")
        caption_lookup = "gallery:image"
    else:
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

    if dataset == "sysu":
        direction = "infrared-to-visible"
        search_mode = str(getattr(config, "test_mode", "all")).lower()
        gallery_mode = str(getattr(config, "gall_mode", "single")).lower()
        official = search_mode == "all" and gallery_mode == "single" and trial_count == 10
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
        official = False
    elif dataset == "llcm":
        direction = "near-infrared-to-visible"
        search_mode = "standard"
        gallery_mode = "single"
        official = trial_count == 10
    else:
        raise ValueError("Unsupported retrieval dataset: {!r}".format(dataset))

    return ProtocolSpec(
        dataset=dataset,
        search_mode=search_mode,
        gallery_mode=gallery_mode,
        gallery_trials=trial_count,
        query_modalities=query_modalities,
        gallery_modalities=gallery_modalities,
        caption_source=caption_source,
        caption_lookup=caption_lookup,
        retrieval_backend=backend_name,
        direction=direction,
        official=official,
    )
