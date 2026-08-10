from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PID_COUNTS = {"sysu": 395, "regdb": 206, "llcm": 713}


def _apply_overrides(config: Any, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise KeyError(f"Unknown SALT config override: {key}")
        setattr(config, key, value)


@dataclass
class SplitSource:
    split: str
    split_tag: str
    loader: Any
    sample_ids: np.ndarray
    labels: np.ndarray
    cameras: np.ndarray


class UniqueTrainDataset:
    def __init__(self, samples: Any, transform: Any, modality: str, views: Any):
        import torch
        from salt_vi.data.dataset import tokenize

        self.samples = samples
        self.transform = transform
        self.modality = modality
        self._tokenize = tokenize
        if modality == "rgb":
            self.store = samples.multiview_stores.get("rgb")
            self.images = samples.train_color_image
            self.labels = np.asarray(samples.train_color_label, dtype=np.int64)
            self.text_attr = "train_text_rgb"
        elif modality == "ir":
            self.store = samples.multiview_stores.get("ir")
            self.images = samples.train_thermal_image
            self.labels = np.asarray(samples.train_thermal_label, dtype=np.int64)
            self.text_attr = "train_text_ir"
        else:
            raise ValueError(f"Unsupported training modality: {modality}")

        if self.store is None:
            requested_views = [0] if views == "all" else list(views)
            if requested_views != [0]:
                raise ValueError(f"Array-backed {modality} training data has only view 0")
        else:
            count = int(samples.sysu_sr_views_per_image)
            requested_views = list(range(count)) if views == "all" else list(views)
            if not all(0 <= value < count for value in requested_views):
                raise ValueError(f"Requested {modality} view outside [0, {count - 1}]")
        self.index = [(sample, view) for sample in range(len(self.labels)) for view in requested_views]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, position: int) -> Dict[str, Any]:
        sample_index, view_index = self.index[position]
        if self.store is None:
            raw_image = self.images[sample_index]
        else:
            raw_image = self.store.image(sample_index, view_index)
        batch = {
            "img": self.transform(raw_image),
            "target": int(self.labels[sample_index]),
            "sample_id": f"train:{self.modality}:{sample_index:06d}:view{view_index:02d}",
        }
        if self.store is not None:
            caption = self.store.caption(sample_index, view_index)
            batch["text"] = self._tokenize(caption, self.samples.tokenizer)
        elif hasattr(self.samples, self.text_attr):
            batch["text"] = getattr(self.samples, self.text_attr)[sample_index]
        return batch


class SaltModelAdapter:
    def __init__(
        self,
        model_spec: Dict[str, Any],
        runtime: Dict[str, Any],
        work_root: Path,
        needs_train_data: bool = False,
    ):
        import torch
        from salt_vi.data.loader import Loader
        from salt_vi.engine import build_model
        from salt_vi.entrypoints.train import _initialize_spatial_backups, _load_compatible_state_dict
        from salt_vi.utils.utils import load_train_configs

        self.torch = torch
        self.model_spec = model_spec
        self.model_id = model_spec["id"]
        self.runtime = runtime
        self.checkpoint = str(Path(model_spec["checkpoint"]).resolve())
        self.config = load_train_configs(model_spec["config"])
        _apply_overrides(self.config, model_spec.get("overrides", {}))
        dataset = str(self.config.dataset).lower()
        if dataset not in PID_COUNTS:
            raise ValueError(f"Unsupported SALT dataset: {dataset}")
        self.config.pid_num = PID_COUNTS[dataset]
        # SALT's Loader materializes unique training arrays only in train mode.
        # This does not start optimization; it only controls which datasets are built.
        self.config.mode = "train" if needs_train_data else "test"
        self.config.DataParallel = False
        self.config.fixed_visual_data_parallel = False
        self.config.output_path = str(work_root / self.model_id)
        device_name = str(runtime["device"])
        if device_name.startswith("cuda:"):
            self.config.gpu_id = device_name.split(":", 1)[1]
        self.device = torch.device(device_name)

        self.loaders = Loader(self.config)
        self.model = build_model(self.config).to(self.device)
        # Inspect on CPU so a full training checkpoint never places optimizer and
        # RNG tensors on the analysis GPU merely to recover model weights.
        payload = torch.load(self.checkpoint, map_location="cpu")
        if isinstance(payload, dict) and "model" in payload:
            # Canonical full-state training checkpoints contain optimizer/RNG state;
            # feature analysis only restores their strict model snapshot.
            self.model.load_state_dict(payload["model"], strict=True)
            del payload
        else:
            state = payload
            if isinstance(payload, dict) and "model_state_dict" in payload:
                state = payload["model_state_dict"]
            elif isinstance(payload, dict) and "state_dict" in payload:
                state = payload["state_dict"]
            try:
                self.model.load_state_dict(state, strict=True)
                del state
                del payload
            except RuntimeError:
                # Historical checkpoints may require the same audited BN expansion
                # or positional interpolation supported by canonical training.
                del state
                del payload
                _load_compatible_state_dict(self.model, self.checkpoint, self.device)
        _initialize_spatial_backups(self.model, self.config)
        self.model.set_eval()
        self.backend = self.model.retrieval_backend

    def split_sources(self, split: str, split_config: Any) -> List[SplitSource]:
        if split == "query":
            return self._query_sources()
        if split == "gallery":
            return self._gallery_sources(split_config)
        if split in {"train_rgb", "train_ir"}:
            return self._train_sources(split, split_config)
        raise ValueError(f"Unsupported split: {split}")

    def _query_sources(self) -> List[SplitSource]:
        if hasattr(self.loaders, "query_loader"):
            loaders = [self.loaders.query_loader]
            labels_list = [np.asarray(self.loaders.query_label, dtype=np.int64)]
            cameras_list = [np.asarray(getattr(self.loaders, "query_cam", []), dtype=np.int64)]
        else:
            loaders = list(self.loaders.query_loaders)
            labels_list = [np.asarray(item, dtype=np.int64) for item in self.loaders.query_labels]
            cameras_list = [np.full(len(item), -1, dtype=np.int64) for item in labels_list]
        sources = []
        for trial, (loader, labels, cameras) in enumerate(zip(loaders, labels_list, cameras_list)):
            if len(cameras) != len(labels):
                cameras = np.full(len(labels), -1, dtype=np.int64)
            split_tag = "query" if len(loaders) == 1 else f"query_trial_{trial:02d}"
            ids = self._eval_sample_ids("query", trial, len(labels))
            sources.append(SplitSource("query", split_tag, loader, ids, labels, cameras))
        return sources

    def _gallery_sources(self, split_config: Dict[str, Any]) -> List[SplitSource]:
        loaders = list(self.loaders.gallery_loaders)
        trials = range(len(loaders)) if split_config["trials"] == "all" else split_config["trials"]
        sources = []
        for trial in trials:
            labels = np.asarray(self.loaders.gallery_labels[trial], dtype=np.int64)
            camera_values = getattr(self.loaders, "gallery_cams", None)
            cameras = (
                np.asarray(camera_values[trial], dtype=np.int64)
                if camera_values is not None
                else np.full(len(labels), -1, dtype=np.int64)
            )
            ids = self._eval_sample_ids("gallery", trial, len(labels))
            sources.append(
                SplitSource("gallery", f"gallery_trial_{trial:02d}", loaders[trial], ids, labels, cameras)
            )
        return sources

    def _train_sources(self, split: str, split_config: Dict[str, Any]) -> List[SplitSource]:
        from torch.utils.data import DataLoader

        modality = "rgb" if split == "train_rgb" else "ir"
        dataset = UniqueTrainDataset(
            self.loaders.samples,
            self.loaders.transform_test,
            modality,
            split_config["views"],
        )
        loader = DataLoader(
            dataset,
            batch_size=int(self.runtime["batch_size"]),
            shuffle=False,
            drop_last=False,
            num_workers=int(self.runtime["num_workers"]),
        )
        labels = np.asarray([dataset.labels[item[0]] for item in dataset.index], dtype=np.int64)
        ids = np.asarray(
            [f"train:{modality}:{item[0]:06d}:view{item[1]:02d}" for item in dataset.index],
            dtype=np.str_,
        )
        cameras = np.full(len(labels), -1, dtype=np.int64)
        return [SplitSource(split, split, loader, ids, labels, cameras)]

    def _eval_sample_ids(self, kind: str, trial: int, count: int) -> np.ndarray:
        if str(self.config.dataset).lower() == "sysu":
            from salt_vi.data.dataset import process_gallery_sysu, process_query_sysu

            if kind == "query":
                paths = process_query_sysu(self.config.sysu_data_path, mode=self.config.test_mode)[0]
            else:
                paths = process_gallery_sysu(
                    self.config.sysu_data_path,
                    mode=self.config.test_mode,
                    trial=trial,
                    gall_mode=self.config.gall_mode,
                )[0]
            root = Path(self.config.sysu_data_path).resolve()
            values = []
            for path in paths:
                resolved = Path(path).resolve()
                try:
                    values.append(str(resolved.relative_to(root)))
                except ValueError:
                    values.append(str(resolved))
            if len(values) == count:
                return np.asarray(values, dtype=np.str_)
        return np.asarray([f"{kind}:trial{trial:02d}:{index:06d}" for index in range(count)], dtype=np.str_)

    def encode(self, batch: Dict[str, Any], spec: Dict[str, Any]) -> Any:
        torch = self.torch
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if torch.is_tensor(value) else value
        encoder = spec["encoder"]
        if encoder == "protocol_query":
            result = self._encode_protocol_query(moved)
        elif encoder == "protocol_gallery":
            result = self._encode_protocol_gallery(moved)
        elif encoder == "image":
            result = self._encode_image(moved, spec)
        elif encoder == "text":
            result = self._encode_text(moved, spec)
        elif encoder == "fusion":
            result = self._encode_fusion(moved, spec)
        else:
            raise AssertionError(f"Unhandled encoder: {encoder}")
        if spec.get("normalize", False):
            result = torch.nn.functional.normalize(result, p=2, dim=1)
        return result.float()

    def _encode_protocol_query(self, batch: Dict[str, Any]) -> Any:
        if self.backend is not None:
            return self.backend.encode_query(self.model, batch, self.device)
        modality = str(self.config.test_modality)
        if "Fusion" in modality:
            self._require_text(batch, "legacy Fusion query")
            return self.model.classifier(self.model.encode_fusion(batch["text"].long(), batch["img"], "ir"), "Fusion")
        if "IR" in modality:
            return self._encode_image(batch, {"modality": "ir", "stage": "post_bn", "use_backup": bool(self.config.Fix_Visual)})
        if "Text" in modality:
            return self._encode_text(batch, {"stage": "post_bn"})
        raise ValueError(f"Cannot infer legacy query encoder from test_modality={modality!r}")

    def _encode_protocol_gallery(self, batch: Dict[str, Any]) -> Any:
        if self.backend is not None:
            return self.backend.encode_gallery(self.model, batch, self.device)
        return self._encode_image(batch, {"modality": "rgb", "stage": "post_bn", "use_backup": False})

    def _encode_image(self, batch: Dict[str, Any], spec: Dict[str, Any]) -> Any:
        from salt_vi.engine.test import _eval_image_feature

        image = batch["img"]
        modality = spec["modality"]
        visual = self.model.encode_image_featmap(image, modality)
        if spec.get("stage", "post_bn") == "pre_bn":
            return self.model.extract_global_feat(visual)
        return _eval_image_feature(
            self.model,
            visual,
            mode=modality.upper(),
            use_backup=bool(spec.get("use_backup", False)),
        )

    def _encode_text(self, batch: Dict[str, Any], spec: Dict[str, Any]) -> Any:
        self._require_text(batch, "text representation")
        features = self.model.encode_text_feat(batch["text"].long())
        if spec.get("stage", "post_bn") == "pre_bn":
            return features
        return self.model.classifier(features, "Text")

    def _encode_fusion(self, batch: Dict[str, Any], spec: Dict[str, Any]) -> Any:
        self._require_text(batch, "fusion representation")
        features = self.model.encode_fusion(batch["text"].long(), batch["img"], spec["modality"])
        if spec.get("stage", "post_bn") == "pre_bn":
            return features
        return self.model.classifier(features, "Fusion")

    @staticmethod
    def _require_text(batch: Dict[str, Any], context: str) -> None:
        if "text" not in batch:
            raise KeyError(f"{context} requires captions, but this split/config did not load text")
