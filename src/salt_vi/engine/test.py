import numpy as np
import torch

from salt_vi.retrieval import get_retrieval_protocol
from salt_vi.utils import eval_llcm, eval_regdb, eval_sysu


def _eval_image_feature(base, visual_output, mode="RGB", use_backup=False):
    if use_backup and base._uses_spatial_map_visual():
        feat = base.backup_pool(visual_output).flatten(1)
        return base.backup_classifier(feat)
    return base.classifier(base.extract_global_feat(visual_output), mode)


def test(base, loader, config, device):
    protocol = get_retrieval_protocol(getattr(config, "retrieval_backend", "identity_text"))
    return protocol.evaluate(base, loader, config, device)


def _append(features, name, value):
    features.setdefault(name, []).append(value.detach().cpu().numpy())


def _concatenate(features):
    return {name: np.concatenate(values, axis=0) for name, values in features.items()}


def _extract_query(base, query_loader, modalities, config, device):
    features = {}
    with torch.no_grad():
        for batch in query_loader:
            image = batch["img"].to(device)
            text = (
                batch["text"].to(device).long()
                if "Fusion" in modalities or "Text" in modalities
                else None
            )
            if "IR" in modalities:
                visual = base.encode_image_featmap(image, "ir")
                _append(
                    features,
                    "IR",
                    _eval_image_feature(
                        base, visual, mode="IR", use_backup=config.Fix_Visual
                    ),
                )
            if "Fusion" in modalities:
                if config.Feat_Filter:
                    text_filter = batch["text_filter"].to(device).long()
                    fusion = base.encode_filtered_fusion(text, text_filter, image)
                else:
                    fusion = base.encode_fusion(text, image, "ir")
                _append(features, "Fusion", base.classifier(fusion, "Fusion"))
            if "Text" in modalities or ("Fusion" in modalities and config.CAT_EVAL):
                text_feature = base.classifier(base.encode_text_feat(text), "Text")
                if "Text" in modalities:
                    _append(features, "Text", text_feature)
                if "Fusion" in modalities and config.CAT_EVAL:
                    _append(features, "Fusion_Text", text_feature)
            if "Fusion" in modalities and config.CAT_EVAL:
                image_feature = base.classifier(base.encode_image_feat(image, "ir"), "IR")
                _append(features, "Fusion_IR", image_feature)
    return _concatenate(features)


def _extract_gallery(base, gallery_loader, modalities, config, device):
    features = {}
    with torch.no_grad():
        for batch in gallery_loader:
            image = batch["img"].to(device)
            visual = base.encode_image_featmap(image, "rgb")
            _append(features, "RGB", _eval_image_feature(base, visual))
            if "IR" in modalities and config.Fix_Visual:
                _append(
                    features,
                    "IR_RGB",
                    _eval_image_feature(base, visual, use_backup=True),
                )
    return _concatenate(features)


def _similarity(mode, query, gallery, config, reverse):
    query_parts = [query[mode]]
    if mode == "Fusion" and config.CAT_EVAL:
        query_parts.extend((query["Fusion_IR"], query["Fusion_Text"]))
    gallery_feature = (
        gallery["IR_RGB"]
        if mode == "IR" and config.Fix_Visual
        else gallery["RGB"]
    )
    if reverse:
        return sum(gallery_feature @ feature.T for feature in query_parts)
    return sum(feature @ gallery_feature.T for feature in query_parts)


def _trial_metric(
    dataset,
    mode,
    query,
    gallery,
    query_label,
    gallery_label,
    query_cam,
    gallery_cam,
    config,
):
    reverse = dataset == "regdb" and config.regdb_test_mode != "t-v"
    distance = -_similarity(mode, query, gallery, config, reverse)
    if dataset == "regdb":
        labels = (
            (gallery_label, query_label) if reverse else (query_label, gallery_label)
        )
        return eval_regdb(distance, *labels)
    metric = eval_sysu if dataset == "sysu" else eval_llcm
    return metric(distance, query_label, gallery_label, query_cam, gallery_cam)


def _average_metrics(trials):
    return (
        float(np.mean([trial[2] for trial in trials])),
        float(np.mean([trial[1] for trial in trials])),
        np.mean([trial[0] for trial in trials], axis=0),
    )


def evaluate_identity_text(base, loader, config, device):
    dataset = loader.dataset
    if dataset not in ("sysu", "regdb", "llcm"):
        raise ValueError(f"Invalid dataset: {dataset}")
    modalities = tuple(
        name for name in ("IR", "Fusion", "Text") if name in config.test_modality
    )
    if not modalities:
        raise ValueError(f"Invalid test modality: {config.test_modality}")

    base.set_eval()
    print("Extracting Query Feature...")
    print("Test Mode: ", config.test_modality)
    shared_query = None
    if dataset != "regdb":
        shared_query = _extract_query(
            base, loader.query_loader, modalities, config, device
        )

    trial_metrics = {mode: [] for mode in modalities}
    print("Extracting Gallery Feature...")
    for trial, gallery_loader in enumerate(loader.gallery_loaders):
        if dataset == "regdb":
            query = _extract_query(
                base, loader.query_loaders[trial], modalities, config, device
            )
            query_label = loader.query_labels[trial]
            query_cam = gallery_cam = None
        else:
            query = shared_query
            query_label = loader.query_label
            query_cam = loader.query_cam
            gallery_cam = loader.gallery_cams[trial]
        gallery_label = loader.gallery_labels[trial]
        gallery = _extract_gallery(base, gallery_loader, modalities, config, device)
        for mode in modalities:
            trial_metrics[mode].append(
                _trial_metric(
                    dataset,
                    mode,
                    query,
                    gallery,
                    query_label,
                    gallery_label,
                    query_cam,
                    gallery_cam,
                    config,
                )
            )
    return {mode: _average_metrics(trials) for mode, trials in trial_metrics.items()}
