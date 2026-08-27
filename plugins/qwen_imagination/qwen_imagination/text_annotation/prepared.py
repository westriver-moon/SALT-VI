"""Consume exact final images and their shared cached pose without inference."""

from person_preprocessing import PersonAssetStore

from ..regional.schema import SourceItem
from ..regional.roi import COCO_KEYPOINTS


class CachedPoseROIGenerator:
    """Adapt shared COCO-17 arrays to Qwen ROI generation."""

    def __init__(self, regions, store):
        self.generator = regions
        self.store = store

    def regions(self, image, modality, *, source_key=None):
        return self.regions_for_source(image, modality, source_key)

    def regions_for_source(self, image, modality, source_key):
        if source_key is None:
            raise ValueError("cached pose lookup requires source_key")
        prediction = self.store.pose(source_key, require_person=True)
        if tuple(prediction["size_hw"].tolist()) != (image.height, image.width):
            raise ValueError("cached pose and Qwen image dimensions differ")
        points = prediction["keypoints"]
        pose = {
            "bbox_xyxy": tuple(float(value) for value in prediction["bbox"]),
            "keypoints": {
                name: tuple(float(value) for value in points[index])
                for name, index in COCO_KEYPOINTS.items()
            },
        }
        return self.generator.regions(image, modality, pose_result=pose)


class PreparedReferenceStore:
    def __init__(self, config, store=None):
        self.store = store or PersonAssetStore(
            config.prepared_data_root, config.dataset
        )
        if self.store.size_hw != tuple(config.output_size_hw):
            raise ValueError("Qwen and prepared image dimensions differ")
        self.store.pose_contract()

    def image(self, source):
        return self.store.image(source.source_key)


def source_split(row, dataset, trial):
    indices = {ref["index"] for ref in row["references"]}
    if dataset == "sysu":
        train, evaluation = bool(indices & {"train", "val"}), "test" in indices
    elif dataset == "regdb":
        modality = "visible" if row["modality"] == "rgb" else "thermal"
        train = f"idx/train_{modality}_{trial}.txt" in indices
        evaluation = f"idx/test_{modality}_{trial}.txt" in indices
    elif dataset == "llcm":
        modality = "vis" if row["modality"] == "rgb" else "nir"
        train = f"idx/train_{modality}.txt" in indices
        evaluation = (
            f"idx/test_{modality}.txt" in indices
            or "test_camera_candidates" in indices
        )
    else:
        raise ValueError("unknown dataset: " + dataset)
    if train and evaluation:
        raise ValueError("source occurs in training and evaluation: " + row["source_key"])
    return "train" if train else "evaluation" if evaluation else None


def collect_prepared_sources(config, split):
    store = PreparedReferenceStore(config).store
    groups = {modality: [] for modality in config.modalities}
    for row in store.records:
        if row["modality"] not in groups:
            continue
        selected = source_split(row, config.dataset, config.trial)
        if selected is None or split not in {"all", selected}:
            continue
        path = store.root / row["image"]
        groups[row["modality"]].append(
            SourceItem(
                source_key=row["source_key"],
                image=path,
                identity=row["identity"],
                camera=row["camera"],
                modality=row["modality"],
                split=selected,
            )
        )
    return [
        rows[index]
        for index in range(max(map(len, groups.values()), default=0))
        for rows in groups.values()
        if index < len(rows)
    ]
