from types import SimpleNamespace

import numpy as np
import torch

from salt_feature_analysis.salt_adapter import UniqueTrainDataset


def test_unique_array_train_dataset_visits_every_sample_once():
    samples = SimpleNamespace(
        multiview_stores={},
        train_color_image=np.zeros((3, 4, 2, 3), dtype=np.uint8),
        train_color_label=np.asarray([4, 5, 6]),
        train_thermal_image=np.zeros((2, 4, 2, 3), dtype=np.uint8),
        train_thermal_label=np.asarray([4, 5]),
        sysu_sr_views_per_image=1,
    )
    transform = lambda image: torch.from_numpy(image.copy()).permute(2, 0, 1)
    dataset = UniqueTrainDataset(samples, transform, "rgb", [0])
    assert len(dataset) == 3
    assert [dataset[index]["target"] for index in range(3)] == [4, 5, 6]
    assert dataset[2]["sample_id"] == "train:rgb:000002:view00"
    assert "text" not in dataset[0]

