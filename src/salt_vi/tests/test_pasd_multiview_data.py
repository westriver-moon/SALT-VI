import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from salt_vi.config.validation import validate_runtime_config
from salt_vi.data.dataset import SYSU_Tri_Data
from salt_vi.data.pasd_multiview import PASDTrainViewStore, eval_view_path


def write_record(root: Path, output_root: Path, relative: str, modality: str, camera: int):
    source = root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 128), "gray").save(source)
    views = []
    for index in range(5):
        output = Path("images") / Path(relative).parent / Path(relative).stem / f"view_{index:02d}.png"
        path = output_root / output
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (256, 512), (index, index, index)).save(path)
        views.append({"view_index": index, "caption": f"caption {index}", "seed": index, "output": str(output)})
    return {
        "source_key": relative,
        "image": str(source),
        "identity": "0001",
        "camera": camera,
        "modality": modality,
        "split": "train",
        "views": views,
    }


def test_manifest_backed_train_and_eval_views(tmp_path: Path):
    root = tmp_path / "SYSU-MM01"
    (root / "exp").mkdir(parents=True)
    (root / "exp" / "train_id.txt").write_text("1", encoding="utf-8")
    (root / "exp" / "val_id.txt").write_text("", encoding="utf-8")
    output_root = tmp_path / "derived"
    records = [
        write_record(root, output_root, "cam1/0001/0001.jpg", "rgb", 1),
        write_record(root, output_root, "cam3/0001/0001.jpg", "ir", 3),
    ]
    manifest = output_root / "source-records.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    rgb = PASDTrainViewStore(root, output_root, manifest, "rgb", np.asarray([0]), 5)
    ir = PASDTrainViewStore(root, output_root, manifest, "ir", np.asarray([0]), 5)
    assert rgb.image(0, 4).shape == (512, 256, 3)
    assert ir.caption(0, 2) == "caption 2"
    assert eval_view_path(root / "cam1/0001/0001.jpg", root, output_root, manifest, 0, 5).endswith("view_00.png")


def test_multiview_store_uses_canonical_source_label_order(tmp_path: Path):
    root = tmp_path / "SYSU-MM01"
    (root / "exp").mkdir(parents=True)
    (root / "exp" / "train_id.txt").write_text("2,1", encoding="utf-8")
    (root / "exp" / "val_id.txt").write_text("", encoding="utf-8")
    output_root = tmp_path / "derived"
    records = [
        write_record(root, output_root, "cam1/0002/0002.jpg", "rgb", 1),
        write_record(root, output_root, "cam1/0001/0001.jpg", "rgb", 1),
    ]
    records[0]["identity"] = "0002"
    manifest = output_root / "source-records.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    store = PASDTrainViewStore(root, output_root, manifest, "rgb", np.asarray([0, 1]), 5)
    assert store.sources == ["cam1/0001/0001.jpg", "cam1/0002/0002.jpg"]
    with pytest.raises(ValueError, match="source/label order mismatch"):
        PASDTrainViewStore(root, output_root, manifest, "rgb", np.asarray([1, 0]), 5)


def test_rgb_only_multiview_keeps_ir_array_input(tmp_path: Path):
    root = tmp_path / "SYSU-MM01"
    (root / "exp").mkdir(parents=True)
    (root / "exp" / "train_id.txt").write_text("1", encoding="utf-8")
    (root / "exp" / "val_id.txt").write_text("", encoding="utf-8")
    np.save(root / "train_rgb_resized_label.npy", np.asarray([0]))
    np.save(root / "train_ir_resized_label.npy", np.asarray([0]))
    np.save(root / "train_ir_resized_img.npy", np.zeros((1, 64, 32, 3), dtype=np.uint8))
    output_root = tmp_path / "derived"
    record = write_record(root, output_root, "cam1/0001/0001.jpg", "rgb", 1)
    manifest = output_root / "source-records.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    transform = lambda image: np.asarray(image)
    dataset = SYSU_Tri_Data(
        str(root) + os.sep,
        transform1=transform,
        transform2=transform,
        transform3=transform,
        colorIndex=np.asarray([0]),
        thermalIndex=np.asarray([0]),
        joint_mode="uni",
        sysu_sr_data_root=str(output_root),
        sysu_sr_modalities=["rgb"],
        sysu_sr_backend="pasd_multiview",
        sysu_sr_view_manifest=str(manifest),
        sysu_sr_views_per_image=5,
        text_modalities=("rgb",),
    )
    batch = dataset[0]
    assert batch["img_rgb_ori"].shape == (512, 256, 3)
    assert batch["img_ir"].shape == (64, 32, 3)
    assert "text_rgb" in batch and "text_ir" not in batch


def test_multiview_config_contract():
    config = SimpleNamespace(
        training_mode="RGB_IR_Text",
        joint_mode="uni",
        uni_BN=False,
        loss_names="id",
        Fix_Visual=True,
        fixed_visual_data_parallel=False,
        visual_unfreeze_last_n_blocks=0,
        dataset="sysu",
        sysu_sr_backend="pasd_multiview",
        sysu_sr_modalities=["rgb"],
        sysu_sr_exact_size=True,
        sysu_sr_views_per_image=5,
        sysu_sr_view_manifest="records.jsonl",
        sysu_sr_view_sampling="independent",
        sysu_sr_eval_view_index=0,
        img_h=512,
        img_w=256,
        retrieval_backend="ir_to_rgb_text",
        test_modality="IR-RGBText",
        gallery_caption_manifest="captions.json",
        gallery_text_dropout=0.3,
        Feat_Filter=False,
    )
    assert validate_runtime_config(config) is config
