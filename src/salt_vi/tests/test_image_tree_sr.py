from pathlib import Path

import numpy as np
from PIL import Image

from salt_vi.data.dataset import _build_sysu_visual_source, _sysu_eval_image_path


def _write_source(root: Path, relative: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 32), "gray").save(path)


def _write_x4(root: Path, relative: str, color):
    path = (root / relative).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 128), color).save(path)
    return path


def test_image_tree_train_source_uses_canonical_sysu_order(tmp_path):
    source = tmp_path / "SYSU-MM01"
    (source / "exp").mkdir(parents=True)
    (source / "exp" / "train_id.txt").write_text("1", encoding="utf-8")
    (source / "exp" / "val_id.txt").write_text("", encoding="utf-8")
    _write_source(source, "cam1/0001/0001.jpg")
    _write_source(source, "cam3/0001/0001.jpg")

    x4 = tmp_path / "x4" / "sysu"
    _write_x4(x4, "cam1/0001/0001.jpg", "red")
    _write_x4(x4, "cam3/0001/0001.jpg", "blue")
    visual = _build_sysu_visual_source(
        str(source), str(x4), {"rgb", "ir"}, "image_tree", None, 1,
        "rgb", np.asarray([0]),
    )
    image, view = visual.sample(0)
    assert image.shape == (128, 64, 3)
    assert image[0, 0].tolist() == [255, 0, 0]
    assert view is None


def test_image_tree_eval_replaces_source_suffix_with_png(tmp_path):
    source = tmp_path / "SYSU-MM01"
    original = source / "cam6/0001/0001.jpg"
    _write_source(source, "cam6/0001/0001.jpg")
    x4 = tmp_path / "x4" / "sysu"
    expected = _write_x4(x4, "cam6/0001/0001.jpg", "blue")
    actual = _sysu_eval_image_path(
        str(original), str(source), str(x4), {"ir"}, "ir",
        sr_backend="image_tree",
    )
    assert Path(actual) == expected
