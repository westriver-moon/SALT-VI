import json

from PIL import Image

from salt_vi.data.dataset import _indexed_images


def test_core_dataset_reads_optional_person_asset_package(tmp_path):
    root = tmp_path / "person/regdb"
    image = root / "images/Visible/0001.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 64), "red").save(image)
    (root / "contract.json").write_text(
        json.dumps({"version": 2, "size_hw": [64, 32]})
    )
    row = {
        "source_key": "Visible/0001.bmp",
        "status": "complete",
        "image": "images/Visible/0001.png",
        "aliases": [],
    }
    (root / "images.jsonl").write_text(json.dumps(row) + "\n")

    source = _indexed_images(
        "/unused", ["Visible/0001.bmp"], "regdb", tmp_path / "person"
    )
    assert len(source) == 1
    assert source[0].shape == (64, 32, 3)
