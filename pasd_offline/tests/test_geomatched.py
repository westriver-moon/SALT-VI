import hashlib
import json
from pathlib import Path

from PIL import Image

from pasd_offline.geomatched import build_geomatched_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_ir_geometry_and_combined_dataset(tmp_path: Path):
    source = tmp_path / "SYSU-MM01"
    (source / "exp").mkdir(parents=True)
    (source / "exp" / "train_id.txt").write_text("1", encoding="utf-8")
    (source / "exp" / "val_id.txt").write_text("", encoding="utf-8")
    (source / "exp" / "test_id.txt").write_text("", encoding="utf-8")
    rgb_source = source / "cam1" / "0001" / "0001.jpg"
    ir_source = source / "cam3" / "0001" / "0001.jpg"
    rgb_source.parent.mkdir(parents=True)
    ir_source.parent.mkdir(parents=True)
    Image.new("RGB", (40, 100), "red").save(rgb_source)
    Image.new("L", (50, 200), 80).save(ir_source)

    rgb_root = tmp_path / "rgb"
    rgb_output = rgb_root / "images" / "cam1" / "0001" / "0001.png"
    rgb_output.parent.mkdir(parents=True)
    Image.new("RGB", (256, 512), "red").save(rgb_output)
    rgb_manifest = rgb_root / "manifest.jsonl"
    rgb_row = {
        "source_key": "cam1/0001/0001.jpg",
        "identity": "0001",
        "camera": 1,
        "modality": "rgb",
        "split": "train",
        "view_index": 0,
        "caption": "red pedestrian",
        "seed": 1,
        "output": "images/cam1/0001/0001.png",
        "output_sha256": _sha256(rgb_output),
        "output_size": [256, 512],
    }
    rgb_manifest.write_text(json.dumps(rgb_row) + "\n", encoding="utf-8")
    (rgb_root / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "view_count": 1,
                "manifest_jsonl_sha256": _sha256(rgb_manifest),
            }
        ),
        encoding="utf-8",
    )

    ir_root = tmp_path / "ir"
    combined_root = tmp_path / "combined"
    result = build_geomatched_dataset(
        source,
        rgb_root,
        ir_root,
        combined_root,
        enforce_official_counts=False,
    )

    assert result["ir"]["modalities"] == {"rgb": 0, "ir": 1}
    assert result["combined"]["modalities"] == {"rgb": 1, "ir": 1}
    ir_row = json.loads((ir_root / "manifest.jsonl").read_text().splitlines()[0])
    assert ir_row["semantic_generation"] is False
    assert ir_row["geometry"]["resized_size"] == [128, 512]
    assert ir_row["geometry"]["padding"] == [64, 0, 64, 0]
    with Image.open(ir_root / ir_row["output"]) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (256, 512)
