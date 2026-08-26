from PIL import Image
import pytest

from pasd_plugin.config import PluginConfig
from pasd_plugin.geometry import (
    PersonDetection,
    prepare_control_image,
    prepare_direct_rewrite,
    restore_blurred_background,
)
from pasd_plugin.validation import validate_geometry


def test_geometry_uniformly_scales_without_stretching() -> None:
    image = Image.new("RGB", (80, 200), "white")
    control, geometry = prepare_control_image(
        image,
        PersonDetection((0, 0, 80, 200), 0.0, "full_frame"),
        target_size=(256, 512),
    )
    assert control.size == (256, 512)
    width, height = geometry["resized_size"]
    assert abs(width - geometry["source_size"][0] * geometry["scale"]) <= 1
    assert abs(height - geometry["source_size"][1] * geometry["scale"]) <= 1


def test_restore_keeps_same_source_canvas_outside_foreground() -> None:
    geometry = {
        "foreground_box": [64, 0, 192, 512],
        "foreground_feather_radius": 0.0,
        "background_blur_radius": 24.0,
    }
    generated = Image.new("RGB", (256, 512), "red")
    source_canvas = Image.new("RGB", (256, 512), "blue")
    restored = restore_blurred_background(generated, geometry, source_canvas)
    assert restored.getpixel((0, 256)) == (0, 0, 255)
    assert restored.getpixel((128, 256)) == (255, 0, 0)


def test_direct_rewrite_preserves_identity_coordinates(tmp_path) -> None:
    image = Image.new("RGB", (256, 512), "white")
    control, geometry = prepare_direct_rewrite(image)
    config = PluginConfig(
        dataset="sysu",
        dataset_root=tmp_path,
        captions={"rgb": tmp_path / "rgb.json", "ir": tmp_path / "ir.json"},
        output_root=tmp_path / "output",
        pretrained_model_path=tmp_path / "sd",
        pasd_model_path=tmp_path / "pasd",
        geometry_mode="direct_rewrite",
    )

    assert control.size == (256, 512)
    assert geometry["source_size"] == [256, 512]
    assert geometry["target_size"] == [256, 512]
    assert geometry["transform"] == {
        "scale_x": 1.0,
        "scale_y": 1.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
    }
    validate_geometry(geometry, config)
    with pytest.raises(ValueError, match="requires source size"):
        prepare_direct_rewrite(Image.new("RGB", (128, 256), "white"))
