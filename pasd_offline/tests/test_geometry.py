import numpy as np
from PIL import Image

from pasd_offline.geometry import PersonDetection, prepare_control_image


def test_full_frame_narrow_image_uses_blurred_background():
    pixels = np.zeros((200, 50, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(50, dtype=np.uint8)
    source = Image.fromarray(pixels)
    control, metadata = prepare_control_image(
        source, PersonDetection((0, 0, 50, 200), 0.0, "full_frame")
    )
    assert control.size == (256, 512)
    assert metadata["mode"] == "person_fit_blurred_background"
    assert sum(metadata["padding"]) > 0
    assert metadata["foreground_box"] == [64, 0, 192, 512]


def test_detected_person_still_preserves_complete_source():
    source = Image.new("RGB", (100, 300), "gray")
    control, metadata = prepare_control_image(
        source, PersonDetection((20, 100, 80, 200), 0.9, "test")
    )
    assert control.size == (256, 512)
    assert metadata["mode"] == "person_fit_blurred_background"
    assert sum(metadata["padding"]) > 0
