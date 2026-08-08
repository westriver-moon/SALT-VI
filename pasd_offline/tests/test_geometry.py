import numpy as np
from PIL import Image

from pasd_offline.geometry import PersonDetection, prepare_control_image


def test_full_frame_narrow_image_uses_edge_padding():
    pixels = np.zeros((200, 50, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(50, dtype=np.uint8)
    source = Image.fromarray(pixels)
    control, metadata = prepare_control_image(
        source, PersonDetection((0, 0, 50, 200), 0.0, "full_frame")
    )
    assert control.size == (256, 512)
    assert metadata["mode"] == "person_fit_edge_pad"
    assert sum(metadata["padding"]) > 0


def test_background_can_be_cropped_when_person_is_safe():
    source = Image.new("RGB", (100, 300), "gray")
    control, metadata = prepare_control_image(
        source, PersonDetection((20, 100, 80, 200), 0.9, "test")
    )
    assert control.size == (256, 512)
    assert metadata["mode"] == "person_safe_cover_crop"
    assert metadata["padding"] == [0, 0, 0, 0]
