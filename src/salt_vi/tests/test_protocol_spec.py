import random
from types import SimpleNamespace

import numpy as np

from salt_vi.data.dataset import process_gallery_sysu
from salt_vi.retrieval import build_protocol_spec, get_retrieval_protocol


def _config(**overrides):
    values = {
        "dataset": "sysu",
        "test_mode": "all",
        "gall_mode": "single",
        "gallery_trials": 10,
        "test_modality": "Fusion",
        "retrieval_backend": "legacy",
        "text_data_root": "/captions",
        "gallery_caption_manifest": None,
        "eval_num_regdb": 1,
        "regdb_test_mode": "t-v",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_legacy_protocol_records_identity_text_query():
    config = _config()
    spec = build_protocol_spec(
        config, get_retrieval_protocol(config.retrieval_backend)
    )
    assert spec.identifier == "all-search-single-shot-10-trial-legacy"
    assert spec.official
    assert spec.query_modalities == (
        "infrared-image",
        "visible-identity-caption",
    )
    assert spec.gallery_modalities == ("visible-image",)
    assert spec.caption_lookup == "query:identity"


def test_ir_to_rgb_text_protocol_records_gallery_caption():
    config = _config(
        retrieval_backend="ir_to_rgb_text",
        test_modality="IR-RGBText",
        gallery_caption_manifest="/captions/gallery.json",
        gallery_trials=3,
    )
    spec = build_protocol_spec(
        config, get_retrieval_protocol(config.retrieval_backend)
    )
    assert not spec.official
    assert spec.gallery_trials == 3
    assert spec.query_modalities == ("infrared-image",)
    assert spec.gallery_modalities == (
        "visible-image",
        "visible-image-caption",
    )
    assert spec.caption_lookup == "gallery:image"


def test_regdb_reverse_direction_swaps_effective_modalities():
    config = _config(
        dataset="regdb",
        regdb_test_mode="v-t",
        eval_num_regdb=3,
    )
    spec = build_protocol_spec(
        config, get_retrieval_protocol(config.retrieval_backend)
    )
    assert spec.direction == "visible-to-thermal"
    assert spec.gallery_trials == 3
    assert spec.query_modalities == ("visible-image",)
    assert spec.gallery_modalities == ("infrared-image", "visible-identity-caption")
    assert spec.caption_lookup == "gallery:identity"


def _make_sysu_gallery(root):
    (root / "exp").mkdir()
    (root / "exp" / "test_id.txt").write_text("1\n", encoding="utf-8")
    for camera_index, camera in enumerate(("cam1", "cam2"), start=1):
        image_dir = root / camera / "0001"
        image_dir.mkdir(parents=True)
        for index in range(10):
            (image_dir / "{}_0001_{:08d}".format(camera_index, index)).touch()


def test_sysu_gallery_sampling_does_not_mutate_global_rng(tmp_path):
    _make_sysu_gallery(tmp_path)

    random.seed(123)
    python_state = random.getstate()
    np.random.seed(456)
    numpy_state = np.random.get_state()
    first = process_gallery_sysu(
        str(tmp_path), mode="indoor", trial=4, gall_mode="multi"
    )[0]

    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]

    np.random.choice(100, 20, replace=False)
    second = process_gallery_sysu(
        str(tmp_path), mode="indoor", trial=4, gall_mode="multi"
    )[0]
    assert first == second
