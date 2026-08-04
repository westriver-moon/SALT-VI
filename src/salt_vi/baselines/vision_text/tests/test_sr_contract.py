from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from salt_vi.baselines.vision_text.config import load_config
from salt_vi.baselines.vision_text.config.defaults import Config
from salt_vi.baselines.vision_text.data.dataset import TestData as SYSUTestData
from salt_vi.baselines.vision_text.data.transforms import ExactScale, SourceTargetScale
from salt_vi.baselines.vision_text.model import build_pmt_model


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
CONFIG_ROOT = PROJECT_ROOT / "configs" / "super_resolution"


def tiny_model_config(*, chunk_size=0, checkpointing=False):
    return Config(
        {
            "data": Config({"height": 32, "width": 16}),
            "model": Config(
                {
                    "num_classes": 4,
                    "embed_dim": 32,
                    "patch_size": [8, 8],
                    "stride_size": [8, 8],
                    "depth": 2,
                    "num_heads": 4,
                    "mlp_ratio": 2.0,
                    "dropout": 0.0,
                    "attention_dropout": 0.0,
                    "drop_path": 0.0,
                    "backbone_chunk_size": chunk_size,
                    "gradient_checkpointing": checkpointing,
                }
            ),
        }
    )


def test_sr_configs_inherit_exact_stage_a_mbpatch_recipe():
    expected = {
        "sr_a0_original_256.yaml": ((256, 128), []),
        "sr_a1_bicubic_x2.yaml": ((512, 256), []),
        "sr_a2_swinir_rgb_x2.yaml": ((512, 256), ["rgb"]),
        "sr_a3_swinir_both_x2.yaml": ((512, 256), ["rgb", "ir"]),
    }
    for name, (size, modalities) in expected.items():
        config = load_config(CONFIG_ROOT / name)
        assert (config.data.height, config.data.width) == size
        assert config.data.source_height == 256
        assert config.data.source_width == 128
        assert config.data.sr_modalities == modalities
        assert config.data.batch_size_per_modality == 32
        assert config.data.num_pos == 4
        assert config.train.max_epoch == 24
        assert config.model.embed_dim == 768
        assert len(config.model.patch_embed.branches) == 2
        assert config.model.gradient_checkpointing is True
        assert config.test.training_trials == 10


def test_source_target_scale_materializes_pmt_lr_before_bicubic_x2():
    array = np.arange(40 * 20 * 3, dtype=np.uint16).reshape(40, 20, 3).astype(np.uint8)
    image = Image.fromarray(array)
    transform = SourceTargetScale(32, 16, 64, 32)
    observed = transform(image)
    expected = image.resize((16, 32), Image.BILINEAR).resize((32, 64), Image.BICUBIC)
    assert np.array_equal(np.asarray(observed), np.asarray(expected))


def test_exact_scale_rejects_wrong_derived_asset_size():
    with pytest.raises(ValueError, match="Derived SR image"):
        ExactScale(64, 32)(Image.new("RGB", (16, 32)))


def test_testdata_remaps_only_selected_modality_to_derived_tree(tmp_path):
    source_root = tmp_path / "source"
    source = source_root / "cam3" / "0001" / "a.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(source)
    sr_root = tmp_path / "sr"
    derived = sr_root / "eval" / source.relative_to(source_root)
    derived.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(4, 5, 6)).save(derived, format="PNG")
    dataset = SYSUTestData(
        [source],
        [1],
        source_root=source_root,
        sr_data_dir=sr_root,
        modality="ir",
    )
    assert dataset._image_path(source) == derived.resolve()


def test_backbone_chunking_preserves_global_bn_and_outputs():
    torch.manual_seed(3)
    full = build_pmt_model(tiny_model_config(chunk_size=0))
    chunked = build_pmt_model(tiny_model_config(chunk_size=2))
    chunked.load_state_dict(full.state_dict())
    full.train()
    chunked.train()
    inputs = torch.randn(8, 3, 32, 16)
    expected = full(inputs, return_dict=True)
    observed = chunked(inputs, return_dict=True)
    assert torch.allclose(observed["features"], expected["features"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(observed["logits"], expected["logits"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(chunked.bottleneck.running_mean, full.bottleneck.running_mean)


def test_gradient_checkpointed_backbone_produces_finite_gradients():
    model = build_pmt_model(tiny_model_config(chunk_size=2, checkpointing=True))
    model.train()
    output = model(torch.randn(8, 3, 32, 16), return_dict=True)
    output["logits"].sum().backward()
    gradients = [parameter.grad for parameter in model.base.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
