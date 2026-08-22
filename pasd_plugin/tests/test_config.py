from pathlib import Path

from pasd_plugin.config import PluginConfig


def test_from_yaml_expands_environment_paths(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    pretrained = tmp_path / "stable-diffusion-v1-5"
    pasd = tmp_path / "pasd-checkpoint"
    dataset.mkdir()
    pretrained.mkdir()
    pasd.mkdir()
    rgb_caption = tmp_path / "rgb.json"
    ir_caption = tmp_path / "ir.json"
    rgb_caption.write_text("{}", encoding="utf-8")
    ir_caption.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PASD_TEST_ROOT", str(tmp_path))

    config_path = tmp_path / "pasd.yaml"
    config_path.write_text(
        """dataset: sysu
dataset_root: ${PASD_TEST_ROOT}/dataset
captions:
  rgb: ${PASD_TEST_ROOT}/rgb.json
  ir: ${PASD_TEST_ROOT}/ir.json
output_root: ${PASD_TEST_ROOT}/output
pretrained_model_path: ${PASD_TEST_ROOT}/stable-diffusion-v1-5
pasd_model_path: ${PASD_TEST_ROOT}/pasd-checkpoint
geometry_mode: direct_rewrite
""",
        encoding="utf-8",
    )

    config = PluginConfig.from_yaml(config_path)

    assert config.dataset_root == dataset
    assert config.captions == {"rgb": rgb_caption, "ir": ir_caption}
    assert config.output_root == tmp_path / "output"
    assert config.pretrained_model_path == pretrained
    assert config.pasd_model_path == pasd
    assert config.geometry_mode == "direct_rewrite"
