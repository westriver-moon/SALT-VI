from pathlib import Path
import sys


SALT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SALT_ROOT / "src"))

from salt_vi.imagination import default_plugin_root, load_imagination_plugin
from salt_vi.utils.utils import load_train_configs


def test_mainline_bridge_resolves_both_versions(monkeypatch):
    monkeypatch.setenv("SALT_VI_ROOT", str(SALT_ROOT))
    monkeypatch.setenv("SALT_QRI_DATA_ROOT", "/home/cgv841/datasets/SYSU-MM01")
    monkeypatch.setenv("SALT_QRI_MODEL_ROOT", "/home/lab929/ybj/models")
    monkeypatch.setenv("SALT_QRI_RUNTIME_ROOT", "/home/lab929/ybj/models/qri-v1")
    monkeypatch.setenv("SALT_SWINIR_ROOT", "/home/cgv841/third_party/SwinIR-official-6545850-v2")
    monkeypatch.setenv("SALT_SWINIR_MODEL", "/home/cgv841/weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth")
    monkeypatch.setenv(
        "SALT_QRI_IDENTITY_CHECKPOINT",
        "/home/lab929/ybj/experiments/archive/stage_a/SALT-VI-safe-tricks-pairs-c1-c3-20260818/c3_camera_diverse_cosine/sysu/Base/Baseline_train[RGB_IR]_pmt_recipe/models/model_IR_19.pth",
    )
    monkeypatch.setenv("QRI_OUTPUT_ROOT", "/home/lab929/ybj/experiments/qri-v1/runtime")
    monkeypatch.setenv("QRI_V2_OUTPUT_ROOT", "/home/lab929/ybj/experiments/qri-v2/runtime")
    assert default_plugin_root() == SALT_ROOT / "plugins" / "qwen_imagination"
    v1 = load_imagination_plugin("qri-v1")
    v2 = load_imagination_plugin("qri-v2")
    assert v1.config.plugin_id == "qwen-regional-imagination-v1"
    assert v2.config.plugin_id == "qwen-regional-imagination-v2"
    expected_dataset = Path("/home/cgv841/datasets/SYSU-MM01")
    expected_model = (
        Path("/home/lab929/ybj/models")
        / "qwen3.8-27b-gguf"
        / "Qwen3.8-27B-UD-Q4_K_XL.gguf"
    )
    expected_identity_config = (
        SALT_ROOT / "configs/stage_a/safe_tricks/c3_b96_camera_diverse_cosine.yaml"
    )
    for plugin in (v1, v2):
        assert plugin.config.dataset_root == expected_dataset
        assert plugin.config.assets["qwen_model"].path == expected_model
        assert Path(plugin.config.identity["config_path"]) == expected_identity_config


def test_stage_a_qri_configs_expand_environment_extends(monkeypatch, tmp_path):
    monkeypatch.setenv("SALT_VI_ROOT", str(SALT_ROOT))
    v1_data = tmp_path / "qri-v1"
    v2_data = tmp_path / "qri-v2"
    v1_manifest = v1_data / "manifests/manifest.uniform.jsonl"
    v2_manifest = v2_data / "manifests/manifest.uniform.jsonl"
    v1_experiments = tmp_path / "experiments-v1"
    v2_experiments = tmp_path / "experiments-v2"
    monkeypatch.setenv("SALT_QRI_DATA_ROOT", str(v1_data))
    monkeypatch.setenv("SALT_QRI_VIEW_MANIFEST", str(v1_manifest))
    monkeypatch.setenv("SALT_QRI_EXPERIMENT_ROOT", str(v1_experiments))
    monkeypatch.setenv("SALT_QRI_V2_DATA_ROOT", str(v2_data))
    monkeypatch.setenv("SALT_QRI_V2_VIEW_MANIFEST", str(v2_manifest))
    monkeypatch.setenv("SALT_QRI_V2_EXPERIMENT_ROOT", str(v2_experiments))

    config_root = default_plugin_root() / "configs/stage_a"
    v1 = load_train_configs(config_root / "qri-v1/c3_b96_qri_uniform.yaml")
    v2 = load_train_configs(config_root / "qri-v2/c3_b96_qri_v2_uniform.yaml")

    assert v1.sampler_type == "identity_camera_diverse"
    assert v2.sampler_type == "identity_camera_diverse"
    assert v1.sysu_sr_data_root == str(v1_data)
    assert v2.sysu_sr_data_root == str(v2_data)
    assert v1.sysu_sr_view_manifest == str(v1_manifest)
    assert v2.sysu_sr_view_manifest == str(v2_manifest)
    assert v1.output_root == str(v1_experiments / "c3_b96_qri_uniform")
    assert v2.output_root == str(v2_experiments / "c3_b96_qri_v2_uniform")
