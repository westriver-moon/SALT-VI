from pathlib import Path
import sys


SALT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SALT_ROOT / "src"))

from salt_vi.imagination import default_plugin_root, load_imagination_plugin


def test_mainline_bridge_loads_both_versions(monkeypatch):
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
