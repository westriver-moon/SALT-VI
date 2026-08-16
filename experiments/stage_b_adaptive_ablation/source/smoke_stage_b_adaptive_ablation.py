import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salt_vi.engine import build_model
from salt_vi.data.loader import Loader, validate_rgb_ir_text_batch_dict
from salt_vi.utils.utils import load_train_configs


CONFIG_PATHS = (
    "configs/stage_b/adaptive_no_sff/b1_scalar_alpha.yaml",
    "configs/stage_b/adaptive_no_sff/b2_sample_gate.yaml",
    "configs/stage_b/adaptive_no_sff/b3_channel_gate.yaml",
    "configs/stage_b/adaptive_no_sff/b4_residual_gate.yaml",
    "configs/stage_b/adaptive_no_sff/b5_norm_residual_gate.yaml",
)


def seed_torch(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_mode_for_visual(pretrain_choice):
    if pretrain_choice in ["RN50", "RN50_ORI"]:
        return "1/3"
    if pretrain_choice in ["ViT-B/16", "PMT_VIT"]:
        return None
    raise ValueError(f"Unsupported pretrain_choice: {pretrain_choice}")


def assert_pass(name, condition, detail=""):
    if not condition:
        raise AssertionError(f"[FAIL] {name}: {detail}".strip())
    suffix = f" | {detail}" if detail else ""
    print(f"[PASS] {name}{suffix}")


def assert_finite(name, tensor):
    assert_pass(name, torch.is_tensor(tensor) and torch.isfinite(tensor).all().item(), f"value={tensor}")


def load_fixed_visual_init(model, config):
    state_dict = torch.load(config.training_weight_init, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)


def adaptive_named_params(model):
    return [(name, param) for name, param in model.named_parameters() if "adaptive_" in name]


def run_smoke(config_path):
    print(f"\n===== Adaptive Smoke: {config_path} =====")
    config = load_train_configs(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    seed_torch(config.seed)
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")

    assert_pass("1. config can load", config is not None, config.config_select)
    assert_pass("2. fusion_way is adaptive_add", getattr(config, "fusion_way", None) == "adaptive_add", config.fusion_way)
    assert_pass("3. Feat_Filter stays disabled", getattr(config, "Feat_Filter", None) is False, str(config.Feat_Filter))
    assert_pass(
        "4. checkpoint resolved",
        bool(config.training_weight_init and os.path.isfile(config.training_weight_init)),
        config.training_weight_init,
    )

    loaders = Loader(config)
    train_loader = loaders.get_train_loader()
    batch_dict = next(iter(train_loader))
    validate_rgb_ir_text_batch_dict(batch_dict)
    assert_pass("5. batch_dict has required fields", True, ",".join(sorted(batch_dict.keys())))

    model = build_model(config)
    load_fixed_visual_init(model, config)
    model = model.to(device)
    model.set_train()
    model.zero_grad(set_to_none=True)

    adaptive_params = adaptive_named_params(model)
    assert_pass("6. adaptive params exist", len(adaptive_params) > 0, ",".join(name for name, _ in adaptive_params))
    assert_pass(
        "7. adaptive params are trainable",
        all(param.requires_grad for _, param in adaptive_params),
        ",".join(name for name, param in adaptive_params if not param.requires_grad),
    )

    batch_device = {key: value.to(device) for key, value in batch_dict.items()}
    mode = train_mode_for_visual(config.pretrain_choice)
    ret = model(batch_device, mode=mode, current_epoch=0)
    assert_pass("8. forward can run", isinstance(ret, dict), f"keys={sorted(ret.keys())}")

    id_loss = ret.get("id_loss")
    wrt_loss = ret.get("wrt_loss")
    total_loss = sum(value for key, value in ret.items() if "loss" in key)
    assert_finite("9. id loss is finite", id_loss)
    assert_finite("10. wrt loss is finite", wrt_loss)
    assert_finite("11. total loss is finite", total_loss)

    total_loss.backward()
    assert_pass("12. backward can run", True)

    missing_grads = [name for name, param in adaptive_params if param.grad is None]
    zero_grads = [
        name
        for name, param in adaptive_params
        if param.grad is not None and float(param.grad.detach().abs().sum().item()) == 0.0
    ]
    assert_pass("13. adaptive params have gradients", not missing_grads and not zero_grads, f"missing={missing_grads} zero={zero_grads}")

    visual_grads = []
    for name, param in model.base_model.visual.named_parameters():
        if param.grad is None:
            continue
        grad_norm = float(param.grad.detach().abs().sum().item())
        if grad_norm > 0:
            visual_grads.append((name, grad_norm))
    assert_pass("14. Fix_Visual keeps visual backbone gradient-free", len(visual_grads) == 0, str(visual_grads[:3]))

    grad_summary = {name: float(param.grad.detach().abs().sum().item()) for name, param in adaptive_params}
    print(f"Adaptive grad summary: {grad_summary}")
    print(f"Trainable/Frozen summary: {model.fix_visual_summary}")


def main():
    for config_path in CONFIG_PATHS:
        run_smoke(str(REPO_ROOT / config_path))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\nAll adaptive Stage B smoke tests passed.")


if __name__ == "__main__":
    main()
