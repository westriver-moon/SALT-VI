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
    "configs/stage_b/vit_source_core_sysu_no_sff.yaml",
    "configs/stage_b/vit_source_core_sysu_sff.yaml",
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


def run_smoke(config_path):
    print(f"\n===== Smoke: {config_path} =====")
    config = load_train_configs(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    seed_torch(config.seed)
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")

    assert_pass("1. config can load", config is not None, config.config_select)
    assert_pass(
        "2. checkpoint resolved",
        bool(config.training_weight_init and os.path.isfile(config.training_weight_init)),
        config.training_weight_init,
    )

    loaders = Loader(config)
    train_loader = loaders.get_train_loader()
    batch_dict = next(iter(train_loader))
    assert_pass("3. loader can fetch batch", isinstance(batch_dict, dict), f"keys={sorted(batch_dict.keys())}")

    validate_rgb_ir_text_batch_dict(batch_dict)
    assert_pass("4. batch_dict has required fields", True, ",".join(sorted(batch_dict.keys())))

    model = build_model(config)
    load_fixed_visual_init(model, config)
    model = model.to(device)
    model.set_train()
    model.zero_grad(set_to_none=True)

    batch_device = {key: value.to(device) for key, value in batch_dict.items()}
    batch_size = batch_device["img_ir"].shape[0]
    mode = train_mode_for_visual(config.pretrain_choice)

    with torch.no_grad():
        visual_output = model.encode_image_featmap(
            torch.cat((batch_device["img_rgb_ori"], batch_device["img_rgb_aug"], batch_device["img_ir"]), dim=0),
            mode=mode,
        )
        rgb_visual = model._slice_visual_output(visual_output, 0, int(2 * batch_size))
        ir_visual = model._slice_visual_output(visual_output, int(2 * batch_size), None)
        rgb_feat = model.extract_global_feat(rgb_visual)
        ir_feat = model.extract_global_feat(ir_visual)
        text_feat = model.encode_text_feat(batch_device["text_rgb"])
        sff_feat = model.encode_filtered_fusion(
            batch_device["text_rgb"],
            batch_device["text_ir"],
            batch_device["img_ir"],
        )

    assert_pass("5. PMT_VIT visual feature shape correct", rgb_feat.shape == (batch_size * 2, config.prj_output_dim), f"rgb={tuple(rgb_feat.shape)}")
    assert_pass("5.1 PMT_VIT IR feature shape correct", ir_feat.shape == (batch_size, config.prj_output_dim), f"ir={tuple(ir_feat.shape)}")
    assert_pass("6. text feature shape correct", text_feat.shape == (batch_size, config.prj_output_dim), f"text={tuple(text_feat.shape)}")
    assert_pass("7. SFF feature shape correct", sff_feat.shape == ir_feat.shape, f"sff={tuple(sff_feat.shape)}")

    ret = model(batch_device, mode=mode, current_epoch=0)
    assert_pass("8. forward can run", isinstance(ret, dict), f"keys={sorted(ret.keys())}")

    id_loss = ret.get("id_loss")
    wrt_loss = ret.get("wrt_loss")
    total_loss = sum(value for key, value in ret.items() if "loss" in key)
    assert_finite("9. id loss is finite", id_loss)
    assert_finite("9.1 wrt loss is finite", wrt_loss)
    assert_finite("9.2 total loss is finite", total_loss)

    total_loss.backward()
    assert_pass("10. backward can run", True)

    visual_grads = []
    for name, param in model.base_model.visual.named_parameters():
        if param.grad is None:
            continue
        grad_norm = float(param.grad.detach().abs().sum().item())
        if grad_norm > 0:
            visual_grads.append((name, grad_norm))
    assert_pass("11. Fix_Visual keeps visual backbone gradient-free", len(visual_grads) == 0, str(visual_grads[:3]))

    print(f"Trainable/Frozen summary: {model.fix_visual_summary}")
    print(f"Resolved checkpoint: {config.training_weight_init}")


def main():
    for config_path in CONFIG_PATHS:
        run_smoke(str(REPO_ROOT / config_path))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
