import torch
from salt_vi.utils import MultiItemAverageMeter


def handle_nonfinite_gradients(scaler, optimizer, nonfinite_gradients):
    """Handle an AMP overflow without letting a disabled scaler update weights."""
    if not scaler.is_enabled():
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            "Non-finite gradients encountered without an active AMP scaler; "
            f"first non-finite gradient: {nonfinite_gradients[0]}"
        )
    previous_scale = float(scaler.get_scale())
    # GradScaler recorded the overflow during unscale_.  With AMP enabled,
    # step() skips the optimizer update and update() reduces the scale.
    scaler.step(optimizer)
    scaler.update()
    return previous_scale, float(scaler.get_scale())


def train(base, loaders, scaler, config, optimizer, current_epoch=None, ema=None):
    base.set_train()
    base.configure_qbn_running_stats(current_epoch)
    meter = MultiItemAverageMeter()
    loader = loaders.get_train_loader()
    consecutive_amp_overflows = 0
    amp_skipped_steps = 0
    max_consecutive_amp_overflows = int(getattr(config, "max_consecutive_amp_overflows", 8))
    if config.pretrain_choice in ["RN50", "RN50_ORI"]:
        mode = "1/3"
    elif config.pretrain_choice in ["ViT-B/16", "PMT_VIT"]:
        mode = None
    else: 
        raise ValueError(f"Pretrain model {config.pretrain_choice} choice not supported")

    # for i, (input1_0, input1_1, input2, input3, label1, label2) in enumerate(loader):
    for i, batch_dict in enumerate(loader):
        # data preparing
        # rgb_imgs0, rgb_imgs1, rgb_pids = input1_0, input1_1, label1
        # ir_imgs, ir_pids = input2, label2
        # text = input3
        # rgb_imgs0, rgb_imgs1, rgb_pids = rgb_imgs0.to(base.device), \
        #                                 rgb_imgs1.to(base.device),\
        #                                 rgb_pids.to(base.device).long()
        # ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()
        # text = text.to(base.device).long()

        # data preparing
        batch_dict = {key: value.to(base.device) for key, value in batch_dict.items()}
        # 清空所有梯度
        optimizer.zero_grad()

        # feature and loss computing. Prefer torch.amp on modern PyTorch while
        # retaining a warning-free fallback for the validated legacy runtime.
        amp_enabled = base.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_context = torch.amp.autocast("cuda", enabled=amp_enabled)
        else:
            autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled)
        with autocast_context:

            # get loss
            ret = base(batch_dict, mode, current_epoch=current_epoch)
            losses = [value for key, value in ret.items() if 'loss' in key]
            if not losses:
                raise RuntimeError("Training step produced no loss tensors")
            for key, value in ret.items():
                if 'loss' in key and not torch.isfinite(value).all():
                    raise FloatingPointError(f"Non-finite loss {key}: {value}")
            total_loss = sum(losses)
        
        # backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        gradients_seen = 0
        nonfinite_gradients = []
        for name, parameter in base.named_parameters():
            if parameter.grad is None:
                continue
            gradients_seen += 1
            if not torch.isfinite(parameter.grad).all():
                nonfinite_gradients.append(name)
        if gradients_seen == 0:
            raise RuntimeError("No gradients were produced for any trainable parameter")
        if nonfinite_gradients:
            previous_scale, current_scale = handle_nonfinite_gradients(
                scaler, optimizer, nonfinite_gradients
            )
            consecutive_amp_overflows += 1
            amp_skipped_steps += 1
            print(
                "AMP overflow: skipped optimizer step "
                f"({consecutive_amp_overflows}/{max_consecutive_amp_overflows}); "
                f"scale {previous_scale:g} -> {current_scale:g}; "
                f"first non-finite gradient: {nonfinite_gradients[0]}"
            )
            if consecutive_amp_overflows >= max_consecutive_amp_overflows:
                raise FloatingPointError(
                    "AMP gradient overflow persisted for "
                    f"{consecutive_amp_overflows} consecutive steps"
                )
            continue
        consecutive_amp_overflows = 0
        clip_norm = float(getattr(config, "gradient_clip_norm", 0.0))
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(base)

        # update meter
        acc_sign = False
        acc_value = 0
        for key, value in ret.items():
            if "loss" in key:
                meter.update({key: value})
            if 'acc' in key:
                acc_sign = True
                acc_value = value
                # meter.update({key: value})
        meter.update({'total_loss': total_loss})
        if acc_sign:
            meter.update({'acc': acc_value})

    meter.update({'amp_skipped_steps': float(amp_skipped_steps)})

    return meter.get_val(), meter.get_str()







