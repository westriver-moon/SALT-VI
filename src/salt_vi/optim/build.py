import inspect

import torch
from torch.optim import _functional as optim_F

_ADAMW_HAS_MODERN_SIGNATURE = "maximize" in inspect.signature(optim_F.adamw).parameters

from .lr_scheduler import LRSchedulerWithWarmup


class AdamWSkipEmptyGrad(torch.optim.AdamW):
    """AdamW compatible with older PyTorch builds that fail on empty grad groups."""

    @torch.no_grad()
    def step(self, closure=None):
        # Modern PyTorch handles empty-gradient groups itself and also changed
        # the functional AdamW state/signature contract. Delegate to the
        # public optimizer implementation instead of calling the private
        # functional API with the legacy positional arguments below.
        if _ADAMW_HAS_MODERN_SIGNATURE:
            return super().step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                params_with_grad.append(param)
                if param.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                grads.append(param.grad)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state["step"] += 1
                state_steps.append(state["step"])

            if not params_with_grad:
                continue

            beta1, beta2 = group["betas"]
            optim_F.adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad,
                beta1,
                beta2,
                group["lr"],
                group["weight_decay"],
                group["eps"],
            )

        return loss


def pmt_visual_layer_id(name, depth):
    prefix = "base_model.visual.vit.blocks."
    if name.startswith(prefix):
        return int(name[len(prefix) :].split(".", 1)[0]) + 1
    if name.startswith((
        "base_model.visual.vit.patch_embed",
        "base_model.visual.vit.pos_embed",
        "base_model.visual.vit.cls_token",
    )):
        return 0
    if name.startswith((
        "base_model.visual.vit.norm",
        "base_model.visual.projection",
    )):
        return int(depth) + 1
    return None


def no_weight_decay_parameter(name, parameter):
    return parameter.ndim <= 1 or name.rsplit(".", 1)[-1] in {
        "pos_embed",
        "cls_token",
        "class_embedding",
        "positional_embedding",
    }


def build_optimizer(args, model):
    params = []
    visual_layer_decay = float(getattr(args, "visual_layer_decay", 1.0))
    visual_depth = int(getattr(args, "pmt_depth", 12))
    use_no_weight_decay = bool(getattr(args, "optimizer_no_weight_decay", False))

    print(f'Using {args.lr_factor} times learning rate for random init module ')

    def is_visual_param(name):
        return name.startswith("base_model.visual") or ".visual." in name

    def is_text_param(name):
        return (
            name.startswith("base_model.transformer")
            or name.startswith("base_model.token_embedding")
            or name.startswith("base_model.positional_embedding")
            or name.startswith("base_model.ln_final")
            or name.startswith("base_model.text_projection")
        )

    def is_pmt_backbone_param(name):
        return (
            name.startswith("base_model.visual.vit.patch_embed")
            or name.startswith("base_model.visual.vit.blocks")
        )
    
    for key, value in model.named_parameters():
        scheduled_visual = bool(
            hasattr(model, "is_scheduled_visual_parameter")
            and model.is_scheduled_visual_parameter(key)
        )
        if not value.requires_grad and not scheduled_visual:
            continue
        lr = args.lr_visual
        weight_decay = args.visual_weight_decay
        metric_boost_role = None
        
        if is_visual_param(key):
            lr = getattr(args, "visual_lr", args.lr_visual) if scheduled_visual else args.lr_visual
            if getattr(args, "pmt_recipe", False) and is_pmt_backbone_param(key):
                lr = args.lr_visual * getattr(args, "pmt_backbone_lr_factor", 0.5)
            if "bias" in key:
                lr = (getattr(args, "visual_lr", args.lr_visual) if scheduled_visual else args.lr_visual) * args.visual_bias_lr_factor
                weight_decay = args.visual_weight_decay_bias
                if getattr(args, "pmt_recipe", False) and is_pmt_backbone_param(key):
                    lr = args.lr_visual * getattr(args, "pmt_backbone_lr_factor", 0.5) * args.visual_bias_lr_factor
            if "cross" in key:
                # use large learning rate for random initialized cross modal module
                lr =  args.lr_visual * args.lr_factor # default 5.0

        elif is_text_param(key):
            lr = args.lr_txt
            weight_decay = args.text_weight_decay
            if "bias" in key:
                lr = args.lr_txt * args.text_bias_lr_factor
                weight_decay = args.text_weight_decay_bias
            if "cross" in key:
                lr =  args.lr_txt * args.lr_factor # default 5.0

        elif "classifier" in key or "mcm_head" in key or "mlm_head" in key:
                lr = args.lr_visual * args.classifier_lr_factor
        
        elif "bias" in key:
                lr = args.lr_visual * args.visual_bias_lr_factor
                weight_decay = args.visual_weight_decay_bias

        elif "cross" in key:
                # use large learning rate for random initialized cross modal module
                lr =  args.lr_visual * args.lr_factor # default 5.0

        layer_id = pmt_visual_layer_id(key, visual_depth) if is_visual_param(key) else None
        lr_scale = 1.0
        if layer_id is not None:
            lr_scale = visual_layer_decay ** (visual_depth + 1 - layer_id)
            lr *= lr_scale
        if use_no_weight_decay and no_weight_decay_parameter(key, value):
            weight_decay = 0.0

        group = {
            "params": [value],
            "lr": lr,
            "weight_decay": weight_decay,
            "layer_id": layer_id,
            "lr_scale": lr_scale,
        }
        if metric_boost_role is None:
            metric_boost_role = "scheduled_visual" if scheduled_visual else "standard"
        group["metric_boost_role"] = metric_boost_role
        if args.optimizer in ["Adam", "AdamW"]:
            group["betas"] = (args.alpha, args.beta)
            group["eps"] = 1e-8
        params.append(group)

    optimizer_lr = getattr(args, "lr", None)
    if optimizer_lr is None:
        optimizer_lr = args.lr_visual

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=optimizer_lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam": # default
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr_visual,
            weight_decay=args.visual_weight_decay,
            betas=(args.alpha, args.beta),
            eps=1e-8, # 1e-3 --> 1e-8 !!!
        )
    elif args.optimizer == "AdamW":
        optimizer = AdamWSkipEmptyGrad(
            params,
            lr=optimizer_lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer: {args.optimizer}")

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.total_train_epoch, # 120
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        target_lr_factor=getattr(args, "target_lr_factor", None),
        power=args.power,
    )
