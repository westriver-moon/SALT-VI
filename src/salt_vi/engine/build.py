import os
import math
import re
from copy import deepcopy
from salt_vi.utils import os_walk
import salt_vi.models.clip_model.objectives as objectives
from salt_vi.models.clip_model.clip_model import LayerNorm, build_CLIP_from_openai_pretrained
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import parallel_apply
from salt_vi.data.loader import validate_rgb_ir_text_batch_dict
from salt_vi.config.validation import validate_runtime_config
from salt_vi.retrieval import get_retrieval_backend
from salt_vi.utils import (
    TripletLoss_WRT,
    kl_align_loss,
    TripletLoss_WRT_local,
    L_i2t,
    L_t2i,
    PMTTripletLoss,
    CrossModalPMTTripletLoss,
    PMTMSEL,
    PMTDCL,
    LabelSmoothingCrossEntropy,
)


CROSS_MODAL_PAIR_NAMES = (
    "RGB-IR",
    "RGB-Fusion",
    "RGB-Text",
    "IR-Fusion",
    "IR-Text",
    "Fusion-Text",
)

IMAGE_TEXT_FUSION_WAYS = frozenset({
    "norm_add",
    "add",
    "parameter_add",
    "adaptive_add",
    "cross_attention",
})
DUAL_TEXT_FUSION_WAYS = frozenset({
    "norm_add",
    "add",
    "cross_attention",
    "attention_rgb_text",
    "attention_ir_text",
    "global_attention",
    "global_attention_rgb_text",
    "global_attention_ir_text",
})
FUSION_WAYS = IMAGE_TEXT_FUSION_WAYS | DUAL_TEXT_FUSION_WAYS
JOINT_MODE_FUSION_WAYS = {
    "uni": IMAGE_TEXT_FUSION_WAYS,
    "ir_crossfusion": IMAGE_TEXT_FUSION_WAYS,
    "ir_selffusion": IMAGE_TEXT_FUSION_WAYS,
    "rgb_selffusion": IMAGE_TEXT_FUSION_WAYS,
    "dual_text": DUAL_TEXT_FUSION_WAYS,
}


def validate_fusion_compatibility(training_mode, joint_mode, fusion_way):
    if fusion_way not in FUSION_WAYS:
        raise ValueError(
            f"Unsupported fusion_way {fusion_way!r}; expected one of {sorted(FUSION_WAYS)}"
        )
    if training_mode != "RGB_IR_Text":
        return
    if joint_mode not in JOINT_MODE_FUSION_WAYS:
        raise ValueError(
            f"Unsupported joint_mode {joint_mode!r}; expected one of "
            f"{sorted(JOINT_MODE_FUSION_WAYS)}"
        )
    allowed = JOINT_MODE_FUSION_WAYS[joint_mode]
    if fusion_way not in allowed:
        raise ValueError(
            f"fusion_way {fusion_way!r} is incompatible with joint_mode "
            f"{joint_mode!r}; expected one of {sorted(allowed)}"
        )


def _checkpoint_epoch(filename, mode, family=None):
    """Parse only supported model checkpoint names; never use substring matches."""
    escaped_mode = re.escape(mode)
    patterns = {
        "legacy": rf"^model_{escaped_mode}_(-?\d+)\.pth$",
        "metric": rf"^model_{escaped_mode}_epoch_(-?\d+)\.pth$",
    }
    families = (family,) if family else ("metric", "legacy")
    for candidate_family in families:
        match = re.fullmatch(patterns[candidate_family], filename)
        if match:
            return int(match.group(1))
    return None


class _FixedVisualEncoder(nn.Module):
    """Unregistered adapter used by the frozen-visual multi-GPU executor."""

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, images, mode=None):
        if hasattr(self.visual, "input_dtype"):
            dtype = self.visual.input_dtype
        elif hasattr(self.visual, "conv1"):
            dtype = self.visual.conv1.weight.dtype
        else:
            dtype = next(self.visual.parameters()).dtype
        images = images.to(dtype=dtype)
        if mode is None:
            return self.visual(images)
        return self.visual(images, mode)


def resolve_cross_modal_pair_weights(raw_weights=None):
    """Validate and canonicalize the six unordered modality-pair weights."""
    if raw_weights is None:
        return {name: 1.0 for name in CROSS_MODAL_PAIR_NAMES}
    if not isinstance(raw_weights, dict):
        raise TypeError("cross_modal_pair_weights must be a YAML mapping")
    unknown = sorted(set(raw_weights) - set(CROSS_MODAL_PAIR_NAMES))
    missing = sorted(set(CROSS_MODAL_PAIR_NAMES) - set(raw_weights))
    if unknown or missing:
        raise ValueError(
            f"cross_modal_pair_weights keys mismatch; missing={missing}, unknown={unknown}"
        )
    result = {}
    for name in CROSS_MODAL_PAIR_NAMES:
        value = float(raw_weights[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid cross-modal pair weight {name}={value!r}")
        result[name] = value
    if sum(result.values()) <= 0.0:
        raise ValueError("cross_modal_pair_weights must have a positive total")
    return result


def weighted_cross_modal_pair_loss(pair_losses, raw_weights=None):
    """Return a scale-preserving weighted mean; all-one defaults preserve H1."""
    weights = resolve_cross_modal_pair_weights(raw_weights)
    missing = sorted(set(CROSS_MODAL_PAIR_NAMES) - set(pair_losses))
    unknown = sorted(set(pair_losses) - set(CROSS_MODAL_PAIR_NAMES))
    if missing or unknown:
        raise ValueError(f"pair loss keys mismatch; missing={missing}, unknown={unknown}")
    ordered = torch.stack([pair_losses[name] for name in CROSS_MODAL_PAIR_NAMES])
    if all(value == 1.0 for value in weights.values()):
        return ordered.mean()
    weight_tensor = ordered.new_tensor([weights[name] for name in CROSS_MODAL_PAIR_NAMES])
    return (ordered * weight_tensor).sum() / weight_tensor.sum()


def configure_qbn_running_stats(classifier, current_epoch, freeze_epoch):
    """Freeze only QBN running statistics while leaving affine parameters trainable."""
    if not getattr(classifier, "uni_BN", False) or freeze_epoch is None:
        return False
    freeze_epoch = int(freeze_epoch)
    if freeze_epoch < 0:
        return False
    frozen = int(current_epoch) >= freeze_epoch
    for branch in ("RGB", "IR", "Fusion", "Text"):
        bn = getattr(classifier, f"BN_{branch}")
        bn.train(not frozen)
        bn.weight.requires_grad_(True)
        bn.bias.requires_grad_(True)
    return frozen


class Normalize(nn.Module):
    def __init__(self, power=2, eps=1e-12):
        super(Normalize, self).__init__()
        self.power = power
        self.eps = float(eps)

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power).clamp_min(self.eps)
        out = x.div(norm)
        return out


class PatchGeM(nn.Module):
    """Signed power mean for transformer tokens whose channels cross zero."""

    def __init__(self, p=3.0, eps=1e-6, learnable=True):
        super().__init__()
        value = torch.tensor(float(p))
        if learnable:
            self.p = nn.Parameter(value)
        else:
            self.register_buffer("p", value)
        self.eps = float(eps)

    def forward(self, patch_tokens):
        if patch_tokens.ndim != 3 or patch_tokens.shape[1] < 1:
            raise ValueError(f"PatchGeM expects [B,N,D] patch tokens, got {tuple(patch_tokens.shape)}")
        p = self.p.clamp(min=1e-3) if torch.is_tensor(self.p) else self.p
        signed_power = patch_tokens.sign() * patch_tokens.abs().clamp_min(self.eps).pow(p)
        pooled_power = signed_power.mean(dim=1)
        return pooled_power.sign() * pooled_power.abs().clamp_min(self.eps).pow(1.0 / p)


def probability_to_logit(value):
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("learnable pa initialization must be strictly between 0 and 1")
    return math.log(value / (1.0 - value))

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


_GLOBAL_FEAT_KEYS = (
    "global_feat",
    "feat",
    "feats",
    "features",
    "cls",
    "embedding",
    "embeddings",
    "tokens",
)


def extract_global_feat(output):
    if torch.is_tensor(output):
        if output.ndim == 2:
            return output.float()
        if output.ndim == 3:
            return output[:, 0, :].float()
        raise TypeError(f"Expected visual tensor with 2 or 3 dims, got shape {tuple(output.shape)}")
    if isinstance(output, dict):
        for key in _GLOBAL_FEAT_KEYS:
            value = output.get(key)
            if torch.is_tensor(value):
                return extract_global_feat(value)
        raise KeyError(
            f"Unable to extract global feature from dict output. Available keys: {sorted(output.keys())}"
        )
    raise TypeError(f"Unsupported visual output type: {type(output)!r}")


def extract_text_token_feat(text_map, caption_ids):
    return text_map[torch.arange(text_map.shape[0]), caption_ids.argmax(dim=-1)].float()


def ensure_matching_feature_shape(**named_tensors):
    shape_map = {name: tuple(tensor.shape) for name, tensor in named_tensors.items()}
    if len(set(shape_map.values())) != 1:
        print(f"SFF shape mismatch: {shape_map}")
        raise ValueError(f"SFF requires matching feature shapes, got {shape_map}")


def build_multihead_attention(embed_dim, num_heads):
    try:
        attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        attn._expects_batch_first = True
    except TypeError:
        attn = nn.MultiheadAttention(embed_dim, num_heads)
        attn._expects_batch_first = False
    return attn


def build_adaptive_gate(input_dim, hidden_dim, output_dim, dropout):
    gate = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_dim, output_dim),
    )
    gate.apply(weights_init_kaiming)
    return gate


class Classifier(nn.Module):
    def __init__(self, pid_num, dim=512, Return_B4_BN=False, uni_BN=False, joint_mode='uni',modal='RGB,IR,Text,Fusion'):
        super(Classifier, self, ).__init__()
        self.pid_num = pid_num
        # self.GAP = GeneralizedMeanPoolingP()
        self.Return_B4_BN = Return_B4_BN
        self.modal = modal
        self.uni_BN = uni_BN
        self.joint_mode = joint_mode
        if uni_BN:
            assert joint_mode == 'uni'
            if joint_mode == 'uni':
                self.BN_RGB = nn.BatchNorm1d(dim)
                self.BN_RGB.apply(weights_init_kaiming)
                self.BN_IR = nn.BatchNorm1d(dim)
                self.BN_IR.apply(weights_init_kaiming)
                self.BN_Fusion = nn.BatchNorm1d(dim)
                self.BN_Fusion.apply(weights_init_kaiming)
                self.BN_Text = nn.BatchNorm1d(dim)
                self.BN_Text.apply(weights_init_kaiming)
        else:
            self.BN = nn.BatchNorm1d(dim)
            self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(dim, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

    def forward(self, features, mode="RGB"): # IR, Fusion, Text, RGB
        # features = self.GAP(features_map)
        bn_input = features.flatten(1) if features.ndim > 1 else features.unsqueeze(0)
        if self.uni_BN:
            if self.training:
                len_feat = len(bn_input)
                if len_feat % 5 != 0:
                    raise ValueError(
                        "uni_BN training expects five equally sized modality groups; "
                        f"received {len_feat} features"
                    )
                b = len_feat // 5
                rgb_features = self.BN_RGB(bn_input[:2*b])
                ir_features = self.BN_IR(bn_input[2*b:3*b])
                fusion_features = self.BN_Fusion(bn_input[3*b:4*b])
                text_features = self.BN_Text(bn_input[4*b:5*b])
                bn_features = torch.cat((rgb_features, ir_features, fusion_features, text_features),dim=0)

            else:
                if mode == 'RGB':
                    bn_features = self.BN_RGB(bn_input)
                elif mode == 'IR':
                    bn_features = self.BN_IR(bn_input)
                elif mode == 'Fusion':
                    bn_features = self.BN_Fusion(bn_input)
                elif mode == 'Text':
                    bn_features = self.BN_Text(bn_input)
                else:
                    raise ValueError("mode must be in ['IR', 'Fusion', 'Text', 'RGB']")
        else:
            bn_features = self.BN(bn_input)

        cls_score = self.classifier(bn_features)

        if self.training:
            return features, cls_score
        else:
            # if self.Return_B4_BN:
            #     return features
            return self.l2_norm(bn_features)


class FM_cat(nn.Module):
    def __init__(self,in_channels):
        super(FM_cat, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels)
        )
        nn.init.normal_(self.W[1].weight.data, 1.0, 0.01)
        nn.init.zeros_(self.W[1].bias.data)


        # self.bottleneck = nn.BatchNorm1d(in_channels)
        # self.bottleneck.bias.requires_grad_(False)  # no shift

        # nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        # nn.init.zeros_(self.bottleneck.bias.data)

    def forward(self,f):

        f = f.view(f.size(0),f.size(1),1,1)
        f = self.W(f)
        f = f.view(f.size(0),-1)
        # f = self.bottleneck(f+feat)

        return f

class CLIP2ReID(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        validate_runtime_config(args)
        validate_fusion_compatibility(args.training_mode, args.joint_mode, args.fusion_way)
        self.retrieval_backend = get_retrieval_backend(
            getattr(args, "retrieval_backend", "legacy")
        )
        self.max_save_model_num = args.max_save_model_num
        self.output_path = args.output_path
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')
        self._init_device()

        self.num_classes = num_classes

        self._set_task()

        # self.Return_B4_BN = args.Return_B4_BN
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice,
            args.img_size,
            args.stride_size,
            download_root=getattr(self.args, "clip_download_root", "~/.cache/clip"),
            prj_output_dim=self.args.prj_output_dim,
            pooling=self.args.pooling,
            pmt_pretrained=getattr(self.args, "pmt_pretrained", None),
            pmt_patch_size=getattr(self.args, "pmt_patch_size", (16, 16)),
            pmt_stride_size=getattr(self.args, "pmt_stride_size", (12, 12)),
            pmt_embed_dim=getattr(self.args, "pmt_embed_dim", 768),
            pmt_depth=getattr(self.args, "pmt_depth", 12),
            pmt_num_heads=getattr(self.args, "pmt_num_heads", 12),
            pmt_mlp_ratio=getattr(self.args, "pmt_mlp_ratio", 4.0),
            pmt_dropout=getattr(self.args, "pmt_dropout", 0.03),
            pmt_attention_dropout=getattr(self.args, "pmt_attention_dropout", 0.0),
            pmt_drop_path_rate=getattr(self.args, "pmt_drop_path_rate", 0.1),
            pmt_patch_embed=getattr(self.args, "pmt_patch_embed", None),
        )
        self.embed_dim = base_cfg['embed_dim']
        if args.pretrain_choice == 'RN50':
            # 复制conv1...的权重到conv1_...
            print("copy conv1 weight to conv1_")
            self.base_model.visual.conv1_.load_state_dict(self.base_model.visual.conv1.state_dict())
            # self.base_model.visual.conv2_.load_state_dict(self.base_model.visual.conv2.state_dict())
            # self.base_model.visual.conv3_.load_state_dict(self.base_model.visual.conv3.state_dict())


        temperature = float(args.temperature)
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be finite and > 0; got {args.temperature!r}")
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))  # 0.07
        # self.logit_scale = torch.ones([]) * np.log(1 / args.temperature)  # 0.07
        if getattr(args, "freeze_text_in_image_only", False) and args.training_mode == "RGB_IR":
            self.freeze_text_encoder_for_image_only()

        if 'attention' in args.fusion_way:
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)
        if 'global' in args.fusion_way:
            self.global_attn_s = build_multihead_attention(
                self.embed_dim,
                self.embed_dim // 64,
            )
            self.global_attn_t = build_multihead_attention(
                self.embed_dim,
                self.embed_dim // 64,
            )
             # init cross attn
            scale = 512**-0.5
            proj_std = scale * ((2 * 4)**-0.5)
            attn_std = scale
            nn.init.normal_(self.global_attn_s.in_proj_weight, std=attn_std)
            nn.init.normal_(self.global_attn_s.out_proj.weight, std=proj_std)
             # init cross attn
            nn.init.normal_(self.global_attn_t.in_proj_weight, std=attn_std)
            nn.init.normal_(self.global_attn_t.out_proj.weight, std=proj_std)
        if 'attention' in args.fusion_way and 'global' not in args.fusion_way:
            self.cross_attn = build_multihead_attention(
                self.embed_dim,
                self.embed_dim // 64,
            )
            scale = self.embed_dim ** -0.5
            proj_std = scale * ((2 * args.cmt_depth)**-0.5)
            attn_std = scale
            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        # Loss definition
        self.classifier = Classifier(self.num_classes,self.embed_dim,args.Return_B4_BN,args.uni_BN,args.joint_mode)
        self.pid_criterion = LabelSmoothingCrossEntropy(getattr(args, "label_smoothing", 0.0))
        self.tri_criterion = TripletLoss_WRT()
        self.wrt_local = TripletLoss_WRT_local()
        self.pmt_tri_criterion = PMTTripletLoss(
            margin=getattr(args, "pmt_triplet_margin", 0.1),
            feat_norm="no",
        )
        self.cross_modal_tri_criterion = CrossModalPMTTripletLoss(
            margin=getattr(args, "pmt_triplet_margin", 0.1),
            feat_norm="no",
        )
        self.pmt_msel_criterion = PMTMSEL(getattr(args, "num_pos", 4), feat_norm="no")
        self.pmt_dcl_criterion = PMTDCL(getattr(args, "num_pos", 4), feat_norm="no")
        self.adaptive_alpha = None
        self.adaptive_gate = None
        self.raw_pa = None
        self.patch_gem = None
        self.patch_pool_gamma = None
        self._configure_adaptive_fusion()
        self._configure_learnable_pa()
        self._configure_patch_pooling()
        self._visual_unfrozen = False
        self._visual_unfreeze_announced = False
        self.visual_unfreeze_summary = None
        self.fix_visual_summary = None
        self._qbn_freeze_announced = False
        self._metric_checkpoint_paths = {}
        # Keep these plain Python containers unregistered so replica weights never
        # enter state_dict(), named_parameters(), or the optimizer.
        self._fixed_visual_parallel_enabled = False
        self._fixed_visual_parallel_devices = ()
        self._fixed_visual_parallel_replicas = []
        self._configure_fix_visual_training()

    def _init_device(self):
        self.device = torch.device(
            'cuda:{}'.format(self.args.gpu_id) if torch.cuda.is_available() else 'cpu')
        print('Model is using device: {}'.format(self.device))

    def set_train(self):
        self.train()
        if getattr(self.args, "Fix_Visual", False):
            self._set_scheduled_visual_mode()
        self.training = True

    def set_eval(self):
        self.eval()
        for replica in self._fixed_visual_parallel_replicas:
            replica.eval()
        self.training = False

    def configure_fixed_visual_data_parallel(self):
        """Create persistent read-only visual replicas after the primary model is placed."""
        enabled = bool(getattr(self.args, "fixed_visual_data_parallel", False))
        if not enabled:
            return None
        if bool(getattr(self.args, "DataParallel", False)):
            raise RuntimeError("fixed_visual_data_parallel cannot be combined with legacy DataParallel")
        if not getattr(self.args, "Fix_Visual", False):
            raise RuntimeError("fixed_visual_data_parallel requires Fix_Visual=true")
        if int(getattr(self.args, "visual_unfreeze_last_n_blocks", 0)) != 0:
            raise RuntimeError("fixed_visual_data_parallel forbids visual unfreezing")
        if not torch.cuda.is_available():
            raise RuntimeError("fixed_visual_data_parallel requires CUDA")

        raw_devices = getattr(self.args, "fixed_visual_device_ids", None)
        if not isinstance(raw_devices, (list, tuple)) or len(raw_devices) < 2:
            raise RuntimeError("fixed_visual_device_ids must contain at least two logical CUDA devices")
        devices = tuple(int(value) for value in raw_devices)
        if len(set(devices)) != len(devices):
            raise RuntimeError(f"fixed_visual_device_ids contains duplicates: {devices}")
        visible_count = torch.cuda.device_count()
        if any(value < 0 or value >= visible_count for value in devices):
            raise RuntimeError(
                f"fixed_visual_device_ids {devices} exceed visible CUDA device count {visible_count}"
            )
        primary = int(self.device.index or 0)
        if devices[0] != primary or primary not in devices:
            raise RuntimeError(
                f"Primary logical device cuda:{primary} must be first in fixed_visual_device_ids {devices}"
            )
        if any(parameter.requires_grad for parameter in self.base_model.visual.parameters()):
            raise RuntimeError("Visual parameters must all be frozen before replica creation")

        primary_adapter = _FixedVisualEncoder(self.base_model.visual).eval()
        replicas = [primary_adapter]
        for device_id in devices[1:]:
            visual_copy = deepcopy(self.base_model.visual).to(torch.device(f"cuda:{device_id}"))
            visual_copy.eval()
            for parameter in visual_copy.parameters():
                parameter.requires_grad_(False)
            replicas.append(_FixedVisualEncoder(visual_copy).eval())
        self._fixed_visual_parallel_devices = devices
        self._fixed_visual_parallel_replicas = replicas
        self._fixed_visual_parallel_enabled = True
        summary = {
            "strategy": "frozen_visual_chunk_data_parallel",
            "logical_devices": list(devices),
            "chunk_size": int(getattr(self.args, "visual_forward_chunk_size", 0)),
            "global_batch_preserved": True,
            "replicas_registered": False,
        }
        print(f"Configured frozen visual data parallel: {summary}")
        return summary

    def configure_qbn_running_stats(self, current_epoch):
        freeze_epoch = getattr(self.args, "qbn_freeze_running_stats_epoch", None)
        frozen = configure_qbn_running_stats(self.classifier, current_epoch, freeze_epoch)
        if frozen and not self._qbn_freeze_announced:
            print(
                "Freezing QBN running statistics from zero-based epoch "
                f"{int(freeze_epoch)}; affine weight/bias remain trainable"
            )
            self._qbn_freeze_announced = True
        return frozen

    def freeze_text_encoder_for_image_only(self):
        text_modules = [
            self.base_model.transformer,
            self.base_model.token_embedding,
            self.base_model.ln_final,
        ]
        for module in text_modules:
            for param in module.parameters():
                param.requires_grad_(False)
        self.base_model.positional_embedding.requires_grad_(False)
        self.base_model.text_projection.requires_grad_(False)

    def _configure_adaptive_fusion(self):
        if getattr(self.args, "fusion_way", "") != "adaptive_add":
            return

        adaptive_type = getattr(self.args, "adaptive_fusion_type", "scalar_alpha")
        if adaptive_type == "scalar_alpha":
            alpha_init = float(getattr(self.args, "adaptive_alpha_init", 1.0))
            self.adaptive_alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
            return

        supported_types = {
            "sample_gate",
            "channel_gate",
            "residual_gate",
            "norm_residual_gate",
        }
        if adaptive_type not in supported_types:
            raise ValueError(f"Unsupported adaptive_fusion_type: {adaptive_type}")

        hidden_dim = int(getattr(self.args, "adaptive_gate_hidden_dim", 256))
        if hidden_dim < 1:
            raise ValueError(f"adaptive_gate_hidden_dim must be positive, got {hidden_dim}")
        dropout = float(getattr(self.args, "adaptive_gate_dropout", 0.1))
        output_dim = 1 if adaptive_type == "sample_gate" else self.embed_dim
        self.adaptive_gate = build_adaptive_gate(self.embed_dim * 3, hidden_dim, output_dim, dropout)

    def _configure_learnable_pa(self):
        if not bool(getattr(self.args, "learnable_pa", False)):
            return
        self.raw_pa = nn.Parameter(
            torch.tensor(probability_to_logit(getattr(self.args, "pa_init", 0.5)), dtype=torch.float32)
        )

    def current_pa(self):
        if self.raw_pa is None:
            return float(self.args.pa)
        return torch.sigmoid(self.raw_pa)

    def _configure_patch_pooling(self):
        mode = str(getattr(self.args, "visual_pooling", "cls"))
        supported = {"cls", "mean_patch", "gem_patch", "cls_gem"}
        if mode not in supported:
            raise ValueError(f"Unsupported visual_pooling {mode!r}; expected {sorted(supported)}")
        if mode in {"gem_patch", "cls_gem"}:
            variant = str(getattr(self.args, "patch_gem_variant", "signed_power_mean"))
            if variant != "signed_power_mean":
                raise ValueError(
                    "PatchGeM supports only patch_gem_variant='signed_power_mean'; "
                    "positive clamping is invalid for signed transformer tokens"
                )
            self.patch_gem = PatchGeM(
                p=float(getattr(self.args, "patch_gem_p", 3.0)),
                learnable=bool(getattr(self.args, "patch_gem_learnable", True)),
            )
        if mode == "cls_gem":
            self.patch_pool_gamma = nn.Parameter(
                torch.tensor(float(getattr(self.args, "patch_pool_gamma_init", 0.0)))
            )

    def _set_module_trainable(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def _configure_fix_visual_training(self):
        if not getattr(self.args, "Fix_Visual", False):
            return

        if self._is_pmt_visual() and "Text" in self.args.training_mode:
            for param in self.parameters():
                param.requires_grad_(False)

            self._set_module_trainable(self.base_model.transformer, True)
            self._set_module_trainable(self.base_model.token_embedding, True)
            self._set_module_trainable(self.base_model.ln_final, True)
            self.base_model.positional_embedding.requires_grad_(True)
            self.base_model.text_projection.requires_grad_(True)
            self._set_module_trainable(self.classifier, True)

            for module_name in (
                "ln_pre_t",
                "ln_pre_i",
                "ln_post",
                "global_attn_s",
                "global_attn_t",
                "cross_attn",
            ):
                if hasattr(self, module_name):
                    self._set_module_trainable(getattr(self, module_name), True)
            if self.adaptive_gate is not None:
                self._set_module_trainable(self.adaptive_gate, True)
            if self.adaptive_alpha is not None:
                self.adaptive_alpha.requires_grad_(True)
            if self.raw_pa is not None:
                self.raw_pa.requires_grad_(True)
            if self.patch_gem is not None:
                self._set_module_trainable(self.patch_gem, True)
            if self.patch_pool_gamma is not None:
                self.patch_pool_gamma.requires_grad_(True)
        else:
            self._set_module_trainable(self.base_model.visual, False)

        self.logit_scale.requires_grad_(False)
        self._record_fix_visual_summary()

    def _scheduled_visual_named_parameters(self):
        count = int(getattr(self.args, "visual_unfreeze_last_n_blocks", 0))
        if count <= 0 or not self._is_pmt_visual():
            return []
        visual = self.base_model.visual
        blocks = visual.vit.blocks
        if count > len(blocks):
            raise ValueError(f"Cannot unfreeze {count} blocks; PMT-ViT has {len(blocks)}")
        selected = []
        start = len(blocks) - count
        for index in range(start, len(blocks)):
            for name, parameter in blocks[index].named_parameters():
                selected.append((f"base_model.visual.vit.blocks.{index}.{name}", parameter))
        for name, parameter in visual.vit.norm.named_parameters():
            selected.append((f"base_model.visual.vit.norm.{name}", parameter))
        for name, parameter in visual.projection.named_parameters():
            selected.append((f"base_model.visual.projection.{name}", parameter))
        return selected

    def is_scheduled_visual_parameter(self, name):
        return any(candidate == name for candidate, _ in self._scheduled_visual_named_parameters())

    def _set_scheduled_visual_mode(self):
        self.base_model.visual.eval()
        if not self._visual_unfrozen:
            return
        count = int(getattr(self.args, "visual_unfreeze_last_n_blocks", 0))
        for block in self.base_model.visual.vit.blocks[-count:]:
            block.train()
        self.base_model.visual.vit.norm.train()
        self.base_model.visual.projection.train()

    def configure_epoch_trainability(self, current_epoch):
        selected = self._scheduled_visual_named_parameters()
        if not selected:
            return None
        start_epoch = int(getattr(self.args, "visual_unfreeze_start_epoch", 3))
        should_unfreeze = int(current_epoch) >= start_epoch
        for _, parameter in selected:
            parameter.requires_grad_(should_unfreeze)
        self._visual_unfrozen = should_unfreeze
        self._set_scheduled_visual_mode()
        if should_unfreeze and not self._visual_unfreeze_announced:
            names = [name for name, _ in selected]
            count = sum(parameter.numel() for _, parameter in selected)
            self.visual_unfreeze_summary = {
                "epoch": int(current_epoch),
                "trainable_names": names,
                "new_trainable_parameter_count": int(count),
                "patch_embedding_trainable": False,
                "positional_embedding_trainable": False,
                "class_token_trainable": False,
                "final_norm_trainable": True,
                "projection_trainable": True,
                "classifier_trainable": any(
                    parameter.requires_grad for parameter in self.classifier.parameters()
                ),
            }
            print(f"Unfreezing PMT-ViT at epoch {current_epoch}: {count} parameters")
            for name in names:
                print(f"  {name}")
            self._visual_unfreeze_announced = True
        return self.visual_unfreeze_summary

    def _record_fix_visual_summary(self):
        trainable_names = [name for name, param in self.named_parameters() if param.requires_grad]
        frozen_visual_param_count = sum(
            param.numel()
            for _, param in self.base_model.visual.named_parameters()
            if not param.requires_grad
        )
        trainable_param_count = sum(param.numel() for _, param in self.named_parameters() if param.requires_grad)
        self.fix_visual_summary = {
            "trainable_names": trainable_names,
            "frozen_visual_param_count": frozen_visual_param_count,
            "trainable_param_count": trainable_param_count,
        }

        print("Trainable parameter names:")
        for name in trainable_names:
            print(f"  {name}")
        print(f"Frozen visual parameter count: {frozen_visual_param_count}")
        print(f"Trainable parameter count: {trainable_param_count}")

    def _is_pmt_visual(self):
        return self.args.pretrain_choice == "PMT_VIT"

    def _uses_token_visual(self):
        return self.args.pretrain_choice in ["ViT-B/16", "PMT_VIT"]

    def _uses_spatial_map_visual(self):
        return "RN" in self.args.pretrain_choice

    def _pmt_recipe_enabled(self):
        return bool(getattr(self.args, "pmt_recipe", False))

    def _assert_pmt_batch_layout(self, label_visible, label_ir):
        if not bool(getattr(self.args, "pmt_assert_batch_layout", True)):
            return
        if label_visible.shape != label_ir.shape:
            raise ValueError(
                f"PMT recipe expects aligned visible/IR label shapes, got "
                f"{tuple(label_visible.shape)} and {tuple(label_ir.shape)}"
            )
        if not torch.equal(label_visible, label_ir):
            raise ValueError("PMT recipe requires aligned visible and IR labels in each batch")
        num_pos = int(getattr(self.args, "num_pos", 4))
        if label_visible.numel() % num_pos != 0:
            raise ValueError(
                f"PMT recipe batch size {label_visible.numel()} must be divisible by num_pos={num_pos}"
            )
        chunks = label_visible.view(-1, num_pos)
        if not torch.all(chunks.eq(chunks[:, :1])):
            raise ValueError("PMT recipe requires each num_pos chunk to contain one identity")

    def _slice_visual_output(self, visual_output, start, end):
        if isinstance(visual_output, dict):
            return {
                key: value[start:end] if torch.is_tensor(value) else value
                for key, value in visual_output.items()
            }
        return visual_output[start:end]

    def _concat_visual_outputs(self, outputs):
        if not outputs:
            raise ValueError("Cannot concatenate an empty visual output list")
        first = outputs[0]
        if torch.is_tensor(first):
            return torch.cat(outputs, dim=0)
        if isinstance(first, dict):
            return {
                key: torch.cat([output[key] for output in outputs], dim=0)
                if torch.is_tensor(first[key])
                else first[key]
                for key in first
            }
        raise TypeError(f"Unsupported chunked visual output type: {type(first)!r}")

    def _encode_fixed_visual(self, images, mode):
        chunk_size = int(getattr(self.args, "visual_forward_chunk_size", 0))
        with torch.no_grad():
            if chunk_size <= 0 or images.shape[0] <= chunk_size:
                return self.base_model.encode_image(images, mode)
            if bool(getattr(self, "_fixed_visual_parallel_enabled", False)):
                chunks = [
                    images[start : start + chunk_size]
                    for start in range(0, images.shape[0], chunk_size)
                ]
                ordered_outputs = []
                replica_count = len(self._fixed_visual_parallel_replicas)
                for wave_start in range(0, len(chunks), replica_count):
                    wave = chunks[wave_start : wave_start + replica_count]
                    replicas = self._fixed_visual_parallel_replicas[: len(wave)]
                    devices = self._fixed_visual_parallel_devices[: len(wave)]
                    inputs = [
                        (chunk.to(torch.device(f"cuda:{device_id}"), non_blocking=True), mode)
                        for chunk, device_id in zip(wave, devices)
                    ]
                    outputs = parallel_apply(
                        replicas,
                        inputs,
                        devices=list(devices),
                    )
                    ordered_outputs.extend(
                        self._move_visual_output(output, self.device) for output in outputs
                    )
                return self._concat_visual_outputs(ordered_outputs)
            outputs = [
                self.base_model.encode_image(images[start : start + chunk_size], mode)
                for start in range(0, images.shape[0], chunk_size)
            ]
        return self._concat_visual_outputs(outputs)

    def _move_visual_output(self, output, device):
        if torch.is_tensor(output):
            return output.to(device, non_blocking=True)
        if isinstance(output, dict):
            return {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in output.items()
            }
        raise TypeError(f"Unsupported parallel visual output type: {type(output)!r}")

    def _get_visual_tokens(self, visual_output):
        if isinstance(visual_output, dict):
            if "tokens" not in visual_output:
                raise KeyError("Token visual output must contain 'tokens'")
            return visual_output["tokens"]
        if torch.is_tensor(visual_output) and visual_output.ndim == 3:
            return visual_output
        raise TypeError(f"Expected token visual output, got {type(visual_output)!r}")

    def _get_visual_embedding(self, visual_output):
        if self._uses_spatial_map_visual():
            return self.base_model.visual.__getattr__(self.args.pooling)(visual_output).float().flatten(1)
        mode = str(getattr(self.args, "visual_pooling", "cls"))
        if mode == "cls":
            return extract_global_feat(visual_output)
        tokens = self._get_visual_tokens(visual_output)
        patches = tokens[:, 1:]
        if mode == "mean_patch":
            return patches.mean(dim=1).float()
        if self.patch_gem is None:
            raise RuntimeError("Patch GeM module is not initialized")
        gem = self.patch_gem(patches).float()
        if mode == "gem_patch":
            return gem
        if mode == "cls_gem":
            return tokens[:, 0].float() + self.patch_pool_gamma * gem
        raise ValueError(f"Unsupported visual_pooling: {mode}")

    def extract_global_feat(self, visual_output):
        return self._get_visual_embedding(visual_output)

    def _forward_pmt_recipe(self, batch_dict, mode=None, current_epoch=None):
        if self.args.pretrain_choice != "PMT_VIT":
            raise ValueError("PMT recipe is only valid with pretrain_choice='PMT_VIT'")
        if self.args.training_mode != "RGB_IR":
            raise ValueError("PMT recipe is image-only and requires training_mode='RGB_IR'")

        rgb_imgs = batch_dict['img_rgb_ori']
        gray_imgs = batch_dict['img_rgb_aug']
        ir_imgs = batch_dict['img_ir']
        label_rgb = batch_dict['target_rgb'].long()
        label_ir = batch_dict['target_ir'].long()
        self._assert_pmt_batch_layout(label_rgb, label_ir)

        epoch = 0 if current_epoch is None else int(current_epoch)
        is_gray_stage = epoch < int(getattr(self.args, "pmt_progressive_epoch", 6))
        visible_imgs = gray_imgs if is_gray_stage else rgb_imgs
        stage = "gray_ir" if is_gray_stage else "rgb_ir"

        if self.args.Fix_Visual and not self._visual_unfrozen:
            visual_output = self._encode_fixed_visual(torch.cat((visible_imgs, ir_imgs), dim=0), mode)
        else:
            visual_output = self.base_model.encode_image(torch.cat((visible_imgs, ir_imgs), dim=0), mode)

        b = ir_imgs.size(0)
        visible_visual = self._slice_visual_output(visual_output, 0, b)
        ir_visual = self._slice_visual_output(visual_output, b, None)
        visible_feats = self._get_visual_embedding(visible_visual)
        ir_feats = self._get_visual_embedding(ir_visual)
        features = torch.cat((visible_feats, ir_feats), dim=0)
        labels = torch.cat((label_rgb, label_ir), dim=0)

        _, scores = self.classifier(features)
        score_visible, score_ir = scores.chunk(2, dim=0)

        ret = dict()
        ret.update({"temperature": 1 / self.logit_scale.exp()})
        ret.update({
            "id_loss": (
                self.pid_criterion(score_visible, label_rgb)
                + self.pid_criterion(score_ir, label_ir)
            ) * self.args.id_loss_weight
        })

        triplet_mining = getattr(self.args, "triplet_mining", "pmt_hard")
        if triplet_mining not in {"pmt_hard", "wrt", "pmt_cross_modal_hard"}:
            raise ValueError(f"Unsupported triplet_mining: {triplet_mining}")

        if is_gray_stage:
            if triplet_mining == "wrt":
                tri_loss = (
                    self.tri_criterion(visible_feats, label_rgb)
                    + self.tri_criterion(ir_feats, label_ir)
                )
            else:
                tri_loss = (
                    self.pmt_tri_criterion(visible_feats, visible_feats, label_rgb)
                    + self.pmt_tri_criterion(ir_feats, ir_feats, label_ir)
                )
            zero = features.new_zeros(())
            ret.update({"tri_loss": tri_loss})
            ret.update({"msel_loss": zero})
            ret.update({"dcl_loss": zero})
        else:
            if triplet_mining == "pmt_hard":
                tri_loss = self.pmt_tri_criterion(features, features, labels)
            elif triplet_mining == "wrt":
                tri_loss = self.tri_criterion(features, labels)
            else:
                tri_loss = self.cross_modal_tri_criterion(visible_feats, ir_feats, label_rgb)
                tri_loss = tri_loss * getattr(self.args, "pmt_cross_modal_triplet_weight", 1.0)
            msel_loss = self.pmt_msel_criterion(features, labels) * getattr(self.args, "pmt_msel_weight", 0.5)
            dcl_loss = self.pmt_dcl_criterion(features, labels) * getattr(self.args, "pmt_dcl_weight", 0.5)
            ret.update({"tri_loss": tri_loss})
            ret.update({"msel_loss": msel_loss})
            ret.update({"dcl_loss": dcl_loss})
        ret.update({"triplet_mining": triplet_mining})

        acc_visible = (score_visible.max(1)[1] == label_rgb).float().mean()
        acc_ir = (score_ir.max(1)[1] == label_ir).float().mean()
        ret.update({"acc": (acc_visible + acc_ir) / 2})
        ret.update({"pmt_stage": stage})
        return ret

    def save_model(self, save_epoch, is_best, mode='Fusion'): # mode = ['IR', 'Fusion', 'Text'] or their composition
        if mode not in ('Fusion', 'IR', 'Text'):
            raise ValueError("saving mode must be in ['Fusion', 'IR', 'Text']")
        if is_best:
            model_file_path = os.path.join(self.save_model_path, f'model_{mode}_{save_epoch}.pth')
            if self.args.DataParallel:
                torch.save(self.module.state_dict(), model_file_path)
            else:
                torch.save(self.state_dict(), model_file_path)

        if self.max_save_model_num > 0:
            root, _, files = os_walk(self.save_model_path)
            legacy_files = [
                (file, _checkpoint_epoch(file, mode, family="legacy"))
                for file in files
            ]
            legacy_files = [item for item in legacy_files if item[1] is not None]
            legacy_files.sort(key=lambda item: (item[1], item[0]))
            for filename, _epoch in legacy_files[:-self.max_save_model_num]:
                os.remove(os.path.join(root, filename))

    def save_metric_checkpoints(self, save_epoch, improved_metrics, mode='Fusion'):
        """Retain one physical snapshot per unique epoch referenced by a best metric."""
        if mode not in ('Fusion', 'IR', 'Text'):
            raise ValueError("mode must be in ['Fusion', 'IR', 'Text']")
        improved_metrics = tuple(improved_metrics)
        if not improved_metrics:
            return dict(self._metric_checkpoint_paths)
        model_file_path = os.path.join(self.save_model_path, f'model_{mode}_epoch_{save_epoch}.pth')
        if not os.path.isfile(model_file_path):
            if self.args.DataParallel:
                torch.save(self.module.state_dict(), model_file_path)
            else:
                torch.save(self.state_dict(), model_file_path)
        for metric in improved_metrics:
            if metric not in ('Rank-1', 'mAP', 'mINP'):
                raise ValueError(f'Unsupported selection metric: {metric}')
            self._metric_checkpoint_paths[metric] = model_file_path
        referenced = set(self._metric_checkpoint_paths.values())
        root, _, files = os_walk(self.save_model_path)
        prefix = f'model_{mode}_epoch_'
        for filename in files:
            if filename.startswith(prefix) and filename.endswith('.pth'):
                candidate = os.path.join(root, filename)
                if candidate not in referenced:
                    os.remove(candidate)
        return dict(self._metric_checkpoint_paths)


    def resume_last_model(self,mode='Fusion'):
        if mode not in ('Fusion', 'IR', 'Text'):
            raise ValueError("mode must be in ['Fusion', 'IR', 'Text']")
        root, _, files = os_walk(self.save_model_path)
        valid_epochs = sorted(
            {
                epoch
                for file in files
                for epoch in [_checkpoint_epoch(file, mode)]
                if epoch is not None
            }
        )
        if not valid_epochs:
            return 0
        latest_epoch = valid_epochs[-1]
        self.resume_model(latest_epoch, mode)
        return latest_epoch

    def resume_model(self, resume_epoch, mode='Fusion'):
        candidates = (
            os.path.join(self.save_model_path, f'model_{mode}_epoch_{resume_epoch}.pth'),
            os.path.join(self.save_model_path, f'model_{mode}_{resume_epoch}.pth'),
        )
        model_path = next((path for path in candidates if os.path.isfile(path)), None)
        if model_path is None:
            raise FileNotFoundError(
                f"No {mode} checkpoint for epoch {resume_epoch}; checked {list(candidates)}"
            )
        print('Resume model from {}'.format(model_path))
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        # A resume checkpoint was produced by this model and must be complete.
        # Historical/warm-start migrations use the audited compatibility loader
        # in main.py instead of silently accepting partial resume state here.
        self.load_state_dict(checkpoint, strict=True)
        print('Successfully resume model from {}'.format(model_path))


    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split(',')]
        print(f'Training Model with {self.current_task} tasks')

    def _run_attention(self, attn_module, q, k, v):
        q = self.ln_pre_t(q)
        k = self.ln_pre_i(k)
        v = self.ln_pre_i(v)
        if getattr(attn_module, "_expects_batch_first", False):
            return attn_module(q, k, v, need_weights=False)[0]
        return attn_module(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            need_weights=False,
        )[0].transpose(0, 1)

    def cross_former(self, q, k, v):
        x = self._run_attention(self.cross_attn, q, k, v)
        x = q + x # residual connection (invalid for mcq and mcqmlm, valid for mlm)
        x = self.ln_post(x)
        return x

    def global_former_s(self, q, k, v):
        x = self._run_attention(self.global_attn_s, q, k, v)

        x = q + x # residual connection (invalid for mcq and mcqmlm, valid for mlm)
        x = self.ln_post(x)
        return x

    def global_former_t(self, q, k, v):
        x = self._run_attention(self.global_attn_t, q, k, v)
        x = q + x # residual connection (invalid for mcq and mcqmlm, valid for mlm)
        x = self.ln_post(x)
        return x

    def encode_image_featmap(self, image, mode=None):
        if self.args.Fix_Visual and not self._visual_unfrozen:
            x = self._encode_fixed_visual(image, mode)
        else:
            x = self.base_model.encode_image(image,mode=mode)
        return x
        # return x.float() # for CLIP ResNet visual model

    def encode_text_featmap(self, text):
        x = self.base_model.encode_text(text)
        return x #[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_feat(self, image, mode=None): # return [B, 512]
        x = self.base_model.encode_image(image,mode=mode)
        return self._get_visual_embedding(x)

    def encode_text_feat(self, text): # return [B, 512]
        x = self.base_model.encode_text(text)
        return extract_text_token_feat(x, text)

    def encode_fusion(self, text, ir, mode='ir'):
        # 获取 id 形式的文本原始数据
        caption_ids = text
        # 获取文本Tensor特征
        text = self.encode_text_featmap(text)
        # 获取IR图像Tensor特征
        ir = self.encode_image_featmap(ir,mode=mode)
        # 获取融合后的特征
        x = self.fusion_layer(text,ir,caption_ids,pa=self.current_pa(), way=self.args.fusion_way)
        return x.float()

    def encode_filtered_fusion(self, text, filter, ir):
        # 获取 id 形式的文本原始数据
        caption_ids = text
        filter_caption_ids = filter
        # 获取文本Tensor特征
        text_feat = self.encode_text_feat(text)
        # 获取filter Tensor特征
        filter_text_feat = self.encode_text_feat(filter)
        # 获取IR图像Tensor特征
        ir = self.encode_image_feat(ir,mode='ir')
        # 获取融合后的特征
        ensure_matching_feature_shape(ir_feats=ir, t_feats=text_feat, text_filter_feats=filter_text_feat)
        x = ir + text_feat - filter_text_feat
        return x.float()

    def text_fusion_layer(self, text_rgb_map, text_ir_map, caption_rgb_ids, caption_ir_ids, way='add'):
        if way not in DUAL_TEXT_FUSION_WAYS:
            raise ValueError(
                f"Unsupported dual-text fusion_way {way!r}; "
                f"expected one of {sorted(DUAL_TEXT_FUSION_WAYS)}"
            )
        if way == 'norm_add':
            text_rgb_feats = text_rgb_map[torch.arange(text_rgb_map.shape[0]), caption_rgb_ids.argmax(dim=-1)]
            text_ir_feats = text_ir_map[torch.arange(text_ir_map.shape[0]), caption_ir_ids.argmax(dim=-1)]
            f_text_feats = F.normalize(text_rgb_feats, dim=-1) + F.normalize(text_ir_feats, dim=-1)

        elif way == 'add':
            text_rgb_feats = text_rgb_map[torch.arange(text_rgb_map.shape[0]), caption_rgb_ids.argmax(dim=-1)]
            text_ir_feats = text_ir_map[torch.arange(text_ir_map.shape[0]), caption_ir_ids.argmax(dim=-1)]
            f_text_feats = text_rgb_feats + text_ir_feats

        elif way == 'cross_attention':
            text_rgb_feats = text_rgb_map[torch.arange(text_rgb_map.shape[0]), caption_rgb_ids.argmax(dim=-1)]
            text_ir_feats = text_ir_map[torch.arange(text_ir_map.shape[0]), caption_ir_ids.argmax(dim=-1)]
            f_text_feats = (self.cross_former(text_rgb_feats.unsqueeze(1),text_ir_map,text_ir_map) + self.cross_former(text_ir_feats.unsqueeze(1),text_rgb_map,text_rgb_map))
            f_text_feats = f_text_feats.squeeze(1).contiguous()

        elif way == 'attention_rgb_text':
            text_rgb_feats = text_rgb_map[torch.arange(text_rgb_map.shape[0]), caption_rgb_ids.argmax(dim=-1)]
            f_text_feats = self.cross_former(text_rgb_feats.unsqueeze(1),text_ir_map,text_ir_map).squeeze(1).contiguous()

        elif way == 'attention_ir_text':
            text_ir_feats = text_ir_map[torch.arange(text_ir_map.shape[0]), caption_ir_ids.argmax(dim=-1)]
            f_text_feats = self.cross_former(text_ir_feats.unsqueeze(1),text_rgb_map,text_rgb_map).squeeze(1).contiguous()

        elif way == 'global_attention':
            f_text_feats = (self.global_former_s(text_rgb_map,text_ir_map,text_ir_map)[torch.arange(text_rgb_map.shape[0]), caption_rgb_ids.argmax(dim=-1)] +\
                             self.global_former_t(text_ir_map,text_rgb_map,text_rgb_map)[torch.arange(text_ir_map.shape[0]), caption_ir_ids.argmax(dim=-1)])
            f_text_feats = f_text_feats.squeeze(1).contiguous()

        elif way == 'global_attention_rgb_text':
            f_text_feats = self.global_former_s(text_rgb_map,text_ir_map,text_ir_map)[torch.arange(text_rgb_map.shape[0]), caption_rgb_ids.argmax(dim=-1)].squeeze(1).contiguous()

        elif way == 'global_attention_ir_text':
            f_text_feats = self.global_former_t(text_ir_map,text_rgb_map,text_rgb_map)[torch.arange(text_ir_map.shape[0]), caption_ir_ids.argmax(dim=-1)].squeeze(1).contiguous()

        return f_text_feats

    def adaptive_fusion_layer(self, text_feats, ir_feats):
        adaptive_type = getattr(self.args, "adaptive_fusion_type", "scalar_alpha")
        if adaptive_type == "scalar_alpha":
            if self.adaptive_alpha is None:
                raise RuntimeError("adaptive_alpha is not initialized")
            return ir_feats + self.adaptive_alpha * text_feats

        if self.adaptive_gate is None:
            raise RuntimeError("adaptive_gate is not initialized")

        if adaptive_type == "norm_residual_gate":
            ir_input = F.normalize(ir_feats, dim=-1)
            text_input = F.normalize(text_feats, dim=-1)
        else:
            ir_input = ir_feats
            text_input = text_feats

        gate_input = torch.cat((ir_input, text_input, ir_input * text_input), dim=-1)
        alpha = torch.sigmoid(self.adaptive_gate(gate_input))

        if adaptive_type in {"sample_gate", "channel_gate"}:
            return ir_feats + alpha * text_feats
        if adaptive_type == "residual_gate":
            return ir_feats + alpha * (text_feats - ir_feats)
        if adaptive_type == "norm_residual_gate":
            return ir_input + alpha * text_input
        raise ValueError(f"Unsupported adaptive_fusion_type: {adaptive_type}")

    def fusion_layer(self, text_map, ir_map, caption_ids, pa=0.1, way='add'):
        if way not in IMAGE_TEXT_FUSION_WAYS:
            raise ValueError(
                f"Unsupported image-text fusion_way {way!r}; "
                f"expected one of {sorted(IMAGE_TEXT_FUSION_WAYS)}"
            )
        text_feats = extract_text_token_feat(text_map, caption_ids)
        if self._uses_spatial_map_visual():
            ir_feats = self._get_visual_embedding(ir_map)
            ir_tokens = ir_map
        elif self._uses_token_visual():
            ir_feats = self._get_visual_embedding(ir_map)
            ir_tokens = self._get_visual_tokens(ir_map)
        else:
            raise ValueError(f"pretrain_choice {self.args.pretrain_choice} is not supported")
        if way == 'norm_add':
            text_unit = F.normalize(text_feats, dim=-1)
            ir_scale = ir_feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            ir_unit = ir_feats / ir_scale
            f_feats = 0.5 * (text_unit + ir_unit) * ir_scale
        elif way == 'cross_attention':
            f_feats = (self.cross_former(text_feats.unsqueeze(1),ir_tokens,ir_tokens) + self.cross_former(ir_feats.unsqueeze(1),text_map,text_map))
            f_feats = f_feats.squeeze(1).contiguous()
        elif way == 'parameter_add':
            f_feats = (1-pa)*text_feats + pa*ir_feats
        elif way == 'adaptive_add':
            f_feats = self.adaptive_fusion_layer(text_feats, ir_feats)
        elif way == 'add':
            f_feats = text_feats + ir_feats
        else:  # pragma: no cover - guarded by IMAGE_TEXT_FUSION_WAYS.
            raise AssertionError(f"Unhandled image-text fusion_way: {way}")
        return f_feats.float()

    def forward(self, batch_dict, mode=None, current_epoch=None):
        # get data
        if self.args.training_mode == 'RGB_IR_Text':
            text_modalities = (
                self.retrieval_backend.TRAIN_TEXT_MODALITIES
                if self.retrieval_backend
                else ("rgb", "ir")
            )
            validate_rgb_ir_text_batch_dict(batch_dict, text_modalities)

        rgb_imgs0 = batch_dict['img_rgb_ori']
        rgb_imgs1 = batch_dict['img_rgb_aug']
        ir_imgs = batch_dict['img_ir']
        label_rgb = batch_dict['target_rgb']
        label_ir = batch_dict['target_ir']

        if self._pmt_recipe_enabled():
            return self._forward_pmt_recipe(batch_dict, mode=mode, current_epoch=current_epoch)

        # init return dict
        ret = dict()

        # get feature map
        b = ir_imgs.size(0)
        if self.args.Fix_Visual and not self._visual_unfrozen:
            image_feats_map = self._encode_fixed_visual(
                torch.cat((rgb_imgs0, rgb_imgs1, ir_imgs), dim=0),
                mode,
            )
        else:
            image_feats_map = self.base_model.encode_image(
                torch.cat((rgb_imgs0, rgb_imgs1, ir_imgs), dim=0),
                mode,
            )
        rgb_visual = self._slice_visual_output(image_feats_map, 0, int(2 * b))
        ir_visual = self._slice_visual_output(image_feats_map, int(2 * b), None)
        fusion_ir_visual = ir_visual
        rgb_feats = self._get_visual_embedding(rgb_visual)
        ir_feats = self._get_visual_embedding(ir_visual)


        logit_scale = self.logit_scale.exp()
        ret.update({'temperature': 1 / logit_scale})


        loss_list = [loss_name.strip() for loss_name in self.args.loss_names.split(',') if loss_name.strip()]

        if self.retrieval_backend:
            ret.update(
                self.retrieval_backend.training_losses(
                    self,
                    batch_dict,
                    rgb_visual,
                    rgb_feats,
                    ir_feats,
                    label_rgb,
                    label_ir,
                    loss_list,
                )
            )
            return ret

        if self.args.training_mode == 'RGB_IR_Text': # 如果有文本信息辅助
            if self.args.fusion_way in FUSION_WAYS:

                # 获取rgb图像特征
                ori_vi_feats = rgb_feats[:int(b)]
                aug_vi_feats = rgb_feats[int(b):]

                if self.args.joint_mode == 'dual_text':
                    text_rgb = batch_dict['text_rgb']
                    text_ir = batch_dict['text_ir']
                    text_rgb_feats_map = self.base_model.encode_text(text_rgb).detach()
                    text_ir_feats_map = self.base_model.encode_text(text_ir).detach()
                    t_feats = self.text_fusion_layer(text_rgb_feats_map,text_ir_feats_map,text_rgb,text_ir,way=self.args.fusion_way)
                    pids = torch.cat([label_rgb, label_rgb, label_ir], dim=0)
                    img_feats = torch.cat((rgb_feats, ir_feats), dim=0)
                    if 'id' in loss_list:
                        # get labels
                        _,img_scores = self.classifier(img_feats)
                        ret.update({'id_loss':(self.pid_criterion(img_scores, pids))*self.args.id_loss_weight})
                        img_acc = (img_scores.max(1)[1] == pids).float().mean()
                        ret.update({'acc': img_acc})

                    if 'wrt' in loss_list:
                        ret.update({'wrt_loss':(self.tri_criterion(img_feats, pids))*self.args.wrt_loss_weight})

                    if 'i2t' in loss_list:
                        ret.update({'i2t_loss':L_i2t(ir_feats,t_feats,logit_scale) + \
                                    0.5*(L_i2t(ori_vi_feats,t_feats,logit_scale) + L_i2t(aug_vi_feats,t_feats,logit_scale))})

                    if 't2i' in loss_list:
                        ret.update({'t2i_loss':L_t2i(ir_feats,t_feats,logit_scale,label_ir) + \
                                    0.5*(L_t2i(ori_vi_feats,t_feats,logit_scale,label_rgb) + L_t2i(aug_vi_feats,t_feats,logit_scale,label_rgb))})


                elif self.args.joint_mode == 'uni':
                    text_rgb = batch_dict['text_rgb']
                    text_rgb_feats_map = self.base_model.encode_text(text_rgb)
                    fusion_text_feats_map = text_rgb_feats_map
                    t_feats = extract_text_token_feat(text_rgb_feats_map, text_rgb)
                    imta_enabled = any(name in loss_list for name in ('imta_proto', 'imta_dual', 'imta_rel'))
                    text_ir_feats = None
                    if imta_enabled:
                        if 'text_ir' not in batch_dict:
                            raise KeyError("IMTA losses require batch_dict['text_ir']")
                        if label_rgb.shape != label_ir.shape or not torch.equal(label_rgb, label_ir):
                            raise ValueError("IMTA requires aligned RGB/IR identity labels")
                        text_ir_feats = self.encode_text_feat(batch_dict['text_ir']).float()
                    text_filter_feats = None

                    # 获取融合后的特征
                    if self.args.Feat_Filter:
                        text_filter = batch_dict['text_ir']
                        text_filter_feats = self.encode_text_feat(text_filter)
                        ensure_matching_feature_shape(ir_feats=ir_feats, t_feats=t_feats, text_filter_feats=text_filter_feats)
                        f_feats = ir_feats + t_feats - text_filter_feats

                    else:
                        f_feats = self.fusion_layer(
                            fusion_text_feats_map,
                            fusion_ir_visual,
                            text_rgb,
                            pa=self.current_pa(),
                            way=self.args.fusion_way,
                        )
                        f_feats = f_feats.squeeze()
                        # 获取文本特征
                        t_feats = t_feats.float()

                    # # uni_id
                    # uni_pids = torch.cat([label_rgb,label_rgb,label_ir,label_ir,label_ir], dim=0)
                    # all_feats = torch.cat((ori_vi_feats, aug_vi_feats, ir_feats, f_feats, t_feats), dim=0)
                    uni_pids = torch.cat([label_rgb,label_rgb,label_ir,label_ir,label_ir], dim=0)
                    all_feats = torch.cat((ori_vi_feats, aug_vi_feats, ir_feats, f_feats, t_feats), dim=0)
                    if "id" in loss_list:
                        # uni_id
                        uni_pids = torch.cat([label_rgb,label_rgb,label_ir,label_ir,label_ir], dim=0)
                        all_feats = torch.cat((ori_vi_feats, aug_vi_feats, ir_feats, f_feats, t_feats), dim=0)
                        _, all_feat_scores = self.classifier(all_feats)
                        ret.update({'id_loss':(self.pid_criterion(all_feat_scores, uni_pids))*self.args.id_loss_weight})
                        feat_acc = (all_feat_scores.max(1)[1] == uni_pids).float().mean()
                        ret.update({'acc': feat_acc})

                    uni_woir_pids = torch.cat([label_rgb, label_rgb, label_ir, label_ir], dim=0)
                    uni_woir_feats = torch.cat([rgb_feats, f_feats, t_feats], dim=0)
                    if "id_woir" in loss_list:
                        _, img_scores = self.classifier(uni_woir_feats)
                        ret.update({'uni_id_woir_loss':(self.pid_criterion(img_scores, uni_woir_pids))*self.args.id_loss_weight})
                        feat_acc = (img_scores.max(1)[1] == uni_woir_pids).float().mean()
                        ret.update({'acc': feat_acc})

                    if "wrt" in loss_list:
                        # uni_wrt
                        ret.update({'wrt_loss':(self.tri_criterion(all_feats, uni_pids))*self.args.wrt_loss_weight})

                    if "imta_proto" in loss_list:
                        ret.update({'imta_proto_loss': objectives.imta_prototype_loss(
                            t_feats, text_ir_feats, (ori_vi_feats + aug_vi_feats) * 0.5, ir_feats,
                            label_rgb, temperature=float(getattr(self.args, 'imta_temperature', 0.07))
                        ) * float(getattr(self.args, 'imta_proto_weight', 0.25))})

                    if "imta_dual" in loss_list:
                        ret.update({'imta_dual_loss': objectives.imta_dual_text_supcon_loss(
                            t_feats, text_ir_feats, label_rgb,
                            temperature=float(getattr(self.args, 'imta_temperature', 0.07))
                        ) * float(getattr(self.args, 'imta_dual_weight', 0.10))})

                    if "imta_rel" in loss_list:
                        ret.update({'imta_rel_loss': objectives.imta_relation_loss(
                            t_feats, text_ir_feats, (ori_vi_feats + aug_vi_feats) * 0.5, ir_feats,
                            temperature=float(getattr(self.args, 'imta_relation_temperature', 0.10))
                        ) * float(getattr(self.args, 'imta_relation_weight', 0.10))})

                    if "cross_modal_hard" in loss_list:
                        if label_rgb.shape != label_ir.shape or not torch.equal(label_rgb, label_ir):
                            raise ValueError("Stage B cross-modal hard mining requires aligned RGB and IR labels")
                        modalities = {
                            "RGB": (ori_vi_feats + aug_vi_feats) / 2.0,
                            "IR": ir_feats,
                            "Fusion": f_feats,
                            "Text": t_feats,
                        }
                        shapes = {name: tuple(value.shape) for name, value in modalities.items()}
                        if len(set(shapes.values())) != 1:
                            raise ValueError(f"Stage B modality batch shapes are not aligned: {shapes}")
                        names = list(modalities)
                        pair_losses = {}
                        for left_index, left_name in enumerate(names):
                            for right_name in names[left_index + 1:]:
                                pair_name = f"{left_name}-{right_name}"
                                pair_losses[pair_name] = self.cross_modal_tri_criterion(
                                    modalities[left_name], modalities[right_name], label_rgb
                                )
                        cross_modal_loss = weighted_cross_modal_pair_loss(
                            pair_losses,
                            getattr(self.args, "cross_modal_pair_weights", None),
                        )
                        if not torch.isfinite(cross_modal_loss):
                            raise FloatingPointError("Stage B cross-modal hard loss is not finite")
                        ret.update({
                            'cross_modal_hard_loss': cross_modal_loss
                            * float(getattr(self.args, "cross_modal_hard_weight", 1.0))
                        })

                    if "wrt_woir" in loss_list:
                        # uni_wrt
                        ret.update({'uni_wrt_woir_loss':(self.tri_criterion(uni_woir_feats, uni_woir_pids))*self.args.wrt_loss_weight})

                    if "orth" in loss_list:
                        # uni_orth
                        ret.update({'uni_orth_loss':objectives.orthogonal_loss(ir_feats, t_feats, text_filter_feats)})

                    if "orth2" in loss_list:
                        # uni_orth
                        ret.update({'uni_orth2_loss':objectives.orthogonal_loss2(ir_feats, t_feats, text_filter_feats)})

                    if "T2I_Regular" in loss_list:
                        ret.update({"T2I_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='T2I')})

                    if "I2T_Regular" in loss_list:
                        ret.update({"I2T_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='I2T')})

                elif self.args.joint_mode == 'ir_crossfusion':

                    text_rgb = batch_dict['text_rgb']
                    text_rgb_feats_map = self.base_model.encode_text(text_rgb)
                    # 获取融合后的特征
                    if self.args.Feat_Filter:
                        text_filter = batch_dict['text_ir']
                        text_filter_feats = self.encode_text_feat(text_filter)
                        f_feats = (ir_feats + text_rgb_feats_map[torch.arange(text_rgb_feats_map.shape[0]), text_rgb.argmax(dim=-1)] - text_filter_feats).squeeze()

                    else:
                        f_feats = self.fusion_layer(text_rgb_feats_map, ir_visual, text_rgb, pa=self.current_pa(), way=self.args.fusion_way).squeeze()

                    pids = torch.cat([label_rgb, label_rgb, label_ir], dim=0)
                    img_feats = torch.cat((ori_vi_feats, aug_vi_feats, f_feats), dim=0)
                    if 'id' in loss_list:
                        # get labels
                        _, img_scores = self.classifier(img_feats)
                        ret.update({'id_loss':(self.pid_criterion(img_scores, pids))*self.args.id_loss_weight})
                        img_acc = (img_scores.max(1)[1] == pids).float().mean()
                        ret.update({'acc': img_acc})

                    if 'wrt' in loss_list:
                        ret.update({'wrt_loss':(self.tri_criterion(img_feats, pids))*self.args.wrt_loss_weight})

                    # if "T2I_Regular" in loss_list:
                    #     ret.update({"T2I_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='T2I')})

                    # if "I2T_Regular" in loss_list:
                    #     ret.update({"I2T_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='I2T')})


                elif self.args.joint_mode == 'ir_selffusion':

                    text_rgb_w = batch_dict['text_rgb_w']
                    text_rgb_w_feats_map = self.base_model.encode_text(text_rgb_w)

                    # 获取融合后的特征
                    f_feats = self.fusion_layer(text_rgb_w_feats_map, ir_visual, text_rgb_w, pa=self.current_pa(), way=self.args.fusion_way).squeeze()

                    pids = torch.cat([label_rgb, label_rgb, label_ir], dim=0)
                    img_feats = torch.cat((rgb_feats, f_feats), dim=0)
                    if 'id' in loss_list:
                        # get labels
                        _, img_scores = self.classifier(img_feats)
                        ret.update({'id_loss':(self.pid_criterion(img_scores, pids))*self.args.id_loss_weight})
                        img_acc = (img_scores.max(1)[1] == pids).float().mean()
                        ret.update({'acc': img_acc})

                    if 'wrt' in loss_list:
                        ret.update({'wrt_loss':(self.tri_criterion(img_feats, pids))*self.args.wrt_loss_weight})

                    # if "T2I_Regular" in loss_list:
                    #     ret.update({"T2I_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='T2I')})

                    # if "I2T_Regular" in loss_list:
                    #     ret.update({"I2T_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='I2T')})


                elif self.args.joint_mode == 'rgb_selffusion':

                    text_ir = batch_dict['text_ir']
                    text_ir_feats_map = self.base_model.encode_text(text_ir)

                    # 获取融合后的特征
                    f_feats = self.fusion_layer(torch.cat([text_ir_feats_map,text_ir_feats_map],dim=0), rgb_visual, torch.cat([text_ir,text_ir],dim=0), pa=self.current_pa(), way=self.args.fusion_way).squeeze()

                    pids = torch.cat([label_rgb, label_rgb, label_ir], dim=0)
                    img_feats = torch.cat((f_feats, ir_feats), dim=0)
                    if 'id' in loss_list:
                        # get labels
                        _,img_scores = self.classifier(img_feats)
                        ret.update({'id_loss':(self.pid_criterion(img_scores, pids))*self.args.id_loss_weight})
                        img_acc = (img_scores.max(1)[1] == pids).float().mean()
                        ret.update({'acc': img_acc})

                    if 'wrt' in loss_list:
                        ret.update({'wrt_loss':(self.tri_criterion(img_feats, pids))*self.args.wrt_loss_weight})

                    # if "T2I_Regular" in loss_list:
                    #     ret.update({"T2I_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='T2I')})

                    # if "I2T_Regular" in loss_list:
                    #     ret.update({"I2T_Regular_loss":kl_align_loss(ir_feats,f_feats,t_feats,logit_scale,mode='I2T')})

                else:
                    raise ValueError(
                        f"joint mode must be one of {sorted(JOINT_MODE_FUSION_WAYS)}"
                    )
            else:  # 如果融合方式没有被定义
                raise NotImplementedError(
                    f"Fusion way must be one of {sorted(FUSION_WAYS)}"
                )

        elif self.args.training_mode == "RGB_IR":
            pids = torch.cat([label_rgb,label_rgb,label_ir], dim=0)
            img_feats = torch.cat((rgb_feats, ir_feats), dim=0)
            if 'id' in loss_list:
                _, scores = self.classifier(img_feats)
                ret.update({'id_loss':(self.pid_criterion(scores, pids))*self.args.id_loss_weight})
                acc = (scores.max(1)[1] == pids).float().mean()
                ret.update({'acc': acc})

            if 'wrt' in loss_list:
                ret.update({'wrt_loss':(self.tri_criterion(img_feats, pids))*self.args.wrt_loss_weight})

        elif self.args.training_mode == "RGB_Text":
            text_rgb = batch_dict['text_rgb']
            text_rgb_feats_map = self.base_model.encode_text(text_rgb)
            t_feats = text_rgb_feats_map[torch.arange(text_rgb_feats_map.shape[0]), text_rgb.argmax(dim=-1)].float()
            img_text_feats = torch.cat((rgb_feats, t_feats), dim=0)
            pids = torch.cat([label_rgb,label_rgb,label_ir], dim=0)
            if 'id' in loss_list:
                _, scores = self.classifier(img_text_feats)
                ret.update({'id_loss':(self.pid_criterion(scores, pids))*self.args.id_loss_weight})
                acc = (scores.max(1)[1] == pids).float().mean()
                ret.update({'acc': acc})

            if 'wrt' in loss_list:
                ret.update({'wrt_loss':(self.tri_criterion(img_text_feats, pids))*self.args.wrt_loss_weight})
        else:
            raise ValueError("training mode must be in ['RGB_IR_Text', 'RGB_IR', 'RGB_Text']")

        return ret



def build_model(config):
    model = CLIP2ReID(config, num_classes=config.pid_num)
    return model
