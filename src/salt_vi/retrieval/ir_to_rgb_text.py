import torch
from collections.abc import Mapping


NAME = "ir_to_rgb_text"
RESULT_KEY = "IR-RGBText"
IS_LEGACY = False
TRAIN_TEXT_MODALITIES = ("rgb",)
QUERY_CAPTION_LOOKUP = None
GALLERY_CAPTION_LOOKUP = "image"
QUERY_NAME = "infrared-image"
GALLERY_NAME = "visible-image-plus-image-caption"


def _value(config, name, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def train_text_modalities(config):
    return TRAIN_TEXT_MODALITIES


def query_caption_lookup(config):
    return QUERY_CAPTION_LOOKUP


def gallery_caption_lookup(config):
    return GALLERY_CAPTION_LOOKUP


def training_recipe(config):
    return NAME


def validate(config, sr_backend=None, sr_modalities=None):
    if str(_value(config, "dataset", "")).lower() != "sysu":
        raise ValueError("ir_to_rgb_text is supported only for SYSU-MM01")
    if (
        str(_value(config, "training_mode", "")) != "RGB_IR_Text"
        or str(_value(config, "joint_mode", "")) != "uni"
    ):
        raise ValueError("ir_to_rgb_text requires training_mode=RGB_IR_Text and joint_mode=uni")
    if bool(_value(config, "Feat_Filter", False)):
        raise ValueError("ir_to_rgb_text does not use IR caption filtering")
    if bool(_value(config, "uni_BN", False)):
        raise ValueError("ir_to_rgb_text requires the shared classifier BN")
    if sr_backend == "pasd_multiview" and sr_modalities != {"rgb"}:
        raise ValueError("ir_to_rgb_text PASD mode requires RGB-only multiview SR")
    if str(_value(config, "test_modality", "")) != RESULT_KEY:
        raise ValueError(f"ir_to_rgb_text requires test_modality={RESULT_KEY}")
    if not _value(config, "gallery_caption_manifest"):
        raise ValueError("ir_to_rgb_text requires gallery_caption_manifest")
    text_dropout = float(_value(config, "gallery_text_dropout", 0.0))
    if not 0.0 <= text_dropout < 1.0:
        raise ValueError("gallery_text_dropout must be in [0, 1)")
    return config


def training_losses(
    model,
    batch_dict,
    rgb_visual,
    rgb_feats,
    ir_feats,
    label_rgb,
    label_ir,
    loss_names,
):
    batch_size = ir_feats.shape[0]
    text = batch_dict["text_rgb"]
    text_map = model.base_model.encode_text(text)
    rgb_image_map = model._slice_visual_output(rgb_visual, 0, batch_size)
    rgb_text_feats = model.fusion_layer(
        text_map,
        rgb_image_map,
        text,
        pa=model.current_pa(),
        way=model.args.fusion_way,
    )
    rgb_image_feats = rgb_feats[:batch_size]
    rgb_aux_feats = (rgb_image_feats + rgb_feats[batch_size:]) * 0.5

    text_dropout = float(getattr(model.args, "gallery_text_dropout", 0.0))
    if text_dropout:
        use_text = torch.rand(batch_size, 1, device=rgb_text_feats.device) >= text_dropout
        rgb_text_feats = torch.where(use_text, rgb_text_feats, rgb_image_feats)

    losses = {}
    if "id" in loss_names:
        features = torch.cat((ir_feats, rgb_text_feats), dim=0)
        labels = torch.cat((label_ir, label_rgb), dim=0)
        _, scores = model.classifier(features)
        losses["id_loss"] = (
            model.pid_criterion(scores, labels) * float(model.args.id_loss_weight)
        )
        losses["acc"] = (scores.argmax(dim=1) == labels).float().mean()

    if "cross_modal_hard" in loss_names:
        main = model.cross_modal_tri_criterion(ir_feats, rgb_text_feats, label_ir)
        auxiliary = model.cross_modal_tri_criterion(ir_feats, rgb_aux_feats, label_ir)
        losses["cross_modal_hard_loss"] = float(
            getattr(model.args, "cross_modal_hard_weight", 1.0)
        ) * (
            float(getattr(model.args, "ir_rgb_text_pair_weight", 1.0)) * main
            + float(getattr(model.args, "ir_rgb_aux_weight", 0.5)) * auxiliary
        )
    return losses


def encode_query(model, batch_dict, device):
    image = batch_dict["img"].to(device)
    visual = model.encode_image_featmap(image, "ir")
    return model.classifier(model.extract_global_feat(visual), "IR")


def encode_gallery(model, batch_dict, device):
    image = batch_dict["img"].to(device)
    text = batch_dict["text"].to(device).long()
    return model.classifier(model.encode_fusion(text, image, mode="rgb"), "Fusion")


def evaluate(model, loader, config, device):
    from .evaluator import evaluate_sysu
    from . import ir_to_rgb_text

    return evaluate_sysu(model, loader, device, ir_to_rgb_text)
