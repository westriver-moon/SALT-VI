import torch


NAME = "ir_to_rgb_text"
RESULT_KEY = "IR-RGBText"
TRAIN_TEXT_MODALITIES = ("rgb",)
QUERY_CAPTION_LOOKUP = None
GALLERY_CAPTION_LOOKUP = "image"
QUERY_NAME = "infrared-image"
GALLERY_NAME = "visible-image-plus-image-caption"


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
