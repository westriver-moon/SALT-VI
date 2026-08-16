from dataclasses import dataclass

import torch

import salt_vi.models.clip_model.objectives as objectives
from salt_vi.data.loader import validate_rgb_ir_text_batch_dict
from salt_vi.utils import kl_align_loss

from .common import (
    ensure_matching_feature_shape,
    extract_text_token_feat,
    weighted_cross_modal_pair_loss,
)


@dataclass
class EncodedBatch:
    rgb_visual: object
    ir_visual: object
    rgb_feats: torch.Tensor
    ir_feats: torch.Tensor
    label_rgb: torch.Tensor
    label_ir: torch.Tensor
    batch_size: int
    logit_scale: torch.Tensor


def _loss_names(model):
    return [name.strip() for name in model.args.loss_names.split(",") if name.strip()]


def _encode_batch(model, batch, mode):
    rgb_images = torch.cat((batch["img_rgb_ori"], batch["img_rgb_aug"]), dim=0)
    ir_images = batch["img_ir"]
    images = torch.cat((rgb_images, ir_images), dim=0)
    if model.args.Fix_Visual and not model._visual_unfrozen:
        visual = model._encode_fixed_visual(images, mode)
    else:
        visual = model.base_model.encode_image(images, mode)
    batch_size = ir_images.size(0)
    rgb_visual = model._slice_visual_output(visual, 0, 2 * batch_size)
    ir_visual = model._slice_visual_output(visual, 2 * batch_size, None)
    return EncodedBatch(
        rgb_visual=rgb_visual,
        ir_visual=ir_visual,
        rgb_feats=model._get_visual_embedding(rgb_visual),
        ir_feats=model._get_visual_embedding(ir_visual),
        label_rgb=batch["target_rgb"],
        label_ir=batch["target_ir"],
        batch_size=batch_size,
        logit_scale=model.logit_scale.exp(),
    )


def _base_result(context):
    return {"temperature": 1 / context.logit_scale}


class PMTRecipe:
    name = "pmt"

    def compute_losses(self, model, batch, mode=None, current_epoch=None):
        if model.args.pretrain_choice != "PMT_VIT":
            raise ValueError("PMT recipe is only valid with pretrain_choice='PMT_VIT'")
        if model.args.training_mode != "RGB_IR":
            raise ValueError("PMT recipe is image-only and requires training_mode='RGB_IR'")
        rgb_images = batch["img_rgb_ori"]
        gray_images = batch["img_rgb_aug"]
        ir_images = batch["img_ir"]
        label_rgb = batch["target_rgb"].long()
        label_ir = batch["target_ir"].long()
        model._assert_pmt_batch_layout(label_rgb, label_ir)

        epoch = 0 if current_epoch is None else int(current_epoch)
        gray_stage = epoch < int(getattr(model.args, "pmt_progressive_epoch", 6))
        visible_images = gray_images if gray_stage else rgb_images
        stage = "gray_ir" if gray_stage else "rgb_ir"
        images = torch.cat((visible_images, ir_images), dim=0)
        if model.args.Fix_Visual and not model._visual_unfrozen:
            visual = model._encode_fixed_visual(images, mode)
        else:
            visual = model.base_model.encode_image(images, mode)

        batch_size = ir_images.size(0)
        visible_feats = model._get_visual_embedding(
            model._slice_visual_output(visual, 0, batch_size)
        )
        ir_feats = model._get_visual_embedding(
            model._slice_visual_output(visual, batch_size, None)
        )
        features = torch.cat((visible_feats, ir_feats), dim=0)
        labels = torch.cat((label_rgb, label_ir), dim=0)
        _, scores = model.classifier(features)
        score_visible, score_ir = scores.chunk(2, dim=0)

        result = {
            "temperature": 1 / model.logit_scale.exp(),
            "id_loss": (
                model.pid_criterion(score_visible, label_rgb)
                + model.pid_criterion(score_ir, label_ir)
            )
            * model.args.id_loss_weight,
        }
        mining = getattr(model.args, "triplet_mining", "pmt_hard")
        if mining not in {"pmt_hard", "wrt", "pmt_cross_modal_hard"}:
            raise ValueError(f"Unsupported triplet_mining: {mining}")
        if gray_stage:
            if mining == "wrt":
                tri_loss = model.tri_criterion(visible_feats, label_rgb) + model.tri_criterion(
                    ir_feats, label_ir
                )
            else:
                tri_loss = model.pmt_tri_criterion(
                    visible_feats, visible_feats, label_rgb
                ) + model.pmt_tri_criterion(ir_feats, ir_feats, label_ir)
            result.update(
                tri_loss=tri_loss,
                msel_loss=features.new_zeros(()),
                dcl_loss=features.new_zeros(()),
            )
        else:
            if mining == "pmt_hard":
                tri_loss = model.pmt_tri_criterion(features, features, labels)
            elif mining == "wrt":
                tri_loss = model.tri_criterion(features, labels)
            else:
                tri_loss = model.cross_modal_tri_criterion(
                    visible_feats, ir_feats, label_rgb
                ) * getattr(model.args, "pmt_cross_modal_triplet_weight", 1.0)
            result.update(
                tri_loss=tri_loss,
                msel_loss=model.pmt_msel_criterion(features, labels)
                * getattr(model.args, "pmt_msel_weight", 0.5),
                dcl_loss=model.pmt_dcl_criterion(features, labels)
                * getattr(model.args, "pmt_dcl_weight", 0.5),
            )
        acc_visible = (score_visible.argmax(dim=1) == label_rgb).float().mean()
        acc_ir = (score_ir.argmax(dim=1) == label_ir).float().mean()
        result.update(triplet_mining=mining, acc=(acc_visible + acc_ir) / 2, pmt_stage=stage)
        return result


class IRToRGBTextRecipe:
    name = "ir_to_rgb_text"

    def compute_losses(self, model, batch, mode=None, current_epoch=None):
        validate_rgb_ir_text_batch_dict(
            batch, model.retrieval_protocol.train_text_modalities(model.args)
        )
        context = _encode_batch(model, batch, mode)
        result = _base_result(context)
        result.update(
            model.retrieval_protocol.training_losses(
                model,
                batch,
                context.rgb_visual,
                context.rgb_feats,
                context.ir_feats,
                context.label_rgb,
                context.label_ir,
                _loss_names(model),
            )
        )
        return result


class IdentityTextRGBIRTextRecipe:
    name = "identity_text_rgb_ir_text"

    def compute_losses(self, model, batch, mode=None, current_epoch=None):
        validate_rgb_ir_text_batch_dict(
            batch, model.retrieval_protocol.train_text_modalities(model.args)
        )
        context = _encode_batch(model, batch, mode)
        losses = _loss_names(model)
        result = _base_result(context)
        b = context.batch_size
        original_rgb = context.rgb_feats[:b]
        augmented_rgb = context.rgb_feats[b:]
        text_rgb = batch["text_rgb"]
        text_map = model.base_model.encode_text(text_rgb)

        if model.args.joint_mode == "ir_crossfusion":
            if model.args.Feat_Filter:
                text_filter_feats = model.encode_text_feat(batch["text_ir"])
                text_feats = extract_text_token_feat(text_map, text_rgb)
                fusion_feats = (context.ir_feats + text_feats - text_filter_feats).squeeze()
            else:
                fusion_feats = model.fusion_layer(
                    text_map,
                    context.ir_visual,
                    text_rgb,
                    pa=model.current_pa(),
                    way=model.args.fusion_way,
                ).squeeze()
            labels = torch.cat(
                (context.label_rgb, context.label_rgb, context.label_ir), dim=0
            )
            features = torch.cat((context.rgb_feats, fusion_feats), dim=0)
            self._classification_and_wrt(model, result, losses, features, labels)
            return result

        text_feats = extract_text_token_feat(text_map, text_rgb)
        text_filter_feats = None
        if model.args.Feat_Filter:
            text_filter_feats = model.encode_text_feat(batch["text_ir"])
            ensure_matching_feature_shape(
                ir_feats=context.ir_feats,
                text_feats=text_feats,
                text_filter_feats=text_filter_feats,
            )
            fusion_feats = context.ir_feats + text_feats - text_filter_feats
        else:
            fusion_feats = model.fusion_layer(
                text_map,
                context.ir_visual,
                text_rgb,
                pa=model.current_pa(),
                way=model.args.fusion_way,
            ).squeeze()

        labels = torch.cat(
            (
                context.label_rgb,
                context.label_rgb,
                context.label_ir,
                context.label_ir,
                context.label_ir,
            ),
            dim=0,
        )
        features = torch.cat(
            (original_rgb, augmented_rgb, context.ir_feats, fusion_feats, text_feats),
            dim=0,
        )
        if "id" in losses:
            _, scores = model.classifier(features)
            result["id_loss"] = model.pid_criterion(scores, labels) * model.args.id_loss_weight
            result["acc"] = (scores.argmax(dim=1) == labels).float().mean()
        if "wrt" in losses:
            result["wrt_loss"] = model.tri_criterion(features, labels) * model.args.wrt_loss_weight

        compact_labels = torch.cat(
            (context.label_rgb, context.label_rgb, context.label_ir, context.label_ir),
            dim=0,
        )
        compact_features = torch.cat((context.rgb_feats, fusion_feats, text_feats), dim=0)
        if "id_woir" in losses:
            _, scores = model.classifier(compact_features)
            result["uni_id_woir_loss"] = (
                model.pid_criterion(scores, compact_labels) * model.args.id_loss_weight
            )
            result["acc"] = (scores.argmax(dim=1) == compact_labels).float().mean()
        if "wrt_woir" in losses:
            result["uni_wrt_woir_loss"] = (
                model.tri_criterion(compact_features, compact_labels)
                * model.args.wrt_loss_weight
            )

        text_ir_feats = None
        if any(name in losses for name in ("imta_proto", "imta_dual", "imta_rel")):
            if (
                context.label_rgb.shape != context.label_ir.shape
                or not torch.equal(context.label_rgb, context.label_ir)
            ):
                raise ValueError("IMTA requires aligned RGB/IR identity labels")
            text_ir_feats = model.encode_text_feat(batch["text_ir"]).float()
        rgb_mean = (original_rgb + augmented_rgb) * 0.5
        if "imta_proto" in losses:
            result["imta_proto_loss"] = objectives.imta_prototype_loss(
                text_feats,
                text_ir_feats,
                rgb_mean,
                context.ir_feats,
                context.label_rgb,
                temperature=float(getattr(model.args, "imta_temperature", 0.07)),
            ) * float(getattr(model.args, "imta_proto_weight", 0.25))
        if "imta_dual" in losses:
            result["imta_dual_loss"] = objectives.imta_dual_text_supcon_loss(
                text_feats,
                text_ir_feats,
                context.label_rgb,
                temperature=float(getattr(model.args, "imta_temperature", 0.07)),
            ) * float(getattr(model.args, "imta_dual_weight", 0.10))
        if "imta_rel" in losses:
            result["imta_rel_loss"] = objectives.imta_relation_loss(
                text_feats,
                text_ir_feats,
                rgb_mean,
                context.ir_feats,
                temperature=float(getattr(model.args, "imta_relation_temperature", 0.10)),
            ) * float(getattr(model.args, "imta_relation_weight", 0.10))

        if "cross_modal_hard" in losses:
            if (
                context.label_rgb.shape != context.label_ir.shape
                or not torch.equal(context.label_rgb, context.label_ir)
            ):
                raise ValueError(
                    "Stage B cross-modal hard mining requires aligned RGB and IR labels"
                )
            modalities = {
                "RGB": rgb_mean,
                "IR": context.ir_feats,
                "Fusion": fusion_feats,
                "Text": text_feats,
            }
            shapes = {name: tuple(value.shape) for name, value in modalities.items()}
            if len(set(shapes.values())) != 1:
                raise ValueError(f"Stage B modality batch shapes are not aligned: {shapes}")
            pair_losses = {}
            names = list(modalities)
            for left_index, left_name in enumerate(names):
                for right_name in names[left_index + 1 :]:
                    pair_losses[f"{left_name}-{right_name}"] = model.cross_modal_tri_criterion(
                        modalities[left_name], modalities[right_name], context.label_rgb
                    )
            cross_modal_loss = weighted_cross_modal_pair_loss(
                pair_losses, getattr(model.args, "cross_modal_pair_weights", None)
            )
            if not torch.isfinite(cross_modal_loss):
                raise FloatingPointError("Stage B cross-modal hard loss is not finite")
            result["cross_modal_hard_loss"] = cross_modal_loss * float(
                getattr(model.args, "cross_modal_hard_weight", 1.0)
            )
        if "orth" in losses:
            result["uni_orth_loss"] = objectives.orthogonal_loss(
                context.ir_feats, text_feats, text_filter_feats
            )
        if "orth2" in losses:
            result["uni_orth2_loss"] = objectives.orthogonal_loss2(
                context.ir_feats, text_feats, text_filter_feats
            )
        if "T2I_Regular" in losses:
            result["T2I_Regular_loss"] = kl_align_loss(
                context.ir_feats, fusion_feats, text_feats, context.logit_scale, mode="T2I"
            )
        if "I2T_Regular" in losses:
            result["I2T_Regular_loss"] = kl_align_loss(
                context.ir_feats, fusion_feats, text_feats, context.logit_scale, mode="I2T"
            )
        return result

    @staticmethod
    def _classification_and_wrt(model, result, losses, features, labels):
        if "id" in losses:
            _, scores = model.classifier(features)
            result["id_loss"] = model.pid_criterion(scores, labels) * model.args.id_loss_weight
            result["acc"] = (scores.argmax(dim=1) == labels).float().mean()
        if "wrt" in losses:
            result["wrt_loss"] = model.tri_criterion(features, labels) * model.args.wrt_loss_weight


class IdentityTextRGBIRRecipe:
    name = "identity_text_rgb_ir"

    def compute_losses(self, model, batch, mode=None, current_epoch=None):
        context = _encode_batch(model, batch, mode)
        losses = _loss_names(model)
        result = _base_result(context)
        labels = torch.cat(
            (context.label_rgb, context.label_rgb, context.label_ir), dim=0
        )
        features = torch.cat((context.rgb_feats, context.ir_feats), dim=0)
        IdentityTextRGBIRTextRecipe._classification_and_wrt(
            model, result, losses, features, labels
        )
        return result


_RECIPES = {
    "pmt": PMTRecipe(),
    "ir_to_rgb_text": IRToRGBTextRecipe(),
    "identity_text_rgb_ir_text": IdentityTextRGBIRTextRecipe(),
    "identity_text_rgb_ir": IdentityTextRGBIRRecipe(),
}


def build_training_recipe(config, retrieval_protocol):
    if bool(getattr(config, "pmt_recipe", False)):
        return _RECIPES["pmt"]
    name = retrieval_protocol.training_recipe(config) or str(config.training_mode)
    try:
        return _RECIPES[name]
    except KeyError as error:
        raise ValueError(f"Unsupported training recipe: {name}") from error
