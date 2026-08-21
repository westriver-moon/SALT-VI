
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import os
from salt_vi.data.dataset import process_query_sysu, process_gallery_sysu, \
    process_test_regdb,process_gallery_llcm,process_query_llcm, SYSU_Tri_Data,RegDB_Tri_Data,LLCM_Tri_Data,Test_Tri_Data
from salt_vi.data.processing import (
    ChannelAdapGray,
    ChannelExchange,
    ChannelRandomErasing,
    ChannelScale,
    MSCMChannelAdapGray,
    MSCMChannelExchange,
    MSCMChannelT,
)
from salt_vi.data.sampler import (
    GenIdx,
    IdentitySampler,
    AutoReplaceIdentitySampler,
    CameraDiverseIdentitySampler,
    validate_identity_batch_config,
)
from salt_vi.data.sysu_sources import load_train_source_records
from salt_vi.retrieval import get_retrieval_protocol
import torch.utils.data as data


SYSU_INTERPOLATION = InterpolationMode.BICUBIC


class ExactSize:
    """Validate a derived SR asset without resampling it."""

    def __init__(self, size, *, field_name="SYSU SR input"):
        self.size = tuple(int(value) for value in size)
        self.field_name = field_name

    def __call__(self, image):
        height, width = self.size
        actual = (int(image.height), int(image.width))
        if actual != (height, width):
            raise ValueError(
                f"{self.field_name} has size {actual}, expected {(height, width)}"
            )
        return image


def _with_sep(path):
    return path if path.endswith(os.sep) else path + os.sep


REQUIRED_RGB_IR_TEXT_BATCH_KEYS = (
    "img_rgb_ori",
    "img_rgb_aug",
    "img_ir",
    "target_rgb",
    "target_ir",
)


def validate_rgb_ir_text_batch_dict(batch_dict, text_modalities=("rgb", "ir")):
    required = REQUIRED_RGB_IR_TEXT_BATCH_KEYS + tuple(
        f"text_{modality}" for modality in text_modalities
    )
    missing = [key for key in required if key not in batch_dict]
    if missing:
        raise KeyError(
            "RGB_IR_Text batch_dict is missing required key(s): "
            + ", ".join(missing)
        )


def sysu_resolution_transforms(config, modality):
    """Return the explicit LR-source -> model-size resize contract.

    Derived SwinIR assets can be required to match the model size exactly.
    Every non-derived SYSU modality first passes through the configured shared
    source size before reaching the model input size.
    """
    if getattr(config, "dataset", None) != "sysu":
        return []
    source_size = getattr(config, "sysu_source_size", None)
    if source_size is None:
        return []
    sr_modalities = {str(item).lower() for item in getattr(config, "sysu_sr_modalities", [])}
    target_size = (int(config.img_h), int(config.img_w))
    if modality in sr_modalities and getattr(config, "sysu_sr_exact_size", False):
        return [ExactSize(target_size, field_name=f"SYSU {modality} SR input")]
    steps = []
    if modality not in sr_modalities:
        steps.append(transforms.Resize(
            tuple(int(value) for value in source_size), interpolation=SYSU_INTERPOLATION
        ))
    steps.append(transforms.Resize(target_size, interpolation=SYSU_INTERPOLATION))
    return steps


def build_mscmnet_exact_quadruple_transforms(
    train_size, normalize, rgb_resolution, ir_resolution
):
    """Build the original MSCMNet four-view augmentations after SR loading.

    Resolution transforms are prepended only to validate/select the already
    loaded model input. For the PASD plugin recipe these are ExactSize checks,
    so super-resolution is never treated as an augmentation operation.
    """
    color1 = transforms.Compose([
        transforms.ToPILImage(),
        *rgb_resolution,
        transforms.RandomGrayscale(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop(train_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ChannelRandomErasing(probability=0.5),
    ])
    color2 = transforms.Compose([
        transforms.ToPILImage(),
        *rgb_resolution,
        transforms.Pad(10),
        transforms.RandomCrop(train_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ChannelRandomErasing(probability=0.5),
        MSCMChannelExchange(gray=2),
    ])
    thermal1 = transforms.Compose([
        transforms.ToPILImage(),
        *ir_resolution,
        transforms.Pad(10),
        transforms.RandomCrop(train_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ChannelRandomErasing(probability=0.5),
        MSCMChannelAdapGray(probability=0.5),
    ])
    thermal2 = transforms.Compose([
        transforms.ToPILImage(),
        *ir_resolution,
        transforms.ColorJitter(brightness=0.5),
        transforms.Pad(10),
        transforms.RandomCrop(train_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ChannelRandomErasing(probability=0.5),
        MSCMChannelT(probability=0.5),
    ])
    return color1, color2, thermal1, thermal2


def build_pmt_recipe_transforms(
    train_size, normalize, rgb_resolution, ir_resolution
):
    """Build the unchanged PMT Stage-A transforms used before the switch epoch."""
    random_erasing = lambda: ChannelRandomErasing(
        probability=0.5,
        mean=[0.485, 0.456, 0.406],
    )
    thermal_mix_aug = [
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(21, sigma=(0.1, 3)),
    ]
    color1 = transforms.Compose([
        transforms.ToPILImage(),
        *(rgb_resolution or [transforms.Resize(train_size)]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        random_erasing(),
    ])
    color2 = transforms.Compose([
        transforms.ToPILImage(),
        *(rgb_resolution or [transforms.Resize(train_size)]),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize,
        random_erasing(),
    ])
    thermal1 = transforms.Compose([
        transforms.ToPILImage(),
        *(ir_resolution or [transforms.Resize(train_size)]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice(thermal_mix_aug),
        transforms.ToTensor(),
        normalize,
        random_erasing(),
    ])
    thermal2 = transforms.Compose([
        transforms.ToPILImage(),
        *(ir_resolution or [transforms.Resize(train_size)]),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        random_erasing(),
        ChannelScale(probability=0.5),
    ])
    return color1, color2, thermal1, thermal2

class Loader:

    def __init__(self, config):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_size = (config.img_h, config.img_w)
        rgb_resolution = sysu_resolution_transforms(config, "rgb")
        ir_resolution = sysu_resolution_transforms(config, "ir")
        self.quadruple_input = (
            str(getattr(config, "visual_input_backend", "single")).lower()
            == "quadruple_patch"
        )
        self.pmt_recipe_variant = str(
            getattr(config, "pmt_recipe_variant", "original") or "original"
        ).lower()
        self.phased_mscm_recipe = self.pmt_recipe_variant == "mscm_phased"
        self.pmt_progressive_epoch = int(
            getattr(config, "pmt_progressive_epoch", 6)
        )
        self.phased_mscm_transforms = None

        if self.phased_mscm_recipe:
            (
                self.transform_color1,
                self.transform_color2,
                self.transform_thermal1,
                self.transform_thermal2,
            ) = build_pmt_recipe_transforms(
                train_size, normalize, rgb_resolution, ir_resolution
            )
            self.transform_thermal2 = None
            self.phased_mscm_transforms = build_mscmnet_exact_quadruple_transforms(
                train_size, normalize, rgb_resolution, ir_resolution
            )
        elif self.quadruple_input:
            (
                self.transform_color1,
                self.transform_color2,
                self.transform_thermal1,
                self.transform_thermal2,
            ) = build_mscmnet_exact_quadruple_transforms(
                train_size, normalize, rgb_resolution, ir_resolution
            )
        elif getattr(config, "pmt_recipe_transforms", False):
            (
                self.transform_color1,
                self.transform_color2,
                self.transform_thermal1,
                self.transform_thermal2,
            ) = build_pmt_recipe_transforms(
                train_size, normalize, rgb_resolution, ir_resolution
            )
        else:
            self.transform_color1 = transforms.Compose( [
                transforms.ToPILImage(),
                *rgb_resolution,
                transforms.Pad(10),
                transforms.RandomCrop(train_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p = 0.1),
                transforms.ToTensor(),
                normalize,
                ChannelRandomErasing(probability = 0.6)])

            self.transform_color2 = transforms.Compose( [
                transforms.ToPILImage(),
                *rgb_resolution,
                transforms.Pad(10),
                transforms.RandomCrop(train_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ChannelRandomErasing(probability = 0.6),
                ChannelExchange(gray = 2)])

            self.transform_thermal1 = transforms.Compose([
                transforms.ToPILImage(),
                *ir_resolution,
                transforms.Pad(10),
                transforms.RandomCrop(train_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ChannelRandomErasing(probability=0.5),
                ChannelAdapGray(probability=0.6)])
            self.transform_thermal2 = transforms.Compose([
                transforms.ToPILImage(),
                *ir_resolution,
                transforms.Pad(10),
                transforms.RandomCrop(train_size),
                transforms.ColorJitter(brightness=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ChannelRandomErasing(probability=0.5),
                ChannelScale(probability=0.5),
            ])

        # Preserve every existing single-input dataset and recipe unchanged.
        self.transform_thermal = self.transform_thermal1


        all_sysu_eval_modalities_are_exact_sr = (
            config.dataset == "sysu"
            and bool(getattr(config, "sysu_sr_exact_size", False))
            and {"rgb", "ir"}.issubset(
                {str(item).lower() for item in getattr(config, "sysu_sr_modalities", [])}
            )
        )
        if all_sysu_eval_modalities_are_exact_sr:
            test_resize = ExactSize((config.img_h, config.img_w), field_name="SYSU evaluation SR input")
        else:
            test_resize = transforms.Resize(
                (config.img_h, config.img_w),
                interpolation=(
                    SYSU_INTERPOLATION
                    if config.dataset == "sysu" and getattr(config, "sysu_source_size", None) is not None
                    else InterpolationMode.BILINEAR
                ),
            )
        self.transform_test = transforms.Compose([
            transforms.ToPILImage(),
            test_resize,
            transforms.ToTensor(),
            normalize])

        # dataset name and path
        self.dataset = config.dataset
        self.sysu_data_path = _with_sep(config.sysu_data_path)

        self.llcm_data_path = _with_sep(config.llcm_data_path)

        if config.dataset == 'regdb':
            self.regdb_data_path = _with_sep(config.regdb_data_path)
            self.trial = int(config.trial)
            self.eval_num_regdb = int(config.eval_num_regdb)
            if not 1 <= self.trial <= 10:
                raise ValueError(f"RegDB trial must be in [1, 10], got {self.trial}")
            if self.eval_num_regdb < 1 or self.trial + self.eval_num_regdb - 1 > 10:
                raise ValueError(
                    "RegDB evaluation trials must be a positive consecutive range "
                    f"within [1, 10], got start={self.trial}, count={self.eval_num_regdb}"
                )
            self.regdb_trials = list(
                range(self.trial, self.trial + self.eval_num_regdb)
            )

        # image size
        self.img_w = config.img_w
        self.img_h = config.img_h

        # number of positive identities
        self.num_pos = config.num_pos
        self.sampler_type = getattr(config, "sampler_type", "identity_current_replace")

        # batch size
        self.batch_size = config.batch_size
        self.test_batch_size = int(getattr(config, "test_batch_size", 128))
        if self.test_batch_size < 1:
            raise ValueError(f"test_batch_size must be positive, got {self.test_batch_size}")

        # model setting
        self.mode = config.mode
        self.test_mode = config.test_mode
        self.gall_mode = config.gall_mode
        self.gallery_trials = int(getattr(config, "gallery_trials", 10))
        if self.gallery_trials < 1:
            raise ValueError(
                f"gallery_trials must be positive, got {self.gallery_trials}"
            )
        self.num_workers = config.num_workers
        self.seed = int(getattr(config, "seed", 0))
        self.eval_caption_seed = int(getattr(config, "eval_caption_seed", 0))
        if not 0 <= self.eval_caption_seed <= 2**32 - 1:
            raise ValueError(
                f"eval_caption_seed must be in [0, 2**32 - 1], got {self.eval_caption_seed}"
            )
        self.training_mode = config.training_mode
        self.test_modality = config.test_modality
        self.retrieval_protocol = get_retrieval_protocol(
            getattr(config, "retrieval_backend", "identity_text")
        )
        self.use_train_text = "Text" in self.training_mode
        self.train_text_modalities = self.retrieval_protocol.train_text_modalities(config)
        self.query_caption_lookup = self.retrieval_protocol.query_caption_lookup(config)
        self.gallery_caption_lookup = self.retrieval_protocol.gallery_caption_lookup(config)
        self.use_eval_text = bool(
            self.query_caption_lookup or self.gallery_caption_lookup
        )
        self.joint_mode = config.joint_mode if self.use_train_text else "image_only"

        # nlp augmentation setting
        self.Feat_Filter = config.Feat_Filter if (self.use_train_text or self.use_eval_text) else False
        self.captioner_name = config.captioner_name
        self.text_data_root = getattr(config, "text_data_root", None)
        self.gallery_caption_manifest = getattr(config, "gallery_caption_manifest", None)
        self.sysu_sr_data_root = getattr(config, "sysu_sr_data_root", None)
        self.sysu_sr_modalities = getattr(config, "sysu_sr_modalities", [])
        self.sysu_source_size = getattr(config, "sysu_source_size", None)
        self.sysu_sr_exact_size = bool(getattr(config, "sysu_sr_exact_size", False))
        self.sysu_sr_backend = getattr(config, "sysu_sr_backend", "array")
        self.sysu_sr_view_manifest = getattr(config, "sysu_sr_view_manifest", None)
        self.sysu_sr_views_per_image = int(getattr(config, "sysu_sr_views_per_image", 1))
        self.sysu_sr_view_sampling = getattr(config, "sysu_sr_view_sampling", "independent")
        self.sysu_sr_eval_view_index = int(getattr(config, "sysu_sr_eval_view_index", 0))
        self.llm_aug = config.llm_aug
        self.llm_aug_prob = config.llm_aug_prob
        if "Text" in config.training_mode:
            print(f"Training With Text Generated From: {config.captioner_name}\n Traininig Mode: {config.training_mode}")

        # form dataloader
        self._loader()

    def _loader(self):
        if self.dataset == 'sysu':
            if self.mode == 'train':
                # train sysu data simples
                samples = SYSU_Tri_Data(self.sysu_data_path, transform1=self.transform_color1, transform2=self.transform_color2,
                                transform3=self.transform_thermal1,
                                transform4=self.transform_thermal2 if self.quadruple_input else None,\
                                        phased_transforms=self.phased_mscm_transforms,\
                                        llm_aug_prob=self.llm_aug_prob,\
                                                llm_aug=self.llm_aug,captioner_name=self.captioner_name,\
                                                    joint_mode=self.joint_mode,\
                                                        Feat_Filter=self.Feat_Filter,
                                                        text_data_root=self.text_data_root,
                                                        sysu_sr_data_root=self.sysu_sr_data_root,
                                                        sysu_sr_modalities=self.sysu_sr_modalities,
                                                        sysu_sr_backend=self.sysu_sr_backend,
                                                        sysu_sr_view_manifest=self.sysu_sr_view_manifest,
                                                        sysu_sr_views_per_image=self.sysu_sr_views_per_image,
                                                        sysu_sr_view_sampling=self.sysu_sr_view_sampling,
                                                        text_modalities=self.train_text_modalities)
                self.color_pos, self.thermal_pos = GenIdx(samples.train_color_label, samples.train_thermal_label)
                if self.sampler_type == "identity_camera_diverse":
                    rgb_records = load_train_source_records(self.sysu_data_path, "rgb")
                    ir_records = load_train_source_records(self.sysu_data_path, "ir")
                    if [record.label for record in rgb_records] != samples.train_color_label.tolist():
                        raise ValueError("SYSU RGB camera manifest does not match training labels")
                    if [record.label for record in ir_records] != samples.train_thermal_label.tolist():
                        raise ValueError("SYSU IR camera manifest does not match training labels")
                    self.color_cameras = [record.camera for record in rgb_records]
                    self.thermal_cameras = [record.camera for record in ir_records]
                self.samples = samples

            # test sysu data simples
            query_samples, gallery_samples_list = self._get_test_samples(self.dataset)
            query_loader = data.DataLoader(query_samples, batch_size=self.test_batch_size, shuffle=False, drop_last=False,
                                                num_workers=self.num_workers)
            gallery_loaders = []
            for i in range(self.gallery_trials):
                gallery_loader = data.DataLoader(gallery_samples_list[i], batch_size=self.test_batch_size, shuffle=False,
                                                 drop_last=False, num_workers=self.num_workers)
                gallery_loaders.append(gallery_loader)
            self.query_loader = query_loader
            self.gallery_loaders = gallery_loaders

        elif self.dataset == 'regdb':
            if self.mode == 'train':
                samples = RegDB_Tri_Data(self.regdb_data_path, trial=self.trial, transform1=self.transform_color1, transform2=self.transform_color2,
                                transform3=self.transform_thermal,\
                                        llm_aug_prob=self.llm_aug_prob,\
                                                llm_aug=self.llm_aug,captioner_name=self.captioner_name,\
                                                    joint_mode=self.joint_mode,\
                                                        Feat_Filter=self.Feat_Filter,
                                                        text_data_root=self.text_data_root)
                self.color_pos, self.thermal_pos = GenIdx(samples.train_color_label, samples.train_thermal_label)
                self.samples = samples


            query_samples_list, gallery_samples_list = self._get_test_samples(self.dataset)
            query_loaders = []
            for i in range(self.eval_num_regdb):
                query_loader = data.DataLoader(query_samples_list[i], batch_size=self.test_batch_size, shuffle=False, drop_last=False,
                                                    num_workers=self.num_workers)
                query_loaders.append(query_loader)
            self.query_loaders = query_loaders

            gallery_loaders = []
            for i in range(self.eval_num_regdb):
                gallery_loader = data.DataLoader(gallery_samples_list[i], batch_size=self.test_batch_size, shuffle=False, drop_last=False,
                                             num_workers=self.num_workers)
                gallery_loaders.append(gallery_loader)
            self.gallery_loaders = gallery_loaders

        elif self.dataset == 'llcm':
            if self.mode == 'train':
                samples = LLCM_Tri_Data(self.llcm_data_path, transform1=self.transform_color1, transform2=self.transform_color2,
                                transform3=self.transform_thermal,\
                                        llm_aug_prob=self.llm_aug_prob,\
                                                llm_aug=self.llm_aug,captioner_name=self.captioner_name,\
                                                    joint_mode=self.joint_mode,\
                                                        Feat_Filter=self.Feat_Filter,
                                                        text_data_root=self.text_data_root)
                self.color_pos, self.thermal_pos = GenIdx(samples.train_color_label, samples.train_thermal_label)
                self.samples = samples

            query_samples, gallery_samples_list = self._get_test_samples(self.dataset)
            query_loader = data.DataLoader(query_samples, batch_size=self.test_batch_size, shuffle=False, drop_last=False,
                                                num_workers=self.num_workers)
            gallery_loaders = []
            for i in range(self.gallery_trials):
                gallery_loader = data.DataLoader(gallery_samples_list[i], batch_size=self.test_batch_size, shuffle=False, drop_last=False,
                                             num_workers=self.num_workers)
                gallery_loaders.append(gallery_loader)
            self.query_loader = query_loader
            self.gallery_loaders = gallery_loaders

    def _get_test_samples(self, dataset):
        if dataset == 'sysu':
            query_img, query_label, query_cam = process_query_sysu(self.sysu_data_path, mode=self.test_mode)
            query_samples = Test_Tri_Data(query_img, query_label, transform=self.transform_test,
                                     img_size=(self.img_w, self.img_h), data_path=self.sysu_data_path,\
                                            captioner_name=self.captioner_name, joint_mode=self.joint_mode,gallorquery='query',\
                                            Feat_Filter=self.Feat_Filter,
                                            load_text=self.query_caption_lookup is not None,
                                            caption_lookup=self.query_caption_lookup or "identity",
                                            text_data_root=self.text_data_root,
                                            sysu_source_size=self.sysu_source_size,
                                            sysu_sr_exact_size=self.sysu_sr_exact_size,
                                            sysu_sr_data_root=self.sysu_sr_data_root,
                                            sysu_sr_modalities=self.sysu_sr_modalities,
                                            sysu_sr_backend=self.sysu_sr_backend,
                                            sysu_sr_view_manifest=self.sysu_sr_view_manifest,
                                            sysu_sr_views_per_image=self.sysu_sr_views_per_image,
                                            sysu_sr_eval_view_index=self.sysu_sr_eval_view_index,
                                            source_modality="ir", caption_seed=self.eval_caption_seed)
            self.query_label = query_label
            self.query_cam = query_cam

            self.n_query = len(query_label)

            gallery_samples_list = []
            self.gallery_labels = []
            self.gallery_cams = []
            for i in range(self.gallery_trials):
                gall_img, gall_label, gall_cam = process_gallery_sysu(self.sysu_data_path, mode=self.test_mode, trial=i,
                                                                      gall_mode=self.gall_mode)
                self.gall_cam = gall_cam
                self.gall_label = gall_label
                self.n_gallery = len(gall_label)
                self.gallery_labels.append(gall_label)
                self.gallery_cams.append(gall_cam)

                gallery_samples = Test_Tri_Data(gall_img, gall_label,data_path=self.sysu_data_path,transform=self.transform_test,
                                        img_size=(self.img_w, self.img_h), captioner_name=self.captioner_name,
                                        joint_mode=self.joint_mode,gallorquery=f'gall[{i+1}]',
                                        Feat_Filter=self.Feat_Filter,
                                        load_text=self.gallery_caption_lookup is not None,
                                        caption_lookup=self.gallery_caption_lookup or "identity",
                                        caption_manifest=self.gallery_caption_manifest,
                                        text_data_root=self.text_data_root,
                                        sysu_source_size=self.sysu_source_size,
                                        sysu_sr_exact_size=self.sysu_sr_exact_size,
                                        sysu_sr_data_root=self.sysu_sr_data_root,
                                        sysu_sr_modalities=self.sysu_sr_modalities,
                                        sysu_sr_backend=self.sysu_sr_backend,
                                        sysu_sr_view_manifest=self.sysu_sr_view_manifest,
                                        sysu_sr_views_per_image=self.sysu_sr_views_per_image,
                                        sysu_sr_eval_view_index=self.sysu_sr_eval_view_index,
                                        source_modality="rgb",
                                        caption_seed=self.eval_caption_seed)
                gallery_samples_list.append(gallery_samples)
            return query_samples, gallery_samples_list
        elif self.dataset == 'regdb':
            query_samples_list = []
            self.query_labels = []
            self.query_sizes = []
            for trial in self.regdb_trials:
                query_img, query_label = process_test_regdb(self.regdb_data_path, trial=trial, modal='thermal')
                self.query_labels.append(query_label)
                self.query_sizes.append(len(query_label))
                query_samples = Test_Tri_Data(query_img, query_label, transform=self.transform_test,
                                        img_size=(self.img_w, self.img_h), data_path=self.regdb_data_path,\
                                            captioner_name=self.captioner_name, \
                                                joint_mode=self.joint_mode,gallorquery=f'query[{trial}]',\
                                                Feat_Filter=self.Feat_Filter, load_text=self.use_eval_text,
                                                text_data_root=self.text_data_root, caption_seed=self.eval_caption_seed)
                query_samples_list.append(query_samples)

            gallery_samples_list = []
            self.gallery_labels = []
            self.gallery_sizes = []
            for trial in self.regdb_trials:
                gall_img, gall_label = process_test_regdb(self.regdb_data_path, trial=trial, modal='visible')
                self.gallery_labels.append(gall_label)
                self.gallery_sizes.append(len(gall_label))

                gallery_samples = Test_Tri_Data(gall_img, gall_label,data_path=self.regdb_data_path,transform=self.transform_test,
                                            img_size=(self.img_w, self.img_h), captioner_name=self.captioner_name,\
                                                joint_mode=self.joint_mode,gallorquery=f'gall[{trial}]',
                                                Feat_Filter=self.Feat_Filter, load_text=False,
                                                text_data_root=self.text_data_root)
                gallery_samples_list.append(gallery_samples)
            # Preserve the legacy single-trial attributes for external callers.
            self.query_label = self.query_labels[0]
            self.gall_label = self.gallery_labels[0]
            self.n_query = self.query_sizes[0]
            self.n_gallery = self.gallery_sizes[0]
            return query_samples_list, gallery_samples_list
        elif self.dataset == 'llcm':
            query_img, query_label, query_cam = process_query_llcm(self.llcm_data_path, mode=2) # nir
            query_samples = Test_Tri_Data(query_img, query_label, transform=self.transform_test,
                                     img_size=(self.img_w, self.img_h), data_path=self.llcm_data_path,\
                                        captioner_name=self.captioner_name, \
                                            joint_mode=self.joint_mode,gallorquery='query',\
                                                Feat_Filter=self.Feat_Filter, load_text=self.use_eval_text,
                                                text_data_root=self.text_data_root, caption_seed=self.eval_caption_seed)
            self.query_label = query_label
            self.query_cam = query_cam

            self.n_query = len(query_label)

            gallery_samples_list = []
            self.gallery_labels = []
            self.gallery_cams = []
            for i in range(self.gallery_trials):
                gall_img, gall_label, gall_cam = process_gallery_llcm(self.llcm_data_path, mode=1, trial=i) # vis

                self.gall_cam = gall_cam
                self.gall_label = gall_label
                self.n_gallery = len(gall_label)
                self.gallery_labels.append(gall_label)
                self.gallery_cams.append(gall_cam)

                gallery_samples = Test_Tri_Data(gall_img, gall_label,data_path=self.llcm_data_path,transform=self.transform_test,
                                            img_size=(self.img_w, self.img_h), captioner_name=self.captioner_name,\
                                                joint_mode=self.joint_mode,gallorquery=f'gall[{i+1}]',
                                                Feat_Filter=self.Feat_Filter, load_text=False,
                                                text_data_root=self.text_data_root)
                gallery_samples_list.append(gallery_samples)
            return query_samples, gallery_samples_list
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")


    def set_training_epoch(self, current_epoch):
        self.current_training_epoch = 0 if current_epoch is None else int(current_epoch)

    def get_train_loader(self):
        if self.phased_mscm_recipe:
            epoch = int(getattr(self, "current_training_epoch", 0))
            phase = "pmt" if epoch < self.pmt_progressive_epoch else "mscm"
            self.samples.set_training_phase(phase)
        if self.sampler_type == "identity_current_replace":
            sampler_cls = IdentitySampler
        elif self.sampler_type == "identity_auto_replace":
            sampler_cls = AutoReplaceIdentitySampler
        elif self.sampler_type == "identity_camera_diverse":
            if self.dataset != "sysu":
                raise ValueError("camera-diverse sampling is currently defined only for SYSU")
            sampler_cls = CameraDiverseIdentitySampler
        else:
            raise ValueError(f"Unsupported sampler_type: {self.sampler_type}")

        identities_per_batch = validate_identity_batch_config(
            self.batch_size, self.num_pos, len(self.color_pos)
        )
        sampler_args = [
            self.samples.train_color_label,
            self.samples.train_thermal_label,
            self.color_pos,
            self.thermal_pos,
            self.num_pos,
            identities_per_batch,
        ]
        if sampler_cls is CameraDiverseIdentitySampler:
            sampler_args.extend((self.color_cameras, self.thermal_cameras))
        sampler = sampler_cls(*sampler_args)
        self.samples.cIndex = sampler.index1
        self.samples.tIndex = sampler.index2
        train_loader = data.DataLoader(self.samples, batch_size=self.batch_size,
                                       sampler=sampler, num_workers=self.num_workers, drop_last=True)
        return train_loader
