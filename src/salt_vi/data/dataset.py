import json
import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
from .tokenizer import SimpleTokenizer
from .pasd_multiview import (
    PASDTrainViewStore,
    eval_view_path,
    eval_views,
    normalize_backend,
    normalize_sampling,
)
from .sources import (
    ArrayCaptionSource,
    ArrayVisualSource,
    MultiviewCaptionSource,
    MultiviewVisualSource,
    NoCaptionSource,
)


PIL_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
PIL_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]
    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_text_dir(data_dir, dataset_name, captioner_name, modality, text_data_root=None):
    candidates = [
        os.path.join(data_dir, "Text", f"{captioner_name}_{modality}"),
        os.path.join(_repo_root(), "datasets", dataset_name, "Text", f"{captioner_name}_{modality}"),
    ]
    if text_data_root:
        candidates.insert(0, os.path.join(text_data_root, f"{captioner_name}_{modality}"))
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate if candidate.endswith(os.sep) else candidate + os.sep
    raise FileNotFoundError(
        f"Unable to locate text directory for {dataset_name}/{captioner_name}_{modality}. "
        f"Tried: {candidates}"
    )


def _resolve_caption_path(
    data_dir,
    dataset_name,
    captioner_name,
    modality,
    caption_lookup,
    text_data_root=None,
    caption_manifest=None,
):
    if caption_manifest:
        return caption_manifest
    text_dir = _resolve_text_dir(
        data_dir, dataset_name, captioner_name, modality, text_data_root
    )
    caption_file = {
        "identity": f"id_caption_map_{captioner_name}_{modality}.json",
        "image": f"caption_dict_{captioner_name}_{modality}.json",
    }[caption_lookup]
    return os.path.join(text_dir, caption_file)


def _infer_dataset_name(data_path):
    normalized = data_path.lower().replace("\\", "/").rstrip("/")
    if "sysu-mm01" in normalized or normalized.endswith("/sysu"):
        return "sysu"
    if "regdb" in normalized:
        return "regdb"
    if "llcm" in normalized:
        return "llcm"
    raise ValueError(f"Unable to infer dataset name from data path: {data_path}")


def _lookup_text_description(text_dict, dataset_name, data_path, image_path):
    normalized_path = image_path.replace("\\", "/")
    normalized_root = data_path.replace("\\", "/").rstrip("/") + "/"
    candidates = [normalized_path]

    if normalized_path.startswith(normalized_root):
        relative_path = normalized_path[len(normalized_root) :]
        candidates.append(f"datasets/{dataset_name}/{relative_path}")

    if dataset_name == "sysu" and "SYSU-MM01/" in normalized_path:
        relative_path = normalized_path.split("SYSU-MM01/", 1)[1]
        candidates.append(f"datasets/sysu/{relative_path}")

    if dataset_name == "llcm" and "test_nir" in normalized_path:
        candidates.append(
            normalized_root + "nir/" + normalized_path.replace("test_nir", "nir").split("cam")[1][2:]
        )

    for candidate in candidates:
        if candidate in text_dict:
            return text_dict[candidate]["description"]

    raise KeyError(
        f"Unable to find text description for {image_path}. Tried keys: {candidates}"
    )


_SYSU_SR_TRAIN_ARRAYS = {
    "rgb": "train_rgb_swinir_x2_img.npy",
    "ir": "train_ir_swinir_x2_img.npy",
}


def normalize_sysu_sr_modalities(modalities):
    if modalities is None:
        return frozenset()
    if isinstance(modalities, str):
        modalities = [item.strip() for item in modalities.split(",") if item.strip()]
    normalized = frozenset(str(item).lower() for item in modalities)
    invalid = normalized.difference(_SYSU_SR_TRAIN_ARRAYS)
    if invalid:
        raise ValueError(f"Unsupported SYSU super-resolution modalities: {sorted(invalid)}")
    return normalized


def _sysu_train_image_path(data_dir, sr_data_root, sr_modalities, modality):
    if modality not in sr_modalities:
        filename = "train_rgb_resized_img.npy" if modality == "rgb" else "train_ir_resized_img.npy"
        return os.path.join(data_dir, filename)
    if not sr_data_root:
        raise ValueError(f"sysu_sr_data_root is required when SYSU {modality} super-resolution is enabled")
    path = os.path.join(sr_data_root, _SYSU_SR_TRAIN_ARRAYS[modality])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing SYSU {modality} super-resolution array: {path}")
    return path


def _sysu_eval_image_path(
    image_path,
    data_path,
    sr_data_root,
    sr_modalities,
    modality,
    sr_backend="array",
    sr_view_manifest=None,
    sr_views_per_image=1,
    sr_eval_view_index=0,
):
    if modality not in sr_modalities:
        return image_path
    if not sr_data_root:
        raise ValueError(f"sysu_sr_data_root is required when SYSU {modality} super-resolution is enabled")
    if normalize_backend(sr_backend) == "pasd_multiview":
        if not sr_view_manifest:
            raise ValueError("sysu_sr_view_manifest is required for PASD multiview evaluation")
        return eval_view_path(
            image_path,
            data_path,
            sr_data_root,
            sr_view_manifest,
            sr_eval_view_index,
            sr_views_per_image,
        )
    relative_path = os.path.relpath(image_path, data_path)
    if relative_path.startswith(".."):
        raise ValueError(f"SYSU evaluation image is outside the dataset root: {image_path}")
    path = os.path.join(sr_data_root, "eval", relative_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing SYSU {modality} super-resolution evaluation image: {path}")
    return path


def _sysu_eval_image_paths(
    image_path,
    data_path,
    sr_data_root,
    sr_modalities,
    modality,
    sr_backend,
    sr_view_manifest,
    sr_views_per_image,
    sr_eval_mode,
    sr_eval_top_k,
    sr_eval_view_index,
):
    if str(sr_eval_mode).lower() != "marginalize":
        return [
            _sysu_eval_image_path(
                image_path,
                data_path,
                sr_data_root,
                sr_modalities,
                modality,
                sr_backend=sr_backend,
                sr_view_manifest=sr_view_manifest,
                sr_views_per_image=sr_views_per_image,
                sr_eval_view_index=sr_eval_view_index,
            )
        ], [1.0]
    if modality not in sr_modalities or normalize_backend(sr_backend) != "pasd_multiview":
        return [image_path], [1.0]
    return eval_views(
        image_path,
        data_path,
        sr_data_root,
        sr_view_manifest,
        sr_eval_top_k,
        sr_views_per_image,
    )

def _build_sysu_visual_source(
    data_dir,
    sr_data_root,
    sr_modalities,
    sr_backend,
    sr_manifest,
    views,
    modality,
    labels,
):
    if sr_backend == "pasd_multiview" and modality in sr_modalities:
        store = PASDTrainViewStore(
            data_dir,
            sr_data_root,
            sr_manifest,
            modality,
            labels,
            views,
        )
        return MultiviewVisualSource(store, views)
    path = _sysu_train_image_path(data_dir, sr_data_root, sr_modalities, modality)
    return ArrayVisualSource(path)


def _build_sysu_caption_source(
    data_dir,
    modality,
    visual_source,
    text_modalities,
    captioner_name,
    text_data_root,
    tokenizer,
    llm_aug,
    llm_aug_prob,
    views,
    sampling,
):
    if modality not in text_modalities:
        return NoCaptionSource()
    if isinstance(visual_source, MultiviewVisualSource):
        return MultiviewCaptionSource(
            visual_source.store,
            views,
            sampling,
            lambda caption: tokenize(caption, tokenizer),
        )
    modality_name = modality.upper()
    text_dir = _resolve_text_dir(
        data_dir, "sysu", captioner_name, modality_name, text_data_root
    )
    captions = [
        tokenize(caption, tokenizer)
        for caption in np.load(
            text_dir + f"train_text_{captioner_name}_{modality_name}.npy"
        )
    ]
    augmented = None
    if llm_aug:
        augmented = [
            tokenize(caption, tokenizer)
            for caption in np.load(
                text_dir + f"train_llm_text_{captioner_name}_{modality_name}.npy"
            )
        ]
    return ArrayCaptionSource(captions, augmented, llm_aug_prob)


class SYSU_Tri_Data(data.Dataset):
    def __init__(
        self,
        data_dir,
        transform1=None,
        transform2=None,
        transform3=None,
        colorIndex=None,
        thermalIndex=None,
        text_length=77,
        llm_aug_prob=0.6,
        llm_aug=False,
        captioner_name="GIT",
        joint_mode="ir_crossfusion",
        Feat_Filter=False,
        text_data_root=None,
        sysu_sr_data_root=None,
        sysu_sr_modalities=None,
        sysu_sr_backend="array",
        sysu_sr_view_manifest=None,
        sysu_sr_views_per_image=1,
        sysu_sr_view_sampling="independent",
        text_modalities=("rgb", "ir"),
    ):
        self.tokenizer = SimpleTokenizer()
        self.joint_mode = joint_mode
        self.Feat_Filter = Feat_Filter
        self.text_length = text_length
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

        modalities = normalize_sysu_sr_modalities(sysu_sr_modalities)
        backend = normalize_backend(sysu_sr_backend)
        sampling = normalize_sampling(sysu_sr_view_sampling)
        views = int(sysu_sr_views_per_image)
        if backend == "pasd_multiview" and (
            not modalities or not sysu_sr_data_root or not sysu_sr_view_manifest
        ):
            raise ValueError("PASD multiview training requires modalities, data root, and manifest")

        self.train_color_label = np.load(data_dir + "train_rgb_resized_label.npy")
        self.train_thermal_label = np.load(data_dir + "train_ir_resized_label.npy")
        self.rgb_visual_source = _build_sysu_visual_source(
            data_dir,
            sysu_sr_data_root,
            modalities,
            backend,
            sysu_sr_view_manifest,
            views,
            "rgb",
            self.train_color_label,
        )
        self.ir_visual_source = _build_sysu_visual_source(
            data_dir,
            sysu_sr_data_root,
            modalities,
            backend,
            sysu_sr_view_manifest,
            views,
            "ir",
            self.train_thermal_label,
        )
        if len(self.rgb_visual_source) != len(self.train_color_label):
            raise ValueError("SYSU RGB image count does not match train labels")
        if len(self.ir_visual_source) != len(self.train_thermal_label):
            raise ValueError("SYSU IR image count does not match train labels")

        text_modalities = (
            frozenset(text_modalities)
            if joint_mode in ("ir_crossfusion", "uni")
            else frozenset()
        )
        self.rgb_caption_source = _build_sysu_caption_source(
            data_dir,
            "rgb",
            self.rgb_visual_source,
            text_modalities,
            captioner_name,
            text_data_root,
            self.tokenizer,
            llm_aug,
            llm_aug_prob,
            views,
            sampling,
        )
        self.ir_caption_source = _build_sysu_caption_source(
            data_dir,
            "ir",
            self.ir_visual_source,
            text_modalities,
            captioner_name,
            text_data_root,
            self.tokenizer,
            llm_aug,
            llm_aug_prob,
            views,
            sampling,
        )

    def __getitem__(self, index):
        color_index = int(self.cIndex[index])
        thermal_index = int(self.tIndex[index])
        rgb_image, rgb_view = self.rgb_visual_source.sample(color_index)
        ir_image, ir_view = self.ir_visual_source.sample(thermal_index)
        batch = {
            "img_rgb_ori": self.transform1(rgb_image),
            "img_rgb_aug": self.transform2(rgb_image),
            "img_ir": self.transform3(ir_image),
            "target_rgb": self.train_color_label[color_index],
            "target_ir": self.train_thermal_label[thermal_index],
        }
        if self.joint_mode in ("ir_crossfusion", "uni"):
            rgb_caption = self.rgb_caption_source.sample(color_index, rgb_view)
            ir_caption = self.ir_caption_source.sample(thermal_index, ir_view)
            if rgb_caption is not None:
                batch["text_rgb"] = rgb_caption
            if ir_caption is not None:
                batch["text_ir"] = ir_caption
        return batch

    def __len__(self):
        return len(self.train_color_label)

class RegDB_Tri_Data(data.Dataset):
    def __init__(self, data_dir, trial,transform1=None, \
                 transform2=None, transform3=None, \
                    colorIndex=None, thermalIndex=None, \
                            text_length=77, llm_aug_prob=0.5,\
                                    llm_aug=False, captioner_name='Blip', joint_mode="ir_crossfusion", \
                                        Feat_Filter=False, text_data_root=None):
        # initialize text tokenizer
        self.tokenizer = SimpleTokenizer()

        # init text option
        self.Feat_Filter = Feat_Filter
        self.joint_mode = joint_mode
        self.text_length = text_length
        self.llm_aug = llm_aug
        self.llm_aug_prob = llm_aug_prob

        # Load RGB&IR training data
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), PIL_LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), PIL_LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # RGB format
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # IR format
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        # Load text data
        if joint_mode == "ir_crossfusion" or joint_mode == "uni":
            print("Loading RGB Text For Training...")
            self.text_dir_rgb = _resolve_text_dir(
                data_dir, 'regdb', captioner_name, 'RGB', text_data_root
            )
            self.text_rgb_dict = json.load(open(self.text_dir_rgb + f'caption_llm_dict_{captioner_name}_RGB.json'))
            self.train_text_rgb = [tokenize(self.text_rgb_dict[data_dir + i_path]['description'], self.tokenizer) for i_path in color_img_file]
            if llm_aug:
                self.llm_text_rgb = [tokenize(self.text_rgb_dict[data_dir + i_path]['aug_description'], self.tokenizer) for i_path in color_img_file]

            if joint_mode == "ir_crossfusion" or joint_mode == "uni":
                print("Loading IR Text For Training...")
                self.text_dir_ir = _resolve_text_dir(
                    data_dir, 'regdb', captioner_name, 'IR', text_data_root
                )
                self.text_ir_dict = json.load(open(self.text_dir_ir + f'caption_llm_dict_{captioner_name}_IR.json'))
                self.train_text_ir = [tokenize(self.text_ir_dict[data_dir + i_path]['description'], self.tokenizer) for i_path in thermal_img_file]
                if llm_aug:
                    self.llm_text_ir = [tokenize(self.text_ir_dict[data_dir + i_path]['aug_description'], self.tokenizer) for i_path in thermal_img_file]


        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        batch_dict = {}

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1_0 = self.transform1(img1) # color image
        img1_1 = self.transform2(img1) # color image with augmentation
        img2 = self.transform3(img2)

        if self.joint_mode == "ir_crossfusion" or self.joint_mode == "uni":
            caption_id_rgb = self.train_text_rgb[self.cIndex[index]]
            if self.llm_aug and random.random() < self.llm_aug_prob:
                    caption_id_rgb = self.llm_text_rgb[self.cIndex[index]]
            batch_dict['text_rgb'] = caption_id_rgb

            caption_id_ir = self.train_text_ir[self.tIndex[index]]
            if self.llm_aug and random.random() < self.llm_aug_prob:
                caption_id_ir = self.llm_text_ir[self.tIndex[index]]
            batch_dict['text_ir'] = caption_id_ir

        batch_dict['img_rgb_ori'] = img1_0
        batch_dict['img_rgb_aug'] = img1_1
        batch_dict['img_ir'] = img2
        batch_dict['target_rgb'] = target1
        batch_dict['target_ir'] = target2

        return batch_dict

    def __len__(self):
        return len(self.train_color_label)


class LLCM_Tri_Data(data.Dataset):
    def __init__(self, data_dir, transform1=None, \
                 transform2=None, transform3=None, \
                    colorIndex=None, thermalIndex=None, \
                            text_length=77, llm_aug_prob=0.5,\
                                    llm_aug=False, captioner_name='Blip', joint_mode="ir_crossfusion", \
                                        Feat_Filter=False, text_data_root=None):
        # initialize text tokenizer
        self.tokenizer = SimpleTokenizer()

        # init text option
        self.Feat_Filter = Feat_Filter
        self.joint_mode = joint_mode
        self.text_length = text_length
        self.llm_aug = llm_aug
        self.llm_aug_prob = llm_aug_prob

        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_vis.txt'
        train_thermal_list = data_dir + 'idx/train_nir.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), PIL_LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), PIL_LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            #print(pix_array.shape)
        train_thermal_image = np.array(train_thermal_image)

        # RGB format
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # IR format
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        # Load text data
        if joint_mode == "ir_crossfusion" or joint_mode == "uni":
            print("Loading RGB Text For Training...")
            self.text_dir_rgb = _resolve_text_dir(
                data_dir, 'llcm', captioner_name, 'RGB', text_data_root
            )
            self.text_rgb_dict = json.load(open(self.text_dir_rgb + f'caption_llm_dict_{captioner_name}_RGB.json'))
            self.train_text_rgb = [tokenize(self.text_rgb_dict[data_dir + i_path]['description'], self.tokenizer) for i_path in color_img_file]
            if llm_aug:
                self.llm_text_rgb = [tokenize(self.text_rgb_dict[data_dir + i_path]['aug_description'], self.tokenizer) for i_path in color_img_file]

            if joint_mode == "ir_crossfusion" or joint_mode == "uni":
                print("Loading IR Text For Training...")
                self.text_dir_ir = _resolve_text_dir(data_dir, 'llcm', captioner_name, 'IR', text_data_root)
                self.text_ir_dict = json.load(open(self.text_dir_ir + f'caption_llm_dict_{captioner_name}_IR.json'))
                self.train_text_ir = [tokenize(self.text_ir_dict[data_dir + i_path]['description'], self.tokenizer) for i_path in thermal_img_file]
                if llm_aug:
                    self.llm_text_ir = [tokenize(self.text_ir_dict[data_dir + i_path]['aug_description'], self.tokenizer) for i_path in thermal_img_file]

        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.cIndex = colorIndex
        self.tIndex = thermalIndex


    def __getitem__(self, index):
        batch_dict = {}

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1_0 = self.transform1(img1) # color image
        img1_1 = self.transform2(img1)
        img2 = self.transform3(img2)

        if self.joint_mode == "ir_crossfusion" or self.joint_mode == "uni":
            caption_id_rgb = self.train_text_rgb[self.cIndex[index]]
            if self.llm_aug and random.random() < self.llm_aug_prob:
                    caption_id_rgb = self.llm_text_rgb[self.cIndex[index]]
            batch_dict['text_rgb'] = caption_id_rgb

            caption_id_ir = self.train_text_ir[self.tIndex[index]]
            if self.llm_aug and random.random() < self.llm_aug_prob:
                caption_id_ir = self.llm_text_ir[self.tIndex[index]]
            batch_dict['text_ir'] = caption_id_ir

        batch_dict['img_rgb_ori'] = img1_0
        batch_dict['img_rgb_aug'] = img1_1
        batch_dict['img_ir'] = img2
        batch_dict['target_rgb'] = target1
        batch_dict['target_ir'] = target2

        return batch_dict

    def __len__(self):
        return len(self.train_color_label)



class Test_Tri_Data(data.Dataset):
    def __init__(self, test_img_file, test_label, data_path, transform=None, \
                 img_size=(144, 288), captioner_name='GIT', \
                    joint_mode="ir_crossfusion", gallorquery='query', \
                            Feat_Filter=False, load_text=True, text_data_root=None,
                            sysu_source_size=None, sysu_sr_data_root=None,
                            sysu_sr_modalities=None, source_modality=None,
                            sysu_sr_exact_size=False, sysu_sr_backend="array",
                            sysu_sr_view_manifest=None, sysu_sr_views_per_image=1,
                            sysu_sr_eval_view_index=0,
                            sysu_sr_eval_mode="fixed", sysu_sr_eval_top_k=5,
                            caption_lookup="identity", caption_manifest=None,
                            caption_seed=0): # include Feat_Filter=False
        self.tokenizer = SimpleTokenizer() if load_text else None
        self.Feat_Filter = Feat_Filter
        self.load_text = load_text
        self.type = gallorquery
        needs_dataset_name = bool(
            load_text
            or sysu_source_size is not None
            or sysu_sr_data_root
            or sysu_sr_modalities
            or source_modality is not None
        )
        dataset_name = _infer_dataset_name(data_path) if needs_dataset_name else None
        assert 'query' in gallorquery or 'gall' in gallorquery, "gallorquery must be 'query[i]' or 'gall[i]'"

        if load_text:
            caption_path = _resolve_caption_path(
                data_path,
                dataset_name,
                captioner_name,
                "RGB",
                caption_lookup,
                text_data_root=text_data_root,
                caption_manifest=caption_manifest,
            )
            with open(caption_path,'r') as f:
                text_dict_rgb = json.load(f)
            if Feat_Filter:
                text_dir_ir = _resolve_text_dir(data_path, dataset_name, captioner_name, 'IR', text_data_root)
                with open(os.path.join(text_dir_ir, f'caption_dict_{captioner_name}_IR.json'),'r') as f:
                    text_dict_ir = json.load(f)


        sr_modalities = normalize_sysu_sr_modalities(sysu_sr_modalities)
        if source_modality is not None:
            source_modality = str(source_modality).lower()
            if source_modality not in _SYSU_SR_TRAIN_ARRAYS:
                raise ValueError(f"Unsupported SYSU source modality: {source_modality}")
        if sr_modalities and dataset_name != "sysu":
            raise ValueError("SYSU super-resolution inputs may only be used with SYSU-MM01")

        test_image = []
        test_view_weights = []
        test_text_ir = []
        test_text_rgb = []
        caption_rng = np.random.RandomState(int(caption_seed))
        self.joint_mode = joint_mode
        print(f"Loading Test {self.type} Data...")
        for i in range(len(test_img_file)):
            # load img from the test_img_file
            source_image_path = test_img_file[i]
            image_paths = [source_image_path]
            view_weights = [1.0]
            if dataset_name == "sysu" and source_modality is not None:
                image_paths, view_weights = _sysu_eval_image_paths(
                    source_image_path,
                    data_path,
                    sysu_sr_data_root,
                    sr_modalities,
                    source_modality,
                    sysu_sr_backend,
                    sysu_sr_view_manifest,
                    sysu_sr_views_per_image,
                    sysu_sr_eval_mode,
                    sysu_sr_eval_top_k,
                    sysu_sr_eval_view_index,
                )
            image_arrays = []
            for image_path in image_paths:
                with Image.open(image_path) as handle:
                    img = handle.convert("RGB")
                using_sr = source_modality in sr_modalities
                if using_sr and sysu_sr_exact_size:
                    expected_size = (int(img_size[0]), int(img_size[1]))
                    if img.size != expected_size:
                        raise ValueError(
                            f"SYSU {source_modality} SR evaluation image has size {img.size}, "
                            f"expected {expected_size}"
                        )
                elif not using_sr and sysu_source_size is not None and dataset_name == "sysu":
                    source_h, source_w = (int(value) for value in sysu_source_size)
                    img = img.resize((source_w, source_h), PIL_BICUBIC)
                else:
                    img = img.resize((img_size[0], img_size[1]), PIL_BICUBIC)
                image_arrays.append(np.array(img))
            test_image.append(
                np.stack(image_arrays) if len(image_arrays) > 1 else image_arrays[0]
            )
            test_view_weights.append(np.asarray(view_weights, dtype=np.float32))

            if load_text:
                if caption_lookup == "image":
                    caption = _lookup_text_description(
                        text_dict_rgb, dataset_name, data_path, test_img_file[i]
                    )
                else:
                    caption = caption_rng.choice(text_dict_rgb[str(test_label[i])])
                test_text_rgb.append(tokenize(caption, self.tokenizer))
                if Feat_Filter:
                    test_text_ir.append(
                        tokenize(
                            _lookup_text_description(
                                text_dict_ir,
                                dataset_name,
                                data_path,
                                test_img_file[i],
                            ),
                            self.tokenizer,
                        )
                    )

        test_image = np.array(test_image)

        self.test_image = test_image
        self.test_view_weights = test_view_weights
        self.test_text_rgb = test_text_rgb
        self.test_text_ir = test_text_ir
        self.test_label = test_label

        self.transform = transform

    def __getitem__(self, index):
        # define batch_dict
        batch_dict = {}

        if len(self.test_text_rgb):
            text = self.test_text_rgb[index]
            batch_dict['text'] = text
        if len(self.test_text_ir):
            text = self.test_text_ir[index]
            if self.Feat_Filter:
                batch_dict['text_filter'] = text
            else:
                batch_dict['text'] = text

        img1, target1 = self.test_image[index], self.test_label[index]
        if img1.ndim == 4:
            img1 = torch.stack([self.transform(view) for view in img1])
            batch_dict['view_weights'] = torch.as_tensor(
                self.test_view_weights[index], dtype=torch.float32
            )
        else:
            img1 = self.transform(img1)

        # add to batch_dict
        batch_dict['img'] = img1
        batch_dict['target'] = target1
        return batch_dict

    def __len__(self):
        return len(self.test_image)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = f.read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label

def process_query_sysu(data_path, mode='all', relabel=False):

    # mode selection
    if mode == 'all':
        ir_cameras = ['cam3', 'cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3', 'cam6']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)

    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_sysu(data_path, mode='all', trial=0, relabel=False, gall_mode='single'):

    py_rng = random.Random(trial)
    np_rng = np.random.default_rng(trial)

    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                if gall_mode == 'single':
                    files_rgb.append(py_rng.choice(new_files))
                if gall_mode == 'multi':
                    replace = len(new_files) < 10
                    files_rgb.append(np_rng.choice(new_files, 10, replace=replace))
    gall_img = []
    gall_id = []
    gall_cam = []

    for img_path in files_rgb:
        if gall_mode == 'single':
            camid, pid = int(img_path[-15]), int(img_path[-13:-9])
            gall_img.append(img_path)
            gall_id.append(pid)
            gall_cam.append(camid)

        if gall_mode == 'multi':
            for i in img_path:
                camid, pid = int(i[-15]), int(i[-13:-9])
                gall_img.append(i)
                gall_id.append(pid)
                gall_cam.append(camid)

    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_test_regdb(img_dir, trial=1, modal='visible'):
    if modal == 'visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal == 'thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'

    with open(input_data_path) as f:
        data_file_list = f.read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split('/')[1]) for s in data_file_list]

    return file_image, np.array(file_label)


def process_query_llcm(data_path, mode = 1):
    if mode== 1:
        cameras = ['test_vis/cam1','test_vis/cam2','test_vis/cam3','test_vis/cam4','test_vis/cam5','test_vis/cam6','test_vis/cam7','test_vis/cam8','test_vis/cam9']
    elif mode ==2:
        cameras = ['test_nir/cam1','test_nir/cam2','test_nir/cam4','test_nir/cam5','test_nir/cam6','test_nir/cam7','test_nir/cam8','test_nir/cam9']

    file_path = os.path.join(data_path,'idx/test_id.txt')
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_llcm(data_path, mode = 1, trial = 0):

    py_rng = random.Random(trial)

    if mode== 1:
        cameras = ['test_vis/cam1','test_vis/cam2','test_vis/cam3','test_vis/cam4','test_vis/cam5','test_vis/cam6','test_vis/cam7','test_vis/cam8','test_vis/cam9']
    elif mode ==2:
        cameras = ['test_nir/cam1','test_nir/cam2','test_nir/cam4','test_nir/cam5','test_nir/cam6','test_nir/cam7','test_nir/cam8','test_nir/cam9']

    file_path = os.path.join(data_path,'idx/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(py_rng.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)
