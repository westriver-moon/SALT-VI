import json
import os
import re
import time
from pathlib import Path

from easydict import EasyDict as edict
import yaml


AUTO_FIND_PMT_VIT_IMAGE_ONLY_BEST = "AUTO_FIND_PMT_VIT_IMAGE_ONLY_BEST"
_BEST_IR_PATTERN = re.compile(
    r"Best IR_RGB mINP: ([0-9.]+), Best mAP: ([0-9.]+), Best Rank1: ([0-9.]+)"
)
_NEW_BEST_IR_PATTERN = re.compile(r"New Best IR_RGB")
_EPOCH_PATTERN = re.compile(r"\bEpoch:\s*(-?\d+)\b")
_IR_CHECKPOINT_PATTERN = re.compile(r"^model_IR_(?:epoch_)?(-?\d+)\.pth$")


def _load_yaml(path, *, unsafe=False):
    loader = getattr(yaml, "UnsafeLoader", yaml.FullLoader) if unsafe else yaml.FullLoader
    with open(path, "r") as handle:
        return yaml.load(handle, Loader=loader)


def _expand_environment_values(value):
    """Recursively expand ${VAR} placeholders in portable YAML configurations."""
    if isinstance(value, dict):
        return {key: _expand_environment_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_environment_values(item) for item in value)
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _load_yaml_with_extends(path, repo_root, seen=None):
    path = os.path.abspath(path)
    seen = set() if seen is None else set(seen)
    if path in seen:
        chain = " -> ".join(sorted(seen | {path}))
        raise ValueError(f"Cyclic config inheritance: {chain}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = dict(_load_yaml(path) or {})
    parent = payload.pop("extends", None)
    if not parent:
        return _expand_environment_values(payload)
    candidates = []
    if os.path.isabs(parent):
        candidates.append(parent)
    else:
        candidates.extend((os.path.join(os.path.dirname(path), parent), os.path.join(repo_root, parent)))
    parent_path = next((os.path.abspath(candidate) for candidate in candidates if os.path.isfile(candidate)), None)
    if parent_path is None:
        raise FileNotFoundError(f"Unable to resolve parent config {parent!r} from {path}")
    inherited = _load_yaml_with_extends(parent_path, repo_root, seen | {path})
    inherited.update(payload)
    return _expand_environment_values(inherited)


def _resolve_existing_file(path, *base_dirs):
    if not path:
        return None
    candidates = []
    if os.path.isabs(path):
        candidates.append(os.path.abspath(path))
    else:
        for base_dir in base_dirs:
            if not base_dir:
                continue
            candidates.append(os.path.abspath(os.path.join(base_dir, path)))
        candidates.append(os.path.abspath(path))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _is_pmt_vit_image_only_config(config):
    return (
        isinstance(config, dict)
        and config.get("pretrain_choice") == "PMT_VIT"
        and config.get("training_mode") == "RGB_IR"
        and config.get("joint_mode", "image_only") == "image_only"
        and "Text" not in str(config.get("training_mode", ""))
    )


def _normalise_architecture_value(value):
    """Make nested YAML architecture fields safely comparable."""
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _stage_a_signature(config):
    """Extract the visual topology that must match a fixed-visual warm start."""
    if not isinstance(config, dict):
        return None
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    height = config.get("img_h", data.get("height"))
    width = config.get("img_w", data.get("width"))
    patch_embed = config.get("pmt_patch_embed", model.get("patch_embed"))
    try:
        height = None if height is None else int(height)
        width = None if width is None else int(width)
    except (TypeError, ValueError):
        return None
    return (height, width, _normalise_architecture_value(patch_embed))


def _signatures_are_compatible(candidate, expected):
    if expected is None:
        return True
    if candidate is None:
        return False
    return candidate == expected


def _format_signature(signature):
    if signature is None:
        return "unknown visual topology"
    height, width, patch_embed = signature
    return f"{height}x{width}, pmt_patch_embed={patch_embed}"


def _parse_best_ir_metrics(log_path):
    records = _parse_best_ir_records(log_path)
    if not records:
        return None
    return max(records, key=lambda item: _metric_score(item["metrics"]))["metrics"]


def _metric_score(metrics):
    return metrics[2], metrics[1], metrics[0]


def _parse_best_ir_records(log_path):
    """Return cumulative best-IR metrics paired with the epoch that logged them."""
    if not log_path.is_file():
        return []
    current_epoch = None
    records = []
    new_best_pending = False
    last_metrics = None
    for line in log_path.read_text(errors="ignore").splitlines():
        epoch_match = _EPOCH_PATTERN.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        if _NEW_BEST_IR_PATTERN.search(line):
            new_best_pending = True
        match = _BEST_IR_PATTERN.search(line)
        if not match or current_epoch is None:
            continue
        metrics = tuple(float(item) for item in match.groups())
        if (
            new_best_pending
            or last_metrics is None
            or _metric_score(metrics) > _metric_score(last_metrics)
        ):
            records.append({"epoch": current_epoch, "metrics": metrics})
        last_metrics = metrics
        new_best_pending = False
    return records


def _ir_checkpoint_epoch(path):
    match = _IR_CHECKPOINT_PATTERN.fullmatch(path.name)
    return None if match is None else int(match.group(1))


def _find_checkpoint_files(run_root):
    if not run_root.is_dir():
        return []
    checkpoints = []
    for pattern in ("best*.pth", "model*.pth"):
        checkpoints.extend(sorted(run_root.rglob(pattern)))
    return [path for path in checkpoints if path.is_file()]


def _find_canonical_stage_a_checkpoint(repo_root, expected_signature=None):
    """Find a migrated, full-state Stage-A PMT-ViT checkpoint.

    Migration manifests are authoritative when the original run directory was
    retired but its canonical checkpoint was preserved under checkpoints/stage_a.
    """
    checkpoint_root = Path(repo_root).resolve() / "checkpoints" / "stage_a"
    candidates = []
    for manifest_path in checkpoint_root.glob("*.pth.manifest.json"):
        checkpoint_path = Path(str(manifest_path)[: -len(".manifest.json")])
        if not checkpoint_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        source_config = manifest.get("bridge_config")
        if source_config and not Path(source_config).is_file():
            source_config = None
        source_signature = _stage_a_signature(manifest.get("source_config"))
        if not _signatures_are_compatible(source_signature, expected_signature):
            continue
        epoch = manifest.get("source_epoch")
        try:
            epoch = int(epoch)
        except (TypeError, ValueError):
            epoch = -1
        try:
            map_score = float(manifest.get("source_best_mAP", -1.0))
        except (TypeError, ValueError):
            map_score = -1.0
        candidates.append(
            {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "config_path": source_config,
                "metrics": None,
                "best_epoch": epoch,
                "score": (map_score, epoch),
            }
        )
    return max(candidates, key=lambda item: item["score"]) if candidates else None


def find_best_pmt_vit_image_only_checkpoint(project_root, expected_signature=None):
    repo_root = Path(project_root).resolve()
    candidate_runs = []
    for search_root_name in ("logs", "train_outputs"):
        search_root = repo_root / search_root_name
        if not search_root.is_dir():
            continue
        for config_path in search_root.rglob("configs.yaml"):
            try:
                config = _load_yaml(config_path, unsafe=True)
            except Exception:
                continue
            if not _is_pmt_vit_image_only_config(config):
                continue
            if not _signatures_are_compatible(_stage_a_signature(config), expected_signature):
                continue

            run_root = config_path.parent
            checkpoints = _find_checkpoint_files(run_root / "models")
            if not checkpoints:
                checkpoints = _find_checkpoint_files(run_root)
            if not checkpoints:
                continue

            checkpoints_by_epoch = {}
            for checkpoint in checkpoints:
                epoch = _ir_checkpoint_epoch(checkpoint)
                if epoch is not None:
                    checkpoints_by_epoch.setdefault(epoch, []).append(checkpoint)
            records = [
                record
                for record in _parse_best_ir_records(run_root / "logs" / "log.log")
                if record["epoch"] in checkpoints_by_epoch
            ]
            if records:
                selected = max(
                    records,
                    key=lambda item: (*_metric_score(item["metrics"]), item["epoch"]),
                )
                # Prefer the explicit metric-checkpoint spelling if both formats exist.
                checkpoint = max(
                    checkpoints_by_epoch[selected["epoch"]],
                    key=lambda path: ("_epoch_" in path.name, path.name),
                )
                metrics = selected["metrics"]
                best_epoch = selected["epoch"]
            elif len(checkpoints_by_epoch) == 1:
                best_epoch, epoch_paths = next(iter(checkpoints_by_epoch.items()))
                checkpoint = max(epoch_paths, key=lambda path: ("_epoch_" in path.name, path.name))
                metrics = None
            else:
                # Multiple checkpoints without an epoch/metric join are ambiguous.
                continue
            candidate_runs.append(
                {
                    "checkpoint_path": str(checkpoint.resolve()),
                    "config_path": str(config_path.resolve()),
                    "metrics": metrics,
                    "best_epoch": best_epoch,
                    "score": (
                        -1.0 if metrics is None else metrics[2],
                        -1.0 if metrics is None else metrics[1],
                        -1.0 if metrics is None else metrics[0],
                        int(config.get("prj_output_dim", 0) or 0),
                        best_epoch,
                    ),
                }
            )

    if candidate_runs:
        return max(candidate_runs, key=lambda item: item["score"])

    canonical_stage_a = _find_canonical_stage_a_checkpoint(repo_root, expected_signature)
    if canonical_stage_a is not None:
        return canonical_stage_a

    fallback_paths = []
    for search_root_name in ("logs", "train_outputs"):
        search_root = repo_root / search_root_name
        if not search_root.is_dir():
            continue
        for pattern in ("best*.pth", "model*.pth"):
            fallback_paths.extend(
                path for path in search_root.rglob(pattern) if "pmt_vit" in str(path).lower()
            )
    fallback_paths = [path for path in fallback_paths if _ir_checkpoint_epoch(path) is not None]
    if len(fallback_paths) == 1:
        checkpoint = fallback_paths[0]
        return {
            "checkpoint_path": str(checkpoint.resolve()),
            "config_path": None,
            "metrics": None,
            "best_epoch": _ir_checkpoint_epoch(checkpoint),
            "score": (-1.0, -1.0, -1.0, 0, _ir_checkpoint_epoch(checkpoint)),
        }

    if fallback_paths:
        raise RuntimeError(
            "Found multiple PMT_VIT image-only checkpoints without a trustworthy "
            "log epoch/metric mapping; set training_weight_init explicitly."
        )

    detail = "" if expected_signature is None else (
        " compatible with " + _format_signature(expected_signature)
    )
    raise FileNotFoundError(
        "Unable to find a trained PMT_VIT image-only checkpoint"
        f"{detail} in logs/, train_outputs/, or checkpoints/stage_a/."
    )

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files

def time_now():
    return time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Successfully make dirs: {}'.format(dir))
    else:
        print('Existed dirs: {}'.format(dir))

def save_train_configs(path, args):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/configs.yaml', 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

def load_train_configs(path):
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(os.path.dirname(package_root))
    if not os.path.isfile(path):
        project_candidate = os.path.join(project_root, path)
        if os.path.isfile(project_candidate):
            path = project_candidate
    default_path = os.path.join(package_root, 'config', 'default.yaml')
    args = _load_yaml(default_path)
    selected_args = _load_yaml_with_extends(path, project_root)
    if selected_args:
        args.update(selected_args)
    pmt_pretrained = args.get('pmt_pretrained')
    resolved_pmt_pretrained = _resolve_existing_file(
        pmt_pretrained,
        os.path.dirname(path),
        project_root,
        package_root,
    )
    if resolved_pmt_pretrained:
        args["pmt_pretrained"] = resolved_pmt_pretrained

    training_weight_init = args.get("training_weight_init")
    needs_training_weight_init = (
        args.get("mode", "train") == "train"
        and not bool(args.get("auto_resume_training_from_lastest_step", False))
        and int(args.get("resume_train_epoch", -1)) < 0
    )
    should_auto_find_weight = (
        needs_training_weight_init
        and args.get("pretrain_choice") == "PMT_VIT"
        and bool(args.get("Fix_Visual"))
        and not args.get("pmt_recipe", False)
        and (
            training_weight_init in (None, "", AUTO_FIND_PMT_VIT_IMAGE_ONLY_BEST)
        )
    )
    if should_auto_find_weight:
        expected_signature = _stage_a_signature(args)
        best_checkpoint = find_best_pmt_vit_image_only_checkpoint(
            project_root, expected_signature=expected_signature
        )
        args["training_weight_init"] = best_checkpoint["checkpoint_path"]
        args["training_weight_init_source_config"] = best_checkpoint["config_path"]
        args["training_weight_init_metrics"] = best_checkpoint["metrics"]
        args["training_weight_init_epoch"] = best_checkpoint["best_epoch"]
    elif needs_training_weight_init:
        resolved_training_weight_init = _resolve_existing_file(
            training_weight_init,
            os.path.dirname(path),
            project_root,
            package_root,
        )
        if training_weight_init and resolved_training_weight_init is None:
            raise FileNotFoundError(f"training_weight_init not found: {training_weight_init}")
        if resolved_training_weight_init:
            args["training_weight_init"] = resolved_training_weight_init
    args['config_select'] = path
    return edict(args)
