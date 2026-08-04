"""Canonical provenance contracts shared by SYSU SR preflight and runners."""

import hashlib
import json
from pathlib import Path
import subprocess

from salt_vi.utils.utils import load_train_configs


PREFLIGHT_SCHEMA_VERSION = 4
_PROVENANCE_CACHE = {}
_FILE_HASH_CACHE = {}
ALGORITHM_SOURCE_PATHS = (
    "src/salt_vi/entrypoints/train.py",
    "src/salt_vi/engine",
    "configs",
    "src/salt_vi/data",
    "src/salt_vi/models",
    "src/salt_vi/optim",
    "scripts/super_resolution",
    "src/salt_vi/utils/super_resolution",
)
SYSU_EVAL_CAMERAS = {
    "rgb": ("cam1", "cam2", "cam4", "cam5"),
    "ir": ("cam3", "cam6"),
}


def sha256_file(path, chunk_size=8 * 1024 * 1024, use_cache=True):
    path = Path(path).resolve()
    before = path.stat()
    signature = (
        str(path), int(before.st_size), int(before.st_mtime_ns),
        int(before.st_ctime_ns), int(before.st_ino),
    )
    if use_cache and signature in _FILE_HASH_CACHE:
        return _FILE_HASH_CACHE[signature]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    after = path.stat()
    after_signature = (
        str(path), int(after.st_size), int(after.st_mtime_ns),
        int(after.st_ctime_ns), int(after.st_ino),
    )
    if after_signature != signature:
        raise RuntimeError(f"File changed while hashing: {path}")
    value = digest.hexdigest()
    if use_cache:
        _FILE_HASH_CACHE[signature] = value
    return value


def plain_data(value):
    if isinstance(value, dict):
        return {str(key): plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_sha256(value):
    payload = json.dumps(plain_data(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_head(repo_root):
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _git(repo_root, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], text=True
    ).strip()


def algorithm_source_files(repo_root):
    """Return tracked algorithm/config files, excluding generated and cache files."""
    repo_root = Path(repo_root).resolve()
    names = _git(repo_root, "ls-files", "--", *ALGORITHM_SOURCE_PATHS).splitlines()
    return [repo_root / name for name in names if (repo_root / name).is_file()]


def algorithm_source_hash(repo_root):
    repo_root = Path(repo_root).resolve()
    digest = hashlib.sha256()
    files = algorithm_source_files(repo_root)
    for path in files:
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path, use_cache=False)))
    return {"sha256": digest.hexdigest(), "file_count": len(files)}


def hash_file_tree(paths, root):
    root = Path(root).resolve()
    digest = hashlib.sha256()
    count = 0
    for path in sorted(Path(path).resolve() for path in paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path, use_cache=False)))
        count += 1
    return {"sha256": digest.hexdigest(), "file_count": count}


def sysu_evaluation_paths(source_root, modality):
    source_root = Path(source_root).resolve()
    test_id_path = source_root / "exp/test_id.txt"
    values = test_id_path.read_text(encoding="utf-8").replace("\n", ",").split(",")
    identities = [f"{int(value):04d}" for value in values if value.strip()]
    paths = []
    for camera in SYSU_EVAL_CAMERAS[modality]:
        for identity in identities:
            directory = source_root / camera / identity
            if directory.is_dir():
                paths.extend(path for path in directory.iterdir() if path.is_file())
    return sorted(paths)


def verified_text_assets(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = payload.get("output_hashes")
    if not isinstance(expected, dict) or not expected:
        raise ValueError(f"Text manifest has no output_hashes: {manifest_path}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != manifest_path
    }
    undeclared = sorted(actual_files - set(expected))
    if undeclared:
        raise ValueError(f"Text manifest omits output files: {undeclared}")
    assets = {}
    for relative, expected_hash in sorted(expected.items()):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Text manifest path escapes root: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"Missing text asset declared by manifest: {path}")
        actual_hash = sha256_file(path, use_cache=False)
        if actual_hash != expected_hash:
            raise ValueError(f"Text asset hash mismatch: {path}")
        assets[relative] = {"path": str(path), "sha256": actual_hash}
    return assets


def assert_clean_algorithm_source(repo_root):
    """Reject tracked, staged, or untracked algorithm changes before controlled work."""
    repo_root = Path(repo_root).resolve()
    checks = (
        ("diff", "--name-only", "--", *ALGORITHM_SOURCE_PATHS),
        ("diff", "--cached", "--name-only", "--", *ALGORITHM_SOURCE_PATHS),
        ("ls-files", "--others", "--exclude-standard", "--", *ALGORITHM_SOURCE_PATHS),
    )
    labels = ("tracked worktree", "staged", "untracked")
    dirty = {}
    for label, args in zip(labels, checks):
        paths = _git(repo_root, *args).splitlines()
        if paths:
            dirty[label] = paths
    if dirty:
        details = "; ".join(f"{label}: {paths}" for label, paths in dirty.items())
        raise RuntimeError(f"Algorithm source must be clean: {details}")
    return git_head(repo_root)


def _required_file(path, label):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _stat_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return (
        str(path), int(stat.st_size), int(stat.st_mtime_ns),
        int(stat.st_ctime_ns), int(stat.st_ino),
    )


def build_preflight_provenance(config_path, repo_root, reference_preflight=None):
    """Bind a preflight result to every input that can change its meaning."""
    config_path = Path(config_path).resolve()
    repo_root = Path(repo_root).resolve()
    config = load_train_configs(str(config_path))
    source_root = Path(config.sysu_data_path).resolve()
    text_manifest_path = Path(config.text_data_root) / "manifest.json"
    text_assets = verified_text_assets(text_manifest_path)
    evaluation = {
        modality: sysu_evaluation_paths(source_root, modality)
        for modality in SYSU_EVAL_CAMERAS
    }
    preflight_code = repo_root / "scripts/super_resolution/preflight_sysu_sr.py"
    validator_code = repo_root / "src/salt_vi/utils/super_resolution/validate_sysu_swinir_x2.py"
    loader_code = repo_root / "data_loader/loader.py"
    tracked_inputs = [
        config_path,
        source_root / "train_rgb_resized_img.npy",
        source_root / "train_ir_resized_img.npy",
        source_root / "train_rgb_resized_label.npy",
        source_root / "train_ir_resized_label.npy",
        source_root / "exp/test_id.txt",
        Path(config.training_weight_init),
        text_manifest_path,
        *(Path(item["path"]) for item in text_assets.values()),
        *(path for paths in evaluation.values() for path in paths),
        preflight_code,
        validator_code,
        loader_code,
    ]
    if config.sysu_sr_modalities:
        tracked_inputs.append(Path(config.sysu_sr_data_root) / "manifest.json")
    if reference_preflight is not None:
        tracked_inputs.append(Path(reference_preflight))
    head = git_head(repo_root)
    algorithm_inputs = algorithm_source_files(repo_root)
    cache_key = (
        head,
        tuple(_stat_signature(path) for path in tracked_inputs),
        tuple(_stat_signature(path) for path in algorithm_inputs),
    )
    if cache_key in _PROVENANCE_CACHE:
        return _PROVENANCE_CACHE[cache_key]
    provenance = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "experiment_id": str(config.metric_experiment_id),
        "git_commit_sha": head,
        "config_path": str(config_path),
        "config_file_sha256": sha256_file(config_path),
        "resolved_config_sha256": canonical_sha256(dict(config)),
        "dataset_root": str(source_root),
        "train_rgb": _required_file(source_root / "train_rgb_resized_img.npy", "SYSU RGB array"),
        "train_ir": _required_file(source_root / "train_ir_resized_img.npy", "SYSU IR array"),
        "train_rgb_label": _required_file(
            source_root / "train_rgb_resized_label.npy", "SYSU RGB labels"
        ),
        "train_ir_label": _required_file(
            source_root / "train_ir_resized_label.npy", "SYSU IR labels"
        ),
        "test_id": _required_file(source_root / "exp/test_id.txt", "SYSU test identities"),
        "source_evaluation_tree": {
            modality: hash_file_tree(paths, source_root)
            for modality, paths in evaluation.items()
        },
        "warm_start": _required_file(config.training_weight_init, "warm-start checkpoint"),
        "text_manifest": _required_file(
            text_manifest_path, "corrected-text manifest"
        ),
        "text_assets": text_assets,
        "implementation": {
            "schema": "sysu-sr-preflight-v4",
            "preflight_sha256": sha256_file(preflight_code),
            "validator_sha256": sha256_file(validator_code),
            "loader_sha256": sha256_file(loader_code),
            "algorithm_source_tree": algorithm_source_hash(repo_root),
        },
        "sr_data_root": str(Path(config.sysu_sr_data_root).resolve()),
        "sr_modalities": sorted(str(value).lower() for value in config.sysu_sr_modalities),
        "sr_manifest": None,
        "reference_preflight": None,
    }
    if provenance["sr_modalities"]:
        provenance["sr_manifest"] = _required_file(
            Path(config.sysu_sr_data_root) / "manifest.json", "SR manifest"
        )
    if reference_preflight is not None:
        provenance["reference_preflight"] = _required_file(
            reference_preflight, "A0 reference preflight"
        )
    provenance["contract_sha256"] = canonical_sha256(provenance)
    _PROVENANCE_CACHE[cache_key] = provenance
    return provenance


def provenance_matches(result, expected):
    return isinstance(result, dict) and result.get("provenance") == expected
