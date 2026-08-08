from pathlib import Path
import pytest

from pasd_offline.config import GenerationConfig
from pasd_offline.scheduler import GPUStatus, gpu_is_eligible


def config(tmp_path: Path):
    return GenerationConfig(
        pretrained_model_path=tmp_path,
        pasd_model_path=tmp_path,
        output_root=tmp_path,
    )


def test_gpu_zero_is_never_eligible(tmp_path: Path):
    value = config(tmp_path)
    assert not gpu_is_eligible(GPUStatus(0, 24_000, 0), value)
    assert gpu_is_eligible(GPUStatus(1, 24_000, 0), value)
    assert not gpu_is_eligible(GPUStatus(2, 20_000, 0), value)
    assert not gpu_is_eligible(GPUStatus(3, 24_000, 10), value)


def test_worker_chunk_size_must_be_positive(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"pretrained_model_path: {tmp_path / 'sd'}",
                f"pasd_model_path: {tmp_path / 'pasd'}",
                f"output_root: {tmp_path / 'output'}",
                "worker_chunk_size: 0",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="worker_chunk_size must be positive"):
        GenerationConfig.from_yaml(config_path)
