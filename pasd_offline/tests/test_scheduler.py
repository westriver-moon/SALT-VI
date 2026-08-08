from pathlib import Path

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
