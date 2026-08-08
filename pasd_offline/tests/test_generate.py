from pathlib import Path

from PIL import Image

from pasd_offline.generate import generate_task
from pasd_offline.tasks import GenerationTask


class FakeGenerator:
    def generate(self, image_path: Path, caption: str, seed: int) -> Image.Image:
        assert caption == "a person wearing red"
        assert seed == 13
        return Image.new("RGB", (32, 64), (128, 32, 16))


def test_generate_task_writes_public_record(tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (16, 32), (0, 0, 0)).save(source)
    task = GenerationTask(
        image=source,
        caption="a person wearing red",
        output=Path("images/rgb/source.png"),
        seed=13,
        modality="rgb",
        identity="1",
    )

    entry = generate_task(FakeGenerator(), task, tmp_path / "public-data")

    output = Path(entry["output"])
    assert output.is_file()
    assert entry["caption"] == "a person wearing red"
    assert entry["output_size"] == [32, 64]
    assert len(entry["input_sha256"]) == 64
    assert len(entry["output_sha256"]) == 64
