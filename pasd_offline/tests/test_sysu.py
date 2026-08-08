import json
from pathlib import Path

from pasd_offline.sysu import build_sysu_records
from pasd_offline.tasks import load_tasks


def test_build_identity_caption_records(tmp_path: Path):
    captions = tmp_path / "captions.json"
    identities = tmp_path / "identities.json"
    output = tmp_path / "records.jsonl"
    captions.write_text(
        json.dumps(
            {
                "datasets/sysu/cam3/0001/0002.jpg": {
                    "id": "0001",
                    "cam": "3",
                    "img": "0002",
                    "description": "one image caption",
                }
            }
        ),
        encoding="utf-8",
    )
    identities.write_text(json.dumps({"1": ["caption a", "caption b"]}), encoding="utf-8")

    count = build_sysu_records(
        dataset_root=tmp_path / "SYSU-MM01",
        caption_dicts=[captions],
        identity_caption_maps=[identities],
        caption_scope="identity",
        output_path=output,
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    pool = json.loads(output.with_suffix(".caption-pool.json").read_text(encoding="utf-8"))
    assert count == 1
    assert record["caption_pool_key"] == "1"
    assert pool["1"] == ["caption a", "caption b"]
    assert record["modality"] == "ir"
    assert record["output"] == "images/ir/cam3/0001/0002.png"
    tasks = load_tasks(output, "all", 7, output.with_suffix(".caption-pool.json"))
    assert [task.caption for task in tasks] == ["caption a", "caption b"]
