import math

from salt_feature_analysis.generalization import build_generalization_rows


def test_generalization_rows_are_discovered_and_ordered():
    source = [
        {
            "comparison_id": "e2_test_ir_to_text",
            "common_identity_count": "96",
            "label_centroid_cosine_mean": "-0.1",
            "label_centroid_retrieval_top1": "0.2",
            "label_retrieval_top1": "0.1",
        },
        {"comparison_id": "unrelated"},
        {
            "comparison_id": "e2_train_ir_to_rgb",
            "common_identity_count": "395",
            "label_centroid_cosine_mean": "0.9",
            "label_centroid_retrieval_top1": "1.0",
        },
    ]
    rows = build_generalization_rows(source)
    assert [(row["split"], row["target"]) for row in rows] == [("train", "rgb"), ("test", "text")]
    assert rows[0]["centroid_top1"] == 1.0
    assert math.isnan(rows[0]["sample_top1"])
