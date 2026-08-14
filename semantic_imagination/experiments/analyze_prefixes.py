#!/usr/bin/env python3
"""Compare corrected clustering over nested prefixes of one sampled manifest."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from semantic_imagination import cluster_hypothesis_samples
from transformers import AutoModel, AutoTokenizer


PREFIXES = (4, 8, 16, 32, 64, 128, 256, 512)
CATEGORY = re.compile(r"(?:^|;\s*)category=([a-z0-9_]+)(?:;|$)", re.I)


def category_of(text: str) -> str:
    match = CATEGORY.search(text)
    return match.group(1).casefold() if match else "unstructured"


def total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def wilson(count: int, total: int) -> dict[str, float]:
    z = 1.959963984540054
    estimate = count / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    margin = z / denominator * math.sqrt(
        estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)
    )
    return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--embed-model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.92)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.embed_model, local_files_only=True)
    model = AutoModel.from_pretrained(args.embed_model, local_files_only=True).eval().cpu()

    def embed(texts: list[str]) -> torch.Tensor:
        embedding_texts = []
        for text in texts:
            fields = dict(
                part.split("=", 1) for part in text.split("; ") if "=" in part
            )
            value = fields.get("value", text).replace("_", " ")
            location = fields.get("location", "none").replace("_", " ")
            embedding_texts.append(f"{value} at {location}")
        batch = tokenizer(
            embedding_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    items = []
    for manifest_path in sorted(args.artifacts.glob("sample-*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples = [sample["text"] for sample in manifest["samples"]]
        prefixes = tuple(count for count in PREFIXES if count <= len(samples))
        if not prefixes:
            raise ValueError(f"{manifest_path} has no supported sample prefix")
        vectors = embed(samples)
        final_clusters = cluster_hypothesis_samples(
            samples, vectors.tolist(), args.threshold, "complete"
        )
        final_assignment = {}
        final_representatives = {}
        all_categories = sorted({category_of(text) for text in samples})
        for cluster in final_clusters:
            cluster_key = str(cluster["cluster_id"])
            final_representatives[cluster_key] = cluster["representative"]
            for member in cluster["member_indices"]:
                final_assignment[member] = cluster_key
        rows = []
        previous_categories = None
        previous_final_distribution = None
        previous_conditionals = None
        for count in prefixes:
            prefix = samples[:count]
            clusters = cluster_hypothesis_samples(
                prefix,
                vectors[:count].tolist(),
                args.threshold,
                "complete",
            )
            categories: dict[str, float] = {}
            for text in prefix:
                category = category_of(text)
                categories[category] = categories.get(category, 0.0) + 1.0 / count
            ranked_categories = sorted(
                categories.items(), key=lambda item: (-item[1], item[0])
            )
            ranked_clusters = sorted(
                clusters, key=lambda item: (-item["weight"], item["cluster_id"])
            )
            matrix = vectors[:count] @ vectors[:count].T
            minimums = []
            for cluster in clusters:
                members = cluster["member_indices"]
                minimums.append(
                    min(float(matrix[left, right]) for left in members for right in members)
                )
            final_counts: dict[str, int] = {}
            category_counts = {category: 0 for category in all_categories}
            conditional_counts = {category: {} for category in all_categories}
            for index, text in enumerate(prefix):
                cluster_key = final_assignment[index]
                category = category_of(text)
                final_counts[cluster_key] = final_counts.get(cluster_key, 0) + 1
                category_counts[category] += 1
                conditional_counts[category][cluster_key] = (
                    conditional_counts[category].get(cluster_key, 0) + 1
                )
            final_distribution = {
                key: value / count for key, value in final_counts.items()
            }
            conditional_distributions = {}
            conditional_top = {}
            for category in all_categories:
                total = category_counts[category]
                if not total:
                    continue
                distribution = {
                    key: value / total
                    for key, value in conditional_counts[category].items()
                }
                conditional_distributions[category] = distribution
                top_key, top_count = max(
                    conditional_counts[category].items(),
                    key=lambda item: (item[1], -int(item[0])),
                )
                conditional_top[category] = {
                    "effective_n": total,
                    "weight": top_count / total,
                    "interval_95": wilson(top_count, total),
                    "representative": final_representatives[top_key],
                }
            comparable_tvs = []
            if previous_conditionals is not None:
                for category, distribution in conditional_distributions.items():
                    if category in previous_conditionals:
                        comparable_tvs.append(
                            total_variation(previous_conditionals[category], distribution)
                        )
            rows.append(
                {
                    "sample_count": count,
                    "cluster_count": len(clusters),
                    "max_cluster_weight": ranked_clusters[0]["weight"],
                    "max_cluster_interval_95": ranked_clusters[0]["weight_interval_95"],
                    "max_cluster_representative": ranked_clusters[0]["representative"],
                    "category_count": len(categories),
                    "category_masses": dict(ranked_categories),
                    "category_tv_from_previous": (
                        None
                        if previous_categories is None
                        else total_variation(previous_categories, categories)
                    ),
                    "minimum_within_cluster_cosine": min(minimums),
                    "covered_category_count": sum(
                        value > 0 for value in category_counts.values()
                    ),
                    "effective_n_by_category": category_counts,
                    "final_partition_tv_from_previous": (
                        None
                        if previous_final_distribution is None
                        else total_variation(
                            previous_final_distribution, final_distribution
                        )
                    ),
                    "mean_conditional_tv_from_previous": (
                        None
                        if not comparable_tvs
                        else sum(comparable_tvs) / len(comparable_tvs)
                    ),
                    "max_conditional_tv_from_previous": (
                        None if not comparable_tvs else max(comparable_tvs)
                    ),
                    "conditional_top_by_category": conditional_top,
                }
            )
            previous_categories = categories
            previous_final_distribution = final_distribution
            previous_conditionals = conditional_distributions
        items.append(
            {
                "source": manifest["source_key"],
                "observed": manifest["observed"],
                "prefixes": rows,
            }
        )

    output = {
        "schema_version": 1,
        "source_artifacts": str(args.artifacts.resolve()),
        "sample_prefixes": [
            count for count in PREFIXES if count <= max(
                len(json.loads(path.read_text(encoding="utf-8"))["samples"])
                for path in args.artifacts.glob("sample-*/manifest.json")
            )
        ],
        "similarity_threshold": args.threshold,
        "cluster_linkage": "complete",
        "items": items,
    }
    target = args.artifacts / "prefix_comparison.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
