#!/usr/bin/env python3
"""Generate a conflict-corrected SYSU-MM01 text tree without touching sources.

The module deliberately does one thing: when an attribute has contradictory
values inside an identity, it replaces only mentions of that attribute with a
value already observed for the same identity.  Cross-identity rarity and full
identity-signature uniqueness decide which observed value wins.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


ATTRIBUTE_ORDER = (
    "gender",
    "hair_style",
    "upper_type",
    "upper_color",
    "lower_type",
    "lower_color",
    "shoe_type",
    "shoe_color",
    "carry",
    "transient_item",
)

COLORS = {
    "black": "black", "dark": "black", "white": "white", "grey": "gray", "gray": "gray",
    "red": "red", "blue": "blue", "navy": "blue", "green": "green", "yellow": "yellow",
    "brown": "brown", "pink": "pink", "orange": "orange", "purple": "purple",
    "beige": "beige", "cream": "cream", "olive": "olive", "maroon": "red", "violet": "purple",
}

GENDER = {
    "man": "male", "boy": "male", "guy": "male", "male": "male", "gentleman": "male",
    "woman": "female", "girl": "female", "lady": "female", "female": "female",
}

HAIR = {
    "short": "short", "brief": "short", "long": "long", "bald": "bald",
    "ponytail": "ponytail", "bun": "bun", "shoulderlength": "shoulder_length",
}

ENTITIES: Dict[str, Dict[str, str]] = {
    "upper_type": {
        "tshirt": "t-shirt", "shirt": "shirt", "jacket": "jacket", "blazer": "jacket",
        "hoodie": "hoodie", "coat": "coat", "overcoat": "coat", "sweater": "sweater",
        "sweatshirt": "sweater", "blouse": "blouse", "top": "top", "dress": "dress",
        "frock": "dress", "kurta": "kurta", "kurti": "kurta", "jersey": "shirt", "vest": "vest",
    },
    "lower_type": {
        "pants": "pants", "pant": "pants", "trousers": "pants", "trouser": "pants",
        "jeans": "jeans", "denims": "jeans", "denim": "jeans", "shorts": "shorts",
        "leggings": "leggings", "jegging": "leggings", "jeggings": "leggings", "skirt": "skirt",
        "pajama": "pajama", "pyjama": "pajama", "dhoti": "dhoti", "stockings": "leggings",
    },
    "shoe_type": {
        "shoes": "shoes", "shoe": "shoes", "footwear": "shoes", "boots": "boots", "boot": "boots",
        "sneakers": "sneakers", "sneaker": "sneakers", "sandals": "sandals", "slippers": "slippers",
        "flipflops": "slippers", "heels": "heels",
    },
    "carry": {
        "backpack": "backpack", "bagpack": "backpack", "rucksack": "backpack",
        "handbag": "handbag", "purse": "handbag", "bag": "bag", "slingbag": "bag", "tote": "bag",
    },
    "transient_item": {
        "phone": "phone", "mobile": "phone", "cellphone": "phone", "bottle": "bottle",
        "umbrella": "umbrella", "camera": "camera",
    },
}

SURFACE = {
    "gender": {"male": "man", "female": "woman"},
    "hair_style": {"short": "short", "long": "long", "bald": "bald", "ponytail": "ponytail",
                   "bun": "bun", "shoulder_length": "shoulder-length"},
}

GENERIC_SUPPRESSED_BY_SPECIFIC = {
    "upper_type": {"shirt": {"t-shirt"}},
    "lower_type": {"pants": {"jeans", "leggings"}},
    "shoe_type": {"shoes": {"boots", "sneakers", "sandals", "slippers", "heels"}},
    "carry": {"bag": {"backpack", "handbag"}},
}

WORD_RE = re.compile(r"[A-Za-z]+(?:\s*-\s*[A-Za-z]+)*")
CLAUSE_RE = re.compile(r"[.;!?]")


@dataclasses.dataclass(frozen=True)
class Token:
    text: str
    normalized: str
    start: int
    end: int
    clause: int


@dataclasses.dataclass(frozen=True)
class Mention:
    attribute: str
    value: str
    start: int
    end: int
    surface: str


@dataclasses.dataclass(frozen=True)
class Decision:
    pid: str
    attribute: str
    selected: str
    candidates: Tuple[str, ...]
    support: Mapping[str, int]
    other_identity_frequency: Mapping[str, int]
    split: str = "unknown"


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def _tokens(text: str) -> List[Token]:
    result: List[Token] = []
    clause = 0
    previous = 0
    for match in WORD_RE.finditer(str(text)):
        clause += len(CLAUSE_RE.findall(text[previous:match.start()]))
        result.append(Token(match.group(), _normalize_token(match.group()), match.start(), match.end(), clause))
        previous = match.end()
    return result


def _entity_mentions(tokens: Sequence[Token]) -> List[Mention]:
    raw: List[Tuple[int, Mention]] = []
    for index, token in enumerate(tokens):
        for attribute, mapping in ENTITIES.items():
            value = mapping.get(token.normalized)
            if value:
                raw.append((index, Mention(attribute, value, token.start, token.end, token.text)))

    kept: List[Tuple[int, Mention]] = []
    for index, mention in raw:
        suppressors = GENERIC_SUPPRESSED_BY_SPECIFIC.get(mention.attribute, {}).get(mention.value, set())
        nearby = {
            other.value for other_index, other in raw
            if other.attribute == mention.attribute and abs(other_index - index) <= 2
        }
        if suppressors & nearby:
            continue
        kept.append((index, mention))
    return [mention for _, mention in kept]


def extract_mentions(text: str) -> List[Mention]:
    """Extract only attribute mentions that have a concrete replaceable span."""
    tokens = _tokens(text)
    mentions: List[Mention] = []

    for token in tokens:
        value = GENDER.get(token.normalized)
        if value:
            mentions.append(Mention("gender", value, token.start, token.end, token.text))

    hair_indexes = [i for i, token in enumerate(tokens) if token.normalized in {"hair", "hairstyle"}]
    for index, token in enumerate(tokens):
        value = HAIR.get(token.normalized)
        if not value:
            continue
        if value in {"bald", "ponytail", "bun"} or any(
            tokens[hair_index].clause == token.clause and abs(hair_index - index) <= 3
            for hair_index in hair_indexes
        ):
            mentions.append(Mention("hair_style", value, token.start, token.end, token.text))

    entity_pairs: List[Tuple[int, Mention]] = []
    for index, token in enumerate(tokens):
        for attribute, mapping in ENTITIES.items():
            value = mapping.get(token.normalized)
            if value:
                entity_pairs.append((index, Mention(attribute, value, token.start, token.end, token.text)))
    entity_mentions = _entity_mentions(tokens)
    kept_spans = {(m.attribute, m.start, m.end) for m in entity_mentions}
    mentions.extend(entity_mentions)

    color_entities = [
        (index, mention) for index, mention in entity_pairs
        if mention.attribute in {"upper_type", "lower_type", "shoe_type", "carry"}
        and (mention.attribute, mention.start, mention.end) in kept_spans
    ]
    for color_index, token in enumerate(tokens):
        color = COLORS.get(token.normalized)
        if not color:
            continue
        candidates = [
            (abs(entity_index - color_index), entity_index, mention)
            for entity_index, mention in color_entities
            if tokens[entity_index].clause == token.clause and abs(entity_index - color_index) <= 4
        ]
        if not candidates:
            continue
        hair_distance = min(
            (abs(hair_index - color_index) for hair_index in hair_indexes
             if tokens[hair_index].clause == token.clause),
            default=999,
        )
        best_distance = min(item[0] for item in candidates)
        best = [item for item in candidates if item[0] == best_distance]
        best_attributes = {item[2].attribute for item in best}
        if hair_distance <= best_distance or len(best_attributes) != 1:
            continue
        source_attribute = best[0][2].attribute
        target = {
            "upper_type": "upper_color", "lower_type": "lower_color", "shoe_type": "shoe_color",
            "carry": "carry_color",
        }[source_attribute]
        if target in ATTRIBUTE_ORDER:
            mentions.append(Mention(target, color, token.start, token.end, token.text))

    unique = {(m.attribute, m.value, m.start, m.end): m for m in mentions}
    return sorted(unique.values(), key=lambda item: (item.start, item.end, item.attribute, item.value))


def _rewrite_mentions(text: str) -> List[Mention]:
    """Return safe mentions plus companion entity words for decided attributes.

    Observation suppresses generic companions such as ``pants`` in
    ``denim jeans pants``.  Once that attribute is known to conflict, every
    entity word in the same attribute must be normalized or the companion can
    become a new contradiction after the specific word is replaced.
    """
    tokens = _tokens(text)
    mentions = list(extract_mentions(text))
    for token in tokens:
        for attribute, mapping in ENTITIES.items():
            value = mapping.get(token.normalized)
            if value:
                mentions.append(Mention(attribute, value, token.start, token.end, token.text))
    unique = {(m.attribute, m.value, m.start, m.end): m for m in mentions}
    return sorted(unique.values(), key=lambda item: (item.start, item.end, item.attribute, item.value))


def _observations(records: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, collections.Counter[str]]]:
    result: Dict[str, Dict[str, collections.Counter[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter)
    )
    for record in records:
        pid = str(int(str(record["pid"])))
        per_record: Dict[str, set[str]] = collections.defaultdict(set)
        for mention in extract_mentions(record["text"]):
            per_record[mention.attribute].add(mention.value)
        for attribute, values in per_record.items():
            for value in values:
                result[pid][attribute][value] += 1
    return result


def _signature(
    pid: str,
    observations: Mapping[str, Mapping[str, collections.Counter[str]]],
    choices: Mapping[Tuple[str, str], str],
    override: Tuple[str, str, str] | None = None,
) -> Tuple[Tuple[str, str], ...]:
    values: List[Tuple[str, str]] = []
    for attribute in ATTRIBUTE_ORDER:
        counter = observations.get(pid, {}).get(attribute)
        if not counter:
            continue
        selected = choices.get((pid, attribute))
        if override and override[0] == pid and override[1] == attribute:
            selected = override[2]
        if selected is None and len(counter) == 1:
            selected = next(iter(counter))
        if selected is not None:
            values.append((attribute, selected))
    return tuple(values)


def build_decisions(
    records: Sequence[Mapping[str, str]], split_by_pid: Mapping[str, str] | None = None,
) -> Dict[Tuple[str, str], Decision]:
    """Select one already-observed value for every within-ID conflict.

    Ordering is deterministic: minimize other-ID usage, maximize own support,
    then lexical order.  Coordinate refinement first minimizes collisions of
    the full selected identity signature and never invents a value.
    """
    observations = _observations(records)
    document_frequency: Dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for identity in observations.values():
        for attribute, counter in identity.items():
            for value in counter:
                document_frequency[attribute][value] += 1

    choices: Dict[Tuple[str, str], str] = {}
    for pid in sorted(observations, key=lambda value: int(value)):
        for attribute in ATTRIBUTE_ORDER:
            counter = observations[pid].get(attribute)
            if not counter or len(counter) < 2:
                continue
            choices[(pid, attribute)] = min(
                counter,
                key=lambda value: (document_frequency[attribute][value] - 1, -counter[value], value),
            )

    pids = sorted(observations, key=lambda value: int(value))
    signatures = {pid: _signature(pid, observations, choices) for pid in pids}
    signature_counts = collections.Counter(signatures.values())
    for _ in range(4):
        changed = False
        for pid in pids:
            for attribute in ATTRIBUTE_ORDER:
                key = (pid, attribute)
                if key not in choices:
                    continue
                counter = observations[pid][attribute]
                old_signature = signatures[pid]
                signature_counts[old_signature] -= 1
                selected = min(
                    counter,
                    key=lambda value: (
                        signature_counts[_signature(pid, observations, choices, (pid, attribute, value))],
                        document_frequency[attribute][value] - 1,
                        -counter[value],
                        value,
                    ),
                )
                if selected != choices[key]:
                    choices[key] = selected
                    changed = True
                new_signature = _signature(pid, observations, choices)
                signatures[pid] = new_signature
                signature_counts[new_signature] += 1
        if not changed:
            break

    split_by_pid = split_by_pid or {}
    decisions: Dict[Tuple[str, str], Decision] = {}
    for (pid, attribute), selected in sorted(choices.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        counter = observations[pid][attribute]
        candidates = tuple(sorted(counter))
        decisions[(pid, attribute)] = Decision(
            pid=pid,
            attribute=attribute,
            selected=selected,
            candidates=candidates,
            support=dict(sorted(counter.items())),
            other_identity_frequency={
                value: int(document_frequency[attribute][value] - 1) for value in candidates
            },
            split=split_by_pid.get(pid, "unknown"),
        )
    return decisions


def _render(attribute: str, value: str) -> str:
    return SURFACE.get(attribute, {}).get(value, value)


def _preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def correct_caption(
    text: str, pid: str, decisions: Mapping[Tuple[str, str], Decision],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Replace only spans belonging to conflicting attributes for ``pid``."""
    normalized_pid = str(int(str(pid)))
    rendered = str(text)
    changes: List[Dict[str, Any]] = []
    for pass_index in range(8):
        replacements: List[Tuple[int, int, str, Mention, Decision]] = []
        for mention in _rewrite_mentions(rendered):
            decision = decisions.get((normalized_pid, mention.attribute))
            if not decision or mention.value == decision.selected:
                continue
            replacement = _preserve_case(mention.surface, _render(mention.attribute, decision.selected))
            replacements.append((mention.start, mention.end, replacement, mention, decision))
        if not replacements:
            break

        pass_changes: List[Dict[str, Any]] = []
        for start, end, replacement, mention, decision in sorted(replacements, reverse=True):
            rendered = rendered[:start] + replacement + rendered[end:]
            pass_changes.append({
                "attribute": mention.attribute,
                "from": mention.value,
                "to": decision.selected,
                "source_surface": mention.surface,
                "replacement_surface": replacement,
                "start": start,
                "end": end,
                "pass": pass_index + 1,
            })
        pass_changes.reverse()
        changes.extend(pass_changes)

        # A multi-colour/type conflict can become "white and white" after both
        # conflicting spans are normalized. Collapse only the selected value
        # and only when this pass changed that attribute.
        for attribute in sorted({change["attribute"] for change in pass_changes}):
            selected = decisions[(normalized_pid, attribute)].selected
            surface = re.escape(_render(attribute, selected))
            rendered = re.sub(
                rf"\b({surface})(?:\s+(?:and|or)\s+|\s+)\1\b",
                r"\1",
                rendered,
                flags=re.IGNORECASE,
            )
    else:
        raise RuntimeError(f"caption correction did not converge for pid {normalized_pid}: {text!r}")
    return rendered, changes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl_gzip(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _split_map(dataset_root: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for split, filename in (("train", "train_id.txt"), ("validation", "val_id.txt"), ("test", "test_id.txt")):
        values = (dataset_root / "exp" / filename).read_text(encoding="utf-8").strip().split(",")
        for value in values:
            pid = str(int(value))
            if pid in result:
                raise ValueError(f"identity {pid} occurs in multiple splits")
            result[pid] = split
    return result


def _evidence_records(source_root: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for path in sorted(source_root.glob("*/caption_dict_*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for key, row in sorted(payload.items()):
            records.append({"pid": str(row["id"]), "text": str(row["description"]), "source": f"{path.name}:{key}"})
    if not records:
        raise ValueError(f"no caption_dict files found under {source_root}")
    return records


def _change_json_file(
    path: Path, relative: str, decisions: Mapping[Tuple[str, str], Decision], changes: List[Dict[str, Any]],
) -> int:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    changed_strings = 0
    if path.name.startswith("id_caption_map_"):
        for pid, captions in payload.items():
            for index, caption in enumerate(captions):
                rendered, edits = correct_caption(str(caption), pid, decisions)
                if edits:
                    captions[index] = rendered
                    changed_strings += 1
                    changes.append({"file": relative, "record": f"{pid}:{index}", "pid": str(int(pid)),
                                    "field": "description", "before": caption, "after": rendered, "edits": edits})
    else:
        for key, row in payload.items():
            pid = str(row["id"])
            for field in ("description", "aug_description"):
                if field not in row:
                    continue
                before = str(row[field])
                rendered, edits = correct_caption(before, pid, decisions)
                if edits:
                    row[field] = rendered
                    changed_strings += 1
                    changes.append({"file": relative, "record": key, "pid": str(int(pid)), "field": field,
                                    "before": before, "after": rendered, "edits": edits})
    _atomic_json(path, payload)
    return changed_strings


def _label_path_for_text(path: Path) -> Path:
    if path.name.startswith("train_llm_text_"):
        name = "train_text_label_" + path.name[len("train_llm_text_"):]
    elif path.name.startswith("train_text_"):
        name = "train_text_label_" + path.name[len("train_text_"):]
    else:
        raise ValueError(f"unsupported training text filename: {path.name}")
    return path.with_name(name)


def _change_npy_file(
    path: Path, relative: str, decisions: Mapping[Tuple[str, str], Decision], changes: List[Dict[str, Any]],
) -> int:
    labels_path = _label_path_for_text(path)
    if not labels_path.is_file():
        raise FileNotFoundError(f"missing label array for {path}: {labels_path}")
    values = np.load(path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    if len(values) != len(labels):
        raise ValueError(f"text/label length mismatch: {path}")
    rendered_values: List[str] = []
    changed_strings = 0
    for index, (caption, pid) in enumerate(zip(values, labels)):
        before = str(caption)
        rendered, edits = correct_caption(before, str(pid), decisions)
        rendered_values.append(rendered)
        if edits:
            changed_strings += 1
            changes.append({"file": relative, "record": str(index), "pid": str(int(str(pid))),
                            "field": "description", "before": before, "after": rendered, "edits": edits})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(rendered_values, dtype=str), allow_pickle=False)
    os.replace(temporary, path)
    return changed_strings


def _signature_collision_count(records: Sequence[Mapping[str, str]], decisions: Mapping[Tuple[str, str], Decision]) -> int:
    observations = _observations(records)
    choices = {(pid, attr): decision.selected for (pid, attr), decision in decisions.items()}
    signatures = [_signature(pid, observations, choices) for pid in sorted(observations, key=lambda value: int(value))]
    counts = collections.Counter(signatures)
    return sum(count * (count - 1) // 2 for count in counts.values())


def _validate_output(
    source_root: Path,
    output_root: Path,
    decisions: Mapping[Tuple[str, str], Decision],
    source_hashes_before: Mapping[str, str],
    changed_strings: int,
) -> Dict[str, Any]:
    source_hashes_after = _tree_hashes(source_root)
    if dict(source_hashes_before) != source_hashes_after:
        raise RuntimeError("source text tree changed during generation")

    output_records = _evidence_records(output_root)
    observations = _observations(output_records)
    unresolved: List[Dict[str, Any]] = []
    for (pid, attribute), decision in decisions.items():
        remaining = sorted(observations.get(pid, {}).get(attribute, {}))
        if any(value != decision.selected for value in remaining):
            unresolved.append({"pid": pid, "attribute": attribute, "selected": decision.selected,
                               "remaining": remaining})
    if unresolved:
        raise RuntimeError(
            f"{len(unresolved)} conflict decisions remain unresolved; first cases: "
            f"{json.dumps(unresolved[:20], ensure_ascii=False, sort_keys=True)}"
        )

    return {
        "source_unchanged": True,
        "identity_count": len({record["pid"] for record in output_records}),
        "decision_count": len(decisions),
        "changed_string_count": changed_strings,
        "unresolved_conflict_count": 0,
        "unresolved_conflicts": [],
        "output_file_count": len(_tree_hashes(output_root)),
    }


def generate(source_root: Path, dataset_root: Path, output_root: Path) -> Dict[str, Any]:
    source_root = source_root.resolve()
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("output must not be the source tree or a child of it")

    source_hashes = _tree_hashes(source_root)
    split_by_pid = _split_map(dataset_root)
    evidence = _evidence_records(source_root)
    evidence_pids = {str(int(record["pid"])) for record in evidence}
    missing = sorted(set(split_by_pid) - evidence_pids, key=int)
    if missing:
        raise ValueError(f"caption dictionaries do not cover split identities: {missing}")
    decisions = build_decisions(evidence, split_by_pid)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=str(output_root.parent)))
    try:
        shutil.rmtree(temporary)
        shutil.copytree(source_root, temporary)
        changes: List[Dict[str, Any]] = []
        changed_strings = 0
        for path in sorted(temporary.glob("*/*.json")):
            relative = str(path.relative_to(temporary)).replace("\\", "/")
            changed_strings += _change_json_file(path, relative, decisions, changes)
        for path in sorted(temporary.glob("*/train*_text_*.npy")):
            if "label" in path.name:
                continue
            relative = str(path.relative_to(temporary)).replace("\\", "/")
            changed_strings += _change_npy_file(path, relative, decisions, changes)

        decision_rows = [dataclasses.asdict(decision) for decision in decisions.values()]
        _atomic_json(temporary / "decisions.json", decision_rows)
        _atomic_jsonl_gzip(temporary / "changes.jsonl.gz", changes)
        validation = _validate_output(source_root, temporary, decisions, source_hashes, changed_strings)
        validation["signature_collision_pairs_after"] = _signature_collision_count(_evidence_records(temporary), decisions)
        _atomic_json(temporary / "validation.json", validation)
        manifest = {
            "algorithm": "sysu-id-conflict-distinctiveness-v1",
            "source_root": str(source_root),
            "output_root": str(output_root),
            "dataset_root": str(dataset_root),
            "rules": {
                "modify_only_conflicting_attributes": True,
                "selected_values_must_be_observed_in_same_identity": True,
                "selection_order": ["full_signature_collision_count", "other_identity_frequency",
                                    "negative_within_identity_support", "lexical_value"],
                "missing_attributes_are_not_inserted": True,
                "source_tree_is_read_only": True,
            },
            "split_identity_counts": dict(collections.Counter(split_by_pid.values())),
            "source_hashes": source_hashes,
            "output_hashes": _tree_hashes(temporary),
            "decision_count": len(decisions),
            "changed_string_count": changed_strings,
        }
        _atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output_root)
        return {"manifest": str(output_root / "manifest.json"), "validation": validation,
                "decision_count": len(decisions), "changed_string_count": changed_strings}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Original Text directory containing captioner subdirectories")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="SYSU-MM01 root containing exp/train_id.txt, val_id.txt and test_id.txt")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New directory; existing paths are refused")
    args = parser.parse_args(argv)
    result = generate(args.source_root, args.dataset_root, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
