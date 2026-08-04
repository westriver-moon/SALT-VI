#!/usr/bin/env python3
"""Generate four faithful paraphrases for every caption in a JSON index.

The writer is append-only while generation is running.  Each completed item is
written to a JSONL journal immediately, so interrupted jobs can resume without
regenerating successful records.  A canonical JSON dictionary and manifest are
materialized atomically after every checkpoint and at normal completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


MODEL_ID = "Qwen/Qwen3-14B-AWQ"
MODEL_REVISION = "31c69efc29464b6bb0aee1398b5a7b50a99340c3"
PROMPT_VERSION = "faithful-paraphrase-v1"
SYSTEM_PROMPT = """You rewrite pedestrian image captions for person re-identification training.
Faithfulness is more important than fluency or creativity. Never infer a fact that is absent
from the source caption. Return valid JSON only."""
USER_TEMPLATE = """Rewrite the source pedestrian caption into exactly four faithful English paraphrases.

Rules:
1. Preserve every identity-relevant visual fact in the source.
2. Do not add, remove, infer, weaken, or change gender/age words, hair, clothing colors,
   clothing types, sleeve or trouser length, footwear, carried objects, pose, direction,
   body build, or visible accessories.
3. Change only wording, syntax, punctuation, and the order in which existing facts are stated.
4. Each paraphrase must be independently readable and linguistically distinct.
5. Be concise. Each paraphrase MUST contain no more than {max_words} whitespace-separated
   words. Count its words before returning the JSON and shorten it if necessary.
6. Do not mention these instructions or the source caption.
7. Return exactly this JSON shape and nothing else:
   {{"paraphrases":["...","...","...","..."]}}

Source caption:
{caption}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="flat or description-object caption JSON")
    parser.add_argument("--output-dir", required=True, help="external unified-dataset output directory")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-id", default=MODEL_ID, help="canonical model identity stored in outputs")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-words", type=int, default=45)
    parser.add_argument(
        "--prompt-max-words",
        type=int,
        default=None,
        help="optional stricter length requested from the model; validation still uses --max-words",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="smoke-test only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_retries < 1:
        parser.error("batch-size and max-retries must be positive")
    if args.max_words < 1 or (args.prompt_max_words is not None and args.prompt_max_words < 1):
        parser.error("max-words and prompt-max-words must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("require 0 <= shard-id < num-shards")
    return args


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_input(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("input must be a non-empty JSON object")
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("every caption key must be a string")
        if isinstance(value, str):
            value = {"description": value}
        if not isinstance(value, dict):
            raise ValueError("every caption value must be a string or object")
        description = value.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"missing description for {key}")
        normalized[key] = dict(value)
    return normalized


def extract_json_object(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model output contains no JSON object")
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model output JSON root is not an object")
    return payload


def normalize_paraphrases(text: str, max_words: int) -> List[str]:
    payload = extract_json_object(text)
    values = payload.get("paraphrases")
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("model output must contain exactly four paraphrases")
    cleaned = [re.sub(r"\s+", " ", str(value)).strip() for value in values]
    if any(not value for value in cleaned):
        raise ValueError("empty paraphrase")
    if len({value.casefold() for value in cleaned}) != 4:
        raise ValueError("duplicate paraphrases")
    if any(len(value.split()) > max_words for value in cleaned):
        raise ValueError("paraphrase exceeds word limit")
    return cleaned


def valid_journal_paraphrases(values: Any) -> bool:
    return (
        isinstance(values, list)
        and len(values) == 4
        and all(isinstance(value, str) and value.strip() for value in values)
        and len({value.strip().casefold() for value in values}) == 4
    )


def read_journal(path: Path) -> Tuple[Dict[str, Dict[str, Any]], int]:
    completed: Dict[str, Dict[str, Any]] = {}
    invalid = 0
    if not path.is_file():
        return completed, invalid
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                key = item["key"]
                if not isinstance(key, str) or not valid_journal_paraphrases(item.get("paraphrases")):
                    raise ValueError
                completed[key] = item
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                invalid += 1
    return completed, invalid


def selected_items(
    entries: Mapping[str, Mapping[str, Any]], shard_id: int, num_shards: int, limit: int | None
) -> List[Tuple[str, Mapping[str, Any]]]:
    selected = [item for index, item in enumerate(sorted(entries.items())) if index % num_shards == shard_id]
    return selected[:limit] if limit is not None else selected


def build_prompt(caption: str, max_words: int) -> str:
    return USER_TEMPLATE.format(caption=caption.strip(), max_words=max_words)


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, cache_dir=args.cache_dir, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        device_map={"": args.device},
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval()
    return model, tokenizer, torch


def render_chat(tokenizer, caption: str, max_words: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(caption, max_words)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_batch(model, tokenizer, torch, captions: Sequence[str], args, attempt: int) -> List[str]:
    prompt_max_words = getattr(args, "prompt_max_words", None) or args.max_words
    prompts = [render_chat(tokenizer, caption, prompt_max_words) for caption in captions]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    ).to(model.device)
    generation_seed = args.seed + args.shard_id * 1_000_003 + attempt
    torch.manual_seed(generation_seed)
    if str(model.device).startswith("cuda"):
        torch.cuda.manual_seed_all(generation_seed)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1.0e-5),
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_width = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(outputs[:, prompt_width:], skip_special_tokens=True)


def materialize(
    output_path: Path,
    manifest_path: Path,
    source: Mapping[str, Mapping[str, Any]],
    completed: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    payload: Dict[str, Dict[str, Any]] = {}
    for key in sorted(completed):
        record = dict(source[key])
        record["paraphrases"] = list(completed[key]["paraphrases"])
        record["augmentation"] = {
            "model": metadata["model"],
            "prompt_version": metadata["prompt_version"],
            "seed": completed[key]["seed"],
            "attempt": completed[key]["attempt"],
        }
        payload[key] = record
    atomic_json(output_path, payload)
    manifest = dict(metadata)
    manifest.update(
        {
            "completed": len(completed),
            "expected_for_shard": metadata["expected_for_shard"],
            "coverage": len(completed) / max(1, int(metadata["expected_for_shard"])),
            "updated_at_unix": time.time(),
            "complete": len(completed) == int(metadata["expected_for_shard"]),
        }
    )
    atomic_json(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard-{args.shard_id:03d}-of-{args.num_shards:03d}"
    journal_path = output_dir / f"paraphrases.{tag}.jsonl"
    output_path = output_dir / f"caption_qwen3_14b_awq_4x.{tag}.json"
    manifest_path = output_dir / f"manifest.{tag}.json"
    failures_path = output_dir / f"failures.{tag}.json"

    source = load_input(input_path)
    work = selected_items(source, args.shard_id, args.num_shards, args.limit)
    allowed_keys = {key for key, _ in work}
    completed, invalid_journal_lines = read_journal(journal_path)
    completed = {key: value for key, value in completed.items() if key in allowed_keys}
    pending = [(key, value) for key, value in work if key not in completed]
    metadata = {
        "schema_version": 1,
        "model": args.model_id,
        "model_source": args.model,
        "revision": args.revision,
        "prompt_version": PROMPT_VERSION,
        "source": str(input_path),
        "source_sha256": sha256_file(input_path),
        "output": str(output_path),
        "journal": str(journal_path),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "expected_total": len(source),
        "expected_for_shard": len(work),
        "invalid_journal_lines": invalid_journal_lines,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "max_words": args.max_words,
            "prompt_max_words": args.prompt_max_words or args.max_words,
        },
    }
    print(json.dumps({**metadata, "already_completed": len(completed), "pending": len(pending)}, indent=2))
    if args.dry_run:
        return 0
    if not pending:
        materialize(output_path, manifest_path, source, completed, metadata)
        atomic_json(failures_path, {})
        return 0

    random.seed(args.seed + args.shard_id)
    model, tokenizer, torch = load_model(args)
    failures: Dict[str, str] = {}
    journal = journal_path.open("a", encoding="utf-8")
    try:
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            unresolved = list(batch)
            for attempt in range(1, args.max_retries + 1):
                if not unresolved:
                    break
                raw_outputs = generate_batch(
                    model,
                    tokenizer,
                    torch,
                    [value["description"] for _, value in unresolved],
                    args,
                    attempt + batch_start,
                )
                retry: List[Tuple[str, Mapping[str, Any]]] = []
                for (key, value), raw in zip(unresolved, raw_outputs):
                    try:
                        paraphrases = normalize_paraphrases(raw, args.max_words)
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        failures[key] = f"{type(exc).__name__}: {exc}; raw={raw[:500]!r}"
                        retry.append((key, value))
                        continue
                    item = {
                        "key": key,
                        "description": value["description"],
                        "paraphrases": paraphrases,
                        "seed": args.seed + args.shard_id * 1_000_003 + attempt + batch_start,
                        "attempt": attempt,
                    }
                    journal.write(json.dumps(item, ensure_ascii=False) + "\n")
                    journal.flush()
                    completed[key] = item
                    failures.pop(key, None)
                unresolved = retry
            processed = min(batch_start + len(batch), len(pending))
            if processed % args.checkpoint_every < args.batch_size or processed == len(pending):
                os.fsync(journal.fileno())
                materialize(output_path, manifest_path, source, completed, metadata)
                atomic_json(failures_path, failures)
                print(f"completed={len(completed)}/{len(work)} pending={len(work)-len(completed)}", flush=True)
    finally:
        journal.close()

    materialize(output_path, manifest_path, source, completed, metadata)
    atomic_json(failures_path, failures)
    if len(completed) != len(work):
        print(f"incomplete: {len(work)-len(completed)} records failed", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
