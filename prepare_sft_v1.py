#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
LLAMA2C = ROOT / "vendor" / "llama2c"

if not (LLAMA2C / "tokenizer.py").is_file():
    raise SystemExit("Missing vendor/llama2c. Add karpathy/llama2.c as a submodule first.")

sys.path.insert(0, str(LLAMA2C))
from tokenizer import Tokenizer  # noqa: E402

PROMPT_PREFIX = "Instruction: "
PROMPT_SUFFIX = " MAXScript:"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_prompt(instruction: str) -> str:
    normalized = " ".join(instruction.strip().split())
    if not normalized:
        raise ValueError("Instruction cannot be empty")
    return f"{PROMPT_PREFIX}{normalized}{PROMPT_SUFFIX}"


def load_checkpoint_limits(checkpoint_path: Path) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_args = checkpoint.get("model_args")
    if not isinstance(model_args, dict):
        raise ValueError("Checkpoint does not contain a model_args dictionary")
    return int(model_args["max_seq_len"]), int(model_args["vocab_size"])


def task_id_from_record_id(record_id: str) -> str:
    if "::" not in record_id:
        raise ValueError(f"Malformed record ID: {record_id!r}")
    return record_id.split("::", 1)[0]


def encode_record(
    record: dict[str, Any],
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> dict[str, Any]:
    record_id = str(record["id"])
    instruction = str(record["instruction"])
    answer = str(record["answer"])
    prompt = render_prompt(instruction)

    # Encode separately so the first answer token follows exactly the tokenized
    # prompt that llama98.c will receive through its -i argument.
    prompt_ids = tokenizer.encode(prompt, bos=True, eos=False)
    answer_ids = tokenizer.encode(answer, bos=False, eos=False)

    if not answer_ids:
        raise ValueError(f"{record_id}: answer produced no tokens")

    decoded_answer = tokenizer.decode(answer_ids).strip()
    if decoded_answer != answer:
        raise ValueError(
            f"{record_id}: tokenizer round-trip changed the answer\n"
            f"expected: {answer!r}\n"
            f"decoded:  {decoded_answer!r}"
        )

    # llama98.c's generation loop uses BOS token 1 as the stop token.
    sequence = prompt_ids + answer_ids + [tokenizer.bos_id]
    input_ids = sequence[:-1]
    labels = sequence[1:]

    # Do not train the model to reproduce the user prompt. The target directly
    # after the final prompt token—the first script token—remains supervised.
    ignored_prompt_targets = len(prompt_ids) - 1
    labels[:ignored_prompt_targets] = [-1] * ignored_prompt_targets

    if len(input_ids) > max_seq_len:
        raise ValueError(
            f"{record_id}: {len(input_ids)} input tokens exceed "
            f"the model context limit of {max_seq_len}"
        )

    supervised_tokens = sum(token != -1 for token in labels)
    expected_supervised = len(answer_ids) + 1
    if supervised_tokens != expected_supervised:
        raise AssertionError(
            f"{record_id}: expected {expected_supervised} supervised tokens, "
            f"got {supervised_tokens}"
        )

    provenance = record.get("provenance", {})
    return {
        "id": record_id,
        "task_id": task_id_from_record_id(record_id),
        "split": record["split"],
        "eval_group": record.get("eval_group", "unspecified"),
        "style": record.get("style", "unspecified"),
        "template_id": record.get("template_id", "unspecified"),
        "prompt": prompt,
        "answer": answer,
        "input_ids": input_ids,
        "labels": labels,
        "token_counts": {
            "prompt": len(prompt_ids),
            "answer": len(answer_ids),
            "input": len(input_ids),
            "supervised": supervised_tokens,
        },
        "source": {
            "generator": provenance.get("generator", "UNKNOWN"),
            "canonical_script_sha256": provenance.get(
                "canonical_script_sha256", "UNKNOWN"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare answer-masked llama2.c SFT records for Teapot-v1."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("data/paraphrases.teapot-v1.jsonl"),
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("data/sft.teapot-v1.tokens.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/stories15M.pt"),
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=LLAMA2C / "tokenizer.model",
    )
    args = parser.parse_args()

    max_seq_len, checkpoint_vocab_size = load_checkpoint_limits(args.checkpoint)
    tokenizer = Tokenizer(str(args.tokenizer_model))
    if tokenizer.n_words != checkpoint_vocab_size:
        raise ValueError(
            f"Tokenizer has {tokenizer.n_words} tokens, but checkpoint expects "
            f"{checkpoint_vocab_size}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    split_counts: Counter[str] = Counter()
    group_counts: Counter[tuple[str, str]] = Counter()
    input_lengths: list[int] = []
    prompt_lengths: list[int] = []
    answer_lengths: list[int] = []
    supervised_lengths: list[int] = []
    record_count = 0

    with (
        args.input.open("r", encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                prepared = encode_record(json.loads(line), tokenizer, max_seq_len)
            except Exception as exc:
                raise RuntimeError(f"{args.input}:{line_number}: {exc}") from exc

            destination.write(
                json.dumps(prepared, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            counts = prepared["token_counts"]
            split = str(prepared["split"])
            group = str(prepared["eval_group"])
            split_counts[split] += 1
            group_counts[(split, group)] += 1
            input_lengths.append(int(counts["input"]))
            prompt_lengths.append(int(counts["prompt"]))
            answer_lengths.append(int(counts["answer"]))
            supervised_lengths.append(int(counts["supervised"]))
            record_count += 1

    if record_count == 0:
        raise RuntimeError("No records were prepared")

    print(f"Wrote:           {args.output}")
    print(f"Records:         {record_count:,}")
    print(
        "Splits:          "
        + ", ".join(
            f"{split}={split_counts[split]:,}"
            for split in ("train", "validation", "test")
        )
    )
    print(f"Tokenizer vocab: {tokenizer.n_words}")
    print(
        f"Special tokens:  BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}, "
        f"PAD={tokenizer.pad_id}"
    )
    print(f"Model context:   {max_seq_len}")
    print(
        f"Input tokens:    min={min(input_lengths)}, "
        f"mean={statistics.mean(input_lengths):.1f}, max={max(input_lengths)}"
    )
    print(f"Prompt tokens:   mean={statistics.mean(prompt_lengths):.1f}")
    print(f"Answer tokens:   mean={statistics.mean(answer_lengths):.1f}")
    print(f"Supervised:      mean={statistics.mean(supervised_lengths):.1f}")
    print(f"Source SHA256:   {sha256_file(args.input)}")
    print(f"Output SHA256:   {sha256_file(args.output)}")
    print()
    for split in ("validation", "test"):
        for group in ("combo", "name_ood", "number_ood", "full_ood"):
            print(f"{split:10} {group:11} {group_counts[(split, group)]:5} records")
    print("\nSFT token preparation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
