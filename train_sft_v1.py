#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parent
LLAMA2C = ROOT / "vendor" / "llama2c"
if not (LLAMA2C / "model.py").is_file():
    raise SystemExit("Missing vendor/llama2c. Add karpathy/llama2.c as a submodule first.")
sys.path.insert(0, str(LLAMA2C))

from export import model_export  # noqa: E402
from model import ModelArgs, Transformer  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

EVAL_GROUPS = ("combo", "name_ood", "number_ood", "full_ood")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def device_from(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(path: Path, *, dropout: float | None = None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    args = dict(checkpoint["model_args"])
    if dropout is not None:
        args["dropout"] = dropout

    model = Transformer(ModelArgs(**args))
    state = dict(checkpoint["model"])
    for key in list(state):
        if key.startswith("_orig_mod."):
            state[key.removeprefix("_orig_mod.")] = state.pop(key)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model, args


def load_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    output = {"train": [], "validation": [], "test": []}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = row["split"]
            if split not in output:
                raise ValueError(f"{path}:{line_number}: bad split {split!r}")

            x = list(map(int, row["input_ids"]))
            y = list(map(int, row["labels"]))
            if not x or len(x) != len(y) or all(value == -1 for value in y):
                raise ValueError(f"{path}:{line_number}: invalid token arrays")

            output[split].append(
                {
                    "id": row["id"],
                    "task_id": row["task_id"],
                    "eval_group": row.get("eval_group", "unspecified"),
                    "style": row.get("style", "unspecified"),
                    "template_id": row.get("template_id", "unspecified"),
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                    "x": x,
                    "y": y,
                }
            )
    return output


def iter_batches(
    records: list[dict[str, Any]],
    batch_size: int,
    seed: int | None = None,
) -> Iterable[list[dict[str, Any]]]:
    indices = list(range(len(records)))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [records[index] for index in indices[start:start + batch_size]]


def collate(records: list[dict[str, Any]], device: torch.device):
    width = max(len(record["x"]) for record in records)
    x = torch.zeros((len(records), width), dtype=torch.long)
    y = torch.full((len(records), width), -1, dtype=torch.long)

    for index, record in enumerate(records):
        length = len(record["x"])
        x[index, :length] = torch.tensor(record["x"], dtype=torch.long)
        y[index, :length] = torch.tensor(record["y"], dtype=torch.long)

    return x.to(device), y.to(device)


@torch.inference_mode()
def evaluate_loss(
    model: Transformer,
    records: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    weighted_loss = 0.0
    token_count = 0

    for batch in iter_batches(records, batch_size):
        x, y = collate(batch, device)
        model(x, y)
        if model.last_loss is None:
            raise RuntimeError("model returned no loss")
        supervised = int((y != -1).sum().item())
        weighted_loss += float(model.last_loss.item()) * supervised
        token_count += supervised

    return weighted_loss / max(token_count, 1)


@torch.inference_mode()
def generate(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, bool]:
    ids = tokenizer.encode(prompt, bos=True, eos=False)
    new_ids: list[int] = []
    stopped = False

    for _ in range(max_new_tokens):
        if len(ids) >= model.params.max_seq_len:
            break
        x = torch.tensor([ids], dtype=torch.long, device=device)
        next_id = int(model(x)[0, -1].argmax().item())

        # The Windows 98 llama98.c runtime stops generation on BOS token 1.
        if next_id == tokenizer.bos_id:
            stopped = True
            break

        ids.append(next_id)
        new_ids.append(next_id)

    return tokenizer.decode(new_ids).strip(), stopped


@torch.inference_mode()
def evaluate_exact(
    model: Transformer,
    tokenizer: Tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    exact = 0
    group_totals: Counter[str] = Counter()
    group_exact: Counter[str] = Counter()

    for index, record in enumerate(records, start=1):
        text, stopped = generate(
            model,
            tokenizer,
            record["prompt"],
            device,
            max_new_tokens,
        )
        ok = text == record["answer"]
        group = str(record["eval_group"])
        exact += int(ok)
        group_totals[group] += 1
        group_exact[group] += int(ok)
        rows.append(
            {
                "id": record["id"],
                "task_id": record["task_id"],
                "eval_group": group,
                "style": record["style"],
                "template_id": record["template_id"],
                "prompt": record["prompt"],
                "expected": record["answer"],
                "generated": text,
                "exact": ok,
                "stopped_on_bos": stopped,
            }
        )

        if index % 250 == 0:
            print(f"  exact-eval progress: {index}/{len(records)}")

    by_group = {
        group: {
            "records": group_totals[group],
            "exact": group_exact[group],
            "exact_rate": (
                group_exact[group] / group_totals[group]
                if group_totals[group]
                else 0.0
            ),
        }
        for group in sorted(group_totals)
    }

    return (
        {
            "records": len(records),
            "exact": exact,
            "exact_rate": exact / max(len(records), 1),
            "by_group": by_group,
        },
        rows,
    )


def one_per_task(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        result.setdefault(record["task_id"], record)
    return list(result.values())


def sample_probe(
    records: list[dict[str, Any]],
    per_group: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    unique = one_per_task(records)
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in unique:
        grouped[str(record["eval_group"])].append(record)

    result: list[dict[str, Any]] = []
    for group in EVAL_GROUPS:
        candidates = grouped[group]
        rng.shuffle(candidates)
        result.extend(candidates[:per_group])
    return result


def sample_records(records: list[dict[str, Any]], count: int, seed: int):
    unique = one_per_task(records)
    if count <= 0 or count >= len(unique):
        return unique
    return random.Random(seed).sample(unique, count)


def lr_at(
    step: int,
    total: int,
    warmup: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if warmup and step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (
        max_lr - min_lr
    )


def save_checkpoint(
    path: Path,
    model: Transformer,
    model_args: dict[str, Any],
    epoch: int,
    step: int,
    val_loss: float,
    meta: dict[str, Any],
) -> None:
    torch.save(
        {
            "model": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "model_args": model_args,
            "iter_num": step,
            "best_val_loss": val_loss,
            "config": meta,
            "training": {"epoch": epoch, "step": step},
        },
        path,
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_exact_summary(label: str, result: dict[str, Any]) -> None:
    print(
        f"{label:16} {result['exact']:4}/{result['records']:<4} "
        f"({result['exact_rate']:.1%})"
    )
    for group, group_result in result.get("by_group", {}).items():
        print(
            f"  {group:14} {group_result['exact']:4}/"
            f"{group_result['records']:<4} ({group_result['exact_rate']:.1%})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune stories15M on the Teapot-v1 copying curriculum."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/sft.teapot-v1.tokens.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/stories15M.pt"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=LLAMA2C / "tokenizer.model",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/teapot15m-v1"),
    )
    parser.add_argument("--binary-name", default="teapot15m-v1.bin")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--probe-per-group", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--train-final-sample",
        type=int,
        default=256,
        help="Number of unique train tasks used for final greedy evaluation; 0 means all.",
    )
    parser.add_argument(
        "--full-final-eval",
        action="store_true",
        help="Evaluate all four prompt variants instead of one record per semantic task.",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = device_from(args.device)
    tokenizer = Tokenizer(str(args.tokenizer))
    data = load_records(args.data)
    model, model_args = load_model(args.checkpoint, dropout=args.dropout)

    if tokenizer.n_words != model.params.vocab_size:
        raise RuntimeError("tokenizer/model vocabulary mismatch")

    longest = max(len(record["x"]) for records in data.values() for record in records)
    if longest > model.params.max_seq_len:
        raise RuntimeError("record exceeds model context")

    model.to(device)
    optimizer = model.configure_optimizers(
        args.weight_decay,
        args.learning_rate,
        (0.9, 0.95),
        device.type,
    )

    train = data["train"]
    validation = data["validation"]
    test = data["test"]
    val_probe = sample_probe(
        validation,
        per_group=args.probe_per_group,
        seed=args.seed,
    )

    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup_steps, max(total_steps - 1, 0))
    best_loss = float("inf")
    best_path = args.out / "best.pt"
    step = 0
    history: list[dict[str, Any]] = []

    meta = {
        "seed": args.seed,
        "device": str(device),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "data_sha256": sha256(args.data),
        "base_checkpoint_sha256": sha256(args.checkpoint),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
    }

    print(f"device: {device}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"records: train={len(train):,}, val={len(validation):,}, test={len(test):,}")
    print(f"longest sequence: {longest}/{model.params.max_seq_len}")
    print(f"steps per epoch: {steps_per_epoch}")
    print(f"total optimization steps: {total_steps}")
    print(
        "initial validation loss: "
        f"{evaluate_loss(model, validation, args.eval_batch_size, device):.5f}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        weighted = 0.0
        supervised = 0
        last_lr = args.learning_rate

        for batch in iter_batches(train, args.batch_size, seed=args.seed + epoch):
            last_lr = lr_at(
                step,
                total_steps,
                warmup,
                args.learning_rate,
                args.min_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = last_lr

            x, y = collate(batch, device)
            optimizer.zero_grad(set_to_none=True)
            model(x, y)
            loss = model.last_loss

            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, step {step}")

            loss.backward()
            if args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            count = int((y != -1).sum().item())
            weighted += float(loss.item()) * count
            supervised += count
            step += 1

        train_loss = weighted / supervised
        do_eval = (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == args.epochs
        )

        if not do_eval:
            print(
                f"epoch {epoch:3}/{args.epochs} "
                f"train={train_loss:.5f} lr={last_lr:.2e}"
            )
            continue

        val_loss = evaluate_loss(model, validation, args.eval_batch_size, device)
        probe_result, _ = evaluate_exact(
            model,
            tokenizer,
            val_probe,
            device,
            args.max_new_tokens,
        )
        group_text = " ".join(
            f"{group}={probe_result['by_group'].get(group, {}).get('exact', 0)}/"
            f"{probe_result['by_group'].get(group, {}).get('records', 0)}"
            for group in EVAL_GROUPS
        )
        history.append(
            {
                "epoch": epoch,
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "probe": probe_result,
                "lr": last_lr,
            }
        )
        print(
            f"epoch {epoch:3}/{args.epochs} train={train_loss:.5f} "
            f"val={val_loss:.5f} probe={probe_result['exact']}/"
            f"{probe_result['records']} {group_text} lr={last_lr:.2e}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                best_path,
                model,
                model_args,
                epoch,
                step,
                val_loss,
                meta,
            )
            print(f"  saved {best_path}")

    best_model, best_args = load_model(best_path, dropout=0.0)
    best_model.to(device)
    best_model.eval()

    train_eval = sample_records(train, args.train_final_sample, args.seed + 10)
    if args.full_final_eval:
        val_eval = validation
        test_eval = test
    else:
        val_eval = one_per_task(validation)
        test_eval = one_per_task(test)

    print("\nfinal greedy evaluation")
    train_result, train_rows = evaluate_exact(
        best_model,
        tokenizer,
        train_eval,
        device,
        args.max_new_tokens,
    )
    val_result, val_rows = evaluate_exact(
        best_model,
        tokenizer,
        val_eval,
        device,
        args.max_new_tokens,
    )
    test_result, test_rows = evaluate_exact(
        best_model,
        tokenizer,
        test_eval,
        device,
        args.max_new_tokens,
    )

    write_jsonl(args.out / "predictions.train-sample.jsonl", train_rows)
    write_jsonl(args.out / "predictions.validation.jsonl", val_rows)
    write_jsonl(args.out / "predictions.test.jsonl", test_rows)

    best_model.to("cpu")
    binary = args.out / args.binary_name
    model_export(best_model, str(binary), version=0)

    metrics = {
        "metadata": meta,
        "model_args": best_args,
        "history": history,
        "evaluation": {
            "train_sample": train_result,
            "validation": val_result,
            "test": test_result,
            "full_final_eval": args.full_final_eval,
        },
        "binary": {
            "path": str(binary),
            "bytes": binary.stat().st_size,
            "sha256": sha256(binary),
            "format": "llama2.c legacy v0 fp32",
        },
    }
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\ncomplete")
    print_exact_summary("train sample", train_result)
    print_exact_summary("validation", val_result)
    print_exact_summary("test", test_result)
    print(f"model: {binary}")
    print(f"sha256: {sha256(binary)}")


if __name__ == "__main__":
    main()
