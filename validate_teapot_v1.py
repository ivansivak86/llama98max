#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mxs98_domain import format_number, format_vector, render_canonical_script

EVAL_GROUPS = {"combo", "name_ood", "number_ood", "full_ood"}
OOD_NAME_GROUPS = {"name_ood", "full_ood"}
OOD_NUMBER_GROUPS = {"number_ood", "full_ood"}
FAMILIAR_NAME_GROUPS = {"combo", "number_ood"}
FAMILIAR_NUMBER_GROUPS = {"combo", "name_ood"}

FORBIDDEN_PATTERNS = (
    re.compile(r"\btask[_ -]?id\b", re.IGNORECASE),
    re.compile(r"\bcreate_primitive\b", re.IGNORECASE),
    re.compile(r"\bschema\b", re.IGNORECASE),
    re.compile(r"\bjson\b", re.IGNORECASE),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Teapot-v1 tasks and SFT records.")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/tasks.teapot-v1.jsonl"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("data/paraphrases.teapot-v1.jsonl"),
    )
    parser.add_argument("--variants-per-task", type=int, default=4)
    args = parser.parse_args()

    tasks = load_jsonl(args.tasks)
    records = load_jsonl(args.records)
    errors: list[str] = []

    task_by_id: dict[str, dict[str, Any]] = {}
    semantic_keys: set[tuple[Any, ...]] = set()
    split_task_counts: Counter[tuple[str, str]] = Counter()

    for task in tasks:
        task_id = str(task.get("id", ""))
        if not task_id or task_id in task_by_id:
            errors.append(f"duplicate or empty task ID: {task_id!r}")
            continue
        task_by_id[task_id] = task

        split = task.get("split")
        group = task.get("eval_group")
        if split == "train" and group != "train":
            errors.append(f"{task_id}: train task has eval_group={group!r}")
        if split in {"validation", "test"} and group not in EVAL_GROUPS:
            errors.append(f"{task_id}: invalid evaluation group {group!r}")

        spec = task.get("spec")
        if not isinstance(spec, dict):
            errors.append(f"{task_id}: missing spec object")
            continue
        expected_script = render_canonical_script(spec)
        if task.get("canonical_script") != expected_script:
            errors.append(f"{task_id}: canonical script mismatch")

        key = (spec["name"], spec["radius"], *spec["position"])
        if key in semantic_keys:
            errors.append(f"{task_id}: duplicate semantic tuple {key!r}")
        semantic_keys.add(key)
        split_task_counts[(str(split), str(group))] += 1

    train_tasks = [task for task in tasks if task.get("split") == "train"]
    train_names = {task["spec"]["name"] for task in train_tasks}
    train_radii = {task["spec"]["radius"] for task in train_tasks}
    train_positions = {
        value for task in train_tasks for value in task["spec"]["position"]
    }

    val_new_names: set[str] = set()
    test_new_names: set[str] = set()

    for task in tasks:
        split = task.get("split")
        group = task.get("eval_group")
        if split not in {"validation", "test"}:
            continue
        spec = task["spec"]
        name = spec["name"]
        values = [spec["radius"], *spec["position"]]

        if group in FAMILIAR_NAME_GROUPS and name not in train_names:
            errors.append(f"{task['id']}: familiar-name group uses unseen name {name!r}")
        if group in OOD_NAME_GROUPS and name in train_names:
            errors.append(f"{task['id']}: OOD-name group leaks train name {name!r}")
        if group in FAMILIAR_NUMBER_GROUPS:
            if spec["radius"] not in train_radii:
                errors.append(f"{task['id']}: radius is not familiar")
            for value in spec["position"]:
                if value not in train_positions:
                    errors.append(f"{task['id']}: position value {value} is not familiar")
        if group in OOD_NUMBER_GROUPS:
            if spec["radius"] in train_radii:
                errors.append(f"{task['id']}: OOD radius {spec['radius']} appears in train")
            for value in spec["position"]:
                if value in train_positions:
                    errors.append(f"{task['id']}: OOD position value {value} appears in train")

        if group in OOD_NAME_GROUPS:
            (val_new_names if split == "validation" else test_new_names).add(name)

    overlap = val_new_names & test_new_names
    if overlap:
        errors.append(f"validation/test OOD names overlap: {sorted(overlap)[:10]!r}")

    record_ids: set[str] = set()
    normalized_instructions: set[str] = set()
    records_per_task: Counter[str] = Counter()
    styles_per_task: defaultdict[str, set[str]] = defaultdict(set)
    split_record_counts: Counter[tuple[str, str]] = Counter()

    for record in records:
        record_id = str(record.get("id", ""))
        if record_id in record_ids:
            errors.append(f"duplicate record ID: {record_id!r}")
        record_ids.add(record_id)

        if "::det::" not in record_id:
            errors.append(f"malformed deterministic record ID: {record_id!r}")
            continue
        task_id = record_id.split("::det::", 1)[0]
        task = task_by_id.get(task_id)
        if task is None:
            errors.append(f"{record_id}: references unknown task {task_id!r}")
            continue

        records_per_task[task_id] += 1
        styles_per_task[task_id].add(str(record.get("style")))

        if record.get("split") != task.get("split"):
            errors.append(f"{record_id}: split mismatch")
        if record.get("eval_group") != task.get("eval_group"):
            errors.append(f"{record_id}: eval_group mismatch")
        if record.get("task_spec") != task.get("spec"):
            errors.append(f"{record_id}: task_spec mismatch")

        expected_answer = render_canonical_script(task["spec"])
        answer = record.get("answer")
        if answer != expected_answer:
            errors.append(f"{record_id}: answer mismatch")

        expected_hash = hashlib.sha256(expected_answer.encode("utf-8")).hexdigest()
        actual_hash = record.get("provenance", {}).get("canonical_script_sha256")
        if actual_hash != expected_hash:
            errors.append(f"{record_id}: answer hash mismatch")

        instruction = str(record.get("instruction", ""))
        normalized = " ".join(instruction.split()).casefold()
        if not normalized:
            errors.append(f"{record_id}: empty instruction")
        elif normalized in normalized_instructions:
            errors.append(f"{record_id}: duplicate instruction")
        normalized_instructions.add(normalized)

        if len(instruction.split()) > 40:
            errors.append(f"{record_id}: instruction exceeds 40 words")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(instruction):
                errors.append(f"{record_id}: forbidden meta language {pattern.pattern!r}")

        spec = task["spec"]
        required = (
            "teapot",
            spec["name"],
            "radius",
            format_number(spec["radius"]),
            format_vector(spec["position"], spaces=True),
        )
        folded = instruction.casefold()
        for literal in required:
            if str(literal).casefold() not in folded:
                errors.append(f"{record_id}: missing required literal {literal!r}")

        split_record_counts[(task["split"], task["eval_group"])] += 1

    for task_id in task_by_id:
        count = records_per_task[task_id]
        if count != args.variants_per_task:
            errors.append(
                f"{task_id}: expected {args.variants_per_task} records, found {count}"
            )
        if len(styles_per_task[task_id]) != args.variants_per_task:
            errors.append(f"{task_id}: styles are not unique")

    print(f"Tasks:   {len(tasks):,}")
    print(f"Records: {len(records):,}")
    print()
    for split in ("train", "validation", "test"):
        groups = ("train",) if split == "train" else tuple(sorted(EVAL_GROUPS))
        for group in groups:
            print(
                f"{split:10} {group:11} "
                f"{split_task_counts[(split, group)]:4} tasks / "
                f"{split_record_counts[(split, group)]:5} records"
            )

    if errors:
        print(f"\nValidation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors[:100]:
            print(f"- {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"- ... and {len(errors) - 100} more", file=sys.stderr)
        return 1

    print("\nTeapot-v1 static validation passed.")
    print("Legacy MAXScript runtime validation remains a separate optional stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
