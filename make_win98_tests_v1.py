#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HEROES: dict[str, tuple[str, int, list[int]]] = {
    "combo": ("TeapotA", 25, [20, -10, 5]),
    "name_ood": ("RetroPot", 25, [20, -10, 5]),
    "number_ood": ("TeapotA", 17, [7, -13, 4]),
    "full_ood": ("RetroPot", 17, [7, -13, 4]),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print four short Windows 98 commands for Teapot-v1 hero tests."
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("data/paraphrases.teapot-v1.jsonl"),
    )
    parser.add_argument("--run", default="R")
    parser.add_argument("--model", default="V1.BIN")
    parser.add_argument("--tokenizer", default="T.BIN")
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with args.records.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))

    for group, (name, radius, position) in HEROES.items():
        candidates = [
            row
            for row in rows
            if row.get("split") == "test"
            and row.get("eval_group") == group
            and row["task_spec"]["name"] == name
            and row["task_spec"]["radius"] == radius
            and row["task_spec"]["position"] == position
        ]
        if not candidates:
            raise RuntimeError(f"Hero record not found for {group}")

        row = min(candidates, key=lambda item: len(item["instruction"]))
        prompt = f"Instruction: {row['instruction']} MAXScript:"
        command = (
            f'{args.run} {args.model} -z {args.tokenizer} -t 0 '
            f'-n {args.steps} -i "{prompt}"'
        )

        print(f"[{group}]")
        print(command)
        print(f"Expected: {row['answer']}")
        print(f"Command length: {len(command)}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
