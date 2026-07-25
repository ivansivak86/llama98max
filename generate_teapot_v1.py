#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from mxs98_domain import format_number, format_vector, render_canonical_script

GENERATOR_VERSION = "teapot-v1-taskgen-2"
DEFAULT_SEED = 20260725
VARIANTS_PER_TASK = 4
EVAL_GROUPS = ("combo", "name_ood", "number_ood", "full_ood")

# Exact values in these sets are excluded from training. They deliberately
# include the literals from our first failed Windows 98 probe: 17, 7, -13, 4.
VALIDATION_RADII = (13, 27, 41, 58, 72, 86)
TEST_RADII = (17, 29, 43, 61, 73, 89, 97)

VALIDATION_POSITIONS = (
    -94, -81, -66, -51, -38, -24, -11, 3, 16, 28, 44, 57, 71, 84, 96,
)
TEST_POSITIONS = (
    -97, -83, -69, -53, -37, -23, -13, 4, 7, 17, 31, 46, 62, 79, 94,
)

PREFIXES = (
    "Amber", "Azure", "Brass", "Bronze", "Copper", "Coral", "Crimson",
    "Delta", "Echo", "Ember", "Frost", "Golden", "Indigo", "Ivory",
    "Jade", "Lunar", "Neon", "Nova", "Onyx", "Pixel", "Quartz", "Retro",
    "Ruby", "Silver", "Solar", "Steel", "Teal", "Turbo", "Velvet",
    "Violet", "Walnut", "Zenith",
)

STEMS = (
    "Pot", "Kettle", "Vessel", "Teapot", "Node", "Object", "Form",
    "Model", "Cup", "Urn", "Globe", "Shape", "Mesh", "Brew", "Scene",
    "Gizmo",
)

# Familiar anchors from v0 are intentionally retained. Their training specs
# differ from the four public "hero" test cases below.
RESERVED_TRAIN_SPECS: dict[str, tuple[int, tuple[int, int, int]]] = {
    "TeapotA": (15, (40, -10, 40)),
    "TeapotB": (25, (20, -10, 5)),
    "TeapotC": (10, (-20, 0, 5)),
    "TeapotD": (35, (10, 30, 0)),
    "TeapotE": (8, (-40, 20, 10)),
    "TeapotF": (30, (0, -30, 20)),
    "TeapotG": (12, (50, 10, -5)),
    "TeapotH": (40, (-50, -20, 25)),
}

HERO_TEST_SPECS: dict[str, dict[str, Any]] = {
    "combo": {
        "operation": "create_primitive",
        "primitive": "teapot",
        "name": "TeapotA",
        "radius": 25,
        "position": [20, -10, 5],
    },
    "name_ood": {
        "operation": "create_primitive",
        "primitive": "teapot",
        "name": "RetroPot",
        "radius": 25,
        "position": [20, -10, 5],
    },
    "number_ood": {
        "operation": "create_primitive",
        "primitive": "teapot",
        "name": "TeapotA",
        "radius": 17,
        "position": [7, -13, 4],
    },
    "full_ood": {
        "operation": "create_primitive",
        "primitive": "teapot",
        "name": "RetroPot",
        "radius": 17,
        "position": [7, -13, 4],
    },
}

INSTRUCTION_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "direct",
        "Create a teapot named {name} with radius {radius} at position {position}.",
    ),
    (
        "stepwise",
        "Add one teapot called {name}; use radius {radius} and coordinates {position}.",
    ),
    (
        "constraint_first",
        "Place a radius {radius} teapot named {name} at {position}.",
    ),
    (
        "compact",
        "Make {name} as a teapot with radius {radius}, positioned at {position}.",
    ),
)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def random_identifier(rng: random.Random) -> str:
    """Generate heterogeneous but MAXScript-safe copy targets."""
    prefix = rng.choice(PREFIXES)
    stem = rng.choice(STEMS)
    letter = rng.choice(string.ascii_uppercase)
    pattern = rng.randrange(6)

    if pattern == 0:
        return f"{prefix}{stem}"
    if pattern == 1:
        return f"{prefix}{stem}{rng.randrange(100):02d}"
    if pattern == 2:
        return f"{prefix}{stem}{rng.randrange(1000):03d}"
    if pattern == 3:
        return f"{stem}{letter}{rng.randrange(1000):03d}"
    if pattern == 4:
        return f"{prefix}{letter}{rng.randrange(1000):03d}"
    return f"{prefix}{stem}{letter}{rng.randrange(100):02d}"


def unique_names(
    count: int,
    rng: random.Random,
    forbidden: set[str],
) -> list[str]:
    names: list[str] = []
    seen = set(forbidden)
    attempts = 0
    while len(names) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("Unable to generate enough unique identifiers")
        candidate = random_identifier(rng)
        if candidate in seen or len(candidate) > 32:
            continue
        seen.add(candidate)
        names.append(candidate)
    forbidden.update(names)
    return names


def semantic_key(spec: dict[str, Any]) -> tuple[Any, ...]:
    return (spec["name"], spec["radius"], *spec["position"])


def make_spec(name: str, radius: int, position: Iterable[int]) -> dict[str, Any]:
    return {
        "operation": "create_primitive",
        "primitive": "teapot",
        "name": name,
        "radius": int(radius),
        "position": [int(value) for value in position],
    }


def sample_values(
    rng: random.Random,
    radii: tuple[int, ...],
    positions: tuple[int, ...],
) -> tuple[int, tuple[int, int, int]]:
    return (
        rng.choice(radii),
        (
            rng.choice(positions),
            rng.choice(positions),
            rng.choice(positions),
        ),
    )


def render_instruction(spec: dict[str, Any], template: str) -> str:
    return template.format(
        name=spec["name"],
        radius=format_number(spec["radius"]),
        position=format_vector(spec["position"], spaces=True),
    )


def task_record(
    task_id: str,
    split: str,
    eval_group: str,
    spec: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    script = render_canonical_script(spec)
    return {
        "id": task_id,
        "schema_version": 1,
        "split": split,
        "eval_group": eval_group,
        "spec": spec,
        "canonical_script": script,
        "expected": {
            "created_object_count": 1,
            "class": "Teapot",
            "name": spec["name"],
            "radius": spec["radius"],
            "position": spec["position"],
        },
        "provenance": {
            "source": "independently_generated",
            "task_generator": GENERATOR_VERSION,
            "task_generator_seed": seed,
            "template": "create_teapot_v1",
        },
    }


def sft_records(task: dict[str, Any]) -> list[dict[str, Any]]:
    script = task["canonical_script"]
    answer_hash = stable_hash(script)
    records: list[dict[str, Any]] = []

    for index, (style, template) in enumerate(INSTRUCTION_TEMPLATES, start=1):
        instruction = render_instruction(task["spec"], template)
        records.append(
            {
                "id": f"{task['id']}::det::{index:02d}",
                "instruction": instruction,
                "answer": script,
                "style": style,
                "template_id": f"teapot-v1-{index:02d}",
                "task_spec": task["spec"],
                "target": "3d-studio-max-2.5",
                "split": task["split"],
                "eval_group": task["eval_group"],
                "validation": {
                    "required_literals": True,
                    "maxscript_parse": "pending",
                    "maxscript_runtime": "pending",
                    "semantic": "pending",
                },
                "provenance": {
                    "source": "synthetic_clean_room",
                    "answer_source": "deterministic_renderer",
                    "instruction_source": "deterministic_template",
                    "generator": GENERATOR_VERSION,
                    "canonical_script_sha256": answer_hash,
                },
            }
        )

    return records


def add_unique_task(
    tasks: list[dict[str, Any]],
    used: set[tuple[Any, ...]],
    task_id: str,
    split: str,
    group: str,
    spec: dict[str, Any],
    seed: int,
) -> None:
    key = semantic_key(spec)
    if key in used:
        raise RuntimeError(f"Duplicate semantic task generated: {key}")
    used.add(key)
    tasks.append(task_record(task_id, split, group, spec, seed))


def build_dataset(
    *,
    seed: int,
    train_tasks: int,
    validation_per_group: int,
    test_per_group: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_tasks < len(RESERVED_TRAIN_SPECS):
        raise ValueError("train_tasks is too small for reserved anchor tasks")
    if validation_per_group < 1 or test_per_group < 1:
        raise ValueError("each evaluation group must contain at least one task")

    rng = random.Random(seed)

    all_heldout_radii = set(VALIDATION_RADII) | set(TEST_RADII)
    all_heldout_positions = set(VALIDATION_POSITIONS) | set(TEST_POSITIONS)
    train_radii = tuple(value for value in range(1, 100) if value not in all_heldout_radii)
    train_positions = tuple(
        value for value in range(-99, 100) if value not in all_heldout_positions
    )

    forbidden_names = set(RESERVED_TRAIN_SPECS) | {"RetroPot"}
    generated_train_names = unique_names(
        train_tasks - len(RESERVED_TRAIN_SPECS),
        rng,
        forbidden_names,
    )
    train_names = list(RESERVED_TRAIN_SPECS) + generated_train_names

    validation_new_names = unique_names(
        2 * validation_per_group,
        rng,
        forbidden_names,
    )
    test_generated_names = unique_names(
        2 * (test_per_group - 1),
        rng,
        forbidden_names,
    )

    tasks: list[dict[str, Any]] = []
    used: set[tuple[Any, ...]] = set()
    task_counter = 1

    # Every training task gets a unique name. The eight reserved tasks ensure
    # familiar reference values for our historical v0 comparison probes.
    for name in train_names:
        if name in RESERVED_TRAIN_SPECS:
            radius, position = RESERVED_TRAIN_SPECS[name]
        else:
            radius, position = sample_values(rng, train_radii, train_positions)
        spec = make_spec(name, radius, position)
        add_unique_task(
            tasks,
            used,
            f"teapot-v1-{task_counter:05d}",
            "train",
            "train",
            spec,
            seed,
        )
        task_counter += 1

    train_specs = [task["spec"] for task in tasks if task["split"] == "train"]
    seen_train_radii = tuple(sorted({spec["radius"] for spec in train_specs}))
    seen_train_positions = tuple(
        sorted({value for spec in train_specs for value in spec["position"]})
    )

    # Familiar-name pools are disjoint between validation and test, except for
    # the deliberate TeapotA hero probes in two different test categories.
    familiar_candidates = [name for name in train_names if name != "TeapotA"]
    rng.shuffle(familiar_candidates)
    required_familiar = 2 * validation_per_group + 2 * (test_per_group - 1)
    if required_familiar > len(familiar_candidates):
        raise ValueError("Not enough training names for held-out evaluation")

    cursor = 0
    val_combo_names = familiar_candidates[cursor:cursor + validation_per_group]
    cursor += validation_per_group
    val_number_names = familiar_candidates[cursor:cursor + validation_per_group]
    cursor += validation_per_group
    test_combo_names = ["TeapotA"] + familiar_candidates[
        cursor:cursor + test_per_group - 1
    ]
    cursor += test_per_group - 1
    test_number_names = ["TeapotA"] + familiar_candidates[
        cursor:cursor + test_per_group - 1
    ]

    val_name_names = validation_new_names[:validation_per_group]
    val_full_names = validation_new_names[validation_per_group:]
    test_name_names = ["RetroPot"] + test_generated_names[:test_per_group - 1]
    test_full_names = ["RetroPot"] + test_generated_names[test_per_group - 1:]

    def add_split(
        split: str,
        group_name_sources: dict[str, list[str]],
        ood_radii: tuple[int, ...],
        ood_positions: tuple[int, ...],
        hero_specs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        nonlocal task_counter
        hero_specs = hero_specs or {}

        for group in EVAL_GROUPS:
            for index, name in enumerate(group_name_sources[group]):
                if index == 0 and group in hero_specs:
                    spec = dict(hero_specs[group])
                    spec["position"] = list(spec["position"])
                    if spec["name"] != name:
                        raise AssertionError("Hero spec/name source mismatch")
                else:
                    if group in {"combo", "name_ood"}:
                        radius, position = sample_values(
                            rng,
                            seen_train_radii,
                            seen_train_positions,
                        )
                    else:
                        radius, position = sample_values(
                            rng,
                            ood_radii,
                            ood_positions,
                        )
                    spec = make_spec(name, radius, position)

                while semantic_key(spec) in used:
                    if group in {"combo", "name_ood"}:
                        radius, position = sample_values(
                            rng,
                            seen_train_radii,
                            seen_train_positions,
                        )
                    else:
                        radius, position = sample_values(
                            rng,
                            ood_radii,
                            ood_positions,
                        )
                    spec = make_spec(name, radius, position)

                add_unique_task(
                    tasks,
                    used,
                    f"teapot-v1-{task_counter:05d}",
                    split,
                    group,
                    spec,
                    seed,
                )
                task_counter += 1

    add_split(
        "validation",
        {
            "combo": val_combo_names,
            "name_ood": val_name_names,
            "number_ood": val_number_names,
            "full_ood": val_full_names,
        },
        VALIDATION_RADII,
        VALIDATION_POSITIONS,
    )
    add_split(
        "test",
        {
            "combo": test_combo_names,
            "name_ood": test_name_names,
            "number_ood": test_number_names,
            "full_ood": test_full_names,
        },
        TEST_RADII,
        TEST_POSITIONS,
        hero_specs=HERO_TEST_SPECS,
    )

    rng.shuffle(tasks)
    records = [record for task in tasks for record in sft_records(task)]
    rng.shuffle(records)
    return tasks, records


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Teapot-v1 open-literal copying curriculum."
    )
    parser.add_argument(
        "--tasks-output",
        type=Path,
        default=Path("data/tasks.teapot-v1.jsonl"),
    )
    parser.add_argument(
        "--records-output",
        type=Path,
        default=Path("data/paraphrases.teapot-v1.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-tasks", type=int, default=4096)
    parser.add_argument("--validation-per-group", type=int, default=128)
    parser.add_argument("--test-per-group", type=int, default=128)
    args = parser.parse_args()

    tasks, records = build_dataset(
        seed=args.seed,
        train_tasks=args.train_tasks,
        validation_per_group=args.validation_per_group,
        test_per_group=args.test_per_group,
    )
    write_jsonl(args.tasks_output, tasks)
    write_jsonl(args.records_output, records)

    task_counts = Counter((task["split"], task["eval_group"]) for task in tasks)
    record_counts = Counter((row["split"], row["eval_group"]) for row in records)

    print(f"Wrote {len(tasks):,} semantic tasks to {args.tasks_output}")
    print(f"Wrote {len(records):,} SFT records to {args.records_output}")
    print()
    for split in ("train", "validation", "test"):
        groups = ("train",) if split == "train" else EVAL_GROUPS
        for group in groups:
            print(
                f"{split:10} {group:11} "
                f"{task_counts[(split, group)]:4} tasks / "
                f"{record_counts[(split, group)]:5} records"
            )
    print()
    print("Reserved Windows 98 hero tests:")
    for group in EVAL_GROUPS:
        spec = HERO_TEST_SPECS[group]
        print(f"  {group:11} -> {render_canonical_script(spec)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
