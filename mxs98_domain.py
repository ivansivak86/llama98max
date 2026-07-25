from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

_SAFE_OBJECT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")


class SpecError(ValueError):
    """Raised when a semantic task specification is invalid."""


def _require_number(value: Any, field_name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{field_name} must be a real number, got {value!r}")
    if not math.isfinite(float(value)):
        raise SpecError(f"{field_name} must be finite, got {value!r}")
    return value


def format_number(value: Any) -> str:
    """Return a deterministic MAXScript-compatible number."""
    number = _require_number(value, "number")
    if isinstance(number, int):
        return str(number)
    if number.is_integer():
        return str(int(number))
    return format(number, ".8g")


def format_vector(value: Any, *, spaces: bool = False) -> str:
    """Render a three-component MAXScript point/vector."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SpecError(f"Expected a three-component vector, got {value!r}")
    separator = ", " if spaces else ","
    return "[" + separator.join(format_number(item) for item in value) + "]"


def format_instruction_vector(value: Any) -> str:
    """Backward-compatible human-readable vector formatter used by v0."""
    return format_vector(value, spaces=True)


def render_canonical_script(spec: Mapping[str, Any]) -> str:
    """Render the deliberately narrow Teapot-v1 MAXScript subset."""
    operation = spec.get("operation")
    primitive = spec.get("primitive")

    if operation != "create_primitive":
        raise SpecError(f"Unsupported operation: {operation!r}")
    if primitive != "teapot":
        raise SpecError(f"Unsupported primitive: {primitive!r}")

    name = spec.get("name")
    if not isinstance(name, str) or not _SAFE_OBJECT_NAME.fullmatch(name):
        raise SpecError(
            "Object name must begin with a letter and contain only letters, "
            f"digits, or underscores: {name!r}"
        )

    radius = _require_number(spec.get("radius"), "radius")
    if radius <= 0:
        raise SpecError(f"radius must be greater than zero: {radius!r}")

    return (
        f'teapot name:"{name}" '
        f"radius:{format_number(radius)} "
        f"pos:{format_vector(spec.get('position'))}"
    )
