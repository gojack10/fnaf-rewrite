"""Name Resolver — numeric IDs → human-readable names, loud on unknowns.

Consumes the dict outputs of `decoders.frame_events` and
`decoders.event_parameters` and injects `*_name` fields alongside every
resolvable numeric ID. Every unknown ID raises `NameResolutionError` so
missing Name Tables entries are caught *before* they silently corrupt
downstream LLM invariant extraction.

Scope ([[Name Resolver|c010e0f4-b420-4bbb-ad32-ee8ab0c0c7da]]):

- Condition / Action `(object_type, num)` → code name.
- Condition / Action `object_type` → type name (or the literal
  "Extension" for IDs >= `EXTENSION_BASE`, per Anaconda's
  `getObjectType()` contract).
- Expression-AST `(object_type, num)` → expression name, recursively.
- Top-level EventParameter `code` → parameter-type name.
- `Click.click` button int → ``{"Left", "Middle", "Right"}``.
- `Time.comparison` / `ExpressionParameter.comparison` int → operator
  symbol (``= <> <= < >= >``).

Out of scope this iteration:

- ``object_info`` / ``object_info_list`` instance handles → per-frame
  object-instance names. Those live in the frame's 0x2229 FrameItems +
  0x3340 FrameItemInstances chunks, which require cross-chunk context.
  Deferred to [[Output Emission|9f3a24e5-37cc-4b36-96b0-90543820a421]].

Contract:

- Pure: inputs are never mutated; every call returns a fresh dict.
- Loud: unknown IDs raise `NameResolutionError` with a location path
  ("frame_events/grp[12]/cond[0]/param[1]/expr[3]") for zero-ambiguity
  diagnostics.
- Tolerant of partially-decoded parameter shapes: if a parameter has
  only `{code, size, data_hex}` (pre-payload-decode), we tag
  `code_name` and leave the rest alone.
"""

from __future__ import annotations

from typing import Any

from ..name_tables.actions import (
    ACTION_EXTENSION_NAMES,
    ACTION_SYSTEM_NAMES,
)
from ..name_tables.conditions import (
    CONDITION_EXTENSION_NAMES,
    CONDITION_SYSTEM_NAMES,
)
from ..name_tables.expressions import (
    EXPRESSION_EXTENSION_NAMES,
    EXPRESSION_SYSTEM_NAMES,
)
from ..name_tables.object_types import EXTENSION_BASE, OBJECT_TYPE_NAMES
from ..name_tables.parameters import PARAMETER_NAMES

# Ported verbatim from
# reference/Anaconda/mmfparser/data/chunkloaders/parameters/loaders.py
# CLICK_NAMES (lines 453-457) and OPERATOR_LIST (lines 35-42). Kept here
# rather than in name_tables/ because they are one-dimensional and
# exclusively consumed by this resolver.
CLICK_NAMES: tuple[str, ...] = ("Left", "Middle", "Right")
OPERATOR_NAMES: tuple[str, ...] = ("=", "<>", "<=", "<", ">=", ">")


class NameResolutionError(LookupError):
    """Raised when a numeric ID isn't in the Name Tables.

    No silent fallback is the whole point of this layer — any missing
    ID is either a gap in the Name Tables port (port it) or a wire-level
    mismatch against Anaconda/CTFAK2.0 (fix the decoder first).
    """


# --- Leaf resolvers -----------------------------------------------------


def _resolve_object_type(object_type: int, *, path: str) -> str:
    """object_type id → name.

    Matches Anaconda's `getObjectType(id)`: `>= EXTENSION_BASE` returns
    the literal "Extension" (per-extension names would need a separate
    extensionDict we don't maintain).
    """
    if object_type >= EXTENSION_BASE:
        return "Extension"
    if object_type not in OBJECT_TYPE_NAMES:
        raise NameResolutionError(
            f"Unknown object_type={object_type} at {path}"
        )
    return OBJECT_TYPE_NAMES[object_type]


def _resolve_code(
    object_type: int,
    num: int,
    *,
    system_table: dict[int, dict[int, str]],
    extension_table: dict[int, str],
    kind: str,
    path: str,
) -> str:
    """Generic nested-vs-flat lookup for conditions / actions."""
    if object_type >= EXTENSION_BASE:
        if num not in extension_table:
            raise NameResolutionError(
                f"Unknown {kind} num={num} for extension object_type="
                f"{object_type} at {path}"
            )
        return extension_table[num]
    if object_type not in system_table:
        raise NameResolutionError(
            f"Unknown {kind} system-slot object_type={object_type} "
            f"(num={num}) at {path}"
        )
    table = system_table[object_type]
    if num not in table:
        object_type_name = OBJECT_TYPE_NAMES.get(object_type, "?")
        raise NameResolutionError(
            f"Unknown {kind} num={num} for object_type={object_type} "
            f"({object_type_name}) at {path}"
        )
    return table[num]


def _resolve_expression_name(
    object_type: int, num: int, *, path: str
) -> str:
    """Expression `(object_type, num)` → name.

    Special cases:
      * `(0, 0)` is the END marker and resolves via the operator slot.
      * `object_type == 0` is the operator slot (Plus/Minus/...).
      * `object_type >= EXTENSION_BASE` falls through to the flat
        extension table.
    """
    if object_type >= EXTENSION_BASE:
        if num not in EXPRESSION_EXTENSION_NAMES:
            raise NameResolutionError(
                f"Unknown expression num={num} for extension "
                f"object_type={object_type} at {path}"
            )
        return EXPRESSION_EXTENSION_NAMES[num]
    if object_type not in EXPRESSION_SYSTEM_NAMES:
        raise NameResolutionError(
            f"Unknown expression system-slot object_type={object_type} "
            f"(num={num}) at {path}"
        )
    table = EXPRESSION_SYSTEM_NAMES[object_type]
    if num not in table:
        object_type_name = OBJECT_TYPE_NAMES.get(object_type, "?")
        raise NameResolutionError(
            f"Unknown expression num={num} for object_type={object_type} "
            f"({object_type_name}) at {path}"
        )
    return table[num]


def _resolve_operator(comparison: int, *, path: str) -> str:
    if not (0 <= comparison < len(OPERATOR_NAMES)):
        raise NameResolutionError(
            f"Unknown comparison operator={comparison} at {path} "
            f"(valid 0..{len(OPERATOR_NAMES) - 1})"
        )
    return OPERATOR_NAMES[comparison]


def _resolve_click(click: int, *, path: str) -> str:
    if not (0 <= click < len(CLICK_NAMES)):
        raise NameResolutionError(
            f"Unknown click button={click} at {path} "
            f"(valid 0..{len(CLICK_NAMES) - 1})"
        )
    return CLICK_NAMES[click]


# --- Expression AST recursive resolver ---------------------------------


def resolve_expression(
    expr: dict[str, Any], *, path: str = "expr"
) -> dict[str, Any]:
    """Return a new dict with `object_type_name` + `num_name` injected.

    Idempotent: resolving an already-resolved record returns the same
    shape (re-resolution overwrites with identical values).
    """
    out = dict(expr)
    object_type = expr["object_type"]
    num = expr["num"]

    # object_type_name: operator-slot 0 doesn't map via OBJECT_TYPE_NAMES;
    # give it the stable label "Operator".
    if object_type == 0:
        out["object_type_name"] = "Operator"
    else:
        out["object_type_name"] = _resolve_object_type(
            object_type, path=path
        )

    # num_name: END marker + operator slot + system slots + per-type slots
    # + extensions, all routed through EXPRESSION_*_NAMES.
    out["num_name"] = _resolve_expression_name(
        object_type, num, path=path
    )
    return out


# --- Parameter resolver (outer + inner) --------------------------------


def resolve_parameter(
    param: dict[str, Any], *, path: str = "param"
) -> dict[str, Any]:
    """Resolve one EventParameter record.

    Handles both the opaque shape (`{code, size, data_hex}`, from
    `frame_events.EventParameter.as_dict`) and the decoded shape
    (`{code, kind, ...}`, from `event_parameters.decode_event_parameter`).
    """
    out = dict(param)

    code = param.get("code")
    if code is None:
        raise NameResolutionError(
            f"Parameter record missing `code` at {path}: {param!r}"
        )
    if code not in PARAMETER_NAMES:
        raise NameResolutionError(
            f"Unknown parameter code={code} at {path}"
        )
    out["code_name"] = PARAMETER_NAMES[code]

    kind = param.get("kind")
    if kind is None:
        # Opaque — nothing more to resolve here.
        return out

    if kind == "Click":
        out["click_name"] = _resolve_click(param["click"], path=path)
    elif kind == "Time":
        out["comparison_name"] = _resolve_operator(
            param["comparison"], path=path
        )
    elif kind == "Object":
        out["object_type_name"] = _resolve_object_type(
            param["object_type"], path=path
        )
    elif kind == "ExpressionParameter":
        out["comparison_name"] = _resolve_operator(
            param["comparison"], path=path
        )
        out["expressions"] = [
            resolve_expression(e, path=f"{path}/expr[{i}]")
            for i, e in enumerate(param["expressions"])
        ]
    # Other kinds (Short, Int, Key, Sample, Position, Create) carry no
    # additional table-resolvable IDs. Their semantic label comes from
    # the containing condition/action, which Name Resolver already
    # stamped onto the record.

    return out


# --- Condition / Action / EventGroup / FrameEvents resolvers -----------


def resolve_condition(
    cond: dict[str, Any], *, path: str = "cond"
) -> dict[str, Any]:
    """Return a new dict with object/num names + every parameter resolved."""
    out = dict(cond)
    object_type = cond["object_type"]
    num = cond["num"]
    out["object_type_name"] = _resolve_object_type(object_type, path=path)
    out["num_name"] = _resolve_code(
        object_type,
        num,
        system_table=CONDITION_SYSTEM_NAMES,
        extension_table=CONDITION_EXTENSION_NAMES,
        kind="condition",
        path=path,
    )
    out["parameters"] = [
        resolve_parameter(p, path=f"{path}/param[{i}]")
        for i, p in enumerate(cond.get("parameters", []))
    ]
    return out


def resolve_action(
    act: dict[str, Any], *, path: str = "act"
) -> dict[str, Any]:
    """Return a new dict with object/num names + every parameter resolved."""
    out = dict(act)
    object_type = act["object_type"]
    num = act["num"]
    out["object_type_name"] = _resolve_object_type(object_type, path=path)
    out["num_name"] = _resolve_code(
        object_type,
        num,
        system_table=ACTION_SYSTEM_NAMES,
        extension_table=ACTION_EXTENSION_NAMES,
        kind="action",
        path=path,
    )
    out["parameters"] = [
        resolve_parameter(p, path=f"{path}/param[{i}]")
        for i, p in enumerate(act.get("parameters", []))
    ]
    return out


def resolve_event_group(
    group: dict[str, Any], *, path: str = "group"
) -> dict[str, Any]:
    """Resolve every condition + action inside one event group."""
    out = dict(group)
    out["conditions"] = [
        resolve_condition(c, path=f"{path}/cond[{i}]")
        for i, c in enumerate(group.get("conditions", []))
    ]
    out["actions"] = [
        resolve_action(a, path=f"{path}/act[{i}]")
        for i, a in enumerate(group.get("actions", []))
    ]
    return out


def resolve_frame_events(
    fe: dict[str, Any], *, path: str = "frame_events"
) -> dict[str, Any]:
    """Resolve every event group inside a FrameEvents.as_dict() dict."""
    out = dict(fe)
    out["event_groups"] = [
        resolve_event_group(g, path=f"{path}/grp[{i}]")
        for i, g in enumerate(fe.get("event_groups", []))
    ]
    return out
