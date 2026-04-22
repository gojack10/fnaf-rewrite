"""Tests for `fnaf_parser.algorithm.name_resolver`.

The Name Resolver is the last layer before Output Emission. Every test
here either

  (a) confirms a numeric ID is correctly injected as `*_name`, or
  (b) confirms the resolver fails *loudly* — `NameResolutionError` with
      a path-qualified message — on any unknown ID.

The loud-failure contract is the whole reason this layer exists: a
silent fallback would let Name Tables gaps silently corrupt the LLM
invariant extractor downstream. We pin it here so a well-meaning
"just add a default" refactor fails immediately.

Shape pins
----------

* `_resolve_object_type`: `>= EXTENSION_BASE` → literal `"Extension"`
  (matches Anaconda's `getObjectType(id)`).
* `resolve_expression`: operator slot (object_type=0) → `"Operator"`
  label — there's no `OBJECT_TYPE_NAMES[0]` entry for that slot.
* `resolve_parameter` tolerates both the opaque shape
  (`{code, size, data_hex}`) and the decoded shape (`{code, kind, ...}`).
"""

from __future__ import annotations

import pytest

from fnaf_parser.algorithm.name_resolver import (
    CLICK_NAMES,
    OPERATOR_NAMES,
    NameResolutionError,
    resolve_action,
    resolve_condition,
    resolve_event_group,
    resolve_expression,
    resolve_frame_events,
    resolve_parameter,
)
from fnaf_parser.name_tables.object_types import EXTENSION_BASE


# --- Module-level constants --------------------------------------------


def test_click_names_match_anaconda_ordering():
    """Anaconda `parameters/loaders.py`: LEFT=0, MIDDLE=1, RIGHT=2."""
    assert CLICK_NAMES == ("Left", "Middle", "Right")


def test_operator_names_match_anaconda_ordering():
    """Anaconda `parameters/loaders.py`:
    EQUAL=0, NOT_EQUAL=1, LESS_EQUAL=2, LESS=3, GREATER_EQUAL=4, GREATER=5.
    """
    assert OPERATOR_NAMES == ("=", "<>", "<=", "<", ">=", ">")


def test_name_resolution_error_is_lookup_error():
    """Callers should be able to catch it via `except LookupError`."""
    assert issubclass(NameResolutionError, LookupError)


# --- resolve_condition: object_type + num ------------------------------


def test_resolve_condition_builtin_system():
    """System/-3 code -1 = StartOfFrame."""
    cond = {"object_type": -3, "num": -1, "parameters": []}
    out = resolve_condition(cond)
    assert out["object_type_name"] == "Game"
    assert out["num_name"] == "StartOfFrame"


def test_resolve_condition_per_type_slot():
    """object_type=2 (Active) + code=-81 = ObjectClicked (per-type slot)."""
    cond = {"object_type": 2, "num": -81, "parameters": []}
    out = resolve_condition(cond)
    assert out["object_type_name"] == "Active"
    assert out["num_name"] == "ObjectClicked"


def test_resolve_condition_extension_object_type():
    """object_type >= EXTENSION_BASE routes to CONDITION_EXTENSION_NAMES
    (flat table) and labels the object_type "Extension"."""
    cond = {"object_type": EXTENSION_BASE, "num": -14, "parameters": []}
    out = resolve_condition(cond)
    assert out["object_type_name"] == "Extension"
    assert out["num_name"] == "OnCollision"


def test_resolve_condition_is_pure():
    """Input dict is never mutated — returns a fresh dict every call."""
    cond = {"object_type": -3, "num": -1, "parameters": []}
    snapshot = dict(cond)
    _ = resolve_condition(cond)
    assert cond == snapshot
    assert "object_type_name" not in cond
    assert "num_name" not in cond


def test_resolve_condition_preserves_other_fields():
    """Resolver only injects names — all other fields pass through."""
    cond = {
        "object_type": -3,
        "num": -1,
        "parameters": [],
        "flags": 0x1234,
        "identifier": 0xDEAD,
    }
    out = resolve_condition(cond)
    assert out["flags"] == 0x1234
    assert out["identifier"] == 0xDEAD


def test_resolve_condition_unknown_object_type_raises():
    # 25 sits in the 10..31 gap: below EXTENSION_BASE (so not an
    # extension) and not in OBJECT_TYPE_NAMES (built-ins are -7..-1
    # and 0..9).
    cond = {"object_type": 25, "num": -1, "parameters": []}
    with pytest.raises(NameResolutionError, match="object_type=25"):
        resolve_condition(cond, path="grp[0]/cond[0]")


def test_resolve_condition_unknown_num_falls_through_both_tables():
    """Anaconda `common.pyx:138-142` `getName()` contract:
    `systemDict[objectType]` first, else flat `extensionDict`. A num
    missing from *both* raises, and the message cites both lookup paths
    so triage can tell whether the gap is in the nested dict or the flat
    table.
    """
    # Backdrop (object_type=1) has no CONDITION_SYSTEM_NAMES entry;
    # num=-9999 is absent from CONDITION_EXTENSION_NAMES too.
    cond = {"object_type": 1, "num": -9999, "parameters": []}
    with pytest.raises(
        NameResolutionError,
        match=r"Unknown condition num=-9999.*not in systemDict.*not in extensionDict",
    ):
        resolve_condition(cond, path="grp[0]/cond[0]")


def test_resolve_condition_system_to_extension_fallback():
    """Built-in visual object types inherit common-object conditions
    (AnimationFrame, IsOverlapping, …) via the `extensionDict` fallback.
    Mirrors Anaconda's `getName()` — see common.pyx:138-142.
    """
    # Backdrop has no CONDITION_SYSTEM_NAMES entry, so num=-1 must fall
    # through to CONDITION_EXTENSION_NAMES[-1] = "AnimationFrame".
    cond = {"object_type": 1, "num": -1, "parameters": []}
    out = resolve_condition(cond)
    assert out["object_type_name"] == "Backdrop"
    assert out["num_name"] == "AnimationFrame"


def test_resolve_condition_unknown_num_raises():
    """Valid object_type, missing num — must name the object_type so a
    triage pass can tell whether it's a name-table gap or a decoder
    mismatch."""
    cond = {"object_type": -3, "num": -999, "parameters": []}
    with pytest.raises(NameResolutionError, match="Unknown condition num=-999"):
        resolve_condition(cond, path="grp[0]/cond[0]")


def test_resolve_condition_error_path_is_propagated():
    """Unknown-ID errors must include the supplied path for triage."""
    cond = {"object_type": 25, "num": -1, "parameters": []}
    with pytest.raises(
        NameResolutionError, match="at frame_events/grp\\[12\\]/cond\\[3\\]"
    ):
        resolve_condition(cond, path="frame_events/grp[12]/cond[3]")


# --- resolve_action ----------------------------------------------------


def test_resolve_action_builtin_system():
    """Speaker (-2) code=0 = PlaySample."""
    act = {"object_type": -2, "num": 0, "parameters": []}
    out = resolve_action(act)
    assert out["object_type_name"] == "Speaker"
    assert out["num_name"] == "PlaySample"


def test_resolve_action_extension_object_type():
    """object_type >= 32 routes to ACTION_EXTENSION_NAMES."""
    act = {"object_type": 45, "num": 24, "parameters": []}
    out = resolve_action(act)
    assert out["object_type_name"] == "Extension"
    assert out["num_name"] == "Destroy"


def test_resolve_action_extension_native_num_stamps_ext_marker():
    """Extension-native action nums (not in the flat ACTION_EXTENSION_NAMES)
    belong to per-extension DLLs whose name tables we can't reach from
    Linux. Anaconda's `getName()` returns `None` for these; we stamp an
    explicit `ExtAction_<num>` marker so the name field is never None
    and extension-native slots are greppable downstream.
    """
    act = {"object_type": 45, "num": 9999, "parameters": []}
    out = resolve_action(act)
    assert out["object_type_name"] == "Extension"
    assert out["num_name"] == "ExtAction_9999"


def test_resolve_action_system_to_extension_fallback():
    """Active (object_type=2) only has nums 80-89 in its nested
    ACTION_SYSTEM_NAMES dict; nums below 80 inherit the common-object
    actions (SetAlphaCoefficient=65, Destroy=24, …). This fallback is
    load-bearing — Anaconda `common.pyx:138-142` pins it as part of the
    shared `getName()` contract.
    """
    act = {"object_type": 2, "num": 65, "parameters": []}
    out = resolve_action(act)
    assert out["object_type_name"] == "Active"
    assert out["num_name"] == "SetAlphaCoefficient"


# --- resolve_expression ------------------------------------------------


def test_resolve_expression_operator_plus():
    """Operator slot (object_type=0): code=2 = Plus."""
    expr = {"object_type": 0, "num": 2}
    out = resolve_expression(expr)
    assert out["object_type_name"] == "Operator"
    assert out["num_name"] == "Plus"


def test_resolve_expression_end_marker():
    """(0, 0) is the Expression-AST END marker — resolves via the
    operator slot to `"End"`."""
    expr = {"object_type": 0, "num": 0}
    out = resolve_expression(expr)
    assert out["object_type_name"] == "Operator"
    assert out["num_name"] == "End"


def test_resolve_expression_system_literal():
    """System literal: object_type=-1, num=0 = Long."""
    expr = {"object_type": -1, "num": 0}
    out = resolve_expression(expr)
    assert out["object_type_name"] == "System"
    assert out["num_name"] == "Long"


def test_resolve_expression_object_ref_per_type_slot():
    """Per-type slot: object_type=7 (Counter), num=80 = CounterValue."""
    expr = {"object_type": 7, "num": 80}
    out = resolve_expression(expr)
    assert out["object_type_name"] == "Counter"
    assert out["num_name"] == "CounterValue"


def test_resolve_expression_extension():
    """object_type >= 32 routes to EXPRESSION_EXTENSION_NAMES."""
    expr = {"object_type": 37, "num": 1}
    out = resolve_expression(expr)
    assert out["object_type_name"] == "Extension"
    assert out["num_name"] == "YPosition"


def test_resolve_expression_unknown_operator_raises():
    expr = {"object_type": 0, "num": 999}
    with pytest.raises(
        NameResolutionError, match="Unknown expression num=999"
    ):
        resolve_expression(expr)


def test_resolve_expression_unknown_object_type_raises():
    """object_type not in OBJECT_TYPE_NAMES and not an extension."""
    expr = {"object_type": 25, "num": 0}  # 25 is an unused slot
    with pytest.raises(NameResolutionError):
        resolve_expression(expr)


def test_resolve_expression_extension_native_num_stamps_ext_marker():
    """Extension-native expression nums (not in EXPRESSION_EXTENSION_NAMES)
    belong to per-extension DLLs. Same Ext<Kind>_<num> marker contract as
    actions/conditions: never None, always greppable."""
    expr = {"object_type": EXTENSION_BASE + 10, "num": 99999}
    out = resolve_expression(expr)
    assert out["object_type_name"] == "Extension"
    assert out["num_name"] == "ExtExpression_99999"


def test_resolve_expression_is_pure():
    expr = {"object_type": 0, "num": 2}
    snapshot = dict(expr)
    _ = resolve_expression(expr)
    assert expr == snapshot


# --- resolve_parameter (opaque vs decoded shape) -----------------------


def test_resolve_parameter_opaque_shape_only_injects_code_name():
    """Opaque `{code, size, data_hex}` shape — we can label the code
    but have nothing else to resolve."""
    param = {"code": 1, "size": 13, "data_hex": "ab" * 9}
    out = resolve_parameter(param)
    assert out["code_name"] == "OBJECT"
    # Nothing else was injected.
    assert "kind" not in out
    assert "click_name" not in out
    assert "comparison_name" not in out


def test_resolve_parameter_click_kind():
    """kind=Click → inject click_name (Left/Middle/Right)."""
    param = {"code": 32, "kind": "Click", "click": 1, "trailing_hex": ""}
    out = resolve_parameter(param)
    assert out["code_name"] == "Click"
    assert out["click_name"] == "Middle"


@pytest.mark.parametrize(
    "click,expected",
    [(0, "Left"), (1, "Middle"), (2, "Right")],
)
def test_resolve_parameter_click_all_values(click, expected):
    param = {"code": 32, "kind": "Click", "click": click, "trailing_hex": ""}
    assert resolve_parameter(param)["click_name"] == expected


def test_resolve_parameter_click_out_of_range_raises():
    param = {"code": 32, "kind": "Click", "click": 3, "trailing_hex": ""}
    with pytest.raises(NameResolutionError, match="Unknown click button=3"):
        resolve_parameter(param)


def test_resolve_parameter_time_kind():
    """kind=Time → inject comparison_name (operator symbol)."""
    param = {
        "code": 2,
        "kind": "Time",
        "timer": 1000,
        "loops": 0,
        "comparison": 4,
        "trailing_hex": "",
    }
    out = resolve_parameter(param)
    assert out["code_name"] == "TIME"
    assert out["comparison_name"] == ">="


@pytest.mark.parametrize(
    "idx,expected",
    [(0, "="), (1, "<>"), (2, "<="), (3, "<"), (4, ">="), (5, ">")],
)
def test_resolve_parameter_time_all_operators(idx, expected):
    param = {
        "code": 2,
        "kind": "Time",
        "timer": 0,
        "loops": 0,
        "comparison": idx,
        "trailing_hex": "",
    }
    assert resolve_parameter(param)["comparison_name"] == expected


def test_resolve_parameter_time_out_of_range_raises():
    param = {
        "code": 2,
        "kind": "Time",
        "timer": 0,
        "loops": 0,
        "comparison": 6,
        "trailing_hex": "",
    }
    with pytest.raises(
        NameResolutionError, match="Unknown comparison operator=6"
    ):
        resolve_parameter(param)


def test_resolve_parameter_object_kind():
    """kind=Object → inject object_type_name."""
    param = {
        "code": 1,
        "kind": "Object",
        "handle": 5,
        "object_type": 2,
        "object_info": 5,
        "object_info_list": 0,
        "type_id": 2,
        "trailing_hex": "",
    }
    out = resolve_parameter(param)
    assert out["code_name"] == "OBJECT"
    assert out["object_type_name"] == "Active"


def test_resolve_parameter_object_kind_extension():
    """object_type >= 32 → "Extension" literal (matches getObjectType)."""
    param = {
        "code": 1,
        "kind": "Object",
        "handle": 5,
        "object_type": 37,
        "object_info": 5,
        "object_info_list": 0,
        "type_id": 37,
        "trailing_hex": "",
    }
    out = resolve_parameter(param)
    assert out["object_type_name"] == "Extension"


def test_resolve_parameter_expression_parameter_recurses():
    """kind=ExpressionParameter → resolve comparison + each expression."""
    param = {
        "code": 22,
        "kind": "ExpressionParameter",
        "comparison": 0,
        "expressions": [
            {"object_type": -1, "num": 0},  # Long literal
            {"object_type": 0, "num": 2},  # Plus operator
            {"object_type": 0, "num": 0},  # End marker
        ],
        "trailing_hex": "",
    }
    out = resolve_parameter(param)
    assert out["code_name"] == "EXPRESSION"
    assert out["comparison_name"] == "="
    assert len(out["expressions"]) == 3
    assert out["expressions"][0]["num_name"] == "Long"
    assert out["expressions"][1]["num_name"] == "Plus"
    assert out["expressions"][2]["num_name"] == "End"
    # Original expressions list was not mutated.
    assert "num_name" not in param["expressions"][0]


def test_resolve_parameter_expression_parameter_error_path_points_to_expr():
    """An unknown-ID inside an ExpressionParameter must cite its
    `.../expr[N]` index for triage.

    Uses `object_type=25` — a slot in the 10..31 non-extension gap that's
    neither in OBJECT_TYPE_NAMES nor >= EXTENSION_BASE, so the object-type
    resolver raises first. Extension territory (>= 32) intentionally
    never raises on unknown nums (it stamps ExtExpression_<num>), so the
    error-path test has to use a non-extension unknown.
    """
    param = {
        "code": 22,
        "kind": "ExpressionParameter",
        "comparison": 0,
        "expressions": [
            {"object_type": -1, "num": 0},
            {"object_type": 25, "num": 0},  # unknown non-extension slot
        ],
        "trailing_hex": "",
    }
    with pytest.raises(NameResolutionError, match=r"expr\[1\]"):
        resolve_parameter(param, path="cond[0]/param[0]")


def test_resolve_parameter_kinds_without_resolvable_ids_passthrough():
    """Short / Int / Key / Sample / Position / Create carry no ID we
    can further resolve — we stamp code_name only and leave the rest."""
    for kind in ("Short", "Int", "Key", "Sample", "Position", "Create"):
        param = {"code": 10, "kind": kind, "trailing_hex": ""}
        out = resolve_parameter(param)
        assert out["code_name"] == "SHORT"
        assert "click_name" not in out
        assert "comparison_name" not in out


def test_resolve_parameter_missing_code_raises():
    param = {"kind": "Click", "click": 0}
    with pytest.raises(NameResolutionError, match="missing `code`"):
        resolve_parameter(param)


def test_resolve_parameter_unknown_code_raises():
    param = {"code": 999, "size": 4, "data_hex": ""}
    with pytest.raises(
        NameResolutionError, match="Unknown parameter code=999"
    ):
        resolve_parameter(param)


def test_resolve_parameter_is_pure():
    param = {"code": 32, "kind": "Click", "click": 0, "trailing_hex": ""}
    snapshot = dict(param)
    _ = resolve_parameter(param)
    assert param == snapshot


# --- resolve_event_group / resolve_frame_events ------------------------


def test_resolve_event_group_walks_conditions_and_actions():
    group = {
        "conditions": [
            {"object_type": -3, "num": -1, "parameters": []},  # StartOfFrame
        ],
        "actions": [
            {"object_type": -2, "num": 0, "parameters": []},  # PlaySample
        ],
    }
    out = resolve_event_group(group)
    assert out["conditions"][0]["num_name"] == "StartOfFrame"
    assert out["actions"][0]["num_name"] == "PlaySample"


def test_resolve_event_group_is_pure():
    group = {
        "conditions": [{"object_type": -3, "num": -1, "parameters": []}],
        "actions": [],
    }
    snapshot_cond = dict(group["conditions"][0])
    _ = resolve_event_group(group)
    assert group["conditions"][0] == snapshot_cond
    assert "num_name" not in group["conditions"][0]


def test_resolve_frame_events_end_to_end():
    """One full FrameEvents.as_dict() roundtrip.

    Mirrors the shape the CLI's `algorithm emit` stage will hand this
    resolver — one frame with one group, one condition + one action, and
    a mix of opaque and decoded parameters."""
    fe = {
        "event_groups": [
            {
                "conditions": [
                    {
                        "object_type": -3,
                        "num": -1,
                        "parameters": [],
                    }
                ],
                "actions": [
                    {
                        "object_type": -2,
                        "num": 0,
                        "parameters": [
                            {  # opaque Sample param (pre-decode)
                                "code": 6,
                                "size": 40,
                                "data_hex": "00" * 36,
                            },
                        ],
                    }
                ],
            }
        ]
    }
    out = resolve_frame_events(fe)
    cond = out["event_groups"][0]["conditions"][0]
    act = out["event_groups"][0]["actions"][0]
    assert cond["num_name"] == "StartOfFrame"
    assert act["num_name"] == "PlaySample"
    assert act["parameters"][0]["code_name"] == "SAMPLE"


def test_resolve_frame_events_error_path_fully_qualified():
    """Errors deep inside a FrameEvents walk name their full chain.

    object_type=25 lives in the non-extension gap (not in OBJECT_TYPE_NAMES
    and not >= EXTENSION_BASE) so the leaf resolver raises with the full
    path propagated back through every walker frame.
    """
    fe = {
        "event_groups": [
            {
                "conditions": [
                    {
                        "object_type": -3,
                        "num": -1,
                        "parameters": [
                            {
                                "code": 22,
                                "kind": "ExpressionParameter",
                                "comparison": 0,
                                "expressions": [
                                    {"object_type": 25, "num": 0},
                                ],
                                "trailing_hex": "",
                            },
                        ],
                    }
                ],
                "actions": [],
            }
        ]
    }
    with pytest.raises(
        NameResolutionError,
        match=r"frame_events/grp\[0\]/cond\[0\]/param\[0\]/expr\[0\]",
    ):
        resolve_frame_events(fe)


def test_resolve_frame_events_handles_empty_groups():
    """Empty groups/frames must be a no-op — the resolver is pure."""
    fe = {"event_groups": []}
    out = resolve_frame_events(fe)
    assert out == {"event_groups": []}


# --- FNAF 1 smoke test -------------------------------------------------


def test_resolve_frame_events_smoke_fnaf1_like_shape():
    """A realistic-ish FNAF 1 event-group shape: a keyboard condition
    (KeyPressed), a counter compare, two actions (PlaySample + SetCounter).
    Verifies the full resolver stack (condition + action + parameters
    with Click/Time/Object/ExpressionParameter kinds) is internally
    consistent end-to-end.
    """
    fe = {
        "event_groups": [
            {
                "conditions": [
                    {  # Keyboard / KeyPressed
                        "object_type": -6,
                        "num": -1,
                        "parameters": [
                            {
                                "code": 14,
                                "kind": "Key",
                                "value": 32,
                                "trailing_hex": "",
                            }
                        ],
                    },
                    {  # Counter compare (object_type=7, num=-81)
                        "object_type": 7,
                        "num": -81,
                        "parameters": [
                            {
                                "code": 22,
                                "kind": "ExpressionParameter",
                                "comparison": 3,
                                "expressions": [
                                    {"object_type": -1, "num": 0},  # Long
                                    {"object_type": 0, "num": 0},  # End
                                ],
                                "trailing_hex": "",
                            }
                        ],
                    },
                ],
                "actions": [
                    {  # Speaker.PlaySample with opaque sample param
                        "object_type": -2,
                        "num": 0,
                        "parameters": [
                            {"code": 6, "size": 40, "data_hex": ""}
                        ],
                    },
                    {  # Counter.SetCounterValue
                        "object_type": 7,
                        "num": 80,
                        "parameters": [
                            {
                                "code": 22,
                                "kind": "ExpressionParameter",
                                "comparison": 0,
                                "expressions": [
                                    {"object_type": -1, "num": 0},  # Long
                                    {"object_type": 0, "num": 0},  # End
                                ],
                                "trailing_hex": "",
                            }
                        ],
                    },
                ],
            }
        ]
    }
    out = resolve_frame_events(fe)
    g = out["event_groups"][0]
    # Conditions
    assert g["conditions"][0]["object_type_name"] == "Keyboard"
    assert g["conditions"][0]["num_name"] == "KeyPressed"
    assert g["conditions"][1]["object_type_name"] == "Counter"
    assert g["conditions"][1]["num_name"] == "CompareCounter"
    assert g["conditions"][1]["parameters"][0]["comparison_name"] == "<"
    # Actions
    assert g["actions"][0]["object_type_name"] == "Speaker"
    assert g["actions"][0]["num_name"] == "PlaySample"
    assert g["actions"][0]["parameters"][0]["code_name"] == "SAMPLE"
    assert g["actions"][1]["object_type_name"] == "Counter"
    assert g["actions"][1]["num_name"] == "SetCounterValue"
    assert g["actions"][1]["parameters"][0]["expressions"][0]["num_name"] == "Long"
    assert g["actions"][1]["parameters"][0]["expressions"][1]["num_name"] == "End"
