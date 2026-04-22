"""Smoke tests for `fnaf_parser.name_tables`.

These tables are pure-data ports of the 5 Anaconda name maps
(`mmfparser/data/chunkloaders/{conditions,actions,expressions,parameters}/names.py`
plus `objectinfo.pyx`). The tables themselves are the ground truth —
the reference tree is the oracle, and duplicating every entry as a
test would just re-encode the same source.

Instead we spot-check ≥10 well-known entries per table for each
non-trivial shape (built-in system IDs, per-type slots, flat
extensionDict, operator slot). The goal is "the port happened, the
keys are typed correctly, and regressions surface loudly" — not
"verify Anaconda's data, again, here."

Shape pin
---------

`CONDITION_SYSTEM_NAMES`, `ACTION_SYSTEM_NAMES`, and
`EXPRESSION_SYSTEM_NAMES` are nested `dict[int, dict[int, str]]` because
the same `code` means different things depending on `object_type`
(e.g. `CONDITION_SYSTEM_NAMES[2][-81]` is `ObjectClicked` while
`CONDITION_SYSTEM_NAMES[9][-81]` is `SubApplicationFrameChanged`). A
flat `dict[int, str]` would collide — we pin that shape here so any
flattening refactor fails loudly.

`PARAMETER_NAMES` and `OBJECT_TYPE_NAMES` are flat by design.
"""

from __future__ import annotations

from fnaf_parser.name_tables import (
    actions,
    conditions,
    expressions,
    object_types,
    parameters,
)

# --- Conditions --------------------------------------------------------


def test_condition_system_names_shape_is_nested():
    """Pin the nested shape: outer key = object_type, inner = code."""
    assert isinstance(conditions.CONDITION_SYSTEM_NAMES, dict)
    # Every outer value must be a dict — that's what makes the (code,
    # object_type_id) disambiguation work.
    for obj_type, inner in conditions.CONDITION_SYSTEM_NAMES.items():
        assert isinstance(obj_type, int)
        assert isinstance(inner, dict), (
            f"CONDITION_SYSTEM_NAMES[{obj_type}] must be nested dict, "
            f"got {type(inner).__name__}"
        )


def test_condition_system_names_disambiguates_code_minus_81():
    """Sentinel collision: same code, different names per object_type."""
    assert conditions.CONDITION_SYSTEM_NAMES[2][-81] == "ObjectClicked"
    assert (
        conditions.CONDITION_SYSTEM_NAMES[9][-81]
        == "SubApplicationFrameChanged"
    )
    assert conditions.CONDITION_SYSTEM_NAMES[7][-81] == "CompareCounter"


def test_condition_system_names_spot_check():
    # 10+ entries across multiple object_types — built-in (-1 System,
    # -2 Speaker, -3 Game, -6 Keyboard, -7 Player) + per-type slots.
    assert conditions.CONDITION_SYSTEM_NAMES[-1][-1] == "Always"
    assert conditions.CONDITION_SYSTEM_NAMES[-1][-2] == "Never"
    assert conditions.CONDITION_SYSTEM_NAMES[-1][-24] == "OrFiltered"
    assert conditions.CONDITION_SYSTEM_NAMES[-1][-25] == "OrLogical"
    assert conditions.CONDITION_SYSTEM_NAMES[-2][-5] == "MusicFinished"
    assert conditions.CONDITION_SYSTEM_NAMES[-3][-1] == "StartOfFrame"
    assert conditions.CONDITION_SYSTEM_NAMES[-3][-2] == "EndOfFrame"
    assert conditions.CONDITION_SYSTEM_NAMES[-6][-1] == "KeyPressed"
    assert conditions.CONDITION_SYSTEM_NAMES[-7][-5] == "PlayerDied"
    assert conditions.CONDITION_SYSTEM_NAMES[4][-81] == "AnswerTrue"


def test_condition_extension_names_flat_and_spot_checked():
    # Flat, not nested.
    assert isinstance(conditions.CONDITION_EXTENSION_NAMES, dict)
    for k, v in conditions.CONDITION_EXTENSION_NAMES.items():
        assert isinstance(k, int)
        assert isinstance(v, str)
    # 10+ entries.
    assert conditions.CONDITION_EXTENSION_NAMES[-1] == "AnimationFrame"
    assert conditions.CONDITION_EXTENSION_NAMES[-14] == "OnCollision"
    assert conditions.CONDITION_EXTENSION_NAMES[-13] == "OnBackgroundCollision"
    assert conditions.CONDITION_EXTENSION_NAMES[-16] == "CompareY"
    assert conditions.CONDITION_EXTENSION_NAMES[-17] == "CompareX"
    assert conditions.CONDITION_EXTENSION_NAMES[-24] == "FlagOff"
    assert conditions.CONDITION_EXTENSION_NAMES[-25] == "FlagOn"
    assert conditions.CONDITION_EXTENSION_NAMES[-27] == "CompareAlterableValue"
    assert conditions.CONDITION_EXTENSION_NAMES[-28] == "ObjectInvisible"
    assert conditions.CONDITION_EXTENSION_NAMES[-29] == "ObjectVisible"
    assert conditions.CONDITION_EXTENSION_NAMES[-33] == "AllDestroyed"


# --- Actions -----------------------------------------------------------


def test_action_system_names_shape_is_nested():
    assert isinstance(actions.ACTION_SYSTEM_NAMES, dict)
    for obj_type, inner in actions.ACTION_SYSTEM_NAMES.items():
        assert isinstance(obj_type, int)
        assert isinstance(inner, dict)


def test_action_system_names_spot_check():
    # 10+ entries — built-in + per-type slots.
    assert actions.ACTION_SYSTEM_NAMES[-1][0] == "Skip"
    assert actions.ACTION_SYSTEM_NAMES[-1][3] == "SetGlobalValue"
    assert actions.ACTION_SYSTEM_NAMES[-1][14] == "StartLoop"
    assert actions.ACTION_SYSTEM_NAMES[-1][15] == "StopLoop"
    assert actions.ACTION_SYSTEM_NAMES[-2][0] == "PlaySample"
    assert actions.ACTION_SYSTEM_NAMES[-2][2] == "PlayMusic"
    assert actions.ACTION_SYSTEM_NAMES[-3][0] == "NextFrame"
    assert actions.ACTION_SYSTEM_NAMES[-3][4] == "EndApplication"
    assert actions.ACTION_SYSTEM_NAMES[-7][0] == "SetScore"
    assert actions.ACTION_SYSTEM_NAMES[-7][1] == "SetLives"
    assert actions.ACTION_SYSTEM_NAMES[2][85] == "SetScale"
    assert actions.ACTION_SYSTEM_NAMES[7][80] == "SetCounterValue"


def test_action_extension_names_flat_and_spot_checked():
    assert isinstance(actions.ACTION_EXTENSION_NAMES, dict)
    # 10+ entries.
    assert actions.ACTION_EXTENSION_NAMES[1] == "SetPosition"
    assert actions.ACTION_EXTENSION_NAMES[2] == "SetX"
    assert actions.ACTION_EXTENSION_NAMES[3] == "SetY"
    assert actions.ACTION_EXTENSION_NAMES[4] == "Stop"
    assert actions.ACTION_EXTENSION_NAMES[5] == "Start"
    assert actions.ACTION_EXTENSION_NAMES[24] == "Destroy"
    assert actions.ACTION_EXTENSION_NAMES[26] == "Hide"
    assert actions.ACTION_EXTENSION_NAMES[27] == "Show"
    assert actions.ACTION_EXTENSION_NAMES[31] == "SetAlterableValue"
    assert actions.ACTION_EXTENSION_NAMES[35] == "EnableFlag"
    assert actions.ACTION_EXTENSION_NAMES[36] == "DisableFlag"


# --- Expressions -------------------------------------------------------


def test_expression_system_names_shape_is_nested():
    assert isinstance(expressions.EXPRESSION_SYSTEM_NAMES, dict)
    for obj_type, inner in expressions.EXPRESSION_SYSTEM_NAMES.items():
        assert isinstance(obj_type, int)
        assert isinstance(inner, dict)


def test_expression_system_names_operator_slot():
    """Operator slot (object_type_id == 0) is how the Expression AST
    encodes +/-/*//, AND/OR/XOR. Critical for probe #4.13."""
    op = expressions.EXPRESSION_SYSTEM_NAMES[0]
    assert op[0] == "End"
    assert op[2] == "Plus"
    assert op[4] == "Minus"
    assert op[6] == "Multiply"
    assert op[8] == "Divide"
    assert op[10] == "Modulus"
    assert op[14] == "AND"
    assert op[16] == "OR"
    assert op[18] == "XOR"


def test_expression_system_names_spot_check():
    # 10+ entries across object_types.
    assert expressions.EXPRESSION_SYSTEM_NAMES[-1][0] == "Long"
    assert expressions.EXPRESSION_SYSTEM_NAMES[-1][3] == "String"
    assert expressions.EXPRESSION_SYSTEM_NAMES[-1][23] == "Double"
    assert expressions.EXPRESSION_SYSTEM_NAMES[-1][24] == "GlobalValue"
    assert expressions.EXPRESSION_SYSTEM_NAMES[-1][50] == "GlobalString"
    assert expressions.EXPRESSION_SYSTEM_NAMES[-3][8] == "CurrentFrame"
    assert expressions.EXPRESSION_SYSTEM_NAMES[-6][0] == "XMouse"
    assert expressions.EXPRESSION_SYSTEM_NAMES[-6][1] == "YMouse"
    assert expressions.EXPRESSION_SYSTEM_NAMES[-7][0] == "PlayerScore"
    assert expressions.EXPRESSION_SYSTEM_NAMES[-4][0] == "TimerValue"


def test_expression_extension_names_flat_and_spot_checked():
    assert isinstance(expressions.EXPRESSION_EXTENSION_NAMES, dict)
    # 10+ entries.
    assert expressions.EXPRESSION_EXTENSION_NAMES[1] == "YPosition"
    assert expressions.EXPRESSION_EXTENSION_NAMES[2] == "AnimationFrame"
    assert expressions.EXPRESSION_EXTENSION_NAMES[3] == "Speed"
    assert expressions.EXPRESSION_EXTENSION_NAMES[7] == "ObjectLeft"
    assert expressions.EXPRESSION_EXTENSION_NAMES[8] == "ObjectRight"
    assert expressions.EXPRESSION_EXTENSION_NAMES[9] == "ObjectTop"
    assert expressions.EXPRESSION_EXTENSION_NAMES[10] == "ObjectBottom"
    assert expressions.EXPRESSION_EXTENSION_NAMES[11] == "XPosition"
    assert expressions.EXPRESSION_EXTENSION_NAMES[16] == "AlterableValue"
    assert expressions.EXPRESSION_EXTENSION_NAMES[19] == "AlterableString"
    assert expressions.EXPRESSION_EXTENSION_NAMES[40] == "GetWidth"
    assert expressions.EXPRESSION_EXTENSION_NAMES[41] == "GetHeight"


# --- Parameters --------------------------------------------------------


def test_parameter_names_flat_and_spot_checked():
    assert isinstance(parameters.PARAMETER_NAMES, dict)
    # The 15 codes known to appear in FNAF 1 are the hottest-path
    # lookups for probe #4.13. Pin every one of them so a port
    # regression surfaces on the exact values the parser will hit.
    fnaf1_codes = {1, 2, 6, 9, 10, 14, 16, 22, 23, 25, 26, 27, 32, 45, 50}
    for code in fnaf1_codes:
        assert code in parameters.PARAMETER_NAMES, (
            f"FNAF 1 parameter code {code} missing from PARAMETER_NAMES"
        )
    # 10+ explicit entries (including coverage of the 15 FNAF 1 codes).
    assert parameters.PARAMETER_NAMES[1] == "OBJECT"
    assert parameters.PARAMETER_NAMES[2] == "TIME"
    assert parameters.PARAMETER_NAMES[6] == "SAMPLE"
    assert parameters.PARAMETER_NAMES[9] == "CREATE"
    assert parameters.PARAMETER_NAMES[14] == "KEY"
    assert parameters.PARAMETER_NAMES[16] == "POSITION"
    assert parameters.PARAMETER_NAMES[22] == "EXPRESSION"
    assert parameters.PARAMETER_NAMES[23] == "COMPARISON"
    assert parameters.PARAMETER_NAMES[25] == "BUFFER4"
    assert parameters.PARAMETER_NAMES[26] == "FRAME"
    assert parameters.PARAMETER_NAMES[32] == "Click"
    assert parameters.PARAMETER_NAMES[45] == "EXPSTRING"
    assert parameters.PARAMETER_NAMES[50] == "AlterableValue"


def test_parameter_names_anaconda_gaps_preserved():
    """Anaconda's source has numeric gaps (8, 20, 30, ...). Preserve
    them so a silent backfill doesn't creep in — an unknown ID must
    hit KeyError, not return a made-up name."""
    for missing in (8, 20, 30, 69, 100):
        assert missing not in parameters.PARAMETER_NAMES


# --- Object types ------------------------------------------------------


def test_object_type_names_spot_checked():
    assert isinstance(object_types.OBJECT_TYPE_NAMES, dict)
    # All 17 built-in types; pin representative samples.
    assert object_types.OBJECT_TYPE_NAMES[-7] == "Player"
    assert object_types.OBJECT_TYPE_NAMES[-6] == "Keyboard"
    assert object_types.OBJECT_TYPE_NAMES[-5] == "Create"
    assert object_types.OBJECT_TYPE_NAMES[-4] == "Timer"
    assert object_types.OBJECT_TYPE_NAMES[-3] == "Game"
    assert object_types.OBJECT_TYPE_NAMES[-2] == "Speaker"
    assert object_types.OBJECT_TYPE_NAMES[-1] == "System"
    assert object_types.OBJECT_TYPE_NAMES[0] == "QuickBackdrop"
    assert object_types.OBJECT_TYPE_NAMES[1] == "Backdrop"
    assert object_types.OBJECT_TYPE_NAMES[2] == "Active"
    assert object_types.OBJECT_TYPE_NAMES[3] == "Text"
    assert object_types.OBJECT_TYPE_NAMES[7] == "Counter"
    assert object_types.OBJECT_TYPE_NAMES[9] == "SubApplication"


def test_object_type_extension_base_is_32():
    """Any object_type_id >= EXTENSION_BASE is a user extension, per
    Anaconda's getObjectType() contract. Pin the magic number so a
    rename that silently drifts it (e.g. to 33) fails this test."""
    assert object_types.EXTENSION_BASE == 32
    # No built-in ID can be >= EXTENSION_BASE — the negative IDs and
    # 0..9 slots all sit well below it.
    for obj_id in object_types.OBJECT_TYPE_NAMES:
        assert obj_id < object_types.EXTENSION_BASE
