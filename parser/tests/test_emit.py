"""Tests for `fnaf_parser.algorithm.emit` — the Output Emission pipeline.

Two layers of coverage:

1. **Pure helper unit tests** — no FNAF binary required. Exercise the
   deterministic pieces (slugging, row flattening, object-name injection,
   expr_str renderer) against hand-built fixtures so failure modes are
   isolated from the pack-walking layer.

2. **CLI smoke** (skipif-gated) — when the real FNAF 1 binary is on disk,
   run the full `dump-algorithm` pipeline end-to-end and assert the
   Algorithm Snapshot Antibody's totals: 17 frames, 584 event groups,
   2532 cond+act pairs. The per-frame + combined + manifest artefacts are
   all written, non-empty, and manifest SHA-256 values reconcile with the
   actual on-disk files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fnaf_parser.algorithm.emit import (
    AlgorithmEmitError,
    DECODER_VERSION,
    _cond_or_act_row,
    _flatten_frame_to_rows,
    _frame_filename,
    _frame_slug,
    _inject_cond_or_act_names,
    _inject_expression_object_names,
    _resolve_handle,
    dump_algorithm,
)
from fnaf_parser.algorithm.expr_str import render_expression_stream
from fnaf_parser.cli import main

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


# --- _frame_slug / _frame_filename --------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Menu", "menu"),
        ("Office Night 1", "office_night_1"),
        ("  Weird-Name!!! ", "weird_name"),
        ("", "unnamed"),
        (None, "unnamed"),
        ("///___", "unnamed"),
    ],
)
def test_frame_slug(name, expected):
    assert _frame_slug(name) == expected


def test_frame_filename_formats_index_and_slug():
    assert _frame_filename(0, "Menu") == "frame_00_menu.json"
    assert _frame_filename(17, None) == "frame_17_unnamed.json"
    assert _frame_filename(9, "Office Night 1") == "frame_09_office_night_1.json"


# --- _resolve_handle ----------------------------------------------------


def test_resolve_handle_sentinel_returns_none():
    assert _resolve_handle(0xFFFF, handles_to_names={}, where="w") is None


def test_resolve_handle_known_returns_name():
    h2n = {5: "Freddy", 7: "Bonnie"}
    assert _resolve_handle(5, handles_to_names=h2n, where="w") == "Freddy"
    assert _resolve_handle(7, handles_to_names=h2n, where="w") == "Bonnie"


def test_resolve_handle_unknown_raises_with_path():
    with pytest.raises(AlgorithmEmitError) as exc_info:
        _resolve_handle(999, handles_to_names={5: "Freddy"}, where="frame[3]/cond[0]")
    msg = str(exc_info.value)
    assert "999" in msg
    assert "frame[3]/cond[0]" in msg


# --- render_expression_stream ------------------------------------------


def _END():
    return {"object_type": 0, "num": 0, "num_name": "End", "body": None}


def _long(value):
    return {
        "object_type": -1,
        "num": 0,
        "num_name": "Long",
        "body": {"body_kind": "Long", "value": value},
    }


def _global_value(index):
    return {
        "object_type": -1,
        "num": 24,
        "num_name": "GlobalValue",
        "body": {"body_kind": "GlobalValue", "value": index},
    }


def _minus():
    return {"object_type": 0, "num": 4, "num_name": "Minus", "body": None}


def test_render_expression_stream_binary_minus_produces_infix():
    # RPN: G_1 1 -    →  "(G_1 - 1)"
    stream = [_global_value(1), _long(1), _minus(), _END()]
    assert render_expression_stream(stream) == "(G_1 - 1)"


def test_render_expression_stream_stack_underflow_is_total():
    # Operator with empty stack — renderer must not raise.
    stream = [_minus(), _END()]
    out = render_expression_stream(stream)
    # Placeholder marker appears instead of crashing.
    assert "<?-?>" in out or "?" in out


def test_render_expression_stream_object_reference_uses_object_name():
    # Active.XPosition  (object_type >= 2 branch, object_name stamped)
    stream = [
        {
            "object_type": 2,
            "num": 11,
            "num_name": "XPosition",
            "object_type_name": "Active",
            "object_info": 5,
            "object_name": "Freddy",
            "body": None,
        },
        _END(),
    ]
    assert render_expression_stream(stream) == "Freddy.XPosition"


def test_render_expression_stream_object_reference_without_name_falls_back_to_handle():
    stream = [
        {
            "object_type": 2,
            "num": 11,
            "num_name": "XPosition",
            "object_type_name": "Active",
            "object_info": 5,
            "body": None,
        },
        _END(),
    ]
    assert render_expression_stream(stream) == "obj[5].XPosition"


def test_render_expression_stream_empty_stream_returns_empty_string():
    assert render_expression_stream([]) == ""
    assert render_expression_stream([_END()]) == ""


# --- _inject_expression_object_names -----------------------------------


def test_inject_expression_object_names_on_object_refs_only():
    h2n = {5: "Freddy"}
    stream = [
        # object ref → gets named
        {
            "object_type": 2,
            "num": 11,
            "num_name": "XPosition",
            "object_info": 5,
            "body": None,
        },
        # system slot (-1, Long) → no object_info → untouched
        _long(42),
        _END(),
    ]
    out = _inject_expression_object_names(
        stream, handles_to_names=h2n, where="w"
    )
    assert out[0]["object_name"] == "Freddy"
    assert "object_name" not in out[1]
    assert "object_name" not in out[2]


def test_inject_expression_object_names_unknown_handle_raises():
    stream = [
        {
            "object_type": 2,
            "num": 11,
            "num_name": "XPosition",
            "object_info": 999,
            "body": None,
        },
    ]
    with pytest.raises(AlgorithmEmitError):
        _inject_expression_object_names(
            stream, handles_to_names={5: "Freddy"}, where="w"
        )


# --- _inject_cond_or_act_names -----------------------------------------


def test_inject_cond_or_act_names_stamps_object_name_and_walks_params():
    h2n = {5: "Freddy", 7: "Bonnie"}
    cond = {
        "object_type": 2,
        "object_type_name": "Active",
        "num": 1,
        "num_name": "Collision",
        "object_info": 5,
        "object_info_list": -1,
        "flags": 0,
        "other_flags": 0,
        "def_type": 0,
        "identifier": 0,
        "parameters": [
            {
                "kind": "ExpressionParameter",
                "code": 22,
                "comparison": 0,
                "comparison_name": "=",
                "expressions": [
                    {
                        "object_type": 2,
                        "num": 11,
                        "num_name": "XPosition",
                        "object_info": 7,
                        "body": None,
                    },
                    _long(100),
                    _minus(),
                    _END(),
                ],
            }
        ],
    }
    out = _inject_cond_or_act_names(
        cond, handles_to_names=h2n, where="frame[0]/grp[0]/cond[0]"
    )
    assert out["object_name"] == "Freddy"
    # Sub-expression `object_info=7` got its object_name stamped.
    assert out["parameters"][0]["expressions"][0]["object_name"] == "Bonnie"
    # expr_str was rendered deterministically.
    assert out["parameters"][0]["expr_str"] == "(Bonnie.XPosition - 100)"


def test_inject_cond_or_act_names_sentinel_handle_resolves_to_none():
    cond = {
        "object_type": -1,
        "object_type_name": "System",
        "num": 0,
        "num_name": "Never",
        "object_info": 0xFFFF,
        "object_info_list": -1,
        "flags": 0,
        "other_flags": 0,
        "def_type": 0,
        "identifier": 0,
        "parameters": [],
    }
    out = _inject_cond_or_act_names(
        cond, handles_to_names={}, where="w"
    )
    assert out["object_name"] is None


# --- _cond_or_act_row / _flatten_frame_to_rows --------------------------


def _minimal_resolved_cond(**overrides):
    base = {
        "object_type": 2,
        "object_type_name": "Active",
        "num": 1,
        "num_name": "Collision",
        "object_info": 5,
        "object_name": "Freddy",
        "object_info_list": -1,
        "flags": 0,
        "other_flags": 0,
        "def_type": 0,
        "identifier": 0,
        "parameters": [],
    }
    base.update(overrides)
    return base


def test_cond_or_act_row_fields_and_order():
    row = _cond_or_act_row(
        _minimal_resolved_cond(),
        row_type="condition",
        frame_index=3,
        frame_name="Night 1",
        event_group_index=12,
        cond_or_act_index=0,
    )
    # Source-citation fields arrive first (per SiftText crystallization).
    keys = list(row.keys())
    assert keys[:5] == [
        "frame",
        "frame_name",
        "event_group_index",
        "type",
        "cond_or_act_index",
    ]
    assert row["frame"] == 3
    assert row["frame_name"] == "Night 1"
    assert row["event_group_index"] == 12
    assert row["type"] == "condition"
    assert row["cond_or_act_index"] == 0
    assert row["object_type_name"] == "Active"
    assert row["num_name"] == "Collision"
    assert row["object_name"] == "Freddy"


def test_cond_or_act_row_action_has_null_identifier():
    # Actions never have `identifier`; row stamps None.
    act = _minimal_resolved_cond()
    del act["identifier"]
    row = _cond_or_act_row(
        act,
        row_type="action",
        frame_index=0,
        frame_name="Menu",
        event_group_index=0,
        cond_or_act_index=0,
    )
    assert row["type"] == "action"
    assert row["identifier"] is None


def test_flatten_frame_to_rows_emits_cond_then_act_per_group():
    fe = {
        "event_groups": [
            {
                "conditions": [
                    _minimal_resolved_cond(num_name="C1"),
                    _minimal_resolved_cond(num_name="C2"),
                ],
                "actions": [_minimal_resolved_cond(num_name="A1")],
            },
            {
                "conditions": [_minimal_resolved_cond(num_name="C3")],
                "actions": [],
            },
        ],
    }
    rows = _flatten_frame_to_rows(fe, frame_index=7, frame_name="X")
    # Group 0: 2 conditions + 1 action, Group 1: 1 condition → 4 rows total.
    assert len(rows) == 4
    assert [r["type"] for r in rows] == [
        "condition",
        "condition",
        "action",
        "condition",
    ]
    assert [r["num_name"] for r in rows] == ["C1", "C2", "A1", "C3"]
    assert [r["event_group_index"] for r in rows] == [0, 0, 0, 1]
    assert [r["cond_or_act_index"] for r in rows] == [0, 1, 0, 0]
    assert all(r["frame"] == 7 for r in rows)
    assert all(r["frame_name"] == "X" for r in rows)


# --- End-to-end smoke (skipif-gated) ------------------------------------

# Pinned Algorithm Snapshot Antibody totals (from SiftText node
# `598d7b05-56c9-4324-a742-4f1e7a9ec496`). Any future decoder change that
# walks a different event count must update these intentionally — silent
# drift is the whole thing this suite exists to catch.
FNAF1_FRAME_COUNT = 17
FNAF1_EVENT_GROUPS = 584
FNAF1_COND_ACT_PAIRS = 2532


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_dump_algorithm_writes_all_four_artefact_types(tmp_path):
    written = dump_algorithm(FNAF_EXE, tmp_path)
    alg_dir = tmp_path / "algorithm"
    frames_dir = alg_dir / "frames"

    # 17 per-frame files + combined.json + combined.jsonl + manifest.json.
    assert frames_dir.is_dir()
    frame_files = sorted(frames_dir.glob("frame_*.json"))
    assert len(frame_files) == FNAF1_FRAME_COUNT

    assert (alg_dir / "combined.json").is_file()
    assert (alg_dir / "combined.jsonl").is_file()
    assert (alg_dir / "manifest.json").is_file()

    # Written-list order reflects emission order: frames first, then
    # combined.json, combined.jsonl, manifest.json.
    assert written[-1] == alg_dir / "manifest.json"
    assert written[-2] == alg_dir / "combined.jsonl"
    assert written[-3] == alg_dir / "combined.json"


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_dump_algorithm_manifest_totals_match_antibody(tmp_path):
    dump_algorithm(FNAF_EXE, tmp_path)
    manifest = json.loads(
        (tmp_path / "algorithm" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["decoder_version"] == DECODER_VERSION
    assert manifest["frame_count"] == FNAF1_FRAME_COUNT
    assert manifest["total_event_groups"] == FNAF1_EVENT_GROUPS
    assert manifest["total_cond_act_pairs"] == FNAF1_COND_ACT_PAIRS
    # source_sha256 is a hex digest of the real binary.
    assert len(manifest["source_sha256"]) == 64
    assert manifest["source_sha256"] == hashlib.sha256(
        FNAF_EXE.read_bytes()
    ).hexdigest()


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_dump_algorithm_manifest_sha256_matches_files(tmp_path):
    dump_algorithm(FNAF_EXE, tmp_path)
    alg_dir = tmp_path / "algorithm"
    manifest = json.loads((alg_dir / "manifest.json").read_text())
    for rel, meta in manifest["files"].items():
        path = alg_dir / rel
        assert path.is_file()
        assert path.stat().st_size == meta["size_bytes"]
        h = hashlib.sha256()
        h.update(path.read_bytes())
        assert h.hexdigest() == meta["sha256"]


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_dump_algorithm_combined_jsonl_row_count_matches_totals(tmp_path):
    dump_algorithm(FNAF_EXE, tmp_path)
    jsonl = (tmp_path / "algorithm" / "combined.jsonl").read_text(
        encoding="utf-8"
    )
    lines = [line for line in jsonl.splitlines() if line.strip()]
    assert len(lines) == FNAF1_COND_ACT_PAIRS

    types = [json.loads(line)["type"] for line in lines]
    # Antibody totals don't split cond vs act, but every row must be one
    # of the two known values — any third value is drift.
    assert set(types) <= {"condition", "action"}


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_dump_algorithm_combined_json_frame_count_matches_files(tmp_path):
    dump_algorithm(FNAF_EXE, tmp_path)
    alg_dir = tmp_path / "algorithm"
    combined = json.loads((alg_dir / "combined.json").read_text())
    frame_files = sorted((alg_dir / "frames").glob("frame_*.json"))
    assert len(combined["frames"]) == len(frame_files) == FNAF1_FRAME_COUNT
    # Each frame in combined.json has the same event_groups list length as
    # its per-frame artefact.
    for frame in combined["frames"]:
        per = json.loads(
            (
                alg_dir
                / "frames"
                / _frame_filename(frame["frame_index"], frame["frame_name"])
            ).read_text()
        )
        assert len(frame["event_groups"]) == len(per["event_groups"])


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_dump_algorithm_every_row_has_source_citation_fields(tmp_path):
    dump_algorithm(FNAF_EXE, tmp_path)
    jsonl = (tmp_path / "algorithm" / "combined.jsonl").read_text(
        encoding="utf-8"
    )
    required = {
        "frame",
        "frame_name",
        "event_group_index",
        "type",
        "cond_or_act_index",
        "object_type_name",
        "num_name",
    }
    for line in jsonl.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        missing = required - row.keys()
        assert not missing, f"row missing fields {missing}: {row}"


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_cli_dump_algorithm_exits_zero_end_to_end(tmp_path):
    exit_code = main(
        ["dump-algorithm", str(FNAF_EXE), "--out", str(tmp_path)]
    )
    assert exit_code == 0
    assert (tmp_path / "algorithm" / "manifest.json").is_file()
