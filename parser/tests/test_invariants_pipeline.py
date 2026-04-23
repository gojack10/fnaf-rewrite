"""Tests for `fnaf_parser.invariants.pipeline` — head-chef orchestrator.

We inject:
- a fake `OpenRouterConfig` so `load_config` isn't called
- a stub `predictor_factory` so `dspy.Predict(ExtractInvariants)` is
  never constructed

No API key required. The stub returns canned InvariantRecords so we
can verify the pipeline's dispatch + JSONL writing + citation checker
hand-off all work end-to-end, independent of the LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fnaf_parser.invariants.config import OpenRouterConfig
from fnaf_parser.invariants.pipeline import RunSummary, run
from fnaf_parser.invariants.signatures import Citation, InvariantRecord


# --- Fixture builders ---------------------------------------------------


def _jsonl(tmp_path: Path) -> Path:
    """One Slice-C-eligible row (code 22 EXPRESSION param)."""
    rows = [
        {
            "frame": 0,
            "frame_name": "Frame 0",
            "event_group_index": 0,
            "type": "condition",
            "cond_or_act_index": 0,
            "num_name": "CompareCounter",
            "object_type_name": "Counter",
            "object_name": "Night",
            "parameters": [
                {"code": 22, "code_name": "EXP", "expr_str": "Night + 1"}
            ],
        }
    ]
    path = tmp_path / "combined.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def _json(tmp_path: Path) -> Path:
    """Mirror the jsonl row in combined.json shape — same coordinates."""
    doc = {
        "decoder_version": "test",
        "frames": [
            {
                "frame_index": 0,
                "frame_name": "Frame 0",
                "event_groups": [
                    {
                        "conditions": [
                            {
                                "num_name": "CompareCounter",
                                "parameters": [
                                    {
                                        "code": 22,
                                        "code_name": "EXP",
                                        "expr_str": "Night + 1",
                                    }
                                ],
                            }
                        ],
                        "actions": [],
                    }
                ],
            }
        ],
    }
    path = tmp_path / "combined.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


class _FakePrediction:
    """Minimal stand-in for `dspy.Prediction` — the pipeline only reads
    `.records` off it."""

    def __init__(self, records: list[InvariantRecord]):
        self.records = records


class _CannedPredictor:
    """Replaces `dspy.Predict(ExtractInvariants)`. Every dispatch
    returns the same canned record list, so the test is deterministic."""

    def __init__(self, records: list[InvariantRecord]):
        self._records = records
        self.calls: list[list[dict[str, Any]]] = []

    def __call__(self, *, tickets: list[dict[str, Any]]) -> _FakePrediction:
        self.calls.append(tickets)
        return _FakePrediction(self._records)


def _fake_config() -> OpenRouterConfig:
    return OpenRouterConfig(
        api_key="sk-or-v1-test", line_cook_model="m", head_chef_model="m"
    )


# --- Tests --------------------------------------------------------------


def test_pipeline_writes_all_three_artefacts(tmp_path: Path):
    """A successful dispatch produces raw_records / accepted /
    quarantine JSONL in the specified out_dir."""
    canned = [
        InvariantRecord(
            claim="Night counter advances by one.",
            citations=[
                Citation(
                    frame=0,
                    event_group_index=0,
                    type="condition",
                    cond_or_act_index=0,
                )
            ],
            pseudo_code="Night + 1",
            kind="numeric_assignment",
        ),
    ]
    predictor = _CannedPredictor(canned)
    out_dir = tmp_path / "out"

    # Patch `dspy` import + `build_lm` by injecting config AND a
    # predictor_factory. We still need dspy installed for the
    # `dspy.configure` + `build_lm` calls — that's fine, dspy is a
    # runtime dep. If the test environment lacks dspy it skips
    # implicitly via ImportError.
    pytest.importorskip("dspy")

    summary = run(
        slice_name="c",
        combined_jsonl=_jsonl(tmp_path),
        combined_json=_json(tmp_path),
        out_dir=out_dir,
        config=_fake_config(),
        predictor_factory=lambda: predictor,
    )
    assert isinstance(summary, RunSummary)
    assert summary.tickets_dispatched == 1
    assert summary.records_emitted == 1
    assert summary.accepted == 1
    assert summary.quarantined == 0
    assert summary.raw_records_path.exists()
    assert summary.accepted_path.exists()
    assert summary.quarantine_path.exists()


def test_pipeline_quarantines_bad_coordinate_records(tmp_path: Path):
    """If the LLM emits a record with out-of-range coords, the
    Citation Checker must reject it — `accepted == 0, quarantined ==
    1`. This is the hallucination-gate contract end-to-end."""
    pytest.importorskip("dspy")
    canned = [
        InvariantRecord(
            claim="Bad coords hallucination.",
            citations=[
                Citation(
                    frame=99,
                    event_group_index=99,
                    type="condition",
                    cond_or_act_index=99,
                )
            ],
            pseudo_code="garbage",
            kind="numeric_assignment",
        ),
    ]
    predictor = _CannedPredictor(canned)
    summary = run(
        slice_name="c",
        combined_jsonl=_jsonl(tmp_path),
        combined_json=_json(tmp_path),
        out_dir=tmp_path / "out",
        config=_fake_config(),
        predictor_factory=lambda: predictor,
    )
    assert summary.accepted == 0
    assert summary.quarantined == 1


def test_pipeline_accepts_dict_records_from_stub(tmp_path: Path):
    """Defensive path: if a stub predictor returns dicts instead of
    Pydantic instances, the pipeline validates them on the way through
    so downstream never sees an un-validated record."""
    pytest.importorskip("dspy")

    class _DictPredictor:
        def __call__(self, *, tickets):
            return _FakePrediction(
                [
                    {
                        "claim": "Night plus one.",
                        "citations": [
                            {
                                "frame": 0,
                                "event_group_index": 0,
                                "type": "condition",
                                "cond_or_act_index": 0,
                            }
                        ],
                        "pseudo_code": "Night + 1",
                        "kind": "numeric_assignment",
                    }
                ]
            )

    summary = run(
        slice_name="c",
        combined_jsonl=_jsonl(tmp_path),
        combined_json=_json(tmp_path),
        out_dir=tmp_path / "out",
        config=_fake_config(),
        predictor_factory=lambda: _DictPredictor(),
    )
    assert summary.records_emitted == 1
    assert summary.accepted == 1


def test_run_summary_log_dict_stringifies_paths(tmp_path: Path):
    """`to_log_dict` is what the CLI `console.print_json` writes —
    every path must be stringified so `json.dumps` doesn't choke on
    `Path` objects."""
    summary = RunSummary(
        slice_name="c",
        tickets_dispatched=1,
        records_emitted=1,
        accepted=1,
        quarantined=0,
        raw_records_path=tmp_path / "raw.jsonl",
        accepted_path=tmp_path / "acc.jsonl",
        quarantine_path=tmp_path / "q.jsonl",
    )
    blob = json.dumps(summary.to_log_dict())  # must not raise
    assert "acc.jsonl" in blob
    assert "q.jsonl" in blob
