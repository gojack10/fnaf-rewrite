"""Tests for `fnaf_parser.invariants.pipeline` — head-chef orchestrator.

We inject:
- a fake `OpenRouterConfig` so `load_config` isn't called
- a stub `line_cook_factory` so `dspy.Predict(ExtractInvariants)` is
  never constructed
- a stub `head_chef_factory` so `dspy.Predict(ReviewInvariants)` is
  never constructed

No API key required. Stubs return canned InvariantRecords so we can
verify the two-phase dispatch + JSONL writing + citation checker
hand-off all work end-to-end, independent of the LLM. The default
head-chef stub (`_PassThroughChef`) preserves the line-cook candidates
unchanged so tests that only care about line-cook wiring don't need to
reason about filter semantics.
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


class _PassThroughChef:
    """Replaces `dspy.Predict(ReviewInvariants)`. Returns the
    line-cook candidates unchanged so tests that only care about
    line-cook wiring don't need to reason about filter semantics.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        tickets: list[dict[str, Any]],
        candidates: list[InvariantRecord],
    ) -> _FakePrediction:
        self.calls.append({"tickets": tickets, "candidates": list(candidates)})
        return _FakePrediction(list(candidates))


def _fake_config() -> OpenRouterConfig:
    # Two distinct slugs so a misrouted build_*_lm call would be
    # obvious in any future test that inspects the LM; the current
    # tests don't exercise the LM build path (stubs replace Predict),
    # but keeping the slugs distinct avoids encoding an assumption of
    # identity that could bite later.
    return OpenRouterConfig(
        api_key="sk-or-v1-test",
        line_cook_model="openrouter/fake/line-cook",
        head_chef_model="openrouter/fake/head-chef",
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
        line_cook_factory=lambda: predictor,
        head_chef_factory=lambda: _PassThroughChef(),
    )
    assert isinstance(summary, RunSummary)
    assert summary.tickets_dispatched == 1
    assert summary.candidates_emitted == 1
    assert summary.records_emitted == 1
    assert summary.accepted == 1
    assert summary.quarantined == 0
    assert summary.candidates_path.exists()
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
        line_cook_factory=lambda: predictor,
        head_chef_factory=lambda: _PassThroughChef(),
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
        line_cook_factory=lambda: _DictPredictor(),
        head_chef_factory=lambda: _PassThroughChef(),
    )
    assert summary.candidates_emitted == 1
    assert summary.records_emitted == 1
    assert summary.accepted == 1


def test_run_summary_log_dict_stringifies_paths(tmp_path: Path):
    """`to_log_dict` is what the CLI `console.print_json` writes —
    every path must be stringified so `json.dumps` doesn't choke on
    `Path` objects."""
    summary = RunSummary(
        slice_name="c",
        tickets_dispatched=1,
        candidates_emitted=2,
        records_emitted=1,
        accepted=1,
        quarantined=0,
        candidates_path=tmp_path / "cand.jsonl",
        raw_records_path=tmp_path / "raw.jsonl",
        accepted_path=tmp_path / "acc.jsonl",
        quarantine_path=tmp_path / "q.jsonl",
    )
    blob = json.dumps(summary.to_log_dict())  # must not raise
    assert "cand.jsonl" in blob
    assert "acc.jsonl" in blob
    assert "q.jsonl" in blob


def test_head_chef_filters_line_cook_candidates(tmp_path: Path):
    """End-to-end: line cook emits 2 candidates, head chef keeps 1.

    Verifies the two-phase wiring: candidates_emitted reflects the
    line-cook output, records_emitted reflects the head-chef-approved
    subset, and each artefact file carries the right count.
    """
    pytest.importorskip("dspy")

    good = InvariantRecord(
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
    )
    noise = InvariantRecord(
        claim="Weakly grounded paraphrase that head chef should drop.",
        citations=[
            Citation(
                frame=0,
                event_group_index=0,
                type="condition",
                cond_or_act_index=0,
            )
        ],
        pseudo_code="noise",
        kind="other",
    )

    class _KeepFirstChef:
        """Drop everything except the first candidate — stand-in for
        the head chef's real filter behaviour."""

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def __call__(
            self,
            *,
            tickets: list[dict[str, Any]],
            candidates: list[InvariantRecord],
        ) -> _FakePrediction:
            self.calls.append(
                {"tickets": tickets, "candidates": list(candidates)}
            )
            return _FakePrediction(list(candidates)[:1])

    out_dir = tmp_path / "out"
    chef = _KeepFirstChef()
    summary = run(
        slice_name="c",
        combined_jsonl=_jsonl(tmp_path),
        combined_json=_json(tmp_path),
        out_dir=out_dir,
        config=_fake_config(),
        line_cook_factory=lambda: _CannedPredictor([good, noise]),
        head_chef_factory=lambda: chef,
    )

    # Counts reflect the filter.
    assert summary.candidates_emitted == 2
    assert summary.records_emitted == 1
    assert summary.accepted == 1
    assert summary.quarantined == 0

    # Artefact files carry the right pre- and post-review counts.
    cand_lines = summary.candidates_path.read_text(
        encoding="utf-8"
    ).splitlines()
    raw_lines = summary.raw_records_path.read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(cand_lines) == 2
    assert len(raw_lines) == 1

    # Head chef actually saw both candidates (1 call for the single
    # Slice C ticket).
    assert len(chef.calls) == 1
    assert len(chef.calls[0]["candidates"]) == 2


def test_head_chef_skipped_when_line_cook_emits_nothing(tmp_path: Path):
    """If the line cook returns no candidates, the head chef must not
    be called — there's nothing to review and we'd waste a token round-
    trip. `candidates_emitted == 0`, `records_emitted == 0`.
    """
    pytest.importorskip("dspy")

    class _TrackingChef(_PassThroughChef):
        pass

    chef = _TrackingChef()
    summary = run(
        slice_name="c",
        combined_jsonl=_jsonl(tmp_path),
        combined_json=_json(tmp_path),
        out_dir=tmp_path / "out",
        config=_fake_config(),
        line_cook_factory=lambda: _CannedPredictor([]),
        head_chef_factory=lambda: chef,
    )

    assert summary.candidates_emitted == 0
    assert summary.records_emitted == 0
    assert chef.calls == []  # never invoked
