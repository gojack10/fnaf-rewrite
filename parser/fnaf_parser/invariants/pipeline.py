"""Head-chef orchestrator — DSPy.RLM dispatch loop.

The only module in the invariant-extraction package that actually calls
an LLM. Its shape:

    pipeline.run(slice="c",
                 combined_jsonl=..., combined_json=...,
                 out_dir=..., config=optional)
      -> RunSummary

Sequence
--------

1. Resolve `OpenRouterConfig` (from env or injected). A missing
   `OPENROUTER_API_KEY` surfaces as `OpenRouterConfigError` — caught by
   the CLI layer and mapped to exit code 3 with a one-shot remediation
   message. DSPy import happens only after the config resolves, so
   running `--help` or probing imports never fails on missing deps.

2. Load the requested slice via `slices.load_slice_{a,b,c}` —
   DuckDB-backed, deterministic, no LLM.

3. For every ticket, run a TWO-PHASE pass — see below.

4. Collect every head-chef-approved `InvariantRecord` into a flat list,
   stream-write it to `<out>/raw_records.jsonl`. Line-cook candidates
   (pre-review) are streamed separately to `<out>/candidates.jsonl` so
   post-run triage can compare before-vs-after without re-running.

5. Hand off to `citation_checker.check_records`, which writes
   `<out>/accepted.jsonl` and `<out>/quarantine.jsonl`.

6. Return a `RunSummary` with counts and paths — the CLI prints it.

Two-phase per-ticket flow
-------------------------

Each ticket goes through exactly TWO LLM calls, in this order:

a. **Line cook** — `dspy.Predict(ExtractInvariants)` over the
   line-cook LM. Emits candidate invariants grounded in the ticket.

b. **Head chef** — `dspy.Predict(ReviewInvariants)` over the
   head-chef LM. Sees the same ticket plus the line cook's
   candidates; returns a (possibly empty) filtered subset.

The head chef is capped at **1 depth**. No recursive refinement, no
"ask again if unhappy" loop. Letting it bounce rejections back at the
line cook risks an infinite loop on a stubborn ticket, and buys little
over the deterministic Citation Checker that runs next. The 3-layer
hallucination gate is:

    Pydantic schema  →  Head chef review  →  Citation Checker
    (re-ask on bad JSON) (one semantic pass) (deterministic cite verify)

If no candidates come back from the line cook, the head chef is
skipped for that ticket — nothing to review. `dspy.context(lm=...)`
scopes each phase's LM so they don't thrash the global config.

Why lazy imports
----------------

`import dspy` is deferred to `run()` (never at module import). That
keeps `from fnaf_parser.invariants.pipeline import RunSummary`
importable in environments where DSPy isn't installed (e.g. a Rust-side
CI job validating only the citation checker). Tests for this module
either (a) don't enter `run()` at all, or (b) patch the factories to
return stubs before calling `run()`.

Why no async
------------

Slice C is ~800 rows; Slices A + B combined are ~400 buckets. Each
ticket now costs TWO calls, so wall-clock roughly doubles — still
under 20 min sequential at OpenRouter's default rate limit. Async
dispatch would save wall-clock but complicates error handling, retry
semantics, and JSONL write ordering. Keep the pilot simple; reconsider
if/when a later probe stretches the ticket count by an order of
magnitude.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from fnaf_parser.invariants.citation_checker import check_records
from fnaf_parser.invariants.config import (
    OpenRouterConfig,
    build_head_chef_lm,
    build_line_cook_lm,
    load_config,
)
from fnaf_parser.invariants.signatures import InvariantRecord
from fnaf_parser.invariants.slices import (
    load_slice_a,
    load_slice_b,
    load_slice_c,
)

SliceName = Literal["a", "b", "c"]


# --- Summary type -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Outcome of a single pipeline run.

    Paths are always absolute so the CLI can print them verbatim
    without worrying about the caller's cwd. Counts are separated
    (dispatched vs candidates vs emitted vs accepted vs quarantined)
    so post-run triage can compare yield rates across slices — and
    compare the head chef's filter rate (candidates_emitted vs
    records_emitted) without re-reading the JSONL artefacts.
    """

    slice_name: SliceName
    tickets_dispatched: int
    candidates_emitted: int
    records_emitted: int
    accepted: int
    quarantined: int
    candidates_path: Path
    raw_records_path: Path
    accepted_path: Path
    quarantine_path: Path

    def to_log_dict(self) -> dict[str, Any]:
        """Shape suitable for dumping via `json.dumps(indent=2)` in the
        CLI. All paths stringified so the log line stays
        cross-platform."""
        return {
            "slice": self.slice_name,
            "tickets_dispatched": self.tickets_dispatched,
            "candidates_emitted": self.candidates_emitted,
            "records_emitted": self.records_emitted,
            "accepted": self.accepted,
            "quarantined": self.quarantined,
            "artefacts": {
                "candidates": str(self.candidates_path),
                "raw_records": str(self.raw_records_path),
                "accepted": str(self.accepted_path),
                "quarantine": str(self.quarantine_path),
            },
        }


# --- Slice dispatch table -----------------------------------------------


_SLICE_LOADERS: dict[SliceName, Callable[[Path], list[dict[str, Any]]]] = {
    "a": load_slice_a,  # type: ignore[dict-item]
    "b": load_slice_b,  # type: ignore[dict-item]
    "c": load_slice_c,  # type: ignore[dict-item]
}


def _ticket_payload(slice_name: SliceName, ticket: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise a slice ticket into the flat `list[dict]` the
    `ExtractInvariants` Signature expects as input.

    Slice A / B tickets carry `rows`, each of which is already a row-
    shaped dict — pass them straight through.

    Slice C tickets are row-shaped themselves (one row per ticket);
    wrap in a singleton list so the Signature input shape is uniform.
    """
    if slice_name in ("a", "b"):
        rows = ticket.get("rows", [])
        return [dict(r) for r in rows]
    return [dict(ticket)]


# --- Predictor factories ------------------------------------------------


PredictorFactory = Callable[[], Callable[..., Any]]


def _default_line_cook_factory() -> Callable[..., Any]:
    """Build the default DSPy `Predict(ExtractInvariants)` callable.

    Isolated behind a factory so tests can inject a stub without
    importing DSPy. The factory is invoked at most once per `run()`
    call, after the config has resolved, so import failures here
    surface only when actually running the pipeline.
    """
    import dspy  # noqa: PLC0415  (lazy — keeps dspy off import-time path)

    # Lazy import too: `extract_signature` imports `dspy` at module
    # scope (the `Signature` metaclass requires it), so pulling it in
    # here keeps both off the parser's import-time path.
    from fnaf_parser.invariants.extract_signature import (  # noqa: PLC0415
        ExtractInvariants,
    )

    return dspy.Predict(ExtractInvariants)


def _default_head_chef_factory() -> Callable[..., Any]:
    """Build the default DSPy `Predict(ReviewInvariants)` callable.

    Mirrors `_default_line_cook_factory`: lazy-imports `dspy` and the
    review signature so the module stays importable without DSPy
    installed. Invoked at most once per `run()` call.
    """
    import dspy  # noqa: PLC0415

    from fnaf_parser.invariants.review_signature import (  # noqa: PLC0415
        ReviewInvariants,
    )

    return dspy.Predict(ReviewInvariants)


# Backward-compat alias — pre-split tests / imports referenced
# `_default_predictor_factory`. Preserve the name so external callers
# don't break; the semantics (line cook, extract phase) are unchanged.
_default_predictor_factory = _default_line_cook_factory


# --- Main entry point ---------------------------------------------------


def run(
    slice_name: SliceName,
    combined_jsonl: Path,
    combined_json: Path,
    out_dir: Path,
    config: OpenRouterConfig | None = None,
    line_cook_factory: PredictorFactory = _default_line_cook_factory,
    head_chef_factory: PredictorFactory = _default_head_chef_factory,
) -> RunSummary:
    """Run the two-phase dispatch loop for one slice.

    Parameters
    ----------
    slice_name
        Which Scout Pass slice to run: `"a"` (by op), `"b"` (by object
        bucket), or `"c"` (EXPRESSION params — pilot target).
    combined_jsonl
        Path to `combined.jsonl` (row-per-line JSONL). Feeds the DuckDB
        slice query.
    combined_json
        Path to `combined.json` (the structured pack). Feeds the
        Citation Checker — which indexes by
        `(frame, event_group_index, type, cond_or_act_index)` — and
        must be the same pack the jsonl was generated from.
    out_dir
        Directory for the four artefacts: `candidates.jsonl` (line-cook
        output, pre-review), `raw_records.jsonl` (head-chef-approved,
        pre-Citation-Checker), `accepted.jsonl`, `quarantine.jsonl`.
        Created if missing.
    config
        Optional pre-resolved config. If None, loaded from env.
        `OpenRouterConfigError` propagates on missing key.
    line_cook_factory
        Optional override for the line-cook predictor — tests pass a
        stub here to exercise the orchestration without network I/O.
    head_chef_factory
        Optional override for the head-chef predictor. Defaulted like
        the line cook; override independently in tests that want to
        verify filter behaviour end-to-end.

    Returns
    -------
    RunSummary
        Counts + artefact paths. CLI serialises this as JSON.
    """
    cfg = config if config is not None else load_config()

    # Only now that the config has resolved do we import dspy. Keeps
    # the CLI snappy (and importable) when the key is missing.
    import dspy  # noqa: PLC0415

    line_cook_lm = build_line_cook_lm(cfg)
    head_chef_lm = build_head_chef_lm(cfg)

    loader = _SLICE_LOADERS[slice_name]
    tickets = loader(combined_jsonl)

    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = out_dir / "candidates.jsonl"
    raw_path = out_dir / "raw_records.jsonl"
    accepted_path = out_dir / "accepted.jsonl"
    quarantine_path = out_dir / "quarantine.jsonl"

    line_cook = line_cook_factory()
    head_chef = head_chef_factory()

    emitted: list[InvariantRecord] = []
    candidates_emitted_count = 0

    with candidates_path.open("w", encoding="utf-8") as cand_fh, raw_path.open(
        "w", encoding="utf-8"
    ) as raw_fh:
        for ticket in tickets:
            payload = _ticket_payload(slice_name, ticket)

            # --- Phase 1: line cook extracts candidates ----------------
            with dspy.context(lm=line_cook_lm):
                cook_pred = line_cook(tickets=payload)
            candidates = _coerce_records(
                getattr(cook_pred, "records", []) or []
            )
            for cand in candidates:
                cand_fh.write(
                    json.dumps(cand.model_dump(mode="json"), sort_keys=True)
                    + "\n"
                )
            candidates_emitted_count += len(candidates)

            # --- Phase 2: head chef reviews ONCE (1-depth) -------------
            # Skip if the line cook emitted nothing — no candidates to
            # grade, and we'd only waste tokens round-tripping an empty
            # list.
            if candidates:
                with dspy.context(lm=head_chef_lm):
                    chef_pred = head_chef(
                        tickets=payload, candidates=candidates
                    )
                reviewed = _coerce_records(
                    getattr(chef_pred, "records", []) or []
                )
            else:
                reviewed = []

            for rec in reviewed:
                raw_fh.write(
                    json.dumps(rec.model_dump(mode="json"), sort_keys=True)
                    + "\n"
                )
                emitted.append(rec)

    accepted_count, quarantined_count = check_records(
        emitted, combined_json, accepted_path, quarantine_path
    )

    return RunSummary(
        slice_name=slice_name,
        tickets_dispatched=len(tickets),
        candidates_emitted=candidates_emitted_count,
        records_emitted=len(emitted),
        accepted=accepted_count,
        quarantined=quarantined_count,
        candidates_path=candidates_path.resolve(),
        raw_records_path=raw_path.resolve(),
        accepted_path=accepted_path.resolve(),
        quarantine_path=quarantine_path.resolve(),
    )


def _coerce_records(raw: Iterable[Any]) -> list[InvariantRecord]:
    """Normalise a predictor's `.records` iterable into validated
    `InvariantRecord` instances.

    DSPy hands us already-parsed Pydantic instances under the typed
    Signature; this is defensive in case a stub or an adapter returns
    plain dicts — we validate on the way through so downstream never
    sees an unvalidated record.
    """
    out: list[InvariantRecord] = []
    for rec in raw:
        if isinstance(rec, InvariantRecord):
            out.append(rec)
        else:
            out.append(InvariantRecord.model_validate(rec))
    return out


__all__ = [
    "RunSummary",
    "SliceName",
    "run",
]
