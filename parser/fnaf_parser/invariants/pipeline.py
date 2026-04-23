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

3. Dispatch each ticket to `dspy.Predict(ExtractInvariants)`. Each
   ticket becomes ONE LLM call — bucket tickets carry multiple rows,
   but the model sees them as a batch and can emit multiple records
   referencing any row in the batch.

4. Collect every emitted `InvariantRecord` into a flat list, stream-
   write it to `<out>/raw_records.jsonl`.

5. Hand off to `citation_checker.check_records`, which writes
   `<out>/accepted.jsonl` and `<out>/quarantine.jsonl`.

6. Return a `RunSummary` with counts and paths — the CLI prints it.

Why lazy imports
----------------

`import dspy` is deferred to `run()` (never at module import). That
keeps `from fnaf_parser.invariants.pipeline import RunSummary`
importable in environments where DSPy isn't installed (e.g. a Rust-side
CI job validating only the citation checker). Tests for this module
either (a) don't enter `run()` at all, or (b) patch `dspy.Predict` to
return a stub before calling `run()`.

Why no async
------------

Slice C is ~800 rows; Slices A + B combined are ~400 buckets. At
OpenRouter's default rate limit the whole Slice C pass is under ten
minutes sequential. Async dispatch would save wall-clock but complicates
error handling, retry semantics, and JSONL write ordering. Keep the
pilot simple; reconsider if/when a later probe stretches the ticket
count by an order of magnitude.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from fnaf_parser.invariants.citation_checker import check_records
from fnaf_parser.invariants.config import (
    OpenRouterConfig,
    build_lm,
    load_config,
)
from fnaf_parser.invariants.signatures import ExtractInvariants, InvariantRecord
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
    (dispatched vs emitted vs accepted vs quarantined) so post-run
    triage can compare yield rates across slices without re-reading
    the JSONL artefacts.
    """

    slice_name: SliceName
    tickets_dispatched: int
    records_emitted: int
    accepted: int
    quarantined: int
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
            "records_emitted": self.records_emitted,
            "accepted": self.accepted,
            "quarantined": self.quarantined,
            "artefacts": {
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


# --- Predictor factory --------------------------------------------------


PredictorFactory = Callable[[], Callable[..., Any]]


def _default_predictor_factory() -> Callable[..., Any]:
    """Build the default DSPy `Predict(ExtractInvariants)` callable.

    Isolated behind a factory so tests can inject a stub without
    importing DSPy. The factory is invoked at most once per `run()`
    call, after the config has resolved, so import failures here
    surface only when actually running the pipeline.
    """
    import dspy  # noqa: PLC0415  (lazy — keeps dspy off import-time path)

    return dspy.Predict(ExtractInvariants)


# --- Main entry point ---------------------------------------------------


def run(
    slice_name: SliceName,
    combined_jsonl: Path,
    combined_json: Path,
    out_dir: Path,
    config: OpenRouterConfig | None = None,
    predictor_factory: PredictorFactory = _default_predictor_factory,
) -> RunSummary:
    """Run the head-chef dispatch loop for one slice.

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
        Directory for the three artefacts: `raw_records.jsonl`,
        `accepted.jsonl`, `quarantine.jsonl`. Created if missing.
    config
        Optional pre-resolved config. If None, loaded from env.
        `OpenRouterConfigError` propagates on missing key.
    predictor_factory
        Optional override for the DSPy predictor — tests pass a stub
        here to exercise the orchestration without network I/O.

    Returns
    -------
    RunSummary
        Counts + artefact paths. CLI serialises this as JSON.
    """
    cfg = config if config is not None else load_config()

    # Only now that the config has resolved do we import dspy. Keeps
    # the CLI snappy (and importable) when the key is missing.
    import dspy  # noqa: PLC0415

    lm = build_lm(cfg)
    dspy.configure(lm=lm)

    loader = _SLICE_LOADERS[slice_name]
    tickets = loader(combined_jsonl)

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_records.jsonl"
    accepted_path = out_dir / "accepted.jsonl"
    quarantine_path = out_dir / "quarantine.jsonl"

    predictor = predictor_factory()
    emitted: list[InvariantRecord] = []
    with raw_path.open("w", encoding="utf-8") as raw_fh:
        for ticket in tickets:
            payload = _ticket_payload(slice_name, ticket)
            prediction = predictor(tickets=payload)
            records: Iterable[Any] = getattr(prediction, "records", []) or []
            for rec in records:
                # DSPy hands us already-parsed Pydantic instances under the
                # typed Signature. Defensive in case a stub returns dicts.
                if isinstance(rec, InvariantRecord):
                    model = rec
                else:
                    model = InvariantRecord.model_validate(rec)
                raw_fh.write(
                    json.dumps(model.model_dump(mode="json"), sort_keys=True)
                    + "\n"
                )
                emitted.append(model)

    accepted_count, quarantined_count = check_records(
        emitted, combined_json, accepted_path, quarantine_path
    )

    return RunSummary(
        slice_name=slice_name,
        tickets_dispatched=len(tickets),
        records_emitted=len(emitted),
        accepted=accepted_count,
        quarantined=quarantined_count,
        raw_records_path=raw_path.resolve(),
        accepted_path=accepted_path.resolve(),
        quarantine_path=quarantine_path.resolve(),
    )


__all__ = [
    "RunSummary",
    "SliceName",
    "run",
]
