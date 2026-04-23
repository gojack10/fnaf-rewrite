"""Deterministic citation verifier — the hallucination gate.

Runs AFTER the DSPy.RLM Pipeline produces its raw JSONL and BEFORE Rust
Test Emission compiles the accepted records into `#[test]` stubs.
For every `InvariantRecord` emitted by the line cook, this module:

1. Loads the cited `(frame, event_group_index, type, cond_or_act_index)`
   row from `combined.json`.
2. Asks "does the claim match what's actually at those coordinates?"
   using a deterministic ladder of verification strategies (below).
3. Writes an `accept` or `reject` decision, with a reason, to one of
   two output JSONL files: `accepted.jsonl` and `quarantine.jsonl`.

**This module NEVER calls an LLM.** Not in the verifier, not in a
fallback, not via `subprocess`. If a claim can't be deterministically
verified, it goes to quarantine with a reason. Owner-reviewed. The
pipeline's credibility rests on that constraint.

Verification ladder (cheapest → most expensive)
------------------------------------------------

1. **Coordinate resolution** — the cited (frame, event_group_index,
   type, cond_or_act_index) 4-tuple must resolve to an actual row in
   combined.json. Missing row → immediate reject.
2. **Parameter-index bounds** — if a citation carries
   `parameter_index`, it must be in range. Out-of-range → reject.
3. **expr_str verbatim match** — for Expression parameters, if the
   record's `pseudo_code` appears verbatim (case-insensitive,
   whitespace-normalised) anywhere in the cited row's `expr_str`,
   auto-accept with reason `"expr_str_match"`.
4. **expr_str reverse match** (pi=int only) — if the cited parameter's
   `expr_str` appears as a standalone token inside the record's
   `pseudo_code`, auto-accept with reason
   `"expr_str_reverse_match"`. Catches whole-action summaries like
   `SetChannelVolume(2, 100)` cited at pi=0 where the per-parameter
   `expr_str` is just `'2'`. Word-boundary guarded so `'1'` doesn't
   trivially match inside `'100'`.
5. **All-exprs-in-pseudo** (pi=None only) — if *every*
   EXPRESSION-bearing parameter's `expr_str` appears as a standalone
   token inside the record's `pseudo_code`, auto-accept with reason
   `"all_exprs_in_pseudo"`. Catches whole-action claims (no
   `parameter_index`) that faithfully summarise the call, like pseudo
   `Ini('beatgame', 0)` against two expression params whose expr_strs
   are `"'beatgame'"` and `'0'`.
6. **SHORT label match** — for codes 10/26 (SHORT) that carry a
   decoded `label`, if the record's `pseudo_code` quotes the exact
   label, auto-accept with reason `"short_label_match"`.
7. **Num-name match** — the cited row's `num_name` appears in the
   record's `claim` or `pseudo_code`. Weak signal → accept with
   reason `"num_name_match"` only when kind is
   `event_trigger`/`state_transition` (where num_name is the
   operative word), reject otherwise.
8. **No match** — quarantine with reason
   `"no_verification_strategy_matched"`.

Output files
------------

- `accepted.jsonl` — one line per accepted record, shape:
  `{"record": {...}, "reasons": [{"citation_index": 0, "reason": "..."}]}`.
- `quarantine.jsonl` — one line per rejected record, same shape with
  `{"reason": "..."}` for every citation that failed.

Neither file is consumed internally — both are artefacts for the owner
and for the Rust Test Emission downstream.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from fnaf_parser.invariants.signatures import Citation, InvariantRecord

# --- Constants ----------------------------------------------------------

# Parameter codes that carry an Expression AST + expr_str flat rendering.
# Keep in sync with the Scout Pass Slice C filter.
_EXPRESSION_CODES = frozenset({22, 23, 27, 45})

# Parameter codes that carry a decoded `label` field when present.
# Currently only SHORT codes 10 and 26 — code 50 (AlterableValue) uses
# the same int16 loader but never carries a label.
_LABEL_CARRYING_CODES = frozenset({10, 26})


# --- Outcomes -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CitationOutcome:
    """Per-citation verification result.

    A record is accepted iff every one of its citations returns an
    outcome with `accepted=True`. One reject in the list kicks the
    whole record to quarantine — a citation that can't be verified is
    treated as strictly worse than "no citation at all" (the
    line-cook Signature validates that citations is non-empty).
    """

    citation_index: int
    accepted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome for one full record.

    `accepted` is True iff every citation was accepted. `outcomes`
    preserves per-citation detail so quarantine triage has everything
    the owner needs to decide whether the line cook was wrong or the
    checker is too strict.
    """

    record: InvariantRecord
    accepted: bool
    outcomes: list[CitationOutcome]


# --- Combined.json loader -----------------------------------------------


# Dict keyed by (frame, event_group_index, type, cond_or_act_index).
# `type` is "condition" or "action" — necessary because the jsonl /
# combined.json structure indexes conditions and actions independently
# (both start at 0 inside the same event group).
CombinedIndex = dict[tuple[int, int, str, int], dict[str, Any]]


def _load_combined(combined_json_path: Path) -> CombinedIndex:
    """Load combined.json into a flat dict keyed by the 4-tuple
    `(frame, event_group_index, type, cond_or_act_index)`.

    combined.json's actual schema (as emitted by `algorithm.emit`) is:

        {
          "decoder_version": "...",
          "frames": [
            {
              "frame_index": 0,           # 0-based frame index
              "frame_name": "...",
              "event_groups": [
                {
                  "conditions": [{...}, ...],
                  "actions":    [{...}, ...],
                  ...
                },
                ...
              ]
            },
            ...
          ]
        }

    Conditions and actions are stored in separate lists inside each
    event group. We flatten to a dict once, up-front, so the hot path
    per citation is a single dict lookup instead of walking the pack.
    """
    with combined_json_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    out: CombinedIndex = {}
    for frame_rec in doc.get("frames", []):
        frame = frame_rec.get("frame_index")
        if not isinstance(frame, int):
            continue
        for eg_index, group in enumerate(frame_rec.get("event_groups", [])):
            for row_type in ("condition", "action"):
                # combined.json uses plural key names ("conditions" /
                # "actions"); keep internal `row_type` singular so it
                # matches the Citation field domain directly.
                plural_key = row_type + "s"
                for coa_index, row in enumerate(group.get(plural_key, [])):
                    out[(frame, eg_index, row_type, coa_index)] = row
    return out


def _row_at(combined: CombinedIndex, cit: Citation) -> dict[str, Any] | None:
    """Return the row at `cit`'s coordinates, or None if any component
    doesn't resolve. Defensive against incomplete combined.json (e.g. a
    partial test fixture)."""
    return combined.get(
        (cit.frame, cit.event_group_index, cit.type, cit.cond_or_act_index)
    )


# --- Normalisers --------------------------------------------------------


_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(s: str) -> str:
    """Lowercase + collapse internal whitespace to single spaces.

    Used for `expr_str` and label matching so cosmetic whitespace in
    the line cook's pseudo_code doesn't cause a false miss. NOT used
    for num_name matching — num_names are emitted by the Name Tables
    Port verbatim and whitespace there would be a signal of a
    mismatch worth rejecting.
    """
    return _WHITESPACE_RE.sub(" ", s.strip().lower())


def _expr_str_token_in(expr_str_norm: str, pseudo_norm: str) -> bool:
    """Return True if `expr_str_norm` appears as a standalone token
    inside `pseudo_norm`.

    Word boundaries (`\\b`) are required only at edges where
    `expr_str_norm` itself has a word character. If it begins or ends
    with a symbol (paren, quote, operator), that symbol is already a
    boundary, so adding `\\b` would be wrong — `\\b` requires a
    word/non-word transition and two adjacent non-word characters
    would fail to match.

    This prevents the classic false accept where a single-digit
    expr_str like `'1'` is naively-substring-matched inside a larger
    numeric literal like `'100'`.
    """
    if not expr_str_norm:
        return False
    left = r"\b" if (expr_str_norm[0].isalnum() or expr_str_norm[0] == "_") else ""
    right = r"\b" if (expr_str_norm[-1].isalnum() or expr_str_norm[-1] == "_") else ""
    pattern = left + re.escape(expr_str_norm) + right
    return re.search(pattern, pseudo_norm) is not None


# --- Verification strategies --------------------------------------------


def _check_expr_str(
    row: dict[str, Any],
    cit: Citation,
    record: InvariantRecord,
) -> str | None:
    """Match the record's `pseudo_code` against EXPRESSION-bearing
    parameters using three nested strategies, returning the first
    accept-reason that fires (or None to fall through to the next
    rung of the ladder).

    Strategies, in priority order:

    1. `expr_str_match` (forward) — `pseudo_code` appears verbatim
       inside some expression param's `expr_str`. Tightest; used when
       the line cook emitted a narrow per-parameter fragment.
    2. `expr_str_reverse_match` — when `pi=int`, the cited param's
       `expr_str` appears as a standalone token inside `pseudo_code`.
       Catches whole-action summaries pinned at one param.
    3. `all_exprs_in_pseudo` — when `pi=None`, *every*
       expression-bearing param's `expr_str` appears as a standalone
       token in `pseudo_code`. Strong evidence the pseudo is a
       faithful whole-action summary.

    Strategies 2 and 3 are guarded by a word-boundary test (see
    `_expr_str_token_in`) so a single-digit `expr_str` can't match
    inside a larger numeric literal.
    """
    params = row.get("parameters", [])
    if cit.parameter_index is not None:
        if cit.parameter_index >= len(params):
            return None
        candidates: Iterable[dict[str, Any]] = [params[cit.parameter_index]]
    else:
        candidates = params

    needle = _normalise(record.pseudo_code)
    if not needle:
        return None

    # 1. Forward substring: pseudo_code ⊆ expr_str.
    for p in candidates:
        if p.get("code") not in _EXPRESSION_CODES:
            continue
        expr_str = p.get("expr_str")
        if not isinstance(expr_str, str):
            continue
        if needle in _normalise(expr_str):
            return "expr_str_match"

    # Collect the subset of candidates that are expression-bearing and
    # carry a non-empty expr_str — both reverse strategies need at
    # least one such param to have anything to match against.
    expr_params: list[dict[str, Any]] = []
    for p in candidates:
        if p.get("code") not in _EXPRESSION_CODES:
            continue
        expr_str = p.get("expr_str")
        if isinstance(expr_str, str) and expr_str:
            expr_params.append(p)
    if not expr_params:
        return None

    # 2. Reverse substring (pi=int only): the cited param's expr_str
    #    appears as a standalone token inside pseudo_code.
    if cit.parameter_index is not None:
        # `candidates` was restricted to the single cited param; so
        # `expr_params` has at most one entry here.
        expr_str_norm = _normalise(expr_params[0]["expr_str"])
        if _expr_str_token_in(expr_str_norm, needle):
            return "expr_str_reverse_match"
        return None

    # 3. All-exprs-in-pseudo (pi=None): require EVERY expression
    #    param's expr_str to be a token in the pseudo. Using the
    #    unrestricted candidates list — if any expression param is
    #    missing an expr_str (empty/non-string), we can't prove the
    #    pseudo faithfully covers the whole call, so bail.
    for p in candidates:
        if p.get("code") not in _EXPRESSION_CODES:
            continue
        expr_str = p.get("expr_str")
        if not isinstance(expr_str, str) or not expr_str:
            return None
        if not _expr_str_token_in(_normalise(expr_str), needle):
            return None
    return "all_exprs_in_pseudo"


def _check_short_label(
    row: dict[str, Any],
    cit: Citation,
    record: InvariantRecord,
) -> str | None:
    """Return an accept reason if a SHORT parameter's decoded `label`
    field appears verbatim (quoted or unquoted, normalised) in the
    record's `pseudo_code` OR `claim`."""
    params = row.get("parameters", [])
    if cit.parameter_index is not None:
        if cit.parameter_index >= len(params):
            return None
        candidates: Iterable[dict[str, Any]] = [params[cit.parameter_index]]
    else:
        candidates = params

    text = _normalise(record.pseudo_code + " " + record.claim)
    for p in candidates:
        if p.get("code") not in _LABEL_CARRYING_CODES:
            continue
        decoded = p.get("decoded") or {}
        label = decoded.get("label")
        if not isinstance(label, str) or not label:
            continue
        if _normalise(label) in text:
            return "short_label_match"
    return None


def _check_num_name(
    row: dict[str, Any],
    record: InvariantRecord,
) -> str | None:
    """Return an accept reason if the row's num_name appears verbatim in
    the record's claim or pseudo_code.

    Only accepted for record kinds where num_name is the operative term:
    event_trigger (conditions like `PlayerPressesKey`) and
    state_transition (actions like `SetAnimationSequence`). For numeric
    kinds the num_name alone is too weak a signal — the expr_str check
    is the operative gate.
    """
    if record.kind not in ("event_trigger", "state_transition"):
        return None
    num_name = row.get("num_name")
    if not isinstance(num_name, str) or not num_name:
        return None
    haystack = record.claim + " " + record.pseudo_code
    if num_name in haystack:
        return "num_name_match"
    return None


# --- Per-citation gate --------------------------------------------------


def _check_citation(
    combined: CombinedIndex,
    record: InvariantRecord,
    cit: Citation,
    citation_index: int,
) -> CitationOutcome:
    """Run the full verification ladder for one citation.

    Cheap checks first (coordinate existence + parameter-index bounds),
    then expr_str, then SHORT label, then num_name fallback, then
    quarantine. Returns a CitationOutcome immediately on the first
    accept-reason hit.
    """
    row = _row_at(combined, cit)
    if row is None:
        return CitationOutcome(
            citation_index=citation_index,
            accepted=False,
            reason="citation_coordinate_out_of_range",
        )

    if cit.parameter_index is not None and cit.parameter_index >= len(
        row.get("parameters", [])
    ):
        return CitationOutcome(
            citation_index=citation_index,
            accepted=False,
            reason="parameter_index_out_of_range",
        )

    expr_reason = _check_expr_str(row, cit, record)
    if expr_reason is not None:
        return CitationOutcome(
            citation_index=citation_index, accepted=True, reason=expr_reason
        )

    label_reason = _check_short_label(row, cit, record)
    if label_reason is not None:
        return CitationOutcome(
            citation_index=citation_index, accepted=True, reason=label_reason
        )

    num_name_reason = _check_num_name(row, record)
    if num_name_reason is not None:
        return CitationOutcome(
            citation_index=citation_index,
            accepted=True,
            reason=num_name_reason,
        )

    return CitationOutcome(
        citation_index=citation_index,
        accepted=False,
        reason="no_verification_strategy_matched",
    )


# --- Public API ---------------------------------------------------------


def verify_record(
    combined: CombinedIndex,
    record: InvariantRecord,
) -> VerificationResult:
    """Verify every citation in `record`. Accept iff all citations pass.

    `combined` is the flattened combined.json produced by
    `_load_combined`. Callers should load it once and pass the same
    object to every `verify_record` call across a pipeline run — the
    checker walks many records and reloading would scale poorly.
    """
    outcomes = [
        _check_citation(combined, record, cit, i)
        for i, cit in enumerate(record.citations)
    ]
    return VerificationResult(
        record=record,
        accepted=all(o.accepted for o in outcomes),
        outcomes=outcomes,
    )


def check_records(
    records: Iterable[InvariantRecord],
    combined_json_path: Path,
    accepted_out: Path,
    quarantine_out: Path,
) -> tuple[int, int]:
    """Stream-verify `records`, write accepts + quarantine JSONL.

    Returns `(accepted_count, quarantined_count)`. Output files are
    overwritten — the Citation Checker is idempotent per run.

    Output rows are sorted by insertion order (which mirrors pipeline
    output order). Deduplication is deliberately NOT performed here —
    that's the Rust Test Emission layer's job so dedup can see the
    accepted+hand-written invariant union.
    """
    combined = _load_combined(combined_json_path)
    accepted = 0
    quarantined = 0
    accepted_out.parent.mkdir(parents=True, exist_ok=True)
    quarantine_out.parent.mkdir(parents=True, exist_ok=True)
    with (
        accepted_out.open("w", encoding="utf-8") as a_fh,
        quarantine_out.open("w", encoding="utf-8") as q_fh,
    ):
        for record in records:
            result = verify_record(combined, record)
            line = json.dumps(
                {
                    "record": result.record.model_dump(mode="json"),
                    "outcomes": [
                        {
                            "citation_index": o.citation_index,
                            "accepted": o.accepted,
                            "reason": o.reason,
                        }
                        for o in result.outcomes
                    ],
                },
                sort_keys=True,
            )
            if result.accepted:
                a_fh.write(line + "\n")
                accepted += 1
            else:
                q_fh.write(line + "\n")
                quarantined += 1
    return accepted, quarantined


__all__ = [
    "CitationOutcome",
    "CombinedIndex",
    "VerificationResult",
    "check_records",
    "verify_record",
]
