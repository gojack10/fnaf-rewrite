"""DSPy typed Signature + Pydantic models for invariant extraction.

The line cook (the only LLM call in the pipeline) exposes exactly one
Signature: `ExtractInvariants`. Its output is a `list[InvariantRecord]`
where every record carries a natural-language claim, one or more
citation coordinates, and a pseudo-code rendering that Citation Checker
can machine-verify against the cited parameter's `expr_str` / `ast`.

Why a single Signature
----------------------

Scout Pass already decided the slice axes — the pipeline just batches
tickets and dispatches them. A single Signature keeps the prompting
surface small: one docstring, one output schema, one set of eval hooks.
Adding more Signatures is a future optimisation (e.g. a separate
"Summarise this bucket" pre-pass) and deliberately out of V1 scope.

Why Pydantic output enforcement
-------------------------------

DSPy 3.x supports `OutputField` with a Pydantic type — when the LLM
replies with malformed or missing-required-field JSON, DSPy re-asks
automatically. That gives us a hallucination gate before Citation
Checker even runs: every record *must* parse into `InvariantRecord`, so
we never ship a record missing its citations or a citation missing its
coordinate triple.
"""

from __future__ import annotations

from typing import Any, Literal

import dspy
from pydantic import BaseModel, Field, field_validator


# --- Citation coordinate ------------------------------------------------


class Citation(BaseModel):
    """A pointer into `combined.json` / `combined.jsonl`.

    Every accepted invariant must cite at least one of these so
    Citation Checker can load the underlying row and verify the claim.
    The 4-tuple `(frame, event_group_index, type, cond_or_act_index)`
    uniquely identifies one condition-or-action row — `type` is
    required because `cond_or_act_index` is per-type in the jsonl
    (conditions index from 0 and actions independently index from 0
    within the same event group). The optional `parameter_index`
    narrows to one parameter within the row — used when the claim
    references a specific Expression AST or a specific SHORT label.

    Coordinate ranges are not validated here against the real pack
    (that would couple this module to a loaded combined.json on every
    construction). Citation Checker is the layer that verifies the
    coordinates actually resolve to a row; this class only enforces
    shape.
    """

    frame: int = Field(ge=0, description="0-based frame index (field name matches combined.jsonl)")
    event_group_index: int = Field(
        ge=0, description="0-based index within the frame's event groups"
    )
    type: Literal["condition", "action"] = Field(
        description=(
            "Whether the cited row is a condition or an action. Required "
            "because cond_or_act_index is type-scoped in the jsonl."
        )
    )
    cond_or_act_index: int = Field(
        ge=0,
        description=(
            "0-based index within the group's conditions list (if "
            "type='condition') or actions list (if type='action')"
        ),
    )
    parameter_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional 0-based parameter index within the cited row. "
            "Omit when citing the row as a whole."
        ),
    )


# --- Invariant record ---------------------------------------------------


class InvariantRecord(BaseModel):
    """One machine-checkable invariant claim.

    The contract is:

    - `claim` — natural-language statement of the invariant. Human-
      readable and LLM-produced; not machine-verified directly.
    - `citations` — at least one `Citation` pointing at the row(s) the
      claim is grounded in. Citation Checker verifies every one.
    - `pseudo_code` — a flat pseudo-code rendering of the invariant.
      For numeric math (Slice C), this is ideally a verbatim copy of
      the cited parameter's `expr_str` field, which makes the
      strongest form of Citation Checker validation (string match)
      possible.
    - `kind` — coarse tag that lets downstream dedup / Rust Test
      Emission route the record without re-parsing `claim`. Free-text
      is deliberately not allowed; the enum is small on purpose.
    - `confidence` — optional self-reported confidence from the line
      cook, clamped to [0, 1]. Used by Citation Checker as a tie-
      breaker, NOT as a gating threshold (the gate is deterministic
      citation verification).
    """

    claim: str = Field(min_length=4)
    citations: list[Citation] = Field(min_length=1)
    pseudo_code: str = Field(min_length=1)
    kind: Literal[
        "numeric_assignment",
        "numeric_comparison",
        "state_transition",
        "event_trigger",
        "other",
    ] = Field(
        description=(
            "Routing tag: numeric_assignment for `var := expr` rules, "
            "numeric_comparison for `var <op> threshold` guards, "
            "state_transition for animation/state label changes, "
            "event_trigger for timers/keypresses/collisions that fire "
            "a group, other for anything else."
        )
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Optional LLM self-rating"
    )

    @field_validator("claim", "pseudo_code")
    @classmethod
    def _strip_whitespace(cls, v: str) -> str:
        """Strip leading/trailing whitespace so prompt-echo garbage
        like newlines or trailing spaces don't cause false inequality
        downstream (e.g. Citation Checker's string-match path)."""
        return v.strip()


# --- DSPy Signature -----------------------------------------------------


class ExtractInvariants(dspy.Signature):
    """Given a batch of FNAF 1 condition/action records, extract every
    machine-verifiable invariant you can ground in the cited JSON.

    Every invariant you emit MUST include at least one citation that
    points at the exact `(frame, event_group_index, type,
    cond_or_act_index)` coordinate in the input. If you can't ground
    the claim in a specific coordinate, DO NOT emit it. Silence is
    preferred over uncited claims.

    For numeric math (Expression parameters, codes 22/23/27/45),
    copy the `expr_str` field verbatim into `pseudo_code` — that makes
    the claim mechanically verifiable by string match.

    For SHORT parameters (codes 10/26), reference the decoded `label`
    field when it's present (e.g. "state label transitions to 'freddy
    attack'") so Citation Checker can confirm by matching against the
    row's label.

    Do not invent ops, parameters, or labels not present in the input.
    Do not combine records across different event groups in a single
    claim unless you cite every group involved.
    """

    tickets: list[dict[str, Any]] = dspy.InputField(
        desc=(
            "Batch of condition/action records from one Scout Pass slice "
            "ticket. Each record carries frame, event_group_index, "
            "cond_or_act_index, type, num_name, object_name, and the "
            "full parameters list with decoded AST / labels."
        )
    )
    records: list[InvariantRecord] = dspy.OutputField(
        desc=(
            "Zero or more invariant records. Empty list is valid when "
            "no claim is groundable. Each record MUST satisfy the "
            "InvariantRecord schema — malformed records will be "
            "re-asked automatically."
        )
    )
