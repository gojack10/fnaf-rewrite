"""DSPy-backed `ExtractInvariants` signature, separated from its
Pydantic co-models so `import fnaf_parser.invariants.signatures` does
not force `import dspy` at collection time.

Why this file exists
--------------------

`dspy` pulls in ~70 sub-modules at import (litellm, tokenizers, plus an
internal deprecation-warning pass over `avatar_optimizer.py`). On the
FNAF parser that adds ~3 s to every `pytest` collection because
`tests/test_invariants_signatures.py` imports the `ExtractInvariants`
class via `fnaf_parser.invariants.signatures`.

The DSPy `Signature` metaclass evaluates `dspy.InputField` /
`dspy.OutputField` calls at class-body time, so the import can't be
hidden inside a factory without losing the declarative signature.
Splitting the signature into its own module isolates that cost: every
other consumer of `Citation` / `InvariantRecord` (Pydantic-only) now
imports them from `signatures.py` without paying the DSPy tax.

The pipeline still lazy-imports this module from inside
`_default_predictor_factory()`, so only the predictor construction
path pulls DSPy in.
"""

from __future__ import annotations

from typing import Any

import dspy

from fnaf_parser.invariants.signatures import InvariantRecord


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
