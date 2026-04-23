"""DSPy-backed `ReviewInvariants` signature — the head-chef review seat.

Why a separate file
-------------------

Same reason `extract_signature.py` is its own file: the DSPy `Signature`
metaclass evaluates `InputField` / `OutputField` calls at class-body
time, so `import dspy` can't be hidden inside a factory without losing
the declarative signature. Keeping this here lets
`fnaf_parser.invariants.signatures` stay DSPy-free (Pydantic models
only, fast `pytest` collection) while the head-chef predictor still
gets a proper typed Signature.

Why a single review Signature (not per-slice)
---------------------------------------------

One review prompt, one schema, one set of rules. Per-slice review
variants would be a future optimisation — for V1, the contract is
uniform: "grade the line cook's candidates against the original
ticket, drop the weakly-grounded ones, don't invent new citations."

Why 1-depth only
----------------

The head chef is invoked EXACTLY ONCE per ticket. No re-prompt loop,
no recursive refinement, no "ask again if unhappy" chain. Recursion is
a loop hazard (the head chef could bounce rejections back to the line
cook until convergence fails to arrive), so we intentionally cap it at
one pass. The Citation Checker is the deterministic tiebreaker if the
head chef over-cleans; over-cleaning is recoverable, an infinite loop
is not.
"""

from __future__ import annotations

from typing import Any

import dspy

from fnaf_parser.invariants.signatures import InvariantRecord


class ReviewInvariants(dspy.Signature):
    """Review the line cook's candidate invariants against the original
    tickets. Keep the records that are well-grounded; drop the weakly-
    grounded, misstated, or redundant ones.

    Hard rules (contracts, not hints):

    1. You MAY ONLY emit records whose citations already appear in the
       candidates list. Do NOT invent new citations, new coordinates,
       or new parameter indices. The downstream Citation Checker
       quarantines any record whose citation does not resolve, so
       invention is pure wasted tokens — and worse, it creates the
       illusion of a verified claim where none exists.

    2. You MAY refine `claim` wording for clarity and MAY tighten
       `pseudo_code` to match the cited `expr_str` more literally, but
       you MUST NOT change the semantic content of either field.
       Paraphrases that alter meaning are worse than the original;
       they defeat the verbatim-match path in Citation Checker.

    3. Prefer silence. If you are unsure whether a candidate is
       correct, DROP it. False positives are worse than false
       negatives — Citation Checker accepts on ANY verification rung,
       so a lenient review leaks hallucinations past the gate.

    4. If two candidates state the same invariant with different
       wording, keep ONE. Pick the one whose `pseudo_code` is closest
       to the cited `expr_str` verbatim; that maximises the chance of
       a string-match acceptance downstream.

    5. Return an empty list when no candidate survives. An empty
       output is a valid review outcome, not a failure.

    You are invoked exactly ONCE per ticket. There is no further
    refinement pass — do all cleaning in this single call.
    """

    tickets: list[dict[str, Any]] = dspy.InputField(
        desc=(
            "The original Scout Pass ticket the line cook saw. Same "
            "shape as ExtractInvariants.tickets — the ground truth you "
            "grade candidates against. Every valid citation MUST point "
            "somewhere inside this payload."
        )
    )
    candidates: list[InvariantRecord] = dspy.InputField(
        desc=(
            "The line cook's candidate invariants for this ticket. "
            "Every emitted record MUST be drawn from this list "
            "(possibly with refined wording per the rules above)."
        )
    )
    records: list[InvariantRecord] = dspy.OutputField(
        desc=(
            "Filtered / refined records. Empty list is valid. Each "
            "record MUST satisfy the InvariantRecord schema — "
            "malformed records will be re-asked automatically."
        )
    )
