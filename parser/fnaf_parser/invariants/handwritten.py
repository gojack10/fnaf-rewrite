"""Owner-written FNAF 1 invariants — pinned to real combined.jsonl rows.

These are the five invariants we know by inspection of the pack, pinned
to exact `(frame, event_group_index, type, cond_or_act_index)`
coordinates that were verified against `out/algorithm/combined.jsonl`.
Each record's `pseudo_code` is chosen so Citation Checker's
`expr_str_match` auto-accepts it — the pseudo-code is the verbatim
`expr_str` substring from the cited parameter, not a rich rendering.

Why bake these in:

1. **Baseline before the LLM runs.** Even with a cold DSPy pipeline,
   the Rust Test Emission downstream has five guaranteed-good records
   it can compile into `#[test]` stubs. That unblocks parser-rebuild
   work on day 1 instead of waiting for the pilot run.
2. **Citation Checker smoke test.** Each record is an end-to-end fire
   of the verification ladder against the real combined.json. If the
   ladder drifts (e.g. a schema change in the Algorithm Emission), the
   handwritten test catches it before the pilot run does.
3. **Human-verified intent anchor.** The `claim` strings describe the
   invariant in owner-voice; the `pseudo_code` string is what Citation
   Checker actually matches. Anyone reading the quarantine file later
   will see the claim + the verbatim expr_str and can verify both
   against the pack at the cited coordinate.

The pseudo_code fields are deliberately terse (the verbatim expr_str)
because Citation Checker's normalized substring match is the gate —
using a richer pseudo_code that ISN'T a substring of expr_str would
silently quarantine these records on the first run. The rich
description lives in `claim`.
"""

from __future__ import annotations

from fnaf_parser.invariants.signatures import Citation, InvariantRecord


# --- Power counter invariants -------------------------------------------

# Invariant #1: power_left initializes to 999 at game start.
# Coord: frame=3, eg=175, action[0] — SetCounterValue(power left) = 999
_INV_POWER_INIT = InvariantRecord(
    claim=(
        "Power counter 'power left' initializes to 999 (the night's "
        "starting energy pool) on the game-start event group."
    ),
    citations=[
        Citation(
            frame=3,
            event_group_index=175,
            type="action",
            cond_or_act_index=0,
            parameter_index=0,
        )
    ],
    pseudo_code="999",
    kind="numeric_assignment",
)


# Invariant #2: power_left drains per tick by the usage meter's counter
# value. The usage meter scales with how many game systems are drawing
# power (cameras open, lights on, doors shut), so this is the canonical
# scaled-drain rule.
# Coord: frame=3, eg=176, action[0] — SubtractCounterValue(power left)
#        by expr "usage meter.CounterValue"
_INV_POWER_DRAIN_SCALES = InvariantRecord(
    claim=(
        "Power left decrements each drain tick by the current value of "
        "the 'usage meter' counter, scaling drain with active systems."
    ),
    citations=[
        Citation(
            frame=3,
            event_group_index=176,
            type="action",
            cond_or_act_index=0,
            parameter_index=0,
        )
    ],
    pseudo_code="usage meter.CounterValue",
    kind="numeric_assignment",
)


# Invariant #5: power_left clamps to 0 when it goes negative. Pairs
# with #2 — the drain can overshoot to -1 or lower on the last tick,
# so the engine floors it.
# Coord: frame=3, eg=284, action[0] — SetCounterValue(power left) = 0
_INV_POWER_CLAMP_ZERO = InvariantRecord(
    claim=(
        "Power left is clamped to 0 whenever a drain tick would drive "
        "it negative, flooring the counter at empty."
    ),
    citations=[
        Citation(
            frame=3,
            event_group_index=284,
            type="action",
            cond_or_act_index=0,
            parameter_index=0,
        )
    ],
    pseudo_code="0",
    kind="numeric_assignment",
)


# --- Animatronic AI invariants ------------------------------------------

# Invariant #3: Freddy's AI level is compared against a ceiling of 20.
# Freddy's AI maxes at 20 — the engine uses `< 20` as the guard to
# decide whether to bump his level.
# Coord: frame=12, eg=4, condition[1] — CompareCounter(freddy AI) vs 20
_INV_FREDDY_AI_CAP = InvariantRecord(
    claim=(
        "Freddy's AI level is gated against a maximum of 20 — the "
        "engine only increments 'freddy AI' while it is under 20."
    ),
    citations=[
        Citation(
            frame=12,
            event_group_index=4,
            type="condition",
            cond_or_act_index=1,
            parameter_index=0,
        )
    ],
    pseudo_code="20",
    kind="numeric_comparison",
)


# Invariant #4: Foxy's progress counter is gated against stage 5 —
# stage 5 is the 'running down the hall' state, guarded by this
# comparison so the run-down transition doesn't re-fire.
# Coord: frame=3, eg=23, condition[2] — CompareCounter(fox progress) vs 5
_INV_FOXY_RUNDOWN_THRESHOLD = InvariantRecord(
    claim=(
        "Foxy's 'fox progress' counter is checked against stage 5 — "
        "the run-down-the-hall trigger, guarded to prevent re-firing."
    ),
    citations=[
        Citation(
            frame=3,
            event_group_index=23,
            type="condition",
            cond_or_act_index=2,
            parameter_index=0,
        )
    ],
    pseudo_code="5",
    kind="numeric_comparison",
)


# --- Export ---------------------------------------------------------------

# Order is intentional: power subsystem first (init → drain → clamp),
# then AI subsystem (Freddy cap, Foxy threshold). Downstream dedup
# relies on insertion order being stable.
HANDWRITTEN: list[InvariantRecord] = [
    _INV_POWER_INIT,
    _INV_POWER_DRAIN_SCALES,
    _INV_POWER_CLAMP_ZERO,
    _INV_FREDDY_AI_CAP,
    _INV_FOXY_RUNDOWN_THRESHOLD,
]


__all__ = ["HANDWRITTEN"]
