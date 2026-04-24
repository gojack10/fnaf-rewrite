"""Tests for `fnaf_parser.invariants.citation_checker`.

Exercises the verification ladder — coordinate resolution, param-index
bounds, expr_str match, SHORT label match, num_name match, quarantine
fallback — end-to-end against a small hand-built combined.json fixture.
No API key required; this is the zero-LLM-fallback layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fnaf_parser.invariants.citation_checker import (
    Citation,
    InvariantRecord,
    _load_combined,
    check_records,
    verify_record,
)


# --- Fixture builder ----------------------------------------------------


def _combined_doc() -> dict[str, Any]:
    """Minimal combined.json shape: two frames, each with one event
    group, each group with one condition and one action. Covers the
    EXPRESSION-param path (code 22) and the SHORT-label path (code 10).
    """
    return {
        "decoder_version": "test-fixture",
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
                        "actions": [
                            {
                                "num_name": "SetAlphaCoefficient",
                                "parameters": [
                                    {
                                        "code": 23,
                                        "code_name": "EXP2",
                                        "expr_str": "50 + Random 100",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "frame_index": 1,
                "frame_name": "Frame 1",
                "event_groups": [
                    {
                        "conditions": [
                            {
                                "num_name": "KeyPressed",
                                "parameters": [],
                            }
                        ],
                        "actions": [
                            {
                                "num_name": "SetAnimSequence",
                                "parameters": [
                                    {
                                        "code": 10,
                                        "code_name": "SHORT",
                                        "value": 3,
                                        "decoded": {"label": "walk"},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                # Frame 2 carries a two-expression-parameter action —
                # the SetChannelVolume shape that stress-tests the
                # reverse-direction + all-exprs-in-pseudo strategies.
                "frame_index": 2,
                "frame_name": "Frame 2",
                "event_groups": [
                    {
                        "conditions": [],
                        "actions": [
                            {
                                "num_name": "SetChannelVolume",
                                "parameters": [
                                    {
                                        "code": 22,
                                        "code_name": "EXP",
                                        "expr_str": "2",
                                    },
                                    {
                                        "code": 22,
                                        "code_name": "EXP",
                                        "expr_str": "100",
                                    },
                                ],
                            },
                            {
                                # Single expression param with expr_str '1' —
                                # used to prove that `'1'` does not
                                # substring-match inside `'100'` under
                                # the word-boundary guard.
                                "num_name": "SetCounterValue",
                                "parameters": [
                                    {
                                        "code": 22,
                                        "code_name": "EXP",
                                        "expr_str": "1",
                                    }
                                ],
                            },
                        ],
                    }
                ],
            },
            {
                # Frame 3 covers V2 strategies: sample-name lookup
                # (code=6) and the normalized-exprs rung that strips
                # `EndParenthesis` tokens and surrounding single-quotes
                # from expr_str before a loose token check.
                "frame_index": 3,
                "frame_name": "Frame 3",
                "event_groups": [
                    {
                        "conditions": [
                            {
                                # `Random 4 EndParenthesis` is how the
                                # decompiler renders `Random(4)` in the
                                # flat expr_str stream — the trailing
                                # `EndParenthesis` marker replaces `)`.
                                "num_name": "CompareCounter",
                                "parameters": [
                                    {
                                        "code": 22,
                                        "code_name": "EXP",
                                        "expr_str": "Random 4 EndParenthesis",
                                    },
                                    {
                                        "code": 23,
                                        "code_name": "EXP2",
                                        "expr_str": "1",
                                    },
                                ],
                            }
                        ],
                        "actions": [
                            {
                                # PlayChannelSample shape: code=6 Sample
                                # param carries the `name` field that
                                # the Event Parameters decoder pulled
                                # out of the binary (non-null for every
                                # code=6 site observed in FNAF 1).
                                # Paired with a code=22 expression that
                                # holds the channel number (no bearing
                                # on the sample-name verifier).
                                "num_name": "PlayChannelSample",
                                "parameters": [
                                    {
                                        "code": 6,
                                        "code_name": "SAMPLE",
                                        "handle": 11,
                                        "name": "deep steps",
                                    },
                                    {
                                        "code": 22,
                                        "code_name": "EXP",
                                        "expr_str": "1",
                                    },
                                ],
                            },
                            {
                                # ExtAction_87 shape: code=45 string
                                # literal arrives wrapped in single
                                # quotes (`"'lives'"`) that the tidier
                                # pseudo drops. Paired with a code=22
                                # identifier-chain that already matches
                                # verbatim.
                                "num_name": "ExtAction_87",
                                "parameters": [
                                    {
                                        "code": 45,
                                        "code_name": "EXPSTR",
                                        "expr_str": "'lives'",
                                    },
                                    {
                                        "code": 22,
                                        "code_name": "EXP",
                                        "expr_str": "lives left.CounterValue",
                                    },
                                ],
                            },
                            {
                                # Negative-control sample row: code=6
                                # name that a hostile pseudo might try
                                # to brush past under a naive substring
                                # check. Used to prove the sample-name
                                # verifier is word-boundary guarded.
                                "num_name": "PlayChannelSample",
                                "parameters": [
                                    {
                                        "code": 6,
                                        "code_name": "SAMPLE",
                                        "handle": 99,
                                        "name": "blip",
                                    },
                                    {
                                        "code": 22,
                                        "code_name": "EXP",
                                        "expr_str": "1",
                                    },
                                ],
                            },
                        ],
                    }
                ],
            },
        ],
    }


@pytest.fixture
def combined_json(tmp_path: Path) -> Path:
    path = tmp_path / "combined.json"
    path.write_text(json.dumps(_combined_doc()), encoding="utf-8")
    return path


# --- Loader -------------------------------------------------------------


def test_load_combined_keyed_by_four_tuple(combined_json: Path):
    """The loader flattens to a dict keyed by (frame, eg, type, coa),
    with conditions and actions indexed independently."""
    combined = _load_combined(combined_json)
    assert (0, 0, "condition", 0) in combined
    assert (0, 0, "action", 0) in combined
    assert (1, 0, "condition", 0) in combined
    assert (1, 0, "action", 0) in combined
    # Frame 2 adds two more action rows (no conditions in that frame).
    assert (2, 0, "action", 0) in combined
    assert (2, 0, "action", 1) in combined
    # Frame 3 adds one condition row + three action rows for V2
    # strategy coverage (sample-name + normalized-exprs).
    assert (3, 0, "condition", 0) in combined
    assert (3, 0, "action", 0) in combined
    assert (3, 0, "action", 1) in combined
    assert (3, 0, "action", 2) in combined
    assert len(combined) == 10


def test_load_combined_preserves_row_fields(combined_json: Path):
    combined = _load_combined(combined_json)
    cond_row = combined[(0, 0, "condition", 0)]
    assert cond_row["num_name"] == "CompareCounter"
    assert cond_row["parameters"][0]["expr_str"] == "Night + 1"


# --- Verification ladder -----------------------------------------------


def test_expr_str_match_accepts_verbatim(combined_json: Path):
    """Slice C happy path: pseudo_code is an exact copy of the cited
    param's expr_str — auto-accept via `expr_str_match`."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Alpha fades randomly between 50 and 150.",
        citations=[
            Citation(
                frame=0,
                event_group_index=0,
                type="action",
                cond_or_act_index=0,
                parameter_index=0,
            )
        ],
        pseudo_code="50 + Random 100",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert result.accepted
    assert result.outcomes[0].reason == "expr_str_match"


def test_expr_str_match_is_case_and_whitespace_insensitive(combined_json: Path):
    """Cosmetic whitespace + case differences shouldn't tank the match
    — the normaliser collapses them."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Irrelevant.",
        citations=[
            Citation(
                frame=0,
                event_group_index=0,
                type="action",
                cond_or_act_index=0,
            )
        ],
        pseudo_code="50   +    random 100",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert result.accepted
    assert result.outcomes[0].reason == "expr_str_match"


def test_expr_str_reverse_match_accepts_whole_action_pseudo(
    combined_json: Path,
):
    """Bucket A recovery: the line cook cites pi=0 (expr_str='2') but
    emits a whole-action pseudo `SetChannelVolume(2, 100)`. The forward
    substring check fails (big pseudo doesn't fit in tiny expr_str),
    but the cited param's expr_str IS a standalone token inside the
    pseudo — accept via `expr_str_reverse_match`."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="The Speaker channel volume is set to 100 for channel 2.",
        citations=[
            Citation(
                frame=2,
                event_group_index=0,
                type="action",
                cond_or_act_index=0,
                parameter_index=0,
            )
        ],
        pseudo_code="SetChannelVolume(2, 100)",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert result.accepted
    assert result.outcomes[0].reason == "expr_str_reverse_match"


def test_expr_str_reverse_match_word_boundary_blocks_digit_smoosh(
    combined_json: Path,
):
    """Word-boundary guard: expr_str='1' must NOT match inside larger
    numeric literals like '100'. The naïve substring would accept;
    `\\b1\\b` rejects. Absent this guard, the reverse strategy would
    silently rubber-stamp any pseudo that merely mentions a digit."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Counter value mentions 100 somewhere.",
        citations=[
            Citation(
                frame=2,
                event_group_index=0,
                type="action",
                cond_or_act_index=1,
                parameter_index=0,
            )
        ],
        pseudo_code="set counter = 100",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert not result.accepted
    assert result.outcomes[0].reason == "no_verification_strategy_matched"


def test_all_exprs_in_pseudo_accepts_when_every_expr_is_token(
    combined_json: Path,
):
    """Bucket B recovery: pi=None citation at a multi-param action
    where both expression params' expr_strs ('2' and '100') appear as
    tokens inside the whole-action pseudo `SetChannelVolume(2, 100)`.
    No single expr_str contains the pseudo (forward fails), but the
    pseudo faithfully covers every expression param → accept via
    `all_exprs_in_pseudo`."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Speaker channel 2 volume set to 100.",
        citations=[
            Citation(
                frame=2,
                event_group_index=0,
                type="action",
                cond_or_act_index=0,
                # Deliberately pi=None — whole-action citation.
            )
        ],
        pseudo_code="SetChannelVolume(2, 100)",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert result.accepted
    assert result.outcomes[0].reason == "all_exprs_in_pseudo"


def test_all_exprs_in_pseudo_rejects_when_one_expr_missing(
    combined_json: Path,
):
    """The all-exprs strategy is strict: if even one expression param's
    expr_str is absent from the pseudo, reject. Prevents a half-truthful
    claim from sneaking through when the pseudo only mentions some of
    the call's arguments."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Partial claim missing the '100' literal.",
        citations=[
            Citation(
                frame=2,
                event_group_index=0,
                type="action",
                cond_or_act_index=0,
            )
        ],
        # Pseudo only mentions '2' — '100' is absent, so the all-exprs
        # gate holds.
        pseudo_code="SetChannelVolume(2, something_else)",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert not result.accepted
    assert result.outcomes[0].reason == "no_verification_strategy_matched"


def test_normalized_exprs_in_pseudo_strips_endparenthesis(
    combined_json: Path,
):
    """V2 straggler: the decompiler's flat expr_str is
    `'Random 4 EndParenthesis'` but the tidier pseudo is
    `Random(4) == 1`. Rung 5 (raw all-exprs) can't match because the
    whole expr_str isn't present verbatim. Rung 6 fires because
    `EndParenthesis` triggers the artifact gate, then tokenises to
    `['Random', '4']` — both standalone tokens in the pseudo, second
    expr_str `'1'` matches verbatim too."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Fires with 1-in-4 odds.",
        citations=[
            Citation(
                frame=3,
                event_group_index=0,
                type="condition",
                cond_or_act_index=0,
                # pi=None — whole-condition claim.
            )
        ],
        pseudo_code="Random(4) == 1",
        kind="numeric_comparison",
    )
    result = verify_record(combined, rec)
    assert result.accepted
    assert result.outcomes[0].reason == "normalized_exprs_in_pseudo"


def test_normalized_exprs_in_pseudo_strips_surrounding_quotes(
    combined_json: Path,
):
    """ExtAction_87 edge case: code=45 string literal `"'lives'"` is
    quote-wrapped in the flat rendering but the pseudo mentions
    `lives` unquoted. Raw all-exprs fails (quotes don't align); rung 6
    strips the outer quotes, tokenises to `['lives']`, finds it as a
    standalone word in `lives = lives left.CounterValue`. The second
    expr_str `lives left.CounterValue` already appears verbatim."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="The 'lives' alterable mirrors the counter.",
        citations=[
            Citation(
                frame=3,
                event_group_index=0,
                type="action",
                cond_or_act_index=1,
            )
        ],
        pseudo_code="lives = lives left.CounterValue",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert result.accepted
    assert result.outcomes[0].reason == "normalized_exprs_in_pseudo"


def test_normalized_exprs_in_pseudo_requires_artifact_marker(
    combined_json: Path,
):
    """Guard rail: the normalized-exprs rung must NOT fire when every
    expr_str is clean. A pseudo that merely happens to contain stray
    word groups from multiple clean expr_strs shouldn't be rubber-
    stamped — rung 6 only compensates for known decompiler artifacts,
    not for everything that fails rung 5.

    Frame 2's SetChannelVolume row has expr_strs `'2'` and `'100'`
    (no `EndParenthesis`, no surrounding quotes). A pseudo that drops
    the `100` should stay quarantined, proving the artifact gate
    fires before the loose tokenisation runs.
    """
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Clean expr_strs, no artifact, loose match must not fire.",
        citations=[
            Citation(
                frame=2,
                event_group_index=0,
                type="action",
                cond_or_act_index=0,
            )
        ],
        # Pseudo mentions `2` but drops `100` — rung 5 rejects, rung 6
        # must also reject because no expr_str has an artifact marker.
        pseudo_code="SetChannelVolume(2, other)",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert not result.accepted
    assert result.outcomes[0].reason == "no_verification_strategy_matched"


def test_normalized_exprs_in_pseudo_rejects_when_word_missing(
    combined_json: Path,
):
    """Artifact triggers rung 6 but one surviving word is absent from
    the pseudo → reject. Prevents a pseudo that strips the
    `EndParenthesis` artifact from squeezing through without actually
    naming every sub-token of the expression."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Claim omits the literal 4 from the Random call.",
        citations=[
            Citation(
                frame=3,
                event_group_index=0,
                type="condition",
                cond_or_act_index=0,
            )
        ],
        # The expr_str is `'Random 4 EndParenthesis'` + `'1'`. After
        # artifact stripping the words are `['Random', '4']` and `['1']`.
        # Pseudo drops the `4` — rung 6 must reject.
        pseudo_code="Random(N) == 1",
        kind="numeric_comparison",
    )
    result = verify_record(combined, rec)
    assert not result.accepted
    assert result.outcomes[0].reason == "no_verification_strategy_matched"


def test_sample_name_in_pseudo_accepts(combined_json: Path):
    """V2 Strategy A: the code=6 Sample param carries a decoded `name`
    field (surfaced directly from the UTF-16 binary). When the pseudo
    mentions that name as a standalone token, auto-accept with reason
    `sample_name_in_pseudo`. Covers the `PlayChannelSample` family
    where code=6 has no expr_str to match against."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Plays the 'deep steps' sample on channel 1.",
        citations=[
            Citation(
                frame=3,
                event_group_index=0,
                type="action",
                cond_or_act_index=0,
                parameter_index=0,
            )
        ],
        pseudo_code="PlayChannelSample(sample='deep steps')",
        kind="other",
    )
    result = verify_record(combined, rec)
    assert result.accepted
    assert result.outcomes[0].reason == "sample_name_in_pseudo"


def test_sample_name_in_pseudo_accepts_without_parameter_index(
    combined_json: Path,
):
    """The sample-name rung must also fire when the citation is
    whole-action (pi=None). Scans every param on the row, picks the
    code=6 one, runs the token check against its `name`."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Plays 'deep steps' whenever the scripted trigger fires.",
        citations=[
            Citation(
                frame=3,
                event_group_index=0,
                type="action",
                cond_or_act_index=0,
                # pi=None — whole-action claim.
            )
        ],
        pseudo_code="PlayChannelSample(sample='deep steps')",
        kind="other",
    )
    result = verify_record(combined, rec)
    assert result.accepted
    assert result.outcomes[0].reason == "sample_name_in_pseudo"


def test_sample_name_in_pseudo_word_boundary_rejects_partial_match(
    combined_json: Path,
):
    """Word-boundary discipline: a sample named `blip` must NOT match
    inside a pseudo that mentions `blipper`. Without `\\b` guards the
    verifier would rubber-stamp any hallucinated claim whose sample
    name is a prefix of a real one — a realistic LM failure mode
    worth guarding against."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="A 'blipper' sample plays — not the one in the row.",
        citations=[
            Citation(
                frame=3,
                event_group_index=0,
                type="action",
                cond_or_act_index=2,
                parameter_index=0,
            )
        ],
        # Pseudo mentions `blipper` but the row's code=6 name is `blip`
        # — word-boundary guard must reject.
        pseudo_code="PlayChannelSample(sample='blipper')",
        kind="other",
    )
    result = verify_record(combined, rec)
    assert not result.accepted
    assert result.outcomes[0].reason == "no_verification_strategy_matched"


def test_sample_name_in_pseudo_rejects_when_name_absent(
    combined_json: Path,
):
    """Sanity: when the row carries a code=6 name but the pseudo
    doesn't mention it at all, fall through to quarantine. This is
    the bare-miss case the deterministic gate is supposed to catch."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Plays something unrelated.",
        citations=[
            Citation(
                frame=3,
                event_group_index=0,
                type="action",
                cond_or_act_index=0,
                parameter_index=0,
            )
        ],
        # Pseudo doesn't mention `deep steps` — no other rung fires
        # for a code=6 param either, so this falls all the way through.
        pseudo_code="PlayChannelSample(sample='something_else_entirely')",
        kind="other",
    )
    result = verify_record(combined, rec)
    assert not result.accepted
    assert result.outcomes[0].reason == "no_verification_strategy_matched"


def test_short_label_match_accepts_quoted_label(combined_json: Path):
    """When a SHORT param's decoded label appears in the claim, we
    auto-accept via `short_label_match`."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Freddy's animation transitions to 'walk' state.",
        citations=[
            Citation(
                frame=1,
                event_group_index=0,
                type="action",
                cond_or_act_index=0,
                parameter_index=0,
            )
        ],
        pseudo_code="set anim_sequence(target=walk)",
        kind="state_transition",
    )
    result = verify_record(combined, rec)
    assert result.accepted
    assert result.outcomes[0].reason == "short_label_match"


def test_num_name_match_only_for_event_trigger_kinds(combined_json: Path):
    """A weak num_name-only signal is accepted for `event_trigger` /
    `state_transition` kinds, rejected for numeric kinds where
    expr_str is the operative gate."""
    combined = _load_combined(combined_json)
    # event_trigger kind -> accepted via num_name match
    good = InvariantRecord(
        claim="A KeyPressed event fires the group.",
        citations=[
            Citation(
                frame=1,
                event_group_index=0,
                type="condition",
                cond_or_act_index=0,
            )
        ],
        pseudo_code="KeyPressed -> trigger",
        kind="event_trigger",
    )
    assert verify_record(combined, good).outcomes[0].reason == "num_name_match"

    # numeric_assignment -> num_name alone is NOT enough
    weak = InvariantRecord(
        claim="Irrelevant KeyPressed claim.",
        citations=[
            Citation(
                frame=1,
                event_group_index=0,
                type="condition",
                cond_or_act_index=0,
            )
        ],
        pseudo_code="KeyPressed",
        kind="numeric_assignment",
    )
    assert verify_record(combined, weak).outcomes[0].reason == (
        "no_verification_strategy_matched"
    )


def test_coordinate_out_of_range_quarantined(combined_json: Path):
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Nonsense coordinate claim.",
        citations=[
            Citation(
                frame=99,
                event_group_index=99,
                type="condition",
                cond_or_act_index=99,
            )
        ],
        pseudo_code="x + 1",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert not result.accepted
    assert result.outcomes[0].reason == "citation_coordinate_out_of_range"


def test_parameter_index_out_of_range_quarantined(combined_json: Path):
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Parameter index past end of params list.",
        citations=[
            Citation(
                frame=0,
                event_group_index=0,
                type="condition",
                cond_or_act_index=0,
                parameter_index=99,
            )
        ],
        pseudo_code="Night + 1",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert not result.accepted
    assert result.outcomes[0].reason == "parameter_index_out_of_range"


def test_record_with_mixed_citations_rejected(combined_json: Path):
    """One good citation + one bad citation → the record goes to
    quarantine. A record is accepted iff *every* citation passes."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Night + 1 is the threshold.",
        citations=[
            Citation(
                frame=0,
                event_group_index=0,
                type="condition",
                cond_or_act_index=0,
            ),
            Citation(
                frame=99,
                event_group_index=99,
                type="condition",
                cond_or_act_index=99,
            ),
        ],
        pseudo_code="Night + 1",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert not result.accepted
    reasons = [o.reason for o in result.outcomes]
    assert "expr_str_match" in reasons
    assert "citation_coordinate_out_of_range" in reasons


# --- Stream writer -----------------------------------------------------


def test_check_records_writes_both_artefacts(combined_json: Path, tmp_path: Path):
    """End-to-end: mixed-quality records stream through
    `check_records` and land in accepted/quarantine JSONL with correct
    counts."""
    accepted_path = tmp_path / "accepted.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    records = [
        InvariantRecord(
            claim="Night + 1 advances the counter.",
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
        InvariantRecord(
            claim="Bad claim with bad coords.",
            citations=[
                Citation(
                    frame=99,
                    event_group_index=99,
                    type="condition",
                    cond_or_act_index=99,
                )
            ],
            pseudo_code="irrelevant",
            kind="numeric_assignment",
        ),
    ]
    accepted_count, quarantined_count = check_records(
        records, combined_json, accepted_path, quarantine_path
    )
    assert accepted_count == 1
    assert quarantined_count == 1
    assert accepted_path.exists()
    assert quarantine_path.exists()
    acc_lines = accepted_path.read_text().splitlines()
    q_lines = quarantine_path.read_text().splitlines()
    assert len(acc_lines) == 1
    assert len(q_lines) == 1
    # Every line is parseable JSON with the expected top-level keys
    for line in (*acc_lines, *q_lines):
        payload = json.loads(line)
        assert "record" in payload
        assert "outcomes" in payload


def test_check_records_overwrites_previous_run(
    combined_json: Path, tmp_path: Path
):
    """Output files are overwritten — the checker is idempotent per
    run, so a stale quarantine from a previous pipeline call doesn't
    silently inflate the rejection pile."""
    accepted_path = tmp_path / "accepted.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    accepted_path.write_text("STALE LINE\n", encoding="utf-8")
    quarantine_path.write_text("STALE LINE\n", encoding="utf-8")
    check_records([], combined_json, accepted_path, quarantine_path)
    assert accepted_path.read_text() == ""
    assert quarantine_path.read_text() == ""
