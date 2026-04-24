"""Literal-Name Gate — enforce the Literal-Until-Proven doctrine.

Three-layer stack (see tree node Literal Until Proven):

  Layer 1  (deterministic vocab-aware scan — this module)
    Flag invariants whose `claim` uses ROLE_WORDS that are not present
    in the data-pack's literal vocabulary.
  Layer 2  (Flash-as-judge at xhigh — this module)
    For candidates that pass Layer 1, a cheap LLM call asks: "does
    this invariant infer game-level meaning, or describe only the
    mechanics using literal record names?"
  Layer 3  (agent-led co-survey — NO CODE)
    Deferred to the first real `extract.py` run. Agent buckets the
    accepted + quarantined piles into patterns, owner verdicts seed
    the Antibody Library on Literal Until Proven.

Session-10 refinement pinned in code
------------------------------------

Probe 0.8 used a Layer-1 logic that subtracted role_words matching any
WORD-TOKEN in the vocabulary. That over-subtracted `power` (token in a
compound like `power left`) and would have blinded the gate to genuine
"power drains" role inferences in production. The session-10 fix:

- **Multi-word role_words** (`right door`, `sound cue`) — subtract if
  the full phrase appears as a substring in any lowercased standalone
  `object_name` / `num_name`.
- **Single-word role_words** (`power`, `jumpscare`) — subtract ONLY if
  the word is ITSELF a standalone literal phrase. Never a
  word-in-compound match.

Tree node Literal Until Proven (session-10 append) carries the full
spec and the "why". This module IS the production implementation of
that spec. Probe 0.8's `build_vocab_words` + `word-in-vocab_words`
check is the known-buggy approach and is deliberately NOT lifted.

Layer 2 judge prompt is lifted verbatim from
`parser/temp-probes/probe_0_7_gate.py` (10/10 after tuning). Do NOT
edit without re-running probe 0.7 or a superseding test.

Output
------

  <out_dir>/accepted.jsonl        — records that passed both layers
  <out_dir>/quarantine_gate.jsonl — records that failed Layer 1 and/or
                                    Layer 2, with reasons

Separate from citation_checker's `accepted.jsonl` / `quarantine.jsonl`
by design — the two gates check orthogonal properties (structural vs.
semantic-literal) and quarantining from either side is informative on
its own.

Exit codes
----------

  0 — all records pass both layers
  1 — at least one record quarantined
  2 — input missing
  3 — OPENROUTER_API_KEY unset (only matters when Layer 2 is enabled)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from openai import APIStatusError, OpenAI


# --- Layer 1 seed lists (role words + forbidden words) ---

# Seed lists transcribed from probe_0_8_chunk_100k.py. Every entry is a
# word or phrase that, if it appears in `claim`, is either a hard
# forbidden (never legitimate without proof) or a role-word (often role
# inference but sometimes a real Clickteam object). The vocab filter
# subtracts per session-10 rule before scanning.

ROLE_WORDS_SEED = [
    "power", "meter", "drain", "decrement",
    "jumpscare", "scare",
    "door button", "left door", "right door",
    "camera feed", "camera view",
    "night", "morning", "6am", "6 am",
    "sound cue", "sound effect", "audio channel", "music",
    "death", "game over", "victory", "win",
    "ambient",
]

FORBIDDEN_WORDS_SEED = [
    "power", "meter", "drain", "decrement",
    "jumpscare", "sound cue", "victory", "death",
]

# Grace words: object names that look role-ish but ARE Clickteam
# literal object names in FNAF 1. Kept as a scan-time grace pool so
# first real runs surface the pattern rather than eat the hit silently.
GRACE_WORDS = ["office", "freddy", "bonnie", "chica", "foxy", "light"]


# --- Layer 2 configuration ---

JUDGE_MODEL = "deepseek/deepseek-v4-flash"
JUDGE_REASONING = {"effort": "xhigh"}
JUDGE_TEMPERATURE = 0
# Reasoning at xhigh + short JSON verdict needs headroom.
JUDGE_MAX_TOKENS = 4096
JUDGE_MAX_WORKERS = 14

# Verbatim from probe_0_7_gate.py (10/10 after tuning). Do NOT edit.
JUDGE_PROMPT = """You are a gate checker for game-logic invariants. Each invariant is a one-sentence description of a rule from Five Nights at Freddy's (a Clickteam Fusion game).

Your job: detect whether the invariant uses ROLE WORDS that infer game-level meaning, versus describing only the MECHANICS of the rule in the raw game-data vocabulary.

LITERAL (answer "no"):
- Uses operation names from the raw records: SetAlterableValue, SubtractFromAlterable, SetChannelVolume, AddToAlterable, Timer, Alterable Value A/B/C, Counter, Active 2, etc.
- Uses object names AS-IS — lowercased phrases like "freddy bear", "office light", "bonnie", "chica", "foxy", "freddy got in" are literal Clickteam object names from the records. Do NOT flag them.
- Describes what condition fires what action, without inferring what the rule MEANS.

ROLE-GUESSING (answer "yes"):
- Uses game-concept words NOT present in raw records: "power", "meter", "drain", "decrement", "jumpscare", "scare", "sound cue", "sound effect", "audio channel", "music", "door button", "camera feed", "victory", "win", "death", "game over", "night", "morning", "6am", "ambient".
- Explains what an object REPRESENTS — especially parentheticals like "(the power meter)", "(the jumpscare trigger)", "(the sound cue)".
- Describes behavior in game terms ("Freddy activates", "player wins", "blocks Bonnie from entering", "scene transitions to death").

Examples (calibration — follow this pattern):

Invariant: "When Active 2 on freddy bear is greater than 0, SetAlterableValue on freddy bear sets it to 2."
Answer: {"role_guessing": "no", "why": "Mechanical description using only literal names (Active 2, freddy bear, SetAlterableValue)."}

Invariant: "When Timer on office light reaches 300, SetAlterableValue on office light to 0."
Answer: {"role_guessing": "no", "why": "'office light' is a literal Clickteam object name; SetAlterableValue and Timer are operation names — no role inference."}

Invariant: "When power drops below a threshold, Freddy activates and walks to the office for a jumpscare."
Answer: {"role_guessing": "yes", "why": "Uses role words 'power' and 'jumpscare' to describe meaning rather than mechanics."}

Invariant: "Active 2 (the power meter) drains by 1 every tick when any door or light is on."
Answer: {"role_guessing": "yes", "why": "Parenthetical '(the power meter)' explains what Active 2 represents — exact shape of role guessing."}

Return strictly in this JSON shape with no code fences and nothing else:
{"role_guessing": "yes" or "no", "why": "<one short sentence>"}

Invariant: {invariant}"""


# --- Layer 1: deterministic vocab-aware scan -----------------------------


def build_vocab_phrases(combined_rows: Iterable[dict]) -> frozenset[str]:
    """Collect lowercased standalone `object_name` / `num_name` strings.

    This is the set single-word ROLE_WORDS are checked against for
    subtraction. The session-10 refinement explicitly drops the
    word-token variant that probe 0.8 used.
    """
    phrases: set[str] = set()
    for row in combined_rows:
        for key in ("object_name", "num_name"):
            val = row.get(key)
            if isinstance(val, str) and val:
                phrases.add(val.lower())
    return frozenset(phrases)


def filter_role_words(
    seed: list[str],
    vocab_phrases: frozenset[str],
) -> tuple[list[str], list[str]]:
    """Subtract role_words that are real literal names.

    - Multi-word seeds match as substrings inside vocab_phrases.
    - Single-word seeds match ONLY if the word is itself a standalone
      vocab phrase (i.e. `word in vocab_phrases`). NEVER a token match
      against word-fragments of compounds.

    Returns (kept, subtracted). `kept` is the effective role-word list
    the scanner runs against.
    """
    kept: list[str] = []
    subtracted: list[str] = []
    for word in seed:
        wl = word.lower()
        if " " in wl:
            match = any(wl in p for p in vocab_phrases)
        else:
            match = wl in vocab_phrases  # standalone only; session-10 fix
        (subtracted if match else kept).append(word)
    return kept, subtracted


def scan_text(
    text: str,
    role_words: list[str],
    forbidden_words: list[str],
    grace_words: list[str] = GRACE_WORDS,
) -> dict:
    """Word-boundary scan of `text` for role / forbidden / grace words.

    Returns a dict describing hit counts per category. `forbidden_hits`
    is a strict subset of `role_hits` — every forbidden is also a
    role, but not every role is forbidden.
    """
    lowered = text.lower()
    role_hits: dict[str, int] = {}
    for word in role_words:
        matches = re.findall(r"\b" + re.escape(word) + r"\b", lowered)
        if matches:
            role_hits[word] = len(matches)
    grace_hits: dict[str, int] = {}
    for word in grace_words:
        matches = re.findall(r"\b" + re.escape(word) + r"\b", lowered)
        if matches:
            grace_hits[word] = len(matches)
    forbidden_hits = {w: n for w, n in role_hits.items() if w in forbidden_words}
    return {
        "role_hits": role_hits,
        "forbidden_hits": forbidden_hits,
        "grace_hits": grace_hits,
        "total_role_hits": sum(role_hits.values()),
        "total_forbidden_hits": sum(forbidden_hits.values()),
    }


def layer1_verdict(
    record: dict,
    effective_role_words: list[str],
    effective_forbidden_words: list[str],
    role_budget: int,
) -> dict:
    """Run Layer 1 against one record's `claim`.

    Pass iff:
      - no forbidden-word hits
      - role-word hits <= role_budget  (default 2; handoff spec)
    """
    claim = str(record.get("claim", ""))
    scan = scan_text(claim, effective_role_words, effective_forbidden_words)
    forbidden_any = scan["total_forbidden_hits"] > 0
    role_exceeds = scan["total_role_hits"] > role_budget
    passed = not forbidden_any and not role_exceeds
    reason_bits: list[str] = []
    if forbidden_any:
        reason_bits.append(
            f"forbidden_hits={scan['forbidden_hits']}"
        )
    if role_exceeds:
        reason_bits.append(
            f"role_hits={scan['role_hits']} > budget={role_budget}"
        )
    return {
        "passed": passed,
        "scan": scan,
        "reason": "; ".join(reason_bits) if reason_bits else "layer1_pass",
    }


# --- Layer 2: Flash-as-judge ---------------------------------------------


def strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.rsplit("```", 1)[0].strip()
    return stripped


def layer2_verdict(
    client: OpenAI,
    claim: str,
) -> dict:
    """One Flash-as-judge call. Returns {passed, role_guessing, why, ...}.

    `passed` is the gate verdict: True iff the judge says
    role_guessing="no". Parse failures and API errors count as
    `passed=False` (fail-closed) with the failure reason captured.
    """
    prompt = JUDGE_PROMPT.replace("{invariant}", claim)
    t0 = time.monotonic()
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
            extra_body={"reasoning": JUDGE_REASONING},
        )
        elapsed = time.monotonic() - t0
        parsed = raw.parse()
        content = parsed.choices[0].message.content or ""
        usage = parsed.usage.model_dump() if parsed.usage else {}
        stripped = strip_fences(content)
        try:
            verdict = json.loads(stripped)
        except json.JSONDecodeError as e:
            return {
                "passed": False,
                "reason": f"parse_error: {e}",
                "raw_content": content,
                "elapsed_s": round(elapsed, 2),
                "usage": usage,
            }
        rg = None
        why = None
        if isinstance(verdict, dict):
            rg = verdict.get("role_guessing")
            why = verdict.get("why")
        if rg not in ("yes", "no"):
            return {
                "passed": False,
                "reason": f"judge_shape_error: role_guessing={rg!r}",
                "raw_content": content,
                "elapsed_s": round(elapsed, 2),
                "usage": usage,
            }
        return {
            "passed": rg == "no",
            "role_guessing": rg,
            "why": why,
            "elapsed_s": round(elapsed, 2),
            "usage": usage,
            "reason": "layer2_pass" if rg == "no" else f"role_guessing: {why}",
        }
    except APIStatusError as e:
        return {
            "passed": False,
            "reason": f"api_error: http={getattr(e, 'status_code', None)}",
            "elapsed_s": round(time.monotonic() - t0, 2),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "passed": False,
            "reason": f"exception: {type(e).__name__}: {e}",
            "elapsed_s": round(time.monotonic() - t0, 2),
        }


# --- Top-level runner ----------------------------------------------------


def run(
    extracted_jsonl: Path,
    combined_jsonl: Path,
    out_dir: Path,
    role_budget: int = 2,
    skip_layer2: bool = False,
    api_key: str | None = None,
    max_workers: int = JUDGE_MAX_WORKERS,
) -> int:
    """Run both Layer 1 and (optionally) Layer 2 on every record.

    Layer 3 is out of scope here — it is an agent-led co-survey pass
    run interactively after this script's outputs land.
    """
    for p in (extracted_jsonl, combined_jsonl):
        if not p.exists():
            print(f"ERROR: missing input {p}", file=sys.stderr)
            return 2

    api_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
    if not skip_layer2 and not api_key:
        print(
            "ERROR: OPENROUTER_API_KEY not set (use --skip-layer2 to run "
            "Layer 1 only)",
            file=sys.stderr,
        )
        return 3

    # --- Build vocab + effective role/forbidden lists ---
    combined_rows = [
        json.loads(line)
        for line in combined_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vocab_phrases = build_vocab_phrases(combined_rows)
    role_kept, role_subtracted = filter_role_words(ROLE_WORDS_SEED, vocab_phrases)
    forbidden_kept, forbidden_subtracted = filter_role_words(
        FORBIDDEN_WORDS_SEED, vocab_phrases
    )

    # --- Load extracted records ---
    records: list[dict] = [
        json.loads(line)
        for line in extracted_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print("=" * 68)
    print("Literal-Name Gate")
    print("=" * 68)
    print(f"records:           {len(records):,}")
    print(f"vocab_phrases:     {len(vocab_phrases):,}  (standalone-literal set)")
    print(f"role_words kept:   {len(role_kept):>2} / {len(ROLE_WORDS_SEED)}  "
          f"subtracted: {role_subtracted}")
    print(f"forbidden kept:    {len(forbidden_kept):>2} / {len(FORBIDDEN_WORDS_SEED)}  "
          f"subtracted: {forbidden_subtracted}")
    print(f"role budget:       {role_budget}")
    print(f"layer 2:           {'skipped' if skip_layer2 else 'flash @ xhigh'}")
    print()

    # --- Layer 1 pass ---
    layer1_results: list[dict] = []
    for rec in records:
        v = layer1_verdict(rec, role_kept, forbidden_kept, role_budget)
        layer1_results.append({"record": rec, "layer1": v})
    layer1_pass = [r for r in layer1_results if r["layer1"]["passed"]]
    layer1_fail = [r for r in layer1_results if not r["layer1"]["passed"]]
    print(f"Layer 1: {len(layer1_pass)}/{len(records)} passed "
          f"(forbidden/role-budget filter).")

    # --- Layer 2 pass (only on layer1_pass) ---
    accepted: list[dict] = []
    layer2_fail: list[dict] = []
    if skip_layer2:
        accepted = layer1_pass
    else:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        t_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(layer2_verdict, client, r["record"]["claim"]): r
                for r in layer1_pass
            }
            done = 0
            for fut in as_completed(futures):
                r = futures[fut]
                r["layer2"] = fut.result()
                done += 1
                if done % 50 == 0 or done == len(layer1_pass):
                    elapsed = time.monotonic() - t_start
                    print(f"  [layer2] {done}/{len(layer1_pass)} "
                          f"({elapsed:.1f}s elapsed)", flush=True)
                if r["layer2"]["passed"]:
                    accepted.append(r)
                else:
                    layer2_fail.append(r)
        elapsed = time.monotonic() - t_start
        print(f"Layer 2: {len(accepted)}/{len(layer1_pass)} passed ({elapsed:.1f}s).")

    quarantined = layer1_fail + layer2_fail

    # --- Write outputs ---
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = out_dir / "accepted.jsonl"
    quarantine_path = out_dir / "quarantine_gate.jsonl"
    with accepted_path.open("w", encoding="utf-8") as fh:
        for r in accepted:
            fh.write(json.dumps(r["record"]) + "\n")
    with quarantine_path.open("w", encoding="utf-8") as fh:
        for r in quarantined:
            fh.write(json.dumps(r, default=str) + "\n")

    # --- Summary ---
    print()
    print("=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"accepted:          {len(accepted):,}  -> {accepted_path}")
    print(f"quarantined:       {len(quarantined):,}  -> {quarantine_path}")
    print(f"  layer1 fail:     {len(layer1_fail)}")
    print(f"  layer2 fail:     {len(layer2_fail)}")

    # A few example failures so the operator can sanity-check the gate
    # shape on every run without opening the JSONL.
    if layer1_fail:
        print()
        print("first 5 Layer 1 failures:")
        for r in layer1_fail[:5]:
            print(f"  [{r['layer1']['reason']}]")
            print(f"    {r['record'].get('claim', '')[:140]!r}")
    if layer2_fail:
        print()
        print("first 5 Layer 2 failures:")
        for r in layer2_fail[:5]:
            reason = r["layer2"].get("reason", "")
            print(f"  [{reason[:100]}]")
            print(f"    {r['record'].get('claim', '')[:140]!r}")

    return 0 if not quarantined else 1


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_extracted = repo_root / "out" / "invariants" / "extracted.jsonl"
    default_combined = repo_root / "out" / "algorithm" / "combined.jsonl"
    default_out = repo_root / "out" / "invariants"

    ap = argparse.ArgumentParser(
        prog="python -m fnaf_parser.invariants.literal_gate",
        description=(
            "Literal-Name Gate: enforce the Literal-Until-Proven doctrine "
            "via a deterministic vocab-aware Layer 1 scan plus a Flash "
            "judge pass at Layer 2. Layer 3 is an agent-led co-survey "
            "run interactively after this script."
        ),
    )
    ap.add_argument("--extracted", type=Path, default=default_extracted)
    ap.add_argument("--combined", type=Path, default=default_combined)
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--role-budget", type=int, default=2)
    ap.add_argument(
        "--skip-layer2",
        action="store_true",
        help="Run Layer 1 only (no LLM calls, no API key required).",
    )
    ap.add_argument("--max-workers", type=int, default=JUDGE_MAX_WORKERS)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args(argv)
    return run(
        extracted_jsonl=args.extracted,
        combined_jsonl=args.combined,
        out_dir=args.out,
        role_budget=args.role_budget,
        skip_layer2=args.skip_layer2,
        api_key=args.api_key,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
