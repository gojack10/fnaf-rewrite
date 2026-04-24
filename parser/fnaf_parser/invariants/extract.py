"""Direct-LLM invariant extraction — session-10 shipping shape.

Reads `combined.jsonl`, groups rows by (frame, event_group_index),
greedy-accumulates groups into ~80K-token chunks, dispatches each chunk
to DeepSeek v4 Flash at `reasoning.effort=xhigh` with streaming +
30-second heartbeat observability, validates output against the inline
Pydantic `InvariantRecord` schema, and writes:

    <out_dir>/extracted.jsonl        — one InvariantRecord per line
    <out_dir>/quarantine_chunks.jsonl — per-chunk failure diagnostics

Why not the CLI
---------------

Runs via `python -m fnaf_parser.invariants.extract ...`. The CLI-level
`extract-invariants` subcommand was retired in the session-10 cleanup
along with the DSPy.RLM scaffolding; extraction is not part of the
data-pack parser's public surface.

Shipping parameters (session-10, Direct LLM Pipeline node)
----------------------------------------------------------

- Worker: `deepseek/deepseek-v4-flash` at `reasoning.effort=xhigh`.
- Chunk size: ~80K cl100k input tokens. Probe 0.8 passed at 100K but
  sat at 89 % of the 200K rot floor; 80K gives ~15K safety margin.
- Parallelism: 14-wide ThreadPool. At ~6 chunks this is one wave.
- Per-call timeout: none. Streaming backpressure is the liveness gate;
  heartbeat log separates "slow xhigh reasoning" from "hung".
- Prompt: v2 verbatim (copied from `parser/temp-probes/probe_0_6.py`
  and `probe_0_8_chunk_100k.py`). Do NOT edit without a fresh probe.

Output shape conversion (prompt v2 → InvariantRecord)
-----------------------------------------------------

Prompt v2 asks for `{frame, event_group_index, plain_english,
triggers[], effects[]}`. `InvariantRecord` is the Citation Checker's
contract — a different shape. We bridge in `item_to_record`:

    claim       = plain_english
    pseudo_code = plain_english  (same string; the literal-names-only
                                  sentence is the verification surface
                                  Citation Checker scans against row
                                  expr_str / label / num_name / sample
                                  content.)
    citations   = triggers mapped type="condition" + effects mapped
                  type="action", both keyed by the item's frame /
                  event_group_index, carrying cond_or_act_index only.
    kind        = "other"  (default; prompt v2 does not surface the
                            closed-set enum. The closed set is pinned
                            on the Rust Test Emission tree node. A
                            future refinement may route by num_name
                            heuristics or a separate tagging pass.)

Failure handling
----------------

Each chunk is its own atomic unit. Model error, JSON parse failure,
and per-item Pydantic validation errors are caught and routed to
`quarantine_chunks.jsonl` with a stack trace; the rest of the chunks
keep running. Partial runs are a feature — one bad chunk never sinks
the shipping artefact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import tiktoken
from openai import APIStatusError, OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator


# --- Record schema (inline copy of citation_checker.Citation / InvariantRecord)
#
# Inline so the extractor has no import dependency on the checker; both
# enforce the same five-tuple citation shape and the same
# whitespace-strip validator independently. If the schemas drift, the
# first shipping run surfaces it as a Pydantic mismatch at the checker
# boundary — a loud failure, not silent corruption.


class Citation(BaseModel):
    frame: int = Field(ge=0)
    event_group_index: int = Field(ge=0)
    type: Literal["condition", "action"]
    cond_or_act_index: int = Field(ge=0)
    parameter_index: int | None = Field(default=None, ge=0)


class InvariantRecord(BaseModel):
    claim: str = Field(min_length=4)
    citations: list[Citation] = Field(min_length=1)
    pseudo_code: str = Field(min_length=1)
    kind: Literal[
        "numeric_assignment",
        "numeric_comparison",
        "state_transition",
        "event_trigger",
        "other",
    ]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("claim", "pseudo_code")
    @classmethod
    def _strip_whitespace(cls, v: str) -> str:
        return v.strip()


# --- Constants -----------------------------------------------------------

MODEL = "deepseek/deepseek-v4-flash"
REASONING = {"effort": "xhigh"}
TEMPERATURE = 0
TARGET_INPUT_TOKENS = 80_000
MAX_TOKENS = 100_000
HEARTBEAT_SECONDS = 30
MAX_WORKERS = 14

# Prompt v2 — verbatim from probe_0_6.py / probe_0_8_chunk_100k.py.
# Do not edit without a fresh probe; the tree's "clean at 103/103 items,
# 0 forbidden hits, 0 role hits" empirical claim depends on this
# specific wording.
PROMPT_V2_TEMPLATE = """You are reading game logic from Five Nights at Freddy's 1, built in Clickteam Fusion. Each record below is ONE condition or ONE action inside an event group. An event group is ONE RULE of the game:

    WHEN all conditions in the group are true, DO all actions in the group.

Rows sharing the same `frame` and `event_group_index` belong to the same rule. Conditions come first (type="condition"), then actions (type="action"), ordered by `cond_or_act_index`.

The chunk below contains {n_groups} COMPLETE rules — every condition and every action for each rule is present.

CRITICAL CONSTRAINT — literal naming only:

Use ONLY the LITERAL object names, operation names, and value names that appear in the records. Do NOT infer what any object, value, string, or counter MEANS or DOES in the game. An opaque description is preferable to a wrong description.

- If the record says `object_name: "Active 2"`, write "Active 2". Do NOT write "power" or "power meter".
- If the record says `object_name: "freddy got in"`, write "freddy got in". Do NOT write "jumpscare flag" or "jumpscare trigger".
- If the record says `object_name: "String"` with `num_name: "SetChannelVolume"`, write "String SetChannelVolume". Do NOT write "sound cue" or "audio channel".
- If the record has `num_name: "SubtractFromAlterable"`, write "SubtractFromAlterable". Do NOT write "decrement" or "drain".

If a rule's meaning cannot be described without inferring a role, describe the rule's MECHANICS ("when the condition on Active 2 fires, SetAlterableValue on freddy bear to 2") rather than its semantics ("when power is low, freddy wakes up").

For EACH event_group in the chunk, produce a JSON object:

- frame:                    (int)
- event_group_index:        (int)
- plain_english:            one-sentence MECHANICAL description using ONLY literal names from the records
- triggers:                 list of {{object, op, cond_or_act_index}} objects — the conditions that must be true
- effects:                  list of {{object, op, cond_or_act_index}} objects — the actions that fire

Output format: a JSON array with one object per event_group. Output ONLY the JSON array — no code fences, no prose.

Records:
{records}
"""


# --- Chunk building ------------------------------------------------------


def load_rows(jsonl_path: Path) -> list[dict]:
    """Stream `combined.jsonl` into a list of row dicts."""
    with jsonl_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def group_by_rule(
    rows: list[dict],
) -> "OrderedDict[tuple[int, int], list[dict]]":
    """Group rows by (frame, event_group_index) in first-seen order.

    Within each group, sort conditions-before-actions by
    `cond_or_act_index` so the model reads the rule top-to-bottom.
    """
    groups: "OrderedDict[tuple[int, int], list[dict]]" = OrderedDict()
    for row in rows:
        key = (row["frame"], row["event_group_index"])
        groups.setdefault(key, []).append(row)
    for rows_in_group in groups.values():
        rows_in_group.sort(
            key=lambda r: (
                0 if r["type"] == "condition" else 1,
                r["cond_or_act_index"],
            )
        )
    return groups


def accumulate_chunks(
    ordered_groups: "OrderedDict[tuple[int, int], list[dict]]",
    target_tokens: int,
    encoding: Any,
    overhead_tokens: int,
) -> list[tuple[list[dict], list[tuple[int, int]]]]:
    """Greedy-accumulate event_groups until each chunk hits target_tokens.

    Returns a list of `(rows, keys)` pairs where `rows` is the flat
    row list for that chunk and `keys` is its list of
    `(frame, event_group_index)` tuples in order.
    """
    chunks: list[tuple[list[dict], list[tuple[int, int]]]] = []
    current_rows: list[dict] = []
    current_keys: list[tuple[int, int]] = []
    current_blob_tokens = 0
    for key, group_rows in ordered_groups.items():
        blob = "\n".join(
            json.dumps(r, separators=(",", ":")) for r in group_rows
        )
        group_tokens = len(encoding.encode(blob))
        projected = (
            current_blob_tokens
            + group_tokens
            + overhead_tokens
            + (1 if current_rows else 0)
        )
        if current_keys and projected > target_tokens:
            chunks.append((current_rows, current_keys))
            current_rows = []
            current_keys = []
            current_blob_tokens = 0
        current_rows.extend(group_rows)
        current_keys.append(key)
        current_blob_tokens += group_tokens + 1
    if current_keys:
        chunks.append((current_rows, current_keys))
    return chunks


def build_prompt(chunk_rows: list[dict], n_groups: int) -> str:
    records_blob = "\n".join(
        json.dumps(r, separators=(",", ":")) for r in chunk_rows
    )
    return PROMPT_V2_TEMPLATE.format(n_groups=n_groups, records=records_blob)


# --- Model call with streaming + heartbeat -------------------------------


def run_model_streaming(
    client: OpenAI,
    prompt: str,
    label: str,
) -> dict:
    """Stream-dispatch one chunk. Returns status + usage + content.

    Heartbeat: every HEARTBEAT_SECONDS, prints elapsed + event_count +
    content_chars so an observer can tell reasoning-silent-alive from
    truly-dead. Session-10 probe 0.8 sat silent for 750 s before
    emitting the first content character — without this, the loop
    would blind-kill every run at the 5-min mark.
    """
    t0 = time.monotonic()
    content_parts: list[str] = []
    final_usage: dict | None = None
    event_count = 0
    last_heartbeat = t0
    finish_reason: str | None = None
    http_status: int | None = None
    print(f"[{label}] start: {MODEL} @ {REASONING}, stream=True", flush=True)
    try:
        raw_stream = client.chat.completions.with_raw_response.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            extra_body={"reasoning": REASONING},
            stream=True,
            stream_options={"include_usage": True},
        )
        http_status = raw_stream.http_response.status_code
        stream = raw_stream.parse()
        for event in stream:
            event_count += 1
            now = time.monotonic()
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                elapsed = now - t0
                char_count = sum(len(p) for p in content_parts)
                print(
                    f"[{label}] heartbeat: {elapsed:6.0f}s "
                    f"events={event_count:5d} content_chars={char_count}",
                    flush=True,
                )
                last_heartbeat = now
            if event.choices:
                choice = event.choices[0]
                delta = getattr(choice, "delta", None)
                if delta and getattr(delta, "content", None):
                    content_parts.append(delta.content)
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
            if getattr(event, "usage", None):
                final_usage = event.usage.model_dump()
        elapsed = time.monotonic() - t0
        content = "".join(content_parts)
        print(
            f"[{label}] stream closed {elapsed:.1f}s events={event_count} "
            f"content_chars={len(content)} finish={finish_reason}",
            flush=True,
        )
        return {
            "status": "ok",
            "http_status": http_status,
            "elapsed_s": round(elapsed, 2),
            "finish_reason": finish_reason,
            "event_count": event_count,
            "usage": final_usage or {},
            "content": content,
        }
    except APIStatusError as e:
        elapsed = time.monotonic() - t0
        print(
            f"[{label}] api_error after {elapsed:.1f}s: "
            f"http={getattr(e, 'status_code', None)}",
            flush=True,
        )
        return {
            "status": "api_error",
            "http_status": getattr(e, "status_code", None),
            "elapsed_s": round(elapsed, 2),
            "event_count": event_count,
            "body": e.response.text[:2000] if e.response is not None else None,
        }
    except Exception as e:  # noqa: BLE001 — surface everything; chunk quarantines
        elapsed = time.monotonic() - t0
        print(
            f"[{label}] exception after {elapsed:.1f}s: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return {
            "status": "exception",
            "elapsed_s": round(elapsed, 2),
            "event_count": event_count,
            "partial_content_chars": sum(len(p) for p in content_parts),
            "error": repr(e),
            "type": type(e).__name__,
        }


# --- Output post-processing ----------------------------------------------


def strip_fences(text: str) -> str:
    """Strip ```json ... ``` fences if the model added them despite
    the prompt instruction to the contrary (Gemma's historical habit;
    Flash mostly complies but the safety net is cheap)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.rsplit("```", 1)[0].strip()
    return stripped


def item_to_record(item: dict) -> InvariantRecord:
    """Convert one prompt-v2 output item to an `InvariantRecord`.

    Prompt v2 emits `{frame, event_group_index, plain_english,
    triggers[], effects[]}` where each trigger/effect is
    `{object, op, cond_or_act_index}`. We bridge to the Citation
    Checker's contract by mapping triggers → condition-type citations,
    effects → action-type citations, and using the plain-English
    sentence as both the `claim` and the `pseudo_code` (the
    literal-names-only rendering IS the verification surface the
    checker's `*_in_pseudo` rungs scan against row expr_strs, decoded
    sample names, and decoded SHORT labels).
    """
    frame = int(item["frame"])
    egi = int(item["event_group_index"])
    plain_english = str(item.get("plain_english", "")).strip()
    citations: list[Citation] = []
    for t in item.get("triggers") or []:
        citations.append(
            Citation(
                frame=frame,
                event_group_index=egi,
                type="condition",
                cond_or_act_index=int(t["cond_or_act_index"]),
            )
        )
    for e in item.get("effects") or []:
        citations.append(
            Citation(
                frame=frame,
                event_group_index=egi,
                type="action",
                cond_or_act_index=int(e["cond_or_act_index"]),
            )
        )
    if not citations:
        raise ValueError(
            f"item at frame={frame} eg={egi} has no triggers or effects"
        )
    return InvariantRecord(
        claim=plain_english,
        citations=citations,
        pseudo_code=plain_english,
        kind="other",
    )


def process_chunk(
    client: OpenAI,
    chunk_index: int,
    chunk_rows: list[dict],
    chunk_keys: list[tuple[int, int]],
    prompt_tokens: int,
) -> dict:
    """Run one chunk end-to-end and return records + diagnostics.

    Status is one of:
      - "ok": records extracted (may have per-item validation errors)
      - "model_error": API/network/stream failure; no records
      - "parse_error": content not JSON; no records
      - "shape_error": content not a list; no records
    """
    label = f"chunk-{chunk_index:02d}"
    n_groups = len(chunk_keys)
    first_key = chunk_keys[0]
    last_key = chunk_keys[-1]
    print(
        f"[{label}] launch: n_groups={n_groups} rows={len(chunk_rows)} "
        f"prompt_toks={prompt_tokens} span={first_key}..{last_key}",
        flush=True,
    )
    prompt = build_prompt(chunk_rows, n_groups)
    run = run_model_streaming(client, prompt, label)
    diag_run = {k: v for k, v in run.items() if k != "content"}
    diagnostics: dict = {
        "chunk_index": chunk_index,
        "n_groups": n_groups,
        "n_rows": len(chunk_rows),
        "prompt_cl100k_tokens": prompt_tokens,
        "first_key": first_key,
        "last_key": last_key,
        "run": diag_run,
    }
    if run["status"] != "ok":
        return {"status": "model_error", "records": [], "diagnostics": diagnostics}

    stripped = strip_fences(run["content"])
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        diagnostics["parse_error"] = str(e)
        diagnostics["raw_prefix"] = run["content"][:500]
        return {"status": "parse_error", "records": [], "diagnostics": diagnostics}
    if not isinstance(parsed, list):
        diagnostics["shape_error"] = f"root is {type(parsed).__name__}, not list"
        return {"status": "shape_error", "records": [], "diagnostics": diagnostics}

    records: list[InvariantRecord] = []
    item_errors: list[dict] = []
    for idx, item in enumerate(parsed):
        try:
            rec = item_to_record(item)
            records.append(rec)
        except (ValidationError, ValueError, KeyError, TypeError) as e:
            item_errors.append(
                {
                    "item_index": idx,
                    "error": repr(e),
                    "item": item,
                }
            )
    diagnostics["n_items_parsed"] = len(parsed)
    diagnostics["n_records_valid"] = len(records)
    diagnostics["item_errors"] = item_errors
    return {"status": "ok", "records": records, "diagnostics": diagnostics}


# --- Top-level entry point -----------------------------------------------


def run(
    jsonl_path: Path,
    out_dir: Path,
    target_tokens: int = TARGET_INPUT_TOKENS,
    max_workers: int = MAX_WORKERS,
    max_chunks: int | None = None,
    api_key: str | None = None,
) -> int:
    """Extract invariants end-to-end. Returns a process exit code."""
    api_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
    if not api_key:
        print(
            "ERROR: OPENROUTER_API_KEY not set (pass --api-key or export it)",
            file=sys.stderr,
        )
        return 3
    if not jsonl_path.exists():
        print(f"ERROR: combined.jsonl not found at {jsonl_path}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    extracted_path = out_dir / "extracted.jsonl"
    quarantine_path = out_dir / "quarantine_chunks.jsonl"

    enc = tiktoken.get_encoding("cl100k_base")
    overhead = len(enc.encode(PROMPT_V2_TEMPLATE.format(n_groups=0, records="")))

    rows = load_rows(jsonl_path)
    groups = group_by_rule(rows)
    chunks = accumulate_chunks(
        groups,
        target_tokens=target_tokens,
        encoding=enc,
        overhead_tokens=overhead,
    )
    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    print(f"=== extract.py ===")
    print(f"input:          {jsonl_path}")
    print(f"output dir:     {out_dir}")
    print(f"rows:           {len(rows):,}")
    print(f"event_groups:   {len(groups):,}")
    print(f"chunks:         {len(chunks)}  (target {target_tokens:,} cl100k each)")
    print(f"model:          {MODEL} @ {REASONING}")
    print(f"max_workers:    {max_workers}")
    print(f"heartbeat:      every {HEARTBEAT_SECONDS}s per chunk")
    if max_chunks is not None:
        print(f"max_chunks:     {max_chunks}  (dry-run cap)")
    print()

    # Pre-compute prompt token size per chunk for logging sanity.
    chunk_prompt_tokens: list[int] = []
    for rows_c, keys_c in chunks:
        pt = len(enc.encode(build_prompt(rows_c, len(keys_c))))
        chunk_prompt_tokens.append(pt)
    print("chunk sizing (prompt cl100k tokens):")
    for i, (pt, (_, keys)) in enumerate(zip(chunk_prompt_tokens, chunks)):
        print(f"  chunk-{i:02d}: {pt:>6,} toks / {len(keys):>4} groups "
              f"/ span {keys[0]}..{keys[-1]}")
    print()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    t_start = time.monotonic()
    all_records: list[InvariantRecord] = []
    all_quarantine: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                process_chunk,
                client,
                idx,
                rows_c,
                keys_c,
                chunk_prompt_tokens[idx],
            ): idx
            for idx, (rows_c, keys_c) in enumerate(chunks)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001
                all_quarantine.append(
                    {
                        "chunk_index": idx,
                        "status": "worker_exception",
                        "error": repr(e),
                        "type": type(e).__name__,
                    }
                )
                print(f"[chunk-{idx:02d}] worker exception: {e!r}", flush=True)
                continue
            if result["status"] == "ok":
                all_records.extend(result["records"])
                if result["diagnostics"].get("item_errors"):
                    # Some items inside an otherwise-OK chunk failed validation.
                    # Log them as partial-quarantine so the run still emits
                    # the good records while the failures stay inspectable.
                    all_quarantine.append(
                        {**result["diagnostics"], "status": "partial_items"}
                    )
            else:
                all_quarantine.append(
                    {**result["diagnostics"], "status": result["status"]}
                )

    elapsed = time.monotonic() - t_start

    with extracted_path.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(rec.model_dump_json(exclude_none=True) + "\n")
    with quarantine_path.open("w", encoding="utf-8") as fh:
        for q in all_quarantine:
            fh.write(json.dumps(q, default=str) + "\n")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"wall-clock:        {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"chunks ok:         {len(chunks) - sum(1 for q in all_quarantine if q['status'] not in ('partial_items',))} / {len(chunks)}")
    print(f"records extracted: {len(all_records):,}")
    print(f"quarantined:       {len(all_quarantine)}")
    print(f"extracted:         {extracted_path}")
    print(f"quarantine:        {quarantine_path}")
    chunk_failures = [
        q for q in all_quarantine if q["status"] != "partial_items"
    ]
    if chunk_failures:
        print(f"\n  {len(chunk_failures)} whole-chunk failure(s):")
        for q in chunk_failures:
            print(f"    chunk-{q.get('chunk_index'):02d}: {q['status']}")
    return 0 if not chunk_failures else 1


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_jsonl = repo_root / "out" / "algorithm" / "combined.jsonl"
    default_out = repo_root / "out" / "invariants"

    ap = argparse.ArgumentParser(
        prog="python -m fnaf_parser.invariants.extract",
        description=(
            "Direct-LLM invariant extraction via DeepSeek v4 Flash at "
            "reasoning.effort=xhigh. Reads combined.jsonl, writes "
            "extracted.jsonl + quarantine_chunks.jsonl. Requires "
            "OPENROUTER_API_KEY."
        ),
    )
    ap.add_argument("--jsonl", type=Path, default=default_jsonl)
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--target-tokens", type=int, default=TARGET_INPUT_TOKENS)
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    ap.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Dry-run cap: process at most N chunks. Useful for "
        "first-run smoke tests before burning the full budget.",
    )
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args(argv)
    return run(
        jsonl_path=args.jsonl,
        out_dir=args.out,
        target_tokens=args.target_tokens,
        max_workers=args.max_workers,
        max_chunks=args.max_chunks,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    raise SystemExit(main())
