"""Algorithm emission entry point — the `dump-algorithm` pipeline.

Output Emission takes every parsing stage built in earlier Algorithm
Extraction work units (Name Tables Port, Probe #4.13 EventParameter
payload decode, Name Resolver) and stitches them into the CLI's
`dump-algorithm` seam:

    load_pack (pipeline.py)
      → 0x2229 FrameItems → `decode_frame_items` → handle→name map
      → every 0x3333 Frame record → `decode_frame` → `Frame.events`
      → walk every EventParameter → `decode_event_parameter`
      → `resolve_frame_events` (Name Resolver) stamps *_name fields
      → inject `object_name` on every cond/action + object-referencing
        expression (looked up in the pack-level FrameItems handle map)
      → render `expr_str` for every ExpressionParameter
      → flatten to one row per condition-or-action
      → write 4 artefacts under `<out>/algorithm/`:
          - `frames/frame_NN_<slug>.json` (pretty per-frame)
          - `combined.json` (pretty, all frames concatenated)
          - `combined.jsonl` (one row per condition / action)
          - `manifest.json` (SHA-256 + size for every file)

Loud-failure posture (inherited from Name Resolver):

- Unknown `object_info` handle → `AlgorithmEmitError`.
- Unknown parameter code / Name Tables gap → propagated from the
  underlying decoder/resolver.
- Sentinel `0xFFFF` is the documented "no-target" handle and resolves
  cleanly to `object_name=None` (not an error).

The canonical record is the structured `parameters` / expression `ast`.
`expr_str` is a deterministic projection for LLM readability; downstream
consumers should parse `ast` when semantic fidelity matters.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from fnaf_parser.algorithm.expr_str import render_expression_stream
from fnaf_parser.algorithm.name_resolver import resolve_frame_events
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.event_parameters import decode_event_parameter
from fnaf_parser.decoders.frame import decode_frame
from fnaf_parser.decoders.frame_events import (
    EventAction,
    EventCondition,
    EventGroup,
    EventParameter,
    FrameEvents,
)
from fnaf_parser.decoders.frame_items import decode_frame_items
from fnaf_parser.pipeline import load_pack

# --- Constants -----------------------------------------------------------

#: Chunk IDs consumed by this pipeline. Defined locally so `grep 0x2229`
#: finds every user, and so the module is self-contained w.r.t. the outer
#: chunk walker.
_CHUNK_ID_FRAME_ITEMS = 0x2229
_CHUNK_ID_FRAME = 0x3333

#: "No-target" sentinel for condition/action `object_info`. Matches
#: Anaconda's `0xFFFF` / CTFAK2.0's `-1` convention — a condition that
#: applies to the event-system generically rather than a specific object.
_OBJECT_INFO_NO_TARGET: int = 0xFFFF

#: Semantic version of this decoder. Surfaced in `manifest.json` so
#: downstream tooling (Algorithm Snapshot Antibody, LLM-driven invariant
#: extraction) can gate on the shape. Bump on any wire-level change to
#: the emitted schema.
DECODER_VERSION: str = "0.1.0"

class AlgorithmEmitError(ValueError):
    """Raised when the algorithm-extraction pipeline sees a wire-level
    or invariant violation that Name Resolver / decoders didn't already
    catch. Typical trigger: a condition / action `object_info` handle
    that isn't in the pack's 0x2229 FrameItems bank (excluding the
    documented `0xFFFF` no-target sentinel)."""


# --- Decoded-event-dict builders ---------------------------------------


def _parameter_to_decoded_dict(
    param: EventParameter, *, unicode: bool
) -> dict[str, Any]:
    """One EventParameter → decoded payload dict.

    Routes through `decode_event_parameter`, which dispatches on `code`
    to the right structured decoder. The returned dict already carries
    the `code` field (stamped by `decode_event_parameter`) and a stable
    `kind` string for every parameter shape (Short/Int/Key/Sample/Time/
    Position/Create/Click/Object/ExpressionParameter).
    """
    return decode_event_parameter(param.code, param.data, unicode=unicode)


def _condition_to_decoded_dict(
    cond: EventCondition, *, unicode: bool
) -> dict[str, Any]:
    return {
        "size": cond.size,
        "object_type": cond.object_type,
        "num": cond.num,
        "object_info": cond.object_info,
        "object_info_list": cond.object_info_list,
        "flags": cond.flags,
        "other_flags": cond.other_flags,
        "def_type": cond.def_type,
        "identifier": cond.identifier,
        "parameters": [
            _parameter_to_decoded_dict(p, unicode=unicode)
            for p in cond.parameters
        ],
    }


def _action_to_decoded_dict(
    act: EventAction, *, unicode: bool
) -> dict[str, Any]:
    return {
        "size": act.size,
        "object_type": act.object_type,
        "num": act.num,
        "object_info": act.object_info,
        "object_info_list": act.object_info_list,
        "flags": act.flags,
        "other_flags": act.other_flags,
        "def_type": act.def_type,
        "parameters": [
            _parameter_to_decoded_dict(p, unicode=unicode)
            for p in act.parameters
        ],
    }


def _group_to_decoded_dict(
    group: EventGroup, *, unicode: bool
) -> dict[str, Any]:
    return {
        "size": group.size,
        "flags": group.flags,
        "is_restricted": group.is_restricted,
        "restrict_cpt": group.restrict_cpt,
        "conditions": [
            _condition_to_decoded_dict(c, unicode=unicode)
            for c in group.conditions
        ],
        "actions": [
            _action_to_decoded_dict(a, unicode=unicode)
            for a in group.actions
        ],
    }


def _frame_events_to_decoded_dict(
    fe: FrameEvents, *, unicode: bool
) -> dict[str, Any]:
    """FrameEvents dataclass → pre-resolution decoded dict.

    This is the shape Name Resolver's `resolve_frame_events` expects:
    a mutable dict tree with numeric IDs in every slot. Running the
    resolver on this returns a *new* tree with `*_name` fields injected
    alongside every resolvable ID.
    """
    return {
        "max_objects": fe.max_objects,
        "max_object_info": fe.max_object_info,
        "num_players": fe.num_players,
        "number_of_conditions": list(fe.number_of_conditions),
        "qualifiers": [q.as_dict() for q in fe.qualifiers],
        "event_groups": [
            _group_to_decoded_dict(g, unicode=unicode)
            for g in fe.event_groups
        ],
        "extension_data_len": len(fe.extension_data),
        "parameter_codes_seen": sorted(fe.parameter_codes_seen),
    }


# --- Object-name injection & expr_str rendering -------------------------


def _resolve_handle(
    handle: int, *, handles_to_names: dict[int, str | None], where: str
) -> str | None:
    """Handle → object name. Loud-fails on unknown handles (≠ sentinel)."""
    if handle == _OBJECT_INFO_NO_TARGET:
        return None
    if handle not in handles_to_names:
        raise AlgorithmEmitError(
            f"Unknown object_info handle={handle} at {where}: not in "
            f"0x2229 FrameItems bank (known handles: "
            f"{sorted(handles_to_names)[:8]}{'…' if len(handles_to_names) > 8 else ''})"
        )
    return handles_to_names[handle]


def _inject_expression_object_names(
    expressions: list[dict[str, Any]],
    *,
    handles_to_names: dict[int, str | None],
    where: str,
) -> list[dict[str, Any]]:
    """Walk a resolved expression list, stamping `object_name` on every
    object-referencing expression. Returns a new list; inputs untouched.
    """
    out: list[dict[str, Any]] = []
    for i, expr in enumerate(expressions):
        e = dict(expr)
        object_type = e["object_type"]
        oi = e.get("object_info")
        # Only object-referencing expressions carry an object_info; others
        # either won't have the key or will have it set to None.
        if (object_type >= 2 or object_type == -7) and oi is not None:
            e["object_name"] = _resolve_handle(
                oi, handles_to_names=handles_to_names, where=f"{where}/expr[{i}]"
            )
        out.append(e)
    return out


def _inject_parameter_names(
    parameters: list[dict[str, Any]],
    *,
    handles_to_names: dict[int, str | None],
    where: str,
) -> list[dict[str, Any]]:
    """Walk the resolved parameter list, injecting `object_name`s on
    ExpressionParameter sub-expressions and rendering `expr_str`.
    """
    out: list[dict[str, Any]] = []
    for i, param in enumerate(parameters):
        p = dict(param)
        if p.get("kind") == "ExpressionParameter":
            p["expressions"] = _inject_expression_object_names(
                p["expressions"],
                handles_to_names=handles_to_names,
                where=f"{where}/param[{i}]",
            )
            p["expr_str"] = render_expression_stream(p["expressions"])
        out.append(p)
    return out


def _inject_cond_or_act_names(
    record: dict[str, Any],
    *,
    handles_to_names: dict[int, str | None],
    where: str,
) -> dict[str, Any]:
    """Condition/Action dict → same dict with `object_name` + expression
    names injected. Returns a new dict; input untouched.
    """
    out = dict(record)
    out["object_name"] = _resolve_handle(
        record["object_info"], handles_to_names=handles_to_names, where=where
    )
    out["parameters"] = _inject_parameter_names(
        record.get("parameters", []),
        handles_to_names=handles_to_names,
        where=where,
    )
    return out


def _inject_frame_events_names(
    fe: dict[str, Any],
    *,
    handles_to_names: dict[int, str | None],
    where: str = "frame_events",
) -> dict[str, Any]:
    """Walk a resolved FrameEvents dict and inject object_names + expr_str.

    Runs after Name Resolver. Input is the tree Name Resolver returned
    (every numeric ID has its `*_name` stamped); output is the same tree
    with `object_name` on every cond/act and object-referencing
    expression, plus `expr_str` on every ExpressionParameter.
    """
    out = dict(fe)
    groups: list[dict[str, Any]] = []
    for gi, group in enumerate(fe.get("event_groups", [])):
        g = dict(group)
        g["conditions"] = [
            _inject_cond_or_act_names(
                c,
                handles_to_names=handles_to_names,
                where=f"{where}/grp[{gi}]/cond[{ci}]",
            )
            for ci, c in enumerate(group.get("conditions", []))
        ]
        g["actions"] = [
            _inject_cond_or_act_names(
                a,
                handles_to_names=handles_to_names,
                where=f"{where}/grp[{gi}]/act[{ai}]",
            )
            for ai, a in enumerate(group.get("actions", []))
        ]
        groups.append(g)
    out["event_groups"] = groups
    return out


# --- Flatten to JSONL rows ---------------------------------------------


def _flatten_frame_to_rows(
    fe: dict[str, Any], *, frame_index: int, frame_name: str
) -> list[dict[str, Any]]:
    """Project a resolved frame-events dict to one row per condition
    OR per action. Source citation fields are stamped on every row so
    downstream consumers can re-associate back to the frame graph.
    """
    rows: list[dict[str, Any]] = []
    for gi, group in enumerate(fe.get("event_groups", [])):
        for ci, cond in enumerate(group.get("conditions", [])):
            rows.append(
                _cond_or_act_row(
                    cond,
                    row_type="condition",
                    frame_index=frame_index,
                    frame_name=frame_name,
                    event_group_index=gi,
                    cond_or_act_index=ci,
                )
            )
        for ai, act in enumerate(group.get("actions", [])):
            rows.append(
                _cond_or_act_row(
                    act,
                    row_type="action",
                    frame_index=frame_index,
                    frame_name=frame_name,
                    event_group_index=gi,
                    cond_or_act_index=ai,
                )
            )
    return rows


def _cond_or_act_row(
    record: dict[str, Any],
    *,
    row_type: str,
    frame_index: int,
    frame_name: str,
    event_group_index: int,
    cond_or_act_index: int,
) -> dict[str, Any]:
    """Build one JSONL row with stable field ordering.

    Field ordering mirrors the SiftText Output Emission crystallization:
    source citation first (so humans / jq pipelines can filter by frame
    before reading the payload), then identity, then parameters.
    """
    return {
        "frame": frame_index,
        "frame_name": frame_name,
        "event_group_index": event_group_index,
        "type": row_type,
        "cond_or_act_index": cond_or_act_index,
        "object_type": record["object_type"],
        "object_type_name": record["object_type_name"],
        "num": record["num"],
        "num_name": record["num_name"],
        "object_info": record["object_info"],
        "object_name": record.get("object_name"),
        "object_info_list": record["object_info_list"],
        "identifier": record.get("identifier"),
        "flags": record["flags"],
        "other_flags": record["other_flags"],
        "def_type": record["def_type"],
        "parameters": record["parameters"],
    }


# --- Per-frame pretty JSON + slugging -----------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _frame_slug(name: str | None) -> str:
    """Stable `frame_NN_<slug>.json` slug: lowercase, `_`-separated."""
    if not name:
        return "unnamed"
    slug = _SLUG_RE.sub("_", name.lower()).strip("_")
    return slug or "unnamed"


def _frame_filename(frame_index: int, frame_name: str | None) -> str:
    return f"frame_{frame_index:02d}_{_frame_slug(frame_name)}.json"


# --- Artefact writer -----------------------------------------------------


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file. Streamed so this works on combined.jsonl too."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    """Pretty 2-space-indented JSON with trailing newline (POSIX)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """One JSON object per line, no trailing comma, trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# --- Public entry point -------------------------------------------------


def dump_algorithm(exe: Path, out_dir: Path) -> list[Path]:
    """Emit the algorithm-extraction artefact suite under `out_dir`.

    Pipeline:

    1. `load_pack` — PE walk + chunk walk + RC4 transform prime.
    2. Find the single 0x2229 FrameItems chunk and decode it. Build the
       pack-level `handle → name` map used to name every `object_info`
       reference in the event graph.
    3. For each 0x3333 Frame chunk (in wire order):
         a. `decode_frame` — walk sub-chunks, decrypt events payload.
         b. Build the pre-resolution decoded events dict, running
            `decode_event_parameter` on every EventParameter.
         c. `resolve_frame_events` — stamp `*_name` fields.
         d. Inject `object_name` + `expr_str` via the FrameItems map.
         e. Flatten to one row per condition / action (source citation
            stamped on every row).
         f. Write `<out>/algorithm/frames/frame_NN_<slug>.json`.
    4. Concatenate all rows, write `combined.jsonl` + `combined.json`.
    5. Hash every written file (SHA-256 + size) + the source `.exe`;
       write `manifest.json`.

    Returns every path written, in the same stable order the files were
    emitted so a snapshot antibody can zip the list against expected
    SHA-256 values for a deterministic regression floor.

    Raises
    ------
    AlgorithmEmitError
        On unknown object_info handles (excluding the 0xFFFF sentinel).
    fnaf_parser.algorithm.name_resolver.NameResolutionError
        On any unknown numeric ID (condition/action/expression/parameter
        code / object type) — Name Resolver's loud-fail contract.
    fnaf_parser.decoders.event_parameters.EventParameterDecodeError
        On malformed parameter payloads.
    fnaf_parser.decoders.frame.FrameDecodeError
        On malformed frame containers.
    fnaf_parser.decoders.frame_items.FrameItemsDecodeError
        On malformed FrameItems.
    """
    exe = Path(exe)
    out_dir = Path(out_dir)

    pack = load_pack(exe)
    unicode = pack.walk.header.unicode

    # --- Step 2: 0x2229 FrameItems → handle → name ---
    frame_items_rec = next(
        (r for r in pack.walk.records if r.id == _CHUNK_ID_FRAME_ITEMS), None
    )
    if frame_items_rec is None:
        raise AlgorithmEmitError(
            "0x2229 FrameItems chunk is missing; without the pack-level "
            "object bank we can't name `object_info` handles. FNAF 1 ships "
            "exactly one. Refusing to emit a partial artefact."
        )
    frame_items_payload = read_chunk_payload(
        pack.blob, frame_items_rec, transform=pack.transform
    )
    frame_items = decode_frame_items(
        frame_items_payload, unicode=unicode, transform=pack.transform
    )
    handles_to_names: dict[int, str | None] = {
        h: obj.name for h, obj in frame_items.by_handle.items()
    }

    # --- Step 3: walk every 0x3333 Frame ---
    algorithm_dir = out_dir / "algorithm"
    frames_dir = algorithm_dir / "frames"
    algorithm_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_records = [r for r in pack.walk.records if r.id == _CHUNK_ID_FRAME]
    per_frame_artefacts: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    written: list[Path] = []

    for frame_index, frame_rec in enumerate(frame_records):
        frame_payload = read_chunk_payload(
            pack.blob, frame_rec, transform=pack.transform
        )
        frame = decode_frame(
            frame_payload, unicode=unicode, transform=pack.transform
        )
        if frame.events is None:
            # FNAF 1: every frame carries a 0x333D FrameEvents sub-chunk.
            # Missing one is loudly anomalous — refuse to emit.
            raise AlgorithmEmitError(
                f"Frame {frame_index} ({frame.name!r}) has no 0x333D "
                f"FrameEvents sub-chunk — can't emit a frame artefact "
                f"without an event graph. deferred_encrypted="
                f"{[f'0x{r.id:04X}' for r in frame.deferred_encrypted]}"
            )

        decoded_events = _frame_events_to_decoded_dict(
            frame.events, unicode=unicode
        )
        resolved = resolve_frame_events(
            decoded_events, path=f"frame[{frame_index}]/events"
        )
        named = _inject_frame_events_names(
            resolved,
            handles_to_names=handles_to_names,
            where=f"frame[{frame_index}]/events",
        )

        frame_name = frame.name or ""
        frame_artefact = {
            "frame_index": frame_index,
            "frame_name": frame.name,
            "event_groups": named["event_groups"],
            "parameter_codes_seen": named["parameter_codes_seen"],
            "number_of_conditions": named["number_of_conditions"],
            "qualifiers": named["qualifiers"],
        }

        frame_path = frames_dir / _frame_filename(frame_index, frame.name)
        _write_json(frame_path, frame_artefact)
        written.append(frame_path)

        per_frame_artefacts.append(frame_artefact)
        all_rows.extend(
            _flatten_frame_to_rows(
                named, frame_index=frame_index, frame_name=frame_name
            )
        )

    # --- Step 4: combined JSON + JSONL ---
    combined_json_path = algorithm_dir / "combined.json"
    _write_json(
        combined_json_path,
        {
            "decoder_version": DECODER_VERSION,
            "source_file": exe.name,
            "frames": per_frame_artefacts,
        },
    )
    written.append(combined_json_path)

    combined_jsonl_path = algorithm_dir / "combined.jsonl"
    _write_jsonl(combined_jsonl_path, all_rows)
    written.append(combined_jsonl_path)

    # --- Step 5: manifest.json with SHA-256 + size per file ---
    source_sha256 = hashlib.sha256(pack.blob).hexdigest()
    total_event_groups = sum(
        len(f["event_groups"]) for f in per_frame_artefacts
    )
    total_cond_act_pairs = len(all_rows)

    files_manifest: dict[str, dict[str, Any]] = {}
    for path in written:
        rel = path.relative_to(algorithm_dir).as_posix()
        files_manifest[rel] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    manifest = {
        "decoder_version": DECODER_VERSION,
        "source_file": exe.name,
        "source_sha256": source_sha256,
        "frame_count": len(per_frame_artefacts),
        "total_event_groups": total_event_groups,
        "total_cond_act_pairs": total_cond_act_pairs,
        "files": files_manifest,
    }
    manifest_path = algorithm_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    written.append(manifest_path)

    return written
