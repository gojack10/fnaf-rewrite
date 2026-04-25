"""Runtime-pack emission entry point — the ``dump-runtime-pack`` pipeline.

Emits the boundary-contract artefacts between the Python parser and the
Rust Clickteam runtime. V0 = JSON; compact binary only if loading is
ever slow enough to matter.

Output layout (under ``<out>/runtime_pack/``):

    frames_state/frame_NN_<slug>.json  × 17
        Per-frame scene-start state. Each file captures every field
        of the decoded ``Frame`` except ``events`` (which belongs to
        the algorithm/ pack), raw debug fields (``sub_records``,
        ``deferred_encrypted``), and any empty/absent fade blocks.

    manifest.json
        Master pack manifest. Covers every file in the pack
        (algorithm/, images/, audio/, runtime_pack/frames_state/,
        NOT self-referencing). Lists pack + decoder versions, source
        .exe + SHA-256, per-file paths + sizes + SHA-256, and summary
        counts.

Pre-requisites
--------------

Must be run AFTER ``dump-algorithm`` and ``dump-assets`` so the
``algorithm/``, ``images/``, and ``audio/`` directories exist under
``<out>/``. The manifest index is incomplete without them, and we'd
rather fail loudly than silently emit a partial pack. The CLI layer
refuses to proceed if they are missing.

Scope cuts
----------

- Does NOT decode ObjectCommon (the opaque ``properties_raw`` bytes on
  each FrameItems entry). Animation tables, alterable defaults,
  hotspots, and default movement live in those bytes. A future probe
  + decoder unlocks files 2 (animation tables) and 3 (per-object
  properties); see Runtime Pack Extraction node on the SiftText tree.
- Does NOT re-run the algorithm or asset emit — those are independent
  subcommands. The runtime pack is purely additive.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.frame import decode_frame
from fnaf_parser.pipeline import load_pack

# --- Constants ----------------------------------------------------------

#: 0x3333 Frame chunk id. Defined locally so ``grep 0x3333`` finds every
#: user and the module is self-contained w.r.t. the outer chunk walker.
_CHUNK_ID_FRAME = 0x3333

#: Semantic version of the runtime pack shape emitted by this module.
#: Bump on any wire-level change to the V0 schema (new fields are fine;
#: renamed or removed fields require a version bump). Surfaced in
#: ``manifest.json`` so Rust consumers can gate loader versions.
PACK_VERSION: str = "0.1.0"

#: Semantic version of the decoder chain producing this pack. Distinct
#: from PACK_VERSION so a consumer can distinguish "pack shape changed"
#: from "decoder internals changed but shape is stable".
DECODER_VERSION: str = "0.1.0"


class RuntimePackEmitError(ValueError):
    """Raised when the runtime-pack pipeline hits an invariant violation
    decoders didn't already catch. Typical triggers: missing
    pre-requisite directories, or manifest-time file reconciliation
    failures."""


# --- Per-frame serialization -------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _frame_slug(name: str | None) -> str:
    """Stable frame_NN_<slug>.json slug: lowercase, underscore-separated.

    Mirrors the slug logic in ``algorithm/emit.py`` so frame files in
    ``algorithm/frames/`` and ``runtime_pack/frames_state/`` share the
    same base name. Lets a downstream consumer cross-reference by slug
    without a separate lookup table.
    """
    if not name:
        return "unnamed"
    slug = _SLUG_RE.sub("_", name.lower()).strip("_")
    return slug or "unnamed"


def _frame_state_filename(frame_index: int, frame_name: str | None) -> str:
    return f"frame_{frame_index:02d}_{_frame_slug(frame_name)}.json"


def _frame_to_scene_start_dict(frame: Any, *, frame_index: int) -> dict[str, Any]:
    """Serialize a decoded Frame's non-events fields to a JSON-safe dict.

    Fields included (all from ``decode_frame()``):

    - ``frame_index`` (synthesized, ordinal position)
    - ``frame_name``
    - ``mvt_timer_base``
    - ``virtual_rect``
    - ``palette``
    - ``layers``
    - ``layer_effects``
    - ``item_instances``
    - ``fade_in`` (or None)
    - ``fade_out`` (or None)

    Fields excluded on purpose:

    - ``events`` — emitted separately by ``algorithm/emit.py``.
    - ``sub_records`` — raw debug walker records; not pack content.
    - ``deferred_encrypted`` — walker-level encrypted sub-chunks we
      couldn't decode; empty on FNAF 1.

    Every included sub-structure calls its own ``as_dict()`` method
    (already defined on every decoder dataclass), so we inherit clean
    JSON-safe output with no nested bytes or tuples.
    """
    return {
        "frame_index": frame_index,
        "frame_name": frame.name,
        "mvt_timer_base": frame.mvt_timer_base,
        "virtual_rect": frame.virtual_rect.as_dict(),
        "palette": frame.palette.as_dict(),
        "layers": frame.layers.as_dict(),
        "layer_effects": frame.layer_effects.as_dict(),
        "item_instances": frame.item_instances.as_dict(),
        "fade_in": frame.fade_in.as_dict() if frame.fade_in is not None else None,
        "fade_out": frame.fade_out.as_dict() if frame.fade_out is not None else None,
    }


# --- Writers -----------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed so it works on large images/audio."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    """Pretty 2-space-indented JSON with trailing newline (POSIX).

    ``sort_keys=True`` pins dict-key ordering so any future pack-shape
    antibody can compute stable SHA-256s without depending on Python's
    insertion-order guarantee.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# --- Manifest builder --------------------------------------------------


#: Sub-directories the master manifest indexes. Ordered so the emitted
#: manifest is stable regardless of filesystem enumeration order.
_MANIFEST_SUBDIRS: tuple[str, ...] = (
    "algorithm",
    "audio",
    "images",
    "runtime_pack/frames_state",
)


def _collect_pack_files(out_dir: Path) -> list[Path]:
    """Collect every file the master manifest should index.

    Walks each sub-directory in :data:`_MANIFEST_SUBDIRS`, ``sorted()``
    so output ordering is deterministic across runs and filesystems.

    Excludes the master manifest itself (``runtime_pack/manifest.json``)
    because it is written last and cannot self-reference its own
    SHA-256. Mirrors the convention used by ``algorithm/manifest.json``.
    """
    files: list[Path] = []
    for sub in _MANIFEST_SUBDIRS:
        sub_dir = out_dir / sub
        if not sub_dir.exists():
            continue
        files.extend(sorted(p for p in sub_dir.rglob("*") if p.is_file()))
    return files


def _require_prerequisite_dirs(out_dir: Path) -> None:
    """Fail loudly if the pre-emit directories haven't been populated.

    The runtime-pack manifest is the master index. If ``algorithm/``
    or ``images/`` or ``audio/`` is missing, the manifest would silently
    omit those files and downstream consumers would see an incomplete
    pack. Refuse to emit a partial manifest.
    """
    missing: list[str] = []
    for sub in ("algorithm", "images", "audio"):
        if not (out_dir / sub).exists():
            missing.append(sub)
    if missing:
        raise RuntimePackEmitError(
            f"runtime-pack emit requires {missing} under {out_dir} — "
            f"run `fnaf-parser dump-algorithm` and `fnaf-parser "
            f"dump-assets` first, then `dump-runtime-pack`."
        )


def _build_master_manifest(
    *,
    exe: Path,
    source_sha256: str,
    out_dir: Path,
    per_frame_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the master manifest dict.

    Files are indexed at paths relative to ``out_dir`` (e.g.
    ``algorithm/combined.jsonl``, ``images/0001.png``). Counts section
    carries cross-frame aggregates Rust consumers or future antibodies
    can gate on without re-reading every frame file.
    """
    all_files = _collect_pack_files(out_dir)
    files_manifest: dict[str, dict[str, Any]] = {}
    for f in all_files:
        rel = f.relative_to(out_dir).as_posix()
        files_manifest[rel] = {
            "sha256": _sha256_file(f),
            "size_bytes": f.stat().st_size,
        }

    images_count = sum(1 for f in all_files if f.suffix == ".png")
    audio_count = sum(1 for f in all_files if f.suffix == ".wav")
    total_instances = sum(s["instance_count"] for s in per_frame_summaries)
    total_layers = sum(s["layer_count"] for s in per_frame_summaries)
    per_frame_instance_counts = {
        str(s["frame_index"]): s["instance_count"] for s in per_frame_summaries
    }

    return {
        "pack_version": PACK_VERSION,
        "decoder_version": DECODER_VERSION,
        "source_file": exe.name,
        "source_sha256": source_sha256,
        "files": files_manifest,
        "counts": {
            "frames": len(per_frame_summaries),
            "images": images_count,
            "audio": audio_count,
            "total_instances": total_instances,
            "total_layers": total_layers,
            "per_frame_instance_counts": per_frame_instance_counts,
        },
    }


# --- Public entry point -----------------------------------------------


def dump_runtime_pack(exe: Path, out_dir: Path) -> list[Path]:
    """Emit the runtime-pack artefact suite under ``<out_dir>/runtime_pack/``.

    Pipeline:

    1. ``_require_prerequisite_dirs`` — assert ``algorithm/``,
       ``images/``, ``audio/`` exist. Loud-fail if not.
    2. ``load_pack`` — PE walk + chunk walk + RC4 transform prime.
    3. For every 0x3333 Frame chunk: ``decode_frame`` + serialize
       non-events fields to
       ``runtime_pack/frames_state/frame_NN_<slug>.json``.
    4. Compute source ``.exe`` SHA-256.
    5. Build + write ``runtime_pack/manifest.json`` pointing at every
       file across algorithm/, images/, audio/, runtime_pack/.

    Returns every path written, in the same stable order the files
    were emitted.

    Raises
    ------
    RuntimePackEmitError
        If ``algorithm/``, ``images/``, or ``audio/`` are missing under
        ``out_dir``.
    fnaf_parser.decoders.frame.FrameDecodeError
        On malformed frame containers — propagated from ``decode_frame``.
    """
    exe = Path(exe)
    out_dir = Path(out_dir)

    _require_prerequisite_dirs(out_dir)

    pack = load_pack(exe)
    unicode = pack.walk.header.unicode

    frame_records = [r for r in pack.walk.records if r.id == _CHUNK_ID_FRAME]

    runtime_pack_dir = out_dir / "runtime_pack"
    frames_state_dir = runtime_pack_dir / "frames_state"
    runtime_pack_dir.mkdir(parents=True, exist_ok=True)
    frames_state_dir.mkdir(parents=True, exist_ok=True)

    per_frame_summaries: list[dict[str, Any]] = []
    written: list[Path] = []

    for frame_index, frame_rec in enumerate(frame_records):
        frame_payload = read_chunk_payload(
            pack.blob, frame_rec, transform=pack.transform
        )
        frame = decode_frame(
            frame_payload, unicode=unicode, transform=pack.transform
        )

        scene_start = _frame_to_scene_start_dict(
            frame, frame_index=frame_index
        )
        frame_path = frames_state_dir / _frame_state_filename(
            frame_index, frame.name
        )
        _write_json(frame_path, scene_start)
        written.append(frame_path)

        per_frame_summaries.append(
            {
                "frame_index": frame_index,
                "frame_name": frame.name,
                "instance_count": len(frame.item_instances.instances),
                "layer_count": len(frame.layers.layers),
            }
        )

    # Manifest last — must see every frame_state file on disk before
    # hashing, and self-referencing would produce an undefined digest.
    source_sha256 = hashlib.sha256(pack.blob).hexdigest()
    manifest = _build_master_manifest(
        exe=exe,
        source_sha256=source_sha256,
        out_dir=out_dir,
        per_frame_summaries=per_frame_summaries,
    )
    manifest_path = runtime_pack_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    written.append(manifest_path)

    return written
