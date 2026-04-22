"""Shared end-to-end pipeline helpers for the FNAF parser CLI.

This module consolidates the *one* incantation used by every CLI
subcommand that actually cracks the pack: locate the data-pack start,
walk every chunk, and prime the RC4 `TransformState` so flag=2/3
encrypted chunks decrypt-on-read via `read_chunk_payload(...,
transform=...)`.

History
-------

Prior to CLI Scaffolding the only end-to-end entry point was
`parser/main.py` (probe #2 + probe #3 only — no decryption, no decode),
plus a `_fnaf1_transform_and_records()` helper duplicated across
`tests/test_sounds.py` and `tests/test_images_pixels.py`. The CLI
subcommands need the same shape, and each subcommand doing its own
ad-hoc transform setup would be the third copy.

What's in scope here (CLI Scaffolding)
--------------------------------------

Just the transform setup + walker call. `dump-assets` and
`dump-algorithm` import `load_pack` and keep their own decode logic.
Future refactors can migrate the test helpers onto this module, but
that's outside this work unit's mandate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fnaf_parser.chunk_walker import ChunkWalkResult, walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import TransformState, make_transform
from fnaf_parser.pe_walker import pe_data_pack_start


# --- Chunk IDs we use during transform bootstrap ------------------------

# The three strings feed `make_transform` alongside product_build. Their
# values are commented in the decoders themselves; we just need the IDs.
_CHUNK_ID_APP_NAME = 0x2224
_CHUNK_ID_APP_EDITOR_FILENAME = 0x222E
_CHUNK_ID_COPYRIGHT = 0x223B


@dataclass(frozen=True, slots=True)
class LoadedPack:
    """Everything downstream subcommands need from a parsed `.exe`.

    - `exe`: the path as supplied by the user.
    - `blob`: full file contents, cached so subcommands can call
      `read_chunk_payload(blob, rec, transform=...)` without re-reading.
    - `walk`: the ChunkWalkResult from `walk_chunks` — header, records,
      histogram inputs.
    - `transform`: the primed RC4 TransformState, ready to pass into
      `read_chunk_payload` for flag=2/3 chunks. `None` is never
      returned — if the three seed strings were missing we'd have
      errored out; the app can't start without them.
    """

    exe: Path
    blob: bytes
    walk: ChunkWalkResult
    transform: TransformState


def load_pack(exe: Path) -> LoadedPack:
    """Locate the data-pack, walk every chunk, and prime the RC4 transform.

    This is the one entry point CLI subcommands call before doing any
    chunk-specific decode work. The function is intentionally simple —
    it does not decode Images / Sounds / Frames. Those happen in the
    subcommand handlers because each handler has different scoping needs
    (e.g. `parse` doesn't need decryption primed, but we prime it anyway
    to keep the return type stable).

    Raises
    ------
    StopIteration
        If any of the three transform-seed string chunks is missing.
        That's a structurally broken pack — the Clickteam runtime itself
        couldn't start it — so we let the error propagate loudly instead
        of silently swapping in empty strings.
    """
    blob = exe.read_bytes()
    pack_start = pe_data_pack_start(exe)
    walk = walk_chunks(exe, pack_start=pack_start)

    def _str_of(chunk_id: int) -> str:
        rec = next(r for r in walk.records if r.id == chunk_id)
        return decode_string_chunk(
            read_chunk_payload(blob, rec), unicode=walk.header.unicode
        )

    editor = _str_of(_CHUNK_ID_APP_EDITOR_FILENAME)
    name = _str_of(_CHUNK_ID_APP_NAME)
    copyright_records = [r for r in walk.records if r.id == _CHUNK_ID_COPYRIGHT]
    copyright_str = (
        decode_string_chunk(
            read_chunk_payload(blob, copyright_records[0]),
            unicode=walk.header.unicode,
        )
        if copyright_records
        else ""
    )
    transform = make_transform(
        editor=editor,
        name=name,
        copyright_str=copyright_str,
        build=walk.header.product_build,
        unicode=walk.header.unicode,
    )
    return LoadedPack(exe=exe, blob=blob, walk=walk, transform=transform)
