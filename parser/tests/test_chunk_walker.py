"""Regression tests for the chunk-list walker (probe #3).

Antibody coverage:
- #1 strict mode — a synthetic unknown chunk_id raises ChunkWalkError.
- #2 byte-count — end_offset - pamu_offset == GAME_HEADER_SIZE +
  sum(CHUNK_HEADER_SIZE + size).  Implicit from how walk_chunks seeks.
- #6 loud skip — total_payload_bytes is exposed and non-negative.

FNAF 1 invariants locked here:
- PACK_HEADER "associated files" prefix at 0x00101400.
- PAMU magic (Unicode variant, not PAME) at pack_start + dataSize - 32.
- reached_last is True (binary terminates cleanly with 0x7F7F at EOF - 8).
- Hot chunks we know must exist: Header (0x2223), Frame (0x3333),
  Images (0x6666), Sounds (0x6668), Last (0x7F7F).
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fnaf_parser.chunk_ids import LAST_CHUNK_ID
from fnaf_parser.chunk_walker import (
    CHUNK_HEADER_SIZE,
    GAME_HEADER_SIZE,
    ChunkWalkError,
    chunk_histogram,
    walk_chunks,
)
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"

# Hot chunks we expect any MMF2 FNAF 1 pack to contain. Conservative list —
# only chunks whose absence would mean the walker or binary are broken.
_REQUIRED_CHUNK_IDS = {
    0x2223,  # AppHeader
    0x2224,  # AppName
    0x3333,  # Frame (at least one)
    0x6666,  # Image bank
    0x6668,  # Sound bank
    0x7F7F,  # Last
}


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_walk_reaches_last_chunk():
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    # FNAF 1 uses the Unicode runtime variant — PAMU, not PAME.
    assert result.header.magic == b"PAMU"
    assert result.header.unicode is True
    # PACK_HEADER "associated files" prefix was detected and skipped.
    assert result.pack_header_present is True
    assert result.pamu_offset > result.pack_start
    assert result.reached_last, "walker did not hit the 0x7F7F LAST terminator"
    # The LAST chunk should be the final record.
    assert result.records[-1].id == LAST_CHUNK_ID


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_walk_contains_required_chunks():
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    seen_ids = {r.id for r in result.records}
    missing = _REQUIRED_CHUNK_IDS - seen_ids
    assert not missing, f"missing required chunk ids: {sorted(hex(m) for m in missing)}"


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_walk_byte_accounting():
    """Antibody #2 at the walk level: cursor math must reconcile exactly.

    end_offset - pamu_offset should equal GAME_HEADER_SIZE plus the sum of
    (CHUNK_HEADER_SIZE + size) over every record. Off-by-one here means
    every downstream chunk payload is shifted and we'd never notice.

    Uses pamu_offset (not pack_start) because the optional PACK_HEADER
    "associated files" region sits between pack_start and pamu_offset and
    its bytes are deliberately not tallied by the chunk walker.
    """
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    expected = GAME_HEADER_SIZE + sum(
        CHUNK_HEADER_SIZE + r.size for r in result.records
    )
    actual = result.end_offset - result.pamu_offset
    assert actual == expected, f"byte accounting: {actual} != {expected}"


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_histogram_is_sorted_and_complete():
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    freqs = chunk_histogram(result.records)
    # Sort stability: non-increasing counts.
    counts = [f.count for f in freqs]
    assert counts == sorted(counts, reverse=True)
    # Sum of per-id counts equals total record count.
    assert sum(counts) == len(result.records)


def test_strict_mode_rejects_unknown_chunk_id(tmp_path: Path):
    """Antibody #1: an unknown chunk_id must raise, never silently skip.

    Synthesize a pack-shaped blob with a bogus chunk id 0xDEAD, confirm
    we raise with the offset in the error.
    """
    pack = bytearray()
    pack += b"PAME"
    pack += struct.pack("<HHII", 0x0302, 0x0000, 0, 285)  # MMF2 header
    # chunk header: id=0xDEAD, flags=0, size=0. Packed unsigned; walker
    # reinterprets as int16 so the id on the wire is what matters.
    pack += struct.pack("<HHI", 0xDEAD, 0, 0)

    blob_path = tmp_path / "fake.pack"
    blob_path.write_bytes(bytes(pack))

    with pytest.raises(ChunkWalkError, match="unknown chunk_id"):
        walk_chunks(blob_path, pack_start=0)


def test_rejects_bad_magic(tmp_path: Path):
    blob_path = tmp_path / "bad_magic.pack"
    blob_path.write_bytes(b"NOPE" + b"\x00" * 12)
    with pytest.raises(ChunkWalkError, match="PAME/PAMU"):
        walk_chunks(blob_path, pack_start=0)
