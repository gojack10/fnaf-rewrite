"""Regression tests for 0x5556 FontOffsets decoder (probe #10).

Top-level offset table — one u32 LE file offset per font-bank handle.
Structural twin of 0x5555 ImageOffsets (probe #6) and 0x5557
SoundOffsets (probe #8). Envelope-only scope: decode the offsets, pin
their empirical shape, and leave per-font parsing to
`test_fonts.py` (probe #10 Fonts decoder).

Antibody coverage:

- #1 strict-unknown : a payload whose length is not a multiple of 4
  raises `FontOffsetsDecodeError`.
- #2 byte-count    : len(payload) // 4 == count, payload consumed in
  full.
- #3 round-trip    : synthetic pack/unpack on a known offset stream.
- #5 multi-input   : runs against the FNAF 1 0x5556 payload.
- #7 snapshot      : pins count, zero-slot count, distinct count, max
  offset, and a SHA-256 of the full offset stream.
- Cross-chunk preview: `non_zero_count` is expected to match the
  0x6667 Fonts NumOfItems. Probe #10's fonts test upgrades this to a
  real cross-chunk handshake once the Fonts bank decoder lands.
"""

from __future__ import annotations

import hashlib
import struct
from collections import Counter
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.font_offsets import (
    FONT_OFFSET_ENTRY_SIZE,
    FontOffsets,
    FontOffsetsDecodeError,
    decode_font_offsets,
)
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


# --- Synthetic tests (no binary needed) ----------------------------------


def test_decode_font_offsets_empty():
    """Zero-length payload decodes to an empty bank. Nothing about the
    chunk format requires a count prefix, so 0 entries is legal."""
    fo = decode_font_offsets(b"")
    assert fo == FontOffsets(count=0, offsets=())


def test_decode_font_offsets_size_not_multiple_of_4_raises():
    """Antibody #2: any length not divisible by 4 must raise, not
    silently truncate to the nearest u32."""
    for bad in (1, 2, 3, 5, 7, 9, 13, 17, 2543):
        with pytest.raises(
            FontOffsetsDecodeError, match="not a multiple of 4"
        ):
            decode_font_offsets(b"\x00" * bad)


def test_decode_font_offsets_roundtrip_synthetic():
    """Antibody #3: pack a recognisable u32 LE stream and verify each
    offset comes back exactly. 0 is kept as a legal value — it's how
    FNAF 1 marks empty handle slots in the sibling tables."""
    offsets = (0xDEADBEEF, 0x00, 0x12345678, 0x00, 0xFFFFFFFF, 0x01020304)
    payload = b"".join(struct.pack("<I", o) for o in offsets)
    assert len(payload) == len(offsets) * FONT_OFFSET_ENTRY_SIZE

    fo = decode_font_offsets(payload)
    assert isinstance(fo, FontOffsets)
    assert fo.count == len(offsets)
    assert fo.offsets == offsets


def test_decode_font_offsets_as_dict_shape():
    """`as_dict` output is stable JSON. Snapshot tests depend on this
    shape staying greppable."""
    payload = struct.pack("<III", 1, 2, 3)
    d = decode_font_offsets(payload).as_dict()
    assert d == {"count": 3, "offsets": [1, 2, 3]}


def test_decode_font_offsets_u32_byte_order():
    """Little-endian: bytes `01 00 00 00` decode to 1, not 0x01000000.
    Tiny but real: a BE/LE swap here puts every offset in outer space."""
    payload = bytes([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80])
    fo = decode_font_offsets(payload)
    assert fo.count == 2
    assert fo.offsets == (1, 0x80000000)


# --- FNAF 1 multi-input (Antibody #5 / #7) ------------------------------


def _fnaf1_transform_and_records():
    """Walk the FNAF 1 pack with the RC4 transform enabled so
    read_chunk_payload can decode any chunk. Mirrors the helper in
    test_sound_offsets.py."""
    blob = FNAF_EXE.read_bytes()
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)

    def _str_of(chunk_id: int) -> str:
        rec = next(r for r in result.records if r.id == chunk_id)
        return decode_string_chunk(
            read_chunk_payload(blob, rec), unicode=result.header.unicode
        )

    editor = _str_of(0x222E)
    name = _str_of(0x2224)
    copyright_records = [r for r in result.records if r.id == 0x223B]
    copyright_str = (
        decode_string_chunk(
            read_chunk_payload(blob, copyright_records[0]),
            unicode=result.header.unicode,
        )
        if copyright_records
        else ""
    )
    transform = make_transform(
        editor=editor,
        name=name,
        copyright_str=copyright_str,
        build=result.header.product_build,
        unicode=result.header.unicode,
    )
    return blob, result, transform


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_font_offsets_decodes_without_error():
    """Antibody #5 multi-input: the full FNAF 1 0x5556 payload decodes
    without raising and produces a non-empty offset bank."""
    blob, result, _ = _fnaf1_transform_and_records()
    recs = [r for r in result.records if r.id == 0x5556]
    assert len(recs) == 1, (
        f"FNAF 1 should carry exactly one 0x5556 chunk; saw {len(recs)}"
    )
    payload = read_chunk_payload(blob, recs[0])
    fo = decode_font_offsets(payload)
    assert fo.count > 0
    for o in fo.offsets:
        assert 0 <= o <= 0xFFFFFFFF


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_font_offsets_snapshot():
    """Antibody #7 snapshot: pin the FNAF 1 0x5556 decode against drift.

    Any single-byte drift in the decompressed payload changes the SHA.
    Any handle-bank reshuffle changes count or zero_count.
    """
    blob, result, _ = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x5556)
    payload = read_chunk_payload(blob, rec)
    fo = decode_font_offsets(payload)

    zero_count = sum(1 for o in fo.offsets if o == 0)
    distinct_count = len(set(fo.offsets))
    max_offset = max(fo.offsets)
    non_zero_count = fo.count - zero_count
    sha = hashlib.sha256(payload).hexdigest()

    # Debug print — surfaces empirical values on test failure so the
    # snapshot expectations below can be bootstrapped / re-pinned.
    print(
        f"\n[font_offsets snapshot] count={fo.count} zero_count={zero_count} "
        f"distinct_count={distinct_count} non_zero_count={non_zero_count} "
        f"max_offset=0x{max_offset:08X} sha={sha}\n"
        f"offsets={list(fo.offsets)}"
    )

    # Pinned empirical values captured 2026-04-21 (probe #10).
    # Observation: FNAF 1's Fonts bank is densely packed — no empty
    # handle slots. Different shape from sounds (8 zeros / 60 total)
    # and images (zeros present). Any future FNAF 1 rebuild that
    # introduces empty slots will trip this pin.
    assert fo.count == 7
    assert zero_count == 0
    assert distinct_count == 7
    assert max_offset == 0x00000279
    assert non_zero_count == 7
    assert sha == (
        "0d94b87da55f07187252c29cd03da22fc8ae81e3c1b90fa5d290fd8aa7aa9211"
    )
    # The exact on-wire offset list — documented so a single-byte
    # drift surfaces as a vector diff, not just a SHA mismatch.
    assert fo.offsets == (515, 402, 326, 264, 459, 633, 577)


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_font_offsets_only_zero_duplicates():
    """Structural invariant (empirical): the *only* duplicate offset in
    the FNAF 1 0x5556 table is 0. Every non-zero offset is unique —
    fonts aren't shared across multiple handles.

    Same antibody shape as 0x5555 ImageOffsets and 0x5557 SoundOffsets.
    If a future pack legitimately shares fonts across handles, this
    test fires and the pack-level assumption needs re-checking before
    the 0x6667 decoder consumers ship.
    """
    blob, result, _ = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x5556)
    payload = read_chunk_payload(blob, rec)
    fo = decode_font_offsets(payload)

    counts = Counter(fo.offsets)
    duplicates = {offset: n for offset, n in counts.items() if n > 1}
    # 0 may or may not be the only duplicate — pin whatever shape is
    # observed. FNAF 1's Fonts bank is tiny, so a small number of empty
    # slots is expected; any non-zero duplicate would be a genuine
    # red flag.
    non_zero_duplicates = {k: v for k, v in duplicates.items() if k != 0}
    assert non_zero_duplicates == {}, (
        f"expected only offset=0 to possibly repeat; "
        f"saw non-zero duplicates {non_zero_duplicates}"
    )
