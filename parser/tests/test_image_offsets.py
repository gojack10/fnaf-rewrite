"""Regression tests for 0x5555 ImageOffsets decoder (probe #6).

Top-level offset table — one u32 LE file offset per image-bank handle.
Envelope-only scope: decode the offsets, pin their empirical shape, and
leave per-image parsing to probe #7 (0x6666 Images).

Antibody coverage:

- #1 strict-unknown : a payload whose length is not a multiple of 4
  raises `ImageOffsetsDecodeError`.
- #2 byte-count    : len(payload) // 4 == count, payload consumed in
  full.
- #3 round-trip    : synthetic pack/unpack on a known offset stream.
- #5 multi-input   : runs against the FNAF 1 0x5555 payload.
- #7 snapshot      : pins count, zero-slot count, distinct count, max
  offset, and a SHA-256 of the full offset stream. Any single-byte
  drift in the decompressed payload changes the SHA; any handle-bank
  reshuffle changes count or zero_count.
- Structural invariant (empirical): the *only* duplicate offset is 0.
  Every non-zero offset is unique. Pinned as a test so a future game's
  bank structure that violates this surfaces the change loudly.
"""

from __future__ import annotations

import hashlib
import struct
from collections import Counter
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.image_offsets import (
    IMAGE_OFFSET_ENTRY_SIZE,
    ImageOffsets,
    ImageOffsetsDecodeError,
    decode_image_offsets,
)
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


# --- Synthetic tests (no binary needed) ----------------------------------


def test_decode_image_offsets_empty():
    """Zero-length payload decodes to an empty bank. Nothing about the
    chunk format requires a count prefix, so 0 entries is legal."""
    io = decode_image_offsets(b"")
    assert io == ImageOffsets(count=0, offsets=())


def test_decode_image_offsets_size_not_multiple_of_4_raises():
    """Antibody #2: any length not divisible by 4 must raise, not
    silently truncate to the nearest u32."""
    for bad in (1, 2, 3, 5, 7, 9, 13, 17, 2543):
        with pytest.raises(
            ImageOffsetsDecodeError, match="not a multiple of 4"
        ):
            decode_image_offsets(b"\x00" * bad)


def test_decode_image_offsets_roundtrip_synthetic():
    """Antibody #3: pack a recognisable u32 LE stream and verify each
    offset comes back exactly. 0 is kept as a legal value — it's how
    FNAF 1 marks empty handle slots."""
    offsets = (0xDEADBEEF, 0x00, 0x12345678, 0x00, 0xFFFFFFFF, 0x01020304)
    payload = b"".join(struct.pack("<I", o) for o in offsets)
    assert len(payload) == len(offsets) * IMAGE_OFFSET_ENTRY_SIZE

    io = decode_image_offsets(payload)
    assert isinstance(io, ImageOffsets)
    assert io.count == len(offsets)
    assert io.offsets == offsets


def test_decode_image_offsets_as_dict_shape():
    """`as_dict` output is stable JSON. Snapshot tests depend on this
    shape staying greppable."""
    payload = struct.pack("<III", 1, 2, 3)
    d = decode_image_offsets(payload).as_dict()
    assert d == {"count": 3, "offsets": [1, 2, 3]}


def test_decode_image_offsets_u32_byte_order():
    """Little-endian: bytes `01 00 00 00` decode to 1, not 0x01000000.
    Tiny but real: a BE/LE swap here puts every offset in outer space."""
    payload = bytes([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80])
    io = decode_image_offsets(payload)
    assert io.count == 2
    assert io.offsets == (1, 0x80000000)


# --- FNAF 1 multi-input (Antibody #5 / #7) ------------------------------


def _fnaf1_transform_and_records():
    """Walk the FNAF 1 pack with the RC4 transform enabled so
    read_chunk_payload can decode any chunk. Mirrors the helper in
    test_frame_items.py."""
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
def test_fnaf1_image_offsets_decodes_without_error():
    """Antibody #5 multi-input: the full FNAF 1 0x5555 payload decodes
    without raising and produces a non-empty offset bank."""
    blob, result, _ = _fnaf1_transform_and_records()
    recs = [r for r in result.records if r.id == 0x5555]
    assert len(recs) == 1, (
        f"FNAF 1 should carry exactly one 0x5555 chunk; saw {len(recs)}"
    )
    payload = read_chunk_payload(blob, recs[0])
    io = decode_image_offsets(payload)
    assert io.count > 0
    # Every offset must be u32-sized; the decoder guarantees this
    # structurally, but make it visible at the test layer too.
    for o in io.offsets:
        assert 0 <= o <= 0xFFFFFFFF


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_image_offsets_snapshot():
    """Antibody #7 snapshot: pin the FNAF 1 0x5555 decode against drift.

    Captured empirically on 2026-04-21 (probe #6):

    - count             : 636 offsets
    - zero_count        : 31 (empty handle slots — see duplicate test)
    - distinct_count    : 606 (= 605 non-zero unique + 1 zero bucket)
    - max_offset        : 0x08E2AA22 (≈ 149 MB) — upper bound on the
                          0x6666 Images chunk body size; a future
                          decoder of 0x6666 can use this as a sanity
                          check on its own size.
    - offsets_sha256    : hash of the full 2544-byte payload.

    Any single-byte drift in the decompressed payload changes the SHA.
    Any handle-bank reshuffle changes count or zero_count.
    """
    blob, result, _ = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x5555)
    payload = read_chunk_payload(blob, rec)
    io = decode_image_offsets(payload)

    zero_count = sum(1 for o in io.offsets if o == 0)
    distinct_count = len(set(io.offsets))
    max_offset = max(io.offsets)
    sha = hashlib.sha256(payload).hexdigest()

    assert io.count == 636
    assert zero_count == 31
    assert distinct_count == 606
    assert max_offset == 0x08E2AA22
    assert sha == (
        "7289f07102cd22b28956081caf0eded0d03cbb546af03506896e02c4efc57d97"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_image_offsets_only_zero_duplicates():
    """Structural invariant (empirical): the *only* duplicate offset in
    the FNAF 1 0x5555 table is 0. Every non-zero offset is unique —
    images aren't shared by multiple handles.

    This is a load-bearing antibody because:

    - If RC4 / zlib drift silently corrupted the offset stream, the
      probability of the collision set staying exactly {0 → 31×} is
      vanishingly small.
    - If a future pack legitimately shares images across handles, this
      test fires and the pack-level assumption needs re-checking before
      the 0x6666 decoder ships.
    """
    blob, result, _ = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x5555)
    payload = read_chunk_payload(blob, rec)
    io = decode_image_offsets(payload)

    counts = Counter(io.offsets)
    duplicates = {offset: n for offset, n in counts.items() if n > 1}
    assert duplicates == {0: 31}, (
        f"expected only offset=0 to repeat (31×), got {duplicates}"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_image_offsets_not_monotonic():
    """Cross-check against CTFAK2's misleading 'Image Handles' name: if
    0x5555 were a handle map (like 0x222B FrameHandles, i.e. an i16
    array where `items[i] = handle`) we'd expect the table to look
    index-like — roughly monotonic or densely packed small integers.
    It isn't: offsets are random-access byte addresses into the Images
    chunk body, keyed by image-bank index. Pin non-monotonicity so any
    future refactor that accidentally starts sorting or rewriting the
    stream fires this test.
    """
    blob, result, _ = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x5555)
    payload = read_chunk_payload(blob, rec)
    io = decode_image_offsets(payload)

    # Filter zeros (empty slots) before checking monotonicity — they
    # would artificially break any ordering even if the non-zero tail
    # was monotonic.
    non_zero = [o for o in io.offsets if o != 0]
    is_monotonic = all(
        non_zero[i] <= non_zero[i + 1] for i in range(len(non_zero) - 1)
    )
    assert not is_monotonic, (
        "FNAF 1 0x5555 offsets were unexpectedly monotonic — that would "
        "imply a different semantic (handle map, not offset table). "
        "Re-investigate before trusting this decoder."
    )
