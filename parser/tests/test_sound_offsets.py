"""Regression tests for 0x5557 SoundOffsets decoder (probe #8a).

Top-level offset table — one u32 LE file offset per sound-bank handle.
Structural twin of 0x5555 ImageOffsets (probe #6). Envelope-only
scope: decode the offsets, pin their empirical shape, and leave
per-sound parsing to probe #8b (0x6668 Sounds).

Antibody coverage:

- #1 strict-unknown : a payload whose length is not a multiple of 4
  raises `SoundOffsetsDecodeError`.
- #2 byte-count    : len(payload) // 4 == count, payload consumed in
  full.
- #3 round-trip    : synthetic pack/unpack on a known offset stream.
- #5 multi-input   : runs against the FNAF 1 0x5557 payload.
- #7 snapshot      : pins count, zero-slot count, distinct count, max
  offset, and a SHA-256 of the full offset stream. Any single-byte
  drift in the decompressed payload changes the SHA; any handle-bank
  reshuffle changes count or zero_count.
- Structural invariant (empirical): the *only* duplicate offset is 0.
  Every non-zero offset is unique. Pinned as a test so a future game's
  bank structure that violates this surfaces the change loudly.
- Cross-chunk preview: non_zero_count (52) matches the anticipated
  0x6668 Sounds NumOfItems. Probe #8b will upgrade this to a real
  cross-chunk handshake once the Sounds bank decoder lands.
"""

from __future__ import annotations

import hashlib
import struct
from collections import Counter
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.sound_offsets import (
    SOUND_OFFSET_ENTRY_SIZE,
    SoundOffsets,
    SoundOffsetsDecodeError,
    decode_sound_offsets,
)
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


# --- Synthetic tests (no binary needed) ----------------------------------


def test_decode_sound_offsets_empty():
    """Zero-length payload decodes to an empty bank. Nothing about the
    chunk format requires a count prefix, so 0 entries is legal."""
    so = decode_sound_offsets(b"")
    assert so == SoundOffsets(count=0, offsets=())


def test_decode_sound_offsets_size_not_multiple_of_4_raises():
    """Antibody #2: any length not divisible by 4 must raise, not
    silently truncate to the nearest u32."""
    for bad in (1, 2, 3, 5, 7, 9, 13, 17, 2543):
        with pytest.raises(
            SoundOffsetsDecodeError, match="not a multiple of 4"
        ):
            decode_sound_offsets(b"\x00" * bad)


def test_decode_sound_offsets_roundtrip_synthetic():
    """Antibody #3: pack a recognisable u32 LE stream and verify each
    offset comes back exactly. 0 is kept as a legal value — it's how
    FNAF 1 marks empty handle slots."""
    offsets = (0xDEADBEEF, 0x00, 0x12345678, 0x00, 0xFFFFFFFF, 0x01020304)
    payload = b"".join(struct.pack("<I", o) for o in offsets)
    assert len(payload) == len(offsets) * SOUND_OFFSET_ENTRY_SIZE

    so = decode_sound_offsets(payload)
    assert isinstance(so, SoundOffsets)
    assert so.count == len(offsets)
    assert so.offsets == offsets


def test_decode_sound_offsets_as_dict_shape():
    """`as_dict` output is stable JSON. Snapshot tests depend on this
    shape staying greppable."""
    payload = struct.pack("<III", 1, 2, 3)
    d = decode_sound_offsets(payload).as_dict()
    assert d == {"count": 3, "offsets": [1, 2, 3]}


def test_decode_sound_offsets_u32_byte_order():
    """Little-endian: bytes `01 00 00 00` decode to 1, not 0x01000000.
    Tiny but real: a BE/LE swap here puts every offset in outer space."""
    payload = bytes([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80])
    so = decode_sound_offsets(payload)
    assert so.count == 2
    assert so.offsets == (1, 0x80000000)


# --- FNAF 1 multi-input (Antibody #5 / #7) ------------------------------


def _fnaf1_transform_and_records():
    """Walk the FNAF 1 pack with the RC4 transform enabled so
    read_chunk_payload can decode any chunk. Mirrors the helper in
    test_image_offsets.py."""
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
def test_fnaf1_sound_offsets_decodes_without_error():
    """Antibody #5 multi-input: the full FNAF 1 0x5557 payload decodes
    without raising and produces a non-empty offset bank."""
    blob, result, _ = _fnaf1_transform_and_records()
    recs = [r for r in result.records if r.id == 0x5557]
    assert len(recs) == 1, (
        f"FNAF 1 should carry exactly one 0x5557 chunk; saw {len(recs)}"
    )
    payload = read_chunk_payload(blob, recs[0])
    so = decode_sound_offsets(payload)
    assert so.count > 0
    # Every offset must be u32-sized; the decoder guarantees this
    # structurally, but make it visible at the test layer too.
    for o in so.offsets:
        assert 0 <= o <= 0xFFFFFFFF


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_sound_offsets_snapshot():
    """Antibody #7 snapshot: pin the FNAF 1 0x5557 decode against drift.

    Captured empirically on 2026-04-21 (probe #8a):

    - count             : 60 offsets
    - zero_count        : 8 (empty handle slots — see duplicate test)
    - distinct_count    : 53 (= 52 non-zero unique + 1 zero bucket)
    - max_offset        : 0x04B8EB93 (≈ 75.56 MB) — upper bound on the
                          0x6668 Sounds chunk body size; the full
                          Sounds chunk is ~76 MB decompressed, so the
                          offset table fits entirely within it.
    - offsets_sha256    : hash of the full 240-byte payload.

    Non-zero count (52) matches the expected 0x6668 NumOfItems. Probe
    #8b will upgrade this cross-chunk observation to a real handshake.

    Any single-byte drift in the decompressed payload changes the SHA.
    Any handle-bank reshuffle changes count or zero_count.
    """
    blob, result, _ = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x5557)
    payload = read_chunk_payload(blob, rec)
    so = decode_sound_offsets(payload)

    zero_count = sum(1 for o in so.offsets if o == 0)
    distinct_count = len(set(so.offsets))
    max_offset = max(so.offsets)
    sha = hashlib.sha256(payload).hexdigest()

    assert so.count == 60
    assert zero_count == 8
    assert distinct_count == 53
    assert max_offset == 0x04B8EB93
    assert sha == (
        "9fa90db55b91bdb0cbcfe928d06e6e4ac9f29ed03b3866c9164286e86562666c"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_sound_offsets_only_zero_duplicates():
    """Structural invariant (empirical): the *only* duplicate offset in
    the FNAF 1 0x5557 table is 0. Every non-zero offset is unique —
    sounds aren't shared by multiple handles.

    This is a load-bearing antibody because:

    - If RC4 / zlib drift silently corrupted the offset stream, the
      probability of the collision set staying exactly {0 → 8×} is
      vanishingly small.
    - If a future pack legitimately shares sounds across handles, this
      test fires and the pack-level assumption needs re-checking before
      the 0x6668 decoder ships.

    Same shape as the equivalent antibody on 0x5555 ImageOffsets —
    which makes its failure mode symmetric too: a drift that breaks
    both offset tables will trip two tests, not one.
    """
    blob, result, _ = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x5557)
    payload = read_chunk_payload(blob, rec)
    so = decode_sound_offsets(payload)

    counts = Counter(so.offsets)
    duplicates = {offset: n for offset, n in counts.items() if n > 1}
    assert duplicates == {0: 8}, (
        f"expected only offset=0 to repeat (8×), got {duplicates}"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_sound_offsets_not_monotonic():
    """Cross-check against the 'Sound Handles' name in CTFAK2: if
    0x5557 were a handle map (like 0x222B FrameHandles, i.e. an i16
    array where `items[i] = handle`) we'd expect the table to look
    index-like — roughly monotonic or densely packed small integers.
    It isn't: offsets are random-access byte addresses into the Sounds
    chunk body, keyed by sound-bank index. Pin non-monotonicity so any
    future refactor that accidentally starts sorting or rewriting the
    stream fires this test.
    """
    blob, result, _ = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x5557)
    payload = read_chunk_payload(blob, rec)
    so = decode_sound_offsets(payload)

    # Filter zeros (empty slots) before checking monotonicity — they
    # would artificially break any ordering even if the non-zero tail
    # was monotonic.
    non_zero = [o for o in so.offsets if o != 0]
    is_monotonic = all(
        non_zero[i] <= non_zero[i + 1] for i in range(len(non_zero) - 1)
    )
    assert not is_monotonic, (
        "FNAF 1 0x5557 offsets were unexpectedly monotonic — that would "
        "imply a different semantic (handle map, not offset table). "
        "Re-investigate before trusting this decoder."
    )
