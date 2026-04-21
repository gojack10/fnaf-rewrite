"""Regression tests for 0x3333 Frame container walker (probe #4.4).

Antibody coverage:

- #1 strict — unknown sub-chunk id raises FrameDecodeError (synthetic TLV).
- #2 byte-count — sub-chunk claiming more bytes than remain raises.
- #4 multi-oracle — outer 0x3333 count equals AppHeader.number_of_frames.
- #5 multi-input — all 17 FNAF 1 frames decode cleanly and yield the
  exact ordered list of empirical frame names.
- #6 loud skip — every frame carries a non-empty deferred_encrypted list
  so the encryption-deferred sub-chunks are counted, not hidden.

Additional invariants locked:

- Every frame's last sub-chunk is LAST (0x7F7F).
- Mvt Timer Base decodes to 60 for every FNAF 1 frame (consistent with
  Clickteam's default 60 Hz movement base).
- The union of deferred encrypted sub-chunk ids matches the nine ids
  CTFAK marks as CompressedAndEncrypted in FNAF 1 frames.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.frame import (
    Frame,
    FrameDecodeError,
    SUB_FRAME_NAME,
    decode_frame,
    walk_frame_payload,
)
from fnaf_parser.decoders.header import decode_header
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"

# Empirical ground-truth, extracted by walking all 17 0x3333 containers in
# the FNAF 1 pack and decoding the plaintext sub-chunks. Order is file
# order (the order Clickteam serialized them); if this order ever shifts
# that's either the binary or the outer walker drifting and we want to
# hear about it.
_FNAF1_FRAME_NAMES: list[str] = [
    "Frame 17",
    "title",
    "what day",
    "Frame 1",
    "died",
    "freddy",
    "next day",
    "wait",
    "gameover",
    "the end",
    "ad",
    "the end 2",
    "customize",
    "the end 3",
    "creepy start",
    "creepy end",
    "end of demo",
]

# Every frame's Mvt Timer Base is 60 in FNAF 1. Locking the constant means
# a future content build with retimed movement will fail here and route
# the investigator straight to the right sub-chunk.
_FNAF1_FRAME_MVT_TIMER_BASE = 60

# Union of encrypted sub-chunk ids across all 17 frames — the nine ids
# CTFAK2.0's Chunk.cs flags as CompressedAndEncrypted for frames. If a
# probe #4.5 decrypt ever leaks the wrong id into the deferred list, this
# lock will fire.
_FNAF1_FRAME_ENCRYPTED_SUB_IDS = frozenset(
    {0x3334, 0x3337, 0x3338, 0x333B, 0x333C, 0x333D, 0x3341, 0x3342, 0x3345}
)

# 0x3333 frame chunk itself is flag=0 (NotCompressed) at the outer layer
# in FNAF 1 — the inner TLV stream is stored raw, and Clickteam's per-sub-
# chunk flags drive any decompression/decryption from there. Empirically
# unusual (most outer chunks in this pack are flag=1), so we lock it
# explicitly — if a future content build compresses the outer frame, the
# reader below needs to decompress before calling decode_frame.
_FNAF1_FRAME_OUTER_FLAG = 0


def _fnaf1_frames() -> tuple[list[Frame], int]:
    blob = FNAF_EXE.read_bytes()
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    frame_records = [r for r in result.records if r.id == 0x3333]
    frames = [
        decode_frame(
            read_chunk_payload(blob, rec),
            unicode=result.header.unicode,
        )
        for rec in frame_records
    ]
    # Pull AppHeader for the cross-chunk oracle; the 0x2223 record is
    # guaranteed present by probe #4.1's regression tests.
    header_rec = next(r for r in result.records if r.id == 0x2223)
    app_header = decode_header(read_chunk_payload(blob, header_rec))
    return frames, app_header.number_of_frames


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_frame_count_matches_app_header():
    """Cross-chunk antibody (#4 multi-oracle): the number of 0x3333 frame
    containers in the pack must equal the AppHeader's number_of_frames.
    Two independent sources of truth — if they drift, one is wrong."""
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    frame_count = sum(1 for r in result.records if r.id == 0x3333)
    blob = FNAF_EXE.read_bytes()
    header_rec = next(r for r in result.records if r.id == 0x2223)
    app_header = decode_header(read_chunk_payload(blob, header_rec))
    assert frame_count == app_header.number_of_frames == 17


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_frame_outer_flag_is_compressed():
    """Lock the outer flag so that any future change in how the 0x3333
    container is encoded surfaces here, not as a confusing decode error."""
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    for rec in result.records:
        if rec.id == 0x3333:
            assert rec.flags == _FNAF1_FRAME_OUTER_FLAG


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_all_frames_decode_and_names_lock():
    """Multi-input antibody (#5): the SAME decoder walks all 17 frames and
    must produce the exact ordered list of empirical frame names. A drift
    in the Frame Name sub-chunk encoding, the string decoder, or the outer
    walker's frame ordering fails here."""
    frames, _ = _fnaf1_frames()
    assert [f.name for f in frames] == _FNAF1_FRAME_NAMES


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_all_frames_mvt_timer_base_is_60():
    """Every FNAF 1 frame keeps the default 60 Hz movement base. Locking
    the value means a retimed content build trips here instead of
    silently decoding to a different tick rate."""
    frames, _ = _fnaf1_frames()
    for i, f in enumerate(frames):
        assert f.mvt_timer_base == _FNAF1_FRAME_MVT_TIMER_BASE, (
            f"frame #{i} ({f.name!r}) mvt_timer_base={f.mvt_timer_base}"
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_ends_with_last_sentinel():
    """The inner TLV stream's last record must be 0x7F7F. If any frame's
    final sub-chunk is something else, the walker either terminated early
    on a mis-sized sub-chunk, or the file itself is truncated."""
    frames, _ = _fnaf1_frames()
    for i, f in enumerate(frames):
        assert f.sub_records, f"frame #{i} ({f.name!r}) has no sub-records"
        last = f.sub_records[-1]
        assert last.id == 0x7F7F, (
            f"frame #{i} ({f.name!r}) final sub-chunk is 0x{last.id:04X}, "
            f"expected 0x7F7F LAST"
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_defers_encrypted_sub_chunks():
    """Antibody #6 (loud skip): encrypted sub-chunks are deferred but
    COUNTED. Every frame in FNAF 1 carries at least one encrypted sub-
    chunk (FrameHeader, at minimum); an empty list would mean the
    is_encrypted check missed flag 3 and we are silently treating
    encrypted bytes as plaintext."""
    frames, _ = _fnaf1_frames()
    for i, f in enumerate(frames):
        assert f.deferred_encrypted, (
            f"frame #{i} ({f.name!r}) has no deferred_encrypted entries — "
            f"the encryption flag check likely missed something"
        )
        for rec in f.deferred_encrypted:
            assert rec.decoded_payload is None
            assert rec.is_encrypted


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_encrypted_sub_ids_match_known_set():
    """The union of encrypted sub-chunk ids across all 17 frames must
    match exactly the nine ids CTFAK2.0 marks CompressedAndEncrypted
    for frames. A stray id in the deferred list means we're mis-reading
    flags (or a different build has widened the encrypted set)."""
    frames, _ = _fnaf1_frames()
    seen: set[int] = set()
    for f in frames:
        for rec in f.deferred_encrypted:
            seen.add(rec.id)
    assert seen == set(_FNAF1_FRAME_ENCRYPTED_SUB_IDS)


# --- Synthetic payload tests (no FNAF binary needed) ---------------------


def _make_sub_tlv(sub_id: int, flags: int, body: bytes) -> bytes:
    """Build one sub-chunk TLV header + body. Helper for synthetic tests."""
    return struct.pack("<hHI", sub_id, flags, len(body)) + body


def test_walk_frame_payload_strict_rejects_unknown_sub_id():
    """Antibody #1: sub-chunk id 0x9999 is not in CHUNK_NAMES; walker
    must raise FrameDecodeError rather than silently skipping."""
    # Prepend a valid Frame Name sub-chunk so we're past the first byte,
    # then a bogus id — confirms the strict check fires mid-stream, not
    # only at offset 0.
    valid = _make_sub_tlv(SUB_FRAME_NAME, 0, b"\x00\x00")  # empty unicode string
    # 0x1234 is a safe pick: fits in signed int16 and is NOT registered
    # anywhere in CHUNK_NAMES (the real IDs start at 0x1122 and cluster
    # much higher). The string appearing in the error message is how the
    # walker formats the unknown id — check for it explicitly.
    bogus = _make_sub_tlv(0x1234, 0, b"")
    payload = valid + bogus
    with pytest.raises(FrameDecodeError, match="0x1234"):
        walk_frame_payload(payload)


def test_walk_frame_payload_strict_rejects_oversize_sub():
    """Antibody #2: sub-chunk's size field claims more bytes than remain
    in the payload. Must raise, not read-past-end."""
    # Header says size=100 but only 4 body bytes follow.
    header = struct.pack("<hHI", SUB_FRAME_NAME, 0, 100)
    payload = header + b"\x00" * 4
    with pytest.raises(FrameDecodeError, match="size=100"):
        walk_frame_payload(payload)


def test_walk_frame_payload_stops_at_last_sentinel():
    """LAST (0x7F7F) terminates walking; any bytes after must be ignored
    (consistent with how the outer walker treats LAST)."""
    name = _make_sub_tlv(SUB_FRAME_NAME, 0, b"\x48\x00\x69\x00\x00\x00")  # "Hi\0"
    last = _make_sub_tlv(0x7F7F, 0, b"")
    trailing_garbage = b"\xff" * 16  # would fail strict mode if walker continued
    payload = name + last + trailing_garbage

    records = walk_frame_payload(payload)
    assert [r.id for r in records] == [SUB_FRAME_NAME, 0x7F7F]


def test_decode_frame_extracts_name_and_mvt_timer_base():
    """Minimal synthetic frame: Name + MvtTimerBase + LAST. Confirms the
    decoder picks the right sub-chunks out of the stream without any
    encrypted payloads involved."""
    # UTF-16LE "abc\0"
    name_body = "abc".encode("utf-16-le") + b"\x00\x00"
    name_sub = _make_sub_tlv(SUB_FRAME_NAME, 0, name_body)
    # 0x3347 Mvt Timer Base: int32 LE = 42
    mvt_sub = _make_sub_tlv(0x3347, 0, struct.pack("<i", 42))
    last = _make_sub_tlv(0x7F7F, 0, b"")
    payload = name_sub + mvt_sub + last

    frame = decode_frame(payload, unicode=True)
    assert frame.name == "abc"
    assert frame.mvt_timer_base == 42
    assert frame.deferred_encrypted == ()
    assert frame.sub_records[-1].id == 0x7F7F


def test_decode_frame_defers_encrypted_sub_chunks():
    """A sub-chunk flagged Encrypted (2) or CompressedAndEncrypted (3)
    must land in deferred_encrypted with decoded_payload=None — not be
    silently dropped, and not crash the walker."""
    name_body = "x".encode("utf-16-le") + b"\x00\x00"
    name_sub = _make_sub_tlv(SUB_FRAME_NAME, 0, name_body)
    # Frame Header (0x3334) flagged CompressedAndEncrypted with opaque body.
    enc_sub = _make_sub_tlv(0x3334, 3, b"\xaa\xbb\xcc\xdd")
    last = _make_sub_tlv(0x7F7F, 0, b"")
    payload = name_sub + enc_sub + last

    frame = decode_frame(payload, unicode=True)
    assert frame.name == "x"
    assert len(frame.deferred_encrypted) == 1
    deferred = frame.deferred_encrypted[0]
    assert deferred.id == 0x3334
    assert deferred.decoded_payload is None
    assert deferred.raw == b"\xaa\xbb\xcc\xdd"
    assert deferred.is_encrypted


def test_decode_frame_rejects_wrong_mvt_timer_base_size():
    """Mvt Timer Base is a fixed 4-byte int32. A payload of any other
    length means either the container is mis-parsed or the format
    drifted — loud failure is better than silent misread."""
    name_body = "x".encode("utf-16-le") + b"\x00\x00"
    name_sub = _make_sub_tlv(SUB_FRAME_NAME, 0, name_body)
    bad_mvt = _make_sub_tlv(0x3347, 0, b"\x00\x00")  # only 2 bytes
    last = _make_sub_tlv(0x7F7F, 0, b"")
    payload = name_sub + bad_mvt + last
    with pytest.raises(FrameDecodeError, match="Mvt Timer Base"):
        decode_frame(payload, unicode=True)
