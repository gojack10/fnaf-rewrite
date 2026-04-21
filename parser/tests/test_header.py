"""Regression tests for 0x2223 AppHeader decoder (probe #4.1).

Antibody coverage locked here:

- #1 strict mode — bogus flag value raises ChunkPayloadError.
- #2 byte-count — decoded payload must be exactly APP_HEADER_SIZE (112);
  decoder raises if not. Also: inner-size field inside the payload must
  equal APP_HEADER_SIZE (format-drift canary).
- #6 compression: decompSize/compSize cross-check — corrupt zlib or
  truncated payload surfaces as ChunkPayloadError, not silent garbage.

FNAF 1 ground-truth invariants locked here:

- window = 1280×720, frame_rate = 60, number_of_frames = 17
  (matches probe #3's 17×0x3333 Frame count — a cross-chunk invariant).
- graphics_mode = 4 ("16 million colors").
- initial_score = 0, initial_lives = 3.
- flags include MultiSamples, Protected, OneFile, DontDisplayMenu
  (all expected for a packaged single-exe fullscreen-capable game).
- other_flags include Direct3D9or11 (FNAF 1 uses D3D9).
- controls_raw is 72 bytes and all four players share the same
  keyboard-type + default keymap (verifying the CTFAK-interleaved
  layout choice in decoders/header.py).

These values came from the actual binary; if any ever changes, either the
binary or the parser drifted. The cross-chunk antibody
(number_of_frames == count(0x3333)) is the load-bearing one — it's the
first signal that our parser's decode agrees with its own walk.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import ChunkRecord, walk_chunks
from fnaf_parser.compression import ChunkPayloadError, read_chunk_payload
from fnaf_parser.decoders.header import (
    APP_HEADER_SIZE,
    HeaderDecodeError,
    decode_header,
)
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"

_HEADER_CHUNK_ID = 0x2223


def _read_fnaf_header():
    blob = FNAF_EXE.read_bytes()
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    record = next(r for r in result.records if r.id == _HEADER_CHUNK_ID)
    payload = read_chunk_payload(blob, record)
    return record, payload, decode_header(payload)


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_header_decompresses_to_112_bytes():
    """Antibody #2 at the chunk level: decompressed payload must equal spec."""
    _, payload, _ = _read_fnaf_header()
    assert len(payload) == APP_HEADER_SIZE == 112


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_header_window_and_frame_rate():
    _, _, hdr = _read_fnaf_header()
    # Window is the visible render resolution, not some internal logical size.
    assert hdr.window_width == 1280
    assert hdr.window_height == 720
    assert hdr.frame_rate == 60


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_header_frame_count_matches_frame_chunks():
    """Cross-chunk antibody: AppHeader.numberOfFrames must equal the number
    of 0x3333 Frame chunks the walker actually saw. If they disagree,
    either the walker missed a Frame or the header is wrong — both
    important to know before we decode any frame payloads."""
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    frame_count = sum(1 for r in result.records if r.id == 0x3333)

    _, _, hdr = _read_fnaf_header()
    assert hdr.number_of_frames == frame_count == 17


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_header_runtime_flags():
    """Confirm the headline flags every FNAF 1 build has. Catches a
    catastrophic off-by-n in the BitFlags layout (if bits shifted, these
    names wouldn't all be simultaneously set)."""
    _, _, hdr = _read_fnaf_header()
    flags = hdr.flags.set_flags
    # These four flags are characteristic of a packaged, MSAA-capable,
    # protected single-exe build. If any drop out we want to know.
    for expected in ("DontDisplayMenu", "MultiSamples", "Protected", "OneFile"):
        assert expected in flags, f"expected flag {expected!r} missing from {flags}"

    # FNAF 1 ships with D3D9or11 enabled (and DirectX enabled in other_flags
    # under Anaconda's naming — under CTFAK's naming, Unknown5 is set too).
    assert "Direct3D9or11" in hdr.other_flags.set_flags


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_header_inner_size_matches_spec():
    """The redundant inner size field inside the payload is the format-drift
    canary. CTFAK/Anaconda both expect 112; if FNAF 1 ever reports
    something else the spec moved and we want to fail loudly."""
    _, _, hdr = _read_fnaf_header()
    assert hdr.inner_size == APP_HEADER_SIZE


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_header_defaults_for_score_and_lives():
    _, _, hdr = _read_fnaf_header()
    assert hdr.initial_score == 0
    assert hdr.initial_lives == 3


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_header_controls_uniform_keyboard():
    """All four players should share one keyboard mapping. Also encodes the
    empirical resolution of REFERENCE DISAGREEMENT #1 (see decoders/header.py):
    the 72 bytes are laid out as 4×int16 controlType THEN 4×16-byte keys
    blocks (Anaconda's separated layout), not CTFAK's interleaved
    [type,keys,type,keys,...]. Under interleaved slicing these bytes would
    NOT produce four identical players."""
    _, _, hdr = _read_fnaf_header()
    assert len(hdr.controls_raw) == 72

    # Four int16 controlType values at bytes 0..7, all Keyboard (=5).
    types = [hdr.controls_raw[i : i + 2] for i in range(0, 8, 2)]
    assert types == [b"\x05\x00"] * 4

    # Four 16-byte keys blocks at bytes 8..72, all identical.
    key_blocks = [hdr.controls_raw[8 + i : 8 + i + 16] for i in range(0, 64, 16)]
    assert key_blocks[0] == key_blocks[1] == key_blocks[2] == key_blocks[3]

    # Sanity-check a couple of VK codes in the shared block — arrow keys are
    # the characteristic FNAF-1 default and would not appear here under any
    # wrong layout interpretation. VK_UP=0x26, VK_DOWN=0x28, VK_LEFT=0x25,
    # VK_RIGHT=0x27 stored as int16 LE.
    first_four_keys = struct.unpack_from("<4H", key_blocks[0])
    assert first_four_keys == (0x26, 0x28, 0x25, 0x27)


def test_decoder_rejects_wrong_length_payload():
    """Antibody #2: length mismatch must raise, never return truncated data."""
    short = b"\x00" * (APP_HEADER_SIZE - 1)
    with pytest.raises(HeaderDecodeError, match="byte count"):
        decode_header(short)

    long = b"\x00" * (APP_HEADER_SIZE + 1)
    with pytest.raises(HeaderDecodeError, match="byte count"):
        decode_header(long)


def test_decoder_rejects_inner_size_drift():
    """Inner `size` field must equal APP_HEADER_SIZE — canary for format drift."""
    # Build a minimum-valid 112-byte payload but with wrong inner size.
    payload = bytearray(APP_HEADER_SIZE)
    struct.pack_into("<i", payload, 0, 999)  # wrong inner size
    with pytest.raises(HeaderDecodeError, match="Format drift"):
        decode_header(bytes(payload))


def test_payload_reader_rejects_bad_flags():
    """Antibody #1 spirit in the compression layer: unknown flag values
    (4..0xFFFF) must raise rather than silently return raw bytes."""
    record = ChunkRecord(id=0x2223, flags=0x0099, size=0, offset=0)
    blob = b"\x00" * 32
    with pytest.raises(ChunkPayloadError, match="unknown.*flags"):
        read_chunk_payload(blob, record)


def test_payload_reader_rejects_truncated_compressed_payload():
    """Corrupt/truncated compressed payloads must not silently return an
    empty or wrong-sized byte string."""
    # Record claims size=4, but a "Compressed" chunk needs >= 8 bytes just
    # for the decompSize+compSize header.
    record = ChunkRecord(id=0x2223, flags=0x0001, size=4, offset=0)
    blob = b"\x00" * 16
    with pytest.raises(ChunkPayloadError, match="Compressed but payload"):
        read_chunk_payload(blob, record)
