"""Regression tests for 0x3334 FrameHeader decoder (probe #4.5).

Antibody coverage:

- #2 byte-count — payload must be exactly 16 bytes.
- #4 multi-oracle — 16 of 17 decrypted FrameHeaders match the
  AppHeader's window_width/window_height. Those two values arrive
  through **completely independent channels**: AppHeader is a flag=1
  (zlib only) outer chunk, FrameHeader is a flag=3 (zlib + RC4) sub-
  chunk nested inside a flag=1 Frame container. If the decryption
  stack (key derivation → S-box KSA → RC4 PRGA) drifts by a single
  byte, the decoded FrameHeader dimensions become garbage. The
  exception is the gameplay office frame, wider than the window to
  accommodate FNAF 1's camera-pan mechanic — locked by name below.
- #5 multi-input — the same FrameHeader decoder runs against all 17
  FNAF 1 frame headers; each must report a plausible value.
- #7 snapshot — the per-frame (name, width, height, background, flags)
  tuples are pinned below; any drift in the decrypt pipeline turns
  multiple values to garbage simultaneously, and pinning all four
  fields per frame makes a false pass vanishingly unlikely.

FNAF 1 ground truth (empirical, cross-referenced with test_header.py's
window = 1280x720):
    16 frames at 1280x720 (title, day-select, office cams, etc.)
    1 frame  at 1600x720 ('Frame 1' — the office, panned wide)
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.frame import decode_frame
from fnaf_parser.decoders.frame_header import (
    FRAME_HEADER_SIZE,
    FrameHeader,
    FrameHeaderDecodeError,
    decode_frame_header,
)
from fnaf_parser.decoders.header import decode_header
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"

_FNAF1_FRAME_WIDTH = 1280
_FNAF1_FRAME_HEIGHT = 720

# Per-frame ground truth for FNAF 1 (empirical, 17 frames in pack order).
# Tuple: (name, width, height, background, flags).
# flags=0x00008020 = HandleCollision (bit 5) + TimeMovements (bit 15), uniform
# across every frame. Backgrounds are pure black or pure white with alpha=0.
_FNAF1_EXPECTED_FRAMES: tuple[
    tuple[str, int, int, tuple[int, int, int, int], int], ...
] = (
    ("Frame 17",     1280, 720, (0, 0, 0, 0),           0x00008020),
    ("title",        1280, 720, (0, 0, 0, 0),           0x00008020),
    ("what day",     1280, 720, (0, 0, 0, 0),           0x00008020),
    ("Frame 1",      1600, 720, (0, 0, 0, 0),           0x00008020),
    ("died",         1280, 720, (0, 0, 0, 0),           0x00008020),
    ("freddy",       1280, 720, (0, 0, 0, 0),           0x00008020),
    ("next day",     1280, 720, (0, 0, 0, 0),           0x00008020),
    ("wait",         1280, 720, (0, 0, 0, 0),           0x00008020),
    ("gameover",     1280, 720, (0, 0, 0, 0),           0x00008020),
    ("the end",      1280, 720, (255, 255, 255, 0),     0x00008020),
    ("ad",           1280, 720, (0, 0, 0, 0),           0x00008020),
    ("the end 2",    1280, 720, (255, 255, 255, 0),     0x00008020),
    ("customize",    1280, 720, (0, 0, 0, 0),           0x00008020),
    ("the end 3",    1280, 720, (255, 255, 255, 0),     0x00008020),
    ("creepy start", 1280, 720, (255, 255, 255, 0),     0x00008020),
    ("creepy end",   1280, 720, (0, 0, 0, 0),           0x00008020),
    ("end of demo",  1280, 720, (0, 0, 0, 0),           0x00008020),
)


def _fnaf1_transform_and_frames():
    """Walk FNAF 1, build the pack-level TransformState, decode every
    frame with decryption enabled. Also returns the decoded AppHeader
    so tests can cross-check FrameHeader dimensions against the window
    size from an independent channel.
    """
    blob = FNAF_EXE.read_bytes()
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)

    # Pull the three seed strings from the pack itself — no hard-coded
    # paths. This way a future binary with a different editor path
    # still decrypts, and we exercise the full app-metadata→transform
    # pipeline every test run.
    def _str_of(chunk_id: int) -> str:
        rec = next(r for r in result.records if r.id == chunk_id)
        return decode_string_chunk(
            read_chunk_payload(blob, rec), unicode=result.header.unicode
        )

    editor = _str_of(0x222E)
    name = _str_of(0x2224)
    # 0x223B (Copyright) is absent in FNAF 1 — empty string per Anaconda
    # create_transform's None-guard.
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

    app_header_rec = next(r for r in result.records if r.id == 0x2223)
    app_header = decode_header(read_chunk_payload(blob, app_header_rec))

    frame_records = [r for r in result.records if r.id == 0x3333]
    frames = [
        decode_frame(
            read_chunk_payload(blob, rec),
            unicode=result.header.unicode,
            transform=transform,
        )
        for rec in frame_records
    ]
    return transform, app_header, frames


# --- Synthetic tests (no binary needed) ----------------------------------


def test_decode_frame_header_exact_size_required():
    """Antibody #2: any byte count other than 16 raises loudly."""
    with pytest.raises(FrameHeaderDecodeError, match="expected 16 bytes"):
        decode_frame_header(b"\x00" * 15)
    with pytest.raises(FrameHeaderDecodeError, match="expected 16 bytes"):
        decode_frame_header(b"\x00" * 17)


def test_decode_frame_header_roundtrip_synthetic():
    """Pack a known FrameHeader and verify every field comes back
    exactly. Cheap guard against endian / field-order drift."""
    payload = struct.pack(
        "<ii4BI",
        640,                 # width
        480,                 # height
        0x11, 0x22, 0x33, 0,  # R, G, B, A
        0x00010005,          # flags
    )
    assert len(payload) == FRAME_HEADER_SIZE
    header = decode_frame_header(payload)
    assert isinstance(header, FrameHeader)
    assert header.width == 640
    assert header.height == 480
    assert header.background == (0x11, 0x22, 0x33, 0)
    assert header.flags == 0x00010005
    # DisplayTitle (bit 0) + HandleCollision (bit 5) + bit 16 are the bits
    # set in 0x00010005 + bit 2 = 0x00010005 & 0b ... let's just compute.
    # flags=0x00010005 → bits 0, 2, 16
    names = set(header.flag_names_set())
    assert "DisplayTitle" in names
    assert "KeepDisplay" in names
    assert "Bit16" in names


# --- FNAF 1 end-to-end antibody (requires binary) ------------------------


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_header_snapshot():
    """Snapshot antibody (#7) pinning all 17 FrameHeader tuples.

    Locks (width, height, background, flags) per frame. This single
    assertion exercises every moving part:
    - string decoder (three seed chunks) must be correct
    - keystring + MakeKeyCombined must be correct
    - InitDecryptionTable KSA must be correct
    - TransformChunk RC4 PRGA must be correct
    - flag=3 size framing (decompSize / compSize / zlib) must be correct
    - FrameHeader struct layout must be correct

    17 frames × 4 pinned fields = 68 independent byte-patterns any one
    of which would break under a wrong-byte decrypt. A false positive
    is vanishingly unlikely.
    """
    _, _, frames = _fnaf1_transform_and_frames()
    assert len(frames) == len(_FNAF1_EXPECTED_FRAMES)
    for i, (f, expected) in enumerate(zip(frames, _FNAF1_EXPECTED_FRAMES)):
        exp_name, exp_w, exp_h, exp_bg, exp_flags = expected
        assert f.name == exp_name, (
            f"frame #{i}: name {f.name!r} != expected {exp_name!r}"
        )
        header_rec = next(
            (r for r in f.sub_records if r.id == 0x3334), None
        )
        assert header_rec is not None, f"frame #{i} missing FrameHeader"
        assert header_rec.decoded_payload is not None, (
            f"frame #{i} FrameHeader still encrypted — transform did not "
            f"reach the Frame container"
        )
        fh = decode_frame_header(header_rec.decoded_payload)
        assert (fh.width, fh.height, fh.background, fh.flags) == (
            exp_w, exp_h, exp_bg, exp_flags
        ), (
            f"frame #{i} ({f.name!r}) drifted: "
            f"got {fh.width}x{fh.height} bg={fh.background} "
            f"flags=0x{fh.flags:08X}, expected {exp_w}x{exp_h} "
            f"bg={exp_bg} flags=0x{exp_flags:08X}"
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_frame_header_majority_matches_app_window():
    """Cross-channel antibody (#4 multi-oracle).

    AppHeader.window_width/height arrive through a flag=1 (zlib-only)
    chunk; FrameHeader width/height arrive through a flag=3 (zlib+RC4)
    sub-chunk nested in a flag=1 container. They share nothing but the
    underlying binary. Most frames render at the window size, with the
    office frame ('Frame 1') intentionally wider for camera panning —
    so we assert the *modal* resolution equals the window.
    """
    _, app_header, frames = _fnaf1_transform_and_frames()
    dims = []
    for f in frames:
        hr = next(r for r in f.sub_records if r.id == 0x3334)
        fh = decode_frame_header(hr.decoded_payload)
        dims.append((fh.width, fh.height))
    window = (app_header.window_width, app_header.window_height)
    matches = dims.count(window)
    assert matches == 16, (
        f"expected 16 of 17 frames at AppHeader window size "
        f"{window}, got {matches}. All frame dims: {dims}"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_transform_empties_deferred_encrypted():
    """Once a transform is supplied, deferred_encrypted must be empty for
    every frame — no encrypted sub-chunk should slip through the
    decrypt path."""
    _, _, frames = _fnaf1_transform_and_frames()
    for i, f in enumerate(frames):
        assert f.deferred_encrypted == (), (
            f"frame #{i} ({f.name!r}) still has {len(f.deferred_encrypted)} "
            f"deferred_encrypted entries despite transform being supplied"
        )
