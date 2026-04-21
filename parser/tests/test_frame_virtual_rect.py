"""Regression tests for 0x3342 FrameVirtualRect decoder (probe #4.7).

Antibody coverage:

- #2 byte-count - payload must be exactly 16 bytes.
- #3 round-trip - synthetic pack/unpack verifies byte order.
- #4 multi-oracle - 16 of 17 frames should carry a virtual rect with
  `(width, height)` equal to AppHeader.window_width/height. AppHeader
  arrives through a flag=1 (zlib only) chunk; VirtualRect arrives
  through a flag=3 (zlib + RC4) sub-chunk nested inside a flag=0
  Frame container. Independent channels - any RC4 drift desynchronises
  all 17 rects at once.
- #5 multi-input - the same decoder runs against all 17 FNAF 1 frame
  virtual rects.
- #6 loud-skip - inherited from the frame walker; deferred_encrypted
  must stay empty once a transform is supplied.
- #7 snapshot - per-frame (left, top, right, bottom) tuples pinned
  below. 17 frames * 4 int32s = 68 independent byte-patterns any one
  of which breaks under a wrong-byte decrypt.

Third independent flag=3 shape after FrameHeader (16 B) and
FramePalette (1028 B). Three distinct shapes all passing clean means
the RC4 plumbing is not overfit to any single decoder.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.frame import decode_frame
from fnaf_parser.decoders.frame_virtual_rect import (
    FRAME_VIRTUAL_RECT_SIZE,
    FrameVirtualRect,
    FrameVirtualRectDecodeError,
    decode_frame_virtual_rect,
)
from fnaf_parser.decoders.header import decode_header
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


def _fnaf1_app_header_and_frames():
    """Walk FNAF 1 end-to-end with transform enabled; return the decoded
    AppHeader (for the cross-channel window dims) and the list of
    decoded Frame objects in pack order. Mirrors the helpers in
    test_frame_header.py / test_frame_palette.py so all three cross-
    channel antibodies share exactly the same decrypt path - if one
    drifts, they all drift.
    """
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
    return app_header, frames


# --- Synthetic tests (no binary needed) ----------------------------------


def test_decode_frame_virtual_rect_exact_size_required():
    """Antibody #2: any byte count other than 16 raises loudly."""
    with pytest.raises(FrameVirtualRectDecodeError, match="expected 16 bytes"):
        decode_frame_virtual_rect(b"\x00" * (FRAME_VIRTUAL_RECT_SIZE - 1))
    with pytest.raises(FrameVirtualRectDecodeError, match="expected 16 bytes"):
        decode_frame_virtual_rect(b"\x00" * (FRAME_VIRTUAL_RECT_SIZE + 1))


def test_decode_frame_virtual_rect_roundtrip_synthetic():
    """Antibody #3: pack known fields, verify every one returns exactly.
    Signed negatives on `left`/`top` exercise the int32-signed decode
    path that a uint32 bug would silently mangle."""
    payload = struct.pack("<iiii", -10, -20, 1280, 720)
    assert len(payload) == FRAME_VIRTUAL_RECT_SIZE
    rect = decode_frame_virtual_rect(payload)
    assert isinstance(rect, FrameVirtualRect)
    assert rect.left == -10
    assert rect.top == -20
    assert rect.right == 1280
    assert rect.bottom == 720
    assert rect.width == 1290
    assert rect.height == 740


def test_decode_frame_virtual_rect_as_dict_shape():
    """as_dict output is stable JSON - snapshot tests rely on this shape."""
    payload = struct.pack("<iiii", 0, 0, 1600, 720)
    d = decode_frame_virtual_rect(payload).as_dict()
    assert d == {
        "left": 0,
        "top": 0,
        "right": 1600,
        "bottom": 720,
        "width": 1600,
        "height": 720,
    }


# --- FNAF 1 end-to-end antibodies (require binary) -----------------------


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_has_virtual_rect():
    """Antibody #5 multi-input: the virtual-rect decoder fires 17 times
    and every frame emits a VirtualRect. If the byte-count antibody in
    decode_frame_virtual_rect catches a bad size, this test fails via
    an uncaught FrameVirtualRectDecodeError inside decode_frame.
    """
    _, frames = _fnaf1_app_header_and_frames()
    assert len(frames) == 17
    for i, f in enumerate(frames):
        assert f.virtual_rect is not None, (
            f"frame #{i} ({f.name!r}) produced no virtual_rect - 0x3342 "
            f"either missing from sub_records or decoder did not run."
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_virtual_rect_snapshot():
    """Snapshot antibody (#7) pinning every virtual rect.

    Locks per-frame (name, left, top, right, bottom). 17 frames *
    4 int32 fields = 68 independent byte-patterns any one of which
    breaks under a wrong-byte decrypt.
    """
    _, frames = _fnaf1_app_header_and_frames()
    expected = _FNAF1_VIRTUAL_RECT_SNAPSHOT
    assert len(frames) == len(expected)
    for i, (f, exp) in enumerate(zip(frames, expected)):
        exp_name, exp_l, exp_t, exp_r, exp_b = exp
        assert f.name == exp_name, (
            f"frame #{i}: name {f.name!r} != expected {exp_name!r}"
        )
        assert f.virtual_rect is not None, f"frame #{i} missing virtual_rect"
        got = (
            f.virtual_rect.left,
            f.virtual_rect.top,
            f.virtual_rect.right,
            f.virtual_rect.bottom,
        )
        want = (exp_l, exp_t, exp_r, exp_b)
        assert got == want, (
            f"frame #{i} ({f.name!r}) virtual rect drifted: "
            f"got {got}, want {want}"
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_frame_virtual_rect_matches_app_window():
    """Cross-channel antibody (#4 multi-oracle).

    AppHeader.window_width/height arrive through a flag=1 (zlib-only)
    chunk; VirtualRect width/height arrive through a flag=3 (zlib+RC4)
    sub-chunk nested in a flag=0 container. Independent channels, so
    this equality catches RC4 drift that a single-channel test would
    miss. Office frame ('Frame 1') is intentionally wider for camera
    panning - so we assert the *modal* dimension equals the window.
    """
    app_header, frames = _fnaf1_app_header_and_frames()
    window = (app_header.window_width, app_header.window_height)
    dims = [
        (f.virtual_rect.width, f.virtual_rect.height)
        for f in frames
        if f.virtual_rect is not None
    ]
    assert len(dims) == 17
    matches = dims.count(window)
    assert matches == 16, (
        f"expected 16 of 17 virtual rects at AppHeader window size "
        f"{window}, got {matches}. All dims: {dims}"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_frame_virtual_rect_origins_are_zero():
    """Every FNAF 1 virtual rect should sit at origin (0, 0). Locked as
    a regression so if a future file with a non-zero origin ever lands
    the drift is obvious. Also implicitly verifies the decoder is not
    misaligned by 4 bytes (which would shuffle left<->top<->right<->bottom).
    """
    _, frames = _fnaf1_app_header_and_frames()
    origins = [
        (f.virtual_rect.left, f.virtual_rect.top)
        for f in frames
        if f.virtual_rect is not None
    ]
    assert origins == [(0, 0)] * 17, (
        f"expected every FNAF 1 virtual rect at origin (0, 0); got {origins}"
    )


# --- Empirical snapshots (captured from probe #4.5 decrypt path) ---------

# FrameHeader (probe #4.5) already locked 16 frames at 1280x720 and one
# office frame ('Frame 1') at 1600x720. FrameVirtualRect should mirror
# those dimensions with origin (0, 0) per Clickteam convention. Pinned
# below in pack order so the multi-oracle cross-check has a third
# independent anchor against the same numbers.

_FNAF1_VIRTUAL_RECT_SNAPSHOT: tuple[
    tuple[str, int, int, int, int], ...
] = (
    ("Frame 17",     0, 0, 1280, 720),
    ("title",        0, 0, 1280, 720),
    ("what day",     0, 0, 1280, 720),
    ("Frame 1",      0, 0, 1600, 720),
    ("died",         0, 0, 1280, 720),
    ("freddy",       0, 0, 1280, 720),
    ("next day",     0, 0, 1280, 720),
    ("wait",         0, 0, 1280, 720),
    ("gameover",     0, 0, 1280, 720),
    ("the end",      0, 0, 1280, 720),
    ("ad",           0, 0, 1280, 720),
    ("the end 2",    0, 0, 1280, 720),
    ("customize",    0, 0, 1280, 720),
    ("the end 3",    0, 0, 1280, 720),
    ("creepy start", 0, 0, 1280, 720),
    ("creepy end",   0, 0, 1280, 720),
    ("end of demo",  0, 0, 1280, 720),
)
