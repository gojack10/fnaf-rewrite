"""Regression tests for 0x3337 FramePalette decoder (probe #4.6).

Antibody coverage:

- #2 byte-count — payload must be exactly 1028 bytes (4 reserved + 256*4).
- #5 multi-input — the same decoder runs against all 17 FNAF 1 palettes.
- #7 snapshot  — per-frame (entries_sha256, entry[0], entry[255]) tuples
  pinned below. The hash locks the full 1024-byte entry table so any
  one-byte drift surfaces immediately; entry[0] and entry[255] are
  human-readable sentinels that make the diagnostic obvious when a
  hash changes.
- #4 multi-oracle — every frame's palette arrives via the flag=3
  decrypt+decompress path (probe #4.5) on a channel independent of the
  flag=3 FrameHeader. If the RC4 key stream drifts by one byte, the
  1028-byte byte-count antibody fires for some frames and the entry
  snapshots drift for the rest. A silent false-positive requires the
  decrypt path to land the exact same 1024 palette bytes across 17
  frames plus the 4 reserved bytes — vanishingly unlikely.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.frame import decode_frame
from fnaf_parser.decoders.frame_palette import (
    FRAME_PALETTE_ENTRY_COUNT,
    FRAME_PALETTE_SIZE,
    FramePalette,
    FramePaletteDecodeError,
    decode_frame_palette,
)
from fnaf_parser.decoders.header import decode_header
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


def _fnaf1_frames_with_palette():
    """Walk FNAF 1 end-to-end with transform enabled and return the list
    of decoded Frame objects in pack order. Mirrors the helper in
    test_frame_header.py so the two cross-channel antibodies share
    exactly the same decrypt path — if one drifts, both drift.
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

    frame_records = [r for r in result.records if r.id == 0x3333]
    frames = [
        decode_frame(
            read_chunk_payload(blob, rec),
            unicode=result.header.unicode,
            transform=transform,
        )
        for rec in frame_records
    ]
    return frames


# --- Synthetic tests (no binary needed) ----------------------------------


def test_decode_frame_palette_exact_size_required():
    """Antibody #2: any byte count other than 1028 raises loudly."""
    with pytest.raises(FramePaletteDecodeError, match="expected 1028 bytes"):
        decode_frame_palette(b"\x00" * (FRAME_PALETTE_SIZE - 1))
    with pytest.raises(FramePaletteDecodeError, match="expected 1028 bytes"):
        decode_frame_palette(b"\x00" * (FRAME_PALETTE_SIZE + 1))


def test_decode_frame_palette_roundtrip_synthetic():
    """Pack a recognisable 256-entry gradient and verify byte order,
    entry count, and the reserved prefix all come back exactly. Cheap
    guard against R/G/B/A swap or off-by-one slicing."""
    reserved = bytes([0xDE, 0xAD, 0xBE, 0xEF])
    entry_bytes = bytearray()
    for i in range(FRAME_PALETTE_ENTRY_COUNT):
        # R=i, G=255-i, B=(i*2) & 0xFF, A=0 — each byte position has a
        # distinct pattern so a byte-order bug shows up on the first
        # assert.
        entry_bytes.extend([i, 255 - i, (i * 2) & 0xFF, 0])
    payload = reserved + bytes(entry_bytes)
    assert len(payload) == FRAME_PALETTE_SIZE

    pal = decode_frame_palette(payload)
    assert isinstance(pal, FramePalette)
    assert pal.reserved == reserved
    assert len(pal.entries) == FRAME_PALETTE_ENTRY_COUNT
    assert pal.entries[0] == (0, 255, 0, 0)
    assert pal.entries[1] == (1, 254, 2, 0)
    assert pal.entries[128] == (128, 127, 0, 0)     # (128*2) & 0xFF == 0
    assert pal.entries[255] == (255, 0, (255 * 2) & 0xFF, 0)


def test_decode_frame_palette_as_dict_shape():
    """as_dict output is stable JSON — snapshot tests rely on this shape."""
    reserved = bytes([1, 2, 3, 4])
    payload = reserved + bytes(
        [(i, i, i, 0xFF)[k] for i in range(256) for k in range(4)]
    )
    d = decode_frame_palette(payload).as_dict()
    assert d["reserved"] == "01020304"
    assert d["entry_count"] == 256
    assert len(d["entries"]) == 256
    assert d["entries"][0] == {"r": 0, "g": 0, "b": 0, "a": 0xFF}
    assert d["entries"][255] == {"r": 255, "g": 255, "b": 255, "a": 0xFF}


# --- FNAF 1 end-to-end antibodies (require binary) -----------------------


def _entries_to_bytes(pal: FramePalette) -> bytes:
    buf = bytearray()
    for r, g, b, a in pal.entries:
        buf.extend([r, g, b, a])
    return bytes(buf)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_has_palette_and_size():
    """Antibody #5 multi-input: the palette decoder fires 17 times and
    every palette reports exactly 256 entries. If the byte-count
    antibody in decode_frame_palette catches a bad size, this test
    fails via an uncaught FramePaletteDecodeError inside decode_frame.
    """
    frames = _fnaf1_frames_with_palette()
    assert len(frames) == 17
    for i, f in enumerate(frames):
        assert f.palette is not None, (
            f"frame #{i} ({f.name!r}) produced no palette — 0x3337 "
            f"either missing from sub_records or decoder did not run."
        )
        assert len(f.palette.entries) == FRAME_PALETTE_ENTRY_COUNT


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_palette_snapshot():
    """Snapshot antibody (#7) pinning every palette.

    Locks per-frame (entries_sha256, entry[0], entry[255]). The hash
    covers the full 1024-byte entry table; entry[0] and entry[255] are
    human-readable sentinels so a drift message names concrete colours
    instead of just "hash mismatch". 17 frames × 3 pinned values = 51
    independent fields that all drift together if the RC4 stream slips
    by a single byte.

    Snapshots captured from commit 2de6360 with probe #4.5 decrypt
    path. Re-pin only on an intentional decoder change.
    """
    frames = _fnaf1_frames_with_palette()
    expected = _FNAF1_PALETTE_SNAPSHOT
    assert len(frames) == len(expected)
    for i, (f, exp) in enumerate(zip(frames, expected)):
        exp_name, exp_sha, exp_first, exp_last = exp
        assert f.name == exp_name, (
            f"frame #{i}: name {f.name!r} != expected {exp_name!r}"
        )
        assert f.palette is not None, f"frame #{i} missing palette"
        sha = _sha256_hex(_entries_to_bytes(f.palette))
        first = f.palette.entries[0]
        last = f.palette.entries[-1]
        assert (sha, first, last) == (exp_sha, exp_first, exp_last), (
            f"frame #{i} ({f.name!r}) palette drifted:\n"
            f"  got  sha={sha} first={first} last={last}\n"
            f"  want sha={exp_sha} first={exp_first} last={exp_last}"
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_all_frame_palettes_identical():
    """Cross-frame consistency antibody (#4 multi-oracle, second channel).

    Empirical: every FNAF 1 frame ships the same 1024-byte palette
    (the default Windows 256-colour table). FNAF 1 renders in true
    colour so the palette is vestigial — Clickteam emits the default
    table unconditionally. That makes "all 17 sha256 hashes equal" a
    highly load-bearing antibody:

    - Any RC4 key-stream drift that desynchronises one frame changes
      its palette bytes but leaves the others intact → set of unique
      hashes becomes >1 → this test fires.
    - A wrong byte-order in the decoder would produce a self-consistent
      but mangled palette everywhere. The snapshot test above catches
      that case.

    The two antibodies together cover both symmetric and asymmetric
    decode failures.
    """
    frames = _fnaf1_frames_with_palette()
    hashes = {
        _sha256_hex(_entries_to_bytes(f.palette))
        for f in frames
        if f.palette is not None
    }
    assert len(hashes) == 1, (
        f"expected all 17 FNAF 1 palettes to be byte-identical, got "
        f"{len(hashes)} distinct palettes: {hashes}"
    )
    assert hashes == {_FNAF1_DEFAULT_PALETTE_SHA}, (
        f"FNAF 1 default palette hash drifted: got {hashes}, "
        f"expected {{{_FNAF1_DEFAULT_PALETTE_SHA}}}"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_frame_palette_reserved_bytes_consistent():
    """The 4-byte reserved prefix is Anaconda's "XXX figure this out"
    region. If Clickteam's format version matches across frames we
    expect the reserved bytes to be consistent — either identical, or
    drawn from a small known set. Pin the observation so a future
    format change announces itself."""
    frames = _fnaf1_frames_with_palette()
    reserved_values = {f.palette.reserved.hex() for f in frames if f.palette}
    # Discovery: all 17 should converge to one or two values. Pin the
    # set below once the first run reports them; a surprise extra value
    # means something shifted in the decrypt path or the format.
    assert reserved_values == _FNAF1_PALETTE_RESERVED_SET, (
        f"FramePalette reserved bytes drifted: got {reserved_values}, "
        f"expected {_FNAF1_PALETTE_RESERVED_SET}"
    )


# --- Empirical snapshots (captured from probe #4.5 decrypt path) ---------

# Observation: all 17 FNAF 1 frame palettes are byte-identical. FNAF 1
# renders in true colour (AppHeader.unicode + 24-bit sprites) so this
# palette is vestigial — Clickteam appears to emit the default Windows
# 256-colour palette for every frame regardless of use. That alone is
# a powerful antibody: any single-byte RC4 drift would desynchronise at
# least one frame, and the "all identical" invariant would fire.
#
# entry[0] = (0, 0, 0, 0) — black    (low end of default palette)
# entry[255] = (255, 255, 255, 0) — white (high end of default palette)
_FNAF1_DEFAULT_PALETTE_SHA = (
    "1e1e0c1f8507edf45439ca5b29fddf81ae9d103d034c05f9dd09485a9786080c"
)
_FNAF1_DEFAULT_PALETTE_ENTRY0 = (0, 0, 0, 0)
_FNAF1_DEFAULT_PALETTE_ENTRY255 = (255, 255, 255, 0)

_FNAF1_PALETTE_SNAPSHOT: tuple[
    tuple[str, str, tuple[int, int, int, int], tuple[int, int, int, int]], ...
] = tuple(
    (
        name,
        _FNAF1_DEFAULT_PALETTE_SHA,
        _FNAF1_DEFAULT_PALETTE_ENTRY0,
        _FNAF1_DEFAULT_PALETTE_ENTRY255,
    )
    for name in (
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
    )
)

# Reserved prefix is `00 03 00 01` across every frame. Read as a
# little-endian u32 that's 0x01000300. Best-guess semantic: (u16
# legacy_rgb_size=0x0300 = 256*3, u16 format_version=0x0001). Not
# load-bearing for this probe — just pinned so a format-version bump
# in a later game announces itself.
_FNAF1_PALETTE_RESERVED_SET: frozenset[str] = frozenset({"00030001"})
