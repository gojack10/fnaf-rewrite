"""Regression tests for 0x3341 FrameLayers decoder (probe #4.8).

Antibody coverage:

- #2 byte-count - payload must consume every byte (u32 count + N
  variable-length layer records). Trailing garbage or short-consume
  raises `FrameLayersDecodeError` loudly.
- #3 round-trip - synthetic pack/unpack in both Unicode and ASCII
  modes verifies each field returns exactly. Signed negative
  `background_index` exercises the int32-signed path (Clickteam uses
  -1 as "none"); a mixed-flags layer (`0x00000013`) mirrors the real
  office-frame parallax flag pattern so the bit ordering is pinned.
- #4 multi-oracle (cross-channel) - the `unicode` flag driving layer-
  name parsing is the same flag AppHeader ships the product name /
  editor name strings under. AppHeader arrives through a flag=1
  (zlib-only) chunk; FrameLayers arrives through a flag=3 (zlib+RC4)
  sub-chunk nested in a flag=0 Frame container. Independent channels -
  any RC4 drift scrambles the layer-name byte pattern and every
  per-frame snapshot breaks at once.
- #5 multi-input - the same decoder runs against all 17 FNAF 1 frames.
- #7 snapshot - per-frame (layer_count, flags tuple, coeffs tuple,
  nbg tuple, first-layer name) pinned below. 16 frames * 1 layer +
  1 office frame * 4 layers = 20 layer records, each with 6 fields,
  so 120 independent byte-patterns any one of which breaks under a
  wrong-byte decrypt.

This is the fourth independent flag=3 shape after FrameHeader (16 B),
FramePalette (1028 B), and FrameVirtualRect (16 B) - and the first
variable-length nested-TLV shape. Every prior post-decrypt decoder
was a fixed-size struct; this one proves the decrypt plumbing is not
overfit to a single payload length. The office frame's 4 layers
carrying the full variety of flag+coefficient combinations (standard
vs parallax) is especially load-bearing: a wrong-byte decrypt would
produce either a bad count (byte-count antibody fires) or scrambled
names (snapshot antibody fires).
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.frame import decode_frame
from fnaf_parser.decoders.frame_layers import (
    FRAME_LAYER_FIXED_SIZE,
    FRAME_LAYERS_COUNT_SIZE,
    FrameLayer,
    FrameLayers,
    FrameLayersDecodeError,
    decode_frame_layers,
)
from fnaf_parser.decoders.header import decode_header
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


def _fnaf1_app_header_and_frames():
    """Walk FNAF 1 end-to-end with transform enabled; return the decoded
    AppHeader (for cross-channel Unicode-flag check) and the list of
    decoded Frame objects in pack order. Mirrors the helpers in
    test_frame_header.py / test_frame_palette.py / test_frame_virtual_rect.py
    so every cross-channel antibody shares exactly the same decrypt path.
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
    return result.header.unicode, app_header, frames


# --- Synthetic helpers ---------------------------------------------------


def _pack_layer_fixed(
    flags: int, x: float, y: float, nbg: int, bgi: int
) -> bytes:
    """Pack the 20-byte fixed layer header exactly as the decoder expects."""
    return struct.pack("<Iffii", flags, x, y, nbg, bgi)


def _pack_name(name: str, *, unicode: bool) -> bytes:
    """Pack a null-terminated layer name in the requested encoding."""
    if unicode:
        return name.encode("utf-16le") + b"\x00\x00"
    return name.encode("ascii") + b"\x00"


def _pack_layers_payload(
    layers: list[tuple[int, float, float, int, int, str]], *, unicode: bool
) -> bytes:
    """Pack a full 0x3341 payload: u32 count + N (fixed + name) records."""
    buf = bytearray()
    buf.extend(len(layers).to_bytes(4, "little", signed=False))
    for flags, x, y, nbg, bgi, name in layers:
        buf.extend(_pack_layer_fixed(flags, x, y, nbg, bgi))
        buf.extend(_pack_name(name, unicode=unicode))
    return bytes(buf)


# --- Synthetic tests (no binary needed) ----------------------------------


def test_frame_layers_module_constants_are_stable():
    """Antibody #2 sanity: if either fixed-size constant ever shifts,
    every byte-count error message in the decoder goes stale. Pin them."""
    assert FRAME_LAYERS_COUNT_SIZE == 4
    assert FRAME_LAYER_FIXED_SIZE == 20


def test_decode_frame_layers_empty_count_consumes_all_bytes():
    """Zero-layer payload is just the u32 count - must decode cleanly and
    consume every byte. Guards against an off-by-one in the count-only
    path."""
    payload = (0).to_bytes(4, "little", signed=False)
    fl = decode_frame_layers(payload, unicode=True)
    assert isinstance(fl, FrameLayers)
    assert fl.count == 0
    assert fl.layers == ()


def test_decode_frame_layers_short_prefix_raises():
    """Antibody #2: payload smaller than the 4-byte count prefix raises."""
    with pytest.raises(FrameLayersDecodeError, match="at least the 4-byte"):
        decode_frame_layers(b"\x00\x00\x00", unicode=True)


def test_decode_frame_layers_short_layer_body_raises():
    """Antibody #2: count claims 1 layer but fewer than 20 fixed bytes
    follow - must raise before attempting to unpack out of bounds."""
    payload = (1).to_bytes(4, "little", signed=False) + b"\x00" * 10
    with pytest.raises(
        FrameLayersDecodeError, match="needs 20 bytes for its fixed header"
    ):
        decode_frame_layers(payload, unicode=True)


def test_decode_frame_layers_trailing_garbage_raises():
    """Antibody #2: decoder consumes exactly `count` layers and then
    must land precisely at end-of-payload. Trailing bytes raise."""
    good = _pack_layers_payload(
        [(0x00000010, 1.0, 1.0, 0, 0, "Layer 1")], unicode=True
    )
    # Tack on a single junk byte - must trip the "trailing bytes
    # unaccounted for" antibody.
    with pytest.raises(
        FrameLayersDecodeError, match="trailing bytes unaccounted for"
    ):
        decode_frame_layers(good + b"\x00", unicode=True)


def test_decode_frame_layers_unicode_missing_terminator_raises():
    """Antibody #2: Unicode name without a wide NUL within the payload
    must raise rather than run off the end silently."""
    # count=1, fixed header, then a single 'A' (2 UTF-16 bytes) with NO
    # trailing \x00\x00 and nothing beyond it.
    payload = (
        (1).to_bytes(4, "little", signed=False)
        + _pack_layer_fixed(0, 1.0, 1.0, 0, 0)
        + "A".encode("utf-16le")
    )
    with pytest.raises(FrameLayersDecodeError, match="not NUL-terminated"):
        decode_frame_layers(payload, unicode=True)


def test_decode_frame_layers_ascii_missing_terminator_raises():
    """Antibody #2 (ASCII path mirror): ASCII name without a trailing
    NUL must raise."""
    payload = (
        (1).to_bytes(4, "little", signed=False)
        + _pack_layer_fixed(0, 1.0, 1.0, 0, 0)
        + b"Layer"
    )
    with pytest.raises(FrameLayersDecodeError, match="not NUL-terminated"):
        decode_frame_layers(payload, unicode=False)


def test_decode_frame_layers_unicode_roundtrip_synthetic():
    """Antibody #3 (Unicode): two layers with distinct flag / coefficient
    / count patterns round-trip through pack + unpack exactly. The
    flags=0x13 + x=0.0 pattern on layer 2 mirrors the real FNAF 1 office
    frame's parallax layers, so a future RC4 drift that happens to clear
    those specific bits is caught here too."""
    specs = [
        (0x00000010, 1.0, 1.0, 0, 0, "Layer 1"),
        (0x00000013, 0.0, 0.0, 1, -1, "Parallax"),
    ]
    payload = _pack_layers_payload(specs, unicode=True)
    fl = decode_frame_layers(payload, unicode=True)
    assert fl.count == 2
    for got, exp in zip(fl.layers, specs):
        f, x, y, n, b, nm = exp
        assert isinstance(got, FrameLayer)
        assert got.flags == f
        assert got.x_coefficient == x
        assert got.y_coefficient == y
        assert got.number_of_backgrounds == n
        assert got.background_index == b
        assert got.name == nm


def test_decode_frame_layers_ascii_roundtrip_synthetic():
    """Antibody #3 (ASCII): same round-trip through the single-byte
    terminator path. FNAF 1 itself is Unicode but the ASCII branch ships
    inside the same module, so it must stay in lockstep with Unicode."""
    specs = [
        (0x00000010, 1.0, 1.0, 0, 0, "Layer 1"),
        (0x00000013, 0.5, 0.5, 2, 7, "Scroll"),
    ]
    payload = _pack_layers_payload(specs, unicode=False)
    fl = decode_frame_layers(payload, unicode=False)
    assert fl.count == 2
    for got, exp in zip(fl.layers, specs):
        f, x, y, n, b, nm = exp
        assert got.flags == f
        assert got.x_coefficient == x
        assert got.y_coefficient == y
        assert got.number_of_backgrounds == n
        assert got.background_index == b
        assert got.name == nm


def test_decode_frame_layers_as_dict_shape():
    """`as_dict` output is stable JSON; snapshot-style tests in other
    modules rely on this shape staying put."""
    payload = _pack_layers_payload(
        [(0x00000013, 0.25, 0.5, 3, -1, "Parallax")], unicode=True
    )
    d = decode_frame_layers(payload, unicode=True).as_dict()
    assert d["count"] == 1
    assert d["layers"] == [
        {
            "flags": "0x00000013",
            "x_coefficient": 0.25,
            "y_coefficient": 0.5,
            "number_of_backgrounds": 3,
            "background_index": -1,
            "name": "Parallax",
        }
    ]


# --- FNAF 1 end-to-end antibodies (require binary) -----------------------


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_has_layers():
    """Antibody #5 multi-input: the FrameLayers decoder fires 17 times
    and every frame emits a FrameLayers. Any byte-count failure in
    `decode_frame_layers` surfaces as an uncaught
    `FrameLayersDecodeError` inside `decode_frame`.
    """
    _, _, frames = _fnaf1_app_header_and_frames()
    assert len(frames) == 17
    for i, f in enumerate(frames):
        assert f.layers is not None, (
            f"frame #{i} ({f.name!r}) produced no layers - 0x3341 "
            f"either missing from sub_records or decoder did not run."
        )
        assert f.layers.count >= 1, (
            f"frame #{i} ({f.name!r}) reported zero layers - every real "
            f"Clickteam frame has at least one."
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_unicode_flag_cross_channel_consistent():
    """Cross-channel antibody (#4 multi-oracle).

    The `unicode` flag that gates FrameLayers name parsing (flag=3
    zlib+RC4 channel) is the same flag AppHeader ships product-name /
    editor-name strings under (flag=1 zlib-only channel). If those two
    channels ever disagree on the flag, every layer name would end up
    mojibake or a size-mismatch error; pin the invariant here so the
    cross-channel dependency is explicit rather than implicit.

    FNAF 1 is a PAMU pack => unicode=True. This test would fail if a
    future regression flipped the flag plumbing on either channel.
    """
    pack_unicode, app_header, frames = _fnaf1_app_header_and_frames()
    assert pack_unicode is True, "FNAF 1 is PAMU, expected unicode=True"
    # AppHeader existing proves the flag=1 channel parsed cleanly.
    assert app_header is not None
    # Every layer name is a real ASCII-safe string ('Layer 1'..'Layer 4'),
    # so if UTF-16LE had decoded as ASCII by mistake, names would contain
    # NUL bytes. Assert the round-trips are clean.
    for f in frames:
        assert f.layers is not None
        for L in f.layers.layers:
            assert "\x00" not in L.name, (
                f"layer name {L.name!r} contains NUL - unicode flag "
                f"inconsistency between channels."
            )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_layers_snapshot():
    """Snapshot antibody (#7) pinning every layer in every frame.

    Locks per-frame (frame_name, [(flags, x, y, nbg, bgi, name), ...]).
    16 single-layer frames + 1 four-layer office frame = 20 layer
    records, each with 6 fields = 120 pinned byte-patterns. Any RC4
    drift that desynchronises a single byte surfaces here.

    The office frame ('Frame 1') carries the full variety of the
    format: standard layers (flags=0x10, coeffs=1.0) alongside
    parallax layers (flags=0x13, coeffs=0.0). That pattern is
    load-bearing - the parallax flag bits on layers 3/4 would be
    the first things to drift under a wrong-byte decrypt.
    """
    _, _, frames = _fnaf1_app_header_and_frames()
    expected = _FNAF1_LAYERS_SNAPSHOT
    assert len(frames) == len(expected)
    for i, (f, exp) in enumerate(zip(frames, expected)):
        exp_name, exp_layers = exp
        assert f.name == exp_name, (
            f"frame #{i}: name {f.name!r} != expected {exp_name!r}"
        )
        assert f.layers is not None, f"frame #{i} missing layers"
        got = tuple(
            (
                L.flags,
                L.x_coefficient,
                L.y_coefficient,
                L.number_of_backgrounds,
                L.background_index,
                L.name,
            )
            for L in f.layers.layers
        )
        assert got == exp_layers, (
            f"frame #{i} ({f.name!r}) layers drifted:\n"
            f"  got  {got}\n"
            f"  want {exp_layers}"
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_office_frame_is_only_multi_layer_frame():
    """Empirical invariant: 'Frame 1' (the office) is the only FNAF 1
    frame with more than one layer - the multi-layer camera-panning
    setup. Every other frame has a single 'Layer 1' sprite atop its
    background. Pin this so a future binary change announces itself.
    """
    _, _, frames = _fnaf1_app_header_and_frames()
    multi_layer = [f.name for f in frames if f.layers and f.layers.count > 1]
    assert multi_layer == ["Frame 1"], (
        f"expected only 'Frame 1' (office) to be multi-layer in FNAF 1, "
        f"got: {multi_layer}"
    )


# --- Empirical snapshots (captured from probe #4.8 decrypt path) ---------

# Pinned in pack order against FrameHeader / FramePalette / FrameVirtualRect
# snapshots. The office frame's 4-layer structure (standard layers 1-2
# + parallax layers 3-4) is the richest single snapshot and catches
# the widest range of decrypt drift modes.

_LayerTuple = tuple[int, float, float, int, int, str]  # flags, x, y, nbg, bgi, name

# Default single-layer pattern used by every non-office frame. Slight
# per-frame variation in `number_of_backgrounds` (ground-truth below)
# prevents a single-value copy-paste shortcut.
def _std(nbg: int) -> tuple[_LayerTuple, ...]:
    return ((0x00000010, 1.0, 1.0, nbg, 0, "Layer 1"),)


_FNAF1_LAYERS_SNAPSHOT: tuple[
    tuple[str, tuple[_LayerTuple, ...]], ...
] = (
    ("Frame 17",     _std(0)),
    ("title",        _std(0)),
    ("what day",     _std(0)),
    ("Frame 1",      (
        (0x00000010, 1.0, 1.0, 0, 0, "Layer 1"),
        (0x00000010, 1.0, 1.0, 0, 0, "Layer 2"),
        (0x00000013, 0.0, 0.0, 0, 0, "Layer 3"),
        (0x00000013, 0.0, 0.0, 0, 0, "Layer 4"),
    )),
    ("died",         _std(0)),
    ("freddy",       _std(0)),
    ("next day",     _std(0)),
    ("wait",         _std(0)),
    ("gameover",     _std(2)),
    ("the end",      _std(1)),
    ("ad",           _std(1)),
    ("the end 2",    _std(1)),
    ("customize",    _std(9)),
    ("the end 3",    _std(1)),
    ("creepy start", _std(1)),
    ("creepy end",   _std(1)),
    ("end of demo",  _std(1)),
)
