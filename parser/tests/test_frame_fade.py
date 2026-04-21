"""Regression tests for 0x333B/0x333C FrameFade decoder (probe #4.11).

First **optional** frame sub-chunk: unlike palette/rect/layers/effects/
instances which every frame ships, only 10 of 17 FNAF 1 frames carry
at least one fade (17 records total across them). Also the first sub-
chunk with a **variable-length trailing string** (after the fixed
32-byte Transition header).

Antibody coverage:

- #1 strict-unknown: offsets outside the payload, negative param_size,
  overlapping / unaccounted-for bytes all raise loudly; missing NUL
  terminator on the module_file string raises loudly.
- #2 byte-count: every payload byte must be covered by exactly one of
  {fixed header, module_file+NUL, parameter_data}. No overlap, no gaps.
- #3 round-trip: synthetic pack/unpack in both FadeIn (flags=0) and
  FadeOut (flags=1) shapes. A third synthetic record exercises a
  non-empty parameterData tail (not hit by FNAF 1 but covered here so
  the variable-tail code path is locked).
- #4 multi-oracle: CTFAK Transition.Read and Anaconda Transition.read
  field orders are identical - verified this probe. Cross-channel:
  the same `unicode` flag that drives AppHeader string decoding drives
  the module_file string here. Any RC4 drift breaks both at once.
- #5 multi-input: 34 fade slots (17 frames x {fade_in, fade_out})
  pass through the same decoder; absent slots resolve to None.
- #7 snapshot: per-frame (name, fade_in_tuple, fade_out_tuple) pinned
  below; covers the 17 `(duration_ms, flags, color, module_file,
  module, name, name_offset, param_offset, param_size, parameter_data)`
  tuples that exist.

Seventh independent flag=3 shape after FrameHeader (16 B), FramePalette
(1028 B), FrameVirtualRect (16 B), FrameLayers (variable w/ string
tail), FrameLayerEffects (parallel array), and FrameItemInstances
(length-prefixed). Second sub-chunk to carry a null-terminated
Unicode string tail (after FrameLayers' per-layer name).
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.frame import decode_frame
from fnaf_parser.decoders.frame_fade import (
    FLAG_COLOR_BIT,
    FRAME_FADE_FIXED_SIZE,
    FrameFade,
    FrameFadeDecodeError,
    decode_frame_fade,
)
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


def _fnaf1_frames():
    """Walk FNAF 1 end-to-end with transform enabled. Mirrors the helper
    in sibling flag=3 tests - any drift in the shared pipeline surfaces
    across all of them at once.
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


# --- Synthetic helpers ---------------------------------------------------


def _pack_fade(
    *,
    module: bytes = b"STDT",
    name: bytes = b"FADE",
    duration_ms: int = 1000,
    flags: int = 0,
    color: bytes = b"\x00\x00\x00\x00",
    module_file: str = "cctrans.dll",
    parameter_data: bytes = b"",
    unicode: bool = True,
    # Escape hatches so tests can fuzz offset fields.
    override_name_offset: int | None = None,
    override_param_offset: int | None = None,
    override_param_size: int | None = None,
) -> bytes:
    """Pack a synthetic fade payload: 32-byte fixed header + module_file
    NUL-terminated string + parameter_data. Offsets default to the
    natural "header, then string, then parameter_data" layout that
    FNAF 1 uses."""
    assert len(module) == 4 and len(name) == 4 and len(color) == 4
    if unicode:
        module_file_bytes = module_file.encode("utf-16le") + b"\x00\x00"
    else:
        module_file_bytes = module_file.encode("ascii") + b"\x00"
    name_offset = FRAME_FADE_FIXED_SIZE if override_name_offset is None else override_name_offset
    param_offset = (
        name_offset + len(module_file_bytes)
        if override_param_offset is None
        else override_param_offset
    )
    param_size = (
        len(parameter_data) if override_param_size is None else override_param_size
    )
    header = struct.pack(
        "<4s4sii4siii",
        module,
        name,
        duration_ms,
        flags,
        color,
        name_offset,
        param_offset,
        param_size,
    )
    return header + module_file_bytes + parameter_data


# --- Synthetic tests (no binary needed) ----------------------------------


def test_module_constants_stable():
    """Antibody #2: the fixed-header size is the load-bearing constant
    in every byte-count error message. Pin it."""
    assert FRAME_FADE_FIXED_SIZE == 32
    assert FLAG_COLOR_BIT == 0x1


def test_roundtrip_fade_in():
    """Antibody #3: FadeIn shape with flags=0, empty parameter_data.
    Exercises the primary FNAF 1 configuration: standard transition,
    no parameter tail, 11-char UTF-16 module_file."""
    payload = _pack_fade(duration_ms=1010, flags=0, module_file="cctrans.dll")
    fade = decode_frame_fade(payload)
    assert isinstance(fade, FrameFade)
    assert fade.module == b"STDT"
    assert fade.name == b"FADE"
    assert fade.duration_ms == 1010
    assert fade.flags == 0
    assert fade.has_color_bit is False
    assert fade.color == (0, 0, 0, 0)
    assert fade.module_file == "cctrans.dll"
    assert fade.parameter_data == b""
    assert fade.name_offset == 32
    assert fade.param_offset == 32 + len("cctrans.dll\x00".encode("utf-16le"))
    assert fade.param_size == 0


def test_roundtrip_fade_out():
    """Antibody #3: FadeOut shape with flags=1 (Color bit) - the other
    primary FNAF 1 configuration. Color bytes are still zero (fade-to-
    black); the Color bit just marks the color as meaningful rather
    than changing its value."""
    payload = _pack_fade(duration_ms=1010, flags=1, module_file="cctrans.dll")
    fade = decode_frame_fade(payload)
    assert fade.flags == 1
    assert fade.has_color_bit is True
    assert fade.color == (0, 0, 0, 0)


def test_roundtrip_with_parameter_data():
    """Antibody #3: synthetic record with a non-empty parameterData
    tail. FNAF 1 never ships one, but the variable-tail code path must
    work so any future game using custom .mvx transitions doesn't
    need a new probe. 8-byte tail is arbitrary."""
    tail = b"\xde\xad\xbe\xef\x01\x02\x03\x04"
    payload = _pack_fade(
        module_file="custom.mvx", parameter_data=tail, duration_ms=500, flags=0
    )
    fade = decode_frame_fade(payload)
    assert fade.module_file == "custom.mvx"
    assert fade.parameter_data == tail
    assert fade.param_size == 8


def test_roundtrip_custom_color():
    """Antibody #3: non-zero color bytes round-trip unchanged. Proves
    the ReadColor 4-byte raw slice is lossless through the decoder."""
    payload = _pack_fade(flags=1, color=b"\xff\x80\x40\xaa")
    fade = decode_frame_fade(payload)
    assert fade.color == (0xFF, 0x80, 0x40, 0xAA)


def test_roundtrip_ascii_module_file():
    """Antibody #3: the ASCII string path (non-Unicode pack). FNAF 1 is
    Unicode, so this path only runs here - it's load-bearing for any
    future non-Unicode pack that lands in this decoder."""
    payload = _pack_fade(module_file="cctrans.dll", unicode=False)
    fade = decode_frame_fade(payload, unicode=False)
    assert fade.module_file == "cctrans.dll"
    # ASCII: 11 chars + 1 NUL = 12 bytes (vs 24 in Unicode).
    assert fade.param_offset == 32 + 12


def test_payload_too_short_raises():
    """Antibody #2: a payload smaller than the 32-byte fixed header."""
    with pytest.raises(FrameFadeDecodeError, match="fixed header"):
        decode_frame_fade(b"\x00" * 16)


def test_name_offset_before_header_raises():
    """Antibody #1: name_offset can't point back into the fixed header -
    that would alias string bytes with header fields."""
    payload = _pack_fade(override_name_offset=0)
    with pytest.raises(FrameFadeDecodeError, match="name_offset=0 is outside"):
        decode_frame_fade(payload)


def test_name_offset_past_end_raises():
    """Antibody #1: name_offset past the payload end is nonsense."""
    payload = _pack_fade(override_name_offset=10_000)
    with pytest.raises(FrameFadeDecodeError, match="name_offset=10000 is outside"):
        decode_frame_fade(payload)


def test_negative_param_size_raises():
    """Antibody #1: param_size negative means RC4 drift decoded a
    garbage value in that slot. Raise before allocating a negative-
    sized blob."""
    payload = _pack_fade(override_param_size=-1)
    with pytest.raises(FrameFadeDecodeError, match="param_size=-1"):
        decode_frame_fade(payload)


def test_param_overruns_payload_raises():
    """Antibody #2: param_offset + param_size running past len(payload)."""
    payload = _pack_fade(override_param_size=999)
    with pytest.raises(FrameFadeDecodeError, match="exceeds payload length"):
        decode_frame_fade(payload)


def test_missing_null_terminator_raises():
    """Antibody #1: a non-terminated string runs off the end of the
    payload. The reader must raise rather than returning partial bytes
    or reading random stack."""
    # Build a valid header that points name_offset at the *last* 4 bytes
    # of the payload - not enough room for a wide NUL terminator.
    header = struct.pack(
        "<4s4sii4siii",
        b"STDT",
        b"FADE",
        100,
        0,
        b"\x00\x00\x00\x00",
        32,  # name_offset - start of tail
        36,  # param_offset
        0,   # param_size
    )
    # 4 bytes of UTF-16 text, no NUL terminator.
    tail = b"a\x00b\x00"
    payload = header + tail
    with pytest.raises(FrameFadeDecodeError, match="no UTF-16 NUL terminator"):
        decode_frame_fade(payload)


def test_region_overlap_raises():
    """Antibody #1: if name_offset and param_offset overlap, the byte-
    coverage check fires and names the overlap."""
    # Build a payload where the string region (32..56) and the
    # parameter_data region (40..48) overlap.
    module_file_bytes = "cctrans.dll".encode("utf-16le") + b"\x00\x00"  # 24 B
    header = struct.pack(
        "<4s4sii4siii",
        b"STDT",
        b"FADE",
        100,
        0,
        b"\x00\x00\x00\x00",
        32,  # name_offset: 32..56
        40,  # param_offset: 40..48 (overlaps string!)
        8,   # param_size
    )
    payload = header + module_file_bytes  # 32 + 24 = 56 bytes total
    with pytest.raises(FrameFadeDecodeError, match="covered by more than one region"):
        decode_frame_fade(payload)


def test_gap_between_regions_raises():
    """Antibody #2: an unaccounted-for byte between the string and the
    parameter_data raises. This is the primary signal for "the pack
    grew a new field we haven't pinned yet" / drift on offset fields."""
    module_file_bytes = "ab".encode("utf-16le") + b"\x00\x00"  # 6 bytes
    header = struct.pack(
        "<4s4sii4siii",
        b"STDT",
        b"FADE",
        100,
        0,
        b"\x00\x00\x00\x00",
        32,  # name_offset: 32..38
        40,  # param_offset: 40..42 (gap at offset 38-39!)
        2,   # param_size
    )
    # Total payload: 32 + 6 (string) + 2 (gap) + 2 (parameter) = 42.
    payload = header + module_file_bytes + b"\x00\x00" + b"\x42\x42"
    with pytest.raises(FrameFadeDecodeError, match="not covered by any region"):
        decode_frame_fade(payload)


def test_as_dict_shape():
    """Snapshot-style pin of the as_dict output. If a field is added,
    removed, or renamed on `FrameFade`, this breaks."""
    payload = _pack_fade(duration_ms=1010, flags=1, module_file="cctrans.dll")
    d = decode_frame_fade(payload).as_dict()
    assert d == {
        "module": "STDT",
        "name": "FADE",
        "duration_ms": 1010,
        "flags": 1,
        "color": [0, 0, 0, 0],
        "has_color_bit": True,
        "name_offset": 32,
        "param_offset": 56,
        "param_size": 0,
        "module_file": "cctrans.dll",
        "parameter_data": "",
    }


# --- FNAF 1 end-to-end antibodies (require binary) -----------------------


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_fade_presence_count():
    """Antibody #5 multi-input: across the 17 FNAF 1 frames, fade slots
    are **optional**. Pin the aggregate presence counts so a future
    drift that silently drops or adds fade records to frames fires
    loudly rather than shifting the snapshot underneath us."""
    frames = _fnaf1_frames()
    n_fade_in = sum(1 for f in frames if f.fade_in is not None)
    n_fade_out = sum(1 for f in frames if f.fade_out is not None)
    n_has_any = sum(
        1 for f in frames if f.fade_in is not None or f.fade_out is not None
    )
    assert n_fade_in == 9
    assert n_fade_out == 8
    assert n_has_any == 10


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_fade_is_standard_cctrans():
    """Antibody #1 strict-unknown: every FNAF 1 fade uses the standard
    Clickteam `STDT` transition marker and `cctrans.dll` module file.
    If this ever changes, Scott shipped a custom transition module and
    the probe needs to expand (or the RC4 pipeline drifted on the tail
    string bytes)."""
    frames = _fnaf1_frames()
    for i, f in enumerate(frames):
        for label, fade in (("fade_in", f.fade_in), ("fade_out", f.fade_out)):
            if fade is None:
                continue
            assert fade.module == b"STDT", (
                f"frame #{i} ({f.name!r}) {label}: module={fade.module!r} "
                f"!= b'STDT' (standard Clickteam transition)"
            )
            assert fade.name == b"FADE", (
                f"frame #{i} ({f.name!r}) {label}: name={fade.name!r} "
                f"!= b'FADE'"
            )
            assert fade.module_file == "cctrans.dll", (
                f"frame #{i} ({f.name!r}) {label}: module_file="
                f"{fade.module_file!r} != 'cctrans.dll'"
            )
            assert fade.param_size == 0, (
                f"frame #{i} ({f.name!r}) {label}: param_size="
                f"{fade.param_size} != 0 (FNAF 1 ships no parameter data)"
            )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_fade_in_flags_are_zero_out_flags_are_one():
    """Antibody #1: across FNAF 1, the flags convention is perfectly
    symmetric - every FadeIn has flags=0 (Color bit clear), every
    FadeOut has flags=1 (Color bit set; fade-to-black target). Pin it
    so a drift that e.g. flips a bit in the flags field surfaces."""
    frames = _fnaf1_frames()
    for i, f in enumerate(frames):
        if f.fade_in is not None:
            assert f.fade_in.flags == 0, (
                f"frame #{i} ({f.name!r}) fade_in.flags={f.fade_in.flags} "
                f"!= 0 (FNAF 1 FadeIn records don't set the Color bit)"
            )
            assert f.fade_in.has_color_bit is False
        if f.fade_out is not None:
            assert f.fade_out.flags == 1, (
                f"frame #{i} ({f.name!r}) fade_out.flags={f.fade_out.flags} "
                f"!= 1 (FNAF 1 FadeOut records always set the Color bit)"
            )
            assert f.fade_out.has_color_bit is True


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_fade_snapshot():
    """Snapshot antibody (#7) pinning the fade transition configuration
    of every FNAF 1 frame. Structure:

        (frame_name, fade_in_tuple_or_None, fade_out_tuple_or_None)

    where the tuple is (duration_ms, flags, color, module_file, module,
    name, name_offset, param_offset, param_size, parameter_data).

    Covers 10 tuple fields across 17 fade_in slots + 17 fade_out slots
    (most resolving to None for absent records, 17 actual records).
    Any byte drift on any field surfaces as a mismatch here.
    """
    frames = _fnaf1_frames()
    expected = _FNAF1_FADE_SNAPSHOT
    assert len(frames) == len(expected)

    def _tup(fade):
        if fade is None:
            return None
        return (
            fade.duration_ms,
            fade.flags,
            fade.color,
            fade.module_file,
            fade.module,
            fade.name,
            fade.name_offset,
            fade.param_offset,
            fade.param_size,
            fade.parameter_data,
        )

    for i, (f, exp) in enumerate(zip(frames, expected)):
        exp_name, exp_in, exp_out = exp
        assert f.name == exp_name, (
            f"frame #{i}: name {f.name!r} != expected {exp_name!r}"
        )
        got_in = _tup(f.fade_in)
        got_out = _tup(f.fade_out)
        assert got_in == exp_in, (
            f"frame #{i} ({f.name!r}) fade_in drifted:\n"
            f"  got  {got_in}\n  want {exp_in}"
        )
        assert got_out == exp_out, (
            f"frame #{i} ({f.name!r}) fade_out drifted:\n"
            f"  got  {got_out}\n  want {exp_out}"
        )


# --- Empirical snapshot (captured from probe #4.11 decrypt path) ---------

# Pinned empirical snapshot: fade_in and fade_out configuration across
# every FNAF 1 frame in pack order. 10 frames carry at least one fade
# record (17 total); 7 frames (title, Frame 1, died, freddy, wait,
# creepy start, creepy end) carry neither. Captured against probe
# #4.11's first passing decrypt run.
_FadeTuple = tuple[
    int,                 # duration_ms
    int,                 # flags
    tuple[int, int, int, int],  # color (r, g, b, a)
    str,                 # module_file
    bytes,               # module (4 B)
    bytes,               # name (4 B)
    int,                 # name_offset
    int,                 # param_offset
    int,                 # param_size
    bytes,               # parameter_data
]

_STD_STDT_CCTRANS_0 = (
    (0, 0, 0, 0),       # color
    "cctrans.dll",      # module_file
    b"STDT",            # module
    b"FADE",            # name
    32,                 # name_offset
    56,                 # param_offset
    0,                  # param_size
    b"",                # parameter_data
)


def _fade(duration_ms: int, flags: int) -> _FadeTuple:
    """Short-hand for the FNAF 1 fade tuple - every record has the same
    trailing 8 fields and only differs in duration+flags."""
    return (duration_ms, flags) + _STD_STDT_CCTRANS_0


_FNAF1_FADE_SNAPSHOT: tuple[
    tuple[str, _FadeTuple | None, _FadeTuple | None], ...
] = (
    ("Frame 17",     _fade(1010, 0), _fade(1010, 1)),
    ("title",        None,           None),
    ("what day",     None,           _fade(1010, 1)),
    ("Frame 1",      None,           None),
    ("died",         None,           None),
    ("freddy",       None,           None),
    ("next day",     _fade(1010, 0), _fade(900,  1)),
    ("wait",         None,           None),
    ("gameover",     _fade(1010, 0), None),
    ("the end",      _fade(2000, 0), _fade(2000, 1)),
    ("ad",           _fade(2000, 0), _fade(2000, 1)),
    ("the end 2",    _fade(2000, 0), _fade(2000, 1)),
    ("customize",    _fade(560,  0), None),
    ("the end 3",    _fade(2000, 0), _fade(2000, 1)),
    ("creepy start", None,           None),
    ("creepy end",   None,           None),
    ("end of demo",  _fade(1130, 0), _fade(1010, 1)),
)
