"""Regression tests for 0x3338 FrameItemInstances decoder (probe #4.10).

First frame sub-chunk carrying **world state** (placed object instances)
rather than configuration. Second length-prefixed variable-length
flag=3 shape after FrameLayers (#4.8). Per-instance fixed size 20 B,
outer u32 count prefix.

This probe also settles a real reference discrepancy empirically:

- CTFAK2.0 reads `4 + 20 * N` bytes and stops.
- Anaconda reads `4 + 20 * N + 4` bytes (a trailing checksum).

The decoder accepts either layout and records the empirical outcome on
`FrameItemInstances.has_trailing_checksum`; the FNAF 1 snapshot pins
which one matches.

Antibody coverage:

- #1 strict-unknown: `parent_type` out-of-enum raises; bytes matching
  neither candidate layout raise with both sizes named.
- #2 byte-count: strict equality against whichever candidate total
  the bytes match. Drift on either axis raises with signed diffs.
- #3 round-trip: synthetic pack/unpack in both candidate layouts.
- #4 multi-oracle: (cross-chunk) every instance's `layer` must be
  `< FrameLayers.count`. Enforced in `decode_frame`, pinned here.
- #5 multi-input: runs against all 17 FNAF 1 frames.
- #7 snapshot: per-frame (count, [(handle, object_info, x, y,
  parent_type, parent_handle, layer, instance)]) pinned below.

Sixth independent flag=3 shape after FrameHeader (16 B), FramePalette
(1028 B), FrameVirtualRect (16 B), FrameLayers (variable, with string
tail), and FrameLayerEffects (parallel array). First to carry *non-
configuration* world-state bytes.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.frame import decode_frame
from fnaf_parser.decoders.frame_item_instances import (
    FRAME_ITEM_INSTANCE_FIXED_SIZE,
    FRAME_ITEM_INSTANCES_COUNT_SIZE,
    PARENT_TYPE_FRAME,
    PARENT_TYPE_FRAMEITEM,
    PARENT_TYPE_NONE,
    PARENT_TYPE_QUALIFIER,
    FrameItemInstance,
    FrameItemInstances,
    FrameItemInstancesDecodeError,
    decode_frame_item_instances,
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


def _pack_instance(
    handle: int,
    object_info: int,
    x: int,
    y: int,
    parent_type: int,
    parent_handle: int,
    layer: int,
    instance: int,
) -> bytes:
    """Pack one 20-byte FrameItemInstance record exactly as the decoder
    expects: u16 u16 + i32 i32 + i16 i16 i16 i16 = 20 bytes."""
    return struct.pack(
        "<HHiihhhh",
        handle,
        object_info,
        x,
        y,
        parent_type,
        parent_handle,
        layer,
        instance,
    )


def _pack_payload(
    entries: list[bytes], *, trailing_checksum: bytes | None = None
) -> bytes:
    """Build a full payload: u32 count (signed LE) + entries + optional
    4-byte trailing checksum (Anaconda layout)."""
    count = len(entries).to_bytes(
        FRAME_ITEM_INSTANCES_COUNT_SIZE, "little", signed=True
    )
    body = b"".join(entries)
    tail = trailing_checksum if trailing_checksum is not None else b""
    return count + body + tail


# --- Synthetic tests (no binary needed) ----------------------------------


def test_module_constants_stable():
    """Antibody #2: per-record / prefix sizes are the load-bearing
    constants in every byte-count error message. Pin them."""
    assert FRAME_ITEM_INSTANCE_FIXED_SIZE == 20
    assert FRAME_ITEM_INSTANCES_COUNT_SIZE == 4
    assert PARENT_TYPE_NONE == 0
    assert PARENT_TYPE_FRAME == 1
    assert PARENT_TYPE_FRAMEITEM == 2
    assert PARENT_TYPE_QUALIFIER == 3


def test_zero_instances_ctfak_layout():
    """Zero-instance payload in CTFAK layout is just a u32 zero. Must
    decode cleanly. Guards against an off-by-one on the count prefix."""
    fii = decode_frame_item_instances(_pack_payload([]))
    assert isinstance(fii, FrameItemInstances)
    assert fii.count == 0
    assert fii.instances == ()
    assert fii.has_trailing_checksum is False
    assert fii.trailing_checksum == b""


def test_zero_instances_anaconda_layout():
    """Zero-instance payload in Anaconda layout: u32 zero + 4 trailing.
    Exercises the trailing-checksum branch without any per-instance
    bytes - proves the layout-discriminator keys on total size, not on
    having actual instance records."""
    fii = decode_frame_item_instances(
        _pack_payload([], trailing_checksum=b"\xde\xad\xbe\xef")
    )
    assert fii.count == 0
    assert fii.instances == ()
    assert fii.has_trailing_checksum is True
    assert fii.trailing_checksum == b"\xde\xad\xbe\xef"


def test_payload_shorter_than_count_prefix_raises():
    """Antibody #2: a payload too short to even hold the count u32."""
    with pytest.raises(
        FrameItemInstancesDecodeError, match="must hold at least"
    ):
        decode_frame_item_instances(b"\x00\x00\x00")


def test_negative_count_raises():
    """Antibody #1: a negative count (signed int32) is nonsense - the
    decoder treats it as a drift signal and raises loudly rather than
    attempting to allocate a negative-sized loop."""
    # 0xFFFFFFFF decoded as signed int32 is -1.
    payload = b"\xff\xff\xff\xff"
    with pytest.raises(FrameItemInstancesDecodeError, match="Negative counts"):
        decode_frame_item_instances(payload)


def test_wrong_size_between_layouts_raises():
    """Antibody #1/#2: payload matching neither CTFAK nor Anaconda layout.
    Error message must name both candidate sizes and the signed diffs
    so the caller can tell immediately which of (a) RC4 drift or
    (b) a new layout variant is most likely."""
    # count=1 would imply 24 B (CTFAK) or 28 B (Anaconda). Supply 25 B.
    entries = [_pack_instance(1, 1, 0, 0, 0, -1, 0, 0)]
    bad = _pack_payload(entries) + b"\x00"  # 25 bytes, matches neither
    with pytest.raises(FrameItemInstancesDecodeError) as exc:
        decode_frame_item_instances(bad)
    msg = str(exc.value)
    assert "24" in msg and "28" in msg, (
        f"error should name both candidate totals; got: {msg}"
    )
    assert "+1" in msg and "-3" in msg, (
        f"error should include signed diffs vs both layouts; got: {msg}"
    )


def test_invalid_parent_type_raises():
    """Antibody #1 strict-unknown: parent_type must be in {0, 1, 2, 3}.
    Any other value indicates drift or a new Clickteam code, and we
    raise rather than silently accepting it."""
    bad = _pack_instance(1, 1, 0, 0, parent_type=7, parent_handle=-1, layer=0, instance=0)
    with pytest.raises(FrameItemInstancesDecodeError, match="parent_type=7"):
        decode_frame_item_instances(_pack_payload([bad]))


def test_roundtrip_ctfak_layout():
    """Antibody #3: multi-record round-trip through the no-tail layout.
    Exercises signed negatives on x/y (can be negative in FNAF 1 when
    instances sit off-screen at frame start) and parent_handle=-1."""
    specs = [
        # (handle, object_info, x, y, parent_type, parent_handle, layer, instance)
        (0, 1, 100, 200, PARENT_TYPE_NONE, -1, 0, 0),
        (1, 5, -50, -25, PARENT_TYPE_FRAME, 0, 0, 1),
        (2, 9, 0, 0, PARENT_TYPE_FRAMEITEM, 1, 3, 7),
        (3, 12, 640, 480, PARENT_TYPE_QUALIFIER, 42, 2, -1),
    ]
    payload = _pack_payload([_pack_instance(*s) for s in specs])
    fii = decode_frame_item_instances(payload)
    assert fii.count == len(specs)
    assert fii.has_trailing_checksum is False
    assert fii.trailing_checksum == b""
    for got, exp in zip(fii.instances, specs):
        handle, obj_info, x, y, ptype, phandle, layer, inst = exp
        assert isinstance(got, FrameItemInstance)
        assert got.handle == handle
        assert got.object_info == obj_info
        assert got.x == x
        assert got.y == y
        assert got.parent_type == ptype
        assert got.parent_handle == phandle
        assert got.layer == layer
        assert got.instance == inst


def test_roundtrip_anaconda_layout():
    """Antibody #3: same round-trip through the trailing-checksum layout.
    Verifies the 4-byte tail is captured verbatim (bit-exact) so a
    future caller can validate or echo it."""
    specs = [(0, 1, 10, 20, PARENT_TYPE_NONE, -1, 0, 0)]
    tail = b"\x01\x02\x03\x04"
    payload = _pack_payload(
        [_pack_instance(*s) for s in specs], trailing_checksum=tail
    )
    fii = decode_frame_item_instances(payload)
    assert fii.count == 1
    assert fii.has_trailing_checksum is True
    assert fii.trailing_checksum == tail


def test_as_dict_shape_ctfak():
    """Snapshot-style pin of the as_dict output shape (no trailing tail)."""
    specs = [(7, 3, 1, 2, PARENT_TYPE_NONE, -1, 0, 0)]
    d = decode_frame_item_instances(
        _pack_payload([_pack_instance(*s) for s in specs])
    ).as_dict()
    assert d == {
        "count": 1,
        "has_trailing_checksum": False,
        "trailing_checksum": None,
        "instances": [
            {
                "handle": 7,
                "object_info": 3,
                "x": 1,
                "y": 2,
                "parent_type": 0,
                "parent_handle": -1,
                "layer": 0,
                "instance": 0,
            }
        ],
    }


def test_as_dict_shape_anaconda():
    """Snapshot-style pin of the as_dict output shape with trailing tail."""
    specs = [(7, 3, 1, 2, PARENT_TYPE_NONE, -1, 0, 0)]
    d = decode_frame_item_instances(
        _pack_payload(
            [_pack_instance(*s) for s in specs],
            trailing_checksum=b"\xaa\xbb\xcc\xdd",
        )
    ).as_dict()
    assert d["has_trailing_checksum"] is True
    assert d["trailing_checksum"] == "aabbccdd"


# --- FNAF 1 end-to-end antibodies (require binary) -----------------------


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_has_item_instances():
    """Antibody #5 multi-input: every FNAF 1 frame must carry a 0x3338
    sub-chunk and it must decode. If decryption drifts on any frame,
    the byte-count antibody inside `decode_frame_item_instances` fires
    before we get here."""
    frames = _fnaf1_frames()
    assert len(frames) == 17
    for i, f in enumerate(frames):
        assert f.item_instances is not None, (
            f"frame #{i} ({f.name!r}) produced no item_instances - 0x3338 "
            f"either missing from sub_records or decoder did not run."
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_item_instance_layers_within_layer_count():
    """Cross-chunk antibody (#4 multi-oracle): every instance's `layer`
    must be a valid index into the peer FrameLayers tuple. Enforced at
    the decode_frame layer - if it fires, RC4 drifted on one side or
    the other."""
    frames = _fnaf1_frames()
    for i, f in enumerate(frames):
        assert f.layers is not None, f"frame #{i} ({f.name!r}) missing layers"
        assert f.item_instances is not None
        for j, inst in enumerate(f.item_instances.instances):
            assert 0 <= inst.layer < f.layers.count, (
                f"frame #{i} ({f.name!r}) instance #{j}: layer="
                f"{inst.layer} out of range [0, {f.layers.count})"
            )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_item_instances_parent_types_valid():
    """Antibody #1 strict-unknown: every instance's parent_type must be
    in the documented enum {0, 1, 2, 3}. Already enforced during decode;
    this pins the post-decode state so any future change to the
    validation table (e.g. adding a fifth code) surfaces here rather
    than quietly expanding the allowed set."""
    valid = {PARENT_TYPE_NONE, PARENT_TYPE_FRAME, PARENT_TYPE_FRAMEITEM, PARENT_TYPE_QUALIFIER}
    frames = _fnaf1_frames()
    for i, f in enumerate(frames):
        assert f.item_instances is not None
        for j, inst in enumerate(f.item_instances.instances):
            assert inst.parent_type in valid, (
                f"frame #{i} instance #{j} parent_type={inst.parent_type} "
                f"not in {sorted(valid)}"
            )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_item_instances_layout_discrepancy_resolved():
    """Cross-reference antibody: CTFAK and Anaconda disagree on whether
    0x3338 has a 4-byte trailing checksum. Pin the empirical outcome
    here - every FNAF 1 frame must agree on the layout (all with tail
    or all without), or we're looking at drift on specific frames."""
    frames = _fnaf1_frames()
    flags = {f.item_instances.has_trailing_checksum for f in frames
             if f.item_instances is not None}
    assert len(flags) == 1, (
        f"FNAF 1 frames disagree on FrameItemInstances trailing-checksum "
        f"layout: {flags}. Either RC4 drifted on a subset of frames or "
        f"the chunk format varies within a pack (unlikely)."
    )
    # Pin the resolved outcome. Expected: Anaconda layout (has_trailing_checksum=True)
    # per the Anaconda reference having a `skipBytes(4)` after the loop.
    observed = flags.pop()
    assert observed is _EXPECTED_HAS_TRAILING_CHECKSUM, (
        f"FrameItemInstances trailing-checksum layout shifted from the "
        f"pinned empirical outcome: got has_trailing_checksum={observed}, "
        f"expected {_EXPECTED_HAS_TRAILING_CHECKSUM}. If this flip is "
        f"legitimate, update _EXPECTED_HAS_TRAILING_CHECKSUM and the "
        f"snapshot literal in this file + the probe #4.10 crystallization."
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_item_instances_snapshot():
    """Snapshot antibody (#7) pinning the world-state placement of every
    object instance in every FNAF 1 frame. Structure:

        (frame_name, instance_count, [
            (handle, object_info, x, y, parent_type, parent_handle,
             layer, instance),
            ...
        ])

    Any byte drift on ANY of these fields (including RC4 drift,
    unintended re-ordering, or a unit bug on x/y int32) surfaces as
    a mismatch here. Pins what is arguably the most information-dense
    cross-chunk antibody in the whole frame-decode chain.
    """
    frames = _fnaf1_frames()
    expected = _FNAF1_ITEM_INSTANCES_SNAPSHOT
    assert len(frames) == len(expected)
    for i, (f, exp) in enumerate(zip(frames, expected)):
        exp_name, exp_count, exp_instances = exp
        assert f.name == exp_name, (
            f"frame #{i}: name {f.name!r} != expected {exp_name!r}"
        )
        assert f.item_instances is not None, (
            f"frame #{i} ({f.name!r}) missing item_instances"
        )
        assert f.item_instances.count == exp_count, (
            f"frame #{i} ({f.name!r}): instance count "
            f"{f.item_instances.count} != expected {exp_count}"
        )
        got = tuple(
            (
                inst.handle,
                inst.object_info,
                inst.x,
                inst.y,
                inst.parent_type,
                inst.parent_handle,
                inst.layer,
                inst.instance,
            )
            for inst in f.item_instances.instances
        )
        assert got == exp_instances, (
            f"frame #{i} ({f.name!r}) item_instances drifted:\n"
            f"  got  {got}\n"
            f"  want {exp_instances}"
        )


# --- Empirical snapshot (captured from probe #4.10 decrypt path) ---------

# Layout discriminator: True = Anaconda layout (trailing 4-byte checksum),
# False = CTFAK layout (no tail). Empirically pinned from the first
# successful FNAF 1 run.
_EXPECTED_HAS_TRAILING_CHECKSUM = True

_InstanceTuple = tuple[int, int, int, int, int, int, int, int]

# Pinned empirical snapshot: every placed object instance across every
# FNAF 1 frame in pack order. 17 frames, 232 total instances. Captured
# against probe #4.10's first passing decrypt run. The office (Frame 1)
# carries 120 instances - every door button, light button, camera LED,
# security monitor element, and the animatronics themselves. Changing
# any byte of any field (handle, object_info, x, y, parent_*, layer,
# instance) anywhere in the pack triggers a diff here.
_FNAF1_ITEM_INSTANCES_SNAPSHOT: tuple[
    tuple[str, int, tuple[_InstanceTuple, ...]], ...
] = (
    ('Frame 17', 2, (
        (0, 0, 310, -325, 0, 0, 0, 0),
        (3, 1, 388, 242, 0, 0, 0, 0),
    )),
    ('title', 33, (
        (5, 4, 0, 0, 0, 0, 0, 0),
        (1, 2, 0, 0, 0, 0, 0, 0),
        (13, 12, 18, -6, 0, 0, 0, 0),
        (2, 3, 44, -246, 0, 0, 0, 0),
        (21, 24, 117, -337, 0, 0, 0, 0),
        (6, 5, 172, 68, 0, 0, 0, 0),
        (7, 6, 275, 420, 0, 0, 0, 0),
        (8, 7, 275, 492, 0, 0, 0, 0),
        (9, 8, 132, 493, 0, 0, 0, 0),
        (10, 9, -24, -30, 0, 0, 0, 0),
        (12, 11, 263, 535, 0, 0, 0, 0),
        (20, 21, 172, 907, 0, 0, 0, 0),
        (18, 20, -151, 362, 0, 0, 0, 0),
        (0, 13, -19, -38, 0, 0, 0, 0),
        (3, 14, 1044, 686, 0, 0, 0, 0),
        (4, 15, -150, 742, 0, 0, 0, 0),
        (14, 16, 285, 571, 0, 0, 0, 0),
        (15, 17, 200, 338, 0, 0, 0, 0),
        (16, 18, 277, 338, 0, 0, 0, 0),
        (33, 33, 352, 338, 0, 0, 0, 0),
        (17, 19, 174, 512, 0, 0, 0, 0),
        (22, 22, 60, 889, 0, 0, 0, 0),
        (19, 23, -190, -91, 0, 0, 0, 0),
        (23, 25, 324, 639, 0, 0, 0, 0),
        (24, 26, -65, 67, 0, 0, 0, 0),
        (26, 27, -65, 92, 0, 0, 0, 0),
        (27, 28, -65, 120, 0, 0, 0, 0),
        (28, 29, -67, 149, 0, 0, 0, 0),
        (29, 30, -65, 173, 0, 0, 0, 0),
        (30, 31, -64, 270, 0, 0, 0, 0),
        (31, 32, 171, 292, 0, 0, 0, 0),
        (32, 34, 26, 682, 0, 0, 0, 0),
        (11, 10, 182, -108, 0, 0, 0, 0),
    )),
    ('what day', 7, (
        (1, 35, 0, 0, 1, 0, 0, 0),
        (2, 36, -136, -492, 0, 0, 0, 0),
        (0, 11, 60, -48, 0, 0, 0, 0),
        (4, 37, 646, 318, 0, 0, 0, 0),
        (5, 15, 1628, 471, 0, 0, 0, 0),
        (6, 38, 1553, 446, 0, 0, 0, 0),
        (3, 39, 764, -218, 0, 0, 0, 0),
    )),
    ('Frame 1', 120, (
        (119, 151, 428, -236, 0, 0, 0, 0),
        (6, 44, 0, 0, 0, 0, 0, 0),
        (19, 58, 868, 400, 0, 0, 0, 0),
        (116, 148, 734, 485, 0, 0, 0, 0),
        (2, 41, 640, 368, 0, 0, 0, 0),
        (10, 46, 0, 0, 2, 45, 0, 0),
        (16, 55, 640, 200, 0, 0, 0, 0),
        (5, 59, 72, -1, 0, 0, 0, 0),
        (21, 60, 1270, -2, 0, 0, 0, 0),
        (22, 61, 48, 390, 0, 0, 0, 0),
        (26, 65, 1546, 400, 0, 0, 0, 0),
        (23, 62, 54, 307, 0, 0, 0, 0),
        (27, 66, 1548, 323, 0, 0, 0, 0),
        (24, 63, 54, 449, 0, 0, 0, 0),
        (28, 67, 1548, 454, 0, 0, 0, 0),
        (25, 64, 180, -78, 0, 0, 0, 0),
        (117, 149, 360, -158, 0, 0, 0, 0),
        (118, 150, 360, -140, 0, 0, 0, 0),
        (33, 68, 0, 0, 2, 71, 0, 0),
        (29, 73, 0, 0, 2, 71, 0, 0),
        (98, 135, 678, 240, 0, 0, 0, 0),
        (114, 146, 660, 478, 0, 0, 0, 0),
        (115, 147, 0, -78, 0, 0, 0, 0),
        (1, 40, -22, -22, 0, 0, 1, 0),
        (3, 42, 0, 0, 0, 0, 2, 0),
        (4, 43, 92, 76, 0, 0, 2, 0),
        (0, 45, 0, 0, 0, 0, 2, 0),
        (8, 47, 60, -136, 0, 0, 2, 0),
        (7, 48, -78, -94, 0, 0, 2, 0),
        (11, 49, -78, -76, 0, 0, 2, 0),
        (9, 50, 0, -1, 0, 0, 2, 0),
        (12, 51, 241, 324, 0, 0, 2, 0),
        (15, 54, 1018, 327, 0, 0, 2, 0),
        (31, 70, 442, 691, 0, 0, 2, 0),
        (34, 72, 589, 599, 0, 0, 2, 0),
        (13, 52, 145, 332, 0, 0, 2, 0),
        (17, 56, 64, 334, 0, 0, 2, 0),
        (14, 53, 1130, 336, 0, 0, 2, 0),
        (18, 57, 1218, 344, 0, 0, 2, 0),
        (35, 74, 848, 313, 0, 0, 2, 0),
        (109, 141, 92, 920, 0, 0, 2, 0),
        (110, 142, 160, 916, 0, 0, 2, 0),
        (111, 143, 218, 918, 0, 0, 2, 0),
        (112, 144, 280, 922, 0, 0, 2, 0),
        (30, 69, 220, -94, 0, 0, 3, 0),
        (32, 71, 0, 0, 0, 0, 3, 0),
        (37, 75, 983, 353, 0, 0, 3, 0),
        (38, 76, 963, 409, 0, 0, 3, 0),
        (59, 95, 931, 487, 0, 0, 3, 0),
        (39, 77, 983, 603, 0, 0, 3, 0),
        (45, 81, 983, 643, 0, 0, 3, 0),
        (49, 85, 1089, 604, 0, 0, 3, 0),
        (50, 86, 1089, 644, 0, 0, 3, 0),
        (55, 91, 1186, 568, 0, 0, 3, 0),
        (58, 94, 1195, 437, 0, 0, 3, 0),
        (53, 89, 857, 436, 0, 0, 3, 0),
        (46, 82, 899, 585, 0, 0, 3, 0),
        (42, 78, 961, 341, 0, 0, 3, 0),
        (43, 79, 939, 397, 0, 0, 3, 0),
        (44, 80, 960, 590, 0, 0, 3, 0),
        (47, 83, 960, 630, 0, 0, 3, 0),
        (48, 84, 877, 574, 0, 0, 3, 0),
        (51, 87, 1066, 592, 0, 0, 3, 0),
        (52, 88, 1066, 632, 0, 0, 3, 0),
        (54, 90, 834, 424, 0, 0, 3, 0),
        (56, 92, 1163, 556, 0, 0, 3, 0),
        (57, 93, 1172, 424, 0, 0, 3, 0),
        (60, 96, 908, 475, 0, 0, 3, 0),
        (20, 97, -138, -258, 0, 0, 3, 0),
        (36, 98, 384, 69, 0, 0, 3, 0),
        (41, 99, 832, 292, 0, 0, 3, 0),
        (74, 100, -61, 637, 0, 0, 3, 0),
        (78, 104, -67, 697, 0, 0, 3, 0),
        (79, 105, 221, 646, 0, 0, 3, 0),
        (96, 15, 1208, 810, 0, 0, 3, 0),
        (84, 123, 1185, 59, 0, 0, 3, 0),
        (75, 101, 74, 674, 0, 0, 3, 0),
        (76, 102, 106, 638, 0, 0, 3, 0),
        (77, 103, -269, 625, 0, 0, 3, 0),
        (82, 106, -196, 632, 0, 0, 3, 0),
        (83, 107, -564, 77, 0, 0, 3, 0),
        (85, 108, 120, 657, 0, 0, 3, 0),
        (61, 109, 727, 794, 0, 0, 3, 0),
        (66, 114, 906, 662, 0, 0, 3, 0),
        (68, 116, 988, 698, 0, 0, 3, 0),
        (69, 117, 1085, 698, 0, 0, 3, 0),
        (67, 115, 1160, 662, 0, 0, 3, 0),
        (108, 140, 1218, 663, 0, 0, 3, 0),
        (63, 111, 978, 359, 0, 0, 3, 0),
        (73, 120, 978, 358, 0, 0, 3, 0),
        (107, 139, 980, 350, 0, 0, 3, 0),
        (62, 110, 771, 876, 0, 0, 3, 0),
        (64, 112, 800, 825, 0, 0, 3, 0),
        (80, 121, 845, 827, 0, 0, 3, 0),
        (87, 127, 884, 826, 0, 0, 3, 0),
        (106, 138, 921, 829, 0, 0, 3, 0),
        (65, 113, 1042, 787, 0, 0, 3, 0),
        (70, 118, 1294, 850, 0, 0, 3, 0),
        (72, 119, 586, -120, 0, 0, 3, 0),
        (81, 122, 1198, 31, 0, 0, 3, 0),
        (86, 124, 1350, -54, 0, 0, 3, 0),
        (71, 125, 980, 785, 0, 0, 3, 0),
        (90, 128, 1112, 783, 0, 0, 3, 0),
        (88, 11, 1237, 89, 0, 0, 3, 0),
        (95, 132, 1130, -87, 0, 0, 3, 0),
        (89, 126, 1040, -58, 0, 0, 3, 0),
        (91, 129, 1112, 819, 0, 0, 3, 0),
        (92, 130, 572, 769, 0, 0, 3, 0),
        (93, 131, 554, 668, 0, 0, 3, 0),
        (97, 133, 754, 74, 0, 0, 3, 0),
        (40, 134, 912, -140, 0, 0, 3, 0),
        (99, 136, 1108, 848, 0, 0, 3, 0),
        (100, 26, 464, -145, 0, 0, 3, 0),
        (101, 27, 464, -127, 0, 0, 3, 0),
        (102, 28, 464, -109, 0, 0, 3, 0),
        (103, 29, 464, -91, 0, 0, 3, 0),
        (104, 30, 464, -73, 0, 0, 3, 0),
        (105, 137, 87, 37, 0, 0, 3, 0),
        (113, 145, 988, -956, 0, 0, 3, 0),
        (94, 10, 794, -134, 0, 0, 3, 0),
    )),
    ('died', 3, (
        (1, 2, 0, 0, 0, 0, 0, 0),
        (2, 35, 0, 0, 0, 0, 0, 0),
        (0, 15, -72, 152, 0, 0, 0, 0),
    )),
    ('freddy', 5, (
        (1, 2, 0, 0, 0, 0, 0, 0),
        (0, 152, 0, 0, 0, 0, 0, 0),
        (3, 35, 0, 0, 2, 2, 0, 0),
        (2, 15, 200, -74, 0, 0, 0, 0),
        (4, 10, 324, -82, 0, 0, 0, 0),
    )),
    ('next day', 12, (
        (1, 153, 544, 298, 0, 0, 0, 0),
        (3, 154, 640, 296, 0, 0, 0, 0),
        (4, 155, 548, 408, 0, 0, 0, 0),
        (5, 156, 573, 440, 0, 0, 0, 0),
        (6, 157, 572, 224, 0, 0, 0, 0),
        (8, 11, -74, 12, 0, 0, 0, 0),
        (2, 31, 170, -104, 0, 0, 0, 0),
        (7, 141, 300, -126, 0, 0, 0, 0),
        (9, 142, 366, -128, 0, 0, 0, 0),
        (10, 143, 434, -128, 0, 0, 0, 0),
        (11, 144, 492, -132, 0, 0, 0, 0),
        (0, 10, 22, -76, 0, 0, 0, 0),
    )),
    ('wait', 2, (
        (4, 158, -98, -354, 0, 0, 0, 0),
        (1, 159, 1226, 675, 0, 0, 0, 0),
    )),
    ('gameover', 4, (
        (2, 161, 0, 0, 0, 0, 0, 0),
        (4, 162, 1046, 660, 0, 0, 0, 0),
        (0, 160, -126, -272, 0, 0, 0, 0),
        (1, 163, -82, 16, 0, 0, 0, 0),
    )),
    ('the end', 2, (
        (0, 164, 0, 0, 0, 0, 0, 0),
        (1, 10, 376, -128, 0, 0, 0, 0),
    )),
    ('ad', 1, (
        (0, 165, 0, 0, 0, 0, 0, 0),
    )),
    ('the end 2', 2, (
        (0, 166, 0, 0, 0, 0, 0, 0),
        (1, 10, 376, -128, 0, 0, 0, 0),
    )),
    ('customize', 32, (
        (9, 174, 118, 187, 0, 0, 0, 0),
        (10, 175, 403, 187, 0, 0, 0, 0),
        (11, 176, 682, 187, 0, 0, 0, 0),
        (12, 177, 957, 187, 0, 0, 0, 0),
        (15, 178, 118, 425, 0, 0, 0, 0),
        (16, 178, 402, 425, 0, 0, 0, 0),
        (17, 178, 684, 425, 0, 0, 0, 0),
        (18, 178, 960, 425, 0, 0, 0, 0),
        (5, 188, 116, 654, 0, 0, 0, 0),
        (1, 167, -50, -522, 0, 0, 0, 0),
        (22, 179, -50, -405, 0, 0, 0, 0),
        (2, 168, 448, 29, 0, 0, 0, 0),
        (3, 169, 1044, 603, 0, 0, 0, 0),
        (4, 170, 140, 108, 0, 0, 0, 0),
        (6, 171, 714, 108, 0, 0, 0, 0),
        (7, 172, 1004, 108, 0, 0, 0, 0),
        (8, 173, 425, 108, 0, 0, 0, 0),
        (20, 11, 18, -44, 0, 0, 0, 0),
        (23, 180, 311, 470, 0, 0, 0, 0),
        (31, 184, 122, 470, 0, 0, 0, 0),
        (32, 185, 406, 470, 0, 0, 0, 0),
        (33, 186, 690, 470, 0, 0, 0, 0),
        (34, 187, 969, 470, 0, 0, 0, 0),
        (28, 181, 593, 470, 0, 0, 0, 0),
        (29, 182, 876, 470, 0, 0, 0, 0),
        (30, 183, 1154, 470, 0, 0, 0, 0),
        (24, 141, 267, 532, 0, 0, 0, 0),
        (25, 142, 550, 532, 0, 0, 0, 0),
        (26, 143, 832, 532, 0, 0, 0, 0),
        (27, 144, 1112, 532, 0, 0, 0, 0),
        (13, 189, -114, 348, 0, 0, 0, 0),
        (35, 10, 408, -116, 0, 0, 0, 0),
    )),
    ('the end 3', 2, (
        (0, 190, 0, 0, 0, 0, 0, 0),
        (1, 10, 376, -128, 0, 0, 0, 0),
    )),
    ('creepy start', 3, (
        (0, 191, 0, 0, 0, 0, 0, 0),
        (1, 192, 510, 192, 0, 0, 0, 0),
        (2, 192, 804, 196, 0, 0, 0, 0),
    )),
    ('creepy end', 1, (
        (0, 193, 0, 0, 0, 0, 0, 0),
    )),
    ('end of demo', 2, (
        (2, 195, 452, 222, 0, 0, 0, 0),
        (0, 194, -228, -740, 0, 0, 0, 0),
    )),
)
