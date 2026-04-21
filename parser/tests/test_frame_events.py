"""Regression tests for 0x333D FrameEvents decoder (probe #4.12).

The runtime event graph. All 17 FNAF 1 frames ship this sub-chunk; it
is the largest and most deeply nested structure inside a Frame (Frame 1
alone carries 435 event groups / 1108 conditions / 929 actions). Unlike
the peer flag=3 sub-chunks, 0x333D is an *internal* TLV envelope with
4-byte ASCII section tags.

Antibody coverage:

- #1 strict-unknown: unknown section tags raise loudly with the tag
  bytes and payload offset; negative qualifier / region / body sizes
  raise; non-negated EventGroup size raises; duplicate non-TAG_END
  tags raise; envelope without the b"<<ER" end marker raises.
- #2 byte-count: every length-prefixed record (EventGroup, Condition,
  Action, Parameter) is validated by reconciling its `size` field
  against the bytes consumed by its children. Parameter-under-run
  inside a condition fires, action walk over-running its group fires,
  ERev region walk ending past its declared byte range fires.
- #3 round-trip: synthetic pack/unpack covers an empty envelope, a
  single-group envelope with one condition and one action, a
  condition with a non-empty opaque parameter tail, and a record that
  omits the optional ERop section.
- #4 multi-oracle: CTFAK Events.cs / Anaconda events.pyx wire formats
  agree on every field order. One divergence on ERop shape (CTFAK:
  single int32; Anaconda: length-prefixed body). We adopted Anaconda's
  length-prefixed form - verified below that the FNAF 1 ERop sections
  decode with zero bytes in the body (Anaconda's reading).
- #5 multi-input: all 17 frames shipping 0x333D route through this
  decoder; the snapshot below pins per-frame (num_groups, total
  conditions, total actions).
- #6 loud-skip: `parameter_codes_seen` surfaces the 15 distinct
  parameter codes FNAF 1 actually uses; any RC4 drift would produce
  garbage codes outside this closed set and fail the snapshot.
- #7 snapshot: per-frame triples pinned below; total parameter-codes
  set pinned globally.

Eighth flag=3 shape after FrameHeader (16 B), FramePalette (1028 B),
FrameVirtualRect (16 B), FrameLayers (variable), FrameLayerEffects
(parallel array), FrameItemInstances (length-prefixed), and FrameFade
(optional; two slots). First sub-chunk that is itself a TLV envelope
rather than a flat record - drift inside this walker never touches
the peer decoders because they are all fully flat.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.frame import decode_frame
from fnaf_parser.decoders.frame_events import (
    ACTION_FIXED_SIZE,
    CONDITION_FIXED_SIZE,
    EVENT_GROUP_HEADER_SIZE,
    HEADER_PRELUDE_SIZE,
    KNOWN_TAGS,
    QUALIFIER_SIZE,
    TAG_END,
    TAG_EVENT_COUNT,
    TAG_EVENTGROUP_DATA,
    TAG_EXTENSION_DATA,
    TAG_HEADER,
    EventAction,
    EventCondition,
    EventGroup,
    EventParameter,
    FrameEvents,
    FrameEventsDecodeError,
    Qualifier,
    decode_frame_events,
)
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


def _fnaf1_frames():
    """Walk FNAF 1 end-to-end with transform enabled. Same helper shape
    as the sibling flag=3 tests - any RC4 or pipeline drift surfaces
    simultaneously across every flag=3 decoder."""
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


# --- Synthetic pack helpers ---------------------------------------------


def _pack_parameter(code: int, data: bytes = b"") -> bytes:
    """Build one Parameter record: int16 size + int16 code + data."""
    size = 4 + len(data)
    return struct.pack("<hh", size, code) + data


def _pack_condition(
    *,
    object_type: int = 0,
    num: int = 0,
    object_info: int = 0,
    object_info_list: int = -1,
    flags: int = 0,
    other_flags: int = 0,
    def_type: int = 0,
    identifier: int = 0,
    parameters: tuple[bytes, ...] = (),
) -> bytes:
    """Build one Condition record: u16 size + fixed 14-B header + params."""
    params_blob = b"".join(parameters)
    size = 2 + CONDITION_FIXED_SIZE + len(params_blob)
    header = struct.pack(
        "<Hhhhhbb BBh",
        size,
        object_type,
        num,
        object_info,
        object_info_list,
        flags,
        other_flags,
        len(parameters),
        def_type,
        identifier,
    )
    return header + params_blob


def _pack_action(
    *,
    object_type: int = 0,
    num: int = 0,
    object_info: int = 0,
    object_info_list: int = -1,
    flags: int = 0,
    other_flags: int = 0,
    def_type: int = 0,
    parameters: tuple[bytes, ...] = (),
) -> bytes:
    """Build one Action record: u16 size + fixed 12-B header + params."""
    params_blob = b"".join(parameters)
    size = 2 + ACTION_FIXED_SIZE + len(params_blob)
    header = struct.pack(
        "<Hhhhhbb BB",
        size,
        object_type,
        num,
        object_info,
        object_info_list,
        flags,
        other_flags,
        len(parameters),
        def_type,
    )
    return header + params_blob


def _pack_group(
    *,
    flags: int = 0,
    line_padding: int = 0,
    is_restricted: int = 0,
    restrict_cpt: int = 0,
    conditions: tuple[bytes, ...] = (),
    actions: tuple[bytes, ...] = (),
    override_size: int | None = None,
) -> bytes:
    """Build one EventGroup: 16-B header (with *negated* size) + body."""
    body = b"".join(conditions) + b"".join(actions)
    size = EVENT_GROUP_HEADER_SIZE + len(body)
    wire_size = override_size if override_size is not None else -size
    header = struct.pack(
        "<hBBHhii",
        wire_size,
        len(conditions),
        len(actions),
        flags,
        line_padding,
        is_restricted,
        restrict_cpt,
    )
    return header + body


def _pack_envelope(
    *,
    max_objects: int = 1000,
    max_object_info: int = 0,
    num_players: int = 1,
    number_of_conditions: tuple[int, ...] = (0,) * 17,
    qualifiers: tuple[tuple[int, int], ...] = (),
    groups: tuple[bytes, ...] = (),
    extension_data: bytes | None = b"",
    include_event_count: bool = True,
    include_end: bool = True,
    extra_sections: bytes = b"",
) -> bytes:
    """Build a full 0x333D envelope in natural section order.

    Any of `extension_data=None`, `include_event_count=False`, or
    `include_end=False` produces a deliberately malformed envelope so
    the negative-path tests can exercise the missing-section raises.
    """
    assert len(number_of_conditions) == 17
    # ER>> header
    prelude = struct.pack(
        "<" + "h" * 21,
        max_objects,
        max_object_info,
        num_players,
        *number_of_conditions,
        len(qualifiers),
    )
    qual_blob = b"".join(struct.pack("<Hh", oi, t) for oi, t in qualifiers)
    out = TAG_HEADER + prelude + qual_blob

    # ERes (4-byte size field, body ignored)
    if include_event_count:
        out += TAG_EVENT_COUNT + struct.pack("<i", 0)

    # ERev (region_size int32 + group bytes)
    group_blob = b"".join(groups)
    out += TAG_EVENTGROUP_DATA + struct.pack("<i", len(group_blob)) + group_blob

    # ERop (optional; length-prefixed body)
    if extension_data is not None:
        out += TAG_EXTENSION_DATA + struct.pack("<i", len(extension_data))
        out += extension_data

    out += extra_sections

    if include_end:
        out += TAG_END

    return out


# --- Synthetic tests (no binary needed) ---------------------------------


def test_module_constants_stable():
    """Antibody #2: the fixed-record sizes are load-bearing in every
    byte-count error message. Pin them so any accidental edit to the
    struct.Struct definitions is caught at import time."""
    assert HEADER_PRELUDE_SIZE == 42
    assert QUALIFIER_SIZE == 4
    assert EVENT_GROUP_HEADER_SIZE == 16
    assert CONDITION_FIXED_SIZE == 14
    assert ACTION_FIXED_SIZE == 12
    assert KNOWN_TAGS == frozenset(
        {b"ER>>", b"ERes", b"ERev", b"ERop", b"<<ER"}
    )
    assert TAG_HEADER == b"ER>>"
    assert TAG_EVENT_COUNT == b"ERes"
    assert TAG_EVENTGROUP_DATA == b"ERev"
    assert TAG_EXTENSION_DATA == b"ERop"
    assert TAG_END == b"<<ER"


def test_roundtrip_empty_envelope():
    """Antibody #3: a minimal envelope with no groups, no qualifiers,
    empty ERop. All five section tags appear; all defaults resolve."""
    payload = _pack_envelope()
    ev = decode_frame_events(payload)
    assert isinstance(ev, FrameEvents)
    assert ev.max_objects == 1000
    assert ev.max_object_info == 0
    assert ev.num_players == 1
    assert ev.number_of_conditions == (0,) * 17
    assert ev.qualifiers == ()
    assert ev.event_groups == ()
    assert ev.extension_data == b""
    assert ev.parameter_codes_seen == frozenset()
    assert ev.total_conditions == 0
    assert ev.total_actions == 0


def test_roundtrip_without_erop():
    """Antibody #3: ERop is optional. An envelope that skips it still
    decodes cleanly - `extension_data` comes back as an empty bytes."""
    payload = _pack_envelope(extension_data=None)
    ev = decode_frame_events(payload)
    assert ev.extension_data == b""


def test_roundtrip_single_group_one_condition_one_action():
    """Antibody #3: the smallest non-trivial event graph - one group
    with one condition + one action, each carrying zero parameters."""
    cond = _pack_condition(
        object_type=1, num=-2, object_info=3, identifier=42
    )
    act = _pack_action(object_type=4, num=-1, object_info=5)
    group = _pack_group(conditions=(cond,), actions=(act,))
    payload = _pack_envelope(groups=(group,))
    ev = decode_frame_events(payload)

    assert len(ev.event_groups) == 1
    g = ev.event_groups[0]
    assert isinstance(g, EventGroup)
    # Group size is un-negated on read.
    assert g.size == EVENT_GROUP_HEADER_SIZE + len(cond) + len(act)
    assert len(g.conditions) == 1
    assert len(g.actions) == 1

    c = g.conditions[0]
    assert isinstance(c, EventCondition)
    assert c.object_type == 1
    assert c.num == -2
    assert c.object_info == 3
    assert c.identifier == 42
    assert c.parameters == ()

    a = g.actions[0]
    assert isinstance(a, EventAction)
    assert a.object_type == 4
    assert a.num == -1
    assert a.object_info == 5
    assert a.parameters == ()

    assert ev.total_conditions == 1
    assert ev.total_actions == 1


def test_roundtrip_with_opaque_parameter_data():
    """Antibody #3: the parameter body is opaque at this probe scope
    (see module docstring). Pack a condition with a non-empty data
    tail; the decoder should preserve it verbatim and surface the
    code in `parameter_codes_seen`."""
    tail = b"\xde\xad\xbe\xef\x01\x02"
    param = _pack_parameter(code=22, data=tail)
    cond = _pack_condition(parameters=(param,))
    act = _pack_action()
    group = _pack_group(conditions=(cond,), actions=(act,))
    payload = _pack_envelope(groups=(group,))
    ev = decode_frame_events(payload)

    c = ev.event_groups[0].conditions[0]
    assert len(c.parameters) == 1
    p = c.parameters[0]
    assert isinstance(p, EventParameter)
    assert p.code == 22
    assert p.size == 4 + len(tail)
    assert p.data == tail
    assert ev.parameter_codes_seen == frozenset({22})


def test_roundtrip_with_qualifiers():
    """Antibody #3: the ER>> qualifier list round-trips as a tuple of
    `(object_info, type)` records. The low-11-bit qualifier derivation
    matches CTFAK's `ObjectInfo & 0b11111111111`."""
    payload = _pack_envelope(
        qualifiers=((0b10000_00000000 | 7, 3), (0b00001_11111110, 5)),
    )
    ev = decode_frame_events(payload)
    assert len(ev.qualifiers) == 2
    q0, q1 = ev.qualifiers
    assert isinstance(q0, Qualifier)
    assert q0.type == 3
    assert q0.qualifier == 7  # low 11 bits
    assert q1.type == 5
    assert q1.qualifier == 0b00001_11111110 & 0b11111111111


def test_unknown_tag_raises():
    """Antibody #1: an unknown 4-byte section tag raises with the tag
    bytes and offset. Canonical RC4-drift signal."""
    # Build a valid prefix then inject a bogus tag before <<ER.
    payload = (
        _pack_envelope(include_end=False)
        + b"XXXX"
        + TAG_END
    )
    with pytest.raises(FrameEventsDecodeError, match=r"unknown section tag"):
        decode_frame_events(payload)


def test_duplicate_tag_raises():
    """Antibody #1: a given section tag (other than <<ER) must appear
    at most once. Two ER>> headers indicate either a corrupted stream
    or RC4 drift that happened to land on a valid-looking tag."""
    payload = _pack_envelope(
        extra_sections=TAG_HEADER
        + struct.pack("<" + "h" * 21, 1000, 0, 1, *(0,) * 17, 0),
        include_end=True,
    )
    with pytest.raises(FrameEventsDecodeError, match=r"appeared twice"):
        decode_frame_events(payload)


def test_missing_end_marker_raises():
    """Antibody #1: a stream that runs off the end without emitting the
    b'<<ER' terminator is treated as truncated."""
    payload = _pack_envelope(include_end=False)
    with pytest.raises(FrameEventsDecodeError, match=r"without the b'<<ER' end marker"):
        decode_frame_events(payload)


def test_missing_header_raises():
    """Antibody #1: an envelope whose only section is <<ER is missing
    the load-bearing ER>> header and must raise."""
    with pytest.raises(FrameEventsDecodeError, match=r"missing the b'ER>>' header"):
        decode_frame_events(TAG_END)


def test_header_prelude_truncated_raises():
    """Antibody #2: ER>> needs 42 bytes for the fixed prelude. A
    truncated payload raises pointing at the missing room."""
    payload = TAG_HEADER + b"\x00" * 10 + TAG_END
    with pytest.raises(FrameEventsDecodeError, match=r"no room for the 42-byte prelude"):
        decode_frame_events(payload)


def test_negative_qualifier_count_raises():
    """Antibody #1: a negative qualifier count is a common RC4-drift
    artifact (the count lives in the int16 slot right before the
    qualifier array and a bit flip turns it negative)."""
    prelude = struct.pack("<" + "h" * 21, 1000, 0, 1, *(0,) * 17, -1)
    payload = TAG_HEADER + prelude + TAG_END
    with pytest.raises(FrameEventsDecodeError, match=r"qualifier_count is -1"):
        decode_frame_events(payload)


def test_qualifier_count_overruns_payload_raises():
    """Antibody #2: ER>> claims more qualifiers than the payload can
    supply. The byte-count check fires before we read garbage."""
    # Claim 100 qualifiers (400 bytes) but supply none.
    prelude = struct.pack("<" + "h" * 21, 1000, 0, 1, *(0,) * 17, 100)
    payload = TAG_HEADER + prelude + TAG_END
    with pytest.raises(FrameEventsDecodeError, match=r"claims 100 qualifiers"):
        decode_frame_events(payload)


def test_erev_region_negative_raises():
    """Antibody #1: ERev region_size is a signed int32 on the wire; a
    negative value raises immediately."""
    prelude = struct.pack("<" + "h" * 21, 1000, 0, 1, *(0,) * 17, 0)
    payload = (
        TAG_HEADER + prelude
        + TAG_EVENT_COUNT + struct.pack("<i", 0)
        + TAG_EVENTGROUP_DATA + struct.pack("<i", -1)
        + TAG_END
    )
    with pytest.raises(FrameEventsDecodeError, match=r"region_size=-1"):
        decode_frame_events(payload)


def test_erev_region_overruns_payload_raises():
    """Antibody #2: ERev region_size claims more bytes than the payload
    holds. Byte-count reconcile catches it before the group walker
    would run off the end."""
    prelude = struct.pack("<" + "h" * 21, 1000, 0, 1, *(0,) * 17, 0)
    payload = (
        TAG_HEADER + prelude
        + TAG_EVENT_COUNT + struct.pack("<i", 0)
        + TAG_EVENTGROUP_DATA + struct.pack("<i", 9999)
        + TAG_END
    )
    with pytest.raises(FrameEventsDecodeError, match=r"exceeds remaining payload"):
        decode_frame_events(payload)


def test_non_negated_group_size_raises():
    """Antibody #1: EventGroup size is stored *negated* int16. A non-
    negative value means either RC4 drift or a schema-version skew."""
    cond = _pack_condition()
    act = _pack_action()
    # Supply a *positive* wire size -> schema violation.
    body_size = EVENT_GROUP_HEADER_SIZE + len(cond) + len(act)
    group = _pack_group(
        conditions=(cond,), actions=(act,), override_size=body_size  # positive!
    )
    payload = _pack_envelope(groups=(group,))
    with pytest.raises(FrameEventsDecodeError, match=r"non-negative size field"):
        decode_frame_events(payload)


def test_group_size_below_header_raises():
    """Antibody #2: even a zero-content group must carry its 16-byte
    header. abs(size)=10 is below the minimum and raises."""
    # Hand-craft a group: size field = -10 (abs=10 below 16 minimum).
    # We still have to emit a full 16-byte header so the later fields
    # don't run past payload end.
    group = struct.pack("<hBBHhii", -10, 0, 0, 0, 0, 0, 0)
    payload = _pack_envelope(groups=(group,))
    with pytest.raises(FrameEventsDecodeError, match=r"abs\(size\)=10"):
        decode_frame_events(payload)


def test_group_size_exceeds_region_raises():
    """Antibody #2: an EventGroup that claims more bytes than the ERev
    region supplies. Declared by over-sizing the group header and
    injecting it into a group blob sized to match a *smaller* region."""
    cond = _pack_condition()
    # Real content is 16 + len(cond) bytes but we claim abs(size) = 500.
    real_size = EVENT_GROUP_HEADER_SIZE + len(cond)
    group = struct.pack(
        "<hBBHhii", -500, 1, 0, 0, 0, 0, 0
    ) + cond
    # Envelope's ERev region will be len(group) = real_size, which is
    # smaller than the 500 the group claims.
    payload = _pack_envelope(groups=(group,))
    assert len(group) == real_size
    with pytest.raises(FrameEventsDecodeError, match=r"claims abs\(size\)=500"):
        decode_frame_events(payload)


def test_condition_parameter_walk_overruns_raises():
    """Antibody #2: a condition declares N parameters but the supplied
    bytes are sized for N-1. The size-field reconcile fires at the
    condition boundary pointing at the offset where the walk drifted."""
    param = _pack_parameter(code=1)  # 4 bytes
    # Hand-build a condition that says "1 parameter" but supply 0 bytes
    # of parameter data.
    cond_header = struct.pack(
        "<Hhhhhbb BBh",
        2 + CONDITION_FIXED_SIZE + len(param),  # size = room for 1 param
        0, 0, 0, -1, 0, 0,
        1,  # number_of_parameters = 1
        0, 0,
    )
    # But we *don't* append the param bytes - instead we append another
    # condition's worth of zeros that don't fit a valid parameter.
    malformed_cond = cond_header  # no parameter payload!
    # Pad with the bytes that *should* have been the parameter so the
    # payload length matches what the size field promised.
    malformed_cond += b"\x00\x00\x00\x00"  # 4 bytes, looks like size=0 param
    act = _pack_action()
    group = _pack_group(conditions=(malformed_cond,), actions=(act,))
    payload = _pack_envelope(groups=(group,))
    # Inner parameter decoder raises on size<4 before the outer
    # reconcile fires - that's still the right antibody #2 signal.
    with pytest.raises(
        FrameEventsDecodeError,
        match=r"(below the 4-byte minimum|parameter walk drifted)",
    ):
        decode_frame_events(payload)


def test_negative_parameter_size_raises():
    """Antibody #1: a parameter whose 2-byte size field is negative
    (or below 4) is a canonical RC4-drift signal."""
    # Hand-craft a parameter with size=0. We still need to package it
    # inside a valid condition and group envelope.
    bad_param = struct.pack("<hh", 0, 99)  # size=0 -> invalid
    cond_header = struct.pack(
        "<Hhhhhbb BBh",
        2 + CONDITION_FIXED_SIZE + 4,
        0, 0, 0, -1, 0, 0,
        1,  # number_of_parameters
        0, 0,
    )
    malformed_cond = cond_header + bad_param
    act = _pack_action()
    group = _pack_group(conditions=(malformed_cond,), actions=(act,))
    payload = _pack_envelope(groups=(group,))
    with pytest.raises(FrameEventsDecodeError, match=r"below the 4-byte minimum"):
        decode_frame_events(payload)


def test_action_size_below_header_raises():
    """Antibody #2: an action that claims size < 14 bytes cannot possibly
    carry its fixed header. Raises with the offending size."""
    # Hand-build: header says "size=10" (below 14-byte minimum).
    bad_action = struct.pack(
        "<Hhhhhbb BB",
        10,   # size - well below the 2+12 minimum
        0, 0, 0, -1, 0, 0, 0, 0,
    )
    # Pad so the action header bytes don't run past payload end.
    bad_action += b"\x00\x00\x00\x00"
    cond = _pack_condition()
    group = _pack_group(conditions=(cond,), actions=(bad_action,))
    payload = _pack_envelope(groups=(group,))
    with pytest.raises(FrameEventsDecodeError, match=r"below the 14-byte header minimum"):
        decode_frame_events(payload)


def test_as_dict_shape():
    """Snapshot-style pin of the as_dict output. If a field is added,
    removed, or renamed on FrameEvents / EventGroup / EventCondition /
    EventAction / EventParameter / Qualifier, this breaks."""
    param = _pack_parameter(code=7, data=b"\x01\x02")
    cond = _pack_condition(
        object_type=1, num=-2, object_info=3, identifier=42,
        parameters=(param,),
    )
    act = _pack_action(object_type=4, num=-1)
    group = _pack_group(conditions=(cond,), actions=(act,))
    payload = _pack_envelope(
        max_object_info=5,
        qualifiers=((0x1007, 3),),
        groups=(group,),
    )
    d = decode_frame_events(payload).as_dict()
    assert d == {
        "max_objects": 1000,
        "max_object_info": 5,
        "num_players": 1,
        "number_of_conditions": [0] * 17,
        "qualifiers": [{"object_info": 0x1007, "type": 3, "qualifier": 7}],
        "event_groups": [
            {
                "size": EVENT_GROUP_HEADER_SIZE + len(cond) + len(act),
                "flags": 0,
                "is_restricted": 0,
                "restrict_cpt": 0,
                "conditions": [
                    {
                        "size": 2 + CONDITION_FIXED_SIZE + len(param),
                        "object_type": 1,
                        "num": -2,
                        "object_info": 3,
                        "object_info_list": -1,
                        "flags": 0,
                        "other_flags": 0,
                        "def_type": 0,
                        "identifier": 42,
                        "parameters": [
                            {"code": 7, "size": 4 + 2, "data_hex": "0102"},
                        ],
                    },
                ],
                "actions": [
                    {
                        "size": 2 + ACTION_FIXED_SIZE,
                        "object_type": 4,
                        "num": -1,
                        "object_info": 0,
                        "object_info_list": -1,
                        "flags": 0,
                        "other_flags": 0,
                        "def_type": 0,
                        "parameters": [],
                    },
                ],
            },
        ],
        "extension_data_len": 0,
        "parameter_codes_seen": [7],
    }


# --- FNAF 1 end-to-end antibodies (require binary) -----------------------


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_all_frames_ship_events():
    """Antibody #5 multi-input: every FNAF 1 frame carries a 0x333D
    sub-chunk. Unlike fades (optional per-frame), events are mandatory
    - a None slot after the full pipeline means the flag=3 decode
    silently dropped the sub-chunk and we should fail loudly."""
    frames = _fnaf1_frames()
    assert len(frames) == 17
    for i, f in enumerate(frames):
        assert f.events is not None, (
            f"frame #{i} ({f.name!r}) has events=None; every FNAF 1 "
            f"frame should carry 0x333D"
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_event_header_invariants():
    """Antibody #1 strict-unknown: the ER>> header fields that FNAF 1
    sets identically on every frame - pinning them here turns any RC4
    drift on the 42-byte prelude into a named per-frame assertion."""
    frames = _fnaf1_frames()
    for i, f in enumerate(frames):
        ev = f.events
        assert ev is not None
        assert ev.max_objects == 1000, (
            f"frame #{i} ({f.name!r}) max_objects={ev.max_objects} != 1000"
        )
        assert ev.num_players == 1, (
            f"frame #{i} ({f.name!r}) num_players={ev.num_players} != 1"
        )
        assert ev.qualifiers == (), (
            f"frame #{i} ({f.name!r}) unexpectedly has {len(ev.qualifiers)} "
            f"qualifiers; FNAF 1 ships zero across every frame"
        )
        assert len(ev.extension_data) == 0, (
            f"frame #{i} ({f.name!r}) extension_data len="
            f"{len(ev.extension_data)} != 0; FNAF 1 ships an empty ERop body"
        )
        # The 17-tuple per-object-type condition counter is also
        # identical across all FNAF 1 frames - this is a single
        # Clickteam-project-wide value, not a per-frame value.
        assert ev.number_of_conditions == _FNAF1_NUMBER_OF_CONDITIONS, (
            f"frame #{i} ({f.name!r}) number_of_conditions="
            f"{ev.number_of_conditions} != {_FNAF1_NUMBER_OF_CONDITIONS}"
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_parameter_codes_closed_set():
    """Antibody #6 loud-skip: the union of every parameter code emitted
    across every FNAF 1 frame is a closed 15-element set. Any RC4 drift
    inside a parameter record produces a garbage `code` and fails the
    closed-set assertion before the snapshot even runs. This also
    guides probe #4.13+: these are the codes whose dispatch needs
    concrete decoders first."""
    frames = _fnaf1_frames()
    seen_all: set[int] = set()
    for f in frames:
        assert f.events is not None
        seen_all |= set(f.events.parameter_codes_seen)
    assert seen_all == {1, 2, 6, 9, 10, 14, 16, 22, 23, 25, 26, 27, 32, 45, 50}


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_event_totals():
    """Antibody #7 snapshot: the aggregate group/condition/action totals
    across every FNAF 1 frame. Pinning the totals at the top catches
    any byte drift that changes *any* frame's count - the per-frame
    snapshot below localizes it."""
    frames = _fnaf1_frames()
    total_groups = sum(len(f.events.event_groups) for f in frames)
    total_conds = sum(f.events.total_conditions for f in frames)
    total_acts = sum(f.events.total_actions for f in frames)
    assert (total_groups, total_conds, total_acts) == (584, 1346, 1186)


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_per_frame_snapshot():
    """Antibody #7 snapshot: per-frame pin of
    `(frame_name, max_object_info, num_groups, total_conditions,
    total_actions)`. 17 rows * 5 fields = 85 pinned values. Any RC4
    drift or envelope-walker bug on any frame surfaces here with the
    offending frame name called out by the assertion."""
    frames = _fnaf1_frames()
    expected = _FNAF1_EVENTS_SNAPSHOT
    assert len(frames) == len(expected)
    for i, (f, exp) in enumerate(zip(frames, expected)):
        ev = f.events
        assert ev is not None
        got = (
            f.name,
            ev.max_object_info,
            len(ev.event_groups),
            ev.total_conditions,
            ev.total_actions,
        )
        assert got == exp, (
            f"frame #{i} events snapshot drifted:\n"
            f"  got  {got}\n  want {exp}"
        )


# --- Empirical snapshot (captured from probe #4.12 decrypt path) ---------

# Project-wide 17-tuple: one int16 per object-type bucket. Same for
# every frame in the pack (it's a per-project Clickteam value that
# sizes the runtime's per-object condition table).
_FNAF1_NUMBER_OF_CONDITIONS: tuple[int, ...] = (
    7, 13, 24, 9, 11, 10, 41, 0, 0, 82, 0, 84, 0, 0, 82, 0, 85,
)

# Per-frame snapshot: `(name, max_object_info, num_groups,
# total_conditions, total_actions)`. Captured on probe #4.12's first
# clean pipeline run. Biggest frame is "Frame 1" (the actual nightly
# gameplay); smallest is "wait". Total across all 17 frames:
# 584 groups / 1346 conditions / 1186 actions.
_FNAF1_EVENTS_SNAPSHOT: tuple[tuple[str, int, int, int, int], ...] = (
    ("Frame 17",       3,   4,     4,     4),
    ("title",         34,  67,   125,   136),
    ("what day",       8,  11,    11,    13),
    ("Frame 1",      121, 435,  1108,   929),
    ("died",           4,   4,     4,     6),
    ("freddy",         6,   5,     5,    10),
    ("next day",      13,  13,    29,    23),
    ("wait",           3,   2,     2,     2),
    ("gameover",       3,   5,     7,     5),
    ("the end",        2,   3,     3,     8),
    ("ad",             1,   4,     4,     4),
    ("the end 2",      2,   3,     3,     8),
    ("customize",     24,  17,    30,    19),
    ("the end 3",      2,   3,     3,     8),
    ("creepy start",   2,   3,     3,     3),
    ("creepy end",     1,   2,     2,     3),
    ("end of demo",    2,   3,     3,     5),
)
