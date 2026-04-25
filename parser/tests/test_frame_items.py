"""Regression tests for 0x2229 FrameItems decoder (probe #5).

Top-level object-type bank. FrameItems decodes header + name for every
ObjectInfo. Active Properties (0x4446) are decoded as ObjectCommon;
non-Active Properties and Effects remain raw/opaque.

Antibody coverage:

- #1 strict-unknown: unknown inner sub-chunk id, invalid object_type,
  duplicate handle, missing 0x4444 header, missing 0x7F7F terminator,
  two headers, negative count.
- #2 byte-count: 0x4444 body length ≠ 16, inner size overrun, trailing
  junk after last ObjectInfo, truncated payload.
- #3 round-trip: synthetic CTFAK-layout pack/unpack with header + name +
  opaque properties + LAST.
- #4 multi-oracle: header field order cross-checked against CTFAK2.0
  `ObjectInfo.cs` and Anaconda `objectinfo.pyx` (187-189 ink bitfield).
- #5 multi-input: runs against the FNAF 1 0x2229 payload.
- #6 loud-skip: `deferred_sub_chunk_ids_seen` == {0x4446} on FNAF 1.
  (0x4448 Effects absent — noted in the docstring.)
- #7 snapshot: (count, object_type_histogram, sorted-handles-sample,
  first/last item (handle, object_type, name)) pinned below.
- #8 ObjectCommon: all 124 Active property bodies decode, byte coverage
  has only the known eight-zero-byte pad, and animation handles resolve
  against the sparse 605-record ImageBank handle set.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.frame_items import (
    EXTENSION_BASE,
    FRAME_ITEMS_COUNT_SIZE,
    OBJECT_HEADER_SIZE,
    OBJECT_TYPE_ACTIVE,
    OBJECT_TYPE_BACKDROP,
    OBJECT_TYPE_COUNTER,
    OBJECT_TYPE_QUICKBACKDROP,
    OBJECT_TYPE_TEXT,
    OPAQUE_SUB_CHUNK_IDS,
    SUB_HEADER_SIZE,
    SUB_OBJECT_EFFECTS,
    SUB_OBJECT_HEADER,
    SUB_OBJECT_NAME,
    SUB_OBJECT_PROPERTIES,
    FrameItems,
    FrameItemsDecodeError,
    ObjectHeader,
    ObjectInfo,
    decode_frame_items,
    object_type_name,
)
from fnaf_parser.decoders.object_common import (
    KNOWN_ZERO_PAD_BYTES,
    KNOWN_ZERO_PAD_END,
    KNOWN_ZERO_PAD_START,
)
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"
LAST_CHUNK_ID = 0x7F7F

FNAF1_ACTIVE_COUNT = 124
FNAF1_ACTIVE_TOTAL_PROPERTIES_BYTES = 36_988
FNAF1_ACTIVE_ANIMATION_FRAMES = 511
FNAF1_ACTIVE_ANIMATION_DIRECTIONS = 234
FNAF1_ACTIVE_UNIQUE_IMAGE_HANDLES = 389


# --- Synthetic helpers ---------------------------------------------------


def _pack_sub_header(sub_id: int, flags: int, size: int) -> bytes:
    """Pack a 0x2229 inner TLV header: int16 id + uint16 flags + uint32 size."""
    return struct.pack("<hHI", sub_id, flags, size)


def _pack_object_header_body(
    handle: int,
    object_type: int,
    flags: int = 0,
    reserved: int = 0,
    ink_effect: int = 0,
    ink_effect_param: int = 0,
) -> bytes:
    """Pack a 16-byte 0x4444 Object Info Header body."""
    return struct.pack(
        "<hhHhII",
        handle,
        object_type,
        flags,
        reserved,
        ink_effect,
        ink_effect_param,
    )


def _pack_name_body(name: str, *, unicode: bool = True) -> bytes:
    """Pack a 0x4445 name body: null-terminated string."""
    if unicode:
        return name.encode("utf-16-le") + b"\x00\x00"
    return name.encode("cp1252") + b"\x00"


def _pack_sub_chunk(sub_id: int, body: bytes, *, flags: int = 0) -> bytes:
    """Pack one inner sub-chunk: header + body (flag 0, not compressed)."""
    return _pack_sub_header(sub_id, flags, len(body)) + body


def _pack_object_info(
    handle: int,
    object_type: int,
    *,
    name: str | None = "Obj",
    properties: bytes | None = b"\x00\x00\x00\x00",
    effects: bytes | None = None,
    ink_effect: int = 0,
    ink_effect_param: int = 0,
    obj_flags: int = 0,
    reserved: int = 0,
    unicode: bool = True,
) -> bytes:
    """Pack a complete ObjectInfo block (header + optional name + opaque
    body chunks + LAST terminator)."""
    parts = [
        _pack_sub_chunk(
            SUB_OBJECT_HEADER,
            _pack_object_header_body(
                handle=handle,
                object_type=object_type,
                flags=obj_flags,
                reserved=reserved,
                ink_effect=ink_effect,
                ink_effect_param=ink_effect_param,
            ),
        )
    ]
    if name is not None:
        parts.append(
            _pack_sub_chunk(SUB_OBJECT_NAME, _pack_name_body(name, unicode=unicode))
        )
    if properties is not None:
        parts.append(_pack_sub_chunk(SUB_OBJECT_PROPERTIES, properties))
    if effects is not None:
        parts.append(_pack_sub_chunk(SUB_OBJECT_EFFECTS, effects))
    parts.append(_pack_sub_chunk(LAST_CHUNK_ID, b""))
    return b"".join(parts)


def _pack_payload(blocks: list[bytes]) -> bytes:
    """u32 count + N × ObjectInfo blocks."""
    count = len(blocks).to_bytes(FRAME_ITEMS_COUNT_SIZE, "little", signed=True)
    return count + b"".join(blocks)


# --- Module constants ----------------------------------------------------


def test_module_constants_stable():
    """Antibody #2: pin the load-bearing sizes + ids used in every error
    message, and the object-type enum values."""
    assert FRAME_ITEMS_COUNT_SIZE == 4
    assert SUB_HEADER_SIZE == 8
    assert OBJECT_HEADER_SIZE == 16
    assert SUB_OBJECT_HEADER == 0x4444
    assert SUB_OBJECT_NAME == 0x4445
    assert SUB_OBJECT_PROPERTIES == 0x4446
    assert SUB_OBJECT_EFFECTS == 0x4448
    assert EXTENSION_BASE == 32
    assert OPAQUE_SUB_CHUNK_IDS == frozenset({0x4446, 0x4447, 0x4448})
    assert OBJECT_TYPE_QUICKBACKDROP == 0
    assert OBJECT_TYPE_BACKDROP == 1
    assert OBJECT_TYPE_ACTIVE == 2
    assert OBJECT_TYPE_TEXT == 3
    assert OBJECT_TYPE_COUNTER == 7


def test_object_type_name_coverage():
    """Every named fixed enum value + the extension bucket rendering."""
    assert object_type_name(-7) == "Player"
    assert object_type_name(0) == "QuickBackdrop"
    assert object_type_name(2) == "Active"
    assert object_type_name(9) == "SubApplication"
    assert object_type_name(32) == "Extension(32)"
    assert object_type_name(99) == "Extension(99)"
    # Outside any known range — surface "Unknown(X)" so a drift-caused
    # bogus id is visible in error strings rather than blanked out.
    assert object_type_name(-8) == "Unknown(-8)"


# --- Synthetic positive cases -------------------------------------------


def test_zero_count_empty_payload():
    """Zero-count payload is just a u32 zero. Exercises the count-only
    path (no ObjectInfo iterations)."""
    fi = decode_frame_items(_pack_payload([]))
    assert isinstance(fi, FrameItems)
    assert fi.count == 0
    assert fi.items == ()
    assert fi.handles == frozenset()
    assert fi.by_handle == {}
    assert fi.object_type_histogram == {}
    assert fi.deferred_encrypted == ()
    assert fi.deferred_sub_chunk_ids_seen == frozenset()


def test_roundtrip_single_object_info():
    """Antibody #3: header + name + properties + LAST round-trips losslessly."""
    payload = _pack_payload(
        [
            _pack_object_info(
                handle=42,
                object_type=OBJECT_TYPE_ACTIVE,
                name="Freddy",
                properties=b"\xde\xad\xbe\xef",
                ink_effect=(1 << 28) | 0x5,  # transparent + ink_id 5
                ink_effect_param=0x1234,
                obj_flags=0b0011,
            )
        ]
    )
    fi = decode_frame_items(
        payload,
        unicode=True,
        decode_active_properties=False,
    )
    assert fi.count == 1
    assert len(fi.items) == 1
    obj = fi.items[0]
    assert isinstance(obj, ObjectInfo)
    assert isinstance(obj.header, ObjectHeader)
    assert obj.handle == 42
    assert obj.object_type == OBJECT_TYPE_ACTIVE
    assert obj.header.flags == 0b0011
    assert obj.header.ink_effect == (1 << 28) | 0x5
    assert obj.header.ink_effect_id == 0x5
    assert obj.header.transparent is True
    assert obj.header.antialias is False
    assert obj.header.ink_effect_param == 0x1234
    assert obj.name == "Freddy"
    assert obj.properties_raw == b"\xde\xad\xbe\xef"
    assert obj.properties is None
    assert obj.effects_raw == b""
    # by_handle resolves the template
    assert fi.by_handle[42] is obj
    assert fi.handles == frozenset({42})
    assert fi.object_type_histogram == {OBJECT_TYPE_ACTIVE: 1}
    # 0x4446 Properties was present, so it lands in the loud-skip surface.
    assert fi.deferred_sub_chunk_ids_seen == frozenset({SUB_OBJECT_PROPERTIES})


def test_roundtrip_multiple_object_infos_preserve_order():
    """Antibody #3: three ObjectInfos with distinct handles + types all
    round-trip and preserve on-wire order. Also pins `by_handle`
    bijectivity (no collisions on the supplied handles)."""
    specs = [
        (0, OBJECT_TYPE_BACKDROP, "Bg"),
        (1, OBJECT_TYPE_ACTIVE, "Enemy"),
        (2, OBJECT_TYPE_COUNTER, "Score"),
    ]
    payload = _pack_payload(
        [_pack_object_info(h, t, name=n) for h, t, n in specs]
    )
    fi = decode_frame_items(
        payload,
        unicode=True,
        decode_active_properties=False,
    )
    assert fi.count == 3
    assert [it.handle for it in fi.items] == [0, 1, 2]
    assert [it.object_type for it in fi.items] == [
        OBJECT_TYPE_BACKDROP,
        OBJECT_TYPE_ACTIVE,
        OBJECT_TYPE_COUNTER,
    ]
    assert [it.name for it in fi.items] == ["Bg", "Enemy", "Score"]
    assert fi.object_type_histogram == {
        OBJECT_TYPE_BACKDROP: 1,
        OBJECT_TYPE_ACTIVE: 1,
        OBJECT_TYPE_COUNTER: 1,
    }


def test_name_absent_is_allowed():
    """0x4445 is optional (system objects sometimes skip it). Missing
    name leaves `.name is None` without raising."""
    payload = _pack_payload(
        [_pack_object_info(handle=3, object_type=-3, name=None, properties=None)]
    )
    fi = decode_frame_items(payload)
    assert fi.items[0].name is None
    assert fi.items[0].properties_raw == b""
    # No 0x4446 present → deferred set empty.
    assert fi.deferred_sub_chunk_ids_seen == frozenset()


def test_extension_object_type_accepted():
    """Extension ids (>= 32) are an open range. Probe #5 accepts them
    without knowing which plugin is behind them — the follow-up ObjectCommon
    probe will decode the per-type body."""
    payload = _pack_payload(
        [
            _pack_object_info(handle=0, object_type=32, name="Plugin32"),
            _pack_object_info(handle=1, object_type=999, name="Plugin999"),
        ]
    )
    fi = decode_frame_items(payload)
    assert [it.object_type for it in fi.items] == [32, 999]


def test_ink_effect_antialias_bit():
    """Antibody #4 multi-oracle: bit 29 = antialias per Anaconda's
    `if ink_effect & 0x20000000:` check (objectinfo.pyx line 188)."""
    ink = (1 << 29) | (1 << 28) | 0xA  # antialias + transparent + id 10
    payload = _pack_payload(
        [_pack_object_info(handle=0, object_type=2, ink_effect=ink)]
    )
    obj = decode_frame_items(payload, decode_active_properties=False).items[0]
    assert obj.header.antialias is True
    assert obj.header.transparent is True
    assert obj.header.ink_effect_id == 0xA


def test_as_dict_shape():
    """Snapshot-style pin of FrameItems.as_dict() output shape."""
    payload = _pack_payload(
        [_pack_object_info(handle=1, object_type=2, name="X")]
    )
    d = decode_frame_items(payload, decode_active_properties=False).as_dict()
    assert d["count"] == 1
    assert d["deferred_encrypted"] == []
    assert d["deferred_sub_chunk_ids_seen"] == ["0x4446"]
    assert d["object_type_histogram"] == {"2": 1}
    assert len(d["items"]) == 1
    item = d["items"][0]
    assert item["name"] == "X"
    assert item["header"]["handle"] == 1
    assert item["header"]["object_type_name"] == "Active"
    assert item["properties_len"] == 4
    assert item["properties_decoded"] is False
    assert item["properties"] is None
    assert item["effects_len"] == 0


# --- Antibody #1 / #2 negatives -----------------------------------------


def test_payload_shorter_than_count_prefix_raises():
    """Antibody #2: payload too short to even hold the u32 count prefix."""
    with pytest.raises(FrameItemsDecodeError, match="too short"):
        decode_frame_items(b"\x00\x00\x00")


def test_negative_count_raises():
    """Antibody #1: signed-int32 decode yields a negative count when
    RC4 drifts (the top 4 bytes of the plaintext are garbage)."""
    with pytest.raises(FrameItemsDecodeError, match="Negative counts"):
        decode_frame_items(b"\xff\xff\xff\xff")  # -1


def test_count_claims_more_than_payload_holds_raises():
    """Antibody #2: outer count = 1 but no ObjectInfo bytes follow."""
    with pytest.raises(FrameItemsDecodeError, match="payload exhausted"):
        decode_frame_items(b"\x01\x00\x00\x00")


def test_unknown_inner_sub_chunk_id_raises():
    """Antibody #1: an inner TLV with an id not in CHUNK_NAMES is a
    schema violation — raise rather than swallow."""
    header_chunk = _pack_sub_chunk(
        SUB_OBJECT_HEADER, _pack_object_header_body(0, OBJECT_TYPE_ACTIVE)
    )
    # 0x1234 is within int16 range (id field is int16) but not in CHUNK_NAMES.
    bogus = _pack_sub_chunk(0x1234, b"")
    last = _pack_sub_chunk(LAST_CHUNK_ID, b"")
    payload = _pack_payload([header_chunk + bogus + last])
    with pytest.raises(FrameItemsDecodeError, match="unknown inner sub-chunk"):
        decode_frame_items(payload)


def test_missing_last_terminator_raises():
    """Antibody #1: running out of payload without hitting 0x7F7F LAST
    means the envelope is truncated."""
    # One ObjectInfo: header only, no LAST.
    header_chunk = _pack_sub_chunk(
        SUB_OBJECT_HEADER, _pack_object_header_body(0, OBJECT_TYPE_ACTIVE)
    )
    payload = _pack_payload([header_chunk])
    with pytest.raises(FrameItemsDecodeError, match="without hitting 0x7F7F"):
        decode_frame_items(payload)


def test_missing_header_raises():
    """Antibody #1: ObjectInfo with name + LAST but no 0x4444 header."""
    name_chunk = _pack_sub_chunk(SUB_OBJECT_NAME, _pack_name_body("X"))
    last = _pack_sub_chunk(LAST_CHUNK_ID, b"")
    payload = _pack_payload([name_chunk + last])
    with pytest.raises(FrameItemsDecodeError, match="missing the mandatory 0x4444"):
        decode_frame_items(payload)


def test_two_headers_raises():
    """Antibody #1: two 0x4444 chunks in the same ObjectInfo is an
    impossible state (the bank is a flat list, no nesting)."""
    hdr_a = _pack_sub_chunk(
        SUB_OBJECT_HEADER, _pack_object_header_body(0, OBJECT_TYPE_ACTIVE)
    )
    hdr_b = _pack_sub_chunk(
        SUB_OBJECT_HEADER, _pack_object_header_body(1, OBJECT_TYPE_ACTIVE)
    )
    last = _pack_sub_chunk(LAST_CHUNK_ID, b"")
    payload = _pack_payload([hdr_a + hdr_b + last])
    with pytest.raises(FrameItemsDecodeError, match="two 0x4444 headers"):
        decode_frame_items(payload)


def test_header_wrong_body_size_raises():
    """Antibody #2: 0x4444 body must be exactly 16 bytes."""
    short_hdr = _pack_sub_chunk(SUB_OBJECT_HEADER, b"\x00" * 15)
    last = _pack_sub_chunk(LAST_CHUNK_ID, b"")
    payload = _pack_payload([short_hdr + last])
    with pytest.raises(FrameItemsDecodeError, match="body length 15"):
        decode_frame_items(payload)


def test_invalid_object_type_raises():
    """Antibody #1: object_type outside the closed set raises. -10 is
    below OBJECT_TYPE_PLAYER (-7) and below the extension base (32)."""
    payload = _pack_payload(
        [_pack_object_info(handle=0, object_type=-10, name="Bad")]
    )
    with pytest.raises(FrameItemsDecodeError, match="object_type=-10"):
        decode_frame_items(payload)


def test_invalid_object_type_between_fixed_and_extension_raises():
    """Antibody #1: 20 sits between the highest fixed type (9) and the
    extension base (32) — no-man's-land. Must raise."""
    payload = _pack_payload(
        [_pack_object_info(handle=0, object_type=20, name="Bad")]
    )
    with pytest.raises(FrameItemsDecodeError, match="object_type=20"):
        decode_frame_items(payload)


def test_duplicate_handle_raises():
    """Antibody #1: FrameItems is a handle → ObjectInfo bijection; two
    ObjectInfos with the same handle is a schema violation (it also
    breaks the cross-chunk antibody with 0x3338 FrameItemInstances)."""
    payload = _pack_payload(
        [
            _pack_object_info(handle=7, object_type=2, name="A"),
            _pack_object_info(handle=7, object_type=2, name="B"),
        ]
    )
    with pytest.raises(FrameItemsDecodeError, match="duplicate handle 7"):
        decode_frame_items(payload, decode_active_properties=False)


def test_inner_sub_chunk_size_overrun_raises():
    """Antibody #2: an inner TLV claiming a larger size than bytes remain."""
    # Header chunk that claims size=100 but provides only 4 bytes.
    bad = _pack_sub_header(SUB_OBJECT_HEADER, 0, 100) + b"\x00" * 4
    payload = _pack_payload([bad])
    with pytest.raises(FrameItemsDecodeError, match="claims size=100"):
        decode_frame_items(payload)


def test_trailing_junk_after_last_object_info_raises():
    """Antibody #2: if `count` ObjectInfos consume fewer bytes than the
    payload contains, the excess is a drift signal — we must not silently
    swallow it."""
    block = _pack_object_info(handle=0, object_type=2)
    payload = _pack_payload([block]) + b"\xAA\xBB\xCC"
    with pytest.raises(FrameItemsDecodeError, match="unconsumed"):
        decode_frame_items(payload, decode_active_properties=False)


# --- FNAF 1 multi-input (Antibody #5 / #7) ------------------------------


def _fnaf1_transform_and_records():
    """Mirrors the helper in sibling flag=3 tests — walk the full pack
    with the transform wired up so we can read any chunk in the live
    FNAF 1 binary."""
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
def test_fnaf1_frame_items_decodes_without_error():
    """Antibody #5 multi-input: the full FNAF 1 0x2229 payload decodes
    without raising and produces a non-empty FrameItems bank."""
    blob, result, transform = _fnaf1_transform_and_records()
    fi_recs = [r for r in result.records if r.id == 0x2229]
    assert len(fi_recs) == 1, (
        f"FNAF 1 should carry exactly one 0x2229 chunk; saw {len(fi_recs)}"
    )
    payload = read_chunk_payload(blob, fi_recs[0], transform=transform)
    fi = decode_frame_items(
        payload, unicode=result.header.unicode, transform=transform
    )
    assert fi.count > 0


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_frame_items_snapshot():
    """Antibody #7 snapshot: pin the FNAF 1 0x2229 decode against drift.
    Dual-confirmed expected shape from CTFAK + Anaconda:

    - 196 ObjectInfos total
    - Histogram: 15 Backdrop + 124 Active + 10 Text + 44 Counter + 3
      extension plugins (one each at ids 32, 33, 34)
    - Handles dense from 0 upward (no gaps in the first 10)
    - First ObjectInfo is handle=0, Text, name="String"
    - Only 0x4446 Properties appear in the opaque set (no 0x4448 Effects)
    - No flag=2/3 encrypted inner sub-chunks were deferred (transform
      was supplied and covered all of them).
    """
    blob, result, transform = _fnaf1_transform_and_records()
    fi_rec = next(r for r in result.records if r.id == 0x2229)
    payload = read_chunk_payload(blob, fi_rec, transform=transform)
    fi = decode_frame_items(
        payload, unicode=result.header.unicode, transform=transform
    )

    assert fi.count == 196
    assert fi.object_type_histogram == {
        OBJECT_TYPE_BACKDROP: 15,
        OBJECT_TYPE_ACTIVE: 124,
        OBJECT_TYPE_TEXT: 10,
        OBJECT_TYPE_COUNTER: 44,
        32: 1,
        33: 1,
        34: 1,
    }
    assert sorted(fi.handles)[:10] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    first = fi.items[0]
    assert first.handle == 0
    assert first.object_type == OBJECT_TYPE_TEXT
    assert first.name == "String"
    assert fi.deferred_sub_chunk_ids_seen == frozenset({SUB_OBJECT_PROPERTIES})
    assert fi.deferred_encrypted == ()
    # Every ObjectInfo carries a 16-byte 0x4444 header with a valid
    # object_type — the decoder's invariants, re-checked at the snapshot
    # layer so any silent fallback would surface here too.
    for item in fi.items:
        assert isinstance(item.header, ObjectHeader)
        assert item.name is not None and item.name != ""


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_active_object_common_snapshot(fnaf1_image_bank):
    """Antibody #8: Active 0x4446 bodies decode as ObjectCommon.

    Pins the new structured scope:

    - all 124 Active objects decode, while non-Active objects stay raw
    - ObjectCommon header.size lines up with the raw property byte length
    - the only unconsumed coverage gap is the known [62..70) zero pad
    - animation-table counts match the probe inventory
    - every animation image handle resolves into the sparse ImageBank
      handle set (not `handle < image_count`)
    """
    blob, result, transform = _fnaf1_transform_and_records()
    fi_rec = next(r for r in result.records if r.id == 0x2229)
    payload = read_chunk_payload(blob, fi_rec, transform=transform)
    fi = decode_frame_items(
        payload, unicode=result.header.unicode, transform=transform
    )

    active = [item for item in fi.items if item.object_type == OBJECT_TYPE_ACTIVE]
    non_active = [item for item in fi.items if item.object_type != OBJECT_TYPE_ACTIVE]
    assert len(active) == FNAF1_ACTIVE_COUNT
    assert all(item.properties is not None for item in active)
    assert all(item.properties is None for item in non_active)
    assert sum(len(item.properties_raw) for item in active) == (
        FNAF1_ACTIVE_TOTAL_PROPERTIES_BYTES
    )

    total_frames = 0
    total_directions = 0
    all_animation_handles: set[int] = set()
    for item in active:
        props = item.properties
        assert props is not None
        assert props.header.identifier == "SPRI"
        assert props.header.size == len(item.properties_raw)
        assert len(props.coverage_gaps) == 1
        gap = props.coverage_gaps[0]
        assert gap.start == KNOWN_ZERO_PAD_START
        assert gap.end == KNOWN_ZERO_PAD_END
        assert gap.data == KNOWN_ZERO_PAD_BYTES
        assert gap.is_known_zero_pad is True
        assert (
            sum(span.size for span in props.coverage_spans)
            + sum(g.size for g in props.coverage_gaps)
        ) == len(item.properties_raw)

        assert props.animations is not None
        total_frames += props.animations.total_frames
        total_directions += props.animations.total_directions
        all_animation_handles.update(props.image_handles)

    assert total_frames == FNAF1_ACTIVE_ANIMATION_FRAMES
    assert total_directions == FNAF1_ACTIVE_ANIMATION_DIRECTIONS
    assert len(all_animation_handles) == FNAF1_ACTIVE_UNIQUE_IMAGE_HANDLES

    assert fnaf1_image_bank.count == 605
    missing_image_handles = sorted(all_animation_handles - fnaf1_image_bank.handles)
    assert missing_image_handles == []
    # Sparse-handle antibody: this must remain membership-in-bank, not
    # handle < image_count. FNAF 1 has valid animation handles above 605.
    assert max(all_animation_handles) > fnaf1_image_bank.count

    largest = max(active, key=lambda item: len(item.properties_raw))
    assert largest.handle == 44
    assert largest.name == "Active 3"
    assert len(largest.properties_raw) == 5304
    assert largest.properties is not None
    assert largest.properties.animations is not None
    assert largest.properties.animations.count == 76
    assert largest.properties.animations.total_frames == 176


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_deferred_set_excludes_decoded_active_properties():
    """Stale-field antibody (2026-04-25): `deferred_sub_chunk_ids_seen` must
    contain 0x4446 only when the body is genuinely undecoded — NOT when
    decode_object_common consumed it successfully. Pre-fix the parser added
    0x4446 unconditionally at the SUB_OBJECT_PROPERTIES branch, so even after
    every Active body decoded the deferred set still claimed work was
    outstanding. Existing assertions (FNAF1 snapshot) couldn't catch this
    because non-Actives also leave 0x4446 in the set legitimately.

    This test constructs a synthetic Active-only payload from a real FNAF1
    Active body (so `decode_object_common` will accept it) and asserts the
    deferred set is empty after a successful decode. If 0x4446 appears here
    the unconditional-add regression has returned.
    """
    # Borrow one real Active body from FNAF1 — guaranteed to decode cleanly.
    blob, result, transform = _fnaf1_transform_and_records()
    fi_rec = next(r for r in result.records if r.id == 0x2229)
    real_payload = read_chunk_payload(blob, fi_rec, transform=transform)
    real_fi = decode_frame_items(
        real_payload, unicode=result.header.unicode, transform=transform
    )
    real_active = next(
        item
        for item in real_fi.items
        if item.object_type == OBJECT_TYPE_ACTIVE and item.properties is not None
    )
    real_body = real_active.properties_raw

    # Active-only synthetic payload using the real body.
    synth = _pack_payload(
        [
            _pack_object_info(
                handle=h,
                object_type=OBJECT_TYPE_ACTIVE,
                name=f"A{h}",
                properties=real_body,
            )
            for h in range(3)
        ]
    )

    fi_decoded = decode_frame_items(
        synth, unicode=True, decode_active_properties=True
    )
    assert all(item.properties is not None for item in fi_decoded.items)
    assert fi_decoded.deferred_sub_chunk_ids_seen == frozenset(), (
        "0x4446 leaked into deferred set despite all Active bodies decoding "
        "— frame_items.py SUB_OBJECT_PROPERTIES unconditional-add regression."
    )

    # Negative control: same synthetic payload, decoding disabled. The body
    # is present but unconsumed, so 0x4446 MUST surface in the deferred set.
    fi_undecoded = decode_frame_items(
        synth, unicode=True, decode_active_properties=False
    )
    assert all(item.properties is None for item in fi_undecoded.items)
    assert fi_undecoded.deferred_sub_chunk_ids_seen == frozenset(
        {SUB_OBJECT_PROPERTIES}
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_cross_chunk_handle_resolution():
    """Cross-chunk antibody: every ObjectInstance.object_info handle
    referenced by 0x3338 FrameItemInstances across all 17 FNAF 1 frames
    must resolve to a known FrameItems.handle. This is the minimum-viable
    cross-check that drove probe #5's envelope-only scope.
    """
    from fnaf_parser.decoders.frame import decode_frame

    blob, result, transform = _fnaf1_transform_and_records()

    fi_rec = next(r for r in result.records if r.id == 0x2229)
    payload = read_chunk_payload(blob, fi_rec, transform=transform)
    fi = decode_frame_items(
        payload, unicode=result.header.unicode, transform=transform
    )
    known = fi.handles

    frame_records = [r for r in result.records if r.id == 0x3333]
    total_instances = 0
    unresolved: list[tuple[int, int]] = []  # (frame_idx, object_info_handle)
    for frame_idx, rec in enumerate(frame_records):
        frame_payload = read_chunk_payload(blob, rec, transform=transform)
        frame = decode_frame(
            frame_payload, unicode=result.header.unicode, transform=transform
        )
        if frame.item_instances is None:
            continue
        for inst in frame.item_instances.instances:
            total_instances += 1
            if inst.object_info not in known:
                unresolved.append((frame_idx, inst.object_info))

    assert total_instances > 0, "expected at least one ObjectInstance"
    assert not unresolved, (
        f"unresolved object_info handles in frames: {unresolved[:20]} "
        f"(total {len(unresolved)} / {total_instances} instances)"
    )
