"""Regression tests for the Counter ObjectInfo body decoder.

Counter is the FNAF 1 named-scalar-global object_type (=7). Its 0x4446
property body is a 152-byte fixed prefix plus a u16-prefixed image-handle
list. The decoder consumes ``display_style`` + ``image_handles`` for the
Rust runtime; ``header_raw`` and ``display_config_raw`` are kept opaque.

Antibody coverage:

- #2 byte-count: payload shorter than the 152-byte prefix raises;
  ``size`` field at offset 0 must match ``len(payload)``; trailing
  handle list slice must equal ``2 * count`` bytes exactly.
- #3 round-trip: synthetic Counter body packs and decodes back to the
  original fields.
- #4 multi-oracle: handle list semantics cross-checked against the
  FNAF 1 fixture (covered by ``test_frame_items.test_fnaf1_counter_body_snapshot``).
"""

from __future__ import annotations

import struct

import pytest

from fnaf_parser.decoders.counter_body import (
    COUNTER_DISPLAY_CONFIG_SIZE,
    COUNTER_DISPLAY_STYLE_OFFSET,
    COUNTER_HANDLE_COUNT_OFFSET,
    COUNTER_HANDLE_LIST_OFFSET,
    COUNTER_HEADER_RAW_SIZE,
    COUNTER_MIN_BODY_SIZE,
    CounterBody,
    CounterBodyDecodeError,
    decode_counter_body,
)


def _pack_counter_body(
    *,
    display_style: int,
    image_handles: tuple[int, ...],
    header_raw: bytes | None = None,
    display_config_raw: bytes | None = None,
    size_override: int | None = None,
) -> bytes:
    """Pack a synthetic Counter property body in the FNAF 1 wire layout.

    Defaults: zero-byte ``header_raw`` (114 B) and zero-byte
    ``display_config_raw`` (28 B) — semantic content does not matter for
    the V0 decoder which keeps both opaque.
    """
    if header_raw is None:
        header_raw = b"\x00" * COUNTER_HEADER_RAW_SIZE
    if display_config_raw is None:
        display_config_raw = b"\x00" * COUNTER_DISPLAY_CONFIG_SIZE
    assert len(header_raw) == COUNTER_HEADER_RAW_SIZE
    assert len(display_config_raw) == COUNTER_DISPLAY_CONFIG_SIZE

    payload_size = COUNTER_HANDLE_LIST_OFFSET + 2 * len(image_handles)
    size_field = payload_size if size_override is None else size_override

    parts = [
        struct.pack("<I", size_field),
        header_raw,
        struct.pack("<I", display_style),
        display_config_raw,
        struct.pack("<H", len(image_handles)),
    ]
    if image_handles:
        parts.append(struct.pack(f"<{len(image_handles)}H", *image_handles))
    return b"".join(parts)


def test_module_constants_stable():
    """Pin offsets so any wire-layout drift surfaces in the smallest
    possible blast radius. Mirrors ``test_frame_items.test_module_constants_stable``.
    """
    assert COUNTER_DISPLAY_STYLE_OFFSET == 118
    assert COUNTER_HANDLE_COUNT_OFFSET == 150
    assert COUNTER_HANDLE_LIST_OFFSET == 152
    assert COUNTER_HEADER_RAW_SIZE == 114
    assert COUNTER_DISPLAY_CONFIG_SIZE == 28
    assert COUNTER_MIN_BODY_SIZE == 152


def test_roundtrip_baseline_180_byte_body():
    """Antibody #3 round-trip: 14 sequential u16 handles (the FNAF 1
    baseline shape) packs + decodes to 180 bytes, matching the size of
    the 43 baseline counters."""
    handles = tuple(range(187, 201))  # 14 handles, like 'night number'
    body = _pack_counter_body(display_style=1, image_handles=handles)
    assert len(body) == 180

    decoded = decode_counter_body(body)
    assert isinstance(decoded, CounterBody)
    assert decoded.size == 180
    assert decoded.display_style == 1
    assert decoded.image_handles == handles
    assert decoded.image_handle_count == 14
    assert len(decoded.header_raw) == 114
    assert len(decoded.display_config_raw) == 28
    assert decoded.unique_image_handles == frozenset(handles)


def test_roundtrip_outlier_162_byte_body():
    """Antibody #3 round-trip: 5-handle body packs + decodes to 162
    bytes, matching the 'usage meter' outlier's wire size."""
    handles = (212, 213, 214, 456, 455)
    body = _pack_counter_body(display_style=1, image_handles=handles)
    assert len(body) == 162

    decoded = decode_counter_body(body)
    assert decoded.size == 162
    assert decoded.image_handles == handles


def test_roundtrip_zero_handles_minimum_body():
    """A Counter with zero image handles yields a 152-byte body — the
    smallest legal Counter wire shape."""
    body = _pack_counter_body(display_style=0, image_handles=())
    assert len(body) == COUNTER_MIN_BODY_SIZE

    decoded = decode_counter_body(body)
    assert decoded.size == 152
    assert decoded.image_handles == ()
    assert decoded.image_handle_count == 0


def test_display_style_round_trips_full_fnaf1_enum():
    """All five FNAF 1 ``display_style`` values pass through unchanged."""
    for ds in (0, 1, 3, 10, 12):
        body = _pack_counter_body(display_style=ds, image_handles=(1, 2, 3))
        assert decode_counter_body(body).display_style == ds


def test_payload_shorter_than_min_size_raises():
    """Antibody #2 byte-count: payloads under the 152-byte fixed prefix
    are rejected loudly so a truncated wire shape can't sneak through."""
    short = b"\x00" * 151
    with pytest.raises(CounterBodyDecodeError, match="smaller than the 152-byte"):
        decode_counter_body(short)


def test_size_field_mismatch_raises():
    """Antibody #2: ``size`` field at offset 0 must equal ``len(payload)``.
    A drifted size (e.g. RC4 misalignment, slicing bug) raises."""
    body = bytearray(_pack_counter_body(display_style=0, image_handles=(1,)))
    # Stamp a wrong size and assert the antibody fires.
    struct.pack_into("<I", body, 0, 999)
    with pytest.raises(CounterBodyDecodeError, match="size field at offset 0 is 999"):
        decode_counter_body(bytes(body))


def test_handle_count_overrun_raises():
    """Antibody #2: a count word that claims more handles than the
    payload tail actually holds raises (no silent truncation)."""
    body = bytearray(_pack_counter_body(display_style=0, image_handles=(1, 2)))
    # Overwrite the handle count to claim 99 handles while the tail only
    # carries 2; the size field still reflects the real wire size.
    struct.pack_into("<H", body, COUNTER_HANDLE_COUNT_OFFSET, 99)
    with pytest.raises(CounterBodyDecodeError, match="handle_count=99"):
        decode_counter_body(bytes(body))


def test_handle_count_underrun_raises():
    """Antibody #2: a count word that under-claims handles leaves
    trailing junk in the payload — also a wire-shape violation."""
    body = bytearray(_pack_counter_body(display_style=0, image_handles=(1, 2, 3, 4)))
    struct.pack_into("<H", body, COUNTER_HANDLE_COUNT_OFFSET, 1)
    with pytest.raises(CounterBodyDecodeError, match="handle_count=1"):
        decode_counter_body(bytes(body))


def test_summary_dict_shape():
    """``summary_dict`` carries the runtime-consumable fields for
    ``runtime_pack/object_bank/objects.json[*].properties_summary``."""
    body = _pack_counter_body(display_style=3, image_handles=(457, 458, 458))
    decoded = decode_counter_body(body)
    summary = decoded.summary_dict()
    assert summary == {
        "size": 158,
        "display_style": 3,
        "image_handle_count": 3,
        "unique_image_handles": [457, 458],
    }


def test_as_dict_shape():
    """``as_dict`` is the debug surface used by ``ObjectInfo.as_dict``;
    it carries raw-span lengths so a snapshot can detect silent layout
    drift without relying on the bytes themselves."""
    body = _pack_counter_body(display_style=10, image_handles=(1, 2, 3, 4, 5))
    decoded = decode_counter_body(body)
    d = decoded.as_dict()
    assert d == {
        "size": 162,
        "display_style": 10,
        "image_handles": [1, 2, 3, 4, 5],
        "header_raw_len": 114,
        "display_config_raw_len": 28,
    }
