"""Regression tests for the Text ObjectInfo body decoder.

Text is the FNAF 1 UI-label object_type (=3). Its 0x4446 property body
wraps a single Clickteam Paragraph (font handle + flags + color + UTF-16-LE
string) inside a 116-byte ObjectCommon-shaped opaque header. The decoder
consumes every wire field except the 112-byte opaque header span, which
is kept as ``opaque_header_raw`` for round-trip and future probing.

Antibody coverage:

- #1 strict-unknown: ``paragraph_count != 1`` raises (multi-paragraph
  Text not supported in FNAF 1); ``paragraph_offsets[0] != 20`` raises;
  embedded UTF-16-LE NUL raises; missing terminator raises.
- #2 byte-count: payloads under 146 bytes raise; ``size`` mirror at
  offset 0 must match ``len(payload)``; ``inner_size`` at offset 116
  must match ``body_size - 116``; UTF-16 region must be a whole number
  of code units.
- #3 round-trip: synthetic single-character and multi-character bodies
  pack and decode back to the original fields.
- #4 multi-oracle: integration cross-check against the FNAF 1 fixture
  is covered by ``test_frame_items.test_fnaf1_text_body_snapshot``.
"""

from __future__ import annotations

import struct

import pytest

from fnaf_parser.decoders.text_body import (
    TEXT_BODY_HEADER_SIZE,
    TEXT_BOX_HEIGHT_OFFSET,
    TEXT_BOX_WIDTH_OFFSET,
    TEXT_FNAF1_PARAGRAPH_COUNT,
    TEXT_FNAF1_PARAGRAPH_OFFSET_0,
    TEXT_INNER_SIZE_OFFSET,
    TEXT_MIN_BODY_SIZE,
    TEXT_OPAQUE_HEADER_END,
    TEXT_OPAQUE_HEADER_SIZE,
    TEXT_OPAQUE_HEADER_START,
    TEXT_PARAGRAPH_COLOR_OFFSET,
    TEXT_PARAGRAPH_COUNT_OFFSET,
    TEXT_PARAGRAPH_FLAGS_OFFSET,
    TEXT_PARAGRAPH_FONT_HANDLE_OFFSET,
    TEXT_PARAGRAPH_OFFSETS_OFFSET,
    TEXT_PARAGRAPH_VALUE_OFFSET,
    TEXT_SIZE_FIELD_OFFSET,
    TextBody,
    TextBodyDecodeError,
    decode_text_body,
)


def _pack_text_body(
    *,
    value: str,
    box_width: int = 1508,
    box_height: int = 189,
    paragraph_count: int = TEXT_FNAF1_PARAGRAPH_COUNT,
    paragraph_offset_0: int = TEXT_FNAF1_PARAGRAPH_OFFSET_0,
    font_handle: int = 20,
    flags: int = 1,
    color: int = 0x00FFFFFF,
    opaque_header_raw: bytes | None = None,
    size_override: int | None = None,
    inner_size_override: int | None = None,
    value_terminator: bytes = b"\x00\x00",
) -> bytes:
    """Pack a synthetic Text property body in the FNAF 1 wire layout.

    Defaults reflect the FNAF 1 fixture: 1 paragraph, offset 20 from inner
    Text-struct base, color white. ``value`` is encoded UTF-16-LE plus a
    trailing 2-byte NUL (overridable for malformed-terminator tests).
    """
    if opaque_header_raw is None:
        # Pad with deterministic bytes; the FNAF 1 fixture has byte-identical
        # 112-byte spans across all 10 bodies but the decoder treats them as
        # opaque, so we don't need to mirror the exact contents here.
        opaque_header_raw = bytes(TEXT_OPAQUE_HEADER_SIZE)
    if len(opaque_header_raw) != TEXT_OPAQUE_HEADER_SIZE:
        raise ValueError(
            f"opaque_header_raw must be {TEXT_OPAQUE_HEADER_SIZE} bytes; "
            f"got {len(opaque_header_raw)}"
        )

    string_bytes = value.encode("utf-16-le") + value_terminator
    body_size = TEXT_PARAGRAPH_VALUE_OFFSET + len(string_bytes)
    inner_size = body_size - TEXT_BODY_HEADER_SIZE
    size_field = body_size if size_override is None else size_override
    inner_size_field = (
        inner_size if inner_size_override is None else inner_size_override
    )

    return b"".join(
        [
            struct.pack("<I", size_field),
            opaque_header_raw,
            struct.pack("<I", inner_size_field),
            struct.pack("<I", box_width),
            struct.pack("<I", box_height),
            struct.pack("<I", paragraph_count),
            struct.pack("<I", paragraph_offset_0),
            struct.pack("<H", font_handle),
            struct.pack("<H", flags),
            struct.pack("<I", color),
            string_bytes,
        ]
    )


def test_module_constants_stable():
    """Pin offsets so any wire-layout drift surfaces in the smallest
    possible blast radius. Mirrors ``test_counter_body.test_module_constants_stable``
    and ``test_backdrop_body.test_module_constants_stable``."""
    assert TEXT_BODY_HEADER_SIZE == 116
    assert TEXT_SIZE_FIELD_OFFSET == 0
    assert TEXT_OPAQUE_HEADER_START == 4
    assert TEXT_OPAQUE_HEADER_END == 116
    assert TEXT_OPAQUE_HEADER_SIZE == 112
    assert TEXT_INNER_SIZE_OFFSET == 116
    assert TEXT_BOX_WIDTH_OFFSET == 120
    assert TEXT_BOX_HEIGHT_OFFSET == 124
    assert TEXT_PARAGRAPH_COUNT_OFFSET == 128
    assert TEXT_PARAGRAPH_OFFSETS_OFFSET == 132
    assert TEXT_PARAGRAPH_FONT_HANDLE_OFFSET == 136
    assert TEXT_PARAGRAPH_FLAGS_OFFSET == 138
    assert TEXT_PARAGRAPH_COLOR_OFFSET == 140
    assert TEXT_PARAGRAPH_VALUE_OFFSET == 144
    assert TEXT_MIN_BODY_SIZE == 146
    assert TEXT_FNAF1_PARAGRAPH_COUNT == 1
    assert TEXT_FNAF1_PARAGRAPH_OFFSET_0 == 20


def test_roundtrip_short_string():
    """Antibody #3 round-trip: the smallest observed FNAF 1 string (">",
    1 char, body_size 148) packs + decodes cleanly."""
    body = _pack_text_body(value=">")
    assert len(body) == 148

    decoded = decode_text_body(body)
    assert isinstance(decoded, TextBody)
    assert decoded.size == 148
    assert decoded.inner_size == 32
    assert decoded.box_width == 1508
    assert decoded.box_height == 189
    assert decoded.paragraph_count == 1
    assert decoded.paragraph_offsets == (20,)
    assert decoded.font_handle == 20
    assert decoded.flags == 1
    assert decoded.color == 0x00FFFFFF
    assert decoded.value == ">"
    assert decoded.char_count == 1


def test_roundtrip_multiline_string():
    """Antibody #3 round-trip: the FNAF 1 night-banner format
    ('12:00 AM\\r\\n\\r\\n7th Night', 21 chars, body_size 188) packs +
    decodes cleanly. Embedded \\r\\n must NOT trip the embedded-NUL
    check (CR=0x000D, LF=0x000A — neither is 0x0000)."""
    value = "12:00 AM\r\n\r\n7th Night"
    body = _pack_text_body(value=value, box_width=700, box_height=271)
    assert len(body) == 188

    decoded = decode_text_body(body)
    assert decoded.value == value
    assert decoded.char_count == 21
    assert decoded.box_width == 700
    assert decoded.box_height == 271


def test_roundtrip_long_string():
    """Antibody #3 round-trip: the largest FNAF 1 Text body (handle 194,
    'Thanks for playing the demo!...', 167 chars, body_size 480) packs
    + decodes cleanly."""
    value = "X" * 167
    body = _pack_text_body(value=value, box_width=812, box_height=625)
    assert len(body) == 480

    decoded = decode_text_body(body)
    assert decoded.size == 480
    assert decoded.value == value
    assert decoded.char_count == 167


def test_payload_too_short_raises():
    """Antibody #2 byte-count: payloads under 146 bytes are rejected.
    The 144-byte fixed prefix plus 2-byte NUL is the shortest legal
    Text body."""
    short = b"\x00" * 145
    with pytest.raises(TextBodyDecodeError, match="payload is 145 bytes"):
        decode_text_body(short)


def test_size_field_mismatch_raises():
    """Antibody #2: ``size`` field at offset 0 must equal ``len(payload)``.
    A drifted size (RC4 misalignment, slicing bug) raises."""
    body = bytearray(_pack_text_body(value="Demo", size_override=999))
    with pytest.raises(TextBodyDecodeError, match="size field at offset 0 is 999"):
        decode_text_body(bytes(body))


def test_inner_size_mismatch_raises():
    """Antibody #2: ``inner_size`` at offset 116 must equal
    ``body_size - 116``. A wrong inner length signals a layout drift
    that would silently mis-locate the Paragraph."""
    body = _pack_text_body(value="Demo", inner_size_override=99)
    with pytest.raises(
        TextBodyDecodeError, match="inner_size field at offset 116 is 99"
    ):
        decode_text_body(body)


def test_paragraph_count_not_one_raises():
    """Antibody #1 strict-unknown: FNAF 1 always emits paragraph_count=1.
    A higher count means multi-paragraph text, which the V0 decoder does
    not support. Loud failure beats silent truncation."""
    body = _pack_text_body(value="Demo", paragraph_count=2)
    with pytest.raises(TextBodyDecodeError, match="paragraph_count is 2"):
        decode_text_body(body)


def test_paragraph_offset_not_twenty_raises():
    """Antibody #1: ``paragraph_offsets[0]`` is always 20 in FNAF 1
    (right after the 5-u32 inner Text header). A different value would
    move the Paragraph and silently mis-locate font/flags/color."""
    body = _pack_text_body(value="Demo", paragraph_offset_0=24)
    with pytest.raises(
        TextBodyDecodeError, match=r"paragraph_offsets\[0\] is 24"
    ):
        decode_text_body(body)


def test_missing_nul_terminator_raises():
    """Antibody #1: value region must end in 0x0000. A non-NUL pair at
    the tail signals truncation or a non-Yuniversal encoding."""
    body = _pack_text_body(value="Demo", value_terminator=b"AB")
    with pytest.raises(
        TextBodyDecodeError, match="does not end in UTF-16-LE NUL"
    ):
        decode_text_body(body)


def test_embedded_nul_raises():
    """Antibody #1: embedded NUL inside the value region means we
    mis-sized the region or the body is malformed (Clickteam's reader
    stops at the first 0x0000). Rejecting it loud beats a silently
    truncated string."""
    # Pack a body whose declared string is "Foo" but the bytes contain
    # an embedded NUL ("Fo\0o"). The terminator is still present at the
    # tail, so the only failure path is the embedded-NUL check.
    bad_string_bytes = (
        "F".encode("utf-16-le")
        + b"\x00\x00"  # embedded NUL where 'o' should be
        + "o".encode("utf-16-le")
    )
    body = _pack_text_body(
        value="",  # placeholder; we replace the string region below
    )
    body_array = bytearray(body)
    # Strip the empty value's NUL and substitute the malformed string.
    new_string_region = bad_string_bytes + b"\x00\x00"
    new_body = bytes(body_array[:TEXT_PARAGRAPH_VALUE_OFFSET]) + new_string_region
    # Re-fix the size / inner_size fields for the new length.
    new_body = bytearray(new_body)
    struct.pack_into("<I", new_body, TEXT_SIZE_FIELD_OFFSET, len(new_body))
    struct.pack_into(
        "<I",
        new_body,
        TEXT_INNER_SIZE_OFFSET,
        len(new_body) - TEXT_BODY_HEADER_SIZE,
    )
    with pytest.raises(TextBodyDecodeError, match="embedded UTF-16-LE NUL"):
        decode_text_body(bytes(new_body))


def test_odd_byte_value_region_raises():
    """Antibody #2 byte-count: value region must be a whole number of
    UTF-16-LE code units. An odd-length tail signals byte-level
    corruption."""
    body = bytearray(_pack_text_body(value="Demo"))
    body.append(0)  # one extra byte → value region is now odd-length
    # Re-fix the size field; without this we fail the size check first.
    struct.pack_into("<I", body, TEXT_SIZE_FIELD_OFFSET, len(body))
    # Re-fix inner_size to match the new total length.
    struct.pack_into(
        "<I",
        body,
        TEXT_INNER_SIZE_OFFSET,
        len(body) - TEXT_BODY_HEADER_SIZE,
    )
    with pytest.raises(
        TextBodyDecodeError,
        match="not a whole number of UTF-16-LE code units",
    ):
        decode_text_body(bytes(body))


def test_flag_names_and_summary_shape():
    """``summary_dict`` carries the runtime-consumable fields for
    ``runtime_pack/object_bank/objects.json[*].properties_summary``,
    including human-readable flag names."""
    # flags=0x0001 → HorizontalCenter only.
    body = _pack_text_body(
        value="Loading...",
        box_width=1508,
        box_height=189,
        font_handle=20,
        flags=0x0001,
        color=0x00FFFFFF,
    )
    decoded = decode_text_body(body)
    assert decoded.flag_names == ("HorizontalCenter",)
    summary = decoded.summary_dict()
    assert summary["box_width"] == 1508
    assert summary["box_height"] == 189
    assert summary["font_handle"] == 20
    assert summary["flags"] == 1
    assert summary["flag_names"] == ["HorizontalCenter"]
    assert summary["color"] == 0x00FFFFFF
    assert summary["value"] == "Loading..."
    assert summary["char_count"] == 10


def test_as_dict_shape():
    """``as_dict`` is the debug surface used by ``ObjectInfo.as_dict``.
    Carries every wire field except the opaque header bytes (their
    length is included for accounting)."""
    body = _pack_text_body(
        value="Game Over", box_width=1508, box_height=189
    )
    decoded = decode_text_body(body)
    as_dict = decoded.as_dict()
    assert as_dict["value"] == "Game Over"
    assert as_dict["paragraph_offsets"] == [20]
    assert as_dict["opaque_header_raw_len"] == TEXT_OPAQUE_HEADER_SIZE
