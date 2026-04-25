"""Text body decoder for ObjectInfo Properties (0x4446) bodies whose
``object_type`` is ``OBJECT_TYPE_TEXT`` (=3).

Text objects are FNAF 1's UI labels: the WARNING screen blurb, "Loading...",
"Game Over", "12:00 AM\\r\\n\\r\\n7th Night", the version number, the demo
ribbons, and similar. There are exactly 10 of them. Each Text body wraps a
single Clickteam Paragraph (font handle + flags + color + UTF-16-LE string)
inside an ObjectCommon-shaped 116-byte header whose inner table happens to
be the Text struct CTFAK names ``Text`` (Width / Height / paragraph count /
paragraph offsets / paragraphs).

Wire format (FNAF 1 build-284, post decompress)
------------------------------------------------

    offset  size  field
    ------  ----  -----
       0     4    u32  body_size  (== len(payload); 148-480 in FNAF 1)
       4   112    bytes opaque_header_raw  (ObjectCommon-shaped wrapper;
                  byte-identical across all 10 bodies; ASCII identifier
                  "TEXT" at body offset 46; back_color 0x00FFFFFF at
                  body offset 50; kept opaque in V0)
     116     4    u32  inner_size  (== body_size - 116; size of the inner
                  Text struct including this length word)
     120     4    u32  box_width   (Clickteam virtual units; 220-1528 in
                  FNAF 1 — not screen pixels)
     124     4    u32  box_height  (65-625 in FNAF 1)
     128     4    u32  paragraph_count  (FNAF 1: always 1)
     132     4    u32  paragraph_offsets[0]  (FNAF 1: always 20, relative
                  to the start of the inner Text struct = body offset 116)
     136     2    u16  font_handle  (resolves against FontBank handle set)
     138     2    u16  flags  (BitDict: HorizontalCenter, RightAligned,
                  VerticalCenter, BottomAligned, ..., Correct, Relief)
     140     4    u32  color  (FNAF 1: always 0x00FFFFFF white)
     144   N+2    UTF-16-LE NUL-terminated string Value
                  (N = string_chars * 2; trailing 0x0000 NUL ends body)

Empirical FNAF 1 inventory (pinned in tests):

* 10 Texts total. Body sizes 148, 154, 156, 160, 164, 166, 188, 238, 314,
  480; one body per distinct size.
* All 10 bodies obey ``body_size == 146 + 2 * string_chars`` (144-byte
  pre-string region + 2*chars UTF-16 payload + 2-byte NUL terminator).
* All 10 carry ``paragraph_count == 1``, ``paragraph_offsets[0] == 20``,
  ``color == 0x00FFFFFF``. These three are FNAF-1 tripwires — not generic
  Clickteam invariants. If a future probe surfaces ``paragraph_count > 1``
  multi-paragraph support is required, and the decoder must split paragraphs
  along ``paragraph_offsets``.
* Strings observed: "WARNING!\\r\\n\\r\\nThis game contains flashing
  lights, loud noises, and lots of jumpscares!", "v 1.132", "Demo",
  "12:00 AM\\r\\n\\r\\n7th Night", "Night", "Loading...", "Game Over",
  "(0-2)easy  (3-6)med  (7-12)hard (13-20)extreme", ">", "Thanks for
  playing the demo!\\r\\n\\r\\nGet the full version...".

Scope cut
---------

V0 keeps ``opaque_header_raw`` as explicit opaque bytes — the 112-byte
ObjectCommon-shaped wrapper has no per-Text variation in FNAF 1 and the
runtime does not need it for rendering text labels. The runtime only
consumes ``font_handle``, ``flags``, ``color``, ``value`` (the Paragraph
fields). ``box_width`` / ``box_height`` are kept for round-trip. They
appear to be Clickteam virtual-coord units (twip-like), not screen
pixels — flag for the renderer pass to interpret.

Antibodies
----------

* ``len(payload) >= 146`` (fixed prefix up to the 2-byte NUL of an empty
  string).
* ``body_size`` field at offset 0 equals ``len(payload)``.
* ``inner_size`` field at offset 116 equals ``body_size - 116``.
* ``paragraph_count == 1`` (FNAF-1 tripwire — multi-paragraph would need
  a different decoder shape).
* ``paragraph_offsets[0] == 20`` (FNAF-1 tripwire).
* The UTF-16-LE region ``[144, body_size)`` is exactly ``string_chars * 2
  + 2`` bytes long, ends in a single ``0x0000`` NUL terminator at byte
  ``body_size - 2``, and contains no embedded NUL.
* Font handle and color values are decoded but not validated here; caller
  cross-checks ``font_handle`` against the FontBank handle set at the
  integration-test layer.

References
----------

* CTFAK2.0 ``Chunks/Objects/Text.cs::Text.Read`` (non-Old branch),
  ``Text.cs::Paragraph.Read`` (non-Old branch).
* CTFAK ``MMFParser/EXE/Loaders/Objects/Text.cs`` (the EXE-loader twin
  of the same shape; ``Settings.GameType.TwoFivePlus`` and
  ``OnePointFive`` branches diverge from the FNAF 1 shape and are
  not relevant here).
* Cross-checked against probe ``parser/temp-probes/text_body_2026_04_25/
  probe.py`` ship-log.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# --- Wire-format constants ---------------------------------------------

#: Pre-string region size: 4 (body_size) + 112 (opaque header) + 4 (inner_size)
#: + 4 (box_width) + 4 (box_height) + 4 (paragraph_count) + 4 (offsets[0])
#: + 2 (font_handle) + 2 (flags) + 4 (color) = 144 bytes. Add 2 bytes for
#: the UTF-16-LE NUL terminator of an empty string to get the smallest
#: legal Text body.
TEXT_BODY_HEADER_SIZE = 116

#: u32 ``body_size`` mirror at offset 0.
TEXT_SIZE_FIELD_OFFSET = 0

#: bytes [4..116) — opaque ObjectCommon-shaped wrapper (identifier "TEXT"
#: at body offset 46, back_color at offset 50). Byte-identical across all
#: 10 FNAF 1 Text bodies. Kept opaque in V0.
TEXT_OPAQUE_HEADER_START = 4
TEXT_OPAQUE_HEADER_END = TEXT_BODY_HEADER_SIZE
TEXT_OPAQUE_HEADER_SIZE = TEXT_OPAQUE_HEADER_END - TEXT_OPAQUE_HEADER_START
assert TEXT_OPAQUE_HEADER_SIZE == 112

#: u32 ``inner_size`` (size of the inner Text struct, including this word).
TEXT_INNER_SIZE_OFFSET = 116

#: u32 ``box_width`` of the text rendering box, in Clickteam virtual units.
TEXT_BOX_WIDTH_OFFSET = 120

#: u32 ``box_height`` of the text rendering box, in Clickteam virtual units.
TEXT_BOX_HEIGHT_OFFSET = 124

#: u32 ``paragraph_count``. FNAF 1: always 1.
TEXT_PARAGRAPH_COUNT_OFFSET = 128

#: u32 ``paragraph_offsets[0]``. FNAF 1: always 20 (relative to start of
#: the inner Text struct = body offset 116, so the paragraph starts at
#: body offset ``TEXT_BODY_HEADER_SIZE + 20 = 136``).
TEXT_PARAGRAPH_OFFSETS_OFFSET = 132

#: u16 Paragraph.FontHandle.
TEXT_PARAGRAPH_FONT_HANDLE_OFFSET = 136

#: u16 Paragraph.Flags (BitDict).
TEXT_PARAGRAPH_FLAGS_OFFSET = 138

#: u32 Paragraph.Color (RGBA / BGRA — Clickteam stores low-byte-red here;
#: FNAF 1 always emits 0x00FFFFFF white).
TEXT_PARAGRAPH_COLOR_OFFSET = 140

#: Offset where the UTF-16-LE NUL-terminated string starts.
TEXT_PARAGRAPH_VALUE_OFFSET = 144

#: Smallest legal FNAF Text body: 144-byte prefix + 2-byte NUL = 146 bytes
#: (an empty string, never observed in FNAF 1 but legally encodable).
TEXT_MIN_BODY_SIZE = TEXT_PARAGRAPH_VALUE_OFFSET + 2
assert TEXT_MIN_BODY_SIZE == 146

#: FNAF-1 tripwire: paragraph_count is always 1.
TEXT_FNAF1_PARAGRAPH_COUNT = 1

#: FNAF-1 tripwire: paragraph_offsets[0] is always 20 bytes from the
#: start of the inner Text struct.
TEXT_FNAF1_PARAGRAPH_OFFSET_0 = 20

#: BitDict bit names matching CTFAK2.0 Paragraph.Flags. Bit i of the u16
#: flags field maps to ``TEXT_FLAG_NAMES[i]`` when set.
TEXT_FLAG_NAMES: tuple[str, ...] = (
    "HorizontalCenter",
    "RightAligned",
    "VerticalCenter",
    "BottomAligned",
    "None",
    "None",
    "None",
    "None",
    "Correct",
    "Relief",
)

_SIZE_STRUCT = struct.Struct("<I")
_INNER_SIZE_STRUCT = struct.Struct("<I")
_BOX_WIDTH_STRUCT = struct.Struct("<I")
_BOX_HEIGHT_STRUCT = struct.Struct("<I")
_PARAGRAPH_COUNT_STRUCT = struct.Struct("<I")
_PARAGRAPH_OFFSET_STRUCT = struct.Struct("<I")
_FONT_HANDLE_STRUCT = struct.Struct("<H")
_FLAGS_STRUCT = struct.Struct("<H")
_COLOR_STRUCT = struct.Struct("<I")


class TextBodyDecodeError(ValueError):
    """Text body decode failure with offset / field context."""


# --- Helpers ------------------------------------------------------------


def _flag_names(flags: int) -> tuple[str, ...]:
    """Return human-readable names for every set bit covered by
    ``TEXT_FLAG_NAMES``. Mirrors ObjectCommon._bits_set."""
    return tuple(
        name
        for i, name in enumerate(TEXT_FLAG_NAMES)
        if (flags >> i) & 1 and name != "None"
    )


# --- Dataclass ---------------------------------------------------------


@dataclass(frozen=True)
class TextBody:
    """Decoded Text ObjectInfo property body (one Paragraph; FNAF 1
    never uses multi-paragraph Text objects).

    Carries the runtime-consumable Paragraph fields (``font_handle``,
    ``flags``, ``color``, ``value``) plus the box dimensions and opaque
    header for round-trip support.
    """

    size: int
    opaque_header_raw: bytes = field(repr=False)
    inner_size: int
    box_width: int
    box_height: int
    paragraph_count: int
    paragraph_offsets: tuple[int, ...]
    font_handle: int
    flags: int
    color: int
    value: str

    @property
    def char_count(self) -> int:
        return len(self.value)

    @property
    def flag_names(self) -> tuple[str, ...]:
        return _flag_names(self.flags)

    def summary_dict(self) -> dict:
        return {
            "size": self.size,
            "inner_size": self.inner_size,
            "box_width": self.box_width,
            "box_height": self.box_height,
            "paragraph_count": self.paragraph_count,
            "font_handle": self.font_handle,
            "flags": self.flags,
            "flag_names": list(self.flag_names),
            "color": self.color,
            "char_count": self.char_count,
            "value": self.value,
        }

    def as_dict(self) -> dict:
        return {
            "size": self.size,
            "inner_size": self.inner_size,
            "box_width": self.box_width,
            "box_height": self.box_height,
            "paragraph_count": self.paragraph_count,
            "paragraph_offsets": list(self.paragraph_offsets),
            "font_handle": self.font_handle,
            "flags": self.flags,
            "flag_names": list(self.flag_names),
            "color": self.color,
            "value": self.value,
            "char_count": self.char_count,
            "opaque_header_raw_len": len(self.opaque_header_raw),
        }


# --- Decoder -----------------------------------------------------------


def decode_text_body(payload: bytes) -> TextBody:
    """Decode one 0x4446 Properties body where ``object_type == 3``.

    Antibodies enforced:

    * ``len(payload) >= 146`` (fixed prefix + 2-byte NUL).
    * ``body_size`` field at offset 0 equals ``len(payload)``.
    * ``inner_size`` field at offset 116 equals ``body_size - 116``.
    * ``paragraph_count == 1`` (FNAF-1 tripwire).
    * ``paragraph_offsets[0] == 20`` (FNAF-1 tripwire).
    * UTF-16-LE region ``[144, body_size)`` is exactly ``string_chars *
      2 + 2`` bytes; trailing 2 bytes form a single ``0x0000`` NUL
      terminator; the codepoint sequence contains no embedded NUL.
    """
    n = len(payload)
    if n < TEXT_MIN_BODY_SIZE:
        raise TextBodyDecodeError(
            f"Text body: payload is {n} bytes, smaller than the "
            f"{TEXT_MIN_BODY_SIZE}-byte minimum (144-byte fixed prefix + "
            "2-byte UTF-16 NUL terminator). Antibody #2 byte-count."
        )

    (size,) = _SIZE_STRUCT.unpack_from(payload, TEXT_SIZE_FIELD_OFFSET)
    if size != n:
        raise TextBodyDecodeError(
            f"Text body: size field at offset 0 is {size} but payload "
            f"length is {n}. Antibody #2 byte-count."
        )

    opaque_header_raw = bytes(
        payload[TEXT_OPAQUE_HEADER_START:TEXT_OPAQUE_HEADER_END]
    )

    (inner_size,) = _INNER_SIZE_STRUCT.unpack_from(
        payload, TEXT_INNER_SIZE_OFFSET
    )
    expected_inner = size - TEXT_BODY_HEADER_SIZE
    if inner_size != expected_inner:
        raise TextBodyDecodeError(
            f"Text body: inner_size field at offset {TEXT_INNER_SIZE_OFFSET} "
            f"is {inner_size} but expected body_size - {TEXT_BODY_HEADER_SIZE} "
            f"= {expected_inner}. Antibody #2 byte-count."
        )

    (box_width,) = _BOX_WIDTH_STRUCT.unpack_from(
        payload, TEXT_BOX_WIDTH_OFFSET
    )
    (box_height,) = _BOX_HEIGHT_STRUCT.unpack_from(
        payload, TEXT_BOX_HEIGHT_OFFSET
    )

    (paragraph_count,) = _PARAGRAPH_COUNT_STRUCT.unpack_from(
        payload, TEXT_PARAGRAPH_COUNT_OFFSET
    )
    if paragraph_count != TEXT_FNAF1_PARAGRAPH_COUNT:
        raise TextBodyDecodeError(
            f"Text body: paragraph_count is {paragraph_count} but FNAF 1 "
            f"only emits {TEXT_FNAF1_PARAGRAPH_COUNT}. Multi-paragraph Text "
            "objects need a different decoder shape (loop over offsets[]). "
            "Antibody #1 strict-unknown."
        )

    (paragraph_offset_0,) = _PARAGRAPH_OFFSET_STRUCT.unpack_from(
        payload, TEXT_PARAGRAPH_OFFSETS_OFFSET
    )
    if paragraph_offset_0 != TEXT_FNAF1_PARAGRAPH_OFFSET_0:
        raise TextBodyDecodeError(
            f"Text body: paragraph_offsets[0] is {paragraph_offset_0} but "
            f"FNAF 1 always emits {TEXT_FNAF1_PARAGRAPH_OFFSET_0} (relative "
            "to start of inner Text struct). Antibody #1 strict-unknown."
        )

    (font_handle,) = _FONT_HANDLE_STRUCT.unpack_from(
        payload, TEXT_PARAGRAPH_FONT_HANDLE_OFFSET
    )
    (flags,) = _FLAGS_STRUCT.unpack_from(
        payload, TEXT_PARAGRAPH_FLAGS_OFFSET
    )
    (color,) = _COLOR_STRUCT.unpack_from(
        payload, TEXT_PARAGRAPH_COLOR_OFFSET
    )

    value_region = payload[TEXT_PARAGRAPH_VALUE_OFFSET:]
    if len(value_region) < 2:
        raise TextBodyDecodeError(
            f"Text body: value region [{TEXT_PARAGRAPH_VALUE_OFFSET}, {n}) "
            f"is {len(value_region)} bytes, too short for the 2-byte "
            "UTF-16-LE NUL terminator. Antibody #2 byte-count."
        )
    if len(value_region) % 2 != 0:
        raise TextBodyDecodeError(
            f"Text body: value region [{TEXT_PARAGRAPH_VALUE_OFFSET}, {n}) "
            f"is {len(value_region)} bytes, not a whole number of UTF-16-LE "
            "code units. Antibody #2 byte-count."
        )
    if value_region[-2:] != b"\x00\x00":
        raise TextBodyDecodeError(
            f"Text body: value region does not end in UTF-16-LE NUL "
            f"terminator (last two bytes {value_region[-2:]!r}). "
            "Antibody #1 strict-unknown."
        )

    string_bytes = value_region[:-2]
    # Reject embedded NULs — the wire format is NUL-terminated, so an
    # embedded NUL means we mis-sized the value region or the body is
    # malformed. This mirrors how Clickteam's own readers stop at the
    # first 0x0000 in ReadYuniversal / ReadUniversal.
    if any(
        string_bytes[i] == 0 and string_bytes[i + 1] == 0
        for i in range(0, len(string_bytes), 2)
    ):
        raise TextBodyDecodeError(
            "Text body: value region contains embedded UTF-16-LE NUL "
            "before the terminator. Antibody #1 strict-unknown."
        )

    try:
        value = string_bytes.decode("utf-16-le")
    except UnicodeDecodeError as exc:
        raise TextBodyDecodeError(
            f"Text body: value region failed UTF-16-LE decode: {exc}. "
            "Antibody #1 strict-unknown."
        ) from exc

    return TextBody(
        size=size,
        opaque_header_raw=opaque_header_raw,
        inner_size=inner_size,
        box_width=box_width,
        box_height=box_height,
        paragraph_count=paragraph_count,
        paragraph_offsets=(paragraph_offset_0,),
        font_handle=font_handle,
        flags=flags,
        color=color,
        value=value,
    )
