"""0x6667 Fonts decoder (probe #10) — envelope-only.

Top-level chunk. The font-bank peer of [[Probe #7]] 0x6666 Images and
[[Probe #9]] 0x6668 Sounds, and the record target for [[Probe #10]]
0x5556 FontOffsets — that chunk is the index table, this one is the
record bank the offsets land inside.

Wire format (post decrypt + zlib on the outer chunk)
----------------------------------------------------

::

    u32 count
    N × FontItem

Each `FontItem` is a 4-byte outer header (the raw handle) followed by a
zlib-compressed body wrapped in the standard 8-byte Clickteam decomp
header:

::

     0  u32 raw_handle         (CTFAK2 stores `raw_handle + offset`; we keep both)
     4  u32 decompressed_size  (size of the inner blob)
     8  u32 compressed_size    (size of the zlib stream on the wire)
    12  compressed_size bytes  → zlib.decompress → inner_blob of length decompressed_size

Inside `inner_blob` (decompressed):

::

     0  i32 checksum
     4  i32 references
     8  i32 size               (unused by readers — declared as 0 on write; Anaconda/CTFAK2 ignore)
    12  LogFont (92 bytes)

LogFont layout (CTFAK2 `LogFont.Read` + Anaconda `data/font.py::LogFont.read`):

::

     0  i32 height
     4  i32 width
     8  i32 escapement
    12  i32 orientation
    16  i32 weight
    20  u8  italic
    21  u8  underline
    22  u8  strike_out
    23  u8  char_set
    24  u8  out_precision
    25  u8  clip_precision
    26  u8  quality
    27  u8  pitch_and_family
    28  64 bytes face_name (32 wchars × UTF-16 LE, NUL-terminated/padded)

Total LogFont = 20 + 8 + 64 = 92 bytes.
Total inner blob = 12 (checksum/references/size) + 92 (LogFont) = 104
bytes.

Oracles (cross-checked — Antibody #4 multi-oracle)
--------------------------------------------------

- CTFAK2.0 `Core/CTFAK.Core/CCN/Chunks/Banks/FontBank.cs`:
  `FontItem.Read` reads `u32 Handle`, then wraps the body via
  `Decompressor.DecompressAsReader` (u32 decompSize + u32 compSize +
  zlib). Bank-level handle adjust: `if (Settings.Build > 284 && !Debug)
  offset = -1`. Strict `> 284`, so FNAF 1 (build=284) gets offset=0 —
  handle stays raw.
- Anaconda `mmfparser/data/chunkloaders/fontbank.py::FontItem.read`:
  Python read path. Same field layout. Bank-level handle adjust:
  `if self.settings['build'] >= 284 and not debug: offset = -1`. Loose
  `>= 284`, so FNAF 1 gets offset=-1 — handle is `raw - 1`.

Oracle split on the handle convention (load-bearing for this probe)
-------------------------------------------------------------------

Unlike sounds/images (where both oracles agree for build=284), fonts
split: CTFAK2 says "keep raw" and Anaconda says "subtract 1". Exact
opposite resolutions on the same input.

This decoder stores BOTH interpretations:

- `raw_handle` — unmodified u32 from the wire.
- `handle` — `raw_handle - FONT_HANDLE_ADJUST` (Anaconda convention,
  mirroring the sound/image pattern).

The empirical winner is picked in the cross-chunk handshake test
(`test_fonts.py::test_fnaf1_fonts_cross_chunk_offset_handshake`) via
the same `_deltas(key_attr)` + XOR-singular `Counter` pattern used to
settle the sound handle convention. Whichever interpretation yields a
single-valued delta against `FontOffsets.offsets[h] -
font.record_start_offset` wins — that's the convention object
references use in practice.

Scope cut — envelope only
-------------------------

This probe decodes the *record bank*. It does NOT:

- Rasterize glyphs — `LogFont` is a Windows GDI `LOGFONTW`
  specification (height, weight, face name, …), not bitmap data. A
  future probe that renders frames would pair it with a TTF file
  bundled via 0x6664 (`TrueTypeFonts`) or the system font cache.
- Interpret the `_italic` / `_underline` / `_strike_out` byte values
  beyond the raw u8 reads. Windows treats non-zero as true; callers
  wanting booleans should branch on `!= 0` themselves.
- Validate `face_name` charset beyond UTF-16 LE decode + NUL strip.

Cross-chunk antibody (load-bearing for this probe)
--------------------------------------------------

Every font record's start offset — its position within the
*decompressed outer 0x6667 body* — must appear in the set of non-zero
offsets produced by Probe #10's 0x5556 FontOffsets decode. Expected on
FNAF 1 (speculative until measured): the Fonts bank is tiny (427 raw
bytes), so expect ≤ a handful of records. This is enforced in the
integration test.

Antibody coverage (this decoder)
--------------------------------

- #1 strict-unknown : negative count / negative decompressed_size /
  negative compressed_size raise `FontBankDecodeError`; a zlib stream
  that overruns the outer payload raises; an inner blob whose actual
  decompressed length disagrees with declared `decompressed_size`
  raises; a too-small inner blob (< `_FONT_INNER_SIZE`) raises.
- #2 byte-count    : `count` prefix + N × record wire size must
  reconcile to exactly `len(payload)`. Inner blob must be exactly 104
  bytes (checksum/references/size triad + 92-byte LogFont).
- #3 round-trip    : synthetic pack/unpack in tests.
- #4 multi-oracle  : field order + sizes mirror CTFAK2 + Anaconda. The
  oracle split on handle adjust is pinned as a test expectation.
- #5 multi-input   : runs against the FNAF 1 0x6667 payload.
- #7 snapshot      : count, raw handle set, face-name set, and a
  SHA-256 of `(raw_handle, weight, italic, face_name)` across all items
  pinned in tests.
- Cross-chunk     : `FontBank.record_start_offsets` drives the
  integration-layer handshake against `FontOffsets.offsets`.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

# --- Wire-format constants ----------------------------------------------

FONT_BANK_COUNT_SIZE = 4

# Per-item outer header (before the zlib-compressed body): u32 raw_handle.
FONT_ITEM_HANDLE_SIZE = 4

# 8-byte Clickteam decomp header that wraps the compressed body:
# u32 decompressed_size + u32 compressed_size. Mirrors CTFAK2's
# `Decompressor.Decompress(reader)` sequence and Anaconda's
# `zlibdata.decompress(reader)`.
_FONT_DECOMP_HEADER = struct.Struct("<II")
FONT_DECOMP_HEADER_SIZE = _FONT_DECOMP_HEADER.size  # 8
assert FONT_DECOMP_HEADER_SIZE == 8

# Inner triad (i32 checksum + i32 references + i32 size), sits at the
# head of the decompressed inner blob.
_FONT_INNER_TRIAD = struct.Struct("<iii")
FONT_INNER_TRIAD_SIZE = _FONT_INNER_TRIAD.size  # 12
assert FONT_INNER_TRIAD_SIZE == 12

# LogFont struct: 5×i32 + 8×u8 + 32-wchar face_name (UTF-16 LE).
_LOG_FONT_FIXED = struct.Struct("<iiiiiBBBBBBBB")
LOG_FONT_FIXED_SIZE = _LOG_FONT_FIXED.size  # 5*4 + 8 = 28
assert LOG_FONT_FIXED_SIZE == 28

LOG_FONT_FACE_NAME_WCHARS = 32
LOG_FONT_FACE_NAME_SIZE = LOG_FONT_FACE_NAME_WCHARS * 2  # 64 bytes
LOG_FONT_SIZE = LOG_FONT_FIXED_SIZE + LOG_FONT_FACE_NAME_SIZE  # 92
assert LOG_FONT_SIZE == 92

# Total inner-blob size (what the zlib stream must decompress to).
FONT_INNER_SIZE = FONT_INNER_TRIAD_SIZE + LOG_FONT_SIZE  # 104
assert FONT_INNER_SIZE == 104

# Handle adjust. Oracle split: CTFAK2 does NOT subtract on build=284
# (strict `> 284`), Anaconda DOES (`>= 284`). We follow the Anaconda
# convention as the default for `handle`, mirroring sounds/images, and
# expose `raw_handle` so the empirical cross-chunk handshake picks the
# winner. Grep-friendly named constant so the intent is visible and a
# future legacy-build code path can flip it off cleanly.
FONT_HANDLE_ADJUST = 1


class FontBankDecodeError(ValueError):
    """0x6667 Fonts decode failure — carries offset + size context."""


# --- Dataclasses --------------------------------------------------------


@dataclass(frozen=True)
class LogFont:
    """One decoded Windows GDI LOGFONTW record.

    Field names mirror the Win32 `LOGFONTW` struct so callers cross-
    referencing MSDN find the same identifiers. Bytes (`italic` etc.)
    are kept as raw u8 — Windows treats any non-zero value as true;
    callers wanting booleans should branch themselves.
    """

    height: int
    width: int
    escapement: int
    orientation: int
    weight: int
    italic: int
    underline: int
    strike_out: int
    char_set: int
    out_precision: int
    clip_precision: int
    quality: int
    pitch_and_family: int
    face_name: str

    @property
    def is_bold(self) -> bool:
        """LOGFONTW weight == 700 is the `FW_BOLD` constant — matches
        Anaconda's `LogFont.isBold()`. Exposed as a property for
        ergonomic access at the test layer."""
        return self.weight == 700

    def as_dict(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "escapement": self.escapement,
            "orientation": self.orientation,
            "weight": self.weight,
            "italic": self.italic,
            "underline": self.underline,
            "strike_out": self.strike_out,
            "char_set": self.char_set,
            "out_precision": self.out_precision,
            "clip_precision": self.clip_precision,
            "quality": self.quality,
            "pitch_and_family": self.pitch_and_family,
            "face_name": self.face_name,
            "is_bold": self.is_bold,
        }


@dataclass(frozen=True)
class Font:
    """One decoded font record inside the 0x6667 FontBank.

    The raw on-wire handle lives in `raw_handle`; `handle` is the
    Anaconda-adjusted logical index (`raw_handle - 1`). CTFAK2 and
    Anaconda disagree on whether the `-1` applies for build=284 — the
    cross-chunk handshake test resolves this empirically. Consumers
    should prefer `by_handle` or `by_raw_handle` on the parent
    `FontBank` rather than reaching into individual records.

    `record_start_offset` is this record's position within the
    *decompressed outer 0x6667 payload* — the load-bearing field for
    the cross-chunk handshake with 0x5556 FontOffsets.
    """

    raw_handle: int
    handle: int
    record_start_offset: int
    record_wire_size: int
    checksum: int
    references: int
    declared_size: int
    decompressed_size: int
    compressed_size: int
    log_font: LogFont
    inner_blob: bytes = field(repr=False)

    @property
    def face_name(self) -> str:
        """Shortcut to `self.log_font.face_name` — the single field
        tests grep for most often."""
        return self.log_font.face_name

    def as_dict(self) -> dict:
        return {
            "raw_handle": self.raw_handle,
            "handle": self.handle,
            "record_start_offset": self.record_start_offset,
            "record_wire_size": self.record_wire_size,
            "checksum": self.checksum,
            "references": self.references,
            "declared_size": self.declared_size,
            "decompressed_size": self.decompressed_size,
            "compressed_size": self.compressed_size,
            "log_font": self.log_font.as_dict(),
        }


@dataclass(frozen=True)
class FontBank:
    """Decoded 0x6667 Fonts payload.

    `fonts` preserves the on-wire order. `record_start_offsets` is the
    tuple of per-record start positions inside the decompressed outer
    payload — the single cross-chunk antibody surface against Probe #10's
    0x5556 FontOffsets. `by_handle` / `by_raw_handle` give either
    adjusted or raw handle resolution depending on whose convention the
    caller needs.
    """

    count: int
    fonts: tuple[Font, ...]
    record_start_offsets: tuple[int, ...]

    @property
    def by_handle(self) -> dict[int, Font]:
        """Logical (`raw - 1`, Anaconda convention) `handle → Font`
        map. Built fresh each call."""
        return {f.handle: f for f in self.fonts}

    @property
    def by_raw_handle(self) -> dict[int, Font]:
        """Raw (on-wire, CTFAK2 convention for build=284)
        `handle → Font` map — exposed so the cross-chunk handshake test
        can pick the winning convention."""
        return {f.raw_handle: f for f in self.fonts}

    @property
    def handles(self) -> frozenset[int]:
        """Set of logical (Anaconda-convention) handles this bank
        exposes."""
        return frozenset(f.handle for f in self.fonts)

    @property
    def face_names(self) -> tuple[str, ...]:
        """Tuple of face names in on-wire order. Handy for snapshot
        tests that want to pin the font list without digging into the
        full LogFont."""
        return tuple(f.face_name for f in self.fonts)

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "fonts": [f.as_dict() for f in self.fonts],
            "record_start_offsets": list(self.record_start_offsets),
        }


# --- Decoder ------------------------------------------------------------


def _decode_face_name(raw_bytes: bytes, raw_handle: int) -> str:
    """Decode the 64-byte face_name field as UTF-16 LE and trim
    trailing NUL / whitespace. Mirrors Anaconda's `readString(32)`
    (which drops at the first NUL) and CTFAK2's `ReadYuniversal(32)`
    (fixed-width unicode read with `.TrimEnd('\\0')` semantics)."""
    assert len(raw_bytes) == LOG_FONT_FACE_NAME_SIZE, (
        f"face_name slice must be {LOG_FONT_FACE_NAME_SIZE} bytes, "
        f"got {len(raw_bytes)}"
    )
    try:
        decoded = raw_bytes.decode("utf-16-le")
    except UnicodeDecodeError as exc:
        raise FontBankDecodeError(
            f"0x6667 Fonts: raw_handle={raw_handle} face_name failed "
            f"UTF-16 LE decode: {exc}. Antibody #1 strict-unknown: likely "
            f"zlib drift inside the item body."
        ) from exc
    # Trim at first NUL (fixed-width Clickteam unicode padding), then
    # whitespace — matches both oracles and keeps snapshot hashes
    # stable even if an upstream writer emits trailing spaces.
    nul_idx = decoded.find("\x00")
    if nul_idx >= 0:
        decoded = decoded[:nul_idx]
    return decoded.strip()


def _decode_log_font(blob: bytes, raw_handle: int) -> LogFont:
    """Decode a 92-byte LogFont slice. `raw_handle` is error-context."""
    if len(blob) != LOG_FONT_SIZE:
        raise FontBankDecodeError(
            f"0x6667 Fonts: raw_handle={raw_handle} LogFont slice is "
            f"{len(blob)} bytes, expected {LOG_FONT_SIZE}. Antibody #2 "
            f"byte-count."
        )
    (
        height,
        width,
        escapement,
        orientation,
        weight,
        italic,
        underline,
        strike_out,
        char_set,
        out_precision,
        clip_precision,
        quality,
        pitch_and_family,
    ) = _LOG_FONT_FIXED.unpack_from(blob, 0)
    face_name = _decode_face_name(blob[LOG_FONT_FIXED_SIZE:], raw_handle)
    return LogFont(
        height=height,
        width=width,
        escapement=escapement,
        orientation=orientation,
        weight=weight,
        italic=italic,
        underline=underline,
        strike_out=strike_out,
        char_set=char_set,
        out_precision=out_precision,
        clip_precision=clip_precision,
        quality=quality,
        pitch_and_family=pitch_and_family,
        face_name=face_name,
    )


def decode_font_bank(payload: bytes, *, is_compressed: bool = True) -> FontBank:
    """Decode a 0x6667 Fonts chunk payload.

    `payload` must be the plaintext bytes returned by
    `compression.read_chunk_payload` — i.e. any outer flag=1/2/3
    decoding has already happened. This decoder handles the *nested*
    per-item zlib layer itself (inline, not via
    `decompress_payload_bytes`) because the inner layer wraps only the
    item body (inner triad + LogFont), not the 4-byte raw_handle.

    `is_compressed` mirrors the bank-level `Compressed` flag in CTFAK2
    — defaults True because that's how FNAF 1 and every other modern
    Fusion pack ships. A debug pack that flips it would pass
    `is_compressed=False`, in which case the 104-byte inner blob is
    read straight from the wire with no zlib layer.

    Envelope only — see module docstring for the scope cut.
    """
    n = len(payload)
    if n < FONT_BANK_COUNT_SIZE:
        raise FontBankDecodeError(
            f"0x6667 Fonts: payload must hold at least the "
            f"{FONT_BANK_COUNT_SIZE}-byte u32 count prefix but got {n}. "
            f"Antibody #2 byte-count."
        )

    count = int.from_bytes(payload[:FONT_BANK_COUNT_SIZE], "little", signed=True)
    if count < 0:
        raise FontBankDecodeError(
            f"0x6667 Fonts: count prefix decoded as {count} (signed "
            f"int32). Negative counts are nonsense; Antibody #1 "
            f"strict-unknown. Likely outer-layer zlib or RC4 drift."
        )

    pos = FONT_BANK_COUNT_SIZE
    fonts: list[Font] = []
    record_starts: list[int] = []

    for i in range(count):
        record_start = pos
        record_starts.append(record_start)

        if pos + FONT_ITEM_HANDLE_SIZE > n:
            raise FontBankDecodeError(
                f"0x6667 Fonts: record #{i} starts at offset 0x{pos:x} "
                f"but only {n - pos} bytes remain (need "
                f"≥{FONT_ITEM_HANDLE_SIZE} for the raw_handle). Antibody "
                f"#2 byte-count."
            )

        raw_handle = int.from_bytes(
            payload[pos : pos + FONT_ITEM_HANDLE_SIZE], "little"
        )
        body_start = pos + FONT_ITEM_HANDLE_SIZE

        if is_compressed:
            if body_start + FONT_DECOMP_HEADER_SIZE > n:
                raise FontBankDecodeError(
                    f"0x6667 Fonts: record #{i} (raw_handle={raw_handle}) "
                    f"at offset 0x{pos:x}: not enough bytes remaining for "
                    f"the {FONT_DECOMP_HEADER_SIZE}-byte decomp header. "
                    f"Antibody #2 byte-count."
                )
            decompressed_size, compressed_size = _FONT_DECOMP_HEADER.unpack_from(
                payload, body_start
            )
            # Signed reinterpret of u32 values to surface negative drift.
            if decompressed_size > 0x7FFFFFFF:
                raise FontBankDecodeError(
                    f"0x6667 Fonts: record #{i} (raw_handle={raw_handle}) "
                    f"at offset 0x{pos:x}: decompressed_size={decompressed_size} "
                    f"has the i32 sign bit set. Antibody #1 strict-unknown."
                )
            if compressed_size > 0x7FFFFFFF:
                raise FontBankDecodeError(
                    f"0x6667 Fonts: record #{i} (raw_handle={raw_handle}) "
                    f"at offset 0x{pos:x}: compressed_size={compressed_size} "
                    f"has the i32 sign bit set. Antibody #1 strict-unknown."
                )
            zlib_start = body_start + FONT_DECOMP_HEADER_SIZE
            zlib_end = zlib_start + compressed_size
            if zlib_end > n:
                raise FontBankDecodeError(
                    f"0x6667 Fonts: record #{i} (raw_handle={raw_handle}) "
                    f"at offset 0x{pos:x}: compressed_size={compressed_size} "
                    f"overruns payload ({n - zlib_start} bytes remaining). "
                    f"Antibody #2 byte-count."
                )
            compressed_body = payload[zlib_start:zlib_end]
            try:
                inner_blob = zlib.decompress(compressed_body)
            except zlib.error as exc:
                raise FontBankDecodeError(
                    f"0x6667 Fonts: record #{i} (raw_handle={raw_handle}) "
                    f"at offset 0x{pos:x}: inner zlib decompress failed: "
                    f"{exc}. Antibody #1 strict-unknown: likely RC4 drift "
                    f"on the outer chunk."
                ) from exc
            if len(inner_blob) != decompressed_size:
                raise FontBankDecodeError(
                    f"0x6667 Fonts: record #{i} (raw_handle={raw_handle}): "
                    f"declared decompressed_size={decompressed_size} but "
                    f"zlib produced {len(inner_blob)} bytes. Antibody #2."
                )
            record_wire_size = (
                FONT_ITEM_HANDLE_SIZE + FONT_DECOMP_HEADER_SIZE + compressed_size
            )
            pos = zlib_end
        else:
            # Uncompressed branch: FONT_INNER_SIZE raw bytes ARE the
            # inner blob. Consumed straight from the payload.
            compressed_size = 0
            decompressed_size = FONT_INNER_SIZE
            body_end = body_start + FONT_INNER_SIZE
            if body_end > n:
                raise FontBankDecodeError(
                    f"0x6667 Fonts: record #{i} (raw_handle={raw_handle}) "
                    f"at offset 0x{pos:x}: uncompressed body needs "
                    f"{FONT_INNER_SIZE} bytes but only {n - body_start} "
                    f"remain. Antibody #2 byte-count."
                )
            inner_blob = bytes(payload[body_start:body_end])
            record_wire_size = FONT_ITEM_HANDLE_SIZE + FONT_INNER_SIZE
            pos = body_end

        if len(inner_blob) != FONT_INNER_SIZE:
            raise FontBankDecodeError(
                f"0x6667 Fonts: record #{i} (raw_handle={raw_handle}): "
                f"inner blob is {len(inner_blob)} bytes, expected "
                f"{FONT_INNER_SIZE} (12-byte triad + 92-byte LogFont). "
                f"Antibody #2 byte-count."
            )

        checksum, references, declared_size = _FONT_INNER_TRIAD.unpack_from(
            inner_blob, 0
        )
        log_font = _decode_log_font(
            inner_blob[FONT_INNER_TRIAD_SIZE:], raw_handle
        )

        fonts.append(
            Font(
                raw_handle=raw_handle,
                handle=raw_handle - FONT_HANDLE_ADJUST,
                record_start_offset=record_start,
                record_wire_size=record_wire_size,
                checksum=checksum,
                references=references,
                declared_size=declared_size,
                decompressed_size=len(inner_blob),
                compressed_size=compressed_size,
                log_font=log_font,
                inner_blob=bytes(inner_blob),
            )
        )

    if pos != n:
        raise FontBankDecodeError(
            f"0x6667 Fonts: decoded {count} records ending at offset "
            f"0x{pos:x} but payload is {n} bytes. Trailing "
            f"{n - pos} bytes unaccounted-for. Antibody #2 byte-count."
        )

    return FontBank(
        count=count,
        fonts=tuple(fonts),
        record_start_offsets=tuple(record_starts),
    )
