"""PNG emission sink for `DecodedPixels` (probe #7.1a visual-inspection).

Turns probe #7.1's `DecodedPixels.rgba` byte grid into an on-disk PNG
file so probe #100 can eyeball the output against real gameplay. Gated
behind the `FNAF_PARSER_EMIT_PNGS` environment variable: test runs
default to *not* spamming the filesystem, and CI never triggers the
path unless the env var is set explicitly.

Why a hand-rolled PNG encoder
-----------------------------

The parser dependency list is deliberately small (pytest + rich). Adding
Pillow / imageio / numpy just to spell `write_png()` is overkill when
the PNG spec has a total of four chunks for the uncompressed-RGBA case
(signature + IHDR + IDAT + IEND) and we already have zlib in the
standard library. Plus: a hand-rolled encoder gives us a single-file
answer to "where does the byte order get locked in?" when probe #100
uncovers a channel-order bug — one module, no third-party indirection.

Format recap (what this module actually emits)
----------------------------------------------

Every PNG starts with the 8-byte signature `89 50 4e 47 0d 0a 1a 0a`.
Then three chunks, each framed as
`[length u32 BE][type 4 ASCII bytes][data][crc32 u32 BE]`:

1. **IHDR** (13 bytes): width u32 BE, height u32 BE, bit_depth=8,
   color_type=6 (RGBA), compression=0, filter=0, interlace=0.
2. **IDAT**: zlib-compressed stream of per-row filter+pixel bytes.
   For every row we prepend filter byte 0 (None — no prediction) and
   the raw RGBA bytes. The filter byte is part of the PNG stream, not
   part of the image — it tells the decoder how to reverse any
   prediction we applied. We apply none, so filter = 0.
3. **IEND**: zero-length sentinel.

The CRC is computed over `type + data` per chunk.

Path layout
-----------

Default output directory is `parser/out/images/` (git-ignored). File
naming: `<handle>.png` (4-digit zero-padded for lexical sort). Collides
with no real FNAF 1 handle (handles are 0-608). Safe to wipe between
runs — nothing else writes to `parser/out/`.
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

from fnaf_parser.decoders.images_pixels import DecodedPixels

# --- PNG wire-format constants ------------------------------------------

# PNG 8-byte file signature (spec §5.2). Identifies the file type and
# detects common transmission corruption (CR/LF / LF/CR translation,
# MSB/LSB swap, 7-bit stripping).
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# IHDR fields (spec §11.2.2). Bit depth 8 + color type 6 = truecolor
# with alpha, the canonical RGBA-8bpc PNG that every decoder supports.
PNG_BIT_DEPTH_8 = 8
PNG_COLOR_TYPE_RGBA = 6
PNG_COMPRESSION_DEFLATE = 0
PNG_FILTER_ADAPTIVE = 0
PNG_INTERLACE_NONE = 0

# Per-row filter byte — 0 means "no filter, raw pixel bytes follow".
# Predictor filters (1..4) would give better compression but the spec
# requires the decoder handle all five; emitting 0 everywhere keeps the
# encoder trivial and still decodes cleanly under every viewer.
PNG_ROW_FILTER_NONE = 0

# Env var that flips the sink on. Any truthy value — we accept the
# standard "1" / "true" / "yes" family. Absence = off.
EMIT_ENV_VAR = "FNAF_PARSER_EMIT_PNGS"

# Default output directory, relative to the parser package root. The
# caller can override per-emit via the `out_dir` argument.
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "out" / "images"


def _should_emit() -> bool:
    """Return True when `FNAF_PARSER_EMIT_PNGS` is set to a truthy value.

    Accepted truthy strings are the POSIX-ish classics — `1`, `true`,
    `yes`, `on` (case-insensitive). Anything else (including absence)
    means don't emit.
    """
    raw = os.environ.get(EMIT_ENV_VAR, "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Frame a single PNG chunk: `length | type | data | crc32`.

    `chunk_type` must be exactly 4 ASCII bytes (e.g. b"IHDR"). Length
    is u32 BE over the data only — the `type` and `crc32` fields are
    framing, not payload. `crc32` is computed over `type + data`.
    """
    assert len(chunk_type) == 4, "PNG chunk type must be 4 bytes"
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


def encode_png_rgba(rgba: bytes, width: int, height: int) -> bytes:
    """Encode `width * height * 4` RGBA bytes into a valid PNG byte string.

    Raises `ValueError` if `len(rgba) != width * height * 4` — a
    drifted input would produce a corrupt PNG, and emitting one
    silently would undermine the whole "visual correctness" point of
    the sink.
    """
    expected = width * height * 4
    if len(rgba) != expected:
        raise ValueError(
            f"encode_png_rgba: rgba is {len(rgba)} bytes but "
            f"width*height*4={expected}. Refusing to emit a corrupt PNG."
        )
    if width <= 0 or height <= 0:
        raise ValueError(
            f"encode_png_rgba: width={width} height={height}; both must be > 0"
        )

    # IHDR: 13 bytes of image metadata.
    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        PNG_BIT_DEPTH_8,
        PNG_COLOR_TYPE_RGBA,
        PNG_COMPRESSION_DEFLATE,
        PNG_FILTER_ADAPTIVE,
        PNG_INTERLACE_NONE,
    )

    # Pack raw scanlines: filter byte + row bytes, for every row. The
    # filter byte is part of the PNG stream spec, not part of the image.
    row_stride = width * 4
    parts = []
    for y in range(height):
        start = y * row_stride
        parts.append(bytes([PNG_ROW_FILTER_NONE]))
        parts.append(rgba[start:start + row_stride])
    raw = b"".join(parts)

    idat = zlib.compress(raw, level=6)

    return b"".join([
        PNG_SIGNATURE,
        _chunk(b"IHDR", ihdr),
        _chunk(b"IDAT", idat),
        _chunk(b"IEND", b""),
    ])


def emit_png(
    decoded: DecodedPixels,
    *,
    out_dir: Path | None = None,
    force: bool = False,
) -> Path | None:
    """Write `decoded` to `<out_dir>/<handle>.png` if emission is enabled.

    Returns the path written to, or `None` when emission is gated off
    by the env var. `force=True` bypasses the env-var gate — used by
    unit tests that need to exercise the write path without setting
    environment state.

    The output directory is created on first write. Filename is
    zero-padded to 4 digits (`0042.png`) so a `ls out/images/` listing
    sorts naturally.
    """
    if not force and not _should_emit():
        return None

    target_dir = out_dir if out_dir is not None else DEFAULT_OUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{decoded.handle:04d}.png"

    png_bytes = encode_png_rgba(decoded.rgba, decoded.width, decoded.height)
    out_path.write_bytes(png_bytes)
    return out_path
