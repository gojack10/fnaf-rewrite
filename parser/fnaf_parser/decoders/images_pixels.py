"""0x6666 Images pixel-payload decoder (probe #7.1a — flag=0 path).

Envelope → payload pivot. Probe #7 decoded every 0x6666 record's
envelope (outer zlib + 32-byte inner header + opaque `image_data`) and
pinned the FNAF 1 distribution: **605 records, all `graphic_mode == 4`,
split into `flags == 0` (520 records) and `flags == 0x10` (85 records)**.
This module is the first step of turning `image_data` into actual RGBA
pixels. Scope is deliberately narrow:

- **In-scope (probe #7.1a):** the 520 `graphic_mode=4, flags=0` records —
  plain 24-bit-per-pixel masked-BGR scanlines with transparent-colour
  keying. The majority of FNAF 1 images, including every office frame
  and every static background.
- **Deferred to probe #7.1b:** the 85 `graphic_mode=4, flags=0x10`
  (Alpha-flagged) records. Same scan loop plus a trailing 8bpp alpha
  plane; shipped separately so flag=0 gets validated against visual
  reality first.
- **Explicitly not in scope:** LZX inner streams, RLE/RLEW/RLET runs,
  Android modes 0-5, 15/16-bit modes 2/3, ACE/Mac flags, the `RGBA`
  CTFAK2 bit. None of these appear in FNAF 1. Any unexpected
  mode/flag combination raises `ImagePixelsDecodeError` — loud deferral,
  no silent fallback.

Algorithm (flag=0 happy path)
-----------------------------

Cross-checked against two independent readers:

- CTFAK2 `Core/CTFAK.Core/Utils/ImageTranslator.cs::Normal24BitMaskedToRGBA`.
- Anaconda `mmfparser/data/chunkloaders/imagebank.pyx::read_rgb`.

Both agree on:

1. Per-row **byte order is B, G, R** (24 bits per pixel, no alpha byte
   in-stream).
2. Rows are **padded so each row's pixel-byte count is a multiple of
   2 bytes, rounded up to the next full pixel (3-byte pointSize
   chunk)**. Concretely for 24bpp: odd `width` → 3 pad bytes, even
   `width` → 0 pad bytes. Derived from CTFAK2's
   `ImageHelper.GetPadding(width, pointSize=3, bytes=2)` with ceil.
3. Alpha is **generated at output time**, not stored in-stream: each
   output RGBA pixel is opaque (A=255) unless its raw BGR bytes
   match the record's `transparent` colour, in which case A=0.

Transparent-colour channel mapping
----------------------------------

The 0x6666 inner header carries `transparent` as four on-wire bytes we
parse as `(R, G, B, A)`. CTFAK2's comparison is written against a .NET
`Color` whose byte-order convention has historically been ambiguous in
their own source; the file sits on top of `.R`, `.G`, `.B` accessors
rather than raw-byte positions. This module compares the raw in-stream
BGR bytes against the colour treating the stored tuple as `(R, G, B)`
— i.e. `in_stream_B == transparent.B, in_stream_G == transparent.G,
in_stream_R == transparent.R`.

For FNAF 1 specifically this distinction is invariant: every observed
transparent colour is `(0, 0, 0, _)` or `(0, 0, 0, 255)`, which
matches under any byte-order reading. If probe #100 (visual inspection)
reveals a systematic magenta/cyan mask on any FNAF 1 image, the
channel-mapping choice here is the first thing to swap. Documented as
open question §10.1 in the Probe #7.1 crystallization.

Error surface — every path is loud
----------------------------------

- `ImagePixelsDecodeError` on: negative or zero width/height, truncated
  `image_data` (shorter than `height * (width*3 + pad_bytes)`), any
  mode other than 4 in the dispatcher, any flag value other than 0 in
  the probe #7.1a dispatcher.
- **Not** an error: trailing bytes. CTFAK's writers append a few
  leftover bytes after the last row in some records; both reference
  readers silently discard them. We do the same but only after the
  required number of bytes has been consumed.

Antibody coverage (see `tests/test_images_pixels.py`)
-----------------------------------------------------

- #1 strict-unknown : mode/flag mismatch raises, truncated stream
  raises.
- #2 byte-count    : output length is exactly `width * height * 4`.
- #3 round-trip    : hand-crafted 2×2 and 3×2 pixel streams decode to
  their expected RGBA sequences, including the odd-width row pad.
- #5 multi-input   : decodes all 520 FNAF 1 flag=0 records without
  raising. Output byte count per record equals `w*h*4`.
- #7 snapshot      : pins decoded-record count, total decoded bytes,
  and a SHA-256 over `(handle, sha256(rgba))` tuples so any per-record
  drift fires with the offending handle called out.
"""

from __future__ import annotations

from dataclasses import dataclass

from fnaf_parser.decoders.images import Image


# --- Wire-format / algorithm constants ----------------------------------

# 24 bits per pixel = 3 bytes per pixel on the wire for mode 4 flag=0.
IMAGE_PIXELS_MODE_4_BYTES_PER_PIXEL = 3

# CTFAK2's ImageHelper.GetPadding(width, pointSize=3, bytes=2) alignment
# target: rows are padded so the raw byte count is even, rounded up to
# the next whole pointSize chunk. For 24bpp this collapses to 0 / 3 bytes.
IMAGE_PIXELS_ROW_ALIGN_BYTES = 2
IMAGE_PIXELS_POINT_SIZE = 3

# Output is four bytes per pixel, RGBA. This is what downstream (PNG
# emission, renderer texture upload) will consume.
IMAGE_PIXELS_OUT_BYTES_PER_PIXEL = 4

# Only mode/flag combination supported by this probe. Everything else
# raises to force the deferral to probe #7.1b or later.
IMAGE_PIXELS_SUPPORTED_MODE = 4
IMAGE_PIXELS_FLAG0 = 0


class ImagePixelsDecodeError(ValueError):
    """Pixel-payload decode failure — carries handle/width/height context."""


@dataclass(frozen=True)
class DecodedPixels:
    """One image's decoded pixel grid.

    `rgba` is exactly `width * height * 4` bytes in row-major R, G, B, A
    order — the wire format every downstream consumer (PNG emit, GPU
    texture upload, renderer sampler) standardises on. `handle` carries
    the source record's logical handle for traceability in snapshot
    fingerprints and emission filenames.
    """

    handle: int
    width: int
    height: int
    rgba: bytes


def get_row_padding_bytes(width: int) -> int:
    """Return the row padding (in bytes) for a 24bpp masked-BGR row.

    Mirror of CTFAK2 `ImageHelper.GetPadding(width, pointSize=3, bytes=2)`
    scaled into bytes. For 24bpp this reduces to:

    - `width` even → 0 pad bytes (row byte count is already a multiple
      of both 2 and 3 when width is even).
    - `width` odd  → 3 pad bytes (round up to the next full pixel).

    Expressed generally instead of as the 0/3 collapse so a future probe
    that opens 16bpp / 15bpp paths can re-use the helper.
    """
    if width <= 0:
        raise ImagePixelsDecodeError(
            f"row-padding is undefined for width={width}; must be > 0"
        )
    # CTFAK2: pad = bytes - (width * pointSize) % bytes; if pad == bytes
    # then no padding; else ceil(pad / pointSize) point-size chunks.
    pad = IMAGE_PIXELS_ROW_ALIGN_BYTES - (
        width * IMAGE_PIXELS_POINT_SIZE
    ) % IMAGE_PIXELS_ROW_ALIGN_BYTES
    if pad == IMAGE_PIXELS_ROW_ALIGN_BYTES:
        return 0
    # ceil(pad / pointSize) chunks, each pointSize bytes wide.
    chunks = (pad + IMAGE_PIXELS_POINT_SIZE - 1) // IMAGE_PIXELS_POINT_SIZE
    return chunks * IMAGE_PIXELS_POINT_SIZE


def decode_flag0_bgr_masked(
    image_data: bytes,
    width: int,
    height: int,
    transparent: tuple[int, int, int, int],
) -> bytes:
    """Decode a mode=4 flag=0 masked-BGR pixel stream to RGBA bytes.

    Returns exactly `width * height * 4` bytes in row-major R, G, B, A
    order. Opaque (A=255) by default; pixels whose raw (B, G, R) bytes
    equal (`transparent[2]`, `transparent[1]`, `transparent[0]`) become
    fully transparent (A=0).

    Raises `ImagePixelsDecodeError` if:

    - `width <= 0` or `height <= 0` (no degenerate dimensions — the
      decoder promises `len(output) == width * height * 4`; zero-area
      images would silently satisfy that but they're not a real case
      we need to support and would silently paper over drift).
    - `image_data` is shorter than
      `height * (width * 3 + row_pad_bytes)`.

    Trailing bytes past the last pixel row are discarded — CTFAK2 and
    Anaconda both do this, and some FNAF 1 records have a handful of
    leftover bytes we don't want to flag.
    """
    if width <= 0 or height <= 0:
        raise ImagePixelsDecodeError(
            f"width={width} height={height}: both must be > 0. "
            f"Antibody #1 strict-unknown (likely inner-header drift)."
        )

    row_pad_bytes = get_row_padding_bytes(width)
    row_payload_bytes = width * IMAGE_PIXELS_MODE_4_BYTES_PER_PIXEL
    row_total_bytes = row_payload_bytes + row_pad_bytes
    required_bytes = height * row_total_bytes

    if len(image_data) < required_bytes:
        raise ImagePixelsDecodeError(
            f"image_data is {len(image_data)} bytes but need "
            f"≥{required_bytes} for width={width} height={height} "
            f"(row_payload={row_payload_bytes}, row_pad={row_pad_bytes}). "
            f"Antibody #2 byte-count — truncated stream."
        )

    # Transparent-key channel mapping: (B, G, R) in-stream against
    # (transparent[2], transparent[1], transparent[0]) for semantic
    # alignment. See module docstring for rationale.
    tr_r, tr_g, tr_b, _tr_a = transparent

    out = bytearray(width * height * IMAGE_PIXELS_OUT_BYTES_PER_PIXEL)
    src = image_data
    src_pos = 0
    dst_pos = 0

    for _y in range(height):
        # Inline the per-pixel loop: Python method-call overhead dwarfs
        # the cost of an indexed read, so an explicit counter + direct
        # byte indexing outperforms slicing/iterating on a decompressed
        # record. 605 images × up to 1280×960 pixels: hot path.
        for _x in range(width):
            b = src[src_pos]
            g = src[src_pos + 1]
            r = src[src_pos + 2]
            out[dst_pos] = r
            out[dst_pos + 1] = g
            out[dst_pos + 2] = b
            if r == tr_r and g == tr_g and b == tr_b:
                out[dst_pos + 3] = 0
            else:
                out[dst_pos + 3] = 255
            src_pos += IMAGE_PIXELS_MODE_4_BYTES_PER_PIXEL
            dst_pos += IMAGE_PIXELS_OUT_BYTES_PER_PIXEL
        src_pos += row_pad_bytes

    return bytes(out)


def decode_image_pixels(image: Image) -> DecodedPixels:
    """Dispatch an `Image` envelope record to the right pixel decoder.

    Probe #7.1a ships the `graphic_mode == 4 and flags == 0` branch
    only. Every other mode/flag combination raises
    `ImagePixelsDecodeError` — the 85 `flags == 0x10` records in FNAF 1
    are expected to fire until probe #7.1b lands the alpha-plane path,
    at which point this dispatcher grows a second branch.

    Returning a `DecodedPixels` dataclass (handle + dims + rgba) rather
    than just bytes gives downstream consumers (PNG sink, renderer,
    snapshot tests) one object to pass around and one place to query
    for provenance.
    """
    if image.graphic_mode != IMAGE_PIXELS_SUPPORTED_MODE:
        raise ImagePixelsDecodeError(
            f"image handle={image.handle} graphic_mode="
            f"{image.graphic_mode}: only mode "
            f"{IMAGE_PIXELS_SUPPORTED_MODE} is supported in probe #7.1a. "
            f"FNAF 1 is empirically mode-4-only; a non-4 record means "
            f"either drift in 0x6666 decode or a non-FNAF-1 pack."
        )
    if image.flags != IMAGE_PIXELS_FLAG0:
        raise ImagePixelsDecodeError(
            f"image handle={image.handle} flags=0x{image.flags:02X}: "
            f"probe #7.1a ships flag=0 (masked-BGR) only. The "
            f"85 FNAF 1 records with flags=0x10 (Alpha) are queued for "
            f"probe #7.1b. This raise is the deferred-work signal."
        )

    rgba = decode_flag0_bgr_masked(
        image.image_data,
        image.width,
        image.height,
        image.transparent,
    )
    return DecodedPixels(
        handle=image.handle,
        width=image.width,
        height=image.height,
        rgba=rgba,
    )
