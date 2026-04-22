"""0x6666 Images pixel-payload decoder (probes #7.1a + #7.1b).

Envelope → payload pivot. Probe #7 decoded every 0x6666 record's
envelope (outer zlib + 32-byte inner header + opaque `image_data`) and
pinned the FNAF 1 distribution: **605 records, all `graphic_mode == 4`,
split into `flags == 0` (520 records) and `flags == 0x10` (85 records)**.
This module turns `image_data` into actual RGBA pixels.

- **In-scope (probe #7.1a):** the 520 `graphic_mode=4, flags=0` records —
  plain 24-bit-per-pixel masked-BGR scanlines with transparent-colour
  keying. The majority of FNAF 1 images, including every office frame
  and every static background.
- **In-scope (probe #7.1b):** the 85 `graphic_mode=4, flags=0x10`
  (Alpha) records. Same BGR colour plane as flag=0 minus the
  transparent-colour keying (superseded by the alpha plane) followed
  by a second pass of 8bpp alpha bytes with 4-byte-aligned row padding.
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

Algorithm (flag=0x10 alpha-plane path)
--------------------------------------

Same two references (CTFAK2 lines 23-76 + Anaconda `read_alpha`)
agree on a strictly two-pass layout:

1. **Colour plane first** — the flag=0 BGR scan loop, verbatim, minus
   the transparent-colour keying step (guarded by `if (!alpha)` at
   CTFAK2:47). Every output pixel starts opaque; the alpha plane
   overrides on pass 2. Consumes `height * (width*3 + colour_pad)`
   bytes with `colour_pad` from `get_row_padding_bytes`.
2. **Alpha plane second** — `width` raw 8bpp alpha bytes per row,
   each row followed by `(4 - width%4) % 4` bytes of padding
   (CTFAK2:62, Anaconda:255 — `GetPadding(width, pointSize=1, bytes=4)`
   gives a strict 4-byte row alignment, distinct from the colour
   plane's 2-byte alignment). Each alpha byte overwrites the A channel
   of its corresponding pixel, addressed by destination index.

The two planes never interleave. Trailing bytes past
`colour_bytes + alpha_bytes` are tolerated, same as the flag=0 path.

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
  `image_data` (shorter than required bytes for the chosen flag path),
  any mode other than 4, any flag value outside {0, 0x10}.
- **Not** an error: trailing bytes. CTFAK's writers append a few
  leftover bytes after the last row in some records; both reference
  readers silently discard them. We do the same but only after the
  required number of bytes has been consumed.

Antibody coverage (see `tests/test_images_pixels.py`)
-----------------------------------------------------

- #1 strict-unknown : mode/flag mismatch raises, truncated stream
  raises.
- #2 byte-count    : output length is exactly `width * height * 4`.
  Flag=16 adds a strict `len(image_data) >= colour_bytes + alpha_bytes`
  check with row-pad arithmetic cross-checked against two references.
- #3 round-trip    : hand-crafted 2×2 and 3×2 streams decode to their
  expected RGBA sequences, including odd-width colour pad AND the
  distinct 4-byte alpha-plane pad.
- #5 multi-input   : all 520 FNAF 1 flag=0 records AND all 85 flag=16
  records decode without raising; every output is `w*h*4` bytes.
- #7 snapshot      : per-flag SHA-256 fingerprint over
  `(handle, sha256(rgba))` tuples pins the entire decoded bank; any
  per-record drift flips the hash.
- Alpha-distribution: flag=16 output alpha channel must be
  non-degenerate across the bank (not uniformly 0 or 255) — catches a
  silent "alpha plane never read" regression that a round-trip hash
  would also catch but less legibly.
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

# Only mode/flag combinations supported by this module. Anything else
# raises — FNAF 1 is empirically mode-4-only with flags ∈ {0, 0x10}.
IMAGE_PIXELS_SUPPORTED_MODE = 4
IMAGE_PIXELS_FLAG0 = 0
IMAGE_PIXELS_FLAG16 = 0x10

# Alpha-plane row alignment (flag=0x10 only): CTFAK2's
# ImageHelper.GetPadding(width, pointSize=1, bytes=4) — each alpha row
# is 4-byte aligned. Collapses to `(4 - width%4) % 4` since pointSize=1
# means no point-size ceiling is required.
IMAGE_PIXELS_ALPHA_ROW_ALIGN_BYTES = 4
IMAGE_PIXELS_ALPHA_POINT_SIZE = 1


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


def get_alpha_row_padding_bytes(width: int) -> int:
    """Return the alpha-plane row padding (in bytes) for flag=0x10 records.

    Mirror of CTFAK2 `ImageHelper.GetPadding(width, pointSize=1, bytes=4)`.
    Since alpha is 8bpp (pointSize=1), the general CTFAK2 formula
    collapses to:

        pad = (4 - width % 4) % 4

    Concretely: width 4 → 0, 3 → 1, 2 → 2, 1 → 3. Independently
    verified against Anaconda `imagebank.pyx::read_alpha` line 255:
    `aPad = getPadding(width, 1, 4)`.
    """
    if width <= 0:
        raise ImagePixelsDecodeError(
            f"alpha row-padding is undefined for width={width}; must be > 0"
        )
    # Same shape as get_row_padding_bytes but pointSize=1: no ceil needed,
    # the padding is always expressible as a whole number of 1-byte chunks.
    pad = IMAGE_PIXELS_ALPHA_ROW_ALIGN_BYTES - (
        width * IMAGE_PIXELS_ALPHA_POINT_SIZE
    ) % IMAGE_PIXELS_ALPHA_ROW_ALIGN_BYTES
    if pad == IMAGE_PIXELS_ALPHA_ROW_ALIGN_BYTES:
        return 0
    return pad


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


def decode_flag16_bgr_with_alpha_plane(
    image_data: bytes,
    width: int,
    height: int,
) -> bytes:
    """Decode a mode=4 flag=0x10 BGR + alpha-plane stream to RGBA bytes.

    Two-pass wire format (CTFAK2 `ImageTranslator.cs:23-76`, Anaconda
    `imagebank.pyx::read_rgb + read_alpha`):

    1. **Colour plane.** Identical to flag=0's BGR scanlines *minus* the
       transparent-colour keying — every pixel starts opaque (A=255).
       The `if (!alpha)` guard at CTFAK2:47 skips the keying when an
       alpha plane is present: the alpha plane fully supersedes any
       transparent-colour hint from the envelope. Row padding is the
       same 2-byte-aligned 0/3-byte scheme as flag=0. Consumes exactly
       `height * (width*3 + colour_pad)` bytes.
    2. **Alpha plane.** `width` raw 8bpp alpha bytes per row, followed
       by `(4 - width%4) % 4` bytes of 4-byte-aligned row padding
       (CTFAK2:62, Anaconda:255). Each alpha byte overwrites the A
       channel of its corresponding pixel. Consumes exactly
       `height * (width + alpha_pad)` bytes.

    The two planes are strictly sequential — no interleaving. Trailing
    bytes past `colour_bytes + alpha_bytes` are tolerated for the same
    reason flag=0 tolerates them (CTFAK writers occasionally append
    stragglers).

    Raises `ImagePixelsDecodeError` if `width <= 0` or `height <= 0`,
    or if `image_data` is shorter than `colour_bytes + alpha_bytes`.

    The `transparent` argument from the envelope is intentionally not
    a parameter here — accepting it would invite confusion about whether
    the alpha plane overrides it (it always does).
    """
    if width <= 0 or height <= 0:
        raise ImagePixelsDecodeError(
            f"width={width} height={height}: both must be > 0. "
            f"Antibody #1 strict-unknown (likely inner-header drift)."
        )

    colour_pad = get_row_padding_bytes(width)
    colour_row_total = (
        width * IMAGE_PIXELS_MODE_4_BYTES_PER_PIXEL + colour_pad
    )
    colour_bytes = height * colour_row_total

    alpha_pad = get_alpha_row_padding_bytes(width)
    alpha_row_total = width + alpha_pad
    alpha_bytes = height * alpha_row_total

    required_bytes = colour_bytes + alpha_bytes

    if len(image_data) < required_bytes:
        raise ImagePixelsDecodeError(
            f"image_data is {len(image_data)} bytes but need "
            f"≥{required_bytes} for width={width} height={height} "
            f"(colour={colour_bytes} with pad={colour_pad}, "
            f"alpha={alpha_bytes} with pad={alpha_pad}). "
            f"Antibody #2 byte-count — truncated stream."
        )

    out = bytearray(width * height * IMAGE_PIXELS_OUT_BYTES_PER_PIXEL)
    src = image_data

    # --- Pass 1: colour plane (BGR → RGBA, no transparent keying) -------
    src_pos = 0
    dst_pos = 0
    for _y in range(height):
        for _x in range(width):
            b = src[src_pos]
            g = src[src_pos + 1]
            r = src[src_pos + 2]
            out[dst_pos] = r
            out[dst_pos + 1] = g
            out[dst_pos + 2] = b
            out[dst_pos + 3] = 255  # overwritten in pass 2
            src_pos += IMAGE_PIXELS_MODE_4_BYTES_PER_PIXEL
            dst_pos += IMAGE_PIXELS_OUT_BYTES_PER_PIXEL
        src_pos += colour_pad

    # --- Pass 2: alpha plane (write A by destination index) -------------
    # src_pos is now exactly `colour_bytes` — we've consumed the whole
    # colour plane. Writing by destination index (y*width + x) rather
    # than running the dst cursor again keeps the 4-byte alpha-pad
    # handling self-evident and mirrors CTFAK2:65-70.
    for y in range(height):
        row_dst_base = y * width * IMAGE_PIXELS_OUT_BYTES_PER_PIXEL
        for x in range(width):
            out[
                row_dst_base + x * IMAGE_PIXELS_OUT_BYTES_PER_PIXEL + 3
            ] = src[src_pos]
            src_pos += 1
        src_pos += alpha_pad

    return bytes(out)


def decode_image_pixels(image: Image) -> DecodedPixels:
    """Dispatch an `Image` envelope record to the right pixel decoder.

    Supported branches:

    - `graphic_mode == 4, flags == 0`   → `decode_flag0_bgr_masked`
      (probe #7.1a — 520 FNAF 1 records).
    - `graphic_mode == 4, flags == 0x10` → `decode_flag16_bgr_with_alpha_plane`
      (probe #7.1b — 85 FNAF 1 records).

    Every other mode/flag combination raises `ImagePixelsDecodeError`.
    FNAF 1 is empirically mode-4-only with flags ∈ {0, 0x10}, so any
    out-of-set value means envelope drift or a non-FNAF-1 pack.

    Returning a `DecodedPixels` dataclass (handle + dims + rgba) rather
    than just bytes gives downstream consumers (PNG sink, renderer,
    snapshot tests) one object to pass around and one place to query
    for provenance.
    """
    if image.graphic_mode != IMAGE_PIXELS_SUPPORTED_MODE:
        raise ImagePixelsDecodeError(
            f"image handle={image.handle} graphic_mode="
            f"{image.graphic_mode}: only mode "
            f"{IMAGE_PIXELS_SUPPORTED_MODE} is supported. "
            f"FNAF 1 is empirically mode-4-only; a non-4 record means "
            f"either drift in 0x6666 decode or a non-FNAF-1 pack."
        )

    if image.flags == IMAGE_PIXELS_FLAG0:
        rgba = decode_flag0_bgr_masked(
            image.image_data,
            image.width,
            image.height,
            image.transparent,
        )
    elif image.flags == IMAGE_PIXELS_FLAG16:
        rgba = decode_flag16_bgr_with_alpha_plane(
            image.image_data,
            image.width,
            image.height,
        )
    else:
        raise ImagePixelsDecodeError(
            f"image handle={image.handle} flags=0x{image.flags:02X}: "
            f"probes #7.1a (flag=0) and #7.1b (flag=0x10) are the only "
            f"supported flag values. FNAF 1 has no records outside "
            f"these two; any other flag means drift or a non-FNAF-1 pack."
        )

    return DecodedPixels(
        handle=image.handle,
        width=image.width,
        height=image.height,
        rgba=rgba,
    )
