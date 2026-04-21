"""0x6666 Images decoder (probe #7) — envelope-only.

Top-level chunk. The bank that every `Active`/`Backdrop`/`QuickBackdrop`
object template ultimately references for its pixels. Peer to
[[Probe #6]] 0x5555 ImageOffsets — that chunk is the index table, this
one is the record bank the offsets land inside.

Wire format (post decrypt + zlib on the outer chunk)
----------------------------------------------------

    u32 count
    N × ImageRecord

Each ImageRecord has *two* layers:

1. Outer 12-byte header:

       handle           : i32  (raw; logical = raw - 1 for build >= 284)
       decompressed_size: u32  (size of the zlib-decompressed inner blob)
       compressed_size  : u32  (size of the zlib stream on the wire)

2. Followed by `compressed_size` bytes of a *nested* zlib stream that
   decompresses to exactly `decompressed_size` bytes. That decompressed
   blob starts with a 32-byte inner header:

       checksum     : i32
       references   : i32
       data_size    : i32  (declared size of the pixel payload)
       width        : i16
       height       : i16
       graphic_mode : u8
       flags        : u8   (IMAGE_FLAGS bit dict — see below)
       reserved     : i16  (two padding bytes per CTFAK ByteReader.ReadInt16())
       hotspot_x    : i16
       hotspot_y    : i16
       action_x     : i16
       action_y     : i16
       transparent  : 4 bytes (R, G, B, A)

    Then the remainder of the decompressed blob is the opaque
    `image_data`. For non-LZX flags this equals `data_size` bytes; for
    LZX-flagged images it's a nested `[u32 decompSize][lzx stream]` that
    this probe does NOT unpack (envelope only).

Schema cross-checked against two independent references:

- CTFAK2.0 `Core/CTFAK.Core/CCN/Chunks/Banks/ImageBank/NormalImage.cs`:
  confirms outer i32/i32/i32 + zlib, then the 32-byte inner layout
  above. `Handle--` for `Settings.Build >= 284` is the build-adjustment.
- Anaconda `mmfparser/data/chunkloaders/imagebank.pyx::ImageItem.load`
  confirms the same fields and the ReadColor() R,G,B,A byte order.

The IMAGE_FLAGS bit dictionary differs slightly between references:
Anaconda stops at `Mac` (bit 6); CTFAK2.0 adds `RGBA` at bit 7. We
carry all 8 named bits forward-compatible with 2.5+ packs; FNAF 1's
flags empirically land in the low bits only.

Scope cut — envelope only
-------------------------

This probe decodes the *record bank*. It does NOT:

- Dispatch on `graphic_mode` to decode pixel data (Mode 0..7 map to
  different RGBA↔RGB masks and are implemented against MSVC intrinsic
  calls in `CTFAK-Native.dll` — blocked until probe #7.1).
- Decompress the LZX inner stream for LZX-flagged images.
- Reinflate alpha channels (Alpha flag) or apply the transparent color
  mask to the masked-RGB modes.

`image_data` is the raw post-outer-zlib byte slice, exactly as it lives
in the decompressed 0x6666 body. Probe #7.1 will take it from here.

Cross-chunk antibody (load-bearing for this probe)
--------------------------------------------------

Every image record's start offset — its position within the
*decompressed outer 0x6666 body* — must appear in the set of non-zero
offsets produced by Probe #6's 0x5555 ImageOffsets decode. Expected
on FNAF 1: `set(record_start_offsets) == set(non_zero 0x5555 offsets)`,
len == 605 (the 606 distinct offsets minus the zero bucket). This is
the single strongest evidence that the two decoders agree on the wire
format; enforced in the integration test, not inside this decoder.

Antibody coverage (this decoder)
--------------------------------

- #1 strict-unknown : negative count or negative per-record sizes
  raise `ImageDecodeError`; a nested zlib compSize that overruns the
  outer payload raises; an inner blob whose actual decompressed length
  disagrees with the declared `decompressed_size` raises.
- #2 byte-count    : outer u32 count + N × (12 + compSize) must
  reconcile to exactly `len(payload)`. Inner blob length must equal
  declared decompressed_size. Inner 32-byte header requires
  `decompressed_size >= 32`.
- #3 round-trip    : synthetic pack/unpack in tests.
- #4 multi-oracle  : field order + sizes mirror CTFAK2 + Anaconda.
- #5 multi-input   : runs against the FNAF 1 0x6666 payload.
- #7 snapshot      : count, first-handle, (width, height) histogram
  samples, and a SHA-256 of the full tuple of `record_start_offsets`
  pinned in tests.
- Cross-chunk     : `ImageBank.record_start_offsets` drives the
  integration-layer handshake against `ImageOffsets.offsets`.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

# --- Wire-format constants ----------------------------------------------

IMAGE_BANK_COUNT_SIZE = 4

# Outer per-record header: handle i32 + decompressed_size u32 + compressed_size u32.
# CTFAK's `ReadInt32` is signed; we unpack as signed so negative values
# surface loudly (Antibody #1) instead of silently wrapping.
_IMAGE_OUTER = struct.Struct("<iii")
IMAGE_OUTER_HEADER_SIZE = _IMAGE_OUTER.size  # 12
assert IMAGE_OUTER_HEADER_SIZE == 12

# Post-zlib inner header (32 bytes): 3×i32 + 2×i16 + 2×u8 + 5×i16 + 4B transparent.
# Transparent is a fixed 4-byte slice (R, G, B, A) extracted separately.
_IMAGE_INNER = struct.Struct("<iiihhBBhhhhh4s")
IMAGE_INNER_HEADER_SIZE = _IMAGE_INNER.size  # 32
assert IMAGE_INNER_HEADER_SIZE == 32

# Build >= 284: the on-wire handle is 1 more than the logical index the
# rest of the pack (0x5555 ImageOffsets, object bank, etc.) uses. Both
# references agree on the `Handle--` adjustment; we store both so
# round-trip is lossless.
IMAGE_BUILD_284_HANDLE_ADJUST = 1

# --- Image-flag bits (CTFAK2 FusionImage BitDict, 8 entries) ------------

IMAGE_FLAG_RLE = 1 << 0
IMAGE_FLAG_RLEW = 1 << 1
IMAGE_FLAG_RLET = 1 << 2
IMAGE_FLAG_LZX = 1 << 3
IMAGE_FLAG_ALPHA = 1 << 4
IMAGE_FLAG_ACE = 1 << 5
IMAGE_FLAG_MAC = 1 << 6
IMAGE_FLAG_RGBA = 1 << 7

IMAGE_FLAG_NAMES: dict[int, str] = {
    IMAGE_FLAG_RLE: "RLE",
    IMAGE_FLAG_RLEW: "RLEW",
    IMAGE_FLAG_RLET: "RLET",
    IMAGE_FLAG_LZX: "LZX",
    IMAGE_FLAG_ALPHA: "Alpha",
    IMAGE_FLAG_ACE: "ACE",
    IMAGE_FLAG_MAC: "Mac",
    IMAGE_FLAG_RGBA: "RGBA",
}


def image_flag_names(flags: int) -> tuple[str, ...]:
    """Return the set names of every bit raised in `flags`.

    Envelope-scope helper: probe #7.1 will consume these when it
    dispatches on LZX / RLE / Alpha / RGBA behaviour, but today it's
    just diagnostic sugar so `as_dict()` produces greppable output.
    """
    return tuple(
        name
        for bit, name in IMAGE_FLAG_NAMES.items()
        if (flags & bit) != 0
    )


Color = tuple[int, int, int, int]  # (R, G, B, A)


class ImageDecodeError(ValueError):
    """0x6666 Images decode failure — carries offset + size context."""


# --- Dataclasses --------------------------------------------------------


@dataclass(frozen=True)
class Image:
    """One decoded image record inside the 0x6666 ImageBank.

    The raw on-wire handle lives in `raw_handle`; `handle` is the
    logical index the rest of the pack uses (subtract 1 for build >= 284).
    `record_start_offset` is this record's position within the
    *decompressed outer 0x6666 payload* — the load-bearing field for
    the cross-chunk handshake with 0x5555 ImageOffsets.

    `image_data` is the opaque byte slice that lives after the 32-byte
    inner header (post-zlib). For non-LZX images this is exactly
    `data_size` bytes; for LZX-flagged images it's a nested
    `[u32 decompSize][lzx stream]` that this probe does not unpack.
    Probe #7.1 owns the graphic-mode dispatch.
    """

    raw_handle: int
    handle: int
    record_start_offset: int
    compressed_size: int
    decompressed_size: int
    checksum: int
    references: int
    data_size: int
    width: int
    height: int
    graphic_mode: int
    flags: int
    reserved: int
    hotspot_x: int
    hotspot_y: int
    action_x: int
    action_y: int
    transparent: Color
    image_data: bytes = field(repr=False)

    @property
    def has_lzx(self) -> bool:
        return bool(self.flags & IMAGE_FLAG_LZX)

    @property
    def has_alpha(self) -> bool:
        return bool(self.flags & IMAGE_FLAG_ALPHA)

    @property
    def has_rgba(self) -> bool:
        return bool(self.flags & IMAGE_FLAG_RGBA)

    def as_dict(self) -> dict:
        r, g, b, a = self.transparent
        return {
            "raw_handle": self.raw_handle,
            "handle": self.handle,
            "record_start_offset": self.record_start_offset,
            "compressed_size": self.compressed_size,
            "decompressed_size": self.decompressed_size,
            "checksum": self.checksum,
            "references": self.references,
            "data_size": self.data_size,
            "width": self.width,
            "height": self.height,
            "graphic_mode": self.graphic_mode,
            "flags": self.flags,
            "flag_names": list(image_flag_names(self.flags)),
            "reserved": self.reserved,
            "hotspot_x": self.hotspot_x,
            "hotspot_y": self.hotspot_y,
            "action_x": self.action_x,
            "action_y": self.action_y,
            "transparent": {"r": r, "g": g, "b": b, "a": a},
            "image_data_len": len(self.image_data),
        }


@dataclass(frozen=True)
class ImageBank:
    """Decoded 0x6666 Images payload.

    `images` preserves the on-wire order. `record_start_offsets` is the
    tuple of per-record start positions inside the decompressed outer
    payload — the single cross-chunk antibody surface against Probe #6's
    0x5555 ImageOffsets. `by_handle` is the bijective
    `logical_handle → Image` map a caller uses to resolve a
    frame-item-instance's image reference back to its pixel record
    (follow-up cross-chunk check).
    """

    count: int
    images: tuple[Image, ...]
    record_start_offsets: tuple[int, ...]

    @property
    def by_handle(self) -> dict[int, Image]:
        """Logical `handle → Image` map. Built fresh each call."""
        return {img.handle: img for img in self.images}

    @property
    def by_raw_handle(self) -> dict[int, Image]:
        """Raw (pre-adjustment) `handle → Image` map — exposed for any
        round-trip tooling that needs the on-wire handle."""
        return {img.raw_handle: img for img in self.images}

    @property
    def handles(self) -> frozenset[int]:
        """Set of logical handles this bank exposes."""
        return frozenset(img.handle for img in self.images)

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "images": [img.as_dict() for img in self.images],
            "record_start_offsets": list(self.record_start_offsets),
        }


# --- Decoder ------------------------------------------------------------


def _decode_inner(blob: bytes, raw_handle: int) -> tuple:
    """Unpack the 32-byte inner header from `blob` and return the
    decoded tuple. The `raw_handle` argument is only used for error
    context — the inner header does NOT carry a handle itself."""
    if len(blob) < IMAGE_INNER_HEADER_SIZE:
        raise ImageDecodeError(
            f"0x6666 Images: inner blob for raw_handle={raw_handle} is "
            f"{len(blob)} bytes, need ≥{IMAGE_INNER_HEADER_SIZE} for the "
            f"fixed inner header. Antibody #2 byte-count."
        )
    return _IMAGE_INNER.unpack_from(blob, 0)


def decode_image_bank(payload: bytes) -> ImageBank:
    """Decode a 0x6666 Images chunk payload.

    `payload` must be the plaintext bytes returned by
    `compression.read_chunk_payload` — i.e. any outer flag=1/2/3
    decoding has already happened. This decoder handles the *nested*
    per-record zlib layer itself (inline, not via
    `decompress_payload_bytes`) because the inner layer has its own
    `[decompSize][compSize][blob]` framing that the TLV-style
    `decompress_payload_bytes` path doesn't understand.

    Envelope only — see module docstring for the scope cut.
    """
    n = len(payload)
    if n < IMAGE_BANK_COUNT_SIZE:
        raise ImageDecodeError(
            f"0x6666 Images: payload must hold at least the "
            f"{IMAGE_BANK_COUNT_SIZE}-byte u32 count prefix but got {n}. "
            f"Antibody #2 byte-count."
        )

    count = int.from_bytes(payload[:IMAGE_BANK_COUNT_SIZE], "little", signed=True)
    if count < 0:
        raise ImageDecodeError(
            f"0x6666 Images: count prefix decoded as {count} (signed "
            f"int32). Negative counts are nonsense; Antibody #1 "
            f"strict-unknown. Likely outer-layer zlib or RC4 drift."
        )

    pos = IMAGE_BANK_COUNT_SIZE
    images: list[Image] = []
    record_starts: list[int] = []

    for i in range(count):
        record_start = pos
        record_starts.append(record_start)

        if pos + IMAGE_OUTER_HEADER_SIZE > n:
            raise ImageDecodeError(
                f"0x6666 Images: record #{i} starts at offset 0x{pos:x} "
                f"but only {n - pos} bytes remain (need "
                f"≥{IMAGE_OUTER_HEADER_SIZE} for the outer header). "
                f"Antibody #2 byte-count."
            )

        raw_handle, decompressed_size, compressed_size = _IMAGE_OUTER.unpack_from(
            payload, pos
        )

        if decompressed_size < 0 or compressed_size < 0:
            raise ImageDecodeError(
                f"0x6666 Images: record #{i} (raw_handle={raw_handle}) "
                f"at offset 0x{pos:x} has negative sizes "
                f"(decompressed_size={decompressed_size}, "
                f"compressed_size={compressed_size}). Antibody #1 "
                f"strict-unknown: likely zlib or RC4 drift."
            )
        if decompressed_size < IMAGE_INNER_HEADER_SIZE:
            raise ImageDecodeError(
                f"0x6666 Images: record #{i} (raw_handle={raw_handle}) "
                f"declares decompressed_size={decompressed_size}, less "
                f"than the fixed {IMAGE_INNER_HEADER_SIZE}-byte inner "
                f"header. Antibody #2 byte-count."
            )

        body_start = pos + IMAGE_OUTER_HEADER_SIZE
        body_end = body_start + compressed_size
        if body_end > n:
            raise ImageDecodeError(
                f"0x6666 Images: record #{i} (raw_handle={raw_handle}) "
                f"at offset 0x{pos:x} claims compressed_size="
                f"{compressed_size} but only {n - body_start} bytes "
                f"remain after the outer header. Antibody #2 byte-count."
            )

        compressed_body = payload[body_start:body_end]
        try:
            inner_blob = zlib.decompress(compressed_body)
        except zlib.error as exc:
            raise ImageDecodeError(
                f"0x6666 Images: record #{i} (raw_handle={raw_handle}) "
                f"at offset 0x{pos:x}: inner zlib decompress failed: "
                f"{exc}. Antibody #1 strict-unknown: likely RC4 drift "
                f"on the outer chunk."
            ) from exc

        if len(inner_blob) != decompressed_size:
            raise ImageDecodeError(
                f"0x6666 Images: record #{i} (raw_handle={raw_handle}): "
                f"declared decompressed_size={decompressed_size} but "
                f"zlib produced {len(inner_blob)} bytes. Antibody #2."
            )

        (
            checksum,
            references,
            data_size,
            width,
            height,
            graphic_mode,
            flags,
            reserved,
            hotspot_x,
            hotspot_y,
            action_x,
            action_y,
            transparent_raw,
        ) = _decode_inner(inner_blob, raw_handle)

        r, g, b, a = transparent_raw[0], transparent_raw[1], transparent_raw[2], transparent_raw[3]
        image_data = bytes(inner_blob[IMAGE_INNER_HEADER_SIZE:])

        # Build-284 handle adjustment. We follow CTFAK here; this module
        # hardcodes the adjust because every pack we target is build ≥ 284.
        # When probe #7 grows a legacy-build code path this becomes a
        # parameter.
        logical_handle = raw_handle - IMAGE_BUILD_284_HANDLE_ADJUST

        images.append(
            Image(
                raw_handle=raw_handle,
                handle=logical_handle,
                record_start_offset=record_start,
                compressed_size=compressed_size,
                decompressed_size=decompressed_size,
                checksum=checksum,
                references=references,
                data_size=data_size,
                width=width,
                height=height,
                graphic_mode=graphic_mode,
                flags=flags,
                reserved=reserved,
                hotspot_x=hotspot_x,
                hotspot_y=hotspot_y,
                action_x=action_x,
                action_y=action_y,
                transparent=(r, g, b, a),
                image_data=image_data,
            )
        )

        pos = body_end

    if pos != n:
        raise ImageDecodeError(
            f"0x6666 Images: decoded {count} records ending at offset "
            f"0x{pos:x} but payload is {n} bytes. Trailing "
            f"{n - pos} bytes unaccounted-for. Antibody #2 byte-count."
        )

    return ImageBank(
        count=count,
        images=tuple(images),
        record_start_offsets=tuple(record_starts),
    )
