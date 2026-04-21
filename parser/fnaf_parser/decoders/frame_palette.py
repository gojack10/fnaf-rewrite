"""0x3337 FramePalette decoder (probe #4.6).

Schema cross-checked against two independent references:

- CTFAK2.0 `Core/CTFAK.Core/CCN/Chunks/Frame/Frame.cs` (class FramePalette):
  reads 257 colors via `reader.ReadColor()` → R, G, B, A byte-order.
- Anaconda `mmfparser/data/chunkloaders/frame.py` (class FramePalette):
  skips 4 bytes ("XXX figure this out"), then reads 256 colors.

Both consume exactly 1028 bytes. They disagree on whether the first 4
bytes are the 257th palette entry (CTFAK) or a reserved/unknown header
(Anaconda). An indexed palette has 256 slots (8-bit index), so we side
with Anaconda's semantic split: a 4-byte reserved prefix followed by
256 RGBA entries. The raw first 4 bytes are kept as `reserved` so a
later probe can reason about them without re-decoding.

Byte order per `ByteReader.ReadColor`:

    R = read_byte
    G = read_byte
    B = read_byte
    A = read_byte

Matches FrameHeader.background (probe #4.5), so a single byte-order
convention covers both decoders.

Total size: 4 + 256*4 = 1028 bytes.

Antibody coverage (this decoder):

- #2 byte-count : 1028 bytes in, 1028 bytes consumed, or raise loudly.
- #5 multi-input: decode runs against all 17 FNAF 1 frame palettes.
- #7 snapshot  : per-frame entry[0] / entry[255] tuples pinned in tests.
- #4 multi-oracle (secondary): this is a second flag=3 sub-chunk after
  FrameHeader, so a clean 1028-byte consume across every frame is
  independent evidence that the probe #4.5 decrypt+decompress plumbing
  isn't overfit to the FrameHeader shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FRAME_PALETTE_ENTRY_COUNT = 256
FRAME_PALETTE_RESERVED_SIZE = 4
FRAME_PALETTE_ENTRY_SIZE = 4
FRAME_PALETTE_SIZE = (
    FRAME_PALETTE_RESERVED_SIZE
    + FRAME_PALETTE_ENTRY_COUNT * FRAME_PALETTE_ENTRY_SIZE
)  # 1028


Color = tuple[int, int, int, int]  # (R, G, B, A)


class FramePaletteDecodeError(ValueError):
    """0x3337 FramePalette decode failure — carries byte-count context."""


@dataclass(frozen=True)
class FramePalette:
    """One decoded 0x3337 FramePalette sub-chunk.

    `reserved` is the opaque first 4 bytes (Anaconda's "XXX figure this
    out" prefix). Kept so a future probe can investigate without having
    to re-decode. `entries` is the 256-entry RGBA palette, each a
    (R, G, B, A) 4-tuple of 0–255 ints.
    """
    reserved: bytes = field(repr=False)
    entries: tuple[Color, ...]

    def as_dict(self) -> dict:
        return {
            "reserved": self.reserved.hex(),
            "entry_count": len(self.entries),
            # Keep the full table; palettes are tiny and dumping them
            # out makes snapshot diffs obvious.
            "entries": [
                {"r": r, "g": g, "b": b, "a": a}
                for (r, g, b, a) in self.entries
            ],
        }


def decode_frame_palette(payload: bytes) -> FramePalette:
    """Decode a 0x3337 FramePalette sub-chunk's plaintext bytes.

    The caller is responsible for having already run the payload through
    the flag=3 decrypt+decompress path (probe #4.5). This function only
    sees the final plaintext.
    """
    if len(payload) != FRAME_PALETTE_SIZE:
        raise FramePaletteDecodeError(
            f"0x3337 FramePalette: expected {FRAME_PALETTE_SIZE} bytes "
            f"(4 reserved + {FRAME_PALETTE_ENTRY_COUNT} RGBA entries) but "
            f"got {len(payload)}. Antibody #2: byte count must reconcile."
        )

    reserved = bytes(payload[:FRAME_PALETTE_RESERVED_SIZE])
    entries_blob = payload[FRAME_PALETTE_RESERVED_SIZE:]

    # Slice the 1024-byte entry table into 256 4-byte chunks. Avoids a
    # struct format string ~1000 chars long and keeps the RGBA byte
    # order visible in code.
    entries: list[Color] = []
    for i in range(FRAME_PALETTE_ENTRY_COUNT):
        base = i * FRAME_PALETTE_ENTRY_SIZE
        r = entries_blob[base]
        g = entries_blob[base + 1]
        b = entries_blob[base + 2]
        a = entries_blob[base + 3]
        entries.append((r, g, b, a))

    return FramePalette(reserved=reserved, entries=tuple(entries))
