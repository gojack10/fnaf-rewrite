"""0x5556 FontOffsets decoder (probe #10).

Top-level chunk. The payload is a flat array of little-endian u32 file
offsets — one per font in the 0x6667 Fonts bank. Each offset points into
the Fonts chunk body at the start of that font's per-entry record, so
0x5556 is the index table the Fonts bank's random-access path consumes.

Naming note. CTFAK2.0's `ChunkList.cs` labels this chunk "Font Handles"
(id 21846 = 0x5556). That name is misleading for the same reason as
0x5555 / 0x5557: it's not a handle map, it's an *offset* table. Both
CTFAK1 and Anaconda (`chunkloaders/offsets.py`) name it
"Font Offsets" / `FontOffsets`, and Anaconda reads it with
`readInt(asUnsigned=True)` — i.e. a u32. We keep "Font Handles" as the
CHUNK_NAMES entry (that's what ships in the histogram) and name this
module / dataclass after the semantically correct term, mirroring
`ImageOffsets` and `SoundOffsets`.

Schema (CTFAK1 `ChunkList.cs` + Anaconda `offsets.py::_OffsetCommon`):

    items: (u32 LE) × (len(payload) / 4)

No count prefix — the length is implicit in the payload size. This
decoder is the structural twin of `image_offsets.decode_image_offsets`
and `sound_offsets.decode_sound_offsets`; FontOffsets is an empty
subclass of `_OffsetCommon` in Anaconda, meaning its schema is
identical to ImageOffsets / SoundOffsets / MusicOffsets.

Antibody coverage (this decoder):

- #1 strict-unknown : a payload whose length is not a multiple of 4
  raises `FontOffsetsDecodeError` instead of silently truncating.
- #2 byte-count    : every byte of the payload is consumed; the offset
  count is reported and verifiable against `len(payload) // 4`.
- #5 multi-input   : runs against the FNAF 1 0x5556 payload.
- #7 snapshot      : (count, offsets_sha256) pinned in the test
  module. Any single-byte drift in the offset table flips the hash.

Cross-chunk antibody (probe #10 integration test): every non-zero offset
in `FontOffsets.offsets` must resolve to a valid Font entry inside the
Fonts chunk body. The handshake uses the same `_deltas(key_attr)` +
XOR-singular pattern used to pick the sound handle convention.
"""

from __future__ import annotations

from dataclasses import dataclass

FONT_OFFSET_ENTRY_SIZE = 4  # u32 LE per offset


class FontOffsetsDecodeError(ValueError):
    """0x5556 FontOffsets decode failure — carries byte-count context."""


@dataclass(frozen=True)
class FontOffsets:
    """Decoded 0x5556 chunk payload.

    `offsets[i]` is the byte offset (within the 0x6667 Fonts chunk body,
    after decompression) where the i-th font entry begins. `count`
    equals `len(offsets)` and is duplicated for ergonomic access —
    callers printing the bank summary want it without a `len(...)`
    wrapper.
    """

    count: int
    offsets: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            # Offsets are tiny (FNAF 1's Fonts bank is a handful of
            # entries). Keep them as ints so JSON stays greppable.
            "offsets": list(self.offsets),
        }


def decode_font_offsets(payload: bytes) -> FontOffsets:
    """Decode a 0x5556 FontOffsets payload.

    The caller has already run the raw TLV payload through
    `compression.read_chunk_payload` (0x5556 is flagged compressed in
    FNAF 1; the pipeline handles that transparently).
    """
    size = len(payload)
    if size % FONT_OFFSET_ENTRY_SIZE != 0:
        raise FontOffsetsDecodeError(
            f"0x5556 FontOffsets: payload is {size} bytes, not a multiple "
            f"of {FONT_OFFSET_ENTRY_SIZE}. Antibody #2: byte count must "
            f"reconcile with u32 LE entry stream."
        )

    count = size // FONT_OFFSET_ENTRY_SIZE
    offsets = tuple(
        int.from_bytes(
            payload[i * FONT_OFFSET_ENTRY_SIZE : (i + 1) * FONT_OFFSET_ENTRY_SIZE],
            "little",
        )
        for i in range(count)
    )
    return FontOffsets(count=count, offsets=offsets)
