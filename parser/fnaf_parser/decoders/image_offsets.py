"""0x5555 ImageOffsets decoder (probe #6).

Top-level chunk. The payload is a flat array of little-endian u32 file
offsets — one per image in the 0x6666 Images bank. Each offset points
into the Images chunk body at the start of that image's per-entry
record, so 0x5555 is the index table the Images bank's random-access
path consumes.

Naming note. CTFAK2.0's `ChunkList.cs` labels this chunk "Image Handles"
(id 21845 = 0x5555). That name is misleading: it's not a handle map
(like `FrameHandles` at 0x222B, which is an `i16` array) — it's an
*offset* table. Both CTFAK1's `ChunkList.cs` and Anaconda's
`chunkloaders/offsets.py` name it "Image Offsets" / `ImageOffsets`, and
the loader in Anaconda reads it with `readInt(asUnsigned=True)` — i.e.
a u32. We keep "Image Handles" as the CHUNK_NAMES entry (that's what
ships in the histogram) and name this module / dataclass after the
semantically correct term.

Schema (CTFAK1 `ChunkList.cs` + Anaconda `offsets.py::_OffsetCommon`):

    items: (u32 LE) × (len(payload) / 4)

No count prefix — the length is implicit in the payload size. This is
the same idiom 0x5556 FontOffsets, 0x5557 SoundOffsets, and 0x5558
MusicOffsets use.

Antibody coverage (this decoder):

- #1 strict-unknown : a payload whose length is not a multiple of 4
  raises `ImageOffsetsDecodeError` instead of silently truncating.
- #2 byte-count    : every byte of the payload is consumed; the offset
  count is reported and verifiable against `len(payload) // 4`.
- #5 multi-input   : runs against the FNAF 1 0x5555 payload.
- #7 snapshot      : (count, offsets_sha256, first_offset, last_offset)
  pinned in the test module. Any single-byte drift in the offset table
  flips the hash.

Cross-chunk antibody deferred to probe #6.x (once 0x6666 Images is
decoded): every offset in `ImageOffsets.offsets` must resolve to a
valid Image entry inside the Images chunk body. Today we only pin the
envelope so that test can fail loudly later if the offset stream ever
stops matching the Images chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

IMAGE_OFFSET_ENTRY_SIZE = 4  # u32 LE per offset


class ImageOffsetsDecodeError(ValueError):
    """0x5555 ImageOffsets decode failure — carries byte-count context."""


@dataclass(frozen=True)
class ImageOffsets:
    """Decoded 0x5555 chunk payload.

    `offsets[i]` is the byte offset (within the 0x6666 Images chunk
    body, after decompression) where the i-th image entry begins.
    `count` equals `len(offsets)` and is duplicated for ergonomic
    access — callers printing the bank summary want it without a
    `len(...)` wrapper.
    """

    count: int
    offsets: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            # Offsets are tiny (≤ tens of thousands typically) and the
            # snapshot test wants the full stream. Keep them as ints so
            # JSON stays greppable.
            "offsets": list(self.offsets),
        }


def decode_image_offsets(payload: bytes) -> ImageOffsets:
    """Decode a 0x5555 ImageOffsets payload.

    The caller has already run the raw TLV payload through
    `compression.read_chunk_payload` (0x5555 is flagged uncompressed in
    FNAF 1, but routing through the pipeline keeps probe #6 forward-
    compatible with packs that compress it).
    """
    size = len(payload)
    if size % IMAGE_OFFSET_ENTRY_SIZE != 0:
        raise ImageOffsetsDecodeError(
            f"0x5555 ImageOffsets: payload is {size} bytes, not a multiple "
            f"of {IMAGE_OFFSET_ENTRY_SIZE}. Antibody #2: byte count must "
            f"reconcile with u32 LE entry stream."
        )

    count = size // IMAGE_OFFSET_ENTRY_SIZE
    offsets = tuple(
        int.from_bytes(
            payload[i * IMAGE_OFFSET_ENTRY_SIZE : (i + 1) * IMAGE_OFFSET_ENTRY_SIZE],
            "little",
        )
        for i in range(count)
    )
    return ImageOffsets(count=count, offsets=offsets)
