"""0x5557 SoundOffsets decoder (probe #8a).

Top-level chunk. The payload is a flat array of little-endian u32 file
offsets — one per sound in the 0x6668 Sounds bank. Each offset points
into the Sounds chunk body at the start of that sound's per-entry
record, so 0x5557 is the index table the Sounds bank's random-access
path consumes.

Naming note. CTFAK2.0's `ChunkList.cs` labels this chunk "Sound Handles"
(id 21847 = 0x5557). That name is misleading for the same reason as
0x5555: it's not a handle map, it's an *offset* table. Both CTFAK1 and
Anaconda (`chunkloaders/offsets.py`) name it "Sound Offsets" /
`SoundOffsets`, and Anaconda reads it with `readInt(asUnsigned=True)`
— i.e. a u32. We keep "Sound Handles" as the CHUNK_NAMES entry (that's
what ships in the histogram) and name this module / dataclass after
the semantically correct term, mirroring `ImageOffsets`.

Schema (CTFAK1 `ChunkList.cs` + Anaconda `offsets.py::_OffsetCommon`):

    items: (u32 LE) × (len(payload) / 4)

No count prefix — the length is implicit in the payload size.
SoundOffsets is an empty subclass of `_OffsetCommon` in Anaconda,
meaning its schema is identical to ImageOffsets / FontOffsets /
MusicOffsets. This decoder is the structural twin of
`image_offsets.decode_image_offsets`.

Antibody coverage (this decoder):

- #1 strict-unknown : a payload whose length is not a multiple of 4
  raises `SoundOffsetsDecodeError` instead of silently truncating.
- #2 byte-count    : every byte of the payload is consumed; the offset
  count is reported and verifiable against `len(payload) // 4`.
- #5 multi-input   : runs against the FNAF 1 0x5557 payload.
- #7 snapshot      : (count, offsets_sha256) pinned in the test
  module. Any single-byte drift in the offset table flips the hash.

Cross-chunk antibody deferred to probe #8b (once 0x6668 Sounds is
decoded): every offset in `SoundOffsets.offsets` must resolve to a
valid Sound entry inside the Sounds chunk body. Today we only pin the
envelope so that test can fail loudly later if the offset stream ever
stops matching the Sounds chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

SOUND_OFFSET_ENTRY_SIZE = 4  # u32 LE per offset


class SoundOffsetsDecodeError(ValueError):
    """0x5557 SoundOffsets decode failure — carries byte-count context."""


@dataclass(frozen=True)
class SoundOffsets:
    """Decoded 0x5557 chunk payload.

    `offsets[i]` is the byte offset (within the 0x6668 Sounds chunk
    body, after decompression) where the i-th sound entry begins.
    `count` equals `len(offsets)` and is duplicated for ergonomic
    access — callers printing the bank summary want it without a
    `len(...)` wrapper.
    """

    count: int
    offsets: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            # Offsets are tiny (≤ tens typically, FNAF 1 = 60) and the
            # snapshot test wants the full stream. Keep them as ints so
            # JSON stays greppable.
            "offsets": list(self.offsets),
        }


def decode_sound_offsets(payload: bytes) -> SoundOffsets:
    """Decode a 0x5557 SoundOffsets payload.

    The caller has already run the raw TLV payload through
    `compression.read_chunk_payload` (0x5557 is flagged compressed in
    FNAF 1; the pipeline handles that transparently).
    """
    size = len(payload)
    if size % SOUND_OFFSET_ENTRY_SIZE != 0:
        raise SoundOffsetsDecodeError(
            f"0x5557 SoundOffsets: payload is {size} bytes, not a multiple "
            f"of {SOUND_OFFSET_ENTRY_SIZE}. Antibody #2: byte count must "
            f"reconcile with u32 LE entry stream."
        )

    count = size // SOUND_OFFSET_ENTRY_SIZE
    offsets = tuple(
        int.from_bytes(
            payload[i * SOUND_OFFSET_ENTRY_SIZE : (i + 1) * SOUND_OFFSET_ENTRY_SIZE],
            "little",
        )
        for i in range(count)
    )
    return SoundOffsets(count=count, offsets=offsets)
