"""Empirical-only chunk decoders (probe #4.3).

Two chunk IDs appear in the FNAF 1 pack that NEITHER CTFAK2.0's
ChunkList nor Anaconda's chunkloaders register:

    0x224D  (8 bytes on disk, flag=0 / NotCompressed)
    0x224F  (23 bytes on disk compressed → 12 bytes / flag=1)

CTFAK's GUI source has a stray `//224F` TODO comment (MainForm.cs:295)
but no loader behind it, confirming the chunk is known-of but
not-spec'd upstream. These live in the chunk_ids.EMPIRICAL set and are
probed here by raw-byte fixture rather than by translating a reference.

Empirical layouts and values (FNAF 1):

    0x224D payload:  `00 00 00 00 03 00 00 00`
        Interpreted as two uint32 LE: (0, 3).
        Provisional meaning: a (flag, level) style pair — the `3`
        matches no other known FNAF1 field directly, so we lock the
        raw value and move on. A different game would tell us more.

    0x224F payload:  `1c 01 00 00 0a 00 00 00 00 00 00 00`
        Interpreted as three uint32 LE: (284, 10, 0).
        The first field (284) is EXACTLY the pack header's
        `product_build`. That gives us Antibody #4 (multi-oracle) for
        free: a cross-chunk invariant test locks it. The other two
        fields are unknown; preserved raw.

Both decoders are defensive: they assert the exact decompressed size
FNAF 1 produces. A different-sized payload from a different build
would fail the length antibody immediately and route the investigator
back here instead of silently misreading.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


class UnknownChunkDecodeError(ValueError):
    """Empirical chunk decode failure — length or struct mismatch."""


# --- 0x224D ------------------------------------------------------------

_CHUNK_224D_SIZE = 8
_CHUNK_224D_STRUCT = struct.Struct("<II")
assert _CHUNK_224D_STRUCT.size == _CHUNK_224D_SIZE


@dataclass(frozen=True)
class UnknownChunk224D:
    """8-byte empirical chunk: two uint32 LE fields."""
    value_a: int
    value_b: int
    raw: bytes

    def as_dict(self) -> dict:
        return {
            "value_a": self.value_a,
            "value_b": self.value_b,
            "raw_hex": self.raw.hex(),
        }


def decode_unknown_224d(payload: bytes) -> UnknownChunk224D:
    if len(payload) != _CHUNK_224D_SIZE:
        raise UnknownChunkDecodeError(
            f"0x224D (empirical): expected {_CHUNK_224D_SIZE} bytes, got "
            f"{len(payload)}. If a non-FNAF-1 build lands here this is the "
            f"place to widen the spec."
        )
    a, b = _CHUNK_224D_STRUCT.unpack(payload)
    return UnknownChunk224D(value_a=a, value_b=b, raw=bytes(payload))


# --- 0x224F ------------------------------------------------------------

_CHUNK_224F_SIZE = 12
_CHUNK_224F_STRUCT = struct.Struct("<III")
assert _CHUNK_224F_STRUCT.size == _CHUNK_224F_SIZE


@dataclass(frozen=True)
class UnknownChunk224F:
    """12-byte empirical chunk: three uint32 LE fields.

    Field names reflect current knowledge:
    - `build_stamp` (field 0) equals pack_header.product_build for FNAF 1.
    - `value_b` / `value_c` meaning unknown; zero for FNAF 1.
    """
    build_stamp: int
    value_b: int
    value_c: int
    raw: bytes

    def as_dict(self) -> dict:
        return {
            "build_stamp": self.build_stamp,
            "value_b": self.value_b,
            "value_c": self.value_c,
            "raw_hex": self.raw.hex(),
        }


def decode_unknown_224f(payload: bytes) -> UnknownChunk224F:
    if len(payload) != _CHUNK_224F_SIZE:
        raise UnknownChunkDecodeError(
            f"0x224F (empirical): expected {_CHUNK_224F_SIZE} bytes, got "
            f"{len(payload)}. If a non-FNAF-1 build lands here this is the "
            f"place to widen the spec."
        )
    a, b, c = _CHUNK_224F_STRUCT.unpack(payload)
    return UnknownChunk224F(build_stamp=a, value_b=b, value_c=c, raw=bytes(payload))
