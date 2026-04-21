"""0x3334 FrameHeader decoder (probe #4.5).

Schema cross-checked against CTFAK2.0 `Core/CTFAK.Core/CCN/Chunks/Frame/
Frame.cs` case 13108 (non-old branch):

    width       i32 LE
    height      i32 LE
    background  R/G/B/A  (4 bytes, Color.FromArgb(a, r, g, b))
    flags       u32 LE   (BitDict: DisplayTitle, GrabDesktop, KeepDisplay,
                          HandleCollision, ResizeAtStart, TimeMovements,
                          DontInclude, DontEraseBG, plus unknown bits)

Total: 16 bytes. FNAF 1 is non-old (Fusion 2.5, build=284); the 8-byte
"old" variant does not apply.

Antibody #2 (byte-count): 16 bytes in, 16 bytes consumed.
Antibody #4 (multi-oracle): this is the first decoder on a flag=3
(compressed-and-encrypted) sub-chunk, so a successful decode with the
expected width=640 / height=480 cross-validates the decryption state
against a channel totally independent of the app-metadata chunks.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

FRAME_HEADER_SIZE = 16
_STRUCT = struct.Struct("<ii4BI")
assert _STRUCT.size == FRAME_HEADER_SIZE


# CTFAK's BitDict for Frame.flags. Unnamed slots stay as "BitN" so a set
# bit we don't have a name for still shows up in the snapshot.
_FLAG_NAMES_BASE: dict[int, str] = {
    0: "DisplayTitle",
    1: "GrabDesktop",
    2: "KeepDisplay",
    5: "HandleCollision",
    8: "ResizeAtStart",
    15: "TimeMovements",
    18: "DontInclude",
    19: "DontEraseBG",
}


def _flag_names(width: int = 32) -> tuple[str, ...]:
    return tuple(_FLAG_NAMES_BASE.get(i, f"Bit{i}") for i in range(width))


FLAG_NAMES: tuple[str, ...] = _flag_names()


class FrameHeaderDecodeError(ValueError):
    """0x3334 FrameHeader decode failure — carries byte-count context."""


@dataclass(frozen=True)
class FrameHeader:
    width: int
    height: int
    background: tuple[int, int, int, int]   # (R, G, B, A)
    flags: int

    def flag_names_set(self) -> tuple[str, ...]:
        return tuple(
            name for i, name in enumerate(FLAG_NAMES) if (self.flags >> i) & 1
        )

    def as_dict(self) -> dict:
        r, g, b, a = self.background
        return {
            "width": self.width,
            "height": self.height,
            "background": {"r": r, "g": g, "b": b, "a": a},
            "flags": f"0x{self.flags:08X}",
            "flags_set": list(self.flag_names_set()),
        }


def decode_frame_header(payload: bytes) -> FrameHeader:
    if len(payload) != FRAME_HEADER_SIZE:
        raise FrameHeaderDecodeError(
            f"0x3334 FrameHeader: expected {FRAME_HEADER_SIZE} bytes but "
            f"got {len(payload)}. Antibody #2: byte count must reconcile."
        )
    width, height, r, g, b, a, flags = _STRUCT.unpack(payload)
    return FrameHeader(
        width=width,
        height=height,
        background=(r, g, b, a),
        flags=flags,
    )
