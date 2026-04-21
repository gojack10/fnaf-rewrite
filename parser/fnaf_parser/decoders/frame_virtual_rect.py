"""0x3342 FrameVirtualRect decoder (probe #4.7).

Schema cross-checked against two independent references:

- CTFAK2.0 `Core/CTFAK.Core/CCN/Chunks/Frame/Frame.cs` (class VirtualRect):
  four `int32` fields read in order `left`, `top`, `right`, `bottom`.
- Anaconda `mmfparser/data/chunkloaders/frame.py` (`class VirtualSize
  (Rectangle): pass`) with `Rectangle` defined in `common.pyx`:
  `reader.readInt()` called four times in the same order.

Both consume exactly 16 bytes. Types are signed int32 LE (Clickteam
allows negative virtual-rect origins for frames that pan past (0, 0),
even though FNAF 1 does not use them).

Total size: 4 * 4 = 16 bytes.

Antibody coverage (this decoder):

- #2 byte-count : 16 bytes in, 16 bytes consumed, or raise loudly.
- #3 round-trip : synthetic pack/unpack in tests.
- #4 multi-oracle: 16 of 17 frames should carry a virtual rect whose
  `(right-left, bottom-top)` equals AppHeader's `window_width/height`.
  AppHeader arrives via flag=1 (zlib only); VirtualRect via flag=3
  (zlib + RC4). Independent channels - any RC4 drift desynchronises
  all 17 rects at once.
- #5 multi-input: decode runs against all 17 FNAF 1 frame rects.
- #7 snapshot  : per-frame (left, top, right, bottom) tuples pinned in
  tests.

Third independent flag=3 shape after FrameHeader (16 B) and
FramePalette (1028 B) - distinct byte shapes all passing clean means
the RC4 plumbing is not overfit to any single decoder.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

FRAME_VIRTUAL_RECT_SIZE = 16
_FRAME_VIRTUAL_RECT_STRUCT = struct.Struct("<iiii")


class FrameVirtualRectDecodeError(ValueError):
    """0x3342 FrameVirtualRect decode failure - carries byte-count context."""


@dataclass(frozen=True)
class FrameVirtualRect:
    """One decoded 0x3342 FrameVirtualRect sub-chunk.

    Fields are signed int32 LE as shipped by Clickteam. For FNAF 1 every
    rect has `left == top == 0` and `(right, bottom)` = frame dimensions,
    but the type keeps the signed representation so a future file with
    negative origins decodes cleanly.
    """
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_dict(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }


def decode_frame_virtual_rect(payload: bytes) -> FrameVirtualRect:
    """Decode a 0x3342 FrameVirtualRect sub-chunk's plaintext bytes.

    The caller is responsible for having already run the payload through
    the flag=3 decrypt+decompress path (probe #4.5). This function only
    sees the final plaintext.
    """
    if len(payload) != FRAME_VIRTUAL_RECT_SIZE:
        raise FrameVirtualRectDecodeError(
            f"0x3342 FrameVirtualRect: expected {FRAME_VIRTUAL_RECT_SIZE} "
            f"bytes (4x int32 LE: left, top, right, bottom) but got "
            f"{len(payload)}. Antibody #2: byte count must reconcile."
        )

    left, top, right, bottom = _FRAME_VIRTUAL_RECT_STRUCT.unpack(payload)
    return FrameVirtualRect(left=left, top=top, right=right, bottom=bottom)
