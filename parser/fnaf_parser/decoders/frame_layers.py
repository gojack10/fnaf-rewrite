"""0x3341 FrameLayers decoder (probe #4.8).

First variable-length nested-TLV flag=3 sub-chunk. Every prior post-
decrypt decoder was a fixed-size shape (FrameHeader 16 B, FramePalette
1028 B, FrameVirtualRect 16 B). FrameLayers is length-prefixed: a u32
count followed by that many `FrameLayer` records, each a 20-byte fixed
header plus a null-terminated Unicode-gated name string.

Schema cross-checked against two independent references:

- CTFAK2.0 `Core/CTFAK.Core/CCN/Chunks/Frame/Frame.cs` (classes `Layers`
  and `Layer`): `ReadUInt32()` count, then per layer `ReadUInt32` flags,
  two `ReadSingle()` coefficients, two `ReadInt32()` counts, an optional
  `reader.Skip(6)` gated on `Settings.Fusion3Seed` (NOT taken for build
  284), then `ReadYuniversal()` name.
- Anaconda `mmfparser/data/chunkloaders/frame.py` (classes `Layers` and
  `Layer`): same field order; `readInt(True)` flags, `readFloat()` x/y
  coefficients, `readInt()` num_backgrounds / background_index,
  `self.readString(reader)` Unicode-gated name.

Both read exactly the five fixed fields plus the null-terminated name.
The CTFAK `Layer` class DECLARES extra fields (ShaderData, InkEffect,
RGBCoeff, blend, etc.) but does NOT read them in `Read()` — those are
Fusion3-only or runtime-only. We mirror the read path, not the
declaration.

Name encoding:
- Unicode pack (FNAF 1 is PAMU magic, unicode=True): UTF-16LE, terminated
  by a single `\x00\x00` wide NUL at a 2-byte-aligned offset.
- ASCII pack: single-byte, terminated by `\x00`.

Per-layer byte count: 20 + (2 * (len(name) + 1)) under unicode, or
(20 + len(name) + 1) under ASCII. Variable-length, so the decoder is
cursor-based rather than a single `struct.Struct`.

Antibody coverage (this decoder):

- #2 byte-count: every byte of `payload` must be consumed after reading
  `count` u32 + N layers; any leftover or short-consume raises loudly.
- #3 round-trip: synthetic pack/unpack in tests, both encodings.
- #4 multi-oracle (cross-channel): the Unicode flag driving `name`
  parsing is the same flag AppHeader is shipped under - a fourth
  independent flag=3 shape after FrameHeader, FramePalette, and
  FrameVirtualRect, now with a real Unicode string inside the
  encrypted payload. Any RC4 drift scrambles the string byte pattern
  and every per-frame name snapshot breaks at once.
- #5 multi-input: runs against all 17 FNAF 1 frames.
- #7 snapshot: per-frame (layer_count, first_layer_name, flags, coeffs)
  pinned in tests.

The `flags` field is kept as a raw u32. The BitDict names (XCoefficient,
YCoefficient, Visible, WrapHorizontally, Redraw, ...) are documented in
CTFAK and Anaconda but decoded layer flag semantics are a rendering
concern - deferred until the rendering stack needs them.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


# Fixed prefix: u32 count.
FRAME_LAYERS_COUNT_SIZE = 4

# Per-layer fixed header (pre-name): flags u32 + x f32 + y f32 + nbg i32
# + bg_index i32 = 20 bytes. `I` is unsigned u32 (flags is a bitfield);
# both counts are signed int32 per CTFAK ReadInt32 and Anaconda readInt.
_FRAME_LAYER_FIXED = struct.Struct("<Iffii")
FRAME_LAYER_FIXED_SIZE = _FRAME_LAYER_FIXED.size  # 20
assert FRAME_LAYER_FIXED_SIZE == 20


class FrameLayersDecodeError(ValueError):
    """0x3341 FrameLayers decode failure - carries offset + byte-count context."""


@dataclass(frozen=True)
class FrameLayer:
    """One decoded layer record inside a 0x3341 FrameLayers payload.

    `flags` is kept as a raw u32 so round-trip is lossless; BitDict
    semantics are a rendering concern deferred past this probe.
    `number_of_backgrounds` and `background_index` are int32 signed per
    both reference readers (Clickteam occasionally uses -1 as a "none"
    sentinel for optional indices).
    """
    flags: int
    x_coefficient: float
    y_coefficient: float
    number_of_backgrounds: int
    background_index: int
    name: str

    def as_dict(self) -> dict:
        return {
            "flags": f"0x{self.flags:08X}",
            "x_coefficient": self.x_coefficient,
            "y_coefficient": self.y_coefficient,
            "number_of_backgrounds": self.number_of_backgrounds,
            "background_index": self.background_index,
            "name": self.name,
        }


@dataclass(frozen=True)
class FrameLayers:
    """One decoded 0x3341 FrameLayers sub-chunk.

    `layers` is the ordered tuple of `FrameLayer`s as they appear in the
    payload; a layer's render order is its position in this tuple.
    """
    layers: tuple[FrameLayer, ...]

    @property
    def count(self) -> int:
        return len(self.layers)

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "layers": [layer.as_dict() for layer in self.layers],
        }


def _read_null_terminated(
    payload: bytes, start: int, *, unicode: bool
) -> tuple[str, int]:
    """Read a null-terminated string starting at `start`.

    Returns `(decoded, next_offset)` where `next_offset` is the byte index
    immediately after the NUL terminator. Raises `FrameLayersDecodeError`
    if no terminator is found before `len(payload)`, matching the strict
    "no silent junk" posture of Antibody #2.
    """
    n = len(payload)
    if unicode:
        # Wide NUL terminator at a 2-byte-aligned position. Scan two bytes
        # at a time, matching CTFAK ReadWideString / Anaconda readUnicodeString.
        pos = start
        while pos + 2 <= n:
            if payload[pos] == 0 and payload[pos + 1] == 0:
                body = payload[start:pos]
                try:
                    decoded = body.decode("utf-16le")
                except UnicodeDecodeError as exc:
                    raise FrameLayersDecodeError(
                        f"FrameLayer name: UTF-16LE decode failed at "
                        f"offset 0x{start:x}: {exc}"
                    ) from exc
                return decoded, pos + 2
            pos += 2
        raise FrameLayersDecodeError(
            f"FrameLayer name: UTF-16 string at offset 0x{start:x} is not "
            f"NUL-terminated within the payload (len={n}). Antibody #2: "
            f"missing terminator indicates decrypt drift or wrong framing."
        )

    # ASCII / single-byte path. FNAF 1 pack is Unicode so this branch is
    # exercised only by synthetic tests, kept in lockstep with strings.py
    # for spec completeness.
    term = payload.find(b"\x00", start)
    if term == -1:
        raise FrameLayersDecodeError(
            f"FrameLayer name: ASCII string at offset 0x{start:x} is not "
            f"NUL-terminated within the payload (len={n})."
        )
    body = payload[start:term]
    try:
        decoded = body.decode("ascii")
    except UnicodeDecodeError as exc:
        raise FrameLayersDecodeError(
            f"FrameLayer name: ASCII decode failed at offset 0x{start:x}: {exc}"
        ) from exc
    return decoded, term + 1


def decode_frame_layers(payload: bytes, *, unicode: bool) -> FrameLayers:
    """Decode a 0x3341 FrameLayers sub-chunk's plaintext bytes.

    The caller is responsible for having already run the payload through
    the flag=3 decrypt+decompress path (probe #4.5). This function only
    sees the final plaintext.

    `unicode` comes from the pack header (PAMU magic) - the same flag
    that gates AppHeader's name strings. Cross-channel antibody #4 relies
    on both channels agreeing on this flag.
    """
    if len(payload) < FRAME_LAYERS_COUNT_SIZE:
        raise FrameLayersDecodeError(
            f"0x3341 FrameLayers: payload must hold at least the "
            f"{FRAME_LAYERS_COUNT_SIZE}-byte u32 count prefix but got "
            f"{len(payload)}. Antibody #2: byte count must reconcile."
        )

    count = int.from_bytes(payload[:FRAME_LAYERS_COUNT_SIZE], "little", signed=False)
    pos = FRAME_LAYERS_COUNT_SIZE
    n = len(payload)

    layers: list[FrameLayer] = []
    for i in range(count):
        if pos + FRAME_LAYER_FIXED_SIZE > n:
            raise FrameLayersDecodeError(
                f"0x3341 FrameLayers: layer #{i} needs {FRAME_LAYER_FIXED_SIZE} "
                f"bytes for its fixed header at offset 0x{pos:x} but only "
                f"{n - pos} bytes remain. Antibody #2."
            )
        flags, x_coeff, y_coeff, num_bg, bg_idx = _FRAME_LAYER_FIXED.unpack_from(
            payload, pos
        )
        pos += FRAME_LAYER_FIXED_SIZE
        name, pos = _read_null_terminated(payload, pos, unicode=unicode)
        layers.append(
            FrameLayer(
                flags=flags,
                x_coefficient=x_coeff,
                y_coefficient=y_coeff,
                number_of_backgrounds=num_bg,
                background_index=bg_idx,
                name=name,
            )
        )

    if pos != n:
        raise FrameLayersDecodeError(
            f"0x3341 FrameLayers: decoded {count} layers and consumed "
            f"{pos} of {n} bytes; {n - pos} trailing bytes unaccounted for. "
            f"Antibody #2: byte count must reconcile."
        )

    return FrameLayers(layers=tuple(layers))
