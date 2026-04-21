"""0x3345 FrameLayerEffects decoder (probe #4.9).

Second variable-length nested-TLV flag=3 sub-chunk, after
[[Probe #4.8 FrameLayers]]. The shape generalises the cursor-based
decoder pattern to a second type and proves the plumbing is not
FrameLayers-overfit.

Unlike FrameLayers this sub-chunk is a **parallel array**, not
length-prefixed. The record count is implicit - the reader iterates
`layers.Items.Count` times, where layers comes from the peer 0x3341
FrameLayers sub-chunk in the same Frame container. That cross-chunk
dependency is this probe's cross-channel antibody: FrameLayers and
FrameLayerEffects ride separate encrypted bytes (both flag=3), so any
RC4 drift on either side breaks the count-parity equality.

Schema cross-checked against two independent references:

- CTFAK2.0 `Core/CTFAK.Core/CCN/Chunks/Frame/Frame.cs` case 13125
  (line 211 as of current checkout): reader loops over
  `layers.Items.Count`. Per entry, reads `ReadInt16` Effect,
  `ReadInt16` EffectParam, `ReadColor` RGBCoeff (4 bytes RGBA via
  ByteReader.ReadColor -> r, g, b, a), `ReadInt32` InkEffect,
  `ReadInt32` NumberOfParams, `ReadInt32` Offset. Then IF
  `InkEffect != -1 && Effect > 0` seeks to `Offset` and reads
  NumberOfParams shader params, each either int32 (type 0/2/3) or
  float (type 1) per the shader's param-list schema. The `shader
  effect taken` branch doesn't fire for FNAF 1 build 284 (no custom
  shaders ship with the game); we assert that empirically and raise
  loudly if it does. Expanding to parse shader-param blobs is a
  follow-up probe gated on FNAF 1 ever containing one.
- Anaconda `mmfparser/data/chunkloaders/frame.py`:

        class LayerEffects(DataLoader):
            def read(self, reader):
                reader.seek(0, 2)

  Anaconda punts - skips the whole payload. CTFAK is the load-bearing
  oracle here; Anaconda is confirmation only that the chunk id binds.

Fixed per-entry size: 2 + 2 + 4 + 4 + 4 + 4 = 20 bytes. Total payload
for FNAF 1 should therefore be exactly `20 * num_layers` bytes; any
other total means either the count-parity broke (decrypt drift) or
the shader-param tail is actually present and we need a bigger probe.

Antibody coverage (this decoder):

- #1 strict-unknown: no hidden fields - every one of the 20 bytes per
  entry is pinned. If `InkEffect != -1 && Effect > 0` (the shader
  branch) we raise loudly instead of silently skipping the tail.
- #2 byte-count: `20 * num_layers` in, `20 * num_layers` consumed, or
  raise loudly. Antibody #2 is tight on this probe because the sum
  crosses RC4-decrypted bytes AND a count from a sibling RC4-decrypted
  chunk.
- #3 round-trip: synthetic pack/unpack in tests.
- #4 multi-oracle: caller threads `num_layers` in from the peer
  FrameLayers sub-chunk. If RC4 drifts on either side the `num_layers`
  arg won't match the actual record count encoded in the bytes and
  the byte-count guard fires.
- #5 multi-input: decode runs against all 17 FNAF 1 frames (20 layer-
  effect records total: 16 single-layer frames + 1 four-layer office
  frame).
- #7 snapshot: per-frame-per-layer (effect, effect_param, rgb_coeff,
  ink_effect) tuples pinned in tests. 20 records * 4 salient fields
  = 80 independent byte-patterns.

`number_of_params` and `offset` are kept but not load-bearing outside
the shader-param tail case. They're exposed on the dataclass so a
future shader-enabled probe can read them without reopening this one.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# Per-entry fixed shape: Effect i16 + EffectParam i16 + RGBCoeff
# (r, g, b, a as 4 raw bytes) + InkEffect i32 + NumberOfParams i32 +
# Offset i32 = 20 bytes. Both counts and `offset` are signed int32 per
# CTFAK's ReadInt32 (InkEffect uses -1 as the "no shader" sentinel, so
# the signed choice is load-bearing).
_FRAME_LAYER_EFFECT_FIXED = struct.Struct("<hh4Biii")
FRAME_LAYER_EFFECT_FIXED_SIZE = _FRAME_LAYER_EFFECT_FIXED.size  # 20
assert FRAME_LAYER_EFFECT_FIXED_SIZE == 20


class FrameLayerEffectsDecodeError(ValueError):
    """0x3345 FrameLayerEffects decode failure - carries offset + byte-count context."""


@dataclass(frozen=True)
class FrameLayerEffect:
    """One decoded layer-effect record inside a 0x3345 FrameLayerEffects payload.

    Fields follow CTFAK2.0 Frame.cs case 13125 exactly:

    - `effect`: int16. Clickteam effect slot id. `<= 0` is "no effect".
    - `effect_param`: int16. Type-dependent param value.
    - `rgb_coeff`: 4-tuple (r, g, b, a) of bytes. RGB blend coefficient
      from ByteReader.ReadColor (reads r, g, b, a in that order); kept
      as the raw 4-byte tuple so round-trip is lossless.
    - `ink_effect`: int32. Shader handle or `-1` for "no shader".
    - `number_of_params`: int32. Count of shader params at `offset`.
    - `offset`: int32. Absolute offset within the LayerEffects payload
      to the shader-param blob (only meaningful when `has_shader_tail`).
    """
    effect: int
    effect_param: int
    rgb_coeff: tuple[int, int, int, int]
    ink_effect: int
    number_of_params: int
    offset: int

    @property
    def has_shader_tail(self) -> bool:
        """True iff the entry points at a shader-param blob per CTFAK's
        branch condition `InkEffect != -1 && Effect > 0`. If any FNAF 1
        entry ever becomes True we need a follow-up probe to actually
        parse the tail bytes - this decoder raises loudly in that case
        rather than silently skipping them (Antibody #1 strict-unknown).
        """
        return self.ink_effect != -1 and self.effect > 0

    def as_dict(self) -> dict:
        return {
            "effect": self.effect,
            "effect_param": self.effect_param,
            "rgb_coeff": list(self.rgb_coeff),
            "ink_effect": self.ink_effect,
            "number_of_params": self.number_of_params,
            "offset": self.offset,
        }


@dataclass(frozen=True)
class FrameLayerEffects:
    """One decoded 0x3345 FrameLayerEffects sub-chunk.

    `effects` is the ordered tuple of `FrameLayerEffect`s parallel to
    the frame's FrameLayers records - `effects[i]` describes
    `layers[i]`'s effect configuration.
    """
    effects: tuple[FrameLayerEffect, ...]

    @property
    def count(self) -> int:
        return len(self.effects)

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "effects": [eff.as_dict() for eff in self.effects],
        }


def decode_frame_layer_effects(
    payload: bytes, *, num_layers: int
) -> FrameLayerEffects:
    """Decode a 0x3345 FrameLayerEffects sub-chunk's plaintext bytes.

    The caller is responsible for having already run the payload through
    the flag=3 decrypt+decompress path (probe #4.5) and for threading
    `num_layers` from the peer 0x3341 FrameLayers sub-chunk in the same
    Frame container. That cross-chunk dependency is this probe's
    multi-oracle antibody - decrypt drift on either chunk breaks the
    byte-count reconcile at the end.
    """
    if num_layers < 0:
        raise FrameLayerEffectsDecodeError(
            f"0x3345 FrameLayerEffects: num_layers must be >= 0, got "
            f"{num_layers}. Caller threads this in from the peer "
            f"FrameLayers sub-chunk."
        )

    expected = FRAME_LAYER_EFFECT_FIXED_SIZE * num_layers
    if len(payload) != expected:
        raise FrameLayerEffectsDecodeError(
            f"0x3345 FrameLayerEffects: expected exactly {expected} "
            f"bytes ({num_layers} layers * "
            f"{FRAME_LAYER_EFFECT_FIXED_SIZE} B/entry, no-shader "
            f"assumption) but got {len(payload)}. Antibody #2: byte "
            f"count must reconcile across chunks. If this fires, "
            f"either (a) RC4 drifted on FrameLayers or "
            f"FrameLayerEffects and num_layers disagrees with the "
            f"encoded count, or (b) FNAF 1 actually ships a shader-"
            f"enabled layer and this probe needs to expand to parse "
            f"the shader-param tail."
        )

    effects: list[FrameLayerEffect] = []
    for i in range(num_layers):
        pos = i * FRAME_LAYER_EFFECT_FIXED_SIZE
        effect, effect_param, r, g, b, a, ink_effect, number_of_params, offset = (
            _FRAME_LAYER_EFFECT_FIXED.unpack_from(payload, pos)
        )
        entry = FrameLayerEffect(
            effect=effect,
            effect_param=effect_param,
            rgb_coeff=(r, g, b, a),
            ink_effect=ink_effect,
            number_of_params=number_of_params,
            offset=offset,
        )
        if entry.has_shader_tail:
            raise FrameLayerEffectsDecodeError(
                f"0x3345 FrameLayerEffects: layer #{i} ships a shader-"
                f"enabled effect (effect={entry.effect}, "
                f"ink_effect={entry.ink_effect}) at offset 0x{pos:x}. "
                f"This decoder (probe #4.9) assumes FNAF 1 build 284 "
                f"ships no custom shaders; the shader-param-tail branch "
                f"is not implemented. Antibody #1 strict-unknown: "
                f"raising loudly instead of silently skipping the tail. "
                f"Expand this probe (or spin #4.9.1) to parse shader "
                f"params from absolute offset {entry.offset}."
            )
        effects.append(entry)

    return FrameLayerEffects(effects=tuple(effects))
