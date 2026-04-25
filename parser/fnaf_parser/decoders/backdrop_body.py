"""Backdrop body decoder for ObjectInfo Properties (0x4446) bodies whose
``object_type`` is ``OBJECT_TYPE_BACKDROP`` (=1).

Backdrops are static-image decoration objects. In FNAF 1 they cover door
frames, hallway photos, posters, and the giant 1280x720 office /
hallway / vent / cam-static panels — pure visuals with no animation
or game logic attached. Position is NOT stored here: this body holds
the *template* (image dimensions + image handle); the per-instance
x/y placement lives in the FrameItemInstance (0x3338) of each Frame.

Wire format (FNAF 1 build-284, post decompress)
------------------------------------------------

    offset  size  field
    ------  ----  -----
       0     4    u32  size  (== len(payload); 18 in FNAF 1)
       4     2    u16  obstacle_type   (∈ {0:None, 1:Solid, 2:Platform,
                                         3:Ladder, 4:Transparent}; FNAF 1
                                         only uses 0)
       6     2    u16  collision_type  (∈ {0:Fine, 1:Box}; FNAF 1 only
                                         uses 0)
       8     4    u32  width
      12     4    u32  height
      16     2    u16  image_handle    (resolves against ImageBank.handles)

Empirical FNAF 1 inventory (pinned in tests):

* 15 Backdrops total, every body exactly 18 bytes.
* ``obstacle_type`` is 0 (None) on all 15 — FNAF 1 does not use
  Clickteam's backdrop obstacle layer for collision; gameplay collision
  is event-driven via Active overlap conditions.
* ``collision_type`` is 0 (Fine) on all 15.
* ``width``/``height`` carry 6 distinct shapes: 1280x720 (full screen),
  200x200 (square sprites for door buttons / cam icons), 206x27 / 426x224
  / 822x25 / 1280x29 (HUD strips), etc. All values fit in the low 16
  bits of the u32 fields.
* 15 distinct ``image_handle`` values, every one resolving into the
  605-record ImageBank handle set (sparse handles, not ``handle <
  image_count``).

Scope cut
---------

V0 keeps every wire field explicit. The runtime needs ``image_handle``
plus ``width``/``height`` to render the backdrop sprite at the
FrameItemInstance position. ``obstacle_type`` and ``collision_type``
are kept for round-trip even though FNAF 1 never uses non-zero values —
if a future Clickteam game uses backdrop obstacles we want the data
present, not silently dropped.

Antibodies
----------

* ``len(payload)`` must equal 18; any other size is a wire-shape
  violation (FNAF 1 always emits the full Backdrop body, never the
  legacy ``Settings.Old`` 10-byte variant).
* ``size`` field at offset 0 must equal ``len(payload)``.
* Image handle validates against the ImageBank handle set at the
  integration-test layer (caller has the bank).

Caveman-parity correction (recorded 2026-04-25)
-----------------------------------------------

The pre-decoder recon block on Non-Active Body Decoders claimed the
layout was ``u32 size + 4-byte zero pad + u16 x + u16 y + u16 w + u16 h
+ u16 image_handle``. Re-probing surfaced three errors:

1. There is no "x/y" field — backdrop bodies do NOT carry position.
   Position is in FrameItemInstance.
2. Width and height are u32, not u16. FNAF 1's high halves happen to
   be zero because no backdrop is ≥ 65536px, but the wire field is
   genuinely 32 bits (CTFAK2.0 ``Backdrop.cs:52-53`` confirms).
3. The "4-byte zero pad" is actually ``u16 obstacle_type`` + ``u16
   collision_type``. Both happen to be zero on every FNAF 1 backdrop,
   so the recon mistook them for padding.

Cross-checked against ``reference/CTFAK2.0/Core/CTFAK.Core/CCN/Chunks/
Objects/Backdrop.cs`` after the probe surfaced the corrected shape.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# --- Wire-format constants ---------------------------------------------

#: Total payload size of every FNAF 1 Backdrop body. The legacy
#: ``Settings.Old`` 10-byte variant (no width/height) is NOT supported;
#: FNAF 1 is built-284, post-Old. Fixed-width.
BACKDROP_BODY_SIZE = 18

#: u32 ``size`` mirror at offset 0.
BACKDROP_SIZE_FIELD_OFFSET = 0

#: u16 ``obstacle_type`` enum (FNAF 1 always 0 = None).
BACKDROP_OBSTACLE_TYPE_OFFSET = 4

#: u16 ``collision_type`` enum (FNAF 1 always 0 = Fine).
BACKDROP_COLLISION_TYPE_OFFSET = 6

#: u32 ``width`` of the backdrop image template, in pixels.
BACKDROP_WIDTH_OFFSET = 8

#: u32 ``height`` of the backdrop image template, in pixels.
BACKDROP_HEIGHT_OFFSET = 12

#: u16 ``image_handle`` resolving against the ImageBank handle set.
BACKDROP_IMAGE_HANDLE_OFFSET = 16

_SIZE_STRUCT = struct.Struct("<I")
_OBSTACLE_STRUCT = struct.Struct("<H")
_COLLISION_STRUCT = struct.Struct("<H")
_WIDTH_STRUCT = struct.Struct("<I")
_HEIGHT_STRUCT = struct.Struct("<I")
_IMAGE_HANDLE_STRUCT = struct.Struct("<H")


class BackdropBodyDecodeError(ValueError):
    """Backdrop body decode failure with offset / handle context."""


# --- Dataclass ---------------------------------------------------------


@dataclass(frozen=True)
class BackdropBody:
    """Decoded Backdrop ObjectInfo property body.

    Carries the runtime-consumable fields (``image_handle``, ``width``,
    ``height``) plus the obstacle/collision enums for round-trip support.
    """

    size: int
    obstacle_type: int
    collision_type: int
    width: int
    height: int
    image_handle: int

    def summary_dict(self) -> dict:
        return {
            "size": self.size,
            "obstacle_type": self.obstacle_type,
            "collision_type": self.collision_type,
            "width": self.width,
            "height": self.height,
            "image_handle": self.image_handle,
        }

    def as_dict(self) -> dict:
        # as_dict is the debug surface used by ObjectInfo.as_dict; it
        # carries the same scalar fields as summary_dict for backdrops
        # since there are no opaque raw spans to track.
        return self.summary_dict()


# --- Decoder -----------------------------------------------------------


def decode_backdrop_body(payload: bytes) -> BackdropBody:
    """Decode one 0x4446 Properties body where ``object_type == 1``.

    Antibodies enforced:

    * ``len(payload) == 18``. Any other size is a wire-shape violation.
    * ``size`` field at offset 0 equals ``len(payload)``.
    """
    n = len(payload)
    if n != BACKDROP_BODY_SIZE:
        raise BackdropBodyDecodeError(
            f"Backdrop body: payload is {n} bytes, expected exactly "
            f"{BACKDROP_BODY_SIZE}. Antibody #2 byte-count."
        )

    (size,) = _SIZE_STRUCT.unpack_from(payload, BACKDROP_SIZE_FIELD_OFFSET)
    if size != n:
        raise BackdropBodyDecodeError(
            f"Backdrop body: size field at offset 0 is {size} but payload "
            f"length is {n}. Antibody #2 byte-count."
        )

    (obstacle_type,) = _OBSTACLE_STRUCT.unpack_from(
        payload, BACKDROP_OBSTACLE_TYPE_OFFSET
    )
    (collision_type,) = _COLLISION_STRUCT.unpack_from(
        payload, BACKDROP_COLLISION_TYPE_OFFSET
    )
    (width,) = _WIDTH_STRUCT.unpack_from(payload, BACKDROP_WIDTH_OFFSET)
    (height,) = _HEIGHT_STRUCT.unpack_from(payload, BACKDROP_HEIGHT_OFFSET)
    (image_handle,) = _IMAGE_HANDLE_STRUCT.unpack_from(
        payload, BACKDROP_IMAGE_HANDLE_OFFSET
    )

    return BackdropBody(
        size=size,
        obstacle_type=obstacle_type,
        collision_type=collision_type,
        width=width,
        height=height,
        image_handle=image_handle,
    )
