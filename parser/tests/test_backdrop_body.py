"""Regression tests for the Backdrop ObjectInfo body decoder.

Backdrop is the FNAF 1 static-image decoration object_type (=1). Its
0x4446 property body is a fixed 18-byte struct (size + obstacle/collision
enums + width + height + image_handle). The decoder consumes every
field; nothing is opaque.

Antibody coverage:

- #2 byte-count: payload not exactly 18 bytes raises; ``size`` field at
  offset 0 must match ``len(payload)``.
- #3 round-trip: synthetic Backdrop body packs and decodes back to the
  original fields.
- #4 multi-oracle: image-handle membership cross-checked against the
  FNAF 1 fixture (covered by ``test_frame_items.test_fnaf1_backdrop_body_snapshot``).
"""

from __future__ import annotations

import struct

import pytest

from fnaf_parser.decoders.backdrop_body import (
    BACKDROP_BODY_SIZE,
    BACKDROP_COLLISION_TYPE_OFFSET,
    BACKDROP_HEIGHT_OFFSET,
    BACKDROP_IMAGE_HANDLE_OFFSET,
    BACKDROP_OBSTACLE_TYPE_OFFSET,
    BACKDROP_SIZE_FIELD_OFFSET,
    BACKDROP_WIDTH_OFFSET,
    BackdropBody,
    BackdropBodyDecodeError,
    decode_backdrop_body,
)


def _pack_backdrop_body(
    *,
    obstacle_type: int = 0,
    collision_type: int = 0,
    width: int,
    height: int,
    image_handle: int,
    size_override: int | None = None,
) -> bytes:
    """Pack a synthetic Backdrop property body in the FNAF 1 wire layout.

    Defaults reflect the FNAF 1 fixture: obstacle=None, collision=Fine.
    """
    size_field = BACKDROP_BODY_SIZE if size_override is None else size_override
    return b"".join(
        [
            struct.pack("<I", size_field),
            struct.pack("<H", obstacle_type),
            struct.pack("<H", collision_type),
            struct.pack("<I", width),
            struct.pack("<I", height),
            struct.pack("<H", image_handle),
        ]
    )


def test_module_constants_stable():
    """Pin offsets so any wire-layout drift surfaces in the smallest
    possible blast radius. Mirrors ``test_counter_body.test_module_constants_stable``.
    """
    assert BACKDROP_BODY_SIZE == 18
    assert BACKDROP_SIZE_FIELD_OFFSET == 0
    assert BACKDROP_OBSTACLE_TYPE_OFFSET == 4
    assert BACKDROP_COLLISION_TYPE_OFFSET == 6
    assert BACKDROP_WIDTH_OFFSET == 8
    assert BACKDROP_HEIGHT_OFFSET == 12
    assert BACKDROP_IMAGE_HANDLE_OFFSET == 16


def test_roundtrip_fnaf1_full_screen_backdrop():
    """Antibody #3 round-trip: a 1280x720 backdrop (the office /
    hallway / vent panels in FNAF 1) packs + decodes cleanly."""
    body = _pack_backdrop_body(width=1280, height=720, image_handle=358)
    assert len(body) == BACKDROP_BODY_SIZE

    decoded = decode_backdrop_body(body)
    assert isinstance(decoded, BackdropBody)
    assert decoded.size == 18
    assert decoded.obstacle_type == 0
    assert decoded.collision_type == 0
    assert decoded.width == 1280
    assert decoded.height == 720
    assert decoded.image_handle == 358


def test_roundtrip_fnaf1_square_button_backdrop():
    """Antibody #3 round-trip: a 200x200 button-shape backdrop packs +
    decodes cleanly. Mirrors handles 174-178 in FNAF 1."""
    body = _pack_backdrop_body(width=200, height=200, image_handle=527)
    decoded = decode_backdrop_body(body)
    assert decoded.width == 200
    assert decoded.height == 200
    assert decoded.image_handle == 527


def test_roundtrip_with_nonzero_obstacle_collision():
    """Future-proofing: even though FNAF 1 always emits obstacle=0 and
    collision=0, a synthetic body with non-zero values round-trips
    cleanly. Keeps the decoder honest if a future Clickteam game uses
    the obstacle layer."""
    body = _pack_backdrop_body(
        obstacle_type=1, collision_type=1, width=100, height=50, image_handle=42
    )
    decoded = decode_backdrop_body(body)
    assert decoded.obstacle_type == 1
    assert decoded.collision_type == 1


def test_payload_wrong_size_raises_too_short():
    """Antibody #2 byte-count: payloads under 18 bytes are rejected
    loudly so a truncated wire shape can't sneak through."""
    short = b"\x00" * 17
    with pytest.raises(BackdropBodyDecodeError, match="payload is 17 bytes"):
        decode_backdrop_body(short)


def test_payload_wrong_size_raises_too_long():
    """Antibody #2 byte-count: payloads over 18 bytes are also rejected.
    Backdrop has a fixed wire shape; trailing junk indicates RC4 drift
    or a Clickteam variant we have not probed."""
    long = b"\x00" * 19
    with pytest.raises(BackdropBodyDecodeError, match="payload is 19 bytes"):
        decode_backdrop_body(long)


def test_size_field_mismatch_raises():
    """Antibody #2: ``size`` field at offset 0 must equal ``len(payload)``.
    A drifted size (e.g. RC4 misalignment, slicing bug) raises."""
    body = bytearray(_pack_backdrop_body(width=100, height=100, image_handle=1))
    struct.pack_into("<I", body, 0, 999)
    with pytest.raises(BackdropBodyDecodeError, match="size field at offset 0 is 999"):
        decode_backdrop_body(bytes(body))


def test_summary_dict_shape():
    """``summary_dict`` carries the runtime-consumable fields for
    ``runtime_pack/object_bank/objects.json[*].properties_summary``."""
    body = _pack_backdrop_body(width=1280, height=720, image_handle=358)
    decoded = decode_backdrop_body(body)
    assert decoded.summary_dict() == {
        "size": 18,
        "obstacle_type": 0,
        "collision_type": 0,
        "width": 1280,
        "height": 720,
        "image_handle": 358,
    }


def test_as_dict_shape():
    """``as_dict`` is the debug surface used by ``ObjectInfo.as_dict``;
    Backdrop has no opaque raw spans so it mirrors ``summary_dict``."""
    body = _pack_backdrop_body(width=426, height=224, image_handle=576)
    decoded = decode_backdrop_body(body)
    assert decoded.as_dict() == {
        "size": 18,
        "obstacle_type": 0,
        "collision_type": 0,
        "width": 426,
        "height": 224,
        "image_handle": 576,
    }
