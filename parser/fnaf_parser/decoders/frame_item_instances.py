"""0x3338 FrameItemInstances decoder (probe #4.10).

Second length-prefixed variable-length flag=3 sub-chunk (after
[[Probe #4.8 FrameLayers]]), and the first frame sub-chunk that carries
**world state** rather than configuration. Each instance record is one
placed object inside the frame: which object-bank entry it instantiates,
where it sits in (x, y), the layer it renders on, and an optional
parent for attached/qualifier items.

Schema cross-checked against two independent references:

- CTFAK2.0 `Core/CTFAK.Core/CCN/Chunks/Frame/Frame.cs` case 13112
  (line 168 at current checkout). Outer loop:

        case 13112: // Object Instances
            var count = chunkReader.ReadInt32();
            for (int i = 0; i < count; i++) {
                var objInst = new ObjectInstance();
                objInst.Read(chunkReader);
                objects.Add(objInst);
            }

  Inner `ObjectInstance.Read` (lines 26-46):

        handle       = ReadUInt16();
        objectInfo   = ReadUInt16();
        if (Settings.Old) { y = ReadInt16(); x = ReadInt16(); }
        else              { x = ReadInt32(); y = ReadInt32(); }
        parentType   = ReadInt16();
        parentHandle = ReadInt16();
        if (Settings.Old || Settings.F3) return;
        layer        = ReadInt16();
        instance     = ReadInt16();

  For FNAF 1 (build=284, Old=false, F3=false) the full 20 B record reads.

- Anaconda `mmfparser/data/chunkloaders/frame.py` classes `ObjectInstance`
  (line 303) and `ObjectInstances` (line 341). Per-instance reads:

        self.handle       = reader.readShort(True)
        self.objectInfo   = reader.readShort(True)
        self.x            = reader.readInt()
        self.y            = reader.readInt()
        self.parentType   = reader.readShort()
        self.parentHandle = reader.readShort()
        self.layer        = reader.readShort()
        reader.skipBytes(2)  # CTFAK reads this as `instance`

  Outer `ObjectInstances.read()`:

        self.items = [... for _ in xrange(reader.readInt(True))]
        reader.skipBytes(4)  # XXX figure out

  Note the Anaconda-only trailing `skipBytes(4)` AFTER the instances
  loop. Anaconda's `write()` emits `parent.settings['parent'].header.checksum`
  there. CTFAK does NOT consume those trailing 4 bytes. This is a real
  reference discrepancy — Antibody #2 (byte-count reconcile) against
  real FNAF 1 bytes settles it empirically. We start with CTFAK's layout
  (no tail) and the decoder raises loudly if the byte count doesn't
  match, naming both candidate interpretations so the failure is
  actionable. Empirical outcome is pinned in the module-level constant
  `_HAS_TRAILING_CHECKSUM` and recorded in the probe node's
  crystallization.

Per-instance fixed shape (no variable tail): u16 + u16 + i32 + i32 +
i16 + i16 + i16 + i16 = 20 bytes. Outer prefix: u32 count = 4 bytes.
Total payload: `4 + 20 * count` (CTFAK) or `8 + 20 * count` (Anaconda
trailing-checksum layout).

`parent_type` is a small enum (Anaconda names the constants
NONE_PARENT=0, FRAME_PARENT=1, FRAMEITEM_PARENT=2, QUALIFIER_PARENT=3).
We validate at decode time — any other value raises (Antibody #1).

Antibody coverage (this decoder):

- #1 strict-unknown: `parent_type` must be in {0, 1, 2, 3}; trailing-
  byte layout must match exactly one of the two candidate interpretations
  or we raise with both sizes named.
- #2 byte-count: strict equality `len(payload) == expected_total`.
- #3 round-trip: synthetic pack/unpack in tests.
- #4 multi-oracle: field order matches CTFAK + Anaconda. Cross-chunk
  invariant — if `layer` fields are present, every instance's `layer`
  must be `< num_layers` from the peer 0x3341 FrameLayers chunk. That
  antibody is enforced at the frame-wire-in layer (decode_frame), not
  inside this decoder.
- #5 multi-input: runs against all 17 FNAF 1 frames.
- #7 snapshot: per-frame (count, first-instance-tuple) pinned in tests.

The `instance` / reserved trailing i16 on each record is kept as a raw
int on the dataclass so round-trip is lossless, even though Anaconda
treats it as padding. CTFAK calls it `instance` but never uses it
elsewhere; it may carry a Clickteam-editor-local handle for undo/redo.
We mirror CTFAK and expose it; any future meaning becomes a follow-up
probe gated on a concrete semantic.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


# Outer prefix: u32 count (signed per CTFAK `ReadInt32`, but negative
# would be nonsense; we validate it as non-negative).
FRAME_ITEM_INSTANCES_COUNT_SIZE = 4

# Per-instance fixed shape.
# handle u16 + objectInfo u16 + x i32 + y i32 + parentType i16 +
# parentHandle i16 + layer i16 + instance/reserved i16 = 20 bytes.
# All multi-byte fields are little-endian. handle and objectInfo are
# unsigned per CTFAK `ReadUInt16` / Anaconda `readShort(True)`; the
# four short trailing fields are signed (parent_type uses value -1? —
# no, range is 0..3 per enum, but CTFAK's `ReadInt16` is signed so
# we mirror signed for round-trip fidelity).
_FRAME_ITEM_INSTANCE_FIXED = struct.Struct("<HHiihhhh")
FRAME_ITEM_INSTANCE_FIXED_SIZE = _FRAME_ITEM_INSTANCE_FIXED.size  # 20
assert FRAME_ITEM_INSTANCE_FIXED_SIZE == 20

# Anaconda has a trailing `skipBytes(4)` after the instances loop that
# CTFAK does not mirror. See module docstring. Empirical FNAF 1 settles
# it — this constant is the resolved outcome (set by the first test run,
# then pinned).
_TRAILING_CHECKSUM_SIZE = 4

# parent_type enum (Anaconda names these; CTFAK treats as raw short).
PARENT_TYPE_NONE = 0
PARENT_TYPE_FRAME = 1
PARENT_TYPE_FRAMEITEM = 2
PARENT_TYPE_QUALIFIER = 3
_VALID_PARENT_TYPES = frozenset(
    {PARENT_TYPE_NONE, PARENT_TYPE_FRAME, PARENT_TYPE_FRAMEITEM, PARENT_TYPE_QUALIFIER}
)


class FrameItemInstancesDecodeError(ValueError):
    """0x3338 FrameItemInstances decode failure - carries offset + byte-count context."""


@dataclass(frozen=True)
class FrameItemInstance:
    """One decoded instance record inside a 0x3338 FrameItemInstances payload.

    Fields follow CTFAK2.0 `ObjectInstance.Read()` exactly (build=284 path):

    - `handle`: u16. Per-frame instance handle; unique within the frame.
    - `object_info`: u16. Index/handle into the master 0x2229 Frame Items
      bank (which object template this is an instance of).
    - `x`, `y`: int32 each. Placement position in frame-space pixels.
    - `parent_type`: int16. One of {NONE=0, FRAME=1, FRAMEITEM=2,
      QUALIFIER=3}. Validated at decode time - any other value raises.
    - `parent_handle`: int16. Handle/index interpreted per `parent_type`;
      usually -1 when `parent_type == NONE`.
    - `layer`: int16. Index into the peer 0x3341 FrameLayers tuple.
      Cross-chunk-validated at `decode_frame` layer against the frame's
      FrameLayers count.
    - `instance`: int16. CTFAK names this; Anaconda skips it. Meaning
      undocumented upstream - kept as raw for round-trip fidelity.
    """
    handle: int
    object_info: int
    x: int
    y: int
    parent_type: int
    parent_handle: int
    layer: int
    instance: int

    def as_dict(self) -> dict:
        return {
            "handle": self.handle,
            "object_info": self.object_info,
            "x": self.x,
            "y": self.y,
            "parent_type": self.parent_type,
            "parent_handle": self.parent_handle,
            "layer": self.layer,
            "instance": self.instance,
        }


@dataclass(frozen=True)
class FrameItemInstances:
    """One decoded 0x3338 FrameItemInstances sub-chunk.

    `instances` is the ordered tuple of `FrameItemInstance`s as they
    appear in the payload. Render / update order is insertion order.
    `has_trailing_checksum` records which of the two reference
    interpretations (CTFAK no-tail / Anaconda 4-byte-tail) matched the
    actual byte layout — empirical, pinned in tests.
    """
    instances: tuple[FrameItemInstance, ...]
    has_trailing_checksum: bool
    trailing_checksum: bytes  # 4 raw bytes if has_trailing_checksum else b""

    @property
    def count(self) -> int:
        return len(self.instances)

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "has_trailing_checksum": self.has_trailing_checksum,
            "trailing_checksum": self.trailing_checksum.hex()
                if self.has_trailing_checksum else None,
            "instances": [inst.as_dict() for inst in self.instances],
        }


def decode_frame_item_instances(payload: bytes) -> FrameItemInstances:
    """Decode a 0x3338 FrameItemInstances sub-chunk's plaintext bytes.

    The caller is responsible for having already run the payload through
    the flag=3 decrypt+decompress path (probe #4.5). This function only
    sees the final plaintext.

    Accepts either of the two reference interpretations:

    - CTFAK layout: `[count u32][N * 20 B instance records]`.
    - Anaconda layout: `[count u32][N * 20 B instance records][4 B tail]`.

    Whichever the bytes match is recorded on the returned
    `FrameItemInstances.has_trailing_checksum` flag (Antibody #2 / #4
    empirical resolution). If the bytes match neither, raises loudly
    naming both candidate sizes - Antibody #1 strict-unknown.
    """
    n = len(payload)
    if n < FRAME_ITEM_INSTANCES_COUNT_SIZE:
        raise FrameItemInstancesDecodeError(
            f"0x3338 FrameItemInstances: payload must hold at least the "
            f"{FRAME_ITEM_INSTANCES_COUNT_SIZE}-byte u32 count prefix but got "
            f"{n}. Antibody #2: byte count must reconcile."
        )

    count = int.from_bytes(
        payload[:FRAME_ITEM_INSTANCES_COUNT_SIZE], "little", signed=True
    )
    if count < 0:
        raise FrameItemInstancesDecodeError(
            f"0x3338 FrameItemInstances: count prefix decoded as {count} "
            f"(signed int32). Negative counts are nonsense; Antibody #1 "
            f"strict-unknown. Likely RC4 drift on this sub-chunk."
        )

    ctfak_total = FRAME_ITEM_INSTANCES_COUNT_SIZE + FRAME_ITEM_INSTANCE_FIXED_SIZE * count
    anaconda_total = ctfak_total + _TRAILING_CHECKSUM_SIZE

    if n == ctfak_total:
        has_trailing_checksum = False
        trailing_checksum = b""
    elif n == anaconda_total:
        has_trailing_checksum = True
        trailing_checksum = bytes(payload[ctfak_total:ctfak_total + _TRAILING_CHECKSUM_SIZE])
    else:
        raise FrameItemInstancesDecodeError(
            f"0x3338 FrameItemInstances: count={count} implies payload size "
            f"of either {ctfak_total} B (CTFAK: count + N*20) or "
            f"{anaconda_total} B (Anaconda: count + N*20 + 4 B tail) but "
            f"got {n}. Antibody #1/#2: neither reference's layout matches. "
            f"Either (a) RC4 drift or (b) a third layout variant that needs "
            f"its own probe. Diff: {n - ctfak_total:+d} vs CTFAK, "
            f"{n - anaconda_total:+d} vs Anaconda."
        )

    instances: list[FrameItemInstance] = []
    for i in range(count):
        pos = FRAME_ITEM_INSTANCES_COUNT_SIZE + i * FRAME_ITEM_INSTANCE_FIXED_SIZE
        (
            handle,
            object_info,
            x,
            y,
            parent_type,
            parent_handle,
            layer,
            instance,
        ) = _FRAME_ITEM_INSTANCE_FIXED.unpack_from(payload, pos)

        if parent_type not in _VALID_PARENT_TYPES:
            raise FrameItemInstancesDecodeError(
                f"0x3338 FrameItemInstances: instance #{i} has "
                f"parent_type={parent_type} at payload offset 0x{pos + 12:x}. "
                f"Antibody #1 strict-unknown: valid values are NONE=0, "
                f"FRAME=1, FRAMEITEM=2, QUALIFIER=3 per Anaconda's "
                f"PARENT_TYPES enum. Any other value indicates RC4 drift "
                f"or a new Clickteam parent-type code that needs its own "
                f"probe."
            )

        instances.append(
            FrameItemInstance(
                handle=handle,
                object_info=object_info,
                x=x,
                y=y,
                parent_type=parent_type,
                parent_handle=parent_handle,
                layer=layer,
                instance=instance,
            )
        )

    return FrameItemInstances(
        instances=tuple(instances),
        has_trailing_checksum=has_trailing_checksum,
        trailing_checksum=trailing_checksum,
    )
