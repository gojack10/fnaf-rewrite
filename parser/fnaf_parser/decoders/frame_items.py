"""0x2229 FrameItems decoder (probe #5).

Top-level object-type bank: the master list of every in-game object
template the frame instances reference via their `object_info` handles.
Biggest still-undecoded outer chunk in FNAF 1 (~34,938 B, dual-
confirmed in CTFAK + Anaconda).

Wire format (post decrypt + zlib)
---------------------------------

    u32 count
    N × ObjectInfo

ObjectInfo is itself a nested ChunkList walk (same TLV shape as the
outer pack and the 0x3333 Frame container) terminated by 0x7F7F LAST.
Inner sub-chunk IDs:

    0x4444  Object Info Header   (16 B fixed)
    0x4445  Object Info Name     (null-terminated string, Unicode-gated)
    0x4446  Object Common / Properties   (OPAQUE at this probe scope)
    0x4448  Object Effects               (OPAQUE at this probe scope)
    0x7F7F  Last                 (terminator, empty body)

Each inner sub-chunk carries its own flags/size prefix and is
independently decompressed (flag 1) or decrypted+decompressed (flag 3)
via the shared `decompress_payload_bytes` path.

Object Info Header (16 B, little-endian):

    int16   handle              (per-ObjectInfo key, unique)
    int16   object_type         (closed enum -7..9 or >= EXTENSION_BASE=32)
    uint16  flags               (OBJECT_FLAGS bit dict; low 4 bits used)
    int16   reserved            (no longer used; kept for round-trip)
    uint32  ink_effect          (low 16 bits = effect id, bit 28 =
                                 transparent, bit 29 = antialias)
    uint32  ink_effect_param    (shader parameter / semi-transparent α)

Scope cut — envelope-only (matches probe #4.12 playbook)
--------------------------------------------------------

Decode header + name for every ObjectInfo. Keep 0x4446 and 0x4448
payloads as opaque `bytes` fields. Per-type bodies become follow-up
sub-probes:

    #5.1 QuickBackdrop (type 0)  — size + obstacle + collision +
                                   width + height + Shape
    #5.2 Backdrop (type 1)       — same prefix + image handle
    #5.3 ObjectCommon (2+)       — huge: 8 offset tables + 8 qualifier
                                   int16s + identifier + backcolor +
                                   per-type sub-structs, all gated on
                                   build/platform flags

Rationale: ObjectCommon alone is 460 lines of branched build/platform
reading in CTFAK2.0. Shipping it in probe #5 would blow the budget.
Envelope-only gets us the cross-chunk antibody today — every
ObjectInstance's `object_info` handle in 0x3338 FrameItemInstances
(232 instances across the 17 FNAF 1 frames) must resolve to a known
FrameItems.handle. That's the minimum-viable-ship.

Antibody coverage
-----------------

- #1 strict-unknown:
    * Unknown inner chunk id in an ObjectInfo ChunkList raises.
    * `object_type` outside the closed set {-7..9} ∪ {>= 32} raises.
    * Missing ObjectHeader (0x4444) inside an ObjectInfo raises.
    * Missing 0x7F7F terminator inside an ObjectInfo raises.
    * Duplicate handle across ObjectInfos raises (handle uniqueness).
- #2 byte-count:
    * 0x4444 header body is exactly 16 bytes.
    * Every inner sub-chunk's raw slice is bounded by its own size field.
    * Outer `count` matches the number of ObjectInfos decoded; the byte
      walk consumes exactly `len(payload)`.
- #3 round-trip: synthetic pack/unpack in tests.
- #4 multi-oracle: field order + sizes cross-checked against CTFAK2.0
  `Core/CTFAK.Core/CCN/Chunks/Objects/ObjectInfo.cs` + Anaconda
  `mmfparser/data/chunkloaders/objectinfo.pyx`.
- #5 multi-input: runs against the FNAF 1 0x2229 payload. Snapshot
  pins count (196), object-type histogram, and handle-set sample.
- Cross-chunk (new): every ObjectInstance.object_info handle referenced
  by 0x3338 FrameItemInstances must appear in FrameItems.by_handle.
  Enforced at the integration-test layer; the decoder exposes `by_handle`
  to make the check a trivial set membership.
- #6 loud-skip: `deferred_sub_chunk_ids_seen` surfaces the set of body
  sub-chunk ids whose bodies were NOT structurally decoded (i.e. genuinely
  need a follow-up decoder). 0x4446 (Properties) is added only when its
  body is present but unconsumed — Active bodies that decode through
  `decode_object_common` are NOT marked deferred. Empirically on FNAF 1
  this set still contains `{0x4446}` because 72 non-Active ObjectInfos
  (Backdrops, Texts, Counters, Extensions) carry undecoded property
  bodies pending the Counter / Non-Active body decoder track. Any other
  id that made it through Antibody #1 but wasn't decoded is listed
  there for the follow-up probe to triage.
- #7 snapshot: (count, sorted-handles-sample, object-type histogram)
  pinned in tests.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from fnaf_parser.chunk_ids import CHUNK_NAMES, LAST_CHUNK_ID
from fnaf_parser.compression import ChunkFlag, decompress_payload_bytes
from fnaf_parser.decoders.object_common import (
    ObjectCommon,
    ObjectCommonDecodeError,
    decode_object_common,
)
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import TransformState

# --- Inner sub-chunk IDs -------------------------------------------------

SUB_OBJECT_HEADER = 0x4444
SUB_OBJECT_NAME = 0x4445
SUB_OBJECT_PROPERTIES = 0x4446  # opaque at this probe scope
SUB_OBJECT_UNKNOWN = 0x4447     # CTFAK "Unknown Object Chunk"; not seen in FNAF 1
SUB_OBJECT_EFFECTS = 0x4448     # opaque at this probe scope

# Sub-chunk ids whose body this probe intentionally keeps as opaque
# bytes (Antibody #6 loud-skip surface). Sub-probes #5.1/#5.2/#5.3
# will promote these out of the set.
OPAQUE_SUB_CHUNK_IDS: frozenset[int] = frozenset(
    {SUB_OBJECT_PROPERTIES, SUB_OBJECT_UNKNOWN, SUB_OBJECT_EFFECTS}
)

# --- Object-type closed set (Antibody #1) ----------------------------------

EXTENSION_BASE = 32

OBJECT_TYPE_PLAYER = -7
OBJECT_TYPE_KEYBOARD = -6
OBJECT_TYPE_CREATE = -5
OBJECT_TYPE_TIMER = -4
OBJECT_TYPE_GAME = -3
OBJECT_TYPE_SPEAKER = -2
OBJECT_TYPE_SYSTEM = -1
OBJECT_TYPE_QUICKBACKDROP = 0
OBJECT_TYPE_BACKDROP = 1
OBJECT_TYPE_ACTIVE = 2
OBJECT_TYPE_TEXT = 3
OBJECT_TYPE_QUESTION = 4
OBJECT_TYPE_SCORE = 5
OBJECT_TYPE_LIVES = 6
OBJECT_TYPE_COUNTER = 7
OBJECT_TYPE_RTF = 8
OBJECT_TYPE_SUBAPPLICATION = 9

# Names for the known fixed enum values; extension types (>= 32) are
# open and carry a plugin-dispatch id in the high range.
OBJECT_TYPE_NAMES: dict[int, str] = {
    OBJECT_TYPE_PLAYER: "Player",
    OBJECT_TYPE_KEYBOARD: "Keyboard",
    OBJECT_TYPE_CREATE: "Create",
    OBJECT_TYPE_TIMER: "Timer",
    OBJECT_TYPE_GAME: "Game",
    OBJECT_TYPE_SPEAKER: "Speaker",
    OBJECT_TYPE_SYSTEM: "System",
    OBJECT_TYPE_QUICKBACKDROP: "QuickBackdrop",
    OBJECT_TYPE_BACKDROP: "Backdrop",
    OBJECT_TYPE_ACTIVE: "Active",
    OBJECT_TYPE_TEXT: "Text",
    OBJECT_TYPE_QUESTION: "Question",
    OBJECT_TYPE_SCORE: "Score",
    OBJECT_TYPE_LIVES: "Lives",
    OBJECT_TYPE_COUNTER: "Counter",
    OBJECT_TYPE_RTF: "RTF",
    OBJECT_TYPE_SUBAPPLICATION: "SubApplication",
}

_FIXED_TYPES: frozenset[int] = frozenset(OBJECT_TYPE_NAMES.keys())


def object_type_name(object_type: int) -> str:
    """Human name for an object_type. 'Extension(<id>)' for >= 32."""
    fixed = OBJECT_TYPE_NAMES.get(object_type)
    if fixed is not None:
        return fixed
    if object_type >= EXTENSION_BASE:
        return f"Extension({object_type})"
    return f"Unknown({object_type})"


def _is_valid_object_type(object_type: int) -> bool:
    """Closed-set membership for Antibody #1."""
    return object_type in _FIXED_TYPES or object_type >= EXTENSION_BASE


# --- Ink-effect bitfield (Anaconda objectinfo.pyx lines 187-189) ---------

INK_EFFECT_MASK = 0xFFFF
INK_EFFECT_TRANSPARENT_BIT = 28
INK_EFFECT_ANTIALIAS_BIT = 29


# --- Object flags (Anaconda OBJECT_FLAGS BitDict, low 4 bits) ------------

OBJECT_FLAG_LOAD_ON_CALL = 1 << 0
OBJECT_FLAG_DISCARDABLE = 1 << 1
OBJECT_FLAG_GLOBAL = 1 << 2
OBJECT_FLAG_RESERVED_1 = 1 << 3


# --- Wire-format struct shapes ------------------------------------------

# Outer u32 count (signed int32 per CTFAK ReadInt32; we validate non-negative).
FRAME_ITEMS_COUNT_SIZE = 4

# Inner TLV sub-chunk header: int16 id + uint16 flags + uint32 size.
# Same shape as the outer pack and the 0x3333 Frame container.
_SUB_HEADER = struct.Struct("<hHI")
SUB_HEADER_SIZE = _SUB_HEADER.size  # 8

# Object Info Header (0x4444) body: 16 bytes.
#   int16 handle, int16 object_type, uint16 flags, int16 reserved,
#   uint32 ink_effect, uint32 ink_effect_param
_OBJECT_HEADER = struct.Struct("<hhHhII")
OBJECT_HEADER_SIZE = _OBJECT_HEADER.size  # 16
assert OBJECT_HEADER_SIZE == 16


class FrameItemsDecodeError(ValueError):
    """0x2229 FrameItems decode failure - carries offset / handle context."""


# --- Dataclasses --------------------------------------------------------


@dataclass(frozen=True)
class ObjectInfoSubChunkRecord:
    """One sub-chunk inside an ObjectInfo's inner ChunkList.

    `raw` is the pre-decompression bytes as they appeared in the outer
    payload. `decoded_payload` is the post-decode plaintext (flag 0/1
    decompressed in-place; flag 2/3 decrypted when a TransformState was
    supplied, else None — the Antibody #6 loud-skip surface).
    """
    id: int
    flags: int
    size: int
    inner_offset: int  # offset within the outer 0x2229 payload
    raw: bytes = field(repr=False)
    decoded_payload: bytes | None = field(repr=False)

    @property
    def is_encrypted(self) -> bool:
        return (self.flags & ChunkFlag.ENCRYPTED) != 0


@dataclass(frozen=True)
class ObjectHeader:
    """Decoded 0x4444 Object Info Header (fixed 16-byte body).

    Ink-effect bits 28/29 carry transparent/antialias per Anaconda; we
    surface both the raw 32-bit `ink_effect` and the derived flags so
    callers have lossless access to the wire value.
    """
    handle: int
    object_type: int
    flags: int
    reserved: int
    ink_effect: int           # raw 32-bit value
    ink_effect_param: int

    @property
    def transparent(self) -> bool:
        return bool((self.ink_effect >> INK_EFFECT_TRANSPARENT_BIT) & 1)

    @property
    def antialias(self) -> bool:
        return bool((self.ink_effect >> INK_EFFECT_ANTIALIAS_BIT) & 1)

    @property
    def ink_effect_id(self) -> int:
        """Low 16 bits of ink_effect - the actual ink-blend enum id."""
        return self.ink_effect & INK_EFFECT_MASK

    def as_dict(self) -> dict:
        return {
            "handle": self.handle,
            "object_type": self.object_type,
            "object_type_name": object_type_name(self.object_type),
            "flags": self.flags,
            "reserved": self.reserved,
            "ink_effect": self.ink_effect,
            "ink_effect_id": self.ink_effect_id,
            "ink_effect_param": self.ink_effect_param,
            "transparent": self.transparent,
            "antialias": self.antialias,
        }


@dataclass(frozen=True)
class ObjectInfo:
    """One decoded ObjectInfo inside the 0x2229 FrameItems bank.

    At this probe scope, `header` and `name` are always populated (missing
    0x4444 raises; missing 0x4445 is allowed and leaves `name=None`).
    `properties_raw` and `effects_raw` hold the post-decompression bytes
    of 0x4446 and 0x4448 respectively — empty `b""` when the sub-chunk
    was absent (so round-trip can distinguish absent-vs-empty via
    `sub_records`). For Active objects, `properties` carries the decoded
    ObjectCommon body when `decode_frame_items(..., decode_active_properties=True)`
    is used. Other object types stay raw until their body shapes are
    probed separately.
    """
    header: ObjectHeader
    name: str | None
    properties_raw: bytes = field(repr=False)
    effects_raw: bytes = field(repr=False)
    sub_records: tuple[ObjectInfoSubChunkRecord, ...]
    properties: ObjectCommon | None = field(default=None, repr=False)

    @property
    def handle(self) -> int:
        return self.header.handle

    @property
    def object_type(self) -> int:
        return self.header.object_type

    def as_dict(self) -> dict:
        return {
            "header": self.header.as_dict(),
            "name": self.name,
            "properties_len": len(self.properties_raw),
            "properties_decoded": self.properties is not None,
            "properties": (
                self.properties.as_dict() if self.properties is not None else None
            ),
            "effects_len": len(self.effects_raw),
            "sub_chunks": [
                {
                    "id": f"0x{r.id:04X}",
                    "flags": r.flags,
                    "size": r.size,
                    "decoded": r.decoded_payload is not None,
                }
                for r in self.sub_records
            ],
        }


@dataclass(frozen=True)
class FrameItems:
    """Decoded 0x2229 FrameItems payload.

    `items` preserves the on-wire order; `by_handle` is the bijective
    handle → ObjectInfo map a caller uses to resolve a FrameItemInstance
    back to its template (Antibody cross-chunk).

    `deferred_encrypted` lists ObjectInfoSubChunkRecords that the decoder
    refused to touch because their flag=2/3 encoding required a
    TransformState that was not supplied (Antibody #6). When the caller
    hands in a transform this is empty and every opaque body lives in
    `ObjectInfo.{properties,effects}_raw`.

    `deferred_sub_chunk_ids_seen` is the set of inner sub-chunk ids whose
    bodies were not structurally decoded by this probe (i.e. still need a
    follow-up decoder). 0x4446 is added only when its body is present but
    not consumed — Actives whose bodies decode via `decode_object_common`
    are NOT marked deferred. On FNAF 1 the set still contains 0x4446
    because 72 non-Active ObjectInfos (Backdrops, Texts, Counters,
    Extensions) carry undecoded property bodies, and 0x4448 because no
    Effects decoder exists yet. Drops to empty once every body sub-chunk
    has a structured decoder.
    """
    items: tuple[ObjectInfo, ...]
    deferred_encrypted: tuple[ObjectInfoSubChunkRecord, ...]
    deferred_sub_chunk_ids_seen: frozenset[int]

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def by_handle(self) -> dict[int, ObjectInfo]:
        """Handle → ObjectInfo map. Built fresh each call (dataclass is
        frozen; we expose it as a property so `items` stays the single
        source of truth for serialisation)."""
        return {item.handle: item for item in self.items}

    @property
    def handles(self) -> frozenset[int]:
        """Set of every handle known to this FrameItems bank. Callers
        use it as the cross-chunk antibody check for FrameItemInstances
        (`inst.object_info in frame_items.handles`)."""
        return frozenset(item.handle for item in self.items)

    @property
    def object_type_histogram(self) -> dict[int, int]:
        """Map from `object_type` value to count of ObjectInfos carrying
        that type. Used by the Antibody #7 snapshot; extension ids are
        bucketed as-is (no Extension folding) so a snapshot drift
        points straight at the offending type id."""
        histogram: dict[int, int] = {}
        for item in self.items:
            histogram[item.object_type] = histogram.get(item.object_type, 0) + 1
        return histogram

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "items": [item.as_dict() for item in self.items],
            "deferred_encrypted": [
                f"0x{r.id:04X}" for r in self.deferred_encrypted
            ],
            "deferred_sub_chunk_ids_seen": sorted(
                f"0x{cid:04X}" for cid in self.deferred_sub_chunk_ids_seen
            ),
            "object_type_histogram": {
                str(k): v for k, v in sorted(self.object_type_histogram.items())
            },
        }


# --- Decoder -------------------------------------------------------------


def _walk_object_info(
    payload: bytes,
    start: int,
    *,
    transform: TransformState | None,
) -> tuple[tuple[ObjectInfoSubChunkRecord, ...], int]:
    """Walk one ObjectInfo's inner ChunkList starting at `start`.

    Terminates on 0x7F7F LAST. Returns the sub-records consumed and the
    absolute end offset (one byte past the LAST chunk's body).

    Mirrors the outer-pack / Frame-container TLV walker with the added
    constraint that LAST is mandatory here: an ObjectInfo that runs out
    of payload without hitting 0x7F7F is a schema violation (Antibody #1).
    """
    n = len(payload)
    pos = start
    records: list[ObjectInfoSubChunkRecord] = []

    while pos + SUB_HEADER_SIZE <= n:
        sid, sflags, ssize = _SUB_HEADER.unpack_from(payload, pos)

        if sid not in CHUNK_NAMES:
            raise FrameItemsDecodeError(
                f"0x2229 FrameItems: unknown inner sub-chunk id 0x{sid:04X} "
                f"at payload offset 0x{pos:x} (flags=0x{sflags:04x}, "
                f"size={ssize}). Antibody #1 strict-unknown: likely RC4 "
                f"drift or a Clickteam inner-chunk variant not yet probed."
            )

        raw_start = pos + SUB_HEADER_SIZE
        raw_end = raw_start + ssize
        if raw_end > n:
            raise FrameItemsDecodeError(
                f"0x2229 FrameItems: inner sub-chunk 0x{sid:04X} at "
                f"payload offset 0x{pos:x} claims size={ssize} but only "
                f"{n - raw_start} bytes remain. Antibody #2 byte-count."
            )
        raw = bytes(payload[raw_start:raw_end])

        decoded: bytes | None
        if (sflags & ChunkFlag.ENCRYPTED) != 0 and transform is None:
            # Antibody #6 loud-skip: caller did not supply a key. Keep
            # the raw ciphertext so a later pass can re-process.
            decoded = None
        else:
            decoded = decompress_payload_bytes(
                raw, flags=sflags, chunk_id=sid, transform=transform
            )

        records.append(
            ObjectInfoSubChunkRecord(
                id=sid,
                flags=sflags,
                size=ssize,
                inner_offset=pos,
                raw=raw,
                decoded_payload=decoded,
            )
        )

        if sid == LAST_CHUNK_ID:
            # Don't advance past LAST's body — we're done with this
            # ObjectInfo. Its end is raw_end (a 0x7F7F body in the wild
            # is almost always 0 bytes, but we respect the declared size
            # anyway for round-trip).
            return tuple(records), raw_end

        pos = raw_end

    raise FrameItemsDecodeError(
        f"0x2229 FrameItems: ObjectInfo ChunkList starting at payload "
        f"offset 0x{start:x} ran out of bytes (end at 0x{pos:x}, payload "
        f"length {n}) without hitting 0x7F7F LAST. Antibody #1 strict-"
        f"unknown: envelope is truncated."
    )


def _decode_object_header(payload: bytes, sub_offset: int) -> ObjectHeader:
    """Unpack a 0x4444 Object Info Header body (exactly 16 bytes)."""
    if len(payload) != OBJECT_HEADER_SIZE:
        raise FrameItemsDecodeError(
            f"0x2229 FrameItems: 0x4444 Object Info Header at sub-chunk "
            f"offset 0x{sub_offset:x} has body length {len(payload)}; "
            f"expected exactly {OBJECT_HEADER_SIZE}. Antibody #2 byte-count."
        )
    (
        handle,
        object_type,
        flags,
        reserved,
        ink_effect,
        ink_effect_param,
    ) = _OBJECT_HEADER.unpack(payload)

    if not _is_valid_object_type(object_type):
        raise FrameItemsDecodeError(
            f"0x2229 FrameItems: 0x4444 Object Info Header at sub-chunk "
            f"offset 0x{sub_offset:x} has object_type={object_type}, "
            f"outside the closed set ({{-7..9}} ∪ {{>= "
            f"{EXTENSION_BASE}}}). Antibody #1 strict-unknown: likely RC4 "
            f"drift or an unseen Clickteam object-type id."
        )

    return ObjectHeader(
        handle=handle,
        object_type=object_type,
        flags=flags,
        reserved=reserved,
        ink_effect=ink_effect,
        ink_effect_param=ink_effect_param,
    )


def _decode_one_object_info(
    payload: bytes,
    start: int,
    *,
    unicode: bool,
    transform: TransformState | None,
    deferred_sub_chunk_ids_seen: set[int],
    deferred_encrypted: list[ObjectInfoSubChunkRecord],
    decode_active_properties: bool,
) -> tuple[ObjectInfo, int]:
    """Decode one ObjectInfo starting at `start` in the outer payload.

    Returns the ObjectInfo and the absolute end offset. Mutates the two
    aggregator parameters so the caller can track cross-ObjectInfo
    deferred state.
    """
    sub_records, end = _walk_object_info(
        payload, start, transform=transform
    )

    header: ObjectHeader | None = None
    name: str | None = None
    properties_raw = b""
    effects_raw = b""
    seen_header_at: int | None = None

    for rec in sub_records:
        if rec.id == LAST_CHUNK_ID:
            continue  # terminator carries no semantic payload

        if rec.decoded_payload is None:
            # Encrypted sub-chunk we couldn't decode (no transform). Log
            # it in deferred_encrypted, skip semantic handling.
            deferred_encrypted.append(rec)
            deferred_sub_chunk_ids_seen.add(rec.id)
            continue

        if rec.id == SUB_OBJECT_HEADER:
            if seen_header_at is not None:
                raise FrameItemsDecodeError(
                    f"0x2229 FrameItems: ObjectInfo starting at payload "
                    f"offset 0x{start:x} has two 0x4444 headers "
                    f"(first at 0x{seen_header_at:x}, second at "
                    f"0x{rec.inner_offset:x}). Antibody #1 strict-unknown."
                )
            header = _decode_object_header(rec.decoded_payload, rec.inner_offset)
            seen_header_at = rec.inner_offset
        elif rec.id == SUB_OBJECT_NAME:
            name = decode_string_chunk(rec.decoded_payload, unicode=unicode)
        elif rec.id == SUB_OBJECT_PROPERTIES:
            properties_raw = rec.decoded_payload
            # Don't preemptively mark 0x4446 as deferred — the body MAY get
            # decoded below (Active path via decode_object_common). We add to
            # the deferred set after the decode block iff decoding was skipped
            # or yielded no structured ObjectCommon. Antibody (2026-04-25):
            # this used to add unconditionally, which left the post-decode
            # deferred set claiming 0x4446 was undecoded even when all 124
            # Actives decoded successfully — a misleading signal for downstream
            # consumers of the runtime pack.
        elif rec.id == SUB_OBJECT_EFFECTS:
            effects_raw = rec.decoded_payload
            deferred_sub_chunk_ids_seen.add(rec.id)
        else:
            # A known id (Antibody #1 passed in _walk) that this probe
            # doesn't have a semantic handler for. Record it so the
            # follow-up probe has a loud-skip surface to prioritise.
            deferred_sub_chunk_ids_seen.add(rec.id)

    if header is None:
        raise FrameItemsDecodeError(
            f"0x2229 FrameItems: ObjectInfo starting at payload offset "
            f"0x{start:x} is missing the mandatory 0x4444 Object Info "
            f"Header. Antibody #1 strict-unknown."
        )

    properties: ObjectCommon | None = None
    if (
        decode_active_properties
        and header.object_type == OBJECT_TYPE_ACTIVE
        and properties_raw
    ):
        try:
            properties = decode_object_common(properties_raw)
        except ObjectCommonDecodeError as exc:
            raise FrameItemsDecodeError(
                f"0x2229 FrameItems: Active ObjectInfo handle={header.handle} "
                f"name={name!r} has malformed ObjectCommon properties "
                f"({len(properties_raw)} bytes) at ObjectInfo payload "
                f"offset 0x{start:x}: {exc}"
            ) from exc

    # Now that the (optional) decode has run, mark 0x4446 as deferred only when
    # the body was present but NOT structurally decoded. This keeps the
    # deferred set honest: it lists sub-chunk ids whose bodies still need a
    # follow-up decoder, not just "ids we saw on the wire".
    if properties_raw and properties is None:
        deferred_sub_chunk_ids_seen.add(SUB_OBJECT_PROPERTIES)

    return (
        ObjectInfo(
            header=header,
            name=name,
            properties_raw=properties_raw,
            effects_raw=effects_raw,
            sub_records=sub_records,
            properties=properties,
        ),
        end,
    )


def decode_frame_items(
    payload: bytes,
    *,
    unicode: bool = True,
    transform: TransformState | None = None,
    decode_active_properties: bool = True,
) -> FrameItems:
    """Decode a 0x2229 FrameItems chunk's plaintext bytes.

    The caller is responsible for having already run the outer payload
    through the flag=3 decrypt+decompress path (probe #4.5). This
    function sees the final plaintext: `u32 count + N × ObjectInfo`.

    `unicode` governs the 0x4445 Object Info Name string decode (PAMU-
    gated; in FNAF 1 this is `True`). `transform` is threaded through
    to inner sub-chunk decompression in case any inner chunk is
    flag=2/3 encrypted.

    When `decode_active_properties` is true (the default), Active
    ObjectInfo 0x4446 bodies are decoded through `object_common.py` and
    exposed as `ObjectInfo.properties`. Non-Active properties stay in
    `properties_raw` only.

    Antibodies enforced:

    - The outer count must be non-negative and reconcile with the
      number of ObjectInfos the walker actually decoded.
    - Every ObjectInfo must carry exactly one 0x4444 header and
      terminate with 0x7F7F LAST (see `_walk_object_info`).
    - Every handle across the bank must be unique (dict invariant).
    - The outer byte walk must consume exactly `len(payload)` bytes
      (no trailing junk).
    """
    n = len(payload)
    if n < FRAME_ITEMS_COUNT_SIZE:
        raise FrameItemsDecodeError(
            f"0x2229 FrameItems: payload is {n} bytes, too short for even "
            f"the {FRAME_ITEMS_COUNT_SIZE}-byte u32 count prefix. Antibody #2."
        )

    count = int.from_bytes(
        payload[:FRAME_ITEMS_COUNT_SIZE], "little", signed=True
    )
    if count < 0:
        raise FrameItemsDecodeError(
            f"0x2229 FrameItems: count prefix decoded as {count} "
            f"(signed int32). Negative counts are nonsense; Antibody #1 "
            f"strict-unknown. Likely RC4 drift on this chunk."
        )

    pos = FRAME_ITEMS_COUNT_SIZE
    items: list[ObjectInfo] = []
    seen_handles: dict[int, int] = {}  # handle -> ObjectInfo index
    deferred_sub_chunk_ids_seen: set[int] = set()
    deferred_encrypted: list[ObjectInfoSubChunkRecord] = []

    for i in range(count):
        if pos >= n:
            raise FrameItemsDecodeError(
                f"0x2229 FrameItems: count declares {count} ObjectInfos "
                f"but payload exhausted after {i}. Last offset 0x{pos:x}, "
                f"payload length {n}. Antibody #2 byte-count."
            )
        obj, pos = _decode_one_object_info(
            payload,
            pos,
            unicode=unicode,
            transform=transform,
            deferred_sub_chunk_ids_seen=deferred_sub_chunk_ids_seen,
            deferred_encrypted=deferred_encrypted,
            decode_active_properties=decode_active_properties,
        )

        existing_idx = seen_handles.get(obj.handle)
        if existing_idx is not None:
            raise FrameItemsDecodeError(
                f"0x2229 FrameItems: duplicate handle {obj.handle} "
                f"(ObjectInfo #{i} collides with ObjectInfo "
                f"#{existing_idx}). Antibody #1 strict-unknown: handle "
                f"uniqueness invariant broken."
            )
        seen_handles[obj.handle] = i
        items.append(obj)

    if pos != n:
        raise FrameItemsDecodeError(
            f"0x2229 FrameItems: decoded {count} ObjectInfos but walk ended "
            f"at offset 0x{pos:x} with payload length {n} "
            f"({n - pos} bytes unconsumed). Antibody #2 byte-count: "
            f"outer walk drifted from count."
        )

    return FrameItems(
        items=tuple(items),
        deferred_encrypted=tuple(deferred_encrypted),
        deferred_sub_chunk_ids_seen=frozenset(deferred_sub_chunk_ids_seen),
    )
