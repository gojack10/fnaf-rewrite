"""ObjectCommon decoder for Active object properties (0x4446 bodies).

Clickteam stores per-object defaults inside the ``ObjectInfo``
``Properties`` sub-chunk. For Active objects in FNAF 1 this body is an
ObjectCommon record: a 62-byte fixed header plus offset-addressed tables.
This module decodes the fixed header and the animation table, while
keeping movement/default-value/string/extension tables as explicit opaque
spans.

Scope cut
---------

V0 decodes the pieces the Rust runtime needs first: Active animation
metadata and image handles. Movements remain opaque, but they are still
covered by byte accounting so a missed table cannot hide behind the
scope cut. Non-Active ObjectInfo property formats (QuickBackdrop,
Backdrop, Text, Counter, extensions) are not decoded here; callers should
only attach this decoder to ``object_type == Active`` until their shapes
are probed separately.

Antibodies
----------

* Header size must be exactly 62 bytes and ``header.size == len(payload)``.
* Every non-zero table offset must point inside the payload and offsets
  may not be duplicated.
* AnimationHeader / Animation / AnimationDirection offsets are range
  checked before every read.
* Coverage must account for every byte except the documented FNAF 1
  zero pad ``[62..70)``; that gap must be exactly eight zero bytes.
* The decoder exposes image handles so tests can assert each handle
  exists in the decoded 605-record image bank (sparse handle set, not
  ``handle < count``).

References
----------

* CTFAK2.0 ``Chunks/Objects/ObjectCommon.cs`` and ``Animations.cs``
* Anaconda ``chunkloaders/objects.pyx::ObjectCommon`` /
  ``AnimationHeader`` / ``AnimationDirection``
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterable

# --- Bit-name tables ----------------------------------------------------

OBJECT_FLAG_NAMES: tuple[str, ...] = (
    "DisplayInFront",
    "Background",
    "Backsave",
    "RunBeforeFadeIn",
    "Movements",
    "Animations",
    "TabStop",
    "WindowProc",
    "Values",
    "Sprites",
    "InternalBacksave",
    "ScrollingIndependant",
    "QuickDisplay",
    "NeverKill",
    "NeverSleep",
    "ManualSleep",
    "Text",
    "DoNotCreateAtStart",
    "FakeSprite",
    "FakeCollisions",
)

NEW_OBJECT_FLAG_NAMES: tuple[str, ...] = (
    "DoNotSaveBackground",
    "SolidBackground",
    "CollisionBox",
    "VisibleAtStart",
    "ObstacleSolid",
    "ObstaclePlatform",
    "AutomaticRotation",
)

OBJECT_PREFERENCE_NAMES: tuple[str, ...] = (
    "Backsave",
    "ScrollingIndependant",
    "QuickDisplay",
    "Sleep",
    "LoadOnCall",
    "Global",
    "BackEffects",
    "Kill",
    "InkEffects",
    "Transitions",
    "FineCollisions",
    "AppletProblems",
)

ANIMATION_NAMES: tuple[str, ...] = (
    "Stopped",
    "Walking",
    "Running",
    "Appearing",
    "Disappearing",
    "Bouncing",
    "Shooting",
    "Jumping",
    "Falling",
    "Climbing",
    "Crouch down",
    "Stand up",
    "User defined 1",
    "User defined 2",
    "User defined 3",
    "User defined 4",
)

# Anaconda's runtime helper treats these animation ids as single-speed
# concepts. The wire still stores both min/max; this tuple is surfaced as
# metadata only and does not rewrite the raw values.
HAS_SINGLE_SPEED_ANIMATION_IDS: frozenset[int] = frozenset({0, 3, 4, 6})

# --- Wire-format constants ---------------------------------------------

_OBJECT_COMMON_HEADER = struct.Struct(
    "<I"   # size
    "h"    # animations_offset
    "h"    # movements_offset
    "h"    # version
    "2x"   # free / padding in the FNAF 1 build-284 NORMAL branch
    "h"    # extension_offset
    "h"    # counter_offset
    "H"    # flags
    "h"    # creation flags; CTFAK/Anaconda historical name: penisFlags
    "8h"   # qualifiers
    "h"    # system_object_offset
    "h"    # values_offset
    "h"    # strings_offset
    "H"    # new_flags
    "H"    # preferences
    "4s"   # identifier ("SPRI" for Active sprites in FNAF 1)
    "I"    # back_color (BGRA u32 in references)
    "I"    # fade_in_offset
    "I"    # fade_out_offset
)

OBJECT_COMMON_HEADER_SIZE = _OBJECT_COMMON_HEADER.size
assert OBJECT_COMMON_HEADER_SIZE == 62

_ANIMATION_HEADER = struct.Struct("<hh")
ANIMATION_HEADER_SIZE = _ANIMATION_HEADER.size  # size + count
assert ANIMATION_HEADER_SIZE == 4

_ANIMATION_DIRECTION = struct.Struct("<bbhhH")
ANIMATION_DIRECTION_HEADER_SIZE = _ANIMATION_DIRECTION.size
assert ANIMATION_DIRECTION_HEADER_SIZE == 8

ANIMATION_DIRECTION_COUNT = 32
ANIMATION_DIRECTION_OFFSETS_SIZE = ANIMATION_DIRECTION_COUNT * 2

# Empirical FNAF 1 gap: after the 62-byte ObjectCommon header and before
# Movements at offset 70. Tests assert every Active carries this exact
# eight-zero-byte gap. If it ever changes, it is a new field, not padding.
KNOWN_ZERO_PAD_START = OBJECT_COMMON_HEADER_SIZE
KNOWN_ZERO_PAD_END = 70
KNOWN_ZERO_PAD_BYTES = b"\x00" * (KNOWN_ZERO_PAD_END - KNOWN_ZERO_PAD_START)

_TABLE_FIELDS: tuple[tuple[str, str], ...] = (
    ("movements", "movements_offset"),
    ("animations", "animations_offset"),
    ("extension", "extension_offset"),
    ("counter", "counter_offset"),
    ("system_object", "system_object_offset"),
    ("values", "values_offset"),
    ("strings", "strings_offset"),
    ("fade_in", "fade_in_offset"),
    ("fade_out", "fade_out_offset"),
)


class ObjectCommonDecodeError(ValueError):
    """ObjectCommon decode failure with offset / table context."""


# --- Helpers ------------------------------------------------------------


def _bits_set(value: int, names: tuple[str, ...]) -> tuple[str, ...]:
    """Return human-readable names for every set bit covered by ``names``."""
    return tuple(name for i, name in enumerate(names) if (value >> i) & 1)


def animation_name(index: int) -> str:
    """Clickteam's fixed names for ids 0..15; user-defined afterwards."""
    if 0 <= index < len(ANIMATION_NAMES):
        return ANIMATION_NAMES[index]
    return f"User defined {index - 12 + 1}"


def _ensure_range(
    *,
    offset: int,
    size: int,
    limit: int,
    context: str,
) -> None:
    """Validate ``[offset, offset + size)`` lies inside ``[0, limit]``."""
    if offset < 0 or size < 0 or offset + size > limit:
        raise ObjectCommonDecodeError(
            f"ObjectCommon: {context} needs byte range "
            f"[{offset}..{offset + size}) but payload length is {limit}. "
            "Antibody #2 byte-count / offset bounds."
        )


# --- Dataclasses --------------------------------------------------------


@dataclass(frozen=True)
class CoverageSpan:
    """Byte range consumed by a decoded or explicitly opaque field."""

    start: int
    end: int
    label: str

    @property
    def size(self) -> int:
        return self.end - self.start

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "size": self.size, "label": self.label}


@dataclass(frozen=True)
class CoverageGap:
    """Unconsumed byte range after coverage reconciliation."""

    start: int
    end: int
    data: bytes = field(repr=False)

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def is_known_zero_pad(self) -> bool:
        return (
            self.start == KNOWN_ZERO_PAD_START
            and self.end == KNOWN_ZERO_PAD_END
            and self.data == KNOWN_ZERO_PAD_BYTES
        )

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "size": self.size,
            "hex": self.data.hex(),
            "known_zero_pad": self.is_known_zero_pad,
        }


@dataclass(frozen=True)
class OpaqueTableSpan:
    """Non-animation table we have not semantically decoded yet."""

    table: str
    start: int
    end: int
    data: bytes = field(repr=False)

    @property
    def size(self) -> int:
        return self.end - self.start

    def as_dict(self) -> dict:
        return {
            "table": self.table,
            "start": self.start,
            "end": self.end,
            "size": self.size,
            "decoded": False,
        }


@dataclass(frozen=True)
class ObjectCommonHeader:
    """Fixed 62-byte ObjectCommon header (FNAF 1 build-284 NORMAL branch)."""

    size: int
    animations_offset: int
    movements_offset: int
    version: int
    extension_offset: int
    counter_offset: int
    flags: int
    creation_flags: int
    qualifiers: tuple[int, ...]
    system_object_offset: int
    values_offset: int
    strings_offset: int
    new_flags: int
    preferences: int
    identifier: str
    back_color_bgra: int
    fade_in_offset: int
    fade_out_offset: int

    @property
    def flag_names(self) -> tuple[str, ...]:
        return _bits_set(self.flags, OBJECT_FLAG_NAMES)

    @property
    def new_flag_names(self) -> tuple[str, ...]:
        return _bits_set(self.new_flags, NEW_OBJECT_FLAG_NAMES)

    @property
    def preference_names(self) -> tuple[str, ...]:
        return _bits_set(self.preferences, OBJECT_PREFERENCE_NAMES)

    @property
    def nonzero_offsets(self) -> dict[str, int]:
        return {
            name: int(getattr(self, attr))
            for name, attr in _TABLE_FIELDS
            if int(getattr(self, attr)) != 0
        }

    def as_dict(self) -> dict:
        return {
            "size": self.size,
            "animations_offset": self.animations_offset,
            "movements_offset": self.movements_offset,
            "version": self.version,
            "extension_offset": self.extension_offset,
            "counter_offset": self.counter_offset,
            "flags": self.flags,
            "flag_names": list(self.flag_names),
            "creation_flags": self.creation_flags,
            "qualifiers": list(self.qualifiers),
            "system_object_offset": self.system_object_offset,
            "values_offset": self.values_offset,
            "strings_offset": self.strings_offset,
            "new_flags": self.new_flags,
            "new_flag_names": list(self.new_flag_names),
            "preferences": self.preferences,
            "preference_names": list(self.preference_names),
            "identifier": self.identifier,
            "back_color_bgra": self.back_color_bgra,
            "fade_in_offset": self.fade_in_offset,
            "fade_out_offset": self.fade_out_offset,
            "nonzero_offsets": self.nonzero_offsets,
        }


@dataclass(frozen=True)
class AnimationDirection:
    """One loaded direction inside an Animation table."""

    direction_index: int
    min_speed: int
    max_speed: int
    repeat: int
    back_to: int
    frame_count: int
    image_handles: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "direction_index": self.direction_index,
            "min_speed": self.min_speed,
            "max_speed": self.max_speed,
            "repeat": self.repeat,
            "back_to": self.back_to,
            "frame_count": self.frame_count,
            "image_handles": list(self.image_handles),
        }


@dataclass(frozen=True)
class Animation:
    """One animation id and its explicitly stored directions."""

    animation_index: int
    animation_name: str
    direction_offsets: tuple[int, ...]
    directions: tuple[AnimationDirection, ...]

    @property
    def has_single_speed_semantics(self) -> bool:
        return self.animation_index in HAS_SINGLE_SPEED_ANIMATION_IDS

    @property
    def frame_count(self) -> int:
        return sum(direction.frame_count for direction in self.directions)

    @property
    def image_handles(self) -> tuple[int, ...]:
        return tuple(
            handle
            for direction in self.directions
            for handle in direction.image_handles
        )

    def as_dict(self) -> dict:
        return {
            "animation_index": self.animation_index,
            "animation_name": self.animation_name,
            "has_single_speed_semantics": self.has_single_speed_semantics,
            "direction_offsets": list(self.direction_offsets),
            "directions": [direction.as_dict() for direction in self.directions],
        }


@dataclass(frozen=True)
class AnimationsBlock:
    """ObjectCommon AnimationHeader block."""

    start: int
    block_size: int
    count: int
    animation_offsets: tuple[int, ...]
    animations: tuple[Animation, ...]

    @property
    def end(self) -> int:
        return self.start + self.block_size

    @property
    def non_empty_count(self) -> int:
        return sum(1 for animation in self.animations if animation.directions)

    @property
    def total_directions(self) -> int:
        return sum(len(animation.directions) for animation in self.animations)

    @property
    def total_frames(self) -> int:
        return sum(animation.frame_count for animation in self.animations)

    @property
    def image_handles(self) -> tuple[int, ...]:
        return tuple(
            handle
            for animation in self.animations
            for handle in animation.image_handles
        )

    @property
    def unique_image_handles(self) -> frozenset[int]:
        return frozenset(self.image_handles)

    def summary_dict(self) -> dict:
        return {
            "total_animations": self.count,
            "non_empty_animations": self.non_empty_count,
            "total_directions": self.total_directions,
            "total_frames": self.total_frames,
            "unique_image_handles": sorted(self.unique_image_handles),
        }

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "block_size": self.block_size,
            "count": self.count,
            "animation_offsets": list(self.animation_offsets),
            "summary": self.summary_dict(),
            "animations": [animation.as_dict() for animation in self.animations],
        }


@dataclass(frozen=True)
class ObjectCommon:
    """Decoded Active ObjectCommon body."""

    header: ObjectCommonHeader
    animations: AnimationsBlock | None
    opaque_tables: tuple[OpaqueTableSpan, ...]
    coverage_spans: tuple[CoverageSpan, ...]
    coverage_gaps: tuple[CoverageGap, ...]

    @property
    def movements_raw(self) -> bytes:
        for span in self.opaque_tables:
            if span.table == "movements":
                return span.data
        return b""

    @property
    def zero_pad_gap(self) -> CoverageGap | None:
        for gap in self.coverage_gaps:
            if gap.is_known_zero_pad:
                return gap
        return None

    @property
    def image_handles(self) -> frozenset[int]:
        if self.animations is None:
            return frozenset()
        return self.animations.unique_image_handles

    @property
    def summary(self) -> dict:
        animation_summary = (
            self.animations.summary_dict()
            if self.animations is not None
            else {
                "total_animations": 0,
                "non_empty_animations": 0,
                "total_directions": 0,
                "total_frames": 0,
                "unique_image_handles": [],
            }
        )
        return {
            **animation_summary,
            "movements_raw_len": len(self.movements_raw),
            "movements_decoded": False,
            "known_zero_pad_gaps": sum(1 for gap in self.coverage_gaps if gap.is_known_zero_pad),
            "unconsumed_gap_count": len(self.coverage_gaps),
            "opaque_tables": {span.table: span.size for span in self.opaque_tables},
        }

    def as_dict(self) -> dict:
        return {
            "header": self.header.as_dict(),
            "movements_raw_len": len(self.movements_raw),
            "movements_decoded": False,
            "animations": self.animations.as_dict() if self.animations else None,
            "opaque_tables": [span.as_dict() for span in self.opaque_tables],
            "coverage": {
                "spans": [span.as_dict() for span in self.coverage_spans],
                "gaps": [gap.as_dict() for gap in self.coverage_gaps],
            },
            "summary": self.summary,
        }


# --- Decoder internals --------------------------------------------------


def _decode_header(payload: bytes) -> ObjectCommonHeader:
    _ensure_range(
        offset=0,
        size=OBJECT_COMMON_HEADER_SIZE,
        limit=len(payload),
        context="fixed header",
    )
    fields = _OBJECT_COMMON_HEADER.unpack_from(payload, 0)
    (
        size,
        animations_offset,
        movements_offset,
        version,
        extension_offset,
        counter_offset,
        flags,
        creation_flags,
        q0,
        q1,
        q2,
        q3,
        q4,
        q5,
        q6,
        q7,
        system_object_offset,
        values_offset,
        strings_offset,
        new_flags,
        preferences,
        identifier_raw,
        back_color_bgra,
        fade_in_offset,
        fade_out_offset,
    ) = fields

    if size != len(payload):
        raise ObjectCommonDecodeError(
            f"ObjectCommon: header.size={size} but payload length is "
            f"{len(payload)}. Antibody #2 byte-count."
        )

    try:
        identifier = identifier_raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ObjectCommonDecodeError(
            f"ObjectCommon: identifier bytes {identifier_raw!r} are not ASCII. "
            "Antibody #1 strict-unknown: likely misaligned header."
        ) from exc

    return ObjectCommonHeader(
        size=size,
        animations_offset=animations_offset,
        movements_offset=movements_offset,
        version=version,
        extension_offset=extension_offset,
        counter_offset=counter_offset,
        flags=flags,
        creation_flags=creation_flags,
        qualifiers=(q0, q1, q2, q3, q4, q5, q6, q7),
        system_object_offset=system_object_offset,
        values_offset=values_offset,
        strings_offset=strings_offset,
        new_flags=new_flags,
        preferences=preferences,
        identifier=identifier,
        back_color_bgra=back_color_bgra,
        fade_in_offset=fade_in_offset,
        fade_out_offset=fade_out_offset,
    )


def _validate_table_offsets(header: ObjectCommonHeader, payload_len: int) -> None:
    seen: dict[int, str] = {}
    for name, offset in header.nonzero_offsets.items():
        if offset < OBJECT_COMMON_HEADER_SIZE or offset > payload_len:
            raise ObjectCommonDecodeError(
                f"ObjectCommon: {name}_offset={offset} is outside the "
                f"payload body (valid non-zero offsets are "
                f"[{OBJECT_COMMON_HEADER_SIZE}, {payload_len}]). "
                "Antibody #2 byte-count / offset bounds."
            )
        prior = seen.get(offset)
        if prior is not None:
            raise ObjectCommonDecodeError(
                f"ObjectCommon: {name}_offset and {prior}_offset both point "
                f"to {offset}. This decoder requires distinct table starts "
                "so byte coverage is unambiguous."
            )
        seen[offset] = name


def _next_table_offset(header: ObjectCommonHeader, table: str, payload_len: int) -> int:
    offsets = header.nonzero_offsets
    start = offsets[table]
    later = [offset for offset in offsets.values() if offset > start]
    return min(later) if later else payload_len


def _mark(span_list: list[CoverageSpan], start: int, end: int, label: str) -> None:
    span_list.append(CoverageSpan(start=start, end=end, label=label))


def _read_i16_array(payload: bytes, *, offset: int, count: int, context: str) -> tuple[int, ...]:
    _ensure_range(offset=offset, size=2 * count, limit=len(payload), context=context)
    return struct.unpack_from(f"<{count}h", payload, offset)


def _decode_animations(
    payload: bytes,
    *,
    base_offset: int,
    coverage_spans: list[CoverageSpan],
) -> AnimationsBlock:
    _ensure_range(
        offset=base_offset,
        size=ANIMATION_HEADER_SIZE,
        limit=len(payload),
        context="AnimationHeader.size+count",
    )
    block_size, count = _ANIMATION_HEADER.unpack_from(payload, base_offset)
    if block_size < ANIMATION_HEADER_SIZE:
        raise ObjectCommonDecodeError(
            f"ObjectCommon: AnimationHeader at offset {base_offset} has "
            f"block_size={block_size}, smaller than the fixed "
            f"{ANIMATION_HEADER_SIZE}-byte header."
        )
    if count < 0:
        raise ObjectCommonDecodeError(
            f"ObjectCommon: AnimationHeader at offset {base_offset} has "
            f"negative count={count}. Antibody #1 strict-unknown."
        )
    block_end = base_offset + block_size
    _ensure_range(
        offset=base_offset,
        size=block_size,
        limit=len(payload),
        context=f"AnimationHeader block_size={block_size}",
    )

    _mark(coverage_spans, base_offset, base_offset + ANIMATION_HEADER_SIZE, "AnimationHeader.size+count")

    offsets_start = base_offset + ANIMATION_HEADER_SIZE
    offsets_size = count * 2
    if ANIMATION_HEADER_SIZE + offsets_size > block_size:
        raise ObjectCommonDecodeError(
            f"ObjectCommon: AnimationHeader count={count} needs "
            f"{offsets_size} offset bytes, but block_size={block_size}. "
            "Antibody #2 byte-count."
        )
    animation_offsets = _read_i16_array(
        payload,
        offset=offsets_start,
        count=count,
        context=f"AnimationHeader.offsets[{count}]",
    )
    _mark(
        coverage_spans,
        offsets_start,
        offsets_start + offsets_size,
        f"AnimationHeader.offsets[{count}]",
    )

    animations: list[Animation] = []
    min_animation_payload_offset = ANIMATION_HEADER_SIZE + offsets_size
    for animation_index, animation_offset in enumerate(animation_offsets):
        if animation_offset == 0:
            animations.append(
                Animation(
                    animation_index=animation_index,
                    animation_name=animation_name(animation_index),
                    direction_offsets=(),
                    directions=(),
                )
            )
            continue
        if animation_offset < min_animation_payload_offset or animation_offset >= block_size:
            raise ObjectCommonDecodeError(
                f"ObjectCommon: AnimationHeader offset for animation "
                f"{animation_index} is {animation_offset}, outside the "
                f"payload area [{min_animation_payload_offset}, {block_size})."
            )

        animation_pos = base_offset + animation_offset
        _ensure_range(
            offset=animation_pos,
            size=ANIMATION_DIRECTION_OFFSETS_SIZE,
            limit=block_end,
            context=f"Animation[{animation_index}].direction_offsets",
        )
        direction_offsets = _read_i16_array(
            payload,
            offset=animation_pos,
            count=ANIMATION_DIRECTION_COUNT,
            context=f"Animation[{animation_index}].direction_offsets",
        )
        _mark(
            coverage_spans,
            animation_pos,
            animation_pos + ANIMATION_DIRECTION_OFFSETS_SIZE,
            f"Animation[{animation_index}].direction_offsets",
        )

        directions: list[AnimationDirection] = []
        for direction_index, direction_offset in enumerate(direction_offsets):
            if direction_offset == 0:
                continue
            if direction_offset < ANIMATION_DIRECTION_OFFSETS_SIZE:
                raise ObjectCommonDecodeError(
                    f"ObjectCommon: Animation[{animation_index}].dir[{direction_index}] "
                    f"offset {direction_offset} points into the 32-entry "
                    "direction-offset table. Antibody #2 byte-count."
                )
            direction_pos = animation_pos + direction_offset
            _ensure_range(
                offset=direction_pos,
                size=ANIMATION_DIRECTION_HEADER_SIZE,
                limit=block_end,
                context=f"Animation[{animation_index}].dir[{direction_index}].header",
            )
            min_speed, max_speed, repeat, back_to, frame_count = _ANIMATION_DIRECTION.unpack_from(
                payload, direction_pos
            )
            _mark(
                coverage_spans,
                direction_pos,
                direction_pos + ANIMATION_DIRECTION_HEADER_SIZE,
                f"Animation[{animation_index}].dir[{direction_index}].header",
            )

            frames_pos = direction_pos + ANIMATION_DIRECTION_HEADER_SIZE
            frames_size = 2 * frame_count
            _ensure_range(
                offset=frames_pos,
                size=frames_size,
                limit=block_end,
                context=f"Animation[{animation_index}].dir[{direction_index}].frames[{frame_count}]",
            )
            image_handles = _read_i16_array(
                payload,
                offset=frames_pos,
                count=frame_count,
                context=f"Animation[{animation_index}].dir[{direction_index}].frames",
            )
            _mark(
                coverage_spans,
                frames_pos,
                frames_pos + frames_size,
                f"Animation[{animation_index}].dir[{direction_index}].frames[{frame_count}]",
            )
            directions.append(
                AnimationDirection(
                    direction_index=direction_index,
                    min_speed=min_speed,
                    max_speed=max_speed,
                    repeat=repeat,
                    back_to=back_to,
                    frame_count=frame_count,
                    image_handles=tuple(image_handles),
                )
            )

        animations.append(
            Animation(
                animation_index=animation_index,
                animation_name=animation_name(animation_index),
                direction_offsets=tuple(direction_offsets),
                directions=tuple(directions),
            )
        )

    return AnimationsBlock(
        start=base_offset,
        block_size=block_size,
        count=count,
        animation_offsets=tuple(animation_offsets),
        animations=tuple(animations),
    )


def _opaque_table_spans(
    payload: bytes,
    header: ObjectCommonHeader,
    *,
    skip_tables: Iterable[str],
    coverage_spans: list[CoverageSpan],
) -> tuple[OpaqueTableSpan, ...]:
    skip = set(skip_tables)
    spans: list[OpaqueTableSpan] = []
    for table, start in sorted(header.nonzero_offsets.items(), key=lambda item: item[1]):
        if table in skip:
            continue
        end = _next_table_offset(header, table, len(payload))
        _ensure_range(
            offset=start,
            size=end - start,
            limit=len(payload),
            context=f"{table} opaque span",
        )
        data = bytes(payload[start:end])
        spans.append(OpaqueTableSpan(table=table, start=start, end=end, data=data))
        _mark(coverage_spans, start, end, f"{table} (opaque)")
    return tuple(spans)


def _coverage_gaps(payload: bytes, spans: Iterable[CoverageSpan]) -> tuple[CoverageGap, ...]:
    gaps: list[CoverageGap] = []
    prev_end = 0
    for span in sorted(spans, key=lambda item: (item.start, item.end, item.label)):
        if not (0 <= span.start <= span.end <= len(payload)):
            raise ObjectCommonDecodeError(
                f"ObjectCommon: invalid coverage span {span.label} "
                f"[{span.start}..{span.end}) for payload length {len(payload)}."
            )
        if span.start < prev_end:
            raise ObjectCommonDecodeError(
                f"ObjectCommon: coverage overlap before {span.label} "
                f"[{span.start}..{span.end}); previous end was {prev_end}. "
                "Likely duplicate offsets or overlapping animation records."
            )
        if span.start > prev_end:
            gaps.append(
                CoverageGap(
                    start=prev_end,
                    end=span.start,
                    data=bytes(payload[prev_end:span.start]),
                )
            )
        prev_end = span.end
    if prev_end < len(payload):
        gaps.append(CoverageGap(start=prev_end, end=len(payload), data=bytes(payload[prev_end:])))
    return tuple(gaps)


def _validate_only_known_gaps(gaps: tuple[CoverageGap, ...]) -> None:
    bad = [gap for gap in gaps if not gap.is_known_zero_pad]
    if bad:
        rendered = ", ".join(
            f"[{gap.start}..{gap.end}) {gap.size} B hex={gap.data.hex()}"
            for gap in bad
        )
        raise ObjectCommonDecodeError(
            f"ObjectCommon: unconsumed coverage gaps outside the known "
            f"[62..70) zero pad: {rendered}. Antibody #2 byte-count / "
            "loud skip — this is probably a real field."
        )


# --- Public decoder -----------------------------------------------------


def decode_object_common(payload: bytes) -> ObjectCommon:
    """Decode an Active ObjectCommon body.

    Parameters
    ----------
    payload:
        Plaintext bytes from an ObjectInfo ``0x4446 Properties`` sub-chunk.

    Returns
    -------
    ObjectCommon
        Header + decoded animation table + explicit opaque table spans.

    Raises
    ------
    ObjectCommonDecodeError
        On header drift, out-of-range offsets, overlapping coverage, or
        any unconsumed gap other than FNAF 1's documented eight-zero-byte
        pad at ``[62..70)``.
    """
    header = _decode_header(payload)
    _validate_table_offsets(header, len(payload))

    coverage_spans: list[CoverageSpan] = []
    _mark(coverage_spans, 0, OBJECT_COMMON_HEADER_SIZE, "ObjectCommon.header")

    animations: AnimationsBlock | None = None
    if header.animations_offset != 0:
        animations = _decode_animations(
            payload,
            base_offset=header.animations_offset,
            coverage_spans=coverage_spans,
        )

    opaque_tables = _opaque_table_spans(
        payload,
        header,
        skip_tables={"animations"},
        coverage_spans=coverage_spans,
    )

    gaps = _coverage_gaps(payload, coverage_spans)
    _validate_only_known_gaps(gaps)

    return ObjectCommon(
        header=header,
        animations=animations,
        opaque_tables=opaque_tables,
        coverage_spans=tuple(sorted(coverage_spans, key=lambda item: (item.start, item.end, item.label))),
        coverage_gaps=gaps,
    )
