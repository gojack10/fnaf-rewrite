"""0x333D FrameEvents decoder (probe #4.12).

The runtime event graph: condition/action tuples grouped into
EventGroups, themselves wrapped in a 0x333D TLV envelope that uses
4-byte ASCII section tags:

    b"ER>>"  header: object/player counters + per-object-type condition
             counters + qualifier list
    b"ERes"  event count: an int32 size that both oracles read-and-
             discard (body is immediately followed by ERev)
    b"ERev"  event-group data: int32 size-in-bytes, then N EventGroups
             back-to-back until the region end is reached
    b"ERop"  extension / option data: int32 size-in-bytes, then that
             many bytes of body (content ignored at this probe scope)
    b"<<ER"  end marker, no body, terminates the envelope loop

Scope cut for this probe
------------------------

The Clickteam parameter loader dispatches 30+ distinct shapes
(ParamObject, Time, Short, IntParam, Sample, Create, Every,
KeyParameter, ExpressionParameter, Position, Shoot, Zone, Colour,
Filename, AlterableValue, Click, Program, Extension, Group,
GroupPointer, GlobalValue, StringParam, TwoShorts, MultipleVariables,
ChildEvent, ...). Decoding every one in one probe would make this
file an unreviewable monolith.

This probe therefore stops at the (condition|action) parameter
boundary: each parameter is captured as a `(code, data)` pair where
`data` is the raw opaque body. Every parameter carries its own
length prefix in the wire format, so the opaque-blob approach is
lossless - a follow-up probe #4.13+ will decode specific parameter
shapes as the game surface demands them.

Also deliberately NOT applied here: CTFAK's `Events.IdentifierCounter`
post-processing transform that mutates condition identifiers for the
-25/-41 MultipleVariables edge cases. That's a runtime-semantics
rewrite, not wire format. Raw pre-transform identifiers are
preserved verbatim.

Wire format (verified field-for-field across CTFAK2.0 Events.cs and
Anaconda mmfparser/data/chunkloaders/events.pyx)
------------------------------------------------

    ER>>  header region:
        int16   max_objects
        int16   max_object_info
        int16   num_players
        int16[17]  number_of_conditions (per object-type bucket)
        int16   qualifier_count
        Qualifier[qualifier_count]  (4 B each: uint16 object_info,
                                     int16 type)

    ERes  event count:
        int32   size   (read and discarded)

    ERev  event-group data:
        int32   region_size_bytes
        EventGroup[...]  until reader position >= region_start +
                         region_size_bytes

    ERop  extension / option data:
        int32   body_size_bytes
        u8[body_size_bytes]   body (ignored this probe)

    <<ER  end marker, no body

    EventGroup (length-prefixed, *negated* size):
        int16   negated_size    (abs(size) = total group bytes
                                 including this field)
        uint8   number_of_conditions
        uint8   number_of_actions
        uint16  flags           (GROUP_FLAGS BitDict)
        int16   line_padding    (only for build >= 284, skipped)
        int32   is_restricted   (build >= 284 uses wide form)
        int32   restrict_cpt
        Condition[number_of_conditions]
        Action[number_of_actions]
        (seek back to group_start + abs(size))

    Condition (length-prefixed, positive size):
        uint16  size    (total condition bytes including this field)
        int16   object_type
        int16   num
        uint16  object_info
        int16   object_info_list
        int8    flags           (sbyte)
        int8    other_flags     (sbyte)
        uint8   number_of_parameters
        uint8   def_type
        int16   identifier
        Parameter[number_of_parameters]
        (seek to condition_start + size)

    Action (length-prefixed, positive size):
        uint16  size
        int16   object_type
        int16   num
        uint16  object_info
        int16   object_info_list
        int8    flags
        int8    other_flags
        uint8   number_of_parameters
        uint8   def_type
        Parameter[number_of_parameters]
        (seek to action_start + size)

        NB: no `identifier` - that field belongs to Condition only.

    Parameter (length-prefixed, positive size):
        int16   size    (total parameter bytes including this field
                         and the code field)
        int16   code    (maps into the parameter-loader dispatch)
        u8[size - 4]  data (opaque at this probe scope)
        (seek to parameter_start + size)

Antibody coverage
-----------------

- #1 strict-unknown: every section tag must be in the known set
  {b"ER>>", b"ERes", b"ERev", b"ERop", b"<<ER"}; anything else raises
  `FrameEventsDecodeError` immediately. Parameter codes are NOT
  validated against a closed set (that's the follow-up probe's job)
  but are surfaced via `parameter_codes_seen` so the caller can
  enforce its own whitelist.
- #2 byte-count: every size field is authoritative. After reading
  all children of a (group|condition|action|parameter), the current
  position must equal `start + abs(size)`. Drift raises loudly with
  the offending record's start offset, claimed size, and actual
  bytes consumed.
- #3 round-trip: synthetic pack/unpack covers an empty envelope
  (header + ERes + empty ERev + <<ER), a single-group envelope with
  one condition and one action, and a record with a parameter whose
  opaque data is non-empty.
- #4 multi-oracle: CTFAK Events.cs and Anaconda events.pyx agree on
  every field order and size. One divergence noted: CTFAK treats
  ERop as a single int32 OptionFlags; Anaconda treats it as length-
  prefixed with an ignored body. We adopt Anaconda's shape (more
  general, supports ERop bodies longer than 4 bytes). For FNAF 1
  the distinction is moot because the observed ERop body length is
  consistent with either reading.
- #5 multi-input: every 0x333D sub-chunk across every FNAF 1 frame
  routes through this decoder; the FNAF 1 snapshot in the test
  suite pins total event-group / condition / action counts per
  frame.
- #6 loud-skip: the `parameter_codes_seen` set on `FrameEvents`
  surfaces every opaque parameter code encountered. Future probes
  can inspect it to decide which parameter-type decoder to write
  next.
- #7 snapshot: per-frame (num_groups, sum_conditions, sum_actions)
  triples pinned in tests; any RC4 drift or envelope-walker bug
  fires a regression at snapshot time.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# --- Section tags (4-byte ASCII) -----------------------------------------

TAG_HEADER = b"ER>>"
TAG_EVENT_COUNT = b"ERes"
TAG_EVENTGROUP_DATA = b"ERev"
TAG_EXTENSION_DATA = b"ERop"
TAG_END = b"<<ER"

KNOWN_TAGS: frozenset[bytes] = frozenset(
    {TAG_HEADER, TAG_EVENT_COUNT, TAG_EVENTGROUP_DATA, TAG_EXTENSION_DATA, TAG_END}
)

# --- Wire-format struct shapes ------------------------------------------

# ER>> fixed prelude: max_objects/max_object_info/num_players (3 * int16)
# + 17 * int16 number_of_conditions + 1 * int16 qualifier_count = 21 shorts.
_HEADER_PRELUDE = struct.Struct("<" + "h" * (3 + 17 + 1))
HEADER_PRELUDE_SIZE = _HEADER_PRELUDE.size  # 42
assert HEADER_PRELUDE_SIZE == 42

# Qualifier: uint16 object_info, int16 type.
_QUALIFIER = struct.Struct("<Hh")
QUALIFIER_SIZE = _QUALIFIER.size  # 4
assert QUALIFIER_SIZE == 4

# EventGroup header: size(int16 negated) + n_cond(u8) + n_act(u8) +
# flags(u16) + line_padding(int16) + is_restricted(int32) + restrict_cpt(int32).
# Total = 2 + 1 + 1 + 2 + 2 + 4 + 4 = 16 bytes for build >= 284.
_EVENT_GROUP_HEADER = struct.Struct("<hBBHhii")
EVENT_GROUP_HEADER_SIZE = _EVENT_GROUP_HEADER.size  # 16
assert EVENT_GROUP_HEADER_SIZE == 16

# Condition fixed fields after the 2-byte size prefix:
#   object_type(int16) + num(int16) + object_info(uint16) +
#   object_info_list(int16) + flags(int8) + other_flags(int8) +
#   number_of_parameters(uint8) + def_type(uint8) + identifier(int16)
# = 2*4 + 1*4 + 2 = 14 bytes, plus 2-byte size prefix = 16 bytes of header.
_CONDITION_FIXED = struct.Struct("<hhHhbbBBh")
CONDITION_FIXED_SIZE = _CONDITION_FIXED.size  # 14
assert CONDITION_FIXED_SIZE == 14

# Action is the same shape as Condition minus `identifier` = 12 bytes
# after the size prefix.
_ACTION_FIXED = struct.Struct("<hhHhbbBB")
ACTION_FIXED_SIZE = _ACTION_FIXED.size  # 12
assert ACTION_FIXED_SIZE == 12


class FrameEventsDecodeError(ValueError):
    """0x333D FrameEvents decode failure - carries tag / offset context."""


# --- Dataclasses --------------------------------------------------------


@dataclass(frozen=True)
class Qualifier:
    """One entry in the ER>> qualifier list.

    `object_info` is a 16-bit handle; the low 11 bits are the qualifier
    id per CTFAK's `Qualifier = ObjectInfo & 0b11111111111`, the high
    bits are flags. We surface the raw handle; callers that need the
    masked qualifier compute it themselves.
    """
    object_info: int
    type: int

    @property
    def qualifier(self) -> int:
        """Low 11 bits of `object_info` - matches CTFAK's derivation."""
        return self.object_info & 0b11111111111

    def as_dict(self) -> dict:
        return {
            "object_info": self.object_info,
            "type": self.type,
            "qualifier": self.qualifier,
        }


@dataclass(frozen=True)
class EventParameter:
    """One `(code, data)` parameter inside a Condition or Action.

    Opaque at this probe scope (see module docstring). `data` is the
    raw bytes between the code field and the next parameter boundary;
    `size` is the on-disk total including the 2-byte size field and
    the 2-byte code field (so `len(data) == size - 4`). The `code`
    maps into the 70+ parameter-loader dispatch table that probe
    #4.13+ will decode.
    """
    code: int
    size: int
    data: bytes = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "size": self.size,
            "data_hex": self.data.hex(),
        }


@dataclass(frozen=True)
class EventCondition:
    """One condition inside an EventGroup.

    `parameters` is a tuple of `EventParameter` opaque records; their
    codes surface in the containing `FrameEvents.parameter_codes_seen`
    set for follow-up probing.
    """
    size: int
    object_type: int
    num: int
    object_info: int
    object_info_list: int
    flags: int
    other_flags: int
    def_type: int
    identifier: int
    parameters: tuple[EventParameter, ...]

    def as_dict(self) -> dict:
        return {
            "size": self.size,
            "object_type": self.object_type,
            "num": self.num,
            "object_info": self.object_info,
            "object_info_list": self.object_info_list,
            "flags": self.flags,
            "other_flags": self.other_flags,
            "def_type": self.def_type,
            "identifier": self.identifier,
            "parameters": [p.as_dict() for p in self.parameters],
        }


@dataclass(frozen=True)
class EventAction:
    """One action inside an EventGroup.

    Shape mirrors `EventCondition` but lacks `identifier` (that field
    is Condition-only in the wire format).
    """
    size: int
    object_type: int
    num: int
    object_info: int
    object_info_list: int
    flags: int
    other_flags: int
    def_type: int
    parameters: tuple[EventParameter, ...]

    def as_dict(self) -> dict:
        return {
            "size": self.size,
            "object_type": self.object_type,
            "num": self.num,
            "object_info": self.object_info,
            "object_info_list": self.object_info_list,
            "flags": self.flags,
            "other_flags": self.other_flags,
            "def_type": self.def_type,
            "parameters": [p.as_dict() for p in self.parameters],
        }


@dataclass(frozen=True)
class EventGroup:
    """One event group inside an ERev region.

    `size` is the on-disk total in bytes (positive - we un-negate the
    wire value on read). `is_restricted` / `restrict_cpt` are the
    build>=284 wide fields. `flags` is the GROUP_FLAGS 16-bit bitfield.
    """
    size: int
    flags: int
    is_restricted: int
    restrict_cpt: int
    conditions: tuple[EventCondition, ...]
    actions: tuple[EventAction, ...]

    def as_dict(self) -> dict:
        return {
            "size": self.size,
            "flags": self.flags,
            "is_restricted": self.is_restricted,
            "restrict_cpt": self.restrict_cpt,
            "conditions": [c.as_dict() for c in self.conditions],
            "actions": [a.as_dict() for a in self.actions],
        }


@dataclass(frozen=True)
class FrameEvents:
    """One decoded 0x333D FrameEvents sub-chunk.

    Top-level accessor for the runtime event graph. `parameter_codes_seen`
    is the Antibody #6 loud-skip surface: every distinct parameter code
    observed in any condition or action, so the caller can tell which
    parameter-type decoders to prioritise in a follow-up probe.
    """
    max_objects: int
    max_object_info: int
    num_players: int
    number_of_conditions: tuple[int, ...]  # always 17 entries
    qualifiers: tuple[Qualifier, ...]
    event_groups: tuple[EventGroup, ...]
    extension_data: bytes = field(repr=False)
    parameter_codes_seen: frozenset[int]

    def as_dict(self) -> dict:
        return {
            "max_objects": self.max_objects,
            "max_object_info": self.max_object_info,
            "num_players": self.num_players,
            "number_of_conditions": list(self.number_of_conditions),
            "qualifiers": [q.as_dict() for q in self.qualifiers],
            "event_groups": [g.as_dict() for g in self.event_groups],
            "extension_data_len": len(self.extension_data),
            "parameter_codes_seen": sorted(self.parameter_codes_seen),
        }

    @property
    def total_conditions(self) -> int:
        return sum(len(g.conditions) for g in self.event_groups)

    @property
    def total_actions(self) -> int:
        return sum(len(g.actions) for g in self.event_groups)


# --- Decoder -------------------------------------------------------------


def _decode_parameter(
    payload: bytes, start: int
) -> tuple[EventParameter, int]:
    """Decode one Parameter record starting at `start`. Returns the
    record and the absolute end offset.

    Enforces byte-count reconcile: the 2-byte size prefix is the
    authoritative advance. `size` must be >= 4 (size field + code field)
    and must not run past `len(payload)`.
    """
    n = len(payload)
    if start + 2 > n:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: parameter at offset 0x{start:x} has "
            f"no room for a 2-byte size prefix (payload length {n})."
        )
    (size,) = struct.unpack_from("<h", payload, start)
    # Clickteam encodes size as positive int16 but the practical upper
    # bound is well under int16 max. A negative or zero value means RC4
    # drift or corruption.
    if size < 4:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: parameter at offset 0x{start:x} has "
            f"size={size}, below the 4-byte minimum (size field + code "
            f"field). Antibody #2 byte-count: likely RC4 drift."
        )
    end = start + size
    if end > n:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: parameter at offset 0x{start:x} claims "
            f"size={size} but only {n - start} bytes remain in the "
            f"payload (length {n}). Antibody #2 byte-count."
        )
    (code,) = struct.unpack_from("<h", payload, start + 2)
    data = bytes(payload[start + 4:end])
    return EventParameter(code=code, size=size, data=data), end


def _decode_condition(
    payload: bytes, start: int, codes_seen: set[int]
) -> tuple[EventCondition, int]:
    """Decode one Condition record starting at `start`. Returns the
    record and the absolute end offset. Populates `codes_seen` with
    every parameter code observed.
    """
    n = len(payload)
    if start + 2 + CONDITION_FIXED_SIZE > n:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: condition at offset 0x{start:x} has "
            f"no room for the {2 + CONDITION_FIXED_SIZE}-byte header "
            f"(payload length {n}). Antibody #2 byte-count."
        )
    (size,) = struct.unpack_from("<H", payload, start)
    if size < 2 + CONDITION_FIXED_SIZE:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: condition at offset 0x{start:x} has "
            f"size={size}, below the {2 + CONDITION_FIXED_SIZE}-byte "
            f"header minimum. Antibody #2 byte-count: likely RC4 drift."
        )
    end = start + size
    if end > n:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: condition at offset 0x{start:x} claims "
            f"size={size} but only {n - start} bytes remain. Antibody #2."
        )
    (
        object_type,
        num,
        object_info,
        object_info_list,
        flags,
        other_flags,
        number_of_parameters,
        def_type,
        identifier,
    ) = _CONDITION_FIXED.unpack_from(payload, start + 2)

    pos = start + 2 + CONDITION_FIXED_SIZE
    parameters: list[EventParameter] = []
    for _ in range(number_of_parameters):
        param, pos = _decode_parameter(payload, pos)
        codes_seen.add(param.code)
        parameters.append(param)

    if pos != end:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: condition at offset 0x{start:x} with "
            f"size={size} declared {number_of_parameters} parameters but "
            f"consumed {pos - start} bytes (expected {size}). Antibody "
            f"#2 byte-count: parameter walk drifted from size field."
        )

    return (
        EventCondition(
            size=size,
            object_type=object_type,
            num=num,
            object_info=object_info,
            object_info_list=object_info_list,
            flags=flags,
            other_flags=other_flags,
            def_type=def_type,
            identifier=identifier,
            parameters=tuple(parameters),
        ),
        end,
    )


def _decode_action(
    payload: bytes, start: int, codes_seen: set[int]
) -> tuple[EventAction, int]:
    """Decode one Action record starting at `start`. Same pattern as
    `_decode_condition` but 2 bytes shorter (no identifier).
    """
    n = len(payload)
    if start + 2 + ACTION_FIXED_SIZE > n:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: action at offset 0x{start:x} has no "
            f"room for the {2 + ACTION_FIXED_SIZE}-byte header (payload "
            f"length {n}). Antibody #2 byte-count."
        )
    (size,) = struct.unpack_from("<H", payload, start)
    if size < 2 + ACTION_FIXED_SIZE:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: action at offset 0x{start:x} has "
            f"size={size}, below the {2 + ACTION_FIXED_SIZE}-byte "
            f"header minimum. Antibody #2 byte-count: likely RC4 drift."
        )
    end = start + size
    if end > n:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: action at offset 0x{start:x} claims "
            f"size={size} but only {n - start} bytes remain. Antibody #2."
        )
    (
        object_type,
        num,
        object_info,
        object_info_list,
        flags,
        other_flags,
        number_of_parameters,
        def_type,
    ) = _ACTION_FIXED.unpack_from(payload, start + 2)

    pos = start + 2 + ACTION_FIXED_SIZE
    parameters: list[EventParameter] = []
    for _ in range(number_of_parameters):
        param, pos = _decode_parameter(payload, pos)
        codes_seen.add(param.code)
        parameters.append(param)

    if pos != end:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: action at offset 0x{start:x} with "
            f"size={size} declared {number_of_parameters} parameters but "
            f"consumed {pos - start} bytes (expected {size}). Antibody "
            f"#2 byte-count: parameter walk drifted from size field."
        )

    return (
        EventAction(
            size=size,
            object_type=object_type,
            num=num,
            object_info=object_info,
            object_info_list=object_info_list,
            flags=flags,
            other_flags=other_flags,
            def_type=def_type,
            parameters=tuple(parameters),
        ),
        end,
    )


def _decode_event_group(
    payload: bytes, start: int, codes_seen: set[int]
) -> tuple[EventGroup, int]:
    """Decode one EventGroup record starting at `start`. The wire size
    field is *negated* int16 - we un-negate it for the returned
    `EventGroup.size`. Returns the group and the absolute end offset.
    """
    n = len(payload)
    if start + EVENT_GROUP_HEADER_SIZE > n:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: event-group at offset 0x{start:x} has "
            f"no room for the {EVENT_GROUP_HEADER_SIZE}-byte header "
            f"(payload length {n}). Antibody #2 byte-count."
        )
    (
        negated_size,
        number_of_conditions,
        number_of_actions,
        flags,
        _line_padding,
        is_restricted,
        restrict_cpt,
    ) = _EVENT_GROUP_HEADER.unpack_from(payload, start)

    # Clickteam ships the group size negated. Flip it back to positive
    # for every downstream use. If the wire value is non-negative, that's
    # a schema violation (or RC4 drift) - raise.
    if negated_size >= 0:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: event-group at offset 0x{start:x} has "
            f"non-negative size field {negated_size}; wire format requires "
            f"negated int16. Antibody #1 strict-unknown: likely RC4 drift."
        )
    size = -negated_size
    if size < EVENT_GROUP_HEADER_SIZE:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: event-group at offset 0x{start:x} has "
            f"abs(size)={size} below the {EVENT_GROUP_HEADER_SIZE}-byte "
            f"header minimum. Antibody #2 byte-count."
        )
    end = start + size
    if end > n:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: event-group at offset 0x{start:x} "
            f"claims abs(size)={size} but only {n - start} bytes remain. "
            f"Antibody #2 byte-count."
        )

    pos = start + EVENT_GROUP_HEADER_SIZE
    conditions: list[EventCondition] = []
    for _ in range(number_of_conditions):
        cond, pos = _decode_condition(payload, pos, codes_seen)
        conditions.append(cond)

    actions: list[EventAction] = []
    for _ in range(number_of_actions):
        act, pos = _decode_action(payload, pos, codes_seen)
        actions.append(act)

    if pos != end:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: event-group at offset 0x{start:x} "
            f"with abs(size)={size} declared {number_of_conditions} "
            f"conditions + {number_of_actions} actions but consumed "
            f"{pos - start} bytes (expected {size}). Antibody #2 "
            f"byte-count: condition/action walk drifted from size field."
        )

    return (
        EventGroup(
            size=size,
            flags=flags,
            is_restricted=is_restricted,
            restrict_cpt=restrict_cpt,
            conditions=tuple(conditions),
            actions=tuple(actions),
        ),
        end,
    )


def decode_frame_events(payload: bytes) -> FrameEvents:
    """Decode a 0x333D FrameEvents sub-chunk's plaintext bytes.

    The caller is responsible for having already run the payload
    through the flag=3 decrypt+decompress path (probe #4.5). This
    function only sees the final plaintext. Each section tag is
    processed exactly once except b"<<ER" which terminates the walk.

    Antibodies enforced:

    - Every 4-byte section tag must be in `KNOWN_TAGS`. An unknown
      tag raises immediately with the tag bytes and the offset where
      it was read.
    - The ER>> header consumes exactly
      `HEADER_PRELUDE_SIZE + qualifier_count * QUALIFIER_SIZE` bytes;
      qualifier_count negative or oversized raises.
    - ERev declares a byte-region size; event groups fill the region
      exactly - neither over-run nor leftover bytes are accepted.
    - Each event group, condition, action, and parameter size field
      is the authoritative advance and must reconcile. See the
      `_decode_*` helpers for the per-record checks.
    """
    n = len(payload)
    pos = 0

    # Defaults so we can detect missing sections at the end (all five
    # tags must appear exactly once in a well-formed envelope, except
    # ERop which can be omitted - Anaconda's loader tolerates that).
    max_objects: int | None = None
    max_object_info: int | None = None
    num_players: int | None = None
    number_of_conditions: tuple[int, ...] | None = None
    qualifiers: tuple[Qualifier, ...] = ()
    event_groups: list[EventGroup] = []
    extension_data = b""
    codes_seen: set[int] = set()
    seen_tags: set[bytes] = set()

    while pos + 4 <= n:
        tag = bytes(payload[pos:pos + 4])
        pos += 4
        if tag not in KNOWN_TAGS:
            raise FrameEventsDecodeError(
                f"0x333D FrameEvents: unknown section tag {tag!r} at "
                f"payload offset 0x{pos - 4:x}. Antibody #1 strict-"
                f"unknown: likely RC4 drift or a schema version skew."
            )
        if tag in seen_tags and tag != TAG_END:
            raise FrameEventsDecodeError(
                f"0x333D FrameEvents: section tag {tag!r} appeared twice "
                f"(second occurrence at payload offset 0x{pos - 4:x}). "
                f"Antibody #1 strict-unknown."
            )
        seen_tags.add(tag)

        if tag == TAG_HEADER:
            if pos + HEADER_PRELUDE_SIZE > n:
                raise FrameEventsDecodeError(
                    f"0x333D FrameEvents: ER>> header at offset "
                    f"0x{pos - 4:x} has no room for the "
                    f"{HEADER_PRELUDE_SIZE}-byte prelude."
                )
            prelude = _HEADER_PRELUDE.unpack_from(payload, pos)
            pos += HEADER_PRELUDE_SIZE
            max_objects = prelude[0]
            max_object_info = prelude[1]
            num_players = prelude[2]
            number_of_conditions = tuple(prelude[3:3 + 17])
            qualifier_count = prelude[3 + 17]
            if qualifier_count < 0:
                raise FrameEventsDecodeError(
                    f"0x333D FrameEvents: ER>> qualifier_count is "
                    f"{qualifier_count} (negative). Antibody #1 strict-"
                    f"unknown: likely RC4 drift."
                )
            needed = qualifier_count * QUALIFIER_SIZE
            if pos + needed > n:
                raise FrameEventsDecodeError(
                    f"0x333D FrameEvents: ER>> claims {qualifier_count} "
                    f"qualifiers ({needed} bytes) but only {n - pos} "
                    f"bytes remain. Antibody #2 byte-count."
                )
            quals: list[Qualifier] = []
            for _ in range(qualifier_count):
                obj_info, qtype = _QUALIFIER.unpack_from(payload, pos)
                quals.append(Qualifier(object_info=obj_info, type=qtype))
                pos += QUALIFIER_SIZE
            qualifiers = tuple(quals)

        elif tag == TAG_EVENT_COUNT:
            if pos + 4 > n:
                raise FrameEventsDecodeError(
                    f"0x333D FrameEvents: ERes at offset 0x{pos - 4:x} "
                    f"has no room for its 4-byte size field."
                )
            # Read-and-discard per both oracles - the body is
            # implicitly part of the ERev region.
            pos += 4

        elif tag == TAG_EVENTGROUP_DATA:
            if pos + 4 > n:
                raise FrameEventsDecodeError(
                    f"0x333D FrameEvents: ERev at offset 0x{pos - 4:x} "
                    f"has no room for its 4-byte size field."
                )
            (region_size,) = struct.unpack_from("<i", payload, pos)
            pos += 4
            if region_size < 0:
                raise FrameEventsDecodeError(
                    f"0x333D FrameEvents: ERev region_size={region_size} "
                    f"is negative. Antibody #1 strict-unknown."
                )
            region_end = pos + region_size
            if region_end > n:
                raise FrameEventsDecodeError(
                    f"0x333D FrameEvents: ERev region_size={region_size} "
                    f"exceeds remaining payload ({n - pos} bytes). "
                    f"Antibody #2 byte-count."
                )
            while pos < region_end:
                group, pos = _decode_event_group(payload, pos, codes_seen)
                event_groups.append(group)
            if pos != region_end:
                raise FrameEventsDecodeError(
                    f"0x333D FrameEvents: ERev walk ended at offset "
                    f"0x{pos:x} but region ends at 0x{region_end:x}. "
                    f"Antibody #2 byte-count: event-group walk drifted."
                )

        elif tag == TAG_EXTENSION_DATA:
            if pos + 4 > n:
                raise FrameEventsDecodeError(
                    f"0x333D FrameEvents: ERop at offset 0x{pos - 4:x} "
                    f"has no room for its 4-byte size field."
                )
            (body_size,) = struct.unpack_from("<i", payload, pos)
            pos += 4
            if body_size < 0:
                raise FrameEventsDecodeError(
                    f"0x333D FrameEvents: ERop body_size={body_size} is "
                    f"negative. Antibody #1 strict-unknown."
                )
            if pos + body_size > n:
                raise FrameEventsDecodeError(
                    f"0x333D FrameEvents: ERop body_size={body_size} "
                    f"exceeds remaining payload ({n - pos} bytes). "
                    f"Antibody #2 byte-count."
                )
            extension_data = bytes(payload[pos:pos + body_size])
            pos += body_size

        elif tag == TAG_END:
            break

    if TAG_END not in seen_tags:
        raise FrameEventsDecodeError(
            f"0x333D FrameEvents: payload ended at offset 0x{pos:x} "
            f"without the b'<<ER' end marker. Antibody #1 strict-"
            f"unknown: envelope is truncated."
        )
    if TAG_HEADER not in seen_tags:
        raise FrameEventsDecodeError(
            "0x333D FrameEvents: envelope is missing the b'ER>>' header "
            "section. Antibody #1 strict-unknown."
        )
    # Type-narrowing - guaranteed non-None because TAG_HEADER was seen.
    assert max_objects is not None
    assert max_object_info is not None
    assert num_players is not None
    assert number_of_conditions is not None

    return FrameEvents(
        max_objects=max_objects,
        max_object_info=max_object_info,
        num_players=num_players,
        number_of_conditions=number_of_conditions,
        qualifiers=qualifiers,
        event_groups=tuple(event_groups),
        extension_data=extension_data,
        parameter_codes_seen=frozenset(codes_seen),
    )
