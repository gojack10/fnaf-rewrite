"""Counter body decoder for ObjectInfo Properties (0x4446) bodies whose
``object_type`` is ``OBJECT_TYPE_COUNTER`` (=7).

Counters in FNAF 1 are the named scalar globals that drive game state
('night number', 'freddy AI', 'foxy AI', 'bonnie activity', 'power left',
'click cooldown', 'time of day', etc.). Counter is a separate
``object_type=7``; a counter is one whole object, NOT a slot on Active.

Wire format (FNAF 1 build-284, post decompress)
------------------------------------------------

    offset  size  field
    ------  ----  -----
       0     4    u32  size  (== len(payload); 180 or 162 in FNAF 1)
       4   114    bytes  header_raw  (mostly constant; only byte 42 ∈ {4,12} varies)
     118     4    u32  display_style  (∈ {0,1,3,10,12} in FNAF 1)
     122    28    bytes  display_config_raw  (font / color / digit-style; opaque)
     150     2    u16  image_handle_count
     152  2*N    u16[N]  image_handles  (resolve against ImageBank.handles)

Empirical FNAF 1 inventory (pinned in tests):

* 44 Counters total. 43 baseline 180-byte bodies + 1 outlier 162-byte
  body ('usage meter', handle 108) which carries fewer image handles
  (5 instead of 14) and a different ``display_config_raw``.
* ``display_style`` histogram: {0: 29, 1: 8, 3: 4, 10: 2, 12: 1}.
* 201 distinct image handles referenced across all 44 counters; every
  one resolves into the 605-record ImageBank handle set (sparse handles,
  not ``handle < image_count``).

Scope cut
---------

V0 keeps ``header_raw`` and ``display_config_raw`` as explicit opaque
bytes. The runtime only consumes ``display_style`` + ``image_handles``
(it renders the digit / sprite display and treats the counter value as
a runtime-defaulted scalar — see ObjectCommon antibody: counter init
values are NOT in this body, the runtime defaults to 0 and events run
explicit ``SetCounterValue`` actions).

Antibodies
----------

* ``size`` field at offset 0 must equal ``len(payload)``.
* Handle list slice ``[152, size)`` must be exactly ``2 * count`` bytes
  (no trailing junk, no underflow).
* ``display_style`` is decoded as u32 but logged as ``int`` — the FNAF 1
  set of distinct values is ``{0, 1, 3, 10, 12}``; tests pin the histogram.
* Image handles validate against the ImageBank handle set at the
  integration-test layer (caller has the bank).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# --- Wire-format constants ---------------------------------------------

#: u32 size, present at offset 0 of every Counter body.
COUNTER_SIZE_FIELD_BYTES = 4

#: Bytes [4..118) — mostly-constant template header. Tests assert it is
#: byte-identical across all 43 baseline 180-byte counters (modulo byte 42,
#: which ∈ {4, 12}). Kept opaque in V0; runtime does not consume it.
COUNTER_HEADER_RAW_START = 4
COUNTER_HEADER_RAW_END = 118
COUNTER_HEADER_RAW_SIZE = COUNTER_HEADER_RAW_END - COUNTER_HEADER_RAW_START
assert COUNTER_HEADER_RAW_SIZE == 114

#: u32 ``display_style`` enum (FNAF 1 values: {0, 1, 3, 10, 12}).
COUNTER_DISPLAY_STYLE_OFFSET = 118
COUNTER_DISPLAY_STYLE_BYTES = 4

#: Bytes [122..150) — opaque display-config block (font / color / digit
#: style). 28 bytes; baseline-stable across 43/44 counters; usage meter
#: (handle 108) is the only one whose contents differ. Kept opaque in V0.
COUNTER_DISPLAY_CONFIG_START = 122
COUNTER_DISPLAY_CONFIG_END = 150
COUNTER_DISPLAY_CONFIG_SIZE = (
    COUNTER_DISPLAY_CONFIG_END - COUNTER_DISPLAY_CONFIG_START
)
assert COUNTER_DISPLAY_CONFIG_SIZE == 28

#: u16 ``image_handle_count`` precedes the handle list.
COUNTER_HANDLE_COUNT_OFFSET = 150
COUNTER_HANDLE_LIST_OFFSET = 152

#: Smallest legal Counter body: empty handle list (count = 0) =>
#: 152 bytes total. Larger bodies hold 2 * count more bytes.
COUNTER_MIN_BODY_SIZE = COUNTER_HANDLE_LIST_OFFSET
assert COUNTER_MIN_BODY_SIZE == 152

_SIZE_STRUCT = struct.Struct("<I")
_DISPLAY_STYLE_STRUCT = struct.Struct("<I")
_HANDLE_COUNT_STRUCT = struct.Struct("<H")


class CounterBodyDecodeError(ValueError):
    """Counter body decode failure with offset / handle context."""


# --- Dataclasses --------------------------------------------------------


@dataclass(frozen=True)
class CounterBody:
    """Decoded Counter ObjectInfo property body.

    Carries the two runtime-consumable fields (``display_style``,
    ``image_handles``) plus the raw opaque spans for round-trip and
    future-probe support.
    """

    size: int
    header_raw: bytes = field(repr=False)
    display_style: int
    display_config_raw: bytes = field(repr=False)
    image_handles: tuple[int, ...]

    @property
    def image_handle_count(self) -> int:
        return len(self.image_handles)

    @property
    def unique_image_handles(self) -> frozenset[int]:
        return frozenset(self.image_handles)

    def summary_dict(self) -> dict:
        return {
            "size": self.size,
            "display_style": self.display_style,
            "image_handle_count": self.image_handle_count,
            "unique_image_handles": sorted(self.unique_image_handles),
        }

    def as_dict(self) -> dict:
        return {
            "size": self.size,
            "display_style": self.display_style,
            "image_handles": list(self.image_handles),
            "header_raw_len": len(self.header_raw),
            "display_config_raw_len": len(self.display_config_raw),
        }


# --- Decoder -----------------------------------------------------------


def decode_counter_body(payload: bytes) -> CounterBody:
    """Decode one 0x4446 Properties body where ``object_type == 7``.

    Antibodies enforced:

    * ``len(payload) >= 152`` (room for header + display-style + display-
      config + handle-count word).
    * ``size`` field at offset 0 equals ``len(payload)``.
    * Handle list slice ``[152, size)`` is exactly ``2 * count`` bytes.
    """
    n = len(payload)
    if n < COUNTER_MIN_BODY_SIZE:
        raise CounterBodyDecodeError(
            f"Counter body: payload is {n} bytes, smaller than the "
            f"{COUNTER_MIN_BODY_SIZE}-byte fixed prefix (header + style + "
            f"display-config + count word). Antibody #2 byte-count."
        )

    (size,) = _SIZE_STRUCT.unpack_from(payload, 0)
    if size != n:
        raise CounterBodyDecodeError(
            f"Counter body: size field at offset 0 is {size} but payload "
            f"length is {n}. Antibody #2 byte-count."
        )

    header_raw = bytes(payload[COUNTER_HEADER_RAW_START:COUNTER_HEADER_RAW_END])

    (display_style,) = _DISPLAY_STYLE_STRUCT.unpack_from(
        payload, COUNTER_DISPLAY_STYLE_OFFSET
    )

    display_config_raw = bytes(
        payload[COUNTER_DISPLAY_CONFIG_START:COUNTER_DISPLAY_CONFIG_END]
    )

    (handle_count,) = _HANDLE_COUNT_STRUCT.unpack_from(
        payload, COUNTER_HANDLE_COUNT_OFFSET
    )

    expected_tail = COUNTER_HANDLE_LIST_OFFSET + 2 * handle_count
    if expected_tail != n:
        raise CounterBodyDecodeError(
            f"Counter body: handle_count={handle_count} at offset "
            f"{COUNTER_HANDLE_COUNT_OFFSET} would require {expected_tail} "
            f"total bytes (152 + 2 * count) but payload length is {n}. "
            "Antibody #2 byte-count."
        )

    image_handles: tuple[int, ...]
    if handle_count == 0:
        image_handles = ()
    else:
        image_handles = struct.unpack_from(
            f"<{handle_count}H", payload, COUNTER_HANDLE_LIST_OFFSET
        )

    return CounterBody(
        size=size,
        header_raw=header_raw,
        display_style=display_style,
        display_config_raw=display_config_raw,
        image_handles=image_handles,
    )
