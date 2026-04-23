# Shapes ported from:
#   reference/Anaconda/mmfparser/data/chunkloaders/parameters/loaders.py
#   reference/Anaconda/mmfparser/data/chunkloaders/expressions/loader.pyx
#   reference/Anaconda/mmfparser/data/chunkloaders/expressions/loaders.pyx
# Drift-corrected against:
#   reference/CTFAK2.0/Core/CTFAK.Core/CCN/Chunks/Frame/Events/*.cs
# Anaconda is (c) Mathias Kaerlev 2012, GPLv3.
"""EventParameter payload decode (probe #4.13).

Probe #4.12 surfaced 15 distinct parameter `code` values inside FNAF 1's
event graph and left `data` as opaque bytes. This module closes that
decode — each of the 15 codes maps to a typed shape, including the
recursive Expression AST on codes 22, 23, 27, 45.

15 FNAF 1 codes:
    {1, 2, 6, 9, 10, 14, 16, 22, 23, 25, 26, 27, 32, 45, 50}

CTFAK2.0 padding contract (load-and-pad)
----------------------------------------
CTFAK2.0's `Parameter.Read` at `Events/Events.cs:525-540` is:

    currentPosition = reader.Tell()
    size            = reader.ReadInt16()
    Code            = reader.ReadInt16()
    Loader          = LoadParameter(Code, reader)
    Loader.Read(reader)                         # may under-consume
    reader.Seek(currentPosition + size)         # <-- pad to full size

Loaders are allowed to read fewer bytes than `size` carries. The
wrapper unconditionally advances to the end. Empirically every FNAF 1
parameter code carries trailing padding past its documented loader
length (see per-code histogram in probe #4.13 crystallization). We
mirror CTFAK2.0's contract: each fixed-width loader reads its N
bytes from the start of `data` and surfaces any `data[N:]` as
`trailing_hex` — a loud-but-non-fatal field that lets tests pin
unexpected drift without breaking FNAF 1 decode.

The same contract applies inside the Expression AST (Expression.Read
also does `reader.Seek(currentPosition + size)` — `Events/Expression.cs:77`).
ExpressionParameter bodies may likewise have trailing bytes past the
END marker (the outer Parameter wrapper pads).

CTFAK2.0 vs Anaconda drift
--------------------------
CTFAK2.0 was authored against MMF 2.5; Anaconda predates MMF 2.5.
Where they disagree we follow CTFAK2.0 for on-disk shapes:

- Time: CTFAK2.0 = 10 bytes (timer int32 + loops int32 + comparison
  int16). Anaconda = 8 bytes (timer + loops). FNAF 1 has 10-byte
  Time bodies.
- Create: CTFAK2.0 = 26 bytes (Position + ObjectInstance + ObjectInfo).
  Anaconda = 30 bytes (adds `skipBytes(4)`). FNAF 1 Create bodies
  are 32 bytes — the first 26 match both decoders, the remaining 6
  surface as `trailing_hex`.
- GlobalCommon (num=24/50 expression body): CTFAK2.0 = 8 bytes
  (skip int32 + value int32). Anaconda = 6 bytes (skip 4 + int16).
  We follow CTFAK2.0. Any additional trailing surfaces.

Code 50 is `AlterableValue`, a subclass of `Short` in CTFAK2.0
(`Events/AlterableValue.cs`). Same 2-byte on-disk layout; we emit
`kind="Short"` for uniformity with codes 10 and 26 — the
per-parameter semantic difference (`AlterableValue` vs plain `Short`)
is a name-resolver concern.

Non-goals at this probe scope
-----------------------------
- Name resolution is iteration 4's job. This decoder emits
  `{code, object_type, num, ...payload}` so the Name Resolver can
  look up `(object_type, num)` in EXPRESSION_SYSTEM_NAMES /
  EXPRESSION_EXTENSION_NAMES downstream.
- Evaluating the Expression AST. We parse it into a tree; execution
  is the Rust rebuild's problem.

Failure contract
----------------
- Loud failure on any code outside the 15-code set (matches Name
  Resolver's loud-unknown contract).
- Loud failure on insufficient bytes for the documented loader shape
  (`data` shorter than the loader needs).
- Loud failure on missing string terminator.
- Loud failure on missing Expression END marker.
- Loud failure on Expression size field inside the body claiming
  more bytes than the body contains, or less than the 6-byte header
  minimum.
- Trailing padding past a loader's consumed length is **tolerated**
  and surfaced as `trailing_hex` — matching CTFAK2.0's
  `reader.Seek(currentPosition + size)` pad-to-full-size idiom.
"""

from __future__ import annotations

import struct
from typing import Any

# 15 parameter codes that actually appear in FNAF 1 (pinned from probe #4.12
# `parameter_codes_seen`). Any other code is rejected loudly.
FNAF1_PARAMETER_CODES: frozenset[int] = frozenset(
    {1, 2, 6, 9, 10, 14, 16, 22, 23, 25, 26, 27, 32, 45, 50}
)


class EventParameterDecodeError(ValueError):
    """EventParameter payload decode failure.

    Raised with the offending code/offset and, where relevant, the slice
    hex so the failure site is unambiguous in test output.
    """


# --- Null-terminated string helper --------------------------------------


def _read_null_terminated(buf: bytes, *, unicode: bool) -> tuple[str, int]:
    """Decode one null-terminated string from the head of `buf`.

    Returns `(decoded_string, bytes_consumed_including_terminator)`.
    For unicode packs the terminator is a wide NUL (2 bytes, aligned).
    Raises EventParameterDecodeError if no terminator is found.
    """
    if unicode:
        # Wide-NUL scan on 2-byte boundaries. Accept len(buf) odd only
        # if we find the NUL before the last byte — a trailing half-
        # codepoint is drift and surfaces as "missing terminator".
        for i in range(0, len(buf) - 1, 2):
            if buf[i] == 0 and buf[i + 1] == 0:
                try:
                    return buf[:i].decode("utf-16-le"), i + 2
                except UnicodeDecodeError as exc:
                    raise EventParameterDecodeError(
                        f"UTF-16LE string body decode failed at byte {i}: {exc}"
                    ) from exc
        raise EventParameterDecodeError(
            f"UTF-16LE string missing NUL terminator in "
            f"{len(buf)}-byte body: {buf.hex()}"
        )
    idx = buf.find(b"\x00")
    if idx == -1:
        raise EventParameterDecodeError(
            f"ASCII string missing NUL terminator in "
            f"{len(buf)}-byte body: {buf.hex()}"
        )
    try:
        return buf[:idx].decode("ascii"), idx + 1
    except UnicodeDecodeError as exc:
        raise EventParameterDecodeError(
            f"ASCII string decode failed: {exc}"
        ) from exc


def _trailing(data: bytes, consumed: int) -> str:
    """Return `data[consumed:]` as hex for `trailing_hex` fields."""
    return data[consumed:].hex()


def _require_min(kind: str, data: bytes, needed: int) -> None:
    """Enforce that `data` is at least `needed` bytes for `kind`."""
    if len(data) < needed:
        raise EventParameterDecodeError(
            f"{kind}: expected >= {needed} bytes, got {len(data)} ({data.hex()})"
        )


# --- Fixed-width scalar loaders (codes 10, 25, 26, 50, 14) --------------


# Minimum trailing length at which we attempt UTF-16 LE label decode on
# a Short parameter. Empirically FNAF 1 SHORT bodies (codes 10 and 26)
# carry 32 bytes of trailing data containing an embedded animation-state
# label; AlterableValue (code 50) carries only 4 bytes of zero padding
# that must NOT trip the label decode path. The 16-byte floor sits
# comfortably between those two regimes and also guards against tiny
# test-synthetic trailing inputs ("aabbccdd", "dead") being misread as
# labels.
_SHORT_LABEL_MIN_TRAILING = 16


def _try_decode_utf16_label(data: bytes) -> str | None:
    """Decode `data` as UTF-16 LE up to the first wide-NUL terminator,
    or — when the label fills the whole buffer — as the entire body.

    Returns the decoded string iff it is non-empty ASCII-printable text
    (labels are English FNAF 1 game state names like ``"freddy attack"``,
    ``"power off"``, ``"animal cam"``, ``"door open"``, ``"closing"``,
    ``"2nd"``/``"3rd"``/...``"7th"`` night counter). Returns ``None`` on
    any of:

    - zero-padding only (no codepoint before the wide-NUL)
    - non-ASCII-printable codepoints (filters out garbage byte patterns
      that happen to be UTF-16 decodable — e.g. ``0xbbaa`` is a valid
      Unicode codepoint but not a game-state label)

    Truncated-label contract
    ------------------------
    Clickteam reserves a fixed 32-byte slot for the label but does not
    emit a wide-NUL when the label text fills it exactly (16 UTF-16 LE
    codepoints). Empirically 23/163 FNAF 1 SHORT labels hit this case
    (``"stage no chica o"`` truncating ``"stage no chica on cam"``). We
    fall back to decoding the full buffer when no NUL is found so those
    labels still surface verbatim.
    """
    if len(data) < 4:
        return None
    end = len(data) - (len(data) % 2)
    for i in range(0, end, 2):
        if data[i] == 0 and data[i + 1] == 0:
            if i < 2:
                return None
            try:
                s = data[:i].decode("utf-16-le")
            except UnicodeDecodeError:
                return None
            if s and all(32 <= ord(ch) < 127 for ch in s):
                return s
            return None
    # No NUL — buffer-filling truncated label. Decode the whole slot.
    try:
        s = data[:end].decode("utf-16-le")
    except UnicodeDecodeError:
        return None
    if s and all(32 <= ord(ch) < 127 for ch in s):
        return s
    return None


def _dec_short(data: bytes) -> dict[str, Any]:
    """Codes 10, 26, 50 — int16 value + trailing pad.

    For codes 10 and 26 (SHORT) Clickteam embeds a 32-byte UTF-16 LE
    animation-state label in the trailing pad. These labels are
    semantically critical for downstream invariant extraction — they
    carry the door states (``"door open"`` / ``"closing"``), animatronic
    positions (``"control room chi[ca]"``, ``"corner 1 bonnie"``,
    ``"hallway 2"``), jumpscare triggers (``"freddy attack"`` /
    ``"fox attack"``), power-state transitions (``"power off"``), night
    counter (``"2nd"`` ... ``"7th"``), and death labels (``"rip"``) that
    integer values alone don't convey. Empirically 163/163 FNAF 1 SHORT
    parameters expose a decodable label.

    Code 50 (AlterableValue) uses the same int16 loader but its trailing
    pad is 4 bytes of zeros — no label surfaces there. The length-based
    gate (`_SHORT_LABEL_MIN_TRAILING`) keeps test-synthetic trailing
    (2–4 bytes) out of the label path too.
    """
    _require_min("Short", data, 2)
    (value,) = struct.unpack_from("<h", data, 0)
    result: dict[str, Any] = {
        "kind": "Short",
        "value": value,
        "trailing_hex": _trailing(data, 2),
    }
    trailing = data[2:]
    if len(trailing) >= _SHORT_LABEL_MIN_TRAILING:
        label = _try_decode_utf16_label(trailing)
        if label is not None:
            result["label"] = label
    return result


def _dec_int(data: bytes) -> dict[str, Any]:
    """Code 25 — int32 value + trailing pad."""
    _require_min("Int", data, 4)
    (value,) = struct.unpack_from("<i", data, 0)
    return {"kind": "Int", "value": value, "trailing_hex": _trailing(data, 4)}


def _dec_key(data: bytes) -> dict[str, Any]:
    """Code 14 — uint16 virtual-key code + trailing pad.

    CTFAK2.0's `KeyParameter.Read` reads `ReadUInt16()`. Anaconda reads
    int16; the two agree for the standard VK_* range but we follow
    CTFAK2.0's unsigned shape.
    """
    _require_min("Key", data, 2)
    (value,) = struct.unpack_from("<H", data, 0)
    return {"kind": "Key", "value": value, "trailing_hex": _trailing(data, 2)}


# --- Code 1 Object ------------------------------------------------------


def _dec_object(data: bytes) -> dict[str, Any]:
    """Code 1 — int16 objectInfoList + uint16 objectInfo + int16 objectType
    + trailing pad.

    Middle field is unsigned (CTFAK2.0 `ParamObject.Read` uses
    `ReadUInt16()`; Anaconda `reader.readShort(True)`).
    """
    _require_min("Object", data, 6)
    object_info_list, object_info, object_type = struct.unpack_from(
        "<hHh", data, 0
    )
    return {
        "kind": "Object",
        "object_info_list": object_info_list,
        "object_info": object_info,
        "object_type": object_type,
        "trailing_hex": _trailing(data, 6),
    }


# --- Code 2 Time --------------------------------------------------------


def _dec_time(data: bytes) -> dict[str, Any]:
    """Code 2 — int32 timer (ms) + int32 loops + int16 comparison
    + trailing pad.

    CTFAK2.0 shape (`Events/Time.cs`): timer + loops + comparison = 10
    bytes. Anaconda's 8-byte (timer + loops) shape is stale for MMF 2.5;
    FNAF 1 carries 10-byte Time bodies.
    """
    _require_min("Time", data, 10)
    timer, loops, comparison = struct.unpack_from("<iih", data, 0)
    return {
        "kind": "Time",
        "timer": timer,
        "loops": loops,
        "comparison": comparison,
        "trailing_hex": _trailing(data, 10),
    }


# --- Code 6 Sample ------------------------------------------------------


def _dec_sample(data: bytes, *, unicode: bool) -> dict[str, Any]:
    """Code 6 — int16 handle + uint16 flags + null-term string
    + trailing pad.

    CTFAK2.0's `Sample.Write` emits a fixed-width 128-byte layout
    (4 header + UTF-16 name + skip 120 + int16 0). FNAF 1 Sample
    bodies are 134 bytes. We read as much as the loader needs (header
    + null-terminated name) and surface the remainder as
    `trailing_hex`.
    """
    _require_min("Sample", data, 4)
    handle, flags = struct.unpack_from("<hH", data, 0)
    name, string_bytes = _read_null_terminated(data[4:], unicode=unicode)
    consumed = 4 + string_bytes
    return {
        "kind": "Sample",
        "handle": handle,
        "flags": flags,
        "name": name,
        "trailing_hex": _trailing(data, consumed),
    }


# --- Position (used by Code 16 and embedded inside Code 9 Create) -------

# Position: 10 fields = 22 bytes total.
#   uint16 objectInfoParent
#   int16  flags
#   int16  x
#   int16  y
#   int16  slope
#   int16  angle
#   int32  direction
#   int16  typeParent
#   int16  objectInfoList
#   int16  layer
_POSITION_STRUCT = struct.Struct("<HhhhhhihhH")
_POSITION_SIZE = _POSITION_STRUCT.size  # 22


def _unpack_position(data: bytes, offset: int = 0) -> dict[str, Any]:
    """Unpack a 22-byte Position record from `data[offset:]`."""
    (
        object_info_parent,
        flags,
        x,
        y,
        slope,
        angle,
        direction,
        type_parent,
        object_info_list,
        layer,
    ) = _POSITION_STRUCT.unpack_from(data, offset)
    return {
        "kind": "Position",
        "object_info_parent": object_info_parent,
        "flags": flags,
        "x": x,
        "y": y,
        "slope": slope,
        "angle": angle,
        "direction": direction,
        "type_parent": type_parent,
        "object_info_list": object_info_list,
        "layer": layer,
    }


def _dec_position(data: bytes) -> dict[str, Any]:
    """Code 16 — 22-byte Position record + trailing pad."""
    _require_min("Position", data, _POSITION_SIZE)
    result = _unpack_position(data)
    result["trailing_hex"] = _trailing(data, _POSITION_SIZE)
    return result


# --- Code 9 Create ------------------------------------------------------

_CREATE_LOADER_SIZE = _POSITION_SIZE + 2 + 2  # 26 bytes (CTFAK2.0 shape)


def _dec_create(data: bytes) -> dict[str, Any]:
    """Code 9 — Position(22) + uint16 objectInstances + uint16 objectInfo
    + trailing pad.

    CTFAK2.0 shape (`Events/Create.cs`): 26 bytes. Anaconda's 30-byte
    shape adds `skipBytes(4)`; we fold that skip into the generic
    `trailing_hex` field so both models round-trip. FNAF 1 Create bodies
    are 32 bytes — 26 loader + 6 trailing.
    """
    _require_min("Create", data, _CREATE_LOADER_SIZE)
    position = _unpack_position(data, 0)
    # Drop the nested "kind" key — it's redundant inside a Create record
    # where the outer kind is already "Create".
    position.pop("kind")
    object_instances, object_info = struct.unpack_from(
        "<HH", data, _POSITION_SIZE
    )
    return {
        "kind": "Create",
        "position": position,
        "object_instances": object_instances,
        "object_info": object_info,
        "trailing_hex": _trailing(data, _CREATE_LOADER_SIZE),
    }


# --- Code 32 Click ------------------------------------------------------


def _dec_click(data: bytes) -> dict[str, Any]:
    """Code 32 — uint8 button + uint8 double + trailing pad.

    Button 0/1/2 = Left/Middle/Right (Anaconda CLICK_NAMES). Name
    resolution is the Name Resolver's job; here we just emit the int.
    """
    _require_min("Click", data, 2)
    click, double = struct.unpack_from("<BB", data, 0)
    return {
        "kind": "Click",
        "click": click,
        "double": bool(double),
        "trailing_hex": _trailing(data, 2),
    }


# --- Expression AST (codes 22, 23, 27, 45) -------------------------------

# systemLoaders / extensionLoaders dispatch tables from
# `expressions/names.py` lines 266-279. Only object_type=-1 has system
# loaders in FNAF 1's Clickteam build; the 5 sub-codes are the numeric
# literal / string literal / double literal / global-value / global-string
# leaves of the expression tree.
#
# extensionLoaders keys are object-extension expression codes with a
# structured payload — 16 (AlterableValue index) and 19 (AlterableString
# index). Every other num on an object_type >= 2 or == -7 reads just the
# 4-byte (objectInfo, objectInfoList) object-reference header and no body.


def _expr_long(data: bytes, *, unicode: bool) -> dict[str, Any]:
    """objectType=-1, num=0 — int32 numeric literal + trailing pad."""
    _require_min("Expression Long", data, 4)
    (value,) = struct.unpack_from("<i", data, 0)
    return {
        "body_kind": "Long",
        "value": value,
        "trailing_hex": _trailing(data, 4),
    }


def _expr_string(data: bytes, *, unicode: bool) -> dict[str, Any]:
    """objectType=-1, num=3 — null-term string literal + trailing pad.

    Anaconda's `expressions/loader.pyx` forces
    `reader.seek(currentPosition + size)` regardless of what the String
    loader actually consumed. We surface any trailing slack as
    `trailing_hex` so drift is visible but non-fatal.
    """
    value, consumed = _read_null_terminated(data, unicode=unicode)
    return {
        "body_kind": "String",
        "value": value,
        "trailing_hex": _trailing(data, consumed),
    }


def _expr_double(data: bytes, *, unicode: bool) -> dict[str, Any]:
    """objectType=-1, num=23 — float64 value + float32 floatValue
    + trailing pad.

    Both decoders agree: an 8-byte double and a trailing 4-byte float.
    (The duplicated precision is a Clickteam legacy artefact.)
    """
    _require_min("Expression Double", data, 12)
    value, float_value = struct.unpack_from("<df", data, 0)
    return {
        "body_kind": "Double",
        "value": value,
        "float_value": float_value,
        "trailing_hex": _trailing(data, 12),
    }


def _expr_global_value(data: bytes, *, unicode: bool) -> dict[str, Any]:
    """objectType=-1, num=24 — skip int32 + int32 value + trailing pad.

    CTFAK2.0 shape (`Events/Expression.cs::GlobalCommon`): 8 bytes.
    Anaconda's shape is 6 bytes (skip 4 + int16); we follow CTFAK2.0
    for robustness on large indices. Low 16 bits match either way.
    """
    _require_min("Expression GlobalValue", data, 8)
    (value,) = struct.unpack_from("<i", data, 4)
    return {
        "body_kind": "GlobalValue",
        "value": value,
        "trailing_hex": _trailing(data, 8),
    }


def _expr_global_string(data: bytes, *, unicode: bool) -> dict[str, Any]:
    """objectType=-1, num=50 — skip int32 + int32 value + trailing pad.

    Same GlobalCommon shape as GlobalValue.
    """
    _require_min("Expression GlobalString", data, 8)
    (value,) = struct.unpack_from("<i", data, 4)
    return {
        "body_kind": "GlobalString",
        "value": value,
        "trailing_hex": _trailing(data, 8),
    }


def _expr_extension_value(data: bytes, *, unicode: bool) -> dict[str, Any]:
    """num=16 — int16 AlterableValue index + trailing pad."""
    _require_min("Expression ExtensionValue", data, 2)
    (value,) = struct.unpack_from("<h", data, 0)
    return {
        "body_kind": "ExtensionValue",
        "value": value,
        "trailing_hex": _trailing(data, 2),
    }


def _expr_extension_string(data: bytes, *, unicode: bool) -> dict[str, Any]:
    """num=19 — int16 AlterableString index + trailing pad."""
    _require_min("Expression ExtensionString", data, 2)
    (value,) = struct.unpack_from("<h", data, 0)
    return {
        "body_kind": "ExtensionString",
        "value": value,
        "trailing_hex": _trailing(data, 2),
    }


# Dispatch: (object_type == -1, num) → body loader
_SYSTEM_EXPR_LOADERS: dict[int, Any] = {
    0: _expr_long,
    3: _expr_string,
    23: _expr_double,
    24: _expr_global_value,
    50: _expr_global_string,
}

# Dispatch: num → body loader for object_type >= 2 or == -7
_EXTENSION_EXPR_LOADERS: dict[int, Any] = {
    16: _expr_extension_value,
    19: _expr_extension_string,
}


def _decode_one_expression(
    data: bytes, start: int, *, unicode: bool
) -> tuple[dict[str, Any], int]:
    """Decode one Expression from `data` starting at byte `start`.
    Returns `(decoded, next_offset)`.

    Wire format per `expressions/loader.pyx` and
    `Events/Expression.cs`:

        int16  objectType
        int16  num
        if (objectType, num) == (0, 0):
            # END marker — 4 bytes total, no size field.
            return
        int16  size           # unsigned; total incl. objectType+num+size
        [body dispatched on (objectType, num)]
        # caller seeks to start + size

    The dispatched body falls in three branches:
      1. objectType == -1 and num in _SYSTEM_EXPR_LOADERS
         → Long/String/Double/GlobalValue/GlobalString body.
      2. objectType >= 2 or objectType == -7 → object-reference header
         (uint16 info, int16 infoList). Then if num in
         _EXTENSION_EXPR_LOADERS, a structured body; else no body.
      3. Everything else (e.g. objectType ∈ {-2, -3, -4, -5, -6} for
         Speaker/Game/Timer/Create/Keyboard expressions, or operator
         slots with objectType == 0) → no body, total size == 6.
    """
    n = len(data)
    if start + 4 > n:
        raise EventParameterDecodeError(
            f"Expression header at offset {start} needs 4 bytes, only "
            f"{n - start} left ({data[start:].hex()})"
        )
    object_type, num = struct.unpack_from("<hh", data, start)
    if object_type == 0 and num == 0:
        # END marker — no size field, 4 bytes total.
        return (
            {
                "object_type": 0,
                "num": 0,
                "size": 4,
                "end": True,
                "body_kind": None,
                "body": None,
            },
            start + 4,
        )

    if start + 6 > n:
        raise EventParameterDecodeError(
            f"Expression (type={object_type}, num={num}) at offset "
            f"{start} needs a 6-byte header, only {n - start} bytes left"
        )
    (size,) = struct.unpack_from("<H", data, start + 4)
    if size < 6:
        raise EventParameterDecodeError(
            f"Expression (type={object_type}, num={num}) at offset "
            f"{start} has size={size}, below 6-byte header minimum"
        )
    end = start + size
    if end > n:
        raise EventParameterDecodeError(
            f"Expression (type={object_type}, num={num}) at offset "
            f"{start} claims size={size}, but only {n - start} bytes left"
        )

    record: dict[str, Any] = {
        "object_type": object_type,
        "num": num,
        "size": size,
        "end": False,
        "object_info": None,
        "object_info_list": None,
        "body_kind": None,
        "body": None,
    }
    body_start = start + 6
    body_slice = data[body_start:end]

    if object_type == -1 and num in _SYSTEM_EXPR_LOADERS:
        loader = _SYSTEM_EXPR_LOADERS[num]
        record["body"] = loader(body_slice, unicode=unicode)
        record["body_kind"] = record["body"]["body_kind"]
    elif object_type >= 2 or object_type == -7:
        # Object-reference header: uint16 objectInfo + int16 objectInfoList.
        if len(body_slice) < 4:
            raise EventParameterDecodeError(
                f"Expression (type={object_type}, num={num}) at offset "
                f"{start} has size={size} but no room for the 4-byte "
                f"(objectInfo, objectInfoList) header"
            )
        object_info, object_info_list = struct.unpack_from(
            "<Hh", body_slice, 0
        )
        record["object_info"] = object_info
        record["object_info_list"] = object_info_list
        inner = body_slice[4:]
        if num in _EXTENSION_EXPR_LOADERS:
            loader = _EXTENSION_EXPR_LOADERS[num]
            record["body"] = loader(inner, unicode=unicode)
            record["body_kind"] = record["body"]["body_kind"]
        elif inner:
            # No structured body. Any bytes past the object-reference
            # header are slack — surface them.
            record["trailing_hex"] = inner.hex()
    else:
        # Operator slot (objectType == 0) or single-system expression
        # (-2 Speaker, -3 Game, -4 Timer, -5 Create, -6 Keyboard) — no
        # body. Any bytes past the header are slack.
        if body_slice:
            record["trailing_hex"] = body_slice.hex()

    return record, end


def _dec_expression_parameter(
    data: bytes, *, unicode: bool
) -> dict[str, Any]:
    """Codes 22, 23, 27, 45 — ExpressionParameter.

    Wire format per `parameters/loaders.py` ExpressionParameter.read:

        int16 comparison    # OPERATOR_LIST index (= <> <= < >= >)
        Expression*         # until END marker (objectType=0, num=0)
        [trailing pad]      # from outer Parameter.Read seek-to-size

    Comparison is meaningful on condition-side (code 22) ExpressionParameters
    and ignored on action-side (codes 23, 27, 45) but the field is always
    on-disk — we surface it verbatim.

    Trailing bytes after the END marker are tolerated (they are the
    outer Parameter wrapper's pad — `Events/Events.cs:539`
    `reader.Seek(currentPosition + size)`). We surface them as
    `trailing_hex`.
    """
    _require_min("ExpressionParameter", data, 2)
    (comparison,) = struct.unpack_from("<h", data, 0)
    pos = 2
    expressions: list[dict[str, Any]] = []
    # Hard upper bound to catch missing END -> infinite loop. Each
    # expression consumes at least 4 bytes so len(data) iterations is
    # strictly above any legal walk.
    for _ in range(len(data) + 1):
        expr, pos = _decode_one_expression(data, pos, unicode=unicode)
        expressions.append(expr)
        if expr["end"]:
            break
    else:
        raise EventParameterDecodeError(
            "ExpressionParameter: no END marker (objectType=0, num=0) "
            f"found within {len(data)}-byte body"
        )
    return {
        "kind": "ExpressionParameter",
        "comparison": comparison,
        "expressions": expressions,
        "trailing_hex": _trailing(data, pos),
    }


# --- Top-level dispatch -------------------------------------------------

# We can't put _dec_sample / _dec_expression_parameter directly in the
# dispatch table because they need the `unicode` kwarg. The caller wraps
# them via `decode_event_parameter` below.
_SIMPLE_DECODERS: dict[int, Any] = {
    1: _dec_object,
    2: _dec_time,
    9: _dec_create,
    10: _dec_short,
    14: _dec_key,
    16: _dec_position,
    25: _dec_int,
    26: _dec_short,
    32: _dec_click,
    50: _dec_short,
}

_STRING_DECODERS: frozenset[int] = frozenset({6})  # Sample needs unicode
_EXPRESSION_DECODERS: frozenset[int] = frozenset({22, 23, 27, 45})


def decode_event_parameter(
    code: int, data: bytes, *, unicode: bool
) -> dict[str, Any]:
    """Decode a single EventParameter payload.

    `code` must be one of the 15 FNAF 1 parameter codes
    (`FNAF1_PARAMETER_CODES`). `data` is the raw bytes between the code
    field and the next parameter boundary (i.e. `len(data) == size - 4`
    where `size` is the EventParameter's on-disk size field).

    `unicode` must match the pack's unicode flag — FNAF 1 is Unicode=True
    so all embedded strings are UTF-16LE.

    Returns a JSON-ready dict tagged with a stable `kind` field and a
    `trailing_hex` field (empty string when the loader consumed all
    of `data`, hex of the trailing slice otherwise — see module docstring
    for the CTFAK2.0 pad-to-size contract).

    Raises `EventParameterDecodeError` on:
      - unknown code
      - insufficient bytes for the documented loader shape
      - missing string terminator
      - missing Expression END marker
      - malformed Expression size field
    """
    if code not in FNAF1_PARAMETER_CODES:
        raise EventParameterDecodeError(
            f"code {code} is outside the FNAF 1 parameter-code set "
            f"{sorted(FNAF1_PARAMETER_CODES)} — refusing to decode. "
            f"If you are parsing a non-FNAF-1 pack, extend "
            f"FNAF1_PARAMETER_CODES and add a loader for this code."
        )
    if code in _SIMPLE_DECODERS:
        result = _SIMPLE_DECODERS[code](data)
    elif code in _STRING_DECODERS:
        # code 6 Sample — needs unicode.
        result = _dec_sample(data, unicode=unicode)
    elif code in _EXPRESSION_DECODERS:
        result = _dec_expression_parameter(data, unicode=unicode)
    else:  # pragma: no cover — FNAF1_PARAMETER_CODES is closed.
        raise EventParameterDecodeError(
            f"code {code} is in FNAF1_PARAMETER_CODES but has no decoder — "
            f"programmer error in dispatch table"
        )
    # Stamp the code on every result so downstream consumers don't need
    # to track it separately.
    result["code"] = code
    return result
