"""0x333B FrameFadeIn / 0x333C FrameFadeOut decoder (probe #4.11).

Both sub-chunks share the same on-disk shape — Clickteam's `Transition`
record — so a single decoder handles both. The distinction between
fade-in and fade-out lives at the sub-chunk id layer; `decode_frame`
stores the decoded value in `Frame.fade_in` or `Frame.fade_out`
accordingly.

Shape per the Transition class in both reference readers
(verified field-for-field identical this probe):

- CTFAK2.0 `Core/CTFAK.Core/CCN/Chunks/Frame/Transition.cs`:

        Module        = reader.ReadAscii(4);
        Name          = reader.ReadAscii(4);
        Duration      = reader.ReadInt32();
        Flags         = reader.ReadInt32();
        Color         = reader.ReadColor();
        var nameOffset      = reader.ReadInt32();
        var parameterOffset = reader.ReadInt32();
        var parameterSize   = reader.ReadInt32();
        reader.Seek(currentPos + nameOffset);
        ModuleFile = reader.ReadYuniversal();
        reader.Seek(currentPos + parameterOffset);
        ParameterData = reader.ReadBytes(parameterSize);

  Dispatched from Frame.cs cases 13113/13115 (FadeIn) and 13114/13116
  (FadeOut) — both construct a `Transition` and call Read.

- Anaconda `mmfparser/data/chunkloaders/transition.py`:

        self.module          = reader.read(4)
        self.name            = reader.read(4)
        self.duration        = reader.readInt()           # int32 ms
        self.flags.setFlags(reader.readInt(True))         # uint32, BitDict('Color')
        self.color           = reader.readColor()         # 4 raw bytes RGBA
        nameOffset      = reader.readInt()
        parameterOffset = reader.readInt()
        parameterSize   = reader.readInt()
        reader.seek(currentPosition + nameOffset)
        self.moduleFile      = str(reader.readUnicodeString())
        reader.seek(currentPosition + parameterOffset)
        self.parameterData   = reader.read(parameterSize)

  Classes `FadeIn(Transition)` and `FadeOut(Transition)` are identical
  to the base - only the id/direction differs.

Fixed header size: 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 = 32 B. Followed by
a variable tail: a null-terminated UTF-16LE moduleFile string at
absolute offset `nameOffset`, then `parameterSize` bytes at absolute
offset `parameterOffset`.

FNAF 1 empirical (all 17 frames, both fade slots):

- 10 of 17 frames ship at least one fade record (17 records total:
  9 FadeIn + 8 FadeOut). 7 frames (#1, #3, #4, #5, #7, #14, #15) ship
  neither - field is None at the Frame layer.
- Every fade record is **exactly 56 bytes** after decrypt+decompress.
- Module == b"STDT" (standard transition marker - `Transition.isStandard()`
  in Anaconda).
- Name == b"FADE".
- nameOffset == 32 (start of tail), parameterOffset == 56, parameterSize
  == 0 (no shader/module parameter data).
- moduleFile == "cctrans.dll" (Clickteam's ships-with-engine standard-
  transition DLL, 11 chars + NUL = 24 UTF-16 bytes; 32 + 24 = 56).
- FadeIn records use flags == 0 (Color bit clear).
- FadeOut records use flags == 1 (Color bit set); color bytes are
  00 00 00 00 (fading to black, which matches the game's look).
- Durations vary per frame between 560 ms (fastest transition, frame
  #12) and 2000 ms (frames #9-#13, office-night intros).

Antibody coverage (this decoder):

- #1 strict-unknown: the 32-byte header is fully pinned - any byte
  within the payload unaccounted for raises. We do NOT assert
  Module=='STDT' here (Anaconda's `isStandard()` is a property query,
  not a format invariant - a non-STDT custom transition is still a
  valid Transition record, just using a different .mvx). But we DO
  assert byte-count reconcile: the fixed 32-byte header + moduleFile
  bytes (including terminator) + parameterSize must exactly equal
  `len(payload)`.
- #2 byte-count: tight equality enforced at the end; off-by-one or RC4
  drift fires immediately.
- #3 round-trip: synthetic pack/unpack in tests covers both FadeIn
  (flags=0, parameterSize=0) and FadeOut (flags=1) plus a synthetic
  record with a non-empty parameterData tail to prove the variable
  tail works even though FNAF 1 never exercises it.
- #4 multi-oracle: CTFAK and Anaconda field orders are identical
  (verified in the probe node).
- #5 multi-input: 34 fade slots (17 frames x {fade_in, fade_out}) pass
  through the same decoder; absent slots stay None at decode_frame,
  present slots all hit the strict byte-count.
- #7 snapshot: per-frame (has_fade_in, has_fade_out, duration_in,
  duration_out, flags_in, flags_out) pinned in tests.

The moduleFile is decoded as UTF-16LE regardless of the pack's
Unicode flag, because Anaconda unconditionally calls
`readUnicodeString()` and CTFAK's `ReadYuniversal` reads the pack-
Unicode path. FNAF 1 is Unicode-on anyway so the distinction is moot
for this build, but if a non-Unicode pack ever arrives at this
decoder the `unicode` parameter is threaded through so the caller can
flip to ASCII without reopening this probe.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# Fixed-header layout:
#   module         4 bytes ASCII  ('STDT' for Clickteam standard transition)
#   name           4 bytes ASCII  ('FADE' for fade transitions)
#   duration       int32 LE       transition time in ms
#   flags          int32 LE       BitDict('Color'): bit 0 = color valid
#   color          4 bytes        RGBA (r, g, b, a) per ByteReader.ReadColor
#   name_offset    int32 LE       offset to moduleFile string, relative to
#                                 sub-chunk payload start
#   param_offset   int32 LE       offset to parameterData blob, relative
#                                 to sub-chunk payload start
#   param_size     int32 LE       length of parameterData blob
# All multi-byte ints little-endian. Signed per CTFAK's ReadInt32;
# Anaconda treats `flags` as unsigned via readInt(True) but the sign
# never matters since FNAF 1 only uses bits 0..0.
_FRAME_FADE_FIXED = struct.Struct("<4s4sii4siii")
FRAME_FADE_FIXED_SIZE = _FRAME_FADE_FIXED.size  # 32
assert FRAME_FADE_FIXED_SIZE == 32

# Flags bits - currently only bit 0 ('Color') is defined by Anaconda's
# BitDict. Expose as a mask so consumers can test without reopening
# this probe.
FLAG_COLOR_BIT = 0x1


class FrameFadeDecodeError(ValueError):
    """0x333B/0x333C FrameFade decode failure - carries byte-count context."""


@dataclass(frozen=True)
class FrameFade:
    """One decoded 0x333B FadeIn or 0x333C FadeOut sub-chunk.

    Fields follow both CTFAK.Transition and Anaconda.Transition exactly:

    - `module`: 4 raw bytes. b"STDT" for standard (Clickteam-ships-it)
      transitions. Other values indicate a custom .mvx module.
    - `name`: 4 raw bytes. b"FADE" for fade transitions. Kept raw so
      a future probe against non-fade transitions doesn't need to
      re-decide the field type.
    - `duration_ms`: int32. Transition duration in milliseconds.
    - `flags`: int32. BitDict('Color') - only bit 0 is defined.
    - `color`: 4-tuple (r, g, b, a) of bytes. The fade target color
      (valid iff `has_color_bit`). FNAF 1 always fades to 00 00 00 00
      (black) but this is kept as raw so round-trip is lossless.
    - `module_file`: str. The .mvx module filename.
    - `parameter_data`: bytes. Module-specific parameter blob;
      empty in FNAF 1's standard fades.
    - `name_offset`, `param_offset`, `param_size`: int. The three
      offset fields from the fixed header; kept on the dataclass so
      round-trip is lossless even though `param_size == len(parameter_data)`
      and the offsets are fully derived from `name_offset`,
      `len(module_file)`, etc.
    """
    module: bytes
    name: bytes
    duration_ms: int
    flags: int
    color: tuple[int, int, int, int]
    name_offset: int
    param_offset: int
    param_size: int
    module_file: str
    parameter_data: bytes

    @property
    def has_color_bit(self) -> bool:
        """True iff the Color flag bit is set (bit 0 of `flags`)."""
        return (self.flags & FLAG_COLOR_BIT) != 0

    @property
    def is_standard(self) -> bool:
        """True iff this is a Clickteam standard transition (module=='STDT').

        Mirrors Anaconda's `Transition.isStandard()`. FNAF 1's fades all
        return True.
        """
        return self.name == b"STDT" or self.module == b"STDT"

    def as_dict(self) -> dict:
        return {
            "module": self.module.decode("ascii", errors="replace"),
            "name": self.name.decode("ascii", errors="replace"),
            "duration_ms": self.duration_ms,
            "flags": self.flags,
            "color": list(self.color),
            "has_color_bit": self.has_color_bit,
            "name_offset": self.name_offset,
            "param_offset": self.param_offset,
            "param_size": self.param_size,
            "module_file": self.module_file,
            "parameter_data": self.parameter_data.hex(),
        }


def _read_null_terminated(
    payload: bytes, start: int, *, unicode: bool
) -> tuple[str, int]:
    """Read a null-terminated string from `payload` starting at absolute
    offset `start`. Returns (decoded_string, bytes_consumed_including_nul).

    Unicode path reads 2-byte code units until a wide NUL (b"\\x00\\x00")
    at an even offset. ASCII path reads bytes until a single NUL.
    Raises FrameFadeDecodeError if no terminator is found before the
    payload ends (Antibody #1 strict-unknown).
    """
    if unicode:
        pos = start
        while pos + 2 <= len(payload):
            if payload[pos:pos + 2] == b"\x00\x00":
                body = payload[start:pos]
                try:
                    decoded = body.decode("utf-16le")
                except UnicodeDecodeError as exc:
                    raise FrameFadeDecodeError(
                        f"0x333B/0x333C FrameFade: UTF-16LE decode of "
                        f"module_file at offset 0x{start:x} failed: {exc}"
                    ) from exc
                return decoded, (pos - start) + 2  # include the 2-byte NUL
            pos += 2
        raise FrameFadeDecodeError(
            f"0x333B/0x333C FrameFade: no UTF-16 NUL terminator found for "
            f"module_file starting at offset 0x{start:x} (payload length "
            f"{len(payload)}). Antibody #1 strict-unknown: string must "
            f"terminate inside the payload."
        )

    # ASCII path - kept for completeness, even though FNAF 1 is Unicode.
    pos = start
    while pos < len(payload):
        if payload[pos] == 0:
            body = payload[start:pos]
            try:
                decoded = body.decode("ascii")
            except UnicodeDecodeError as exc:
                raise FrameFadeDecodeError(
                    f"0x333B/0x333C FrameFade: ASCII decode of module_file "
                    f"at offset 0x{start:x} failed: {exc}"
                ) from exc
            return decoded, (pos - start) + 1
        pos += 1
    raise FrameFadeDecodeError(
        f"0x333B/0x333C FrameFade: no ASCII NUL terminator found for "
        f"module_file starting at offset 0x{start:x} (payload length "
        f"{len(payload)}). Antibody #1 strict-unknown."
    )


def decode_frame_fade(payload: bytes, *, unicode: bool = True) -> FrameFade:
    """Decode a 0x333B FrameFadeIn or 0x333C FrameFadeOut sub-chunk's
    plaintext bytes.

    The caller is responsible for having already run the payload through
    the flag=3 decrypt+decompress path (probe #4.5). This function only
    sees the final plaintext. The caller also decides whether the record
    came from 0x333B or 0x333C via the sub-chunk id - both decode
    through here; `decode_frame` slots the result into `fade_in` or
    `fade_out` accordingly.

    Antibody #2 (byte-count): the sum of fixed-header (32 B) +
    module_file bytes (including the NUL terminator) + param_size must
    exactly equal `len(payload)`. Any gap or overlap raises loudly.
    """
    n = len(payload)
    if n < FRAME_FADE_FIXED_SIZE:
        raise FrameFadeDecodeError(
            f"0x333B/0x333C FrameFade: payload must hold at least the "
            f"{FRAME_FADE_FIXED_SIZE}-byte fixed header but got {n}. "
            f"Antibody #2: byte count must reconcile."
        )

    (
        module,
        name,
        duration_ms,
        flags,
        color_raw,
        name_offset,
        param_offset,
        param_size,
    ) = _FRAME_FADE_FIXED.unpack_from(payload, 0)

    # Antibody #1 strict-unknown: all three offsets must point inside
    # the payload and the implied regions must not overlap or leave
    # gaps. name_offset can't be before the fixed header (would alias
    # back into it). param_size must be non-negative.
    if name_offset < FRAME_FADE_FIXED_SIZE or name_offset > n:
        raise FrameFadeDecodeError(
            f"0x333B/0x333C FrameFade: name_offset={name_offset} is "
            f"outside [{FRAME_FADE_FIXED_SIZE}, {n}]. Antibody #1 "
            f"strict-unknown: likely RC4 drift."
        )
    if param_offset < FRAME_FADE_FIXED_SIZE or param_offset > n:
        raise FrameFadeDecodeError(
            f"0x333B/0x333C FrameFade: param_offset={param_offset} is "
            f"outside [{FRAME_FADE_FIXED_SIZE}, {n}]. Antibody #1 "
            f"strict-unknown: likely RC4 drift."
        )
    if param_size < 0:
        raise FrameFadeDecodeError(
            f"0x333B/0x333C FrameFade: param_size={param_size} is "
            f"negative. Antibody #1 strict-unknown: likely RC4 drift."
        )
    if param_offset + param_size > n:
        raise FrameFadeDecodeError(
            f"0x333B/0x333C FrameFade: param_offset={param_offset} + "
            f"param_size={param_size} = {param_offset + param_size} "
            f"exceeds payload length {n}. Antibody #2 byte-count."
        )

    module_file, name_bytes_consumed = _read_null_terminated(
        payload, name_offset, unicode=unicode
    )

    # Byte-count reconcile. The on-disk layout is:
    #   [0 .. 32)                     fixed header
    #   [name_offset .. name_offset + name_bytes_consumed)   module_file + NUL
    #   [param_offset .. param_offset + param_size)          parameter_data
    # For FNAF 1 these regions are adjacent: name_offset == 32 and
    # param_offset == name_offset + name_bytes_consumed. We don't hard-
    # require that ordering (Transition allows reordered tails in
    # principle) but we DO require every payload byte to be accounted
    # for by exactly one region.
    used = bytearray(n)
    for lo, hi in (
        (0, FRAME_FADE_FIXED_SIZE),
        (name_offset, name_offset + name_bytes_consumed),
        (param_offset, param_offset + param_size),
    ):
        for i in range(lo, hi):
            if used[i] != 0:
                raise FrameFadeDecodeError(
                    f"0x333B/0x333C FrameFade: byte at offset 0x{i:x} is "
                    f"covered by more than one region (fixed header / "
                    f"module_file / parameter_data overlap). Antibody #1/"
                    f"#2: layout is not internally consistent. "
                    f"name_offset={name_offset}, "
                    f"name_bytes_consumed={name_bytes_consumed}, "
                    f"param_offset={param_offset}, param_size={param_size}."
                )
            used[i] = 1
    if any(u == 0 for u in used):
        first_gap = next(i for i, u in enumerate(used) if u == 0)
        raise FrameFadeDecodeError(
            f"0x333B/0x333C FrameFade: byte at offset 0x{first_gap:x} is "
            f"not covered by any region (fixed header / module_file / "
            f"parameter_data). Antibody #2 byte-count: payload has "
            f"unaccounted-for bytes. len(payload)={n}, "
            f"name_offset={name_offset}, "
            f"name_bytes_consumed={name_bytes_consumed}, "
            f"param_offset={param_offset}, param_size={param_size}."
        )

    parameter_data = bytes(payload[param_offset:param_offset + param_size])

    return FrameFade(
        module=bytes(module),
        name=bytes(name),
        duration_ms=duration_ms,
        flags=flags,
        color=(color_raw[0], color_raw[1], color_raw[2], color_raw[3]),
        name_offset=name_offset,
        param_offset=param_offset,
        param_size=param_size,
        module_file=module_file,
        parameter_data=parameter_data,
    )
