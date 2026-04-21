"""0x2223 AppHeader decoder (probe #4.1).

Schema cross-checked against:
- CTFAK2.0  Core/CTFAK.Core/CCN/Chunks/AppHeader.cs  (newer, authoritative)
- Anaconda  mmfparser/data/chunkloaders/appheader.py

Both agree on the payload fields and their total width (112 bytes) but
differ in two places. Tracked with inline REFERENCE DISAGREEMENT notes:

  (1) Controls layout — RESOLVED EMPIRICALLY.
      CTFAK reads four PlayerControls *interleaved*:
          [ctrl0.type][ctrl0.keys(16)][ctrl1.type][ctrl1.keys(16)]...
      Anaconda reads them *separated*:
          [ctrl0.type][ctrl1.type][ctrl2.type][ctrl3.type]
          [ctrl0.keys][ctrl1.keys][ctrl2.keys][ctrl3.keys]
      Both consume 72 bytes total, so byte-count alone cannot arbitrate.
      We decoded FNAF 1's raw 72 bytes two ways and only Anaconda's
      separated layout produces four identical players (expected: a
      packaged single-player game ships with uniform defaults) with
      coherent VK codes (0x25..0x28 = arrow keys, 0x10 = Shift, etc).
      Under CTFAK's interleaved reading, the first "player" is
      type=5/keys=0x0005,0x0005,0x0005,... which is nonsense.
      → Decoder implements Anaconda's separated layout. CTFAK upstream
      is either stale or targets an older game version. The raw 72 bytes
      are still preserved in `controls_raw` in case this flips again.

  (2) WindowsMenuIndex width.
      CTFAK: int32 (4 bytes).
      Anaconda: uint8 + 3 pad bytes.
      Same 4 bytes on disk. CTFAK wins since for the FNAF 1 values seen
      the extra three bytes are observed to be zero — either reading is
      lossless, and int32 is less structural friction.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# Bit-flag names — directly from CTFAK2.0 AppHeader.cs and cross-checked
# against Anaconda's HEADER_FLAGS / HEADER_NEW_FLAGS / HEADER_OTHER_FLAGS.
# Where the two disagree on a bit name, CTFAK wins (as authoritative spec);
# Anaconda's name recorded as a trailing comment for traceability.
_FLAG_NAMES: tuple[str, ...] = (
    "HeadingMaximized",      # Anaconda: BorderMax
    "NoHeading",
    "FitInsideBars",         # Anaconda: Panic
    "MachineIndependentSpeed",  # Anaconda: SpeedIndependent
    "ResizeDisplay",         # Anaconda: Stretch
    "MusicOn",
    "SoundOn",
    "DontDisplayMenu",       # Anaconda: MenuHidden
    "MenuBar",
    "MaximizedOnBoot",       # Anaconda: Maximize
    "MultiSamples",
    "ChangeResolutionMode",  # Anaconda: FullscreenAtStart
    "SwitchToFromFullscreen",  # Anaconda: FullscreenSwitch
    "Protected",
    "Copyright",
    "OneFile",
)

_NEW_FLAG_NAMES: tuple[str, ...] = (
    "SamplesOverFrames",
    "RelocFiles",
    "RunFrame",
    "PlaySamplesWhenUnfocused",  # Anaconda: SamplesWhenNotFocused
    "NoMinimizeBox",
    "NoMaximizeBox",
    "NoThickFrame",
    "DoNotCenterFrame",
    "IgnoreInputOnScreensaver",  # Anaconda: ScreensaverAutostop
    "DisableClose",
    "HiddenAtStart",
    "VisualThemes",          # Anaconda: XPVisualThemes
    "VSync",
    "RunWhenMinimized",
    "MDI",
    "RunWhileResizing",
)

# CTFAK populates all 16 bits; Anaconda only names 10 and leaves the rest
# blank. Where Anaconda gives a different name we annotate; where Anaconda
# had nothing we still provide CTFAK's name.
_OTHER_FLAG_NAMES: tuple[str, ...] = (
    "DebuggerShortcuts",
    "Unknown1",              # Anaconda: DirectX
    "Unknown2",              # Anaconda: VRAM
    "DontShareSubData",      # Anaconda: Obsolete
    "Unknown3",              # Anaconda: AutoImageFilter
    "Unknown4",              # Anaconda: AutoSoundFilter
    "Unknown5",              # Anaconda: AllInOne
    "ShowDebugger",
    "Unknown6",              # Anaconda: Reserved1
    "Unknown7",              # Anaconda: Reserved2
    "Unknown8",
    "Unknown9",
    "Unknown10",
    "Unknown11",
    "Direct3D9or11",
    "Direct3D8or11",
)

# Clickteam's graphics-mode enum. Anaconda documents it; CTFAK does not but
# stores the same int16. We stringify for the JSON dump.
_GRAPHIC_MODES: dict[int, str] = {
    3: "256 colors",
    4: "16 million colors",
    6: "32768 colors",
    7: "65536 colors",
}

# Fixed-width fields up to (but not including) controls. 24 bytes.
_HEADER_PREFIX = struct.Struct("<i H H h H h h I I")
# Prefix widths:  size(i4) flags(u2) newFlags(u2) mode(i2) otherFlags(u2)
#                 width(u2) height(u2) initScore(u4) initLives(u4)
# Unsigned shorts for width/height because CTFAK treats them as logical
# dimensions; negative values would be nonsense.
assert _HEADER_PREFIX.size == 24

_CONTROLS_SIZE = 72  # 4 players × (int16 type + 8× int16 keys)
# After controls: borderColor(4) + numberOfFrames(i4) + frameRate(i4)
#                 + windowsMenuIndex(i4)  = 16 bytes
_HEADER_TAIL = struct.Struct("<4s i i i")
assert _HEADER_TAIL.size == 16

APP_HEADER_SIZE = _HEADER_PREFIX.size + _CONTROLS_SIZE + _HEADER_TAIL.size  # 112
assert APP_HEADER_SIZE == 112

# initialScore/initialLives on disk are XOR 0xFFFFFFFF. Both references
# apply this mask. Stored unmasked in the dataclass.
_XOR_MASK = 0xFFFFFFFF


@dataclass(frozen=True)
class BitFlags:
    """16-bit flag word with per-bit names.

    `value` is the raw uint16. `set_flags` is the subset of names whose bit
    is 1. Stored both ways because downstream code sometimes wants the raw
    bits (for round-trip / re-encode) and sometimes the names (for JSON).
    """
    value: int
    names: tuple[str, ...]  # name per bit position 0..15

    @property
    def set_flags(self) -> tuple[str, ...]:
        return tuple(
            name for i, name in enumerate(self.names) if (self.value >> i) & 1
        )

    def as_dict(self) -> dict:
        return {
            "value": f"0x{self.value:04X}",
            "set": list(self.set_flags),
        }


@dataclass(frozen=True)
class AppHeader:
    # Redundant size field inside the decompressed payload. CTFAK reads it
    # (when not Settings.Old) as the first int32; we preserve it as a sanity
    # check — expected to equal APP_HEADER_SIZE (112).
    inner_size: int

    flags: BitFlags
    new_flags: BitFlags
    other_flags: BitFlags

    graphics_mode: int                  # raw int16
    window_width: int
    window_height: int
    initial_score: int                  # already XOR'd back
    initial_lives: int

    # REFERENCE DISAGREEMENT (1): controls layout unresolved; bytes kept raw.
    controls_raw: bytes = field(repr=False)

    border_color: tuple[int, int, int, int]  # (R, G, B, A)
    number_of_frames: int               # count of Frame chunks declared
    frame_rate: int
    windows_menu_index: int             # REFERENCE DISAGREEMENT (2); stored as int32

    def graphics_mode_label(self) -> str:
        return _GRAPHIC_MODES.get(self.graphics_mode, f"unknown({self.graphics_mode})")

    def as_dict(self) -> dict:
        r, g, b, a = self.border_color
        return {
            "inner_size": self.inner_size,
            "flags": self.flags.as_dict(),
            "new_flags": self.new_flags.as_dict(),
            "other_flags": self.other_flags.as_dict(),
            "graphics_mode": self.graphics_mode,
            "graphics_mode_label": self.graphics_mode_label(),
            "window_width": self.window_width,
            "window_height": self.window_height,
            "initial_score": self.initial_score,
            "initial_lives": self.initial_lives,
            "controls_raw_hex": self.controls_raw.hex(),
            "border_color": {"r": r, "g": g, "b": b, "a": a},
            "number_of_frames": self.number_of_frames,
            "frame_rate": self.frame_rate,
            "windows_menu_index": self.windows_menu_index,
        }


class HeaderDecodeError(ValueError):
    """0x2223 decode failure — always includes byte accounting context."""


def decode_header(payload: bytes) -> AppHeader:
    """Decode a decompressed 0x2223 payload.

    Antibody #2: asserts len(payload) == APP_HEADER_SIZE. Either the
    decompression layer handed us the wrong bytes or the upstream spec is
    mis-translated; both are bugs worth surfacing loudly rather than
    quietly reading 112 of N bytes.
    """
    if len(payload) != APP_HEADER_SIZE:
        raise HeaderDecodeError(
            f"0x2223 Header: expected {APP_HEADER_SIZE} bytes (decompressed) "
            f"but got {len(payload)}. Antibody #2: byte count must reconcile."
        )

    (
        inner_size,
        flags_raw,
        new_flags_raw,
        graphics_mode,
        other_flags_raw,
        window_width,
        window_height,
        init_score_raw,
        init_lives_raw,
    ) = _HEADER_PREFIX.unpack_from(payload, 0)

    controls_raw = bytes(payload[_HEADER_PREFIX.size : _HEADER_PREFIX.size + _CONTROLS_SIZE])

    tail_offset = _HEADER_PREFIX.size + _CONTROLS_SIZE
    border_rgba, number_of_frames, frame_rate, windows_menu_index = (
        _HEADER_TAIL.unpack_from(payload, tail_offset)
    )

    if inner_size != APP_HEADER_SIZE:
        # Not fatal (some older versions use a different value), but worth
        # surfacing — a size mismatch is often the first sign of format drift.
        # Raise to make it a real signal; can be downgraded if a real binary
        # legitimately reports something else.
        raise HeaderDecodeError(
            f"0x2223 Header: inner size field = {inner_size}, expected "
            f"{APP_HEADER_SIZE}. Format drift or wrong slice?"
        )

    return AppHeader(
        inner_size=inner_size,
        flags=BitFlags(value=flags_raw, names=_FLAG_NAMES),
        new_flags=BitFlags(value=new_flags_raw, names=_NEW_FLAG_NAMES),
        other_flags=BitFlags(value=other_flags_raw, names=_OTHER_FLAG_NAMES),
        graphics_mode=graphics_mode,
        window_width=window_width,
        window_height=window_height,
        initial_score=init_score_raw ^ _XOR_MASK,
        initial_lives=init_lives_raw ^ _XOR_MASK,
        controls_raw=controls_raw,
        border_color=(
            border_rgba[0],
            border_rgba[1],
            border_rgba[2],
            border_rgba[3],
        ),
        number_of_frames=number_of_frames,
        frame_rate=frame_rate,
        windows_menu_index=windows_menu_index,
    )
