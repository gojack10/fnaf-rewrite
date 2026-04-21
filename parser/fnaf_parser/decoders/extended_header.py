"""0x2245 ExtendedHeader decoder (probe #4.2).

Schema cross-checked against:
- CTFAK2.0 Core/CTFAK.Core/CCN/Chunks/UselessChunks.cs (ExtendedHeader)
  — authoritative.
- Anaconda does not define this chunk loader, so there is no
  second-opinion spec. We lock values against FNAF 1's actual bytes as
  the empirical check.

Layout (20 bytes total):

    flags             u32 LE    bit flags (see _FLAG_NAMES)
    build_type        u8        Clickteam build enum; 0 = Windows EXE
    _pad              3 bytes   padding after build_type
    compression_flags u32 LE    pack-level compression flags
    screen_ratio      i16 LE
    screen_angle      i16 LE
    view_flags        u16 LE    always 0 in CTFAK's BitDict for FNAF-class games
    new_flags         u16 LE

CTFAK's BitDict leaves many bits as unnamed placeholder strings
("1", "2", ...). We carry CTFAK's names verbatim and keep the unnamed
slots as "BitN" so a set bit is always reportable — even if we don't
know its meaning yet, we can flag it for later investigation.

Antibody #2: 20 bytes in, 20 bytes consumed, no slack.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

EXTENDED_HEADER_SIZE = 20

_STRUCT = struct.Struct("<I B 3x I h h H H")
assert _STRUCT.size == EXTENDED_HEADER_SIZE

# BuildType enum from CTFAK ExtendedHeader.Read switch. We only name the
# ones Clickteam actually ships; anything else becomes "unknown(N)" via
# the .build_type_label helper.
_BUILD_TYPES: dict[int, str] = {
    0: "Windows EXE Application",
    1: "Windows Screen Saver",
    2: "Sub-Application",
    3: "Java Sub-Application",
    4: "Java Application",
    5: "Java Internet Applet",
    6: "Java Web Start",
    7: "Java for Mobile Devices",
    9: "Java Mac Application",
    10: "Adobe Flash",
    11: "Java for BlackBerry",
    12: "Android / OUYA Application",
    13: "iOS Application",
    14: "iOS Xcode Project",
    15: "Final iOS Xcode Project",
    18: "XNA Windows Project",
    19: "XNA Xbox Project",
    20: "XNA Phone Project",
    27: "HTML5 Development",
    28: "HTML5 Final Project",
    33: "UWP Project",
    34: "Android App Bundle",
    74: "Nintendo Switch",
    75: "Xbox One",
    78: "Playstation",
}

# Names per bit position for the flags field. Placeholder "BitN" where
# CTFAK also did not assign a name — keeps indexing obvious when a set
# bit turns up in the wild.
def _pad_names(named: dict[int, str], width: int = 32) -> tuple[str, ...]:
    return tuple(named.get(i, f"Bit{i}") for i in range(width))


_FLAG_NAMES: tuple[str, ...] = _pad_names({
    0: "KeepScreenRatio",
    2: "AntiAliasingWhenResizing",
    4: "RightToLeftReading",
    5: "RightToLeftLayout",
    20: "DontOptimizeStrings",
    24: "DontIgnoreDestroy",
    25: "DisableIME",
    26: "ReduceCPUUsage",
    28: "PremultipliedAlpha",
    29: "OptimizePlaySample",
})

_COMPRESSION_FLAG_NAMES: tuple[str, ...] = _pad_names({
    0: "CompressionLevelMax",
    1: "CompressSounds",
    2: "IncludeExternalFiles",
    3: "NoAutoImageFilters",
    4: "NoAutoSoundFilters",
    8: "DontDisplayBuildWarning",
    9: "OptimizeImageSize",
})


def _set_bits(value: int, names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for i, name in enumerate(names) if (value >> i) & 1)


@dataclass(frozen=True)
class ExtendedHeader:
    flags: int
    build_type: int
    compression_flags: int
    screen_ratio: int
    screen_angle: int
    view_flags: int
    new_flags: int

    def build_type_label(self) -> str:
        return _BUILD_TYPES.get(self.build_type, f"unknown({self.build_type})")

    def flag_names_set(self) -> tuple[str, ...]:
        return _set_bits(self.flags, _FLAG_NAMES)

    def compression_flag_names_set(self) -> tuple[str, ...]:
        return _set_bits(self.compression_flags, _COMPRESSION_FLAG_NAMES)

    def as_dict(self) -> dict:
        return {
            "flags": f"0x{self.flags:08X}",
            "flags_set": list(self.flag_names_set()),
            "build_type": self.build_type,
            "build_type_label": self.build_type_label(),
            "compression_flags": f"0x{self.compression_flags:08X}",
            "compression_flags_set": list(self.compression_flag_names_set()),
            "screen_ratio": self.screen_ratio,
            "screen_angle": self.screen_angle,
            "view_flags": f"0x{self.view_flags:04X}",
            "new_flags": f"0x{self.new_flags:04X}",
        }


class ExtendedHeaderDecodeError(ValueError):
    """0x2245 decode failure — always includes byte-count context."""


def decode_extended_header(payload: bytes) -> ExtendedHeader:
    if len(payload) != EXTENDED_HEADER_SIZE:
        raise ExtendedHeaderDecodeError(
            f"0x2245 ExtendedHeader: expected {EXTENDED_HEADER_SIZE} bytes "
            f"but got {len(payload)}. Antibody #2: byte count must reconcile."
        )
    (
        flags,
        build_type,
        compression_flags,
        screen_ratio,
        screen_angle,
        view_flags,
        new_flags,
    ) = _STRUCT.unpack(payload)
    return ExtendedHeader(
        flags=flags,
        build_type=build_type,
        compression_flags=compression_flags,
        screen_ratio=screen_ratio,
        screen_angle=screen_angle,
        view_flags=view_flags,
        new_flags=new_flags,
    )
