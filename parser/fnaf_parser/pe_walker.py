"""PE shell walker.

Locates the end of the last mapped PE section in a 32-bit Windows executable.
For our FiveNightsatFreddys.exe this should equal 0x00101400, which is where
the Clickteam Fusion 2.5 data pack begins. See Binary Layout - Rizin Decomp
in the FNAF Speedrun tree for the measured byte layout.

This is the first deterministic invariant of the parser. Any drift here means
either the binary changed or our PE parsing is wrong — both cases we want
to fail loudly before touching a single chunk.
"""

from __future__ import annotations

import struct
from pathlib import Path

# Known-good data pack start for the FNAF 1 binary we own.
# Sourced from [[Binary Layout - Rizin Decomp|48529e04-be83-41ae-b5f3-ded9c1b2551e]].
FNAF1_DATA_PACK_START = 0x00101400

_DOS_E_LFANEW_OFFSET = 0x3C
_PE_SIGNATURE = b"PE\x00\x00"
_COFF_HEADER_SIZE = 20
_SECTION_HEADER_SIZE = 40


def pe_data_pack_start(path: Path) -> int:
    """Return the file offset immediately after the last mapped PE section.

    For a Clickteam-packed binary this is where the trailing data pack starts.
    Raises ValueError on malformed input.
    """
    blob = Path(path).read_bytes()
    if blob[:2] != b"MZ":
        raise ValueError("not a PE file: missing MZ signature")

    (e_lfanew,) = struct.unpack_from("<I", blob, _DOS_E_LFANEW_OFFSET)
    if blob[e_lfanew : e_lfanew + 4] != _PE_SIGNATURE:
        raise ValueError(f"missing PE signature at e_lfanew=0x{e_lfanew:x}")

    coff_off = e_lfanew + 4
    # COFF header: Machine(H) NumberOfSections(H) TimeDateStamp(I)
    # PointerToSymbolTable(I) NumberOfSymbols(I) SizeOfOptionalHeader(H)
    # Characteristics(H)
    (_machine, n_sections, _ts, _psym, _nsym, size_opt, _char) = struct.unpack_from(
        "<HHIIIHH", blob, coff_off
    )
    sections_off = coff_off + _COFF_HEADER_SIZE + size_opt

    # Section header: Name[8], VirtualSize(I), VirtualAddress(I),
    # SizeOfRawData(I), PointerToRawData(I), ... we only need raw data fields.
    end = 0
    for i in range(n_sections):
        off = sections_off + i * _SECTION_HEADER_SIZE
        size_raw, ptr_raw = struct.unpack_from("<II", blob, off + 16)
        end = max(end, ptr_raw + size_raw)
    return end
