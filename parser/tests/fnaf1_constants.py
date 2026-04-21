"""Shared empirical constants pinned against FNAF 1 (build 284).

These are pack-level invariants observed across multiple probes. Each
has been independently confirmed against the FNAF 1 binary; a mismatch
here is a loud, clearly-named failure rather than a buried `== 260`
literal in one decoder's test.

`FNAF1_BANK_OFFSET_DELTA`
    Reference-frame shift between `record_start_offset` (position
    inside the decompressed bank body) and the corresponding entry in
    the sibling offsets table. Confirmed at:

    - probe #7 (Images / 0x6666 vs 0x5555): 605 records, delta = 260
    - probe #9 (Sounds / 0x6668 vs 0x5557): 52 non-zero records, 260
    - probe #10 (Fonts / 0x6667 vs 0x5556): 7 records, 260

    Three independent banks, all agreeing on +260. The leading theory
    is a CTFAK-era per-bank header footprint (≈260 bytes) that the
    offsets table accounts for but our per-record decode strips away.
    Any future FNAF 1 rebuild that changes the delta invalidates the
    cross-chunk handshake in all three banks simultaneously; pinning it
    once here makes that loud.

    Callers import this constant instead of re-declaring a literal
    260 so a change requires editing exactly one line.
"""

from __future__ import annotations

FNAF1_BANK_OFFSET_DELTA = 260
