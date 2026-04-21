"""Regression test for the PE shell walker.

Asserts the data pack start for our FNAF 1 binary is exactly 0x00101400.
This is the first Parser Antibody in practice — a deterministic invariant
that will fail loudly if the binary we ship against changes, or if our
PE header parsing ever regresses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START, pe_data_pack_start

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_data_pack_starts_at_known_offset():
    assert pe_data_pack_start(FNAF_EXE) == FNAF1_DATA_PACK_START


def test_rejects_non_pe_input(tmp_path: Path):
    bad = tmp_path / "not-a-pe.bin"
    bad.write_bytes(b"\x00" * 1024)
    with pytest.raises(ValueError, match="MZ"):
        pe_data_pack_start(bad)
