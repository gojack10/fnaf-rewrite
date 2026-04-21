"""Regression tests for empirical-only chunks (probe #4.3).

Antibody coverage:

- #2 byte-count — decoded payload must equal the exact size observed
  empirically in FNAF 1 (8 for 0x224D, 12 for 0x224F). Any non-FNAF-1
  build with different sized payloads must fail here, not silently.
- #4 multi-oracle — 0x224F's first u32 equals pack_header.product_build,
  providing a cross-chunk invariant that ties this empirical chunk to
  the pack metadata. If either side drifts, the test fires.

FNAF 1 ground-truth locked here:

- 0x224D payload = `00 00 00 00 03 00 00 00`  → (value_a=0, value_b=3)
- 0x224F payload = `1c 01 00 00 0a 00 00 00 00 00 00 00`
                 → (build_stamp=284, value_b=10, value_c=0)

These are placeholders: if we ever learn what these fields MEAN we
rename the decoder fields, keep the tests, and move on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.unknowns import (
    UnknownChunkDecodeError,
    decode_unknown_224d,
    decode_unknown_224f,
)
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


def _read_chunk(chunk_id: int):
    blob = FNAF_EXE.read_bytes()
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    record = next(r for r in result.records if r.id == chunk_id)
    payload = read_chunk_payload(blob, record)
    return result, payload


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_unknown_224d_exact_values():
    """8-byte fixture lock: if these ever change, either the binary or
    the compression layer drifted."""
    _, payload = _read_chunk(0x224D)
    assert payload == bytes.fromhex("0000000003000000")
    chunk = decode_unknown_224d(payload)
    assert chunk.value_a == 0
    assert chunk.value_b == 3


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_unknown_224f_exact_values():
    _, payload = _read_chunk(0x224F)
    assert payload == bytes.fromhex("1c0100000a00000000000000")
    chunk = decode_unknown_224f(payload)
    assert chunk.build_stamp == 284
    assert chunk.value_b == 10
    assert chunk.value_c == 0


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_unknown_224f_build_stamp_matches_pack_header():
    """Cross-chunk antibody: 0x224F's first uint32 must equal the pack
    header's product_build. Ties the empirical chunk to the known-good
    pack metadata — a second, independent confirmation that 284 isn't a
    coincidence but a load-bearing value."""
    result, payload = _read_chunk(0x224F)
    chunk = decode_unknown_224f(payload)
    assert chunk.build_stamp == result.header.product_build == 284


def test_decoder_224d_rejects_wrong_length():
    with pytest.raises(UnknownChunkDecodeError, match="0x224D"):
        decode_unknown_224d(b"\x00" * 7)
    with pytest.raises(UnknownChunkDecodeError, match="0x224D"):
        decode_unknown_224d(b"\x00" * 9)


def test_decoder_224f_rejects_wrong_length():
    with pytest.raises(UnknownChunkDecodeError, match="0x224F"):
        decode_unknown_224f(b"\x00" * 11)
    with pytest.raises(UnknownChunkDecodeError, match="0x224F"):
        decode_unknown_224f(b"\x00" * 13)
