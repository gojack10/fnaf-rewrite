"""Regression tests for 0x2245 ExtendedHeader decoder (probe #4.2).

Antibody coverage:

- #2 byte-count — 20-byte payload must decode to 20 bytes consumed.
- #4 multi-oracle — only CTFAK defines this chunk; Anaconda has no
  second opinion. The FNAF 1 invariants below act as the empirical
  oracle instead.

FNAF 1 ground-truth invariants:

- build_type == 0 (Windows EXE Application). Cross-reference with the
  pack header's product_version=3/build=284 (see chunks_seen.json) —
  a packaged Windows exe.
- Raw payload bytes (post-decompression, 20 bytes):
    0200000000000000 240010000800 00000000 0000
  → flags=0x00000002, build_type=0, pad=000000,
    compression_flags=0x00100024, screen_ratio=8, screen_angle=0,
    view_flags=0, new_flags=0.
- 3 CompressionFlags bits are set: IncludeExternalFiles (bit 2),
  Bit5, Bit20. CTFAK does not name bits 5/20; we surface them as
  Bit5/Bit20 rather than silently dropping — an unknown-set-bit is
  worth knowing about when format drift happens.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.extended_header import (
    EXTENDED_HEADER_SIZE,
    ExtendedHeaderDecodeError,
    decode_extended_header,
)
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"

_EXT_HEADER_CHUNK_ID = 0x2245


def _read_fnaf_ext_header():
    blob = FNAF_EXE.read_bytes()
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    record = next(r for r in result.records if r.id == _EXT_HEADER_CHUNK_ID)
    payload = read_chunk_payload(blob, record)
    return record, payload, decode_extended_header(payload)


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_ext_header_decompresses_to_20_bytes():
    _, payload, _ = _read_fnaf_ext_header()
    assert len(payload) == EXTENDED_HEADER_SIZE == 20


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_ext_header_is_windows_exe():
    """The packaged binary IS a Windows EXE, so build_type must reflect
    that. If this drifts we're either looking at a different game or a
    repack."""
    _, _, hdr = _read_fnaf_ext_header()
    assert hdr.build_type == 0
    assert hdr.build_type_label() == "Windows EXE Application"


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_ext_header_exact_field_values():
    """Lock the full struct. Cheap change detector: any single-byte
    drift in the payload flips exactly one assertion here."""
    _, _, hdr = _read_fnaf_ext_header()
    assert hdr.flags == 0x00000002
    assert hdr.compression_flags == 0x00100024
    assert hdr.screen_ratio == 8
    assert hdr.screen_angle == 0
    assert hdr.view_flags == 0
    assert hdr.new_flags == 0


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_ext_header_compression_flag_includes_external_files():
    """IncludeExternalFiles is the only named CTFAK bit set in FNAF 1's
    compression_flags. Its presence is a characteristic marker of a
    packaged-with-assets Clickteam build."""
    _, _, hdr = _read_fnaf_ext_header()
    names = hdr.compression_flag_names_set()
    assert "IncludeExternalFiles" in names


def test_decoder_rejects_wrong_length_payload():
    """Antibody #2: length mismatch raises, never silently truncates."""
    with pytest.raises(ExtendedHeaderDecodeError, match="byte count"):
        decode_extended_header(b"\x00" * (EXTENDED_HEADER_SIZE - 1))
    with pytest.raises(ExtendedHeaderDecodeError, match="byte count"):
        decode_extended_header(b"\x00" * (EXTENDED_HEADER_SIZE + 1))


def test_decoder_unknown_build_type_gets_labelled():
    """A build_type value outside the CTFAK enum must still decode and be
    labelled `unknown(N)` — we never want a KeyError for format drift."""
    # Synthetic 20-byte payload with build_type=99.
    payload = (
        b"\x00\x00\x00\x00"  # flags
        b"\x63"              # build_type = 99
        b"\x00\x00\x00"      # pad
        b"\x00\x00\x00\x00"  # compression_flags
        b"\x00\x00"          # screen_ratio
        b"\x00\x00"          # screen_angle
        b"\x00\x00"          # view_flags
        b"\x00\x00"          # new_flags
    )
    hdr = decode_extended_header(payload)
    assert hdr.build_type == 99
    assert hdr.build_type_label() == "unknown(99)"
