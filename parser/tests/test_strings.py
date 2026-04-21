"""Regression tests for string chunk decoder (probe #4.2).

Antibody coverage:

- #2 byte-count — payload must be exactly string + one NUL terminator.
  Truncated, over-sized, or non-NUL-terminated payloads raise rather
  than silently decoding to a shorter / garbage string.
- #5 multi-input — same decoder drives four chunks: Name, Author,
  Editor-Filename, Target-Filename. A format-drift bug would break at
  least one of them.

FNAF 1 ground-truth locked here:

- 0x2224 Name                   = "Five Nights at Freddy's"
- 0x2225 Author                 = "Scott Cawthon"
- 0x222E App Editor-Filename    = 'C:\\Users\\Scott\\Desktop\\Five Nights\\FiveNights-55.mfa'
- 0x222F App Target-Filename    = 'C:\\Users\\Scott\\Desktop\\sdk\\tools\\ContentBuilder\\content\\windows_content\\FiveNightsatFreddys.exe'

These came directly from decompressing the binary; if any changes,
either the binary, the decompression layer, or the string decoder has
drifted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.strings import StringDecodeError, decode_string_chunk
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"

_STRING_CHUNK_EXPECTED: dict[int, str] = {
    0x2224: "Five Nights at Freddy's",
    0x2225: "Scott Cawthon",
    0x222E: r"C:\Users\Scott\Desktop\Five Nights\FiveNights-55.mfa",
    0x222F: r"C:\Users\Scott\Desktop\sdk\tools\ContentBuilder\content\windows_content\FiveNightsatFreddys.exe",
}


def _decode_fnaf1_string(chunk_id: int) -> str:
    blob = FNAF_EXE.read_bytes()
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    record = next(r for r in result.records if r.id == chunk_id)
    payload = read_chunk_payload(blob, record)
    return decode_string_chunk(payload, unicode=True)


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
@pytest.mark.parametrize("chunk_id,expected", sorted(_STRING_CHUNK_EXPECTED.items()))
def test_fnaf1_string_chunk_exact_value(chunk_id: int, expected: str):
    """Multi-input antibody: one decoder handles all four string chunks
    and must produce the exact known value for each."""
    actual = _decode_fnaf1_string(chunk_id)
    assert actual == expected, (
        f"0x{chunk_id:04X}: expected {expected!r}, got {actual!r}"
    )


def test_unicode_decoder_rejects_odd_byte_count():
    """UTF-16 payloads must have even byte count; odd count is a red flag
    that the decompression layer handed us a truncated slice."""
    with pytest.raises(StringDecodeError, match="even byte count"):
        decode_string_chunk(b"\x41\x00\x42", unicode=True)


def test_unicode_decoder_rejects_missing_terminator():
    """String chunks are NUL-terminated on the wire. Missing terminator
    means we're reading a slice that doesn't contain a complete string."""
    # "Hi" in UTF-16LE with NO terminator.
    payload = b"\x48\x00\x69\x00"
    with pytest.raises(StringDecodeError, match="not NUL-terminated"):
        decode_string_chunk(payload, unicode=True)


def test_unicode_decoder_rejects_interior_nul():
    """Anything past the first NUL is either junk or a different string —
    either way, silently truncating would be a multi-value bug waiting
    to happen. Raise instead."""
    # "A\0B\0" — NUL in the middle followed by more characters then NUL.
    payload = b"\x41\x00\x00\x00\x42\x00\x00\x00"
    with pytest.raises(StringDecodeError, match="interior NUL"):
        decode_string_chunk(payload, unicode=True)


def test_unicode_decoder_accepts_empty_string():
    """A payload of just the NUL terminator is a legitimate empty string
    and must decode cleanly — some chunks in other games are empty."""
    assert decode_string_chunk(b"\x00\x00", unicode=True) == ""


def test_unicode_decoder_rejects_too_short_payload():
    """Below 2 bytes can't even contain a terminator. Using empty bytes
    so we don't also trip the odd-length check first."""
    with pytest.raises(StringDecodeError, match="at least a NUL terminator"):
        decode_string_chunk(b"", unicode=True)


def test_ascii_decoder_roundtrip():
    """ASCII path exists for non-Unicode packs. Lock it via a synthetic
    payload — FNAF 1 doesn't exercise this path."""
    assert decode_string_chunk(b"Hello\x00", unicode=False) == "Hello"


def test_ascii_decoder_rejects_missing_terminator():
    with pytest.raises(StringDecodeError, match="not NUL-terminated"):
        decode_string_chunk(b"Hello", unicode=False)


def test_ascii_decoder_rejects_interior_nul():
    with pytest.raises(StringDecodeError, match="interior NUL"):
        decode_string_chunk(b"Hi\x00there\x00", unicode=False)
