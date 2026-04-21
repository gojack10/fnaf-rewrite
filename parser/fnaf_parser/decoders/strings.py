"""String chunk decoder (probe #4.2).

Handles every chunk whose payload is a single null-terminated string:

    0x2224 Name            ("Five Nights at Freddy's")
    0x2225 Author          ("Scott Cawthon")
    0x222E App Editor-Filename
    0x222F App Target-Filename

Schema cross-checked against:
- CTFAK2.0 Core/CTFAK.Core/CCN/Chunks/StringChunk.cs
- Anaconda mmfparser/data/chunkloaders/stringchunk.py (pattern)

Both read `ReadYuniversal` — UTF-16LE if the pack is Unicode, otherwise
ASCII. FNAF 1's pack header reports Unicode=true (see pe_walker) so this
decoder takes unicode as a caller-provided flag rather than hiding a
hidden global; the unit tests below lock both paths.

The reference readers loop-until-NUL and ignore any trailing bytes past
the terminator (CTFAK: `while (b != 0) ...`). FNAF 1 has zero trailing
junk in any of the four chunks so we enforce a stricter antibody: the
payload must be exactly (len(str)+1) code units, i.e. one NUL terminator
and no unconsumed padding. A mismatch means either our decompressed
slice is wrong or the format has drifted — both worth surfacing loudly.
"""

from __future__ import annotations


class StringDecodeError(ValueError):
    """String chunk decode failure — preserves context for triage."""


def decode_string_chunk(payload: bytes, *, unicode: bool) -> str:
    """Decode a null-terminated string chunk payload.

    Antibody #2 (byte count): caller hands us `payload` already bounded
    by compression.read_chunk_payload; we require every byte to be part
    of either the string or its single NUL terminator, nothing else.
    Raises StringDecodeError if the invariant breaks.
    """
    if unicode:
        if len(payload) % 2 != 0:
            raise StringDecodeError(
                f"UTF-16 string payload must have even byte count, got {len(payload)}"
            )
        if len(payload) < 2:
            raise StringDecodeError(
                f"UTF-16 string payload must contain at least a NUL terminator "
                f"(2 bytes); got {len(payload)}"
            )
        # Expect exactly one NUL wide-char at the very end.
        if payload[-2:] != b"\x00\x00":
            raise StringDecodeError(
                f"UTF-16 string payload is not NUL-terminated: last 2 bytes = "
                f"{payload[-2:].hex()}"
            )
        body = payload[:-2]
        # No interior wide-NUL allowed — would mean we're reading past a
        # legitimate terminator into trailing junk, the exact silent-skip
        # bug the antibody exists to catch.
        for i in range(0, len(body), 2):
            if body[i : i + 2] == b"\x00\x00":
                raise StringDecodeError(
                    f"UTF-16 string payload has interior NUL at offset {i} "
                    f"(only a single trailing NUL is expected)"
                )
        try:
            return body.decode("utf-16le")
        except UnicodeDecodeError as exc:
            raise StringDecodeError(f"UTF-16LE decode failed: {exc}") from exc

    # ASCII path — kept for spec completeness even though FNAF 1 pack is
    # Unicode. Mirrors CTFAK's ReadAscii: single-byte code units, NUL
    # terminated, no trailing junk.
    if len(payload) < 1:
        raise StringDecodeError("ASCII string payload must contain at least a NUL terminator")
    if payload[-1:] != b"\x00":
        raise StringDecodeError(
            f"ASCII string payload is not NUL-terminated: last byte = "
            f"0x{payload[-1]:02x}"
        )
    body = payload[:-1]
    if b"\x00" in body:
        raise StringDecodeError(
            "ASCII string payload has interior NUL (only a single trailing "
            "NUL is expected)"
        )
    try:
        return body.decode("ascii")
    except UnicodeDecodeError as exc:
        raise StringDecodeError(f"ASCII decode failed: {exc}") from exc
