"""Chunk payload decompression (probe #4 onwards).

Probe #3's walker seeks past every payload without touching it. Probe #4
needs the payload bytes themselves. Clickteam encodes the TLV flags field
as a compression/encryption mode (CTFAK `ChunkFlags`); this module translates
that mode into a plain bytes object the decoders consume.

Format (per CTFAK Chunk.cs + Decompression.cs):

    flag = 0 (NotCompressed)
        payload is `size` raw bytes.

    flag = 1 (Compressed)
        payload is:
            decompSize: uint32 LE   (uncompressed size)
            compSize:   uint32 LE   (compressed size; == size - 8)
            blob:       zlib stream of `compSize` bytes
        decompress → `decompSize` bytes.

    flag = 2 (Encrypted) and flag = 3 (CompressedAndEncrypted)
        FNAF 1 does not use these. We raise NotImplementedError with enough
        context to tell which chunk tripped — Antibody #1 in spirit: the
        moment we see one, it's a priority probe, not a silent skip.
"""

from __future__ import annotations

import zlib
from enum import IntEnum

from fnaf_parser.chunk_walker import ChunkRecord

_COMPRESSED_HEADER_SIZE = 8  # two uint32 LE fields


class ChunkFlag(IntEnum):
    NOT_COMPRESSED = 0
    COMPRESSED = 1
    ENCRYPTED = 2
    COMPRESSED_AND_ENCRYPTED = 3


class ChunkPayloadError(ValueError):
    """Decompression/decryption layer errors.

    Includes chunk id, offset, and flag so the failure maps directly to a
    rizin/hexdump target.
    """


def read_chunk_payload(blob: bytes, record: ChunkRecord) -> bytes:
    """Return the *decoded* payload bytes for a chunk record.

    Handles the four flag modes. Compressed payloads are zlib-decompressed
    with a size cross-check (decompSize header vs actual output length) —
    catches truncated/corrupt zlib streams before downstream decoders see
    mismatched byte counts.

    `blob` is the full file contents; `record.offset` points at the TLV
    header, so the payload lives at `record.offset + 8` for `record.size`
    bytes.
    """
    from fnaf_parser.chunk_walker import CHUNK_HEADER_SIZE

    payload_start = record.offset + CHUNK_HEADER_SIZE
    payload_end = payload_start + record.size
    if payload_end > len(blob):
        raise ChunkPayloadError(
            f"chunk 0x{record.id:04X} at 0x{record.offset:x} claims size="
            f"{record.size} but file ends at 0x{len(blob):x}"
        )
    raw = blob[payload_start:payload_end]

    try:
        flag = ChunkFlag(record.flags)
    except ValueError as exc:
        raise ChunkPayloadError(
            f"chunk 0x{record.id:04X} at 0x{record.offset:x} has unknown "
            f"flags=0x{record.flags:04x} (expected 0..3)"
        ) from exc

    if flag is ChunkFlag.NOT_COMPRESSED:
        return raw

    if flag is ChunkFlag.COMPRESSED:
        if len(raw) < _COMPRESSED_HEADER_SIZE:
            raise ChunkPayloadError(
                f"chunk 0x{record.id:04X} at 0x{record.offset:x} flagged "
                f"Compressed but payload is only {len(raw)} bytes "
                f"(need ≥{_COMPRESSED_HEADER_SIZE})"
            )
        decomp_size = int.from_bytes(raw[:4], "little")
        comp_size = int.from_bytes(raw[4:8], "little")
        stream = raw[_COMPRESSED_HEADER_SIZE:]
        if comp_size != len(stream):
            raise ChunkPayloadError(
                f"chunk 0x{record.id:04X}: compSize={comp_size} header says "
                f"but {len(stream)} bytes remain in payload"
            )
        try:
            decompressed = zlib.decompress(stream)
        except zlib.error as exc:
            raise ChunkPayloadError(
                f"chunk 0x{record.id:04X} at 0x{record.offset:x}: "
                f"zlib decompress failed: {exc}"
            ) from exc
        if len(decompressed) != decomp_size:
            raise ChunkPayloadError(
                f"chunk 0x{record.id:04X}: decompSize={decomp_size} header "
                f"but {len(decompressed)} actual bytes decoded"
            )
        return decompressed

    # Encrypted / CompressedAndEncrypted — not used by FNAF 1 pack. If this
    # ever fires it's an interesting finding, not a bug in us.
    raise NotImplementedError(
        f"chunk 0x{record.id:04X} at 0x{record.offset:x} has flag={flag.name} "
        f"(0x{record.flags:04x}). FNAF 1 was not expected to use encrypted "
        f"chunks; decode paths for flags 2/3 are not implemented."
    )
