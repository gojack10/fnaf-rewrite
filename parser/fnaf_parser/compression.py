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


def decompress_payload_bytes(raw: bytes, *, flags: int, chunk_id: int) -> bytes:
    """Decode a raw TLV payload into its plaintext bytes.

    Outer chunks and Frame-container sub-chunks share this same flag→mode
    encoding (CTFAK Chunk.cs), so the decompression logic lives here — one
    place to reason about size bookkeeping and zlib errors.

    Supports flags 0 (NotCompressed) and 1 (Compressed). Flags 2 and 3
    (Encrypted / CompressedAndEncrypted) require a keyed RC4-like transform
    seeded off (editor, name, copyright) from the app-metadata chunks —
    deferred to probe #4.5. Raises NotImplementedError with enough context
    to identify the offending chunk.
    """
    try:
        flag = ChunkFlag(flags)
    except ValueError as exc:
        raise ChunkPayloadError(
            f"chunk 0x{chunk_id:04X} has unknown flags=0x{flags:04x} "
            f"(expected 0..3)"
        ) from exc

    if flag is ChunkFlag.NOT_COMPRESSED:
        return raw

    if flag is ChunkFlag.COMPRESSED:
        if len(raw) < _COMPRESSED_HEADER_SIZE:
            raise ChunkPayloadError(
                f"chunk 0x{chunk_id:04X} flagged Compressed but payload is "
                f"only {len(raw)} bytes (need ≥{_COMPRESSED_HEADER_SIZE})"
            )
        decomp_size = int.from_bytes(raw[:4], "little")
        comp_size = int.from_bytes(raw[4:8], "little")
        stream = raw[_COMPRESSED_HEADER_SIZE:]
        if comp_size != len(stream):
            raise ChunkPayloadError(
                f"chunk 0x{chunk_id:04X}: compSize={comp_size} header says "
                f"but {len(stream)} bytes remain in payload"
            )
        try:
            decompressed = zlib.decompress(stream)
        except zlib.error as exc:
            raise ChunkPayloadError(
                f"chunk 0x{chunk_id:04X}: zlib decompress failed: {exc}"
            ) from exc
        if len(decompressed) != decomp_size:
            raise ChunkPayloadError(
                f"chunk 0x{chunk_id:04X}: decompSize={decomp_size} header "
                f"but {len(decompressed)} actual bytes decoded"
            )
        return decompressed

    raise NotImplementedError(
        f"chunk 0x{chunk_id:04X} has flag={flag.name} (0x{flags:04x}). "
        f"Encrypted chunk decoding is deferred to probe #4.5 (Clickteam "
        f"RC4-like transform seeded from editor+name+copyright)."
    )


def read_chunk_payload(blob: bytes, record: ChunkRecord) -> bytes:
    """Return the *decoded* payload bytes for a chunk record.

    `blob` is the full file contents; `record.offset` points at the TLV
    header, so the payload lives at `record.offset + 8` for `record.size`
    bytes. Delegates the flag-specific decoding to
    `decompress_payload_bytes` so sub-chunk decoders share the same path.
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
    return decompress_payload_bytes(raw, flags=record.flags, chunk_id=record.id)
