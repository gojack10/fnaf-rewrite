"""Chunk payload decompression + decryption (probes #4, #4.5).

Probe #3's walker seeks past every payload without touching it. Probe #4
needs the payload bytes themselves. Clickteam encodes the TLV flags field
as a compression/encryption mode (CTFAK `ChunkFlags`); this module translates
that mode into a plain bytes object the decoders consume.

Format (per CTFAK Chunk.cs + Decompression.cs + Decryption.cs):

    flag = 0 (NotCompressed)
        payload is `size` raw bytes.

    flag = 1 (Compressed)
        payload is:
            decompSize: uint32 LE   (uncompressed size)
            compSize:   uint32 LE   (compressed size; == size - 8)
            blob:       zlib stream of `compSize` bytes
        decompress → `decompSize` bytes.

    flag = 2 (Encrypted)
        optional odd-id first-byte XOR (build-gated), then transform the
        whole payload against the pack-level S-box. No zlib.

    flag = 3 (CompressedAndEncrypted)
        payload is:
            decompSize: uint32 LE       (plaintext, NOT encrypted)
            rest:       encrypted blob
        decrypt `rest` against the pack S-box, optionally XOR its first
        byte for odd chunk ids (build > 284), then parse the decrypted
        blob as [compSize: u32 LE][zlib stream] and decompress.

Flags 2/3 require a pack-level `TransformState` (see encryption.py). If
the caller does not have one wired up yet, the module raises
`ChunkPayloadError` with enough context to trace the chunk back to its
file offset rather than silently skipping.
"""

from __future__ import annotations

import zlib
from enum import IntEnum

from fnaf_parser.chunk_walker import ChunkRecord
from fnaf_parser.encryption import TransformState, apply_odd_id_xor

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


def _decompress_stream(stream: bytes, expected: int, chunk_id: int) -> bytes:
    """zlib-decompress and verify the decompressed-size field matches."""
    try:
        decompressed = zlib.decompress(stream)
    except zlib.error as exc:
        raise ChunkPayloadError(
            f"chunk 0x{chunk_id:04X}: zlib decompress failed: {exc}"
        ) from exc
    if len(decompressed) != expected:
        raise ChunkPayloadError(
            f"chunk 0x{chunk_id:04X}: decompSize={expected} header "
            f"but {len(decompressed)} actual bytes decoded"
        )
    return decompressed


def decompress_payload_bytes(
    raw: bytes,
    *,
    flags: int,
    chunk_id: int,
    transform: TransformState | None = None,
) -> bytes:
    """Decode a raw TLV payload into its plaintext bytes.

    Outer chunks and Frame-container sub-chunks share this same flag→mode
    encoding (CTFAK Chunk.cs), so the decompression logic lives here — one
    place to reason about size bookkeeping, zlib errors, and the decrypt
    layer's size framing.

    Flags 2 / 3 require `transform` (a pack-level `TransformState` built
    from the editor/name/copyright seed). Raising instead of silently
    returning raw ciphertext enforces Antibody #1 — an encrypted chunk
    we cannot decrypt is a pipeline-configuration bug, not a skip.
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
        return _decompress_stream(stream, decomp_size, chunk_id)

    if transform is None:
        raise ChunkPayloadError(
            f"chunk 0x{chunk_id:04X} has flag={flag.name} (0x{flags:04x}) "
            f"but no TransformState was supplied. Build one via "
            f"encryption.make_transform(editor=..., name=..., "
            f"copyright_str=..., build=...) from the pack's app metadata "
            f"and pass it through."
        )

    if flag is ChunkFlag.ENCRYPTED:
        body = apply_odd_id_xor(raw, chunk_id=chunk_id, build=transform.build)
        return transform.transform(body)

    # CompressedAndEncrypted: [decompSize u32 plaintext][encrypted blob]
    # where the encrypted blob decrypts to [compSize u32][zlib stream].
    if len(raw) < 4:
        raise ChunkPayloadError(
            f"chunk 0x{chunk_id:04X} flagged CompressedAndEncrypted but "
            f"payload is only {len(raw)} bytes (need ≥4 for decompSize)"
        )
    decomp_size = int.from_bytes(raw[:4], "little")
    ciphertext = apply_odd_id_xor(
        raw[4:], chunk_id=chunk_id, build=transform.build
    )
    plaintext = transform.transform(ciphertext)
    if len(plaintext) < 4:
        raise ChunkPayloadError(
            f"chunk 0x{chunk_id:04X} decrypted body is {len(plaintext)} "
            f"bytes, need ≥4 for embedded compSize"
        )
    comp_size = int.from_bytes(plaintext[:4], "little")
    stream = plaintext[4:]
    if comp_size > len(stream):
        raise ChunkPayloadError(
            f"chunk 0x{chunk_id:04X}: embedded compSize={comp_size} but "
            f"only {len(stream)} bytes decrypted — decryption likely "
            f"misaligned (wrong key?)"
        )
    # Some CTFAK captures show compSize <= len(stream) with trailing
    # padding; feed zlib only the declared window to avoid spurious noise.
    return _decompress_stream(stream[:comp_size], decomp_size, chunk_id)


def read_chunk_payload(
    blob: bytes,
    record: ChunkRecord,
    *,
    transform: TransformState | None = None,
) -> bytes:
    """Return the *decoded* payload bytes for a chunk record.

    `blob` is the full file contents; `record.offset` points at the TLV
    header, so the payload lives at `record.offset + 8` for `record.size`
    bytes. Threads `transform` through to `decompress_payload_bytes` so
    callers that already hold a pack-level `TransformState` get flag-2
    and flag-3 decoding for free.
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
    return decompress_payload_bytes(
        raw, flags=record.flags, chunk_id=record.id, transform=transform
    )
