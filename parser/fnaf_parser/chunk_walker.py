"""Chunk-list walker (probe #3).

Walks the Clickteam Fusion 2.5 data-pack TLV structure from the PE data-pack
start to the LAST chunk (0x7F7F), without decoding payloads. Produces a
per-chunk-id frequency histogram that becomes the probe #4 decode queue.

Data-pack layout (empirically confirmed against FNAF1 + both references):

    offset 0x00101400  (FNAF 1; start of pack region)
    ┌─────────────────────────────────┐
    │ PACK_HEADER magic       (8)     │  77 77 77 77 49 87 47 12
    │ header_size  uint32             │  (always 32)
    │ data_size    uint32             │  PAMU offset = pack_start+data_size-32
    │ format_version uint32           │
    │ (pad)        uint32 x2          │
    │ count        uint32             │
    │ pack_files × count              │  associated-files blob, we skip whole
    ├─────────────────────────────────┤  pack_start + data_size - 32
    │ "PAME" or "PAMU" magic  (4)     │
    │ runtime_version   uint16        │
    │ runtime_subver    uint16        │
    │ product_version   uint32        │
    │ product_build     uint32        │
    ├─────────────────────────────────┤  GAME_HEADER_SIZE = 16 past PAMU start
    │ chunk[0]  id(i16) flags(u16)    │
    │           size(u32)             │
    │           payload(size bytes)   │
    │ chunk[1]  ...                   │
    │ ...                             │
    │ chunk[n]  id = 0x7F7F  (LAST)   │
    └─────────────────────────────────┘

The PACK_HEADER region is Clickteam's "associated files" blob (zlib DLL etc.
the runtime extracts at launch). Probe #3 does not need to decode pack files
— it just jumps past them using `data_size` and lands on the GAME_HEADER.

Parser Antibodies applied at line one:

- #1 strict mode — any chunk_id not in `chunk_ids.CHUNK_NAMES` raises.
- #6 loud skip log — every byte we seek past is tallied; total reported at
  end. Caller decides whether to surface it or suppress.

This walker does NOT decode chunk payloads. Container chunks (frames, etc.)
wrap nested chunk lists whose IDs are invisible at this level — that's a
known gap closed by probe #4 when decode-time descent grows the set.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from fnaf_parser.chunk_ids import (
    CHUNK_NAMES,
    CTFAK_ONLY,
    DUAL_CONFIRMED,
    EMPIRICAL,
    LAST_CHUNK_ID,
    is_known,
)

# TLV record header: id (int16, signed to match references) + flags (uint16)
# + size (uint32 LE).  Signed int16 is intentional: both C# (short) and
# Cython (short) use signed; for the FNAF 1 ID space (0x1122..0x7F7F) it's
# irrelevant, but any drift would still land in positive territory and we'd
# see it downstream.
_CHUNK_HEADER = struct.Struct("<hHI")
CHUNK_HEADER_SIZE = _CHUNK_HEADER.size  # 8 bytes

_GAME_HEADER = struct.Struct("<4sHHII")
GAME_HEADER_SIZE = _GAME_HEADER.size  # 16 bytes past PAMU start (4+2+2+4+4)

VALID_GAME_MAGICS = (b"PAME", b"PAMU")

# "Associated files" blob prefix. When present at pack_start, PAMU lives at
# pack_start + dataSize - 32 (formula cribbed from Anaconda's packdata.py and
# empirically verified on FNAF 1). headerSize is always 32 in the wild; we
# assert it so a format drift would surface loudly instead of mis-seeking.
PACK_HEADER_MAGIC = b"\x77\x77\x77\x77\x49\x87\x47\x12"
_PACK_PREFIX = struct.Struct("<8sII")  # magic(8) + headerSize(4) + dataSize(4)
_PACK_PREFIX_SIZE = _PACK_PREFIX.size  # 16
_EXPECTED_PACK_HEADER_SIZE = 32


class ChunkWalkError(ValueError):
    """Raised when probe #3 hits something the references don't cover.

    Includes the unknown chunk_id, the offset in-file where it was seen, and
    a running count of chunks already walked. All three are the minimum info
    needed to go look at the binary with rizin/hexdump.
    """


@dataclass(frozen=True)
class PackHeader:
    magic: bytes                 # b"PAME" or b"PAMU"
    unicode: bool                # PAMU variant?
    runtime_version: int
    runtime_subversion: int
    product_version: int
    product_build: int


@dataclass(frozen=True)
class ChunkRecord:
    """One TLV record as seen by the outer walker.

    offset is the absolute file offset of the record's header (so payload
    starts at offset + CHUNK_HEADER_SIZE). `flags` is kept for Antibody
    debugging later — compression/encryption modes (see CTFAK Chunk.cs).
    """
    id: int
    flags: int
    size: int
    offset: int


@dataclass(frozen=True)
class ChunkWalkResult:
    pack_start: int              # file offset of the pack region (PACK_HEADER or PAMU)
    pamu_offset: int             # file offset where PAME/PAMU magic begins
    pack_header_present: bool    # True if "associated files" prefix was skipped
    pack_data_size: int          # dataSize field from PACK_HEADER, 0 if absent
    header: PackHeader
    records: tuple[ChunkRecord, ...]
    total_payload_bytes: int     # Antibody #6 running tally
    end_offset: int              # absolute file offset past the LAST chunk's header
    file_size: int
    reached_last: bool           # True if we terminated on 0x7F7F LAST


def walk_chunks(path: Path, *, pack_start: int) -> ChunkWalkResult:
    """Walk TLV records from `pack_start` until LAST or EOF.

    Strict mode (Antibody #1): any chunk_id not in CHUNK_NAMES raises
    ChunkWalkError immediately. No silent skips.
    """
    blob = Path(path).read_bytes()
    file_size = len(blob)

    if pack_start + _PACK_PREFIX_SIZE > file_size:
        raise ChunkWalkError(
            f"pack_start=0x{pack_start:x} leaves <16 bytes for pack/game "
            f"header (file size=0x{file_size:x})"
        )

    # Optional PACK_HEADER ("associated files" blob). If present, its dataSize
    # field tells us where PAMU lives; otherwise pack_start IS the PAMU offset.
    if blob[pack_start : pack_start + 8] == PACK_HEADER_MAGIC:
        _, hdr_size, data_size = _PACK_PREFIX.unpack_from(blob, pack_start)
        if hdr_size != _EXPECTED_PACK_HEADER_SIZE:
            raise ChunkWalkError(
                f"PACK_HEADER at 0x{pack_start:x} has headerSize={hdr_size}, "
                f"expected {_EXPECTED_PACK_HEADER_SIZE}. Format drift?"
            )
        pamu_offset = pack_start + data_size - 32
        pack_header_present = True
        pack_data_size = data_size
    else:
        pamu_offset = pack_start
        pack_header_present = False
        pack_data_size = 0

    if pamu_offset + GAME_HEADER_SIZE > file_size:
        raise ChunkWalkError(
            f"pamu_offset=0x{pamu_offset:x} leaves <{GAME_HEADER_SIZE} bytes "
            f"for game header (file size=0x{file_size:x})"
        )

    magic, rt_ver, rt_sub, prod_ver, prod_build = _GAME_HEADER.unpack_from(
        blob, pamu_offset
    )
    if magic not in VALID_GAME_MAGICS:
        raise ChunkWalkError(
            f"expected PAME/PAMU magic at 0x{pamu_offset:x}, got {magic!r}"
        )
    header = PackHeader(
        magic=magic,
        unicode=(magic == b"PAMU"),
        runtime_version=rt_ver,
        runtime_subversion=rt_sub,
        product_version=prod_ver,
        product_build=prod_build,
    )

    records: list[ChunkRecord] = []
    total_payload = 0
    cursor = pamu_offset + GAME_HEADER_SIZE
    reached_last = False

    while cursor + CHUNK_HEADER_SIZE <= file_size:
        chunk_id, flags, size = _CHUNK_HEADER.unpack_from(blob, cursor)

        if not is_known(chunk_id):
            raise ChunkWalkError(
                f"unknown chunk_id 0x{chunk_id:04X} at offset 0x{cursor:x} "
                f"(record #{len(records)}, flags=0x{flags:04x}, size={size}). "
                "Not in CTFAK2.0 or Anaconda reference; Antibody #1 rejects "
                "silent skips. Either the binary is unusual or the walker "
                "is misaligned; check nested container handling."
            )

        records.append(
            ChunkRecord(id=chunk_id, flags=flags, size=size, offset=cursor)
        )

        if chunk_id == LAST_CHUNK_ID:
            reached_last = True
            # Advance past the LAST header itself; LAST.size is typically 0
            # but we tally whatever it claims for consistency.
            total_payload += size
            cursor += CHUNK_HEADER_SIZE + size
            break

        # Seek past payload without decoding. Antibody #6: tally every byte.
        if cursor + CHUNK_HEADER_SIZE + size > file_size:
            raise ChunkWalkError(
                f"chunk 0x{chunk_id:04X} at 0x{cursor:x} claims size={size} "
                f"but only {file_size - (cursor + CHUNK_HEADER_SIZE)} bytes "
                f"remain in file (EOF=0x{file_size:x})"
            )
        total_payload += size
        cursor += CHUNK_HEADER_SIZE + size

    return ChunkWalkResult(
        pack_start=pack_start,
        pamu_offset=pamu_offset,
        pack_header_present=pack_header_present,
        pack_data_size=pack_data_size,
        header=header,
        records=tuple(records),
        total_payload_bytes=total_payload,
        end_offset=cursor,
        file_size=file_size,
        reached_last=reached_last,
    )


@dataclass(frozen=True)
class ChunkFrequency:
    id: int
    id_hex: str
    name: str
    confidence: str              # "dual" | "ctfak-only" | "empirical"
    count: int
    first_offset: int            # file offset of earliest occurrence
    total_bytes: int             # sum of payload sizes across all occurrences


def _confidence_for(chunk_id: int) -> str:
    if chunk_id in EMPIRICAL:
        return "empirical"
    if chunk_id in DUAL_CONFIRMED:
        return "dual"
    return "ctfak-only"


def chunk_histogram(records: tuple[ChunkRecord, ...]) -> list[ChunkFrequency]:
    """Aggregate records by chunk_id, sorted by count descending.

    Sorted by count desc so the probe #4 decode queue is hottest-first.
    Ties broken by id asc for stable output across runs (matters for the
    snapshot-test Antibody #7 in a future probe).
    """
    seen: dict[int, dict[str, int]] = {}
    for rec in records:
        slot = seen.setdefault(
            rec.id,
            {"count": 0, "first_offset": rec.offset, "total_bytes": 0},
        )
        slot["count"] += 1
        slot["total_bytes"] += rec.size

    out: list[ChunkFrequency] = []
    for chunk_id, stats in seen.items():
        out.append(
            ChunkFrequency(
                id=chunk_id,
                id_hex=f"0x{chunk_id:04X}",
                name=CHUNK_NAMES.get(chunk_id, "(unknown)"),
                confidence=_confidence_for(chunk_id),
                count=stats["count"],
                first_offset=stats["first_offset"],
                total_bytes=stats["total_bytes"],
            )
        )

    out.sort(key=lambda f: (-f.count, f.id))
    return out


def histogram_to_json_payload(
    result: ChunkWalkResult, freqs: list[ChunkFrequency]
) -> dict:
    """JSON-serializable dict for chunks_seen.json.

    Schema choice: top-level object with pack metadata + `chunks` array.
    Prettified (caller uses json.dump indent=2), stable order (sorted by
    frequency then id), all offsets as hex strings for readability.
    """
    return {
        "pack": {
            "start": f"0x{result.pack_start:08X}",
            "pamu_offset": f"0x{result.pamu_offset:08X}",
            "pack_header_present": result.pack_header_present,
            "pack_data_size": result.pack_data_size,
            "magic": result.header.magic.decode("ascii"),
            "unicode": result.header.unicode,
            "runtime_version": f"0x{result.header.runtime_version:04X}",
            "runtime_subversion": f"0x{result.header.runtime_subversion:04X}",
            "product_version": result.header.product_version,
            "product_build": result.header.product_build,
        },
        "walk": {
            "total_records": len(result.records),
            "unique_chunk_ids": len(freqs),
            "total_payload_bytes": result.total_payload_bytes,
            "end_offset": f"0x{result.end_offset:08X}",
            "file_size": f"0x{result.file_size:08X}",
            "reached_last": result.reached_last,
            "trailing_bytes_after_last": result.file_size - result.end_offset,
            "dual_confirmed_ids_seen": sum(
                1 for f in freqs if f.confidence == "dual"
            ),
            "ctfak_only_ids_seen": sum(
                1 for f in freqs if f.confidence == "ctfak-only"
            ),
            "empirical_ids_seen": sum(
                1 for f in freqs if f.confidence == "empirical"
            ),
        },
        "chunks": [
            {
                "id": f.id_hex,
                "name": f.name,
                "confidence": f.confidence,
                "count": f.count,
                "first_offset": f"0x{f.first_offset:08X}",
                "total_bytes": f.total_bytes,
            }
            for f in freqs
        ],
    }
