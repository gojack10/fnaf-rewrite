"""0x3333 Frame container walker (probe #4.4).

A Frame chunk is not a flat record — it's a nested TLV stream of
sub-chunks terminated by 0x7F7F (LAST), mirroring the outer ChunkList
structure at the file level. This module walks that inner stream and
decodes the sub-chunks we can, without yet implementing Clickteam's
RC4-like encryption transform.

Sub-chunk encoding (per CTFAK Chunk.cs, cross-checked against Anaconda
mmfparser/data/chunk.pyx):

    id      int16 LE
    flags   uint16 LE     (ChunkFlags: 0=NotCompressed, 1=Compressed,
                           2=Encrypted, 3=CompressedAndEncrypted)
    size    uint32 LE
    raw     size bytes

FNAF 1 frame sub-chunk flag distribution (empirical):

    flag=0 (plain)          : 0x3347 MvtTimerBase and occasionally 0x3335
    flag=1 (zlib)           : 0x3335 FrameName (usual), 0x3349 FrameEffects
    flag=3 (zlib+encrypted) : the rest — FrameHeader, VirtualRect, Palette,
                              Layers, LayerEffects, ItemInstances, Events,
                              FadeIn/Out

Scope of probe #4.4:
- Walk sub-chunks in strict mode (Antibody #1): every id must be known.
- Decode flag 0/1 sub-chunks (FrameName, MvtTimerBase, FrameEffects).
- Capture encrypted sub-chunks as `FrameSubChunkRecord` with
  `decoded_payload=None` and surface them in `Frame.deferred_encrypted`
  (Antibody #6 loud-skip — counted, not silently ignored).

Cross-chunk antibody (#4 multi-oracle): the number of Frame chunks in the
outer pack must equal `AppHeader.number_of_frames`. Enforced at the test
layer, not here, because this module is scoped to a single frame payload.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from fnaf_parser.chunk_ids import CHUNK_NAMES, LAST_CHUNK_ID
from fnaf_parser.compression import ChunkFlag, decompress_payload_bytes
from fnaf_parser.decoders.frame_palette import FramePalette, decode_frame_palette
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import TransformState

# Sub-chunk IDs appearing inside a 0x3333 Frame container. Named here so
# grep locates all the places that know about them; the canonical label
# still comes from chunk_ids.CHUNK_NAMES.
SUB_FRAME_HEADER = 0x3334
SUB_FRAME_NAME = 0x3335
SUB_FRAME_PASSWORD = 0x3336
SUB_FRAME_PALETTE = 0x3337
SUB_FRAME_ITEM_INSTANCES = 0x3338
SUB_FRAME_FADE_IN_FRAME = 0x3339
SUB_FRAME_FADE_OUT_FRAME = 0x333A
SUB_FRAME_FADE_IN = 0x333B
SUB_FRAME_FADE_OUT = 0x333C
SUB_FRAME_EVENTS = 0x333D
SUB_FRAME_PLAY_HEADER = 0x333E
SUB_ADDITIONAL_FRAME_ITEM = 0x333F
SUB_ADDITIONAL_FRAME_ITEM_INSTANCE = 0x3340
SUB_FRAME_LAYERS = 0x3341
SUB_FRAME_VIRTUAL_RECT = 0x3342
SUB_DEMO_FILE_PATH = 0x3343
SUB_RANDOM_SEED = 0x3344
SUB_FRAME_LAYER_EFFECTS = 0x3345
SUB_BLU_RAY_FRAME_OPTIONS = 0x3346
SUB_MVT_TIMER_BASE = 0x3347
SUB_MOSAIC_IMAGE_TABLE = 0x3348
SUB_FRAME_EFFECTS = 0x3349
SUB_FRAME_IPHONE_OPTIONS = 0x334A

_SUB_HEADER = struct.Struct("<hHI")
SUB_HEADER_SIZE = _SUB_HEADER.size  # 8, same shape as outer TLV


class FrameDecodeError(ValueError):
    """Frame container walk / sub-chunk decode failure.

    Always carries the offending sub-chunk id and offset-within-payload so
    the failure maps directly to a hexdump position inside the frame's
    decompressed bytes (not the file).
    """


@dataclass(frozen=True)
class FrameSubChunkRecord:
    """One sub-chunk found inside a Frame container.

    `raw` is the pre-decompression bytes. `decoded_payload` is the
    plaintext bytes after applying the flag's decoding (None when the flag
    requires the encryption transform we have not implemented yet).
    """
    id: int
    flags: int
    size: int                # on-disk size (raw length)
    inner_offset: int        # offset within the frame's decompressed payload
    raw: bytes = field(repr=False)
    decoded_payload: bytes | None = field(repr=False)

    @property
    def is_encrypted(self) -> bool:
        # Flags 2 (Encrypted) and 3 (CompressedAndEncrypted) both set bit 1.
        return (self.flags & ChunkFlag.ENCRYPTED) != 0


@dataclass(frozen=True)
class Frame:
    """One decoded 0x3333 Frame container.

    Carries every sub-chunk record we saw (so the raw bytes are always
    recoverable) plus the decoded-payload shortcuts for the sub-chunks we
    actually understand. When the caller supplies a `TransformState`
    encrypted sub-chunks are decrypted in-place and land in
    `sub_records[i].decoded_payload`; `deferred_encrypted` then lists only
    the sub-chunks we had a TransformState for but whose decoded payload
    the top-level `decode_frame` has not yet learned to parse.

    Without a TransformState, `deferred_encrypted` is the Antibody #6
    loud-skip list (sub-chunks present-but-not-yet-decrypted).
    """
    sub_records: tuple[FrameSubChunkRecord, ...]
    name: str | None
    mvt_timer_base: int | None
    palette: FramePalette | None
    deferred_encrypted: tuple[FrameSubChunkRecord, ...]

    def sub_by_id(self, chunk_id: int) -> FrameSubChunkRecord | None:
        for r in self.sub_records:
            if r.id == chunk_id:
                return r
        return None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "mvt_timer_base": self.mvt_timer_base,
            "palette": self.palette.as_dict() if self.palette is not None else None,
            "sub_chunks": [
                {
                    "id": f"0x{r.id:04X}",
                    "flags": r.flags,
                    "size": r.size,
                    "decoded": r.decoded_payload is not None,
                }
                for r in self.sub_records
            ],
            "deferred_encrypted": [
                f"0x{r.id:04X}" for r in self.deferred_encrypted
            ],
        }


def walk_frame_payload(
    payload: bytes, *, transform: TransformState | None = None
) -> tuple[FrameSubChunkRecord, ...]:
    """Walk the inner TLV stream of a Frame container's decompressed payload.

    Strict mode (Antibody #1): any sub-chunk id not in CHUNK_NAMES raises
    FrameDecodeError immediately. Terminates on 0x7F7F LAST or end-of-buffer.
    For flag 0/1 sub-chunks, also runs the decompression so the caller gets
    plaintext bytes. For flag 2/3 sub-chunks: when `transform` is supplied,
    decrypt (and decompress for flag 3) via the shared compression layer;
    when it is absent, leave `decoded_payload=None` so the caller can still
    surface the sub-chunk as deferred (Antibody #6 loud-skip).
    """
    records: list[FrameSubChunkRecord] = []
    pos = 0
    n = len(payload)

    while pos + SUB_HEADER_SIZE <= n:
        sid, sflags, ssize = _SUB_HEADER.unpack_from(payload, pos)

        if sid not in CHUNK_NAMES:
            raise FrameDecodeError(
                f"unknown sub-chunk id 0x{sid:04X} at frame-offset "
                f"0x{pos:x} (flags=0x{sflags:04x}, size={ssize}). "
                "Antibody #1: strict mode rejects silent skips."
            )

        raw_start = pos + SUB_HEADER_SIZE
        raw_end = raw_start + ssize
        if raw_end > n:
            raise FrameDecodeError(
                f"sub-chunk 0x{sid:04X} at frame-offset 0x{pos:x} claims "
                f"size={ssize} but only {n - raw_start} bytes remain in the "
                f"frame payload (payload size={n})"
            )
        raw = payload[raw_start:raw_end]

        decoded: bytes | None
        if (sflags & ChunkFlag.ENCRYPTED) != 0 and transform is None:
            # Deferred: caller did not hand us a TransformState. Keep raw
            # so a later pass can re-process once the key is available.
            decoded = None
        else:
            decoded = decompress_payload_bytes(
                raw, flags=sflags, chunk_id=sid, transform=transform
            )

        records.append(
            FrameSubChunkRecord(
                id=sid,
                flags=sflags,
                size=ssize,
                inner_offset=pos,
                raw=raw,
                decoded_payload=decoded,
            )
        )

        if sid == LAST_CHUNK_ID:
            # Don't advance past LAST — we're done.
            break

        pos = raw_end

    return tuple(records)


def decode_frame(
    payload: bytes,
    *,
    unicode: bool = True,
    transform: TransformState | None = None,
) -> Frame:
    """Decode a 0x3333 Frame container's decompressed payload.

    Walks the inner TLV stream (see walk_frame_payload) and extracts every
    sub-chunk we can at the current probe scope:
    - Frame Name (0x3335): string, flag 0 or 1 in FNAF 1.
    - Mvt Timer Base (0x3347): int32 LE, always flag 0 in FNAF 1.

    When `transform` is supplied, encrypted sub-chunks (flags 2/3) are
    decrypted during the walk. Their `decoded_payload` then holds the
    plaintext bytes; `deferred_encrypted` lists only those we do not yet
    know how to interpret at this probe scope. Without a transform, every
    encrypted sub-chunk is surfaced in `deferred_encrypted` untouched.
    """
    sub_records = walk_frame_payload(payload, transform=transform)

    name: str | None = None
    mvt_timer_base: int | None = None
    palette: FramePalette | None = None
    deferred: list[FrameSubChunkRecord] = []

    for rec in sub_records:
        if rec.is_encrypted and rec.decoded_payload is None:
            deferred.append(rec)
            continue

        if rec.id == SUB_FRAME_NAME:
            if rec.decoded_payload is None:
                # Shouldn't happen: non-encrypted means we decompressed it.
                raise FrameDecodeError(
                    "Frame Name (0x3335) sub-chunk had no decoded payload "
                    "despite non-encrypted flag — decoder state is inconsistent."
                )
            name = decode_string_chunk(rec.decoded_payload, unicode=unicode)
        elif rec.id == SUB_MVT_TIMER_BASE:
            body = rec.decoded_payload or b""
            if len(body) != 4:
                raise FrameDecodeError(
                    f"Mvt Timer Base (0x3347): expected 4 bytes, got "
                    f"{len(body)} at frame-offset 0x{rec.inner_offset:x}"
                )
            mvt_timer_base = int.from_bytes(body, "little", signed=True)
        elif rec.id == SUB_FRAME_PALETTE:
            # decoded_payload is guaranteed non-None here: either the
            # sub-chunk was flag=0/1 and decompress_payload_bytes ran,
            # or it was flag=2/3 and we only fall through the deferred
            # branch above when transform is None.
            if rec.decoded_payload is None:
                raise FrameDecodeError(
                    "Frame Palette (0x3337) sub-chunk had no decoded payload "
                    "despite transform being supplied — decoder state is "
                    "inconsistent."
                )
            palette = decode_frame_palette(rec.decoded_payload)

    return Frame(
        sub_records=sub_records,
        name=name,
        mvt_timer_base=mvt_timer_base,
        palette=palette,
        deferred_encrypted=tuple(deferred),
    )
