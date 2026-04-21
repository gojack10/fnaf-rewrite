"""0x6668 Sounds decoder (probe #9) — envelope-only.

Top-level chunk. The sound-bank peer of [[Probe #7]] 0x6666 Images and
the record target for [[Probe #8]] 0x5557 SoundOffsets — that chunk is
the index table, this one is the record bank the offsets land inside.

Wire format (post decrypt + zlib on the outer chunk)
----------------------------------------------------

::

    u32 count                 (== NumOfItems; expected 0x34 = 52 for FNAF 1)
    N × SoundItem

Each `SoundItem` is a 28-byte header + a body whose framing depends on
the bank-level `IsCompressed` flag and the item's own `Flags`:

::

     0  u32 raw_handle        (CTFAK2 stores `raw_handle - 1`; we keep both)
     4  i32 checksum
     8  u32 references
    12  i32 decompressed_size (size of the inner blob = name bytes + audio bytes)
    16  u32 flags             (Anaconda reads u32; CTFAK2 reads u8 + skip 3 — same wire size)
    20  i32 reserved          (== 0 empirically on FNAF 1)
    24  i32 name_length       (in WCHARS — multiply by 2 for UTF-16 LE byte count)
    28  <body>

Body branch selection — mirrors both oracles:

- If bank-level `IsCompressed` is True AND `flags != 33` (PlayFromDisk
  + Wave, i.e. `0x21`):
  - `i32 compressed_size`
  - `compressed_size` bytes → `zlib.decompress` → inner_blob of length
    `decompressed_size`
- Else (PlayFromDisk-uncompressed path):
  - `decompressed_size` raw bytes → inner_blob of length
    `decompressed_size`

Inside `inner_blob`:

- `name`: first `2 * name_length` bytes decoded as UTF-16 LE. We trim
  trailing NULs and whitespace to match CTFAK2's `.Trim()`.
- `audio_data`: remaining `decompressed_size - 2 * name_length` bytes,
  **opaque**. Envelope scope — no OGG/WAV container parse, no format
  dispatch. Probe #9.1 owns that if it ever needs to ship.

CTFAK2 quirk on `flags == 33` (PlayFromDisk + Wave): after reading
`name`, CTFAK2 rewinds `soundData` to offset 0 before reading `Data`,
i.e. the resulting `Data` includes the name bytes. Anaconda does NOT
do this. For FNAF 1 we pin as an antibody that no item has
`flags == 33`, so the quirk never fires; if a future pack trips it,
the test fires before the decoder produces silently-wrong audio.

Oracles (cross-checked — Antibody #4 multi-oracle)
--------------------------------------------------

- CTFAK2.0 `Core/CTFAK.Core/CCN/Chunks/Banks/SoundBank/SoundBank.cs`:
  C# read path. Does `Handle - 1` unconditionally. 28-byte header is
  read field-by-field; note `ReadByte()` + `Skip(3)` for flags,
  equivalent to a u32 read modulo high-bit discard.
- Anaconda `mmfparser/data/chunkloaders/soundbank.py::SoundItem.read`:
  Python read path. Keeps raw handle. Reads flags as full u32, decodes
  via `SOUND_FLAGS` BitDict (Wave, MIDI, _, _, LoadOnCall,
  PlayFromDisk, Loaded).

Oracle discrepancies resolved here:

1. **Handle**: both `raw_handle` and `handle = raw_handle - 1` are
   stored, mirroring `images.Image`. Whichever interpretation object
   references use is settled empirically by the cross-chunk handshake
   and by a future object-bank probe, not here.
2. **Flags byte vs u32**: we read as `u32` (Anaconda) because it's
   forward-compatible with packs that set high bits. FNAF 1 empirically
   lands in the low byte, so behavior matches CTFAK2 on the target pack.

Sound flag bits (Anaconda SOUND_FLAGS)
--------------------------------------

+---+------+--------------+
|Bit| Mask | Name         |
+===+======+==============+
| 0 | 0x01 | Wave         |
+---+------+--------------+
| 1 | 0x02 | MIDI         |
+---+------+--------------+
| 4 | 0x10 | LoadOnCall   |
+---+------+--------------+
| 5 | 0x20 | PlayFromDisk |
+---+------+--------------+
| 6 | 0x40 | Loaded       |
+---+------+--------------+

Bits 2 and 3 are reserved / unused in both oracles.

Scope cut — envelope only
-------------------------

This probe decodes the *record bank*. It does NOT:

- Parse OGG / WAV / AIFF / MOD container headers inside `audio_data`.
- Dispatch on `flags` to normalize audio formats.
- Handle the `flags == 33` seek-0 quirk (pinned-absent on FNAF 1).

`audio_data` is the raw post-inner-decompress byte slice, exactly as it
lives in the decompressed 0x6668 body after the name is stripped.

Cross-chunk antibody (load-bearing for this probe)
--------------------------------------------------

Every sound record's start offset — its position within the
*decompressed outer 0x6668 body* — must appear in the set of non-zero
offsets produced by Probe #8's 0x5557 SoundOffsets decode. Expected on
FNAF 1:

::

    set(record_start_offsets) == {o for o in sound_offsets.offsets if o != 0}
    len(record_start_offsets) == 52
    sound_bank.count         == 52

This is the single strongest evidence that the two decoders agree on
the wire format; enforced in the integration test, not inside this
decoder.

Antibody coverage (this decoder)
--------------------------------

- #1 strict-unknown : negative count / negative per-record sizes /
  negative `name_length` raise `SoundBankDecodeError`; a `name_length *
  2` that exceeds `decompressed_size` raises; a nested zlib compSize
  that overruns the outer payload raises; an inner blob whose actual
  decompressed length disagrees with declared `decompressed_size`
  raises.
- #2 byte-count    : `count` prefix + N × record size must reconcile
  to exactly `len(payload)`. Inner blob byte count must equal declared
  `decompressed_size`. Name bytes (`2 * name_length`) must fit inside
  `decompressed_size`.
- #3 round-trip    : synthetic pack/unpack in tests.
- #4 multi-oracle  : field order + sizes mirror CTFAK2 + Anaconda.
- #5 multi-input   : runs against the FNAF 1 0x6668 payload.
- #7 snapshot      : count, first raw handle, name histogram samples,
  total audio bytes, and a SHA-256 of the `(raw_handle, flags, name,
  decompressed_size)` tuple across all items pinned in tests.
- Cross-chunk     : `SoundBank.record_start_offsets` drives the
  integration-layer handshake against `SoundOffsets.offsets`.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

# --- Wire-format constants ----------------------------------------------

SOUND_BANK_COUNT_SIZE = 4

# 28-byte per-item header: u32 handle + i32 checksum + u32 references +
# i32 decompressed_size + u32 flags + i32 reserved + i32 name_length.
# We use `I` / `i` to match the signed/unsigned semantics of each field
# — signed reads surface negative drift loudly (Antibody #1) on fields
# that can't be negative in reality.
_SOUND_ITEM_HEADER = struct.Struct("<IiIiIii")
SOUND_ITEM_HEADER_SIZE = _SOUND_ITEM_HEADER.size  # 28
assert SOUND_ITEM_HEADER_SIZE == 28

# Compressed-size prefix before the zlib stream (compressed branch only).
SOUND_ITEM_COMPSIZE_FIELD = 4

# CTFAK2 subtracts 1 from the wire handle unconditionally. Mirrors the
# Image bank's build>=284 handle adjust. Kept as a named constant so the
# intent is grep-friendly and a future legacy-build code path can flip
# it off cleanly.
SOUND_HANDLE_ADJUST = 1

# Flag bit that selects the `flags == 33` PlayFromDisk-Wave branch in
# CTFAK2. Exposed so tests can pin "no item trips this" cleanly.
SOUND_FLAGS_PLAYFROMDISK_WAVE = 0x21

# --- Sound-flag bits (Anaconda SOUND_FLAGS BitDict) ---------------------

SOUND_FLAG_WAVE = 1 << 0
SOUND_FLAG_MIDI = 1 << 1
SOUND_FLAG_LOAD_ON_CALL = 1 << 4
SOUND_FLAG_PLAY_FROM_DISK = 1 << 5
SOUND_FLAG_LOADED = 1 << 6

SOUND_FLAG_NAMES: dict[int, str] = {
    SOUND_FLAG_WAVE: "Wave",
    SOUND_FLAG_MIDI: "MIDI",
    SOUND_FLAG_LOAD_ON_CALL: "LoadOnCall",
    SOUND_FLAG_PLAY_FROM_DISK: "PlayFromDisk",
    SOUND_FLAG_LOADED: "Loaded",
}


def sound_flag_names(flags: int) -> tuple[str, ...]:
    """Return the set names of every raised bit in `flags`.

    Diagnostic sugar. The decoder itself dispatches on `flags ==
    SOUND_FLAGS_PLAYFROMDISK_WAVE`, not on the bit-name decomposition
    — but the snapshot test and any caller dumping the bank summary
    wants greppable names.
    """
    return tuple(
        name
        for bit, name in SOUND_FLAG_NAMES.items()
        if (flags & bit) != 0
    )


class SoundBankDecodeError(ValueError):
    """0x6668 Sounds decode failure — carries offset + size context."""


# --- Dataclasses --------------------------------------------------------


@dataclass(frozen=True)
class Sound:
    """One decoded sound record inside the 0x6668 SoundBank.

    The raw on-wire handle lives in `raw_handle`; `handle` is the
    CTFAK2-adjusted logical index (`raw_handle - 1`), matching the
    Image bank's convention. `record_start_offset` is this record's
    position within the *decompressed outer 0x6668 payload* — the
    load-bearing field for the cross-chunk handshake with 0x5557
    SoundOffsets.

    `audio_data` is the opaque byte slice that lives after the inner
    header + name bytes. Envelope scope — no container-format parsing.
    """

    raw_handle: int
    handle: int
    record_start_offset: int
    record_wire_size: int
    checksum: int
    references: int
    decompressed_size: int
    flags: int
    reserved: int
    name_length: int
    name: str
    is_compressed: bool
    compressed_size: int
    audio_data: bytes = field(repr=False)

    @property
    def has_wave(self) -> bool:
        return bool(self.flags & SOUND_FLAG_WAVE)

    @property
    def has_play_from_disk(self) -> bool:
        return bool(self.flags & SOUND_FLAG_PLAY_FROM_DISK)

    @property
    def has_midi(self) -> bool:
        return bool(self.flags & SOUND_FLAG_MIDI)

    def as_dict(self) -> dict:
        return {
            "raw_handle": self.raw_handle,
            "handle": self.handle,
            "record_start_offset": self.record_start_offset,
            "record_wire_size": self.record_wire_size,
            "checksum": self.checksum,
            "references": self.references,
            "decompressed_size": self.decompressed_size,
            "flags": self.flags,
            "flag_names": list(sound_flag_names(self.flags)),
            "reserved": self.reserved,
            "name_length": self.name_length,
            "name": self.name,
            "is_compressed": self.is_compressed,
            "compressed_size": self.compressed_size,
            "audio_data_len": len(self.audio_data),
        }


@dataclass(frozen=True)
class SoundBank:
    """Decoded 0x6668 Sounds payload.

    `sounds` preserves the on-wire order. `record_start_offsets` is the
    tuple of per-record start positions inside the decompressed outer
    payload — the single cross-chunk antibody surface against Probe #8's
    0x5557 SoundOffsets. `by_handle` / `by_raw_handle` give either
    adjusted or raw handle resolution depending on whose convention the
    caller needs.
    """

    count: int
    sounds: tuple[Sound, ...]
    record_start_offsets: tuple[int, ...]

    @property
    def by_handle(self) -> dict[int, Sound]:
        """Logical (`raw - 1`) `handle → Sound` map. Built fresh each call."""
        return {s.handle: s for s in self.sounds}

    @property
    def by_raw_handle(self) -> dict[int, Sound]:
        """Raw (on-wire) `handle → Sound` map — exposed for round-trip
        tooling and for a future object-bank probe to decide which
        convention object references use."""
        return {s.raw_handle: s for s in self.sounds}

    @property
    def handles(self) -> frozenset[int]:
        """Set of logical handles this bank exposes."""
        return frozenset(s.handle for s in self.sounds)

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "sounds": [s.as_dict() for s in self.sounds],
            "record_start_offsets": list(self.record_start_offsets),
        }


# --- Decoder ------------------------------------------------------------


def _decode_name(inner_blob: bytes, name_length: int, raw_handle: int) -> str:
    """Decode `name_length` WCHARS of UTF-16 LE from the head of
    `inner_blob` and trim trailing NUL / whitespace to match CTFAK2's
    `.Trim()`.

    `raw_handle` is error-context only.
    """
    name_bytes = 2 * name_length
    if name_bytes < 0 or name_bytes > len(inner_blob):
        raise SoundBankDecodeError(
            f"0x6668 Sounds: raw_handle={raw_handle} name_length="
            f"{name_length} (= {name_bytes} bytes) does not fit inside "
            f"inner blob of {len(inner_blob)} bytes. Antibody #2 byte-count: "
            f"name_length is wrong or the zlib stream produced a short blob."
        )
    try:
        raw_name = inner_blob[:name_bytes].decode("utf-16-le")
    except UnicodeDecodeError as exc:
        raise SoundBankDecodeError(
            f"0x6668 Sounds: raw_handle={raw_handle} name failed UTF-16 LE "
            f"decode: {exc}. Antibody #1 strict-unknown: likely zlib drift "
            f"inside the item body."
        ) from exc
    return raw_name.rstrip("\x00").strip()


def decode_sound_bank(payload: bytes, *, is_compressed: bool = True) -> SoundBank:
    """Decode a 0x6668 Sounds chunk payload.

    `payload` must be the plaintext bytes returned by
    `compression.read_chunk_payload` — i.e. any outer flag=1/2/3
    decoding has already happened. This decoder handles the *nested*
    per-item zlib layer itself (inline, not via
    `decompress_payload_bytes`) because the inner layer wraps only the
    item body (name + audio), not the 28-byte item header.

    `is_compressed` mirrors the bank-level `IsCompressed` flag in
    CTFAK2 — defaults True because that's how FNAF 1 and every other
    modern Fusion pack ships. A debug pack that flips it would pass
    `is_compressed=False`.

    Envelope only — see module docstring for the scope cut.
    """
    n = len(payload)
    if n < SOUND_BANK_COUNT_SIZE:
        raise SoundBankDecodeError(
            f"0x6668 Sounds: payload must hold at least the "
            f"{SOUND_BANK_COUNT_SIZE}-byte u32 count prefix but got {n}. "
            f"Antibody #2 byte-count."
        )

    count = int.from_bytes(payload[:SOUND_BANK_COUNT_SIZE], "little", signed=True)
    if count < 0:
        raise SoundBankDecodeError(
            f"0x6668 Sounds: count prefix decoded as {count} (signed "
            f"int32). Negative counts are nonsense; Antibody #1 "
            f"strict-unknown. Likely outer-layer zlib or RC4 drift."
        )

    pos = SOUND_BANK_COUNT_SIZE
    sounds: list[Sound] = []
    record_starts: list[int] = []

    for i in range(count):
        record_start = pos
        record_starts.append(record_start)

        if pos + SOUND_ITEM_HEADER_SIZE > n:
            raise SoundBankDecodeError(
                f"0x6668 Sounds: record #{i} starts at offset 0x{pos:x} "
                f"but only {n - pos} bytes remain (need "
                f"≥{SOUND_ITEM_HEADER_SIZE} for the fixed header). "
                f"Antibody #2 byte-count."
            )

        (
            raw_handle,
            checksum,
            references,
            decompressed_size,
            flags,
            reserved,
            name_length,
        ) = _SOUND_ITEM_HEADER.unpack_from(payload, pos)

        if decompressed_size < 0:
            raise SoundBankDecodeError(
                f"0x6668 Sounds: record #{i} (raw_handle={raw_handle}) "
                f"at offset 0x{pos:x} has decompressed_size="
                f"{decompressed_size} (signed). Antibody #1 strict-unknown: "
                f"likely zlib or RC4 drift."
            )
        if name_length < 0:
            raise SoundBankDecodeError(
                f"0x6668 Sounds: record #{i} (raw_handle={raw_handle}) "
                f"at offset 0x{pos:x} has name_length={name_length} "
                f"(signed). Antibody #1 strict-unknown."
            )

        use_compressed_branch = is_compressed and flags != SOUND_FLAGS_PLAYFROMDISK_WAVE
        body_start = pos + SOUND_ITEM_HEADER_SIZE

        if use_compressed_branch:
            if body_start + SOUND_ITEM_COMPSIZE_FIELD > n:
                raise SoundBankDecodeError(
                    f"0x6668 Sounds: record #{i} (raw_handle={raw_handle}) "
                    f"at offset 0x{pos:x}: not enough bytes remaining for "
                    f"the compressed-size prefix. Antibody #2 byte-count."
                )
            compressed_size = int.from_bytes(
                payload[body_start : body_start + SOUND_ITEM_COMPSIZE_FIELD],
                "little",
                signed=True,
            )
            if compressed_size < 0:
                raise SoundBankDecodeError(
                    f"0x6668 Sounds: record #{i} (raw_handle={raw_handle}) "
                    f"at offset 0x{pos:x}: negative compressed_size="
                    f"{compressed_size}. Antibody #1 strict-unknown."
                )
            zlib_start = body_start + SOUND_ITEM_COMPSIZE_FIELD
            zlib_end = zlib_start + compressed_size
            if zlib_end > n:
                raise SoundBankDecodeError(
                    f"0x6668 Sounds: record #{i} (raw_handle={raw_handle}) "
                    f"at offset 0x{pos:x}: compressed_size={compressed_size} "
                    f"overruns payload ({n - zlib_start} bytes remaining). "
                    f"Antibody #2 byte-count."
                )
            compressed_body = payload[zlib_start:zlib_end]
            try:
                inner_blob = zlib.decompress(compressed_body)
            except zlib.error as exc:
                raise SoundBankDecodeError(
                    f"0x6668 Sounds: record #{i} (raw_handle={raw_handle}) "
                    f"at offset 0x{pos:x}: inner zlib decompress failed: "
                    f"{exc}. Antibody #1 strict-unknown: likely RC4 drift "
                    f"on the outer chunk."
                ) from exc
            if len(inner_blob) != decompressed_size:
                raise SoundBankDecodeError(
                    f"0x6668 Sounds: record #{i} (raw_handle={raw_handle}): "
                    f"declared decompressed_size={decompressed_size} but "
                    f"zlib produced {len(inner_blob)} bytes. Antibody #2."
                )
            record_wire_size = (
                SOUND_ITEM_HEADER_SIZE + SOUND_ITEM_COMPSIZE_FIELD + compressed_size
            )
            pos = zlib_end
        else:
            # Uncompressed branch: decompressed_size raw bytes ARE the
            # inner blob. Consumed straight from the payload.
            compressed_size = 0
            body_end = body_start + decompressed_size
            if body_end > n:
                raise SoundBankDecodeError(
                    f"0x6668 Sounds: record #{i} (raw_handle={raw_handle}) "
                    f"at offset 0x{pos:x}: decompressed_size={decompressed_size} "
                    f"overruns payload ({n - body_start} bytes remaining). "
                    f"Antibody #2 byte-count."
                )
            inner_blob = bytes(payload[body_start:body_end])
            record_wire_size = SOUND_ITEM_HEADER_SIZE + decompressed_size
            pos = body_end

        name = _decode_name(inner_blob, name_length, raw_handle)
        audio_data = bytes(inner_blob[2 * name_length :])

        sounds.append(
            Sound(
                raw_handle=raw_handle,
                handle=raw_handle - SOUND_HANDLE_ADJUST,
                record_start_offset=record_start,
                record_wire_size=record_wire_size,
                checksum=checksum,
                references=references,
                decompressed_size=decompressed_size,
                flags=flags,
                reserved=reserved,
                name_length=name_length,
                name=name,
                is_compressed=use_compressed_branch,
                compressed_size=compressed_size,
                audio_data=audio_data,
            )
        )

    if pos != n:
        raise SoundBankDecodeError(
            f"0x6668 Sounds: decoded {count} records ending at offset "
            f"0x{pos:x} but payload is {n} bytes. Trailing "
            f"{n - pos} bytes unaccounted-for. Antibody #2 byte-count."
        )

    return SoundBank(
        count=count,
        sounds=tuple(sounds),
        record_start_offsets=tuple(record_starts),
    )
