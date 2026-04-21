"""Regression tests for 0x6668 Sounds decoder (probe #9).

Top-level chunk. Envelope-only scope — this probe decodes the record
bank and pins its structural invariants against FNAF 1, but defers
audio-format parsing (OGG/WAV container decode) to a tail probe #9.1
if downstream ever needs format-aware extraction.

Antibody coverage:

- #1 strict-unknown : negative count, negative per-record sizes,
  oversize name_length, short zlib stream, or an inner-blob size
  mismatch raise `SoundBankDecodeError`.
- #2 byte-count    : u32 count + N × (28 + body_size) must reconcile
  to `len(payload)` exactly; inner blob length must equal declared
  `decompressed_size`; name bytes (`2*name_length`) must fit inside
  `decompressed_size`.
- #3 round-trip    : synthetic pack/unpack on a 2-sound bank covering
  both the compressed and uncompressed branches.
- #5 multi-input   : runs against the FNAF 1 0x6668 payload.
- #7 snapshot      : pins count, first/last handles, total audio bytes,
  flags histogram, and a SHA-256 of the tuple of `record_start_offsets`.
- Cross-chunk (load-bearing): for every sound record `R` in the bank,
  the 0x5557 offset keyed by `R`'s handle equals
  `R.record_start_offset + FNAF1_SOUND_OFFSET_DELTA`. The delta is
  measured empirically and pinned as a named constant so any future
  drift surfaces with the offending handle identified. 52 distinct
  non-zero offsets, 8 zero-bucket slots — together the full 60-slot
  offset table.
- Structural invariant: `flags == 33` (PlayFromDisk + Wave) never fires
  on FNAF 1. Pinned because that branch carries a CTFAK2 seek-0 quirk
  we've chosen not to implement; if a future pack trips it, this test
  fires before the decoder produces silently-wrong audio.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from collections import Counter
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.sound_offsets import decode_sound_offsets
from fnaf_parser.decoders.sounds import (
    SOUND_FLAG_PLAY_FROM_DISK,
    SOUND_FLAG_WAVE,
    SOUND_FLAGS_PLAYFROMDISK_WAVE,
    SOUND_HANDLE_ADJUST,
    SOUND_ITEM_HEADER_SIZE,
    Sound,
    SoundBank,
    SoundBankDecodeError,
    decode_sound_bank,
    sound_flag_names,
)
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START
from tests.fnaf1_constants import FNAF1_BANK_OFFSET_DELTA

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"

# Alias for local readability — see `tests/fnaf1_constants.py`. Probe #9
# confirmed the same +260 delta first observed in probe #7 (images) and
# later in probe #10 (fonts). Kept as a named constant so a sound-bank
# drift fires with a clearly-scoped failure.
FNAF1_SOUND_OFFSET_DELTA = FNAF1_BANK_OFFSET_DELTA


# --- Synthetic helpers --------------------------------------------------


def _encode_name_wchars(name: str) -> bytes:
    """Encode `name` as UTF-16 LE bytes. Caller owns NUL-terminator
    semantics (pass a trailing `\\x00` if matching CTFAK2's
    `WriteUnicode` behaviour)."""
    return name.encode("utf-16-le")


def _pack_sound_compressed(
    *,
    raw_handle: int,
    checksum: int = 0,
    references: int = 0,
    flags: int = SOUND_FLAG_WAVE,
    reserved: int = 0,
    name: str,
    audio: bytes,
) -> bytes:
    """Pack one SoundItem using the *compressed* body branch:

    ::

        [28-byte header][i32 compressed_size][zlib(inner_blob)]

    where `inner_blob = name_wchars + audio`.
    """
    name_bytes = _encode_name_wchars(name)
    # name_length is in WCHARS — len(name_bytes) // 2. For ASCII-only
    # names len(name_bytes) == 2 * len(name), but we compute from bytes
    # so surrogate-pair cases don't silently miscount.
    assert len(name_bytes) % 2 == 0
    name_length = len(name_bytes) // 2
    inner_blob = name_bytes + audio
    compressed = zlib.compress(inner_blob)
    header = struct.pack(
        "<IiIiIii",
        raw_handle,
        checksum,
        references,
        len(inner_blob),  # decompressed_size
        flags,
        reserved,
        name_length,
    )
    return header + struct.pack("<i", len(compressed)) + compressed


def _pack_sound_uncompressed(
    *,
    raw_handle: int,
    checksum: int = 0,
    references: int = 0,
    flags: int = SOUND_FLAGS_PLAYFROMDISK_WAVE,
    reserved: int = 0,
    name: str,
    audio: bytes,
) -> bytes:
    """Pack one SoundItem using the *uncompressed* body branch:

    ::

        [28-byte header][name_wchars + audio]
    """
    name_bytes = _encode_name_wchars(name)
    assert len(name_bytes) % 2 == 0
    name_length = len(name_bytes) // 2
    inner_blob = name_bytes + audio
    header = struct.pack(
        "<IiIiIii",
        raw_handle,
        checksum,
        references,
        len(inner_blob),  # decompressed_size
        flags,
        reserved,
        name_length,
    )
    return header + inner_blob


def _pack_bank(records: list[bytes]) -> bytes:
    return struct.pack("<i", len(records)) + b"".join(records)


# --- Synthetic tests (no binary needed) ---------------------------------


def test_decode_sound_bank_empty():
    """Zero-count bank decodes to an empty SoundBank. Only the 4-byte
    u32 count prefix is required on the wire."""
    bank = decode_sound_bank(struct.pack("<i", 0))
    assert bank == SoundBank(count=0, sounds=(), record_start_offsets=())


def test_decode_sound_bank_roundtrip_compressed_and_uncompressed():
    """Antibody #3: pack two recognisable records — one compressed,
    one uncompressed (flags==33) — and verify every inner field
    survives a pack → zlib → decode cycle."""
    rec_a = _pack_sound_compressed(
        raw_handle=5,
        checksum=0x11223344,
        references=1,
        flags=SOUND_FLAG_WAVE,
        reserved=0,
        name="intro.ogg",
        audio=b"OggS\x00\x00\x00\x00fake-audio",
    )
    rec_b = _pack_sound_uncompressed(
        raw_handle=9,
        checksum=-1,
        references=0,
        flags=SOUND_FLAGS_PLAYFROMDISK_WAVE,
        reserved=0,
        name="door.wav",
        audio=b"RIFFfakewavbody",
    )
    payload = _pack_bank([rec_a, rec_b])
    bank = decode_sound_bank(payload)
    assert isinstance(bank, SoundBank)
    assert bank.count == 2
    assert len(bank.sounds) == 2
    assert bank.record_start_offsets == (4, 4 + len(rec_a))

    a, b = bank.sounds
    assert a.raw_handle == 5
    assert a.handle == 5 - SOUND_HANDLE_ADJUST == 4
    assert a.checksum == 0x11223344
    assert a.references == 1
    assert a.flags == SOUND_FLAG_WAVE
    assert a.name == "intro.ogg"
    assert a.is_compressed is True
    assert a.audio_data == b"OggS\x00\x00\x00\x00fake-audio"
    # Sanity: audio + name*2 == decompressed_size
    assert len(a.audio_data) + 2 * a.name_length == a.decompressed_size

    assert b.raw_handle == 9
    assert b.handle == 8
    assert b.checksum == -1
    assert b.flags == SOUND_FLAGS_PLAYFROMDISK_WAVE
    assert b.is_compressed is False
    assert b.compressed_size == 0
    assert b.name == "door.wav"
    assert b.audio_data == b"RIFFfakewavbody"


def test_decode_sound_bank_name_trims_trailing_nul_and_whitespace():
    """CTFAK2 does `.Trim()` on names; we match by stripping trailing
    NUL and whitespace. A name written with a trailing NUL terminator
    (common Fusion convention) comes back clean."""
    rec = _pack_sound_compressed(
        raw_handle=1,
        flags=SOUND_FLAG_WAVE,
        name="menu.ogg\x00",  # trailing NUL written
        audio=b"AUDIO",
    )
    bank = decode_sound_bank(_pack_bank([rec]))
    assert bank.sounds[0].name == "menu.ogg"


def test_decode_sound_bank_payload_too_small_for_count():
    """Antibody #2: a 3-byte payload can't hold even the u32 count."""
    with pytest.raises(SoundBankDecodeError, match="count prefix"):
        decode_sound_bank(b"\x00\x00\x00")


def test_decode_sound_bank_negative_count_raises():
    """Antibody #1: a negative signed-int32 count surfaces as an error.
    Likely a sign of outer-zlib corruption or RC4 drift."""
    with pytest.raises(SoundBankDecodeError, match="Negative counts"):
        decode_sound_bank(struct.pack("<i", -1))


def test_decode_sound_bank_record_header_truncated_raises():
    """Antibody #2: count=1 but payload only carries the count prefix
    (no room for the 28-byte item header) must raise, not silently
    return count=0."""
    payload = struct.pack("<i", 1)  # count=1, zero bytes of record body
    with pytest.raises(SoundBankDecodeError, match="fixed header"):
        decode_sound_bank(payload)


def test_decode_sound_bank_compressed_size_overruns_payload():
    """Antibody #2: a compressed-branch record that claims more bytes
    than remain in the payload must raise."""
    # count=1, 28-byte header claiming decompressed_size=10, flags=0
    # (so compressed branch is selected), then i32 compressed_size=9999
    # but no compressed body follows.
    header = struct.pack(
        "<IiIiIii",
        1,       # raw_handle
        0,       # checksum
        0,       # references
        10,      # decompressed_size
        0,       # flags -- not the PlayFromDisk+Wave combo, compressed branch
        0,       # reserved
        0,       # name_length
    )
    payload = struct.pack("<i", 1) + header + struct.pack("<i", 9999)
    with pytest.raises(SoundBankDecodeError, match="compressed_size"):
        decode_sound_bank(payload)


def test_decode_sound_bank_uncompressed_decompressed_size_overruns():
    """Antibody #2: uncompressed branch (flags==33) that claims more
    body bytes than remain must raise."""
    header = struct.pack(
        "<IiIiIii",
        1,
        0,
        0,
        9999,    # decompressed_size — way too big
        SOUND_FLAGS_PLAYFROMDISK_WAVE,  # flags → uncompressed branch
        0,
        0,
    )
    payload = struct.pack("<i", 1) + header
    with pytest.raises(SoundBankDecodeError, match="overruns payload"):
        decode_sound_bank(payload)


def test_decode_sound_bank_name_length_overruns_inner_blob():
    """Antibody #2: name_length * 2 must fit within the inner blob.
    If it doesn't, either the name_length is wrong or the zlib stream
    produced a short blob — either way, antibody #2 fires."""
    # Compressed branch with name_length=100 but an inner blob of only
    # 4 bytes — wildly inconsistent.
    inner_blob = b"abcd"
    compressed = zlib.compress(inner_blob)
    header = struct.pack(
        "<IiIiIii",
        1,
        0,
        0,
        len(inner_blob),
        0,       # flags — compressed branch
        0,
        100,     # name_length = 100 wchars = 200 bytes > 4
    )
    payload = (
        struct.pack("<i", 1)
        + header
        + struct.pack("<i", len(compressed))
        + compressed
    )
    with pytest.raises(SoundBankDecodeError, match="does not fit"):
        decode_sound_bank(payload)


def test_decode_sound_bank_decompressed_size_mismatch_raises():
    """Antibody #2: declared decompressed_size must match what zlib
    actually produces. Drift here is the loudest symptom of a
    corrupted record."""
    real_inner = _encode_name_wchars("a") + b"AUDIO"
    compressed = zlib.compress(real_inner)
    header = struct.pack(
        "<IiIiIii",
        1,
        0,
        0,
        len(real_inner) + 1,  # lie: claim one more byte than reality
        0,
        0,
        1,
    )
    payload = (
        struct.pack("<i", 1)
        + header
        + struct.pack("<i", len(compressed))
        + compressed
    )
    with pytest.raises(SoundBankDecodeError, match="zlib produced"):
        decode_sound_bank(payload)


def test_decode_sound_bank_trailing_bytes_raise():
    """Antibody #2: one valid record followed by an extra byte must
    raise — no silent truncation, no silent extension."""
    rec = _pack_sound_compressed(
        raw_handle=1,
        flags=SOUND_FLAG_WAVE,
        name="x",
        audio=b"A",
    )
    payload = _pack_bank([rec]) + b"\x00"
    with pytest.raises(SoundBankDecodeError, match="unaccounted-for"):
        decode_sound_bank(payload)


def test_sound_flag_names_helper():
    """`sound_flag_names` returns the set names of every raised bit."""
    assert sound_flag_names(0) == ()
    assert sound_flag_names(SOUND_FLAG_WAVE) == ("Wave",)
    combo = SOUND_FLAG_WAVE | SOUND_FLAG_PLAY_FROM_DISK
    assert combo == SOUND_FLAGS_PLAYFROMDISK_WAVE
    names = sound_flag_names(combo)
    assert set(names) == {"Wave", "PlayFromDisk"}


def test_as_dict_shape_is_stable():
    """`as_dict()` outputs a fixed JSON-friendly shape. Snapshot tests
    lean on this key list staying greppable."""
    rec = _pack_sound_compressed(
        raw_handle=2,
        flags=SOUND_FLAG_WAVE,
        name="hit.ogg",
        audio=b"OggS\x00\x01",
    )
    bank = decode_sound_bank(_pack_bank([rec]))
    d = bank.as_dict()
    assert set(d.keys()) == {"count", "sounds", "record_start_offsets"}
    snd_d = d["sounds"][0]
    for k in (
        "raw_handle", "handle", "record_start_offset", "record_wire_size",
        "checksum", "references", "decompressed_size", "flags",
        "flag_names", "reserved", "name_length", "name", "is_compressed",
        "compressed_size", "audio_data_len",
    ):
        assert k in snd_d, f"missing key: {k}"
    assert snd_d["name"] == "hit.ogg"


# --- FNAF 1 multi-input / snapshot / cross-chunk ------------------------


def _fnaf1_transform_and_records():
    """Walk the FNAF 1 pack with the RC4 transform enabled so
    read_chunk_payload can decode any chunk. Mirrors the helper in
    test_images.py."""
    blob = FNAF_EXE.read_bytes()
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)

    def _str_of(chunk_id: int) -> str:
        rec = next(r for r in result.records if r.id == chunk_id)
        return decode_string_chunk(
            read_chunk_payload(blob, rec), unicode=result.header.unicode
        )

    editor = _str_of(0x222E)
    name = _str_of(0x2224)
    copyright_records = [r for r in result.records if r.id == 0x223B]
    copyright_str = (
        decode_string_chunk(
            read_chunk_payload(blob, copyright_records[0]),
            unicode=result.header.unicode,
        )
        if copyright_records
        else ""
    )
    transform = make_transform(
        editor=editor,
        name=name,
        copyright_str=copyright_str,
        build=result.header.product_build,
        unicode=result.header.unicode,
    )
    return blob, result, transform


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_sound_bank_decodes_without_error():
    """Antibody #5 multi-input: the full FNAF 1 0x6668 payload decodes
    without raising and produces a non-empty SoundBank."""
    blob, result, transform = _fnaf1_transform_and_records()
    recs = [r for r in result.records if r.id == 0x6668]
    assert len(recs) == 1, (
        f"FNAF 1 should carry exactly one 0x6668 chunk; saw {len(recs)}"
    )
    payload = read_chunk_payload(blob, recs[0], transform=transform)
    bank = decode_sound_bank(payload)
    assert bank.count > 0
    # Every sound must carry at least the 28-byte header's worth of
    # metadata and a non-negative audio payload.
    for snd in bank.sounds:
        assert isinstance(snd, Sound)
        assert snd.decompressed_size >= 0
        assert 0 <= snd.flags <= 0xFFFFFFFF
        # name_length must fit inside decompressed_size (structural)
        assert 2 * snd.name_length <= snd.decompressed_size


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_sound_bank_no_playfromdisk_wave_items():
    """Structural invariant: no item on FNAF 1 has `flags == 33`
    (PlayFromDisk + Wave). That combo triggers CTFAK2's
    `soundData.Seek(0)` quirk we've deliberately NOT implemented.

    If a future pack trips this, the test fires before the decoder
    silently mis-splits the name from the audio body. At that point
    the fix is to teach the decoder the seek-0 semantics; don't just
    delete the assertion.
    """
    blob, result, transform = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x6668)
    payload = read_chunk_payload(blob, rec, transform=transform)
    bank = decode_sound_bank(payload)

    trippers = [
        snd for snd in bank.sounds if snd.flags == SOUND_FLAGS_PLAYFROMDISK_WAVE
    ]
    assert not trippers, (
        f"FNAF 1 unexpectedly has {len(trippers)} SoundItem(s) with "
        f"flags==33 (PlayFromDisk+Wave). CTFAK2's seek-0 quirk is NOT "
        f"implemented here; re-investigate before trusting the decoder. "
        f"First 3 offenders: "
        f"{[(s.raw_handle, s.name) for s in trippers[:3]]}"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_sound_bank_snapshot():
    """Antibody #7 snapshot: pin the FNAF 1 0x6668 decode against drift.

    Captured empirically on 2026-04-21 (probe #9). Values will be filled
    in from the first green test run; the structural invariants are
    verified before the pinned constants so a future drift reports a
    structural problem first, not a mystery integer mismatch.
    """
    blob, result, transform = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x6668)
    payload = read_chunk_payload(blob, rec, transform=transform)
    bank = decode_sound_bank(payload)

    # Structural invariants (true regardless of pack):
    assert len(bank.record_start_offsets) == bank.count
    assert bank.record_start_offsets == tuple(
        s.record_start_offset for s in bank.sounds
    )
    assert all(s.is_compressed for s in bank.sounds), (
        "FNAF 1 is expected to ship every sound via the compressed branch "
        "(bank-level IsCompressed=True + no flags==33 item)."
    )

    # Tie to 0x5557 non-zero count for reassurance before pinning:
    offsets_rec = next(r for r in result.records if r.id == 0x5557)
    offsets_payload = read_chunk_payload(blob, offsets_rec, transform=transform)
    sound_offsets = decode_sound_offsets(offsets_payload)
    non_zero_offsets = {o for o in sound_offsets.offsets if o != 0}
    assert bank.count == len(non_zero_offsets), (
        f"SoundBank.count={bank.count} must equal the number of distinct "
        f"non-zero 0x5557 offsets ({len(non_zero_offsets)})"
    )

    starts_blob = b"".join(
        struct.pack("<I", off & 0xFFFFFFFF) for off in bank.record_start_offsets
    )
    starts_sha = hashlib.sha256(starts_blob).hexdigest()

    # Full-item fingerprint: (raw_handle, flags, name, decompressed_size)
    # across every record. Catches name-decoder drift even if the layout
    # stays the same.
    items_fp_blob = b"".join(
        struct.pack("<IIiI", s.raw_handle, s.flags, s.decompressed_size, len(s.name))
        + s.name.encode("utf-8")
        for s in bank.sounds
    )
    items_sha = hashlib.sha256(items_fp_blob).hexdigest()

    flags_hist = Counter(s.flags for s in bank.sounds)
    total_audio = sum(len(s.audio_data) for s in bank.sounds)

    # Pinned snapshot values for FNAF 1 (captured 2026-04-21, probe #9):
    assert bank.count == 52
    # NumOfItems (0x34) matches the non-zero 0x5557 count, confirmed above.
    # Note: raw_handles are NOT a contiguous 1..52 range — the pack ships
    # 52 records whose raw_handle values run from 1 to 47 with gaps. The
    # non-zero 0x5557 slot positions reveal the same skipping pattern, so
    # cardinality reconciles even though the sparse handle namespace does.
    assert bank.sounds[0].raw_handle == 1
    assert bank.sounds[0].handle == 0
    assert bank.sounds[-1].raw_handle == 47
    assert bank.sounds[-1].handle == 46
    assert bank.sounds[-1].name == "XSCREAM2"
    # Every sound is Wave-flagged and compressed (flags=1 = SOUND_FLAG_WAVE).
    assert flags_hist == {SOUND_FLAG_WAVE: 52}
    assert starts_sha == (
        "5bf4c69b220f5e314a2f76bb3d4a5299038700f86e04329a0364335c3b9d03e7"
    )
    assert items_sha == (
        "7a80935dd95c39b9bf7e6af6b3661942a3eb6a50e7c988d4ef050118e847d273"
    )
    # Sum of all audio_data bytes ≈ 92.8 MiB (MP3/Wave streams).
    assert total_audio == 97345706


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_sound_bank_cross_chunk_handshake_with_delta():
    """Cross-chunk antibody (load-bearing): for every sound record `R`
    in the bank, the 0x5557 offset keyed by `R`'s handle must equal
    `R.record_start_offset + FNAF1_SOUND_OFFSET_DELTA`.

    Delta is measured empirically once and pinned. On FNAF 1 Probe #7
    (Images) showed a +260 reference-frame shift between record starts
    and 0x5555 offsets; SoundOffsets may or may not use the same delta.
    The test computes the observed delta before asserting so the first
    green run tells us what to pin.

    Investigation log
    -----------------

    - The cardinality check (bank.count == len(non_zero_offsets)) is the
      first line of defence: any decoder desync trips it immediately.
    - The per-record delta check is next: offset[handle] - record_start
      must be a single constant across all 52 records. A multi-valued
      Counter means a drifted decoder; an empty Counter means the handle
      indexing is wrong.
    - Whichever handle convention (raw vs raw-1) makes ALL 52 records
      line up wins — that's how we close the CTFAK2-vs-Anaconda oracle
      disagreement captured in probe #9's ideation node.
    """
    blob, result, transform = _fnaf1_transform_and_records()

    offsets_rec = next(r for r in result.records if r.id == 0x5557)
    offsets_payload = read_chunk_payload(blob, offsets_rec, transform=transform)
    sound_offsets = decode_sound_offsets(offsets_payload)

    sounds_rec = next(r for r in result.records if r.id == 0x6668)
    sounds_payload = read_chunk_payload(blob, sounds_rec, transform=transform)
    bank = decode_sound_bank(sounds_payload)

    non_zero_offsets = [o for o in sound_offsets.offsets if o != 0]
    assert len(non_zero_offsets) == bank.count == 52, (
        f"cardinality mismatch: non_zero_offsets={len(non_zero_offsets)}, "
        f"bank.count={bank.count} — expected 52 both"
    )

    # Observe the delta under BOTH handle conventions (raw and
    # CTFAK2-adjusted) so whichever is correct is self-evident.
    def _deltas(key_attr: str) -> Counter[int]:
        deltas: list[int] = []
        for snd in bank.sounds:
            h = getattr(snd, key_attr)
            if 0 <= h < len(sound_offsets.offsets):
                deltas.append(sound_offsets.offsets[h] - snd.record_start_offset)
        return Counter(deltas)

    raw_deltas = _deltas("raw_handle")
    adj_deltas = _deltas("handle")

    # Exactly one convention should produce a single-valued Counter.
    # Pin the outcome — this nails down the CTFAK2-vs-Anaconda handle
    # disagreement once and for all for FNAF 1.
    raw_singular = len(raw_deltas) == 1 and sum(raw_deltas.values()) == bank.count
    adj_singular = len(adj_deltas) == 1 and sum(adj_deltas.values()) == bank.count
    assert raw_singular ^ adj_singular, (
        f"expected exactly one handle convention to produce a single-value "
        f"delta Counter. raw={raw_deltas}, adj={adj_deltas}"
    )
    winning_attr = "raw_handle" if raw_singular else "handle"
    winning_delta = next(iter(
        (raw_deltas if raw_singular else adj_deltas).keys()
    ))

    # Oracle-resolution pin: probe #9 measured this on FNAF 1 and found
    # CTFAK2's `handle = raw - 1` convention is correct, with the same
    # +260 reference-frame shift as 0x5555 ImageOffsets (probe #7's
    # FNAF1_IMAGE_OFFSET_DELTA). If a future pack flips the winning
    # convention or changes the delta, this breaks and forces us to
    # re-examine the handshake before trusting either decoder.
    assert winning_attr == "handle", (
        f"expected CTFAK2-style adjusted handle to win on FNAF 1, "
        f"got {winning_attr!r}"
    )
    assert winning_delta == FNAF1_SOUND_OFFSET_DELTA, (
        f"expected same +{FNAF1_SOUND_OFFSET_DELTA} offset delta as "
        f"0x5555 ImageOffsets (probe #7 FNAF1_BANK_OFFSET_DELTA), "
        f"got {winning_delta}"
    )

    # Per-record enforcement with the winning convention. Named loop so
    # any single drifted record is called out.
    mismatches: list[tuple[int, int, int, int]] = []
    for snd in bank.sounds:
        h = getattr(snd, winning_attr)
        if not (0 <= h < len(sound_offsets.offsets)):
            mismatches.append((h, snd.record_start_offset, -1, -1))
            continue
        expected = snd.record_start_offset + winning_delta
        actual = sound_offsets.offsets[h]
        if actual != expected:
            mismatches.append((h, snd.record_start_offset, expected, actual))
    assert not mismatches, (
        f"{len(mismatches)} record(s) whose 0x5557 offset ≠ "
        f"record_start + {winning_delta} under {winning_attr}. "
        f"First 5: {mismatches[:5]}"
    )

    # Coverage: every non-zero offset maps back to exactly one record's
    # `record_start + delta`. No stranded offsets.
    record_keyed = frozenset(
        snd.record_start_offset + winning_delta for snd in bank.sounds
    )
    stranded = frozenset(non_zero_offsets) - record_keyed
    assert not stranded, (
        f"{len(stranded)} 0x5557 offset(s) don't correspond to any record's "
        f"start + {winning_delta}: {sorted(stranded)[:5]}"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_sound_bank_handles_are_unique():
    """Structural invariant: both raw and logical handles must be
    unique. A collision would mean either the bank repeats a record or
    the `-1` adjustment lost information."""
    blob, result, transform = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x6668)
    payload = read_chunk_payload(blob, rec, transform=transform)
    bank = decode_sound_bank(payload)

    raw = [s.raw_handle for s in bank.sounds]
    assert len(raw) == len(set(raw)), (
        f"duplicate raw handles in SoundBank "
        f"({len(raw) - len(set(raw))} collision(s))"
    )
    adj = [s.handle for s in bank.sounds]
    assert len(adj) == len(set(adj))


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_sound_bank_names_are_nonempty():
    """Structural invariant: every sound has a non-empty name. A sound
    with an empty name is either decoder drift or a corrupted record —
    empirically FNAF 1 labels every sound."""
    blob, result, transform = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x6668)
    payload = read_chunk_payload(blob, rec, transform=transform)
    bank = decode_sound_bank(payload)

    empty = [s for s in bank.sounds if not s.name]
    assert not empty, (
        f"FNAF 1 has {len(empty)} SoundItem(s) with empty names — "
        f"first 3 raw_handles: {[s.raw_handle for s in empty[:3]]}"
    )
    # Basic smoke: at least 28 bytes of header were consumed per sound.
    assert all(s.record_wire_size > SOUND_ITEM_HEADER_SIZE for s in bank.sounds)
