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

import pytest

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
from tests.fnaf1_constants import FNAF1_BANK_OFFSET_DELTA

# FNAF 1 integration tests below pull their blob / walk_chunks result /
# RC4 transform / decoded SoundBank from session-scoped fixtures in
# conftest.py (`fnaf1_exe_bytes`, `fnaf1_walk_result`, `fnaf1_transform`,
# `fnaf1_sound_bank`). The per-module `_fnaf1_transform_and_records()`
# helper + `FNAF_EXE` constant + `walk_chunks` / `make_transform` /
# `decode_string_chunk` imports are all gone.

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


def test_fnaf1_sound_bank_decodes_without_error(fnaf1_walk_result, fnaf1_sound_bank: SoundBank):
    """Antibody #5 multi-input: the full FNAF 1 0x6668 payload decodes
    without raising and produces a non-empty SoundBank."""
    recs = [r for r in fnaf1_walk_result.records if r.id == 0x6668]
    assert len(recs) == 1, (
        f"FNAF 1 should carry exactly one 0x6668 chunk; saw {len(recs)}"
    )
    assert fnaf1_sound_bank.count > 0
    # Every sound must carry at least the 28-byte header's worth of
    # metadata and a non-negative audio payload.
    for snd in fnaf1_sound_bank.sounds:
        assert isinstance(snd, Sound)
        assert snd.decompressed_size >= 0
        assert 0 <= snd.flags <= 0xFFFFFFFF
        # name_length must fit inside decompressed_size (structural)
        assert 2 * snd.name_length <= snd.decompressed_size


def test_fnaf1_sound_bank_no_playfromdisk_wave_items(fnaf1_sound_bank: SoundBank):
    """Structural invariant: no item on FNAF 1 has `flags == 33`
    (PlayFromDisk + Wave). That combo triggers CTFAK2's
    `soundData.Seek(0)` quirk we've deliberately NOT implemented.

    If a future pack trips this, the test fires before the decoder
    silently mis-splits the name from the audio body. At that point
    the fix is to teach the decoder the seek-0 semantics; don't just
    delete the assertion.
    """
    trippers = [
        snd for snd in fnaf1_sound_bank.sounds if snd.flags == SOUND_FLAGS_PLAYFROMDISK_WAVE
    ]
    assert not trippers, (
        f"FNAF 1 unexpectedly has {len(trippers)} SoundItem(s) with "
        f"flags==33 (PlayFromDisk+Wave). CTFAK2's seek-0 quirk is NOT "
        f"implemented here; re-investigate before trusting the decoder. "
        f"First 3 offenders: "
        f"{[(s.raw_handle, s.name) for s in trippers[:3]]}"
    )


def test_fnaf1_sound_bank_snapshot(
    fnaf1_exe_bytes: bytes,
    fnaf1_walk_result,
    fnaf1_transform,
    fnaf1_sound_bank: SoundBank,
):
    """Antibody #7 snapshot: pin the FNAF 1 0x6668 decode against drift.

    Captured empirically on 2026-04-21 (probe #9). Values will be filled
    in from the first green test run; the structural invariants are
    verified before the pinned constants so a future drift reports a
    structural problem first, not a mystery integer mismatch.
    """
    bank = fnaf1_sound_bank

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
    offsets_rec = next(r for r in fnaf1_walk_result.records if r.id == 0x5557)
    offsets_payload = read_chunk_payload(
        fnaf1_exe_bytes, offsets_rec, transform=fnaf1_transform
    )
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


def test_fnaf1_sound_bank_cross_chunk_handshake_with_delta(
    fnaf1_exe_bytes: bytes,
    fnaf1_walk_result,
    fnaf1_transform,
    fnaf1_sound_bank: SoundBank,
):
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
    offsets_rec = next(r for r in fnaf1_walk_result.records if r.id == 0x5557)
    offsets_payload = read_chunk_payload(
        fnaf1_exe_bytes, offsets_rec, transform=fnaf1_transform
    )
    sound_offsets = decode_sound_offsets(offsets_payload)

    bank = fnaf1_sound_bank

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


def test_fnaf1_sound_bank_handles_are_unique(fnaf1_sound_bank: SoundBank):
    """Structural invariant: both raw and logical handles must be
    unique. A collision would mean either the bank repeats a record or
    the `-1` adjustment lost information."""
    bank = fnaf1_sound_bank

    raw = [s.raw_handle for s in bank.sounds]
    assert len(raw) == len(set(raw)), (
        f"duplicate raw handles in SoundBank "
        f"({len(raw) - len(set(raw))} collision(s))"
    )
    adj = [s.handle for s in bank.sounds]
    assert len(adj) == len(set(adj))


def test_fnaf1_sound_bank_names_are_nonempty(fnaf1_sound_bank: SoundBank):
    """Structural invariant: every sound has a non-empty name. A sound
    with an empty name is either decoder drift or a corrupted record —
    empirically FNAF 1 labels every sound."""
    bank = fnaf1_sound_bank

    empty = [s for s in bank.sounds if not s.name]
    assert not empty, (
        f"FNAF 1 has {len(empty)} SoundItem(s) with empty names — "
        f"first 3 raw_handles: {[s.raw_handle for s in empty[:3]]}"
    )
    # Basic smoke: at least 28 bytes of header were consumed per sound.
    assert all(s.record_wire_size > SOUND_ITEM_HEADER_SIZE for s in bank.sounds)


# --- Probe #9.1 payload-layer antibodies --------------------------------
#
# Probe #9 deferred audio_data-content decisions to #9.1. The recon ran
# on 2026-04-21 and found every FNAF 1 audio_data blob is a complete
# byte-valid RIFF/WAVE PCM container. These tests pin that empirical
# fact: if a future pack ships ADPCM / OGG / headerless PCM the decoder
# fails here before the env-gated audio_emit sink silently writes a
# file with a .wav extension that isn't actually a WAV.


def _parse_riff_fmt(data: bytes) -> dict | None:
    """Walk top-level RIFF chunks to pull `fmt ` and `data` chunk info.

    Returns None when `data` isn't a valid RIFF/WAVE container (short,
    wrong magic, or missing WAVE sub-type). On success returns the
    declared riff size, actual size, a structural consistency flag, the
    `fmt ` chunk unpacked fields, and the `data` chunk declared size.

    Minimal chunk walker — word-aligns odd-size chunks per RIFF spec.
    This isn't a full WAV parser: we don't recurse into LIST chunks,
    don't interpret extensible format headers, don't validate the
    `data` chunk's content. All that matters here is that `fmt ` is
    0x0001 (PCM) and the declared lengths reconcile to the blob size.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None

    (riff_size,) = struct.unpack("<I", data[4:8])
    pos = 12
    fmt_info: dict | None = None
    data_chunk_size: int | None = None
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        (chunk_size,) = struct.unpack("<I", data[pos + 4:pos + 8])
        body_start = pos + 8
        body_end = body_start + chunk_size
        if chunk_id == b"fmt " and chunk_size >= 16:
            (
                fmt_code,
                channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
            ) = struct.unpack("<HHIIHH", data[body_start:body_start + 16])
            fmt_info = {
                "fmt_code": fmt_code,
                "channels": channels,
                "sample_rate": sample_rate,
                "byte_rate": byte_rate,
                "block_align": block_align,
                "bits_per_sample": bits_per_sample,
                "fmt_chunk_size": chunk_size,
            }
        elif chunk_id == b"data":
            data_chunk_size = chunk_size
        # RIFF chunks are word-aligned — pad odd sizes.
        pos = body_end + (chunk_size & 1)

    return {
        "riff_declared_size": riff_size,
        "actual_total_size": len(data),
        "riff_declared_fits": riff_size + 8 == len(data),
        "fmt": fmt_info,
        "data_chunk_size": data_chunk_size,
    }


def test_fnaf1_sound_bank_all_riff_wave(fnaf1_sound_bank: SoundBank):
    """Probe #9.1 antibody: every `audio_data` blob must start with
    `RIFF` + `WAVE`. If this fails, the passthrough `emit_wav` sink
    would silently produce a `.wav` file that isn't a wav — catching it
    here before the sink writes anything."""
    bank = fnaf1_sound_bank

    non_riff = [
        (s.raw_handle, s.name, s.audio_data[:4].hex())
        for s in bank.sounds
        if not (len(s.audio_data) >= 12
                and s.audio_data[:4] == b"RIFF"
                and s.audio_data[8:12] == b"WAVE")
    ]
    assert not non_riff, (
        f"FNAF 1 has {len(non_riff)} audio_data blob(s) that aren't "
        f"RIFF/WAVE — first 3: {non_riff[:3]}"
    )


def test_fnaf1_sound_bank_all_pcm_format_code(fnaf1_sound_bank: SoundBank):
    """Probe #9.1 antibody: every fmt chunk must declare fmt_code
    0x0001 (uncompressed PCM). ADPCM (0x0002), IMA ADPCM (0x0011), and
    every other coded format would need real decoding before playback —
    emit_wav's passthrough only works for PCM.

    If a future pack ships a non-PCM sound, this fires, and the fix
    belongs at a `decoders/sounds_audio.py` module (not shipped yet —
    no demand on FNAF 1)."""
    bank = fnaf1_sound_bank

    non_pcm: list[tuple[int, str, int]] = []
    for snd in bank.sounds:
        info = _parse_riff_fmt(snd.audio_data)
        assert info is not None, (
            f"raw_handle={snd.raw_handle} name={snd.name!r}: "
            f"couldn't parse RIFF/WAVE header"
        )
        fmt = info["fmt"]
        assert fmt is not None, (
            f"raw_handle={snd.raw_handle} name={snd.name!r}: "
            f"no fmt chunk found"
        )
        if fmt["fmt_code"] != 0x0001:
            non_pcm.append((snd.raw_handle, snd.name, fmt["fmt_code"]))

    assert not non_pcm, (
        f"FNAF 1 has {len(non_pcm)} non-PCM sound(s) — first 3: "
        f"{non_pcm[:3]}. emit_wav's passthrough path only handles "
        f"fmt_code 0x0001; a real decoder is needed for the rest."
    )


def test_fnaf1_sound_bank_riff_size_matches_audio_data_size(
    fnaf1_sound_bank: SoundBank,
):
    """Probe #9.1 antibody: the RIFF-declared size field must reconcile
    to `len(audio_data)`. On disk this is the `riff_size + 8 ==
    total_file_size` invariant for single-RIFF files.

    A mismatch would mean our decoder's inner-zlib stripped or
    appended bytes, or that Clickteam wrapped something other than the
    raw file — either case breaks the passthrough contract."""
    bank = fnaf1_sound_bank

    mismatches: list[tuple[int, str, int, int]] = []
    for snd in bank.sounds:
        info = _parse_riff_fmt(snd.audio_data)
        assert info is not None, f"raw_handle={snd.raw_handle} not RIFF/WAVE"
        if not info["riff_declared_fits"]:
            mismatches.append((
                snd.raw_handle,
                snd.name,
                info["riff_declared_size"],
                info["actual_total_size"],
            ))
    assert not mismatches, (
        f"{len(mismatches)} sound(s) where RIFF declared size + 8 != "
        f"len(audio_data). First 3: {mismatches[:3]}"
    )


def test_fnaf1_sound_bank_audio_params_snapshot(fnaf1_sound_bank: SoundBank):
    """Probe #9.1 antibody #7 snapshot: pin the FNAF 1 audio parameter
    distribution (rate/channels/bps/data-size) against drift.

    Captured empirically 2026-04-21 (probe #9.1 recon):
    - 52/52 fmt_code 0x0001 PCM
    - sample rates: 22050 Hz × 42, 44100 Hz × 9, 11025 Hz × 1
    - channels:     mono × 49, stereo × 3
    - bps:          16 × 51, 8 × 1

    The SHA fingerprint captures the per-record tuple
    `(raw_handle, fmt_code, sample_rate, channels, bps, data_chunk_size)`
    — catches any per-record drift even if histograms happen to match."""
    bank = fnaf1_sound_bank

    fmt_codes = Counter()
    sample_rates = Counter()
    channels_hist = Counter()
    bps_hist = Counter()
    params_blob = b""
    for snd in bank.sounds:
        info = _parse_riff_fmt(snd.audio_data)
        assert info is not None and info["fmt"] is not None
        f = info["fmt"]
        fmt_codes[f["fmt_code"]] += 1
        sample_rates[f["sample_rate"]] += 1
        channels_hist[f["channels"]] += 1
        bps_hist[f["bits_per_sample"]] += 1
        params_blob += struct.pack(
            "<IHHIHHI",
            snd.raw_handle,
            f["fmt_code"],
            f["channels"],
            f["sample_rate"],
            f["bits_per_sample"],
            0,  # pad to align — keeps the struct layout stable
            info["data_chunk_size"] or 0,
        )

    params_sha = hashlib.sha256(params_blob).hexdigest()

    # Histogram pins — easy to read, fail loudly:
    assert fmt_codes == {0x0001: 52}
    assert sample_rates == {22050: 42, 44100: 9, 11025: 1}
    assert channels_hist == {1: 49, 2: 3}
    assert bps_hist == {16: 51, 8: 1}
    # Per-record fingerprint pin — catches name-file-drift or any
    # single-record parameter change that leaves histograms intact.
    assert params_sha == (
        "5641ab825255e726527b778862b79ef649ef96dac032a69d59cbaa890729e859"
    )
