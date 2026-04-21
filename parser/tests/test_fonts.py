"""Regression tests for 0x6667 Fonts decoder (probe #10) — envelope-only.

Record bank — font peer of 0x6666 Images / 0x6668 Sounds, indexed by
0x5556 FontOffsets. Envelope scope only: decode the bank header, each
item's zlib-wrapped inner blob, and the LogFont struct; pin the
cross-chunk handshake with FontOffsets.

Oracle split resolved here (load-bearing for probe #10)
-------------------------------------------------------

CTFAK2 FontBank.cs uses strict `> 284` → FNAF 1 (build=284) keeps raw
handles. Anaconda fontbank.py uses `>= 284` → FNAF 1 subtracts 1.
Exact opposite resolutions on the same input.

`test_fnaf1_fonts_cross_chunk_offset_handshake` picks the winner
empirically using the same `_deltas(key_attr)` + XOR-singular `Counter`
pattern that settled the sound handle convention — whichever
interpretation yields a single-valued delta against
`FontOffsets.offsets[h] - font.record_start_offset` wins.

Antibody coverage:

- #1 strict-unknown : negative count / negative sizes / sign-bit-set
  u32 sizes raise `FontBankDecodeError`; zlib overrun raises; mismatched
  inner blob length raises.
- #2 byte-count    : count prefix + N × record wire size reconciles to
  exactly `len(payload)`.
- #3 round-trip    : synthetic pack/unpack covers both compressed and
  uncompressed item bodies.
- #4 multi-oracle  : field order + sizes mirror CTFAK2 + Anaconda.
- #5 multi-input   : runs against the FNAF 1 0x6667 payload.
- #7 snapshot      : count, raw handle set, face-name set, and a
  SHA-256 of `(raw_handle, weight, italic, face_name_utf8)` tuples
  pinned against drift.
- Cross-chunk     : `FontBank.record_start_offsets` must equal
  `{o for o in FontOffsets.offsets if o != 0}` — or equivalently
  produce a single-valued delta under one handle convention.
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
from fnaf_parser.decoders.font_offsets import decode_font_offsets
from fnaf_parser.decoders.fonts import (
    FONT_BANK_COUNT_SIZE,
    FONT_DECOMP_HEADER_SIZE,
    FONT_HANDLE_ADJUST,
    FONT_INNER_SIZE,
    FONT_INNER_TRIAD_SIZE,
    FONT_ITEM_HANDLE_SIZE,
    LOG_FONT_FACE_NAME_SIZE,
    LOG_FONT_FIXED_SIZE,
    LOG_FONT_SIZE,
    Font,
    FontBank,
    FontBankDecodeError,
    LogFont,
    decode_font_bank,
)
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START
from tests.fnaf1_constants import FNAF1_BANK_OFFSET_DELTA

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"

# Alias for local readability — see `tests/fnaf1_constants.py`. Probe
# #10 confirmed the same +260 delta first seen in images (#7) and
# sounds (#9). Three independent banks, one pack-level constant.
FNAF1_FONT_OFFSET_DELTA = FNAF1_BANK_OFFSET_DELTA


# --- Synthetic pack helpers ---------------------------------------------


def _pack_log_font(
    *,
    height: int = -12,
    width: int = 0,
    escapement: int = 0,
    orientation: int = 0,
    weight: int = 400,
    italic: int = 0,
    underline: int = 0,
    strike_out: int = 0,
    char_set: int = 0,
    out_precision: int = 0,
    clip_precision: int = 0,
    quality: int = 0,
    pitch_and_family: int = 0,
    face_name: str = "Tahoma",
) -> bytes:
    """Pack a LogFont synthetic blob. Mirrors the decoder's struct
    definition exactly — any drift surfaces as a length mismatch."""
    fixed = struct.pack(
        "<iiiiiBBBBBBBB",
        height,
        width,
        escapement,
        orientation,
        weight,
        italic,
        underline,
        strike_out,
        char_set,
        out_precision,
        clip_precision,
        quality,
        pitch_and_family,
    )
    assert len(fixed) == LOG_FONT_FIXED_SIZE
    # NUL-pad face_name to 32 wchars (64 bytes).
    encoded = face_name.encode("utf-16-le")
    assert len(encoded) <= LOG_FONT_FACE_NAME_SIZE, (
        f"face_name {face_name!r} too long for test packer"
    )
    padded = encoded + b"\x00" * (LOG_FONT_FACE_NAME_SIZE - len(encoded))
    blob = fixed + padded
    assert len(blob) == LOG_FONT_SIZE
    return blob


def _pack_inner_blob(
    *,
    checksum: int = 0,
    references: int = 1,
    declared_size: int = 0,
    **log_font_kwargs,
) -> bytes:
    """Pack the 104-byte inner blob (triad + LogFont)."""
    triad = struct.pack("<iii", checksum, references, declared_size)
    assert len(triad) == FONT_INNER_TRIAD_SIZE
    lf = _pack_log_font(**log_font_kwargs)
    blob = triad + lf
    assert len(blob) == FONT_INNER_SIZE
    return blob


def _pack_font_item_compressed(raw_handle: int, inner_blob: bytes) -> bytes:
    """Pack a compressed FontItem: u32 handle + u32 decomp + u32 comp +
    zlib stream."""
    compressed = zlib.compress(inner_blob)
    return (
        struct.pack("<III", raw_handle, len(inner_blob), len(compressed))
        + compressed
    )


def _pack_font_item_uncompressed(raw_handle: int, inner_blob: bytes) -> bytes:
    """Pack an uncompressed FontItem: u32 handle + raw 104-byte body."""
    assert len(inner_blob) == FONT_INNER_SIZE
    return struct.pack("<I", raw_handle) + inner_blob


def _pack_bank(items: list[bytes]) -> bytes:
    """Pack the FontBank envelope: u32 count + concatenated items."""
    return struct.pack("<I", len(items)) + b"".join(items)


# --- Synthetic tests (no binary needed) ---------------------------------


def test_font_inner_size_invariants():
    """Pin the derived layout constants — any accidental field drift
    surfaces here before anything touches real data."""
    assert FONT_INNER_TRIAD_SIZE == 12
    assert LOG_FONT_FIXED_SIZE == 28
    assert LOG_FONT_FACE_NAME_SIZE == 64
    assert LOG_FONT_SIZE == 92
    assert FONT_INNER_SIZE == 104
    assert FONT_BANK_COUNT_SIZE == 4
    assert FONT_ITEM_HANDLE_SIZE == 4
    assert FONT_DECOMP_HEADER_SIZE == 8
    assert FONT_HANDLE_ADJUST == 1


def test_decode_font_bank_empty():
    """A payload with count=0 and nothing else decodes to an empty bank."""
    fb = decode_font_bank(struct.pack("<I", 0))
    assert fb == FontBank(count=0, fonts=(), record_start_offsets=())


def test_decode_font_bank_negative_count_raises():
    """Antibody #1: count reinterpreted as i32 must reject negatives."""
    payload = struct.pack("<i", -1)
    with pytest.raises(FontBankDecodeError, match="count prefix decoded as -1"):
        decode_font_bank(payload)


def test_decode_font_bank_short_payload_raises():
    """Antibody #2: payload shorter than the 4-byte count prefix."""
    for n in range(FONT_BANK_COUNT_SIZE):
        with pytest.raises(FontBankDecodeError, match="count prefix"):
            decode_font_bank(b"\x00" * n)


def test_decode_font_bank_roundtrip_compressed():
    """Antibody #3: synthetic compressed bank round-trips."""
    items = [
        _pack_font_item_compressed(
            raw_handle=1,
            inner_blob=_pack_inner_blob(
                checksum=-1, references=1, face_name="Tahoma", weight=400
            ),
        ),
        _pack_font_item_compressed(
            raw_handle=2,
            inner_blob=_pack_inner_blob(
                checksum=0xABCD, references=3, face_name="Arial", weight=700
            ),
        ),
    ]
    payload = _pack_bank(items)
    fb = decode_font_bank(payload)
    assert fb.count == 2
    assert len(fb.fonts) == 2

    f0, f1 = fb.fonts
    assert f0.raw_handle == 1
    assert f0.handle == 0  # raw - 1
    assert f0.face_name == "Tahoma"
    assert f0.log_font.weight == 400
    assert f0.log_font.is_bold is False

    assert f1.raw_handle == 2
    assert f1.handle == 1
    assert f1.face_name == "Arial"
    assert f1.log_font.weight == 700
    assert f1.log_font.is_bold is True

    # Record start offsets must be strictly ascending (bank is a linear
    # stream of records).
    assert list(fb.record_start_offsets) == sorted(fb.record_start_offsets)
    assert fb.record_start_offsets[0] == FONT_BANK_COUNT_SIZE


def test_decode_font_bank_roundtrip_uncompressed():
    """Antibody #3: uncompressed branch produces the same fields."""
    items = [
        _pack_font_item_uncompressed(
            raw_handle=42,
            inner_blob=_pack_inner_blob(face_name="Consolas"),
        ),
    ]
    payload = _pack_bank(items)
    fb = decode_font_bank(payload, is_compressed=False)
    assert fb.count == 1
    (f,) = fb.fonts
    assert f.raw_handle == 42
    assert f.handle == 41
    assert f.face_name == "Consolas"
    assert f.compressed_size == 0
    assert f.decompressed_size == FONT_INNER_SIZE


def test_decode_font_bank_by_handle_maps():
    """`by_handle` vs `by_raw_handle`: the former applies the Anaconda
    -1 adjustment, the latter keeps the on-wire value."""
    items = [
        _pack_font_item_compressed(
            raw_handle=5, inner_blob=_pack_inner_blob(face_name="X")
        ),
        _pack_font_item_compressed(
            raw_handle=10, inner_blob=_pack_inner_blob(face_name="Y")
        ),
    ]
    fb = decode_font_bank(_pack_bank(items))
    assert set(fb.by_raw_handle) == {5, 10}
    assert set(fb.by_handle) == {4, 9}
    assert fb.handles == frozenset({4, 9})
    assert fb.face_names == ("X", "Y")


def test_decode_font_bank_zlib_overrun_raises():
    """Antibody #1/#2: a compressed_size that overruns the payload."""
    # Handcraft a single item that claims a compressed_size > remaining.
    payload = (
        struct.pack("<I", 1)  # count
        + struct.pack("<I", 1)  # raw_handle
        + struct.pack("<II", FONT_INNER_SIZE, 9_000_000)  # decomp, comp
        + b"\x00" * 4  # only 4 bytes of "zlib" stream
    )
    with pytest.raises(FontBankDecodeError, match="overruns payload"):
        decode_font_bank(payload)


def test_decode_font_bank_wrong_decompressed_size_raises():
    """Antibody #2: declared decompressed_size disagrees with zlib output."""
    inner = _pack_inner_blob()
    compressed = zlib.compress(inner)
    # Claim decomp_size = FONT_INNER_SIZE + 1 (a single byte too large).
    payload = (
        struct.pack("<I", 1)
        + struct.pack("<I", 1)
        + struct.pack("<II", FONT_INNER_SIZE + 1, len(compressed))
        + compressed
    )
    with pytest.raises(FontBankDecodeError, match="decompressed_size"):
        decode_font_bank(payload)


def test_decode_font_bank_face_name_nul_trimmed():
    """`face_name` must be trimmed at the first NUL wchar. Feed a name
    with internal padding and verify it's stripped."""
    items = [
        _pack_font_item_compressed(
            raw_handle=1,
            inner_blob=_pack_inner_blob(face_name="Verdana"),
        )
    ]
    fb = decode_font_bank(_pack_bank(items))
    (f,) = fb.fonts
    assert f.face_name == "Verdana"
    # Full face_name slice is 64 bytes; the rest is NUL-padded.
    assert len(f.face_name.encode("utf-16-le")) < LOG_FONT_FACE_NAME_SIZE


def test_decode_font_bank_trailing_bytes_raise():
    """Antibody #2: trailing bytes after the last record fire the
    tail-consumption check."""
    items = [
        _pack_font_item_compressed(
            raw_handle=1, inner_blob=_pack_inner_blob()
        )
    ]
    payload = _pack_bank(items) + b"\xFF\xFF\xFF"
    with pytest.raises(FontBankDecodeError, match="unaccounted-for"):
        decode_font_bank(payload)


# --- FNAF 1 multi-input (Antibody #5 / #7 / cross-chunk) ----------------


def _fnaf1_transform_and_records():
    """Walk the FNAF 1 pack with the RC4 transform enabled so
    read_chunk_payload can decode any chunk."""
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
def test_fnaf1_fonts_decodes_without_error():
    """Antibody #5 multi-input: the full FNAF 1 0x6667 payload decodes
    without raising and produces at least one font record."""
    blob, result, _ = _fnaf1_transform_and_records()
    recs = [r for r in result.records if r.id == 0x6667]
    assert len(recs) == 1, (
        f"FNAF 1 should carry exactly one 0x6667 chunk; saw {len(recs)}"
    )
    payload = read_chunk_payload(blob, recs[0])
    fb = decode_font_bank(payload)
    assert fb.count >= 1
    for f in fb.fonts:
        assert f.raw_handle >= 0
        assert f.decompressed_size == FONT_INNER_SIZE
        assert isinstance(f.log_font, LogFont)


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_fonts_snapshot():
    """Antibody #7 snapshot: pin FNAF 1 0x6667 decode against drift.

    Snapshot digest captures the load-bearing per-font fields
    (raw_handle, weight, italic, face_name as UTF-8) so any single-byte
    drift in the decompressed payload flips the hash.
    """
    blob, result, _ = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x6667)
    payload = read_chunk_payload(blob, rec)
    fb = decode_font_bank(payload)

    # Snapshot digest — grep-friendly binary blob of the load-bearing
    # per-font fields.
    parts: list[bytes] = []
    for f in fb.fonts:
        name_bytes = f.face_name.encode("utf-8")
        parts.append(
            struct.pack(
                "<IiB I",
                f.raw_handle,
                f.log_font.weight,
                f.log_font.italic,
                len(name_bytes),
            )
        )
        parts.append(name_bytes)
    items_sha = hashlib.sha256(b"".join(parts)).hexdigest()

    # Debug print — surfaces empirical values on first run so the
    # snapshot expectations below can be bootstrapped.
    print(
        f"\n[fonts snapshot] count={fb.count}\n"
        f"raw_handles={[f.raw_handle for f in fb.fonts]}\n"
        f"face_names={list(fb.face_names)}\n"
        f"weights={[f.log_font.weight for f in fb.fonts]}\n"
        f"italics={[f.log_font.italic for f in fb.fonts]}\n"
        f"record_start_offsets={list(fb.record_start_offsets)}\n"
        f"items_sha={items_sha}"
    )

    # Pinned empirical values captured 2026-04-21 (probe #10).
    #
    # FNAF 1 uses 7 fonts: two bold variants of "LCD Solid" (the
    # door/clock digital readouts), one "Lucida Sans Unicode" (likely
    # the story-mode body text), three Consolas weights (debug /
    # menu / HUD), one "Tahoma" (default Windows GDI fallback). Order
    # is NOT by raw_handle — it's by appearance order in the pack.
    assert fb.count == 7
    assert [f.raw_handle for f in fb.fonts] == [4, 3, 2, 5, 1, 7, 6]
    # Logical handles (Anaconda convention, `raw - 1`) — the cross-
    # chunk handshake below pins this as the winning interpretation.
    assert [f.handle for f in fb.fonts] == [3, 2, 1, 4, 0, 6, 5]
    assert list(fb.face_names) == [
        "LCD Solid",
        "Lucida Sans Unicode",
        "Consolas",
        "Consolas",
        "LCD Solid",
        "Consolas",
        "Tahoma",
    ]
    assert [f.log_font.weight for f in fb.fonts] == [700, 700, 400, 400, 700, 400, 400]
    assert all(f.log_font.italic == 0 for f in fb.fonts)
    # Record start offsets within the decompressed outer 0x6667 body.
    # The cross-chunk test proves these correspond 1:1 to non-zero
    # entries of FontOffsets (with a constant +260 delta).
    assert fb.record_start_offsets == (4, 66, 142, 199, 255, 317, 373)
    # Load-bearing SHA — flips on any per-font drift.
    assert items_sha == (
        "e51d9bec996821e2c334d729a4437117c70b7cecbbc969e9263986475e871afa"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_fonts_cross_chunk_offset_handshake():
    """Cross-chunk antibody: each font's `record_start_offset` (in the
    decompressed outer 0x6667 body) must correspond to a non-zero offset
    in 0x5556 FontOffsets — modulo a single constant delta.

    This mirrors the handshake pattern from `test_sounds.py`:

    1. Compute `delta[h] = font_offsets.offsets[h] - font.record_start_offset`
       under each handle interpretation (`raw_handle` vs `handle`).
    2. Whichever interpretation yields a single-valued delta across all
       records wins — that's the handle convention the pack uses.
    3. XOR-singular: exactly one of the two must be singular. Both
       singular would be suspicious, neither singular means the whole
       offset table is wrong.

    This resolves the CTFAK2 (`> 284`) vs Anaconda (`>= 284`) oracle
    split empirically. Records the winning delta for comparison against
    the +260 sound/image deltas.
    """
    blob, result, _ = _fnaf1_transform_and_records()
    fo_rec = next(r for r in result.records if r.id == 0x5556)
    fb_rec = next(r for r in result.records if r.id == 0x6667)

    font_offsets = decode_font_offsets(read_chunk_payload(blob, fo_rec))
    font_bank = decode_font_bank(read_chunk_payload(blob, fb_rec))

    non_zero_offsets = [o for o in font_offsets.offsets if o != 0]

    # The offset table and the record bank must agree on population
    # size after filtering FontOffsets zeros.
    assert len(non_zero_offsets) == font_bank.count, (
        f"0x5556 non-zero offsets ({len(non_zero_offsets)}) disagree with "
        f"0x6667 font count ({font_bank.count}). Envelope handshake "
        f"failed before handle resolution."
    )

    def _deltas(key_attr: str) -> Counter[int]:
        """Compute per-record `offset[h] - record_start_offset` under
        the given handle attribute and return a frequency Counter.

        Missing handles (where `font_offsets.offsets[h]` is out of
        bounds or zero) are skipped — the caller wants the delta only
        for handles that actually index into the offset table."""
        counter: Counter[int] = Counter()
        for f in font_bank.fonts:
            h = getattr(f, key_attr)
            if not 0 <= h < font_offsets.count:
                continue
            offset = font_offsets.offsets[h]
            if offset == 0:
                continue
            counter[offset - f.record_start_offset] += 1
        return counter

    raw_deltas = _deltas("raw_handle")
    adj_deltas = _deltas("handle")

    raw_singular = len(raw_deltas) == 1 and sum(raw_deltas.values()) == font_bank.count
    adj_singular = len(adj_deltas) == 1 and sum(adj_deltas.values()) == font_bank.count

    print(
        f"\n[fonts cross-chunk] "
        f"raw_deltas={dict(raw_deltas)} adj_deltas={dict(adj_deltas)} "
        f"raw_singular={raw_singular} adj_singular={adj_singular}"
    )

    # XOR: exactly one convention must produce a single delta across
    # all records. Both True is suspicious (coincidence at this scale
    # is implausible); both False means the offset table disagrees
    # with the bank — cross-chunk drift.
    assert raw_singular ^ adj_singular, (
        f"Expected exactly one of raw/adjusted handle to yield a "
        f"singular delta. Got raw={dict(raw_deltas)}, adj={dict(adj_deltas)}."
    )

    if raw_singular:
        winning_attr = "raw_handle"
        winning_delta = next(iter(raw_deltas))
    else:
        winning_attr = "handle"
        winning_delta = next(iter(adj_deltas))

    print(
        f"[fonts cross-chunk] winning_attr={winning_attr} "
        f"winning_delta={winning_delta} (=0x{winning_delta:x})"
    )

    # Pin the resolution. If this fires after a future parser change,
    # the pack's handle convention flipped — re-investigate before
    # trusting `FONT_HANDLE_ADJUST`.
    #
    # Winning attribute identifies the oracle: "raw_handle" → CTFAK2
    # (strict `> 284`, no adjust), "handle" → Anaconda (`>= 284`, apply
    # -1). FNAF 1 empirically resolved as Anaconda's convention.
    assert winning_attr == "handle", (
        f"expected Anaconda-style adjusted handle (-1) to win on FNAF 1, "
        f"got {winning_attr!r}"
    )
    # Third independent confirmation of the pack-level +260 delta —
    # after probe #7 (images) and probe #9 (sounds). The shared
    # `FNAF1_BANK_OFFSET_DELTA` now lives in `tests/fnaf1_constants.py`.
    assert winning_delta == FNAF1_FONT_OFFSET_DELTA, (
        f"expected same +{FNAF1_FONT_OFFSET_DELTA} offset delta as the "
        f"image/sound banks (tests.fnaf1_constants.FNAF1_BANK_OFFSET_DELTA), "
        f"got {winning_delta}"
    )
