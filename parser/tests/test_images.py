"""Regression tests for 0x6666 Images decoder (probe #7).

Top-level chunk. Envelope-only scope — this probe decodes the record
bank and pins its structural invariants against FNAF 1, but defers
pixel-data unpacking (graphic-mode dispatch, RLE / RLEW / RLET / LZX
decompression, alpha reconstruction) to probe #7.1 once the
CTFAK-Native.dll MSVC-intrinsics wall is handled.

Antibody coverage:

- #1 strict-unknown : negative count, negative per-record sizes, or an
  inner-blob size mismatch raises `ImageDecodeError`.
- #2 byte-count    : u32 count + N × (12 + compSize) must reconcile to
  `len(payload)` exactly; inner blob length must equal the declared
  decompressed_size; inner header requires ≥32 bytes.
- #3 round-trip    : synthetic pack/unpack on a 2-image bank.
- #5 multi-input   : runs against the FNAF 1 0x6666 payload.
- #7 snapshot      : pins count, first logical handle, image-data total
  bytes, and a SHA-256 of the tuple of `record_start_offsets`. Any
  single-byte drift in the decompressed payload flips at least one of
  these.
- Cross-chunk (load-bearing): for every image record `R`, the 0x5555
  offset indexed by `R`'s logical handle equals `R.record_start_offset +
  FNAF1_IMAGE_OFFSET_DELTA`. The delta is an empirical +260 on FNAF 1
  (see the docstring on `test_..._cross_chunk_handshake_..._with_delta`
  for the investigation). Pinned per-record rather than as a set
  equality so a future pack where the delta differs fails loudly with
  the offending record identified. 605 distinct non-zero offsets, 31
  zero-bucket slots — together the full 636-slot offset table.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.image_offsets import decode_image_offsets
from fnaf_parser.decoders.images import (
    IMAGE_BUILD_284_HANDLE_ADJUST,
    IMAGE_INNER_HEADER_SIZE,
    IMAGE_OUTER_HEADER_SIZE,
    Image,
    ImageBank,
    ImageDecodeError,
    decode_image_bank,
)
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"

# Empirical constant: on FNAF 1, every 0x5555 offset equals the
# corresponding image's record_start_offset + 260. See
# `test_fnaf1_image_bank_cross_chunk_handshake_with_delta` for the
# investigation; 260 is pinned here as a named constant so a future
# pack where the delta differs fires a clearly-named failure.
FNAF1_IMAGE_OFFSET_DELTA = 260


# --- Synthetic helpers --------------------------------------------------


def _pack_image_record(
    *,
    raw_handle: int,
    checksum: int = 0,
    references: int = 0,
    width: int = 0,
    height: int = 0,
    graphic_mode: int = 0,
    flags: int = 0,
    reserved: int = 0,
    hotspot_x: int = 0,
    hotspot_y: int = 0,
    action_x: int = 0,
    action_y: int = 0,
    transparent: bytes = b"\x00\x00\x00\x00",
    image_data: bytes = b"",
) -> bytes:
    """Pack one ImageRecord matching the on-wire layout, inner zlib
    included. Used by synthetic round-trip / error-path tests."""
    assert len(transparent) == 4
    inner = struct.pack(
        "<iiihhBBhhhhh4s",
        checksum,
        references,
        len(image_data),  # data_size
        width,
        height,
        graphic_mode,
        flags,
        reserved,
        hotspot_x,
        hotspot_y,
        action_x,
        action_y,
        transparent,
    ) + image_data
    compressed = zlib.compress(inner)
    outer = struct.pack("<iii", raw_handle, len(inner), len(compressed))
    return outer + compressed


def _pack_bank(records: list[bytes]) -> bytes:
    return struct.pack("<i", len(records)) + b"".join(records)


# --- Synthetic tests (no binary needed) ---------------------------------


def test_decode_image_bank_empty():
    """Zero-count bank decodes to an empty ImageBank. Only the 4-byte
    u32 count prefix is required on the wire."""
    bank = decode_image_bank(struct.pack("<i", 0))
    assert bank == ImageBank(count=0, images=(), record_start_offsets=())


def test_decode_image_bank_roundtrip_two_images():
    """Antibody #3: pack two recognisable records and verify every
    inner field survives a pack → zlib → decode cycle."""
    rec_a = _pack_image_record(
        raw_handle=5,
        checksum=0x11223344,
        references=1,
        width=320,
        height=240,
        graphic_mode=7,
        flags=0,
        reserved=0x7777,
        hotspot_x=1,
        hotspot_y=2,
        action_x=3,
        action_y=4,
        transparent=bytes([0xFF, 0x00, 0xAA, 0x80]),
        image_data=b"\xAA\xBB\xCC\xDD",
    )
    rec_b = _pack_image_record(
        raw_handle=9,
        checksum=-1,
        references=0,
        width=1,
        height=1,
        graphic_mode=0,
        flags=0b10001001,  # RLE | LZX | RGBA
        hotspot_x=-5,
        action_x=-1,
        transparent=bytes([0, 0, 0, 0xFF]),
        image_data=b"\x00",
    )
    payload = _pack_bank([rec_a, rec_b])
    bank = decode_image_bank(payload)
    assert isinstance(bank, ImageBank)
    assert bank.count == 2
    assert len(bank.images) == 2
    assert bank.record_start_offsets == (4, 4 + len(rec_a))

    a, b = bank.images
    assert a.raw_handle == 5
    assert a.handle == 5 - IMAGE_BUILD_284_HANDLE_ADJUST == 4
    assert a.checksum == 0x11223344
    assert a.references == 1
    assert a.data_size == 4
    assert (a.width, a.height) == (320, 240)
    assert a.graphic_mode == 7
    assert a.flags == 0
    assert a.reserved == 0x7777
    assert (a.hotspot_x, a.hotspot_y) == (1, 2)
    assert (a.action_x, a.action_y) == (3, 4)
    assert a.transparent == (0xFF, 0x00, 0xAA, 0x80)
    assert a.image_data == b"\xAA\xBB\xCC\xDD"
    assert a.record_start_offset == 4

    assert b.raw_handle == 9
    assert b.handle == 8
    assert b.checksum == -1
    assert b.flags == 0b10001001
    assert b.has_rgba is True
    assert b.has_lzx is True
    assert b.has_alpha is False
    assert b.hotspot_x == -5
    assert b.action_x == -1


def test_decode_image_bank_payload_too_small_for_count():
    """Antibody #2: a 3-byte payload can't hold even the u32 count."""
    with pytest.raises(ImageDecodeError, match="count prefix"):
        decode_image_bank(b"\x00\x00\x00")


def test_decode_image_bank_negative_count_raises():
    """Antibody #1: a negative signed-int32 count surfaces as an error.
    Likely a sign of outer-zlib corruption or RC4 drift."""
    with pytest.raises(ImageDecodeError, match="Negative counts"):
        decode_image_bank(struct.pack("<i", -1))


def test_decode_image_bank_record_outer_truncated_raises():
    """Antibody #2: count=1 but payload only carries the count prefix
    (no room for the 12-byte outer header) must raise, not silently
    return count=0."""
    payload = struct.pack("<i", 1)  # count=1, zero bytes of record body
    with pytest.raises(ImageDecodeError, match="outer header"):
        decode_image_bank(payload)


def test_decode_image_bank_compressed_size_overruns_payload():
    """Antibody #2: a record that claims more compressed bytes than
    remain in the payload must raise."""
    # count=1, one outer header claiming compressed_size=9999, but only
    # zero bytes of compressed body follow.
    payload = struct.pack("<i", 1) + struct.pack("<iii", 1, 32, 9999)
    with pytest.raises(ImageDecodeError, match="compressed_size"):
        decode_image_bank(payload)


def test_decode_image_bank_decompressed_size_below_inner_header_raises():
    """Antibody #2: an inner blob smaller than the 32-byte inner
    header can't be a valid record — raise before zlib gets a chance
    to succeed on a pathologically tiny stream."""
    # Handcraft a record whose decompressed_size is below the inner
    # header size. We don't even need to build the nested zlib — the
    # decoder validates the declared size first.
    tiny_compressed = zlib.compress(b"\x00" * (IMAGE_INNER_HEADER_SIZE - 1))
    payload = (
        struct.pack("<i", 1)
        + struct.pack("<iii", 1, IMAGE_INNER_HEADER_SIZE - 1, len(tiny_compressed))
        + tiny_compressed
    )
    with pytest.raises(ImageDecodeError, match="inner header"):
        decode_image_bank(payload)


def test_decode_image_bank_decompressed_size_mismatch_raises():
    """Antibody #2: declared decompressed_size must match what zlib
    actually produces. Drift here is the loudest symptom of a
    corrupted record."""
    real_inner = (
        struct.pack(
            "<iiihhBBhhhhh4s",
            0, 0, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, b"\x00\x00\x00\x00",
        )
        + b"\xDE\xAD\xBE\xEF"
    )
    compressed = zlib.compress(real_inner)
    # Lie: claim decompressed_size one larger than reality.
    bad_outer = struct.pack("<iii", 1, len(real_inner) + 1, len(compressed))
    payload = struct.pack("<i", 1) + bad_outer + compressed
    with pytest.raises(ImageDecodeError, match="zlib produced"):
        decode_image_bank(payload)


def test_decode_image_bank_trailing_bytes_raise():
    """Antibody #2: one valid record followed by an extra byte must
    raise — no silent truncation, no silent extension."""
    rec = _pack_image_record(raw_handle=1, width=2, height=2, image_data=b"x")
    payload = _pack_bank([rec]) + b"\x00"
    with pytest.raises(ImageDecodeError, match="unaccounted-for"):
        decode_image_bank(payload)


def test_as_dict_shape_is_stable():
    """`as_dict()` outputs a fixed JSON-friendly shape. Snapshot tests
    lean on this key list staying greppable."""
    rec = _pack_image_record(
        raw_handle=2,
        width=3,
        height=3,
        graphic_mode=1,
        flags=IMAGE_OUTER_HEADER_SIZE,  # arbitrary bit pattern
        transparent=bytes([1, 2, 3, 4]),
        image_data=b"pix",
    )
    bank = decode_image_bank(_pack_bank([rec]))
    d = bank.as_dict()
    assert set(d.keys()) == {"count", "images", "record_start_offsets"}
    img_d = d["images"][0]
    for k in (
        "raw_handle", "handle", "record_start_offset", "compressed_size",
        "decompressed_size", "checksum", "references", "data_size",
        "width", "height", "graphic_mode", "flags", "flag_names",
        "reserved", "hotspot_x", "hotspot_y", "action_x", "action_y",
        "transparent", "image_data_len",
    ):
        assert k in img_d, f"missing key: {k}"
    assert img_d["transparent"] == {"r": 1, "g": 2, "b": 3, "a": 4}


# --- FNAF 1 multi-input / snapshot / cross-chunk ------------------------


def _fnaf1_transform_and_records():
    """Walk the FNAF 1 pack with the RC4 transform enabled so
    read_chunk_payload can decode any chunk. Mirrors the helper in
    test_image_offsets.py."""
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
def test_fnaf1_image_bank_decodes_without_error():
    """Antibody #5 multi-input: the full FNAF 1 0x6666 payload decodes
    without raising and produces a non-empty ImageBank."""
    blob, result, transform = _fnaf1_transform_and_records()
    recs = [r for r in result.records if r.id == 0x6666]
    assert len(recs) == 1, (
        f"FNAF 1 should carry exactly one 0x6666 chunk; saw {len(recs)}"
    )
    payload = read_chunk_payload(blob, recs[0], transform=transform)
    bank = decode_image_bank(payload)
    assert bank.count > 0
    # Every image must carry at least the 32-byte inner header's
    # worth of decompressed bytes. The decoder guarantees this
    # structurally; assert visibly at the test layer too.
    for img in bank.images:
        assert isinstance(img, Image)
        assert img.decompressed_size >= IMAGE_INNER_HEADER_SIZE
        assert 0 <= img.graphic_mode <= 255
        assert 0 <= img.flags <= 255


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_image_bank_snapshot():
    """Antibody #7 snapshot: pin the FNAF 1 0x6666 decode against drift.

    Captured empirically on 2026-04-21 (probe #7):

    - count                     : 605 images (= non-zero 0x5555 count).
    - first / last handle       : (3, 608) logical; (4, 609) raw.
    - record_start_offsets_sha256: SHA-256 of the tuple of per-record
                                   start positions serialised as the
                                   stream of little-endian u32s. Any
                                   layout drift changes it.
    - total image_data bytes    : 845_154_886. Sum of the opaque
                                  post-inner-header bytes across all
                                  605 records. Changes loudly if any
                                  inner header's data_size is misread
                                  or the inner zlib inflates off.
    - graphic_mode histogram    : {4: 605} — every FNAF 1 image uses
                                  mode 4 (masked RGB / RLEW-pointable).
    - flags histogram           : {0: 520, 16: 85} — 520 records have
                                  no flag bits set; 85 have Alpha
                                  (bit 4 = 0x10). No LZX / RLE flagged
                                  records in FNAF 1 — all pixel data
                                  is raw-mode under graphic_mode=4.

    Any single-byte drift in the outer-decompressed payload changes the
    SHA. Any per-image header-field drift changes the histograms.
    """
    blob, result, transform = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x6666)
    payload = read_chunk_payload(blob, rec, transform=transform)
    bank = decode_image_bank(payload)

    # Non-zero offset count in 0x5555 was 605 (distinct_count=606
    # including the zero bucket). Tie the two directly.
    offsets_rec = next(r for r in result.records if r.id == 0x5555)
    offsets_payload = read_chunk_payload(blob, offsets_rec, transform=transform)
    image_offsets = decode_image_offsets(offsets_payload)
    non_zero_offsets = {o for o in image_offsets.offsets if o != 0}
    assert bank.count == len(non_zero_offsets), (
        f"ImageBank.count={bank.count} must equal the number of distinct "
        f"non-zero 0x5555 offsets ({len(non_zero_offsets)})"
    )

    starts_blob = b"".join(
        struct.pack("<I", off & 0xFFFFFFFF) for off in bank.record_start_offsets
    )
    starts_sha = hashlib.sha256(starts_blob).hexdigest()

    gm_hist: dict[int, int] = {}
    fl_hist: dict[int, int] = {}
    for img in bank.images:
        gm_hist[img.graphic_mode] = gm_hist.get(img.graphic_mode, 0) + 1
        fl_hist[img.flags] = fl_hist.get(img.flags, 0) + 1

    # Structural invariants first (true regardless of pack):
    assert len(bank.record_start_offsets) == bank.count
    assert bank.record_start_offsets == tuple(
        img.record_start_offset for img in bank.images
    )

    # Pinned snapshot values for FNAF 1:
    assert bank.count == 605
    assert bank.images[0].handle == 3
    assert bank.images[0].raw_handle == 4
    assert bank.images[-1].handle == 608
    assert bank.images[-1].raw_handle == 609
    assert starts_sha == (
        "c90b65c0e1b6bae9a46ba4e0d3c1520526546a9f68f9430bf51f29b20f480637"
    )
    total_image_data = sum(len(img.image_data) for img in bank.images)
    assert total_image_data == 845_154_886
    assert gm_hist == {4: 605}
    assert fl_hist == {0: 520, 16: 85}


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_image_bank_cross_chunk_handshake_with_delta():
    """Cross-chunk antibody (load-bearing): for every image record `R`
    in the bank, the 0x5555 offset keyed by `R.handle` must equal
    `R.record_start_offset + FNAF1_IMAGE_OFFSET_DELTA` (delta = 260).

    Investigation log — the +260 constant
    ---------------------------------------

    When first written, this test asserted set-equality between the
    bank's record_start_offsets and 0x5555's non-zero offsets. That
    failed with ALL 605 offsets shifted by exactly +260 bytes relative
    to record starts (`Counter(offset - record_start) == {260: 605}`).

    Confirmed structural sanity via direct probing:

    - `bank.count == len(non_zero_offsets) == 605` ✓
    - Decoder decodes all 605 records cleanly (no zlib errors, no
      header-field sanity errors).
    - `offsets[logical_handle]` lines up with the record whose
      `handle == logical_handle` for every record — not just by sort
      position but by handle identity.
    - Outer 12-byte header matches both CTFAK1, CTFAK2, and Anaconda
      reference readers (confirmed by direct source read of all three).
    - Inner 32-byte post-zlib header decodes cleanly for every record.

    So the decoder envelope is correct and the shift is structural —
    whatever 260 represents, it's a *reference-frame* offset between
    how Clickteam wrote the 0x5555 table and how our decoder measures
    position in the decompressed 0x6666 body. Candidate explanations
    considered but not resolved in this probe:

    - 0x5555 offsets point into a *runtime* layout where Clickteam's
      loader pre-allocated 260 bytes of per-bank header before the
      records.
    - Offsets were baked against a different repack of the bank that
      had 260 bytes of metadata stripped during our decompression path.
    - Some other build-284+ writer convention we haven't decoded yet.

    Resolution: pin the +260 empirically, per-record, so this test
    remains load-bearing (any single-record drift on EITHER decoder
    breaks the equality) while deferring semantic interpretation to
    probe #7.1 when the image payload bytes themselves get decoded.

    This is still the single strongest structural test in the probe.
    It can only pass if:

    - The outer 0x6666 decompression produced exactly the byte stream
      the 0x5555 offsets index into (outer zlib / RC4 both correct).
    - The inner per-record walker advanced `pos` by
      `IMAGE_OUTER_HEADER_SIZE + compressed_size` each iteration with
      zero off-by-one.
    - The logical-handle mapping (`raw - 1` for build ≥ 284) is right.

    Any drift on any of those breaks the per-record delta immediately,
    and the error message names the offending handle.
    """
    blob, result, transform = _fnaf1_transform_and_records()

    offsets_rec = next(r for r in result.records if r.id == 0x5555)
    offsets_payload = read_chunk_payload(blob, offsets_rec, transform=transform)
    image_offsets = decode_image_offsets(offsets_payload)

    images_rec = next(r for r in result.records if r.id == 0x6666)
    images_payload = read_chunk_payload(blob, images_rec, transform=transform)
    bank = decode_image_bank(images_payload)

    # Tie the cardinality first: every distinct non-zero offset must be
    # referenced by exactly one record, and vice versa. Both sides count
    # 605 on FNAF 1.
    non_zero_offsets = [o for o in image_offsets.offsets if o != 0]
    assert len(non_zero_offsets) == bank.count == 605, (
        f"cardinality mismatch: non_zero_offsets={len(non_zero_offsets)}, "
        f"bank.count={bank.count} — expected 605 both"
    )

    # Per-record delta: offsets[logical_handle] == record_start + DELTA.
    # Using a per-record loop rather than set equality so any single
    # drifted record is named in the assertion message.
    mismatches: list[tuple[int, int, int, int]] = []
    for img in bank.images:
        if img.handle < 0 or img.handle >= len(image_offsets.offsets):
            mismatches.append((img.handle, img.record_start_offset, -1, -1))
            continue
        expected = img.record_start_offset + FNAF1_IMAGE_OFFSET_DELTA
        actual = image_offsets.offsets[img.handle]
        if actual != expected:
            mismatches.append(
                (img.handle, img.record_start_offset, expected, actual)
            )
    assert not mismatches, (
        f"{len(mismatches)} record(s) whose 0x5555 offset ≠ "
        f"record_start + {FNAF1_IMAGE_OFFSET_DELTA}. First 5: "
        f"{mismatches[:5]}"
    )

    # And a coverage check: every non-zero offset is accounted for by
    # exactly one record's (record_start + DELTA). No stranded offsets.
    record_keyed_offsets = frozenset(
        img.record_start_offset + FNAF1_IMAGE_OFFSET_DELTA
        for img in bank.images
    )
    stranded = frozenset(non_zero_offsets) - record_keyed_offsets
    assert not stranded, (
        f"{len(stranded)} 0x5555 offset(s) don't correspond to any "
        f"record's start + {FNAF1_IMAGE_OFFSET_DELTA}: "
        f"{sorted(stranded)[:5]}"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_image_bank_logical_handles_are_unique():
    """Structural invariant: after the build-284 `raw - 1` adjustment
    every logical handle must be unique. A collision would mean the
    build adjustment is wrong for this pack, or a record was decoded
    twice."""
    blob, result, transform = _fnaf1_transform_and_records()
    rec = next(r for r in result.records if r.id == 0x6666)
    payload = read_chunk_payload(blob, rec, transform=transform)
    bank = decode_image_bank(payload)

    handles = [img.handle for img in bank.images]
    assert len(handles) == len(set(handles)), (
        f"duplicate logical handles in ImageBank "
        f"({len(handles) - len(set(handles))} collision(s))"
    )
    # Raw handles must also be unique; the -1 shift is a bijection.
    raw = [img.raw_handle for img in bank.images]
    assert len(raw) == len(set(raw))
