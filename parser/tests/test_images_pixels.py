"""Regression tests for 0x6666 pixel-payload decoder (probe #7.1a).

Scope: the `graphic_mode == 4, flags == 0` path — 520 of FNAF 1's 605
image records. Every other mode/flag combination is expected to raise,
so flag=16 (85 records) staying as a deferred-work signal until probe
#7.1b lands is enforced here.

Antibody coverage:

- #1 strict-unknown : mode != 4, flags != 0, truncated image_data,
  non-positive width/height — all raise `ImagePixelsDecodeError`.
- #2 byte-count    : output is always exactly `w * h * 4` bytes; row
  padding (3 bytes on odd-width rows, 0 on even) is consumed from the
  input and not emitted to the output.
- #3 round-trip    : synthetic 2×2 and 3×2 streams decode to exactly
  the expected RGBA byte sequences including the odd-width pad.
- #5 multi-input   : all 520 FNAF 1 flag=0 records decode cleanly,
  every output is `w * h * 4` bytes long.
- #7 snapshot      : pins total decoded bytes (sum of w*h*4 across
  flag=0 records) and a SHA-256 fingerprint computed over the
  per-record pairs `(handle, sha256(rgba))`. Any single-pixel drift
  on any record flips the fingerprint and names the handle.
- Deferred-work  : every flag=16 record raises; pinned as 85 raises
  so a future pack that silently reclassifies a flag=0 record trips
  the count mismatch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.images import Image, decode_image_bank
from fnaf_parser.decoders.images_pixels import (
    DecodedPixels,
    ImagePixelsDecodeError,
    decode_flag0_bgr_masked,
    decode_image_pixels,
    get_row_padding_bytes,
)
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


# --- Row padding (pure function) ----------------------------------------


def test_row_padding_even_widths_are_zero():
    """Antibody #2: even widths need zero row padding at 24bpp because
    `width * 3` is always a multiple of 2. Spot-check a range."""
    for w in (2, 4, 6, 8, 10, 100, 320, 1280):
        assert get_row_padding_bytes(w) == 0, f"even width {w} wants 0 pad"


def test_row_padding_odd_widths_are_three():
    """Antibody #2: odd widths need exactly 3 bytes of pad at 24bpp —
    CTFAK2's `ceil(1 / 3) = 1` point-size chunk = 3 bytes."""
    for w in (1, 3, 5, 7, 9, 11, 101, 321, 1279):
        assert get_row_padding_bytes(w) == 3, f"odd width {w} wants 3 pad"


def test_row_padding_rejects_non_positive():
    """A width ≤ 0 is undefined — pushing the error up to the caller
    where it's disambiguated against the rest of the input."""
    for w in (0, -1, -7):
        with pytest.raises(ImagePixelsDecodeError):
            get_row_padding_bytes(w)


# --- Synthetic pixel decode (no binary needed) --------------------------


def test_decode_2x2_opaque_produces_expected_rgba():
    """Antibody #3 round-trip: a 2×2 BGR stream with a non-matching
    transparent colour produces 4 opaque RGBA pixels. Byte order is
    checked explicitly — this is the test that catches a B/R swap
    before it ships."""
    # 4 pixels, BGR on the wire: (B=1, G=2, R=3), (B=10, G=20, R=30),
    # (B=4, G=5, R=6), (B=40, G=50, R=60). Width even → no row pad.
    src = bytes([
        1, 2, 3,   10, 20, 30,
        4, 5, 6,   40, 50, 60,
    ])
    # transparent is (R, G, B, A) — pick something no pixel matches.
    rgba = decode_flag0_bgr_masked(src, width=2, height=2, transparent=(200, 200, 200, 0))
    assert len(rgba) == 2 * 2 * 4
    assert list(rgba) == [
        3, 2, 1, 255,   30, 20, 10, 255,
        6, 5, 4, 255,   60, 50, 40, 255,
    ]


def test_decode_2x2_transparent_masks_match():
    """Antibody #3: a pixel whose (R,G,B) matches the transparent
    colour becomes A=0 while every other pixel stays A=255. Channel
    order is still checked via the non-matching pixels around it."""
    src = bytes([
        1, 2, 3,   10, 20, 30,
        4, 5, 6,   40, 50, 60,
    ])
    # Mark the first pixel (R=3, G=2, B=1) transparent.
    rgba = decode_flag0_bgr_masked(src, width=2, height=2, transparent=(3, 2, 1, 255))
    # First pixel: RGB preserved, alpha zeroed.
    assert rgba[0:4] == bytes([3, 2, 1, 0])
    # Second pixel: not a match → alpha stays 255.
    assert rgba[4:8] == bytes([30, 20, 10, 255])
    # Rows 2: both still opaque.
    assert rgba[8:12] == bytes([6, 5, 4, 255])
    assert rgba[12:16] == bytes([60, 50, 40, 255])


def test_decode_3x2_consumes_odd_width_row_padding():
    """Antibody #3: with width=3 each row has 3 bytes of pad after the
    9 data bytes. The padded bytes must be consumed from the stream
    (advancing src_pos) but never emitted to output."""
    # Row 0: pixels (1,2,3), (4,5,6), (7,8,9) on wire as BGR. Then 3
    # byte row pad filled with sentinel 0xCC. Row 1 same shape.
    row0 = bytes([1, 2, 3,  4, 5, 6,  7, 8, 9,  0xCC, 0xCC, 0xCC])
    row1 = bytes([10, 20, 30,  40, 50, 60,  70, 80, 90,  0xCC, 0xCC, 0xCC])
    src = row0 + row1
    rgba = decode_flag0_bgr_masked(src, width=3, height=2, transparent=(255, 255, 255, 0))
    assert len(rgba) == 3 * 2 * 4
    # Output must NOT contain 0xCC — if the pad isn't stripped the
    # sentinel would bleed into downstream pixels.
    assert 0xCC not in rgba
    assert list(rgba[:12]) == [3, 2, 1, 255,  6, 5, 4, 255,  9, 8, 7, 255]
    assert list(rgba[12:24]) == [30, 20, 10, 255,  60, 50, 40, 255,  90, 80, 70, 255]


def test_decode_tolerates_trailing_bytes_after_last_row():
    """Some FNAF 1 records have a handful of leftover bytes past the
    required height × row stride — CTFAK2 and Anaconda both silently
    discard them. Pin that behaviour explicitly: an input longer than
    required must still decode cleanly.

    The test deliberately appends sentinel 0xCC bytes to make sure
    they don't bleed into the decoded RGBA output."""
    src = bytes([1, 2, 3,  4, 5, 6]) + b"\xCC\xCC\xCC\xCC\xCC"
    rgba = decode_flag0_bgr_masked(src, width=2, height=1, transparent=(9, 9, 9, 0))
    assert len(rgba) == 2 * 1 * 4
    assert list(rgba) == [3, 2, 1, 255,  6, 5, 4, 255]


def test_decode_truncated_image_data_raises():
    """Antibody #1 strict-unknown / #2 byte-count: input shorter than
    `height * (width*3 + pad)` raises. A silent truncation would
    surface later as a wrong-size PNG."""
    # width=2 height=2 needs 12 bytes; give 11.
    src = bytes([1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11])
    with pytest.raises(ImagePixelsDecodeError, match="truncated"):
        decode_flag0_bgr_masked(src, width=2, height=2, transparent=(0, 0, 0, 0))


def test_decode_non_positive_dimensions_raise():
    """Antibody #1: zero or negative width/height is nonsense at the
    pixel layer. Inner-header drift would be the most likely source."""
    for w, h in [(0, 2), (2, 0), (-1, 2), (2, -1)]:
        with pytest.raises(ImagePixelsDecodeError):
            decode_flag0_bgr_masked(b"", width=w, height=h, transparent=(0, 0, 0, 0))


# --- Dispatcher tests (synthetic Image records) -------------------------


def _synth_image(
    *,
    handle: int = 1,
    width: int = 2,
    height: int = 2,
    graphic_mode: int = 4,
    flags: int = 0,
    transparent: tuple[int, int, int, int] = (0, 0, 0, 0),
    image_data: bytes = b"",
) -> Image:
    """Build an `Image` envelope dataclass without round-tripping
    through the 0x6666 decoder. Dispatcher tests don't care about the
    zlib layer."""
    return Image(
        raw_handle=handle + 1,
        handle=handle,
        record_start_offset=0,
        compressed_size=0,
        decompressed_size=32 + len(image_data),
        checksum=0,
        references=0,
        data_size=len(image_data),
        width=width,
        height=height,
        graphic_mode=graphic_mode,
        flags=flags,
        reserved=0,
        hotspot_x=0,
        hotspot_y=0,
        action_x=0,
        action_y=0,
        transparent=transparent,
        image_data=image_data,
    )


def test_dispatcher_rejects_non_mode4():
    """Antibody #1: FNAF 1 is mode-4-only. A mode-7 record means either
    a different pack shipped into this pipeline, or envelope drift —
    either way, stop before making garbage pixels."""
    img = _synth_image(graphic_mode=7, image_data=b"\x00" * 12)
    with pytest.raises(ImagePixelsDecodeError, match="graphic_mode"):
        decode_image_pixels(img)


def test_dispatcher_rejects_flag16_as_deferred_work():
    """Antibody #1 / deferred-work: flag=0x10 (Alpha) raises with a
    pointer to probe #7.1b. The 85 FNAF 1 records in this bucket all
    surface through this raise until the alpha-plane path ships."""
    img = _synth_image(flags=0x10, image_data=b"\x00" * 12)
    with pytest.raises(ImagePixelsDecodeError, match="probe #7.1b"):
        decode_image_pixels(img)


def test_dispatcher_rejects_arbitrary_unknown_flag():
    """Any non-zero flag combination beyond the 0x10 deferred bucket
    must also raise — no silent fall-through into flag=0's main loop
    for, e.g., RLE or LZX packs that don't exist in FNAF 1."""
    for bad in (0x01, 0x04, 0x08, 0x20, 0x80, 0xFF):
        img = _synth_image(flags=bad, image_data=b"\x00" * 12)
        with pytest.raises(ImagePixelsDecodeError):
            decode_image_pixels(img)


def test_dispatcher_returns_decoded_pixels_dataclass():
    """Happy path: a minimal 2×2 flag=0 record returns a
    `DecodedPixels` carrying handle, dims, and rgba bytes."""
    src = bytes([1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12])
    img = _synth_image(
        handle=42,
        width=2,
        height=2,
        image_data=src,
        transparent=(200, 200, 200, 0),
    )
    dp = decode_image_pixels(img)
    assert isinstance(dp, DecodedPixels)
    assert dp.handle == 42
    assert (dp.width, dp.height) == (2, 2)
    assert len(dp.rgba) == 2 * 2 * 4
    assert list(dp.rgba[:4]) == [3, 2, 1, 255]


# --- FNAF 1 multi-input / snapshot (Antibody #5 / #7) -------------------


def _fnaf1_image_bank():
    """Decode the FNAF 1 0x6666 ImageBank envelope. Helper mirrors the
    one in test_images.py — RC4 transform derived from header strings."""
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

    rec = next(r for r in result.records if r.id == 0x6666)
    payload = read_chunk_payload(blob, rec, transform=transform)
    return decode_image_bank(payload)


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_all_flag0_records_decode_without_error():
    """Antibody #5 multi-input: all 520 flag=0 FNAF 1 records decode
    cleanly; their output is always exactly `w * h * 4` bytes."""
    bank = _fnaf1_image_bank()
    flag0 = [img for img in bank.images if img.flags == 0]
    assert len(flag0) == 520, (
        f"expected 520 flag=0 records in FNAF 1; saw {len(flag0)} "
        f"— envelope drift or pack mismatch"
    )
    for img in flag0:
        dp = decode_image_pixels(img)
        expected_len = img.width * img.height * 4
        assert len(dp.rgba) == expected_len, (
            f"handle={img.handle} w={img.width} h={img.height}: rgba "
            f"len={len(dp.rgba)} expected {expected_len}"
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_flag16_records_all_raise_as_deferred_work():
    """Deferred-work pin: every flag=0x10 record surfaces as
    `ImagePixelsDecodeError` until probe #7.1b ships. Pin the count so
    any future silent reclassification into flag=0 fires loudly."""
    bank = _fnaf1_image_bank()
    flag16 = [img for img in bank.images if img.flags == 0x10]
    assert len(flag16) == 85, (
        f"expected 85 flag=0x10 records in FNAF 1; saw {len(flag16)}"
    )
    raised = 0
    for img in flag16:
        try:
            decode_image_pixels(img)
        except ImagePixelsDecodeError:
            raised += 1
    assert raised == 85, (
        f"expected every flag=0x10 record to raise in probe #7.1a; "
        f"{85 - raised} fell through the dispatcher"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_flag0_decoded_pixels_snapshot():
    """Antibody #7 snapshot: pin total decoded bytes and a SHA-256
    fingerprint over `(handle, sha256(rgba))` pairs across all 520
    flag=0 records.

    Captured empirically on 2026-04-21 (probe #7.1a). Any single-pixel
    drift in any flag=0 record — whether from a byte-order swap, a
    row-pad miscalculation, or 0x6666 envelope regression — flips the
    combined hash. The error message names nothing, but a failure here
    points straight at `decode_flag0_bgr_masked` or its inputs.
    """
    bank = _fnaf1_image_bank()
    flag0 = [img for img in bank.images if img.flags == 0]
    assert len(flag0) == 520

    total_bytes = 0
    h = hashlib.sha256()
    per_record = []
    for img in flag0:
        dp = decode_image_pixels(img)
        total_bytes += len(dp.rgba)
        rec_sha = hashlib.sha256(dp.rgba).digest()
        h.update(img.handle.to_bytes(4, "little", signed=True))
        h.update(rec_sha)
        per_record.append((img.handle, rec_sha.hex()))

    fingerprint = h.hexdigest()

    # Debug print — surfaces the empirical values on first run so the
    # snapshot expectations below can be bootstrapped without a separate
    # inspection pass.
    print(
        f"\n[images_pixels snapshot] flag0_count={len(flag0)} "
        f"total_bytes={total_bytes} fingerprint={fingerprint}\n"
        f"first_3={per_record[:3]}\n"
        f"last_3={per_record[-3:]}"
    )

    # Pinned empirical values — first captured 2026-04-21 (probe #7.1a).
    assert total_bytes == sum(
        img.width * img.height * 4 for img in flag0
    ), "total_bytes must equal sum of w*h*4 across flag=0 records"

    # Pinned empirical values captured 2026-04-21 (probe #7.1a
    # bootstrap). On bootstrap the test's debug print surfaces both
    # values; a drift here flips the fingerprint loudly while the
    # total_bytes sanity check above catches dimension-level drift
    # separately.
    assert total_bytes == 982_084_248
    assert fingerprint == (
        "c7998dc6fee847904346be8171dcbc7f5e80d725a836ddb7fbb7a14667c5a88f"
    )
