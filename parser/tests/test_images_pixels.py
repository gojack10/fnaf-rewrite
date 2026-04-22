"""Regression tests for 0x6666 pixel-payload decoder (probes #7.1a + #7.1b).

Scope: both `graphic_mode == 4` paths in FNAF 1 — `flags == 0` (520
records, probe #7.1a) and `flags == 0x10` (85 records, probe #7.1b).
Every other mode/flag combination is expected to raise.

Antibody coverage:

- #1 strict-unknown : mode != 4, flag ∉ {0, 0x10}, truncated image_data
  on either flag path, non-positive width/height — all raise
  `ImagePixelsDecodeError`.
- #2 byte-count    : output is always exactly `w * h * 4` bytes.
  Flag=0 row pad is 0/3 bytes; flag=16 colour pad is 0/3 bytes AND the
  alpha plane adds an independent 4-byte-aligned row pad that
  exercises every width-mod-4 class.
- #3 round-trip    : synthetic 2×2 and 3×2 streams decode for both
  flag paths to exactly the expected RGBA sequences, including
  odd-width colour pad AND the distinct 4-byte alpha-plane pad.
- #5 multi-input   : all 520 flag=0 records AND all 85 flag=16
  records in FNAF 1 decode cleanly, every output is `w * h * 4` bytes.
- #7 snapshot      : per-flag SHA-256 fingerprint pins the entire
  decoded bank; any single-pixel drift on any record flips the hash.
- Alpha-distribution: flag=16 alpha channel across the whole bank is
  non-degenerate (at least two distinct values, AND at least one
  value that isn't 0 or 255) — catches an "alpha plane never read"
  regression that round-trip hashes would also catch but less legibly.
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
    decode_flag16_bgr_with_alpha_plane,
    decode_image_pixels,
    get_alpha_row_padding_bytes,
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


# --- Alpha-plane row padding (flag=0x10) --------------------------------


def test_alpha_row_padding_matches_four_byte_alignment():
    """Antibody #2: `(4 - width%4) % 4`. Spot-check every residue class
    plus a few real-world widths from FNAF 1 sprite sheets."""
    expected = {
        1: 3, 2: 2, 3: 1, 4: 0,
        5: 3, 6: 2, 7: 1, 8: 0,
        100: 0,   # 100 % 4 == 0
        101: 3,   # 101 % 4 == 1
        102: 2,
        103: 1,
        320: 0, 321: 3,
        1279: 1, 1280: 0,
    }
    for w, expected_pad in expected.items():
        assert get_alpha_row_padding_bytes(w) == expected_pad, (
            f"width={w}: expected alpha pad {expected_pad}, "
            f"got {get_alpha_row_padding_bytes(w)}"
        )


def test_alpha_row_padding_rejects_non_positive():
    """Same strict-unknown handling as the colour-plane helper."""
    for w in (0, -1, -7):
        with pytest.raises(ImagePixelsDecodeError):
            get_alpha_row_padding_bytes(w)


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


# --- Synthetic flag=16 decode (alpha plane) -----------------------------


def test_decode_flag16_2x2_applies_alpha_plane_not_transparent_keying():
    """Antibody #3 round-trip: a 2×2 flag=16 stream applies the alpha
    plane to the A channel of every pixel and *never* consults a
    transparent colour. Width 2 → colour pad 0 (even width) but
    alpha pad 2 (2 % 4 = 2 → pad = 2), so every alpha row is
    `[a0, a1, pad, pad]`. This test specifically exercises that the
    alpha pad is NOT zero for even widths — a common miscount
    collapses it to the colour pad."""
    # Colour plane: every pixel (B=1, G=2, R=3). Width even → no colour pad.
    colour = bytes([1, 2, 3,  1, 2, 3,  1, 2, 3,  1, 2, 3])
    # Alpha plane: rows of [a0, a1, pad, pad]. Sentinel 0xAA in pad so
    # any pad miscount bleeds into visible alpha and the assert fires.
    alpha = bytes([0, 64, 0xAA, 0xAA,   128, 255, 0xAA, 0xAA])
    rgba = decode_flag16_bgr_with_alpha_plane(
        colour + alpha, width=2, height=2
    )
    # All pixels share RGB=(3, 2, 1); alpha comes strictly from the
    # non-pad positions of the alpha plane.
    assert list(rgba) == [
        3, 2, 1, 0,    3, 2, 1, 64,
        3, 2, 1, 128,  3, 2, 1, 255,
    ]
    # Explicit sanity: 0xAA is sentinel for "alpha pad bled through".
    assert 0xAA not in rgba


def test_decode_flag16_3x2_consumes_both_row_paddings():
    """Antibody #2 byte-count: a 3×2 flag=16 stream has *two* kinds of
    row pad — 3 bytes after each colour row (odd width @ 24bpp) and 1
    byte after each alpha row (3 % 4 = 3 → pad = 1). If either pad is
    miscounted the decoder reads alpha bytes from inside the colour
    pad (or vice versa) and the output diverges.

    Sentinel bytes 0xCC in both pad slots would bleed into output if
    the row-advance logic is off by one."""
    # Colour plane rows: 3 pixels × 3 bytes + 3 bytes pad each.
    c_row0 = bytes([1, 2, 3,  4, 5, 6,  7, 8, 9,  0xCC, 0xCC, 0xCC])
    c_row1 = bytes([
        10, 20, 30,  40, 50, 60,  70, 80, 90,  0xCC, 0xCC, 0xCC,
    ])
    # Alpha plane rows: 3 alpha bytes + 1 byte pad each.
    a_row0 = bytes([11, 22, 33,  0xCC])
    a_row1 = bytes([44, 55, 66,  0xCC])
    src = c_row0 + c_row1 + a_row0 + a_row1
    rgba = decode_flag16_bgr_with_alpha_plane(src, width=3, height=2)
    assert len(rgba) == 3 * 2 * 4
    # Output must never contain 0xCC — if any pad is unread, the
    # sentinel bleeds into a pixel channel.
    assert 0xCC not in rgba
    assert list(rgba[:12]) == [3, 2, 1, 11,  6, 5, 4, 22,  9, 8, 7, 33]
    assert list(rgba[12:24]) == [
        30, 20, 10, 44,  60, 50, 40, 55,  90, 80, 70, 66,
    ]


def test_decode_flag16_ignores_transparent_colour_semantics():
    """Explicit antibody: flag=16 decoder has no `transparent` param
    and never consults one. A pixel whose BGR would match a
    "would-be transparent" sentinel gets its alpha from the alpha
    plane only — not forced to 0. Closes the inherited §10.3 open
    question from Probe #7.1a."""
    # Pixel at (0,0) has RGB=(3,2,1) — exactly what flag=0's
    # transparent test used to key off.
    colour = bytes([1, 2, 3,  10, 20, 30])
    # Alpha row = [a0, a1, pad, pad] (width=2 → pad=2).
    alpha = bytes([200, 100, 0xAA, 0xAA])
    rgba = decode_flag16_bgr_with_alpha_plane(colour + alpha, width=2, height=1)
    # Alpha plane wins: first pixel A=200, NOT 0.
    assert rgba[3] == 200
    assert rgba[7] == 100


def test_decode_flag16_tolerates_trailing_bytes():
    """Trailing-byte tolerance matches flag=0: some records have a few
    leftover bytes past the last alpha row pad; both reference readers
    silently discard them."""
    colour = bytes([1, 2, 3,  4, 5, 6])
    # width=2 → alpha pad=2 → row is [a0, a1, pad, pad].
    alpha = bytes([128, 64, 0xAA, 0xAA])
    src = colour + alpha + b"\xCC\xCC\xCC\xCC"
    rgba = decode_flag16_bgr_with_alpha_plane(src, width=2, height=1)
    assert list(rgba) == [3, 2, 1, 128,  6, 5, 4, 64]


def test_decode_flag16_truncated_colour_plane_raises():
    """Antibody #2: if the stream is shorter than `colour_bytes`,
    the unified byte-count check fires at the top of the function."""
    # width=2 height=2 needs 12 colour + 8 alpha = 20 bytes. Give 15.
    src = bytes([1] * 15)
    with pytest.raises(ImagePixelsDecodeError, match="truncated"):
        decode_flag16_bgr_with_alpha_plane(src, width=2, height=2)


def test_decode_flag16_truncated_alpha_plane_raises():
    """Antibody #2: a stream long enough for the colour plane but
    short in the alpha plane still raises — the byte-count check is
    over the *combined* requirement. A subtle alpha-row-pad miscount
    would otherwise quietly read OOB past image_data, masked by
    Python's bounds checks into a cryptic IndexError."""
    # width=2 height=2: need 20 bytes, give 19 (colour complete,
    # alpha missing 1 byte).
    src = bytes([1] * 19)
    with pytest.raises(ImagePixelsDecodeError, match="truncated"):
        decode_flag16_bgr_with_alpha_plane(src, width=2, height=2)


def test_decode_flag16_non_positive_dimensions_raise():
    """Antibody #1: zero or negative dims raise before any byte
    indexing — same defensive posture as flag=0."""
    for w, h in [(0, 2), (2, 0), (-1, 2), (2, -1)]:
        with pytest.raises(ImagePixelsDecodeError):
            decode_flag16_bgr_with_alpha_plane(b"", width=w, height=h)


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


def test_dispatcher_routes_flag16_to_alpha_decoder():
    """Happy path: flag=0x10 routes to the alpha-plane decoder and
    the result carries per-pixel alpha from the trailing plane. This
    is the test that proves probe #7.1b has flipped from "raises" to
    "decodes" end-to-end."""
    # 2×2 stream: 12 colour bytes + 2*(2+2)=8 alpha bytes = 20 total.
    colour = bytes([1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12])
    # Alpha rows [a0, a1, pad, pad] × 2 rows.
    alpha = bytes([10, 90, 0xAA, 0xAA,   200, 255, 0xAA, 0xAA])
    img = _synth_image(
        handle=77,
        width=2,
        height=2,
        flags=0x10,
        # Include a transparent colour that would key a pixel under
        # flag=0: we're proving it's *ignored* on the flag=16 path.
        transparent=(3, 2, 1, 0),
        image_data=colour + alpha,
    )
    dp = decode_image_pixels(img)
    assert isinstance(dp, DecodedPixels)
    assert dp.handle == 77
    # First pixel: BGR=(1,2,3) → RGB=(3,2,1). Alpha from plane, NOT
    # zeroed by the would-be transparent match.
    assert list(dp.rgba[:4]) == [3, 2, 1, 10]
    assert list(dp.rgba[4:8]) == [6, 5, 4, 90]
    assert list(dp.rgba[8:12]) == [9, 8, 7, 200]
    assert list(dp.rgba[12:16]) == [12, 11, 10, 255]


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
def test_fnaf1_all_flag16_records_decode_without_error():
    """Antibody #5 multi-input: all 85 flag=0x10 FNAF 1 records decode
    cleanly; their output is always `w * h * 4` bytes. Any truncation
    or row-pad miscalculation (most likely the 4-byte alpha pad) would
    surface here before a snapshot even runs."""
    bank = _fnaf1_image_bank()
    flag16 = [img for img in bank.images if img.flags == 0x10]
    assert len(flag16) == 85, (
        f"expected 85 flag=0x10 records in FNAF 1; saw {len(flag16)} "
        f"— envelope drift or pack mismatch"
    )
    for img in flag16:
        dp = decode_image_pixels(img)
        expected_len = img.width * img.height * 4
        assert len(dp.rgba) == expected_len, (
            f"handle={img.handle} w={img.width} h={img.height}: rgba "
            f"len={len(dp.rgba)} expected {expected_len}"
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


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_flag16_decoded_pixels_snapshot():
    """Antibody #7 snapshot: pin total decoded bytes and a SHA-256
    fingerprint over `(handle, sha256(rgba))` pairs across all 85
    flag=0x10 records.

    First captured empirically on 2026-04-21 (probe #7.1b). Any
    single-pixel drift — colour-plane byte-order swap, alpha-plane
    row-pad miscount, alpha-plane byte order (not expected but
    possible), or a silent reclassification of any flag=0 record into
    this bucket — flips the combined hash."""
    bank = _fnaf1_image_bank()
    flag16 = [img for img in bank.images if img.flags == 0x10]
    assert len(flag16) == 85

    total_bytes = 0
    h = hashlib.sha256()
    per_record = []
    for img in flag16:
        dp = decode_image_pixels(img)
        total_bytes += len(dp.rgba)
        rec_sha = hashlib.sha256(dp.rgba).digest()
        h.update(img.handle.to_bytes(4, "little", signed=True))
        h.update(rec_sha)
        per_record.append((img.handle, rec_sha.hex()))

    fingerprint = h.hexdigest()

    # Bootstrap debug print: surfaces empirical values on first run so
    # the pinned expectations below can be filled in without a separate
    # inspection pass.
    print(
        f"\n[images_pixels flag16 snapshot] flag16_count={len(flag16)} "
        f"total_bytes={total_bytes} fingerprint={fingerprint}\n"
        f"first_3={per_record[:3]}\n"
        f"last_3={per_record[-3:]}"
    )

    assert total_bytes == sum(
        img.width * img.height * 4 for img in flag16
    ), "total_bytes must equal sum of w*h*4 across flag=0x10 records"

    # Pinned empirical values — populate on first successful run from
    # the debug print above. Sentinel assertions below intentionally
    # fail at `None` so an unprepared test environment doesn't silently
    # record the "wrong" snapshot.
    assert total_bytes == _FLAG16_SNAPSHOT_TOTAL_BYTES
    assert fingerprint == _FLAG16_SNAPSHOT_FINGERPRINT


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF1 binary not available")
def test_fnaf1_flag16_alpha_channel_is_non_degenerate():
    """Alpha-distribution antibody: across all 85 flag=16 records, the
    output alpha channel must contain **at least three distinct values**
    AND at least one value that is neither 0 nor 255.

    This catches a regression the snapshot would also catch, but more
    legibly: if the alpha-plane read is off-by-one or reading from
    colour-pad bytes, the alpha channel often collapses to constant 0
    or constant 255 across entire records. The bank-wide histogram
    gives a loud, labelled error — "alpha channel degenerate" points
    straight at the alpha plane, while a fingerprint flip doesn't."""
    bank = _fnaf1_image_bank()
    flag16 = [img for img in bank.images if img.flags == 0x10]
    assert len(flag16) == 85

    alpha_values: set[int] = set()
    for img in flag16:
        dp = decode_image_pixels(img)
        # Every 4th byte from offset 3 is an alpha byte.
        alpha_values.update(dp.rgba[3::4])
        # Early exit: once we have 3+ distinct values AND at least one
        # mid-range value, further records can't disprove the claim.
        if len(alpha_values) >= 3 and any(
            0 < v < 255 for v in alpha_values
        ):
            break

    assert len(alpha_values) >= 3, (
        f"flag=16 alpha channel is degenerate: only "
        f"{len(alpha_values)} distinct value(s) across 85 records. "
        f"Alpha plane likely not being read — colour pad or alpha pad "
        f"miscount is the first place to look."
    )
    assert any(0 < v < 255 for v in alpha_values), (
        f"flag=16 alpha channel only ever takes values in {{0, 255}} "
        f"across all 85 records — no anti-aliased edges made it through. "
        f"Suggests the alpha plane is being read as a 1-bit mask."
    )


# --- Flag=16 snapshot pins ---------------------------------------------

# Pinned empirical values — first captured 2026-04-21 (probe #7.1b
# bootstrap). Total bytes is separately cross-checked against
# `sum(w*h*4)` inside the test to separate dimension drift from per-byte
# drift.
_FLAG16_SNAPSHOT_TOTAL_BYTES = 108_519_636
_FLAG16_SNAPSHOT_FINGERPRINT = (
    "dd2955d1d6f01eeb64b102eea06860985594cb31508ed648c522f8fde762b05e"
)
