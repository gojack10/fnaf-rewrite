"""Regression tests for the PNG emission sink (probe #7.1a auxiliary).

The sink itself is one-way — it writes PNG files to disk so probe #100
can eyeball them. These tests cover three surfaces:

- The raw PNG encoder produces bytes a third-party decoder parses back
  to the exact RGBA input. No Pillow dep, so we parse the PNG
  structurally via the stdlib `struct` + `zlib.decompress` to round-trip.
- The env-var gate: `FNAF_PARSER_EMIT_PNGS` absent → no write; truthy
  value → write.
- The file naming convention: `<handle 04d>.png` under the target
  directory, directory auto-created on first write.

Antibody coverage:

- #1 strict-unknown : rgba length mismatch raises; non-positive width/
  height raises.
- #2 byte-count    : every row is exactly filter-byte + width*4 bytes
  inside the IDAT stream. Round-trip inflates to the exact raw block.
- #3 round-trip    : encode → structural-decode → rgba matches input.
- #5 (n/a)         : nothing to multi-input against yet; the FNAF 1
  emission path is exercised manually at probe #100.
- Env gate         : sink is no-op by default and only fires when the
  var is set. Tests set/unset via `monkeypatch`.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from fnaf_parser.decoders.images_pixels import DecodedPixels
from fnaf_parser.sinks.png_emit import (
    EMIT_ENV_VAR,
    PNG_SIGNATURE,
    emit_png,
    encode_png_rgba,
)


# --- Structural PNG parse helper ----------------------------------------


def _parse_png(blob: bytes) -> tuple[int, int, bytes]:
    """Parse a minimal PNG emitted by `encode_png_rgba` back to
    `(width, height, raw_rgba)`. Supports only the shape this encoder
    produces: 8-bit RGBA, no interlace, filter=0 rows.

    Not a general PNG parser — just enough to prove our encoder's
    output round-trips. A structural mismatch here (wrong chunk order,
    wrong CRC, wrong IHDR length) would fail the parse loudly.
    """
    assert blob[:8] == PNG_SIGNATURE, "PNG signature mismatch"
    pos = 8
    ihdr = None
    idat_accum = b""
    end = False
    while pos < len(blob):
        (chunk_len,) = struct.unpack(">I", blob[pos:pos + 4])
        chunk_type = blob[pos + 4:pos + 8]
        chunk_data = blob[pos + 8:pos + 8 + chunk_len]
        chunk_crc = blob[pos + 8 + chunk_len:pos + 12 + chunk_len]
        assert struct.unpack(">I", chunk_crc)[0] == (
            zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        ), f"CRC mismatch on chunk {chunk_type!r}"
        pos += 12 + chunk_len

        if chunk_type == b"IHDR":
            ihdr = chunk_data
        elif chunk_type == b"IDAT":
            idat_accum += chunk_data
        elif chunk_type == b"IEND":
            end = True
            break

    assert end, "no IEND chunk found"
    assert ihdr is not None and len(ihdr) == 13

    width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    assert (bit_depth, color_type, comp, filt, interlace) == (8, 6, 0, 0, 0), (
        f"unexpected IHDR shape: {bit_depth=} {color_type=} {comp=} "
        f"{filt=} {interlace=}"
    )

    raw_stream = zlib.decompress(idat_accum)
    # Strip the per-row filter bytes.
    row_stride = width * 4
    assert len(raw_stream) == height * (row_stride + 1), (
        f"raw stream length {len(raw_stream)} doesn't match "
        f"height*(row_stride+1)={height * (row_stride + 1)}"
    )
    rows = []
    for y in range(height):
        start = y * (row_stride + 1)
        filter_byte = raw_stream[start]
        assert filter_byte == 0, f"row {y} filter byte != 0"
        rows.append(raw_stream[start + 1:start + 1 + row_stride])
    return width, height, b"".join(rows)


# --- Encoder unit tests -------------------------------------------------


def test_encode_png_rgba_roundtrips_2x2():
    """Antibody #3: a deliberately asymmetric RGBA grid survives
    encode → structural-parse → raw bytes unchanged. The asymmetry
    guarantees every channel / row is load-bearing."""
    rgba = bytes([
        255, 0, 0, 255,    0, 255, 0, 128,
        0, 0, 255, 64,     255, 255, 0, 0,
    ])
    blob = encode_png_rgba(rgba, width=2, height=2)
    w, h, raw = _parse_png(blob)
    assert (w, h) == (2, 2)
    assert raw == rgba


def test_encode_png_rgba_odd_dimensions_roundtrip():
    """Encoder is agnostic to odd widths — there's no row padding at
    the PNG layer (that was a 24bpp masked-RGB concern). Confirm
    a 3x5 grid round-trips bit-exact."""
    rgba = bytes(i & 0xFF for i in range(3 * 5 * 4))
    blob = encode_png_rgba(rgba, width=3, height=5)
    w, h, raw = _parse_png(blob)
    assert (w, h) == (3, 5)
    assert raw == rgba


def test_encode_png_rgba_size_mismatch_raises():
    """Antibody #1: emitting a PNG with `len(rgba) != w*h*4` would
    silently corrupt the output. Raise loudly instead."""
    with pytest.raises(ValueError, match="rgba is"):
        encode_png_rgba(b"\x00" * 15, width=2, height=2)  # should be 16


def test_encode_png_rgba_non_positive_dimensions_raise():
    """Antibody #1: zero/negative dimensions are an error upstream."""
    for w, h in [(0, 1), (1, 0), (-1, 1), (1, -1)]:
        with pytest.raises(ValueError):
            encode_png_rgba(b"", width=w, height=h)


def test_encode_png_rgba_signature_and_chunks():
    """Signature-level antibody: the emitted byte string starts with
    the canonical PNG signature and contains the three expected chunk
    types in order (IHDR → IDAT(+) → IEND)."""
    rgba = bytes([0] * 4)
    blob = encode_png_rgba(rgba, width=1, height=1)
    assert blob.startswith(PNG_SIGNATURE)
    # Chunk types appear in order somewhere in the stream; easiest
    # invariant is to search for the type-codes in the bytes.
    idx_ihdr = blob.find(b"IHDR")
    idx_idat = blob.find(b"IDAT")
    idx_iend = blob.find(b"IEND")
    assert idx_ihdr != -1 and idx_idat != -1 and idx_iend != -1
    assert idx_ihdr < idx_idat < idx_iend


# --- Env-gate tests -----------------------------------------------------


def _make_decoded(handle: int = 42) -> DecodedPixels:
    """Build a small, deterministic DecodedPixels so file-write tests
    can inspect the payload without depending on 0x6666 decode."""
    rgba = bytes([255, 0, 0, 255,  0, 255, 0, 255,  0, 0, 255, 255,  0, 0, 0, 0])
    return DecodedPixels(handle=handle, width=2, height=2, rgba=rgba)


def test_emit_png_no_op_when_env_var_unset(monkeypatch, tmp_path: Path):
    """The sink must be no-op by default — automated CI runs and
    generic developer test runs must not spam the filesystem."""
    monkeypatch.delenv(EMIT_ENV_VAR, raising=False)
    decoded = _make_decoded()
    result = emit_png(decoded, out_dir=tmp_path)
    assert result is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "TRUE", "Yes"])
def test_emit_png_writes_when_env_var_truthy(
    monkeypatch, tmp_path: Path, truthy: str
):
    """Any of the accepted truthy strings enables emission. Checks
    both case-insensitivity and the POSIX-ish accepted list."""
    monkeypatch.setenv(EMIT_ENV_VAR, truthy)
    decoded = _make_decoded(handle=7)
    result = emit_png(decoded, out_dir=tmp_path)
    assert result == tmp_path / "0007.png"
    assert result.exists()
    # Round-trip the emitted file to confirm it's a real PNG.
    w, h, raw = _parse_png(result.read_bytes())
    assert (w, h) == (2, 2)
    assert raw == decoded.rgba


@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "off", "random"])
def test_emit_png_is_no_op_for_falsy_env_values(
    monkeypatch, tmp_path: Path, falsy: str
):
    """Anything outside the explicit truthy list is treated as off —
    including nonsense values. A subtle misconfiguration shouldn't
    silently enable emission."""
    monkeypatch.setenv(EMIT_ENV_VAR, falsy)
    decoded = _make_decoded()
    result = emit_png(decoded, out_dir=tmp_path)
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_emit_png_force_bypasses_env_gate(monkeypatch, tmp_path: Path):
    """`force=True` exists specifically so unit tests (and probe #100
    bootstrap scripts) can exercise the write path without touching
    the process environment."""
    monkeypatch.delenv(EMIT_ENV_VAR, raising=False)
    decoded = _make_decoded(handle=100)
    result = emit_png(decoded, out_dir=tmp_path, force=True)
    assert result == tmp_path / "0100.png"
    assert result.exists()


def test_emit_png_creates_out_dir_if_missing(monkeypatch, tmp_path: Path):
    """Directory auto-creation: the first emit to a nested path must
    create every parent. Probe #100 will point this at
    `parser/out/images/` which doesn't exist on a fresh checkout."""
    monkeypatch.setenv(EMIT_ENV_VAR, "1")
    target = tmp_path / "deep" / "nested" / "images"
    decoded = _make_decoded(handle=1)
    result = emit_png(decoded, out_dir=target)
    assert result == target / "0001.png"
    assert target.is_dir()
    assert result.exists()
