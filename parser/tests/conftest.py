"""Shared session-scope fixtures for FNAF 1 integration tests.

Before this file existed, every `test_sounds.py` / `test_images.py` /
`test_images_pixels.py` test did its own `read_bytes` on the 220 MB
binary, re-ran `walk_chunks`, re-derived the RC4 transform, and re-
decoded the 0x6666 / 0x6668 envelopes. With ~20 tests each paying that
cost, the default run was ~500 s — dominated by redundant decoding.

These fixtures amortise the work across the whole test session, so:

- the 220 MB binary is read once
- `walk_chunks` runs once
- the RC4 transform is derived once
- each top-level bank (0x6666 Images, 0x6668 Sounds) is decoded once

Tests opt in by declaring the fixture name as a parameter. Fixtures
auto-skip when the binary isn't on disk so unit-only runs (CI without
the proprietary binary) still pass.

The `slow` marker registered here gates `test_cli.py`'s full asset-
dump smoke test out of the default run — it takes ~3 minutes and is
redundant with the dedicated decoder / sink tests. Run the full suite
with `uv run pytest -m "not slow or slow"` or just `pytest -m slow` for
the smoke test alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.images import Image, ImageBank, decode_image_bank
from fnaf_parser.decoders.images_pixels import DecodedPixels, decode_image_pixels
from fnaf_parser.decoders.sounds import SoundBank, decode_sound_bank
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so `--strict-markers` (and future linting)
    doesn't flag our in-tree markers as unknown."""
    config.addinivalue_line(
        "markers",
        "slow: end-to-end smoke tests (full asset dump, ~3 min). "
        "Excluded from the default run via addopts in pyproject.toml.",
    )


@pytest.fixture(scope="session")
def fnaf1_exe_path() -> Path:
    """Absolute path to the FNAF 1 binary. Skips if missing so unit-only
    environments can still run the non-integration suite."""
    if not FNAF_EXE.exists():
        pytest.skip("FNAF 1 binary not on disk")
    return FNAF_EXE


@pytest.fixture(scope="session")
def fnaf1_exe_bytes(fnaf1_exe_path: Path) -> bytes:
    """The FNAF 1 binary, read exactly once per session."""
    return fnaf1_exe_path.read_bytes()


@pytest.fixture(scope="session")
def fnaf1_walk_result(fnaf1_exe_path: Path):
    """Cached `walk_chunks` result for the FNAF 1 pack."""
    return walk_chunks(fnaf1_exe_path, pack_start=FNAF1_DATA_PACK_START)


@pytest.fixture(scope="session")
def fnaf1_transform(fnaf1_exe_bytes: bytes, fnaf1_walk_result):
    """The RC4 transform derived from the FNAF 1 header strings.

    Mirrors the per-file helpers that existed in test_images.py /
    test_sounds.py / test_images_pixels.py, hoisted so the 0x222E /
    0x2224 / 0x223B string decodes happen exactly once.
    """

    def _str_of(chunk_id: int) -> str:
        rec = next(r for r in fnaf1_walk_result.records if r.id == chunk_id)
        return decode_string_chunk(
            read_chunk_payload(fnaf1_exe_bytes, rec),
            unicode=fnaf1_walk_result.header.unicode,
        )

    editor = _str_of(0x222E)
    name = _str_of(0x2224)
    copyright_records = [r for r in fnaf1_walk_result.records if r.id == 0x223B]
    copyright_str = (
        decode_string_chunk(
            read_chunk_payload(fnaf1_exe_bytes, copyright_records[0]),
            unicode=fnaf1_walk_result.header.unicode,
        )
        if copyright_records
        else ""
    )
    return make_transform(
        editor=editor,
        name=name,
        copyright_str=copyright_str,
        build=fnaf1_walk_result.header.product_build,
        unicode=fnaf1_walk_result.header.unicode,
    )


@pytest.fixture(scope="session")
def fnaf1_image_bank(
    fnaf1_exe_bytes: bytes, fnaf1_walk_result, fnaf1_transform
) -> ImageBank:
    """Decoded 0x6666 ImageBank for FNAF 1. Session-scoped so the two
    flag0 pixel-decoding suites (~110 s apiece) share one decode."""
    rec = next(r for r in fnaf1_walk_result.records if r.id == 0x6666)
    payload = read_chunk_payload(fnaf1_exe_bytes, rec, transform=fnaf1_transform)
    return decode_image_bank(payload)


@pytest.fixture(scope="session")
def fnaf1_sound_bank(
    fnaf1_exe_bytes: bytes, fnaf1_walk_result, fnaf1_transform
) -> SoundBank:
    """Decoded 0x6668 SoundBank for FNAF 1. Session-scoped so
    test_sounds.py's 11+ integration tests share one decode."""
    rec = next(r for r in fnaf1_walk_result.records if r.id == 0x6668)
    payload = read_chunk_payload(fnaf1_exe_bytes, rec, transform=fnaf1_transform)
    return decode_sound_bank(payload)


@pytest.fixture(scope="session")
def fnaf1_flag0_decoded_pixels(
    fnaf1_image_bank: ImageBank,
) -> tuple[tuple[Image, DecodedPixels], ...]:
    """Every (img, DecodedPixels) pair for FNAF 1's 520 `flags == 0`
    records — the most expensive slice in the whole suite (~100 s of
    per-record bgr_masked decoding).

    Before this fixture, two independent tests
    (`test_fnaf1_all_flag0_records_decode_without_error` and
    `test_fnaf1_flag0_decoded_pixels_snapshot`) each paid the full cost
    — ~200 s total. Folding the work into a session fixture keeps both
    assertions (no-raise on every record + SHA-256 snapshot) while only
    running the decoder once.
    """
    flag0 = [img for img in fnaf1_image_bank.images if img.flags == 0]
    return tuple((img, decode_image_pixels(img)) for img in flag0)


@pytest.fixture(scope="session")
def fnaf1_flag16_decoded_pixels(
    fnaf1_image_bank: ImageBank,
) -> tuple[tuple[Image, DecodedPixels], ...]:
    """Every (img, DecodedPixels) pair for FNAF 1's 85 `flags == 0x10`
    records. Three tests share the decode (no-raise, snapshot, alpha-
    distribution) — session-scope avoids doing it three times (~30 s
    → ~10 s)."""
    flag16 = [img for img in fnaf1_image_bank.images if img.flags == 0x10]
    return tuple((img, decode_image_pixels(img)) for img in flag16)
