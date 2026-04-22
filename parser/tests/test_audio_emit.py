"""Regression tests for the WAV emission sink (probe #9.1 auxiliary).

The sink is a one-way passthrough — it writes `Sound.audio_data` to disk
so probe #100 can `aplay out/audio/*.wav` and confirm audibility against
gameplay. Unlike `png_emit` there's no encoder to round-trip; the tests
cover the env-gate, the file-naming contract, and the write invariant
(bytes on disk equal `sound.audio_data` exactly).

Antibody coverage:

- #1 strict-unknown : `_make_sound` helper takes the minimum fields
  actually consumed by the sink (`handle`, `audio_data`); a drift in
  which fields the sink reaches for would surface as an AttributeError.
- #2 byte-count    : the on-disk file size must equal
  `len(sound.audio_data)` exactly — no padding, no truncation.
- #3 round-trip    : write → read back → compare bytes.
- Env gate         : default off; truthy-list enables; falsy-list
  disables; unknown values treated as off.
- Force flag       : `force=True` bypasses env without touching process
  env — the only way CI tests can exercise the write path safely.
- Directory create : emit to a nested path must auto-create parents.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fnaf_parser.decoders.sounds import SOUND_FLAG_WAVE, Sound
from fnaf_parser.sinks.audio_emit import EMIT_ENV_VAR, emit_wav


# --- Helpers ------------------------------------------------------------


def _make_sound(
    *,
    raw_handle: int = 5,
    audio_data: bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
                        b"\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00"
                        b"\x02\x00\x10\x00data\x00\x00\x00\x00",
) -> Sound:
    """Construct a minimal-but-structurally-valid `Sound` for sink tests.

    The `audio_data` default is a real (if silent) 44.1kHz mono 16-bit
    PCM RIFF/WAVE — 44 bytes of header + zero-byte data chunk. Using a
    real-shape WAV keeps the tests representative of the FNAF 1
    passthrough path: the sink writes exactly what decoders/sounds.py
    produces, no more, no less.

    We keep the frozen dataclass contract honest by filling every field;
    passing dummies for the audio-agnostic fields (checksum, references,
    etc.) keeps the sink's contract "reads handle + audio_data" explicit.
    """
    return Sound(
        raw_handle=raw_handle,
        handle=raw_handle - 1,
        record_start_offset=0,
        record_wire_size=len(audio_data) + 28,
        checksum=0,
        references=0,
        decompressed_size=len(audio_data),
        flags=SOUND_FLAG_WAVE,
        reserved=0,
        name_length=0,
        name="test.wav",
        is_compressed=True,
        compressed_size=0,
        audio_data=audio_data,
    )


# --- Env-gate behaviour -------------------------------------------------


def test_emit_wav_no_op_when_env_var_unset(monkeypatch, tmp_path: Path):
    """Default-off: the sink must not write when the env var is absent.
    Automated CI runs and `uv run pytest` should never spam the
    filesystem — that's the whole reason for the env gate."""
    monkeypatch.delenv(EMIT_ENV_VAR, raising=False)
    sound = _make_sound()
    result = emit_wav(sound, out_dir=tmp_path)
    assert result is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "TRUE", "Yes"])
def test_emit_wav_writes_when_env_var_truthy(
    monkeypatch, tmp_path: Path, truthy: str
):
    """Each accepted truthy value enables emission. Covers case-
    insensitivity and the full POSIX-ish truthy set (1/true/yes/on)."""
    monkeypatch.setenv(EMIT_ENV_VAR, truthy)
    sound = _make_sound(raw_handle=8)
    result = emit_wav(sound, out_dir=tmp_path)
    assert result == tmp_path / "0007.wav"  # handle = raw_handle - 1 = 7
    assert result.exists()
    # The on-disk file must be byte-identical to sound.audio_data —
    # passthrough, no re-encoding, no trailing NULs.
    assert result.read_bytes() == sound.audio_data


@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "off", "random"])
def test_emit_wav_is_no_op_for_falsy_env_values(
    monkeypatch, tmp_path: Path, falsy: str
):
    """Only the explicit truthy list enables emission. `random` and
    similar unknown values must be treated as off — a subtle misconfig
    shouldn't silently write 97 MiB of audio to disk."""
    monkeypatch.setenv(EMIT_ENV_VAR, falsy)
    sound = _make_sound()
    result = emit_wav(sound, out_dir=tmp_path)
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_emit_wav_force_bypasses_env_gate(monkeypatch, tmp_path: Path):
    """`force=True` exercises the write path without touching process
    env — required by unit tests and by probe #100 bootstrap scripts
    that want to dump all 52 sounds without shell-level env plumbing."""
    monkeypatch.delenv(EMIT_ENV_VAR, raising=False)
    sound = _make_sound(raw_handle=101)
    result = emit_wav(sound, out_dir=tmp_path, force=True)
    assert result == tmp_path / "0100.wav"
    assert result.exists()
    assert result.read_bytes() == sound.audio_data


def test_emit_wav_creates_out_dir_if_missing(monkeypatch, tmp_path: Path):
    """Directory auto-creation: emitting to a nested path must create
    every parent. Probe #100 will point this at `parser/out/audio/`
    which doesn't exist on a fresh checkout."""
    monkeypatch.setenv(EMIT_ENV_VAR, "1")
    target = tmp_path / "deep" / "nested" / "audio"
    sound = _make_sound(raw_handle=2)
    result = emit_wav(sound, out_dir=target)
    assert result == target / "0001.wav"
    assert target.is_dir()
    assert result.exists()


def test_emit_wav_filename_uses_logical_handle(monkeypatch, tmp_path: Path):
    """Naming contract pinned: filename is `<sound.handle:04d>.wav` —
    the logical (CTFAK2-adjusted) handle, not the raw on-wire handle.
    Same convention as `png_emit.py` so `0007.png` and `0007.wav` refer
    to the same logical asset slot."""
    monkeypatch.setenv(EMIT_ENV_VAR, "1")
    sound = _make_sound(raw_handle=48)  # handle = 47
    result = emit_wav(sound, out_dir=tmp_path)
    assert result.name == "0047.wav"


def test_emit_wav_passthrough_preserves_exact_bytes(
    monkeypatch, tmp_path: Path
):
    """Byte-count antibody: the on-disk file must have exactly
    `len(sound.audio_data)` bytes and the content must match byte-for-
    byte. A silent padding/truncation bug would make the audio
    un-playable without breaking a higher-level test."""
    monkeypatch.setenv(EMIT_ENV_VAR, "1")
    weird_audio = bytes(range(256)) * 7  # non-trivial, non-aligned length
    sound = _make_sound(raw_handle=1, audio_data=weird_audio)
    result = emit_wav(sound, out_dir=tmp_path)
    assert result.exists()
    on_disk = result.read_bytes()
    assert len(on_disk) == len(weird_audio)
    assert on_disk == weird_audio
