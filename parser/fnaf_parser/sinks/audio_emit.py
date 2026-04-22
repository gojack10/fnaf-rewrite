"""WAV emission sink for `Sound` records (probe #9.1 audio inspection).

Pivot note (2026-04-21 recon)
-----------------------------

Probe #9.1's first move was a magic-byte + format-code histogram across
the full FNAF 1 0x6668 bank. Result: 52/52 sounds are `RIFF/WAVE`
containers with `fmt_code == 0x0001` (uncompressed PCM), and every blob's
RIFF-declared length reconciles to `len(audio_data) - 8` exactly. That
means `Sound.audio_data` — the opaque byte slice probe #9 hands us — is
*already* a complete, byte-valid `.wav` file. No container dispatcher,
no ADPCM decoder, no OGG re-wrapping is on the critical path for FNAF 1.

Consequence: this sink passes `audio_data` through to disk verbatim. The
module mirrors `png_emit.py`'s env-gate, path layout, and force-flag
shape deliberately — if a future pack ships ADPCM or OGG we'll add a
dispatcher at the *encoder* layer; the sink's write surface stays the
same.

Env-gate and path layout
------------------------

Gated behind the `FNAF_PARSER_EMIT_AUDIO` environment variable. Default
off so CI and generic developer test runs don't write 97 MiB of audio to
disk. Default output directory is `parser/out/audio/`, file naming is
`<handle 04d>.wav` (logical handle, matching the PNG sink convention).
Safe to wipe between runs — nothing else writes to `parser/out/`.

Why no re-encoding path
-----------------------

The whole point of the envelope-only scope on probe #9 was to push the
opacity decision to here. The recon collapsed that decision to
"passthrough" for FNAF 1, so this module contains one `write_bytes` call
and its env-gate — no hand-rolled RIFF writer, no soundfile dep. If
`Sound.audio_data` ever fails the "already a valid RIFF/WAVE" invariant
on some other pack, the decoder antibody
`test_fnaf1_sound_bank_all_riff_wave_pcm` fires first; we don't try to
paper over it here.
"""

from __future__ import annotations

import os
from pathlib import Path

from fnaf_parser.decoders.sounds import Sound

# Env var that flips the sink on. Any of the POSIX-ish truthy strings —
# `1` / `true` / `yes` / `on` (case-insensitive). Absence = off.
EMIT_ENV_VAR = "FNAF_PARSER_EMIT_AUDIO"

# Default output directory, relative to the parser package root. Mirrors
# `png_emit.DEFAULT_OUT_DIR` (`parser/out/images`) so probe #100's
# inspection layout is `parser/out/{images,audio}/` — same shape.
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "out" / "audio"


def _should_emit() -> bool:
    """Return True when `FNAF_PARSER_EMIT_AUDIO` is set to a truthy value.

    Accepted truthy strings: `1`, `true`, `yes`, `on` (case-insensitive).
    Anything else — including absence, empty string, and arbitrary
    nonsense values like `random` — means don't emit. A nonsense value
    silently enabling emission would be an unpleasant surprise; we
    enforce the explicit allow-list instead.
    """
    raw = os.environ.get(EMIT_ENV_VAR, "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def emit_wav(
    sound: Sound,
    *,
    out_dir: Path | None = None,
    force: bool = False,
) -> Path | None:
    """Write `sound.audio_data` to `<out_dir>/<handle>.wav` if emission
    is enabled.

    Returns the path written to, or `None` when emission is gated off by
    the env var. `force=True` bypasses the env-var gate — used by unit
    tests that need to exercise the write path without setting
    environment state.

    The output directory is created on first write. Filename uses the
    logical (CTFAK2-adjusted) handle padded to 4 digits so
    `ls out/audio/` sorts naturally — e.g. `0007.wav` for
    `sound.handle == 7`. `sound.raw_handle` is deliberately NOT in the
    filename: logical handle is what object references use, so `0007.wav`
    matches `0007.png` from the image sink for the same logical slot.
    """
    if not force and not _should_emit():
        return None

    target_dir = out_dir if out_dir is not None else DEFAULT_OUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{sound.handle:04d}.wav"

    # Passthrough: `audio_data` is already a valid RIFF/WAVE file for
    # every FNAF 1 record per probe #9.1 recon. If a future pack ships
    # non-RIFF audio, the decoder-level antibodies will fire before this
    # sink silently writes a `.wav` file that isn't a wav.
    out_path.write_bytes(sound.audio_data)
    return out_path
