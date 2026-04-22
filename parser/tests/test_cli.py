"""Tests for `fnaf_parser.cli` — the CLI Scaffolding surface.

Coverage is deliberately narrow: we verify the argparse surface
(subcommand names, required flags, help text), that `parse` dispatches
end-to-end against a real FNAF 1 binary when present, and that
`dump-algorithm` cleanly surfaces the NotImplementedError from the
stub in `fnaf_parser.algorithm.emit`.

The actual decode paths (`decode_image_bank`, `decode_sound_bank`,
`read_chunk_payload`, sinks) already have deep coverage in the
dedicated tests; `cmd_dump_assets` is tested via a skipif-gated smoke
run that asserts the sinks get called, not the byte-for-byte output.

Future work units (Output Emission in particular) will add their own
tests as they ship the real `dump-algorithm` body — this file owns
only the CLI-scaffolding-level assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fnaf_parser.algorithm.emit import NEXT_WORK_UNIT
from fnaf_parser.cli import build_parser, main

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


# --- Argparse surface --------------------------------------------------


def test_build_parser_exposes_three_subcommands():
    """The CLI exposes exactly `parse`, `dump-assets`, `dump-algorithm`.

    If a future work unit adds a fourth subcommand, update this list
    deliberately — silently widening the CLI surface is the kind of
    drift the Parity Maintenance habit is meant to catch.
    """
    parser = build_parser()
    subparsers_action = next(
        a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"
    )
    assert set(subparsers_action.choices.keys()) == {
        "parse",
        "dump-assets",
        "dump-algorithm",
    }


def test_parse_requires_exe_positional():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["parse"])


def test_dump_assets_requires_out_flag():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["dump-assets", "some.exe"])


def test_dump_algorithm_requires_out_flag():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["dump-algorithm", "some.exe"])


def test_no_subcommand_errors():
    """Calling `fnaf-parser` bare should fail with argparse's usage error."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# --- dump-algorithm stub --------------------------------------------------


def test_dump_algorithm_surfaces_not_implemented(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Stub path: `dump-algorithm` must return 1 and print the NEXT_WORK_UNIT
    pointer to stdout so a consumer running against the stub sees a
    pointer to the work unit that will ship the real pipeline.
    """
    exit_code = main(["dump-algorithm", "nonexistent.exe", "--out", str(tmp_path)])
    assert exit_code == 1
    captured = capsys.readouterr()
    # Part of the NEXT_WORK_UNIT text must leak through — it's the
    # actionable feedback. We pick a distinctive substring rather than
    # assert equality so rich's markup doesn't trip the check.
    assert "Output Emission" in captured.out
    assert "stub" in captured.out.lower()
    # NEXT_WORK_UNIT itself is the source of truth for that message;
    # assert the constant is non-empty so accidental truncation fails.
    assert NEXT_WORK_UNIT
    assert "Output Emission" in NEXT_WORK_UNIT


# --- Real-binary smoke tests (skipif-gated) ------------------------------


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_parse_dispatch_emits_chunks_seen(
    capsys: pytest.CaptureFixture[str],
):
    """End-to-end smoke: `fnaf-parser parse <exe>` must succeed and write
    parser/out/chunks_seen.json with a non-empty records list.

    Uses the real binary because the structural walk invariants
    (magic, chunk counts, reached_last) are what we're verifying the
    CLI routes through correctly.
    """
    exit_code = main(["parse", str(FNAF_EXE)])
    assert exit_code == 0

    out_artefact = (
        Path(__file__).resolve().parent.parent / "out" / "chunks_seen.json"
    )
    assert out_artefact.exists()
    payload = json.loads(out_artefact.read_text())
    assert payload["pack"]["magic"] == "PAMU"
    assert payload["walk"]["reached_last"] is True
    assert len(payload["chunks"]) > 0


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_dump_assets_dispatch_writes_images_and_audio(tmp_path: Path):
    """End-to-end smoke: `fnaf-parser dump-assets` writes PNGs + WAVs.

    We don't byte-compare against a golden — the dedicated decoder /
    sink tests cover that. This test verifies the CLI reaches the sinks
    with `force=True` (bypassing the legacy env gate) and that both
    subdirs receive at least one file.
    """
    exit_code = main(
        ["dump-assets", str(FNAF_EXE), "--out", str(tmp_path)]
    )
    assert exit_code == 0

    images_dir = tmp_path / "images"
    audio_dir = tmp_path / "audio"
    assert images_dir.is_dir()
    assert audio_dir.is_dir()
    pngs = list(images_dir.glob("*.png"))
    wavs = list(audio_dir.glob("*.wav"))
    # FNAF 1: 605 images, 52 sounds. Assert lower bounds rather than
    # exact counts so bank-growth in later probes doesn't break this.
    assert len(pngs) >= 50, f"expected at least 50 PNGs, got {len(pngs)}"
    assert len(wavs) >= 10, f"expected at least 10 WAVs, got {len(wavs)}"
