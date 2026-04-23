"""Tests for `fnaf_parser.cli` — the CLI Scaffolding surface.

Coverage is deliberately narrow: we verify the argparse surface
(subcommand names, required flags, help text) and that `parse` +
`dump-assets` dispatch end-to-end against a real FNAF 1 binary when
present. The `dump-algorithm` subcommand's real pipeline is exercised
end-to-end in `test_emit.py`; this file only checks its argparse
surface so the stub-era CLI-level coverage doesn't evaporate.

The actual decode paths (`decode_image_bank`, `decode_sound_bank`,
`read_chunk_payload`, sinks) already have deep coverage in the
dedicated tests; `cmd_dump_assets` is tested via a skipif-gated smoke
run that asserts the sinks get called, not the byte-for-byte output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fnaf_parser.cli import build_parser, main

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


# --- Argparse surface --------------------------------------------------


def test_build_parser_exposes_four_subcommands():
    """The CLI exposes exactly `parse`, `dump-assets`, `dump-algorithm`,
    and `extract-invariants`.

    If a future work unit adds a fifth subcommand, update this list
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
        "extract-invariants",
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


def test_extract_invariants_requires_all_four_flags():
    """All four --slice / --combined-jsonl / --combined-json / --out flags
    are required; argparse should reject a call missing any of them."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["extract-invariants"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "extract-invariants",
                "--slice",
                "c",
                "--combined-jsonl",
                "x.jsonl",
                "--out",
                "out",
            ]
        )  # missing --combined-json


def test_extract_invariants_slice_choices_enforced():
    """--slice only accepts a, b, or c; `d` must be rejected by argparse."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "extract-invariants",
                "--slice",
                "d",
                "--combined-jsonl",
                "x.jsonl",
                "--combined-json",
                "x.json",
                "--out",
                "out",
            ]
        )


def test_extract_invariants_missing_api_key_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When OPENROUTER_API_KEY is unset, the CLI must exit 3 (not crash
    with a DSPy traceback). The remediation message is the actual
    contract — if this exits 1 the user-facing UX is broken.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Create minimal combined.jsonl + combined.json so argparse doesn't
    # trip on file-not-found before the config error has a chance to
    # fire. The handler must fail on the missing API key, not on I/O.
    jsonl_path = tmp_path / "combined.jsonl"
    json_path = tmp_path / "combined.json"
    jsonl_path.write_text("", encoding="utf-8")
    json_path.write_text('{"frames": []}', encoding="utf-8")

    exit_code = main(
        [
            "extract-invariants",
            "--slice",
            "c",
            "--combined-jsonl",
            str(jsonl_path),
            "--combined-json",
            str(json_path),
            "--out",
            str(tmp_path / "inv-out"),
        ]
    )
    assert exit_code == 3


def test_no_subcommand_errors():
    """Calling `fnaf-parser` bare should fail with argparse's usage error."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


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
