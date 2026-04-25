"""Runtime-Pack Emit Antibody — V0 scene-start + master manifest pin.

Pins the structural invariants the runtime-pack emit must keep holding
across refactors. Mirrors ``test_algorithm_snapshot.py``'s shape: one
module-scope fixture runs the full emit once; each antibody is a
narrow test that reads the output and asserts one fact.

Pinned facts (see SiftText ``Runtime Pack Extraction`` node,
2026-04-24 scene-start section):

1. Master manifest exists; every listed path resolves on disk; every
   SHA-256 verifies; every size_bytes matches disk size.
2. `runtime_pack/frames_state/` contains exactly 17 files (one per
   0x3333 Frame chunk).
3. Every item_instance.object_info in every frame_state file resolves
   to a handle in the pack-level FrameItems bank (196 handles).
4. Every item_instance.layer in every frame_state file is within the
   valid layer range for that frame.
5. Manifest `source_sha256` matches the `.exe` bytes it read.
6. Manifest does NOT self-reference (writing last means the digest
   would otherwise be undefined).
7. Manifest `counts` section is self-consistent: counts.frames matches
   the number of frame_state files, counts.images matches `*.png`
   count in files, counts.audio matches `*.wav` count.
8. Pre-requisite check: calling `dump_runtime_pack` without prior
   `dump-algorithm` + `dump-assets` raises `RuntimePackEmitError`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fnaf_parser.algorithm.emit import dump_algorithm
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.frame_items import decode_frame_items
from fnaf_parser.pipeline import load_pack
from fnaf_parser.runtime_pack.emit import (
    PACK_VERSION,
    RuntimePackEmitError,
    dump_runtime_pack,
)
from fnaf_parser.sinks.audio_emit import emit_wav
from fnaf_parser.sinks.png_emit import emit_png
from fnaf_parser.decoders.images import decode_image_bank
from fnaf_parser.decoders.images_pixels import decode_image_pixels
from fnaf_parser.decoders.sounds import decode_sound_bank

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


# --- Pinned antibody facts ----------------------------------------------

FNAF1_FRAME_COUNT = 17
FNAF1_FRAME_ITEMS_HANDLES = 196
FNAF1_IMAGE_COUNT = 605
FNAF1_AUDIO_COUNT = 52


# --- Fixture ------------------------------------------------------------


@pytest.fixture(scope="module")
def _pack_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the full emit pipeline once and share the output dir.

    Runs dump_algorithm + dump_assets (needed for the master manifest
    to index something) + dump_runtime_pack in sequence, all against
    one shared tmp directory. All subsequent antibodies read from this
    directory. ~2-3s total (dominated by dump_algorithm).
    """
    if not FNAF_EXE.exists():
        pytest.skip("FNAF 1 binary not on disk")

    out = tmp_path_factory.mktemp("runtime_pack")

    # Algorithm dump (gives us algorithm/)
    dump_algorithm(FNAF_EXE, out)

    # Asset dump equivalent — call the decoders directly rather than
    # through cmd_dump_assets (avoids argparse ceremony in tests).
    pack = load_pack(FNAF_EXE)
    image_rec = next(r for r in pack.walk.records if r.id == 0x6666)
    image_bank = decode_image_bank(
        read_chunk_payload(pack.blob, image_rec, transform=pack.transform)
    )
    images_dir = out / "images"
    for img in image_bank.images:
        emit_png(decode_image_pixels(img), out_dir=images_dir, force=True)

    sound_rec = next(r for r in pack.walk.records if r.id == 0x6668)
    sound_bank = decode_sound_bank(
        read_chunk_payload(pack.blob, sound_rec, transform=pack.transform)
    )
    audio_dir = out / "audio"
    for snd in sound_bank.sounds:
        emit_wav(snd, out_dir=audio_dir, force=True)

    # Runtime-pack dump (the thing under test)
    dump_runtime_pack(FNAF_EXE, out)

    return out


# --- Antibodies ---------------------------------------------------------


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_manifest_exists(_pack_out: Path) -> None:
    """The master manifest must exist at the pinned path."""
    manifest_path = _pack_out / "runtime_pack" / "manifest.json"
    assert manifest_path.is_file(), (
        f"master manifest missing at {manifest_path}"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_manifest_versions_present(_pack_out: Path) -> None:
    """pack_version + decoder_version must be stamped on the manifest."""
    manifest = json.loads(
        (_pack_out / "runtime_pack" / "manifest.json").read_text()
    )
    assert manifest["pack_version"] == PACK_VERSION
    assert "decoder_version" in manifest


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_source_sha256_matches_exe(_pack_out: Path) -> None:
    """`source_sha256` must match the `.exe` byte hash."""
    manifest = json.loads(
        (_pack_out / "runtime_pack" / "manifest.json").read_text()
    )
    exe_sha = hashlib.sha256(FNAF_EXE.read_bytes()).hexdigest()
    assert manifest["source_sha256"] == exe_sha


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_frames_state_file_count(_pack_out: Path) -> None:
    """Exactly 17 per-frame-state JSON files (one per 0x3333 Frame)."""
    frames_state_dir = _pack_out / "runtime_pack" / "frames_state"
    files = sorted(frames_state_dir.glob("frame_*.json"))
    assert len(files) == FNAF1_FRAME_COUNT, (
        f"expected {FNAF1_FRAME_COUNT} frame_state files, "
        f"got {len(files)}: {[f.name for f in files]}"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_manifest_files_reconcile_on_disk(_pack_out: Path) -> None:
    """Every file the master manifest lists must exist with matching
    SHA-256 + size_bytes. Prevents silent manifest/artefact drift."""
    manifest = json.loads(
        (_pack_out / "runtime_pack" / "manifest.json").read_text()
    )
    files = manifest["files"]
    for rel, meta in files.items():
        path = _pack_out / rel
        assert path.is_file(), f"manifest references missing file: {rel}"
        assert path.stat().st_size == meta["size_bytes"], (
            f"size mismatch on {rel}: manifest={meta['size_bytes']} "
            f"disk={path.stat().st_size}"
        )
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        assert h == meta["sha256"], f"sha256 mismatch on {rel}"


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_manifest_does_not_self_reference(_pack_out: Path) -> None:
    """The master manifest must NOT appear in its own `files` list.
    It is written last and a self-reference would have an undefined
    SHA-256 until two-pass emission is implemented."""
    manifest = json.loads(
        (_pack_out / "runtime_pack" / "manifest.json").read_text()
    )
    assert "runtime_pack/manifest.json" not in manifest["files"]


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_counts_are_self_consistent(_pack_out: Path) -> None:
    """Manifest `counts` section must agree with what's on disk."""
    manifest = json.loads(
        (_pack_out / "runtime_pack" / "manifest.json").read_text()
    )
    files = manifest["files"]
    counts = manifest["counts"]

    assert counts["frames"] == FNAF1_FRAME_COUNT
    assert counts["images"] == FNAF1_IMAGE_COUNT
    assert counts["audio"] == FNAF1_AUDIO_COUNT

    # Cross-check counts against the files list
    png_count = sum(1 for k in files if k.endswith(".png"))
    wav_count = sum(1 for k in files if k.endswith(".wav"))
    frame_state_count = sum(
        1 for k in files if k.startswith("runtime_pack/frames_state/")
    )
    assert png_count == counts["images"]
    assert wav_count == counts["audio"]
    assert frame_state_count == counts["frames"]


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_instance_object_info_handles_resolve(_pack_out: Path) -> None:
    """Every item_instance.object_info in every frame must resolve to
    a handle in the pack-level FrameItems bank. Catches any decoder
    drift where a frame references an undefined object type."""
    # Build the pack-level handle set fresh
    pack = load_pack(FNAF_EXE)
    frame_items_rec = next(r for r in pack.walk.records if r.id == 0x2229)
    frame_items = decode_frame_items(
        read_chunk_payload(pack.blob, frame_items_rec, transform=pack.transform),
        unicode=pack.walk.header.unicode,
        transform=pack.transform,
    )
    valid_handles = set(frame_items.by_handle.keys())
    assert len(valid_handles) == FNAF1_FRAME_ITEMS_HANDLES

    frames_state_dir = _pack_out / "runtime_pack" / "frames_state"
    for frame_path in sorted(frames_state_dir.glob("frame_*.json")):
        frame_state = json.loads(frame_path.read_text())
        for inst in frame_state["item_instances"]["instances"]:
            assert inst["object_info"] in valid_handles, (
                f"{frame_path.name}: item_instance object_info="
                f"{inst['object_info']} handle=/{inst['handle']} "
                f"not in FrameItems bank"
            )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_instance_layer_is_in_range(_pack_out: Path) -> None:
    """Every item_instance.layer in every frame must be a valid index
    into that frame's layer list. Cross-chunk sanity check that the
    emit didn't shuffle layers vs instances across frames."""
    frames_state_dir = _pack_out / "runtime_pack" / "frames_state"
    for frame_path in sorted(frames_state_dir.glob("frame_*.json")):
        frame_state = json.loads(frame_path.read_text())
        layer_count = frame_state["layers"]["count"]
        for inst in frame_state["item_instances"]["instances"]:
            layer = inst["layer"]
            assert 0 <= layer < layer_count, (
                f"{frame_path.name}: item_instance handle={inst['handle']} "
                f"layer={layer} outside [0, {layer_count})"
            )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_refuses_without_prerequisites(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """`dump_runtime_pack` must raise `RuntimePackEmitError` if
    algorithm/, images/, or audio/ are missing. Guards against silent
    partial emits that would produce a misleading manifest."""
    out = tmp_path_factory.mktemp("runtime_pack_no_prereqs")
    # No dump-algorithm or dump-assets called — pre-req dirs don't exist.
    with pytest.raises(RuntimePackEmitError) as exc_info:
        dump_runtime_pack(FNAF_EXE, out)
    # Error message should name the missing dirs so the human can act.
    msg = str(exc_info.value)
    assert "algorithm" in msg
    assert "images" in msg
    assert "audio" in msg
