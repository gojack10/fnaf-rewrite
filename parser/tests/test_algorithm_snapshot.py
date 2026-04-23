"""Algorithm Snapshot Antibody — byte-level regression sentinel.

Pins the five integration-layer facts Algorithm Extraction must keep
producing byte-for-byte across refactors. The narrow unit/shape tests
live in `test_emit.py`; this file is the discoverable pin — when a
decoder refactor shifts output in any way, *this* is the test that
fails, and the commit message convention is:

    test(parser): re-pin algorithm snapshot after <reason>

Pinned facts (see SiftText node
`598d7b05-56c9-4324-a742-4f1e7a9ec496`):

1. Frame count = 17
2. Event group total = 584
3. Cond+act pair total = 2532
4. Parameter code set = {1, 2, 6, 9, 10, 14, 16, 22, 23, 25, 26, 27,
   32, 45, 50} — the 15 FNAF 1 codes the decoder is closed over.
5. SHA-256 of `combined.jsonl` = a pinned hex digest. This is the
   byte-level antibody — every key-order, whitespace, or value
   change trips it.

Hash-stability invariant
------------------------

The hash is only stable because `emit._write_jsonl` serialises every
row with `json.dumps(..., sort_keys=True)`. If that flag is ever
removed, the pin silently drifts on any dict-construction reorder —
so this file also cross-checks that every key in every JSONL row is
emitted in sorted order, which catches the regression without needing
a second Python version to flap.

Gating
------

Every test here is skipif-gated on the real FNAF 1 binary because
these pins are semantic claims about the FNAF 1 data-pack, not about
the pure decoder. Running without the binary is a silent no-op, not a
false green — the skip reason is `"FNAF 1 binary not on disk"`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fnaf_parser.algorithm.emit import dump_algorithm
from fnaf_parser.decoders.event_parameters import FNAF1_PARAMETER_CODES

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


# --- Pinned antibody facts ----------------------------------------------

FNAF1_FRAME_COUNT = 17
FNAF1_EVENT_GROUPS = 584
FNAF1_COND_ACT_PAIRS = 2532

# Re-pin whenever the decoder output intentionally changes. The commit
# message convention is: `test(parser): re-pin algorithm snapshot after
# <reason>` so the hash bump is always traceable to a stated cause.
FNAF1_COMBINED_JSONL_SHA256 = (
    "bae32d609a5ce4804154f9a236d9307a04a963bcde9f52071c419cf22d9a24a1"
)


# --- Antibodies ---------------------------------------------------------


@pytest.fixture(scope="module")
def _dump(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run `dump_algorithm` once per module and share the output dir.

    The full pipeline takes ~1s, and all six antibodies below read the
    same artefact set — sharing avoids six redundant decode passes.
    """
    if not FNAF_EXE.exists():
        pytest.skip("FNAF 1 binary not on disk")
    out = tmp_path_factory.mktemp("algorithm_snapshot")
    dump_algorithm(FNAF_EXE, out)
    return out / "algorithm"


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_frame_count(_dump: Path) -> None:
    manifest = json.loads((_dump / "manifest.json").read_text())
    assert manifest["frame_count"] == FNAF1_FRAME_COUNT


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_event_group_total(_dump: Path) -> None:
    manifest = json.loads((_dump / "manifest.json").read_text())
    assert manifest["total_event_groups"] == FNAF1_EVENT_GROUPS


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_cond_act_pair_total(_dump: Path) -> None:
    manifest = json.loads((_dump / "manifest.json").read_text())
    assert manifest["total_cond_act_pairs"] == FNAF1_COND_ACT_PAIRS


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_parameter_code_set(_dump: Path) -> None:
    """Every EventParameter in every row must carry one of the 15 FNAF 1
    parameter codes. A new code surfacing = a new decoder probe is
    overdue; an expected code going missing = a decoder regression."""
    observed: set[int] = set()
    for line in (_dump / "combined.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for p in row.get("parameters", []):
            observed.add(p["code"])
    assert observed == set(FNAF1_PARAMETER_CODES), (
        f"unexpected parameter codes: "
        f"+{sorted(observed - set(FNAF1_PARAMETER_CODES))}  "
        f"-{sorted(set(FNAF1_PARAMETER_CODES) - observed)}"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_combined_jsonl_sha256(_dump: Path) -> None:
    """The byte-level pin. Tripping this test means *something* in the
    decoder changed — re-pin deliberately with a conventional commit."""
    h = hashlib.sha256((_dump / "combined.jsonl").read_bytes()).hexdigest()
    assert h == FNAF1_COMBINED_JSONL_SHA256, (
        f"combined.jsonl SHA-256 drifted\n"
        f"  expected: {FNAF1_COMBINED_JSONL_SHA256}\n"
        f"  actual:   {h}\n"
        f"If the decoder change was intentional, re-pin via commit "
        f"'test(parser): re-pin algorithm snapshot after <reason>'."
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_manifest_files_reconcile_on_disk(_dump: Path) -> None:
    """Every file the manifest claims must exist on disk with matching
    SHA-256 + size_bytes. Prevents silent manifest/artefact drift —
    e.g. a writer emitting a stale file the manifest doesn't know
    about, or vice versa."""
    manifest = json.loads((_dump / "manifest.json").read_text())
    files = manifest["files"]
    # 17 per-frame JSONs + combined.json + combined.jsonl = 19 entries.
    # Manifest does NOT self-reference (it's written last so the digest
    # would be undefined) — exactly 19 entries expected.
    assert len(files) == FNAF1_FRAME_COUNT + 2
    for rel, meta in files.items():
        path = _dump / rel
        assert path.is_file(), f"manifest references missing file: {rel}"
        assert path.stat().st_size == meta["size_bytes"], (
            f"size mismatch on {rel}: "
            f"manifest={meta['size_bytes']} disk={path.stat().st_size}"
        )
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
        ), f"sha256 mismatch on {rel}"


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_antibody_jsonl_keys_are_sorted(_dump: Path) -> None:
    """The pinned SHA-256 only stays stable because `_write_jsonl` uses
    `json.dumps(..., sort_keys=True)`. If that flag is ever dropped, the
    hash silently drifts on any dict-construction reorder. Guard the
    invariant directly: every row (and every nested dict) must have its
    keys emitted in sorted order."""

    def _assert_sorted(obj: object, path: str = "") -> None:
        if isinstance(obj, dict):
            keys = list(obj.keys())
            assert keys == sorted(keys), (
                f"unsorted keys at {path or '<root>'}: {keys}"
            )
            for k, v in obj.items():
                _assert_sorted(v, f"{path}.{k}" if path else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _assert_sorted(v, f"{path}[{i}]")

    for ln_no, line in enumerate(
        (_dump / "combined.jsonl").read_text().splitlines()
    ):
        if not line.strip():
            continue
        _assert_sorted(json.loads(line), path=f"line{ln_no}")
