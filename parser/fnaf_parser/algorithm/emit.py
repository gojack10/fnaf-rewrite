"""Algorithm emission entry point (stub until Output Emission lands).

`dump_algorithm` is the seam the CLI's `dump-algorithm` subcommand
calls. CLI Scaffolding ships it as a stub raising
`NotImplementedError` so the CLI surface is complete *today* without
letting a consumer read a partial artefact. Output Emission (the next
Algorithm Extraction child after Name Tables Port, Probe #4.13, and
Name Resolver) will replace the body with the real pipeline:

    1. Walk frames, call `FrameEvents.as_dict()` on each.
    2. Resolve IDs -> names via the ported Clickteam name tables.
    3. Decode all 15 parameter codes into structured payloads.
    4. Write per-frame pretty JSON, combined.json, combined.jsonl, and
       manifest.json with SHA-256 per file.

Signature and return contract
-----------------------------

`dump_algorithm(exe, out_dir)` writes to `out_dir` and returns the
set of created paths. Keeping the signature concrete now means the
CLI plumbing doesn't need to change when Output Emission lands —
only this module's body does.
"""

from __future__ import annotations

from pathlib import Path


# Work-unit pointer that replaces this stub. Written out as a constant
# so the CLI can surface the next step in its NotImplementedError
# message without hard-coding a sentence in two places.
NEXT_WORK_UNIT = (
    "Output Emission (SiftText node 9f3a24e5-37cc-4b36-96b0-90543820a421) "
    "ships the real dump_algorithm. Until it lands, `fnaf-parser "
    "dump-algorithm` is a stub."
)


def dump_algorithm(exe: Path, out_dir: Path) -> list[Path]:
    """Emit the algorithm-extraction artefact suite under `out_dir`.

    **Status: stub.** Raises `NotImplementedError`. The real
    implementation is the Output Emission work unit's deliverable.

    Parameters
    ----------
    exe
        Path to the FNAF .exe or equivalent Clickteam 2.5 pack.
    out_dir
        Directory to write `fnaf1_frame_*.json`, `combined.json`,
        `combined.jsonl`, and `manifest.json` into. Auto-created.

    Returns
    -------
    list[Path]
        Every file written, in emission order. Stable across runs so
        the Algorithm Snapshot Antibody can SHA-256 them.
    """
    raise NotImplementedError(NEXT_WORK_UNIT)
