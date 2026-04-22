"""`fnaf-parser` CLI — argparse dispatcher for the three subcommands.

History
-------

During Asset Extraction the parser's only entry point was
`parser/main.py` (probe #2 + probe #3 structural walk), and the two
emission sinks (`sinks.png_emit`, `sinks.audio_emit`) were gated behind
`FNAF_PARSER_EMIT_PNGS=1` / `FNAF_PARSER_EMIT_AUDIO=1` env vars —
convenient for tests but awkward for humans driving the parser.

CLI Scaffolding replaces that with a proper argparse tree:

    fnaf-parser parse <exe>
        Structural walk: PE locate, chunk histogram, strict-mode
        checks. Equivalent to the pre-CLI `uv run main.py <exe>`
        behaviour.

    fnaf-parser dump-assets <exe> --out <dir>
        Decode every 0x6666 Images chunk + 0x6668 Sounds chunk and
        emit PNGs to `<dir>/images/` and WAVs to `<dir>/audio/`.
        Calls the existing sinks with `force=True` so the env-var gate
        stays specific to test runs and doesn't double-gate the CLI.

    fnaf-parser dump-algorithm <exe> --out <dir>
        Algorithm-extraction entry point; currently a stub that
        surfaces the Output Emission work unit's status. Lands fully
        when Output Emission ships.

Env-var compatibility
---------------------

The legacy `FNAF_PARSER_EMIT_*` gates still work for test suites that
set them explicitly — the sink modules are untouched. The CLI's
`dump-assets` handler uses `force=True`, which bypasses the env gate
entirely; humans calling the CLI get emission unconditionally. The
env vars will be formally deprecated in a later probe — they're kept
functional for now so test suites don't flap during this transition.

Exit codes
----------

- 0: success (expected case)
- 1: semantic failure (e.g. PE data-pack start mismatches expected)
- 2: argparse usage error — raised by argparse itself
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from fnaf_parser.algorithm.emit import dump_algorithm
from fnaf_parser.chunk_walker import (
    chunk_histogram,
    histogram_to_json_payload,
    walk_chunks,
)
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.images import decode_image_bank
from fnaf_parser.decoders.images_pixels import decode_image_pixels
from fnaf_parser.decoders.sounds import decode_sound_bank
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START, pe_data_pack_start
from fnaf_parser.pipeline import load_pack
from fnaf_parser.sinks.audio_emit import emit_wav
from fnaf_parser.sinks.png_emit import emit_png


# --- Chunk IDs used by the CLI dispatch layer ---------------------------

_CHUNK_ID_IMAGES = 0x6666
_CHUNK_ID_SOUNDS = 0x6668


# --- Subcommand: parse ---------------------------------------------------


def cmd_parse(args: argparse.Namespace) -> int:
    """`fnaf-parser parse <exe>` — structural walk only.

    Mirrors the pre-CLI `main.py` behaviour: run probe #2 (PE walker)
    and probe #3 (chunk walker), print a frequency table, and write
    `<exe>/../out/chunks_seen.json`. This is the "does the binary
    even look right?" smoke command.
    """
    console = Console()
    exe: Path = args.exe

    start = pe_data_pack_start(exe)
    console.print(f"data pack start: [cyan]0x{start:08x}[/cyan]")
    console.print(f"expected (FNAF 1): [cyan]0x{FNAF1_DATA_PACK_START:08x}[/cyan]")
    if start != FNAF1_DATA_PACK_START:
        console.print(
            "[yellow]note[/yellow] — start differs from the FNAF 1 reference. "
            "That's informational for non-FNAF-1 packs; proceeding anyway."
        )

    result = walk_chunks(exe, pack_start=start)
    freqs = chunk_histogram(result.records)

    console.print()
    console.print(
        f"pack magic: [cyan]{result.header.magic.decode('ascii')}[/cyan] "
        f"(unicode={result.header.unicode})  "
        f"build: [cyan]{result.header.product_build}[/cyan]  "
        f"runtime: [cyan]0x{result.header.runtime_version:04X}[/cyan]"
    )
    console.print(
        f"records walked:  [cyan]{len(result.records)}[/cyan]   "
        f"unique chunk ids: [cyan]{len(freqs)}[/cyan]   "
        f"reached LAST: "
        f"{'[green]yes[/green]' if result.reached_last else '[red]NO[/red]'}"
    )
    console.print(
        f"[yellow]payload bytes seeked past: "
        f"{result.total_payload_bytes:,}[/yellow] "
        f"(end offset 0x{result.end_offset:08X}, file 0x{result.file_size:08X}, "
        f"trailing {result.file_size - result.end_offset:,} bytes)"
    )

    table = Table(title="Chunks seen (sorted by frequency)", expand=False)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("name")
    table.add_column("conf", style="magenta")
    table.add_column("count", justify="right", style="green")
    table.add_column("total bytes", justify="right")
    table.add_column("first offset", no_wrap=True)
    for f in freqs:
        table.add_row(
            f.id_hex,
            f.name,
            f.confidence,
            str(f.count),
            f"{f.total_bytes:,}",
            f"0x{f.first_offset:08X}",
        )
    console.print(table)

    # Emit the chunks_seen.json artefact next to the exe so it's discoverable
    # without a --out flag — matches the pre-CLI main.py behaviour.
    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    artifact = out_dir / "chunks_seen.json"
    payload = histogram_to_json_payload(result, freqs)
    artifact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]wrote[/green] {artifact}")

    return 0


# --- Subcommand: dump-assets --------------------------------------------


def cmd_dump_assets(args: argparse.Namespace) -> int:
    """`fnaf-parser dump-assets <exe> --out <dir>` — emit PNGs + WAVs.

    Walks the pack, decodes 0x6666 Images + 0x6668 Sounds, and writes
    per-handle artefacts to `<dir>/images/` and `<dir>/audio/`. Uses
    the existing sinks with `force=True` so the env-var gate stays
    specific to tests; humans driving the CLI always get emission.

    Parallelism is deliberately absent — FNAF 1 is 605 images + 52
    sounds, ~200 MB total; serial decode is seconds and the
    correctness story (single-threaded, deterministic ordering) is
    cheaper to reason about than any scheduling layer.
    """
    console = Console()
    exe: Path = args.exe
    out_dir: Path = args.out

    pack = load_pack(exe)

    # Images
    image_rec = next(
        (r for r in pack.walk.records if r.id == _CHUNK_ID_IMAGES), None
    )
    image_count = 0
    if image_rec is not None:
        image_payload = read_chunk_payload(
            pack.blob, image_rec, transform=pack.transform
        )
        image_bank = decode_image_bank(image_payload)
        images_dir = out_dir / "images"
        for img in image_bank.images:
            decoded = decode_image_pixels(img)
            emit_png(decoded, out_dir=images_dir, force=True)
            image_count += 1
        console.print(
            f"[green]wrote[/green] {image_count} PNG(s) -> {images_dir}"
        )
    else:
        console.print("[yellow]no 0x6666 Images chunk found[/yellow]")

    # Sounds
    sound_rec = next(
        (r for r in pack.walk.records if r.id == _CHUNK_ID_SOUNDS), None
    )
    sound_count = 0
    if sound_rec is not None:
        sound_payload = read_chunk_payload(
            pack.blob, sound_rec, transform=pack.transform
        )
        sound_bank = decode_sound_bank(sound_payload)
        audio_dir = out_dir / "audio"
        for snd in sound_bank.sounds:
            emit_wav(snd, out_dir=audio_dir, force=True)
            sound_count += 1
        console.print(
            f"[green]wrote[/green] {sound_count} WAV(s) -> {audio_dir}"
        )
    else:
        console.print("[yellow]no 0x6668 Sounds chunk found[/yellow]")

    return 0


# --- Subcommand: dump-algorithm -----------------------------------------


def cmd_dump_algorithm(args: argparse.Namespace) -> int:
    """`fnaf-parser dump-algorithm <exe> --out <dir>` — STUB.

    Delegates to `fnaf_parser.algorithm.emit.dump_algorithm`, which is
    currently a stub that raises `NotImplementedError`. Output
    Emission replaces the stub with the real pipeline. We catch the
    NotImplementedError here so the CLI surfaces a clean message
    instead of a traceback — users running against this stub need
    actionable feedback, not a stack dump.
    """
    console = Console()
    try:
        written = dump_algorithm(args.exe, args.out)
    except NotImplementedError as exc:
        console.print(f"[red]dump-algorithm: not yet implemented.[/red] {exc}")
        return 1

    console.print(f"[green]wrote[/green] {len(written)} file(s) -> {args.out}")
    return 0


# --- Argparse tree ------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the argparse tree. Kept as a function so tests can
    inspect the surface (help text, subcommand names) without spinning
    up the full CLI runtime."""
    parser = argparse.ArgumentParser(
        prog="fnaf-parser",
        description=(
            "FNAF 1 Clickteam Fusion 2.5 data-pack parser. "
            "Three subcommands: structural parse, asset dump "
            "(PNG + WAV), and algorithm dump (JSON + JSONL)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # parse
    parse_p = subparsers.add_parser(
        "parse",
        help="Structural walk: PE locate + chunk histogram, no decode.",
        description=(
            "Structural walk of the data-pack. Prints the chunk-frequency "
            "table and writes parser/out/chunks_seen.json."
        ),
    )
    parse_p.add_argument("exe", type=Path, help="Path to the Clickteam .exe")
    parse_p.set_defaults(func=cmd_parse)

    # dump-assets
    assets_p = subparsers.add_parser(
        "dump-assets",
        help="Decode 0x6666 Images + 0x6668 Sounds and emit PNGs + WAVs.",
        description=(
            "Full asset-extraction pipeline. Writes per-handle PNGs to "
            "<out>/images/ and per-handle WAVs to <out>/audio/. Replaces "
            "the FNAF_PARSER_EMIT_PNGS=1 / FNAF_PARSER_EMIT_AUDIO=1 "
            "env-var entry points used during Asset Extraction."
        ),
    )
    assets_p.add_argument("exe", type=Path, help="Path to the Clickteam .exe")
    assets_p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (subdirs images/ and audio/ are created under it).",
    )
    assets_p.set_defaults(func=cmd_dump_assets)

    # dump-algorithm
    alg_p = subparsers.add_parser(
        "dump-algorithm",
        help="Emit the algorithm JSON + JSONL spec (stub until Output Emission).",
        description=(
            "Algorithm-extraction dump: per-frame JSON + combined.json + "
            "combined.jsonl + manifest.json. Currently a stub that will "
            "ship fully when the Output Emission work unit lands."
        ),
    )
    alg_p.add_argument("exe", type=Path, help="Path to the Clickteam .exe")
    alg_p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (auto-created).",
    )
    alg_p.set_defaults(func=cmd_dump_algorithm)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Installed as `fnaf-parser` via `[project.scripts]`."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
