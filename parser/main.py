"""CLI for the FNAF parser. Today: probe #2 + probe #3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from fnaf_parser.chunk_walker import (
    chunk_histogram,
    histogram_to_json_payload,
    walk_chunks,
)
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START, pe_data_pack_start


def main() -> int:
    console = Console()
    if len(sys.argv) != 2:
        console.print("[red]usage:[/red] uv run main.py <exe>")
        return 2

    exe = Path(sys.argv[1])

    # ── Probe #2: PE shell walker ────────────────────────────────────────
    start = pe_data_pack_start(exe)
    console.print(f"data pack start: [cyan]0x{start:08x}[/cyan]")
    console.print(f"expected:        [cyan]0x{FNAF1_DATA_PACK_START:08x}[/cyan]")
    if start != FNAF1_DATA_PACK_START:
        console.print("[red]MISMATCH[/red] — binary or parser drifted")
        return 1
    console.print("[green]probe #2 OK[/green]")

    # ── Probe #3: chunk-list walker ──────────────────────────────────────
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
    # Antibody #6: loud skip log.
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

    # Emit the chunks_seen.json artifact.
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    artifact = out_dir / "chunks_seen.json"
    payload = histogram_to_json_payload(result, freqs)
    artifact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]probe #3 OK[/green] → {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
