"""Legacy entry point — delegates to `fnaf_parser.cli:main`.

History
-------

Asset Extraction used `uv run main.py <exe>` as the only way to drive
the parser (probe #2 + probe #3 structural walk). CLI Scaffolding
replaces that surface with a proper argparse dispatcher living at
`fnaf_parser.cli:main`, exposed as the `fnaf-parser` script via
`[project.scripts]` in `pyproject.toml`.

This file stays around as a thin shim so muscle memory and existing
docs / notebooks that say `uv run main.py <exe>` keep working:

    uv run main.py FiveNightsatFreddys.exe

is now equivalent to

    uv run fnaf-parser parse FiveNightsatFreddys.exe

The shim prepends `parse` when argv[1] looks like a file path rather
than a subcommand name so the pre-CLI invocation style continues to do
the same thing (structural walk + chunks_seen.json emission). If
argv[1] is already a known subcommand (`parse`, `dump-assets`,
`dump-algorithm`) we pass through untouched.
"""

from __future__ import annotations

import sys

from fnaf_parser.cli import main as cli_main

_KNOWN_SUBCOMMANDS = frozenset(
    {"parse", "dump-assets", "dump-algorithm", "dump-runtime-pack", "-h", "--help"}
)


def main() -> int:
    argv = sys.argv[1:]
    # Pre-CLI invocation: `uv run main.py <exe>`. Rewrite to `parse <exe>`
    # so the old one-arg form keeps producing the structural walk output.
    if argv and argv[0] not in _KNOWN_SUBCOMMANDS:
        argv = ["parse", *argv]
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
