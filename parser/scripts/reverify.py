"""Re-run the Citation Checker against an existing `raw_records.jsonl`.

The checker is deterministic and LM-free. When its verification ladder
grows a new strategy (e.g. reverse-direction match, all-exprs-in-pseudo),
a full pipeline rerun would waste every LM call the line cook + head
chef already made. This script skips straight to the gate:

    raw_records.jsonl  +  combined.json  →  accepted.jsonl + quarantine.jsonl

Overwrites the accepted/quarantine JSONL in place, prints a delta
summary (how many records moved from quarantine to accepted, how many
stayed quarantined, reason distribution before/after).

Usage
-----

    uv run python scripts/reverify.py                     # default out/invariants
    uv run python scripts/reverify.py --dir out/probe     # probe artefacts
    uv run python scripts/reverify.py --dir out/foo --combined out/algorithm/combined.json

Intentionally NOT a `[project.scripts]` entry — diagnostic tool, not a
product surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from fnaf_parser.invariants.citation_checker import check_records
from fnaf_parser.invariants.signatures import InvariantRecord


REPO_ROOT = Path(__file__).resolve().parent.parent


def _reason_histogram(jsonl: Path) -> Counter[str]:
    """Count outcome reasons across every citation of every record in
    an accepted/quarantine JSONL. Multi-citation records contribute
    multiple entries — mirrors the run.log accounting."""
    counts: Counter[str] = Counter()
    if not jsonl.exists():
        return counts
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        for o in payload.get("outcomes", []):
            counts[o.get("reason", "<missing>")] += 1
    return counts


def _iter_records(raw_path: Path):
    """Stream `raw_records.jsonl` → validated `InvariantRecord`s."""
    with raw_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                yield InvariantRecord.model_validate(payload)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[reverify] line {lineno}: skip ({type(exc).__name__}: {exc})",
                    file=sys.stderr,
                )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-run the Citation Checker on existing raw_records.jsonl."
    )
    ap.add_argument(
        "--dir",
        type=Path,
        default=REPO_ROOT / "out" / "invariants",
        help="Directory containing raw_records.jsonl + accepted/quarantine "
        "(default: out/invariants).",
    )
    ap.add_argument(
        "--combined",
        type=Path,
        default=REPO_ROOT / "out" / "algorithm" / "combined.json",
        help="Path to combined.json (default: out/algorithm/combined.json).",
    )
    args = ap.parse_args(argv)

    raw_path = args.dir / "raw_records.jsonl"
    accepted_path = args.dir / "accepted.jsonl"
    quarantine_path = args.dir / "quarantine.jsonl"

    for p in (raw_path, args.combined):
        if not p.exists():
            print(f"[reverify] missing {p}", file=sys.stderr)
            return 2

    # Snapshot BEFORE counts so we can print a delta.
    before_accepted = _reason_histogram(accepted_path)
    before_quarantine = _reason_histogram(quarantine_path)

    records = list(_iter_records(raw_path))
    print(f"[reverify] re-verifying {len(records)} records from {raw_path}")
    print(f"[reverify] combined: {args.combined}")

    accepted_count, quarantined_count = check_records(
        records, args.combined, accepted_path, quarantine_path
    )

    after_accepted = _reason_histogram(accepted_path)
    after_quarantine = _reason_histogram(quarantine_path)

    print()
    print(f"[reverify] accepted:    {accepted_count}")
    print(f"[reverify] quarantined: {quarantined_count}")
    print()
    print("=== ACCEPTED reason distribution ===")
    print(f"{'reason':<40} {'before':>8} {'after':>8} {'delta':>8}")
    keys = sorted(set(before_accepted) | set(after_accepted))
    for k in keys:
        b = before_accepted[k]
        a = after_accepted[k]
        print(f"{k:<40} {b:>8} {a:>8} {a - b:>+8}")

    print()
    print("=== QUARANTINE reason distribution ===")
    keys = sorted(set(before_quarantine) | set(after_quarantine))
    for k in keys:
        b = before_quarantine[k]
        a = after_quarantine[k]
        print(f"{k:<40} {b:>8} {a:>8} {a - b:>+8}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
