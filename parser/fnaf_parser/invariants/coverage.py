"""Coverage Antibody — deterministic post-run breadth test.

Asserts that every (op, object) pair appearing in `combined.jsonl`
surfaces at least once as a citation in the extractor's output. Acts
as the cross-record breadth gate that sits downstream of Citation
Checker's per-record structural gate:

    combined.jsonl
      └─ extract.py       (produces records with citations)
          └─ citation_checker (per-record structural)
              └─ coverage.py  (this file — cross-record breadth)
                  └─ literal_gate.py / Rust Test Emission

Unit of coverage
----------------

One (type, num_name, object_name) triple. `type` is "condition" or
"action", `num_name` is the operation name (`SetAlterableValue`,
`CompareCounter`, ...), `object_name` is the target object (`Active 2`,
`freddy bear`, `String`, ...). A pair is "covered" when some row with
that triple appears as the resolved target of some citation.

Input file
----------

The input defaults to `out/invariants/extracted.jsonl` but can be
pointed at `accepted.jsonl` post-Citation-Checker via `--input`. The
earlier position (extracted.jsonl) catches shape-level coverage gaps
before the checker's structural gate narrows the pile further; the
later position (accepted.jsonl) catches post-checker gaps introduced
by records that quarantined.

Zero LLM. Zero heuristics. DuckDB set diff, runs in seconds.

Exit codes
----------

- 0 — every (type, num_name, object_name) triple covered at least once
- 1 — gaps exist; count and grouping printed to stdout, full list in
      coverage_report.json
- 2 — input file missing / unreadable
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import duckdb


def run(
    combined_jsonl: Path,
    input_jsonl: Path,
    report_path: Path,
) -> int:
    """Run the coverage check. Returns a process exit code."""
    for p in (combined_jsonl, input_jsonl):
        if not p.exists():
            print(f"ERROR: missing input {p}", file=sys.stderr)
            return 2

    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE combined AS
        SELECT
            frame::INTEGER            AS frame,
            event_group_index::INTEGER AS event_group_index,
            type                       AS type,
            cond_or_act_index::INTEGER AS cond_or_act_index,
            num_name                   AS num_name,
            object_name                AS object_name
        FROM read_json_auto(?, format='newline_delimited')
        """,
        [str(combined_jsonl)],
    )

    con.execute(
        """
        CREATE TABLE expected AS
        SELECT DISTINCT type, num_name, object_name FROM combined
        WHERE num_name IS NOT NULL AND object_name IS NOT NULL
        """
    )

    con.execute(
        """
        CREATE TABLE extracted AS
        SELECT * FROM read_json_auto(?, format='newline_delimited')
        """,
        [str(input_jsonl)],
    )

    # Flatten citations: each record's citations[] becomes one row per
    # citation. UNNEST on a list-of-structs exposes the struct fields
    # as a sub-struct we can project off.
    con.execute(
        """
        CREATE TABLE cited_rows AS
        SELECT DISTINCT
            UNNEST(citations).frame              AS frame,
            UNNEST(citations).event_group_index  AS event_group_index,
            UNNEST(citations).type               AS type,
            UNNEST(citations).cond_or_act_index  AS cond_or_act_index
        FROM extracted
        """
    )

    con.execute(
        """
        CREATE TABLE observed AS
        SELECT DISTINCT c.type, c.num_name, c.object_name
        FROM cited_rows cr
        JOIN combined c
          ON cr.frame             = c.frame
         AND cr.event_group_index = c.event_group_index
         AND cr.type              = c.type
         AND cr.cond_or_act_index = c.cond_or_act_index
        WHERE c.num_name IS NOT NULL AND c.object_name IS NOT NULL
        """
    )

    gap_rows = con.execute(
        """
        SELECT type, num_name, object_name
        FROM (SELECT * FROM expected EXCEPT SELECT * FROM observed)
        ORDER BY type, num_name, object_name
        """
    ).fetchall()

    expected_count = con.execute("SELECT COUNT(*) FROM expected").fetchone()[0]
    observed_count = con.execute("SELECT COUNT(*) FROM observed").fetchone()[0]
    combined_rows = con.execute("SELECT COUNT(*) FROM combined").fetchone()[0]
    extracted_records = con.execute("SELECT COUNT(*) FROM extracted").fetchone()[0]
    cited_row_count = con.execute("SELECT COUNT(*) FROM cited_rows").fetchone()[0]

    gaps = [
        {"type": t, "num_name": o, "object_name": n}
        for (t, o, n) in gap_rows
    ]

    # Group gaps by shape to help triage:
    #   - per op: gaps keyed by (type, num_name)
    #   - per object: gaps keyed by (type, object_name)
    #   - isolated: any (type, num_name, object_name) that is NOT part
    #              of a bulk cluster (count=1 in both per-op and per-object)
    per_op: Counter = Counter()
    per_obj: Counter = Counter()
    for g in gaps:
        per_op[(g["type"], g["num_name"])] += 1
        per_obj[(g["type"], g["object_name"])] += 1

    report = {
        "combined_rows": combined_rows,
        "extracted_records": extracted_records,
        "citations": cited_row_count,
        "expected_triples": expected_count,
        "observed_triples": observed_count,
        "gap_count": len(gaps),
        "pass": len(gaps) == 0,
        "gaps": gaps,
        "per_op_clusters": sorted(
            (
                {"type": t, "num_name": o, "missing": c}
                for (t, o), c in per_op.items()
                if c > 1
            ),
            key=lambda d: (-d["missing"], d["type"], d["num_name"]),
        ),
        "per_object_clusters": sorted(
            (
                {"type": t, "object_name": n, "missing": c}
                for (t, n), c in per_obj.items()
                if c > 1
            ),
            key=lambda d: (-d["missing"], d["type"], d["object_name"]),
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # --- Stdout summary ---
    status = "PASS" if report["pass"] else "FAIL"
    print("=" * 68)
    print(f"Coverage Antibody: {status}")
    print("=" * 68)
    print(f"combined rows:      {combined_rows:,}")
    print(f"extracted records:  {extracted_records:,}")
    print(f"citations resolved: {cited_row_count:,}")
    print(f"expected triples:   {expected_count:,}")
    print(f"observed triples:   {observed_count:,}")
    print(f"coverage:           {observed_count}/{expected_count} "
          f"({observed_count / max(expected_count, 1) * 100:.1f}%)")
    print(f"gap count:          {len(gaps)}")
    print(f"report written:     {report_path}")

    if report["per_op_clusters"]:
        print()
        print("top per-op gap clusters (>1 missing per op):")
        for row in report["per_op_clusters"][:10]:
            print(f"  {row['type']:<9} {row['num_name']:<30} missing={row['missing']}")
    if report["per_object_clusters"]:
        print()
        print("top per-object gap clusters (>1 missing per object):")
        for row in report["per_object_clusters"][:10]:
            print(f"  {row['type']:<9} {row['object_name']:<30} missing={row['missing']}")
    if gaps:
        print()
        print(f"first {min(20, len(gaps))} gaps:")
        for g in gaps[:20]:
            print(f"  {g['type']:<9} {g['num_name']:<30} {g['object_name']}")
        if len(gaps) > 20:
            print(f"  ... and {len(gaps) - 20} more (see report)")

    return 0 if report["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_combined = repo_root / "out" / "algorithm" / "combined.jsonl"
    default_input = repo_root / "out" / "invariants" / "extracted.jsonl"
    default_report = repo_root / "out" / "invariants" / "coverage_report.json"

    ap = argparse.ArgumentParser(
        prog="python -m fnaf_parser.invariants.coverage",
        description=(
            "Coverage Antibody: assert every (type, num_name, "
            "object_name) triple in combined.jsonl is cited at least "
            "once by the extractor output."
        ),
    )
    ap.add_argument("--combined", type=Path, default=default_combined)
    ap.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Citations source: extracted.jsonl (default) or "
        "accepted.jsonl post-Citation-Checker.",
    )
    ap.add_argument("--report", type=Path, default=default_report)
    args = ap.parse_args(argv)
    return run(
        combined_jsonl=args.combined,
        input_jsonl=args.input,
        report_path=args.report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
