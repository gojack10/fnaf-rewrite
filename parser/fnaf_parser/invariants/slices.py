"""DuckDB slice loaders for the three Scout Pass axes.

One function per Scout Pass child node. Each returns a list of
ticket-shaped dicts matching the contract documented on the
corresponding SiftText node:

- `load_slice_a` -> one ticket per `(type, num_name)` bucket
  (exhaustiveness backbone, Slice A)
- `load_slice_b` -> one ticket per `(object_type_name, object_name)`
  bucket (FNAF-flavored, Slice B)
- `load_slice_c` -> one ticket per row with an EXPRESSION-bearing
  parameter (pilot target, Slice C)

The SQL is the same SQL pinned on the slice nodes — if it ever drifts
there is a bug on one side or the other. Tests compare this module's
output to a small hand-built JSONL fixture so the query contract stays
machine-checked.

Why DuckDB and not pandas / a hand-rolled loader
------------------------------------------------

- `combined.jsonl` is big-ish (~2532 rows, each a nested dict with up
  to ~20 parameters) — DuckDB reads it in a single pass, handles
  lazy column projection, and keeps memory flat.
- DuckDB's `list_filter(..., lambda)` lets Slice C filter inside an
  array column without exploding rows — cleaner than a Python loop.
- The same queries are portable to `duckdb` CLI so the owner can run
  them by hand against the same artefact — no "only works under the
  Python driver" trap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import duckdb


# --- Ticket shapes (typed for editor help, not runtime-enforced) --------


class RowRecord(TypedDict, total=False):
    """One cond/action row, as it appears inside a ticket's `rows`."""

    frame: int
    event_group_index: int
    cond_or_act_index: int
    type: str
    num_name: str
    object_type_name: str | None
    object_name: str | None
    parameters: list[dict[str, Any]]


class BucketTicket(TypedDict):
    """Slice A / B ticket — many rows sharing a bucket key."""

    bucket_key: dict[str, str]
    occurrence_count: int
    rows: list[RowRecord]


class ExpressionTicket(TypedDict):
    """Slice C ticket — a single row with its EXPRESSION params called out."""

    frame: int
    event_group_index: int
    cond_or_act_index: int
    type: str
    num_name: str
    object_type_name: str | None
    object_name: str | None
    parameters: list[dict[str, Any]]
    expr_params: list[dict[str, Any]]


# --- Query constants (re-pin if Scout Pass children rewrite their SQL) --

# NB: Keep these in sync with the `DuckDB query` sections of the Scout
# Pass slice nodes. Any divergence is a bug; the unit tests crosscheck
# the contract via fixture.

_SLICE_A_SQL = """
WITH rows AS (
  SELECT
    frame,
    event_group_index,
    cond_or_act_index,
    type,
    num_name,
    object_type_name,
    object_name,
    parameters
  FROM read_json_auto(?, format='newline_delimited')
)
SELECT
  type,
  num_name,
  COUNT(*) AS occurrence_count,
  list({
    'frame': frame,
    'event_group_index': event_group_index,
    'type': type,
    'cond_or_act_index': cond_or_act_index,
    'object_type_name': object_type_name,
    'object_name': object_name,
    'parameters': parameters
  }) AS rows
FROM rows
GROUP BY type, num_name
ORDER BY occurrence_count DESC, type, num_name
"""

_SLICE_B_SQL = """
WITH rows AS (
  SELECT
    frame,
    event_group_index,
    cond_or_act_index,
    type,
    num_name,
    COALESCE(object_type_name, '(system)') AS object_type_name,
    COALESCE(object_name, '(system)') AS object_name,
    parameters
  FROM read_json_auto(?, format='newline_delimited')
)
SELECT
  object_type_name,
  object_name,
  COUNT(*) AS occurrence_count,
  list({
    'frame': frame,
    'event_group_index': event_group_index,
    'type': type,
    'cond_or_act_index': cond_or_act_index,
    'num_name': num_name,
    'parameters': parameters
  }) AS rows
FROM rows
GROUP BY object_type_name, object_name
ORDER BY object_type_name, object_name, occurrence_count DESC
"""

_SLICE_C_SQL = """
WITH rows AS (
  SELECT
    frame,
    event_group_index,
    cond_or_act_index,
    type,
    num_name,
    object_type_name,
    object_name,
    parameters,
    list_filter(parameters, p -> p.code IN (22, 23, 27, 45)) AS expr_params
  FROM read_json_auto(?, format='newline_delimited')
)
SELECT *
FROM rows
WHERE length(expr_params) > 0
ORDER BY frame, event_group_index, type, cond_or_act_index
"""


# --- Loaders ------------------------------------------------------------


def _connect() -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with JSON loaded.

    Kept private so callers can't accidentally hold a handle after the
    loader returns — each loader opens, queries, and closes in one
    span, which keeps test isolation trivial.
    """
    return duckdb.connect(":memory:")


def _row_to_record(row: tuple[Any, ...], columns: list[str]) -> dict[str, Any]:
    """Turn a DuckDB row tuple into a plain dict, preserving column order
    so callers can rely on a stable shape."""
    return {name: value for name, value in zip(columns, row, strict=True)}


def load_slice_a(jsonl_path: Path) -> list[BucketTicket]:
    """Return Slice A tickets — one per `(type, num_name)` bucket.

    The returned list is ordered by `occurrence_count DESC` then by
    `type, num_name` for stability across runs.
    """
    with _connect() as con:
        cur = con.execute(_SLICE_A_SQL, [str(jsonl_path)])
        columns = [c[0] for c in cur.description]
        rows = cur.fetchall()
    tickets: list[BucketTicket] = []
    for row in rows:
        rec = _row_to_record(row, columns)
        tickets.append(
            {
                "bucket_key": {"type": rec["type"], "num_name": rec["num_name"]},
                "occurrence_count": int(rec["occurrence_count"]),
                "rows": list(rec["rows"]),
            }
        )
    return tickets


def load_slice_b(jsonl_path: Path) -> list[BucketTicket]:
    """Return Slice B tickets — one per `(object_type_name, object_name)`
    bucket.

    Null object_type_name / object_name are coalesced to `'(system)'`
    in the SQL so system-level rules (timer conditions, global-var
    writes with no Active target) land in one dedicated bucket instead
    of being spread across many `None` buckets."""
    with _connect() as con:
        cur = con.execute(_SLICE_B_SQL, [str(jsonl_path)])
        columns = [c[0] for c in cur.description]
        rows = cur.fetchall()
    tickets: list[BucketTicket] = []
    for row in rows:
        rec = _row_to_record(row, columns)
        tickets.append(
            {
                "bucket_key": {
                    "object_type_name": rec["object_type_name"],
                    "object_name": rec["object_name"],
                },
                "occurrence_count": int(rec["occurrence_count"]),
                "rows": list(rec["rows"]),
            }
        )
    return tickets


def load_slice_c(jsonl_path: Path) -> list[ExpressionTicket]:
    """Return Slice C tickets — one per row with an EXPRESSION-bearing
    parameter (codes 22, 23, 27, 45). Row-per-ticket is deliberate: the
    numeric-math pilot treats each expression as its own invariant
    candidate, so bucketing would smear distinct expressions together.
    """
    with _connect() as con:
        cur = con.execute(_SLICE_C_SQL, [str(jsonl_path)])
        columns = [c[0] for c in cur.description]
        rows = cur.fetchall()
    tickets: list[ExpressionTicket] = []
    for row in rows:
        rec = _row_to_record(row, columns)
        tickets.append(
            {
                "frame": int(rec["frame"]),
                "event_group_index": int(rec["event_group_index"]),
                "type": rec["type"],
                "cond_or_act_index": int(rec["cond_or_act_index"]),
                "num_name": rec["num_name"],
                "object_type_name": rec["object_type_name"],
                "object_name": rec["object_name"],
                "parameters": list(rec["parameters"]),
                "expr_params": list(rec["expr_params"]),
            }
        )
    return tickets


__all__ = [
    "BucketTicket",
    "ExpressionTicket",
    "RowRecord",
    "load_slice_a",
    "load_slice_b",
    "load_slice_c",
]
