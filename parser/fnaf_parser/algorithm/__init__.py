"""Algorithm-extraction package — `dump-algorithm` pipeline home.

This subpackage houses the work that turns the parser's decoded event
data into a machine-readable JSON / JSONL spec for downstream LLM
invariant extraction. Scope is split across the Algorithm Extraction
work units:

- **CLI Scaffolding** (this module's reason for existing) — argparse
  wires `fnaf-parser dump-algorithm` into `emit.dump_algorithm`, which
  is currently a stub.
- **Name Tables Port** — will add `name_tables/` with pure-dict Python
  ports of Anaconda's conditions / actions / expressions / parameters /
  objectTypes name maps.
- **Probe #4.13 (rehydrated)** — will decode all 15 FNAF 1 parameter
  codes into structured Python (including the recursive Expression AST).
- **Name Resolver** — will glue (code=12, object=5) ->
  ("PlayerPressesKey", "Freddy"), loud-failing on unknown IDs.
- **Output Emission** — will replace the `emit.dump_algorithm` stub
  with the real per-frame JSON + combined.json + combined.jsonl +
  manifest.json emission.
- **Algorithm Snapshot Antibody** — will pin frame=17 / event_groups=
  584 / cond+act=2532 / SHA-256 of combined.jsonl as regression floors.

Until Output Emission lands, `fnaf-parser dump-algorithm` surfaces a
`NotImplementedError` rather than emitting a partial artefact that
downstream consumers could accidentally read.
"""
