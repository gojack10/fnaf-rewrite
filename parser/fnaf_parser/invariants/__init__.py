"""Invariant-extraction package — DSPy.RLM pipeline home.

This subpackage turns the algorithm-extraction artefact
(`combined.jsonl`) into a checked, citation-backed invariant spec that
the Rust rebuild can compile into a test suite. The pipeline is split
into a creative lane and a skepticism lane, intentionally separated:

    Scout Pass children (SiftText)
      └─ DuckDB slice query -> tickets
          └─ DSPy.RLM head-chef  (pipeline.py)
              └─ line-cook Signature -> InvariantRecord (signatures.py)
                  └─ Citation Checker -> accepted.jsonl + quarantine.jsonl
                      └─ Rust #[test] stub emitter (downstream, not yet
                        implemented)

Modules
-------

- `config`
    OpenRouter-backed LLM config. Reads `OPENROUTER_API_KEY` from the
    environment and raises a clear `OpenRouterConfigError` (not a silent
    DSPy traceback) when the key is absent. The pilot model defaults
    are captured here so changing them is a one-file edit.

- `slices`
    DuckDB query layer. One function per Scout Pass slice — returns an
    iterator of dicts shaped like the ticket contract documented on
    each slice node.

- `signatures`
    The single DSPy typed `Signature` the line cook calls, plus the
    Pydantic `InvariantRecord` / `Citation` models that constrain its
    output. Loud-failing Pydantic validation is the first hallucination
    gate (before Citation Checker even runs).

- `citation_checker`
    Deterministic, non-LLM verifier. Loads combined.json into memory
    once, walks every record's citations, accepts the trivially
    verifiable ones, quarantines everything else with a reason. NEVER
    has an LLM fallback — that rule is the point of the design.

- `pipeline`
    Head-chef orchestrator. Loads a slice, dispatches tickets to the
    line-cook Signature via DSPy.RLM, writes raw JSONL output for the
    Citation Checker to consume. The only module that requires a live
    API key at import time of its `run()` entry point.

OpenRouter API key gate
-----------------------

`config`, `slices`, `signatures`, and `citation_checker` are all fully
testable without an API key — they are pure-Python / pure-DuckDB /
pure-Pydantic. Only `pipeline.run()` requires `OPENROUTER_API_KEY` to
be set. The CLI surfaces the missing-key condition as a clean exit
code + error message; it is never hidden behind a DSPy traceback.
"""
