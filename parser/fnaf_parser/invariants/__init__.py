"""Invariant-extraction package — direct-LLM pipeline home.

This subpackage turns the algorithm-extraction artefact
(`combined.jsonl`) into a checked, citation-backed invariant spec that
the Rust rebuild can compile into a test suite. The pipeline is a
three-stage machine with deterministic and LLM-judged gates:

    combined.jsonl
      └─ extract.py                 (DeepSeek Flash @ xhigh, ~6 chunks
          |                          of ~80K tokens, streaming with
          |                          30s heartbeat)
          └─ extracted.jsonl
              └─ citation_checker   (deterministic V2 ladder, zero LLM)
                  └─ accepted.jsonl + quarantine.jsonl
                      └─ coverage.py       (Coverage Antibody, breadth)
                      └─ literal_gate.py   (Layer 1 vocab scan,
                                            Layer 2 Flash-as-judge,
                                            Layer 3 agent co-survey)

Modules
-------

- `citation_checker`
    Deterministic, non-LLM verifier. Loads combined.json into memory
    once, walks every record's citations, accepts the trivially
    verifiable ones, quarantines everything else with a reason. NEVER
    has an LLM fallback — that rule is the point of the design. Also
    hosts the `Citation` / `InvariantRecord` Pydantic models used as
    the schema contract between producer and consumer.

- `extract`
    Direct-LLM extraction pipeline. Greedy-accumulates event_groups
    into ~80K-token chunks, dispatches parallel model calls with
    streaming + heartbeat observability, validates output against
    the inline Pydantic schema, and writes extracted.jsonl +
    quarantine_chunks.jsonl.

- `coverage`
    Coverage Antibody. Post-run DuckDB set-diff of (op, object) pairs
    between combined.jsonl and extracted.jsonl citations. Asserts no
    (op, object) pair is silently dropped.

- `literal_gate`
    Three-layer Literal-Name Gate enforcing the Literal-Until-Proven
    doctrine. Layer 1 is a deterministic vocab-aware scanner (session-10
    refined — single-word role_words match vocab_phrases standalone
    only, not word tokens). Layer 2 is Flash-as-judge at xhigh with the
    probe-0.7 tuned prompt. Layer 3 is an agent-led co-survey run at
    pipeline-run granularity (no code — interactive).

History
-------

The DSPy.RLM-era scaffolding (`slices`, `pipeline`, `extract_signature`,
`review_signature`, `signatures`, `config`) was removed in the session-10
cleanup after probe 0.8 validated a direct-LLM architecture at 100K
chunk scale. See the tree nodes Direct LLM Pipeline, Literal Until
Proven, and Citation Checker for the decision trail.
"""
