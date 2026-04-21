"""Per-chunk decoders (probe #4).

One module per chunk type. Each decoder:

- Takes a raw payload `bytes` (post-decompression, via compression.read_chunk_payload).
- Returns a typed dataclass with all fields populated.
- Enforces Parser Antibody #2: consumed byte count must equal len(payload).
- Emits stable JSON via an `as_dict()` (or module-level helper) so the
  snapshot-test antibody (#7) can freeze output.

Decoders never open files, never care about compression, and never look
past their payload slice. The caller (main / integration tests) feeds
them bytes. That keeps unit tests trivial and makes every decoder
re-usable for in-memory fixtures.
"""
