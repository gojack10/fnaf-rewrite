"""Runtime-pack emit pipeline.

Boundary contract between the Python parser and the Rust Clickteam
runtime. Emits scene-start state per frame + a master manifest that
indexes every pack file (algorithm/, images/, audio/, runtime_pack/).

V0 scope: wire-up only, no new decoders. Consumes what decode_frame()
already produces in memory. Future work (V0.5+) adds ObjectCommon
decoding for animation tables + per-object defaults.
"""
