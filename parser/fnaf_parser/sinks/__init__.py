"""Emission sinks — opt-in side-effect outputs for visual inspection.

Sinks take decoder outputs (pixel grids, audio blobs, font metrics) and
write them to disk in human-inspectable formats. They're the visual
correctness layer that catches silent corruption which structural
decoder tests can't: wrong channel order, wrong stride, wrong endianness
on the transparent-colour mask. Every sink is gated behind a
`FNAF_PARSER_EMIT_*` environment variable so automated test runs don't
spam disk unless explicitly asked.
"""
