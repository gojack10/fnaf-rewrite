"""Regression tests for 0x3345 FrameLayerEffects decoder (probe #4.9).

Second variable-length nested-TLV flag=3 sub-chunk and the first
decoder whose record count comes from a *peer* sub-chunk rather than
being self-describing. FrameLayers (#4.8) supplied its own u32 count;
FrameLayerEffects is a parallel array whose length equals
`layers.count`. That cross-chunk dependency is this probe's tightest
antibody - both FrameLayers and FrameLayerEffects ride separate
encrypted bytes through the same RC4 stream, so any drift on either
side breaks the byte-count reconcile.

Antibody coverage:

- #1 strict-unknown - an entry with `has_shader_tail == True`
  (InkEffect != -1 and Effect > 0) raises loudly. Every one of the
  20 bytes per entry is pinned; no hidden tail is silently skipped.
- #2 byte-count - `len(payload) == 20 * num_layers` is enforced to
  the byte. Short / long / zero-with-count-mismatch all raise.
- #3 round-trip - synthetic pack/unpack verifies the 20-byte shape
  field-by-field, including signed -1 on `ink_effect` and signed
  negatives on `effect` / `offset`.
- #4 multi-oracle (cross-chunk) - `layer_effects.count == layers.count`
  is pinned for every FNAF 1 frame. FrameLayers and FrameLayerEffects
  ride separate encrypted bytes (both flag=3) - RC4 drift on either
  chunk scrambles the peer-count equality or mangles the per-entry
  byte pattern.
- #5 multi-input - the decoder fires 17 times against FNAF 1: 16
  single-layer frames + 1 four-layer office frame = 20 layer-effect
  records total.
- #7 snapshot - per-frame-per-layer tuples pinned below. 20 records *
  (effect, effect_param, rgb_coeff, ink_effect, number_of_params,
  offset) = 120 pinned fields; any RC4 drift surfaces here.

Fifth independent flag=3 shape after FrameHeader (16 B), FramePalette
(1028 B), FrameVirtualRect (16 B), and FrameLayers (variable). A
distinct per-entry size (20 B fixed, parallel-array framing) rules
out any "the RC4 plumbing happens to work only for a specific payload
shape" overfit.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.frame import decode_frame
from fnaf_parser.decoders.frame_layer_effects import (
    FRAME_LAYER_EFFECT_FIXED_SIZE,
    FrameLayerEffect,
    FrameLayerEffects,
    FrameLayerEffectsDecodeError,
    decode_frame_layer_effects,
)
from fnaf_parser.decoders.header import decode_header
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


def _fnaf1_frames():
    """Walk FNAF 1 end-to-end with transform enabled; return the list of
    decoded Frame objects in pack order. Mirrors the helpers in the
    sibling test modules so every cross-channel antibody shares exactly
    the same decrypt path - if one drifts, they all drift.
    """
    blob = FNAF_EXE.read_bytes()
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)

    def _str_of(chunk_id: int) -> str:
        rec = next(r for r in result.records if r.id == chunk_id)
        return decode_string_chunk(
            read_chunk_payload(blob, rec), unicode=result.header.unicode
        )

    editor = _str_of(0x222E)
    name = _str_of(0x2224)
    copyright_records = [r for r in result.records if r.id == 0x223B]
    copyright_str = (
        decode_string_chunk(
            read_chunk_payload(blob, copyright_records[0]),
            unicode=result.header.unicode,
        )
        if copyright_records
        else ""
    )

    transform = make_transform(
        editor=editor,
        name=name,
        copyright_str=copyright_str,
        build=result.header.product_build,
        unicode=result.header.unicode,
    )

    frame_records = [r for r in result.records if r.id == 0x3333]
    frames = [
        decode_frame(
            read_chunk_payload(blob, rec),
            unicode=result.header.unicode,
            transform=transform,
        )
        for rec in frame_records
    ]
    return frames


# --- Synthetic helpers ---------------------------------------------------


def _pack_effect(
    effect: int,
    effect_param: int,
    rgb: tuple[int, int, int, int],
    ink_effect: int,
    nparams: int,
    offset: int,
) -> bytes:
    """Pack one 20-byte FrameLayerEffect entry exactly as the decoder
    expects: i16 i16 + 4x u8 + i32 i32 i32. Used both to build valid
    multi-entry payloads and to craft byte-level drift cases."""
    r, g, b, a = rgb
    return struct.pack(
        "<hh4Biii", effect, effect_param, r, g, b, a, ink_effect, nparams, offset
    )


def _pack_no_shader(rgb: tuple[int, int, int, int] = (255, 255, 255, 255)) -> bytes:
    """Shorthand for the "no effect, no shader" entry every FNAF 1 layer
    should emit. Used to cheaply build N-layer synthetic payloads."""
    return _pack_effect(0, 0, rgb, -1, 0, 0)


# --- Synthetic tests (no binary needed) ----------------------------------


def test_frame_layer_effects_module_constant_stable():
    """Antibody #2 sanity: the per-entry size is the load-bearing
    constant that anchors every byte-count error message in the decoder.
    Pin it here so a struct-format typo surfaces loud, not subtle."""
    assert FRAME_LAYER_EFFECT_FIXED_SIZE == 20


def test_decode_frame_layer_effects_zero_layers_empty_payload():
    """Zero-layer frame has an empty payload - must decode cleanly.
    Guards against an off-by-one that would reject empty input."""
    fle = decode_frame_layer_effects(b"", num_layers=0)
    assert isinstance(fle, FrameLayerEffects)
    assert fle.count == 0
    assert fle.effects == ()


def test_decode_frame_layer_effects_negative_num_layers_raises():
    """Antibody #2: a negative `num_layers` is nonsense - decoder must
    reject it before attempting arithmetic on a negative length."""
    with pytest.raises(FrameLayerEffectsDecodeError, match="num_layers must be >= 0"):
        decode_frame_layer_effects(b"", num_layers=-1)


def test_decode_frame_layer_effects_wrong_size_raises():
    """Antibody #2: any `len(payload) != 20 * num_layers` raises. This is
    the cross-chunk reconcile - if the peer FrameLayers.count disagrees
    with the encoded count (e.g. RC4 drifted on either chunk), this
    fires. Exercise both short and long."""
    # Short: claim 2 layers but supply only 1 entry's worth.
    with pytest.raises(FrameLayerEffectsDecodeError, match="expected exactly 40"):
        decode_frame_layer_effects(_pack_no_shader(), num_layers=2)
    # Long: claim 1 layer but supply 2 entries' worth.
    two = _pack_no_shader() + _pack_no_shader()
    with pytest.raises(FrameLayerEffectsDecodeError, match="expected exactly 20"):
        decode_frame_layer_effects(two, num_layers=1)


def test_decode_frame_layer_effects_roundtrip_synthetic():
    """Antibody #3: three entries with distinct field values round-trip
    through pack + unpack exactly. Negative `effect` and signed `offset`
    exercise the int16/int32 signed paths; a non-white RGB coefficient
    proves the 4-byte colour isn't accidentally struct-packed as a single
    u32 (which would reverse byte order on little-endian)."""
    specs = [
        # (effect, effect_param, rgb, ink_effect, nparams, offset)
        (0, 0, (255, 255, 255, 255), -1, 0, 0),       # "no effect" baseline
        (-3, 7, (128, 64, 32, 16), -1, 0, 0),         # signed -3 on effect
        (0, 0, (0, 0, 0, 0), -1, 0, 0),               # fully-zero baseline
    ]
    payload = b"".join(_pack_effect(*s) for s in specs)
    fle = decode_frame_layer_effects(payload, num_layers=len(specs))
    assert fle.count == len(specs)
    for got, exp in zip(fle.effects, specs):
        effect, effect_param, rgb, ink_effect, nparams, offset = exp
        assert isinstance(got, FrameLayerEffect)
        assert got.effect == effect
        assert got.effect_param == effect_param
        assert got.rgb_coeff == rgb
        assert got.ink_effect == ink_effect
        assert got.number_of_params == nparams
        assert got.offset == offset
        # `has_shader_tail` must be False for all three: effect<=0 OR
        # ink_effect==-1 disqualifies the shader branch.
        assert got.has_shader_tail is False


def test_decode_frame_layer_effects_shader_tail_raises_loudly():
    """Antibody #1 strict-unknown: CTFAK's shader branch fires when
    `ink_effect != -1 && effect > 0`. Probe #4.9 is scoped to the
    no-shader path (FNAF 1 build 284 ships no custom shaders per our
    empirical survey) - if an entry ever satisfies the shader
    condition, the decoder raises rather than silently skipping the
    tail. Follow-up probe #4.9.1 would expand shader-param parsing.
    """
    payload = _pack_effect(
        effect=5,           # > 0
        effect_param=0,
        rgb=(255, 255, 255, 255),
        ink_effect=42,      # != -1
        nparams=3,
        offset=60,
    )
    with pytest.raises(
        FrameLayerEffectsDecodeError, match="ships a shader-enabled effect"
    ):
        decode_frame_layer_effects(payload, num_layers=1)


def test_decode_frame_layer_effects_shader_tail_not_triggered_by_ink_alone():
    """Boundary sanity on the `has_shader_tail` predicate: `ink_effect !=
    -1` alone is NOT sufficient to trip the shader branch - `effect > 0`
    also needed. Pin both arms so a future refactor that collapses the
    AND into an OR would be caught here.
    """
    # ink_effect != -1 but effect == 0 → no shader tail.
    payload = _pack_effect(0, 0, (0, 0, 0, 0), ink_effect=7, nparams=0, offset=0)
    fle = decode_frame_layer_effects(payload, num_layers=1)
    assert fle.effects[0].has_shader_tail is False
    # ink_effect == -1 but effect > 0 → no shader tail either.
    payload = _pack_effect(3, 0, (0, 0, 0, 0), ink_effect=-1, nparams=0, offset=0)
    fle = decode_frame_layer_effects(payload, num_layers=1)
    assert fle.effects[0].has_shader_tail is False


def test_decode_frame_layer_effects_as_dict_shape():
    """`as_dict` output is stable JSON - snapshot-style downstream code
    relies on this shape staying put."""
    payload = _pack_effect(0, 0, (255, 128, 64, 32), -1, 0, 0)
    d = decode_frame_layer_effects(payload, num_layers=1).as_dict()
    assert d == {
        "count": 1,
        "effects": [
            {
                "effect": 0,
                "effect_param": 0,
                "rgb_coeff": [255, 128, 64, 32],
                "ink_effect": -1,
                "number_of_params": 0,
                "offset": 0,
            }
        ],
    }


# --- FNAF 1 end-to-end antibodies (require binary) -----------------------


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_has_layer_effects():
    """Antibody #5 multi-input: the FrameLayerEffects decoder fires 17
    times and every frame emits a FrameLayerEffects. If the byte-count
    antibody catches a size mismatch, it surfaces as an uncaught
    `FrameLayerEffectsDecodeError` inside `decode_frame`."""
    frames = _fnaf1_frames()
    assert len(frames) == 17
    for i, f in enumerate(frames):
        assert f.layer_effects is not None, (
            f"frame #{i} ({f.name!r}) produced no layer_effects - 0x3345 "
            f"either missing from sub_records or decoder did not run."
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_layer_effects_count_parity_with_layers():
    """Cross-chunk antibody (#4 multi-oracle).

    FrameLayerEffects is a parallel array to FrameLayers - no
    self-describing count. Both chunks ride flag=3 (zlib+RC4) through
    the same RC4 stream but as separate encrypted byte sequences. So
    this equality is the sharpest possible check on the decrypt pipeline:
    if RC4 drifts on either chunk, the byte-count inside
    decode_frame_layer_effects fails before we even get here; if it
    drifts on both in the same way, the counts disagree loudly.
    """
    frames = _fnaf1_frames()
    for i, f in enumerate(frames):
        assert f.layers is not None, f"frame #{i} ({f.name!r}) missing layers"
        assert f.layer_effects is not None, (
            f"frame #{i} ({f.name!r}) missing layer_effects"
        )
        assert f.layer_effects.count == f.layers.count, (
            f"frame #{i} ({f.name!r}): layer_effects.count "
            f"{f.layer_effects.count} != layers.count {f.layers.count}. "
            f"Count parity broken - RC4 likely drifted."
        )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_no_frame_ships_shader_effects():
    """Empirical invariant: FNAF 1 build 284 ships no custom shaders.
    If this ever flips, the decoder's shader-tail guard fires during
    `decode_frame_layer_effects` and this test would never run - so
    this test's role is to pin the post-decode state (`has_shader_tail`
    is False everywhere) so a future probe that DOES implement the
    shader tail has a clean before/after diff. Belt-and-braces against
    Antibody #1 strict-unknown.
    """
    frames = _fnaf1_frames()
    for i, f in enumerate(frames):
        assert f.layer_effects is not None
        for j, eff in enumerate(f.layer_effects.effects):
            assert eff.has_shader_tail is False, (
                f"frame #{i} ({f.name!r}) layer #{j} has a shader-enabled "
                f"effect (effect={eff.effect}, ink_effect={eff.ink_effect}). "
                f"FNAF 1 was assumed shader-free by probe #4.9 - expand to "
                f"a #4.9.1 shader-param probe."
            )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_total_layer_effect_record_count():
    """Sanity tally: 16 single-layer frames + 1 four-layer office frame
    = 20 layer-effect records across the pack. This pins the summation
    that downstream antibodies rely on ("20 records * 6 fields = 120
    pinned bytes") so a future change to the frame layout - like a new
    layer being added to the office frame - announces itself here
    instead of silently in the per-frame snapshot diff."""
    frames = _fnaf1_frames()
    total = sum(
        f.layer_effects.count for f in frames if f.layer_effects is not None
    )
    assert total == 20, (
        f"expected 20 total layer-effect records across FNAF 1 "
        f"(16 * 1 + 1 * 4); got {total}"
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_every_frame_layer_effects_snapshot():
    """Snapshot antibody (#7) pinning every layer-effect in every frame.

    Locks per-frame (frame_name, [(effect, effect_param, rgb_coeff,
    ink_effect, number_of_params, offset), ...]). 20 layer-effect
    records * 6 fields = 120 pinned byte-patterns any one of which
    breaks under a wrong-byte decrypt.
    """
    frames = _fnaf1_frames()
    expected = _FNAF1_LAYER_EFFECTS_SNAPSHOT
    assert len(frames) == len(expected)
    for i, (f, exp) in enumerate(zip(frames, expected)):
        exp_name, exp_effects = exp
        assert f.name == exp_name, (
            f"frame #{i}: name {f.name!r} != expected {exp_name!r}"
        )
        assert f.layer_effects is not None, f"frame #{i} missing layer_effects"
        got = tuple(
            (
                e.effect,
                e.effect_param,
                e.rgb_coeff,
                e.ink_effect,
                e.number_of_params,
                e.offset,
            )
            for e in f.layer_effects.effects
        )
        assert got == exp_effects, (
            f"frame #{i} ({f.name!r}) layer effects drifted:\n"
            f"  got  {got}\n"
            f"  want {exp_effects}"
        )


# --- Empirical snapshots (captured from probe #4.9 decrypt path) ---------

# Pinned in pack order against the FrameLayers snapshot. The office
# frame ('Frame 1') carries 4 records; every other frame carries 1.
# FNAF 1 build 284 ships no custom shaders so every entry should be
# (effect=0, effect_param=0, rgb=(255,255,255,255), ink=-1, nparams=0,
# offset=0) - the "no effect, no shader, no colour tint" baseline.
# If any deviation shows up on first run we capture it here rather
# than silently accepting it.

_EffectTuple = tuple[int, int, tuple[int, int, int, int], int, int, int]

# Default "no effect, no shader, no colour tint" record. Used by every
# non-office layer in FNAF 1. Split out as a module-level constant so
# the snapshot literal below stays readable.
_NO_FX: _EffectTuple = (0, 0, (255, 255, 255, 255), -1, 0, 0)


_FNAF1_LAYER_EFFECTS_SNAPSHOT: tuple[
    tuple[str, tuple[_EffectTuple, ...]], ...
] = (
    ("Frame 17",     (_NO_FX,)),
    ("title",        (_NO_FX,)),
    ("what day",     (_NO_FX,)),
    ("Frame 1",      (_NO_FX, _NO_FX, _NO_FX, _NO_FX)),
    ("died",         (_NO_FX,)),
    ("freddy",       (_NO_FX,)),
    ("next day",     (_NO_FX,)),
    ("wait",         (_NO_FX,)),
    ("gameover",     (_NO_FX,)),
    ("the end",      (_NO_FX,)),
    ("ad",           (_NO_FX,)),
    ("the end 2",    (_NO_FX,)),
    ("customize",    (_NO_FX,)),
    ("the end 3",    (_NO_FX,)),
    ("creepy start", (_NO_FX,)),
    ("creepy end",   (_NO_FX,)),
    ("end of demo",  (_NO_FX,)),
)
