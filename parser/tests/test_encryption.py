"""Synthetic tests for the Clickteam decryption port (probe #4.5).

We do not have an independent reference keystream to compare against,
so the authoritative end-to-end antibody lives in `test_frame_header.py`
(FNAF 1 FrameHeader decrypts to 640x480). The tests here are fast,
local sanity checks that trip if the primitives drift.
"""

from __future__ import annotations

from fnaf_parser.encryption import (
    MAGIC_CHAR,
    EncryptionError,
    TransformState,
    _init_decryption_table,
    _keystring,
    _make_key_combined,
    _transform,
    apply_odd_id_xor,
    make_transform,
)
import pytest


def test_keystring_drops_zero_bytes():
    """ASCII in → one byte per char (high byte dropped); wide char in →
    both bytes emitted in LE order."""
    assert _keystring("abc") == b"abc"
    # U+0041 'A' (lo=0x41, hi=0x00) → "A"
    # U+1041 (Myanmar digit one; lo=0x41, hi=0x10) → bytes 0x41, 0x10
    assert _keystring("A၁") == b"\x41\x41\x10"


def test_keystring_drops_nulls_entirely():
    """Char U+0100 has lo=0x00, hi=0x01 — low byte is zero so only
    the high byte is emitted."""
    assert _keystring("Ā") == b"\x01"


def test_make_key_combined_is_256_bytes():
    key = _make_key_combined(b"hello")
    assert len(key) == 256
    # lastKeyByte lives at data_len + 1 = 6 per CTFAK.
    # Anything past that should still be the zero padding from resize.
    assert all(b == 0 for b in key[7:])


def test_make_key_combined_refuses_oversize_input():
    with pytest.raises(EncryptionError, match="too long"):
        _make_key_combined(b"\x00" * 255)


def test_init_decryption_table_is_permutation():
    """Regardless of the key, the KSA output is a permutation of
    0..255 — every byte appears exactly once."""
    key = _make_key_combined(b"some-seed")
    s_box = _init_decryption_table(key)
    assert sorted(s_box) == list(range(256))


def test_transform_is_xor_involution():
    """RC4-like PRGA with a fresh S-box copy per call means
    transform(transform(x)) == x (classic stream-cipher involution)."""
    key = _make_key_combined(b"round-trip")
    s_box = _init_decryption_table(key)
    plaintext = b"The quick brown fox jumps over the lazy dog" * 3
    ciphertext = _transform(s_box, plaintext)
    assert ciphertext != plaintext
    assert _transform(s_box, ciphertext) == plaintext


def test_transform_state_reuse_across_chunks():
    """The pack-level S-box is shared across every chunk in the pack
    (CTFAK copies it per TransformChunk call). Confirm that two
    transforms in a row don't mutate state across each other."""
    state = make_transform(
        editor="editor", name="name", copyright_str="", build=284
    )
    a = state.transform(b"AAAA")
    b = state.transform(b"AAAA")
    assert a == b   # same input + shared S-box ⇒ identical output
    # And the involution still holds.
    assert state.transform(a) == b"AAAA"


def test_apply_odd_id_xor_no_op_for_fnaf1_build():
    """Build=284 sits at the CTFAK threshold (strict `> 284`), so even
    odd chunk ids pass through unchanged. Locking this keeps the flag-
    2/3 path deterministic for FNAF 1."""
    assert apply_odd_id_xor(b"\xAA\xBB", chunk_id=0x3335, build=284) == b"\xAA\xBB"
    assert apply_odd_id_xor(b"\xAA\xBB", chunk_id=0x3334, build=284) == b"\xAA\xBB"


def test_apply_odd_id_xor_fires_for_newer_builds_and_odd_ids():
    """For build > 284 and odd chunk id, the first byte is XORed with
    `(id & 0xFF) ^ (id >> 8)`."""
    out = apply_odd_id_xor(b"\x00\x11", chunk_id=0x3335, build=285)
    expected_mask = (0x35) ^ (0x33)
    assert out[0] == expected_mask
    assert out[1] == 0x11
    # Even chunk id: untouched.
    assert apply_odd_id_xor(b"\x00\x11", chunk_id=0x3334, build=285) == b"\x00\x11"


def test_make_transform_exposes_build_metadata():
    state = make_transform(
        editor="e", name="n", copyright_str="c", build=284
    )
    assert isinstance(state, TransformState)
    assert state.build == 284
    assert len(state.s_box) == 256
    assert len(state.magic_key) == 256


def test_make_transform_build_threshold_swaps_arg_order():
    """CTFAK: build > 284 ⇒ (name, copyright, editor); else ⇒
    (editor, name, copyright). Two states that share all three strings
    but straddle the threshold must produce different S-boxes."""
    old = make_transform(editor="E", name="N", copyright_str="C", build=284)
    new = make_transform(editor="E", name="N", copyright_str="C", build=285)
    assert old.s_box != new.s_box


def test_magic_char_value_is_54():
    """Locking CTFAK's current magic char. Anaconda's was 99 in an
    older build — if someone reverts this we want a fast fail."""
    assert MAGIC_CHAR == 54
