"""Clickteam RC4-like chunk decryption (probe #4.5).

Clickteam Fusion encrypts chunk payloads with a keyed stream cipher whose
S-box init and PRGA are RC4-*like* but not RC4. Two layers:

1. Key derivation (one-shot per pack)
   - Concatenate the three seed strings (editor, name, copyright) into
     a keystring of low/high UTF-16LE bytes (dropping zero bytes).
   - Fold that keystring through `MakeKeyCombined`: right-rotate-by-1 a
     running byte, XOR into the key buffer, then stamp a terminator byte
     derived from the product of those XORed bytes.
   - Feed that 256-byte key into `InitDecryptionTable`, a KSA variant
     that mixes a custom `hash` rotation and a trigger (`hash ==
     magic_key[key]`) that resets the key index and stops accumulator
     updates. Output: a 256-byte S-box.

2. Per-chunk transform (many times per pack)
   - Copy the S-box, walk it with standard RC4 PRGA indices, XOR the
     chunk bytes. Each chunk starts fresh from the pack S-box — the
     state machine does not carry across chunks.

Cross-references:
- CTFAK2.0 `Core/CTFAK.Core/Memory/Decryption.cs` — authoritative for
  arithmetic; we transcribe its byte-wise semantics, masking to uint8
  at every step.
- Anaconda `mmfparser/data/chunk.pyx::create_transform` — independent
  second-opinion on the call signature and arg order. Its actual
  `prepare_transform` / `transform` live in a C++ sidecar we do not have
  checked in, so CTFAK is the ground truth for the inner loops.

Build-threshold note: CTFAK guards the odd-id first-byte XOR behind
`Settings.Build > 284`; Anaconda uses `> 285`. FNAF 1's build is 284, so
neither fires — we still implement the XOR for future packs but keep it
conservative (CTFAK's threshold, since it is the more current reference).
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAGIC_CHAR = 54
_S_BOX_SIZE = 256
_KEY_MAX = 128  # Anaconda truncates the keystring to 128 before padding.


class EncryptionError(ValueError):
    """Decryption layer failure — bad key length, bad state, etc."""


def _keystring(s: str) -> bytes:
    """Emit low/high UTF-16LE bytes of each char, dropping zero bytes.

    Mirrors CTFAK `KeyString` and Anaconda `create_transform_part`. For
    pure-ASCII input this yields one byte per char (the high byte is
    zero and gets dropped). For wider chars, both bytes appear.
    """
    out = bytearray()
    for ch in s:
        code = ord(ch)
        lo = code & 0xFF
        hi = (code >> 8) & 0xFF
        if lo:
            out.append(lo)
        if hi:
            out.append(hi)
    return bytes(out)


def _make_key_combined(keystr: bytes, magic_char: int = MAGIC_CHAR) -> bytes:
    """256-byte padded key table after the right-rotate-by-1 scramble.

    Transcribes CTFAK `MakeKeyCombined`: iterate `dataLen + 1` times
    (note the `<=` in CTFAK's for loop), XOR each byte with the rotated
    running value `v34`, and stamp `lastKeyByte` at index `dataLen + 1`.
    """
    data_len = len(keystr)
    if data_len >= _S_BOX_SIZE - 1:
        raise EncryptionError(
            f"keystring too long: {data_len} bytes (max {_S_BOX_SIZE - 2}). "
            f"Truncate to {_KEY_MAX} before calling."
        )

    buf = bytearray(_S_BOX_SIZE)
    buf[:data_len] = keystr

    last_key_byte = magic_char & 0xFF
    v34 = magic_char & 0xFF

    for i in range(data_len + 1):
        v34 = ((v34 << 7) | (v34 >> 1)) & 0xFF
        buf[i] ^= v34
        last_key_byte = (last_key_byte + buf[i] * ((v34 & 1) + 2)) & 0xFF

    buf[data_len + 1] = last_key_byte
    return bytes(buf)


def _init_decryption_table(
    magic_key: bytes, magic_char: int = MAGIC_CHAR
) -> bytes:
    """256-byte S-box via Clickteam's custom KSA.

    Line-for-line port of CTFAK `InitDecryptionTable`. Differs from stock
    RC4 KSA in two ways: (a) the accumulator + hash path runs only while
    `never_reset_key` is True, and (b) when `hash == magic_key[key]`
    fires, `hash` is re-seeded from the magic char and `key` resets to 0
    (but `i` keeps advancing).
    """
    if len(magic_key) < _S_BOX_SIZE:
        raise EncryptionError(
            f"magic_key must be {_S_BOX_SIZE} bytes, got {len(magic_key)}"
        )

    s = bytearray(range(_S_BOX_SIZE))

    def _rotate(v: int) -> int:
        return ((v << 7) | (v >> 1)) & 0xFF

    accum = magic_char & 0xFF
    hash_ = magic_char & 0xFF
    never_reset_key = True

    i2 = 0
    key = 0
    for i in range(_S_BOX_SIZE):
        hash_ = _rotate(hash_)

        if never_reset_key:
            accum = (accum + (2 if (hash_ & 1) == 0 else 3)) & 0xFF
            accum = (accum * magic_key[key]) & 0xFF

        if hash_ == magic_key[key]:
            hash_ = _rotate(magic_char & 0xFF)
            key = 0
            never_reset_key = False

        i2 = (i2 + (hash_ ^ magic_key[key]) + s[i]) & 0xFF
        s[i], s[i2] = s[i2], s[i]

        key = (key + 1) & 0xFF

    return bytes(s)


def _transform(s_box: bytes, data: bytes) -> bytes:
    """RC4 PRGA over a per-call copy of the S-box.

    Each chunk must transform against a fresh copy (CTFAK
    `TransformChunk` does `Array.Copy(decodeBuffer, tempBuf, 256)`), so
    chunk order does not matter and the pack-level S-box is immutable.
    """
    temp = bytearray(s_box)
    out = bytearray(data)
    i = 0
    j = 0
    for k in range(len(out)):
        i = (i + 1) & 0xFF
        j = (j + temp[i]) & 0xFF
        temp[i], temp[j] = temp[j], temp[i]
        out[k] ^= temp[(temp[i] + temp[j]) & 0xFF]
    return bytes(out)


@dataclass(frozen=True)
class TransformState:
    """Pack-level decryption state.

    Immutable: `transform` copies `s_box` internally per call, so the
    state is reusable across every encrypted chunk in the pack. Carrying
    `magic_key` and `build` makes the state self-describing for tests
    and JSON snapshots.
    """
    s_box: bytes = field(repr=False)
    magic_key: bytes = field(repr=False)
    build: int

    def transform(self, data: bytes) -> bytes:
        return _transform(self.s_box, data)


def make_transform(
    *,
    editor: str,
    name: str,
    copyright_str: str,
    build: int,
    unicode: bool = True,
    magic_char: int = MAGIC_CHAR,
) -> TransformState:
    """Derive a `TransformState` from the three seed strings.

    Arg-order switch follows CTFAK (`build > 284`). For FNAF 1
    (build=284, unicode pack) the `else` branch fires and the seed is
    `editor + name + copyright`. The unicode path routes each string
    through `_keystring` before concatenation; the ASCII path concats
    raw latin-1 bytes (Anaconda's is_ascii branch).
    """
    if unicode:
        parts = (
            (_keystring(name), _keystring(copyright_str), _keystring(editor))
            if build > 284
            else (_keystring(editor), _keystring(name), _keystring(copyright_str))
        )
        key_bytes = b"".join(parts)
    else:
        ordered = (
            (name, copyright_str, editor)
            if build > 284
            else (editor, name, copyright_str)
        )
        key_bytes = "".join(ordered).encode("latin-1")

    if len(key_bytes) > _KEY_MAX:
        key_bytes = key_bytes[:_KEY_MAX]

    magic_key = _make_key_combined(key_bytes, magic_char=magic_char)
    s_box = _init_decryption_table(magic_key, magic_char=magic_char)
    return TransformState(s_box=s_box, magic_key=magic_key, build=build)


def apply_odd_id_xor(data: bytes, chunk_id: int, build: int) -> bytes:
    """XOR the first byte when chunk id is odd and build > 284.

    CTFAK `DecodeMode3` / `Chunk.Read` do this before calling
    `TransformChunk`. The threshold is build-gated; for FNAF 1 (build=
    284) this is a no-op. Split out so flag-2 / flag-3 call sites can
    share the same guard without duplicating the bit math.
    """
    if (chunk_id & 1) == 0 or build <= 284 or not data:
        return data
    mask = (chunk_id & 0xFF) ^ ((chunk_id >> 8) & 0xFF)
    out = bytearray(data)
    out[0] ^= mask
    return bytes(out)
