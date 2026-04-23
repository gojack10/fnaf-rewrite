"""Tests for probe #4.13 EventParameter payload decode.

Covers all 15 FNAF 1 parameter codes, the recursive Expression AST, every
loud-failure path, and a smoke test that walks every EventParameter in
every FNAF 1 frame through `decode_event_parameter`.

Contract under test: loaders consume their documented bytes, tolerate
trailing padding (CTFAK2.0 `reader.Seek(currentPosition + size)` idiom),
and raise loudly on insufficient bytes / missing terminators / missing
Expression END markers.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.compression import read_chunk_payload
from fnaf_parser.decoders.event_parameters import (
    FNAF1_PARAMETER_CODES,
    EventParameterDecodeError,
    decode_event_parameter,
)
from fnaf_parser.decoders.frame import decode_frame
from fnaf_parser.decoders.strings import decode_string_chunk
from fnaf_parser.encryption import make_transform
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"


# --- String helpers -----------------------------------------------------


def _u16le(value: str) -> bytes:
    """Encode `value` as UTF-16LE + wide NUL terminator."""
    return value.encode("utf-16-le") + b"\x00\x00"


def _ascii(value: str) -> bytes:
    return value.encode("ascii") + b"\x00"


# --- Fixed-width shape tests -------------------------------------------


class TestShortInt:
    """Codes 10, 26, 50 (Short) and 25 (Int) and 14 (Key)."""

    @pytest.mark.parametrize("code", [10, 26, 50])
    def test_short_positive(self, code: int) -> None:
        out = decode_event_parameter(code, struct.pack("<h", 42), unicode=True)
        assert out == {
            "kind": "Short",
            "value": 42,
            "trailing_hex": "",
            "code": code,
        }

    @pytest.mark.parametrize("code", [10, 26, 50])
    def test_short_negative(self, code: int) -> None:
        out = decode_event_parameter(code, struct.pack("<h", -1), unicode=True)
        assert out["value"] == -1

    @pytest.mark.parametrize("code", [10, 26, 50])
    def test_short_trailing_tolerated(self, code: int) -> None:
        data = struct.pack("<h", 7) + b"\xAA\xBB\xCC\xDD"
        out = decode_event_parameter(code, data, unicode=True)
        assert out["value"] == 7
        assert out["trailing_hex"] == "aabbccdd"

    @pytest.mark.parametrize("code", [10, 26, 50])
    def test_short_wrong_size(self, code: int) -> None:
        with pytest.raises(EventParameterDecodeError, match=">= 2 bytes"):
            decode_event_parameter(code, b"\x00", unicode=True)

    @pytest.mark.parametrize("code", [10, 26])
    def test_short_label_nul_terminated(self, code: int) -> None:
        """Codes 10/26 surface the 32-byte UTF-16 LE animation-state
        label when it's shorter than the buffer (NUL-terminated)."""
        label = "freddy attack"
        label_bytes = label.encode("utf-16-le") + b"\x00\x00"
        pad = b"\x00" * (32 - len(label_bytes))
        data = struct.pack("<h", 3) + label_bytes + pad
        out = decode_event_parameter(code, data, unicode=True)
        assert out["value"] == 3
        assert out["label"] == label

    @pytest.mark.parametrize("code", [10, 26])
    def test_short_label_buffer_filling(self, code: int) -> None:
        """Codes 10/26 surface a buffer-filling label (no wide-NUL)
        when the label fills all 16 UTF-16 LE slots. Empirically
        Clickteam truncates long labels this way — 23/163 FNAF 1 SHORT
        labels hit this case (e.g. `"stage no chica o"`)."""
        label = "stage no chica o"  # exactly 16 chars → 32 bytes UTF-16 LE
        data = struct.pack("<h", 1) + label.encode("utf-16-le")
        out = decode_event_parameter(code, data, unicode=True)
        assert out["label"] == label

    @pytest.mark.parametrize("code", [10, 26, 50])
    def test_short_label_absent_on_zero_pad(self, code: int) -> None:
        """All-zero trailing of any length yields no `label` field."""
        data = struct.pack("<h", 0) + b"\x00" * 32
        out = decode_event_parameter(code, data, unicode=True)
        assert "label" not in out

    @pytest.mark.parametrize("code", [10, 26, 50])
    def test_short_label_absent_on_short_trailing(self, code: int) -> None:
        """Trailing shorter than the 16-byte label floor is ignored —
        guards regression from tiny test-synthetic inputs like
        `b"\\xAA\\xBB\\xCC\\xDD"` being misread as labels."""
        data = struct.pack("<h", 42) + b"\xAA\xBB\xCC\xDD"
        out = decode_event_parameter(code, data, unicode=True)
        assert "label" not in out
        assert out["trailing_hex"] == "aabbccdd"

    def test_short_label_absent_on_non_ascii(self) -> None:
        """Non-ASCII-printable trailing (garbage bytes that happen to
        UTF-16-decode) is NOT surfaced as a label."""
        # 0xBBAA 0xDDCC ... — valid Unicode codepoints but not printable
        # ASCII — so the label gate rejects them.
        data = struct.pack("<h", 5) + b"\xAA\xBB\xCC\xDD" * 8
        out = decode_event_parameter(10, data, unicode=True)
        assert "label" not in out

    def test_int_value(self) -> None:
        out = decode_event_parameter(25, struct.pack("<i", 1_000_000), unicode=True)
        assert out == {
            "kind": "Int",
            "value": 1_000_000,
            "trailing_hex": "",
            "code": 25,
        }

    def test_int_trailing_tolerated(self) -> None:
        data = struct.pack("<i", -42) + b"\x00\x00"
        out = decode_event_parameter(25, data, unicode=True)
        assert out["value"] == -42
        assert out["trailing_hex"] == "0000"

    def test_int_wrong_size(self) -> None:
        with pytest.raises(EventParameterDecodeError, match=">= 4 bytes"):
            decode_event_parameter(25, b"\x00\x00", unicode=True)

    def test_key_value(self) -> None:
        # Key is unsigned (CTFAK2.0 KeyParameter.Read = ReadUInt16).
        out = decode_event_parameter(14, struct.pack("<H", 65), unicode=True)
        assert out == {
            "kind": "Key",
            "value": 65,
            "trailing_hex": "",
            "code": 14,
        }

    def test_key_wrong_size(self) -> None:
        with pytest.raises(EventParameterDecodeError, match=">= 2 bytes"):
            decode_event_parameter(14, b"", unicode=True)


class TestObject:
    """Code 1 — 3 shorts (listIdx, info, type). Middle field unsigned."""

    def test_roundtrip(self) -> None:
        # object_info_list=-1 int16, object_info=5 uint16, object_type=2 int16
        data = struct.pack("<hHh", -1, 5, 2)
        out = decode_event_parameter(1, data, unicode=True)
        assert out == {
            "kind": "Object",
            "object_info_list": -1,
            "object_info": 5,
            "object_type": 2,
            "trailing_hex": "",
            "code": 1,
        }

    def test_unsigned_middle(self) -> None:
        # 0xFFFF in the middle field must decode as 65535, not -1.
        data = struct.pack("<hHh", 0, 0xFFFF, 0)
        out = decode_event_parameter(1, data, unicode=True)
        assert out["object_info"] == 0xFFFF

    def test_trailing_tolerated(self) -> None:
        # FNAF 1 shape: 8 bytes (6 loader + 2 pad).
        data = struct.pack("<hHh", 1, 2, 3) + b"\x00\x00"
        out = decode_event_parameter(1, data, unicode=True)
        assert out["object_info_list"] == 1
        assert out["trailing_hex"] == "0000"

    def test_wrong_size(self) -> None:
        with pytest.raises(EventParameterDecodeError, match=">= 6 bytes"):
            decode_event_parameter(1, b"\x00" * 4, unicode=True)


class TestTime:
    """Code 2 — int32 timer + int32 loops + int16 comparison (CTFAK2.0 shape)."""

    def test_roundtrip(self) -> None:
        data = struct.pack("<iih", 5000, 3, 0)
        out = decode_event_parameter(2, data, unicode=True)
        assert out == {
            "kind": "Time",
            "timer": 5000,
            "loops": 3,
            "comparison": 0,
            "trailing_hex": "",
            "code": 2,
        }

    def test_negative_loops(self) -> None:
        data = struct.pack("<iih", 1000, -1, 0)
        out = decode_event_parameter(2, data, unicode=True)
        assert out["loops"] == -1

    def test_comparison_field(self) -> None:
        # CTFAK2.0 adds a 2-byte comparison field past timer/loops.
        data = struct.pack("<iih", 0, 0, 4)
        out = decode_event_parameter(2, data, unicode=True)
        assert out["comparison"] == 4

    def test_trailing_tolerated(self) -> None:
        data = struct.pack("<iih", 0, 0, 0) + b"\xDE\xAD"
        out = decode_event_parameter(2, data, unicode=True)
        assert out["trailing_hex"] == "dead"

    def test_wrong_size(self) -> None:
        # Anaconda's 8-byte shape is now rejected — Time needs 10.
        with pytest.raises(EventParameterDecodeError, match=">= 10 bytes"):
            decode_event_parameter(2, b"\x00" * 8, unicode=True)


class TestClick:
    """Code 32 — uint8 button + uint8 double."""

    def test_left_single(self) -> None:
        out = decode_event_parameter(32, b"\x00\x00", unicode=True)
        assert out == {
            "kind": "Click",
            "click": 0,
            "double": False,
            "trailing_hex": "",
            "code": 32,
        }

    def test_right_double(self) -> None:
        out = decode_event_parameter(32, b"\x02\x01", unicode=True)
        assert out["click"] == 2 and out["double"] is True

    def test_trailing_tolerated(self) -> None:
        # FNAF 1 shape: 6 bytes (2 loader + 4 pad).
        data = b"\x00\x00" + b"\xAA\xBB\xCC\xDD"
        out = decode_event_parameter(32, data, unicode=True)
        assert out["trailing_hex"] == "aabbccdd"

    def test_wrong_size(self) -> None:
        with pytest.raises(EventParameterDecodeError, match=">= 2 bytes"):
            decode_event_parameter(32, b"\x00", unicode=True)


class TestSample:
    """Code 6 — handle(int16) + flags(uint16) + null-term string + pad."""

    def test_roundtrip_unicode(self) -> None:
        data = struct.pack("<hH", 7, 0x0002) + _u16le("sting.wav")
        out = decode_event_parameter(6, data, unicode=True)
        assert out == {
            "kind": "Sample",
            "handle": 7,
            "flags": 0x0002,
            "name": "sting.wav",
            "trailing_hex": "",
            "code": 6,
        }

    def test_roundtrip_empty_name(self) -> None:
        data = struct.pack("<hH", 0, 0) + _u16le("")
        out = decode_event_parameter(6, data, unicode=True)
        assert out["name"] == ""

    def test_roundtrip_ascii(self) -> None:
        data = struct.pack("<hH", 3, 1) + _ascii("boo")
        out = decode_event_parameter(6, data, unicode=False)
        assert out["name"] == "boo"

    def test_missing_terminator(self) -> None:
        data = struct.pack("<hH", 0, 0) + "no-nul".encode("utf-16-le")
        with pytest.raises(EventParameterDecodeError, match="missing NUL"):
            decode_event_parameter(6, data, unicode=True)

    def test_trailing_tolerated(self) -> None:
        # CTFAK2.0 Sample.Write pads to a fixed 128+ byte width — any
        # bytes past the name's terminator are trailing_hex, not drift.
        data = struct.pack("<hH", 0, 0) + _u16le("x") + b"\xAA\xBB"
        out = decode_event_parameter(6, data, unicode=True)
        assert out["name"] == "x"
        assert out["trailing_hex"] == "aabb"

    def test_header_too_short(self) -> None:
        with pytest.raises(EventParameterDecodeError, match=">= 4 bytes"):
            decode_event_parameter(6, b"\x00\x00", unicode=True)


class TestPosition:
    """Code 16 — 22-byte Position record + pad."""

    def test_roundtrip(self) -> None:
        # 10 fields: HhhhhhihhH
        data = struct.pack(
            "<HhhhhhihhH", 11, 0x40, 100, 200, 0, 45, 90, -1, 5, 2
        )
        out = decode_event_parameter(16, data, unicode=True)
        assert out == {
            "kind": "Position",
            "object_info_parent": 11,
            "flags": 0x40,
            "x": 100,
            "y": 200,
            "slope": 0,
            "angle": 45,
            "direction": 90,
            "type_parent": -1,
            "object_info_list": 5,
            "layer": 2,
            "trailing_hex": "",
            "code": 16,
        }

    def test_trailing_tolerated(self) -> None:
        # FNAF 1 shape: 24 bytes (22 loader + 2 pad).
        data = (
            struct.pack("<HhhhhhihhH", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            + b"\xDE\xAD"
        )
        out = decode_event_parameter(16, data, unicode=True)
        assert out["trailing_hex"] == "dead"

    def test_wrong_size(self) -> None:
        with pytest.raises(EventParameterDecodeError, match=">= 22 bytes"):
            decode_event_parameter(16, b"\x00" * 20, unicode=True)


class TestCreate:
    """Code 9 — Position(22) + uint16 inst + uint16 info + pad.

    CTFAK2.0 loader is 26 bytes; FNAF 1 carries 32. We surface the
    trailing 6 as `trailing_hex`.
    """

    def test_roundtrip_exact(self) -> None:
        position = struct.pack(
            "<HhhhhhihhH", 1, 0, 10, 20, 0, 0, 0, -1, -1, 0
        )
        tail = struct.pack("<HH", 3, 7)
        data = position + tail
        assert len(data) == 26
        out = decode_event_parameter(9, data, unicode=True)
        assert out["kind"] == "Create"
        assert out["object_instances"] == 3
        assert out["object_info"] == 7
        assert out["trailing_hex"] == ""
        assert out["position"]["x"] == 10
        assert out["position"]["y"] == 20
        assert out["code"] == 9

    def test_trailing_fnaf1_shape(self) -> None:
        # FNAF 1 Create payloads are 32 bytes: 26 loader + 6 pad.
        position = struct.pack(
            "<HhhhhhihhH", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
        tail = struct.pack("<HH", 0, 0) + b"\xDE\xAD\xBE\xEF\xCA\xFE"
        out = decode_event_parameter(9, position + tail, unicode=True)
        assert out["trailing_hex"] == "deadbeefcafe"

    def test_position_has_no_kind_key(self) -> None:
        # Nested position dict drops its "kind" — outer kind is "Create".
        position = struct.pack(
            "<HhhhhhihhH", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
        tail = struct.pack("<HH", 0, 0)
        out = decode_event_parameter(9, position + tail, unicode=True)
        assert "kind" not in out["position"]

    def test_wrong_size(self) -> None:
        with pytest.raises(EventParameterDecodeError, match=">= 26 bytes"):
            decode_event_parameter(9, b"\x00" * 24, unicode=True)


# --- Expression AST tests -----------------------------------------------

# Expression header fields: int16 objectType, int16 num, uint16 size (omitted
# for END marker). Total record size includes all three + body.
END_MARKER = struct.pack("<hh", 0, 0)


def _expr(object_type: int, num: int, body: bytes) -> bytes:
    """Build one Expression: int16 type + int16 num + uint16 size + body.

    size = 6 + len(body).
    """
    return struct.pack("<hhH", object_type, num, 6 + len(body)) + body


def _param_22(comparison: int, expressions: bytes) -> bytes:
    """Build a code-22 (ExpressionParameter) body: int16 comparison + expr* + END."""
    return struct.pack("<h", comparison) + expressions + END_MARKER


class TestExpressionLeaves:
    """Every expression body-kind on objectType=-1 leaf."""

    def test_long_literal(self) -> None:
        body = struct.pack("<i", 42)
        data = _param_22(0, _expr(-1, 0, body))
        out = decode_event_parameter(22, data, unicode=True)
        assert out["kind"] == "ExpressionParameter"
        assert out["comparison"] == 0
        assert out["trailing_hex"] == ""
        [literal, end] = out["expressions"]
        assert literal["object_type"] == -1
        assert literal["num"] == 0
        assert literal["body_kind"] == "Long"
        assert literal["body"]["value"] == 42
        assert literal["body"]["trailing_hex"] == ""
        assert end["end"] is True

    def test_string_literal_unicode(self) -> None:
        body = _u16le("hello")
        data = _param_22(0, _expr(-1, 3, body))
        out = decode_event_parameter(22, data, unicode=True)
        [literal, _end] = out["expressions"]
        assert literal["body_kind"] == "String"
        assert literal["body"]["value"] == "hello"
        assert literal["body"]["trailing_hex"] == ""

    def test_string_trailing_tolerated(self) -> None:
        # Anaconda's seek(start+size) forces progression regardless of
        # inner String consumption — our port mirrors that leniency.
        body = _u16le("a") + b"\xAA\xBB"
        data = _param_22(0, _expr(-1, 3, body))
        out = decode_event_parameter(22, data, unicode=True)
        literal = out["expressions"][0]
        assert literal["body"]["value"] == "a"
        assert literal["body"]["trailing_hex"] == "aabb"

    def test_double_literal(self) -> None:
        body = struct.pack("<df", 3.14, 3.14)
        data = _param_22(0, _expr(-1, 23, body))
        out = decode_event_parameter(22, data, unicode=True)
        literal = out["expressions"][0]
        assert literal["body_kind"] == "Double"
        assert literal["body"]["value"] == pytest.approx(3.14)
        assert literal["body"]["float_value"] == pytest.approx(3.14, rel=1e-6)

    def test_global_value(self) -> None:
        # CTFAK2.0 shape: skip int32 + int32 value = 8 bytes.
        body = b"\x00\x00\x00\x00" + struct.pack("<i", 12)
        data = _param_22(0, _expr(-1, 24, body))
        out = decode_event_parameter(22, data, unicode=True)
        literal = out["expressions"][0]
        assert literal["body_kind"] == "GlobalValue"
        assert literal["body"]["value"] == 12

    def test_global_string(self) -> None:
        body = b"\x00\x00\x00\x00" + struct.pack("<i", 4)
        data = _param_22(0, _expr(-1, 50, body))
        out = decode_event_parameter(22, data, unicode=True)
        literal = out["expressions"][0]
        assert literal["body_kind"] == "GlobalString"
        assert literal["body"]["value"] == 4


class TestExpressionObjectRefs:
    """objectType >= 2 or == -7 expressions carry an (info, infoList) header."""

    def test_alterable_value_extension(self) -> None:
        # num=16 => ExtensionValue body (int16 alterable index)
        header = struct.pack("<Hh", 11, 5)  # object_info=11, list=5
        body = header + struct.pack("<h", 3)  # index 3
        data = _param_22(0, _expr(2, 16, body))
        out = decode_event_parameter(22, data, unicode=True)
        expr = out["expressions"][0]
        assert expr["object_info"] == 11
        assert expr["object_info_list"] == 5
        assert expr["body_kind"] == "ExtensionValue"
        assert expr["body"]["value"] == 3

    def test_alterable_string_extension(self) -> None:
        header = struct.pack("<Hh", 0, -1)
        body = header + struct.pack("<h", 7)
        data = _param_22(0, _expr(2, 19, body))
        out = decode_event_parameter(22, data, unicode=True)
        expr = out["expressions"][0]
        assert expr["body_kind"] == "ExtensionString"
        assert expr["body"]["value"] == 7

    def test_object_ref_no_body(self) -> None:
        # num that has no extension loader — just the (info, infoList) header.
        header = struct.pack("<Hh", 9, 2)
        data = _param_22(0, _expr(2, 1, header))  # num=1 (YPosition)
        out = decode_event_parameter(22, data, unicode=True)
        expr = out["expressions"][0]
        assert expr["object_info"] == 9
        assert expr["object_info_list"] == 2
        assert expr["body_kind"] is None
        assert expr.get("trailing_hex", "") == ""

    def test_player_object_ref(self) -> None:
        # objectType == -7 (Player) uses the same object-ref header path.
        header = struct.pack("<Hh", 0, 0)
        data = _param_22(0, _expr(-7, 0, header))
        out = decode_event_parameter(22, data, unicode=True)
        expr = out["expressions"][0]
        assert expr["object_type"] == -7
        assert expr["object_info"] == 0
        assert expr["object_info_list"] == 0


class TestExpressionNoBody:
    """objectType in {0, -2, -3, -4, -5, -6} and not -1 => no body."""

    @pytest.mark.parametrize("object_type", [0, -2, -3, -4, -5, -6])
    def test_no_body(self, object_type: int) -> None:
        data = _param_22(0, _expr(object_type, 1, b""))
        out = decode_event_parameter(22, data, unicode=True)
        expr = out["expressions"][0]
        assert expr["object_type"] == object_type
        assert expr["body_kind"] is None
        assert expr["size"] == 6

    def test_operator_plus(self) -> None:
        # objectType=0, num=2 is EXPRESSION_SYSTEM_NAMES[0][2] = "Plus"
        # (name resolution is not this module's job, but verify the shape).
        data = _param_22(0, _expr(0, 2, b""))
        out = decode_event_parameter(22, data, unicode=True)
        expr = out["expressions"][0]
        assert expr["object_type"] == 0
        assert expr["num"] == 2


class TestExpressionSequence:
    """Multi-expression sequences: Plus-tree, nested operations."""

    def test_two_leaves_plus_end(self) -> None:
        # Equivalent to expression: 1 + 2
        body = _expr(-1, 0, struct.pack("<i", 1))  # Long 1
        body += _expr(0, 2, b"")  # Plus operator
        body += _expr(-1, 0, struct.pack("<i", 2))  # Long 2
        data = _param_22(0, body)
        out = decode_event_parameter(22, data, unicode=True)
        exprs = out["expressions"]
        assert len(exprs) == 4  # 3 real + END
        assert exprs[0]["body"]["value"] == 1
        assert exprs[1]["num"] == 2  # Plus
        assert exprs[2]["body"]["value"] == 2
        assert exprs[3]["end"] is True

    def test_comparison_preserved(self) -> None:
        # comparison=4 is typically EQUAL in Clickteam.
        data = _param_22(4, _expr(-1, 0, struct.pack("<i", 99)))
        out = decode_event_parameter(22, data, unicode=True)
        assert out["comparison"] == 4


class TestExpressionFailures:
    """Loud-failure paths on malformed Expression bodies."""

    def test_missing_end_marker(self) -> None:
        # Long literal with no END.
        body = _expr(-1, 0, struct.pack("<i", 1))
        data = struct.pack("<h", 0) + body  # no END marker
        with pytest.raises(EventParameterDecodeError):
            decode_event_parameter(22, data, unicode=True)

    def test_size_too_small(self) -> None:
        # Manually build an expression with size=4 (illegal, < 6 minimum).
        body = struct.pack("<hhH", -1, 0, 4)
        data = struct.pack("<h", 0) + body + END_MARKER
        with pytest.raises(EventParameterDecodeError, match="below 6-byte"):
            decode_event_parameter(22, data, unicode=True)

    def test_size_overruns_buffer(self) -> None:
        # size=100 but only a few bytes left.
        body = struct.pack("<hhH", -1, 0, 100)
        data = struct.pack("<h", 0) + body + END_MARKER
        with pytest.raises(EventParameterDecodeError, match="claims size="):
            decode_event_parameter(22, data, unicode=True)

    def test_trailing_tolerated_after_end(self) -> None:
        # Outer Parameter.Read seeks to currentPosition + size, so
        # trailing bytes past the END marker are pad, not drift.
        data = (
            _param_22(0, _expr(-1, 0, struct.pack("<i", 1))) + b"\xAA\xBB"
        )
        out = decode_event_parameter(22, data, unicode=True)
        assert out["trailing_hex"] == "aabb"

    def test_header_truncated(self) -> None:
        # Only 1 byte in the comparison header.
        with pytest.raises(EventParameterDecodeError):
            decode_event_parameter(22, b"\x00", unicode=True)

    def test_missing_size_field(self) -> None:
        # objectType=-1, num=0 (not END) but no room for size field.
        data = struct.pack("<h", 0) + struct.pack("<hh", -1, 0) + END_MARKER
        with pytest.raises(EventParameterDecodeError):
            decode_event_parameter(22, data, unicode=True)

    @pytest.mark.parametrize("code", [22, 23, 27, 45])
    def test_all_four_expression_codes(self, code: int) -> None:
        """23, 27, 45 dispatch through the same ExpressionParameter path."""
        data = _param_22(0, _expr(-1, 0, struct.pack("<i", 1)))
        out = decode_event_parameter(code, data, unicode=True)
        assert out["kind"] == "ExpressionParameter"
        assert out["code"] == code


# --- Dispatch tests -----------------------------------------------------


class TestDispatch:
    def test_all_15_fnaf1_codes_reachable(self) -> None:
        """Every code in FNAF1_PARAMETER_CODES must have a decoder."""
        covered = {
            1, 2, 6, 9, 10, 14, 16, 22, 23, 25, 26, 27, 32, 45, 50,
        }
        assert FNAF1_PARAMETER_CODES == covered

    def test_unknown_code_rejected(self) -> None:
        with pytest.raises(EventParameterDecodeError, match="outside the FNAF 1"):
            decode_event_parameter(99, b"", unicode=True)

    def test_anaconda_unused_code_rejected(self) -> None:
        # Code 3 is a known Anaconda Short loader but not in FNAF 1 — loud.
        with pytest.raises(EventParameterDecodeError, match="outside the FNAF 1"):
            decode_event_parameter(3, b"\x00\x00", unicode=True)

    def test_code_stamped_on_result(self) -> None:
        out = decode_event_parameter(10, b"\x00\x00", unicode=True)
        assert out["code"] == 10


# --- FNAF 1 smoke test --------------------------------------------------


@pytest.mark.skipif(
    not FNAF_EXE.exists(),
    reason=f"FNAF 1 exe not checked in at {FNAF_EXE}",
)
class TestFnaf1Smoke:
    """Walk every EventParameter in every FNAF 1 frame through the decoder.

    Zero-tolerance antibody: if any parameter fails to decode, probe #4.13
    is unfinished. The test also pins the per-code population so future
    drift (e.g. Clickteam build change) is caught loudly.
    """

    @pytest.fixture(scope="class")
    def fnaf1(self) -> tuple[list, bool]:
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
        return frames, result.header.unicode

    def test_every_parameter_decodes(self, fnaf1) -> None:
        frames, unicode = fnaf1
        total = 0
        per_code: dict[int, int] = {}
        for frame in frames:
            if frame.events is None:
                continue
            for group in frame.events.event_groups:
                for cond in group.conditions:
                    for param in cond.parameters:
                        decode_event_parameter(
                            param.code, param.data, unicode=unicode
                        )
                        total += 1
                        per_code[param.code] = per_code.get(param.code, 0) + 1
                for act in group.actions:
                    for param in act.parameters:
                        decode_event_parameter(
                            param.code, param.data, unicode=unicode
                        )
                        total += 1
                        per_code[param.code] = per_code.get(param.code, 0) + 1

        # Sanity: there must be parameters, and every seen code must be
        # in the FNAF 1 set.
        assert total > 0
        assert set(per_code) == FNAF1_PARAMETER_CODES

    def test_expression_codes_present_and_parsed(self, fnaf1) -> None:
        """At least one ExpressionParameter body is non-trivial (length >= 2
        expressions after the first one, excluding the END marker). This
        pins that we're not accidentally treating all Expression bodies as
        empty comparisons."""
        frames, unicode = fnaf1
        found_nontrivial = False
        for frame in frames:
            if frame.events is None:
                continue
            for group in frame.events.event_groups:
                for holder in list(group.conditions) + list(group.actions):
                    for param in holder.parameters:
                        if param.code in {22, 23, 27, 45}:
                            decoded = decode_event_parameter(
                                param.code, param.data, unicode=unicode
                            )
                            non_end = [
                                e for e in decoded["expressions"] if not e["end"]
                            ]
                            if len(non_end) >= 2:
                                found_nontrivial = True
                                break
                    if found_nontrivial:
                        break
                if found_nontrivial:
                    break
            if found_nontrivial:
                break
        assert found_nontrivial, (
            "no non-trivial ExpressionParameter found — either FNAF 1 really "
            "only has single-leaf expressions (unlikely) or the walker is "
            "short-circuiting"
        )
