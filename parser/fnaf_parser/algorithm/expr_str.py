"""Flat pseudo-code renderer for resolved Clickteam Expression AST streams.

An Expression parameter (codes 22/23/27/45) carries a flat RPN token stream
terminated by the `(object_type=0, num=0)` END marker. The canonical record
is the `ast` list itself (fully structured, every body resolved); `expr_str`
is a *projection* meant to be cheap for LLMs to read.

Design choices:

- The renderer is **total**: no input shape raises. If the stack underflows
  on an operator (malformed stream), we push a bracketed placeholder
  (``<?OP?>``) and keep going — the canonical `ast` is always there as
  ground truth.
- Operator arities are pinned to Anaconda's `EXPRESSION_SYSTEM_NAMES[0]`.
  All nine arithmetic/logical ops (Plus/Minus/Multiply/Divide/Modulus/Power/
  AND/OR/XOR) are binary; the `End` entry is the stream terminator and is
  skipped during rendering.
- Literals (Long/String/Double) and globals (GlobalValue/GlobalString) are
  rendered as their body `value`, with a stable prefix for globals.
- Object-referencing expressions (object_type ≥ 2 or == -7) render as
  ``<object_name>.<num_name>`` when Name Resolver + Output Emission have
  already stamped `object_name`; otherwise fall back to
  ``obj[<handle>].<num_name>``.
- Anything we don't explicitly handle falls through to the resolved
  `num_name` token. The renderer prefers stable output over clever
  rendering — `ast` is what downstream consumers should parse.
"""

from __future__ import annotations

from typing import Any

# Anaconda `EXPRESSION_SYSTEM_NAMES[0]` — operator slot. Mapped here with
# their infix symbols; all nine are binary (arity 2).
_BINARY_OP_SYMBOLS: dict[int, str] = {
    2: "+",
    4: "-",
    6: "*",
    8: "/",
    10: "%",
    12: "^",
    14: "AND",
    16: "OR",
    18: "XOR",
}


def _render_literal_token(expr: dict[str, Any]) -> str | None:
    """Return a pseudo-code token for `object_type == -1` literals/globals.

    Returns None if the expression doesn't carry a renderable literal body
    (e.g. system calls like `Sin`, `Random`, parenthesis markers).
    """
    num = expr["num"]
    body = expr.get("body")
    if body is None:
        return None
    body_kind = body.get("body_kind")
    if body_kind == "Long":
        return str(body["value"])
    if body_kind == "Double":
        # Keep repr stable across platforms — str(float) does the right
        # thing here (no locale, no thousand separators).
        return str(body["value"])
    if body_kind == "String":
        # repr() keeps quoting + escapes deterministic.
        return repr(body["value"])
    if body_kind == "GlobalValue":
        return f"G_{body['value']}"
    if body_kind == "GlobalString":
        return f"GS_{body['value']}"
    # GlobalValueExpression (num=2) / GlobalStringExpression (num=49) share
    # the loader map but come through with body_kind=GlobalValue /
    # GlobalString already — the num-name is a separate concern.
    del num  # silence lint; kept in the signature for future dispatching
    return None


def _render_object_token(expr: dict[str, Any]) -> str:
    """Token for object-referencing expressions (object_type >= 2 or -7)."""
    object_name = expr.get("object_name")
    num_name = expr.get("num_name", f"?{expr['num']}")
    body = expr.get("body")
    # Extension-value/extension-string carry the index as body["value"].
    suffix = ""
    if body is not None:
        body_kind = body.get("body_kind")
        if body_kind in ("ExtensionValue", "ExtensionString"):
            suffix = f"({body['value']})"
    if object_name:
        return f"{object_name}.{num_name}{suffix}"
    # Fall back to the raw handle so the output is still unambiguous even
    # when object-name injection failed / wasn't attempted.
    oi = expr.get("object_info")
    if oi is not None:
        return f"obj[{oi}].{num_name}{suffix}"
    return num_name + suffix


def render_expression_stream(expressions: list[dict[str, Any]]) -> str:
    """Render a resolved Expression RPN stream to a flat pseudo-code string.

    `expressions` is the list produced by Name Resolver after walking an
    `ExpressionParameter.expressions`. Every element carries `object_type`,
    `num`, `num_name`, `body` (optional), and — when Output Emission has
    injected it — `object_name` (for object-referencing expressions).

    The renderer never raises; malformed streams produce placeholder
    tokens so the canonical `ast` remains the single source of truth.
    """
    stack: list[str] = []
    for expr in expressions:
        object_type = expr["object_type"]
        num = expr["num"]

        # END marker — stream terminator.
        if object_type == 0 and num == 0:
            continue

        # Operator slot — binary ops only in the current Anaconda table.
        if object_type == 0:
            symbol = _BINARY_OP_SYMBOLS.get(num)
            if symbol is not None:
                if len(stack) < 2:
                    stack.append(f"<?{symbol}?>")
                    continue
                b = stack.pop()
                a = stack.pop()
                stack.append(f"({a} {symbol} {b})")
                continue
            # Unknown operator slot — use num_name if available.
            stack.append(expr.get("num_name", f"?op{num}"))
            continue

        # Literal / global from the -1 system slot.
        if object_type == -1:
            token = _render_literal_token(expr)
            if token is not None:
                stack.append(token)
                continue
            # Non-literal system slot (Sin/Cos/Random/Parenthesis/...) —
            # fall through to the num_name token verbatim.
            stack.append(expr.get("num_name", f"?sys{num}"))
            continue

        # Object-referencing expressions.
        if object_type >= 2 or object_type == -7:
            stack.append(_render_object_token(expr))
            continue

        # Speaker/Game/Timer/Create/Keyboard system slots (object_type ∈
        # {-2, -3, -4, -5, -6}) — no body, just a labelled token.
        stack.append(expr.get("num_name", f"?{object_type}:{num}"))

    return " ".join(stack)
