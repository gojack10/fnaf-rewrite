"""OpenRouter-backed DSPy.RLM configuration.

Single source of truth for:

- which model the line cook runs on (OpenRouter model slug),
- which model the head chef runs on (usually the same, cheaper-tier
  reserved for future split if router-style meta-prompting gets added),
- where the API key comes from (`OPENROUTER_API_KEY` env var),
- how a missing key surfaces to the caller (a clear
  `OpenRouterConfigError`, not a DSPy internal traceback).

Why OpenRouter
--------------

The pilot pipeline is cost-sensitive — Slice C alone is ~800 rows,
Slices A + B scale the dispatch count further. OpenRouter exposes
dozens of models behind one billing surface, which lets the owner swap
the model in one env-var bump without touching the pipeline code. The
default slugs here point at a mid-tier analytic model (Claude Sonnet
class) that empirically handles small-context structured extraction
cleanly.

Environment contract
--------------------

- `OPENROUTER_API_KEY` — the API key. The ONLY required env var.
  Missing → `OpenRouterConfigError` with an explicit how-to-fix
  message; never a silent DSPy 401.
- `FNAF_LINE_COOK_MODEL` — optional override for the line-cook model
  slug. Defaults to `_DEFAULT_LINE_COOK_MODEL`.
- `FNAF_HEAD_CHEF_MODEL` — optional override for the head-chef model
  slug. Defaults to the line-cook model (single-model mode).

Testing posture
---------------

Every function here is synchronous, pure, and touches the environment
via a `dict` parameter so tests can inject a fake env without
monkey-patching `os.environ`. No network I/O at import time. The LM
handle is built lazily only when `build_lm()` is called.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

# --- Defaults (edit in one place to change the pilot model) -------------

# Default OpenRouter slug for the line cook. Chosen for:
#   - solid structured-output compliance,
#   - moderate context window (64k+) — comfortably larger than any
#     single-slice ticket batch we'll ship in V1,
#   - priced at roughly $0.13/M input + $0.40/M output at the time the
#     pilot was planned — a full Slice C run (~800 rows) fits in a
#     single-digit-dollar budget at our expected token counts.
_DEFAULT_LINE_COOK_MODEL = "openrouter/anthropic/claude-sonnet-4.5"

# OpenRouter's OpenAI-compatible chat-completions endpoint. Exposed as a
# constant so tests can swap it for a local mock server.
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# Env var names. Centralised so the CLI help text and error messages
# can reference them without drift.
ENV_API_KEY = "OPENROUTER_API_KEY"
ENV_LINE_COOK_MODEL = "FNAF_LINE_COOK_MODEL"
ENV_HEAD_CHEF_MODEL = "FNAF_HEAD_CHEF_MODEL"


# --- Errors -------------------------------------------------------------


class OpenRouterConfigError(RuntimeError):
    """Raised when the OpenRouter env contract is not satisfied.

    The message always includes:

    - which env var is missing or malformed,
    - how to set it (`export OPENROUTER_API_KEY=...`),
    - a link back to the project's invariant-extraction docs so the
      caller can recover without reading the source.

    Preferred over re-raising `KeyError` / `ValueError` so the CLI
    layer can catch *just this one exception type* and map it to a
    clean exit code + user-facing message.
    """


# --- Config object ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpenRouterConfig:
    """Resolved config for a pipeline run.

    Frozen + slotted so accidental mutation can't leak a stale model
    slug across a multi-slice run. Built once at `run()` entry, passed
    down to every sub-call.
    """

    api_key: str
    line_cook_model: str
    head_chef_model: str
    api_base: str = OPENROUTER_API_BASE

    def redacted(self) -> dict[str, str]:
        """Return a dict safe to log — api_key is masked to its last 4.

        Logs / error messages / CLI --dry-run banners should print
        `cfg.redacted()` instead of the raw dataclass repr so the key
        never leaks to stdout or the parser's `out/` artefacts.
        """
        masked = (
            "****" + self.api_key[-4:] if len(self.api_key) >= 4 else "****"
        )
        return {
            "api_key": masked,
            "line_cook_model": self.line_cook_model,
            "head_chef_model": self.head_chef_model,
            "api_base": self.api_base,
        }


# --- Loader -------------------------------------------------------------


def load_config(env: Mapping[str, str] | None = None) -> OpenRouterConfig:
    """Resolve `OpenRouterConfig` from the environment.

    Pass `env=` to inject a fake environment in tests; defaults to
    `os.environ`. Raises `OpenRouterConfigError` with a clear remediation
    message when `OPENROUTER_API_KEY` is missing or empty.

    Model slugs fall back to the pilot defaults in this module when the
    corresponding override env vars are absent. The same default slug is
    used for both line cook and head chef — splitting them is a future
    optimisation, not a V1 requirement.
    """
    effective_env: Mapping[str, str] = env if env is not None else os.environ

    api_key = effective_env.get(ENV_API_KEY, "").strip()
    if not api_key:
        raise OpenRouterConfigError(
            f"{ENV_API_KEY} is not set. Export it before running the "
            f"invariant-extraction pipeline:\n"
            f"  export {ENV_API_KEY}=sk-or-v1-...\n"
            f"Get a key from https://openrouter.ai/keys — the key is "
            f"only needed when actually running the pipeline; every "
            f"other part of fnaf-parser is usable without it."
        )

    line_cook = (
        effective_env.get(ENV_LINE_COOK_MODEL, "").strip()
        or _DEFAULT_LINE_COOK_MODEL
    )
    head_chef = (
        effective_env.get(ENV_HEAD_CHEF_MODEL, "").strip() or line_cook
    )

    return OpenRouterConfig(
        api_key=api_key,
        line_cook_model=line_cook,
        head_chef_model=head_chef,
    )


# --- LM handle builder --------------------------------------------------


def build_lm(cfg: OpenRouterConfig):  # type: ignore[no-untyped-def]
    """Return a `dspy.LM` instance wired to OpenRouter.

    Import of `dspy` is intentionally lazy — `load_config` stays usable
    and fully testable even in environments where DSPy isn't installed
    (e.g. a Rust-side CI job validating only the citation checker).

    The returned LM has `max_tokens` set generously (line cook needs
    room to emit a full `list[InvariantRecord]` for larger ticket
    buckets) and `temperature=0` because invariant extraction is a
    grounded claim-plus-citation task — creativity is a liability, not
    a feature.
    """
    import dspy  # noqa: PLC0415  (lazy; keeps dspy off import-time path)

    return dspy.LM(
        cfg.line_cook_model,
        api_key=cfg.api_key,
        api_base=cfg.api_base,
        temperature=0.0,
        max_tokens=4096,
    )
