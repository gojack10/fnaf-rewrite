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
the model in one env-var bump without touching the pipeline code.

The defaults now split line cook from head chef: a smaller, cheaper
instruction-tuned model (Gemma 4 31B) handles the per-ticket extraction
grind, while a larger reasoning-class model (GLM 5.1) is reserved for
the head-chef seat when router-style meta-prompting lands. Until
`build_lm` grows a sibling head-chef builder, only the line-cook slug
is actually instantiated — the head-chef default is forward-looking.

Environment contract
--------------------

- `OPENROUTER_API_KEY` — the API key. The ONLY required env var.
  Missing → `OpenRouterConfigError` with an explicit how-to-fix
  message; never a silent DSPy 401.
- `FNAF_LINE_COOK_MODEL` — optional override for the line-cook model
  slug. Defaults to `_DEFAULT_LINE_COOK_MODEL`.
- `FNAF_HEAD_CHEF_MODEL` — optional override for the head-chef model
  slug. Defaults to `_DEFAULT_HEAD_CHEF_MODEL` when line cook is also
  unset; falls back to the line-cook slug when the user has explicitly
  pinned line cook but not head chef (don't surprise a pinning user
  with a second, unrelated model).

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
#   - native structured-output support (OpenRouter catalog lists
#     `response_format` + `structured_outputs` in supported_parameters),
#   - 262k-token context window — comfortable headroom over any single
#     Slice A/B bucket ticket,
#   - small-model economics: Gemma 4 31B IT sits at the cheap end of
#     OpenRouter's instruction-tuned tier, keeping a full Slice C run
#     (~800 rows) well inside a single-digit-dollar budget.
# The `openrouter/` prefix is the LiteLLM/DSPy provider routing scheme;
# the OpenRouter catalog slug itself is `google/gemma-4-31b-it`.
_DEFAULT_LINE_COOK_MODEL = "openrouter/google/gemma-4-31b-it"

# Default OpenRouter slug for the head chef. Reserved for the future
# router-style meta-prompting seat — NOT currently instantiated by
# `build_lm`, which still builds a single LM from the line-cook slug.
# GLM 5.1 was chosen because:
#   - it's a larger reasoning-class model — right fit for the head
#     chef's eventual job of routing tickets into per-slice strategies,
#   - 202k context + 65k max output cap per OpenRouter — plenty for
#     reasoning traces and long batch emissions,
#   - structured-output support is declared in the catalog.
# Wire a sibling `build_head_chef_lm()` here when the split actually
# matters; until then this constant is documentation of intent.
_DEFAULT_HEAD_CHEF_MODEL = "openrouter/z-ai/glm-5.1"

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
    corresponding override env vars are absent. Line cook and head chef
    now have *different* pilot defaults (smaller worker + larger
    reasoner), but we preserve one ergonomic: if the user has pinned
    only `FNAF_LINE_COOK_MODEL`, head chef copies that override rather
    than quietly reaching for `_DEFAULT_HEAD_CHEF_MODEL` — a user who
    overrides line cook is almost always trying to isolate the pipeline
    on a single model, and surprising them with a second model defeats
    the purpose of the override.
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

    line_cook_override = effective_env.get(ENV_LINE_COOK_MODEL, "").strip()
    head_chef_override = effective_env.get(ENV_HEAD_CHEF_MODEL, "").strip()

    line_cook = line_cook_override or _DEFAULT_LINE_COOK_MODEL

    if head_chef_override:
        head_chef = head_chef_override
    elif line_cook_override:
        # User pinned line cook explicitly → don't conjure an unrelated
        # head chef. Copy the override.
        head_chef = line_cook_override
    else:
        # Pure defaults path — use the split pilot defaults.
        head_chef = _DEFAULT_HEAD_CHEF_MODEL

    return OpenRouterConfig(
        api_key=api_key,
        line_cook_model=line_cook,
        head_chef_model=head_chef,
    )


# --- LM handle builder --------------------------------------------------


def _build_lm_for_model(cfg: OpenRouterConfig, model_slug: str):  # type: ignore[no-untyped-def]
    """Shared LM construction — one place to tune temp / max_tokens.

    Import of `dspy` is intentionally lazy — `load_config` stays usable
    and fully testable even in environments where DSPy isn't installed
    (e.g. a Rust-side CI job validating only the citation checker).

    LM parameters — grounded in measured ticket sizes, not guessed:

    `temperature=0.0`
        Invariant extraction is a grounded claim-plus-citation task.
        Slice C (pilot) wants `pseudo_code` to be a **verbatim** copy
        of the cited `expr_str`; Citation Checker's `expr_str_match`
        verification rewards literal string match, so any creativity
        (paraphrasing `20` -> `20.0`, renaming identifiers) just moves
        records from accepted to quarantine. DSPy's JSON adapter +
        `InvariantRecord` Pydantic enforcement damps Gemma's historical
        temp-0 repetition tendency on long free-form output — the
        schema forces structure. If the first pilot surfaces repetition
        artefacts in `quarantine.jsonl`, bump Gemma to 0.1 before
        reaching for anything higher.

        The head chef inherits the same `0.0` setting: review is a
        filtering task, not a creative one — same rationale applies.

    `max_tokens=16384`
        Sized for the real worst case, measured against
        `out/algorithm/combined.jsonl`:

        - Slice C (1484 row-per-ticket calls): 250-850 tokens/call.
          4k would be fine, but...
        - Slice A p90 bucket (182 rows): 2.5k-7.5k tokens/call.
        - Slice A max bucket (CompareCounter, 471 rows): 5k-12.5k
          tokens/call. A 4k cap silently truncates the emission into
          malformed JSON, triggering a DSPy re-ask (doubles cost).
        - Slice B max (Speaker/String, 234 rows): 3.75k-10k tokens.
        - Future RLM / ChainOfThought upgrade: +300-2000 reasoning
          tokens per call; true self-reflection multiplies by the
          iteration count.

        16384 absorbs all of that with zero cost when unused
        (OpenRouter bills emission, not the cap). Gemma declares no
        output cap (262k context); GLM 5.1 supports 65,535. 16k is far
        below both provider limits but caps runaway-loop damage at a
        bounded worst-case bill — the rationale for not going higher.

        The head chef sees the same cap because its output (filtered
        `list[InvariantRecord]`) is bounded above by the line cook's
        output — it can only shrink the set, never grow it.

    `timeout=180`
        Hard ceiling (seconds) on any single HTTP call to OpenRouter.
        Passed through to litellm / httpx. Without this, httpx defaults
        to unbounded read, and a single hung streaming response from
        a slow reasoning-class model (observed in the wild with
        GLM 5.1 on the head-chef seat) stalls the entire pipeline —
        no retry, no error, just a silent `do_sys_poll` forever.

        180s is generous enough that productive-but-slow responses
        (15k reasoning + emission tokens on GLM 5.1) still complete,
        but bounded enough that hangs cost us three minutes, not
        hours. Litellm raises `Timeout` on expiry; DSPy's LM wrapper
        retries per its own `num_retries` default, so a single
        stuck response doesn't kill the whole run.
    """
    import dspy  # noqa: PLC0415  (lazy; keeps dspy off import-time path)

    return dspy.LM(
        model_slug,
        api_key=cfg.api_key,
        api_base=cfg.api_base,
        temperature=0.0,
        max_tokens=16384,
        timeout=180,
    )


def build_line_cook_lm(cfg: OpenRouterConfig):  # type: ignore[no-untyped-def]
    """Build the line-cook LM — per-ticket candidate extraction.

    Bound to `cfg.line_cook_model` (Gemma 4 31B by default). Invoked
    once per ticket during the line-cook phase of the pipeline.
    """
    return _build_lm_for_model(cfg, cfg.line_cook_model)


def build_head_chef_lm(cfg: OpenRouterConfig):  # type: ignore[no-untyped-def]
    """Build the head-chef LM — per-ticket 1-depth review.

    Bound to `cfg.head_chef_model` (GLM 5.1 by default). Invoked at
    most once per ticket — never in a loop. The head chef filters the
    line cook's candidates against the original ticket; 1-depth is
    intentional (see `review_signature.ReviewInvariants` docstring for
    why recursion is rejected).
    """
    return _build_lm_for_model(cfg, cfg.head_chef_model)


# Backward-compat alias. Pre-split callers asked for a single
# `build_lm(cfg)` and got a line-cook LM; preserve that shape so
# existing tests / downstream imports don't break. New code should
# pick `build_line_cook_lm` or `build_head_chef_lm` explicitly.
build_lm = build_line_cook_lm
