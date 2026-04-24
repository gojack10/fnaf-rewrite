#!/usr/bin/env bash
# Overnight Step 5 runner — FNAF invariants pipeline end-to-end.
#
# Sequenced:
#   1. SMOKE    — extract.py --max-chunks 1 (catches end-to-end wiring)
#   2. EXTRACT  — extract.py full corpus   (serial, reliability > speed)
#   3. CITATION — scripts/reverify.py      (citation-checker gate)
#   4. COVERAGE — coverage.py              (breadth gate on accepted)
#   5. LITERAL  — literal_gate.py          (literal-purity gate on accepted)
#
# Tuning:
#   --target-tokens 15000 + --max-workers 1  → ~38 serial chunks, ~5.5-6 hours
#   Literal Gate Layer 2:  --max-workers 5   → conservative judge parallelism
#
# Each phase gates the next: if SMOKE fails, full run aborts.
# Partial state survives individual phase failures; logs persist in
# out/step5-log/ for morning post-mortem.

set -u
set -o pipefail

REPO_ROOT="/home/jack/projects/fnaf-rewrite/parser"
LOG_DIR="$REPO_ROOT/out/step5-log"
RUN_LOG="$LOG_DIR/run.log"

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

# Load API key from disk (strip any trailing whitespace).
if [ ! -f "$HOME/.openrouter_key" ]; then
  echo "ABORT: $HOME/.openrouter_key missing" | tee -a "$RUN_LOG"
  exit 3
fi
export OPENROUTER_API_KEY
OPENROUTER_API_KEY=$(tr -d '[:space:]' < "$HOME/.openrouter_key")
if [ -z "$OPENROUTER_API_KEY" ]; then
  echo "ABORT: OPENROUTER_API_KEY empty after load" | tee -a "$RUN_LOG"
  exit 3
fi

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$RUN_LOG"
}

log "=== STEP 5 START ==="
log "repo: $REPO_ROOT"
log "log dir: $LOG_DIR"
log "key loaded (sha256 first 20): $(printf '%s' "$OPENROUTER_API_KEY" | sha256sum | cut -c1-20)"

# -----------------------------------------------------------------------
# Phase 1: SMOKE — single chunk end-to-end, proves wiring before full burn.
# -----------------------------------------------------------------------
log "--- PHASE 1: SMOKE ---"
uv run python -m fnaf_parser.invariants.extract \
  --out out/invariants-smoke \
  --max-chunks 1 \
  --max-workers 1 \
  --target-tokens 15000 \
  2>&1 | tee "$LOG_DIR/smoke.log"
SMOKE_EXIT=${PIPESTATUS[0]}
log "SMOKE exit=$SMOKE_EXIT"
if [ "$SMOKE_EXIT" -ne 0 ]; then
  log "ABORT: smoke failed; not starting full extract"
  exit 1
fi

# -----------------------------------------------------------------------
# Phase 2: FULL EXTRACT — the long serial grind.
# -----------------------------------------------------------------------
log "--- PHASE 2: FULL EXTRACT ---"
uv run python -m fnaf_parser.invariants.extract \
  --out out/invariants \
  --max-workers 1 \
  --target-tokens 15000 \
  2>&1 | tee "$LOG_DIR/extract.log"
EXTRACT_EXIT=${PIPESTATUS[0]}
log "EXTRACT exit=$EXTRACT_EXIT"
# Non-zero here means some chunks whole-failed; downstream gates still
# meaningful on the surviving records. Continue.

# -----------------------------------------------------------------------
# Phase 3: CITATION CHECKER — via scripts/reverify.py.
# reverify expects raw_records.jsonl in --dir; extract writes extracted.jsonl.
# Copy (not move) so the extract.py artefact stays intact for inspection.
# -----------------------------------------------------------------------
log "--- PHASE 3: CITATION CHECKER ---"
if [ ! -f "out/invariants/extracted.jsonl" ]; then
  log "SKIP citation: out/invariants/extracted.jsonl missing"
else
  cp out/invariants/extracted.jsonl out/invariants/raw_records.jsonl
  uv run python scripts/reverify.py \
    --dir out/invariants \
    --combined out/algorithm/combined.json \
    2>&1 | tee "$LOG_DIR/citation.log"
  CITATION_EXIT=${PIPESTATUS[0]}
  log "CITATION exit=$CITATION_EXIT"
fi

# -----------------------------------------------------------------------
# Phase 4: COVERAGE ANTIBODY — on accepted.jsonl (post-citation-check).
# -----------------------------------------------------------------------
log "--- PHASE 4: COVERAGE ---"
if [ ! -f "out/invariants/accepted.jsonl" ]; then
  log "SKIP coverage: accepted.jsonl missing"
else
  uv run python -m fnaf_parser.invariants.coverage \
    --combined out/algorithm/combined.jsonl \
    --input out/invariants/accepted.jsonl \
    --report out/invariants/coverage_report.json \
    2>&1 | tee "$LOG_DIR/coverage.log"
  COVERAGE_EXIT=${PIPESTATUS[0]}
  log "COVERAGE exit=$COVERAGE_EXIT"
fi

# -----------------------------------------------------------------------
# Phase 5: LITERAL GATE — Layer 1 free, Layer 2 judge with light parallelism.
# -----------------------------------------------------------------------
log "--- PHASE 5: LITERAL GATE ---"
if [ ! -f "out/invariants/accepted.jsonl" ]; then
  log "SKIP literal_gate: accepted.jsonl missing"
else
  uv run python -m fnaf_parser.invariants.literal_gate \
    --extracted out/invariants/accepted.jsonl \
    --combined out/algorithm/combined.jsonl \
    --out out/invariants \
    --max-workers 5 \
    2>&1 | tee "$LOG_DIR/literal_gate.log"
  LITERAL_EXIT=${PIPESTATUS[0]}
  log "LITERAL exit=$LITERAL_EXIT"
fi

log "=== STEP 5 DONE ==="
