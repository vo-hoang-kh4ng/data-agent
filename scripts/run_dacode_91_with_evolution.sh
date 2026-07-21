#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/khoa/agent_benchmark/data-agent}"
PYTHON_BIN="${PYTHON_BIN:-/home/khoa/agent_benchmark/venv/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/config/evolution_91.paper.json}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/evolution_runs/$RUN_ID}"
RESUME="${RESUME:-0}"

if [[ -e "$OUTPUT_ROOT" && "$RESUME" != "1" ]]; then
  echo "Refusing to overwrite existing run directory: $OUTPUT_ROOT" >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(--config "$CONFIG_FILE" --output-root "$OUTPUT_ROOT")
if [[ "$RESUME" == "1" ]]; then
  ARGS+=(--resume)
fi
"$PYTHON_BIN" -m dgm_agent.evolution.cli "${ARGS[@]}"

echo "Completed evolution + frozen 91-task evaluation."
echo "Summary: $OUTPUT_ROOT/run_summary.json"
