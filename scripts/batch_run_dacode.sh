#!/usr/bin/env bash
# Batched DA-Code runner: invokes dacode_runner in --max_tasks chunks and loops until
# all tasks are finished (resume-skip chains batches). Each invocation reloads E5 and
# exits after N tasks, releasing RAM — avoids the ~47-task OOM on this 16GB machine.
#
# Usage: bash scripts/batch_run_dacode.sh <manifest> <sandbox_dir> <max_tasks> [log_file]
set -u
MANIFEST="${1:-data/dacode_unified_manifest.jsonl}"
SANDBOX="${2:-data/dacode_sandbox_baseline}"
MAXT="${3:-20}"
LOG="${4:-data/batch_run.log}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
VENV_PYTHON="${PYTHON_BIN:-python3}"

# Disable E5+KMeans semantic clustering — use prefix fallback only.
# Per-task isolated lake has 1-5 files so KMeans adds no value and wastes tokens.
export DACODE_USE_SEMANTIC=0

TOTAL=$("$VENV_PYTHON" -c "import json;print(sum(1 for l in open('$MANIFEST',encoding='utf-8') if l.strip()))")
echo "[$(date +%H:%M:%S)] batch_run START  manifest=$MANIFEST  total=$TOTAL  sandbox=$SANDBOX  max_tasks=$MAXT  kmeans=OFF" | tee "$LOG"

ITER=0
while true; do
    ITER=$((ITER+1))
    FINISHED=$(ls "$SANDBOX"/*/dabench/result.json 2>/dev/null | xargs -I{} "$VENV_PYTHON" -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));print(1 if d.get('finished') else 0)" {} 2>/dev/null | awk '{s+=$1} END{print s+0}')
    FINISHED=${FINISHED:-0}
    echo "[$(date +%H:%M:%S)] iter=$ITER  finished=$FINISHED/$TOTAL" | tee -a "$LOG"
    if [ "$FINISHED" -ge "$TOTAL" ]; then
        echo "[$(date +%H:%M:%S)] ALL FINISHED ($FINISHED/$TOTAL). Done." | tee -a "$LOG"
        break
    fi
    # Run one batch; capture final done count to detect stalls
    BEFORE=$FINISHED
    "$VENV_PYTHON" -m dgm_agent.dacode_runner --manifest "$MANIFEST" --sandbox_dir "$SANDBOX" --max_tasks "$MAXT" >> "$LOG" 2>&1
    RC=$?
    FINISHED=$(ls "$SANDBOX"/*/dabench/result.json 2>/dev/null | xargs -I{} "$VENV_PYTHON" -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));print(1 if d.get('finished') else 0)" {} 2>/dev/null | awk '{s+=$1} END{print s+0}')
    FINISHED=${FINISHED:-0}
    echo "[$(date +%H:%M:%S)] iter=$ITER batch exit=$RC  finished=$FINISHED/$TOTAL  (+$((FINISHED-BEFORE)) this batch)" | tee -a "$LOG"
    if [ "$FINISHED" -le "$BEFORE" ]; then
        echo "[$(date +%H:%M:%S)] STALL: no new finished tasks ($FINISHED<=$BEFORE). Stopping to avoid infinite loop." | tee -a "$LOG"
        break
    fi
    # Safety: count attempted (result.json exists, finished or not) to know if we've covered all
    ATTEMPTED=$(ls "$SANDBOX"/*/dabench/result.json 2>/dev/null | wc -l)
    if [ "$ATTEMPTED" -ge "$TOTAL" ] && [ "$RC" -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] All $TOTAL attempted. Stopping." | tee -a "$LOG"
        break
    fi
done
echo "[$(date +%H:%M:%S)] batch_run END  finished=$FINISHED/$TOTAL" | tee -a "$LOG"
