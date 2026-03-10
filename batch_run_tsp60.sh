#!/bin/bash
# Tsp60 批量实验（可与 batch_run_tsp30.sh 并行： ./batch_run_tsp30.sh & ./batch_run_tsp60.sh & wait）

LOG_DIR="logs/batch_run_tsp60_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/batch_run.log"
PROGRESS_LOG="$LOG_DIR/progress.log"
ERROR_LOG="$LOG_DIR/errors.log"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

log_info()    { echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG" | tee -a "$ERROR_LOG"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"; }

update_progress() {
    local current=$1 total=$2
    echo "[$current/$total] ($((current * 100 / total))%) - $(date '+%Y-%m-%d %H:%M:%S')" > "$PROGRESS_LOG"
    log_info "Progress: [$current/$total]"
}

run_experiment() {
    local cmd="$1" exp_name="$2"
    local exp_log="$LOG_DIR/exp_$(echo "$exp_name" | tr '/' '_' | tr ':' '_').log"
    log_info "Starting: $exp_name"
    local start_time=$(date +%s)
    if eval "$cmd" > "$exp_log" 2>&1; then
        log_success "Completed: $exp_name (Duration: $(($(date +%s) - start_time))s)"
        COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))
    else
        log_error "Failed: $exp_name"
        FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        echo "Failed command: $cmd" >> "$ERROR_LOG"
        echo "Log: $exp_log" >> "$ERROR_LOG"
    fi
    update_progress $((COMPLETED_EXPERIMENTS + FAILED_EXPERIMENTS)) "$TOTAL_EXPERIMENTS"
}

log_info "=== Batch Tsp60 ==="
log_info "Log directory: $LOG_DIR"

declare -a EXPERIMENTS=(
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=0 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp0"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=0.1 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp0.1"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=0.3 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp0.3"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=0.5 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp0.5"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=0.7 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp0.7"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=0.9 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp0.9"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=1.1 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp1.1"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=1.3 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp1.3"
)

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
log_info "Total experiments: $TOTAL_EXPERIMENTS"
BATCH_START_TIME=$(date +%s)

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r cmd name <<< "$exp"
    run_experiment "$cmd" "$name"
done

BATCH_END_TIME=$(date +%s)
D=$((BATCH_END_TIME - BATCH_START_TIME))
log_info "=== Tsp60 done: $COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS, ${D}s ==="
echo "Tsp60: $COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS, ${D}s" > "$LOG_DIR/summary.txt"
[ $FAILED_EXPERIMENTS -gt 0 ] && exit 1 || exit 0
