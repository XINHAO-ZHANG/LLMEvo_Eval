#!/bin/bash

# =================================================================
# EvoEval 批量实验脚本 (增强版)
# 用途: 晚上批量运行多种任务和模型组合的实验
# 作者: EvoEval Team
# 日期: 2025-07-08
# 新增: 进度日志、错误处理、时间统计
# =================================================================

# 配置
LOG_DIR="logs/batch_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/batch_run.log"
PROGRESS_LOG="$LOG_DIR/progress.log"
ERROR_LOG="$LOG_DIR/errors.log"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 实验计数器
TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG" | tee -a "$ERROR_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"
}

# 进度更新函数
update_progress() {
    local current=$1
    local total=$2
    local percentage=$((current * 100 / total))
    echo "[$current/$total] ($percentage%) - $(date '+%Y-%m-%d %H:%M:%S')" > "$PROGRESS_LOG"
    log_info "Progress: [$current/$total] ($percentage%) experiments completed"
}

# 运行实验函数
run_experiment() {
    local cmd="$1"
    local exp_name="$2"
    
    log_info "Starting experiment: $exp_name"
    log_info "Command: $cmd"
    
    # 为每个实验创建单独的日志文件
    local exp_log="$LOG_DIR/exp_$(echo "$exp_name" | tr '/' '_' | tr ':' '_').log"
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 运行实验
    if eval "$cmd" > "$exp_log" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Completed experiment: $exp_name (Duration: ${duration}s)"
        COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "Failed experiment: $exp_name (Duration: ${duration}s)"
        FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        echo "Failed command: $cmd" >> "$ERROR_LOG"
        echo "Log file: $exp_log" >> "$ERROR_LOG"
        echo "---" >> "$ERROR_LOG"
    fi
    
    # 更新进度
    local total_completed=$((COMPLETED_EXPERIMENTS + FAILED_EXPERIMENTS))
    update_progress "$total_completed" "$TOTAL_EXPERIMENTS"
}

# 开始批量实验
log_info "=== Starting Batch Experiments ==="
log_info "Log directory: $LOG_DIR"
log_info "Main log: $MAIN_LOG"
log_info "Progress log: $PROGRESS_LOG"
log_info "Error log: $ERROR_LOG"

# 预先计算实验总数
log_info "Counting total experiments..."

# 定义所有实验
declare -a EXPERIMENTS=(

    #deepseek-chat-v3-0324
    
    # meta-llama/llama-3.2-1b-instruct
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=tsp tasks.tsp.city_num=30 enable_zero_shot_eval=True|gemma-3n-e2b-it_tsp30_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=tsp tasks.tsp.city_num=30 enable_zero_shot_eval=False|gemma-3n-e2b-it_tsp30_evolve"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=tsp tasks.tsp.city_num=60 enable_zero_shot_eval=True|gemma-3n-e2b-it_tsp60_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=tsp tasks.tsp.city_num=60 enable_zero_shot_eval=False|gemma-3n-e2b-it_tsp60_evolve"
    # #promptopt 任务
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=True|gemma-3n-e2b-it_promptopt_sum_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|gemma-3n-e2b-it_promptopt_sum_evolve"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|gemma-3n-e2b-it_promptopt_sim_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=False|gemma-3n-e2b-it_promptopt_sim_evolve"

    # #oscillator1 任务
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=symboreg_oscillator1 enable_zero_shot_eval=True|gemma-3n-e2b-it_oscillator1_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=symboreg_oscillator1 enable_zero_shot_eval=False|gemma-3n-e2b-it_oscillator1_evolve"

    # #oscillator2 任务
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=symboreg_oscillator2 enable_zero_shot_eval=True|gemma-3n-e2b-it_oscillator2_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=symboreg_oscillator2 enable_zero_shot_eval=False|gemma-3n-e2b-it_oscillator2_evolve"

    # #bin_packing 任务
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=bin_packing enable_zero_shot_eval=True|gemma-3n-e2b-it_bin_packing_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=bin_packing enable_zero_shot_eval=False|gemma-3n-e2b-it_bin_packing_evolve"

    #重跑bin_packing 任务
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-3b-instruct task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|llama-3.2-3b-it_bin_packing_or3_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-3b-instruct task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|llama-3.2-3b-it_bin_packing_or3_evolve"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-3b-instruct task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|llama-3.2-3b-it_bin_packing_weibull_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-3b-instruct task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|llama-3.2-3b-it_bin_packing_weibull_evolve"
    
    # # meta-llama/llama-3.1-70b-instruct
    # "python -m scripts.run_exp model=meta-llama/llama-3.1-70b-instruct task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|llama-3.1-70b-it_bin_packing_or3_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.1-70b-instruct task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|llama-3.1-70b-it_bin_packing_or3_evolve"
    # "python -m scripts.run_exp model=meta-llama/llama-3.1-70b-instruct task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|llama-3.1-70b-it_bin_packing_weibull_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.1-70b-instruct task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|llama-3.1-70b-it_bin_packing_weibull_evolve"

    # # meta-llama/llama-3.1-8b-instruct
    # "python -m scripts.run_exp model=meta-llama/llama-3.1-8b-instruct task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|llama-3.1-8b-it_bin_packing_or3_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.1-8b-instruct task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|llama-3.1-8b-it_bin_packing_or3_evolve"
    # "python -m scripts.run_exp model=meta-llama/llama-3.1-8b-instruct task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|llama-3.1-8b-it_bin_packing_weibull_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.1-8b-instruct task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|llama-3.1-8b-it_bin_packing_weibull_evolve"

    #deepseek-chat-v3-0324
    # "python -m scripts.run_exp model=deepseek/deepseek-chat-v3-0324 task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|deepseek-chat-v3-0324_bin_packing_or3_init"
    # "python -m scripts.run_exp model=deepseek/deepseek-chat-v3-0324 task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|deepseek-chat-v3-0324_bin_packing_or3_evolve"
    # "python -m scripts.run_exp model=deepseek/deepseek-chat-v3-0324 task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|deepseek-chat-v3-0324_bin_packing_weibull_init"
    # "python -m scripts.run_exp model=deepseek/deepseek-chat-v3-0324 task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|deepseek-chat-v3-0324_bin_packing_weibull_evolve"

    # #mistral-large
    # "python -m scripts.run_exp model=vertex_ai/mistral-large task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|mistral-large_bin_packing_or3_init"
    # "python -m scripts.run_exp model=vertex_ai/mistral-large task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|mistral-large_bin_packing_or3_evolve"
    # "python -m scripts.run_exp model=vertex_ai/mistral-large task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|mistral-large_bin_packing_weibull_init"
    # "python -m scripts.run_exp model=vertex_ai/mistral-large task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|mistral-large_bin_packing_weibull_evolve"

    # #mistral-7b-instruct-v0.3
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|mistral-7b-instruct-v0.3_bin_packing_or3_init"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_bin_packing_or3_evolve"    

    # mistralai/mistral-7b-instruct-v0.3
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|mistral-7b-instruct-v0.3_bin_packing_or3_init"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_bin_packing_or3_evolve"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|mistral-7b-instruct-v0.3_bin_packing_weibull_init"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_bin_packing_weibull_evolve"  

    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|mistral-7b-instruct-v0.3_bin_packing_weibull_init"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_bin_packing_weibull_evolve"

    # # mistralai/magistral-small-2506
    # "python -m scripts.run_exp model=mistralai/magistral-small-2506 task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|magistral-small-2506_bin_packing_or3_init"   
    # "python -m scripts.run_exp model=mistralai/magistral-small-2506 task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|magistral-small-2506_bin_packing_or3_evolve"    
    # "python -m scripts.run_exp model=mistralai/magistral-small-2506 task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|magistral-small-2506_bin_packing_weibull_init"   
    # "python -m scripts.run_exp model=mistralai/magistral-small-2506 task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|magistral-small-2506_bin_packing_weibull_evolve"    

    # #gpt-4o
    # "python -m scripts.run_exp model=openai/gpt-4o task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|gpt-4o_bin_packing_or3_init"  
    # "python -m scripts.run_exp model=openai/gpt-4o task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|gpt-4o_bin_packing_or3_evolve"   
    # "python -m scripts.run_exp model=openai/gpt-4o task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|gpt-4o_bin_packing_weibull_init"  
    # "python -m scripts.run_exp model=openai/gpt-4o task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|gpt-4o_bin_packing_weibull_evolve"   

    # #gpt-4o-mini
    # "python -m scripts.run_exp model=openai/gpt-4o-mini task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|gpt-4o-mini_bin_packing_or3_init"    
    # "python -m scripts.run_exp model=openai/gpt-4o-mini task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|gpt-4o-mini_bin_packing_or3_evolve"
    # "python -m scripts.run_exp model=openai/gpt-4o-mini task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|gpt-4o-mini_bin_packing_weibull_init"
    # "python -m scripts.run_exp model=openai/gpt-4o-mini task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|gpt-4o-mini_bin_packing_weibull_evolve"

    # #gpt-3.5-turbo
    # "python -m scripts.run_exp model=openai/gpt-3.5-turbo task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|gpt-3.5-turbo_bin_packing_or3_init"    
    # "python -m scripts.run_exp model=openai/gpt-3.5-turbo task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|gpt-3.5-turbo_bin_packing_or3_evolve" 
    # "python -m scripts.run_exp model=openai/gpt-3.5-turbo task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|gpt-3.5-turbo_bin_packing_weibull_init"    
    # "python -m scripts.run_exp model=openai/gpt-3.5-turbo task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|gpt-3.5-turbo_bin_packing_weibull_evolve"

    # #vertex_ai/gemini-1.5-flash
    # "python -m scripts.run_exp model=vertex_ai/gemini-1.5-flash task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|gemini-1.5-flash_bin_packing_or3_init"   
    # "python -m scripts.run_exp model=vertex_ai/gemini-1.5-flash task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|gemini-1.5-flash_bin_packing_or3_evolve"    
    # "python -m scripts.run_exp model=vertex_ai/gemini-1.5-flash task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|gemini-1.5-flash_bin_packing_weibull_init"   
    # "python -m scripts.run_exp model=vertex_ai/gemini-1.5-flash task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|gemini-1.5-flash_bin_packing_weibull_evolve"    

    #vertex_ai/gemini-1.5
    # "python -m scripts.run_exp model=vertex_ai/gemini-1.5 task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|gemini-1.5_bin_packing_or3_init"   
    # "python -m scripts.run_exp model=vertex_ai/gemini-1.5 task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|gemini-1.5_bin_packing_or3_evolve"    
    # "python -m scripts.run_exp model=vertex_ai/gemini-1.5 task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|gemini-1.5_bin_packing_weibull_init"   


    #google/gemma-3n-e4b-it
    # "python -m scripts.run_exp model=google/gemma-3n-e4b-it task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|gemma-3n-e4b-it_bin_packing_or3_init"    
    # "python -m scripts.run_exp model=google/gemma-3n-e4b-it task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|gemma-3n-e4b-it_bin_packing_or3_evolve"
    ## meta-llama/llama-3.2-1b-instruct
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|llama-3.2-1b_bin_packing_or3_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|llama-3.2-1b_bin_packing_or3_evolve"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|llama-3.2-1b_bin_packing_weibull_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|llama-3.2-1b_bin_packing_weibull_evolve"
    # "python -m scripts.run_exp model=google/gemma-3n-e4b-it task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=True|gemma-3n-e4b-it_bin_packing_weibull_init"
    # "python -m scripts.run_exp model=google/gemma-3n-e4b-it task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|gemma-3n-e4b-it_bin_packing_weibull_evolve"
    # "python -m scripts.run_exp model=vertex_ai/gemini-1.5 task=bin_packing tasks.bin_packing.dataset_type=weibull enable_zero_shot_eval=False|gemini-1.5_bin_packing_weibull_evolve"
    #llama-3.2-1b dataset_type=or3
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=True|llama-3.2-1b_bin_packing_or3_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-1b-instruct task=bin_packing tasks.bin_packing.dataset_type=or3 enable_zero_shot_eval=False|llama-3.2-1b_bin_packing_or3_evolve"

    # mistralai/mistral-7b-instruct-v0.3: Tsp30, Tsp60, symbolic regression (oscillator1, oscillator2) @ temps 0, 0.1, 0.3, 0.5, 0.9, 1.1, 1.3

    # # Tsp30
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=30 temperature=0 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp30_temp0"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=30 temperature=0.1 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp30_temp0.1"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=30 temperature=0.3 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp30_temp0.3"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=30 temperature=0.5 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp30_temp0.5"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=30 temperature=0.7 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp30_temp0.7"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=30 temperature=0.9 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp30_temp0.9"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=30 temperature=1.1 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp30_temp1.1"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=30 temperature=1.3 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp30_temp1.3"
    # Tsp60
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=0 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp0"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=0.1 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp0.1"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=0.3 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp0.3"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=0.5 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp0.5"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=0.7 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp0.7"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=0.9 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp0.9"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=1.1 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp1.1"
    "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=tsp tasks.tsp.city_num=60 temperature=1.3 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_tsp60_temp1.3"
    # # Symbolic regression (oscillator1)
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator1 temperature=0 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc1_temp0"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator1 temperature=0.1 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc1_temp0.1"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator1 temperature=0.3 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc1_temp0.3"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator1 temperature=0.5 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc1_temp0.5"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator1 temperature=0.9 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc1_temp0.9"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator1 temperature=1.1 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc1_temp1.1"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator1 temperature=1.3 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc1_temp1.3"
    # # Symbolic regression (oscillator2)
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator2 temperature=0 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc2_temp0"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator2 temperature=0.1 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc2_temp0.1"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator2 temperature=0.3 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc2_temp0.3"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator2 temperature=0.5 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc2_temp0.5"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator2 temperature=0.9 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc2_temp0.9"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator2 temperature=1.1 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc2_temp1.1"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=symboreg_oscillator2 temperature=1.3 enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_symboreg_osc2_temp1.3"

# ---------------------------------

    # "python -m scripts.run_exp model=meta-llama/llama-3.1-70b-instruct task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|llama-3.1-70b_promptopt_sim_init"
    # "python -m scripts.run_exp model=meta-llama/llama-3.1-8b-instruct task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|llama-3.1-8b_promptopt_sim_init"

    # #deepseek-chat-v3-0324
    # "python -m scripts.run_exp model=deepseek/deepseek-chat-v3-0324 task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|deepseek-chat-v3-0324_promptopt_sim_init"

    # #mistral
    # "python -m scripts.run_exp model=vertex_ai/mistral-large task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|mistral-large_promptopt_sim_init"

    # #mistral-7b-instruct-v0.3
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|mistral-7b-instruct-v0.3_promptopt_sim_init"
    # # mistralai/mistral-7b-instruct-v0.3
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|mistral-7b-instruct-v0.3_promptopt_sim_init"
    # # mistralai/magistral-small-2506
    # "python -m scripts.run_exp model=mistralai/magistral-small-2506 task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|magistral-small-2506_promptopt_sim_init"

    # #openai/gpt-4o
    # "python -m scripts.run_exp model=openai/gpt-4o task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|gpt-4o_promptopt_sim_init"

    # #openai/gpt-4o-mini
    # "python -m scripts.run_exp model=openai/gpt-4o-mini task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|gpt-4o-mini_promptopt_sim_init"
    # #openai/gpt-3.5-turbo
    # "python -m scripts.run_exp model=openai/gpt-3.5-turbo task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|gpt-3.5-turbo_promptopt_sim_init"

    # #vertex_ai/gemini-1.5-flash
    # "python -m scripts.run_exp model=vertex_ai/gemini-1.5-flash task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|gemini-1.5-flash_promptopt_sim_init"
    # #vertex_ai/gemini-1.5
    # "python -m scripts.run_exp model=vertex_ai/gemini-1.5 task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|gemini-1.5_promptopt_sim_init"
    # #google/gemma-3n-e4b-it
    # "python -m scripts.run_exp model=google/gemma-3n-e4b-it task=promptopt tasks.promptopt.eval_task=sim enable_zero_shot_eval=True|gemma-3n-e4b-it_promptopt_sim_init"

    # "python -m scripts.run_exp model=deepseek/deepseek-chat-v3-0324 task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|deepseek-chat-v3-0324_promptopt_sum_evolve"
    # "python -m scripts.run_exp model=vertex_ai/mistral-large task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|mistral-large_promptopt_sum_evolve"
    # "python -m scripts.run_exp model=openai/gpt-4o task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|gpt-4o_promptopt_sum_evolve"
    # "python -m scripts.run_exp model=openai/gpt-4o-mini task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|gpt-4o-mini_promptopt_sum_evolve"
    # "python -m scripts.run_exp model=vertex_ai/gemini-1.5-flash task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|gemini-1.5-flash_promptopt_sum_evolve"
    # "python -m scripts.run_exp model=openai/gpt-3.5-turbo task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|gpt-3.5-turbo_promptopt_sum_evolve"
    # "python -m scripts.run_exp model=vertex_ai/gemini-1.5 task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|gemini-1.5_promptopt_sum_evolve"
    # #  - meta-llama/llama-3.1-70b-instruct
    # #   - meta-llama/llama-3.1-8b-instruct
    # #   - meta-llama/llama-3.2-3b-instruct
    # "python -m scripts.run_exp model=meta-llama/llama-3.1-8b-instruct task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|llama-3.1-8b_promptopt_sum_evolve"
    # "python -m scripts.run_exp model=meta-llama/llama-3.1-70b-instruct task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|llama-3.1-70b_promptopt_sum_evolve"
    # "python -m scripts.run_exp model=meta-llama/llama-3.2-3b-instruct task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|llama-3.2-3b_promptopt_sum_evolve"
    # #   - mistralai/mistral-7b-instruct-v0.3
    # #   - mistralai/magistral-small-2506
    # #   - mistralai/mistral-7b-instruct-v0.3
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_promptopt_sum_evolve"
    # "python -m scripts.run_exp model=mistralai/magistral-small-2506 task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|magistral-small-2506_promptopt_sum_evolve"
    # "python -m scripts.run_exp model=mistralai/mistral-7b-instruct-v0.3 task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|mistral-7b-instruct-v0.3_promptopt_sum_evolve"
    # # "google/gemma-3n-e4b-it"
    # "python -m scripts.run_exp model=google/gemma-3n-e4b-it task=promptopt tasks.promptopt.eval_task=sum enable_zero_shot_eval=False|gemma-3n-e4b-it_promptopt_sum_evolve"


)

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
log_info "Total experiments to run: $TOTAL_EXPERIMENTS"

# 开始运行实验
log_info "=== Starting Experiments ==="
BATCH_START_TIME=$(date +%s)

for exp in "${EXPERIMENTS[@]}"; do
    # 分割命令和名称
    IFS='|' read -r cmd name <<< "$exp"
    
    # 运行实验
    run_experiment "$cmd" "$name"
    
    # 如果有失败，询问是否继续
    if [ $FAILED_EXPERIMENTS -gt 0 ] && [ $((FAILED_EXPERIMENTS % 3)) -eq 0 ]; then
        log_warning "Already $FAILED_EXPERIMENTS experiments failed. Check error log if needed."
    fi
done

# 批量实验结束
BATCH_END_TIME=$(date +%s)
TOTAL_DURATION=$((BATCH_END_TIME - BATCH_START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

log_info "=== Batch Experiments Completed ==="
log_info "Total experiments: $TOTAL_EXPERIMENTS"
log_info "Completed successfully: $COMPLETED_EXPERIMENTS"
log_info "Failed: $FAILED_EXPERIMENTS"
log_info "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log_info "Average time per experiment: $((TOTAL_DURATION / TOTAL_EXPERIMENTS))s"

# 生成总结报告
SUMMARY_FILE="$LOG_DIR/summary.txt"
cat > "$SUMMARY_FILE" << EOF
=== Batch Experiment Summary ===
Date: $(date)
Total experiments: $TOTAL_EXPERIMENTS
Completed successfully: $COMPLETED_EXPERIMENTS
Failed: $FAILED_EXPERIMENTS
Success rate: $((COMPLETED_EXPERIMENTS * 100 / TOTAL_EXPERIMENTS))%
Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s
Average time per experiment: $((TOTAL_DURATION / TOTAL_EXPERIMENTS))s

Log files:
- Main log: $MAIN_LOG
- Progress log: $PROGRESS_LOG
- Error log: $ERROR_LOG
- Individual experiment logs: $LOG_DIR/exp_*.log
EOF

log_info "Summary report saved to: $SUMMARY_FILE"

if [ $FAILED_EXPERIMENTS -gt 0 ]; then
    log_warning "Some experiments failed. Check the error log: $ERROR_LOG"
    exit 1
else
    log_success "All experiments completed successfully!"
    exit 0
fi