#!/bin/bash
#SBATCH --job-name meditron-eval
#SBATCH --output reports/R-%x.%j.err
#SBATCH --error reports/R-%x.%j.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 64
#SBATCH --time 05:00:00
#SBATCH --environment ../.edf/apertus.toml
#SBATCH -A a127

# =========================================================
# PHASE 1: SUBMISSION LOGIC (Runs on Login Node)
# =========================================================

if [ -z "$SLURM_JOB_ID" ]; then
    # Optional: first CLI arg = run name (for logs)
    RUN_NAME="${1:-eval-apertus-8b}"
    SCRIPT_PATH="$0"

    # Load env (PROJECT_ROOT, USER_STORAGE, etc.)
    if [ -f .env ]; then
        set -o allexport
        source .env
        set +o allexport
    fi

    echo "This script is self-submitting..."
    echo "🏷️  Run Name:  $RUN_NAME"

    SUBMISSION_OUTPUT=$(sbatch -J "$RUN_NAME" "$SCRIPT_PATH" "$RUN_NAME")
    JOB_ID=$(echo "$SUBMISSION_OUTPUT" | awk '{print $4}')

    echo "🚀 Submitted Job: $JOB_ID"

    LOG_FILE="$PROJECT_ROOT/reports/R-${RUN_NAME}.${JOB_ID}.err"
    echo "Waiting for log file: $LOG_FILE"

    while [ ! -f "$LOG_FILE" ]; do
        sleep 1
    done

    echo "✅ Log found! Tailing (Ctrl+C to stop watching, job will continue)..."
    echo "-------------------------------------------------------------------"
    tail -f "$LOG_FILE"

    exit 0
fi



# =========================================================
# PHASE 2: WORKER LOGIC (Runs on Compute Node)
# =========================================================

RUN_NAME="$1"
if [ -z "$RUN_NAME" ]; then
    RUN_NAME="eval-apertus-8b"
fi

# 1. Project root & env
export PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$PROJECT_ROOT"

echo "Project Root detected as: $PROJECT_ROOT"

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

cd "/users/theimer/lm-evaluation-harness"
pip install -e .
pip install git+https://github.com/huggingface/datasets.git

cd "/users/theimer/lm-evaluation-harness/lm_eval/tasks"

export WORLD_SIZE=$SLURM_NNODES
export MASTER_ADDR=$(hostname)
export MASTER_PORT=6300
export HF_HOME=$USER_STORAGE/hf
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRITON_CACHE_DIR=$USER_STORAGE/triton
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers

export MODEL_PATH="/capstor/store/cscs/swissai/a127/apertus/huggingface/Apertus8B"

echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MODEL_PATH=$MODEL_PATH"

echo "START TIME: $(date)"
set -eo pipefail
set -x

accelerate launch -m lm_eval \
  --model hf \
  --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,attn_implementation=flash_attention_2,trust_remote_code=True" \
  --tasks medqa_g,medmcqa_g \
  --batch_size 16 \
  --verbosity DEBUG \
  --log_samples \
  --output_path $PROJECT_ROOT/eval_results/ \
  --gen_kwargs '{"max_new_tokens": 1024}' \
  --limit 100 \
  --apply_chat_template tokenizer_default 

echo "END TIME: $(date)"

#pubmedqa,medmcqa,medqa_4options,pubmedqa_g,medmcqa_g,medqa_g,medxpertqa,medxpertqa_g,mmlu_flan_cot_zeroshot
