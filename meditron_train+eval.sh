#!/bin/bash
#SBATCH --job-name meditron-default-job
#SBATCH --output train_reports/R-%x.%j.err
#SBATCH --error train_reports/R-%x.%j.err
#SBATCH --nodes 32
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 2:59:59
#SBATCH --environment ../.edf/new_axolotl.toml
#SBATCH -A a127

ulimit -c 0 # prevents core dumps

# =========================================================
# PHASE 1: SUBMISSION LOGIC (Runs on Login Node)
# =========================================================
if [ -z "$SLURM_JOB_ID" ]; then
    if [ -z "$1" ]; then
        echo "❌ Error: No config file provided."
        exit 1
    fi
    TEMPLATE_CFG="$1"
    
    # Load Environment
    set -o allexport
    if [ -f "$PROJECT_ROOT/.env" ]; then source "$PROJECT_ROOT/.env"; elif [ -f .env ]; then source .env; fi
    set +o allexport

    SRC_CFG="$PROJECT_ROOT/$TEMPLATE_CFG"
    JOB_NAME=$(basename "$TEMPLATE_CFG" .yaml)
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    
    GEN_CONFIG_DIR="$PROJECT_ROOT/axolotl_config/generated"
    mkdir -p "$GEN_CONFIG_DIR"
    DEST_CFG="$GEN_CONFIG_DIR/${JOB_NAME}-${TIMESTAMP}.yaml"
    
    echo "📝 Using template: $SRC_CFG"
    envsubst < "$SRC_CFG" > "$DEST_CFG"

    SCRIPT_PATH="$0"
    # Ensure Slurm doesn't try to create a non-existent TMPDIR before scratch exists.
    SUBMISSION_OUTPUT=$(TMPDIR=/tmp sbatch -J "$JOB_NAME" "$SCRIPT_PATH" "$DEST_CFG")
    JOB_ID=$(echo "$SUBMISSION_OUTPUT" | awk '{print $4}')
    echo "🚀 Submitted Job: $JOB_ID"

    LOG_FILE="$PROJECT_ROOT/train_reports/R-${JOB_NAME}.${JOB_ID}.err"
    while [ ! -f "$LOG_FILE" ]; do sleep 1; done
    tail -n 0 -F "$LOG_FILE" &
    TAIL_PID=$!
    while squeue -j "$JOB_ID" >/dev/null 2>&1; do sleep 5; done
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
    exit 0
fi

# =========================================================
# PHASE 2: WORKER LOGIC (Runs on Compute Nodes)
# =========================================================

FROZEN_CONFIG_PATH="$1"
if [ ! -f "$FROZEN_CONFIG_PATH" ]; then
    echo "CRITICAL ERROR: Config not found: $FROZEN_CONFIG_PATH"
    exit 1
fi

# --- ADDED: Initialize Vars for Slack ---
export START_TS=$(date +%s)
export START_HUMAN=$(date -Is)
# Extract just the filename without extension for the run name
export RUN_NAME=$(basename "$FROZEN_CONFIG_PATH" .yaml)
# Ensure Node count is available (Slurm usually sets this, but we explicitly export it)
export NODE_COUNT=${SLURM_NNODES:-1}
# ----------------------------------------

export PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}

# 1. Load Environment
set -o allexport
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
elif [ -f .env ]; then
    source .env
fi
set +o allexport

# 2. Fix Slack Helper Import
# Using absolute path ensures workers find the script despite spooling
if ! declare -F slack_notify >/dev/null 2>&1; then
    if [ -f "$PROJECT_ROOT/scripts/slack_helpers.sh" ]; then
        source "$PROJECT_ROOT/scripts/slack_helpers.sh"
    fi
fi

# 3. Define Scratch Paths (Do NOT export TMPDIR yet!)
JOB_SCRATCH_BASE=${TMPDIR_BASE:-/iopsstor/scratch/cscs/theimer/axolotl-cache}
mkdir -p $JOB_SCRATCH_BASE
JOB_SCRATCH="$JOB_SCRATCH_BASE/${SLURM_JOB_ID:-nojob}"
LOCAL_TMP="$JOB_SCRATCH/tmp"
LOCAL_HF="$JOB_SCRATCH/hf_cache"
LOCAL_DS="$JOB_SCRATCH/hf_datasets"
LOCAL_TRITON="$JOB_SCRATCH/triton"
LOCAL_WANDB="$JOB_SCRATCH/wandb"
mkdir -p "$LOCAL_TMP" "$LOCAL_HF" "$LOCAL_DS" "$LOCAL_TRITON" "$LOCAL_WANDB"

# 4. PRE-FLIGHT CHECK: Create Directories on ALL Nodes
# Added: -A and --reservation to match the main job parameters
echo "🛠️  Pre-creating scratch directories on all $SLURM_NNODES nodes..."

export TMPDIR=/iopsstor/scratch/cscs/theimer/tmp
mkdir -p "$TMPDIR"
srun --ntasks-per-node=1 \
     --cpus-per-task=1 \
     --nodes=$SLURM_NNODES \
     -A a127 \
     --reservation=sai-a127 \
     bash -c "mkdir -p $LOCAL_TMP $LOCAL_HF $LOCAL_DS $LOCAL_TRITON $LOCAL_WANDB/wandb && ulimit -n 65535"

if [ $? -ne 0 ]; then
    echo "❌ CRITICAL: Failed to create scratch directories on remote nodes."
    exit 1
fi
echo "✅ Scratch directories ready."

# 5. Export Environment
export TMPDIR="$LOCAL_TMP"
export HF_HOME="$LOCAL_HF"
export TRANSFORMERS_CACHE="$LOCAL_HF/transformers"
export HF_DATASETS_CACHE="$LOCAL_DS"
export TRITON_CACHE_DIR="$LOCAL_TRITON"
export WANDB_DIR="$LOCAL_WANDB"
export AXOLOTL_CONFIG_FILE="$FROZEN_CONFIG_PATH"
export AXOLOTL_DO_NOT_TRACK=1 
export WANDB_MODE="online"

# Quiet noisy logs
export DEEPSPEED_DISABLE_ASYNC_IO=1
export DS_BUILD_SPARSE_ATTN=0
export SLACK_INSECURE="${SLACK_INSECURE:-1}"

trap 'rc=$?; slack_notify "$rc"; exit "$rc"' EXIT

# 6. Network Setup
GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6200

LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 0"

# 7. Launch Training
FULL_CMD="$LAUNCHER -m axolotl.cli.train $AXOLOTL_CONFIG_FILE"

echo "🚀 Launching Axolotl on $SLURM_NNODES nodes..."
echo "Command: $FULL_CMD"

LOG_FILE="$PROJECT_ROOT/train_reports/R-${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    -A a127 \
    --reservation=sai-a127 \
    bash -c "$FULL_CMD"

if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
fi

# Slack runtime bookkeeping (worker side)
START_TS="$(date +%s)"
START_HUMAN="$(date -Is)"
SLACK_JOB_ID="${SLURM_JOB_ID:-?}"
FAILED_CMD=""
trap 'FAILED_CMD=$BASH_COMMAND' ERR
trap 'rc=$?; slack_notify "$rc" "compute"; exit "$rc"' EXIT
set -eo pipefail

cd "$PROJECT_ROOT/../lm-evaluation-harness"
pip install -e .
pip install --upgrade --no-deps "datasets>=2.19.0,<3.0.0"

cd "$PROJECT_ROOT/../lm-evaluation-harness/lm_eval/tasks"

export WORLD_SIZE=$SLURM_NNODES
export MASTER_ADDR=$(hostname)
export MASTER_PORT=6300
export HF_HOME=$USER_STORAGE/hf
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRITON_CACHE_DIR=$USER_STORAGE/triton
export HF_DATASETS_CACHE=$HF_HOME/datasets
unset TRANSFORMERS_CACHE
export LM_EVAL_INCLUDE_PATH="/users/theimer/lm-evaluation-harness/lm_eval/tasks"

echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MODEL_PATH=$MODEL_PATH"

echo "START TIME: $(date)"
set -x

SAFE_MODEL_TAG="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_.-' '_' | sed 's/^_//;s/_$//')"
RUN_TAG="${RUN_NAME:-eval}"
JOB_TAG="${SLURM_JOB_ID:-nojob}"
OUTPUT_DIR="$PROJECT_ROOT/eval_results/${RUN_TAG}_${SAFE_MODEL_TAG}_${JOB_TAG}"
mkdir -p "$OUTPUT_DIR"

VERBOSITY_LEVEL="INFO"
LIMIT_ARGS=()
if [ "$DEBUG_FLAG" -eq 1 ]; then
    VERBOSITY_LEVEL="DEBUG"
    LIMIT_ARGS=(--limit 100)
fi

#MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,attn_implementation=flash_attention_3,trust_remote_code=True"
MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,attn_implementation=flash_attention_2,trust_remote_code=True"
if [ "$MODEL_PARALLELISM" -eq 1 ]; then
    MODEL_ARGS="$MODEL_ARGS,parallelize=True"
fi

if [ "$MODEL_PARALLELISM" -eq 1 ]; then
python3 -m lm_eval \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks pubmedqa_g,medmcqa_g,medqa_g \
  --batch_size 16 \
  --verbosity "$VERBOSITY_LEVEL" \
  --log_samples \
  --output_path "$OUTPUT_DIR" \
  --include_path "$LM_EVAL_INCLUDE_PATH" \
  --gen_kwargs max_new_tokens=1024 \
  "${LIMIT_ARGS[@]}" \
  --apply_chat_template tokenizer_default 
else
accelerate launch --num_processes 4 --num_machines 1 --mixed_precision bf16 --dynamo_backend no -m lm_eval \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks pubmedqa_g,medmcqa_g,medqa_g \
  --batch_size 16 \
  --verbosity "$VERBOSITY_LEVEL" \
  --log_samples \
  --output_path "$OUTPUT_DIR" \
  --include_path "$LM_EVAL_INCLUDE_PATH" \
  --gen_kwargs max_new_tokens=1024 \
  "${LIMIT_ARGS[@]}" \
  --apply_chat_template tokenizer_default 
fi

echo "END TIME: $(date)"
