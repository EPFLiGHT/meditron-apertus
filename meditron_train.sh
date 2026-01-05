#!/bin/bash
#SBATCH --job-name meditron-default-job
#SBATCH --output reports/R-%x.%j.err
#SBATCH --error reports/R-%x.%j.err
#SBATCH --nodes 16
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 11:59:59
#SBATCH --environment ../.edf/apertus.toml
#SBATCH -A a127

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/slack_helpers.sh"

# =========================================================
# PHASE 1: SUBMISSION LOGIC (Runs on Login Node)
# =========================================================
if [ -z "$SLURM_JOB_ID" ]; then
    
    # 1. Check for Config Argument
    if [ -z "$1" ]; then
        echo "❌ Error: No config file provided."
        echo "Usage: $0 <path/to/config.yaml>"
        exit 1
    fi
    CONFIG_ARG="$1"

    # 2. Load Environment (use absolute path so compute nodes can also source it)
    set -o allexport
    if [ -f "$PROJECT_ROOT/.env" ]; then
        source "$PROJECT_ROOT/.env"
    elif [ -f .env ]; then
        # fallback to local .env if present
        source .env
    else
        echo "CRITICAL: .env not found at $PROJECT_ROOT/.env or ./env. Please ensure the .env file is present before submitting."
        exit 1
    fi
    set +o allexport

    SRC_CFG="$PROJECT_ROOT/$CONFIG_ARG"
    DEST_CFG="$PROJECT_ROOT/axolotl_config/config.yaml"

    echo "Using template config: $SRC_CFG"
    echo "Writing substituted config to: $DEST_CFG"

    envsubst < "$SRC_CFG" > "$DEST_CFG"

    export AXOLOTL_CONFIG_FILE="$DEST_CFG"

    


    SCRIPT_PATH="$0"

    # 3. Derive Job Name from Config Filename (e.g., 'apertus-8b' from 'config/apertus-8b.yaml')
    # This overrides the #SBATCH --job-name above
    JOB_NAME=$(basename "$CONFIG_ARG" .yaml)

    echo "This script is self-submitting..."
    echo "📄 Config:      $CONFIG_ARG"
    echo "🏷️  Job Name:    $JOB_NAME"

    # 4. Submit *this script* with the config as an argument
    # We use -J to override the job name so logs match the config
    SUBMISSION_OUTPUT=$(sbatch -J "$JOB_NAME" "$SCRIPT_PATH" "$CONFIG_ARG")
    JOB_ID=$(echo "$SUBMISSION_OUTPUT" | awk '{print $4}')

    echo "🚀 Submitted Job: $JOB_ID"

    # 5. Construct Log Path (Matches #SBATCH --error reports/R-%x.%j.err)
    LOG_FILE="$PROJECT_ROOT/reports/R-${JOB_NAME}.${JOB_ID}.err"
    echo "Waiting for log file: $LOG_FILE"

    while [ ! -f "$LOG_FILE" ]; do
        sleep 1
    done

    echo "✅ Log found! Tailing (Ctrl+C to stop watching, job will continue)..."
    echo "-------------------------------------------------------------------"
    tail -f "$LOG_FILE"
    echo "🔧 Axolotl Config (after envsubst):"
    cat $AXOLOTL_CONFIG_FILE
    exit 0
fi

# =========================================================
# PHASE 2: WORKER LOGIC (Runs on Compute Nodes)
# =========================================================

# 1. Retrieve Config Argument (Passed from Phase 1)
CONFIG_ARG="$1"
if [ -z "$CONFIG_ARG" ]; then
    echo "CRITICAL ERROR: Script running on node but no config argument received."
    exit 1
fi

# 2. Environment Setup
export PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
echo "Project Root detected as: $PROJECT_ROOT"

set -o allexport
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
elif [ -f .env ]; then
    source .env
else
    echo "CRITICAL: .env not found at $PROJECT_ROOT/.env or ./env. Compute node cannot continue without environment variables."
    echo "DEBUG: SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
    echo "DEBUG: pwd=$(pwd)"
    echo "DEBUG: listing project root contents:"; ls -la "$PROJECT_ROOT" || true
    exit 1
fi
set +o allexport

export WANDB_MODE="online"
export AXOLOTL_CONFIG_FILE="$PROJECT_ROOT/axolotl_config/config.yaml"

# Ensure Slack helpers are loaded on worker even if earlier source failed
if ! declare -F slack_notify >/dev/null 2>&1; then
    if [ -f "$SCRIPT_DIR/slack_helpers.sh" ]; then
        source "$SCRIPT_DIR/slack_helpers.sh"
    elif [ -f "$PROJECT_ROOT/slack_helpers.sh" ]; then
        source "$PROJECT_ROOT/slack_helpers.sh"
    fi
fi

# Use per-job scratch to avoid NFS cache thrash and missing TMPDIR
JOB_SCRATCH_BASE=${TMPDIR_BASE:-/tmp/theimer/axolotl-cache}
mkdir -p "$JOB_SCRATCH_BASE" || true
JOB_SCRATCH="$JOB_SCRATCH_BASE/${SLURM_JOB_ID:-nojob}"
mkdir -p "$JOB_SCRATCH/tmp" "$JOB_SCRATCH/hf_cache" "$JOB_SCRATCH/hf_datasets" "$JOB_SCRATCH/triton" "$JOB_SCRATCH/wandb/wandb" || true
export TMPDIR="$JOB_SCRATCH/tmp"
export HF_HOME="$JOB_SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$JOB_SCRATCH/hf_datasets"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export WANDB_DIR="$JOB_SCRATCH/wandb"

# Quiet noisy DeepSpeed/Triton warnings on this cluster
export DEEPSPEED_DISABLE_ASYNC_IO=1
export DS_BUILD_SPARSE_ATTN=0

export SLACK_INSECURE="${SLACK_INSECURE:-1}"
START_TS="$(date +%s)"
START_HUMAN="$(date -Is)"
SLACK_JOB_ID="${SLURM_JOB_ID:-?}"
FAILED_CMD=""
trap 'FAILED_CMD=$BASH_COMMAND' ERR
trap 'rc=$?; slack_notify "$rc"; exit "$rc"' EXIT

# Bump file descriptor limit to avoid "too many open files" during dataset packing
ulimit -n 65535 || echo "WARN: unable to raise open files limit"
echo "Open files limit: $(ulimit -n)"

echo "START TIME: $(date)"
set -eo pipefail
set -x

# 3. Network & Topology
GPUS_PER_NODE=4
echo "NODES: $SLURM_NNODES"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6200

# 4. Execution Command
LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 0"

FULL_CMD="$LAUNCHER -m axolotl.cli.train $AXOLOTL_CONFIG_FILE"

# 5. SLURM Launch
SRUN_ARGS=" \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    -A a127 \
    --reservation=sai-a127"

echo "Command: $FULL_CMD"
srun $SRUN_ARGS bash -c "$FULL_CMD"

echo "END TIME: $(date)"
