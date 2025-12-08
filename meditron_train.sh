#!/bin/bash
#SBATCH --job-name meditron-default-job
#SBATCH --output reports/R-%x.%j.out
#SBATCH --error reports/R-%x.%j.err
#SBATCH --nodes 16
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 11:59:59
#SBATCH --environment ../.edf/apertus.toml
#SBATCH -A a127

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

    # 2. Load Environment
    if [ -f .env ]; then
        set -a; source .env; set +a
    else
        echo "ERROR: .env file not found."
        exit 1
    fi

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

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
    echo "Successfully loaded .env"
else
    echo "CRITICAL ERROR: .env file not found"
    exit 1
fi

export HF_HOME="$USER_STORAGE/hf"
export WANDB_DIR="$USER_STORAGE/wandb"
export WANDB_MODE="online"

# Construct full path to config (Handling relative paths from Project Root)
export AXOLOTL_CONFIG_FILE="$PROJECT_ROOT/$CONFIG_ARG"

echo "🔧 Axolotl Config: $AXOLOTL_CONFIG_FILE"

# Validate the resolved DeepSpeed config path early so rank 0 fails fast with a clear error.
DEEPSPEED_CFG_PATH=$(python3 - <<'PY'
import os, yaml
cfg_path = os.environ["AXOLOTL_CONFIG_FILE"]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
ds = cfg.get("deepspeed")
if isinstance(ds, str):
    print(os.path.expandvars(ds))
PY
)
if [ -n "$DEEPSPEED_CFG_PATH" ]; then
    echo "🔧 DeepSpeed Config (resolved): $DEEPSPEED_CFG_PATH"
    if [ ! -f "$DEEPSPEED_CFG_PATH" ]; then
        echo "CRITICAL ERROR: DeepSpeed config file not found at resolved path."
        exit 1
    fi
fi

# Caching locations
export XDG_CACHE_HOME="$USER_STORAGE/cache"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export PYTORCH_KERNEL_CACHE_PATH="$XDG_CACHE_HOME/torch_extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"

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
