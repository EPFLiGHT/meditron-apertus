#!/bin/bash
#SBATCH --job-name meditron-apertus-8b-only-med-new-2
#SBATCH --output reports/R-%x.%j.err
#SBATCH --error reports/R-%x.%j.err
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 11:59:59
#SBATCH --environment ../.edf/apertus.toml
#SBATCH -A a127

# ========================
# 1. Environment 
# ========================

echo "Job launched from: ${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "ERROR: .env file not found. Please create one based on .env.example"
    exit 1
fi

export HF_HOME="$USER_STORAGE/hf"
export WANDB_DIR="$USER_STORAGE/wandb"
export WANDB_MODE="online"

TEMPLATE_CONFIG="$PROJECT_ROOT/axolotl_config/apertus-8b-ablation-no-mediset.yaml"
AXOLOTL_CONFIG_FILE="config_generated_${SLURM_JOB_ID}.yaml"
envsubst < "$TEMPLATE_CONFIG" > "$AXOLOTL_CONFIG_FILE"

# SHOULD not be tmp Caching locations
export XDG_CACHE_HOME="$USER_STORAGE/cache"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export PYTORCH_KERNEL_CACHE_PATH="$XDG_CACHE_HOME/torch_extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"

#export CUDA_LAUNCH_BLOCKING=1 #CRITICAL REMOVE this is for debugging

# Script setup
echo "START TIME: $(date)"
set -eo pipefail
set -x

# ========================
# 2. Network & Topology
# ========================
GPUS_PER_NODE=4
echo "NODES: $SLURM_NNODES"

# Get the first node name as the master address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6200

# ========================
# 3. Execution Command
# ========================
# Note: Variables like \$SLURM_PROCID are escaped so they evaluate 
# inside the srun tasks, not on the launch node.

LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 0" # SHOULD be 0 because 3 logs all workers not only master

FULL_CMD="$LAUNCHER -m axolotl.cli.train $AXOLOTL_CONFIG_FILE"

# ========================
# 4. SLURM Launch
# ========================
SRUN_ARGS=" \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    -A a127 \
    --reservation=sai-a127"

echo "Command: $FULL_CMD"
srun $SRUN_ARGS bash -c "$FULL_CMD"

echo "END TIME: $(date)"
