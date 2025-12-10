# meditron-apertus

Axolotl configs and Slurm helpers for training/evaluating Apertus-based Meditron models on CSCS.

## Prerequisites
- CSCS account with access to the storage paths referenced in the configs.
- Python environment described by your EDF file (see `ENV` below).
- Clone of the lm-evaluation-harness fork alongside this repo: `git clone https://github.com/Xkrilandar/lm-evaluation-harness`.

## Environment setup
1. Create a `.env` in the repo root with your paths and tokens (do not commit secrets):
   ```
   # Paths
   PROJECT_ROOT=/users/<user>/meditron-apertus
   STORAGE_ROOT=/capstor/store/cscs/swissai/a127
   USER_STORAGE=$STORAGE_ROOT/homes/<user>
   ENV=/users/<user>/.edf/apertus.toml

   # Auth
   WANDB_API_KEY=<wandb_token>
   HF_TOKEN=<hf_token>

   # Logging
   WANDB_PROJECT=<wandb-project>
   WANDB_ENTITY=<wandb-entity>
   ```
2. Log in to CSCS and load your environment (quick commands live in `setup.md` and `script_login.bash`).

## Training
- Pick a config in `axolotl_config/` (e.g., `apertus-8b-only-mediset.yaml`, `apertus-8b-ablation-no-mediset.yaml`, `apertus-70b.yaml`).
- Submit via Slurm (self-submits and tails logs):
  ```
  bash meditron_train.sh axolotl_config/apertus-8b-only-mediset.yaml
  ```
  The script:
  - injects your `.env` values into the template and writes `axolotl_config/config.yaml`,
  - submits itself with `sbatch -J <config-name> ...`,
  - tails `reports/R-<job>.<jobid>.err` once the log appears.
- Adjust SBATCH resources at the top of `meditron_train.sh` if you need different GPUs/time.

## Evaluation
- Model parallel run (uses torchrun/parallelize):
  ```
  bash run_eval_model_parallelism.sh
  ```
- Data parallel run (uses accelerate launch):
  ```
  bash run_eval_data_parallelism.sh
  ```
Both scripts expect:
- `SLURM_NNODES` set by the scheduler,
- lm-evaluation-harness installed from your local clone (`git clone https://github.com/Xkrilandar/lm-evaluation-harness`),
- `MODEL_PATH` pointing to the HF repo or local checkpoint you want to score.
