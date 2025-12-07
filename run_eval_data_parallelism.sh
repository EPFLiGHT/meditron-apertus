if [ -f .env ]; then set -a; source .env; set +a; fi

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

accelerate launch -m lm_eval --model hf \
  --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,parallelize=False,trust_remote_code=True" \
  --tasks pubmedqa,medmcqa,medqa_4options \
  --batch_size 1 \
  --verbosity DEBUG \
  --log_samples \
  --output_path $PROJECT_ROOT/eval_results/ \
  --gen_kwargs '{"max_new_tokens": 2048}' \
  --apply_chat_template tokenizer_default 



#pubmedqa,medmcqa,medqa_4options,pubmedqa_g,medmcqa_g,medqa_g,medxpertqa,medxpertqa_g,mmlu_flan_cot_zeroshot
