if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

bash run_eval_data_parallelism_debug.sh "$STORAGE_ROOT/meditron/models/Meditron-Apertus-8B-only-med-no-moove" &
bash run_eval_data_parallelism_debug.sh "$STORAGE_ROOT/apertus/huggingface/Apertus8B" &
wait