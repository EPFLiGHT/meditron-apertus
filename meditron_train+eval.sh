#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ] || [ -z "${1:-}" ]; then
  echo "Usage: $0 <train_config.yaml> [eval_args...]"
  echo "Example: $0 axolotl_config/apertus-8b.yaml --debug"
  exit 1
fi

TRAIN_CONFIG="$1"
shift

EVAL_ARGS=()
while [ "$#" -gt 0 ]; do
  EVAL_ARGS+=("$1")
  shift
done

if [ -f "$SCRIPT_DIR/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "$SCRIPT_DIR/.env"
  set +o allexport
fi

resolve_output_dir() {
  local cfg="$1"
  local raw=""
  raw="$(awk -F': ' '/^output_dir:/{print $2; exit}' "$cfg")"
  if [ -z "$raw" ]; then
    echo ""
    return
  fi
  if command -v envsubst >/dev/null 2>&1; then
    printf '%s' "$raw" | envsubst
  else
    printf '%s' "$raw"
  fi
}

has_weights() {
  local dir="$1"
  ls "$dir"/*.safetensors "$dir"/*.bin "$dir"/*.pt "$dir"/adapter_model.safetensors >/dev/null 2>&1
}

echo "Running training: $TRAIN_CONFIG"
bash "$SCRIPT_DIR/meditron_train.sh" "$TRAIN_CONFIG"

OUTPUT_DIR="$(resolve_output_dir "$TRAIN_CONFIG")"
if [ -z "$OUTPUT_DIR" ]; then
  echo "Could not resolve output_dir from $TRAIN_CONFIG" >&2
  exit 1
fi

EVAL_MODEL="$OUTPUT_DIR"

if [ ! -d "$EVAL_MODEL" ]; then
  echo "Eval model path not found: $EVAL_MODEL" >&2
  exit 1
fi

if ! has_weights "$EVAL_MODEL"; then
  echo "No weights found in $EVAL_MODEL (expected .safetensors/.bin/.pt)" >&2
  exit 1
fi

echo "Running eval: $EVAL_MODEL ${EVAL_ARGS[*]}"
bash "$SCRIPT_DIR/meditron_eval.sh" "$EVAL_MODEL" "${EVAL_ARGS[@]}"
