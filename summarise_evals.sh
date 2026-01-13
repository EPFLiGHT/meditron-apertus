#!/usr/bin/env bash
set -euo pipefail

REPORTS_DIR="${1:-eval_reports}"

if [ ! -d "$REPORTS_DIR" ]; then
  echo "Reports directory not found: $REPORTS_DIR" >&2
  exit 1
fi

shopt -s nullglob
mapfile -d '' report_files < <(find "$REPORTS_DIR" -maxdepth 1 -type f -print0)
if [ "${#report_files[@]}" -eq 0 ]; then
  echo "No report files found in $REPORTS_DIR" >&2
  exit 0
fi

sorted_reports=()
while IFS= read -r report; do
  sorted_reports+=("$report")
done < <(
  printf '%s\0' "${report_files[@]}" \
    | xargs -0 stat -c '%Y %n' \
    | sort -n \
    | awk '{$1=""; sub(/^ /,""); print}'
)

if command -v rg >/dev/null 2>&1; then
  line_cmd=(rg -n -m1)
  grep_cmd=(rg -n)
else
  line_cmd=(grep -n -m1)
  grep_cmd=(grep -n)
fi

for report in "${sorted_reports[@]}"; do
  echo "== $report =="

  model_line="$("${line_cmd[@]}" "MODEL_PATH=" "$report" || true)"
  model=""
  if [ -n "$model_line" ]; then
    model="${model_line#*MODEL_PATH=}"
  else
    model_line="$("${line_cmd[@]}" "pretrained=" "$report" || true)"
    if [ -n "$model_line" ]; then
      model="$(printf '%s' "$model_line" | sed -E 's/.*pretrained=([^, ]+).*/\1/')"
    fi
  fi

  if [ -n "$model" ]; then
    echo "Model: $model"
  else
    echo "Model: UNKNOWN"
  fi

  exit_code="$(awk -F'Exit code: ' '/Exit code:/{code=$2} END{if(code!="") print code}' "$report" | sed -E 's/[^0-9].*$//')"
  if [ -n "$exit_code" ]; then
    if [ "$exit_code" = "0" ]; then
      echo "Status: COMPLETED"
    else
      echo "Status: FAILED (exit code $exit_code)"
    fi
  else
    echo "Status: UNKNOWN"
  fi

  results_table="$(awk '
    /^\|[[:space:]]*Tasks[[:space:]]*\|/ {in_table=1; print; next}
    in_table && /^\|/ {print; next}
    in_table {exit}
  ' "$report")"

  if [ -n "$results_table" ]; then
    echo "Results:"
    printf '%s\n' "$results_table"
  else
    error_summary=""
    if "${grep_cmd[@]}" "Traceback" "$report" >/dev/null 2>&1; then
      error_summary="$(awk '
        /Traceback/ {tb=1}
        tb && ($0 ~ /(Error|Exception|KeyError|RuntimeError|ValueError|TypeError|AssertionError|ChildFailedError)/) {last=$0}
        END {if (last!="") print last}
      ' "$report")"
    fi

    if [ -z "$error_summary" ]; then
      error_summary="$("${line_cmd[@]}" "ERROR:|Error:|FAILED|exit code: [1-9]" "$report" 2>/dev/null || true)"
    fi

    if [ -n "$error_summary" ]; then
      echo "Error: $error_summary"
    else
      echo "Error: UNKNOWN"
    fi
  fi

  echo
done
