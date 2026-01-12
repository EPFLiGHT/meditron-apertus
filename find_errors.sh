#!/usr/bin/env bash
set -euo pipefail

REPORTS_DIR="${1:-reports}"

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
  search_cmd=(rg -n -m1 -i)
  nan_pattern="\\bnan\\b|loss is nan|nan loss"
else
  search_cmd=(grep -n -m1 -i -E)
  nan_pattern="\\<nan\\>|loss is nan|nan loss"
fi

for report in "${sorted_reports[@]}"; do
  echo "== $report =="

  if "${search_cmd[@]}" "cuda out of memory|cudnn_status_alloc_failed|out of memory" "$report"; then
    :
  elif "${search_cmd[@]}" "$nan_pattern" "$report"; then
    :
  elif "${search_cmd[@]}" "too many open files|errno 24" "$report"; then
    :
  else
    echo "IDK"
  fi

  echo
done
