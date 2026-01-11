#!/usr/bin/env bash
# Shared Slack helpers for training/eval scripts.
# Expects the caller to define START_TS, START_HUMAN, RUN_NAME, SLURM_JOB_ID/SLACK_JOB_ID, and optionally FAILED_CMD.

if [ -n "${SLACK_HELPERS_LOADED:-}" ]; then
  return 0
fi
SLACK_HELPERS_LOADED=1

: "${SLACK_INSECURE:=1}"

format_duration() {
    local s="$1"
    printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

_slack_build_payload() {
    local text="$1"
    local payload=""

    local PY_BIN=""
    if command -v python3 >/dev/null 2>&1; then
        PY_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PY_BIN="python"
    fi

    if [ -n "$PY_BIN" ]; then
        payload="$("$PY_BIN" - <<'PY' "$text"
import json, sys
msg = sys.argv[1] if len(sys.argv) > 1 else ""
print(json.dumps({"text": msg}))
PY
)"
    else
        local escaped_msg
        escaped_msg="$(printf '%s' "$text" | sed 's/\"/\\\\\"/g')"
        payload="{\"text\": \"${escaped_msg}\"}"
    fi

    printf '%s' "$payload"
}

slack_notify() {
    # First arg: exit code. Additional args ignored.
    local rc="$1"
    local end_ts end_human elapsed status text payload job_id

    [ -n "${SLACK_WEBHOOK_URL:-}" ] || return 0

    end_ts="$(date +%s)"
    end_human="$(date -Is)"
    elapsed="$((end_ts - ${START_TS:-end_ts}))"
    job_id="${SLACK_JOB_ID:-${SLURM_JOB_ID:-?}}"

    if [ "$rc" -eq 0 ]; then
        status="COMPLETED"
    else
        status="FAILED"
    fi

    text="[$status] ${RUN_NAME:-job} (job ${job_id}) on $(hostname)
Config: ${RUN_NAME:-unknown}
Nodes:  ${node_count}
Start:  ${START_HUMAN:-unknown}
End:    ${end_human}
Elapsed: $(format_duration "$elapsed")
Exit code: ${rc}"

    if [ "$rc" -ne 0 ] && [ -n "${FAILED_CMD:-}" ]; then
        text="${text}
Failed at: ${FAILED_CMD}"
    fi

    if [ -n "${SLACK_TAG:-}" ]; then
        text="${SLACK_TAG} ${text}"
    fi

    payload="$(_slack_build_payload "$text")"

    curl_ssl_flag=()
    if [ "${SLACK_INSECURE:-0}" = "1" ]; then
        curl_ssl_flag+=(--insecure)
    fi

    { set +x; } 2>/dev/null
    curl -sS "${curl_ssl_flag[@]}" -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK_URL" >/dev/null || true
    { set -x; } 2>/dev/null
}
