#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
REPO=${REPO:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)}
BUILD=${BUILD:-"$REPO/build-gb200"}
MASTER_PORT=${MASTER_PORT:-50051}
MASTER_BIN=${MASTER_BIN:-"$BUILD/mooncake-store/src/mooncake_master"}
MASTER_LOG=${MASTER_LOG:-/tmp/mooncake_master.gb200.log}
MASTER_PID_FILE=${MASTER_PID_FILE:-/tmp/mooncake_master.gb200.pid}

if [ ! -x "$MASTER_BIN" ]; then
    printf 'missing mooncake_master: %s\n' "$MASTER_BIN" >&2
    printf 'run: sh scripts/gb200_build.sh\n' >&2
    exit 1
fi

if [ "${STOP_OLD_MASTER:-0}" = "1" ]; then
    if command -v pgrep >/dev/null 2>&1; then
        pids=$(pgrep -f "mooncake_master.*--port=$MASTER_PORT" || true)
        if [ -n "$pids" ]; then
            printf '%s\n' "$pids" | while IFS= read -r pid; do
                [ -n "$pid" ] || continue
                kill -9 "$pid" 2>/dev/null || true
            done
        fi
    fi
fi

: > "$MASTER_LOG"
export REPO BUILD MASTER_SERVER="127.0.0.1:$MASTER_PORT"
nohup sh "$REPO/scripts/gb200_env_exec.sh" \
    "$MASTER_BIN" \
    --rpc_address=0.0.0.0 \
    --port="$MASTER_PORT" \
    --logtostderr=1 \
    > "$MASTER_LOG" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$MASTER_PID_FILE"

sleep "${MASTER_STARTUP_SLEEP:-1}"
if ! kill -0 "$pid" 2>/dev/null; then
    printf 'mooncake_master exited early, log follows:\n' >&2
    cat "$MASTER_LOG" >&2 || true
    exit 1
fi

printf 'MASTER_PID=%s\n' "$pid"
printf 'MASTER_SERVER=%s\n' "127.0.0.1:$MASTER_PORT"
printf 'MASTER_LOG=%s\n' "$MASTER_LOG"
if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep ":$MASTER_PORT" || true
fi
