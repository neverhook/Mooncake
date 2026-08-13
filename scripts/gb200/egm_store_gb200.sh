#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
detected_repo_root="$(cd "${script_dir}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/gb200/egm_store_gb200.sh [--config PATH] ACTION [ARGS]

Run on both nodes:
  print-config       Print deterministic endpoints, paths, branch, and SHA
  build              Build Store/TE focused targets and run focused tests
  preflight          Validate GB200/Fabric/IMEX prerequisites on this node

Run on Node A:
  master-start       Start Master and embedded HTTP metadata service
  master-status      Check Master RPC, metadata, and admin endpoints
  master-stop        Stop the recorded Master process
  provider-start     Start EGM Provider and wait for Master publication
  provider-status    Check Provider PID and show readiness evidence
  provider-stop      Stop Provider and execute retryable Store cleanup
  diagnose           Capture endpoints, ports, and Master segment state

Run on Node B:
  consumer DEVICE     Run one HBM Consumer smoke/performance stream
  bench              Run concurrent HBM -> EGM -> HBM validation
  report READY LOG   Render PR Markdown using copied Node A readiness/log files

Any node:
  stop               Stop locally recorded Provider and Master processes
EOF
}

config_path="${detected_repo_root}/egm-store-gb200.conf"
if [[ "${1:-}" == "--config" ]]; then
  [[ $# -ge 3 ]] || { usage >&2; exit 2; }
  config_path="$2"
  shift 2
fi
action="${1:-}"
[[ -n "$action" ]] || { usage >&2; exit 2; }
shift
case "$action" in
  -h|--help|help) usage; exit 0 ;;
esac

if [[ ! -r "$config_path" ]]; then
  printf 'config is not readable: %s\n' "$config_path" >&2
  printf 'create it with: cp %s/egm_store_gb200.conf.example %s\n' \
    "$script_dir" "$config_path" >&2
  exit 2
fi

# Reset supported values before loading the trusted local config so unrelated
# inherited shell variables cannot silently change a validation run.
REPO_ROOT="$detected_repo_root"
BUILD_DIR="${detected_repo_root}/build-egm-store-gb200"
RESULT_ROOT="/tmp/mooncake-egm-gb200"
NODE_A_IP=""
NODE_B_IP=""
RUN_ID=""
MASTER_RPC_PORT=50051
MASTER_ADMIN_PORT=9003
METADATA_PORT=8079
PROVIDER_PORT=12345
CONSUMER_PORT_BASE=12400
EGM_POOL_SIZE="10 GB"
EGM_NUMA_NODES="auto"
MC_MAX_MR_SIZE_BYTES=157286400
MC_IMEX_DAEMON_EXTERNAL=1
MC_CUDART_LIBRARY=""
DEVICES="0,1,2,3"
PAYLOAD_SIZES="4096,1048576,16777216,134217728"
ITERATIONS=4
THRESHOLD_PAYLOAD_SIZE=134217728
MIN_PUT_GIB_S=0
MIN_GET_GIB_S=0
BUILD_JOBS=""
BUILD_UNIT_TESTS=0
CMAKE_EXTRA_ARGS=""
MASTER_START_TIMEOUT_SEC=30
PROVIDER_READINESS_TIMEOUT_SEC=120
EXTRA_LD_LIBRARY_PATH=""

# shellcheck disable=SC1090
source "$config_path"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_nonempty() {
  local name="$1" value="${!1:-}"
  [[ -n "$value" ]] || fail "$name must be set in $config_path"
  [[ "$value" != *[[:space:]/]* ]] || fail "$name contains whitespace or /: $value"
  [[ "$value" != "CHANGE_ME" && "$value" != *-CHANGE_ME ]] || \
    fail "$name still contains CHANGE_ME"
}
require_uint() {
  local name="$1" value="${!1:-}"
  [[ "$value" =~ ^[0-9]+$ ]] || fail "$name must be an integer: $value"
}
require_positive_uint() {
  require_uint "$1"
  (( ${!1} > 0 )) || fail "$1 must be greater than zero"
}
require_port() {
  require_positive_uint "$1"
  (( ${!1} <= 65535 )) || fail "$1 is outside 1..65535"
}
require_nonnegative_number() {
  local name="$1" value="${!1:-}"
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "$name must be nonnegative: $value"
}

require_nonempty NODE_A_IP
require_nonempty NODE_B_IP
require_nonempty RUN_ID
[[ "$NODE_A_IP" != *:* && "$NODE_B_IP" != *:* ]] || \
  fail 'NODE_A_IP and NODE_B_IP currently accept IPv4/DNS names, not IPv6 literals'
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsupported RUN_ID: $RUN_ID"
for name in MASTER_RPC_PORT MASTER_ADMIN_PORT METADATA_PORT PROVIDER_PORT \
    CONSUMER_PORT_BASE; do require_port "$name"; done
for name in MC_MAX_MR_SIZE_BYTES ITERATIONS THRESHOLD_PAYLOAD_SIZE \
    MASTER_START_TIMEOUT_SEC PROVIDER_READINESS_TIMEOUT_SEC; do
  require_positive_uint "$name"
done
(( ITERATIONS >= 2 )) || fail 'ITERATIONS must be at least two'
[[ "$MC_IMEX_DAEMON_EXTERNAL" == 0 || "$MC_IMEX_DAEMON_EXTERNAL" == 1 ]] || \
  fail 'MC_IMEX_DAEMON_EXTERNAL must be 0 or 1'
require_nonnegative_number MIN_PUT_GIB_S
require_nonnegative_number MIN_GET_GIB_S
[[ "$EGM_NUMA_NODES" == auto || "$EGM_NUMA_NODES" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
  fail 'EGM_NUMA_NODES must be auto or comma-separated nonnegative IDs'
[[ "$DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail 'invalid DEVICES list'
[[ "$PAYLOAD_SIZES" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || \
  fail 'invalid PAYLOAD_SIZES list'
seen=","; IFS=, read -r -a device_list <<<"$DEVICES"
for device in "${device_list[@]}"; do
  [[ "$seen" != *",${device},"* ]] || fail "duplicate GPU $device"
  seen+="${device},"
  (( CONSUMER_PORT_BASE + device <= 65535 )) || fail "GPU $device port overflows"
done
IFS=, read -r -a payload_list <<<"$PAYLOAD_SIZES"
for size in "${payload_list[@]}"; do
  (( size <= 128 * 1024 * 1024 )) || fail "payload exceeds 128 MiB: $size"
done
[[ -z "$BUILD_JOBS" || "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || \
  fail 'BUILD_JOBS must be empty or positive'
[[ "$BUILD_UNIT_TESTS" == 0 || "$BUILD_UNIT_TESTS" == 1 ]] || \
  fail 'BUILD_UNIT_TESTS must be 0 or 1'

SOURCE_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SOURCE_BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"
MASTER_SERVER="${NODE_A_IP}:${MASTER_RPC_PORT}"
METADATA_BASE_URL="http://${NODE_A_IP}:${METADATA_PORT}"
METADATA_SERVER="${METADATA_BASE_URL}/metadata"
MASTER_ADMIN_URL="http://${NODE_A_IP}:${MASTER_ADMIN_PORT}"
PROVIDER_HOSTNAME="${NODE_A_IP}:${PROVIDER_PORT}"
RESULT_DIR="${RESULT_ROOT%/}/${RUN_ID}"
MASTER_BIN="${BUILD_DIR}/mooncake-store/src/mooncake_master"
MASTER_PID_FILE="${RESULT_DIR}/master.pid"
MASTER_LOG="${RESULT_DIR}/master.log"
PROVIDER_PID_FILE="${RESULT_DIR}/provider.pid"
PROVIDER_LOG="${RESULT_DIR}/provider.log"
PROVIDER_READY_FILE="${RESULT_DIR}/provider.ready.json"
BENCH_LOG="${RESULT_DIR}/bench.jsonl"
BENCH_STDERR_LOG="${RESULT_DIR}/bench.stderr.log"

python_path="${BUILD_DIR}/mooncake-integration:${script_dir}"
library_paths=(
  "${BUILD_DIR}/mooncake-common/etcd"
  "${BUILD_DIR}/mooncake-common"
  "${BUILD_DIR}/mooncake-common/src"
  "${BUILD_DIR}/mooncake-store/src"
  "${BUILD_DIR}/mooncake-transfer-engine/src"
  "${BUILD_DIR}/mooncake-integration"
  "/usr/local/cuda/lib64"
  "/usr/local/cuda/targets/sbsa-linux/lib"
  "/usr/local/cuda/targets/aarch64-linux/lib"
)
ld_library_path="$(IFS=:; printf '%s' "${library_paths[*]}")"
[[ -z "$EXTRA_LD_LIBRARY_PATH" ]] || \
  ld_library_path="${ld_library_path}:${EXTRA_LD_LIBRARY_PATH}"
common_env=(
  "PYTHONPATH=${python_path}"
  "LD_LIBRARY_PATH=${ld_library_path}"
  "MC_MS_AUTO_DISC=0"
  "MC_FORCE_MNNVL=1"
  "MC_MAX_MR_SIZE=${MC_MAX_MR_SIZE_BYTES}"
  "MC_IMEX_DAEMON_EXTERNAL=${MC_IMEX_DAEMON_EXTERNAL}"
  "MOONCAKE_TE_META_DATA_SERVER=${METADATA_SERVER}"
)
[[ -z "$MC_CUDART_LIBRARY" ]] || common_env+=("MC_CUDART_LIBRARY=${MC_CUDART_LIBRARY}")

run_env() {
  local bind_address="$1"; shift
  env -u MC_USE_NVLINK_IPC -u MC_RPC_PROTOCOL -u MC_INTRANODE_NVLINK \
    -u MC_FORCE_TCP -u MC_FORCE_HCA -u MOONCAKE_CONFIG_PATH \
    -u MC_CUDART_LIBRARY "${common_env[@]}" \
    "MC_TCP_BIND_ADDRESS=${bind_address}" "$@"
}
run_common() {
  env -u MC_USE_NVLINK_IPC -u MC_RPC_PROTOCOL -u MC_INTRANODE_NVLINK \
    -u MC_FORCE_TCP -u MC_FORCE_HCA -u MOONCAKE_CONFIG_PATH \
    -u MC_CUDART_LIBRARY -u MC_TCP_BIND_ADDRESS "${common_env[@]}" "$@"
}
pid_matches() {
  local pid="$1" expected="$2" command_line=""
  if [[ -r "/proc/${pid}/cmdline" ]]; then
    command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  elif command -v ps >/dev/null 2>&1; then
    command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  fi
  [[ "$command_line" == *"$expected"* ]]
}
pid_live() {
  local file="$1" expected="$2" pid
  [[ -r "$file" ]] || return 1
  read -r pid <"$file"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && pid_matches "$pid" "$expected"
}
pid_exists() {
  local file="$1" pid
  [[ -r "$file" ]] || return 1
  read -r pid <"$file"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}
wait_for_pid_command() {
  local file="$1" expected="$2" timeout="$3"
  local deadline=$((SECONDS + timeout))
  while pid_exists "$file"; do
    pid_live "$file" "$expected" && return 0
    (( SECONDS < deadline )) || return 1
    sleep 0.1
  done
  return 1
}
stop_pid() {
  local label="$1" file="$2" expected="$3" pid deadline
  if ! pid_live "$file" "$expected"; then
    rm -f "$file"
    printf '%s is not running\n' "$label"
    return 0
  fi
  read -r pid <"$file"; kill "$pid"; deadline=$((SECONDS + 30))
  while pid_live "$file" "$expected" && (( SECONDS < deadline )); do sleep 1; done
  pid_live "$file" "$expected" && fail "$label PID $pid did not stop"
  rm -f "$file"
  printf '%s stopped: pid=%s\n' "$label" "$pid"
}
tcp_reachable() {
  python3 - "$1" "$2" <<'PY'
import socket
import sys
try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
PY
}
wait_for_tcp() {
  local host="$1" port="$2" timeout="$3"
  local deadline=$((SECONDS + timeout))
  until tcp_reachable "$host" "$port"; do
    (( SECONDS < deadline )) || return 1
    sleep 1
  done
}
master_remote_status() {
  tcp_reachable "$NODE_A_IP" "$MASTER_RPC_PORT" || fail "Master RPC unreachable: $MASTER_SERVER"
  [[ "$(curl -fsS --max-time 5 "${METADATA_BASE_URL}/health")" == OK ]] || \
    fail 'metadata health check failed'
  curl -fsS --max-time 5 "${MASTER_ADMIN_URL}/health" >/dev/null || \
    fail 'Master admin health check failed'
  local key encoded url expected actual
  key="egm-gb200/${RUN_ID}/metadata-probe"
  encoded="$(python3 - "$key" <<'PY'
import sys
import urllib.parse
print(urllib.parse.quote(sys.argv[1], safe=""))
PY
)"
  url="${METADATA_SERVER}?key=${encoded}"
  expected="{\"run_id\":\"${RUN_ID}\"}"
  curl -fsS --max-time 5 -X PUT --data-binary "$expected" "$url" >/dev/null
  actual="$(curl -fsS --max-time 5 "$url")"
  [[ "$actual" == "$expected" ]] || fail "metadata readback mismatch: $actual"
  curl -fsS --max-time 5 -X DELETE "$url" >/dev/null
  printf 'Master endpoints PASS: %s %s %s\n' \
    "$MASTER_SERVER" "$METADATA_SERVER" "$MASTER_ADMIN_URL"
}
diagnose() {
  printf 'RUN_ID=%s SOURCE_SHA=%s\n' "$RUN_ID" "$SOURCE_SHA"
  printf 'PROVIDER=%s MASTER=%s METADATA=%s\n' \
    "$PROVIDER_HOSTNAME" "$MASTER_SERVER" "$METADATA_SERVER"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null || true
  fi
  for path in health get_all_segments get_segments_detail; do
    printf '\n=== %s ===\n' "$path"
    curl -sS --max-time 5 "${MASTER_ADMIN_URL}/${path}" || true
    printf '\n'
  done
  printf '\n=== provider segment ===\n'
  curl --globoff -sS --max-time 5 \
    "${MASTER_ADMIN_URL}/query_segment?segment=${PROVIDER_HOSTNAME}" || true
  printf '\n'
}
verify_provider_unpublished() {
  local segments matches status
  segments="$(curl -fsS --max-time 5 "${MASTER_ADMIN_URL}/get_all_segments")" || \
    fail 'could not query Master after Provider cleanup'
  matches="$(printf '%s\n' "$segments" | grep -Fxc "$PROVIDER_HOSTNAME" || true)"
  if (( matches == 0 )); then status=PASS; else status=FAIL; fi
  python3 - "$status" "$RUN_ID" "$SOURCE_SHA" "$PROVIDER_HOSTNAME" "$matches" <<'PY' \
    | tee -a "$PROVIDER_LOG"
import json
import sys
print(json.dumps({
    "event": "provider_unpublished",
    "status": sys.argv[1],
    "run_id": sys.argv[2],
    "source_sha": sys.argv[3],
    "local_hostname": sys.argv[4],
    "remaining_chunks": int(sys.argv[5]),
}, sort_keys=True))
PY
  [[ "$status" == PASS ]] || fail "Master still contains $matches Provider chunks"
}
verify_provider_cleanup() {
  python3 - "$PROVIDER_LOG" "$RUN_ID" "$SOURCE_SHA" <<'PY'
import json
import pathlib
import sys

records = []
for line in pathlib.Path(sys.argv[1]).read_text(errors="replace").splitlines():
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if (
        isinstance(value, dict)
        and value.get("event") == "provider_cleanup"
        and value.get("run_id") == sys.argv[2]
        and value.get("source_sha") == sys.argv[3]
    ):
        records.append(value)
if len(records) != 1 or records[0].get("status") != "PASS":
    raise SystemExit("Provider cleanup is not exactly one PASS record")
print(json.dumps(records[0], sort_keys=True))
PY
}

case "$action" in
  print-config)
    printf 'CONFIG=%s\nREPO_ROOT=%s\nBUILD_DIR=%s\nRESULT_DIR=%s\n' \
      "$config_path" "$REPO_ROOT" "$BUILD_DIR" "$RESULT_DIR"
    printf 'SOURCE_BRANCH=%s\nSOURCE_SHA=%s\nRUN_ID=%s\n' \
      "$SOURCE_BRANCH" "$SOURCE_SHA" "$RUN_ID"
    printf 'NODE_A_IP=%s\nNODE_B_IP=%s\nMASTER_SERVER=%s\n' \
      "$NODE_A_IP" "$NODE_B_IP" "$MASTER_SERVER"
    printf 'METADATA_SERVER=%s\nMASTER_ADMIN_URL=%s\nPROVIDER_HOSTNAME=%s\n' \
      "$METADATA_SERVER" "$MASTER_ADMIN_URL" "$PROVIDER_HOSTNAME"
    printf 'EGM_POOL_SIZE=%s\nEGM_NUMA_NODES=%s\nDEVICES=%s\nPAYLOAD_SIZES=%s\n' \
      "$EGM_POOL_SIZE" "$EGM_NUMA_NODES" "$DEVICES" "$PAYLOAD_SIZES"
    printf 'ITERATIONS=%s\nMIN_PUT_GIB_S=%s\nMIN_GET_GIB_S=%s\n' \
      "$ITERATIONS" "$MIN_PUT_GIB_S" "$MIN_GET_GIB_S"
    printf 'BUILD_UNIT_TESTS=%s\nMC_IMEX_DAEMON_EXTERNAL=%s\n' \
      "$BUILD_UNIT_TESTS" "$MC_IMEX_DAEMON_EXTERNAL"
    ;;
  build)
    build_env=("BUILD_DIR=${BUILD_DIR}" "BUILD_UNIT_TESTS=${BUILD_UNIT_TESTS}"
      "CMAKE_EXTRA_ARGS=${CMAKE_EXTRA_ARGS}")
    [[ -z "$BUILD_JOBS" ]] || build_env+=("JOBS=${BUILD_JOBS}")
    run_common env "${build_env[@]}" "${script_dir}/egm_store_build.sh" "$@"
    ;;
  preflight)
    mkdir -p "$RESULT_DIR"
    run_common "${script_dir}/egm_store_preflight.sh" "$@" | \
      tee "${RESULT_DIR}/preflight-$(hostname).log"
    ;;
  master-start)
    mkdir -p "$RESULT_DIR"
    [[ -x "$MASTER_BIN" ]] || fail "Master binary missing: $MASTER_BIN"
    pid_live "$MASTER_PID_FILE" "$MASTER_BIN" && fail 'Master is already running'
    for port in "$MASTER_RPC_PORT" "$MASTER_ADMIN_PORT" "$METADATA_PORT"; do
      if tcp_reachable 127.0.0.1 "$port" || tcp_reachable "$NODE_A_IP" "$port"; then
        fail "port $port is occupied"
      fi
    done
    : >"$MASTER_LOG"
    nohup env -u MC_USE_NVLINK_IPC -u MC_RPC_PROTOCOL -u MC_INTRANODE_NVLINK \
      -u MC_FORCE_TCP -u MC_FORCE_HCA -u MOONCAKE_CONFIG_PATH \
      -u MC_CUDART_LIBRARY "${common_env[@]}" "MC_TCP_BIND_ADDRESS=${NODE_A_IP}" \
      "$MASTER_BIN" --rpc_address=0.0.0.0 --rpc_port="$MASTER_RPC_PORT" \
      --enable_http_metadata_server=true --http_metadata_server_host=0.0.0.0 \
      --http_metadata_server_port="$METADATA_PORT" \
      --metrics_port="$MASTER_ADMIN_PORT" --logtostderr=1 \
      >"$MASTER_LOG" 2>&1 &
    printf '%s\n' "$!" >"$MASTER_PID_FILE"
    if ! wait_for_tcp 127.0.0.1 "$MASTER_RPC_PORT" "$MASTER_START_TIMEOUT_SEC" ||
       ! wait_for_tcp 127.0.0.1 "$METADATA_PORT" "$MASTER_START_TIMEOUT_SEC" ||
       ! wait_for_tcp 127.0.0.1 "$MASTER_ADMIN_PORT" "$MASTER_START_TIMEOUT_SEC"; then
      tail -n 100 "$MASTER_LOG" >&2 || true
      stop_pid Master "$MASTER_PID_FILE" "$MASTER_BIN" || true
      fail 'Master did not open all configured ports'
    fi
    master_remote_status
    printf 'Master started: pid=%s log=%s\n' "$(<"$MASTER_PID_FILE")" "$MASTER_LOG"
    ;;
  master-status)
    pid_live "$MASTER_PID_FILE" "$MASTER_BIN" || fail 'recorded Master PID is not live'
    master_remote_status
    ;;
  master-stop) stop_pid Master "$MASTER_PID_FILE" "$MASTER_BIN" ;;
  provider-start)
    mkdir -p "$RESULT_DIR"
    master_remote_status
    compgen -G "${BUILD_DIR}/mooncake-integration/store*.so" >/dev/null || \
      fail "Store Python binding missing under ${BUILD_DIR}/mooncake-integration"
    pid_live "$PROVIDER_PID_FILE" egm_store_provider.py && fail 'Provider is already running'
    rm -f "$PROVIDER_READY_FILE" "$PROVIDER_PID_FILE"
    : >"$PROVIDER_LOG"
    nohup env -u MC_USE_NVLINK_IPC -u MC_RPC_PROTOCOL -u MC_INTRANODE_NVLINK \
      -u MC_FORCE_TCP -u MC_FORCE_HCA -u MOONCAKE_CONFIG_PATH \
      -u MC_CUDART_LIBRARY "${common_env[@]}" "MC_TCP_BIND_ADDRESS=${NODE_A_IP}" \
      python3 "${script_dir}/egm_store_provider.py" \
      --local-hostname "$PROVIDER_HOSTNAME" --metadata-server "$METADATA_SERVER" \
      --master-server "$MASTER_SERVER" --master-admin-url "$MASTER_ADMIN_URL" \
      --pool-size "$EGM_POOL_SIZE" --numa-nodes "$EGM_NUMA_NODES" \
      --run-id "$RUN_ID" --source-sha "$SOURCE_SHA" \
      --ready-file "$PROVIDER_READY_FILE" \
      --readiness-timeout-sec "$PROVIDER_READINESS_TIMEOUT_SEC" \
      >"$PROVIDER_LOG" 2>&1 &
    printf '%s\n' "$!" >"$PROVIDER_PID_FILE"
    if ! wait_for_pid_command "$PROVIDER_PID_FILE" egm_store_provider.py 5; then
      tail -n 160 "$PROVIDER_LOG" >&2 || true
      if pid_exists "$PROVIDER_PID_FILE"; then
        kill "$(<"$PROVIDER_PID_FILE")" 2>/dev/null || true
      fi
      rm -f "$PROVIDER_PID_FILE"
      fail 'Provider exited during launcher handoff'
    fi
    deadline=$((SECONDS + PROVIDER_READINESS_TIMEOUT_SEC + 5))
    while [[ ! -s "$PROVIDER_READY_FILE" ]] &&
        pid_live "$PROVIDER_PID_FILE" egm_store_provider.py &&
        (( SECONDS < deadline )); do sleep 1; done
    if [[ -s "$PROVIDER_READY_FILE" ]] && pid_live "$PROVIDER_PID_FILE" egm_store_provider.py; then
      cat "$PROVIDER_READY_FILE"
      printf 'Provider started: pid=%s log=%s\n' "$(<"$PROVIDER_PID_FILE")" "$PROVIDER_LOG"
    else
      diagnose >"${RESULT_DIR}/provider-start-diagnose.log" 2>&1 || true
      tail -n 160 "$PROVIDER_LOG" >&2 || true
      stop_pid Provider "$PROVIDER_PID_FILE" egm_store_provider.py || true
      fail 'Provider did not become ready'
    fi
    ;;
  provider-status)
    pid_live "$PROVIDER_PID_FILE" egm_store_provider.py || fail 'Provider PID is not live'
    [[ -s "$PROVIDER_READY_FILE" ]] || fail 'Provider readiness file is missing'
    cat "$PROVIDER_READY_FILE"
    diagnose
    ;;
  provider-stop)
    stop_pid Provider "$PROVIDER_PID_FILE" egm_store_provider.py
    verify_provider_cleanup
    verify_provider_unpublished
    ;;
  diagnose) diagnose ;;
  consumer)
    [[ $# -ge 1 ]] || fail 'consumer requires a GPU device number'
    device="$1"; shift
    [[ "$device" =~ ^[0-9]+$ ]] || fail "invalid GPU device: $device"
    (( CONSUMER_PORT_BASE + device <= 65535 )) || fail 'Consumer port overflows'
    master_remote_status
    mkdir -p "$RESULT_DIR"
    run_env "$NODE_B_IP" python3 "${script_dir}/egm_store_consumer.py" \
      --local-hostname "${NODE_B_IP}:$((CONSUMER_PORT_BASE + device))" \
      --metadata-server "$METADATA_SERVER" --master-server "$MASTER_SERVER" \
      --device "$device" --payload-size "$THRESHOLD_PAYLOAD_SIZE" \
      --iterations "$ITERATIONS" --run-id "$RUN_ID" --source-sha "$SOURCE_SHA" \
      "$@" >"${RESULT_DIR}/consumer-gpu-${device}.jsonl" \
      2>"${RESULT_DIR}/consumer-gpu-${device}.stderr.log"
    cat "${RESULT_DIR}/consumer-gpu-${device}.jsonl"
    ;;
  bench)
    master_remote_status
    mkdir -p "$RESULT_DIR"
    run_env "$NODE_B_IP" python3 "${script_dir}/egm_store_bench.py" \
      --node-b-ip "$NODE_B_IP" --consumer-port-base "$CONSUMER_PORT_BASE" \
      --metadata-server "$METADATA_SERVER" --master-server "$MASTER_SERVER" \
      --devices "$DEVICES" --payload-sizes "$PAYLOAD_SIZES" \
      --iterations "$ITERATIONS" --run-id "$RUN_ID" --source-sha "$SOURCE_SHA" \
      --threshold-payload-size "$THRESHOLD_PAYLOAD_SIZE" \
      --min-put-gib-s "$MIN_PUT_GIB_S" --min-get-gib-s "$MIN_GET_GIB_S" \
      "$@" >"$BENCH_LOG" 2>"$BENCH_STDERR_LOG"
    cat "$BENCH_LOG"
    printf 'Benchmark PASS: jsonl=%s stderr=%s\n' "$BENCH_LOG" "$BENCH_STDERR_LOG"
    ;;
  report)
    [[ $# -eq 2 ]] || fail 'report requires copied Node A provider.ready.json and provider.log paths'
    provider_evidence="$1"
    provider_log_evidence="$2"
    [[ -s "$provider_evidence" ]] || fail "Provider evidence missing: $provider_evidence"
    [[ -s "$provider_log_evidence" ]] || fail "Provider log missing: $provider_log_evidence"
    [[ -s "$BENCH_LOG" ]] || fail "Benchmark evidence missing: $BENCH_LOG"
    python3 "${script_dir}/egm_store_report.py" \
      --provider-ready "$provider_evidence" --provider-log "$provider_log_evidence" \
      --benchmark-log "$BENCH_LOG" \
      --branch "$SOURCE_BRANCH" | tee "${RESULT_DIR}/pr-report.md"
    ;;
  stop)
    stop_pid Provider "$PROVIDER_PID_FILE" egm_store_provider.py || true
    stop_pid Master "$MASTER_PID_FILE" "$MASTER_BIN" || true
    ;;
  *) usage >&2; fail "unknown action: $action" ;;
esac
