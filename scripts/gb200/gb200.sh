#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
detected_repo_root="$(cd "${script_dir}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/gb200/gb200.sh [--config PATH] ACTION [ARGS]

Actions:
  print-config            Print the resolved, deterministic process config
  build                   Build and run configured GB200 validation tests
  preflight               Run strict GB200 Fabric/RDMA preflight
  master-start            Start Master + embedded HTTP metadata on Node A
  master-status           Verify RPC, metadata CRUD, admin API, and PID
  master-stop             Stop the Master started by this wrapper
  provider-start          Start HOST_NUMA Provider and wait for readiness
  provider-status         Show Provider PID/readiness and Master publication
  provider-stop           Stop the Provider started by this wrapper
  diagnose                Capture ports and all Master segment query responses
  consumer DEVICE [ARGS]  Run no-Torch HBM Consumer on one Node B GPU
  bench [ARGS]            Run concurrent no-Torch HBM benchmark on Node B
  stop                    Stop Provider, then Master, using recorded PID files

Default config: <repo>/gb200.conf
Create it with:
  cp scripts/gb200/gb200.conf.example gb200.conf
EOF
}

config_path="${detected_repo_root}/gb200.conf"
if [[ "${1:-}" == "--config" ]]; then
  [[ $# -ge 3 ]] || { usage >&2; exit 2; }
  config_path="$2"
  shift 2
fi
action="${1:-}"
[[ -n "$action" ]] || { usage >&2; exit 2; }
shift

case "$action" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

if [[ ! -r "$config_path" ]]; then
  printf 'GB200 config is not readable: %s\n' "$config_path" >&2
  printf 'Create it with: cp %s/gb200.conf.example %s\n' \
    "$script_dir" "$config_path" >&2
  exit 2
fi

# Reset every supported value before loading the config. Inherited variables
# therefore cannot silently alter a later invocation from another shell.
REPO_ROOT="$detected_repo_root"
BUILD_DIR="${detected_repo_root}/build-nvlink-host-numa"
RESULT_ROOT="/tmp/mooncake-gb200"
NODE_A_IP=""
NODE_B_IP=""
RUN_ID=""
MASTER_RPC_PORT=50051
MASTER_ADMIN_PORT=9003
METADATA_PORT=8079
PROVIDER_PORT=12345
CONSUMER_PORT_BASE=12400
GLOBAL_SEGMENT_SIZE="600 GB"
LOCAL_BUFFER_SIZE="0"
HOST_NUMA_NODES="auto"
MC_MAX_MR_SIZE_BYTES=161061273600
MC_IMEX_DAEMON_EXTERNAL=0
MC_CUDART_LIBRARY=""
DEVICES="0,1,2,3"
PAYLOAD_SIZES="4096,1048576,16777216"
ITERATIONS=4
SINGLE_PAYLOAD_SIZE=16777216
RUN_HARDWARE_TESTS=1
SKIP_PREFLIGHT=0
BUILD_JOBS=""
CMAKE_EXTRA_ARGS=""
MASTER_START_TIMEOUT_SEC=30
PROVIDER_READINESS_TIMEOUT_SEC=120
PROVIDER_DIAGNOSTIC_DELAY_SEC=5
EXTRA_LD_LIBRARY_PATH=""

# The config is a trusted local shell assignment file. shellcheck disable=SC1090
source "$config_path"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_nonempty() {
  local name="$1" value="${!1:-}"
  [[ -n "$value" ]] || fail "$name must be set in $config_path"
  [[ "$value" != *[[:space:]/]* ]] || fail "$name contains whitespace or '/': $value"
  [[ "$value" != "CHANGE_ME" && "$value" != *-CHANGE_ME ]] || \
    fail "$name still contains CHANGE_ME in $config_path"
}

require_uint() {
  local name="$1" value="${!1:-}"
  [[ "$value" =~ ^[0-9]+$ ]] || fail "$name must be an integer, got: $value"
}

require_positive_uint() {
  local name="$1" value="${!1:-}"
  require_uint "$name"
  (( value > 0 )) || fail "$name must be greater than zero"
}

require_port() {
  local name="$1" value="${!1:-}"
  require_uint "$name"
  (( value >= 1 && value <= 65535 )) || fail "$name is outside 1..65535: $value"
}

require_bool() {
  local name="$1" value="${!1:-}"
  [[ "$value" == 0 || "$value" == 1 ]] || fail "$name must be 0 or 1, got: $value"
}

require_nonempty NODE_A_IP
require_nonempty NODE_B_IP
[[ "$NODE_A_IP" != *:* && "$NODE_B_IP" != *:* ]] || \
  fail "NODE_A_IP and NODE_B_IP currently support IPv4 addresses or DNS names, not IPv6 literals"
require_nonempty RUN_ID
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "RUN_ID contains unsupported characters: $RUN_ID"
for name in MASTER_RPC_PORT MASTER_ADMIN_PORT METADATA_PORT PROVIDER_PORT CONSUMER_PORT_BASE; do
  require_port "$name"
done
for name in ITERATIONS SINGLE_PAYLOAD_SIZE MC_MAX_MR_SIZE_BYTES \
    MASTER_START_TIMEOUT_SEC PROVIDER_READINESS_TIMEOUT_SEC; do
  require_positive_uint "$name"
done
require_uint PROVIDER_DIAGNOSTIC_DELAY_SEC
(( ITERATIONS >= 2 )) || fail "ITERATIONS must be at least 2"
(( PROVIDER_DIAGNOSTIC_DELAY_SEC < PROVIDER_READINESS_TIMEOUT_SEC )) || \
  fail "PROVIDER_DIAGNOSTIC_DELAY_SEC must be below PROVIDER_READINESS_TIMEOUT_SEC"
for name in RUN_HARDWARE_TESTS SKIP_PREFLIGHT MC_IMEX_DAEMON_EXTERNAL; do
  require_bool "$name"
done
if [[ -n "$BUILD_JOBS" ]]; then
  require_positive_uint BUILD_JOBS
fi
(( CONSUMER_PORT_BASE % 10 == 0 )) || \
  fail "CONSUMER_PORT_BASE must end in 0 so benchmark ports can append GPU IDs"
[[ "$MASTER_RPC_PORT" != "$MASTER_ADMIN_PORT" && \
   "$MASTER_RPC_PORT" != "$METADATA_PORT" && \
   "$MASTER_ADMIN_PORT" != "$METADATA_PORT" ]] || \
  fail "MASTER_RPC_PORT, MASTER_ADMIN_PORT, and METADATA_PORT must be distinct"
[[ "$DEVICES" =~ ^[0-9](,[0-9])*$ ]] || \
  fail "DEVICES must be a comma-separated unique list of GPU IDs 0..9"
seen_devices=","
IFS=, read -r -a configured_devices <<<"$DEVICES"
for device in "${configured_devices[@]}"; do
  [[ "$seen_devices" != *",${device},"* ]] || fail "DEVICES contains duplicate GPU $device"
  seen_devices+="${device},"
  (( CONSUMER_PORT_BASE + device <= 65535 )) || \
    fail "consumer port for GPU $device exceeds 65535"
done
[[ "$PAYLOAD_SIZES" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || \
  fail "PAYLOAD_SIZES must be a comma-separated list of positive integers"

MASTER_SERVER="${NODE_A_IP}:${MASTER_RPC_PORT}"
METADATA_BASE_URL="http://${NODE_A_IP}:${METADATA_PORT}"
METADATA_SERVER="${METADATA_BASE_URL}/metadata"
MASTER_ADMIN_URL="http://${NODE_A_IP}:${MASTER_ADMIN_PORT}"
PROVIDER_HOSTNAME="${NODE_A_IP}:${PROVIDER_PORT}"
CONSUMER_HOSTNAME_PREFIX="${NODE_B_IP}:$((CONSUMER_PORT_BASE / 10))"
RESULT_DIR="${RESULT_ROOT%/}/${RUN_ID}"
MASTER_BIN="${BUILD_DIR}/mooncake-store/src/mooncake_master"
FABRIC_PROBE="${BUILD_DIR}/mooncake-transfer-engine/tests/nvlink_host_numa_fabric_test"
MASTER_PID_FILE="${RESULT_DIR}/master.pid"
MASTER_LOG="${RESULT_DIR}/master.log"
PROVIDER_PID_FILE="${RESULT_DIR}/provider.pid"
PROVIDER_LOG="${RESULT_DIR}/provider.log"
PROVIDER_READY_FILE="${RESULT_DIR}/provider.ready.json"

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
if [[ -n "$EXTRA_LD_LIBRARY_PATH" ]]; then
  ld_library_path="${ld_library_path}:${EXTRA_LD_LIBRARY_PATH}"
fi

common_env=(
  "PYTHONPATH=${python_path}"
  "LD_LIBRARY_PATH=${ld_library_path}"
  "MC_MS_AUTO_DISC=0"
  "MC_FORCE_MNNVL=1"
  "MC_MAX_MR_SIZE=${MC_MAX_MR_SIZE_BYTES}"
  "MC_IMEX_DAEMON_EXTERNAL=${MC_IMEX_DAEMON_EXTERNAL}"
  "MC_STORE_CLIENT_METRIC_BANDWIDTH=1"
  "MOONCAKE_TE_META_DATA_SERVER=${METADATA_SERVER}"
)
if [[ -n "$MC_CUDART_LIBRARY" ]]; then
  common_env+=("MC_CUDART_LIBRARY=${MC_CUDART_LIBRARY}")
fi

run_env() {
  local bind_address="$1"
  shift
  env -u MC_USE_NVLINK_IPC -u MC_RPC_PROTOCOL -u MC_INTRANODE_NVLINK \
    -u MC_FORCE_TCP -u MC_FORCE_HCA -u MOONCAKE_CONFIG_PATH \
    -u MC_CUDART_LIBRARY \
    "${common_env[@]}" "MC_TCP_BIND_ADDRESS=${bind_address}" "$@"
}

run_common() {
  env -u MC_USE_NVLINK_IPC -u MC_RPC_PROTOCOL -u MC_INTRANODE_NVLINK \
    -u MC_FORCE_TCP -u MC_FORCE_HCA -u MOONCAKE_CONFIG_PATH \
    -u MC_CUDART_LIBRARY -u MC_TCP_BIND_ADDRESS \
    "${common_env[@]}" "$@"
}

build_env=("CMAKE_EXTRA_ARGS=${CMAKE_EXTRA_ARGS}")
if [[ -n "$BUILD_JOBS" ]]; then
  build_env+=("JOBS=${BUILD_JOBS}")
fi

pid_command_matches() {
  local pid="$1" expected="$2" command=""
  if [[ -r "/proc/${pid}/cmdline" ]]; then
    command="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  elif command -v ps >/dev/null 2>&1; then
    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  fi
  [[ "$command" == *"$expected"* ]]
}

pid_is_live() {
  local file="$1" expected="$2" pid
  [[ -r "$file" ]] || return 1
  read -r pid <"$file"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && \
    pid_command_matches "$pid" "$expected"
}

show_port_owner() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | awk -v suffix=":${port}" '$4 ~ suffix "$" {print}' || true
  else
    printf 'ss is unavailable; cannot print owner for port %s\n' "$port"
  fi
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
  local host="$1" port="$2" timeout="$3" deadline=$((SECONDS + timeout))
  until tcp_reachable "$host" "$port"; do
    (( SECONDS < deadline )) || return 1
    sleep 1
  done
}

metadata_crud() {
  local key="gb200-wrapper/${RUN_ID}/probe" encoded url expected actual
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
  [[ "$actual" == "$expected" ]] || fail "metadata CRUD readback mismatch: $actual"
  curl -fsS --max-time 5 -X DELETE "$url" >/dev/null
}

master_remote_status() {
  tcp_reachable "$NODE_A_IP" "$MASTER_RPC_PORT" || fail "Master RPC is unreachable at $MASTER_SERVER"
  [[ "$(curl -fsS --max-time 5 "${METADATA_BASE_URL}/health")" == "OK" ]] || \
    fail "metadata health check failed at ${METADATA_BASE_URL}/health"
  curl -fsS --max-time 5 "${MASTER_ADMIN_URL}/health" >/dev/null || \
    fail "Master admin is unreachable at ${MASTER_ADMIN_URL}/health"
  metadata_crud
  printf 'Master endpoints PASS: rpc=%s metadata=%s admin=%s\n' \
    "$MASTER_SERVER" "$METADATA_SERVER" "$MASTER_ADMIN_URL"
}

print_http() {
  local label="$1" url="$2" tmp status
  tmp="$(mktemp)"
  status="$(curl --globoff -sS --max-time 5 -o "$tmp" -w '%{http_code}' "$url" || true)"
  printf '\n=== %s ===\nGET %s\nHTTP %s\n' "$label" "$url" "${status:-curl_failed}"
  cat "$tmp"
  printf '\n'
  rm -f "$tmp"
}

diagnose() {
  printf '=== resolved endpoints ===\n'
  printf 'RUN_ID=%s\nPROVIDER_HOSTNAME=%s\nMETADATA_SERVER=%s\nMASTER_SERVER=%s\nMASTER_ADMIN_URL=%s\n' \
    "$RUN_ID" "$PROVIDER_HOSTNAME" "$METADATA_SERVER" "$MASTER_SERVER" "$MASTER_ADMIN_URL"
  printf '\n=== ports and owning processes ===\n'
  for port in "$MASTER_RPC_PORT" "$METADATA_PORT" "$MASTER_ADMIN_PORT" "$PROVIDER_PORT"; do
    printf -- '-- port %s --\n' "$port"
    show_port_owner "$port"
  done
  print_http 'metadata health' "${METADATA_BASE_URL}/health"
  print_http 'admin health' "${MASTER_ADMIN_URL}/health"
  print_http 'all segments' "${MASTER_ADMIN_URL}/get_all_segments"
  print_http 'segment details' "${MASTER_ADMIN_URL}/get_segments_detail"
  print_http 'raw colon query' "${MASTER_ADMIN_URL}/query_segment?segment=${PROVIDER_HOSTNAME}"
  local encoded
  encoded="$(python3 - "$PROVIDER_HOSTNAME" <<'PY'
import sys
import urllib.parse
print(urllib.parse.urlencode({"segment": sys.argv[1]}))
PY
)"
  print_http 'URL-encoded query' "${MASTER_ADMIN_URL}/query_segment?${encoded}"
}

stop_pid_file() {
  local label="$1" file="$2" expected="$3" pid deadline
  if ! pid_is_live "$file" "$expected"; then
    rm -f "$file"
    printf '%s is not running under %s\n' "$label" "$file"
    return 0
  fi
  read -r pid <"$file"
  kill "$pid"
  deadline=$((SECONDS + 15))
  while pid_is_live "$file" "$expected" && (( SECONDS < deadline )); do sleep 1; done
  if pid_is_live "$file" "$expected"; then
    fail "$label PID $pid did not stop after SIGTERM"
  fi
  rm -f "$file"
  printf '%s stopped: pid=%s\n' "$label" "$pid"
}

case "$action" in
  print-config)
    printf 'CONFIG=%s\nREPO_ROOT=%s\nBUILD_DIR=%s\nRESULT_DIR=%s\n' \
      "$config_path" "$REPO_ROOT" "$BUILD_DIR" "$RESULT_DIR"
    printf 'RUN_ID=%s\nNODE_A_IP=%s\nNODE_B_IP=%s\n' "$RUN_ID" "$NODE_A_IP" "$NODE_B_IP"
    printf 'MASTER_SERVER=%s\nMETADATA_SERVER=%s\nMASTER_ADMIN_URL=%s\n' \
      "$MASTER_SERVER" "$METADATA_SERVER" "$MASTER_ADMIN_URL"
    printf 'PROVIDER_HOSTNAME=%s\nCONSUMER_HOSTNAME_PREFIX=%s\n' \
      "$PROVIDER_HOSTNAME" "$CONSUMER_HOSTNAME_PREFIX"
    printf 'PYTHONPATH=%s\nLD_LIBRARY_PATH=%s\n' "$python_path" "$ld_library_path"
    printf 'GLOBAL_SEGMENT_SIZE=%s\nHOST_NUMA_NODES=%s\nDEVICES=%s\n' \
      "$GLOBAL_SEGMENT_SIZE" "$HOST_NUMA_NODES" "$DEVICES"
    printf 'MC_MS_AUTO_DISC=0\nMC_FORCE_MNNVL=1\nMC_MAX_MR_SIZE=%s\n' \
      "$MC_MAX_MR_SIZE_BYTES"
    ;;
  build)
    run_common env -u MC_MNNVL_FABRIC_PROBE -u MC_REQUIRE_MNNVL_FABRIC \
      -u MC_REQUIRE_NVLINK_HOST_NUMA_RDMA -u MC_USE_NVLINK_IPC \
      -u JOBS -u CMAKE_EXTRA_ARGS "${build_env[@]}" \
      BUILD_DIR="$BUILD_DIR" RUN_HARDWARE_TESTS="$RUN_HARDWARE_TESTS" \
      SKIP_PREFLIGHT="$SKIP_PREFLIGHT" \
      "${script_dir}/nvlink_host_numa_build.sh" "$@"
    ;;
  preflight)
    [[ -x "$FABRIC_PROBE" ]] || fail "Fabric probe is missing: $FABRIC_PROBE"
    run_common env \
      MC_REQUIRE_MNNVL_FABRIC=1 MC_REQUIRE_NVLINK_HOST_NUMA_RDMA=1 \
      MC_MNNVL_FABRIC_PROBE="$FABRIC_PROBE" \
      "${script_dir}/nvlink_host_numa_preflight.sh" "$@"
    ;;
  master-start)
    mkdir -p "$RESULT_DIR"
    [[ -x "$MASTER_BIN" ]] || fail "Master binary is missing: $MASTER_BIN (run build first)"
    if pid_is_live "$MASTER_PID_FILE" "$MASTER_BIN"; then
      fail "Master is already running with PID $(<"$MASTER_PID_FILE")"
    fi
    for port in "$MASTER_RPC_PORT" "$METADATA_PORT" "$MASTER_ADMIN_PORT"; do
      if tcp_reachable 127.0.0.1 "$port" || tcp_reachable "$NODE_A_IP" "$port"; then
        show_port_owner "$port" >&2
        fail "port $port is already occupied"
      fi
    done
    rm -f "$MASTER_PID_FILE"
    : >"$MASTER_LOG"
    nohup env -u MC_USE_NVLINK_IPC -u MC_RPC_PROTOCOL -u MC_INTRANODE_NVLINK \
      -u MC_FORCE_TCP -u MC_FORCE_HCA -u MOONCAKE_CONFIG_PATH \
      -u MC_CUDART_LIBRARY \
      "${common_env[@]}" "MC_TCP_BIND_ADDRESS=${NODE_A_IP}" \
      "$MASTER_BIN" \
      --rpc_address=0.0.0.0 \
      --rpc_port="$MASTER_RPC_PORT" \
      --enable_http_metadata_server=true \
      --http_metadata_server_host=0.0.0.0 \
      --http_metadata_server_port="$METADATA_PORT" \
      --metrics_port="$MASTER_ADMIN_PORT" \
      --logtostderr=1 >"$MASTER_LOG" 2>&1 &
    printf '%s\n' "$!" >"$MASTER_PID_FILE"
    if ! wait_for_tcp 127.0.0.1 "$MASTER_RPC_PORT" "$MASTER_START_TIMEOUT_SEC" || \
       ! wait_for_tcp 127.0.0.1 "$METADATA_PORT" "$MASTER_START_TIMEOUT_SEC" || \
       ! wait_for_tcp 127.0.0.1 "$MASTER_ADMIN_PORT" "$MASTER_START_TIMEOUT_SEC"; then
      tail -n 100 "$MASTER_LOG" >&2 || true
      stop_pid_file Master "$MASTER_PID_FILE" "$MASTER_BIN" || true
      fail "Master did not open all configured ports"
    fi
    master_remote_status
    printf 'Master started: pid=%s log=%s\n' "$(<"$MASTER_PID_FILE")" "$MASTER_LOG"
    ;;
  master-status)
    if pid_is_live "$MASTER_PID_FILE" "$MASTER_BIN"; then
      printf 'Recorded Master PID is live: %s\n' "$(<"$MASTER_PID_FILE")"
    else
      printf 'No live local Master PID in %s (remote checks still follow)\n' "$MASTER_PID_FILE"
    fi
    master_remote_status
    ;;
  master-stop)
    stop_pid_file Master "$MASTER_PID_FILE" "$MASTER_BIN"
    ;;
  provider-start)
    mkdir -p "$RESULT_DIR"
    master_remote_status
    compgen -G "${BUILD_DIR}/mooncake-integration/store*.so" >/dev/null || \
      fail "build-tree Store binding is missing under ${BUILD_DIR}/mooncake-integration"
    if pid_is_live "$PROVIDER_PID_FILE" "nvlink_host_numa_provider.py"; then
      fail "Provider is already running with PID $(<"$PROVIDER_PID_FILE")"
    fi
    rm -f "$PROVIDER_PID_FILE" "$PROVIDER_READY_FILE"
    : >"$PROVIDER_LOG"
    nohup env -u MC_USE_NVLINK_IPC -u MC_RPC_PROTOCOL -u MC_INTRANODE_NVLINK \
      -u MC_FORCE_TCP -u MC_FORCE_HCA -u MOONCAKE_CONFIG_PATH \
      -u MC_CUDART_LIBRARY \
      "${common_env[@]}" "MC_TCP_BIND_ADDRESS=${NODE_A_IP}" \
      python3 "${script_dir}/nvlink_host_numa_provider.py" \
      --local-hostname "$PROVIDER_HOSTNAME" \
      --metadata-server "$METADATA_SERVER" \
      --master-server "$MASTER_SERVER" \
      --master-admin-url "$MASTER_ADMIN_URL" \
      --global-segment-size "$GLOBAL_SEGMENT_SIZE" \
      --local-buffer-size "$LOCAL_BUFFER_SIZE" \
      --nodes "$HOST_NUMA_NODES" \
      --run-id "$RUN_ID" \
      --ready-file "$PROVIDER_READY_FILE" \
      --readiness-timeout-sec "$PROVIDER_READINESS_TIMEOUT_SEC" \
      >"$PROVIDER_LOG" 2>&1 &
    printf '%s\n' "$!" >"$PROVIDER_PID_FILE"
    deadline=$((SECONDS + PROVIDER_READINESS_TIMEOUT_SEC + 5))
    diagnostic_at=$((SECONDS + PROVIDER_DIAGNOSTIC_DELAY_SEC))
    diagnosed=0
    while [[ ! -s "$PROVIDER_READY_FILE" ]] && \
        pid_is_live "$PROVIDER_PID_FILE" "nvlink_host_numa_provider.py"; do
      if (( diagnosed == 0 && SECONDS >= diagnostic_at )); then
        printf 'Provider is not ready after %ss; capturing live diagnostics...\n' \
          "$PROVIDER_DIAGNOSTIC_DELAY_SEC"
        diagnose | tee "${RESULT_DIR}/provider-start-diagnose.log"
        diagnosed=1
      fi
      (( SECONDS < deadline )) || break
      sleep 1
    done
    if [[ -s "$PROVIDER_READY_FILE" ]] && \
        pid_is_live "$PROVIDER_PID_FILE" "nvlink_host_numa_provider.py"; then
      cat "$PROVIDER_READY_FILE"
      printf 'Provider started: pid=%s log=%s\n' "$(<"$PROVIDER_PID_FILE")" "$PROVIDER_LOG"
      exit 0
    fi
    if (( diagnosed == 0 )); then
      diagnose | tee "${RESULT_DIR}/provider-start-diagnose.log" || true
    fi
    tail -n 120 "$PROVIDER_LOG" >&2 || true
    stop_pid_file Provider "$PROVIDER_PID_FILE" "nvlink_host_numa_provider.py" || true
    fail "Provider did not reach readiness; diagnostics are in ${RESULT_DIR}"
    ;;
  provider-status)
    pid_is_live "$PROVIDER_PID_FILE" "nvlink_host_numa_provider.py" || \
      fail "Provider PID is not live: $PROVIDER_PID_FILE"
    printf 'Provider PID is live: %s\n' "$(<"$PROVIDER_PID_FILE")"
    [[ -s "$PROVIDER_READY_FILE" ]] || fail "Provider ready file is missing: $PROVIDER_READY_FILE"
    cat "$PROVIDER_READY_FILE"
    diagnose
    ;;
  provider-stop)
    stop_pid_file Provider "$PROVIDER_PID_FILE" "nvlink_host_numa_provider.py"
    rm -f "$PROVIDER_READY_FILE"
    ;;
  diagnose)
    diagnose
    ;;
  consumer)
    [[ $# -ge 1 ]] || fail "consumer requires a GPU device number"
    device="$1"
    shift
    [[ "$device" =~ ^[0-9]+$ ]] || fail "consumer device must be nonnegative: $device"
    port=$((CONSUMER_PORT_BASE + device))
    (( port <= 65535 )) || fail "consumer port is out of range: $port"
    master_remote_status
    mkdir -p "$RESULT_DIR"
    run_env "$NODE_B_IP" python3 "${script_dir}/nvlink_host_numa_consumer.py" \
      --local-hostname "${NODE_B_IP}:${port}" \
      --metadata-server "$METADATA_SERVER" \
      --master-server "$MASTER_SERVER" \
      --device "$device" \
      --payload-size "$SINGLE_PAYLOAD_SIZE" \
      --iterations "$ITERATIONS" \
      --run-id "$RUN_ID" "$@" | tee "${RESULT_DIR}/consumer-gpu-${device}.jsonl"
    ;;
  bench)
    master_remote_status
    mkdir -p "$RESULT_DIR"
    run_env "$NODE_B_IP" python3 "${script_dir}/nvlink_host_numa_bench.py" \
      --local-hostname-prefix "$CONSUMER_HOSTNAME_PREFIX" \
      --metadata-server "$METADATA_SERVER" \
      --master-server "$MASTER_SERVER" \
      --devices "$DEVICES" \
      --payload-sizes "$PAYLOAD_SIZES" \
      --iterations "$ITERATIONS" \
      --run-id "$RUN_ID" "$@" | tee "${RESULT_DIR}/bench.jsonl"
    ;;
  stop)
    stop_pid_file Provider "$PROVIDER_PID_FILE" "nvlink_host_numa_provider.py"
    rm -f "$PROVIDER_READY_FILE"
    stop_pid_file Master "$MASTER_PID_FILE" "$MASTER_BIN"
    ;;
  *)
    usage >&2
    fail "unknown action: $action"
    ;;
esac
