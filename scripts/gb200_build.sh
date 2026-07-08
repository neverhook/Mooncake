#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
REPO=${REPO:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)}
BUILD=${BUILD:-"$REPO/build-gb200"}
JOBS=${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-RelWithDebInfo}

cd "$REPO"
mkdir -p "$BUILD"

cmake -S "$REPO" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DBUILD_UNIT_TESTS=ON \
    -DBUILD_EXAMPLES=OFF \
    -DWITH_TE=ON \
    -DWITH_STORE=ON \
    -DWITH_STORE_RUST=OFF \
    -DWITH_EP=OFF \
    -DUSE_CUDA=ON \
    -DUSE_MNNVL=ON \
    -DUSE_ETCD=ON \
    -DSTORE_USE_ETCD=ON \
    -DENABLE_MULTI_PROTOCOL=ON \
    ${EXTRA_CMAKE_ARGS:-}

cmake --build "$BUILD" --target \
    mooncake_master \
    mooncake_store \
    store \
    default_config_test \
    nvlink_transport_test \
    transfer_metadata_test \
    config_test \
    transfer_task_test \
    transfer_candidate_selector_test \
    client_read_selection_test \
    -j "$JOBS"

printf 'GB200 build complete: %s\n' "$BUILD"
