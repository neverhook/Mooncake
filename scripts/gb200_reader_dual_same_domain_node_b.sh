#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
REPO=${REPO:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)}

export REPO
export EXPECTED_PATH=${EXPECTED_PATH:-nvlink}
export MC_NVLINK_SCALE_UP_DOMAIN_ID=${MC_NVLINK_SCALE_UP_DOMAIN_ID:-gb200-nvl}

exec sh "$REPO/scripts/gb200_reader_dual_node_b.sh"
