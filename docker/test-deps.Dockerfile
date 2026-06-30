ARG UBUNTU_IMAGE=public.ecr.aws/ubuntu/ubuntu:24.04
FROM ${UBUNTU_IMAGE}

ARG GO_VERSION=1.25.9
ARG PYBIND11_COMMIT=58c382a8e3d7081364d2f5c62e7f429f0412743b
ARG YALANTINGLIBS_COMMIT=6a0e067d9a43492cf8e4e280b531924fbd724dbd

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/go/bin:${PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libasio-dev \
        libboost-all-dev \
        libc-bin \
        libc6-dev \
        libcurl4-openssl-dev \
        libgoogle-glog-dev \
        libgrpc++-dev \
        libgrpc-dev \
        libgtest-dev \
        libhiredis-dev \
        libibverbs-dev \
        libjemalloc-dev \
        libjsoncpp-dev \
        libmsgpack-dev \
        libnuma-dev \
        libprotobuf-dev \
        libpython3-dev \
        libssl-dev \
        libunwind-dev \
        liburing-dev \
        libxxhash-dev \
        libyaml-cpp-dev \
        libzstd-dev \
        ninja-build \
        patchelf \
        pkg-config \
        protobuf-compiler-grpc \
        unzip \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    arch="$(uname -m)"; \
    case "${arch}" in \
        aarch64) goarch=arm64 ;; \
        x86_64) goarch=amd64 ;; \
        *) echo "Unsupported architecture: ${arch}" >&2; exit 1 ;; \
    esac; \
    tarball="go${GO_VERSION}.linux-${goarch}.tar.gz"; \
    for base_url in \
        "https://go.dev/dl" \
        "https://golang.google.cn/dl" \
        "https://mirrors.aliyun.com/golang"; do \
        if wget -q --timeout=30 --tries=2 -O "/tmp/${tarball}" "${base_url}/${tarball}"; then \
            break; \
        fi; \
        rm -f "/tmp/${tarball}"; \
    done; \
    test -s "/tmp/${tarball}"; \
    rm -rf /usr/local/go; \
    tar -C /usr/local -xzf "/tmp/${tarball}"; \
    rm -f "/tmp/${tarball}"; \
    go version

RUN set -eux; \
    mkdir -p /opt/mooncake-submodules; \
    git clone https://github.com/pybind/pybind11.git /opt/mooncake-submodules/pybind11; \
    git -C /opt/mooncake-submodules/pybind11 checkout "${PYBIND11_COMMIT}"; \
    git clone https://github.com/alibaba/yalantinglibs.git /opt/mooncake-submodules/yalantinglibs; \
    git -C /opt/mooncake-submodules/yalantinglibs checkout "${YALANTINGLIBS_COMMIT}"; \
    cmake -S /opt/mooncake-submodules/yalantinglibs \
        -B /opt/mooncake-submodules/yalantinglibs/build \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARK=OFF \
        -DBUILD_UNIT_TESTS=OFF; \
    cmake --build /opt/mooncake-submodules/yalantinglibs/build -j"$(nproc)"; \
    cmake --install /opt/mooncake-submodules/yalantinglibs/build; \
    rm -rf /opt/mooncake-submodules/yalantinglibs/build

RUN cat >/usr/local/bin/mooncake-prepare-source <<'EOF' && chmod +x /usr/local/bin/mooncake-prepare-source
#!/usr/bin/env bash
set -euo pipefail

repo="${1:-/work/Mooncake}"
if [ ! -d "${repo}" ]; then
    echo "repository directory does not exist: ${repo}" >&2
    exit 1
fi

mkdir -p "${repo}/extern"
for module in pybind11 yalantinglibs; do
    target="${repo}/extern/${module}"
    if [ -e "${target}" ] && [ "$(find "${target}" -mindepth 1 -maxdepth 1 | wc -l)" -gt 0 ]; then
        continue
    fi
    rm -rf "${target}"
    cp -a "/opt/mooncake-submodules/${module}" "${target}"
done
EOF

WORKDIR /work
CMD ["/bin/bash"]
