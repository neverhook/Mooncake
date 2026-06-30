# Mooncake Test Dependency Image

This image is for local build and unit-test validation. It preinstalls the
Ubuntu packages from `dependencies.sh`, Go, pybind11, and yalantinglibs so test
runs do not need to reinstall dependencies from scratch.

Build the image from the repository root:

```bash
docker build -f docker/test-deps.Dockerfile \
  -t mooncake-test-deps:ubuntu24.04 \
  docker
```

Prepare a source tree inside a container:

```bash
docker run --rm -it \
  -v "$PWD":/work/Mooncake \
  mooncake-test-deps:ubuntu24.04 \
  bash -lc 'mooncake-prepare-source /work/Mooncake && bash'
```

Run a focused local validation:

```bash
docker run --rm \
  -v "$PWD":/work/Mooncake \
  mooncake-test-deps:ubuntu24.04 \
  bash -lc '
    mooncake-prepare-source /work/Mooncake &&
    cmake -S /work/Mooncake -B /work/Mooncake/build-tests -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DBUILD_UNIT_TESTS=ON \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_BENCHMARK=OFF \
      -DWITH_TE=ON \
      -DWITH_STORE=ON \
      -DWITH_STORE_RUST=OFF \
      -DENABLE_MULTI_PROTOCOL=ON \
      -DUSE_TCP=ON \
      -DUSE_HTTP=ON &&
    cmake --build /work/Mooncake/build-tests -j"$(nproc)" \
      --target transfer_candidate_selector_test client_read_selection_test &&
    ctest --test-dir /work/Mooncake/build-tests --output-on-failure \
      -R "^(transfer_candidate_selector_test|client_read_selection_test)$"
  '
```

The helper command only fills `extern/pybind11` and `extern/yalantinglibs` when
those directories are empty. Existing checked-out submodules are left unchanged.
