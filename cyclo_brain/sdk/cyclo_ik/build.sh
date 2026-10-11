#!/usr/bin/env bash
# Build the unchanged Cyclo C++ core against the policy Python's Pinocchio.
set -euo pipefail
cyclo_sdk_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cyclo_python="${PYTHON:-python3}"
cyclo_build_dir="${CYCLO_IK_BUILD_DIR:-/tmp/cyclo_ik_build}"
cyclo_output_dir="${1:?Usage: bash build.sh /path/to/Python/module/output}"
cyclo_deps_dir="$cyclo_output_dir/deps"
cyclo_prefix="$($cyclo_python -c 'import site; from pathlib import Path; print(next(str(Path(p)/"cmeel.prefix") for p in site.getsitepackages() if (Path(p)/"cmeel.prefix").is_dir()))')"
cyclo_pybind="$($cyclo_python -m pybind11 --cmakedir)"
mkdir -p "$cyclo_build_dir"
# OSQP is a native dependency of the official, vendored OsqpEigen library.
if [ ! -f "$cyclo_deps_dir/lib/cmake/osqp/osqp-config.cmake" ]; then
  if [ ! -d "$cyclo_build_dir/osqp/.git" ]; then
    git clone --branch v0.6.3 --depth 1 --recurse-submodules https://github.com/osqp/osqp.git "$cyclo_build_dir/osqp"
  fi
  cmake -S "$cyclo_build_dir/osqp" -B "$cyclo_build_dir/osqp-build" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$cyclo_deps_dir" \
    -DUNITTESTS=OFF -DPRINTING=OFF
  cmake --build "$cyclo_build_dir/osqp-build" -j2
  cmake --install "$cyclo_build_dir/osqp-build"
fi
cmake -S "$cyclo_sdk_dir/../cyclo_control/osqp_eigen_vendor/third_party/osqp-eigen" \
  -B "$cyclo_build_dir/osqp-eigen-build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$cyclo_prefix;$cyclo_deps_dir" \
  -DCMAKE_INSTALL_PREFIX="$cyclo_deps_dir" -DBUILD_TESTING=OFF
cmake --build "$cyclo_build_dir/osqp-eigen-build" -j2
cmake --install "$cyclo_build_dir/osqp-eigen-build"
cmake -S "$cyclo_sdk_dir" -B "$cyclo_build_dir/native" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$cyclo_prefix;$cyclo_deps_dir;$cyclo_pybind" \
  -DPYTHON_EXECUTABLE="$(command -v "$cyclo_python")" -DCMAKE_INSTALL_PREFIX="$cyclo_output_dir"
# Pinocchio's C++ templates consume significant memory; compile sequentially.
cmake --build "$cyclo_build_dir/native" -j1
cmake --install "$cyclo_build_dir/native"
