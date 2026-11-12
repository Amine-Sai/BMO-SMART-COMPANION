#!/usr/bin/env bash
# Rebuild llama-cpp-python from source with CUDA so n_gpu_layers actually offloads.
#
# Usage:
#   tools/rebuild_llamacpp_cuda.sh
#
# Prerequisites: NVIDIA driver, CUDA toolkit at CUDA_HOME (/opt/cuda on Arch),
# a C/C++ toolchain, system cmake and ninja.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
LLAMA_CPP_VERSION="${LLAMA_CPP_VERSION:-0.3.36}"
CUDA_ARCH="${CUDA_ARCH:-75}"  # GTX 1660 Ti is Turing (sm_75)
JOBS="${JOBS:-6}"
BUILD_ROOT="${BUILD_ROOT:-$ROOT/scratch/llamacpp-build}"

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "nvcc not found at $CUDA_HOME/bin/nvcc; install the CUDA toolkit (e.g. sudo pacman -S --needed extra/cuda)" >&2
  exit 1
fi

export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
export FORCE_CMAKE=1
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"

rm -rf "$BUILD_ROOT/wheels"
mkdir -p "$BUILD_ROOT/wheels" "$BUILD_ROOT/tmp"
export TMPDIR="$BUILD_ROOT/tmp"

echo "Building llama-cpp-python $LLAMA_CPP_VERSION with CUDA (arch sm_$CUDA_ARCH) using $JOBS jobs" >&2
"$VENV/bin/pip" wheel \
  --no-deps --no-cache-dir --no-binary=:all: \
  "llama-cpp-python==$LLAMA_CPP_VERSION" \
  -w "$BUILD_ROOT/wheels"

WHEEL="$(ls -1 "$BUILD_ROOT"/wheels/llama_cpp_python-*.whl | head -n1)"
echo "Installing $WHEEL" >&2
"$VENV/bin/pip" install --force-reinstall --no-deps "$WHEEL"

"$VENV/bin/python" - <<'PY'
import llama_cpp

probe = getattr(llama_cpp, "llama_supports_gpu_offload", None)
supported = bool(probe()) if probe else False
print("supports_gpu_offload:", supported)
raise SystemExit(0 if supported else 1)
PY
