# ggml-aie

Native GGML dynamic backend (`libggml-aie.so`) for **AMD Phoenix / XDNA1 AIE2 NPU** (`/dev/accel/accel0`) via XRT, targeting encoder, embedding, reranker, and hybrid prefill workloads in GGML / `llama-server` / Ollama.

## Architecture & Capabilities

`ggml-aie` implements the GGML Backend API v3 (`GGML_BACKEND_DL`) and dispatches supported tensor operations onto compiled AIE2 XCLBIN kernels (`NPU1` 4-column `BF16` `aie::mmul<4,8,8>` + `NPU1Col1` `F32` ops) through `xrt_coreutil`:

- **AIE2 XRT Kernels (`src/device/*.cc`)**
  - `gemm_f32`: 32x32x32 tiled `F32` matrix multiplication (`GGML_OP_MUL_MAT`)
  - `layernorm`: `GGML_OP_NORM` and `GGML_OP_RMS_NORM` (row width `1..4096`)
  - `softmax`: `GGML_OP_SOFT_MAX` with optional `F32`/`F16` mask and scale (row width `1..4096`)
  - `eltwise`: `GGML_OP_ADD`, `GGML_OP_MUL`, `GGML_OP_SUB`, `GGML_OP_SCALE`, `GGML_OP_DIV`
  - `unary` & `gelu`: `SILU`, `TANH`, `SIGMOID`, `RELU`, `NEG`, `ABS`, `GELU`
  - `l2_norm` & `reduce_sum`: `GGML_OP_L2_NORM` and `GGML_OP_SUM_ROWS` (row width `1..4096`)
  - `GGML_OP_GLU`: `REGLU`, `GEGLU`, `SWIGLU`, `GEGLU_ERF`, `GEGLU_QUICK`
  - `GGML_OP_FLASH_ATTN_EXT`: GQA-aware attention (`n_kv <= 4096`, `logit_softcap == 0`)
- **Host & View Ops**
  - `GET_ROWS`, `SET_ROWS`, `CONT`, `DUP`, `CPY`, `PAD`, `ROPE` (`NORMAL` & `NEOX`, `ext_factor == 0`)
  - Zero-copy views: `NONE`, `RESHAPE`, `VIEW`, `TRANSPOSE`, `PERMUTE`
  - Automatic host dequantization via GGML `to_float` traits for `MUL_MAT`, `GET_ROWS`, and `CPY`/`DUP`/`CONT`

Unsupported ops or shapes outside AIE2 kernel limits return `false` in `supports_op`, allowing GGML/Ollama to place them on CPU or GPU.

## Repository Layout

| Path | Description |
| --- | --- |
| `src/` | C++20 GGML backend (`ggml_backend.cpp`), XRT runners (`xrt_*`), symbol version script (`ggml-aie.ver`), and AIE2 device kernels (`src/device/*.cc`) |
| `scripts/build_kernels.py` | MLIR-AIE (`aie.iron`) build script compiling `src/device/*.cc` into `.xclbin` + `insts.bin` |
| `packaging/aur/` | Arch Linux `PKGBUILD` |
| `.clang-format`, `.clang-tidy` | Formatting and static analysis rules (`-std=c++20 -Wall -Wextra -Werror`) |

## Prerequisites

- **Hardware / Kernel**: AMD Phoenix / XDNA1 NPU (`/dev/accel/accel0`) with `amdxdna` driver
- **Runtime**: `xrt` (`2.21.75`), `xrt-plugin-amdxdna`, Ollama (`0.40.16+`, GGML Backend API v3)
- **Build**: CMake >= 3.24, Ninja, C++20 compiler (`g++` or `clang++`), GGML source headers (`GGML_SOURCE_DIR`)
- **Kernel Compilation** *(optional, only when rebuilding `.xclbin`)*: `mlir-aie` (`aie.iron`), `llvm-aie` (`peano`), `numpy`, `cloudpickle`

## Building

### 1. Build the GGML Backend Library

```bash
cmake -S . -B build -G Ninja \
  -DGGML_SOURCE_DIR=/path/to/llama.cpp/ggml \
  -DOLLAMA_LIBRARY_DIR=/usr/lib/ollama \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-aie -j
```

Output: `build/libggml-aie.so` (exports only `ggml_backend_init` via `src/ggml-aie.ver`).

### 2. Compile AIE2 Device Kernels

```bash
PEANO_INSTALL_DIR=/path/to/llvm-aie \
MLIR_AIE_INSTALL_DIR=/path/to/mlir_aie \
PATH="$MLIR_AIE_INSTALL_DIR/bin:$PEANO_INSTALL_DIR/bin:$PATH" \
PYTHONPATH="$MLIR_AIE_INSTALL_DIR/python" \
python3 scripts/build_kernels.py
```

Each operator produces `<op>/<op>.xclbin` and `<op>/insts.bin`.

## Installation & Configuration

Install the backend library and compiled operator kernels to the default system paths:

```bash
sudo install -Dm755 build/libggml-aie.so /usr/lib/ggml/libggml-aie.so
sudo mkdir -p /usr/share/ggml-aie/kernel
# Copy each compiled <op>/ directory containing <op>.xclbin and insts.bin into /usr/share/ggml-aie/kernel/
```

Configure Ollama (e.g. via a systemd drop-in):

```ini
[Service]
LimitMEMLOCK=512M
Environment=GGML_BACKEND_PATH=/usr/lib/ggml/libggml-aie.so
```

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `AIE2_XRT_OPS_DIR` | `/usr/share/ggml-aie/kernel` | Override directory containing compiled `<op>/<op>.xclbin` and `<op>/insts.bin` |
| `AIE2_TRACE` | unset | Set to any value to log dispatched GGML graph nodes to `stderr` |

## Development & Linting

```bash
clang-format --dry-run --Werror src/*.h src/*.cpp src/device/*.cc
clang-tidy --quiet --warnings-as-errors='*' src/*.cpp -- \
  -std=c++20 -Wall -Wextra -Isrc \
  -isystem /path/to/llama.cpp/ggml/include \
  -isystem /path/to/llama.cpp/ggml/src \
  -DGGML_BACKEND_DL -DGGML_BACKEND_SHARED -DGGML_BACKEND_BUILD
```

## License

See [LICENSE](LICENSE).
