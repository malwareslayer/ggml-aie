# ggml-aie

Native GGML dynamic backend (`libggml-aie.so`) for **AMD Phoenix / XDNA1 AIE2 NPUs** (`/dev/accel/accel0`) via Xilinx Runtime (`xrt` + `xrt-plugin-amdxdna`).

Designed for **heterogeneous compute offloading**—offloading supporting compute workloads such as embeddings, encoders, rerankers, and hybrid prefill to the NPU so `CPU` and `GPU` (`ROCm` / `Vulkan` / `CUDA`) resources remain available for primary inference.

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
  - Automatic host dequantization via GGML `to_float` traits and lazy `BF16` host prepack caching for `MUL_MAT`
  - `xrt::runlist` tile batching to minimize kernel submission overhead
- **XRT Hardware Context QoS**
  - Populates `xrt::hw_context` QoS hints (`priority`, `gops`, `latency`, and optional `fps`, `dma_bandwidth`, `frame_execution_time`) surfaced by `xrt-smi examine -r all`

Unsupported ops or shapes outside AIE2 kernel limits return `false` in `supports_op`, allowing GGML / `llama.cpp` to place them on `CPU` or `GPU` automatically.

## Repository Layout

| Path | Description |
| --- | --- |
| `src/` | C++20 GGML backend (`ggml_backend.cpp`), XRT runners (`xrt_*`), symbol version script (`ggml-aie.ver`), and AIE2 device kernels (`src/device/*.cc`) |
| `scripts/build_kernels.py` | MLIR-AIE (`aie.iron`) build script compiling `src/device/*.cc` into `.xclbin` + `insts.bin` |
| `packaging/aur/` | Arch Linux `PKGBUILD` and `.SRCINFO` |
| `packaging/systemd/` | `llama-server` systemd drop-in (`90-aie.conf`) |
| `.clang-format`, `.clang-tidy` | Formatting and static analysis rules (`-std=c++20 -Wall -Wextra -Werror`) |

## Prerequisites

- **Hardware / Kernel**: AMD Phoenix / XDNA1 NPU (`/dev/accel/accel0`) with `amdxdna` driver
- **Runtime**: `xrt` (`2.21.75`), `xrt-plugin-amdxdna`, `llama.cpp` / `ggml` (`>= 0.29.0`, GGML Backend API v3)
- **Build**: CMake `>= 3.24`, Ninja, C++20 compiler (`g++` or `clang++`), GGML source headers (`GGML_SOURCE_DIR`)
- **Kernel Compilation** *(optional, only when rebuilding `.xclbin`)*: `mlir-aie` (`aie.iron`), `llvm-aie` (`peano`), `numpy`, `cloudpickle`

## Installation

### Arch Linux (AUR)

```bash
yay -S ggml-aie
```

This installs `/usr/lib/ggml/libggml-aie.so`, pre-compiled kernels in `/usr/share/ggml-aie/kernel/`, and the `llama-server` systemd drop-in at `/usr/lib/systemd/system/llama-server.service.d/90-aie.conf`.

### Building from Source

#### 1. Build the GGML Backend Library

```bash
cmake -S . -B build -G Ninja \
  -DGGML_SOURCE_DIR=/path/to/ggml \
  -DGGML_LIBRARY_DIR=/usr/lib \
  -DCMAKE_INSTALL_PREFIX=/usr \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-aie -j"$(nproc)"
```

Output: `build/libggml-aie.so` (exports only `ggml_backend_init` via `src/ggml-aie.ver`).

#### 2. Compile AIE2 Device Kernels *(Optional)*

```bash
PEANO_INSTALL_DIR=/path/to/llvm-aie \
MLIR_AIE_INSTALL_DIR=/path/to/mlir_aie \
PATH="$MLIR_AIE_INSTALL_DIR/bin:$PEANO_INSTALL_DIR/bin:$PATH" \
PYTHONPATH="$MLIR_AIE_INSTALL_DIR/python" \
python3 scripts/build_kernels.py
```

Each operator produces `<op>/<op>.xclbin` and `<op>/insts.bin`.

#### 3. Install

```bash
sudo cmake --install build
sudo install -dm755 /usr/share/ggml-aie/kernel
sudo cp -a kernel/* /usr/share/ggml-aie/kernel/
```

## Usage

Load the dynamic backend via `GGML_BACKEND_PATH` and select `AIE2-XRT`:

```bash
# Benchmark prompt processing / embedding prefill on NPU
GGML_BACKEND_PATH=/usr/lib/ggml/libggml-aie.so \
  llama-bench -m /path/to/embedding-model.gguf -dev AIE2-XRT -p 512 -n 0

# Run llama-server with NPU offloading
GGML_BACKEND_PATH=/usr/lib/ggml/libggml-aie.so \
  llama-server -m /path/to/embedding-model.gguf --embedding -dev AIE2-XRT
```

### `llama-server` Systemd Drop-in

`/usr/lib/systemd/system/llama-server.service.d/90-aie.conf`:

```ini
[Service]
DeviceAllow=/dev/accel/accel0 rw
SupplementaryGroups=render
LimitMEMLOCK=512M
Environment=GGML_BACKEND_PATH=/usr/lib/ggml/libggml-aie.so
```

## Configuration

### Runtime Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `AIE2_XRT_OPS_DIR` | `/usr/share/ggml-aie/kernel` | Override directory containing compiled `<op>/<op>.xclbin` and `<op>/insts.bin` |
| `AIE2_TRACE` | unset | Set to any value to log dispatched GGML graph nodes to `stderr` |

### XRT Hardware Context QoS

QoS hints are applied when creating `xrt::hw_context` and reported under `HW Contexts` in `xrt-smi examine -r all`. Set any numeric value to `0` (or `AIE2_QOS_PRIORITY=unset`) to omit that field.

| Variable | Default | Accepted Values | Description |
| --- | --- | --- | --- |
| `AIE2_QOS_PRIORITY` | `normal` | `realtime` (`0x100`), `high` (`0x180`), `normal` (`0x200`), `low` (`0x280`), `none`/`unset` (`0`), or integer | Hardware context scheduling priority |
| `AIE2_QOS_GOPS` | `4000` | Unsigned 32-bit integer | Target Giga-OPS/s hint |
| `AIE2_QOS_LATENCY` | `1000` | Unsigned 32-bit integer | Target latency hint |
| `AIE2_QOS_FPS` | `0` | Unsigned 32-bit integer | Target frames per second (unset by default) |
| `AIE2_QOS_DMA_BW` | `0` | Unsigned 32-bit integer | Target DMA bandwidth hint (GB/s) |
| `AIE2_QOS_FRAME_EXEC_TIME` | `0` | Unsigned 32-bit integer | Target frame execution time hint |

## Verification & Development

Inspect active NPU hardware contexts and QoS state while `llama-server` or `llama-bench` is running:

```bash
xrt-smi examine -r all
```

Format and static analysis:

```bash
clang-format --dry-run --Werror src/*.h src/*.cpp src/device/*.cc
clang-tidy --quiet --warnings-as-errors='*' src/*.cpp -- \
  -std=c++20 -Wall -Wextra -Isrc \
  -isystem /path/to/ggml/include \
  -isystem /path/to/ggml/src \
  -DGGML_BACKEND_DL -DGGML_BACKEND_SHARED -DGGML_BACKEND_BUILD
```

## Contributing

Contributions are welcome. Please format changes with `clang-format` and verify `libggml-aie.so` exports only `ggml_backend_init` before opening a pull request.

## License

MIT — see [LICENSE](LICENSE).
