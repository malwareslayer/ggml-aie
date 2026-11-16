#!/usr/bin/env python3
"""Compile AIE2 device kernels in src/device/*.cc into XRT xclbin + insts.bin."""

import os
from pathlib import Path
import numpy as np

from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1
from aie.utils.compile import compile_cxx_core_function, compile_mlir_module

REPO = Path(__file__).resolve().parent.parent
DEVICE_DIR = REPO / "src" / "device"
KERNEL_DIR = REPO / "work" / "fullgraph" / "operators" / "kernel"

MLIR_AIE_DIR = Path(os.environ["MLIR_AIE_INSTALL_DIR"])
INCLUDE_DIRS = [
    str(MLIR_AIE_DIR / "include"),
    str(MLIR_AIE_DIR / "aie_runtime_lib" / "AIE2"),
]


def dtype_for(kind: str):
    if kind == "f32":
        return np.float32
    if kind == "u16":
        return np.uint16
    raise ValueError(f"Unsupported dtype kind: {kind}")


def build_op(
    op_name: str,
    src_file: str,
    fn_name: str,
    args_spec: list[tuple[str, str, int, str]],
) -> None:
    """Compile one AIE2 operator.

    args_spec: list of (arg_name, direction 'in'|'out', count, dtype 'f32'|'u16')
               in exact XRT kernel argument order (group_id 3, 4, ...).
    """
    out_dir = KERNEL_DIR / op_name
    build_dir = out_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    src_path = str(DEVICE_DIR / src_file)
    obj_name = f"{fn_name}.o"
    obj_path = str(build_dir / obj_name)

    compile_cxx_core_function(
        source_path=src_path,
        target_arch="aie2",
        output_path=obj_path,
        include_dirs=INCLUDE_DIRS,
    )

    types = [np.ndarray[(cnt,), np.dtype[dtype_for(dt)]] for _, _, cnt, dt in args_spec]
    fifos = [
        ObjectFifo(ty, name=f"{name}_fifo", depth=1)
        for (name, _, _, _), ty in zip(args_spec, types)
    ]

    kernel_fn = ExternalFunction(
        fn_name,
        object_file_name=obj_name,
        source_file=src_path,
        arg_types=types,
        include_dirs=INCLUDE_DIRS,
    )
    kernel_fn._validate_arg = lambda _i, _a, _e: None

    dirs = [d for _, d, _, _ in args_spec]
    core_handles = [
        fifo.cons() if d == "in" else fifo.prod() for fifo, d in zip(fifos, dirs)
    ]
    rt_handles = [
        fifo.prod() if d == "in" else fifo.cons() for fifo, d in zip(fifos, dirs)
    ]

    def core_body(*args):
        *handles, fn = args
        elems = [h.acquire(1) for h in handles]
        fn(*elems)
        for h in handles:
            h.release(1)

    worker = Worker(core_body, fn_args=[*core_handles, kernel_fn])

    def seq_body(*args):
        n = len(dirs)
        rt_bufs = args[:n]
        handles = args[n:]
        for d, buf, h in zip(dirs, rt_bufs, handles):
            if d == "in":
                h.fill(buf)
        out_idx = [i for i, d in enumerate(dirs) if d == "out"][-1]
        for i, (d, buf, h) in enumerate(zip(dirs, rt_bufs, handles)):
            if d == "out":
                h.drain(buf, wait=(i == out_idx))

    rt = Runtime(seq_body, fn_args=[*types, *rt_handles])
    prog = Program(NPU1Col1(), rt, workers=[worker])
    module = prog.resolve_program()

    compile_mlir_module(
        module,
        insts_path=str(out_dir / "insts.bin"),
        xclbin_path=str(out_dir / f"{op_name}.xclbin"),
        work_dir=str(build_dir),
        device=NPU1Col1(),
    )
    print(f"[OK] {op_name} -> {out_dir / (op_name + '.xclbin')}")


def main() -> None:
    ops = [
        ("layernorm", "layernorm.cc", "layernorm_f32_4096",
         [("in", "in", 4096, "f32"), ("out", "out", 4096, "f32"), ("aux", "in", 16, "f32")]),
        ("softmax", "softmax.cc", "softmax_f32_4096",
         [("in", "in", 4096, "f32"), ("aux", "in", 4112, "f32"), ("out", "out", 4096, "f32")]),
        ("eltwise", "eltwise.cc", "eltwise_f32_4096",
         [("a", "in", 4096, "f32"), ("aux", "in", 4112, "f32"), ("out", "out", 4096, "f32")]),
        ("unary", "unary.cc", "unary_f32_4096",
         [("in", "in", 4096, "f32"), ("out", "out", 4096, "f32"), ("aux", "in", 16, "f32")]),
        ("gelu", "gelu.cc", "gelu_bf16_4096",
         [("in", "in", 4096, "u16"), ("out", "out", 4096, "u16")]),
        ("gemm_f32", "gemm_f32.cc", "gemm_f32_32x32x32",
         [("a", "in", 1024, "f32"), ("b", "in", 1024, "f32"), ("c", "out", 1024, "f32")]),
        ("l2_norm", "reduce.cc", "l2_norm_f32_4096",
         [("in", "in", 4096, "f32"), ("out", "out", 4096, "f32"), ("aux", "in", 16, "f32")]),
        ("reduce_sum", "reduce.cc", "reduce_sum_f32_4096",
         [("in", "in", 4096, "f32"), ("out", "out", 16, "f32"), ("aux", "in", 16, "f32")]),
    ]
    for op_name, src_file, fn_name, args_spec in ops:
        build_op(op_name, src_file, fn_name, args_spec)


if __name__ == "__main__":
    main()
