#!/usr/bin/env python3
"""Compile 4-column AIE2 BF16 GEMM kernel (32x1024x1024 per invocation)."""

import os
from pathlib import Path
import numpy as np

from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1
from aie.utils.compile import compile_cxx_core_function, compile_mlir_module

REPO = Path(__file__).resolve().parent.parent
SRC_PATH = REPO / "src" / "device" / "gemm_bf16.cc"
OUT_DIR = REPO / "kernel" / "gemm_bf16"
MLIR_AIE_DIR = Path(os.environ["MLIR_AIE_INSTALL_DIR"])
INCLUDE_DIRS = [
    str(MLIR_AIE_DIR / "include"),
    str(MLIR_AIE_DIR / "aie_runtime_lib" / "AIE2"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_dir = OUT_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    obj_name = "gemm_bf16.o"
    compile_cxx_core_function(
        source_path=str(SRC_PATH),
        target_arch="aie2",
        output_path=str(build_dir / obj_name),
        include_dirs=INCLUDE_DIRS,
    )

    k_chunks = 16
    nw = 16
    a_tile = 32 * 64
    b_tile = 64 * 64
    c_tile = 32 * 64

    a_ty = np.ndarray[(a_tile,), np.dtype[np.uint16]]
    b_ty = np.ndarray[(b_tile,), np.dtype[np.uint16]]
    c_ty = np.ndarray[(c_tile,), np.dtype[np.uint16]]
    a_full_ty = np.ndarray[(k_chunks * a_tile,), np.dtype[np.uint16]]
    b_col_ty = np.ndarray[(4 * b_tile,), np.dtype[np.uint16]]
    c_col_ty = np.ndarray[(4 * c_tile,), np.dtype[np.uint16]]
    b_all_ty = np.ndarray[(k_chunks * nw * b_tile,), np.dtype[np.uint16]]
    c_all_ty = np.ndarray[(nw * c_tile,), np.dtype[np.uint16]]

    zero_fn = ExternalFunction(
        "gemm_bf16_zero", object_file_name=obj_name, source_file=str(SRC_PATH),
        arg_types=[], include_dirs=INCLUDE_DIRS,
    )
    gemm_fn = ExternalFunction(
        "gemm_bf16_32x64x64_acc", object_file_name=obj_name, source_file=str(SRC_PATH),
        arg_types=[a_ty, b_ty], include_dirs=INCLUDE_DIRS,
    )
    gemm_fn._validate_arg = lambda _i, _a, _e: None
    out_fn = ExternalFunction(
        "gemm_bf16_out", object_file_name=obj_name, source_file=str(SRC_PATH),
        arg_types=[c_ty], include_dirs=INCLUDE_DIRS,
    )
    out_fn._validate_arg = lambda _i, _a, _e: None

    of_a = ObjectFifo(a_ty, name="of_a", depth=2)
    of_b_cols = [ObjectFifo(b_col_ty, name=f"of_b_c{c}", depth=2) for c in range(4)]
    of_b_workers = [
        of_b_cols[c].cons().split(
            [w * b_tile for w in range(4)],
            obj_types=[b_ty] * 4,
            names=[f"of_b_c{c}_w{w}" for w in range(4)],
        )
        for c in range(4)
    ]
    of_c_cols = [ObjectFifo(c_col_ty, name=f"of_c_c{c}", depth=1) for c in range(4)]
    of_c_workers = [
        of_c_cols[c].prod().join(
            [w * c_tile for w in range(4)],
            obj_types=[c_ty] * 4,
            names=[f"of_c_c{c}_w{w}" for w in range(4)],
        )
        for c in range(4)
    ]

    def core_fn(fa, fb, fc, zfn, gfn, ofn):
        zfn()
        for _ in range_(k_chunks):
            a = fa.acquire(1)
            b = fb.acquire(1)
            gfn(a, b)
            fa.release(1)
            fb.release(1)
        c = fc.acquire(1)
        ofn(c)
        fc.release(1)

    workers = []
    for col in range(4):
        for row in range(4):
            workers.append(
                Worker(
                    core_fn,
                    fn_args=[
                        of_a.cons(), of_b_workers[col][row].cons(),
                        of_c_workers[col][row].prod(), zero_fn, gemm_fn, out_fn,
                    ],
                    stack_size=2048,
                )
            )

    def seq_body(a_in, b_in, c_out, pa, pb0, pb1, pb2, pb3, pc0, pc1, pc2, pc3):
        pa.fill(a_in)
        pbs = [pb0, pb1, pb2, pb3]
        pcs = [pc0, pc1, pc2, pc3]
        b_len = k_chunks * 4 * b_tile
        c_len = 4 * c_tile
        for col in range(4):
            pbs[col].fill(b_in, offset=col * b_len, sizes=[1, 1, 1, b_len], strides=[0, 0, 0, 1])
            pcs[col].drain(c_out, offset=col * c_len, sizes=[1, 1, 1, c_len], strides=[0, 0, 0, 1], wait=True)

    rt = Runtime(
        seq_body,
        fn_args=[
            a_full_ty, b_all_ty, c_all_ty,
            of_a.prod(),
            *[f.prod() for f in of_b_cols],
            *[f.cons() for f in of_c_cols],
        ],
    )
    prog = Program(NPU1(), rt, workers=workers)
    compile_mlir_module(
        prog.resolve_program(),
        insts_path=str(OUT_DIR / "insts.bin"),
        xclbin_path=str(OUT_DIR / "gemm_bf16.xclbin"),
        work_dir=str(build_dir),
        device=NPU1(),
    )
    print(f"[OK] gemm_bf16 -> {OUT_DIR / 'gemm_bf16.xclbin'}")


if __name__ == "__main__":
    main()
