// GGML backend for AMD AIE2 (Phoenix/XDNA1) via XRT. Only standard GGML
// backend/device/registry interfaces are implemented; unsupported ops are rejected.
#include "ggml-backend-impl.h"
#include "xrt_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

struct State {
    std::mutex init_lock;
    std::shared_ptr<XrtOpsRunner> ops;
    bool init_attempted = false;
};
State state;
extern ggml_backend_device device;

std::shared_ptr<XrtOpsRunner> get_ops() {
    const char* env = std::getenv("AIE2_XRT_OPS_DIR");
    std::string ops_dir = (env && *env) ? env : "/usr/share/ollama-aie2/kernel";
    const std::scoped_lock guard(state.init_lock);
    if (!state.ops && !state.init_attempted) {
        state.init_attempted = true;
        try {
            state.ops = std::make_shared<XrtOpsRunner>(std::move(ops_dir));
        } catch (const std::exception& e) {
            std::fprintf(stderr, "aie2: XRT ops init failed: %s\n", e.what());
        }
    }
    return state.ops;
}

bool can_read_as_f32(const struct ggml_tensor* t) {
    if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) {
        return true;
    }
    const auto* traits = ggml_get_type_traits(t->type);
    return traits && traits->to_float != nullptr && t->nb[0] == ggml_type_size(t->type);
}

float read_elem_f32(const struct ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char* ptr = static_cast<const char*>(t->data) + (i0 * t->nb[0]) + (i1 * t->nb[1]) +
                      (i2 * t->nb[2]) + (i3 * t->nb[3]);
    if (t->type == GGML_TYPE_F32) {
        return *reinterpret_cast<const float*>(ptr);
    }
    if (t->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(ptr));
    }
    return 0.0f;
}

void read_row_f32(const struct ggml_tensor* t, int64_t i1, int64_t i2, int64_t i3, float* dst) {
    const char* row_ptr =
        static_cast<const char*>(t->data) + (i1 * t->nb[1]) + (i2 * t->nb[2]) + (i3 * t->nb[3]);
    if (t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float)) {
        std::memcpy(dst, row_ptr, static_cast<size_t>(t->ne[0]) * sizeof(float));
        return;
    }
    const auto* traits = ggml_get_type_traits(t->type);
    if (traits && traits->to_float != nullptr && t->nb[0] == ggml_type_size(t->type)) {
        traits->to_float(row_ptr, dst, t->ne[0]);
        return;
    }
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        dst[i0] = read_elem_f32(t, i0, i1, i2, i3);
    }
}

void write_elem_f32(const struct ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3,
                    float val) {
    char* ptr = static_cast<char*>(t->data) + (i0 * t->nb[0]) + (i1 * t->nb[1]) + (i2 * t->nb[2]) +
                (i3 * t->nb[3]);
    if (t->type == GGML_TYPE_F32) {
        *reinterpret_cast<float*>(ptr) = val;
    } else if (t->type == GGML_TYPE_F16) {
        *reinterpret_cast<ggml_fp16_t*>(ptr) = ggml_fp32_to_fp16(val);
    }
}

void pack_tensor_f32(const struct ggml_tensor* t, std::vector<float>& out) {
    const int64_t n = ggml_nelements(t);
    out.resize(static_cast<size_t>(n));
    if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
        std::memcpy(out.data(), t->data, static_cast<size_t>(n) * sizeof(float));
        return;
    }
    const int64_t n0 = t->ne[0];
    int64_t offset = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                read_row_f32(t, i1, i2, i3, out.data() + offset);
                offset += n0;
            }
        }
    }
}

int32_t unary_op_code(enum ggml_unary_op uop) {
    switch (uop) {
    case GGML_UNARY_OP_SILU:
        return 0;
    case GGML_UNARY_OP_TANH:
        return 1;
    case GGML_UNARY_OP_SIGMOID:
        return 2;
    case GGML_UNARY_OP_RELU:
        return 3;
    case GGML_UNARY_OP_NEG:
        return 5;
    case GGML_UNARY_OP_ABS:
        return 6;
    case GGML_UNARY_OP_GELU:
        return 7;
    default:
        return -1;
    }
}

// Ollama 0.40.16 (llama.cpp b12177) inserts 5 ops before UNARY, shifting UNARY=92 and GLU=101.
constexpr int kOpUnary = 92;
constexpr int kOpGlu = 101;

bool op_supported(const ggml_tensor* op) {
    switch (static_cast<int>(op->op)) {
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_PERMUTE:
        return true;
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_CPY:
        return can_read_as_f32(op->src[0]) &&
               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
    case GGML_OP_PAD:
        return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
    case GGML_OP_GET_ROWS:
        return can_read_as_f32(op->src[0]) && op->src[1]->type == GGML_TYPE_I32;
    case GGML_OP_SET_ROWS:
        return can_read_as_f32(op->src[0]) &&
               (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32) &&
               op->type == GGML_TYPE_F32;
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM:
        return op->src[0]->type == GGML_TYPE_F32 && op->src[0]->ne[0] > 0 &&
               op->src[0]->ne[0] <= 4096;
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_SUB:
        return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
    case GGML_OP_SCALE:
        return op->src[0]->type == GGML_TYPE_F32;
    case GGML_OP_MUL_MAT:
        return can_read_as_f32(op->src[0]) && op->src[1]->type == GGML_TYPE_F32;
    case GGML_OP_SOFT_MAX:
        return op->src[0]->type == GGML_TYPE_F32 && op->src[0]->ne[0] > 0 &&
               op->src[0]->ne[0] <= 4096 &&
               (!op->src[1] || op->src[1]->type == GGML_TYPE_F32 ||
                op->src[1]->type == GGML_TYPE_F16);
    case kOpUnary:
        return op->src[0]->type == GGML_TYPE_F32 && unary_op_code(ggml_get_unary_op(op)) >= 0;
    case kOpGlu: {
        const enum ggml_glu_op glu = ggml_get_glu_op(op);
        return op->src[0]->type == GGML_TYPE_F32 &&
               (!op->src[1] || op->src[1]->type == GGML_TYPE_F32) && op->type == GGML_TYPE_F32 &&
               (glu == GGML_GLU_OP_REGLU || glu == GGML_GLU_OP_GEGLU || glu == GGML_GLU_OP_SWIGLU ||
                glu == GGML_GLU_OP_GEGLU_ERF || glu == GGML_GLU_OP_GEGLU_QUICK);
    }
    case GGML_OP_ROPE: {
        const int32_t mode = reinterpret_cast<const int32_t*>(op->op_params)[2];
        float ext_factor = 0.0f;
        std::memcpy(&ext_factor, reinterpret_cast<const int32_t*>(op->op_params) + 7,
                    sizeof(float));
        const bool supported_mode = (mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX);
        return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->src[1] &&
               op->src[1]->type == GGML_TYPE_I32 &&
               (!op->src[2] || op->src[2]->type == GGML_TYPE_F32) && supported_mode &&
               ext_factor == 0.0f;
    }
    case GGML_OP_SUM_ROWS:
        return op->src[0]->type == GGML_TYPE_F32 && op->src[0]->ne[0] > 0 &&
               op->src[0]->ne[0] <= 4096;
    case GGML_OP_DIV:
        return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
               ggml_nelements(op->src[1]) == 1;
    case GGML_OP_L2_NORM:
        return op->src[0]->type == GGML_TYPE_F32 && op->src[0]->ne[0] > 0 &&
               op->src[0]->ne[0] <= 4096;
    case GGML_OP_FLASH_ATTN_EXT: {
        const auto* q = op->src[0];
        const auto* k = op->src[1];
        const auto* v = op->src[2];
        const auto* mask = op->src[3];
        if (op->type != GGML_TYPE_F32 || op->src[4] != nullptr || op->src[5] != nullptr) {
            return false;
        }
        if (q->type != GGML_TYPE_F32) {
            return false;
        }
        if ((k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16) ||
            (v->type != GGML_TYPE_F32 && v->type != GGML_TYPE_F16)) {
            return false;
        }
        if (mask && mask->type != GGML_TYPE_F32 && mask->type != GGML_TYPE_F16) {
            return false;
        }
        float logit_softcap = 0.0f;
        std::memcpy(&logit_softcap, reinterpret_cast<const float*>(op->op_params) + 2,
                    sizeof(float));
        if (logit_softcap != 0.0f) {
            return false;
        }
        return k->ne[1] > 0 && k->ne[1] <= 4096;
    }
    default:
        return false;
    }
}

const char* backend_name(ggml_backend_t /*backend*/) {
    return "AIE2-XRT";
}

struct BackendContext {
    std::shared_ptr<XrtOpsRunner> ops;
};

void backend_free(ggml_backend_t backend) {
    if (!backend) {
        return;
    }
    delete static_cast<BackendContext*>(backend->context);
    delete backend;
}

ggml_status compute_node(XrtOpsRunner& ops, ggml_tensor* node) {
    switch (static_cast<int>(node->op)) {
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_PERMUTE:
        return GGML_STATUS_SUCCESS;
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_CPY: {
        std::vector<float> buf;
        pack_tensor_f32(node->src[0], buf);
        if (node->type == GGML_TYPE_F32 && ggml_is_contiguous(node)) {
            std::memcpy(node->data, buf.data(), buf.size() * sizeof(float));
        } else {
            int64_t idx = 0;
            for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                        for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                            write_elem_f32(node, i0, i1, i2, i3, buf[idx++]);
                        }
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_PAD: {
        const auto* src0 = node->src[0];
        int32_t p[9] = {};
        std::memcpy(p, node->op_params, sizeof(p));
        const int32_t lp0 = p[0];
        const int32_t rp0 = p[1];
        const int32_t lp1 = p[2];
        const int32_t rp1 = p[3];
        const int32_t lp2 = p[4];
        const int32_t rp2 = p[5];
        const int32_t lp3 = p[6];
        const int32_t rp3 = p[7];
        const bool circular = p[8] != 0;
        const auto wrap = [](int64_t x, int64_t n) -> int64_t {
            const int64_t r = x % n;
            return r < 0 ? r + n : r;
        };
        for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                        float val = 0.0f;
                        if (circular) {
                            val = read_elem_f32(
                                src0, wrap(i0 - lp0, src0->ne[0]), wrap(i1 - lp1, src0->ne[1]),
                                wrap(i2 - lp2, src0->ne[2]), wrap(i3 - lp3, src0->ne[3]));
                        } else if (i0 >= lp0 && i0 < node->ne[0] - rp0 && i1 >= lp1 &&
                                   i1 < node->ne[1] - rp1 && i2 >= lp2 && i2 < node->ne[2] - rp2 &&
                                   i3 >= lp3 && i3 < node->ne[3] - rp3) {
                            val = read_elem_f32(src0, i0 - lp0, i1 - lp1, i2 - lp2, i3 - lp3);
                        }
                        char* dst_ptr = static_cast<char*>(node->data) + (i0 * node->nb[0]) +
                                        (i1 * node->nb[1]) + (i2 * node->nb[2]) +
                                        (i3 * node->nb[3]);
                        *reinterpret_cast<float*>(dst_ptr) = val;
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_SET_ROWS: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        const int64_t nc = src0->ne[0];
        const int64_t nr = src0->ne[1];
        for (int64_t i03 = 0; i03 < src0->ne[3]; ++i03) {
            for (int64_t i02 = 0; i02 < src0->ne[2]; ++i02) {
                const int64_t i12 = i03 % src1->ne[2];
                const int64_t i11 = i02 % src1->ne[1];
                for (int64_t i = 0; i < nr; ++i) {
                    const char* idx_p = static_cast<const char*>(src1->data) + (i * src1->nb[0]) +
                                        (i11 * src1->nb[1]) + (i12 * src1->nb[2]);
                    const int64_t dst_row =
                        (src1->type == GGML_TYPE_I64)
                            ? *reinterpret_cast<const int64_t*>(idx_p)
                            : static_cast<int64_t>(*reinterpret_cast<const int32_t*>(idx_p));
                    if (dst_row < 0 || dst_row >= node->ne[1]) {
                        std::fprintf(
                            stderr,
                            "aie2: SET_ROWS dst_row=%lld out of range [0, %lld) (node=%s "
                            "src0=%s[%s] src1=%s[%s] ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] "
                            "data=%p)\n",
                            static_cast<long long>(dst_row), static_cast<long long>(node->ne[1]),
                            node->name, src0->name, ggml_type_name(src0->type), src1->name,
                            ggml_type_name(src1->type), static_cast<long long>(src1->ne[0]),
                            static_cast<long long>(src1->ne[1]),
                            static_cast<long long>(src1->ne[2]),
                            static_cast<long long>(src1->ne[3]), src1->nb[0], src1->nb[1],
                            src1->nb[2], src1->nb[3], src1->data);
                        return GGML_STATUS_FAILED;
                    }
                    char* dst_p = static_cast<char*>(node->data) + (dst_row * node->nb[1]) +
                                  (i02 * node->nb[2]) + (i03 * node->nb[3]);
                    for (int64_t c = 0; c < nc; ++c) {
                        *reinterpret_cast<float*>(dst_p + (c * node->nb[0])) =
                            read_elem_f32(src0, c, i, i02, i03);
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_GET_ROWS: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        auto* dst = static_cast<float*>(node->data);
        const int64_t n0 = src0->ne[0];
        for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < src1->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < src1->ne[0]; ++i0) {
                        const char* idx_p = static_cast<const char*>(src1->data) +
                                            (i0 * src1->nb[0]) + (i1 * src1->nb[1]) +
                                            (i2 * src1->nb[2]) + (i3 * src1->nb[3]);
                        const int32_t row = *reinterpret_cast<const int32_t*>(idx_p);
                        float* out_row =
                            dst + (((((i3 * src1->ne[2]) + i2) * src1->ne[1] * src1->ne[0]) +
                                    (i1 * src1->ne[0]) + i0) *
                                   n0);
                        read_row_f32(src0, row, i2, i3, out_row);
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM: {
        float eps = 1e-5f;
        std::memcpy(&eps, node->op_params, sizeof(float));
        std::vector<float> in_buf;
        pack_tensor_f32(node->src[0], in_buf);
        ops.layernorm_f32(in_buf.data(), static_cast<float*>(node->data), ggml_nrows(node->src[0]),
                          node->src[0]->ne[0], eps, node->op == GGML_OP_RMS_NORM);
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_SUB: {
        int32_t op_code = 0;
        if (node->op == GGML_OP_MUL) {
            op_code = 1;
        } else if (node->op == GGML_OP_SUB) {
            op_code = 2;
        }
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        std::vector<float> a_buf;
        pack_tensor_f32(src0, a_buf);
        const int64_t total = ggml_nelements(node);
        if (ggml_are_same_shape(src0, src1) && ggml_is_contiguous(src1)) {
            ops.eltwise_f32(a_buf.data(), static_cast<const float*>(src1->data),
                            static_cast<float*>(node->data), total, total, op_code);
        } else if (src1->ne[0] == node->ne[0] && ggml_nrows(src1) == 1 &&
                   ggml_is_contiguous(src1)) {
            ops.eltwise_f32(a_buf.data(), static_cast<const float*>(src1->data),
                            static_cast<float*>(node->data), total, node->ne[0], op_code);
        } else {
            std::vector<float> b_buf(static_cast<size_t>(total));
            int64_t idx = 0;
            for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                        for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                            b_buf[idx++] = read_elem_f32(src1, i0 % src1->ne[0], i1 % src1->ne[1],
                                                         i2 % src1->ne[2], i3 % src1->ne[3]);
                        }
                    }
                }
            }
            ops.eltwise_f32(a_buf.data(), b_buf.data(), static_cast<float*>(node->data), total,
                            total, op_code);
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_MUL_MAT: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        auto* dst = static_cast<float*>(node->data);
        const int64_t K = src0->ne[0];
        const int64_t N = src0->ne[1];
        const int64_t M = src1->ne[1];
        const int64_t r2 = src1->ne[2] / src0->ne[2];
        const int64_t r3 = src1->ne[3] / src0->ne[3];
        std::vector<float> a_slice(static_cast<size_t>(M * K));
        std::vector<float> b_slice(static_cast<size_t>(N * K));
        for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                const int64_t i03 = i3 / r3;
                const int64_t i02 = i2 / r2;
                for (int64_t m = 0; m < M; ++m) {
                    read_row_f32(src1, m, i2, i3, a_slice.data() + (m * K));
                }
                for (int64_t n = 0; n < N; ++n) {
                    read_row_f32(src0, n, i02, i03, b_slice.data() + (n * K));
                }
                float* c_ptr = dst + (((i3 * node->ne[2]) + i2) * M * N);
                ops.gemm_f32_nt(a_slice.data(), K, b_slice.data(), K, c_ptr, N, M, N, K);
            }
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_SOFT_MAX: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        float scale = 1.0f;
        std::memcpy(&scale, node->op_params, sizeof(float));
        std::vector<float> in_buf;
        pack_tensor_f32(src0, in_buf);
        std::vector<float> mask_buf;
        int64_t mask_rows = 0;
        if (src1) {
            pack_tensor_f32(src1, mask_buf);
            mask_rows = ggml_nrows(src1);
        }
        ops.softmax_f32(in_buf.data(), src1 ? mask_buf.data() : nullptr,
                        static_cast<float*>(node->data), ggml_nrows(src0), src0->ne[0], mask_rows,
                        scale);
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_SCALE: {
        float scale = 1.0f;
        std::memcpy(&scale, node->op_params, sizeof(float));
        std::vector<float> in_buf;
        pack_tensor_f32(node->src[0], in_buf);
        ops.eltwise_f32(in_buf.data(), &scale, static_cast<float*>(node->data),
                        ggml_nelements(node->src[0]), 1, 1);
        return GGML_STATUS_SUCCESS;
    }
    case kOpUnary: {
        std::vector<float> in_buf;
        pack_tensor_f32(node->src[0], in_buf);
        const enum ggml_unary_op uop = ggml_get_unary_op(node);
        if (uop == GGML_UNARY_OP_GELU) {
            ops.gelu_f32(in_buf.data(), static_cast<float*>(node->data),
                         ggml_nelements(node->src[0]));
        } else {
            const int32_t op_code = unary_op_code(uop);
            if (op_code < 0) {
                return GGML_STATUS_FAILED;
            }
            ops.unary_f32(in_buf.data(), static_cast<float*>(node->data),
                          ggml_nelements(node->src[0]), op_code);
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_SUM_ROWS: {
        std::vector<float> in_buf;
        pack_tensor_f32(node->src[0], in_buf);
        ops.reduce_sum_f32(in_buf.data(), static_cast<float*>(node->data), ggml_nrows(node->src[0]),
                           node->src[0]->ne[0], 1.0f);
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_DIV: {
        std::vector<float> in_buf;
        pack_tensor_f32(node->src[0], in_buf);
        const float inv_div = 1.0f / read_elem_f32(node->src[1], 0, 0, 0, 0);
        ops.eltwise_f32(in_buf.data(), &inv_div, static_cast<float*>(node->data),
                        ggml_nelements(node->src[0]), 1, 1);
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_L2_NORM: {
        float eps = 1e-12f;
        std::memcpy(&eps, node->op_params, sizeof(float));
        if (eps <= 0.0f)
            eps = 1e-12f;
        std::vector<float> in_buf;
        pack_tensor_f32(node->src[0], in_buf);
        ops.l2_norm_f32(in_buf.data(), static_cast<float*>(node->data), ggml_nrows(node->src[0]),
                        node->src[0]->ne[0], eps);
        return GGML_STATUS_SUCCESS;
    }
    case kOpGlu: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        const int64_t nc = node->ne[0];
        const int64_t nr = ggml_nrows(node);
        const int64_t total = nc * nr;
        int32_t swapped = 0;
        std::memcpy(&swapped, reinterpret_cast<const int32_t*>(node->op_params) + 1,
                    sizeof(int32_t));
        std::vector<float> gate(static_cast<size_t>(total));
        std::vector<float> val(static_cast<size_t>(total));
        if (src1) {
            const auto* g_t = swapped ? src1 : src0;
            const auto* v_t = swapped ? src0 : src1;
            pack_tensor_f32(g_t, gate);
            pack_tensor_f32(v_t, val);
        } else {
            std::vector<float> packed;
            pack_tensor_f32(src0, packed);
            for (int64_t r = 0; r < nr; ++r) {
                const float* row = packed.data() + (r * 2 * nc);
                const float* g_ptr = swapped ? (row + nc) : row;
                const float* v_ptr = swapped ? row : (row + nc);
                std::memcpy(gate.data() + (r * nc), g_ptr, static_cast<size_t>(nc) * sizeof(float));
                std::memcpy(val.data() + (r * nc), v_ptr, static_cast<size_t>(nc) * sizeof(float));
            }
        }
        std::vector<float> act(static_cast<size_t>(total));
        const enum ggml_glu_op glu = ggml_get_glu_op(node);
        if (glu == GGML_GLU_OP_SWIGLU) {
            ops.unary_f32(gate.data(), act.data(), total, 0);
        } else if (glu == GGML_GLU_OP_REGLU) {
            ops.unary_f32(gate.data(), act.data(), total, 3);
        } else {
            ops.gelu_f32(gate.data(), act.data(), total);
        }
        ops.eltwise_f32(act.data(), val.data(), static_cast<float*>(node->data), total, total, 1);
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_ROPE: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        const auto* src2 = node->src[2];
        const int32_t n_dims = reinterpret_cast<const int32_t*>(node->op_params)[1];
        const int32_t mode = reinterpret_cast<const int32_t*>(node->op_params)[2];
        float freq_base = 10000.0f;
        float freq_scale = 1.0f;
        float attn_factor = 1.0f;
        std::memcpy(&freq_base, reinterpret_cast<const int32_t*>(node->op_params) + 5,
                    sizeof(float));
        std::memcpy(&freq_scale, reinterpret_cast<const int32_t*>(node->op_params) + 6,
                    sizeof(float));
        std::memcpy(&attn_factor, reinterpret_cast<const int32_t*>(node->op_params) + 8,
                    sizeof(float));
        const bool is_neox =
            (static_cast<uint32_t>(mode) & static_cast<uint32_t>(GGML_ROPE_TYPE_NEOX)) != 0U;
        const auto* pos = static_cast<const int32_t*>(src1->data);
        const float* freq_factors = src2 ? static_cast<const float*>(src2->data) : nullptr;
        const int64_t ne0 = node->ne[0];
        const int64_t ne1 = node->ne[1];
        const int64_t ne2 = node->ne[2];
        const int64_t ne3 = node->ne[3];
        const int64_t half = n_dims / 2;
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                const float theta_base = static_cast<float>(pos[i2]) * freq_scale;
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    const auto* s = reinterpret_cast<const float*>(
                        static_cast<const char*>(src0->data) + (i3 * src0->nb[3]) +
                        (i2 * src0->nb[2]) + (i1 * src0->nb[1]));
                    auto* d = reinterpret_cast<float*>(static_cast<char*>(node->data) +
                                                       (i3 * node->nb[3]) + (i2 * node->nb[2]) +
                                                       (i1 * node->nb[1]));
                    for (int64_t ic = 0; ic < half; ++ic) {
                        float theta =
                            theta_base * std::pow(freq_base, -2.0f * static_cast<float>(ic) /
                                                                 static_cast<float>(n_dims));
                        if (freq_factors) {
                            theta /= freq_factors[ic];
                        }
                        const float c = std::cos(theta) * attn_factor;
                        const float sn = std::sin(theta) * attn_factor;
                        const int64_t idx0 = is_neox ? ic : (2 * ic);
                        const int64_t idx1 = is_neox ? (ic + half) : ((2 * ic) + 1);
                        const float x0 = s[idx0];
                        const float x1 = s[idx1];
                        d[idx0] = (x0 * c) - (x1 * sn);
                        d[idx1] = (x0 * sn) + (x1 * c);
                    }
                    for (int64_t i0 = n_dims; i0 < ne0; ++i0) {
                        d[i0] = s[i0];
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_FLASH_ATTN_EXT: {
        const auto* q = node->src[0];
        const auto* k = node->src[1];
        const auto* v = node->src[2];
        const auto* mask = node->src[3];

        const int64_t DK = q->ne[0];
        const int64_t N = q->ne[1];
        const int64_t n_head = q->ne[2];
        const int64_t n_stream = q->ne[3];
        const int64_t n_kv = k->ne[1];
        const int64_t DV = v->ne[0];
        const int64_t rk2 = n_head / k->ne[2];
        const int64_t rk3 = n_stream / k->ne[3];
        const int64_t rv2 = n_head / v->ne[2];
        const int64_t rv3 = n_stream / v->ne[3];

        float scale = 1.0f;
        float max_bias = 0.0f;
        std::memcpy(&scale, reinterpret_cast<const float*>(node->op_params) + 0, sizeof(float));
        std::memcpy(&max_bias, reinterpret_cast<const float*>(node->op_params) + 1, sizeof(float));

        const uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(n_head)));
        const float m0 = std::pow(2.0f, -max_bias / static_cast<float>(n_head_log2));
        const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<float>(n_head_log2));

        std::vector<float> q_buf(static_cast<size_t>(N * DK));
        std::vector<float> k_buf(static_cast<size_t>(n_kv * DK));
        std::vector<float> kq_buf(static_cast<size_t>(N * n_kv));
        std::vector<float> mask_buf(mask ? static_cast<size_t>(N * n_kv) : 0);
        std::vector<float> probs_buf(static_cast<size_t>(N * n_kv));
        std::vector<float> v_buf(static_cast<size_t>(DV * n_kv));
        std::vector<float> out_buf(static_cast<size_t>(N * DV));

        for (int64_t iq3 = 0; iq3 < n_stream; ++iq3) {
            for (int64_t iq2 = 0; iq2 < n_head; ++iq2) {
                const int64_t ik2 = iq2 / rk2;
                const int64_t ik3 = iq3 / rk3;
                const int64_t iv2 = iq2 / rv2;
                const int64_t iv3 = iq3 / rv3;

                for (int64_t t = 0; t < N; ++t) {
                    for (int64_t d = 0; d < DK; ++d) {
                        q_buf[(t * DK) + d] = read_elem_f32(q, d, t, iq2, iq3);
                    }
                }
                for (int64_t ic = 0; ic < n_kv; ++ic) {
                    for (int64_t d = 0; d < DK; ++d) {
                        k_buf[(ic * DK) + d] = read_elem_f32(k, d, ic, ik2, ik3);
                    }
                }
                ops.gemm_f32_nt(q_buf.data(), DK, k_buf.data(), DK, kq_buf.data(), n_kv, N, n_kv,
                                DK);

                if (mask) {
                    float slope = 1.0f;
                    if (max_bias > 0.0f) {
                        if (std::cmp_less(iq2, n_head_log2)) {
                            slope = std::pow(m0, static_cast<float>(iq2 + 1));
                        } else {
                            slope = std::pow(
                                m1, static_cast<float>(
                                        (2 * (iq2 - static_cast<int64_t>(n_head_log2))) + 1));
                        }
                    }
                    const int64_t im2 = iq2 % mask->ne[2];
                    const int64_t im3 = iq3 % mask->ne[3];
                    for (int64_t t = 0; t < N; ++t) {
                        for (int64_t ic = 0; ic < n_kv; ++ic) {
                            mask_buf[(t * n_kv) + ic] =
                                slope * read_elem_f32(mask, ic, t, im2, im3);
                        }
                    }
                }
                ops.softmax_f32(kq_buf.data(), mask ? mask_buf.data() : nullptr, probs_buf.data(),
                                N, n_kv, mask ? N : 0, scale);

                for (int64_t d = 0; d < DV; ++d) {
                    for (int64_t ic = 0; ic < n_kv; ++ic) {
                        v_buf[(d * n_kv) + ic] = read_elem_f32(v, d, ic, iv2, iv3);
                    }
                }
                ops.gemm_f32_nt(probs_buf.data(), n_kv, v_buf.data(), n_kv, out_buf.data(), DV, N,
                                DV, n_kv);

                for (int64_t t = 0; t < N; ++t) {
                    char* dst_row = static_cast<char*>(node->data) + (iq2 * node->nb[1]) +
                                    (t * node->nb[2]) + (iq3 * node->nb[3]);
                    if (node->nb[0] == sizeof(float)) {
                        std::memcpy(dst_row, out_buf.data() + (t * DV),
                                    static_cast<size_t>(DV) * sizeof(float));
                    } else {
                        for (int64_t d = 0; d < DV; ++d) {
                            *reinterpret_cast<float*>(dst_row + (d * node->nb[0])) =
                                out_buf[(t * DV) + d];
                        }
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
    default:
        std::fprintf(stderr, "aie2: unsupported op %s\n", ggml_op_name(node->op));
        return GGML_STATUS_FAILED;
    }
}

ggml_status graph_compute(ggml_backend_t backend, ggml_cgraph* graph) {
    const auto* ctx = static_cast<const BackendContext*>(backend->context);
    if (!ctx || !ctx->ops) {
        std::fprintf(stderr, "aie2: graph_compute called with null ctx/ops\n");
        return GGML_STATUS_FAILED;
    }
    const bool trace = std::getenv("AIE2_TRACE") != nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto* node = ggml_graph_node(graph, i);
        if ((static_cast<uint32_t>(node->flags) &
             static_cast<uint32_t>(GGML_TENSOR_FLAG_COMPUTE)) == 0U) {
            continue;
        }
        if (trace) {
            std::fprintf(stderr, "aie2: node[%d] op=%s ne=[%lld,%lld,%lld,%lld]\n", i,
                         ggml_op_name(node->op), static_cast<long long>(node->ne[0]),
                         static_cast<long long>(node->ne[1]), static_cast<long long>(node->ne[2]),
                         static_cast<long long>(node->ne[3]));
        }
        try {
            const ggml_status st = compute_node(*ctx->ops, node);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "aie2: node[%d] op=%s returned status %d\n", i,
                             ggml_op_name(node->op), static_cast<int>(st));
                return st;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "aie2: XRT compute failed (op=%s): %s\n", ggml_op_name(node->op),
                         e.what());
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

const ggml_backend_i backend_iface = {
    backend_name, backend_free, nullptr, nullptr, nullptr,       nullptr, nullptr, nullptr,
    nullptr,      nullptr,      nullptr, nullptr, graph_compute, nullptr, nullptr, nullptr,
};

const char* dev_name(ggml_backend_dev_t /*dev*/) {
    return "AIE2-XRT";
}

const char* dev_description(ggml_backend_dev_t /*dev*/) {
    return "AMD AIE2 XRT native full-graph encoder backend";
}

void dev_memory(ggml_backend_dev_t /*dev*/, size_t* free, size_t* total) {
    const long page_size = sysconf(_SC_PAGESIZE);
    const long phys_pages = sysconf(_SC_PHYS_PAGES);
    const long avail_pages = sysconf(_SC_AVPHYS_PAGES);
    if (page_size > 0 && phys_pages > 0) {
        const size_t total_ram = static_cast<size_t>(phys_pages) * static_cast<size_t>(page_size);
        const size_t avail_ram =
            (avail_pages > 0) ? (static_cast<size_t>(avail_pages) * static_cast<size_t>(page_size))
                              : (total_ram / 4);
        *total = total_ram / 4;
        *free = std::min(avail_ram, *total);
    } else {
        *free = *total = 0;
    }
}

enum ggml_backend_dev_type dev_type(ggml_backend_dev_t /*dev*/) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

void dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props* props) {
    props->name = dev_name(dev);
    props->description = dev_description(dev);
    props->type = dev_type(dev);
    dev_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {};
}

ggml_backend_t dev_init(ggml_backend_dev_t dev, const char* /*params*/) {
    auto ops = get_ops();
    if (!ops) {
        return nullptr;
    }
    auto* ctx = new BackendContext{std::move(ops)};
    static ggml_guid guid = {0x41, 0x49, 0x45, 0x32, 0x2d, 0x58, 0x52, 0x54,
                             0,    0,    0,    0,    0,    0,    0,    2};
    return new ggml_backend{&guid, backend_iface, dev, ctx};
}

ggml_backend_buffer_type_t dev_buft(ggml_backend_dev_t /*dev*/) {
    return ggml_backend_cpu_buffer_type();
}

bool dev_supports(ggml_backend_dev_t /*dev*/, const ggml_tensor* op) {
    return op_supported(op);
}

bool dev_supports_buft(ggml_backend_dev_t /*dev*/, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
}

const ggml_backend_device_i device_iface = {
    dev_name, dev_description, dev_memory,        dev_type, dev_props, dev_init, dev_buft, nullptr,
    nullptr,  dev_supports,    dev_supports_buft, nullptr,  nullptr,   nullptr,  nullptr,
};

const char* reg_name(ggml_backend_reg_t /*reg*/) {
    return "AIE2-XRT";
}

size_t reg_count(ggml_backend_reg_t /*reg*/) {
    return get_ops() ? 1 : 0;
}

ggml_backend_dev_t reg_dev(ggml_backend_reg_t /*reg*/, size_t index) {
    return index == 0 && get_ops() ? &device : nullptr;
}

const ggml_backend_reg_i reg_iface = {reg_name, reg_count, reg_dev, nullptr};
// Ollama 0.40.16 (llama.cpp b12177) expects GGML_BACKEND_API_VERSION == 3.
constexpr int kGgmlBackendApiVersion = 3;
ggml_backend_reg registry = {kGgmlBackendApiVersion, reg_iface, nullptr};
ggml_backend_device device = {device_iface, &registry, nullptr};

} // namespace

static ggml_backend_reg_t backend_init() {
    return &registry;
}

GGML_BACKEND_DL_IMPL(backend_init)
