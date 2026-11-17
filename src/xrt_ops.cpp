#include "xrt_ops.h"

#include "xrt_context.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

XrtOpsRunner::XrtOpsRunner(std::string artifacts_dir) : dir_(std::move(artifacts_dir)) {
    XrtContextManager::instance().initialize(dir_);
}

void XrtOpsRunner::layernorm_f32(const float* in, float* out, int64_t n_rows, int64_t row_len,
                                 float eps, bool is_rms) {
    if (row_len <= 0 || row_len > 4096) {
        throw std::runtime_error("layernorm_f32 requires 1 <= row_len <= 4096");
    }
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto& prog = mgr.slot(Aie2OpKind::kLayerNorm);

    constexpr int64_t kCap = 4096;
    const int64_t max_rows = std::max<int64_t>(1, kCap / row_len);

    auto* in_buf = prog.arg_ptr<float>(0);
    const auto* out_buf = prog.arg_ptr<float>(1);
    auto* aux_buf = prog.arg_ptr<float>(2);
    auto& bo_in = prog.arg_bo(0);
    auto& bo_out = prog.arg_bo(1);
    auto& bo_aux = prog.arg_bo(2);

    std::fill_n(aux_buf, 16, 0.0f);
    aux_buf[0] = eps;
    auto rl_i32 = static_cast<int32_t>(row_len);
    int32_t rms_i32 = is_rms ? 1 : 0;
    std::memcpy(&aux_buf[1], &rl_i32, sizeof(rl_i32));
    std::memcpy(&aux_buf[3], &rms_i32, sizeof(rms_i32));

    for (int64_t r = 0; r < n_rows; r += max_rows) {
        const int64_t batch = std::min<int64_t>(max_rows, n_rows - r);
        const int64_t elems = batch * row_len;
        auto br_i32 = static_cast<int32_t>(batch);
        std::memcpy(&aux_buf[2], &br_i32, sizeof(br_i32));

        std::memcpy(in_buf, in + (r * row_len), static_cast<size_t>(elems) * sizeof(float));
        if (elems < kCap) {
            std::fill(in_buf + elems, in_buf + kCap, 0.0f);
        }

        mgr.timed_sync(bo_in, XCL_BO_SYNC_BO_TO_DEVICE);
        mgr.timed_sync(bo_aux, XCL_BO_SYNC_BO_TO_DEVICE);
        mgr.timed_run_wait(prog, bo_in, bo_out, bo_aux);
        mgr.timed_sync(bo_out, XCL_BO_SYNC_BO_FROM_DEVICE);

        std::memcpy(out + (r * row_len), out_buf, static_cast<size_t>(elems) * sizeof(float));
    }
}

void XrtOpsRunner::softmax_f32(const float* in, const float* mask, float* out, int64_t n_rows,
                               int64_t row_len, int64_t mask_rows, float scale) {
    if (row_len <= 0 || row_len > 4096) {
        throw std::runtime_error("softmax_f32 requires 1 <= row_len <= 4096");
    }
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto& prog = mgr.slot(Aie2OpKind::kSoftmax);

    constexpr int64_t kCap = 4096;
    const int64_t stride = row_len;
    const int64_t max_rows = std::max<int64_t>(1, kCap / stride);

    auto* in_buf = prog.arg_ptr<float>(0);
    auto* aux_buf = prog.arg_ptr<float>(1);
    const auto* out_buf = prog.arg_ptr<float>(2);
    auto& bo_in = prog.arg_bo(0);
    auto& bo_aux = prog.arg_bo(1);
    auto& bo_out = prog.arg_bo(2);

    std::fill(aux_buf + kCap, aux_buf + 4112, 0.0f);
    auto valid_len = static_cast<int32_t>(row_len);
    auto stride_i32 = static_cast<int32_t>(stride);
    std::memcpy(&aux_buf[4096], &valid_len, sizeof(valid_len));
    aux_buf[4097] = scale;
    std::memcpy(&aux_buf[4099], &stride_i32, sizeof(stride_i32));

    for (int64_t r0 = 0; r0 < n_rows; r0 += max_rows) {
        const int64_t batch = std::min<int64_t>(max_rows, n_rows - r0);
        const int64_t elems = batch * stride;
        auto br_i32 = static_cast<int32_t>(batch);
        std::memcpy(&aux_buf[4098], &br_i32, sizeof(br_i32));

        std::memcpy(in_buf, in + (r0 * row_len), static_cast<size_t>(elems) * sizeof(float));
        if (elems < kCap) {
            std::fill(in_buf + elems, in_buf + kCap, 0.0f);
        }
        if (mask && mask_rows > 0) {
            for (int64_t r = 0; r < batch; ++r) {
                const int64_t mr = (r0 + r) % mask_rows;
                std::memcpy(aux_buf + (r * stride), mask + (mr * row_len),
                            static_cast<size_t>(row_len) * sizeof(float));
            }
        } else {
            std::fill_n(aux_buf, static_cast<size_t>(elems), 0.0f);
        }

        mgr.timed_sync(bo_in, XCL_BO_SYNC_BO_TO_DEVICE);
        mgr.timed_sync(bo_aux, XCL_BO_SYNC_BO_TO_DEVICE);
        mgr.timed_run_wait(prog, bo_in, bo_aux, bo_out);
        mgr.timed_sync(bo_out, XCL_BO_SYNC_BO_FROM_DEVICE);

        std::memcpy(out + (r0 * row_len), out_buf, static_cast<size_t>(elems) * sizeof(float));
    }
}

void XrtOpsRunner::eltwise_f32(const float* a, const float* b, float* out, int64_t total_elems,
                               int64_t b_period, int32_t op) {
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto& prog = mgr.slot(Aie2OpKind::kEltwise);

    constexpr int64_t kChunk = 4096;
    auto* a_buf = prog.arg_ptr<float>(0);
    auto* aux_buf = prog.arg_ptr<float>(1);
    const auto* out_buf = prog.arg_ptr<float>(2);
    auto& bo_a = prog.arg_bo(0);
    auto& bo_aux = prog.arg_bo(1);
    auto& bo_out = prog.arg_bo(2);

    std::fill(aux_buf + kChunk, aux_buf + 4112, 0.0f);
    auto full_period = static_cast<int32_t>(kChunk);
    std::memcpy(&aux_buf[4096], &op, sizeof(op));
    std::memcpy(&aux_buf[4097], &full_period, sizeof(full_period));

    for (int64_t off = 0; off < total_elems; off += kChunk) {
        const int64_t n = std::min<int64_t>(kChunk, total_elems - off);
        std::memcpy(a_buf, a + off, static_cast<size_t>(n) * sizeof(float));
        if (n < kChunk) {
            std::fill(a_buf + n, a_buf + kChunk, 0.0f);
        }
        if (b_period >= total_elems) {
            std::memcpy(aux_buf, b + off, static_cast<size_t>(n) * sizeof(float));
        } else if (b_period == 1) {
            std::fill_n(aux_buf, static_cast<size_t>(n), b[0]);
        } else {
            for (int64_t i = 0; i < n; ++i) {
                aux_buf[i] = b[(off + i) % b_period];
            }
        }
        mgr.timed_sync(bo_a, XCL_BO_SYNC_BO_TO_DEVICE);
        mgr.timed_sync(bo_aux, XCL_BO_SYNC_BO_TO_DEVICE);
        mgr.timed_run_wait(prog, bo_a, bo_aux, bo_out);
        mgr.timed_sync(bo_out, XCL_BO_SYNC_BO_FROM_DEVICE);

        std::memcpy(out + off, out_buf, static_cast<size_t>(n) * sizeof(float));
    }
}

void XrtOpsRunner::unary_f32(const float* in, float* out, int64_t total_elems, int32_t op) {
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto& prog = mgr.slot(Aie2OpKind::kUnary);

    constexpr int64_t kChunk = 4096;
    auto* in_buf = prog.arg_ptr<float>(0);
    const auto* out_buf = prog.arg_ptr<float>(1);
    auto* aux_buf = prog.arg_ptr<float>(2);
    auto& bo_in = prog.arg_bo(0);
    auto& bo_out = prog.arg_bo(1);
    auto& bo_aux = prog.arg_bo(2);

    std::fill_n(aux_buf, 16, 0.0f);
    std::memcpy(aux_buf, &op, sizeof(op));
    mgr.timed_sync(bo_aux, XCL_BO_SYNC_BO_TO_DEVICE);

    for (int64_t off = 0; off < total_elems; off += kChunk) {
        const int64_t n = std::min<int64_t>(kChunk, total_elems - off);
        std::memcpy(in_buf, in + off, static_cast<size_t>(n) * sizeof(float));
        if (n < kChunk) {
            std::fill(in_buf + n, in_buf + kChunk, 0.0f);
        }

        mgr.timed_sync(bo_in, XCL_BO_SYNC_BO_TO_DEVICE);
        mgr.timed_run_wait(prog, bo_in, bo_out, bo_aux);
        mgr.timed_sync(bo_out, XCL_BO_SYNC_BO_FROM_DEVICE);

        std::memcpy(out + off, out_buf, static_cast<size_t>(n) * sizeof(float));
    }
}

void XrtOpsRunner::gelu_f32(const float* in, float* out, int64_t total_elems) {
    unary_f32(in, out, total_elems, 7);
}

void XrtOpsRunner::gemm_f32_nt(const float* a, int64_t lda, const float* b, int64_t ldb, float* c,
                               int64_t ldc, int64_t M, int64_t N, int64_t K) {
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto& prog = mgr.slot(Aie2OpKind::kGemmF32);

    constexpr int64_t TM = kGemmTM;
    constexpr int64_t TK = kGemmTK;
    constexpr int64_t TN_W = kGemmTNW;
    constexpr int64_t NW = kGemmNW;
    constexpr int64_t TN = kGemmTN;
    constexpr auto kTileElems = static_cast<size_t>(TM * TK);

    auto* a_tile = prog.arg_ptr<float>(0);
    auto* b_tile = prog.arg_ptr<float>(1);
    const auto* c_tile = prog.arg_ptr<float>(2);
    auto& bo_a = prog.arg_bo(0);
    auto& bo_b = prog.arg_bo(1);
    auto& bo_c = prog.arg_bo(2);

    std::vector<float> accum(static_cast<size_t>(TM * N));
    for (int64_t m0 = 0; m0 < M; m0 += TM) {
        const int64_t mb = std::min<int64_t>(TM, M - m0);
        std::fill_n(accum.data(), static_cast<size_t>(mb * N), 0.0f);
        for (int64_t k0 = 0; k0 < K; k0 += TK) {
            const int64_t kb = std::min<int64_t>(TK, K - k0);
            if (mb < TM || kb < TK) {
                std::fill_n(a_tile, kTileElems, 0.0f);
            }
            for (int64_t m = 0; m < mb; ++m) {
                std::memcpy(a_tile + (m * TK), a + ((m0 + m) * lda) + k0,
                            static_cast<size_t>(kb) * sizeof(float));
            }
            mgr.timed_sync(bo_a, XCL_BO_SYNC_BO_TO_DEVICE);

            for (int64_t n0 = 0; n0 < N; n0 += TN) {
                const int64_t nb = std::min<int64_t>(TN, N - n0);
                const int64_t active_w = (nb + TN_W - 1) / TN_W;
                if (kb < TK || nb < TN) {
                    std::fill_n(b_tile, NW * kTileElems, 0.0f);
                }
                for (int64_t w = 0; w < active_w; ++w) {
                    const int64_t wn0 = n0 + (w * TN_W);
                    const int64_t wnb = std::min<int64_t>(TN_W, N - wn0);
                    float* bw = b_tile + (static_cast<size_t>(w) * kTileElems);
                    for (int64_t n = 0; n < wnb; ++n) {
                        std::memcpy(bw + (n * TK), b + ((wn0 + n) * ldb) + k0,
                                    static_cast<size_t>(kb) * sizeof(float));
                    }
                }
                mgr.timed_sync(bo_b, XCL_BO_SYNC_BO_TO_DEVICE);
                mgr.timed_run_wait(prog, bo_a, bo_b, bo_c);
                mgr.timed_sync(bo_c, XCL_BO_SYNC_BO_FROM_DEVICE);

                for (int64_t w = 0; w < active_w; ++w) {
                    const int64_t wn0 = n0 + (w * TN_W);
                    const int64_t wnb = std::min<int64_t>(TN_W, N - wn0);
                    const float* cw = c_tile + (static_cast<size_t>(w) * kTileElems);
                    for (int64_t m = 0; m < mb; ++m) {
                        float* acc_row = accum.data() + (m * N) + wn0;
                        const float* c_row = cw + (m * TN_W);
                        for (int64_t n = 0; n < wnb; ++n) {
                            acc_row[n] += c_row[n];
                        }
                    }
                }
            }
        }
        for (int64_t m = 0; m < mb; ++m) {
            std::memcpy(c + ((m0 + m) * ldc), accum.data() + (m * N),
                        static_cast<size_t>(N) * sizeof(float));
        }
    }
}

void XrtOpsRunner::gemm_f32_nt_prepacked(const float* a, int64_t lda, const float* b_packed,
                                         int64_t n_blocks, int64_t k_blocks, float* c, int64_t ldc,
                                         int64_t M, int64_t N, int64_t K) {
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto& prog = mgr.slot(Aie2OpKind::kGemmF32);

    constexpr int64_t TM = kGemmTM;
    constexpr int64_t TK = kGemmTK;
    constexpr int64_t TN_W = kGemmTNW;
    constexpr int64_t NW = kGemmNW;
    constexpr int64_t TN = kGemmTN;
    constexpr auto kTileElems = static_cast<size_t>(TM * TK);
    constexpr auto kBlockElems = static_cast<size_t>(NW) * kTileElems;

    auto* a_tile = prog.arg_ptr<float>(0);
    auto* b_tile = prog.arg_ptr<float>(1);
    const auto* c_tile = prog.arg_ptr<float>(2);
    auto& bo_a = prog.arg_bo(0);
    auto& bo_b = prog.arg_bo(1);
    auto& bo_c = prog.arg_bo(2);

    std::vector<float> accum(static_cast<size_t>(TM * N));
    for (int64_t m0 = 0; m0 < M; m0 += TM) {
        const int64_t mb = std::min<int64_t>(TM, M - m0);
        std::fill_n(accum.data(), static_cast<size_t>(mb * N), 0.0f);
        for (int64_t kb_idx = 0; kb_idx < k_blocks; ++kb_idx) {
            const int64_t k0 = kb_idx * TK;
            const int64_t kb = std::min<int64_t>(TK, K - k0);
            if (mb < TM || kb < TK) {
                std::fill_n(a_tile, kTileElems, 0.0f);
            }
            for (int64_t m = 0; m < mb; ++m) {
                std::memcpy(a_tile + (m * TK), a + ((m0 + m) * lda) + k0,
                            static_cast<size_t>(kb) * sizeof(float));
            }
            mgr.timed_sync(bo_a, XCL_BO_SYNC_BO_TO_DEVICE);

            for (int64_t nb_idx = 0; nb_idx < n_blocks; ++nb_idx) {
                const int64_t n0 = nb_idx * TN;
                const int64_t nb = std::min<int64_t>(TN, N - n0);
                const int64_t active_w = (nb + TN_W - 1) / TN_W;
                const float* b_src =
                    b_packed + (static_cast<size_t>((nb_idx * k_blocks) + kb_idx) * kBlockElems);
                std::memcpy(b_tile, b_src, kBlockElems * sizeof(float));
                mgr.timed_sync(bo_b, XCL_BO_SYNC_BO_TO_DEVICE);
                mgr.timed_run_wait(prog, bo_a, bo_b, bo_c);
                mgr.timed_sync(bo_c, XCL_BO_SYNC_BO_FROM_DEVICE);

                for (int64_t w = 0; w < active_w; ++w) {
                    const int64_t wn0 = n0 + (w * TN_W);
                    const int64_t wnb = std::min<int64_t>(TN_W, N - wn0);
                    const float* cw = c_tile + (static_cast<size_t>(w) * kTileElems);
                    for (int64_t m = 0; m < mb; ++m) {
                        float* acc_row = accum.data() + (m * N) + wn0;
                        const float* c_row = cw + (m * TN_W);
                        for (int64_t n = 0; n < wnb; ++n) {
                            acc_row[n] += c_row[n];
                        }
                    }
                }
            }
        }
        for (int64_t m = 0; m < mb; ++m) {
            std::memcpy(c + ((m0 + m) * ldc), accum.data() + (m * N),
                        static_cast<size_t>(N) * sizeof(float));
        }
    }
}

void XrtOpsRunner::l2_norm_f32(const float* in, float* out, int64_t n_vecs, int64_t len,
                               float eps) {
    if (len <= 0 || len > 4096) {
        throw std::runtime_error("l2_norm_f32 requires 1 <= len <= 4096");
    }
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto& prog = mgr.slot(Aie2OpKind::kL2Norm);

    constexpr int64_t kCap = 4096;
    const int64_t max_vecs = std::max<int64_t>(1, kCap / len);
    auto* in_buf = prog.arg_ptr<float>(0);
    const auto* out_buf = prog.arg_ptr<float>(1);
    auto* aux_buf = prog.arg_ptr<float>(2);
    auto& bo_in = prog.arg_bo(0);
    auto& bo_out = prog.arg_bo(1);
    auto& bo_aux = prog.arg_bo(2);

    std::fill_n(aux_buf, 16, 0.0f);
    aux_buf[0] = eps;
    auto len_i32 = static_cast<int32_t>(len);
    std::memcpy(&aux_buf[1], &len_i32, sizeof(len_i32));

    for (int64_t v = 0; v < n_vecs; v += max_vecs) {
        const int64_t batch = std::min<int64_t>(max_vecs, n_vecs - v);
        const int64_t elems = batch * len;
        auto bv_i32 = static_cast<int32_t>(batch);
        std::memcpy(&aux_buf[2], &bv_i32, sizeof(bv_i32));

        std::memcpy(in_buf, in + (v * len), static_cast<size_t>(elems) * sizeof(float));
        if (elems < kCap) {
            std::fill(in_buf + elems, in_buf + kCap, 0.0f);
        }

        mgr.timed_sync(bo_in, XCL_BO_SYNC_BO_TO_DEVICE);
        mgr.timed_sync(bo_aux, XCL_BO_SYNC_BO_TO_DEVICE);
        mgr.timed_run_wait(prog, bo_in, bo_out, bo_aux);
        mgr.timed_sync(bo_out, XCL_BO_SYNC_BO_FROM_DEVICE);

        std::memcpy(out + (v * len), out_buf, static_cast<size_t>(elems) * sizeof(float));
    }
}

void XrtOpsRunner::reduce_sum_f32(const float* in, float* out, int64_t n_rows, int64_t row_len,
                                  float scale) {
    if (row_len <= 0 || row_len > 4096) {
        throw std::runtime_error("reduce_sum_f32 requires 1 <= row_len <= 4096");
    }
    std::vector<float> ones(static_cast<size_t>(row_len), scale);
    gemm_f32_nt(in, row_len, ones.data(), row_len, out, 1, n_rows, 1, row_len);
}
