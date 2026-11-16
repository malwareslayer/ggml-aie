#include "xrt_ops.h"

#include "xrt_context.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

uint16_t f32_to_bf16(float v) {
    const auto bits = std::bit_cast<uint32_t>(v);
    const uint32_t bias = ((bits >> 16U) & 1U) + 0x7FFFU;
    return static_cast<uint16_t>((bits + bias) >> 16U);
}

float bf16_to_f32(uint16_t h) {
    return std::bit_cast<float>(static_cast<uint32_t>(h) << 16U);
}

struct LoadedOp {
    XrtContextManager::ProgramHandle prog;
    xrt::bo insts_bo;
};

LoadedOp load_op(XrtContextManager& mgr, const std::string& dir, const char* op_name) {
    const fs::path base = fs::path(dir) / op_name;
    auto prog = mgr.acquire((base / (std::string(op_name) + ".xclbin")).string(),
                            (base / "insts.bin").string(), "MLIR_AIE");
    xrt::bo insts_bo(*prog.device, prog.insts.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
                     prog.kernel->group_id(1));
    insts_bo.write(prog.insts.data());
    insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return {std::move(prog), std::move(insts_bo)};
}

} // namespace

XrtOpsRunner::XrtOpsRunner(std::string artifacts_dir) : dir_(std::move(artifacts_dir)) {}

void XrtOpsRunner::layernorm_f32(const float* in, float* out, int64_t n_rows, int64_t row_len,
                                 float eps, bool is_rms) {
    if (row_len <= 0 || row_len > 4096) {
        throw std::runtime_error("layernorm_f32 requires 1 <= row_len <= 4096");
    }
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto [prog, insts_bo] = load_op(mgr, dir_, "layernorm");

    constexpr int64_t kCap = 4096;
    const int64_t max_rows = std::max<int64_t>(1, kCap / row_len);
    std::vector<float> in_buf(kCap, 0.0f);
    std::vector<float> out_buf(kCap, 0.0f);
    std::vector<float> aux_buf(16, 0.0f);

    aux_buf[0] = eps;
    auto rl_i32 = static_cast<int32_t>(row_len);
    int32_t rms_i32 = is_rms ? 1 : 0;
    std::memcpy(&aux_buf[1], &rl_i32, sizeof(rl_i32));
    std::memcpy(&aux_buf[3], &rms_i32, sizeof(rms_i32));

    xrt::bo bo_in(*prog.device, in_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                  prog.kernel->group_id(3));
    xrt::bo bo_out(*prog.device, out_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                   prog.kernel->group_id(4));
    xrt::bo bo_aux(*prog.device, aux_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                   prog.kernel->group_id(5));

    for (int64_t r = 0; r < n_rows; r += max_rows) {
        const int64_t batch = std::min<int64_t>(max_rows, n_rows - r);
        auto br_i32 = static_cast<int32_t>(batch);
        std::memcpy(&aux_buf[2], &br_i32, sizeof(br_i32));

        std::ranges::fill(in_buf, 0.0f);
        std::memcpy(in_buf.data(), in + (r * row_len),
                    static_cast<size_t>(batch * row_len) * sizeof(float));

        bo_in.write(in_buf.data());
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_aux.write(aux_buf.data());
        bo_aux.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const auto run = (*prog.kernel)(3u, insts_bo, static_cast<uint32_t>(prog.insts.size()),
                                        bo_in, bo_out, bo_aux);
        run.wait();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(out_buf.data());
        std::memcpy(out + (r * row_len), out_buf.data(),
                    static_cast<size_t>(batch * row_len) * sizeof(float));
    }
}

void XrtOpsRunner::softmax_f32(const float* in, const float* mask, float* out, int64_t n_rows,
                               int64_t row_len, int64_t mask_rows, float scale) {
    if (row_len <= 0 || row_len > 4096) {
        throw std::runtime_error("softmax_f32 requires 1 <= row_len <= 4096");
    }
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto [prog, insts_bo] = load_op(mgr, dir_, "softmax");

    constexpr int64_t kCap = 4096;
    const int64_t stride = row_len;
    const int64_t max_rows = std::max<int64_t>(1, kCap / stride);
    std::vector<float> in_buf(kCap, 0.0f);
    std::vector<float> aux_buf(4112, 0.0f);
    std::vector<float> out_buf(kCap, 0.0f);

    auto valid_len = static_cast<int32_t>(row_len);
    auto stride_i32 = static_cast<int32_t>(stride);
    std::memcpy(&aux_buf[4096], &valid_len, sizeof(valid_len));
    aux_buf[4097] = scale;
    std::memcpy(&aux_buf[4099], &stride_i32, sizeof(stride_i32));

    xrt::bo bo_in(*prog.device, in_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                  prog.kernel->group_id(3));
    xrt::bo bo_aux(*prog.device, aux_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                   prog.kernel->group_id(4));
    xrt::bo bo_out(*prog.device, out_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                   prog.kernel->group_id(5));

    for (int64_t r0 = 0; r0 < n_rows; r0 += max_rows) {
        const int64_t batch = std::min<int64_t>(max_rows, n_rows - r0);
        auto br_i32 = static_cast<int32_t>(batch);
        std::memcpy(&aux_buf[4098], &br_i32, sizeof(br_i32));

        std::ranges::fill(in_buf, 0.0f);
        std::ranges::fill(aux_buf.begin(), aux_buf.begin() + kCap, 0.0f);
        for (int64_t r = 0; r < batch; ++r) {
            const int64_t row_idx = r0 + r;
            std::memcpy(in_buf.data() + (r * stride), in + (row_idx * row_len),
                        static_cast<size_t>(row_len) * sizeof(float));
            if (mask && mask_rows > 0) {
                const int64_t mr = row_idx % mask_rows;
                std::memcpy(aux_buf.data() + (r * stride), mask + (mr * row_len),
                            static_cast<size_t>(row_len) * sizeof(float));
            }
        }
        bo_in.write(in_buf.data());
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_aux.write(aux_buf.data());
        bo_aux.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const auto run = (*prog.kernel)(3u, insts_bo, static_cast<uint32_t>(prog.insts.size()),
                                        bo_in, bo_aux, bo_out);
        run.wait();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(out_buf.data());
        for (int64_t r = 0; r < batch; ++r) {
            std::memcpy(out + ((r0 + r) * row_len), out_buf.data() + (r * stride),
                        static_cast<size_t>(row_len) * sizeof(float));
        }
    }
}

void XrtOpsRunner::eltwise_f32(const float* a, const float* b, float* out, int64_t total_elems,
                               int64_t b_period, int32_t op) {
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto [prog, insts_bo] = load_op(mgr, dir_, "eltwise");

    constexpr int64_t kChunk = 4096;
    std::vector<float> a_buf(kChunk, 0.0f);
    std::vector<float> aux_buf(4112, 0.0f);
    std::vector<float> out_buf(kChunk, 0.0f);

    auto full_period = static_cast<int32_t>(kChunk);
    std::memcpy(&aux_buf[4096], &op, sizeof(op));
    std::memcpy(&aux_buf[4097], &full_period, sizeof(full_period));

    xrt::bo bo_a(*prog.device, a_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                 prog.kernel->group_id(3));
    xrt::bo bo_aux(*prog.device, aux_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                   prog.kernel->group_id(4));
    xrt::bo bo_out(*prog.device, out_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                   prog.kernel->group_id(5));

    for (int64_t off = 0; off < total_elems; off += kChunk) {
        const int64_t n = std::min<int64_t>(kChunk, total_elems - off);
        std::ranges::fill(a_buf, 0.0f);
        std::memcpy(a_buf.data(), a + off, static_cast<size_t>(n) * sizeof(float));
        if (b_period >= total_elems) {
            std::ranges::fill(aux_buf.begin(), aux_buf.begin() + kChunk, 0.0f);
            std::memcpy(aux_buf.data(), b + off, static_cast<size_t>(n) * sizeof(float));
        } else {
            for (int64_t i = 0; i < n; ++i) {
                aux_buf[i] = b[(off + i) % b_period];
            }
        }
        bo_a.write(a_buf.data());
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_aux.write(aux_buf.data());
        bo_aux.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const auto run = (*prog.kernel)(3u, insts_bo, static_cast<uint32_t>(prog.insts.size()),
                                        bo_a, bo_aux, bo_out);
        run.wait();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(out_buf.data());
        std::memcpy(out + off, out_buf.data(), static_cast<size_t>(n) * sizeof(float));
    }
}

void XrtOpsRunner::unary_f32(const float* in, float* out, int64_t total_elems, int32_t op) {
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto [prog, insts_bo] = load_op(mgr, dir_, "unary");

    constexpr int64_t kChunk = 4096;
    std::vector<float> in_buf(kChunk, 0.0f);
    std::vector<float> out_buf(kChunk, 0.0f);
    std::vector<float> aux_buf(16, 0.0f);
    std::memcpy(aux_buf.data(), &op, sizeof(op));

    xrt::bo bo_in(*prog.device, in_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                  prog.kernel->group_id(3));
    xrt::bo bo_out(*prog.device, out_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                   prog.kernel->group_id(4));
    xrt::bo bo_aux(*prog.device, aux_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                   prog.kernel->group_id(5));
    bo_aux.write(aux_buf.data());
    bo_aux.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    for (int64_t off = 0; off < total_elems; off += kChunk) {
        const int64_t n = std::min<int64_t>(kChunk, total_elems - off);
        std::ranges::fill(in_buf, 0.0f);
        std::memcpy(in_buf.data(), in + off, static_cast<size_t>(n) * sizeof(float));

        bo_in.write(in_buf.data());
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const auto run = (*prog.kernel)(3u, insts_bo, static_cast<uint32_t>(prog.insts.size()),
                                        bo_in, bo_out, bo_aux);
        run.wait();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(out_buf.data());
        std::memcpy(out + off, out_buf.data(), static_cast<size_t>(n) * sizeof(float));
    }
}

void XrtOpsRunner::gelu_f32(const float* in, float* out, int64_t total_elems) {
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto [prog, insts_bo] = load_op(mgr, dir_, "gelu");

    constexpr int64_t kChunk = 4096;
    std::vector<uint16_t> in_buf(kChunk, 0);
    std::vector<uint16_t> out_buf(kChunk, 0);

    xrt::bo bo_in(*prog.device, in_buf.size() * sizeof(uint16_t), xrt::bo::flags::host_only,
                  prog.kernel->group_id(3));
    xrt::bo bo_out(*prog.device, out_buf.size() * sizeof(uint16_t), xrt::bo::flags::host_only,
                   prog.kernel->group_id(4));

    for (int64_t off = 0; off < total_elems; off += kChunk) {
        const int64_t n = std::min<int64_t>(kChunk, total_elems - off);
        std::ranges::fill(in_buf, 0);
        for (int64_t i = 0; i < n; ++i) {
            in_buf[i] = f32_to_bf16(in[off + i]);
        }
        bo_in.write(in_buf.data());
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const auto run =
            (*prog.kernel)(3u, insts_bo, static_cast<uint32_t>(prog.insts.size()), bo_in, bo_out);
        run.wait();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(out_buf.data());
        for (int64_t i = 0; i < n; ++i) {
            out[off + i] = bf16_to_f32(out_buf[i]);
        }
    }
}

void XrtOpsRunner::gemm_f32_nt(const float* a, int64_t lda, const float* b, int64_t ldb, float* c,
                               int64_t ldc, int64_t M, int64_t N, int64_t K) {
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto [prog, insts_bo] = load_op(mgr, dir_, "gemm_f32");

    constexpr int64_t T = 32;
    std::vector<float> a_tile(T * T, 0.0f);
    std::vector<float> b_tile(T * T, 0.0f);
    std::vector<float> c_tile(T * T, 0.0f);
    std::vector<float> accum(T * T, 0.0f);

    xrt::bo bo_a(*prog.device, a_tile.size() * sizeof(float), xrt::bo::flags::host_only,
                 prog.kernel->group_id(3));
    xrt::bo bo_b(*prog.device, b_tile.size() * sizeof(float), xrt::bo::flags::host_only,
                 prog.kernel->group_id(4));
    xrt::bo bo_c(*prog.device, c_tile.size() * sizeof(float), xrt::bo::flags::host_only,
                 prog.kernel->group_id(5));

    for (int64_t m0 = 0; m0 < M; m0 += T) {
        const int64_t mb = std::min<int64_t>(T, M - m0);
        for (int64_t n0 = 0; n0 < N; n0 += T) {
            const int64_t nb = std::min<int64_t>(T, N - n0);
            std::ranges::fill(accum, 0.0f);
            for (int64_t k0 = 0; k0 < K; k0 += T) {
                const int64_t kb = std::min<int64_t>(T, K - k0);
                std::ranges::fill(a_tile, 0.0f);
                std::ranges::fill(b_tile, 0.0f);
                for (int64_t m = 0; m < mb; ++m) {
                    std::memcpy(a_tile.data() + (m * T), a + ((m0 + m) * lda) + k0,
                                static_cast<size_t>(kb) * sizeof(float));
                }
                for (int64_t n = 0; n < nb; ++n) {
                    std::memcpy(b_tile.data() + (n * T), b + ((n0 + n) * ldb) + k0,
                                static_cast<size_t>(kb) * sizeof(float));
                }
                bo_a.write(a_tile.data());
                bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                bo_b.write(b_tile.data());
                bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                const auto run = (*prog.kernel)(
                    3u, insts_bo, static_cast<uint32_t>(prog.insts.size()), bo_a, bo_b, bo_c);
                run.wait();

                bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                bo_c.read(c_tile.data());
                for (int64_t m = 0; m < mb; ++m) {
                    for (int64_t n = 0; n < nb; ++n) {
                        accum[(m * T) + n] += c_tile[(m * T) + n];
                    }
                }
            }
            for (int64_t m = 0; m < mb; ++m) {
                std::memcpy(c + ((m0 + m) * ldc) + n0, accum.data() + (m * T),
                            static_cast<size_t>(nb) * sizeof(float));
            }
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
    auto [prog, insts_bo] = load_op(mgr, dir_, "l2_norm");

    constexpr int64_t kCap = 4096;
    const int64_t max_vecs = std::max<int64_t>(1, kCap / len);
    std::vector<float> in_buf(kCap, 0.0f);
    std::vector<float> out_buf(kCap, 0.0f);
    std::vector<float> aux_buf(16, 0.0f);
    aux_buf[0] = eps;
    auto len_i32 = static_cast<int32_t>(len);
    std::memcpy(&aux_buf[1], &len_i32, sizeof(len_i32));

    xrt::bo bo_in(*prog.device, in_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                  prog.kernel->group_id(3));
    xrt::bo bo_out(*prog.device, out_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                   prog.kernel->group_id(4));
    xrt::bo bo_aux(*prog.device, aux_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                   prog.kernel->group_id(5));

    for (int64_t v = 0; v < n_vecs; v += max_vecs) {
        const int64_t batch = std::min<int64_t>(max_vecs, n_vecs - v);
        auto bv_i32 = static_cast<int32_t>(batch);
        std::memcpy(&aux_buf[2], &bv_i32, sizeof(bv_i32));

        std::ranges::fill(in_buf, 0.0f);
        std::memcpy(in_buf.data(), in + (v * len),
                    static_cast<size_t>(batch * len) * sizeof(float));

        bo_in.write(in_buf.data());
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_aux.write(aux_buf.data());
        bo_aux.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const auto run = (*prog.kernel)(3u, insts_bo, static_cast<uint32_t>(prog.insts.size()),
                                        bo_in, bo_out, bo_aux);
        run.wait();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(out_buf.data());
        std::memcpy(out + (v * len), out_buf.data(),
                    static_cast<size_t>(batch * len) * sizeof(float));
    }
}

void XrtOpsRunner::reduce_sum_f32(const float* in, float* out, int64_t n_rows, int64_t row_len,
                                  float scale) {
    if (row_len <= 0 || row_len > 4096) {
        throw std::runtime_error("reduce_sum_f32 requires 1 <= row_len <= 4096");
    }
    auto& mgr = XrtContextManager::instance();
    const std::scoped_lock lock(mgr.mutex());
    auto [prog, insts_bo] = load_op(mgr, dir_, "reduce_sum");

    constexpr int64_t kCap = 4096;
    std::vector<float> in_buf(kCap, 0.0f);
    std::vector<float> out_buf(16, 0.0f);
    std::vector<float> aux_buf(16, 0.0f);
    auto valid_len = static_cast<int32_t>(row_len);
    std::memcpy(aux_buf.data(), &valid_len, sizeof(valid_len));
    aux_buf[1] = scale;

    xrt::bo bo_in(*prog.device, in_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                  prog.kernel->group_id(3));
    xrt::bo bo_out(*prog.device, out_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                   prog.kernel->group_id(4));
    xrt::bo bo_aux(*prog.device, aux_buf.size() * sizeof(float), xrt::bo::flags::host_only,
                   prog.kernel->group_id(5));
    bo_aux.write(aux_buf.data());
    bo_aux.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    for (int64_t r = 0; r < n_rows; ++r) {
        std::ranges::fill(in_buf, 0.0f);
        std::memcpy(in_buf.data(), in + (r * row_len),
                    static_cast<size_t>(row_len) * sizeof(float));

        bo_in.write(in_buf.data());
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const auto run = (*prog.kernel)(3u, insts_bo, static_cast<uint32_t>(prog.insts.size()),
                                        bo_in, bo_out, bo_aux);
        run.wait();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(out_buf.data());
        out[r] = out_buf[0];
    }
}
