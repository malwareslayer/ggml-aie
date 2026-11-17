#pragma once

#include <cstdint>
#include <string>

class XrtOpsRunner {
  public:
    static constexpr int64_t kGemmTM = 64;
    static constexpr int64_t kGemmTK = 64;
    static constexpr int64_t kGemmTNW = 64;
    static constexpr int64_t kGemmNW = 4;
    static constexpr int64_t kGemmTN = kGemmTNW * kGemmNW; // 256

    static constexpr int64_t kBf16TM = 32;
    static constexpr int64_t kBf16TK = 64;
    static constexpr int64_t kBf16KChunks = 16;
    static constexpr int64_t kBf16KSlice = kBf16TK * kBf16KChunks; // 1024
    static constexpr int64_t kBf16TNW = 64;
    static constexpr int64_t kBf16NW = 16;
    static constexpr int64_t kBf16TN = kBf16TNW * kBf16NW; // 1024

    explicit XrtOpsRunner(std::string artifacts_dir);

    [[nodiscard]] const std::string& artifacts_dir() const {
        return dir_;
    }

    static void layernorm_f32(const float* in, float* out, int64_t n_rows, int64_t row_len,
                              float eps, bool is_rms = false);

    static void softmax_f32(const float* in, const float* mask, float* out, int64_t n_rows,
                            int64_t row_len, int64_t mask_rows, float scale);

    // op: 0 = add, 1 = mul, 2 = sub
    static void eltwise_f32(const float* a, const float* b, float* out, int64_t total_elems,
                            int64_t b_period, int32_t op);

    // op: 0 = silu, 1 = tanh, 2 = sigmoid, 3 = relu, 4 = sqr, 5 = neg, 6 = abs, 7 = gelu
    static void unary_f32(const float* in, float* out, int64_t total_elems, int32_t op);

    static void gelu_f32(const float* in, float* out, int64_t total_elems);

    // Computes C[M, N] = A[M, K] * B^T[K, N] in float32.
    // A has row stride lda, B has row stride ldb, C has row stride ldc.
    static void gemm_f32_nt(const float* a, int64_t lda, const float* b, int64_t ldb, float* c,
                            int64_t ldc, int64_t M, int64_t N, int64_t K);

    // Computes C[M, N] = A[M, K] * B^T[K, N] using pre-packed B tiles.
    // b_packed is laid out as [n_blocks][k_blocks][kGemmNW][kGemmTNW][kGemmTK] floats.
    static void gemm_f32_nt_prepacked(const float* a, int64_t lda, const float* b_packed,
                                      int64_t n_blocks, int64_t k_blocks, float* c, int64_t ldc,
                                      int64_t M, int64_t N, int64_t K);

    // Computes C[M, N] = A[M, K] * B^T[K, N] on 4-column BF16 aie::mmul<4,8,8> NPU kernel.
    // b_packed is [n_slices][k_slices][4_col][16_kchunk][4_row][8_n8][8_k8][8_kin][8_nin] uint16.
    // If b_bos is non-null and non-empty, uses pre-synced xrt::bo per (ns, ks) slice directly.
    static void gemm_bf16_nt_prepacked(const float* a, int64_t lda, const uint16_t* b_packed,
                                       const void* b_bos_ptr, int64_t n_slices, int64_t k_slices,
                                       float* c, int64_t ldc, int64_t M, int64_t N, int64_t K);

    static void l2_norm_f32(const float* in, float* out, int64_t n_vecs, int64_t len, float eps);

    static void reduce_sum_f32(const float* in, float* out, int64_t n_rows, int64_t row_len,
                               float scale);

  private:
    std::string dir_;
};
