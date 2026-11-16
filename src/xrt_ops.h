#pragma once

#include <cstdint>
#include <string>

class XrtOpsRunner {
  public:
    explicit XrtOpsRunner(std::string artifacts_dir);

    [[nodiscard]] const std::string& artifacts_dir() const {
        return dir_;
    }

    void layernorm_f32(const float* in, float* out, int64_t n_rows, int64_t row_len, float eps,
                       bool is_rms = false);

    void softmax_f32(const float* in, const float* mask, float* out, int64_t n_rows,
                     int64_t row_len, int64_t mask_rows, float scale);

    // op: 0 = add, 1 = mul, 2 = sub
    void eltwise_f32(const float* a, const float* b, float* out, int64_t total_elems,
                     int64_t b_period, int32_t op);

    // op: 0 = silu, 1 = tanh, 2 = sigmoid, 3 = relu, 4 = sqr, 5 = neg, 6 = abs, 7 = gelu
    void unary_f32(const float* in, float* out, int64_t total_elems, int32_t op);

    void gelu_f32(const float* in, float* out, int64_t total_elems);

    // Computes C[M, N] = A[M, K] * B^T[K, N] in float32.
    // A has row stride lda, B has row stride ldb, C has row stride ldc.
    void gemm_f32_nt(const float* a, int64_t lda, const float* b, int64_t ldb, float* c,
                     int64_t ldc, int64_t M, int64_t N, int64_t K);

    void l2_norm_f32(const float* in, float* out, int64_t n_vecs, int64_t len, float eps);

    void reduce_sum_f32(const float* in, float* out, int64_t n_rows, int64_t row_len, float scale);

  private:
    std::string dir_;
};
