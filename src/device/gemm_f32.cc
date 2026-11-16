#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
// Computes C[32x32] = A[32x32] * B^T[32x32] in float32.
// Both A and B are row-major [32, 32] so B^T[k, n] == B[n, k],
// matching GGML MUL_MAT where src0 is [K, N] and src1 is [K, M].
void gemm_f32_32x32x32(const float* __restrict a, const float* __restrict b, float* __restrict c) {
    constexpr int M = 32;
    constexpr int K = 32;
    constexpr int N = 32;
    constexpr int V = 16;

    for (int m = 0; m < M; ++m) {
        const float* __restrict a_row = a + m * K;
        ::aie::vector<float, V> a0 = ::aie::load_v<V>(a_row);
        ::aie::vector<float, V> a1 = ::aie::load_v<V>(a_row + V);
        float* __restrict c_row = c + m * N;
        for (int n = 0; n < N; ++n) {
            const float* __restrict b_row = b + n * K;
            ::aie::accum<accfloat, V> acc = ::aie::mul(a0, ::aie::load_v<V>(b_row));
            acc = ::aie::mac(acc, a1, ::aie::load_v<V>(b_row + V));
            c_row[n] = ::aie::reduce_add(acc.to_vector<float>());
        }
    }
}
}
