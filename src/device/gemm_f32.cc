#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
// Computes C[64x64] = A[64x64] * B^T[64x64] in float32.
// Both A and B are row-major [64, 64] so B^T[k, n] == B[n, k],
// matching GGML MUL_MAT where src0 is [K, N] and src1 is [K, M].
void gemm_f32_64x64x64(const float* __restrict a, const float* __restrict b, float* __restrict c) {
    constexpr int M = 64;
    constexpr int K = 64;
    constexpr int N = 64;
    constexpr int V = 16;

    for (int m = 0; m < M; ++m) {
        const float* __restrict a_row = a + (m * K);
        ::aie::vector<float, V> a0 = ::aie::load_v<V>(a_row);
        ::aie::vector<float, V> a1 = ::aie::load_v<V>(a_row + 16);
        ::aie::vector<float, V> a2 = ::aie::load_v<V>(a_row + 32);
        ::aie::vector<float, V> a3 = ::aie::load_v<V>(a_row + 48);
        float* __restrict c_row = c + (m * N);
        for (int n = 0; n < N; ++n) {
            const float* __restrict b_row = b + (n * K);
            ::aie::accum<accfloat, V> acc = ::aie::mul(a0, ::aie::load_v<V>(b_row));
            acc = ::aie::mac(acc, a1, ::aie::load_v<V>(b_row + 16));
            acc = ::aie::mac(acc, a2, ::aie::load_v<V>(b_row + 32));
            acc = ::aie::mac(acc, a3, ::aie::load_v<V>(b_row + 48));
            c_row[n] = ::aie::reduce_add(acc.to_vector<float>());
        }
    }
}
}
