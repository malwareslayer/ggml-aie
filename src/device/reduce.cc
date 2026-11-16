#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
inline float fast_invsqrt_f32(float x) {
    float xhalf = 0.5f * x;
    int32_t i;
    __builtin_memcpy(&i, &x, sizeof(i));
    i = 0x5f3759df - (i >> 1);
    float y;
    __builtin_memcpy(&y, &i, sizeof(y));
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    return y;
}
} // namespace

extern "C" {
// L2-normalizes n_vecs vectors of length `len` (n_vecs * len <= 4096).
// aux[0] = eps (e.g. 1e-12f)
// aux[1] = bitcast(int32_t len)
// aux[2] = bitcast(int32_t n_vecs)
void l2_norm_f32_4096(const float* __restrict in, float* __restrict out,
                      const float* __restrict aux) {
    const float eps = aux[0];
    int32_t len = 384;
    int32_t n_vecs = 1;
    __builtin_memcpy(&len, &aux[1], sizeof(len));
    __builtin_memcpy(&n_vecs, &aux[2], sizeof(n_vecs));
    constexpr int N = 16;
    const int chunks = len / N;
    const int rem_start = chunks * N;

    for (int v = 0; v < n_vecs; ++v) {
        const float* __restrict vin = in + v * len;
        float* __restrict vout = out + v * len;

        ::aie::accum<accfloat, N> acc = ::aie::zeros<accfloat, N>();
        for (int i = 0; i < chunks; ++i) {
            ::aie::vector<float, N> x = ::aie::load_v<N>(vin + i * N);
            acc = ::aie::mac(acc, x, x);
        }
        float norm_sq = ::aie::reduce_add(acc.to_vector<float>());
        for (int i = rem_start; i < len; ++i) {
            norm_sq += vin[i] * vin[i];
        }
        float inv_norm = fast_invsqrt_f32((norm_sq > eps) ? norm_sq : eps);
        ::aie::vector<float, N> inv_v = ::aie::broadcast<float, N>(inv_norm);
        for (int i = 0; i < chunks; ++i) {
            ::aie::vector<float, N> x = ::aie::load_v<N>(vin + i * N);
            ::aie::store_v(vout + i * N, ::aie::mul(x, inv_v).to_vector<float>());
        }
        for (int i = rem_start; i < len; ++i) {
            vout[i] = vin[i] * inv_norm;
        }
    }
}

// Sums valid_len elements (<= 4096) of `in`, scales by `scale`, writes to `out[0]`.
// aux[0] = bitcast(int32_t valid_len), aux[1] = float scale.
void reduce_sum_f32_4096(const float* __restrict in, float* __restrict out,
                         const float* __restrict aux) {
    int32_t valid_len = 0;
    __builtin_memcpy(&valid_len, &aux[0], sizeof(valid_len));
    const float scale = aux[1];
    constexpr int N = 16;
    const int chunks = valid_len / N;
    const int rem_start = chunks * N;
    ::aie::vector<float, N> sum_v = ::aie::zeros<float, N>();
    for (int i = 0; i < chunks; ++i) {
        sum_v = ::aie::add(sum_v, ::aie::load_v<N>(in + i * N));
    }
    float sum = ::aie::reduce_add(sum_v);
    for (int i = rem_start; i < valid_len; ++i) {
        sum += in[i];
    }
    out[0] = sum * scale;
}
}
