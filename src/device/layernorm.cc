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
// Processes up to n_rows rows packed in 4096 float32 elements (row_len <= 4096).
// aux[0] = float eps
// aux[1] = bitcast(int32_t row_len)
// aux[2] = bitcast(int32_t n_rows)
// aux[3] = bitcast(int32_t is_rms) : 0 = LayerNorm, 1 = RMSNorm
void layernorm_f32_4096(const float* __restrict in, float* __restrict out,
                        const float* __restrict aux) {
    const float eps = aux[0];
    int32_t row_len = 384;
    int32_t n_rows = 1;
    int32_t is_rms = 0;
    __builtin_memcpy(&row_len, &aux[1], sizeof(row_len));
    __builtin_memcpy(&n_rows, &aux[2], sizeof(n_rows));
    __builtin_memcpy(&is_rms, &aux[3], sizeof(is_rms));

    constexpr int N = 16;
    const int chunks = row_len / N;
    const int rem_start = chunks * N;

    for (int r = 0; r < n_rows; ++r) {
        const float* __restrict row_in = in + r * row_len;
        float* __restrict row_out = out + r * row_len;

        float mean = 0.0f;
        if (!is_rms) {
            ::aie::vector<float, N> sum_v = ::aie::zeros<float, N>();
            for (int i = 0; i < chunks; ++i) {
                sum_v = ::aie::add(sum_v, ::aie::load_v<N>(row_in + i * N));
            }
            float sum = ::aie::reduce_add(sum_v);
            for (int i = rem_start; i < row_len; ++i) {
                sum += row_in[i];
            }
            mean = sum / float(row_len);
        }
        ::aie::vector<float, N> mean_v = ::aie::broadcast<float, N>(mean);

        ::aie::accum<accfloat, N> var_acc = ::aie::zeros<accfloat, N>();
        for (int i = 0; i < chunks; ++i) {
            ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(row_in + i * N), mean_v);
            var_acc = ::aie::mac(var_acc, d, d);
        }
        float var_sum = ::aie::reduce_add(var_acc.to_vector<float>());
        for (int i = rem_start; i < row_len; ++i) {
            float d = row_in[i] - mean;
            var_sum += d * d;
        }
        float var = var_sum / float(row_len);
        float inv_std = fast_invsqrt_f32(var + eps);
        ::aie::vector<float, N> inv_v = ::aie::broadcast<float, N>(inv_std);

        for (int i = 0; i < chunks; ++i) {
            ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(row_in + i * N), mean_v);
            ::aie::store_v(row_out + i * N, ::aie::mul(d, inv_v).to_vector<float>());
        }
        for (int i = rem_start; i < row_len; ++i) {
            row_out[i] = (row_in[i] - mean) * inv_std;
        }
    }
}
}
