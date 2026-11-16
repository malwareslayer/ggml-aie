#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
// Fast exp(x) for x <= 0 using 2^(x * log2(e)) with degree-4 polynomial.
inline float fast_exp_nonpos(float x) {
    if (x < -80.0f)
        return 0.0f;
    constexpr float log2e = 1.4426950408889634f;
    float t = x * log2e;
    int32_t k = static_cast<int32_t>(t);
    if (t < static_cast<float>(k))
        k -= 1; // floor(t)
    float f = t - static_cast<float>(k);
    // 2^f on [0, 1) via minimax/Taylor degree-4
    float p = 1.0f + f * (0.69314718f + f * (0.24022651f + f * (0.05550411f + f * 0.00961813f)));
    int32_t exp_bits = (k + 127) << 23;
    float pow2_k;
    __builtin_memcpy(&pow2_k, &exp_bits, sizeof(pow2_k));
    return pow2_k * p;
}
} // namespace

extern "C" {
// Processes n_rows rows of valid_len elements (n_rows * stride <= 4096) in float32.
// aux[0..4095] = mask (float[4096])
// aux[4096]    = bitcast(int32_t valid_len)
// aux[4097]    = float scale
// aux[4098]    = bitcast(int32_t n_rows)
// aux[4099]    = bitcast(int32_t stride)
void softmax_f32_4096(const float* __restrict in, const float* __restrict aux,
                      float* __restrict out) {
    int32_t valid_len = 0;
    int32_t n_rows = 8;
    int32_t stride = 512;
    __builtin_memcpy(&valid_len, &aux[4096], sizeof(valid_len));
    const float scale = aux[4097];
    __builtin_memcpy(&n_rows, &aux[4098], sizeof(n_rows));
    __builtin_memcpy(&stride, &aux[4099], sizeof(stride));

    for (int r = 0; r < n_rows; ++r) {
        const float* __restrict row_in = in + r * stride;
        const float* __restrict row_mask = aux + r * stride;
        float* __restrict row_out = out + r * stride;

        float max_val = -1e30f;
        for (int i = 0; i < valid_len; ++i) {
            float v = row_in[i] * scale + row_mask[i];
            row_out[i] = v;
            if (v > max_val)
                max_val = v;
        }
        float sum = 0.0f;
        for (int i = 0; i < valid_len; ++i) {
            float e = fast_exp_nonpos(row_out[i] - max_val);
            row_out[i] = e;
            sum += e;
        }
        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (int i = 0; i < valid_len; ++i) {
            row_out[i] *= inv_sum;
        }
        for (int i = valid_len; i < stride; ++i) {
            row_out[i] = 0.0f;
        }
    }
}
}
