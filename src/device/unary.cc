#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
inline float sigmoid_approx_f32(float x) {
    float z = -x;
    if (z < -80.0f)
        return 1.0f;
    if (z > 80.0f)
        return 0.0f;
    const float ln2_inv = 1.4426950408889634f;
    const float ln2 = 0.6931471805599453f;
    float k_f = z * ln2_inv;
    int32_t k = static_cast<int32_t>(k_f >= 0.0f ? (k_f + 0.5f) : (k_f - 0.5f));
    float r = z - static_cast<float>(k) * ln2;
    float p =
        1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * 0.008333334f))));
    int32_t pow2_bits = (k + 127) << 23;
    float pow2;
    __builtin_memcpy(&pow2, &pow2_bits, sizeof(pow2));
    float e = p * pow2;
    return 1.0f / (1.0f + e);
}
} // namespace

extern "C" {
// Generalized elementwise unary op for 4096 float32 elements per dispatch.
// aux[0] = bitcast int32 op:
//   0 = silu, 1 = tanh, 2 = sigmoid, 3 = relu, 4 = sqr, 5 = neg, 6 = abs, 7 = gelu
void unary_f32_4096(const float* __restrict in, float* __restrict out,
                    const float* __restrict aux) {
    int32_t op = 0;
    __builtin_memcpy(&op, &aux[0], sizeof(op));

#pragma clang loop unroll(disable)
    for (int32_t i = 0; i < 4096; ++i) {
        float x = in[i];
        float y = x;
        if (op == 0 || op == 1 || op == 2 || op == 7) {
            float arg = x;
            if (op == 1) {
                arg = 2.0f * x;
            } else if (op == 7) {
                arg = 1.5957691216f * x * (1.0f + 0.044715f * x * x);
            }
            float s = sigmoid_approx_f32(arg);
            if (op == 0 || op == 7) {
                y = x * s;
            } else if (op == 1) {
                y = 2.0f * s - 1.0f;
            } else {
                y = s;
            }
        } else if (op == 3) {
            y = x > 0.0f ? x : 0.0f;
        } else if (op == 4) {
            y = x * x;
        } else if (op == 5) {
            y = -x;
        } else if (op == 6) {
            y = x >= 0.0f ? x : -x;
        }
        out[i] = y;
    }
}
} // extern "C"
