#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
// Processes 4096 float32 elements per dispatch.
// aux[0..4095] = b (float[4096])
// aux[4096]    = bitcast(int32_t op): 0 = add, 1 = mul, 2 = sub
// aux[4097]    = bitcast(int32_t b_period): repeat period for b
void eltwise_f32_4096(const float* __restrict a, const float* __restrict aux,
                      float* __restrict out) {
    int32_t op = 0;
    int32_t b_period = 4096;
    __builtin_memcpy(&op, &aux[4096], sizeof(op));
    __builtin_memcpy(&b_period, &aux[4097], sizeof(b_period));
    const float* __restrict b = aux;
    constexpr int total = 4096;
    constexpr int N = 16;

    if (b_period >= total) {
        if (op == 0) {
            for (int i = 0; i < total; i += N) {
                ::aie::store_v(out + i,
                               ::aie::add(::aie::load_v<N>(a + i), ::aie::load_v<N>(b + i)));
            }
        } else if (op == 1) {
            for (int i = 0; i < total; i += N) {
                ::aie::store_v(out + i, ::aie::mul(::aie::load_v<N>(a + i), ::aie::load_v<N>(b + i))
                                            .to_vector<float>());
            }
        } else {
            for (int i = 0; i < total; i += N) {
                ::aie::store_v(out + i,
                               ::aie::sub(::aie::load_v<N>(a + i), ::aie::load_v<N>(b + i)));
            }
        }
    } else {
        for (int base = 0; base < total; base += b_period) {
            int count = (base + b_period <= total) ? b_period : (total - base);
            for (int i = 0; i < count; ++i) {
                if (op == 0) {
                    out[base + i] = a[base + i] + b[i];
                } else if (op == 1) {
                    out[base + i] = a[base + i] * b[i];
                } else {
                    out[base + i] = a[base + i] - b[i];
                }
            }
        }
    }
}
}
