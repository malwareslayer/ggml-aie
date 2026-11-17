#include <aie_api/aie.hpp>
#include <cstdint>

alignas(64) static float g_acc[32 * 64];

extern "C" {

void gemm_bf16_zero() {
    aie::vector<float, 16> z = aie::zeros<float, 16>();
    for (int i = 0; i < (32 * 64) / 16; ++i) {
        aie::store_v(g_acc + (i * 16), z);
    }
}

// A is [8 (m4)][8 (k8)][32 (4x8)]
// B is [8 (n8)][8 (k8)][64 (8x8)]
// g_acc is [8 (m4)][8 (n8)][32 (4x8)]
void gemm_bf16_32x64x64_acc(const bfloat16* __restrict a, const bfloat16* __restrict b) {
    using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16>;
    for (int m = 0; m < 8; ++m) {
        const bfloat16* __restrict a_m = a + (m * (8 * 32));
        for (int n = 0; n < 8; ++n) {
            const bfloat16* __restrict b_n = b + (n * (8 * 64));
            float* __restrict c_mn = g_acc + (((m * 8) + n) * 32);
            MMUL acc;
            acc.mul(aie::load_v<32>(a_m), aie::load_v<64>(b_n));
            for (int k = 1; k < 8; ++k) {
                acc.mac(aie::load_v<32>(a_m + (k * 32)), aie::load_v<64>(b_n + (k * 64)));
            }
            aie::store_v(c_mn, aie::add(aie::load_v<32>(c_mn), acc.template to_vector<float>()));
        }
    }
}

void gemm_bf16_out(bfloat16* __restrict out) {
    for (int i = 0; i < (32 * 64) / 16; ++i) {
        aie::accum<accfloat, 16> acc;
        acc.from_vector(aie::load_v<16>(g_acc + (i * 16)));
        aie::store_v(out + (i * 16), acc.to_vector<bfloat16>());
    }
}

} // extern "C"
