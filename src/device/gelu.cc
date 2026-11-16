#include "aie_kernels/aie2/gelu.cc"
#include "lut_based_ops.cpp"

extern "C" {
void gelu_bf16_4096(bfloat16* __restrict input, bfloat16* __restrict output) {
    gelu_tanh_approx_bf16(input, output, 4096);
}
}
