#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace aie2 {
// Fixed-M Phoenix BF16 GEMM. Inputs are GGML-contiguous weights [N,K] F16
// and activation [T,K] F32; output is [T,N] F32. Not thread-safe.
class Matmul {
  public:
    explicit Matmul(const std::string& artifact_dir);
    ~Matmul();
    Matmul(const Matmul&) = delete;
    Matmul& operator=(const Matmul&) = delete;
    [[nodiscard]] static bool supports(int64_t k, int64_t n, int64_t tokens);
    void compute(int64_t k, int64_t n, int64_t tokens, const uint16_t* weights_nk,
                 const float* activation_tk, float* output_tn);

  private:
    struct Fixture;
    std::string artifact_dir_;
    size_t rows_ = 0;
    // At most one context per supported shape; bounds device and host allocations.
    std::array<std::unique_ptr<Fixture>, 3> fixtures_;
};
} // namespace aie2
