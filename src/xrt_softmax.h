#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace aie2 {
class Softmax {
  public:
    explicit Softmax(const std::string& artifact_dir);
    ~Softmax();
    Softmax(const Softmax&) = delete;
    Softmax& operator=(const Softmax&) = delete;
    [[nodiscard]] static bool supports(int64_t length);
    void compute(int64_t length, const float* input, float* output);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace aie2
