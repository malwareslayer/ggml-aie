#include "xrt_softmax.h"
#include "xrt_context.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

namespace aie2 {
namespace {
uint16_t bf16(float value) {
    const auto bits = std::bit_cast<uint32_t>(value);
    return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
float from_bf16(uint16_t value) {
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}
std::vector<char> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot open softmax instructions: " + path.string());
    return {std::istreambuf_iterator<char>(file), {}};
}
} // namespace

struct Softmax::Impl {
    static constexpr int64_t tile_size = 1024;
    std::vector<char> instructions_data;
    xrt::device device;
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo instructions, input, output;

    explicit Impl(const std::filesystem::path& dir)
        : instructions_data(read_file(dir / "insts.bin")), device(0), context([&]() {
              const auto qos = build_aie2_qos_config();
              const auto uuid =
                  device.register_xclbin(xrt::xclbin((dir / "softmax.xclbin").string()));
              return qos.empty() ? xrt::hw_context(device, uuid)
                                 : xrt::hw_context(device, uuid, qos);
          }()),
          kernel(context, "MLIR_AIE"), instructions(device, instructions_data.size(),
                                                    xrt::bo::flags::cacheable, kernel.group_id(1)),
          input(device, tile_size * sizeof(uint16_t), xrt::bo::flags::host_only,
                kernel.group_id(3)),
          output(device, tile_size * sizeof(uint16_t), xrt::bo::flags::host_only,
                 kernel.group_id(4)) {
        if (instructions_data.empty() || instructions_data.size() % sizeof(uint32_t) != 0)
            throw std::runtime_error("invalid softmax instruction stream");
        std::memcpy(instructions.map<char*>(), instructions_data.data(), instructions_data.size());
        instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
};

Softmax::Softmax(const std::string& artifact_dir) : impl_(std::make_unique<Impl>(artifact_dir)) {}
Softmax::~Softmax() = default;
bool Softmax::supports(int64_t length) {
    return length > 0 && length <= Impl::tile_size;
}
void Softmax::compute(int64_t length, const float* input, float* output) {
    if (!supports(length) || !input || !output)
        throw std::invalid_argument("unsupported or null softmax input");
    auto* in = impl_->input.map<uint16_t*>();
    auto* out = impl_->output.map<uint16_t*>();
    for (int64_t i = 0; i < Impl::tile_size; ++i)
        in[i] = bf16(i < length ? input[i] : -20.0f);
    std::fill(out, out + Impl::tile_size, uint16_t{0});
    impl_->input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    impl_->output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const auto run = impl_->kernel(3, impl_->instructions,
                                   static_cast<uint32_t>(impl_->instructions_data.size()),
                                   impl_->input, impl_->output, 0, 0);
    const auto state = run.wait(30000U);
    if (state != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("softmax NPU dispatch failed");
    impl_->output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (int64_t i = 0; i < length; ++i)
        output[i] = from_bf16(out[i]);
}
} // namespace aie2
