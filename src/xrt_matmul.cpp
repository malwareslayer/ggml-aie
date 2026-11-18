#include "xrt_matmul.h"
#include "xrt_context.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

namespace aie2 {
namespace {
constexpr size_t max_tokens = 4096;
constexpr size_t weight_cache_bytes =
    size_t{16} * 1024U * 1024U; // per shape; three fixtures at most
using Clock = std::chrono::steady_clock;
int64_t micros(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}
struct Profile {
    int64_t context = 0, setup = 0, pack = 0, sync = 0, kernel = 0, output = 0;
};
size_t read_rows(const std::filesystem::path& path) {
    if (std::filesystem::file_size(path) > 16)
        throw std::runtime_error("invalid GEMM rows contract: " + path.string());
    std::ifstream stream(path);
    unsigned rows = 0;
    char extra = 0;
    if (!(stream >> rows) || (stream >> extra) ||
        (rows != 32 && rows != 64 && rows != 128 && rows != 4096))
        throw std::runtime_error("invalid GEMM rows contract: " + path.string());
    return rows;
}
struct Shape {
    int64_t k, n;
    const char* name;
};
constexpr Shape shapes[] = {{384, 384, "qkv"}, {384, 1536, "ffn1"}, {1536, 384, "ffn2"}};
const Shape* shape_for(int64_t k, int64_t n) {
    for (const auto& s : shapes)
        if (k == s.k && n == s.n)
            return &s;
    return nullptr;
}
std::vector<char> read_instructions(const std::filesystem::path& path) {
    const auto size = std::filesystem::file_size(path);
    if (size == 0 || size > size_t{1024} * 1024U || size % sizeof(uint32_t))
        throw std::runtime_error("invalid GEMM instruction size: " + path.string());
    std::vector<char> bytes(static_cast<size_t>(size));
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(bytes.data(), static_cast<std::streamsize>(size)))
        throw std::runtime_error("cannot read GEMM instructions: " + path.string());
    return bytes;
}
xrt::xclbin load_binary(const std::filesystem::path& path) {
    xrt::xclbin binary(path.string());
    const auto args = binary.get_kernel("MLIR_AIE").get_args();
    if (args.size() != 8 || args[0].get_name() != "opcode" || args[1].get_name() != "instr" ||
        args[2].get_name() != "ninstr")
        throw std::runtime_error("not the expected eight-argument GEMM kernel: " + path.string());
    return binary;
}
uint16_t bf16(float value) {
    const auto bits = std::bit_cast<uint32_t>(value);
    // Round to nearest, ties to even; reject non-finite inputs before conversion.
    return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
float f16(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
    const uint32_t exp = (static_cast<uint32_t>(bits) >> 10U) & 31U;
    const uint32_t mantissa = bits & 1023U;
    if (exp == 31)
        throw std::invalid_argument("non-finite F16 weight");
    if (exp != 0)
        return std::bit_cast<float>(sign | ((exp + 112U) << 23U) | (mantissa << 13U));
    if (!mantissa)
        return std::bit_cast<float>(sign);
    // F16 subnormals are also legal GGUF values.
    const float result = static_cast<float>(mantissa) * 0x1p-24f;
    return (sign ? -result : result);
}
float from_bf16(uint16_t bits) {
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16U);
}
} // namespace

struct Matmul::Fixture {
    Shape shape;
    size_t rows;
    std::vector<char> instruction_data;
    xrt::xclbin binary;
    xrt::device device;
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo instructions, a, b, c;
    uint32_t instruction_bytes;
    struct CachedWeight {
        const uint16_t* source; // lookup hint only; always compare the entire snapshot
        uint64_t id;
        std::vector<uint16_t> snapshot, packed;
    };
    // FIFO bounded by both the original F16 snapshot and the packed BF16 data.
    std::vector<CachedWeight> weight_cache;
    uint64_t next_weight_id = 1, resident_weight_id = 0;

    Fixture(const std::filesystem::path& dir, Shape s, size_t capacity)
        : shape(s), rows(capacity),
          instruction_data(read_instructions(dir / (std::string(s.name) + ".insts.txt"))),
          binary(load_binary(dir / (std::string(s.name) + ".xclbin"))), device(0), context([&]() {
              const auto qos = build_aie2_qos_config();
              const auto uuid = device.register_xclbin(binary);
              return qos.empty() ? xrt::hw_context(device, uuid)
                                 : xrt::hw_context(device, uuid, qos);
          }()),
          kernel(context, "MLIR_AIE"), instructions(device, instruction_data.size(),
                                                    xrt::bo::flags::cacheable, kernel.group_id(1)),
          a(device, rows * static_cast<size_t>(s.k) * sizeof(uint16_t), xrt::bo::flags::host_only,
            kernel.group_id(3)),
          b(device, static_cast<size_t>(s.k * s.n) * sizeof(uint16_t), xrt::bo::flags::host_only,
            kernel.group_id(4)),
          c(device, rows * static_cast<size_t>(s.n) * sizeof(uint16_t), xrt::bo::flags::host_only,
            kernel.group_id(5)),
          instruction_bytes(static_cast<uint32_t>(instruction_data.size())) {
        std::memcpy(instructions.map<char*>(), instruction_data.data(), instruction_data.size());
        instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    void run(bool sync_weights, Profile& profile) {
        const auto sync_start = Clock::now();
        a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (sync_weights)
            b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        profile.sync += micros(sync_start, Clock::now());
        const auto kernel_start = Clock::now();
        auto job = kernel(3, instructions, instruction_bytes, a, b, c, 0, 0);
        const auto state = job.wait(30000U);
        profile.kernel += micros(kernel_start, Clock::now());
        if (state == ERT_CMD_STATE_TIMEOUT) {
            job.abort();
            throw std::runtime_error("GEMM timed out after 30 seconds");
        }
        if (state != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("GEMM failed (ERT state " + std::to_string(state) + ")");
        const auto output_sync_start = Clock::now();
        c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        profile.sync += micros(output_sync_start, Clock::now());
    }
};

Matmul::Matmul(const std::string& artifact_dir) : artifact_dir_(artifact_dir) {
    if (artifact_dir.empty())
        throw std::invalid_argument("empty GEMM artifact directory");
    rows_ = read_rows(std::filesystem::path(artifact_dir) / "rows.txt");
    for (const auto& s : shapes) {
        const std::filesystem::path dir(artifact_dir);
        read_instructions(dir / (std::string(s.name) + ".insts.txt"));
        load_binary(dir / (std::string(s.name) + ".xclbin"));
    }
    const xrt::device device(0);
}
Matmul::~Matmul() = default;
bool Matmul::supports(int64_t k, int64_t n, int64_t tokens) {
    return shape_for(k, n) && tokens >= 1 && std::cmp_less_equal(tokens, max_tokens);
}
void Matmul::compute(int64_t k, int64_t n, int64_t tokens, const uint16_t* weights_nk,
                     const float* activation_tk, float* output_tn) {
    if (!supports(k, n, tokens))
        throw std::invalid_argument("unsupported Phoenix GEMM shape");
    if (!weights_nk || !activation_tk || !output_tn)
        throw std::invalid_argument("null GEMM input or output");
    // Validate all chunks before launching any job (avoid partial execution/output).
    for (size_t i = 0; std::cmp_less(i, tokens * k); ++i)
        if (!std::isfinite(activation_tk[i]))
            throw std::invalid_argument("non-finite F32 activation");
    const auto* shape = shape_for(k, n);
    Profile profile;
    const auto context_start = Clock::now();
    auto& fixture = fixtures_[static_cast<size_t>(shape - shapes)];
    if (!fixture)
        fixture = std::make_unique<Fixture>(artifact_dir_, *shape, rows_);
    profile.context = micros(context_start, Clock::now());
    const auto pack_start = Clock::now();
    const auto weight_count = static_cast<size_t>(k * n);
    const size_t weight_bytes = weight_count * sizeof(uint16_t);
    const auto matches = [&](const Fixture::CachedWeight& entry) {
        return std::memcmp(entry.snapshot.data(), weights_nk, weight_bytes) == 0;
    };
    auto& cache = fixture->weight_cache;
    // Check likely pointers first, but never trust identity without a full comparison.
    auto found = std::ranges::find_if(
        cache, [&](const auto& entry) { return entry.source == weights_nk && matches(entry); });
    if (found == cache.end())
        found = std::ranges::find_if(cache, matches);
    const bool cache_hit = found != cache.end();
    if (!cache_hit) {
        Fixture::CachedWeight entry{weights_nk, fixture->next_weight_id++,
                                    std::vector<uint16_t>(weights_nk, weights_nk + weight_count),
                                    std::vector<uint16_t>(weight_count)};
        // GGML F16 weights are [N,K]; Phoenix B is row-major [K,N].
        for (int64_t col = 0; col < n; ++col)
            for (int64_t depth = 0; depth < k; ++depth)
                entry.packed[static_cast<size_t>((depth * n) + col)] =
                    bf16(f16(weights_nk[static_cast<size_t>((col * k) + depth)]));
        const size_t capacity = weight_cache_bytes / (2 * weight_bytes);
        if (cache.size() == capacity) {
            cache.erase(cache.begin());
        }
        cache.push_back(std::move(entry));
        found = std::prev(cache.end());
    }
    const bool weight_sync = fixture->resident_weight_id != found->id;
    if (weight_sync) {
        fixture->resident_weight_id = 0;
        std::memcpy(fixture->b.map<uint16_t*>(), found->packed.data(), weight_bytes);
    }
    profile.pack = micros(pack_start, Clock::now());
    const size_t chunks = (static_cast<size_t>(tokens) + rows_ - 1) / rows_;
    for (size_t offset = 0; std::cmp_less(offset, tokens); offset += rows_) {
        const size_t count = std::min(rows_, static_cast<size_t>(tokens) - offset);
        const auto setup_start = Clock::now();
        auto* a = fixture->a.map<uint16_t*>();
        std::fill(a + (count * static_cast<size_t>(k)), a + (rows_ * static_cast<size_t>(k)),
                  uint16_t{0});
        for (size_t t = 0; t < count; ++t)
            for (int64_t depth = 0; depth < k; ++depth)
                a[(t * static_cast<size_t>(k)) + static_cast<size_t>(depth)] =
                    bf16(activation_tk[((offset + t) * static_cast<size_t>(k)) +
                                       static_cast<size_t>(depth)]);
        profile.setup += micros(setup_start, Clock::now());
        fixture->run(weight_sync && offset == 0, profile);
        fixture->resident_weight_id = found->id;
        const auto output_start = Clock::now();
        const auto* c = fixture->c.map<uint16_t*>();
        for (size_t t = 0; t < count; ++t)
            for (int64_t col = 0; col < n; ++col) {
                const float value =
                    from_bf16(c[(t * static_cast<size_t>(n)) + static_cast<size_t>(col)]);
                if (!std::isfinite(value))
                    throw std::runtime_error("non-finite GEMM result");
                output_tn[((offset + t) * static_cast<size_t>(n)) + static_cast<size_t>(col)] =
                    value;
            }
        profile.output += micros(output_start, Clock::now());
    }
    const char* profiling = std::getenv("AIE2_PROFILE");
    if (profiling && std::strcmp(profiling, "1") == 0)
        std::cerr << R"({"event":"aie2_matmul","shape":")" << shape->name << R"(","tokens":)"
                  << tokens << R"(,"rows":)" << rows_ << R"(,"chunks":)" << chunks
                  << R"(,"cache_hit":)" << (cache_hit ? "true" : "false") << R"(,"weight_sync":)"
                  << (weight_sync ? "true" : "false") << R"(,"context_us":)" << profile.context
                  << R"(,"setup_us":)" << profile.setup << R"(,"pack_us":)" << profile.pack
                  << R"(,"sync_us":)" << profile.sync << R"(,"kernel_us":)" << profile.kernel
                  << R"(,"output_us":)" << profile.output << "}\n";
}
} // namespace aie2
