#include "xrt_context.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace fs = std::filesystem;

namespace {

uint32_t parse_u32_env(const char* name, uint32_t default_val) {
    const char* raw = std::getenv(name);
    if (!raw || *raw == '\0') {
        return default_val;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 0);
    if (end == raw || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
        return default_val;
    }
    return static_cast<uint32_t>(parsed);
}

uint32_t parse_priority_env(const char* name, uint32_t default_val) {
    const char* raw = std::getenv(name);
    if (!raw || *raw == '\0') {
        return default_val;
    }
    std::string lower(raw);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    struct NamedPriority {
        std::string_view key;
        uint32_t value;
    };
    constexpr std::array<NamedPriority, 6> kNamed = {{
        {"realtime", 0x100U},
        {"high", 0x180U},
        {"normal", 0x200U},
        {"low", 0x280U},
        {"none", 0U},
        {"unset", 0U},
    }};
    for (const auto& entry : kNamed) {
        if (lower == entry.key) {
            return entry.value;
        }
    }
    return parse_u32_env(name, default_val);
}

} // namespace

xrt::hw_context::cfg_param_type build_aie2_qos_config() {
    xrt::hw_context::cfg_param_type qos;
    const uint32_t priority = parse_priority_env("AIE2_QOS_PRIORITY", 0x200U);
    const uint32_t gops = parse_u32_env("AIE2_QOS_GOPS", 4000U);
    const uint32_t latency = parse_u32_env("AIE2_QOS_LATENCY", 1000U);
    const uint32_t fps = parse_u32_env("AIE2_QOS_FPS", 0U);
    const uint32_t dma_bw = parse_u32_env("AIE2_QOS_DMA_BW", 0U);
    const uint32_t frame_exec = parse_u32_env("AIE2_QOS_FRAME_EXEC_TIME", 0U);
    if (priority > 0U) {
        qos["priority"] = priority;
    }
    if (gops > 0U) {
        qos["gops"] = gops;
    }
    if (latency > 0U) {
        qos["latency"] = latency;
    }
    if (fps > 0U) {
        qos["fps"] = fps;
    }
    if (dma_bw > 0U) {
        qos["dma_bandwidth"] = dma_bw;
    }
    if (frame_exec > 0U) {
        qos["frame_execution_time"] = frame_exec;
    }
    return qos;
}

std::vector<uint32_t> load_xrt_insts_bin(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Failed to open insts.bin: " + path);
    }
    const std::streamsize bytes = ifs.tellg();
    if (bytes <= 0 || (bytes % 4) != 0) {
        throw std::runtime_error("Invalid insts.bin size: " + path);
    }
    ifs.seekg(0, std::ios::beg);
    std::vector<uint32_t> insts(static_cast<size_t>(bytes / 4));
    if (!ifs.read(reinterpret_cast<char*>(insts.data()), bytes)) {
        throw std::runtime_error("Failed to read insts.bin: " + path);
    }
    return insts;
}

XrtContextManager& XrtContextManager::instance() {
    static XrtContextManager mgr;
    return mgr;
}

XrtContextManager::XrtContextManager() : device_(0) {}

void XrtContextManager::initialize(const std::string& ops_dir) {
    const std::scoped_lock lock(mu_);
    if (initialized_ && initialized_dir_ == ops_dir) {
        return;
    }
    initialized_ = true;
    initialized_dir_ = ops_dir;
}

XrtContextManager::ProgramSlot& XrtContextManager::slot(Aie2OpKind kind) {
    ProgramSlot& prog = slots_[static_cast<size_t>(kind)];
    if (prog.initialized) {
        return prog;
    }
    if (!initialized_) {
        throw std::runtime_error("XrtContextManager::slot called before initialize()");
    }

    struct Spec {
        Aie2OpKind kind;
        const char* name;
        std::array<size_t, 3> arg_bytes;
    };

    constexpr size_t kGemmTileBytes = static_cast<size_t>(64) * 64U * sizeof(float);
    constexpr size_t kBf16ABytes = static_cast<size_t>(16) * 32U * 64U * sizeof(uint16_t);
    constexpr size_t kBf16BBytes = static_cast<size_t>(16) * 16U * 64U * 64U * sizeof(uint16_t);
    constexpr size_t kBf16CBytes = static_cast<size_t>(16) * 32U * 64U * sizeof(uint16_t);
    constexpr size_t kVecBytes = static_cast<size_t>(4096) * sizeof(float);
    constexpr size_t kAuxLargeBytes = static_cast<size_t>(4112) * sizeof(float);
    constexpr size_t kAuxSmallBytes = static_cast<size_t>(16) * sizeof(float);

    constexpr std::array<Spec, static_cast<size_t>(Aie2OpKind::kCount)> kSpecs = {{
        {Aie2OpKind::kGemmF32,
         "gemm_f32",
         {kGemmTileBytes, 4U * kGemmTileBytes, 4U * kGemmTileBytes}},
        {Aie2OpKind::kLayerNorm, "layernorm", {kVecBytes, kVecBytes, kAuxSmallBytes}},
        {Aie2OpKind::kSoftmax, "softmax", {kVecBytes, kAuxLargeBytes, kVecBytes}},
        {Aie2OpKind::kEltwise, "eltwise", {kVecBytes, kAuxLargeBytes, kVecBytes}},
        {Aie2OpKind::kUnary, "unary", {kVecBytes, kVecBytes, kAuxSmallBytes}},
        {Aie2OpKind::kL2Norm, "l2_norm", {kVecBytes, kVecBytes, kAuxSmallBytes}},
        {Aie2OpKind::kGemmBf16, "gemm_bf16", {kBf16ABytes, kBf16BBytes, kBf16CBytes}},
    }};

    const auto& spec = kSpecs[static_cast<size_t>(kind)];
    const fs::path base = fs::path(initialized_dir_) / spec.name;
    const std::string xclbin_path = (base / (std::string(spec.name) + ".xclbin")).string();
    const std::string insts_path = (base / "insts.bin").string();
    if (!fs::exists(xclbin_path)) {
        throw std::runtime_error("Missing xclbin: " + xclbin_path);
    }
    const auto insts = load_xrt_insts_bin(insts_path);
    const xrt::xclbin xb(xclbin_path);
    prog.uuid = device_.register_xclbin(xb);
    const auto qos = build_aie2_qos_config();
    prog.hw_ctx = qos.empty() ? xrt::hw_context(device_, prog.uuid)
                              : xrt::hw_context(device_, prog.uuid, qos);
    prog.kernel = xrt::kernel(prog.hw_ctx, "MLIR_AIE");
    prog.insts_count = static_cast<uint32_t>(insts.size());
    prog.insts_bo = xrt::bo(device_, insts.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
                            prog.kernel.group_id(1));
    prog.insts_bo.write(insts.data());
    timed_sync(prog.insts_bo, XCL_BO_SYNC_BO_TO_DEVICE);

    for (size_t i = 0; i < 3; ++i) {
        const size_t sz = spec.arg_bytes[i];
        if (sz > 0) {
            prog.args[i].bo = xrt::bo(device_, sz, xrt::bo::flags::host_only,
                                      prog.kernel.group_id(static_cast<int>(i + 3)));
            prog.args[i].mapped = prog.args[i].bo.map<void*>();
            prog.args[i].bytes = sz;
        }
    }

    prog.run = xrt::run(prog.kernel);
    prog.run.set_arg(0, 3U);
    prog.run.set_arg(1, prog.insts_bo);
    prog.run.set_arg(2, prog.insts_count);
    prog.initialized = true;
    return prog;
}

void XrtContextManager::timed_sync(xrt::bo& bo, xclBOSyncDirection dir) {
    const auto t0 = std::chrono::steady_clock::now();
    bo.sync(dir);
    const auto t1 = std::chrono::steady_clock::now();
    sync_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                       std::memory_order_relaxed);
}
