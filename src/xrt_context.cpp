#include "xrt_context.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

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

XrtContextManager::ProgramHandle XrtContextManager::acquire(const std::string& xclbin_path,
                                                            const std::string& insts_path,
                                                            const std::string& kernel_name) {
    const std::string key = xclbin_path + "|" + kernel_name;

    for (auto it = lru_list_.begin(); it != lru_list_.end(); ++it) {
        if (it->key == key) {
            lru_list_.splice(lru_list_.begin(), lru_list_, it);
            return ProgramHandle{&device_, &it->hw_ctx, &it->kernel, it->insts};
        }
    }

    auto uuid_it = registered_xclbins_.find(xclbin_path);
    if (uuid_it == registered_xclbins_.end()) {
        if (!fs::exists(xclbin_path)) {
            throw std::runtime_error("Missing xclbin: " + xclbin_path);
        }
        const xrt::xclbin xb(xclbin_path);
        xrt::uuid uuid = device_.register_xclbin(xb);
        uuid_it = registered_xclbins_.emplace(xclbin_path, uuid).first;
    }

    auto insts_it = cached_insts_.find(insts_path);
    if (insts_it == cached_insts_.end()) {
        insts_it = cached_insts_.emplace(insts_path, load_xrt_insts_bin(insts_path)).first;
    }

    while (lru_list_.size() >= kMaxActiveContexts) {
        lru_list_.pop_back();
    }

    xrt::hw_context hw_ctx(device_, uuid_it->second);
    xrt::kernel kernel(hw_ctx, kernel_name);
    lru_list_.push_front(CachedProgram{key, uuid_it->second, std::move(hw_ctx), std::move(kernel),
                                       insts_it->second});
    auto& entry = lru_list_.front();
    return ProgramHandle{&device_, &entry.hw_ctx, &entry.kernel, entry.insts};
}
