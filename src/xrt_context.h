#pragma once

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

class XrtContextManager {
  public:
    static XrtContextManager& instance();

    struct ProgramHandle {
        xrt::device* device = nullptr;
        xrt::hw_context* hw_ctx = nullptr;
        xrt::kernel* kernel = nullptr;
        std::vector<uint32_t> insts;
    };

    // Acquires an active hw_context + kernel for (xclbin_path, insts_path, kernel_name),
    // evicting the least-recently-used hw_context if more than max_contexts are active.
    // Caller must hold mutex() for the duration of BO creation/execution.
    std::mutex& mutex() {
        return mu_;
    }
    ProgramHandle acquire(const std::string& xclbin_path, const std::string& insts_path,
                          const std::string& kernel_name);

  private:
    XrtContextManager();

    struct CachedProgram {
        std::string key;
        xrt::uuid uuid;
        xrt::hw_context hw_ctx;
        xrt::kernel kernel;
        std::vector<uint32_t> insts;
    };

    static constexpr size_t kMaxActiveContexts = 4;

    std::mutex mu_;
    xrt::device device_;
    std::unordered_map<std::string, xrt::uuid> registered_xclbins_;
    std::unordered_map<std::string, std::vector<uint32_t>> cached_insts_;
    std::list<CachedProgram> lru_list_;
};

std::vector<uint32_t> load_xrt_insts_bin(const std::string& path);
