#pragma once

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

enum class Aie2OpKind : uint8_t {
    kGemmF32 = 0,
    kLayerNorm = 1,
    kSoftmax = 2,
    kEltwise = 3,
    kUnary = 4,
    kL2Norm = 5,
    kGemmBf16 = 6,
    kCount = 7,
};

class XrtContextManager {
  public:
    static XrtContextManager& instance();

    struct BufferSlot {
        xrt::bo bo;
        void* mapped = nullptr;
        size_t bytes = 0;
    };

    struct ProgramSlot {
        bool initialized = false;
        xrt::uuid uuid;
        xrt::hw_context hw_ctx;
        xrt::kernel kernel;
        xrt::run run;
        xrt::bo insts_bo;
        uint32_t insts_count = 0;
        std::array<BufferSlot, 3> args;

        template <typename T> T* arg_ptr(size_t idx) {
            return static_cast<T*>(args[idx].mapped);
        }
        template <typename T> [[nodiscard]] const T* arg_ptr(size_t idx) const {
            return static_cast<const T*>(args[idx].mapped);
        }
        xrt::bo& arg_bo(size_t idx) {
            return args[idx].bo;
        }
    };

    // Initializes xrt::device, all 6 xrt::xclbin, xrt::hw_context, xrt::kernel,
    // and pre-allocates + mmaps all workspace/scratch xrt::bo buffers once.
    void initialize(const std::string& ops_dir);

    std::mutex& mutex() {
        return mu_;
    }

    xrt::device& device() {
        return device_;
    }

    ProgramSlot& slot(Aie2OpKind kind);

    void timed_sync(xrt::bo& bo, xclBOSyncDirection dir);

    template <typename... Args> void timed_run_wait(ProgramSlot& prog, Args&&... bo_args) {
        const auto t0 = std::chrono::steady_clock::now();
        int idx = 3;
        (prog.run.set_arg(idx++, std::forward<Args>(bo_args)), ...);
        prog.run.start();
        // Blocks on DRM_IOCTL_AMDXDNA_WAIT_CMD hardware interrupt waitqueue.
        prog.run.wait(std::chrono::milliseconds(0));
        const auto t1 = std::chrono::steady_clock::now();
        kernel_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                             std::memory_order_relaxed);
    }

    void add_prepack_ns(int64_t ns) {
        prepack_ns_.fetch_add(ns, std::memory_order_relaxed);
    }
    void add_kernel_ns(int64_t ns) {
        kernel_ns_.fetch_add(ns, std::memory_order_relaxed);
    }

    [[nodiscard]] int64_t prepack_ns() const {
        return prepack_ns_.load(std::memory_order_relaxed);
    }
    int64_t take_sync_ns() {
        return sync_ns_.exchange(0, std::memory_order_relaxed);
    }
    int64_t take_kernel_ns() {
        return kernel_ns_.exchange(0, std::memory_order_relaxed);
    }

  private:
    XrtContextManager();

    std::mutex mu_;
    xrt::device device_;
    bool initialized_ = false;
    std::string initialized_dir_;
    std::array<ProgramSlot, static_cast<size_t>(Aie2OpKind::kCount)> slots_;
    std::atomic<int64_t> prepack_ns_{0};
    std::atomic<int64_t> sync_ns_{0};
    std::atomic<int64_t> kernel_ns_{0};
};

std::vector<uint32_t> load_xrt_insts_bin(const std::string& path);
