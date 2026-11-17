// GGML backend for AMD AIE2 (Phoenix/XDNA1) via XRT. Implements zero-copy
// mmap buffers (xrt::bo::map), one-time weight pre-packing in set_tensor,
// persistent XRT contexts/scratch BOs, and phase-level 3-way hybrid routing.
#include "ggml-backend-impl.h"
#include "xrt_context.h"
#include "xrt_ops.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct OllamaBufferTypeI {
    const char* (*get_name)(ggml_backend_buffer_type_t buft);
    ggml_backend_buffer_t (*alloc_buffer)(ggml_backend_buffer_type_t buft, size_t size);
    ggml_backend_buffer_t (*alloc_buffer_n)(ggml_backend_buffer_type_t buft,
                                            const struct ggml_tensor* const* tensors,
                                            size_t n_tensors);
    size_t (*get_alignment)(ggml_backend_buffer_type_t buft);
    size_t (*get_max_size)(ggml_backend_buffer_type_t buft);
    size_t (*get_alloc_size)(ggml_backend_buffer_type_t buft, const struct ggml_tensor* tensor);
    size_t (*get_alloc_size_n)(ggml_backend_buffer_type_t buft,
                               const struct ggml_tensor* const* tensors, size_t n_tensors);
    bool (*is_host)(ggml_backend_buffer_type_t buft);
};

struct OllamaBufferType {
    OllamaBufferTypeI iface;
    ggml_backend_dev_t device;
    void* context;
};

struct OllamaBufferI {
    void (*free_buffer)(ggml_backend_buffer_t buffer);
    void* (*get_base)(ggml_backend_buffer_t buffer);
    enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor);
    void (*memset_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor, uint8_t value,
                          size_t offset, size_t size);
    void (*set_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor, const void* data,
                       size_t offset, size_t size);
    void (*get_tensor)(ggml_backend_buffer_t buffer, const struct ggml_tensor* tensor, void* data,
                       size_t offset, size_t size);
    void (*set_tensor_2d)(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor,
                          const void* data, size_t offset, size_t size, size_t n_copies,
                          size_t stride_tensor, size_t stride_data);
    void (*get_tensor_2d)(ggml_backend_buffer_t buffer, const struct ggml_tensor* tensor,
                          void* data, size_t offset, size_t size, size_t n_copies,
                          size_t stride_tensor, size_t stride_data);
    bool (*cpy_tensor)(ggml_backend_buffer_t buffer, const struct ggml_tensor* src,
                       struct ggml_tensor* dst);
    void (*clear)(ggml_backend_buffer_t buffer, uint8_t value);
    void (*reset)(ggml_backend_buffer_t buffer);
    void* reserved_n0;
    void* reserved_n1;
    void* reserved_n2;
};

struct OllamaBuffer {
    OllamaBufferI iface;
    ggml_backend_buffer_type_t buft;
    void* context;
    size_t size;
    enum ggml_backend_buffer_usage usage;
};

static_assert(sizeof(OllamaBufferType) == 0x50);
static_assert(offsetof(OllamaBufferType, device) == 0x40);
static_assert(sizeof(OllamaBufferI) == 0x70);
static_assert(sizeof(OllamaBuffer) == 0x90);
static_assert(offsetof(OllamaBuffer, buft) == 0x70);
static_assert(offsetof(OllamaBuffer, context) == 0x78);

struct State {
    std::mutex init_lock;
    std::shared_ptr<XrtOpsRunner> ops;
    bool init_attempted = false;
};
State state;
extern ggml_backend_device device;
extern OllamaBufferType aie2_buffer_type_storage;

ggml_backend_buffer_type_t aie2_buft_ptr() {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<ggml_backend_buffer_type_t>(&aie2_buffer_type_storage);
}

OllamaBuffer* ollama_buf(ggml_backend_buffer_t buffer) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<OllamaBuffer*>(buffer);
}

std::shared_ptr<XrtOpsRunner> get_ops() {
    const char* env = std::getenv("AIE2_XRT_OPS_DIR");
    std::string ops_dir = (env && *env) ? env : "/usr/share/ggml-aie/kernel";
    const std::scoped_lock guard(state.init_lock);
    if (!state.ops && !state.init_attempted) {
        state.init_attempted = true;
        try {
            state.ops = std::make_shared<XrtOpsRunner>(std::move(ops_dir));
        } catch (const std::exception& e) {
            std::fprintf(stderr, "aie2: XRT ops init failed: %s\n", e.what());
        }
    }
    return state.ops;
}

bool can_read_as_f32(const struct ggml_tensor* t) {
    if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) {
        return true;
    }
    const auto* traits = ggml_get_type_traits(t->type);
    return traits && traits->to_float != nullptr && t->nb[0] == ggml_type_size(t->type);
}

float read_elem_f32(const struct ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char* ptr = static_cast<const char*>(t->data) + (i0 * t->nb[0]) + (i1 * t->nb[1]) +
                      (i2 * t->nb[2]) + (i3 * t->nb[3]);
    if (t->type == GGML_TYPE_F32) {
        return *reinterpret_cast<const float*>(ptr);
    }
    if (t->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(ptr));
    }
    return 0.0f;
}

void read_row_f32(const struct ggml_tensor* t, int64_t i1, int64_t i2, int64_t i3, float* dst) {
    const char* row_ptr =
        static_cast<const char*>(t->data) + (i1 * t->nb[1]) + (i2 * t->nb[2]) + (i3 * t->nb[3]);
    if (t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float)) {
        std::memcpy(dst, row_ptr, static_cast<size_t>(t->ne[0]) * sizeof(float));
        return;
    }
    const auto* traits = ggml_get_type_traits(t->type);
    if (traits && traits->to_float != nullptr && t->nb[0] == ggml_type_size(t->type)) {
        traits->to_float(row_ptr, dst, t->ne[0]);
        return;
    }
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        dst[i0] = read_elem_f32(t, i0, i1, i2, i3);
    }
}

void write_elem_f32(const struct ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3,
                    float val) {
    char* ptr = static_cast<char*>(t->data) + (i0 * t->nb[0]) + (i1 * t->nb[1]) + (i2 * t->nb[2]) +
                (i3 * t->nb[3]);
    if (t->type == GGML_TYPE_F32) {
        *reinterpret_cast<float*>(ptr) = val;
    } else if (t->type == GGML_TYPE_F16) {
        *reinterpret_cast<ggml_fp16_t*>(ptr) = ggml_fp32_to_fp16(val);
    }
}

void pack_tensor_f32(const struct ggml_tensor* t, std::vector<float>& out) {
    const int64_t n = ggml_nelements(t);
    out.resize(static_cast<size_t>(n));
    if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
        std::memcpy(out.data(), t->data, static_cast<size_t>(n) * sizeof(float));
        return;
    }
    const int64_t n0 = t->ne[0];
    int64_t offset = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                read_row_f32(t, i1, i2, i3, out.data() + offset);
                offset += n0;
            }
        }
    }
}

int32_t unary_op_code(enum ggml_unary_op uop) {
    switch (uop) {
    case GGML_UNARY_OP_SILU:
        return 0;
    case GGML_UNARY_OP_TANH:
        return 1;
    case GGML_UNARY_OP_SIGMOID:
        return 2;
    case GGML_UNARY_OP_RELU:
        return 3;
    case GGML_UNARY_OP_NEG:
        return 5;
    case GGML_UNARY_OP_ABS:
        return 6;
    case GGML_UNARY_OP_GELU:
        return 7;
    default:
        return -1;
    }
}

// Ollama 0.40.16 (llama.cpp b12177) inserts 5 ops before UNARY, shifting UNARY=92 and GLU=101.
constexpr int kOpUnary = 92;
constexpr int kOpGlu = 101;

struct PackedWeight {
    std::vector<float> data;
    std::vector<uint16_t> data_bf16;
    std::vector<xrt::bo> slice_bos;
    size_t bo_bytes = 0;
    bool is_bf16 = false;
    int64_t n_blocks = 0;
    int64_t k_blocks = 0;
    int64_t N = 0;
    int64_t K = 0;
};

struct ggml_backend_aie2_buffer {
    void* base = nullptr;
    size_t size = 0;
    size_t prepacked_bytes = 0;
    size_t bo_weight_bytes = 0;
    std::mutex pack_mu;
    std::unordered_map<const void*, PackedWeight> packed_weights;
};

std::atomic<size_t> g_total_prepack_bytes{0};
std::atomic<size_t> g_total_bo_weight_bytes{0};
constexpr size_t kMaxPrepackBytes = 1024ULL * 1024ULL * 1024ULL; // 1 GiB cap for F32/BF16 pre-pack
constexpr size_t kMaxBoWeightBytes = 256ULL * 1024ULL * 1024ULL; // 256 MiB cap for pinned XRT BOs

ggml_backend_aie2_buffer* get_aie2_buffer(const struct ggml_tensor* t) {
    if (!t) {
        return nullptr;
    }
    const struct ggml_tensor* root = t->view_src ? t->view_src : t;
    ggml_backend_buffer_t b = root->buffer;
    if (!b || ollama_buf(b)->buft != aie2_buft_ptr()) {
        return nullptr;
    }
    return static_cast<ggml_backend_aie2_buffer*>(ollama_buf(b)->context);
}

void ggml_backend_aie2_buffer_free(ggml_backend_buffer_t buffer) {
    const auto* ctx = static_cast<ggml_backend_aie2_buffer*>(ollama_buf(buffer)->context);
    if (ctx) {
        if (ctx->prepacked_bytes > 0) {
            g_total_prepack_bytes.fetch_sub(ctx->prepacked_bytes, std::memory_order_relaxed);
        }
        if (ctx->bo_weight_bytes > 0) {
            g_total_bo_weight_bytes.fetch_sub(ctx->bo_weight_bytes, std::memory_order_relaxed);
        }
        std::free(ctx->base);
        delete ctx;
    }
}

void* ggml_backend_aie2_buffer_get_base(ggml_backend_buffer_t buffer) {
    const auto* ctx = static_cast<ggml_backend_aie2_buffer*>(ollama_buf(buffer)->context);
    return ctx->base;
}

void ggml_backend_aie2_buffer_memset_tensor(
    ggml_backend_buffer_t /*buffer*/,
    struct ggml_tensor* tensor, // NOLINT(misc-const-correctness)
    uint8_t value, size_t offset, size_t size) {
    std::memset(static_cast<char*>(tensor->data) + offset, value, size);
}

void ggml_backend_aie2_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor,
                                         const void* data, size_t offset, size_t size) {
    auto* ctx = static_cast<ggml_backend_aie2_buffer*>(ollama_buf(buffer)->context);
    std::memcpy(static_cast<char*>(tensor->data) + offset, data, size);

    const size_t total_bytes = ggml_nbytes(tensor);
    if (offset != 0 || size != total_bytes) {
        return;
    }

    auto& mgr = XrtContextManager::instance();
    const int64_t K = tensor->ne[0];
    const int64_t N = tensor->ne[1];
    const bool is_2d_weight =
        (K > 1 && N > 1 && N <= 16384 && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
         (static_cast<uint32_t>(tensor->flags) & static_cast<uint32_t>(GGML_TENSOR_FLAG_INPUT)) ==
             0U &&
         can_read_as_f32(tensor) && std::strcmp(tensor->name, "token_embd.weight") != 0);

    if (is_2d_weight) {
        const auto t0 = std::chrono::steady_clock::now();
        constexpr int64_t TK = XrtOpsRunner::kBf16TK;
        constexpr int64_t KChunks = XrtOpsRunner::kBf16KChunks;
        constexpr int64_t KSlice = XrtOpsRunner::kBf16KSlice;
        constexpr int64_t TNW = XrtOpsRunner::kBf16TNW;
        constexpr int64_t NW = XrtOpsRunner::kBf16NW;
        constexpr int64_t TN = XrtOpsRunner::kBf16TN;
        constexpr auto kTileElems = static_cast<size_t>(TNW * TK);
        constexpr auto kColChunkElems = static_cast<size_t>(4) * kTileElems;
        constexpr auto kColSliceElems = static_cast<size_t>(KChunks) * kColChunkElems;
        constexpr auto kSliceElems = static_cast<size_t>(NW * KChunks) * kTileElems;

        const int64_t n_slices = (N + TN - 1) / TN;
        const int64_t k_slices = (K + KSlice - 1) / KSlice;
        const size_t packed_elems = static_cast<size_t>(n_slices * k_slices) * kSliceElems;
        const size_t packed_bytes = packed_elems * sizeof(uint16_t);
        const size_t prev = g_total_prepack_bytes.load(std::memory_order_relaxed);
        if (prev + packed_bytes > kMaxPrepackBytes) {
            return;
        }

        PackedWeight pw;
        pw.data_bf16.assign(packed_elems, uint16_t{0});
        pw.is_bf16 = true;
        pw.n_blocks = n_slices;
        pw.k_blocks = k_slices;
        pw.N = N;
        pw.K = K;

        const auto to_bf16 = [](float f) -> uint16_t {
            uint32_t u = 0;
            std::memcpy(&u, &f, sizeof(u));
            u += 0x7FFFu + ((u >> 16U) & 1U);
            return static_cast<uint16_t>(u >> 16U);
        };

        std::vector<float> row_buf(static_cast<size_t>(K));
        for (int64_t n = 0; n < N; ++n) {
            read_row_f32(tensor, n, 0, 0, row_buf.data());
            const int64_t ns = n / TN;
            const int64_t n_rem = n % TN;
            const int64_t w = n_rem / TNW;
            const int64_t col = w / 4;
            const int64_t row = w % 4;
            const int64_t n_local = n_rem % TNW;
            const int64_t n8 = n_local / 8;
            const int64_t n_in = n_local % 8;
            for (int64_t k = 0; k < K; ++k) {
                const int64_t ks = k / KSlice;
                const int64_t k_rem = k % KSlice;
                const int64_t kc = k_rem / TK;
                const int64_t k_local = k_rem % TK;
                const int64_t k8 = k_local / 8;
                const int64_t k_in = k_local % 8;
                const size_t idx = (static_cast<size_t>((ns * k_slices) + ks) * kSliceElems) +
                                   (static_cast<size_t>(col) * kColSliceElems) +
                                   (static_cast<size_t>(kc) * kColChunkElems) +
                                   (static_cast<size_t>(row) * kTileElems) +
                                   static_cast<size_t>((((n8 * 8) + k8) * 64) + (k_in * 8) + n_in);
                pw.data_bf16[idx] = to_bf16(row_buf[static_cast<size_t>(k)]);
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        mgr.add_prepack_ns(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

        const size_t cur_bo = g_total_bo_weight_bytes.load(std::memory_order_relaxed);
        if (cur_bo + packed_bytes <= kMaxBoWeightBytes) {
            try {
                const std::scoped_lock xrt_lock(mgr.mutex());
                const auto& prog = mgr.slot(Aie2OpKind::kGemmBf16);
                const auto n_slices_total = static_cast<size_t>(n_slices * k_slices);
                const size_t slice_bytes = kSliceElems * sizeof(uint16_t);
                const int grp = prog.kernel.group_id(4);
                pw.slice_bos.reserve(n_slices_total);
                for (size_t s = 0; s < n_slices_total; ++s) {
                    xrt::bo bo(mgr.device(), slice_bytes, xrt::bo::flags::host_only, grp);
                    std::memcpy(bo.map<void*>(), pw.data_bf16.data() + (s * kSliceElems),
                                slice_bytes);
                    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    pw.slice_bos.push_back(std::move(bo));
                }
                pw.bo_bytes = packed_bytes;
                pw.data_bf16.clear();
                pw.data_bf16.shrink_to_fit();
            } catch (const std::exception&) {
                pw.slice_bos.clear();
                pw.bo_bytes = 0;
            }
        }

        const std::scoped_lock lock(ctx->pack_mu);
        if (!ctx->packed_weights.contains(tensor->data)) {
            if (pw.bo_bytes > 0) {
                ctx->bo_weight_bytes += pw.bo_bytes;
                g_total_bo_weight_bytes.fetch_add(pw.bo_bytes, std::memory_order_relaxed);
            } else {
                ctx->prepacked_bytes += packed_bytes;
                g_total_prepack_bytes.fetch_add(packed_bytes, std::memory_order_relaxed);
            }
        }
        ctx->packed_weights[tensor->data] = std::move(pw);
    }
}

void ggml_backend_aie2_buffer_get_tensor(ggml_backend_buffer_t /*buffer*/,
                                         const struct ggml_tensor* tensor, void* data,
                                         size_t offset, size_t size) {
    std::memcpy(data, static_cast<const char*>(tensor->data) + offset, size);
}

void ggml_backend_aie2_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const auto* ctx = static_cast<ggml_backend_aie2_buffer*>(ollama_buf(buffer)->context);
    std::memset(ctx->base, value, ctx->size);
}

const OllamaBufferI aie2_buffer_iface = {
    ggml_backend_aie2_buffer_free,
    ggml_backend_aie2_buffer_get_base,
    nullptr,
    ggml_backend_aie2_buffer_memset_tensor,
    ggml_backend_aie2_buffer_set_tensor,
    ggml_backend_aie2_buffer_get_tensor,
    nullptr,
    nullptr,
    nullptr,
    ggml_backend_aie2_buffer_clear,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

const char* aie2_buft_get_name(ggml_backend_buffer_type_t /*buft*/) {
    return "AIE2-XRT-MMAP";
}

ggml_backend_buffer_t aie2_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (!get_ops()) {
        return nullptr;
    }
    const size_t alloc_bytes = std::max<size_t>((size + 63U) & ~static_cast<size_t>(63U), 64U);
    void* base = std::aligned_alloc(64, alloc_bytes);
    if (!base) {
        std::fprintf(stderr, "aie2: buffer alloc failed (%zu bytes)\n", size);
        return nullptr;
    }
    auto* ctx = new ggml_backend_aie2_buffer{base, alloc_bytes, 0, 0, {}, {}};
    using BufferInitFn =
        ggml_backend_buffer_t (*)(ggml_backend_buffer_type_t, OllamaBufferI, void*, size_t);
    const auto init_fn = reinterpret_cast<BufferInitFn>(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,bugprone-casting-through-void)
        reinterpret_cast<void*>(&ggml_backend_buffer_init));
    return init_fn(buft, aie2_buffer_iface, ctx, size);
}

size_t aie2_buft_get_alignment(ggml_backend_buffer_type_t /*buft*/) {
    return 64;
}

bool aie2_buft_is_host(ggml_backend_buffer_type_t /*buft*/) {
    return true;
}

OllamaBufferType aie2_buffer_type_storage = {
    {
        aie2_buft_get_name,
        aie2_buft_alloc_buffer,
        nullptr,
        aie2_buft_get_alignment,
        nullptr,
        nullptr,
        nullptr,
        aie2_buft_is_host,
    },
    &device,
    nullptr,
};

bool is_npu_dma_op(const ggml_tensor* op) {
    switch (static_cast<int>(op->op)) {
    case GGML_OP_MUL_MAT:
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_SUB:
    case GGML_OP_SCALE:
    case GGML_OP_DIV:
    case GGML_OP_UNARY:
    case kOpUnary:
    case kOpGlu:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_L2_NORM:
        return true;
    default:
        return false;
    }
}

bool full_offload_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("AIE2_FULL_OFFLOAD");
        return env && *env && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

int64_t min_gemm_m() {
    static const int64_t val = []() -> int64_t {
        const char* env = std::getenv("AIE2_MIN_GEMM_M");
        if (env && *env) {
            const long parsed = std::strtol(env, nullptr, 10);
            if (parsed >= 2) {
                return static_cast<int64_t>(parsed);
            }
        }
        return 32;
    }();
    return val;
}

bool ggml_backend_aie2_supports_op(ggml_backend_dev_t /*dev*/, const ggml_tensor* op) {
    const int64_t M = op->ne[1];
    switch (static_cast<int>(op->op)) {
    case GGML_OP_MUL_MAT: {
        if (M < min_gemm_m()) {
            return false;
        }
        const auto* src0 = op->src[0];
        const auto* src1 = op->src[1];
        if (!src0 || !src1 || !can_read_as_f32(src0) || src1->type != GGML_TYPE_F32 ||
            op->type != GGML_TYPE_F32) {
            return false;
        }
        const int64_t K = src0->ne[0];
        const int64_t N = src0->ne[1];
        return (M * K * N) >= (8LL * 1024LL * 1024LL);
    }
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM:
        return full_offload_enabled() && M > 1 && op->src[0]->type == GGML_TYPE_F32 &&
               op->src[0]->ne[0] > 0 && op->src[0]->ne[0] <= 4096;
    case GGML_OP_SOFT_MAX:
        return full_offload_enabled() && M > 1 && op->src[0]->type == GGML_TYPE_F32 &&
               op->src[0]->ne[0] > 0 && op->src[0]->ne[0] <= 4096 &&
               (!op->src[1] || op->src[1]->type == GGML_TYPE_F32 ||
                op->src[1]->type == GGML_TYPE_F16);
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_SUB: {
        if (!full_offload_enabled() || M <= 1 || op->src[0]->type != GGML_TYPE_F32 ||
            op->src[1]->type != GGML_TYPE_F32) {
            return false;
        }
        const auto* s0 = op->src[0];
        const auto* s1 = op->src[1];
        return ggml_are_same_shape(s0, s1) || ggml_nelements(s1) == 1 ||
               (s1->ne[0] == s0->ne[0] && s1->ne[1] == 1 && s1->ne[2] == 1 && s1->ne[3] == 1);
    }
    case GGML_OP_SCALE:
        return full_offload_enabled() && M > 1 && op->src[0]->type == GGML_TYPE_F32;
    case GGML_OP_DIV:
        return full_offload_enabled() && M > 1 && op->src[0]->type == GGML_TYPE_F32 &&
               op->src[1]->type == GGML_TYPE_F32 && ggml_nelements(op->src[1]) == 1;
    case GGML_OP_UNARY:
    case kOpUnary:
        return full_offload_enabled() && M > 1 && op->src[0]->type == GGML_TYPE_F32 &&
               unary_op_code(ggml_get_unary_op(op)) >= 0;
    case kOpGlu: {
        if (!full_offload_enabled() || M <= 1) {
            return false;
        }
        const enum ggml_glu_op glu = ggml_get_glu_op(op);
        return op->src[0]->type == GGML_TYPE_F32 &&
               (!op->src[1] || op->src[1]->type == GGML_TYPE_F32) && op->type == GGML_TYPE_F32 &&
               (glu == GGML_GLU_OP_REGLU || glu == GGML_GLU_OP_GEGLU || glu == GGML_GLU_OP_SWIGLU ||
                glu == GGML_GLU_OP_GEGLU_ERF || glu == GGML_GLU_OP_GEGLU_QUICK);
    }
    case GGML_OP_SUM_ROWS:
    case GGML_OP_L2_NORM:
        return full_offload_enabled() && M > 1 && op->src[0]->type == GGML_TYPE_F32 &&
               op->src[0]->ne[0] > 0 && op->src[0]->ne[0] <= 4096;
    case GGML_OP_GET_ROWS:
    case GGML_OP_ROPE:
    case GGML_OP_FLASH_ATTN_EXT:
    case GGML_OP_VIEW:
    case GGML_OP_RESHAPE:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_PAD:
    case GGML_OP_SET_ROWS:
    case GGML_OP_NONE:
    default:
        return false;
    }
}

const char* backend_name(ggml_backend_t /*backend*/) {
    return "AIE2-XRT";
}

struct BackendContext {
    std::shared_ptr<XrtOpsRunner> ops;
};

void backend_free(ggml_backend_t backend) {
    if (!backend) {
        return;
    }
    delete static_cast<BackendContext*>(backend->context);
    delete backend;
}

ggml_status compute_node(ggml_tensor* node) {
    switch (static_cast<int>(node->op)) {
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_PERMUTE:
        return GGML_STATUS_SUCCESS;
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_CPY: {
        const auto* src0 = node->src[0];
        if (src0->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) &&
            node->type == GGML_TYPE_F32 && ggml_is_contiguous(node)) {
            std::memcpy(node->data, src0->data,
                        static_cast<size_t>(ggml_nelements(node)) * sizeof(float));
            return GGML_STATUS_SUCCESS;
        }
        for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                        write_elem_f32(node, i0, i1, i2, i3, read_elem_f32(src0, i0, i1, i2, i3));
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_PAD: {
        const auto* src0 = node->src[0];
        int32_t p[9] = {};
        std::memcpy(p, node->op_params, sizeof(p));
        const int32_t lp0 = p[0];
        const int32_t rp0 = p[1];
        const int32_t lp1 = p[2];
        const int32_t rp1 = p[3];
        const int32_t lp2 = p[4];
        const int32_t rp2 = p[5];
        const int32_t lp3 = p[6];
        const int32_t rp3 = p[7];
        const bool circular = p[8] != 0;
        const auto wrap = [](int64_t x, int64_t n) -> int64_t {
            const int64_t r = x % n;
            return r < 0 ? r + n : r;
        };
        for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                        float val = 0.0f;
                        if (circular) {
                            val = read_elem_f32(
                                src0, wrap(i0 - lp0, src0->ne[0]), wrap(i1 - lp1, src0->ne[1]),
                                wrap(i2 - lp2, src0->ne[2]), wrap(i3 - lp3, src0->ne[3]));
                        } else {
                            const bool in0 = (i0 >= lp0 && i0 < node->ne[0] - rp0);
                            const bool in1 = (i1 >= lp1 && i1 < node->ne[1] - rp1);
                            const bool in2 = (i2 >= lp2 && i2 < node->ne[2] - rp2);
                            const bool in3 = (i3 >= lp3 && i3 < node->ne[3] - rp3);
                            if (in0 && in1 && in2 && in3) {
                                val = read_elem_f32(src0, i0 - lp0, i1 - lp1, i2 - lp2, i3 - lp3);
                            }
                        }
                        write_elem_f32(node, i0, i1, i2, i3, val);
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_GET_ROWS: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        const int64_t nc = node->ne[0];
        for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                    const char* idx_ptr = static_cast<const char*>(src1->data) +
                                          (i1 * src1->nb[0]) + (i2 * src1->nb[1]) +
                                          (i3 * src1->nb[2]);
                    const int32_t row = *reinterpret_cast<const int32_t*>(idx_ptr);
                    if (row < 0 || row >= src0->ne[1]) {
                        return GGML_STATUS_FAILED;
                    }
                    if (node->type == GGML_TYPE_F32 && node->nb[0] == sizeof(float)) {
                        auto* dst_row = reinterpret_cast<float*>(
                            static_cast<char*>(node->data) + (i1 * node->nb[1]) +
                            (i2 * node->nb[2]) + (i3 * node->nb[3]));
                        read_row_f32(src0, row, i2, i3, dst_row);
                    } else {
                        for (int64_t i0 = 0; i0 < nc; ++i0) {
                            write_elem_f32(node, i0, i1, i2, i3,
                                           read_elem_f32(src0, i0, row, i2, i3));
                        }
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_SET_ROWS: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        const int64_t nc = src0->ne[0];
        for (int64_t i3 = 0; i3 < src0->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src0->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < src0->ne[1]; ++i1) {
                    const char* idx_ptr = static_cast<const char*>(src1->data) +
                                          (i1 * src1->nb[0]) + ((i2 % src1->ne[1]) * src1->nb[1]) +
                                          ((i3 % src1->ne[2]) * src1->nb[2]);
                    const int64_t row = (src1->type == GGML_TYPE_I64)
                                            ? *reinterpret_cast<const int64_t*>(idx_ptr)
                                            : *reinterpret_cast<const int32_t*>(idx_ptr);
                    if (row < 0 || row >= node->ne[1]) {
                        return GGML_STATUS_FAILED;
                    }
                    for (int64_t i0 = 0; i0 < nc; ++i0) {
                        write_elem_f32(node, i0, row, i2, i3, read_elem_f32(src0, i0, i1, i2, i3));
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM: {
        const auto* src0 = node->src[0];
        float eps = 1e-5f;
        std::memcpy(&eps, node->op_params, sizeof(float));
        const int64_t row_len = src0->ne[0];
        const int64_t n_rows = ggml_nrows(src0);
        const bool is_rms = (node->op == GGML_OP_RMS_NORM);
        if (ggml_is_contiguous(src0)) {
            XrtOpsRunner::layernorm_f32(static_cast<const float*>(src0->data),
                                        static_cast<float*>(node->data), n_rows, row_len, eps,
                                        is_rms);
        } else {
            std::vector<float> in;
            pack_tensor_f32(src0, in);
            XrtOpsRunner::layernorm_f32(in.data(), static_cast<float*>(node->data), n_rows, row_len,
                                        eps, is_rms);
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_SUB: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        const int64_t total = ggml_nelements(node);
        const int64_t b_elems = ggml_nelements(src1);
        int32_t op_code = 0;
        if (node->op == GGML_OP_MUL) {
            op_code = 1;
        } else if (node->op == GGML_OP_SUB) {
            op_code = 2;
        }
        if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
            XrtOpsRunner::eltwise_f32(static_cast<const float*>(src0->data),
                                      static_cast<const float*>(src1->data),
                                      static_cast<float*>(node->data), total, b_elems, op_code);
        } else {
            std::vector<float> a;
            std::vector<float> b;
            pack_tensor_f32(src0, a);
            pack_tensor_f32(src1, b);
            XrtOpsRunner::eltwise_f32(a.data(), b.data(), static_cast<float*>(node->data), total,
                                      b_elems, op_code);
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_SCALE: {
        const auto* src0 = node->src[0];
        float scale = 1.0f;
        float bias = 0.0f;
        std::memcpy(&scale, reinterpret_cast<const float*>(node->op_params), sizeof(float));
        std::memcpy(&bias, reinterpret_cast<const float*>(node->op_params) + 1, sizeof(float));
        const int64_t total = ggml_nelements(node);
        const auto* in_ptr = static_cast<const float*>(src0->data);
        std::vector<float> in_tmp;
        if (!ggml_is_contiguous(src0)) {
            pack_tensor_f32(src0, in_tmp);
            in_ptr = in_tmp.data();
        }
        auto* out_ptr = static_cast<float*>(node->data);
        XrtOpsRunner::eltwise_f32(in_ptr, &scale, out_ptr, total, 1, 1);
        if (bias != 0.0f) {
            XrtOpsRunner::eltwise_f32(out_ptr, &bias, out_ptr, total, 1, 0);
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_DIV: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        const float inv = 1.0f / read_elem_f32(src1, 0, 0, 0, 0);
        const int64_t total = ggml_nelements(node);
        if (ggml_is_contiguous(src0)) {
            XrtOpsRunner::eltwise_f32(static_cast<const float*>(src0->data), &inv,
                                      static_cast<float*>(node->data), total, 1, 1);
        } else {
            std::vector<float> a;
            pack_tensor_f32(src0, a);
            XrtOpsRunner::eltwise_f32(a.data(), &inv, static_cast<float*>(node->data), total, 1, 1);
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_UNARY:
    case kOpUnary: {
        const auto* src0 = node->src[0];
        const enum ggml_unary_op uop = ggml_get_unary_op(node);
        const int32_t code = unary_op_code(uop);
        if (code < 0) {
            return GGML_STATUS_FAILED;
        }
        const int64_t n = ggml_nelements(node);
        const auto* in_ptr = static_cast<const float*>(src0->data);
        std::vector<float> in_tmp;
        if (!ggml_is_contiguous(src0)) {
            pack_tensor_f32(src0, in_tmp);
            in_ptr = in_tmp.data();
        }
        XrtOpsRunner::unary_f32(in_ptr, static_cast<float*>(node->data), n, code);
        return GGML_STATUS_SUCCESS;
    }
    case kOpGlu: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        const enum ggml_glu_op glu = ggml_get_glu_op(node);
        int32_t act_code = 0;
        if (glu == GGML_GLU_OP_REGLU) {
            act_code = 3;
        } else if (glu == GGML_GLU_OP_GEGLU || glu == GGML_GLU_OP_GEGLU_ERF ||
                   glu == GGML_GLU_OP_GEGLU_QUICK) {
            act_code = 7;
        } else if (glu == GGML_GLU_OP_SWIGLU) {
            act_code = 0;
        } else {
            return GGML_STATUS_FAILED;
        }
        int32_t swapped = 0;
        std::memcpy(&swapped, node->op_params + 1, sizeof(int32_t));
        const int64_t nc = node->ne[0];
        const int64_t nrows = ggml_nrows(node);
        const int64_t total = nc * nrows;
        std::vector<float> gate(static_cast<size_t>(total));
        std::vector<float> up(static_cast<size_t>(total));
        if (src1) {
            pack_tensor_f32(src0, gate);
            pack_tensor_f32(src1, up);
        } else {
            std::vector<float> full;
            pack_tensor_f32(src0, full);
            for (int64_t r = 0; r < nrows; ++r) {
                const float* row = full.data() + (r * 2 * nc);
                const float* first = row;
                const float* second = row + nc;
                std::memcpy(gate.data() + (r * nc), swapped ? second : first,
                            static_cast<size_t>(nc) * sizeof(float));
                std::memcpy(up.data() + (r * nc), swapped ? first : second,
                            static_cast<size_t>(nc) * sizeof(float));
            }
        }
        auto* out_ptr = static_cast<float*>(node->data);
        XrtOpsRunner::unary_f32(gate.data(), out_ptr, total, act_code);
        XrtOpsRunner::eltwise_f32(out_ptr, up.data(), out_ptr, total, total, 1);
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_MUL_MAT: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        const int64_t K = src0->ne[0];
        const int64_t N = src0->ne[1];
        const int64_t M = node->ne[1];
        const int64_t ne2 = node->ne[2];
        const int64_t ne3 = node->ne[3];
        const int64_t r2 = ne2 / src0->ne[2];
        const int64_t r3 = ne3 / src0->ne[3];

        const bool a_contig = (src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) &&
                               src1->nb[1] == static_cast<size_t>(K) * sizeof(float));

        ggml_backend_aie2_buffer* w_buf = get_aie2_buffer(src0);
        const PackedWeight* pw = nullptr;
        if (w_buf && src0->ne[2] == 1 && src0->ne[3] == 1) {
            const std::scoped_lock lock(w_buf->pack_mu);
            const auto it = w_buf->packed_weights.find(src0->data);
            if (it != w_buf->packed_weights.end() && it->second.N == N && it->second.K == K) {
                pw = &it->second;
            }
        }

        std::vector<float> a_slice;
        if (!a_contig) {
            a_slice.resize(static_cast<size_t>(M * K));
        }
        std::vector<float> b_slice;
        const bool b_f32_contig = (src0->type == GGML_TYPE_F32 && src0->nb[0] == sizeof(float) &&
                                   src0->nb[1] == static_cast<size_t>(K) * sizeof(float));
        if (!pw && !b_f32_contig) {
            b_slice.resize(static_cast<size_t>(N * K));
        }

        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            const int64_t i30 = i3 / r3;
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                const int64_t i20 = i2 / r2;
                const float* a_ptr = nullptr;
                if (a_contig) {
                    a_ptr = reinterpret_cast<const float*>(static_cast<const char*>(src1->data) +
                                                           (i2 * src1->nb[2]) + (i3 * src1->nb[3]));
                } else {
                    for (int64_t m = 0; m < M; ++m) {
                        read_row_f32(src1, m, i2, i3, a_slice.data() + (m * K));
                    }
                    a_ptr = a_slice.data();
                }

                auto* c_ptr = reinterpret_cast<float*>(static_cast<char*>(node->data) +
                                                       (i2 * node->nb[2]) + (i3 * node->nb[3]));
                if (pw) {
                    if (pw->is_bf16) {
                        XrtOpsRunner::gemm_bf16_nt_prepacked(
                            a_ptr, K, pw->data_bf16.data(),
                            pw->slice_bos.empty() ? nullptr : &pw->slice_bos, pw->n_blocks,
                            pw->k_blocks, c_ptr, N, M, N, K);
                    } else {
                        XrtOpsRunner::gemm_f32_nt_prepacked(a_ptr, K, pw->data.data(), pw->n_blocks,
                                                            pw->k_blocks, c_ptr, N, M, N, K);
                    }
                } else {
                    const float* b_ptr = nullptr;
                    if (b_f32_contig) {
                        b_ptr = reinterpret_cast<const float*>(
                            static_cast<const char*>(src0->data) + (i20 * src0->nb[2]) +
                            (i30 * src0->nb[3]));
                    } else {
                        for (int64_t n = 0; n < N; ++n) {
                            read_row_f32(src0, n, i20, i30, b_slice.data() + (n * K));
                        }
                        b_ptr = b_slice.data();
                    }
                    XrtOpsRunner::gemm_f32_nt(a_ptr, K, b_ptr, K, c_ptr, N, M, N, K);
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_SOFT_MAX: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        float scale = 1.0f;
        std::memcpy(&scale, node->op_params, sizeof(float));
        const int64_t row_len = src0->ne[0];
        const int64_t n_rows = ggml_nrows(src0);
        const auto* in_ptr = static_cast<const float*>(src0->data);
        std::vector<float> in_tmp;
        if (!ggml_is_contiguous(src0)) {
            pack_tensor_f32(src0, in_tmp);
            in_ptr = in_tmp.data();
        }
        std::vector<float> mask;
        int64_t mask_rows = 0;
        if (src1) {
            pack_tensor_f32(src1, mask);
            mask_rows = ggml_nrows(src1);
        }
        XrtOpsRunner::softmax_f32(in_ptr, src1 ? mask.data() : nullptr,
                                  static_cast<float*>(node->data), n_rows, row_len, mask_rows,
                                  scale);
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_ROPE: {
        const auto* src0 = node->src[0];
        const auto* src1 = node->src[1];
        const auto* src2 = node->src[2];
        const auto* p = reinterpret_cast<const int32_t*>(node->op_params);
        const int32_t n_dims = p[1];
        const int32_t mode = p[2];
        float freq_base = 10000.0f;
        float freq_scale = 1.0f;
        float attn_factor = 1.0f;
        std::memcpy(&freq_base, p + 5, sizeof(float));
        std::memcpy(&freq_scale, p + 6, sizeof(float));
        std::memcpy(&attn_factor, p + 8, sizeof(float));
        const bool is_neox =
            (static_cast<uint32_t>(mode) & static_cast<uint32_t>(GGML_ROPE_TYPE_NEOX)) != 0U;
        const int64_t ne0 = node->ne[0];
        const int64_t ne1 = node->ne[1];
        const int64_t ne2 = node->ne[2];
        const int64_t ne3 = node->ne[3];
        const float theta_scale = std::pow(freq_base, -2.0f / static_cast<float>(n_dims));
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                const auto* pos_ptr = reinterpret_cast<const int32_t*>(
                    static_cast<const char*>(src1->data) + (i2 * src1->nb[0]));
                const auto pos = static_cast<float>(*pos_ptr);
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    float theta_base = pos * freq_scale;
                    for (int64_t d = 0; d < n_dims / 2; ++d) {
                        const float ff = src2 ? read_elem_f32(src2, d, 0, 0, 0) : 1.0f;
                        const float ang = theta_base / ff;
                        const float cos_t = std::cos(ang) * attn_factor;
                        const float sin_t = std::sin(ang) * attn_factor;
                        theta_base *= theta_scale;
                        const int64_t idx0 = is_neox ? d : (2 * d);
                        const int64_t idx1 = is_neox ? (d + (n_dims / 2)) : ((2 * d) + 1);
                        const float x0 = read_elem_f32(src0, idx0, i1, i2, i3);
                        const float x1 = read_elem_f32(src0, idx1, i1, i2, i3);
                        write_elem_f32(node, idx0, i1, i2, i3, (x0 * cos_t) - (x1 * sin_t));
                        write_elem_f32(node, idx1, i1, i2, i3, (x0 * sin_t) + (x1 * cos_t));
                    }
                    for (int64_t d = n_dims; d < ne0; ++d) {
                        write_elem_f32(node, d, i1, i2, i3, read_elem_f32(src0, d, i1, i2, i3));
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_SUM_ROWS: {
        const auto* src0 = node->src[0];
        const int64_t row_len = src0->ne[0];
        const int64_t n_rows = ggml_nrows(src0);
        const auto* in_ptr = static_cast<const float*>(src0->data);
        std::vector<float> in_tmp;
        if (!ggml_is_contiguous(src0)) {
            pack_tensor_f32(src0, in_tmp);
            in_ptr = in_tmp.data();
        }
        XrtOpsRunner::reduce_sum_f32(in_ptr, static_cast<float*>(node->data), n_rows, row_len,
                                     1.0f);
        return GGML_STATUS_SUCCESS;
    }
    case GGML_OP_L2_NORM: {
        const auto* src0 = node->src[0];
        float eps = 1e-12f;
        std::memcpy(&eps, node->op_params, sizeof(float));
        const int64_t len = src0->ne[0];
        const int64_t n_vecs = ggml_nrows(src0);
        const auto* in_ptr = static_cast<const float*>(src0->data);
        std::vector<float> in_tmp;
        if (!ggml_is_contiguous(src0)) {
            pack_tensor_f32(src0, in_tmp);
            in_ptr = in_tmp.data();
        }
        XrtOpsRunner::l2_norm_f32(in_ptr, static_cast<float*>(node->data), n_vecs, len, eps);
        return GGML_STATUS_SUCCESS;
    }
    default:
        return GGML_STATUS_FAILED;
    }
}

ggml_status ggml_backend_aie2_graph_compute(ggml_backend_t backend, ggml_cgraph* cgraph) {
    const auto* ctx = static_cast<const BackendContext*>(backend->context);
    if (!ctx || !ctx->ops) {
        return GGML_STATUS_FAILED;
    }
    auto& mgr = XrtContextManager::instance();
    int npu_nodes = 0;

    const int n_nodes = ggml_graph_n_nodes(cgraph);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor* const node = ggml_graph_node(cgraph, i);
        if ((static_cast<uint32_t>(node->flags) &
             static_cast<uint32_t>(GGML_TENSOR_FLAG_COMPUTE)) == 0U) {
            continue;
        }
        if (is_npu_dma_op(node)) {
            ++npu_nodes;
        }

        try {
            const ggml_status st = compute_node(node);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "aie2: node[%d] op=%s returned status %d\n", i,
                             ggml_op_name(node->op), static_cast<int>(st));
                return st;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "aie2: XRT compute failed (op=%s): %s\n", ggml_op_name(node->op),
                         e.what());
            return GGML_STATUS_FAILED;
        }
    }

    if (npu_nodes > 0) {
        const double prepack_ms = static_cast<double>(mgr.prepack_ns()) / 1e6;
        const double sync_ms = static_cast<double>(mgr.take_sync_ns()) / 1e6;
        const double kernel_ms = static_cast<double>(mgr.take_kernel_ns()) / 1e6;
        std::fprintf(stderr,
                     "[aie2-timing] set_tensor_prepack=%.3f ms | bo_sync=%.3f ms | "
                     "npu_kernel=%.3f ms (nodes=%d)\n",
                     prepack_ms, sync_ms, kernel_ms, npu_nodes);
    }
    return GGML_STATUS_SUCCESS;
}

const ggml_backend_i backend_iface = {
    backend_name,
    backend_free,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    ggml_backend_aie2_graph_compute,
    nullptr,
    nullptr,
    nullptr,
};

const char* dev_name(ggml_backend_dev_t /*dev*/) {
    return "AIE2-XRT";
}

const char* dev_description(ggml_backend_dev_t /*dev*/) {
    return "AMD AIE2 XRT native full-graph encoder backend";
}

void dev_memory(ggml_backend_dev_t /*dev*/, size_t* free, size_t* total) {
    const long page_size = sysconf(_SC_PAGESIZE);
    const long phys_pages = sysconf(_SC_PHYS_PAGES);
    const long avail_pages = sysconf(_SC_AVPHYS_PAGES);
    if (page_size > 0 && phys_pages > 0) {
        const size_t total_ram = static_cast<size_t>(phys_pages) * static_cast<size_t>(page_size);
        const size_t avail_ram =
            (avail_pages > 0) ? (static_cast<size_t>(avail_pages) * static_cast<size_t>(page_size))
                              : (total_ram / 4);
        *total = total_ram / 4;
        *free = std::min(avail_ram, *total);
    } else {
        *free = *total = 0;
    }
}

enum ggml_backend_dev_type dev_type(ggml_backend_dev_t /*dev*/) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

void dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props* props) {
    props->name = dev_name(dev);
    props->description = dev_description(dev);
    props->type = dev_type(dev);
    dev_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {};
}

ggml_backend_t ggml_backend_aie2_init(ggml_backend_dev_t dev, const char* /*params*/) {
    auto ops = get_ops();
    if (!ops) {
        return nullptr;
    }
    XrtContextManager::instance().initialize(ops->artifacts_dir());
    auto* ctx = new BackendContext{std::move(ops)};
    static ggml_guid guid = {0x41, 0x49, 0x45, 0x32, 0x2d, 0x58, 0x52, 0x54,
                             0,    0,    0,    0,    0,    0,    0,    2};
    return new ggml_backend{&guid, backend_iface, dev, ctx};
}

ggml_backend_buffer_type_t dev_buft(ggml_backend_dev_t /*dev*/) {
    return aie2_buft_ptr();
}

bool dev_supports_buft(ggml_backend_dev_t /*dev*/, ggml_backend_buffer_type_t buft) {
    return buft == aie2_buft_ptr() || ggml_backend_buft_is_host(buft);
}

const ggml_backend_device_i device_iface = {
    dev_name,          dev_description,
    dev_memory,        dev_type,
    dev_props,         ggml_backend_aie2_init,
    dev_buft,          nullptr,
    nullptr,           ggml_backend_aie2_supports_op,
    dev_supports_buft, nullptr,
    nullptr,           nullptr,
    nullptr,
};

const char* reg_name(ggml_backend_reg_t /*reg*/) {
    return "AIE2-XRT";
}

size_t reg_count(ggml_backend_reg_t /*reg*/) {
    return get_ops() ? 1 : 0;
}

ggml_backend_dev_t reg_dev(ggml_backend_reg_t /*reg*/, size_t index) {
    return index == 0 && get_ops() ? &device : nullptr;
}

const ggml_backend_reg_i reg_iface = {reg_name, reg_count, reg_dev, nullptr};
// Ollama 0.40.16 (llama.cpp b12177) expects GGML_BACKEND_API_VERSION == 3.
constexpr int kGgmlBackendApiVersion = 3;
ggml_backend_reg registry = {kGgmlBackendApiVersion, reg_iface, nullptr};
ggml_backend_device device = {device_iface, &registry, nullptr};

} // namespace

static ggml_backend_reg_t backend_init() {
    return &registry;
}

GGML_BACKEND_DL_IMPL(backend_init)
