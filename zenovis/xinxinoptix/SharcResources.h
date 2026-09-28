#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace xinxinoptix {

struct SharcResolveSettings {
    float camera_position[3] = {0.0f, 0.0f, 0.0f};
    float camera_position_prev[3] = {0.0f, 0.0f, 0.0f};
    float scene_scale = 50.0f;
    float radiance_scale = 1000.0f;
    uint32_t accumulation_frame_num = 32u;
    uint32_t responsive_frame_num = 8u;
    uint32_t stale_frame_num_max = 64u;
    uint32_t frame_index = 0u;
};

class SharcResources {
public:
    SharcResources() = default;
    ~SharcResources();

    SharcResources(const SharcResources&) = delete;
    SharcResources& operator=(const SharcResources&) = delete;

    bool initialize(uint32_t entry_count, std::string* error = nullptr);
    bool clear(CUstream stream = nullptr, std::string* error = nullptr);
    bool resolve(
        const SharcResolveSettings& settings,
        CUstream stream = nullptr,
        std::string* error = nullptr);
    void reset();

    bool ready() const noexcept;
    uint32_t entryCount() const noexcept;
    CUdeviceptr hashEntries() const noexcept;
    CUdeviceptr accumulation() const noexcept;
    CUdeviceptr resolved() const noexcept;

    static constexpr std::size_t hashEntryStride() noexcept { return 8u; }
    static constexpr std::size_t accumulationStride() noexcept { return 32u; }
    static constexpr std::size_t resolvedStride() noexcept { return 24u; }

private:
    CUmodule module_ = nullptr;
    CUfunction resolve_kernel_ = nullptr;
    CUdeviceptr hash_entries_ = 0;
    CUdeviceptr accumulation_ = 0;
    CUdeviceptr resolved_ = 0;
    uint32_t entry_count_ = 0;
};

} // namespace xinxinoptix
