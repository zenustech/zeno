#include "SharcResources.h"

#include "NvrtcWorker.h"

#include <cuda_runtime_api.h>
#include <sutil/sutil.h>

#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace xinxinoptix {
namespace {

void setError(std::string* error, const std::string& message)
{
    if (error != nullptr) {
        *error = message;
    }
}

std::string driverError(CUresult result, const char* operation)
{
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);

    std::ostringstream stream;
    stream << operation << " failed";
    if (name != nullptr) {
        stream << " (" << name << ')';
    }
    if (message != nullptr) {
        stream << ": " << message;
    }
    return stream.str();
}

bool checkDriver(CUresult result, const char* operation, std::string* error)
{
    if (result == CUDA_SUCCESS) {
        return true;
    }
    setError(error, driverError(result, operation));
    return false;
}

std::string computeArchitectureOption(std::string* error)
{
    int device = 0;
    cudaDeviceProp properties = {};
    cudaError_t result = cudaGetDevice(&device);
    if (result == cudaSuccess) {
        result = cudaGetDeviceProperties(&properties, device);
    }
    if (result != cudaSuccess) {
        setError(error, std::string("Unable to query CUDA device: ") +
            cudaGetErrorString(result));
        return {};
    }
    return "--gpu-architecture=compute_" +
        std::to_string(properties.major) + std::to_string(properties.minor);
}

bool checkedBufferSize(
    uint32_t entry_count,
    std::size_t stride,
    std::size_t& byte_size)
{
    if (entry_count == 0u ||
        std::size_t(entry_count) > std::numeric_limits<std::size_t>::max() / stride) {
        return false;
    }
    byte_size = std::size_t(entry_count) * stride;
    return true;
}

} // namespace

SharcResources::~SharcResources()
{
    reset();
}

bool SharcResources::initialize(uint32_t entry_count, std::string* error)
{
    if (entry_count < 16u) {
        setError(error, "SHARC entry count must be at least the 16-entry hash bucket size");
        return false;
    }
    if (ready() && entry_count_ == entry_count) {
        return true;
    }

    reset();
    if (!checkDriver(cuInit(0), "cuInit", error)) {
        return false;
    }

    const std::string architecture = computeArchitectureOption(error);
    if (architecture.empty()) {
        return false;
    }
    const cudaError_t context_result = cudaFree(nullptr);
    if (context_result != cudaSuccess) {
        setError(error, std::string("Unable to initialize the CUDA primary context: ") +
            cudaGetErrorString(context_result));
        return false;
    }

    const char* source = nullptr;
    try {
        source = sutil::lookupIncFile("SharcResolve.cu");
    } catch (...) {
        setError(error, "Embedded SharcResolve.cu source is unavailable");
        return false;
    }

    const std::vector<const char*> options {
        "-std=c++17",
        "-default-device",
        "--use_fast_math",
        "--zeno-nvrtc-worker=0",
        architecture.c_str(),
    };
    auto compile_result = zeno::nvrtc_worker::compile(
        source, "", "SharcResolve.cu", options);
    if (!compile_result.success) {
        setError(error, "SHARC CUDA proxy compilation failed:\n" + compile_result.log);
        return false;
    }

    if (!checkDriver(
            cuModuleLoadData(&module_, compile_result.data.data()),
            "cuModuleLoadData(SHARC resolve)", error) ||
        !checkDriver(
            cuModuleGetFunction(&resolve_kernel_, module_, "zenoSharcResolve"),
            "cuModuleGetFunction(zenoSharcResolve)", error)) {
        reset();
        return false;
    }

    std::size_t hash_bytes = 0;
    std::size_t accumulation_bytes = 0;
    std::size_t resolved_bytes = 0;
    if (!checkedBufferSize(entry_count, hashEntryStride(), hash_bytes) ||
        !checkedBufferSize(entry_count, accumulationStride(), accumulation_bytes) ||
        !checkedBufferSize(entry_count, resolvedStride(), resolved_bytes)) {
        setError(error, "SHARC buffer size overflow");
        reset();
        return false;
    }

    if (!checkDriver(cuMemAlloc(&hash_entries_, hash_bytes),
                     "cuMemAlloc(SHARC hash entries)", error) ||
        !checkDriver(cuMemAlloc(&accumulation_, accumulation_bytes),
                     "cuMemAlloc(SHARC accumulation)", error) ||
        !checkDriver(cuMemAlloc(&resolved_, resolved_bytes),
                     "cuMemAlloc(SHARC resolved)", error)) {
        reset();
        return false;
    }

    entry_count_ = entry_count;
    if (!clear(nullptr, error) ||
        !checkDriver(cuStreamSynchronize(nullptr),
                     "cuStreamSynchronize(SHARC initialization)", error)) {
        reset();
        return false;
    }
    return true;
}

bool SharcResources::clear(CUstream stream, std::string* error)
{
    if (!ready()) {
        setError(error, "SHARC resources are not initialized");
        return false;
    }

    const std::size_t hash_bytes = std::size_t(entry_count_) * hashEntryStride();
    const std::size_t accumulation_bytes =
        std::size_t(entry_count_) * accumulationStride();
    const std::size_t resolved_bytes = std::size_t(entry_count_) * resolvedStride();
    return checkDriver(cuMemsetD8Async(hash_entries_, 0, hash_bytes, stream),
                       "cuMemsetD8Async(SHARC hash entries)", error) &&
        checkDriver(cuMemsetD8Async(accumulation_, 0, accumulation_bytes, stream),
                    "cuMemsetD8Async(SHARC accumulation)", error) &&
        checkDriver(cuMemsetD8Async(resolved_, 0, resolved_bytes, stream),
                    "cuMemsetD8Async(SHARC resolved)", error);
}

bool SharcResources::resolve(const SharcResolveSettings& settings, CUstream stream, std::string* error)
{
    if (!ready()) {
        setError(error, "SHARC resources are not initialized");
        return false;
    }
    if (!(settings.scene_scale > 0.0f) || !(settings.radiance_scale > 0.0f)) {
        setError(error, "SHARC scene and radiance scales must be positive");
        return false;
    }

    uint32_t capacity = entry_count_;
    CUdeviceptr hash_entries = hash_entries_;
    CUdeviceptr accumulation = accumulation_;
    CUdeviceptr resolved = resolved_;
    float camera_position_x = settings.camera_position[0];
    float camera_position_y = settings.camera_position[1];
    float camera_position_z = settings.camera_position[2];
    float camera_position_prev_x = settings.camera_position_prev[0];
    float camera_position_prev_y = settings.camera_position_prev[1];
    float camera_position_prev_z = settings.camera_position_prev[2];
    float scene_scale = settings.scene_scale;
    float radiance_scale = settings.radiance_scale;
    uint32_t accumulation_frame_num = settings.accumulation_frame_num;
    uint32_t responsive_frame_num = settings.responsive_frame_num;
    uint32_t stale_frame_num_max = settings.stale_frame_num_max;
    uint32_t frame_index = settings.frame_index;

    void* arguments[] = {
        &capacity,
        &hash_entries,
        &accumulation,
        &resolved,
        &camera_position_x,
        &camera_position_y,
        &camera_position_z,
        &camera_position_prev_x,
        &camera_position_prev_y,
        &camera_position_prev_z,
        &scene_scale,
        &radiance_scale,
        &accumulation_frame_num,
        &responsive_frame_num,
        &stale_frame_num_max,
        &frame_index,
    };

    constexpr uint32_t block_size = 256u;
    const uint32_t block_count = (entry_count_ + block_size - 1u) / block_size;
    return checkDriver(
        cuLaunchKernel(
            resolve_kernel_,
            block_count, 1u, 1u,
            block_size, 1u, 1u,
            0u, stream,
            arguments, nullptr),
        "cuLaunchKernel(zenoSharcResolve)", error);
}

void SharcResources::reset()
{
    if (resolved_ != 0) {
        cuMemFree(resolved_);
    }
    if (accumulation_ != 0) {
        cuMemFree(accumulation_);
    }
    if (hash_entries_ != 0) {
        cuMemFree(hash_entries_);
    }
    if (module_ != nullptr) {
        cuModuleUnload(module_);
    }
    resolved_ = 0;
    accumulation_ = 0;
    hash_entries_ = 0;
    resolve_kernel_ = nullptr;
    module_ = nullptr;
    entry_count_ = 0u;
}

bool SharcResources::ready() const noexcept
{
    return module_ != nullptr && resolve_kernel_ != nullptr &&
        hash_entries_ != 0 && accumulation_ != 0 && resolved_ != 0 &&
        entry_count_ != 0u;
}

uint32_t SharcResources::entryCount() const noexcept
{
    return entry_count_;
}

CUdeviceptr SharcResources::hashEntries() const noexcept
{
    return hash_entries_;
}

CUdeviceptr SharcResources::accumulation() const noexcept
{
    return accumulation_;
}

CUdeviceptr SharcResources::resolved() const noexcept
{
    return resolved_;
}

} // namespace xinxinoptix
