#pragma once

#include <cstdio>
#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <memory>
#include <nvrtc.h>
#include <optix.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <stdio.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <sutil/PPMLoader.h>
#include <optix_stack_size.h>
#include "NvrtcWorker.h"
#include "optixVolume.h"
#include "optix_types.h"
#include "raiicuda.h"
#include "zeno/utils/log.h"
#include "zeno/utils/string.h"
#include <filesystem>
#define CRYPTOPP_ENABLE_NAMESPACE_WEAK 1
#include <cryptopp/md5.h>
#include <cryptopp/hex.h>

//#include <GLFW/glfw3.h>

#include <tbb/task_group.h>
#include <parallel_hashmap/phmap.h>

#include <glm/common.hpp>
#include <glm/matrix.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <utility>
#include <vector>
#include <string>
#include <string_view>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <thread>

#include "BCX.h"
#include "ies/ies.h"
#include <tinygltf/json.hpp>

#include "zeno/utils/fileio.h"
#include "zeno/extra/TempNode.h"
#include "zeno/types/PrimitiveObject.h"
#include "ChiefDesignerEXR.h"
#include <stb_image.h>
#include <cudaMemMarco.hpp>

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}
namespace OptixUtil
{
    using namespace xinxinoptix;
////these are all material independent stuffs;
inline raii<OptixDeviceContext>             context                  ;

inline OptixPipelineCompileOptions          pipeline_compile_options ;
inline raii<OptixPipeline>                  pipeline                 ;

inline std::tuple<bool, bool>  raygen_config;
inline raii<OptixModule>                    raygen_module            ;
inline raii<OptixProgramGroup>              raygen_prog_group        ;
inline raii<OptixProgramGroup>              radiance_miss_group      ;
inline raii<OptixProgramGroup>              occlusion_miss_group     ;

inline raii<CUdeviceptr> d_raygen_record;
inline raii<CUdeviceptr> d_miss_records;
inline raii<CUdeviceptr> d_hitgroup_records;
inline raii<CUdeviceptr> d_callable_records;    

inline raii<OptixModule> sphere_ism;

inline raii<OptixModule> round_linear_ism;
inline raii<OptixModule> round_bezier_ism;
inline raii<OptixModule> round_catrom_ism;

inline raii<OptixModule> round_quadratic_ism;
inline raii<OptixModule> flat_quadratic_ism;
inline raii<OptixModule> round_cubic_ism;

inline uint CachedPrimitiveTypeFlags = UINT_MAX;
inline std::vector< std::function<void(void)> > garbageTasks;

inline void clearCallableProgramCache();
inline void resetPipelineProgramGroupsDirty(bool dirty = false);

inline void resetAll() {

    raygen_prog_group.reset();
    radiance_miss_group.reset();
    occlusion_miss_group.reset();

    raygen_module.reset();

    for (auto& task : garbageTasks) {
        task();
    }
    garbageTasks.clear();

    d_miss_records.reset();
    d_raygen_record.reset();
    d_hitgroup_records.reset();
    d_callable_records.reset();  

    clearCallableProgramCache();

    pipeline.reset();
    context.reset();

    CachedPrimitiveTypeFlags = UINT_MAX;
}

typedef std::tuple<uint, uint> PipelineMark;

inline PipelineMark pipelineMark = {};
////end material independent stuffs

inline static auto DefaultCompileOptions() {
    OptixModuleCompileOptions module_compile_options = {};
#if defined( NDEBUG )
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else 
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
    return module_compile_options;
}

inline unsigned int optixLogCallbackLevel()
{
    if (const char* env = std::getenv("ZENO_OPTIX_LOG_LEVEL")) {
        char* end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end != env && *end == '\0' && value >= 0 && value <= 4) {
            return static_cast<unsigned int>(value);
        }
    }

#if defined( NDEBUG )
    return 0;
#else
    return 4;
#endif
}

inline void createContext()
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK_LOG( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = optixLogCallbackLevel();
    options.validationMode            = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    OPTIX_CHECK_LOG( optixDeviceContextCreate( cu_ctx, &options, &context ) );
}

inline void configureAsyncMemoryPool()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaMemPool_t pool = nullptr;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device));

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    // Retain up to half of the VRAM currently available to this process. This
    // is a release threshold, not an eager allocation: the pool only keeps
    // memory that cudaMallocAsync has actually used.
    uint64_t release_threshold = static_cast<uint64_t>(free_bytes) / 2ull;
    CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &release_threshold));
    zeno::log_info("CUDA async pool release threshold: {:.2f} GiB (50% of {:.2f} GiB currently free)",
        static_cast<double>(release_threshold) / (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0));
}

inline bool configPipeline(OptixPrimitiveTypeFlags usesPrimitiveTypeFlags) {

    auto enough = (usesPrimitiveTypeFlags&CachedPrimitiveTypeFlags) == usesPrimitiveTypeFlags;
    if (CachedPrimitiveTypeFlags != UINT_MAX && enough) { return false; }
    CachedPrimitiveTypeFlags = usesPrimitiveTypeFlags;

    pipeline_compile_options = {};
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY; //OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.usesMotionBlur        = false;
    pipeline_compile_options.numPayloadValues      = 2;
    pipeline_compile_options.numAttributeValues    = 2;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_USER;
    //pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | usesPrimitiveTypeFlags;
    pipeline_compile_options.usesPrimitiveTypeFlags = usesPrimitiveTypeFlags;
    pipeline_compile_options.allowOpacityMicromaps = true;

    OptixModuleCompileOptions module_compile_options = DefaultCompileOptions();

    OptixBuiltinISOptions builtin_is_options {};
    builtin_is_options.usesMotionBlur = false;
    builtin_is_options.buildFlags     = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;

    const static auto PrimitiveTypeConfigs = std::vector<std::tuple<OptixPrimitiveTypeFlags, OptixPrimitiveType, OptixModule*>> {

        { OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE, OPTIX_PRIMITIVE_TYPE_SPHERE, &sphere_ism },

        { OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR,       OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR,       &round_linear_ism },
        { OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM,   OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM,   &round_catrom_ism },
        { OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BEZIER, OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER, &round_bezier_ism },
    
        { OPTIX_PRIMITIVE_TYPE_FLAGS_FLAT_QUADRATIC_BSPLINE,  OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE,  &flat_quadratic_ism  },
        { OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE, OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE, &round_quadratic_ism },
        { OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE,     OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE,     &round_cubic_ism     }
    };

    auto count = garbageTasks.size();
    for (auto& task : garbageTasks) {
        task();
    }
    garbageTasks.clear();

    for (auto& [pflag, ptype, module_ptr] : PrimitiveTypeConfigs) {
        if (pflag & pipeline_compile_options.usesPrimitiveTypeFlags) {
            builtin_is_options.builtinISModuleType = ptype;
            OPTIX_CHECK( optixBuiltinISModuleGet( context, &module_compile_options, &pipeline_compile_options, &builtin_is_options, module_ptr ) );
            
            garbageTasks.push_back([module_ptr=module_ptr](){
                optixModuleDestroy(*module_ptr);
                *module_ptr = 0u;
            });
        } //if
    }
    return true;
}

#define COMPILE_WITH_TASKS_CHECK( call ) check( call, #call, __FILE__, __LINE__ )

inline tbb::task_group _compile_group;

inline std::atomic<bool> pipeline_program_groups_dirty{false};

struct ShaderModuleBuildStats {
    unsigned int shader_count = 0;
    unsigned int optix_ir_count = 0;
    unsigned int ptx_count = 0;
    unsigned int optix_module_count = 0;
    unsigned int cuda_module_count = 0;
    uint32_t shader_compile_time_ms = 0;
    uint32_t optix_ir_compile_time_ms = 0;
    uint32_t ptx_compile_time_ms = 0;
    uint32_t optix_module_time_ms = 0;
    uint64_t cuda_module_time_ms = 0;
};

// Count overlapping worker intervals once while still accumulating sequential waves.
struct ShaderBuildTimingIntervals {
    using Clock = std::chrono::steady_clock;
    using Interval = std::pair<Clock::time_point, Clock::time_point>;

    std::mutex mutex;
    std::vector<Interval> intervals;

    void reset()
    {
        std::lock_guard<std::mutex> lock(mutex);
        intervals.clear();
    }

    void add(Clock::time_point begin, Clock::time_point end)
    {
        std::lock_guard<std::mutex> lock(mutex);
        intervals.emplace_back(begin, end);
    }

    uint32_t consumeMilliseconds()
    {
        std::vector<Interval> pending;
        {
            std::lock_guard<std::mutex> lock(mutex);
            pending.swap(intervals);
        }
        if (pending.empty()) {
            return 0u;
        }

        std::sort(pending.begin(), pending.end(), [](const Interval& lhs, const Interval& rhs) {
            return lhs.first < rhs.first;
        });
        auto interval_begin = pending.front().first;
        auto interval_end = pending.front().second;
        Clock::duration elapsed {};
        for (size_t i = 1; i < pending.size(); ++i) {
            if (pending[i].first <= interval_end) {
                interval_end = std::max(interval_end, pending[i].second);
            } else {
                elapsed += interval_end - interval_begin;
                interval_begin = pending[i].first;
                interval_end = pending[i].second;
            }
        }
        elapsed += interval_end - interval_begin;
        return static_cast<uint32_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
    }
};

struct ShaderModuleBuildCounters {
    std::atomic<unsigned int> shader_count {0};
    std::atomic<unsigned int> optix_ir_count {0};
    std::atomic<unsigned int> ptx_count {0};
    std::atomic<unsigned int> optix_module_count {0};
    ShaderBuildTimingIntervals shader_compile_time;
    ShaderBuildTimingIntervals optix_ir_compile_time;
    ShaderBuildTimingIntervals ptx_compile_time;
    ShaderBuildTimingIntervals optix_module_time;
};

inline ShaderModuleBuildCounters shader_module_build_counters;

inline void resetShaderModuleBuildStats()
{
    shader_module_build_counters.shader_count.store(0u, std::memory_order_relaxed);
    shader_module_build_counters.optix_ir_count.store(0u, std::memory_order_relaxed);
    shader_module_build_counters.ptx_count.store(0u, std::memory_order_relaxed);
    shader_module_build_counters.optix_module_count.store(0u, std::memory_order_relaxed);
    shader_module_build_counters.shader_compile_time.reset();
    shader_module_build_counters.optix_ir_compile_time.reset();
    shader_module_build_counters.ptx_compile_time.reset();
    shader_module_build_counters.optix_module_time.reset();
    xinxinoptix::resetVolumeDensityModuleBuildStats();
}

inline ShaderModuleBuildStats consumeShaderModuleBuildStats()
{
    ShaderModuleBuildStats stats;
    stats.shader_count = shader_module_build_counters.shader_count.exchange(0u, std::memory_order_relaxed);
    stats.optix_ir_count = shader_module_build_counters.optix_ir_count.exchange(0u, std::memory_order_relaxed);
    stats.ptx_count = shader_module_build_counters.ptx_count.exchange(0u, std::memory_order_relaxed);
    stats.optix_module_count = shader_module_build_counters.optix_module_count.exchange(0u, std::memory_order_relaxed);
    stats.shader_compile_time_ms = shader_module_build_counters.shader_compile_time.consumeMilliseconds();
    stats.optix_ir_compile_time_ms = shader_module_build_counters.optix_ir_compile_time.consumeMilliseconds();
    stats.ptx_compile_time_ms = shader_module_build_counters.ptx_compile_time.consumeMilliseconds();
    stats.optix_module_time_ms = shader_module_build_counters.optix_module_time.consumeMilliseconds();
    xinxinoptix::consumeVolumeDensityModuleBuildStats(stats.cuda_module_count, stats.cuda_module_time_ms);
    return stats;
}

inline void resetPipelineProgramGroupsDirty(bool dirty)
{
    pipeline_program_groups_dirty.store(dirty, std::memory_order_relaxed);
}

inline void markPipelineProgramGroupsDirty()
{
    pipeline_program_groups_dirty.store(true, std::memory_order_relaxed);
}

inline bool pipelineProgramGroupsDirty()
{
    return pipeline_program_groups_dirty.load(std::memory_order_relaxed);
}

enum class CompileModuleKind : size_t
{
    Raygen = 0,
    CoreShader,
    Callable,
    Other,
    Count
};

inline int nvrtcSplitCompileThreadCount(CompileModuleKind kind);
inline bool nvrtcUseWorkerProcessForModule(CompileModuleKind kind);

inline CompileModuleKind classifyCompileModule(const char* name)
{
    if (name == nullptr) {
        return CompileModuleKind::Other;
    }
    if (std::strstr(name, "PTKernel.cu") != nullptr) {
        return CompileModuleKind::Raygen;
    }
    if (std::strstr(name, "Callable.cu") != nullptr) {
        return CompileModuleKind::Callable;
    }
    if (std::strstr(name, "MatShader.cu") != nullptr) {
        return CompileModuleKind::CoreShader;
    }
    return CompileModuleKind::Other;
}

inline int nvrtcSplitCompileThreadCount(CompileModuleKind kind)
{
    switch (kind) {
        case CompileModuleKind::Callable: return 1;
        default: return 1;
    }
}

inline std::atomic<bool> callable_nvrtc_helper_enabled{true};

inline void setCallableNvrtcHelperEnabled(bool enabled)
{
    callable_nvrtc_helper_enabled.store(enabled, std::memory_order_relaxed);
}

inline int callableNvrtcHelperDirtyThreshold()
{
    return 4;
}

inline bool nvrtcUseWorkerProcessForModule(CompileModuleKind kind)
{
    if (kind == CompileModuleKind::Callable) {
        return callable_nvrtc_helper_enabled.load(std::memory_order_relaxed);
    }
    return true;
}

inline void executeOptixTask(OptixTask theTask, tbb::task_group& _c_group) {
     
    static const auto processor_count = std::thread::hardware_concurrency();

    uint m_maxNumAdditionalTasks = processor_count;
    uint numAdditionalTasksCreated = 0;

    std::vector<OptixTask> additionalTasks( m_maxNumAdditionalTasks );

    optixTaskExecute( theTask, 
                      additionalTasks.data(), 
                      m_maxNumAdditionalTasks, 
                      &numAdditionalTasksCreated );

    for( size_t i = 0; i < numAdditionalTasksCreated; ++i )
    {
        // Capture additionalTasks[i] by value since it will go out of scope.
        OptixTask task = additionalTasks[i];

        _c_group.run([task, &_c_group]() {
            executeOptixTask(task, _c_group);
        });
    }  
}

static std::vector<char> readData(std::string const& filename) {

    std::ifstream inputData(filename, std::ios::binary);

    if (inputData.fail())
    {
        std::cerr << "ERROR: readData() Failed to open file " << filename << '\n';
        return std::vector<char>();
    }

    // Copy the input buffer to a char vector.
    std::vector<char> data(std::istreambuf_iterator<char>(inputData), {});

    if (inputData.fail())
    {
        std::cerr << "ERROR: readData() Failed to read file " << filename << '\n';
        return std::vector<char>();
    }

    return data;
} // readData

inline std::string currentComputeArchitectureOption()
{
    int device = 0;
    cudaDeviceProp prop = {};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return {};
    }
    return "--gpu-architecture=compute_" +
        std::to_string(prop.major) + std::to_string(prop.minor);
}

inline bool isPtxEntryDeclaration(std::string_view line)
{
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string_view::npos || line[first] == '/') {
        return false;
    }

    auto tokenBegin = first;
    while (tokenBegin < line.size()) {
        const auto tokenEnd = line.find_first_of(" \t\r", tokenBegin);
        const auto token = line.substr(tokenBegin, tokenEnd - tokenBegin);
        if (token == ".entry") {
            return true;
        }
        if (token != ".visible" && token != ".weak" && token != ".extern") {
            return false;
        }
        if (tokenEnd == std::string_view::npos) {
            return false;
        }
        tokenBegin = line.find_first_not_of(" \t\r", tokenEnd);
    }
    return false;
}

inline bool makeOptixPtxView(
    const std::string& compiled_ptx,
    std::string& optix_ptx,
    size_t& removed_entry_count)
{
    optix_ptx.clear();
    optix_ptx.reserve(compiled_ptx.size());
    removed_entry_count = 0;

    bool removing_entry = false;
    bool found_entry_body = false;
    int brace_depth = 0;
    size_t line_begin = 0;
    while (line_begin < compiled_ptx.size()) {
        const auto newline = compiled_ptx.find('\n', line_begin);
        const auto line_end = newline == std::string::npos ? compiled_ptx.size() : newline + 1;
        const std::string_view line(compiled_ptx.data() + line_begin, line_end - line_begin);

        if (!removing_entry && isPtxEntryDeclaration(line)) {
            removing_entry = true;
            found_entry_body = false;
            brace_depth = 0;
            ++removed_entry_count;
        }

        if (removing_entry) {
            for (const char ch : line) {
                if (ch == '{') {
                    found_entry_body = true;
                    ++brace_depth;
                } else if (ch == '}' && found_entry_body) {
                    --brace_depth;
                    if (brace_depth < 0) {
                        return false;
                    }
                }
            }
            if (found_entry_body && brace_depth == 0) {
                removing_entry = false;
            }
        } else {
            optix_ptx.append(line.data(), line.size());
        }

        line_begin = line_end;
    }

    return !removing_entry && brace_depth == 0;
}

inline bool createModule(
    OptixModule &module,
    OptixDeviceContext &context,
    const char *source,
    const char *name,
    const std::vector<std::string>& macros = {},
    tbb::task_group* _c_group = nullptr,
    std::string* compiled_ptx = nullptr)
{
    const auto module_kind = classifyCompileModule(name);

    OptixModuleCompileOptions module_compile_options = OptixUtil::DefaultCompileOptions();
    module_compile_options.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    char log[16384] = {};
    size_t sizeof_log = sizeof( log );

    size_t      inputSize = 0;
    //TODO: the file path problem
    bool success=false;

    std::vector<const char*> compilerOptions {
        "-std=c++17", "-default-device",
        "--extra-device-vectorization",
        "--relocatable-device-code=true",
        "--use_fast_math",
    #if !defined( NDEBUG )      
        "-lineinfo" //"-G"//"--dopt=on",
    #endif
    };
    if (compiled_ptx == nullptr) {
        compilerOptions.push_back("--optix-ir");
    }
    std::string architecture_option;
    if (compiled_ptx != nullptr) {
        architecture_option = currentComputeArchitectureOption();
        if (!architecture_option.empty()) {
            compilerOptions.push_back(architecture_option.c_str());
        }
    }
    std::string split_compile_option;
    std::string worker_compile_option;

    {  
        int major, minor; 
        auto result = nvrtcVersion(&major, &minor);
        if (result == NVRTC_SUCCESS) {
            if (major >= 12) {
                split_compile_option = "--split-compile=" + std::to_string(nvrtcSplitCompileThreadCount(module_kind));
                compilerOptions.push_back(split_compile_option.c_str());
            }
        }
        if (!nvrtcUseWorkerProcessForModule(module_kind)) {
            worker_compile_option = "--zeno-nvrtc-worker=0";
            compilerOptions.push_back(worker_compile_option.c_str());
        }
        // printf("NVRTC Version %d.%d \n", major, minor);
    }

    std::string flat_macros = ""; 

    for (auto &ele : macros) {
        compilerOptions.push_back(ele.c_str());
        flat_macros += ele + "\n";
    }

    const auto shader_compile_begin = std::chrono::steady_clock::now();
    auto compile_result = zeno::nvrtc_worker::compile(source, flat_macros.c_str(), name, compilerOptions);
    const auto shader_compile_end = std::chrono::steady_clock::now();
    success = compile_result.success;
    inputSize = compile_result.data.size();

    if(!success) {
        std::cerr << "NVRTC compilation failed {" << name << '}';
        if (!compile_result.log.empty()) {
            std::cerr << ":\n" << compile_result.log;
        }
        std::cerr << std::endl;
        return false;
    }
    shader_module_build_counters.shader_count.fetch_add(1u, std::memory_order_relaxed);
    shader_module_build_counters.shader_compile_time.add(
        shader_compile_begin,
        shader_compile_end);
    if (compiled_ptx != nullptr) {
        shader_module_build_counters.ptx_count.fetch_add(1u, std::memory_order_relaxed);
        shader_module_build_counters.ptx_compile_time.add(
            shader_compile_begin,
            shader_compile_end);
    } else {
        shader_module_build_counters.optix_ir_count.fetch_add(1u, std::memory_order_relaxed);
        shader_module_build_counters.optix_ir_compile_time.add(
            shader_compile_begin,
            shader_compile_end);
    }

    std::string optix_ptx;
    const char* input = compile_result.data.data();
    if (compiled_ptx != nullptr) {
        size_t removed_entry_count = 0;
        if (!makeOptixPtxView(compile_result.data, optix_ptx, removed_entry_count)) {
            std::cerr << "Failed to separate CUDA kernels from OptiX PTX {" << name << "}" << std::endl;
            return false;
        }
        if (removed_entry_count == 0) {
            std::cerr << "Combined volume PTX contains no CUDA kernels {" << name << "}" << std::endl;
            return false;
        }
        input = optix_ptx.data();
        inputSize = optix_ptx.size();
    }

    const auto optix_module_begin = std::chrono::steady_clock::now();
    if (_c_group == nullptr) {
        //OPTIX_CHECK(
        auto resu = optixModuleCreate(context, &module_compile_options, &pipeline_compile_options, input, inputSize, log, &sizeof_log, &module);
        if (resu != OPTIX_SUCCESS) {
            module = nullptr;
            std::cerr
                << "OptiX module creation failed {" << name << "}: "
                << optixGetErrorName(resu) << " (" << static_cast<int>(resu) << "): "
                << optixGetErrorString(resu) << '\n';
            if (log[0] != '\0') {
                std::cerr << "OptiX module log:\n" << log << '\n';
            }
            return false;
        }
        //);
    } else {
        
        OptixTask firstTask;
        auto resu = optixModuleCreateWithTasks(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            input,
            inputSize,
            log,
            &sizeof_log,
            &module,
            &firstTask);
        if (resu != OPTIX_SUCCESS) {
            module = nullptr;
            std::cerr
                << "OptiX module creation failed {" << name << "}: "
                << optixGetErrorName(resu) << " (" << static_cast<int>(resu) << "): "
                << optixGetErrorString(resu) << '\n';
            if (log[0] != '\0') {
                std::cerr << "OptiX module log:\n" << log << '\n';
            }
            return false;
        }

        executeOptixTask(firstTask, *_c_group);
        //COMPILE_WITH_TASKS_CHECK( //);
        _c_group->wait();  
    }

    const auto optix_module_end = std::chrono::steady_clock::now();
    shader_module_build_counters.optix_module_count.fetch_add(1u, std::memory_order_relaxed);
    shader_module_build_counters.optix_module_time.add(
        optix_module_begin,
        optix_module_end);

    if (compiled_ptx != nullptr) {
        *compiled_ptx = std::move(compile_result.data);
    }
    return true;
}

inline void createRenderGroups(OptixDeviceContext &context, OptixModule &_module)
{
    OptixProgramGroupOptions  program_group_options = {};
    char   log[2048];
    size_t sizeof_log = sizeof( log );
    {
        OptixProgramGroupDesc desc    = {};
        desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module            = _module;
        desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context, &desc,
                    1,  // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &raygen_prog_group.reset()
                    ) );
    }

    {   
        OptixProgramGroupDesc desc  = {};
        desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module            = _module;
        desc.miss.entryFunctionName = "__miss__radiance";
        sizeof_log                  = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context, &desc,
                    1,  // num program groups
                    &program_group_options,
                    log, &sizeof_log,
                    &radiance_miss_group.reset()
                    ) );
        memset( &desc, 0, sizeof( OptixProgramGroupDesc ) );
        desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module            = _module;
        desc.miss.entryFunctionName = "__miss__occlusion";
        sizeof_log                  = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context, &desc,
                    1,  // num program groups
                    &program_group_options,
                    log, &sizeof_log,
                    &occlusion_miss_group.reset()
                    ) );
    }
}

inline void createRTProgramGroups(OptixDeviceContext &context, OptixModule &_module, 
                std::string kind, std::string entry, std::string nameIS, OptixModule* moduleIS,
                raii<OptixProgramGroup>& oGroup)
{
    OptixProgramGroupOptions  program_group_options = {};
    char   log[2048];
    size_t sizeof_log = sizeof( log );
//    std::cout<<kind<<std::endl;
//    std::cout<<entry<<std::endl;

    OptixProgramGroupDesc desc        = {};
    desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    const char* entryName = entry.empty()? nullptr:entry.c_str();

if (entryName != nullptr) { 

    if(kind == "OPTIX_PROGRAM_GROUP_KIND_CLOSEHITGROUP")
    {
        desc.hitgroup.moduleCH            = _module;
        desc.hitgroup.entryFunctionNameCH = entryName;
    } 
    else if(kind == "OPTIX_PROGRAM_GROUP_KIND_ANYHITGROUP")
    {
        desc.hitgroup.moduleAH            = _module;
        desc.hitgroup.entryFunctionNameAH = entryName;
    }
}

    if (moduleIS != nullptr) {
        desc.hitgroup.moduleIS            = *moduleIS;
        desc.hitgroup.entryFunctionNameIS = nullptr;
    } else {
        if (!nameIS.empty()) {
            desc.hitgroup.moduleIS            = _module;
            desc.hitgroup.entryFunctionNameIS = nameIS.c_str();
        }
    }

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                context,
                &desc,
                1,  // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &oGroup.reset()
                ) );
}
struct cuTexture{
    std::string md5;
    bool blockCompression = false;
    
    cudaArray_t gpuImageArray = nullptr;
    cudaTextureObject_t texture = 0llu;

    uint8_t channel;
    uint32_t width, height;
    float average = 0.0f;

    std::vector<float> cdf;
    std::vector<float> pdf; 
    std::vector<int> start;

    std::vector<float> rawData;

    cuTexture() {}
    cuTexture(uint32_t w, uint32_t h) : width(w), height(h) {}
    ~cuTexture()
    {
        if(gpuImageArray!=nullptr)
        {
            cudaFreeArray(gpuImageArray);
            texture = 0;
        }
    }
};
inline sutil::Texture loadCubeMap(const std::string& ppm_filename)
{

    return loadPPMTexture( ppm_filename, make_float3(1,1,1), nullptr );
}

inline std::shared_ptr<cuTexture> makeCudaTexture(unsigned char* img, int nx, int ny, int nc, bool blockCompression)
{
    auto texture = std::make_shared<cuTexture>(nx, ny);

    if (nx % 4 || ny % 4) {
        blockCompression = false;
    }

    std::vector<uchar4> alt;
    if (nc == 3 && !blockCompression) { // cuda doesn't support raw rgb, should be raw rgba or compressed rgb 
        auto count = nx * ny;    
        alt.resize(count);

        for (size_t i=0; i<count; ++i) {
            alt[i] = { img[i*nc + 0], img[i*nc + 1], img[i*nc + 2], 255u };
        }
        nc = 4;
        img = (unsigned char*)alt.data();
    }

    cudaError_t rc;
 
    if (blockCompression == false) {
        std::vector<int> xyzw(4, 0);
        for (int i=0; i<nc; ++i) {xyzw[i] = 8;}

        cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc(xyzw[0], xyzw[1], xyzw[2], xyzw[3], cudaChannelFormatKindUnsigned);
        rc = cudaMallocArray(&texture->gpuImageArray, &channelDescriptor, nx, ny, 0);
        if (rc != cudaSuccess) {
            std::cout<<"texture space alloc failed\n";
            return 0;
        }

        rc = cudaMemcpyToArray(texture->gpuImageArray, 0, 0, img, sizeof(unsigned char) * nc * nx * ny, cudaMemcpyHostToDevice);

    } else {

        std::vector<unsigned char> bc_data;
        cudaChannelFormatDesc channelDescriptor;

        if (nc == 1) {
            bc_data = compressBC4(img, nx, ny);
            channelDescriptor = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed4>();
        } else if (nc == 2) {
            bc_data = compressBC5(img, nx, ny);
            channelDescriptor = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed5>();
        } else if (nc == 3) {
            bc_data = compressBC1(img, nx, ny);
            channelDescriptor = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed1>();
        } else if (nc == 4) {
            bc_data = compressBC3(img, nx, ny);
            channelDescriptor = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed3>();
        } else {
            std::cout<<"texture data unsupported \n";
            return 0;
        }
        
        rc = cudaMallocArray(&texture->gpuImageArray, &channelDescriptor, nx, ny, 0);

        if (rc != cudaSuccess) {
            std::cout<<"texture space alloc failed\n";
            return 0;
        }

        rc = cudaMemcpyToArray(texture->gpuImageArray, 0, 0, bc_data.data(), bc_data.size(), cudaMemcpyHostToDevice);
    }

    if (rc != cudaSuccess) {
        std::cout<<"texture data copy failed\n";
        cudaFreeArray(texture->gpuImageArray);
        texture->gpuImageArray = nullptr;
        return 0;
    }

    cudaResourceDesc resourceDescriptor = { };
    resourceDescriptor.resType = cudaResourceTypeArray;
    resourceDescriptor.res.array.array = texture->gpuImageArray;
    cudaTextureDesc textureDescriptor = { };
    textureDescriptor.addressMode[0] = cudaAddressModeWrap;
    textureDescriptor.addressMode[1] = cudaAddressModeWrap;
    textureDescriptor.borderColor[0] = 0.0f;
    textureDescriptor.borderColor[1] = 0.0f;
    textureDescriptor.disableTrilinearOptimization = 1;
    textureDescriptor.filterMode = cudaFilterModeLinear;
    textureDescriptor.normalizedCoords = true;
    textureDescriptor.readMode = cudaReadModeNormalizedFloat ;
    textureDescriptor.sRGB = 0;
    rc = cudaCreateTextureObject(&texture->texture, &resourceDescriptor, &textureDescriptor, nullptr);
    if (rc != cudaSuccess) {
        std::cout<<"texture creation failed\n";
        texture->texture = 0;
        cudaFreeArray(texture->gpuImageArray);
        texture->gpuImageArray = nullptr;
        return 0;
    }

    texture->blockCompression = blockCompression;
    texture->channel = nc;
    return texture;

}

inline std::shared_ptr<cuTexture> makeFallbackAlphaTexture() {
    std::vector<uint8_t> img(4, 255);
    return makeCudaTexture(img.data(), 2, 2, 1, false);
}

inline void changeCudaTexture(std::shared_ptr<cuTexture> &texture, unsigned char* img, int nx, int ny, int nc, bool blockCompression=false)
{
    cudaFreeArray(texture->gpuImageArray);
    std::vector<uchar4> alt;

    if (nc == 3 && !blockCompression) { // cuda doesn't support raw rgb, should be raw rgba or compressed rgb
        auto count = nx * ny;
        alt.resize(count);

        for (size_t i=0; i<count; ++i) {
            alt[i] = { img[i*nc + 0], img[i*nc + 1], img[i*nc + 2], 255u };
        }
        nc = 4;
        img = (unsigned char*)alt.data();
    }

    if (nx%4 || ny%4) {
        blockCompression = false;
    }

    cudaError_t rc;

    if (blockCompression == false) {
        std::vector<int> xyzw(4, 0);
        for (int i=0; i<nc; ++i) {xyzw[i] = 8;}

        cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc(xyzw[0], xyzw[1], xyzw[2], xyzw[3], cudaChannelFormatKindUnsigned);
        rc = cudaMallocArray(&texture->gpuImageArray, &channelDescriptor, nx, ny, 0);
        if (rc != cudaSuccess) {
            std::cout<<"texture space alloc failed\n";
            return;
        }

        rc = cudaMemcpyToArray(texture->gpuImageArray, 0, 0, img, sizeof(unsigned char) * nc * nx * ny, cudaMemcpyHostToDevice);

    } else {

        std::vector<unsigned char> bc_data;
        cudaChannelFormatDesc channelDescriptor;

        if (nc == 1) {
            bc_data = compressBC4(img, nx, ny);
            channelDescriptor = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed4>();
        } else if (nc == 2) {
            bc_data = compressBC5(img, nx, ny);
            channelDescriptor = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed5>();
        } else if (nc == 3) {
            bc_data = compressBC1(img, nx, ny);
            channelDescriptor = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed1>();
        } else if (nc == 4) {
            bc_data = compressBC3(img, nx, ny);
            channelDescriptor = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed3>();
        } else {
            std::cout<<"texture data unsupported \n";
            return;
        }

        rc = cudaMallocArray(&texture->gpuImageArray, &channelDescriptor, nx, ny, 0);

        if (rc != cudaSuccess) {
            std::cout<<"texture space alloc failed\n";
            return;
        }

        rc = cudaMemcpyToArray(texture->gpuImageArray, 0, 0, bc_data.data(), bc_data.size(), cudaMemcpyHostToDevice);
    }

    if (rc != cudaSuccess) {
        std::cout<<"texture data copy failed\n";
        cudaFreeArray(texture->gpuImageArray);
        texture->gpuImageArray = nullptr;
        return;
    }
    texture->blockCompression = blockCompression;
}

template <typename TF=float>
inline void changeCudaTexture(std::shared_ptr<cuTexture> &texture, float* img, int nx, int ny, int nc)
{
    cudaFreeArray(texture->gpuImageArray);
    
    auto channel = (nc==3) ? 4:nc;
    std::vector<TF> data(nx * ny * channel, 0);
    if (nc == channel) {
        for (size_t i=0; i<data.size(); ++i) {
            data[i] = (TF)img[i];
        }
    } else {
        auto count = nx * ny;
        for (size_t i=0; i<count; ++i) {
            size_t dst_idx = i * channel;
            size_t src_idx = i * nc;

            for (int c=0; c<nc; ++c)
                data[dst_idx+c] = (TF)img[src_idx+c];
        }
    }
    
    std::vector<int> xyzw(4, 0);
    for (int i=0; i<channel; ++i) {xyzw[i] = sizeof(TF) * 8;}

    cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc(xyzw[0], xyzw[1], xyzw[2], xyzw[3], cudaChannelFormatKindFloat);
    cudaError_t rc = cudaMallocArray(&texture->gpuImageArray, &channelDescriptor, nx, ny, 0);
    
    if (rc != cudaSuccess) {
        std::cout<<"texture space alloc failed\n";
        return;
    }

    rc = cudaMemcpy2DToArray(texture->gpuImageArray, 0, 0, data.data(),
                             nx * sizeof(TF) * channel,
                             nx * sizeof(TF) * channel,
                             ny,
                             cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) {
        std::cout<<"texture data copy failed\n";
        cudaFreeArray(texture->gpuImageArray);
        texture->gpuImageArray = nullptr;
    }
}

template <typename TF=float>
inline std::shared_ptr<cuTexture> makeCudaTexture(float* img, int nx, int ny, int nc, bool commpress=false)
{
    auto texture = std::make_shared<cuTexture>(nx, ny);
    auto channel = (nc==3) ? 4:nc;

    std::vector<TF> data(nx * ny * channel, 0);
    if (nc == channel) {
        for (size_t i=0; i<data.size(); ++i) {
            data[i] = (TF)img[i];
        }
    } else {
        auto count = nx * ny;
        for (size_t i=0; i<count; ++i) {
            size_t dst_idx = i * channel;
            size_t src_idx = i * nc;

            for (int c=0; c<nc; ++c)
                data[dst_idx+c] = (TF)img[src_idx+c];
        }
    }

    std::vector<int> xyzw(4, 0);
    for (int i=0; i<channel; ++i) {xyzw[i] = sizeof(TF) * 8;}

    cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc(xyzw[0], xyzw[1], xyzw[2], xyzw[3], cudaChannelFormatKindFloat);
    cudaError_t rc = cudaMallocArray(&texture->gpuImageArray, &channelDescriptor, nx, ny, 0);
    if (rc != cudaSuccess) {
        std::cout<<"texture space alloc failed\n";
        return 0;
    }
    rc = cudaMemcpy2DToArray(texture->gpuImageArray, 0, 0, data.data(),
                             nx * sizeof(TF) * channel,
                             nx * sizeof(TF) * channel,
                             ny,
                             cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) {
        std::cout<<"texture data copy failed\n";
        cudaFreeArray(texture->gpuImageArray);
        texture->gpuImageArray = nullptr;
        return 0;
    }
    cudaResourceDesc resourceDescriptor = { };
    resourceDescriptor.resType = cudaResourceTypeArray;
    resourceDescriptor.res.array.array = texture->gpuImageArray;
    cudaTextureDesc textureDescriptor = { };
    textureDescriptor.addressMode[0] = cudaAddressModeWrap;
    textureDescriptor.addressMode[1] = cudaAddressModeWrap;
    textureDescriptor.borderColor[0] = 0.0f;
    textureDescriptor.borderColor[1] = 0.0f;
    textureDescriptor.disableTrilinearOptimization = 1;
    textureDescriptor.filterMode = cudaFilterModeLinear;
    textureDescriptor.normalizedCoords = true;
    textureDescriptor.readMode = cudaReadModeElementType;
    textureDescriptor.sRGB = 0;
    rc = cudaCreateTextureObject(&texture->texture, &resourceDescriptor, &textureDescriptor, nullptr);
    if (rc != cudaSuccess) {
        std::cout<<"texture creation failed\n";
        texture->texture = 0;
        cudaFreeArray(texture->gpuImageArray);
        texture->gpuImageArray = nullptr;
        return 0;
    }
    texture->channel = channel;
    return texture;
}

inline void logInfoVRAM(std::string info) {
    size_t free_byte, total_byte ;

    auto cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

        if ( cudaSuccess != cuda_status ){
            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
            exit(1);
        }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;

    double used_db = total_db - free_db ;

    std::cout << " <<< " << info << " >>> " << std::endl;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

inline std::vector<float> loadIES(const std::string& path, float& coneAngle)
{
    std::filesystem::path filePath = path;

    if ( !std::filesystem::exists(filePath) ) {
        std::cout << filePath.string() << " doesn't exist" << std::endl;
        return {};
    }

    auto iesBuffer = zeno::file_get_binary(path);
    auto iesString = std::string(iesBuffer.data());
    //std::cout << iesString << std::endl;

    blender::IESFile iesFile;
    iesFile.load(iesString);

    std::vector<float> iesData(iesFile.packed_size());
    iesFile.pack(iesData.data());
    coneAngle = iesFile.coneAngle();

    return iesData;
}

struct TexKey {
    std::string path;
    bool blockCompression;

    bool operator == (const TexKey& other) const {
        return path == other.path && blockCompression == other.blockCompression;
    }

    bool operator < (const TexKey& other) const {
        auto l = std::tie(this->path, this->blockCompression);
        auto r = std::tie(other.path, other.blockCompression);
        return l < r;
    }
};

struct TexKeyHash
{
    size_t operator()(const TexKey& key) const
    {
        return hash(key);
    }

    static size_t hash(const TexKey& s) noexcept
    {
        std::size_t h1 = std::hash<std::string>{}(s.path);
        std::size_t h2 = std::hash<bool>{}(s.blockCompression);
        return h1 ^ (h2 << 1); // or use boost::hash_combine (see Discussion)
    }

    static bool equal( const TexKey& x, const TexKey& y ) {
        return x.path == y.path && x.blockCompression == y.blockCompression;
    }
};

inline phmap::parallel_node_hash_map_m<TexKey, std::shared_ptr<cuTexture>, TexKeyHash> tex_lut;
inline phmap::parallel_flat_hash_map_m<TexKey, std::filesystem::file_time_type, TexKeyHash> g_tex_last_write_time;
inline phmap::parallel_flat_hash_map_m<std::string, std::string> md5_path_mapping;

inline std::optional<std::string> sky_tex;
inline std::shared_ptr<cuTexture> sky_tex_ptr;
inline std::string default_sky_tex;

inline std::optional<std::function<void(void)>> portal_delayed;

struct WrapperIES {
    raii<CUdeviceptr> ptr;
    float coneAngle = 0.0f;
};

inline std::map<std::string, WrapperIES> g_ies;

// Create cumulative distribution function for importance sampling of spherical environment lights.
// This is a textbook implementation for the CDF generation of a spherical HDR environment.
// See "Physically Based Rendering" v2, chapter 14.6.5 on Infinite Area Lights.

inline void calc_sky_cdf_map(cuTexture* tex, int nx, int ny, int nc, std::function<float(uint32_t)>& look) {

    tex->width  = nx;
    tex->height = ny;

    auto &sky_avg = tex->average;

    auto &sky_cdf = tex->cdf;
    auto &sky_pdf = tex->pdf;
    auto &sky_start = tex->start;

    //we need to recompute cdf
    sky_cdf.resize(nx*ny);
    sky_cdf.assign(nx*ny, 0);
    sky_pdf.resize(nx*ny);
    sky_pdf.assign(nx*ny, 0);
    sky_start.resize(nx*ny);
    sky_start.assign(nx*ny, 0);
    
    for(int jj=0; jj<ny;jj++)
    {
        for(int ii=0;ii<nx;ii++)
        {
            size_t idx2 = jj*nx*nc + ii*nc;
            size_t idx = jj*nx + ii;
            float illum = 0.0f;
            auto color = zeno::vec3f(look(idx2+0), look(idx2+1), look(idx2+2));
            illum = zeno::dot(color, zeno::vec3f(0.2722287, 0.6740818, 0.0536895));
            //illum = illum > 0.5? illum : 0.0f;
            illum = abs(illum) * sinf(3.1415926f*((float)jj + 0.5f)/(float)ny) * 2.0f * 3.1415926 * 3.1415926 ;

            sky_cdf[idx] += illum + (idx>0? sky_cdf[idx-1]:0);
        }
    }
    float total_illum = sky_cdf[sky_cdf.size()-1];
    sky_avg = total_illum / ((float)nx * (float)ny) ;
    for(int ii=0;ii<sky_cdf.size();ii++)
    {
        sky_cdf[ii] /= total_illum;

        if(ii>0)
        {
            if(sky_cdf[ii]>sky_cdf[ii-1])
            {
                sky_start[ii] = ii;
            }
            else
            {
                sky_start[ii] = sky_start[ii-1];
            }
        }
    }
}

static std::string calculateMD5(const std::vector<char>& input) {
    unsigned char digest[CryptoPP::Weak::MD5::DIGESTSIZE];
    CryptoPP::Weak::MD5().CalculateDigest(digest, (const unsigned char*)input.data(), input.size());
    CryptoPP::HexEncoder encoder;
    std::string output;
    encoder.Attach(new CryptoPP::StringSink(output));
    encoder.Put(digest, sizeof(digest));
    encoder.MessageEnd();
    return output;
}

namespace detail {
    template <typename T> struct is_void {
        static constexpr bool value = false;
    };
    template <> struct is_void<void> {
        static constexpr bool value = true;
    };
}

template<typename T=float>
static std::shared_ptr<cuTexture> changeOrCreateCuImage(const TexKey& tex_key, T* img, int nx, int ny, int nc, bool compress=false) {
        
    auto find = tex_lut.find(tex_key);

     if(find != tex_lut.end()) {
        auto ptr = find->second;
        if constexpr (std::is_same_v<T, float>)
            changeCudaTexture(ptr, img, nx, ny, nc);
        else if constexpr (std::is_same_v<T, unsigned char>)
            changeCudaTexture(ptr, img, nx, ny, nc, compress);

        return ptr;
    } else {
        return makeCudaTexture(img, nx, ny, nc, compress);
    }
}

template<typename TaskType=void>
inline void addTexture(std::string path, bool blockCompression=false, TaskType* task=nullptr)
{
    std::string native_path = std::filesystem::u8path(path).string();

    TexKey tex_key {path, blockCompression};

    zeno::log_debug("loading texture :{}", path);

    if (std::filesystem::exists(native_path)) {
        std::filesystem::file_time_type ftime = std::filesystem::last_write_time(native_path);

        if (g_tex_last_write_time.count(tex_key)) {
            if (g_tex_last_write_time[tex_key] == ftime) return;
        } 
        g_tex_last_write_time.insert( {tex_key, ftime} );

    } else {
        zeno::log_info("file {} doesn't exist", path);
        return;
    }
    
    auto input = readData(native_path);
    std::string md5Hash = calculateMD5(input);

    if ( md5_path_mapping.count(md5Hash) ) {

        auto& alt_path = md5_path_mapping[md5Hash];
        auto alt_key = TexKey { alt_path, blockCompression };

        if (tex_lut.count(alt_key)) {
            tex_lut.insert( {tex_key, tex_lut[alt_key]} );
            //zeno::log_info("path {} reuse {} tex", path, alt_path);
            return;
        }
    }
    else {
        md5_path_mapping.insert({md5Hash, path});
    }

    unsigned char* ldr_image = nullptr;
    float* hdr_image = nullptr;

    int nx, ny, nc;
    stbi_set_flip_vertically_on_load(true);

    std::function<float(uint32_t)> lookupTexture = [](uint32_t x) {return 0.0f;};
    std::function<void(void)>     cleanupTexture = [](){};

    std::shared_ptr<cuTexture> newTexture = nullptr;
    std::shared_ptr<std::vector<unsigned char>> ucdata;

    if (zeno::ends_with(path, ".exr", false)) {
        float* rgba;
        const char* err;
        using namespace zeno::ChiefDesignerEXR; // let a small portion of people drive Cayenne first
        int ret = LoadEXR(&rgba, &nx, &ny, native_path.c_str(), &err);
        if (ret != 0) {
            zeno::log_error("load exr: {}", err);
            FreeEXRErrorMessage(err);
            return;
        }
        nc = 4;
        auto count = nx * ny * nc;
        for (auto i = 0; i < count; i++) {
            rgba[i] = zeno::clamp(rgba[i], 0.f, 100000.0f);
        }
        nx = std::max(nx, 1);
        ny = std::max(ny, 1);
        for (auto i = 0; i < ny / 2; i++) {
            for (auto x = 0; x < nx * 4; x++) {
                auto index1 = i * (nx * 4) + x;
                auto index2 = (ny - 1 - i) * (nx * 4) + x;
                std::swap(rgba[index1], rgba[index2]);
            }
        }
        assert(rgba);
        hdr_image = rgba;

        lookupTexture = [rgba](uint32_t idx) {
            return rgba[idx];
        };
        cleanupTexture = [rgba]() {
            free(rgba);
        };
    }
    else if (zeno::ends_with(path, ".ies", false)) {
        float coneAngle;
        auto iesd = loadIES(path, coneAngle);

        if (iesd.empty()) {
            g_ies.erase(path);
            return;
        }

        raii<CUdeviceptr> iesBuffer;
        size_t data_length = iesd.size() * sizeof(float);
        iesBuffer.resize(data_length);
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)iesBuffer ), iesd.data(), data_length, cudaMemcpyHostToDevice ) );
        
        g_ies[path] = {std::move(iesBuffer), coneAngle };
        return;
    }
    else if (zeno::getSession().nodeClasses.count("ReadPNG16") > 0 && zeno::ends_with(path, ".png", false)) {
        auto outs = zeno::TempNodeSimpleCaller("ReadPNG16")
                .set2("path", path)
                .call();

        // Create nodes
        auto img = outs.get<zeno::PrimitiveObject>("image");
        if (img->verts.size() == 0) {
            newTexture = std::make_shared<cuTexture>();
            return;
        }
        nx = std::max(img->userData().get2<int>("w"), 1);
        ny = std::max(img->userData().get2<int>("h"), 1);
        nc = std::max(img->userData().get2<int>("channels"), 1);

        ucdata = std::make_shared<std::vector<unsigned char>>(img->verts.size() * nc);

        if (nc < 4) {

            for(size_t i=0; i<img->verts.size(); i+=1 ) {

                for (int c=0; c<nc; ++c) {
                    ucdata->at(i*nc+c) = (img->verts[i][c] * 255.0);
                }
            }
            ldr_image = ucdata->data();

        } else {

            assert(nc == 4);
            auto data = (uchar4*)ucdata->data();
            auto &alpha = img->verts.attr<float>("alpha");
            for (auto i = 0; i < nx * ny; i++) {
                data[i].x = (unsigned char)(img->verts[i][0]*255.0);
                data[i].y = (unsigned char)(img->verts[i][1]*255.0);
                data[i].z = (unsigned char)(img->verts[i][2]*255.0);
                data[i].w = (unsigned char)(alpha[i]        *255.0);
            }
            ldr_image = ucdata->data();
        }
        
        lookupTexture = [ucdata=ucdata, img=img](uint32_t idx) {
            auto ptr = ucdata->data();
            return ptr[idx]/255.0;
        };
    }
    else if (stbi_is_hdr(native_path.c_str())) {
        float *img = stbi_loadf(native_path.c_str(), &nx, &ny, &nc, 0);

        if(!img){
            zeno::log_error("loading hdr texture failed:{}", path);
            newTexture = std::make_shared<cuTexture>();
            return;
        }
        auto count = nx * ny * nc;
        for (auto i = 0; i < count; i++) {
            img[i] = zeno::clamp(img[i], 0.f, 100000.0f);
        }
        nx = std::max(nx, 1);
        ny = std::max(ny, 1);
        assert(img);
        hdr_image = img;

        lookupTexture = [img](uint32_t idx) {
            return img[idx];
        };
        cleanupTexture = [img]() {
            stbi_image_free(img);
        };
    }
    else {

        unsigned char *img = stbi_load(native_path.c_str(), &nx, &ny, &nc, 0);
        if(!img){
            zeno::log_error("loading ldr texture failed:{}", path);
            tex_lut.insert( {tex_key, std::make_shared<cuTexture>()} );
            return;
        }
        nx = std::max(nx, 1);
        ny = std::max(ny, 1);

        ldr_image = img;

        lookupTexture = [img](uint32_t idx) {
            return (float)img[idx] / 255;
        };
        cleanupTexture = [img]() {
            stbi_image_free(img);
        };
    }

    if (hdr_image != nullptr) {
        newTexture = changeOrCreateCuImage(tex_key, hdr_image, nx, ny, nc, blockCompression);
    } else if  (ldr_image != nullptr) {
        newTexture = changeOrCreateCuImage(tex_key, ldr_image, nx, ny, nc, blockCompression);
    } else {
        newTexture = makeFallbackAlphaTexture();
    }

    if (newTexture != nullptr) {
        newTexture->md5 = md5Hash;

        if constexpr (!detail::is_void<TaskType>::value) {
            if (task != nullptr) {
                (*task)(newTexture.get(), nx, ny, nc, lookupTexture);
            }
        }
        tex_lut.insert({ tex_key, newTexture });
    }
    cleanupTexture();
}

inline void removeTexture(const TexKey &key) {

    auto& path = key.path;

    if (!path.empty()) {

        if (tex_lut.count(key)) {

            md5_path_mapping.erase(tex_lut[key]->md5);
            tex_lut.erase(key);
        }
        else {
            zeno::log_error("removeTexture: {} not exists!", path);
        }
        g_tex_last_write_time.erase(key);
    }
}

inline std::shared_ptr<cuTexture> getTexturePtr(const std::string& path) {

    auto find = OptixUtil::tex_lut.find({path, false});

    if (find != OptixUtil::tex_lut.end())
        return find->second;
    else
        return nullptr;
}

inline void setSkyTexture(std::string& path) {
    
    auto task = [](cuTexture* tex, uint32_t nx, uint32_t ny, uint32_t nc, std::function<float(uint32_t)> &lookupTexture) {
        
        const auto float_count = nx * ny * nc;

        auto& rawData = tex->rawData;
        rawData.resize(float_count);
        for (uint32_t i=0; i<float_count; ++i) {
            rawData[i] = lookupTexture(i);
        }

        calc_sky_cdf_map(tex, nx, ny, nc, lookupTexture);
    };

    addTexture(path, false, &task);
    sky_tex_ptr = OptixUtil::getTexturePtr(path);
}

struct OptixShaderCore {
    raii<OptixModule>                        module {}; 
    OptixModule*                    moduleIS = nullptr;

    raii<OptixProgramGroup>   m_radiance_hit_group  {};
    raii<OptixProgramGroup>   m_occlusion_hit_group {};

    const char* _source;

    std::string _hittingEntry;
    std::string _shadingEntry;
    std::string _occlusionEntry;

    OptixShaderCore() {}
    ~OptixShaderCore() {
        module.reset();
        moduleIS = nullptr;

        m_radiance_hit_group.reset();
        m_occlusion_hit_group.reset();
    }

    OptixShaderCore(const char *shaderSource, std::string shadingEntry, std::string occlusionEntry)
    {
        _source = shaderSource;
        
        _shadingEntry = shadingEntry;
        _occlusionEntry = occlusionEntry;
    }

    OptixShaderCore(const char *shaderSource, std::string shadingEntry, std::string occlusionEntry, std::string hittingEntry)
    {
        _source = shaderSource;

        _hittingEntry = hittingEntry;
        _shadingEntry = shadingEntry;
        _occlusionEntry = occlusionEntry;
    }

    bool loadProgram(uint idx, const std::vector<std::string> &macro_list = {}, tbb::task_group* _c_group = nullptr)
    {
        std::string tmp_name = "MatShader.cu";
        tmp_name = "$" + std::to_string(idx) + tmp_name;
         
        if(createModule(module.reset(), context, _source, tmp_name.c_str(), macro_list, _c_group))
        {
            // std::cout<<"module created"<<std::endl;

            m_radiance_hit_group.reset();
            m_occlusion_hit_group.reset();

            createRTProgramGroups(context, module, 
                "OPTIX_PROGRAM_GROUP_KIND_CLOSEHITGROUP", 
                _shadingEntry, _hittingEntry, moduleIS, m_radiance_hit_group);

            createRTProgramGroups(context, module, 
                "OPTIX_PROGRAM_GROUP_KIND_ANYHITGROUP", 
                _occlusionEntry, _hittingEntry, moduleIS, m_occlusion_hit_group);

            markPipelineProgramGroupsDirty();
            //_c_group.wait();
            return true;
        }
        return false;
    }
};

struct CallableProgram
{
    std::string cache_key {};
    std::string ptx {};
    raii<OptixModule> module {};
    raii<OptixProgramGroup> prog_group {};
};

struct CallableProgramCacheEntry
{
    std::weak_ptr<CallableProgram> program {};
    bool compiling = false;
};

inline std::mutex callable_program_cache_mutex;
inline std::condition_variable callable_program_cache_cv;
inline std::unordered_map<std::string, CallableProgramCacheEntry> callable_program_cache;

inline void clearCallableProgramCache()
{
    std::lock_guard<std::mutex> lock(callable_program_cache_mutex);
    callable_program_cache.clear();
}

inline void pruneCallableProgramCacheEntry(const std::string& cache_key)
{
    if (cache_key.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(callable_program_cache_mutex);
    auto it = callable_program_cache.find(cache_key);
    if (it != callable_program_cache.end() && !it->second.compiling && it->second.program.expired()) {
        callable_program_cache.erase(it);
    }
}

inline std::string callableProgramCacheKey(
    const std::string& callable,
    const std::vector<std::string>& macros)
{
    auto hash_bytes = [](uint64_t& hash, std::string_view bytes) {
        for (unsigned char ch : bytes) {
            hash ^= ch;
            hash *= 1099511628211ull;
        }
    };

    uint64_t hash0 = 1469598103934665603ull;
    uint64_t hash1 = 1099511628211ull;
    hash_bytes(hash0, callable);
    hash_bytes(hash1, std::to_string(callable.size()));
    hash_bytes(hash1, callable);
    for (const auto& macro : macros) {
        hash_bytes(hash0, std::string_view("\0", 1));
        hash_bytes(hash0, macro);
        hash_bytes(hash1, std::to_string(macro.size()));
        hash_bytes(hash1, macro);
    }

    std::ostringstream key;
    key << std::hex << hash0 << ':' << hash1 << ':' << std::dec << callable.size() << ':' << macros.size();
    return key.str();
}

struct OptixShaderWrapper
{
    bool dirty = true;
    std::shared_ptr<OptixShaderCore> core{};
    
    std::string                       callable_src {};
    std::shared_ptr<CallableProgram>  callable_prg {};

    std::vector<uint64_t>                  texs {};
    std::vector<std::string>               vbds {};

    nlohmann::json                    parameters{};
    std::map<std::string, std::string>   macros {};

    std::string                       density_signature {};
    uint8_t                           density_vdb_primary_slot = 0;
    std::set<uint8_t>                 density_vdb_referenced_slots {};
    bool force_density_bake = false;

    bool isVol() const {
        return macros.count("_volu_");
    }

    bool isHomoVol() const {
        return macros.count("_homo_");
    }

    OptixShaderWrapper() = default;
    ~OptixShaderWrapper() = default;

    OptixShaderWrapper(OptixShaderWrapper&& ref) = default;
    
    OptixShaderWrapper(std::shared_ptr<OptixShaderCore> _core_, const std::string& callableSource) 
    {
        core = _core_; callable_src = callableSource;
    } 

    bool loadProgram(uint idx, bool fallback=false, tbb::task_group* _c_group = nullptr)
    {
        std::string tmp_name = "Callable.cu";
        tmp_name = "$" + std::to_string(idx) + tmp_name;
        const auto old_callable_group = callable_prg ? callable_prg->prog_group.handle : nullptr;
        const auto old_cache_key = callable_prg ? callable_prg->cache_key : std::string{};

        std::vector<std::string> _macros_ {};

        if (!fallback) {
            _macros_.push_back("--define-macro=__FORWARD__");
        }

        for (auto& [k, v] : this->macros) {
            _macros_.push_back("--define-macro=" + k + "=" + v);
        }

        const bool compile_shared_volume_artifacts = isVol();
        std::string cache_key = callableProgramCacheKey(callable_src, _macros_);
        if (compile_shared_volume_artifacts) {
            cache_key += ':' + currentComputeArchitectureOption();
        }
        {
            std::unique_lock<std::mutex> lock(callable_program_cache_mutex);
            while (true) {
                auto& entry = callable_program_cache[cache_key];
                if (auto cached_program = entry.program.lock()) {
                    callable_prg = std::move(cached_program);
                    if (compile_shared_volume_artifacts) {
                        xinxinoptix::prepareVolumeDensityBakeModuleAsync(
                            callable_prg->cache_key,
                            callable_prg->ptx);
                    }
                    if (old_callable_group != callable_prg->prog_group.handle) {
                        markPipelineProgramGroupsDirty();
                    }
                    if (old_cache_key != cache_key) {
                        lock.unlock();
                        pruneCallableProgramCacheEntry(old_cache_key);
                    }
                    return true;
                }

                if (!entry.compiling) {
                    entry.compiling = true;
                    break;
                }

                callable_program_cache_cv.wait(lock);
            }
        }

        try {
            auto program = std::make_shared<CallableProgram>();
            program->cache_key = cache_key;
            auto callable_done = createModule(
                program->module.reset(),
                context,
                callable_src.c_str(),
                tmp_name.c_str(),
                _macros_,
                compile_shared_volume_artifacts ? nullptr : _c_group,
                compile_shared_volume_artifacts ? &program->ptx : nullptr);
            if (callable_done) {

                if (compile_shared_volume_artifacts) {
                    xinxinoptix::prepareVolumeDensityBakeModuleAsync(
                        program->cache_key,
                        program->ptx);
                }

                // Callable programs
                OptixProgramGroupOptions callable_prog_group_options  = {};
                OptixProgramGroupDesc    callable_prog_group_descs[1] = {};

                callable_prog_group_descs[0].kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
                callable_prog_group_descs[0].callables.moduleDC            = program->module;
                callable_prog_group_descs[0].callables.entryFunctionNameDC = "__direct_callable__evalmat";

                char LOG[2048];
                size_t LOG_SIZE = sizeof( LOG );

                OPTIX_CHECK( 
                    optixProgramGroupCreate( context, callable_prog_group_descs, 1, &callable_prog_group_options, LOG, &LOG_SIZE, &program->prog_group.reset())
                );
                callable_prg = program;
                if (old_callable_group != callable_prg->prog_group.handle) {
                    markPipelineProgramGroupsDirty();
                }

                {
                    std::lock_guard<std::mutex> lock(callable_program_cache_mutex);
                    auto& entry = callable_program_cache[cache_key];
                    entry.program = program;
                    entry.compiling = false;
                }
                callable_program_cache_cv.notify_all();
                if (old_cache_key != cache_key) {
                    pruneCallableProgramCacheEntry(old_cache_key);
                }
                return true;
            }

            {
                std::lock_guard<std::mutex> lock(callable_program_cache_mutex);
                callable_program_cache.erase(cache_key);
            }
            callable_program_cache_cv.notify_all();
            return false;
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(callable_program_cache_mutex);
                callable_program_cache.erase(cache_key);
            }
            callable_program_cache_cv.notify_all();
            throw;
        }
    }

    void clearTextureRecords()
    {
        texs = {};
    }
    cudaTextureObject_t getTexture(int i)
    {
        if (i>=texs.size())
            return 0;
        else
            return texs[i];
    }
};

inline std::vector<OptixShaderWrapper> rtMaterialShaders;//just have an arry of shaders

inline void createPipeline(uint tree_depth, bool shaderDirty)
{
    auto shader_count = rtMaterialShaders.size();
    auto newMark = PipelineMark {tree_depth, shader_count};
    if (!shaderDirty && newMark == pipelineMark)
        return;

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = 2;

    size_t num_progs = 3 + rtMaterialShaders.size() * 2;
    num_progs += rtMaterialShaders.size(); // callables;

    std::vector<OptixProgramGroup> program_groups(num_progs, {});
    program_groups[0] = raygen_prog_group;
    program_groups[1] = radiance_miss_group;
    program_groups[2] = occlusion_miss_group;
    for(size_t i=0;i<rtMaterialShaders.size();i++)
    {
        program_groups[3 + i*2] = rtMaterialShaders[i].core->m_radiance_hit_group;
        program_groups[3 + i*2 + 1] = rtMaterialShaders[i].core->m_occlusion_hit_group;

        program_groups[3 + 2 * rtMaterialShaders.size() + i] = rtMaterialShaders[i].callable_prg->prog_group;
    }
    char   log[2048];
    size_t sizeof_log = sizeof( log );

    if (std::get<0>(pipelineMark)!=0)
    {
        OPTIX_CHECK_LOG(optixPipelineDestroy(pipeline));
        pipelineMark = newMark;
    }
    OPTIX_CHECK_LOG( optixPipelineCreate(
                context,
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups.data(),
                num_progs,
                log,
                &sizeof_log,
                &pipeline
                ) );
    pipelineMark = newMark;

    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK( optixUtilAccumulateStackSizes( raygen_prog_group,    &stack_sizes, pipeline ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( radiance_miss_group,  &stack_sizes, pipeline ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( occlusion_miss_group, &stack_sizes, pipeline ) );
    for(int i=0;i<rtMaterialShaders.size();i++)
    {        
        OPTIX_CHECK( optixUtilAccumulateStackSizes( rtMaterialShaders[i].core->m_radiance_hit_group, &stack_sizes, pipeline ) );
        OPTIX_CHECK( optixUtilAccumulateStackSizes( rtMaterialShaders[i].core->m_occlusion_hit_group, &stack_sizes, pipeline ) );
        OPTIX_CHECK( optixUtilAccumulateStackSizes( rtMaterialShaders[i].callable_prg->prog_group, &stack_sizes, pipeline ) );
    }
    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 1;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes(
                &stack_sizes,
                max_trace_depth,
                max_cc_depth,
                max_dc_depth,
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state,
                &continuation_stack_size
                ) );

    const uint32_t max_traversal_depth = tree_depth;
    OPTIX_CHECK( optixPipelineSetStackSize(
                pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_traversal_depth
                ) );
}


}
