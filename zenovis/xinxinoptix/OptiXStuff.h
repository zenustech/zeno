#pragma once

#include <cstdio>
#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <memory>
#include <optix.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

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
#include "optixVolume.h"
#include "optix_types.h"
#include "raiicuda.h"
#include "zeno/types/TextureObject.h"
#include "zeno/utils/log.h"
#include "zeno/utils/string.h"
#include <filesystem>
#define CRYPTOPP_ENABLE_NAMESPACE_WEAK 1
#include <cryptopp/md5.h>
#include <cryptopp/hex.h>

//#include <GLFW/glfw3.h>

#include <tbb/task_group.h>
#include <glm/common.hpp>
#include <glm/matrix.hpp>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <filesystem>

#include "BCX.h"
#include "ies/ies.h"

#include "zeno/utils/fileio.h"
#include "zeno/extra/TempNode.h"
#include "zeno/types/PrimitiveObject.h"
#include "ChiefDesignerEXR.h"
#include <stb_image.h>
#include <cudaMemMarco.hpp>
#include <vector>

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

inline std::vector< std::function<void(void)> > garbageTasks;

inline void resetAll() {

    raygen_prog_group.reset();
    radiance_miss_group.reset();
    occlusion_miss_group.reset();

    raygen_module.reset();

    auto count = garbageTasks.size();
    for (auto& task : garbageTasks) {
        task();
    }
    garbageTasks.clear();

    d_miss_records.reset();
    d_raygen_record.reset();
    d_hitgroup_records.reset();
    d_callable_records.reset();  

    pipeline.reset();
    context.reset();
}

inline bool isPipelineCreated = false;
////end material independent stuffs

inline static auto DefaultCompileOptions() {
    OptixModuleCompileOptions module_compile_options = {};
#if defined( NDEBUG )
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else 
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
#endif
    return module_compile_options;
}

inline void createContext()
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
#if defined( NDEBUG )
    options.logCallbackLevel          = 0;
#else
    options.logCallbackLevel          = 4;
#endif
    options.validationMode            = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );
}

inline uint CachedPrimitiveTypeFlags = UINT_MAX;

inline bool configPipeline(OptixPrimitiveTypeFlags usesPrimitiveTypeFlags) {

    if (CachedPrimitiveTypeFlags != UINT_MAX && (usesPrimitiveTypeFlags&CachedPrimitiveTypeFlags == usesPrimitiveTypeFlags)) { return false; }
    CachedPrimitiveTypeFlags = usesPrimitiveTypeFlags;

    pipeline_compile_options = {};
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY; //OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.usesMotionBlur        = false;
    pipeline_compile_options.numPayloadValues      = 2;
    pipeline_compile_options.numAttributeValues    = 2;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_DEBUG;
    //pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | usesPrimitiveTypeFlags;
    pipeline_compile_options.usesPrimitiveTypeFlags = usesPrimitiveTypeFlags;

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

static std::vector<char> readData(std::string const& filename)
{
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
}

inline bool createModule(OptixModule &module, OptixDeviceContext &context, const char *source, const char *name, const std::vector<std::string>& macros={}, tbb::task_group* _c_group = nullptr)
{
    OptixModuleCompileOptions module_compile_options = OptixUtil::DefaultCompileOptions();
    module_compile_options.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    char log[2048];
    size_t sizeof_log = sizeof( log );

    size_t      inputSize = 0;
    //TODO: the file path problem
    bool is_success=false;

    std::vector<const char*> compilerOptions {
        "-std=c++17", "-default-device" 
        //,"-extra-device-vectorization"
  #if !defined( NDEBUG )      
        ,"-lineinfo" //"-G"//"--dopt=on",
  #endif
        // "--gpu-architecture=compute_60",
        ,"--relocatable-device-code=true"
        // "--extensible-whole-program"
        ,"--split-compile=0"
    };

    std::string flat_macros = ""; 

    for (auto &ele : macros) {
        compilerOptions.push_back(ele.c_str());
        flat_macros += ele;
    }

    const char* input = sutil::getCodePTX( source, flat_macros.c_str(), name, inputSize, is_success, nullptr, compilerOptions);

    if(is_success==false)
    {
        return false;
    }

    if (_c_group == nullptr) {
        OPTIX_CHECK(
            optixModuleCreate( context, &module_compile_options, &pipeline_compile_options, input, inputSize, log, &sizeof_log, &module )
        );
    } else {
        
        OptixTask firstTask;
        OPTIX_CHECK(
            optixModuleCreateWithTasks( 
                context, 
                &module_compile_options, 
                &pipeline_compile_options,
                input, 
                inputSize, 
                log, &sizeof_log, 
                &module, 
                &firstTask)
        );

        executeOptixTask(firstTask, *_c_group);
        //COMPILE_WITH_TASKS_CHECK( //);
        _c_group->wait();  
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
        desc.miss.module            = nullptr;  // NULL miss program for occlusion rays
        desc.miss.entryFunctionName = nullptr;
        sizeof_log                  = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context, &desc,
                    1,  // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
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
    std::cout<<kind<<std::endl;
    std::cout<<entry<<std::endl;

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
    return texture;

}
inline std::shared_ptr<cuTexture> makeCudaTexture(float* img, int nx, int ny, int nc)
{
    auto texture = std::make_shared<cuTexture>(nx, ny);
    std::vector<float4> data;
    data.resize(nx*ny);
    for(int j=0;j<ny;j++)
        for(int i=0;i<nx;i++)
        {
            size_t idx = j*nx + i;
            data[idx] = {
                    nc>=1?img[idx*nc + 0]:0,
                    nc>=2?img[idx*nc + 1]:0,
                    nc>=3?img[idx*nc + 2]:0,
                    nc>=4?img[idx*nc + 3]:0,
            };
        }
    cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaError_t rc = cudaMallocArray(&texture->gpuImageArray, &channelDescriptor, nx, ny, 0);
    if (rc != cudaSuccess) {
        std::cout<<"texture space alloc failed\n";
        return 0;
    }
    rc = cudaMemcpy2DToArray(texture->gpuImageArray, 0, 0, data.data(),
                             nx * sizeof(float) * 4,
                             nx * sizeof(float) * 4,
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

inline std::map<std::string, uint> matIDtoShaderIndex;

inline std::map<std::string, std::shared_ptr<VolumeWrapper>> g_vdb_cached_map;
inline std::map<std::string, std::pair<uint, uint>> g_vdb_indice_visible;

inline std::map<uint, std::vector<std::string>> g_vdb_list_for_each_shader;

inline std::vector<std::tuple<std::string, glm::mat4>> volumeTrans;
inline std::vector<std::tuple<std::string, std::shared_ptr<VolumeWrapper>>> volumeBoxs;

inline bool preloadVolumeBox(std::string& key, std::string& matid, glm::mat4& transform) {

    volumeTrans.push_back( {matid, transform} );
    return true;
}

inline bool processVolumeBox() {

    volumeBoxs.clear();
    for (auto& [key, val] : volumeTrans) {
        auto volume_ptr = std::make_shared<VolumeWrapper>();
        volume_ptr->transform = val;
        buildVolumeAccel(volume_ptr->accel, *volume_ptr, context);
        volumeBoxs.emplace_back( std::tuple{ key, volume_ptr } );
    }
    volumeTrans.clear();
    return true;
}

inline bool preloadVDB(const zeno::TextureObjectVDB& texVDB, 
                       uint index_of_shader, uint index_inside_shader,
                       const glm::mat4& transform, 
                       std::string& combined_key)
{
    auto path = texVDB.path;
    auto channel = texVDB.channel;

    std::filesystem::path filePath = path;

    if ( !std::filesystem::exists(filePath) ) {
        std::cout << filePath.string() << " doesn't exist" << std::endl;
        return false;
    }

    auto fileTime = std::filesystem::last_write_time(filePath);
    // std::filesystem::file_time_type::duration ft = fileTime.time_since_epoch();
    // if (filePath.extension() != ".vdb")
    // {
    //     std::cout << filePath.filename() << " doesn't exist";
    //     return false;
    // }

        auto isNumber = [] (const std::string& s)
        {
            for (char const &ch : s) {
                if (std::isdigit(ch) == 0)
                    return false;
            }
            return true;
        };

    if ( isNumber(channel) ) {
        auto channel_index = (uint)std::stoi(channel);
        channel = fetchGridName(path, channel_index);
    } else {
        checkGridName(path, channel);
    }

    const auto vdb_key = path + "{" + channel + "}";
    combined_key = vdb_key;

    zeno::log_debug("loading VDB :{}", path);

    if (g_vdb_cached_map.count(vdb_key)) {

        auto& cached = g_vdb_cached_map[vdb_key];

        if (transform == cached->transform && fileTime == cached->file_time && texVDB.eleType == cached->type) {

            g_vdb_indice_visible[vdb_key] = std::make_pair(index_of_shader, index_inside_shader);
            return true;
        } else {
            cleanupVolume(*g_vdb_cached_map[vdb_key]);
        }
    }

    auto volume_ptr = std::make_shared<VolumeWrapper>();
    volume_ptr->file_time = fileTime;
    volume_ptr->transform = transform;
    volume_ptr->selected = {channel};
    volume_ptr->type = texVDB.eleType;
    
    auto succ = loadVolume(*volume_ptr, path); 
    
    if (!succ) {return false;}

    g_vdb_cached_map[vdb_key] = volume_ptr;
    g_vdb_indice_visible[vdb_key] = std::make_pair(index_of_shader, index_inside_shader);

    return true;
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

inline std::map<TexKey, std::shared_ptr<cuTexture>> tex_lut;
inline std::map<std::string, std::filesystem::file_time_type> g_tex_last_write_time;
inline std::map<std::string, std::string> md5_path_mapping;
inline std::optional<std::string> sky_tex;
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
            illum = abs(illum) * sinf(3.1415926f*((float)jj + 0.5f)/(float)ny);

            sky_cdf[idx] += illum + (idx>0? sky_cdf[idx-1]:0);
        }
    }
    float total_illum = sky_cdf[sky_cdf.size()-1];
    sky_avg = total_illum / ((float)nx * (float)ny);
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

template<typename TaskType=void>
inline void addTexture(std::string path, bool blockCompression=false, TaskType* task=nullptr)
{
    std::string native_path = std::filesystem::u8path(path).string();

    TexKey tex_key {path, blockCompression}; 

    if (tex_lut.count(tex_key)) {
        return; // do nothing
    }

    zeno::log_debug("loading texture :{}", path);

    bool should_reload = false;
    if (std::filesystem::exists(native_path)) {
        std::filesystem::file_time_type ftime = std::filesystem::last_write_time(native_path);

        if(g_tex_last_write_time[path] != ftime) {
            should_reload = true;
        }
        g_tex_last_write_time[path] = ftime;
    } else {
        zeno::log_info("file {} doesn't exist", path);
        return;
    }
    
    auto input = readData(native_path);
    std::string md5Hash = calculateMD5(input);

    if ( md5_path_mapping.count(md5Hash) && !should_reload) {

        auto& alt_path = md5_path_mapping[md5Hash];
        auto alt_key = TexKey { alt_path, blockCompression };

        if (tex_lut.count(alt_key)) {

            tex_lut[tex_key] = tex_lut[alt_key];
            zeno::log_info("path {} reuse {} tex", path, alt_path);
            return;
        }
    }
    else {
        md5_path_mapping[md5Hash] = path;
    }

    int nx, ny, nc;
    stbi_set_flip_vertically_on_load(true);

    std::function<float(uint32_t)> lookupTexture = [](uint32_t x) {return 0.0f;};
    std::function<void(void)>     cleanupTexture = [](){};

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

        tex_lut[tex_key] = makeCudaTexture(rgba, nx, ny, nc);

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

        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &iesBuffer.reset() ), data_length) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)iesBuffer ), iesd.data(), data_length, cudaMemcpyHostToDevice ) );
        
        g_ies[path] = {std::move(iesBuffer), coneAngle };
    }
    else if (zeno::getSession().nodeClasses.count("ReadPNG16") > 0 && zeno::ends_with(path, ".png", false)) {
        auto outs = zeno::TempNodeSimpleCaller("ReadPNG16")
                .set2("path", path)
                .call();

        // Create nodes
        auto img = outs.get<zeno::PrimitiveObject>("image");
        if (img->verts.size() == 0) {
            tex_lut[tex_key] = std::make_shared<cuTexture>();
            return;
        }
        nx = std::max(img->userData().get2<int>("w"), 1);
        ny = std::max(img->userData().get2<int>("h"), 1);
        nc = std::max(img->userData().get2<int>("channels"), 1);

        auto ucdata = std::make_shared<std::vector<unsigned char>>(img->verts.size() * nc);

        if (nc < 4) {

            for(size_t i=0; i<img->verts.size(); i+=1 ) {

                for (int c=0; c<nc; ++c) {
                    ucdata->at(i*nc+c) = (img->verts[i][c] * 255.0);
                }
            }
            tex_lut[tex_key] = makeCudaTexture(ucdata->data(), nx, ny, nc, blockCompression);

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
            tex_lut[tex_key] = makeCudaTexture((unsigned char *)data, nx, ny, 4, blockCompression);
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
            tex_lut[tex_key] = std::make_shared<cuTexture>();
            return;
        }
        nx = std::max(nx, 1);
        ny = std::max(ny, 1);
        assert(img);
        
        tex_lut[tex_key] = makeCudaTexture(img, nx, ny, nc);

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
            tex_lut[tex_key] = std::make_shared<cuTexture>();
            return;
        }
        nx = std::max(nx, 1);
        ny = std::max(ny, 1);
        
        tex_lut[tex_key] = makeCudaTexture(img, nx, ny, nc, blockCompression);

        lookupTexture = [img](uint32_t idx) {
            return (float)img[idx] / 255;
        };
        cleanupTexture = [img]() {
            stbi_image_free(img);
        };
    }
    tex_lut[tex_key]->md5 = md5Hash;

    if constexpr (!detail::is_void<TaskType>::value) {
        if (task != nullptr) {
            (*task)(tex_lut[tex_key].get(), nx, ny, nc, lookupTexture);
        }
    }

    cleanupTexture();
}
inline void removeTexture(const TexKey &key) {

    auto& path = key.path;

    if (path.size()) {
        if (tex_lut.count(key)) {
            zeno::log_info("removeTexture: {} blockCompresssion: {}", path, key.blockCompression);
            md5_path_mapping.erase(tex_lut[key]->md5);
        }
        else {
            zeno::log_error("removeTexture: {} not exists!", path);
        }
        tex_lut.erase(key);
        g_tex_last_write_time.erase(path);
    }
}

inline void addSkyTexture(std::string path) {
    
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
            std::cout<<"module created"<<std::endl;

            m_radiance_hit_group.reset();
            m_occlusion_hit_group.reset();

            createRTProgramGroups(context, module, 
                "OPTIX_PROGRAM_GROUP_KIND_CLOSEHITGROUP", 
                _shadingEntry, _hittingEntry, moduleIS, m_radiance_hit_group);

            createRTProgramGroups(context, module, 
                "OPTIX_PROGRAM_GROUP_KIND_ANYHITGROUP", 
                _occlusionEntry, _hittingEntry, moduleIS, m_occlusion_hit_group);

            //_c_group.wait();
            return true;
        }
        return false;
    }
};

struct OptixShaderWrapper
{
    std::shared_ptr<OptixShaderCore> core{};
    
    std::string                 callable {};
    raii<OptixModule>           callable_module {};
    raii<OptixProgramGroup> callable_prog_group {};
   
    std::map<int, TexKey>                m_texs {};
    bool                                has_vdb {};
    std::string                       parameters{};

    OptixShaderWrapper() = default;
    ~OptixShaderWrapper() = default;

    OptixShaderWrapper(OptixShaderWrapper&& ref) = default;
    
    OptixShaderWrapper(std::shared_ptr<OptixShaderCore> _core_, const std::string& callableSource) 
    {
        core = _core_; callable = callableSource;
    } 

    bool loadProgram(uint idx, bool fallback=false, tbb::task_group* _c_group = nullptr)
    {
        std::string tmp_name = "Callable.cu";
        tmp_name = "$" + std::to_string(idx) + tmp_name;

        std::vector<std::string> macros {};

        if (fallback) {
            macros.push_back("--define-macro=_FALLBACK_"); 
        }

        auto callable_done = createModule(callable_module.reset(), context, callable.c_str(), tmp_name.c_str(), macros); 
        if (callable_done) {

            // Callable programs
            OptixProgramGroupOptions callable_prog_group_options  = {};
            OptixProgramGroupDesc    callable_prog_group_descs[1] = {};

            callable_prog_group_descs[0].kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            callable_prog_group_descs[0].callables.moduleDC            = callable_module;
            callable_prog_group_descs[0].callables.entryFunctionNameDC = "__direct_callable__evalmat";

            char LOG[2048];
            size_t LOG_SIZE = sizeof( LOG );

            OPTIX_CHECK( 
                optixProgramGroupCreate( context, callable_prog_group_descs, 1, &callable_prog_group_options, LOG, &LOG_SIZE, &callable_prog_group.reset())
            );
            return true;
        }

        return false;
    }

    void clearTextureRecords()
    {
        m_texs.clear();
    }
    void addTexture(int i, TexKey key)
    {
        m_texs[i] = key;
    }
    cudaTextureObject_t getTexture(int i)
    {
        if(m_texs.find(i)!=m_texs.end())
        {
            if(tex_lut.find(m_texs[i])!=tex_lut.end())
            {
                return tex_lut[m_texs[i]]->texture;
            }
            return 0;
        }
        return 0;
    }
};

inline std::vector<OptixShaderWrapper> rtMaterialShaders;//just have an arry of shaders

inline void createPipeline()
{
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

        program_groups[3 + 2 * rtMaterialShaders.size() + i] = rtMaterialShaders[i].callable_prog_group;
    }
    char   log[2048];
    size_t sizeof_log = sizeof( log );

    if (isPipelineCreated)
    {
        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        isPipelineCreated = false;
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
    isPipelineCreated = true;

    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK( optixUtilAccumulateStackSizes( raygen_prog_group,    &stack_sizes, pipeline ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( radiance_miss_group,  &stack_sizes, pipeline ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( occlusion_miss_group, &stack_sizes, pipeline ) );
    for(int i=0;i<rtMaterialShaders.size();i++)
    {        
        OPTIX_CHECK( optixUtilAccumulateStackSizes( rtMaterialShaders[i].core->m_radiance_hit_group, &stack_sizes, pipeline ) );
        OPTIX_CHECK( optixUtilAccumulateStackSizes( rtMaterialShaders[i].core->m_occlusion_hit_group, &stack_sizes, pipeline ) );
        OPTIX_CHECK( optixUtilAccumulateStackSizes( rtMaterialShaders[i].callable_prog_group, &stack_sizes, pipeline ) );
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

    const uint32_t max_traversal_depth = 3;
    OPTIX_CHECK( optixPipelineSetStackSize(
                pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_traversal_depth
                ) );
}


template <typename T = char>
class CuBuffer
{
  public:
    CuBuffer( size_t count = 0 ) { alloc( count ); }
    ~CuBuffer() { free(); }
    void alloc( size_t count )
    {
        free();
        m_allocCount = m_count = count;
        if( m_count )
        {
            CUDA_CHECK( cudaMalloc( &m_ptr, m_allocCount * sizeof( T ) ) );
        }
    }
    void allocIfRequired( size_t count )
    {
        if( count <= m_allocCount )
        {
            m_count = count;
            return;
        }
        alloc( count );
    }
    CUdeviceptr get() const { return reinterpret_cast<CUdeviceptr>( m_ptr ); }
    CUdeviceptr get( size_t index ) const { return reinterpret_cast<CUdeviceptr>( m_ptr + index ); }
    void        free()
    {
        m_count      = 0;
        m_allocCount = 0;
        CUDA_CHECK( cudaFree( m_ptr ) );
        m_ptr = nullptr;
    }
    CUdeviceptr release()
    {
        m_count             = 0;
        m_allocCount        = 0;
        CUdeviceptr current = reinterpret_cast<CUdeviceptr>( m_ptr );
        m_ptr               = nullptr;
        return current;
    }
    void upload( const T* data )
    {
        CUDA_CHECK( cudaMemcpy( m_ptr, data, m_count * sizeof( T ), cudaMemcpyHostToDevice ) );
    }

    void download( T* data ) const
    {
        CUDA_CHECK( cudaMemcpy( data, m_ptr, m_count * sizeof( T ), cudaMemcpyDeviceToHost ) );
    }
    void downloadSub( size_t count, size_t offset, T* data ) const
    {
        assert( count + offset <= m_allocCount );
        CUDA_CHECK( cudaMemcpy( data, m_ptr + offset, count * sizeof( T ), cudaMemcpyDeviceToHost ) );
    }
    size_t count() const { return m_count; }
    size_t reservedCount() const { return m_allocCount; }
    size_t byteSize() const { return m_allocCount * sizeof( T ); }

  private:
    size_t m_count      = 0;
    size_t m_allocCount = 0;
    T*     m_ptr        = nullptr;
};


}
