#pragma once

#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <memory>
#include <optix.h>
#include <optix_function_table_definition.h>
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
#include "raiicuda.h"
#include "zeno/types/TextureObject.h"
#include "zeno/utils/string.h"
#include "tinyexr.h"
#include <filesystem>

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
#include "ies/ies_loader.h"
#include "zeno/utils/fileio.h"
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
inline OptixPipelineCompileOptions        pipeline_compile_options {};
inline raii<OptixPipeline>                  pipeline                 ;
inline raii<OptixModule>                    ray_module               ;
inline raii<OptixModule>                    sphere_module            ;
inline raii<OptixProgramGroup>              raygen_prog_group        ;
inline raii<OptixProgramGroup>              radiance_miss_group      ;
inline raii<OptixProgramGroup>              occlusion_miss_group     ;
inline bool isPipelineCreated = false;
////end material independent stuffs
inline void createContext()
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    options.validationMode            = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );
    pipeline_compile_options = {};
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY; //OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.usesMotionBlur        = false;
    pipeline_compile_options.numPayloadValues      = 2;
    pipeline_compile_options.numAttributeValues    = 2;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_DEBUG;
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    OptixModuleCompileOptions module_compile_options = {};
    #if defined( NDEBUG )
        module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    #else 
        module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    #endif

    OptixBuiltinISOptions builtin_is_options {};

    builtin_is_options.usesMotionBlur      = false;
    builtin_is_options.buildFlags          = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
    builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OPTIX_CHECK( optixBuiltinISModuleGet( context, &module_compile_options, &pipeline_compile_options,
                            &builtin_is_options, &sphere_module ) );
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

    for( unsigned int i = 0; i < numAdditionalTasksCreated; ++i )
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

inline bool createModule(OptixModule &m, OptixDeviceContext &context, const char *source, const char *location, tbb::task_group* _c_group = nullptr)
{
    //OptixModule m;
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#if defined( NDEBUG )
    module_compile_options.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
    module_compile_options.optLevel          = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

#endif

    char log[2048];
    size_t sizeof_log = sizeof( log );

    size_t      inputSize = 0;
    //TODO: the file path problem
    bool is_success=false;

    const std::vector<const char*> compilerOptions {
        "-std=c++17", "-default-device", //"-extra-device-vectorization"
  #if !defined( NDEBUG )      
        //"-lineinfo", "-G"//"--dopt=on",
  #endif
        //"--gpu-architecture=compute_60",
        //"--relocatable-device-code=true"
        //"--extensible-whole-program"
    };

    const char* input = sutil::getInputData( nullptr, nullptr, source, location, inputSize, is_success, nullptr, compilerOptions);

    if(is_success==false)
    {
        return false;
    }

    if (_c_group == nullptr) {

        OPTIX_CHECK(
            optixModuleCreateFromPTX(context, &module_compile_options, &pipeline_compile_options, input, inputSize, log, &sizeof_log, &m)
        );
    } else {
        
        OptixTask firstTask;
        OPTIX_CHECK(
            optixModuleCreateFromPTXWithTasks( 
                context, 
                &module_compile_options, 
                &pipeline_compile_options,
                input, 
                inputSize, 
                log, &sizeof_log, 
                &m, 
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
                    &raygen_prog_group
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
                    &radiance_miss_group
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
                    &occlusion_miss_group
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


    if(kind == "OPTIX_PROGRAM_GROUP_KIND_CLOSEHITGROUP")
    {
        desc.hitgroup.moduleCH            = _module;
        desc.hitgroup.entryFunctionNameCH = entry.c_str();
    } 
    else if(kind == "OPTIX_PROGRAM_GROUP_KIND_ANYHITGROUP")
    {
        desc.hitgroup.moduleAH           = _module;
        desc.hitgroup.entryFunctionNameAH = entry.c_str();
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
    cudaArray_t gpuImageArray;
    cudaTextureObject_t texture;
    cuTexture(){gpuImageArray = nullptr;texture=0;}
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
inline std::shared_ptr<cuTexture> makeCudaTexture(unsigned char* img, int nx, int ny, int nc)
{
    auto texture = std::make_shared<cuTexture>();
    std::vector<float4> data;
    data.resize(nx*ny);
    for(int j=0;j<ny;j++)
    for(int i=0;i<nx;i++)
    {
        size_t idx = j*nx + i;
        data[idx] = {
            nc>=1?(float)(img[idx*nc + 0])/255.0f:0,
            nc>=2?(float)(img[idx*nc + 1])/255.0f:0,
            nc>=3?(float)(img[idx*nc + 2])/255.0f:0,
            nc>=4?(float)(img[idx*nc + 3])/255.0f:0,
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
inline std::shared_ptr<cuTexture> makeCudaTexture(float* img, int nx, int ny, int nc)
{
    auto texture = std::make_shared<cuTexture>();
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

inline bool preloadVDB(const zeno::TextureObjectVDB& texVDB, 
                       uint index_of_shader, uint index_inside_shader,
                       const glm::f64mat4& transform, 
                       std::string& combined_key)
{
    auto path = texVDB.path;
    auto channel = texVDB.channel;

    std::filesystem::path filePath = path;

    if ( !std::filesystem::exists(filePath) ) {
        std::cout << filePath.string() << " doesn't exist";
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

        if (transform == g_vdb_cached_map[vdb_key]->transform && fileTime == cached->file_time && texVDB.eleType == cached->type) {

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
inline std::vector<float> IES2HDR(const std::string& path)
{
    IESFileInfo info;
    auto IESBuffer = zeno::file_get_binary(path);
    IESBuffer.push_back(0);

    IESLoadHelper IESLoader;
    if (!IESLoader.load(IESBuffer.data(), IESBuffer.size() - 1, info)) {
        std::string l = "IESLoader.load";
        throw std::runtime_error(l);
    }

    std::vector<float> img;
    img.resize(256 * 3);

    if (!IESLoader.saveAs1D(info, img.data(), 256, 3)) {
        std::string l = "IESLoader.saveAs1D";
        throw std::runtime_error(l);
    }

    return img;
}
#include <stb_image.h>
inline std::map<std::string, std::shared_ptr<cuTexture>> g_tex;
inline std::map<std::string, std::filesystem::file_time_type> g_tex_last_write_time;
inline std::optional<std::string> sky_tex;
inline std::map<std::string, int> sky_nx_map;
inline std::map<std::string, int> sky_ny_map;


inline std::map<std::string, std::vector<float>> sky_cdf_map;
inline std::map<std::string, std::vector<float>> sky_pdf_map;
inline std::map<std::string, std::vector<int>>   sky_start_map;

template<typename T>
inline void calc_sky_cdf_map(int nx, int ny, int nc, T *img) {
    auto &sky_nx = sky_nx_map[sky_tex.value()];
    auto &sky_ny = sky_ny_map[sky_tex.value()];
    auto &sky_cdf = sky_cdf_map[sky_tex.value()];
    auto &sky_pdf = sky_pdf_map[sky_tex.value()];
    auto &sky_start = sky_start_map[sky_tex.value()];
    sky_nx = nx;
    sky_ny = ny;
    //we need to recompute cdf
    sky_cdf.resize(nx*ny);
    sky_cdf.assign(nx*ny, 0);
    sky_pdf.resize(nx*ny);
    sky_pdf.assign(nx*ny, 0);
    sky_start.resize(nx*ny);
    sky_start.assign(nx*ny, 0);
    std::vector<double> skypdf(nx*ny);
    skypdf.assign(nx*ny,0);
    for(int jj=0; jj<ny;jj++)
    {
        for(int ii=0;ii<nx;ii++)
        {
            size_t idx2 = jj*nx*nc + ii*nc;
            size_t idx = jj*nx + ii;
            float illum = 0.0f;
            auto color = zeno::vec3f(img[idx2+0], img[idx2+1], img[idx2+2]);
            illum = zeno::dot(color, zeno::vec3f(0.33333333f,0.33333333f, 0.33333333f));
            //illum = illum > 0.5? illum : 0.0f;
            illum = abs(illum) * sin(3.1415926f*((float)jj + 0.5f)/(float)ny);

            sky_cdf[idx] += illum + (idx>0? sky_cdf[idx-1]:0);
            skypdf[idx] = illum;
        }
    }
    float total_illum = sky_cdf[sky_cdf.size()-1];
    for(int ii=0;ii<sky_cdf.size();ii++)
    {
        sky_cdf[ii] /= total_illum;
        skypdf[ii] = skypdf[ii] * (double)nx * (double)ny / (double)total_illum;
        sky_pdf[ii] = skypdf[ii];
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

inline void addTexture(std::string path)
{
    zeno::log_debug("loading texture :{}", path);
    std::string native_path = std::filesystem::u8path(path).string();
    if (std::filesystem::exists(native_path)) {
        std::filesystem::file_time_type ftime = std::filesystem::last_write_time(native_path);
        if(g_tex.count(path) && g_tex_last_write_time[path] == ftime) {
            return;
        }
        g_tex_last_write_time[path] = ftime;
    }
    else {
        if(g_tex.count(path)) {
            return;
        }
    }
    int nx, ny, nc;
    stbi_set_flip_vertically_on_load(true);

    if (zeno::ends_with(path, ".exr", false)) {
        float* rgba;
        const char* err;
        int ret = LoadEXR(&rgba, &nx, &ny, native_path.c_str(), &err);
        if (ret != 0) {
            zeno::log_error("load exr: {}", err);
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
        if(sky_tex.value() == path)//if this is a loading of a sky texture
        {
            calc_sky_cdf_map(nx, ny, nc, rgba);
        }
        g_tex[path] = makeCudaTexture(rgba, nx, ny, nc);
        free(rgba);
    }
    else if (zeno::ends_with(path, ".ies", false)) {
        auto img = IES2HDR(path);
        g_tex[path] = makeCudaTexture(img.data(), 256, 1, 3);
    }
    else if (stbi_is_hdr(native_path.c_str())) {
        float *img = stbi_loadf(native_path.c_str(), &nx, &ny, &nc, 0);
        if(!img){
            zeno::log_error("loading texture failed:{}", path);
            g_tex[path] = std::make_shared<cuTexture>();
            return;
        }
        nx = std::max(nx, 1);
        ny = std::max(ny, 1);
        assert(img);
        if(sky_tex.value() == path)//if this is a loading of a sky texture
        {
            calc_sky_cdf_map(nx, ny, nc, img);
        }
        g_tex[path] = makeCudaTexture(img, nx, ny, nc);
        stbi_image_free(img);
    }
    else {
        unsigned char *img = stbi_load(native_path.c_str(), &nx, &ny, &nc, 0);
        if(!img){
            zeno::log_error("loading hdr texture failed:{}", path);
            g_tex[path] = std::make_shared<cuTexture>();
            return;
        }
        nx = std::max(nx, 1);
        ny = std::max(ny, 1);
        assert(img);
        if(sky_tex.value() == path)//if this is a loading of a sky texture
        {
            calc_sky_cdf_map(nx, ny, nc, img);
        }
        g_tex[path] = makeCudaTexture(img, nx, ny, nc);
        stbi_image_free(img);
    }

    for (auto i = g_tex.begin(); i != g_tex.end(); i++) {
        zeno::log_info("-{}", i->first);
    }
}
struct rtMatShader
{
    raii<OptixModule>                    m_ptx_module                ; 
    OptixModule*                         moduleIS = nullptr;
    //the below two things are just like vertex shader and frag shader in real time rendering
    //the two are linked to codes modeling the rayHit and occlusion test of an particular "Material"
    //of an Object.
    raii<OptixProgramGroup>              m_radiance_hit_group        ;
    raii<OptixProgramGroup>              m_occlusion_hit_group       ;
    std::string                          m_shaderFile                ;
    std::string                          m_hittingEntry              ;
    std::string                          m_shadingEntry              ;
    std::string                          m_occlusionEntry            ;
    std::map<int, std::string>           m_texs;
    bool                                 has_vdb{};

    void clearTextureRecords()
    {
        m_texs.clear();
    }
    void addTexture(int i, std::string name)
    {
        m_texs[i] = name;
    }
    cudaTextureObject_t getTexture(int i)
    {
        if(m_texs.find(i)!=m_texs.end())
        {
            if(g_tex.find(m_texs[i])!=g_tex.end())
            {
                return g_tex[m_texs[i]]->texture;
            }
            return 0;
        }
        return 0;
    }
    rtMatShader() {}
    rtMatShader(const char *shaderFile, std::string shadingEntry, std::string occlusionEntry)
    {
        m_shaderFile = shaderFile;
        m_shadingEntry = shadingEntry;
        m_occlusionEntry = occlusionEntry;
    }

    rtMatShader(const char *shaderFile, std::string shadingEntry, std::string occlusionEntry, std::string hittingEntry)
    {
        m_shaderFile = shaderFile;
        m_shadingEntry = shadingEntry;
        m_occlusionEntry = occlusionEntry;

        m_hittingEntry = hittingEntry;
    }

    bool loadProgram(uint idx, tbb::task_group* _c_group = nullptr)
    {
        // try {
        //     createModule(m_ptx_module.reset(), context, m_shaderFile.c_str(), "MatShader.cu");
        //     createRTProgramGroups(context, m_ptx_module, 
        //     "OPTIX_PROGRAM_GROUP_KIND_CLOSEHITGROUP", 
        //     m_shadingEntry, m_radiance_hit_group);

        //     createRTProgramGroups(context, m_ptx_module, 
        //     "OPTIX_PROGRAM_GROUP_KIND_ANYHITGROUP", 
        //     m_occlusionEntry, m_occlusion_hit_group);
        // } catch (sutil::Exception const &e) {
        //     throw std::runtime_error((std::string)"cannot create program group. Log:\n" + e.what() + "\n===BEG===\n" + m_shaderFile + "\n===END===\n");
        // }

        std::string tmp_name = "MatShader.cu";
        tmp_name = "$" + std::to_string(idx) + tmp_name;
        
        if(createModule(m_ptx_module.reset(), context, m_shaderFile.c_str(), tmp_name.c_str(), _c_group))
        {
            std::cout<<"module created"<<std::endl;

            createRTProgramGroups(context, m_ptx_module, 
                "OPTIX_PROGRAM_GROUP_KIND_CLOSEHITGROUP", 
                m_shadingEntry, m_hittingEntry, moduleIS, m_radiance_hit_group);

            createRTProgramGroups(context, m_ptx_module, 
                "OPTIX_PROGRAM_GROUP_KIND_ANYHITGROUP", 
                m_occlusionEntry, m_hittingEntry, moduleIS, m_occlusion_hit_group);

            //_c_group.wait();
            return true;
        }
        return false;

    }

};
inline std::vector<rtMatShader> rtMaterialShaders;//just have an arry of shaders
inline void createPipeline()
{
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = 2;
#if defined( NDEBUG )
    pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
    pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    size_t num_progs = 3 + rtMaterialShaders.size() * 2;
    OptixProgramGroup* program_groups = new OptixProgramGroup[num_progs];
    program_groups[0] = raygen_prog_group;
    program_groups[1] = radiance_miss_group;
    program_groups[2] = occlusion_miss_group;
    for(size_t i=0;i<rtMaterialShaders.size();i++)
    {
        program_groups[3 + i*2] = rtMaterialShaders[i].m_radiance_hit_group;
        program_groups[3 + i*2 + 1] = rtMaterialShaders[i].m_occlusion_hit_group;
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
                program_groups,
                num_progs,
                log,
                &sizeof_log,
                &pipeline
                ) );
    isPipelineCreated = true;

    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK( optixUtilAccumulateStackSizes( raygen_prog_group,    &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( radiance_miss_group,  &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( occlusion_miss_group, &stack_sizes ) );
    for(int i=0;i<rtMaterialShaders.size();i++)
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( rtMaterialShaders[i].m_radiance_hit_group, &stack_sizes ) );
        OPTIX_CHECK( optixUtilAccumulateStackSizes( rtMaterialShaders[i].m_occlusion_hit_group, &stack_sizes ) );
    }
    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
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
    delete[]program_groups;

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
