#pragma once
#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

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
#include <optix_stack_size.h>
#include "raiicuda.h"

//#include <GLFW/glfw3.h>


#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}
namespace OptixUtil
{
    using namespace xinxinoptix;
////these are all material independent stuffs;
inline raii<OptixDeviceContext>             context                  ;
inline OptixPipelineCompileOptions    pipeline_compile_options = {};
inline raii<OptixPipeline>                  pipeline                 ;
inline raii<OptixModule>                    ray_module               ;
inline raii<OptixProgramGroup>              raygen_prog_group        ;
inline raii<OptixProgramGroup>              radiance_miss_group      ;
inline raii<OptixProgramGroup>              occlusion_miss_group     ;
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
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );
    pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur        = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues      = 2;
    pipeline_compile_options.numAttributeValues    = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

}
inline bool createModule(OptixModule &m, OptixDeviceContext &context, const char *source, const char *location)
{
    //OptixModule m;
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    

    char log[2048];
    size_t sizeof_log = sizeof( log );


    size_t      inputSize = 0;
    //TODO: the file path problem
    bool is_success=false;
    const char* input     = sutil::getInputData( nullptr, nullptr, source, location, inputSize, is_success);
    if(is_success==false)
    {
        return false;
    }
    optixModuleCreateFromPTX(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        input,
        inputSize,
        log,
        &sizeof_log,
        &m
    );
    return true;
    //return m;
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
inline void createRTProgramGroups(OptixDeviceContext &context, OptixModule &_module, std::string kind, std::string entry, raii<OptixProgramGroup>& oGroup)
{
    OptixProgramGroupOptions  program_group_options = {};
    char   log[2048];
    size_t sizeof_log = sizeof( log );
    std::cout<<kind<<std::endl;
    std::cout<<entry<<std::endl;
    if(kind == "OPTIX_PROGRAM_GROUP_KIND_CLOSEHITGROUP")
    {
        OptixProgramGroupDesc desc        = {};
        desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH            = _module;
        desc.hitgroup.entryFunctionNameCH = entry.c_str();
        sizeof_log                        = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context,
                    &desc,
                    1,  // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &oGroup.reset()
                    ) );
    } else if(kind == "OPTIX_PROGRAM_GROUP_KIND_ANYHITGROUP")
    {
        OptixProgramGroupDesc desc        = {};
        desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleAH            = _module;
        desc.hitgroup.entryFunctionNameAH = entry.c_str();
        sizeof_log                        = sizeof( log );
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
    cudaError_t rc = cudaMallocArray(&texture->gpuImageArray, &channelDescriptor, nx, ny);
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
#include <stb_image.h>
inline std::map<std::string, std::shared_ptr<cuTexture>> g_tex;
inline void addTexture(std::string path)
{
    std::cout<<"loading texture "<<path<<std::endl;
    if(g_tex.find(path)!=g_tex.end())
    {
        return;
    }
    int nx, ny, nc;
    stbi_set_flip_vertically_on_load(true);
    unsigned char *img = stbi_load(path.c_str(), &nx, &ny, &nc, 0);
    if(!img){
        std::cout<<"load texture "<<path.c_str()<<" failed\n";
        g_tex[path] = std::make_shared<cuTexture>();
        return;
    }
    nx = std::max(nx, 1);
    ny = std::max(ny, 1);
    assert(img);
    g_tex[path] = makeCudaTexture(img, nx, ny, nc);
    stbi_image_free(img);
}
struct rtMatShader
{
    raii<OptixModule>                    m_ptx_module             ;
    
    //the below two things are just like vertex shader and frag shader in real time rendering
    //the two are linked to codes modeling the rayHit and occlusion test of an particular "Material"
    //of an Object.
    raii<OptixProgramGroup>              m_radiance_hit_group        ;
    raii<OptixProgramGroup>              m_occlusion_hit_group       ;
    std::string                          m_shaderFile                ;
    std::string                          m_shadingEntry              ;
    std::string                          m_occlusionEntry            ;
    std::map<int, std::string>           m_texs;
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


    bool loadProgram()
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
        
        if(createModule(m_ptx_module.reset(), context, m_shaderFile.c_str(), "MatShader.cu"))
        {
            std::cout<<"module created"<<std::endl;
            createRTProgramGroups(context, m_ptx_module, 
            "OPTIX_PROGRAM_GROUP_KIND_CLOSEHITGROUP", 
            m_shadingEntry, m_radiance_hit_group);

            createRTProgramGroups(context, m_ptx_module, 
            "OPTIX_PROGRAM_GROUP_KIND_ANYHITGROUP", 
            m_occlusionEntry, m_occlusion_hit_group);
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
    pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    int num_progs = 3 + rtMaterialShaders.size() * 2;
    OptixProgramGroup* program_groups = new OptixProgramGroup[num_progs];
    program_groups[0] = raygen_prog_group;
    program_groups[1] = radiance_miss_group;
    program_groups[2] = occlusion_miss_group;
    for(int i=0;i<rtMaterialShaders.size();i++)
    {
        program_groups[3 + i*2] = rtMaterialShaders[i].m_radiance_hit_group;
        program_groups[3 + i*2 + 1] = rtMaterialShaders[i].m_occlusion_hit_group;
    }
    char   log[2048];
    size_t sizeof_log = sizeof( log );
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

    const uint32_t max_traversal_depth = 1;
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
