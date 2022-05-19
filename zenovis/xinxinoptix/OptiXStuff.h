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
inline raii<OptixModule> createModule(OptixDeviceContext &context, const char *source, const char *location)
{
    raii<OptixModule> m;
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    

    char log[2048];
    size_t sizeof_log = sizeof( log );


    size_t      inputSize = 0;
    //TODO: the file path problem
    const char* input     = sutil::getInputData( nullptr, nullptr, source, location, inputSize );
    
    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        input,
        inputSize,
        log,
        &sizeof_log,
        &m
    ) );
    return m;
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
        desc.hitgroup.moduleCH            = nullptr;
        desc.hitgroup.entryFunctionNameCH = nullptr;
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
                    &oGroup
                    ) );
    }
}
struct rtMatShader
{
    raii<OptixModule>                    m_ptx_module             ;
    
    //the below two things are just like vertex shader and frag shader in real time rendering
    //the two are linked to codes modeling the rayHit and occlusion test of an particular "Material"
    //of an Object.
    raii<OptixProgramGroup>              m_radiance_hit_group     ;
    raii<OptixProgramGroup>              m_occlusion_hit_group    ;
    std::string                    m_shaderFile                ;
    std::string                    m_shadingEntry              ;
    std::string                    m_occlusionEntry            ;
    rtMatShader() {}
    rtMatShader(const char *shaderFile, std::string shadingEntry, std::string occlusionEntry)
    {
        m_shaderFile = std::move(shaderFile);
        m_shadingEntry = std::move(shadingEntry);
        m_occlusionEntry = std::move(occlusionEntry);
    }


    void loadProgram()
    {
        m_ptx_module = createModule(context, m_shaderFile.c_str(), "tmpshader.cu");
        createRTProgramGroups(context, m_ptx_module, 
        "OPTIX_PROGRAM_GROUP_KIND_CLOSEHITGROUP", 
        m_shadingEntry, m_radiance_hit_group);

        createRTProgramGroups(context, m_ptx_module, 
        "OPTIX_PROGRAM_GROUP_KIND_ANYHITGROUP", 
        m_occlusionEntry, m_occlusion_hit_group);

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


}
