//
// this part of code is modified from nvidia's optix example


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
#include <sutil/Scene.h>
#include <optix_stack_size.h>

//#include <GLFW/glfw3.h>

#include "optixPathTracer.h"

#include <zeno/utils/log.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/types/MaterialObject.h>
#include <array>
#include <optional>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include "xinxinoptixapi.h"
#include "OptiXStuff.h"
#include <zeno/utils/vec.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace xinxinoptix {


bool resize_dirty = false;
bool minimized    = false;
static bool using20xx = false;

// Camera state
bool             camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// Mouse state
int32_t mouse_button = -1;

//int32_t samples_per_launch = 16;
int32_t samples_per_launch = 1;

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;


//struct Vertex
//{
    //float x, y, z, pad;
//};


//struct IndexedTriangle
//{
    //uint32_t v1, v2, v3, pad;
//};


//struct Instance
//{
    //float transform[12];
//};


using Vertex = float4;
std::vector<Vertex> g_lightMesh;
std::vector<Vertex> g_lightColor;
struct PathTracerState
{
    OptixDeviceContext context = 0;
    OptixTraversableHandle         m_ias_handle;
    OptixTraversableHandle         gas_handle               = {};  // Traversable handle for triangle AS
    raii<CUdeviceptr>              d_gas_output_buffer;  // Triangle AS memory
    raii<CUdeviceptr>              m_d_ias_output_buffer;
    raii<CUdeviceptr>              d_vertices;
    raii<CUdeviceptr>              d_clr;
    raii<CUdeviceptr>              d_nrm;
    raii<CUdeviceptr>              d_uv;
    raii<CUdeviceptr>              d_tan;
    raii<CUdeviceptr>              d_lightMark;
    raii<CUdeviceptr>              d_mat_indices             ;

    raii<OptixModule>              ptx_module;
    raii<OptixModule>              ptx_module2;
    OptixPipelineCompileOptions    pipeline_compile_options;
    OptixPipeline                  pipeline;

    OptixProgramGroup              raygen_prog_group;
    OptixProgramGroup              radiance_miss_group;
    OptixProgramGroup              occlusion_miss_group;
    OptixProgramGroup              radiance_hit_group;
    OptixProgramGroup              occlusion_hit_group;
    OptixProgramGroup              radiance_hit_group2;
    OptixProgramGroup              occlusion_hit_group2;

    raii<CUstream>                       stream;
    raii<CUdeviceptr> accum_buffer_p;
    raii<CUdeviceptr> lightsbuf_p;
    Params                         params;
    raii<CUdeviceptr>                        d_params;

    raii<CUdeviceptr>  d_raygen_record;
    raii<CUdeviceptr>d_miss_records;
    raii<CUdeviceptr>  d_hitgroup_records;

    OptixShaderBindingTable        sbt                      = {};
};

PathTracerState state;

struct smallMesh{
    std::vector<Vertex> verts;
    std::vector<uint32_t> mat_idx;
    std::vector<uint3>     idx;
    OptixTraversableHandle            gas_handle = 0;
    raii<CUdeviceptr>        d_gas_output_buffer;
    raii<CUdeviceptr>                     dverts;
    raii<CUdeviceptr>                      dmats;
    raii<CUdeviceptr>                       didx;
    smallMesh(){idx.resize(0);verts.resize(0);mat_idx.resize(0);}
    ~smallMesh(){d_gas_output_buffer.reset(); dverts.reset(); dmats.reset();}
};

//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

//const int32_t TRIANGLE_COUNT = 32;
//const int32_t MAT_COUNT      = 5;

static std::vector<Vertex> g_vertices= // TRIANGLE_COUNT*3
{
    {0,0,0},
    {0,0,0},
    {0,0,0},
};
static std::vector<Vertex> g_clr= // TRIANGLE_COUNT*3
{
    {0,0,0},
    {0,0,0},
    {0,0,0},
};
static std::vector<Vertex> g_nrm= // TRIANGLE_COUNT*3
{
    {0,0,0},
    {0,0,0},
    {0,0,0},
};
static std::vector<Vertex> g_uv= // TRIANGLE_COUNT*3
{
    {0,0,0},
    {0,0,0},
    {0,0,0},
};
static std::vector<Vertex> g_tan= // TRIANGLE_COUNT*3
{
    {0,0,0},
    {0,0,0},
    {0,0,0},
};
static std::vector<uint32_t> g_mat_indices= // TRIANGLE_COUNT
{
    0,0,0,
};
static std::vector<uint16_t> g_lightMark = //TRIANGLE_COUNT
{
    0
};
static std::vector<ParallelogramLight> g_lights={
    ParallelogramLight{
        /*corner=*/
        /*v1=*/
        /*v2=*/
        /*normal=*/
        /*emission=*/
    },
};
/*
static std::vector<float3> g_emission_colors= // MAT_COUNT
{
    {0,0,0},
};
static std::vector<float3> g_diffuse_colors= // MAT_COUNT
{
    {0.8f,0.8f,0.8f},
};
*/
std::map<std::string, int> g_mtlidlut; // MAT_COUNT


//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

//static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
//{
    //double xpos, ypos;
    //glfwGetCursorPos( window, &xpos, &ypos );

    //if( action == GLFW_PRESS )
    //{
        //mouse_button = button;
        //trackball.startTracking( static_cast<int>( xpos ), static_cast<int>( ypos ) );
    //}
    //else
    //{
        //mouse_button = -1;
    //}
//}


//static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
//{
    //Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );

    //if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    //{
        //trackball.setViewMode( sutil::Trackball::LookAtFixed );
        //trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        //camera_changed = true;
    //}
    //else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    //{
        //trackball.setViewMode( sutil::Trackball::EyeFixed );
        //trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        //camera_changed = true;
    //}
//}


//static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
//{
    //// Keep rendering at the current resolution when the window is minimized.
    //if( minimized )
        //return;

    //// Output dimensions must be at least 1 in both x and y.
    //sutil::ensureMinimumSize( res_x, res_y );

    //Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );
    //params->width  = res_x;
    //params->height = res_y;
    //camera_changed = true;
    //resize_dirty   = true;
//}


//static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
//{
    //minimized = ( iconified > 0 );
//}


//static void keyCallback( GLFWwindow* window, int32_t key, int32_t [>scancode*/, int32_t action, int32_t /*mods<] )
//{
    //if( action == GLFW_PRESS )
    //{
        //if( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
        //{
            //glfwSetWindowShouldClose( window, true );
        //}
    //}
    //else if( key == GLFW_KEY_G )
    //{
        //// toggle UI draw
    //}
//}


//static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
//{
    //if( trackball.wheelEvent( (int)yscroll ) )
        //camera_changed = true;
//}


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

static void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}


static void initLaunchParams( PathTracerState& state )
{
//#ifdef USING_20XX
    if (using20xx) {
    state.params.handle         = state.gas_handle;
//#else
    } else {
    state.params.handle         = state.m_ias_handle;
    }
//#endif
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &state.accum_buffer_p.reset() ),
                state.params.width * state.params.height * sizeof( float4 )
                ) );
    
    state.params.accum_buffer = (float4*)(CUdeviceptr)state.accum_buffer_p;
    
    state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    state.params.samples_per_launch = samples_per_launch;
    state.params.subframe_index     = 0u;
}


static void handleCameraUpdate( Params& params )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    //params.vp1 = cam_vp1;
    //params.vp2 = cam_vp2;
    //params.vp3 = cam_vp3;
    //params.vp4 = cam_vp4;
    camera.setAspectRatio( static_cast<float>( params.width ) / static_cast<float>( params.height ) );
    //params.eye = camera.eye();
    //camera.UVWFrame( params.U, params.V, params.W );
}


static void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &state.accum_buffer_p .reset()),
                params.width * params.height * sizeof( float4 )
                ) );
    state.params.accum_buffer = (float4*)(CUdeviceptr)state.accum_buffer_p;
    state.params.subframe_index = 0;
}


static void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        params.subframe_index = 0;

    handleCameraUpdate( params );
    handleResize( output_buffer, params );
}


static void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, PathTracerState& state )
{
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer  = result_buffer_data;
    state.params.num_lights = g_lights.size();
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.d_params ),
                &state.params, sizeof( Params ),
                cudaMemcpyHostToDevice
                ) );
    OPTIX_CHECK( optixLaunch(
                state.pipeline,
                0,
                (CUdeviceptr)state.d_params,
                sizeof( Params ),
                &state.sbt,
                state.params.width,   // launch width
                state.params.height,  // launch height
                1                     // launch depth
                ) );   
    output_buffer.unmap();
    //CUDA_SYNC_CHECK();
}


static void displaySubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, PathTracerState& state , int fbo = 0 )
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    framebuf_res_x = (int)state.params.width;
    framebuf_res_y = (int)state.params.height;
    //glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO(),
            fbo);
}





static void initCameraState()
{
    camera.setEye( make_float3( 278.0f, 273.0f, -900.0f ) );
    camera.setLookat( make_float3( 278.0f, 273.0f, 330.0f ) );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 35.0f );
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame(
            make_float3( 1.0f, 0.0f, 0.0f ),
            make_float3( 0.0f, 0.0f, 1.0f ),
            make_float3( 0.0f, 1.0f, 0.0f )
            );
    trackball.setGimbalLock( true );
}

static void buildMeshAccelSplitMesh( PathTracerState& state, std::shared_ptr<smallMesh> mesh)
{
    //
    // copy mesh data to device
    //
    const size_t vertices_size_in_bytes = mesh->verts.size() * sizeof( Vertex );
    //CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_vertices ) ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &mesh->dverts.reset() ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)mesh->dverts ),
                mesh->verts.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    const size_t mat_indices_size_in_bytes = mesh->mat_idx.size() * sizeof( uint32_t );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &mesh->dmats.reset() ), mat_indices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)mesh->dmats ),
                mesh->mat_idx.data(),
                mat_indices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    const size_t idx_indices_size_in_bytes = mesh->idx.size() * sizeof(uint3);
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &mesh->didx.reset() ), idx_indices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)mesh->didx ),
                mesh->idx.data(),
                idx_indices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    // // Build triangle GAS // // One per SBT record for this build input
    std::vector<uint32_t> triangle_input_flags(//MAT_COUNT
        g_mtlidlut.size(),
        OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);

    OptixBuildInput triangle_input                           = {};
    triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes         = sizeof( Vertex );
    triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>( mesh->verts.size() );
    triangle_input.triangleArray.vertexBuffers               = g_vertices.empty() ? nullptr : & state.d_vertices;
    triangle_input.triangleArray.flags                       = triangle_input_flags.data();
    triangle_input.triangleArray.numSbtRecords               = g_vertices.empty() ? 1 : g_mtlidlut.size();
    triangle_input.triangleArray.sbtIndexOffsetBuffer        = mesh->dmats;
    triangle_input.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes          = sizeof(unsigned int)*3;
    triangle_input.triangleArray.numIndexTriplets            = mesh->idx.size();
    triangle_input.triangleArray.indexBuffer                 = mesh->didx;

    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    //char   log[2048]; size_t sizeof_log = sizeof( log );
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                state.context,
                &accel_options,
                &triangle_input,
                1,  // num_build_inputs
                &gas_buffer_sizes
                ) );

    raii<CUdeviceptr> d_temp_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer.reset(), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    raii<CUdeviceptr> d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK(cudaMalloc((void**)&d_buffer_temp_output_gas_and_compacted_size.reset(), compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result             = ( CUdeviceptr )( (char*)(CUdeviceptr)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild(
                state.context,
                0,                                  // CUDA stream
                &accel_options,
                &triangle_input,
                1,                                  // num build inputs
                d_temp_buffer,
                gas_buffer_sizes.tempSizeInBytes,
                d_buffer_temp_output_gas_and_compacted_size,
                gas_buffer_sizes.outputSizeInBytes,
                &mesh->gas_handle,
                &emitProperty,                      // emitted property list
                1                                   // num emitted properties
                ) );

    d_temp_buffer.reset();
    //d_mat_indices.reset();

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK(cudaMalloc((void**)&mesh->d_gas_output_buffer.reset(), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, 
        mesh->gas_handle, mesh->d_gas_output_buffer, compacted_gas_size, &mesh->gas_handle ) );

        d_buffer_temp_output_gas_and_compacted_size.reset();
    }
    else
    {
        mesh->d_gas_output_buffer = std::move(d_buffer_temp_output_gas_and_compacted_size);
    }
    state.gas_handle = mesh->gas_handle;
}
static void buildInstanceAccel(PathTracerState& state, int rayTypeCount, std::vector<std::shared_ptr<smallMesh>> m_meshes)
{
    float mat4x4[12] = {1,0,0,0,0,1,0,0,0,0,1,0};
    const size_t num_instances = m_meshes.size();
    std::vector<OptixInstance> optix_instances( num_instances );
    unsigned int sbt_offset = 0;
    for( size_t i = 0; i < m_meshes.size(); ++i )
    {
        auto  mesh = m_meshes[i];
        auto& optix_instance = optix_instances[i];
        memset( &optix_instance, 0, sizeof( OptixInstance ) );

        optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId        = static_cast<unsigned int>( i );
        //optix_instance.sbtOffset         = sbt_offset;
        optix_instance.visibilityMask    = 1;
        optix_instance.traversableHandle = mesh->gas_handle;
        memcpy( optix_instance.transform, mat4x4, sizeof( float ) * 12 );

        //sbt_offset += static_cast<unsigned int>( mesh->verts.size() ) * rayTypeCount;  // one sbt record per GAS build input per RAY_TYPE
    }
    const size_t instances_size_in_bytes = sizeof( OptixInstance ) * num_instances;
    raii<CUdeviceptr>  d_instances;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_instances.reset() ), instances_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)d_instances ),
                optix_instances.data(),
                instances_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    OptixBuildInput instance_input = {};
    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = d_instances;
    instance_input.instanceArray.numInstances = static_cast<unsigned int>( num_instances );

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags                  = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation                   = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                state.context,
                &accel_options,
                &instance_input,
                1, // num build inputs
                &ias_buffer_sizes
                ) );

    raii<CUdeviceptr> d_temp_buffer;
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_temp_buffer.reset() ),
                ias_buffer_sizes.tempSizeInBytes
                ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &state.m_d_ias_output_buffer.reset() ),
                ias_buffer_sizes.outputSizeInBytes
                ) );

    OPTIX_CHECK( optixAccelBuild(
                state.context,
                nullptr,                  // CUDA stream
                &accel_options,
                &instance_input,
                1,                  // num build inputs
                d_temp_buffer,
                ias_buffer_sizes.tempSizeInBytes,
                state.m_d_ias_output_buffer,
                ias_buffer_sizes.outputSizeInBytes,
                &state.m_ias_handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ) );


}
static void buildMeshAccel( PathTracerState& state )
{
    //
    // copy mesh data to device
    //
    const size_t vertices_size_in_bytes = g_vertices.size() * sizeof( Vertex );
    //CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_vertices ) ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_vertices.reset() ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)state.d_vertices ),
                g_vertices.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof( uint32_t );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_mat_indices.reset() ), mat_indices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.d_mat_indices ),
                g_mat_indices.data(),
                mat_indices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    // // Build triangle GAS // // One per SBT record for this build input
    std::vector<uint32_t> triangle_input_flags(//MAT_COUNT
        g_mtlidlut.size(),
        OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);

    OptixBuildInput triangle_input                           = {};
    triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes         = sizeof( Vertex );
    triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>( g_vertices.size() );
    triangle_input.triangleArray.vertexBuffers               = g_vertices.empty() ? nullptr : & state.d_vertices;
    triangle_input.triangleArray.flags                       = triangle_input_flags.data();
    triangle_input.triangleArray.numSbtRecords               = g_vertices.empty() ? 1 : g_mtlidlut.size();
    triangle_input.triangleArray.sbtIndexOffsetBuffer        = state.d_mat_indices;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    //char   log[2048]; size_t sizeof_log = sizeof( log );
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                state.context,
                &accel_options,
                &triangle_input,
                1,  // num_build_inputs
                &gas_buffer_sizes
                ) );

    raii<CUdeviceptr> d_temp_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer.reset(), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    raii<CUdeviceptr> d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK(cudaMalloc((void**)&d_buffer_temp_output_gas_and_compacted_size.reset(), compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result             = ( CUdeviceptr )( (char*)(CUdeviceptr)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild(
                state.context,
                0,                                  // CUDA stream
                &accel_options,
                &triangle_input,
                1,                                  // num build inputs
                d_temp_buffer,
                gas_buffer_sizes.tempSizeInBytes,
                d_buffer_temp_output_gas_and_compacted_size,
                gas_buffer_sizes.outputSizeInBytes,
                &state.gas_handle,
                &emitProperty,                      // emitted property list
                1                                   // num emitted properties
                ) );

    d_temp_buffer.reset();
    //d_mat_indices.reset();

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK(cudaMalloc((void**)&state.d_gas_output_buffer.reset(), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, 
        state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle ) );

        d_buffer_temp_output_gas_and_compacted_size.reset();
    }
    else
    {
        state.d_gas_output_buffer = std::move(d_buffer_temp_output_gas_and_compacted_size);
    }
}




static void createSBT( PathTracerState& state )
{
        state.d_raygen_record.reset();
        state.d_miss_records.reset();
        state.d_hitgroup_records.reset();
        state.d_gas_output_buffer.reset();
        state.accum_buffer_p.reset();

    raii<CUdeviceptr>  &d_raygen_record = state.d_raygen_record;
    const size_t raygen_record_size = sizeof( RayGenRecord );
        CUDA_CHECK(cudaMalloc((void**)&d_raygen_record.reset(), raygen_record_size));

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt ) );

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)d_raygen_record ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ) );


    raii<CUdeviceptr>  &d_miss_records = state.d_miss_records;
    const size_t miss_record_size = sizeof( MissRecord );
    CUDA_CHECK(cudaMalloc((void**)&d_miss_records.reset(), miss_record_size * RAY_TYPE_COUNT )) ;

    MissRecord ms_sbt[2];
    OPTIX_CHECK( optixSbtRecordPackHeader( state.radiance_miss_group,  &ms_sbt[0] ) );
    ms_sbt[0].data.bg_color = make_float4( 0.0f );
    OPTIX_CHECK( optixSbtRecordPackHeader( state.occlusion_miss_group, &ms_sbt[1] ) );
    ms_sbt[1].data.bg_color = make_float4( 0.0f );

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)d_miss_records ),
                ms_sbt,
                miss_record_size*RAY_TYPE_COUNT,
                cudaMemcpyHostToDevice
                ) );

    raii<CUdeviceptr>  &d_hitgroup_records = state.d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    CUDA_CHECK(cudaMalloc((void**)&d_hitgroup_records.reset(),
                hitgroup_record_size * RAY_TYPE_COUNT * g_mtlidlut.size()
                ));

    std::vector<HitGroupRecord> hitgroup_records(RAY_TYPE_COUNT * g_mtlidlut.size());
    for( int i = 0; i < g_mtlidlut.size(); ++i )
    {
        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material

            OPTIX_CHECK( optixSbtRecordPackHeader( OptixUtil::rtMaterialShaders[i].m_radiance_hit_group, &hitgroup_records[sbt_idx] ) );
            hitgroup_records[sbt_idx].data.uniforms        = nullptr; //TODO uniforms like iTime, iFrame, etc.
            hitgroup_records[sbt_idx].data.vertices        = reinterpret_cast<float4*>( (CUdeviceptr)state.d_vertices );
            hitgroup_records[sbt_idx].data.uv              = reinterpret_cast<float4*>( (CUdeviceptr)state.d_uv );
            hitgroup_records[sbt_idx].data.nrm             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_nrm );
            hitgroup_records[sbt_idx].data.clr             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_clr );
            hitgroup_records[sbt_idx].data.tan             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_tan );
            hitgroup_records[sbt_idx].data.lightMark       = reinterpret_cast<unsigned short*>( (CUdeviceptr)state.d_lightMark );
            for(int t=0;t<32;t++)
            {
                hitgroup_records[sbt_idx].data.textures[t] = OptixUtil::rtMaterialShaders[i].getTexture(t);
            }
        }

        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for ith material
            memset( &hitgroup_records[sbt_idx], 0, hitgroup_record_size );

            OPTIX_CHECK( optixSbtRecordPackHeader( OptixUtil::rtMaterialShaders[i].m_occlusion_hit_group, &hitgroup_records[sbt_idx] ) );
            hitgroup_records[sbt_idx].data.uniforms        = nullptr; //TODO uniforms like iTime, iFrame, etc.
            hitgroup_records[sbt_idx].data.vertices        = reinterpret_cast<float4*>( (CUdeviceptr)state.d_vertices );
            hitgroup_records[sbt_idx].data.uv              = reinterpret_cast<float4*>( (CUdeviceptr)state.d_uv );
            hitgroup_records[sbt_idx].data.nrm             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_nrm );
            hitgroup_records[sbt_idx].data.clr             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_clr );
            hitgroup_records[sbt_idx].data.tan             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_tan );
            hitgroup_records[sbt_idx].data.lightMark       = reinterpret_cast<unsigned short*>( (CUdeviceptr)state.d_lightMark );
            for(int t=0;t<32;t++)
            {
                hitgroup_records[sbt_idx].data.textures[t] = OptixUtil::rtMaterialShaders[i].getTexture(t);
            }
        }
    }
    // {
    //     int i = MAT_COUNT-1;
    //     {
    //         const int sbt_idx = i * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material

    //         OPTIX_CHECK( optixSbtRecordPackHeader( state.radiance_hit_group2, &hitgroup_records[sbt_idx] ) );
    //         hitgroup_records[sbt_idx].data.emission_color = g_emission_colors[i];
    //         hitgroup_records[sbt_idx].data.diffuse_color  = g_diffuse_colors[i];
    //         hitgroup_records[sbt_idx].data.vertices       = reinterpret_cast<float4*>( state.d_vertices );
    //     }

    //     {
    //         const int sbt_idx = i * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for ith material
    //         memset( &hitgroup_records[sbt_idx], 0, hitgroup_record_size );

    //         OPTIX_CHECK( optixSbtRecordPackHeader( state.occlusion_hit_group2, &hitgroup_records[sbt_idx] ) );
    //     }
    // }
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)d_hitgroup_records ),
                hitgroup_records.data(),
                hitgroup_record_size*RAY_TYPE_COUNT*g_mtlidlut.size(),
                cudaMemcpyHostToDevice
                ) );

    state.sbt.raygenRecord                = d_raygen_record;
    state.sbt.missRecordBase              = d_miss_records;
    state.sbt.missRecordStrideInBytes     = static_cast<uint32_t>( miss_record_size );
    state.sbt.missRecordCount             = RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase          = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    state.sbt.hitgroupRecordCount         = RAY_TYPE_COUNT * g_mtlidlut.size();
}


static void cleanupState( PathTracerState& state )
{
    OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.radiance_miss_group ) );
    //OPTIX_CHECK( optixProgramGroupDestroy( state.radiance_hit_group ) );
    //OPTIX_CHECK( optixProgramGroupDestroy( state.occlusion_hit_group ) );
    //OPTIX_CHECK( optixProgramGroupDestroy( state.radiance_hit_group2 ) );
    //OPTIX_CHECK( optixProgramGroupDestroy( state.occlusion_hit_group2 ) );
    //OPTIX_CHECK( optixProgramGroupDestroy( state.occlusion_miss_group ) );
    //OPTIX_CHECK( optixModuleDestroy( state.ptx_module ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );
    OPTIX_CHECK( optixModuleDestroy( OptixUtil::ray_module));


    //CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord ) ) );
    //CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase ) ) );
    //CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
        state.d_raygen_record.reset();
        state.d_miss_records.reset();
        state.d_hitgroup_records.reset();
        state.d_vertices.reset();
        state.d_gas_output_buffer.reset();
        state.accum_buffer_p.reset();
        state.d_params.reset();
}

static void detectHuangrenxunHappiness() {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    zeno::log_info("CUDA graphic card name: {}", prop.name);
    zeno::log_info("CUDA compute capability: {}.{}", prop.major, prop.minor);
    zeno::log_info("CUDA total global memory: {} MB", float(prop.totalGlobalMem / 1048576.0f));

    int driverVersion, runtimeVersion;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    zeno::log_info("CUDA driver version: {}.{}",
           driverVersion / 1000, (driverVersion % 100) / 10);
    zeno::log_info("CUDA runtime version: {}.{}",
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    if ((using20xx = (prop.major <= 7))) {
        // let's buy the stupid bitcoin card to cihou huangrenxun's wallet-happiness
        zeno::log_warn("graphic card <= RTX20** detected, disabling instancing. consider upgrade to RTX30** for full performance.");
    }
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------
std::optional<sutil::CUDAOutputBuffer<uchar4>> output_buffer_o;
std::optional<sutil::GLDisplay> gl_display_o;
sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
void optixinit( int argc, char* argv[] )
{
    state.params.width                             = 768;
    state.params.height                            = 768;

    //
    // Parse command line options
    //

    std::string outfile;
    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--no-gl-interop" )
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            outfile = argv[++i];
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            int w, h;
            sutil::parseDimensions( dims_arg.c_str(), w, h );
            state.params.width  = w;
            state.params.height = h;
        }
        else if( arg == "--launch-samples" || arg == "-s" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            samples_per_launch = atoi( argv[++i] );
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    detectHuangrenxunHappiness();
        initCameraState();

        //
        // Set up OptiX state
        //
        //createContext( state );
        OptixUtil::createContext();
        state.context = OptixUtil::context;

    //CUDA_CHECK( cudaStreamCreate( &state.stream.reset() ) );
    CUDA_CHECK(cudaMalloc((void**)&state.d_params.reset(), sizeof( Params )));

        if (!output_buffer_o) {
            output_buffer_o.emplace(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                    );
            output_buffer_o->setStream( 0 );
        }
        if (!gl_display_o) {
            gl_display_o.emplace(sutil::BufferImageFormat::UNSIGNED_BYTE4);
        }
}


//static std::string get_content(std::string const &path) {
    //std::ifstream ifs("/home/bate/zeno/zenovis/xinxinoptix/" + path);
    //if (!ifs) throw std::runtime_error("cannot open file: " + path);
    //std::string res;
    //std::copy(std::istreambuf_iterator<char>(ifs),
              //std::istreambuf_iterator<char>(),
              //std::back_inserter(res));
    //return res;
//}
void splitMesh(std::vector<Vertex> & verts, 
std::vector<uint32_t> &mat_idx, 
std::vector<std::shared_ptr<smallMesh>> &oMeshes);

std::vector<std::shared_ptr<smallMesh>> g_meshPieces;
static void updatedrawobjects();
void optixupdatemesh(std::map<std::string, int> const &mtlidlut) {
    camera_changed = true;
    g_mtlidlut = mtlidlut;
    updatedrawobjects();
//#ifdef USING_20XX
    if (using20xx) {
    const size_t vertices_size_in_bytes = g_vertices.size() * sizeof( Vertex );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_vertices.reset() ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)state.d_vertices ),
                g_vertices.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_clr.reset() ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)state.d_clr ),
                g_clr.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_uv.reset() ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)state.d_uv ),
                g_uv.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_nrm.reset() ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)state.d_nrm ),
                g_nrm.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_tan.reset() ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)state.d_tan ),
                g_tan.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof( uint32_t );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_mat_indices.reset() ), mat_indices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.d_mat_indices ),
                g_mat_indices.data(),
                mat_indices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    const size_t light_mark_size_in_bytes = g_lightMark.size() * sizeof( unsigned short );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_lightMark.reset() ), light_mark_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.d_lightMark ),
                g_lightMark.data(),
                light_mark_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    buildMeshAccel( state );
//#else
    } else {
    const size_t vertices_size_in_bytes = g_vertices.size() * sizeof( Vertex );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_vertices.reset() ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)state.d_vertices ),
                g_vertices.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_clr.reset() ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)state.d_clr ),
                g_clr.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_uv.reset() ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)state.d_uv ),
                g_uv.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_nrm.reset() ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)state.d_nrm ),
                g_nrm.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_tan.reset() ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)state.d_tan ),
                g_tan.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof( uint32_t );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_mat_indices.reset() ), mat_indices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.d_mat_indices ),
                g_mat_indices.data(),
                mat_indices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    const size_t light_mark_size_in_bytes = g_lightMark.size() * sizeof( unsigned short );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_lightMark.reset() ), light_mark_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.d_lightMark ),
                g_lightMark.data(),
                light_mark_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    splitMesh(g_vertices, g_mat_indices, g_meshPieces);
    std::cout<<"split mesh done\n";
    std::cout<<"mesh pieces:"<<g_meshPieces.size()<<std::endl;
    for(int i=0;i<g_meshPieces.size();i++)
    {
        buildMeshAccelSplitMesh(state, g_meshPieces[i]);
    }
    buildInstanceAccel(state, 2, g_meshPieces);
    }
//#endif
}

struct LightDat{
    std::vector<float> v0;
    std::vector<float> v1;
    std::vector<float> v2;
    std::vector<float> normal;
    std::vector<float> emission;
};
static std::map<std::string, LightDat> lightdats;

void unload_light(){
    lightdats.clear();
}

void load_light(std::string const &key, float const*v0,float const*v1,float const*v2, float const*nor,float const*emi){
    LightDat ld;
    ld.v0.assign(v0, v0 + 3);
    ld.v1.assign(v1, v1 + 3);
    ld.v2.assign(v2, v2 + 3);
    ld.normal.assign(nor, nor + 3);
    ld.emission.assign(emi, emi + 3);
    zeno::log_info("light clr after read: {} {} {}", ld.emission[0],ld.emission[1],ld.emission[2]);
    lightdats[key] = ld;
}

static void addLightMesh(float3 corner, float3 v2, float3 v1, float3 normal, float3 emission)
{
    float3 lc = corner;
    float3 vert0 = lc, vert1 = lc + v1, vert2 = lc + v2, vert3 = lc + v1 + v2;
    g_lightMesh.push_back(make_float4(vert0.x, vert0.y, vert0.z, 0.f));
    g_lightMesh.push_back(make_float4(vert1.x, vert1.y, vert1.z, 0.f));
    g_lightMesh.push_back(make_float4(vert2.x, vert2.y, vert2.z, 0.f));
    g_lightColor.push_back(make_float4(emission.x, emission.y, emission.z, 0.0f));
    g_lightColor.push_back(make_float4(emission.x, emission.y, emission.z, 0.0f));
    g_lightColor.push_back(make_float4(emission.x, emission.y, emission.z, 0.0f));

    g_lightMesh.push_back(make_float4(vert3.x, vert3.y, vert3.z, 0.f));
    g_lightMesh.push_back(make_float4(vert2.x, vert2.y, vert2.z, 0.f));
    g_lightMesh.push_back(make_float4(vert1.x, vert1.y, vert1.z, 0.f));
    g_lightColor.push_back(make_float4(emission.x, emission.y, emission.z, 0.0f));
    g_lightColor.push_back(make_float4(emission.x, emission.y, emission.z, 0.0f));
    g_lightColor.push_back(make_float4(emission.x, emission.y, emission.z, 0.0f));
}

void optixupdatelight() {
    camera_changed = true;

    zeno::log_info("lights size {}", lightdats.size());

    g_lights.clear();
    g_lightMesh.clear();
    g_lightColor.clear();
    if(lightdats.empty())
    {
        for (int i = 0; i < 1; i++) {
            auto &light = g_lights.emplace_back();
            light.emission = make_float3( 30000.f,30000.f,30000.f );
            light.corner   = make_float3( 343.0f, 548.5f, 227.0f );
            light.v2       = make_float3( -10.0f, 0.0f, 0.0f );
            light.normal   = make_float3( 0.0f, -1.0f, 0.0f );
            light.v1       = cross( light.v2, light.normal );
            addLightMesh(light.corner, light.v2, light.v1, light.normal, light.emission);
        }
    }
    else
    {
        for (auto const &[key, dat]: lightdats) {
            auto &light = g_lights.emplace_back();
            light.emission = make_float3( (float)(dat.emission[0]), (float)dat.emission[1], (float)dat.emission[2] );
            zeno::log_info("light clr after read: {} {} {}", light.emission.x,light.emission.y,light.emission.z);
            light.corner   = make_float3( dat.v0[0], dat.v0[1], dat.v0[2] );
            zeno::log_info("light clr after read: {} {} {}", light.corner.x,light.corner.y,light.corner.z);
            light.v1       = make_float3( dat.v1[0], dat.v1[1], dat.v1[2] );
            zeno::log_info("light clr after read: {} {} {}", light.v1.x,light.v1.y,light.v1.z);
            light.v2       = make_float3( dat.v2[0], dat.v2[1], dat.v2[2] );
            zeno::log_info("light clr after read: {} {} {}", light.v2.x,light.v2.y,light.v2.z);
            light.normal   = make_float3( dat.normal[0], dat.normal[1], dat.normal[2] );
            zeno::log_info("light clr after read: {} {} {}", light.normal.x,light.normal.y,light.normal.z);
            addLightMesh(light.corner, light.v2, light.v1, light.normal, light.emission);
        }
    }

    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &state.lightsbuf_p.reset() ),
                sizeof( ParallelogramLight ) * g_lights.size()
                ) );
    state.params.lights = (ParallelogramLight*)(CUdeviceptr)state.lightsbuf_p;
    if (g_lights.size())
        CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.lightsbuf_p ),
                g_lights.data(), sizeof( ParallelogramLight ) * g_lights.size(),
                cudaMemcpyHostToDevice
                ) );
}

void optixupdatematerial(std::vector<std::string> const &shaders, 
std::vector<std::vector<std::string>> &texs) {
    camera_changed = true;

        static bool hadOnce = false;
        if (!hadOnce) {
            //OPTIX_CHECK( optixModuleDestroy( OptixUtil::ray_module ) );
    if (!OptixUtil::createModule(
        OptixUtil::ray_module,
        state.context,
        sutil::lookupIncFile("PTKernel.cu"),
        "PTKernel.cu")) throw std::runtime_error("base ray module failed to compile");
        } hadOnce = true;
    OptixUtil::rtMaterialShaders.resize(0);
    for (int i = 0; i < shaders.size(); i++) {
        if (shaders[i].empty()) zeno::log_error("shader {} is empty", i);
        //OptixUtil::rtMaterialShaders.push_back(OptixUtil::rtMatShader(shaders[i].c_str(),"__closesthit__radiance", "__anyhit__shadow_cutout"));
        OptixUtil::rtMaterialShaders.emplace_back(shaders[i].c_str(),"__closesthit__radiance", "__anyhit__shadow_cutout");
        if(texs.size()>0){
            std::cout<<"texSize:"<<texs[i].size()<<std::endl;
            for(int j=0;j<texs[i].size();j++)
            {
                std::cout<<"texName:"<<texs[i][j]<<std::endl;
                OptixUtil::rtMaterialShaders[i].addTexture(j, texs[i][j]);
            }
        }
    }
    for(int i=0;i<OptixUtil::rtMaterialShaders.size();i++)
    {
        std::cout<<"now compiling "<<i<<"'th shader"<<std::endl;
        if(OptixUtil::rtMaterialShaders[i].loadProgram()==false)
        {
            std::cout<<"program compile failed, using default"<<std::endl;
            
            OptixUtil::rtMaterialShaders[i].m_shaderFile     = shaders[0];
            OptixUtil::rtMaterialShaders[i].m_shadingEntry   = "__closesthit__radiance";
            OptixUtil::rtMaterialShaders[i].m_occlusionEntry = "__anyhit__shadow_cutout";
            std::cout<<OptixUtil::rtMaterialShaders[i].loadProgram()<<std::endl;
            std::cout<<"shader restored to default\n";
        }
    }
    OptixUtil::createRenderGroups(state.context, OptixUtil::ray_module);
}

void optixupdateend() {
    camera_changed = true;
        OptixUtil::createPipeline();
        //static bool hadOnce = false;
        //if (hadOnce) {
    //OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
    //state.raygen_prog_group ) );
    //state.radiance_miss_group ) );
    //state.occlusion_miss_group ) );
    //OPTIX_CHECK( optixProgramGroupDestroy( state.radiance_hit_group ) );
    //OPTIX_CHECK( optixProgramGroupDestroy( state.occlusion_hit_group ) );
    //OPTIX_CHECK( optixProgramGroupDestroy( state.radiance_hit_group2 ) );
    //OPTIX_CHECK( optixProgramGroupDestroy( state.occlusion_hit_group2 ) );
    //OPTIX_CHECK( optixModuleDestroy( state.ptx_module ) );
    //OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );
        //} hadOnce = true;

        state.pipeline_compile_options = OptixUtil::pipeline_compile_options;
        state.pipeline = OptixUtil::pipeline;
        state.raygen_prog_group = OptixUtil::raygen_prog_group;
        state.radiance_miss_group = OptixUtil::radiance_miss_group;
        state.occlusion_miss_group = OptixUtil::occlusion_miss_group;
        //state.radiance_hit_group = OptixUtil::radiance_hit_group;
        //state.occlusion_hit_group = OptixUtil::occlusion_hit_group;
        //state.radiance_hit_group2 = OptixUtil::radiance_hit_group2;
        //state.occlusion_hit_group2 = OptixUtil::occlusion_hit_group2;
        //state.ptx_module2 = createModule(state.context, "optixPathTracer.cu");
        //createModule( state );
        //createProgramGroups( state );
        //createPipeline( state );
        createSBT( state );
        initLaunchParams( state );
}

struct DrawDat {
    std::string mtlid;
    std::vector<float> verts;
    std::vector<int> tris;
    std::map<std::string, std::vector<float>> vertattrs;
    auto const &getAttr(std::string const &s) const
    {
        //if(vertattrs.find(s)!=vertattrs.end())
        //{
            return vertattrs.at(s);//->second;
        //}
        
    }
};
static std::map<std::string, DrawDat> drawdats;

void splitMesh(std::vector<Vertex> & verts, std::vector<uint32_t> &mat_idx, 
std::vector<std::shared_ptr<smallMesh>> &oMeshes)
{
    size_t num_tri = verts.size()/3;
    oMeshes.resize(0);
    size_t tris_per_mesh = 1024;
    size_t num_iter = num_tri/tris_per_mesh + 1;
    for(int i=0; i<num_iter;i++)
    {
        auto m = std::make_shared<smallMesh>();
        for(int j=0;j<tris_per_mesh;j++)
        {
            size_t idx = i*tris_per_mesh + j;
            if(idx<num_tri){
                m->verts.emplace_back(verts[idx*3+0]);
                m->verts.emplace_back(verts[idx*3+1]);
                m->verts.emplace_back(verts[idx*3+2]);
                m->mat_idx.emplace_back(mat_idx[idx]);
                m->idx.emplace_back(make_uint3(idx*3+0,idx*3+1,idx*3+2));
            }
        }
        if(m->verts.size()>0)
            oMeshes.push_back(std::move(m));
    }
}
static void updatedrawobjects() {
    g_vertices.clear();
    g_clr.clear();
    g_nrm.clear();
    g_uv.clear();
    g_tan.clear();
    g_mat_indices.clear();
    g_lightMark.clear();
    size_t n = 0;
    for (auto const &[key, dat]: drawdats) {
        n += dat.tris.size()/3;
    }
    g_vertices.resize(n * 3);
    g_clr.resize(n*3);
    g_nrm.resize(n*3);
    g_uv.resize(n*3);
    g_tan.resize(n*3);
    g_mat_indices.resize(n);
    g_lightMark.resize(n);
    n = 0;
    for (auto const &[key, dat]: drawdats) {
        auto it = g_mtlidlut.find(dat.mtlid);
        int mtlindex = it != g_mtlidlut.end() ? it->second : 0;
        //zeno::log_error("{} {}", dat.mtlid, mtlindex);
//#pragma omp parallel for
        for (size_t i = 0; i < dat.tris.size() / 3; i++) {
            g_mat_indices[n + i] = mtlindex;
            g_lightMark[n+i] = 0;
            g_vertices[(n + i) * 3 + 0] = {
                dat.verts[dat.tris[i * 3 + 0] * 3 + 0],
                dat.verts[dat.tris[i * 3 + 0] * 3 + 1],
                dat.verts[dat.tris[i * 3 + 0] * 3 + 2],
                0,
            };
            g_vertices[(n + i) * 3 + 1] = {
                dat.verts[dat.tris[i * 3 + 1] * 3 + 0],
                dat.verts[dat.tris[i * 3 + 1] * 3 + 1],
                dat.verts[dat.tris[i * 3 + 1] * 3 + 2],
                0,
            };
            g_vertices[(n + i) * 3 + 2] = {
                dat.verts[dat.tris[i * 3 + 2] * 3 + 0],
                dat.verts[dat.tris[i * 3 + 2] * 3 + 1],
                dat.verts[dat.tris[i * 3 + 2] * 3 + 2],
                0,
            };

            g_clr[(n + i) * 3 + 0] = {
                dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 0],
                dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 1],
                dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 2],
                0,
            };
            g_clr[(n + i) * 3 + 1] = {
                dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 0],
                dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 1],
                dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 2],
                0,
            };
            g_clr[(n + i) * 3 + 2] = {
                dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 0],
                dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 1],
                dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 2],
                0,
            };

            g_nrm[(n + i) * 3 + 0] = {
                dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 0],
                dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 1],
                dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 2],
                0,
            };
            g_nrm[(n + i) * 3 + 1] = {
                dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 0],
                dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 1],
                dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 2],
                0,
            };
            g_nrm[(n + i) * 3 + 2] = {
                dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 0],
                dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 1],
                dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 2],
                0,
            };

            g_uv[(n + i) * 3 + 0] = {
                dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 0],
                dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 1],
                dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 2],
                0,
            };
            g_uv[(n + i) * 3 + 1] = {
                dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 0],
                dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 1],
                dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 2],
                0,
            };
            g_uv[(n + i) * 3 + 2] = {
                dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 0],
                dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 1],
                dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 2],
                0,
            };

            g_tan[(n + i) * 3 + 0] = {
                dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 0],
                dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 1],
                dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 2],
                0,
            };
            g_tan[(n + i) * 3 + 1] = {
                dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 0],
                dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 1],
                dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 2],
                0,
            };
            g_tan[(n + i) * 3 + 2] = {
                dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 0],
                dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 1],
                dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 2],
                0,
            };
        }
        n += dat.tris.size() / 3;
    }
    size_t ltris = g_lightMesh.size()/3;
    for(int l=0;l<ltris;l++)
    {
        g_vertices.push_back(g_lightMesh[l*3+0]);
        g_vertices.push_back(g_lightMesh[l*3+1]);
        g_vertices.push_back(g_lightMesh[l*3+2]);
        auto l1 = g_lightMesh[l*3+1] - g_lightMesh[l*3+0];
        auto l2 = g_lightMesh[l*3+2] - g_lightMesh[l*3+0];
        auto zl1= zeno::vec3f(l1.x, l1.y, l1.z);
        auto zl2= zeno::vec3f(l2.x, l2.y, l2.z);
        auto normal = zeno::normalize(zeno::cross(zl1,zl2));
        g_nrm.push_back(make_float4(normal[0], normal[1], normal[2],0));
        g_nrm.push_back(make_float4(normal[0], normal[1], normal[2],0));
        g_nrm.push_back(make_float4(normal[0], normal[1], normal[2],0));
        g_clr.push_back(g_lightColor[l*3+0]);
        g_clr.push_back(g_lightColor[l*3+1]);
        g_clr.push_back(g_lightColor[l*3+2]);
        g_tan.push_back(make_float4(0));
        g_tan.push_back(make_float4(0));
        g_tan.push_back(make_float4(0));
        g_uv.push_back( make_float4(0));
        g_uv.push_back( make_float4(0));
        g_uv.push_back( make_float4(0));
        g_mat_indices.push_back(0);
        g_lightMark.push_back(1);
    }
}

void load_object(std::string const &key, std::string const &mtlid, float const *verts, size_t numverts, int const *tris, size_t numtris, std::map<std::string, std::pair<float const *, size_t>> const &vtab) {
    DrawDat &dat = drawdats[key];
    //ZENO_P(mtlid);
    dat.mtlid = mtlid;
    dat.verts.assign(verts, verts + numverts * 3);
    dat.tris.assign(tris, tris + numtris * 3);
    //TODO: flatten just here... or in renderengineoptx.cpp
    for (auto const &[key, fptr]: vtab) {
        dat.vertattrs[key].assign(fptr.first, fptr.first + numverts * fptr.second);
    }
}

void unload_object(std::string const &key) {
    drawdats.erase(key);
}

void set_window_size(int nx, int ny) {
    state.params.width = nx;
    state.params.height = ny;
    camera_changed = true;
    resize_dirty = true;
}

void set_perspective(float const *U, float const *V, float const *W, float const *E, float aspect, float fov) {
    auto &cam = state.params.cam;
    cam.eye = make_float3(E[0], E[1], E[2]);
    cam.right = make_float3(U[0], U[1], U[2]);
    cam.right *= aspect;
    cam.up = make_float3(V[0], V[1], V[2]);
    cam.front = make_float3(W[0], W[1], W[2]);
    if (fov > 0) {
        float radfov = fov * float(M_PI) / 180;
        float tanfov = std::tan(radfov / 2);
        cam.front /= tanfov;
        float focallen = 0.018f / tanfov;
        cam.eye -= focallen * cam.front;
    }
    camera_changed = true;
    //cam.aspect = aspect;
    //cam.fov = fov;
    //camera.setZxxViewMatrix(U, V, W);
    //camera.setAspectRatio(aspect);
    //camera.setFovY(fov * aspect * (float)M_PI / 180.0f);
}


void optixrender(int fbo) {
    zeno::log_debug("[optix] rendering subframe {}", state.params.subframe_index);
    if (!output_buffer_o) throw sutil::Exception("no output_buffer_o");
    if (!gl_display_o) throw sutil::Exception("no gl_display_o");
    updateState( *output_buffer_o, state.params );
    for(int f=0;f<16;f++){
        launchSubframe( *output_buffer_o, state );
        state.params.subframe_index++;
    }
    displaySubframe( *output_buffer_o, *gl_display_o, state, fbo );
                    
}

void *optixgetimg(int &w, int &h) {
    w = output_buffer_o->width();
    h = output_buffer_o->height();
    return output_buffer_o->getHostPointer();
}

//void optixsaveimg(const char *outfile) {
    //sutil::ImageBuffer buffer;
    //buffer.data         = output_buffer_o->getHostPointer();
    //buffer.width        = output_buffer_o->width();
    //buffer.height       = output_buffer_o->height();
    //buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

    //sutil::saveImage( outfile, buffer, false );
//}

void optixcleanup() {
    return;             // other wise will crasu
    CUDA_SYNC_CHECK();
    cleanupState( state );
}
#if 0
        if( outfile.empty() )
        {
            //GLFWwindow* window = sutil::initUI( "optixPathTracer", state.params.width, state.params.height );
            //glfwSetMouseButtonCallback( window, mouseButtonCallback );
            //glfwSetCursorPosCallback( window, cursorPosCallback );
            //glfwSetWindowSizeCallback( window, windowSizeCallback );
            //glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            //glfwSetKeyCallback( window, keyCallback );
            //glfwSetScrollCallback( window, scrollCallback );
            //glfwSetWindowUserPointer( window, &state.params );

            //
            // Render loop
            //
            {

                //sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState( output_buffer, state.params );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe( output_buffer, state );
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe( output_buffer, gl_display, window );
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats( state_update_time, render_time, display_time );

                    glfwSwapBuffers( window );

                    ++state.params.subframe_index;
                } while( !glfwWindowShouldClose( window ) );
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI( window );
        }
        else
        {
            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                    );

            handleCameraUpdate( state.params );
            handleResize( output_buffer, state.params );
            launchSubframe( output_buffer, state );

            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = output_buffer.width();
            buffer.height       = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

            sutil::saveImage( outfile.c_str(), buffer, false );

            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                glfwTerminate();
            }
        }

#endif
}
