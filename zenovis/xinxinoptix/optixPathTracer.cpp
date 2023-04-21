//
// this part of code is modified from nvidia's optix example


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
#include <sutil/Scene.h>
#include <optix_stack_size.h>
#include <stb_image_write.h>

//#include <GLFW/glfw3.h>

#include "optixPathTracer.h"

#include <zeno/utils/log.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/UserData.h>
#include "optixVolume.h"
#include "zeno/core/Session.h"

#include <thread>
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
#include <unordered_map>
#include "xinxinoptixapi.h"
#include "OptiXStuff.h"
#include <zeno/utils/vec.h>
#include <zeno/utils/envconfig.h>
#include <zeno/utils/orthonormal.h>
#include <unordered_map>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <string_view>
struct CppTimer {
    void tick() {
        struct timespec t;
        std::timespec_get(&t, TIME_UTC);
        last = t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
    }
    void tock() {
        struct timespec t;
        std::timespec_get(&t, TIME_UTC);
        cur = t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
    }
    float elapsed() const noexcept {
        return cur - last;
    }
    void tock(std::string_view tag) {
        tock();
        printf("%s: %f ms\n", tag.data(), elapsed());
    }

  private:
    double last, cur;
};
static CppTimer timer, localTimer;

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
//int32_t samples_per_launch = 16;

std::vector<std::shared_ptr<VolumeWrapper>> list_volume;
std::vector<std::shared_ptr<VolumeAccel>>   list_volume_accel;

std::vector<uint> list_volume_index_in_shader_list;

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

    OptixTraversableHandle         root_handle;
    raii<CUdeviceptr>              root_output_buffer;

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
    raii<CUdeviceptr>              d_mat_indices;
    raii<CUdeviceptr>              d_meshIdxs;
    raii<CUdeviceptr>              d_meshMats;
    raii<CUdeviceptr>              d_instPos;
    raii<CUdeviceptr>              d_instNrm;
    raii<CUdeviceptr>              d_instUv;
    raii<CUdeviceptr>              d_instClr;
    raii<CUdeviceptr>              d_instTang;
    raii<CUdeviceptr>              d_uniforms;

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
    CUdeviceptr                              d_params2=0;

    raii<CUdeviceptr>  d_raygen_record;
    raii<CUdeviceptr>  d_miss_records;
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

struct InstData
{
    std::vector<Vertex> vertices = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    std::vector<Vertex> clr = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    std::vector<Vertex> nrm = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    std::vector<Vertex> uv = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    std::vector<Vertex> tan = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    std::vector<uint32_t> mat_indices = {0, 0, 0};
    std::vector<uint16_t> lightMark = {0};

    std::size_t staticMeshNum = 0;
    std::size_t staticVertNum = 0;

    std::vector<std::shared_ptr<smallMesh>> meshPieces;
};
static std::map<std::string, InstData> g_instLUT;
std::unordered_map<std::string, std::vector<glm::mat4>> g_instMatsLUT;
struct InstAttr
{
    std::vector<float3> pos;
    std::vector<float3> nrm;
    std::vector<float3> uv;
    std::vector<float3> clr;
    std::vector<float3> tang;
};
std::unordered_map<std::string, InstAttr> g_instAttrsLUT;

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
    state.params.handle         = state.root_handle;
    }
//#endif
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &state.accum_buffer_p.reset() ),
                state.params.width * state.params.height * sizeof( float4 )
                ) );
    
    state.params.accum_buffer = (float4*)(CUdeviceptr)state.accum_buffer_p;
    
    state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    //state.params.samples_per_launch = samples_per_launch;
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

    CUDA_SYNC_CHECK();
    CUDA_CHECK( cudaMemcpy((void*)state.d_params2 ,
                &state.params, sizeof( Params ),
                cudaMemcpyHostToDevice
                ) );
    CUDA_SYNC_CHECK();
    OPTIX_CHECK( optixLaunch(
                state.pipeline,
                0,
                (CUdeviceptr)state.d_params2,
                sizeof( Params ),
                &state.sbt,
                state.params.width,   // launch width
                state.params.height,  // launch height
                1                     // launch depth
                ) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
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
    triangle_input.triangleArray.vertexBuffers               = g_vertices.empty() ? nullptr : & mesh->dverts;
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
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION; //| OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
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

static size_t g_staticMeshNum = 0;
static size_t g_staticVertNum = 0;
static size_t g_staticAndDynamicMeshNum = 0;
static size_t g_staticAndDynamicVertNum = 0;

static void buildInstanceAccel(PathTracerState& state, int rayTypeCount, std::vector<std::shared_ptr<smallMesh>> m_meshes)
{
    //zeno::log_info("build IAS begin");
    std::cout<<"IAS begin"<<std::endl;
    timer.tick();
    const float mat4x4[12] = {1,0,0,0,0,1,0,0,0,0,1,0};

    float3 defaultInstPos = {0, 0, 0};
    float3 defaultInstNrm = {0, 1, 0};
    float3 defaultInstUv = {0, 0, 0};
    float3 defaultInstClr = {1, 1, 1};
    float3 defaultInstTang = {1, 0, 0};
    std::size_t num_instances = g_staticAndDynamicMeshNum;
    for (const auto &[instID, instData] : g_instLUT)
    {
        auto it = g_instMatsLUT.find(instID);
        if (it != g_instMatsLUT.end())
        {
            num_instances += it->second.size() * instData.meshPieces.size();
        }
        else
        {
            num_instances += instData.meshPieces.size();
        }
    }

    std::vector<OptixInstance> optix_instances( num_instances );
    memset( optix_instances.data(), 0, sizeof( OptixInstance ) * num_instances );

    for (auto& ins : optix_instances ) {
        ins.sbtOffset = 0;
        ins.visibilityMask = DefaultMatMask;
    }
    
    std::vector<int> meshIdxs(num_instances);
    std::vector<float> meshMats(16 * num_instances);
    std::vector<float3> instPos(num_instances);
    std::vector<float3> instNrm(num_instances);
    std::vector<float3> instUv(num_instances);
    std::vector<float3> instClr(num_instances);
    std::vector<float3> instTang(num_instances);
    unsigned int sbt_offset = 0;
    for( size_t i = 0; i < g_staticAndDynamicMeshNum; ++i )
    {
        auto  mesh = m_meshes[i];
        auto& optix_instance = optix_instances[i];
        memset( &optix_instance, 0, sizeof( OptixInstance ) );

        optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId        = static_cast<unsigned int>( i );
        //optix_instance.sbtOffset         = 0;
        optix_instance.visibilityMask    = DefaultMatMask;
        optix_instance.traversableHandle = mesh->gas_handle;
        memcpy( optix_instance.transform, mat4x4, sizeof( float ) * 12 );

        meshIdxs[i] = i; 
        for (int j = 0; j < 12; ++j)
        {
            meshMats[16 * i + j] = mat4x4[j];
        }
        meshMats[16 * i + 12] = 0;
        meshMats[16 * i + 13] = 0;
        meshMats[16 * i + 14] = 0;
        meshMats[16 * i + 15] = 1;
        instPos[i] = defaultInstPos;
        instNrm[i] = defaultInstNrm;
        instUv[i] = defaultInstUv;
        instClr[i] = defaultInstClr;
        instTang[i] = defaultInstTang;
    }

    std::size_t instanceId = g_staticAndDynamicMeshNum;
    std::size_t meshesOffset = g_staticAndDynamicMeshNum;
    for (auto &[instID, instData] : g_instLUT)
    {
        auto it = g_instMatsLUT.find(instID);
        if (it != g_instMatsLUT.end())
        {
            const auto &instMats = it->second;
            const auto &instAttrs = g_instAttrsLUT[instID];
            for (std::size_t i = 0; i < instData.meshPieces.size(); ++i)
            {
                auto mesh = m_meshes[meshesOffset];
                for (std::size_t k = 0; k < instMats.size(); ++k)
                {
                    const auto &instMat = instMats[k];
                    float instMat4x4[12] = {
                        instMat[0][0], instMat[1][0], instMat[2][0], instMat[3][0],
                        instMat[0][1], instMat[1][1], instMat[2][1], instMat[3][1],
                        instMat[0][2], instMat[1][2], instMat[2][2], instMat[3][2]};
                    auto& optix_instance = optix_instances[instanceId];
                    optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                    optix_instance.instanceId = static_cast<unsigned int>(instanceId);
                    optix_instance.visibilityMask = DefaultMatMask;
                    optix_instance.traversableHandle = mesh->gas_handle;
                    memcpy(optix_instance.transform, instMat4x4, sizeof(float) * 12);

                    meshIdxs[instanceId] = meshesOffset; 
                    for (int j = 0; j < 12; ++j)
                    {
                        meshMats[16 * instanceId + j] = instMat4x4[j];
                    }
                    meshMats[16 * instanceId + 12] = 0;
                    meshMats[16 * instanceId + 13] = 0;
                    meshMats[16 * instanceId + 14] = 0;
                    meshMats[16 * instanceId + 15] = 1;
                    instPos[instanceId] = instAttrs.pos[k];
                    instNrm[instanceId] = instAttrs.nrm[k];
                    instUv[instanceId] = instAttrs.uv[k];
                    instClr[instanceId] = instAttrs.clr[k];
                    instTang[instanceId] = instAttrs.tang[k];

                    ++instanceId;
                }
                ++meshesOffset;
            }
        }
        else
        {
            for (std::size_t i = 0; i < instData.meshPieces.size(); ++i)
            {
                auto mesh = m_meshes[meshesOffset];
                auto &optix_instance = optix_instances[instanceId];
                optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                optix_instance.instanceId = static_cast<unsigned int>(instanceId);
                optix_instance.visibilityMask = DefaultMatMask;
                optix_instance.traversableHandle = mesh->gas_handle;
                memcpy(optix_instance.transform, mat4x4, sizeof(float) * 12);

                meshIdxs[instanceId] = meshesOffset; 
                for (int j = 0; j < 12; ++j)
                {
                    meshMats[16 * instanceId + j] = mat4x4[j];
                }
                meshMats[16 * instanceId + 12] = 0;
                meshMats[16 * instanceId + 13] = 0;
                meshMats[16 * instanceId + 14] = 0;
                meshMats[16 * instanceId + 15] = 1;
                instPos[instanceId] = defaultInstPos;
                instNrm[instanceId] = defaultInstNrm;
                instUv[instanceId] = defaultInstUv;
                instClr[instanceId] = defaultInstClr;
                instTang[instanceId] = defaultInstTang;

                ++instanceId;
                ++meshesOffset;
            }
        }
    }

    state.d_meshIdxs.resize(sizeof(meshIdxs[0]) * meshIdxs.size(), 0);
    // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_meshIdxs.reset() ), sizeof(meshIdxs[0]) * meshIdxs.size()) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.d_meshIdxs ),
                meshIdxs.data(),
                sizeof(meshIdxs[0]) * meshIdxs.size(),
                cudaMemcpyHostToDevice
                ) );
    state.d_meshMats.resize(sizeof(meshMats[0]) * meshMats.size(), 0);
    // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_meshMats.reset() ), sizeof(meshMats[0]) * meshMats.size()) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.d_meshMats ),
                meshMats.data(),
                sizeof(meshMats[0]) * meshMats.size(),
                cudaMemcpyHostToDevice
                ) );
    state.d_instPos.resize(sizeof(instPos[0]) * instPos.size(), 0);
    // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_instPos.reset() ), sizeof(instPos[0]) * instPos.size()) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.d_instPos ),
                instPos.data(),
                sizeof(instPos[0]) * instPos.size(),
                cudaMemcpyHostToDevice
                ) );
    state.d_instNrm.resize(sizeof(instNrm[0]) * instNrm.size(), 0);
    // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_instNrm.reset() ), sizeof(instNrm[0]) * instNrm.size()) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.d_instNrm ),
                instNrm.data(),
                sizeof(instNrm[0]) * instNrm.size(),
                cudaMemcpyHostToDevice
                ) );
    state.d_instUv.resize(sizeof(instUv[0]) * instUv.size(), 0);
    // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_instUv.reset() ), sizeof(instUv[0]) * instUv.size()) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.d_instUv ),
                instUv.data(),
                sizeof(instUv[0]) * instUv.size(),
                cudaMemcpyHostToDevice
                ) );
    state.d_instClr.resize(sizeof(instClr[0]) * instClr.size(), 0);
    // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_instClr.reset() ), sizeof(instClr[0]) * instClr.size()) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.d_instClr ),
                instClr.data(),
                sizeof(instClr[0]) * instClr.size(),
                cudaMemcpyHostToDevice
                ) );
    state.d_instTang.resize(sizeof(instTang[0]) * instTang.size(), 0);
    // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_instTang.reset() ), sizeof(instTang[0]) * instTang.size()) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.d_instTang ),
                instTang.data(),
                sizeof(instTang[0]) * instTang.size(),
                cudaMemcpyHostToDevice
                ) );

    timer.tock("done IAS middle");
    std::cout<<"IAS middle\n";
    timer.tick();

    const size_t instances_size_in_bytes = sizeof( OptixInstance ) * optix_instances.size();
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
    instance_input.instanceArray.numInstances = static_cast<unsigned int>( optix_instances.size() );

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags                  = OPTIX_BUILD_FLAG_ALLOW_COMPACTION; //| OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
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

    OptixInstance bigger_instance {};
        bigger_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        bigger_instance.instanceId = 0;
        bigger_instance.sbtOffset = 0;
        bigger_instance.visibilityMask = DefaultMatMask;
        bigger_instance.traversableHandle = state.m_ias_handle;
        memcpy( bigger_instance.transform, mat4x4, sizeof( float ) * 12 );

        auto _optix_instances = std::vector<OptixInstance>{bigger_instance};

        // process volume
        {  
            for ( uint i=0; i<list_volume.size(); ++i ) {

                OptixInstance optix_instance {};

                sbt_offset = list_volume_index_in_shader_list[i] * RAY_TYPE_COUNT;

                optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                optix_instance.instanceId = static_cast<unsigned int>( optix_instances.size() );
                optix_instance.sbtOffset = sbt_offset;
                optix_instance.visibilityMask = VolumeMatMask; //VOLUME_OBJECT;
                optix_instance.traversableHandle = list_volume_accel[i]->handle;
                getOptixTransform( *(list_volume[i]), optix_instance.transform ); // transform as stored in Grid

                _optix_instances.push_back( optix_instance );
            }
        }

        const size_t _instances_size_in_bytes = sizeof( OptixInstance ) * _optix_instances.size();
        raii<CUdeviceptr>  _d_instances;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &_d_instances.reset() ), _instances_size_in_bytes ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( (CUdeviceptr)_d_instances ),
                    _optix_instances.data(),
                    _instances_size_in_bytes,
                    cudaMemcpyHostToDevice
                    ) );

        OptixBuildInput _instance_input = {};
        _instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        _instance_input.instanceArray.instances    = _d_instances;
        _instance_input.instanceArray.numInstances = static_cast<unsigned int>( _optix_instances.size() );

        OptixAccelBuildOptions _accel_options = {};
        _accel_options.buildFlags                  = OPTIX_BUILD_FLAG_ALLOW_COMPACTION; //| OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
        _accel_options.operation                   = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes _ias_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage(
                    state.context,
                    &_accel_options,
                    &_instance_input,
                    1, // num build inputs
                    &_ias_buffer_sizes
                    ) );

        raii<CUdeviceptr> _d_temp_buffer;
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &_d_temp_buffer.reset() ),
                    _ias_buffer_sizes.tempSizeInBytes
                    ) );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &state.root_output_buffer.reset() ),
                    _ias_buffer_sizes.outputSizeInBytes
                    ) );

        OPTIX_CHECK( optixAccelBuild(
                    state.context,
                    nullptr,                  // CUDA stream
                    &_accel_options,
                    &_instance_input,
                    1,                  // num build inputs
                    _d_temp_buffer,
                    _ias_buffer_sizes.tempSizeInBytes,
                    state.root_output_buffer,
                    _ias_buffer_sizes.outputSizeInBytes,
                    &state.root_handle,
                    nullptr,            // emitted property list
                    0                   // num emitted properties
                    ) );

    timer.tock("done IAS build");
    std::cout<<"IAS end\n";
    //zeno::log_info("build IAS end");
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
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION; //| OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
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

    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    const size_t hitgroup_record_count = g_mtlidlut.size() * RAY_TYPE_COUNT;

    raii<CUdeviceptr>  &d_hitgroup_records = state.d_hitgroup_records;
    
    CUDA_CHECK(cudaMalloc((void**)&d_hitgroup_records.reset(),
                hitgroup_record_size * hitgroup_record_count
                ));

    std::vector<HitGroupRecord> hitgroup_records(hitgroup_record_count);
    uint sbt_idx = 0;

    for( int j = 0; j < g_mtlidlut.size(); ++j ) {

        auto has_vdb = OptixUtil::rtMaterialShaders[j].has_vdb;

        if (!has_vdb) {

            sbt_idx = RAY_TYPE_COUNT*j;  // SBT for radiance ray-type for ith material
            hitgroup_records[sbt_idx] = {};

            OPTIX_CHECK( optixSbtRecordPackHeader( OptixUtil::rtMaterialShaders[j].m_radiance_hit_group, &hitgroup_records[sbt_idx] ) );
            hitgroup_records[sbt_idx].data.uniforms        = reinterpret_cast<float4*>( (CUdeviceptr)state.d_uniforms );
            hitgroup_records[sbt_idx].data.vertices        = reinterpret_cast<float4*>( (CUdeviceptr)state.d_vertices );
            hitgroup_records[sbt_idx].data.uv              = reinterpret_cast<float4*>( (CUdeviceptr)state.d_uv );
            hitgroup_records[sbt_idx].data.nrm             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_nrm );
            hitgroup_records[sbt_idx].data.clr             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_clr );
            hitgroup_records[sbt_idx].data.tan             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_tan );
            hitgroup_records[sbt_idx].data.lightMark       = reinterpret_cast<unsigned short*>( (CUdeviceptr)state.d_lightMark );
            hitgroup_records[sbt_idx].data.meshIdxs        = reinterpret_cast<int*>( (CUdeviceptr)state.d_meshIdxs );
            hitgroup_records[sbt_idx].data.meshMats        = reinterpret_cast<float*>( (CUdeviceptr)state.d_meshMats );
            hitgroup_records[sbt_idx].data.instPos         = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instPos );
            hitgroup_records[sbt_idx].data.instNrm         = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instNrm );
            hitgroup_records[sbt_idx].data.instUv          = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instUv );
            hitgroup_records[sbt_idx].data.instClr         = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instClr );
            hitgroup_records[sbt_idx].data.instTang        = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instTang );
            for(int t=0;t<32;t++)
            {
                hitgroup_records[sbt_idx].data.textures[t] = OptixUtil::rtMaterialShaders[j].getTexture(t);
            }

            sbt_idx = RAY_TYPE_COUNT*j + 1;  // SBT for occlusion ray-type for ith material
            memset( &hitgroup_records[sbt_idx], 0, hitgroup_record_size );

            OPTIX_CHECK( optixSbtRecordPackHeader( OptixUtil::rtMaterialShaders[j].m_occlusion_hit_group, &hitgroup_records[sbt_idx] ) );
            hitgroup_records[sbt_idx].data.uniforms        = reinterpret_cast<float4*>( (CUdeviceptr)state.d_uniforms );
            hitgroup_records[sbt_idx].data.vertices        = reinterpret_cast<float4*>( (CUdeviceptr)state.d_vertices );
            hitgroup_records[sbt_idx].data.uv              = reinterpret_cast<float4*>( (CUdeviceptr)state.d_uv );
            hitgroup_records[sbt_idx].data.nrm             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_nrm );
            hitgroup_records[sbt_idx].data.clr             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_clr );
            hitgroup_records[sbt_idx].data.tan             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_tan );
            hitgroup_records[sbt_idx].data.lightMark       = reinterpret_cast<unsigned short*>( (CUdeviceptr)state.d_lightMark );
            hitgroup_records[sbt_idx].data.meshIdxs        = reinterpret_cast<int*>( (CUdeviceptr)state.d_meshIdxs );
            hitgroup_records[sbt_idx].data.meshMats        = reinterpret_cast<float*>( (CUdeviceptr)state.d_meshMats );
            hitgroup_records[sbt_idx].data.instPos         = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instPos );
            hitgroup_records[sbt_idx].data.instNrm         = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instNrm );
            hitgroup_records[sbt_idx].data.instUv          = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instUv );
            hitgroup_records[sbt_idx].data.instClr         = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instClr );
            hitgroup_records[sbt_idx].data.instTang        = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instTang );
            for(int t=0;t<32;t++)
            {
                hitgroup_records[sbt_idx].data.textures[t] = OptixUtil::rtMaterialShaders[j].getTexture(t);
            }
        } else {

            sbt_idx = 2*j; 
            HitGroupRecord rec = {};

            if (OptixUtil::g_vdb_list_for_each_shader.count(j) == 0) {
                continue;
            }

            auto& vdb_list = OptixUtil::g_vdb_list_for_each_shader.at(j);

            //if (OptixUtil::g_cached_vdb_map.count(key_vdb) == 0) continue;
            //auto& volumeWrapper = OptixUtil::g_vdb[key_vdb];

            rec.data.opacityHDDA = 0.25f;
            rec.data.sigma_a = 1.0f;
            rec.data.sigma_s = 1.0f;
            rec.data.greenstein = 0;

            for(uint t=0; t<min(vdb_list.size(), 8ull); ++t)
            {
                auto vdb_key = vdb_list[t];
                auto vdb_ptr = OptixUtil::g_vdb_cached_map.at(vdb_key);

                rec.data.vdb_grids[t] = vdb_ptr->grids.front().deviceptr;
                rec.data.vdb_max_v[t] = vdb_ptr->grids.front().max_value;
            }

             for(int t=0;t<32;t++)
            {
                rec.data.textures[t] = OptixUtil::rtMaterialShaders[j].getTexture(t);
            }

            OPTIX_CHECK(optixSbtRecordPackHeader( OptixUtil::rtMaterialShaders[j].m_radiance_hit_group, &rec ) );
            hitgroup_records[sbt_idx] = rec;

            OPTIX_CHECK(optixSbtRecordPackHeader( OptixUtil::rtMaterialShaders[j].m_occlusion_hit_group, &rec ) );
            hitgroup_records[sbt_idx+1] = rec;
        }
    }

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)d_hitgroup_records ),
                hitgroup_records.data(),
                hitgroup_record_size*hitgroup_records.size(),
                cudaMemcpyHostToDevice
                ) );

    state.sbt.raygenRecord                = d_raygen_record;
    state.sbt.missRecordBase              = d_miss_records;
    state.sbt.missRecordStrideInBytes     = static_cast<uint32_t>( miss_record_size );
    state.sbt.missRecordCount             = RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase          = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    state.sbt.hitgroupRecordCount         = hitgroup_records.size();
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

    for (auto& ele : list_volume_accel) {
        cleanupVolumeAccel(*ele);
    }
    list_volume_accel.clear();
    
    for (auto& ele : list_volume) {
        cleanupVolume(*ele);
    }
    list_volume.clear();

    for (auto const& [key, val] : OptixUtil::g_vdb_cached_map) {
        cleanupVolume(*val);
    }
    OptixUtil::g_vdb_cached_map.clear();

        state.d_raygen_record.reset();
        state.d_miss_records.reset();
        state.d_hitgroup_records.reset();
        state.d_vertices.reset();
        state.d_gas_output_buffer.reset();
        state.accum_buffer_p.reset();
        state.d_params.reset();
    state = {};
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
            //samples_per_launch = atoi( argv[++i] );
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
    if(state.d_params2==0)
        CUDA_CHECK(cudaMalloc((void**)&state.d_params2, sizeof( Params )));

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
    xinxinoptix::update_procedural_sky(zeno::vec2f(-60, 45), 1, zeno::vec2f(0, 0), 0, 0.1,
                                       1.0, 0.0, 6500.0);
    xinxinoptix::using_hdr_sky(false);
}

void updateVolume() {

    OptixUtil::logInfoVRAM("Before update Volume");

    for (auto const& [key, val] : OptixUtil::g_vdb_cached_map) {

        if (OptixUtil::g_vdb_indice_visible.count(key) > 0) {
            // UPLOAD to GPU
            for (auto& task : val->tasks) {
                task();
            } //val->uploadTasks.clear();
        } else {      
            cleanupVolume(*val); // Remove from GPU-RAM, but keep in SYS-RAM 
        }
    }

    OptixUtil::logInfoVRAM("After update Volume");

    list_volume.clear();
    list_volume_index_in_shader_list.clear();

    for (uint i=0; i<list_volume_accel.size(); ++i) {
        auto ele = list_volume_accel[i];
        cleanupVolumeAccel(*ele);
    }
    list_volume_accel.clear();

    OptixUtil::logInfoVRAM("Before Volume GAS");

    for (auto const& [index, val] : OptixUtil::g_vdb_list_for_each_shader) {
        auto base_key = val.front();

        if (OptixUtil::g_vdb_indice_visible.count(base_key) == 0) continue;

        list_volume.push_back( OptixUtil::g_vdb_cached_map[base_key] );
        list_volume_index_in_shader_list.push_back(index);
   }

    for (uint i=0; i<list_volume.size(); ++i) {
        VolumeAccel accel;
        buildVolumeAccel( accel, *(list_volume[i]), state.context );

        list_volume_accel.push_back(std::make_shared<VolumeAccel>(std::move(accel)) );
    }

    OptixUtil::logInfoVRAM("After Volume GAS");
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
std::vector<std::shared_ptr<smallMesh>> &oMeshes, int meshesStart, int vertsStart);
std::vector<std::shared_ptr<smallMesh>> g_StaticMeshPieces;
std::vector<std::shared_ptr<smallMesh>> g_meshPieces;
static void updateDynamicDrawObjects();
static void updateStaticDrawObjects();
static void updateStaticDrawInstObjects();
static void updateDynamicDrawInstObjects();
void UpdateStaticMesh(std::map<std::string, int> const &mtlidlut) {
    camera_changed = true;
    g_mtlidlut = mtlidlut;
    updateStaticDrawObjects();
    g_staticMeshNum = 0;
    g_staticVertNum = 0;
    if(!using20xx) {
        splitMesh(g_vertices, g_mat_indices, g_meshPieces, 0, 0);
        g_staticMeshNum = g_meshPieces.size();
        size_t vertSize = 1024 * 3 * g_meshPieces.size();
        g_staticVertNum = vertSize;
        g_vertices.resize(vertSize);
        g_clr.resize(vertSize);
        g_nrm.resize(vertSize);
        g_tan.resize(vertSize);
        g_uv.resize(vertSize);
        g_mat_indices.resize(vertSize/3);
        g_lightMark.resize(vertSize/3);
    }
}
void UpdateDynamicMesh(std::map<std::string, int> const &mtlidlut) {
    camera_changed = true;
    g_mtlidlut = mtlidlut;
    updateDynamicDrawObjects();
    if(!using20xx) {
        splitMesh(g_vertices, g_mat_indices, g_meshPieces, g_staticMeshNum, g_staticVertNum);
        g_staticAndDynamicMeshNum = g_meshPieces.size();
        size_t vertSize = 1024 * 3 * g_meshPieces.size();
        g_staticAndDynamicVertNum = vertSize;
        g_vertices.resize(vertSize);
        g_clr.resize(vertSize);
        g_nrm.resize(vertSize);
        g_tan.resize(vertSize);
        g_uv.resize(vertSize);
        g_mat_indices.resize(vertSize/3);
        g_lightMark.resize(vertSize/3);
    }
}
void UpdateStaticInstMesh(const std::map<std::string, int> &mtlidlut)
{
    camera_changed = true;
    g_mtlidlut = mtlidlut;
    updateStaticDrawInstObjects();
    for (auto &[_, instData] : g_instLUT)
    {
        instData.staticMeshNum = 0;
        instData.staticVertNum = 0;
    }
    if (!using20xx)
    {
        for (auto &[instID, instData] : g_instLUT)
        {
            auto &vertices = instData.vertices;
            auto &clr = instData.clr;
            auto &nrm = instData.nrm;
            auto &uv = instData.uv;
            auto &tan = instData.tan;
            auto &mat_indices = instData.mat_indices;
            auto &lightMark = instData.lightMark;
            auto &staticMeshNum = instData.staticMeshNum;
            auto &staticVertNum = instData.staticVertNum;
            auto &meshPieces = instData.meshPieces;

            splitMesh(vertices, mat_indices, meshPieces, 0, 0);
            staticMeshNum = meshPieces.size();
            std::size_t vertSize = 1024 * 3 * meshPieces.size();
            staticVertNum = vertSize;
            vertices.resize(vertSize);
            clr.resize(vertSize);
            nrm.resize(vertSize);
            tan.resize(vertSize);
            uv.resize(vertSize);
            mat_indices.resize(vertSize / 3);
            lightMark.resize(vertSize / 3);
        }
    }
}
void UpdateDynamicInstMesh(std::map<std::string, int> const &mtlidlut)
{
    camera_changed = true;
    g_mtlidlut = mtlidlut;
    updateDynamicDrawInstObjects();
    if(!using20xx)
    {
        for (auto &[instID, instData] : g_instLUT)
        {
            auto &vertices = instData.vertices;
            auto &clr = instData.clr;
            auto &nrm = instData.nrm;
            auto &uv = instData.uv;
            auto &tan = instData.tan;
            auto &mat_indices = instData.mat_indices;
            auto &lightMark = instData.lightMark;
            auto &staticMeshNum = instData.staticMeshNum;
            auto &staticVertNum = instData.staticVertNum;
            auto &meshPieces = instData.meshPieces;

            splitMesh(vertices, mat_indices, meshPieces, staticMeshNum, staticVertNum);
            std::size_t vertSize = 1024 * 3 * meshPieces.size();
            vertices.resize(vertSize);
            clr.resize(vertSize);
            nrm.resize(vertSize);
            tan.resize(vertSize);
            uv.resize(vertSize);
            mat_indices.resize(vertSize / 3);
            lightMark.resize(vertSize / 3);
        }
    }
}

void CopyInstMeshToGlobalMesh()
{
    if(!using20xx)
    {
        auto numVerts = g_staticAndDynamicVertNum;
        auto numMeshPieces = g_staticAndDynamicMeshNum;
        for (auto &[_, instData] : g_instLUT)
        {
            numVerts += instData.vertices.size();
            numMeshPieces += instData.meshPieces.size();
        }
        g_vertices.resize(numVerts);
        g_clr.resize(numVerts);
        g_nrm.resize(numVerts);
        g_tan.resize(numVerts);
        g_uv.resize(numVerts);
        g_mat_indices.resize(numVerts / 3);
        g_lightMark.resize(numVerts / 3);
        g_meshPieces.resize(numMeshPieces);

        auto vertsOffset = g_staticAndDynamicVertNum;
        auto meshPiecesOffset = g_staticAndDynamicMeshNum;
        for (auto &[_, instData] : g_instLUT)
        {
            auto &vertices = instData.vertices;
            auto &clr = instData.clr;
            auto &nrm = instData.nrm;
            auto &uv = instData.uv;
            auto &tan = instData.tan;
            auto &mat_indices = instData.mat_indices;
            auto &lightMark = instData.lightMark;
            auto &meshPieces = instData.meshPieces;

            for (int i = 0; i < vertices.size(); ++i)
            {
                g_vertices[vertsOffset + i] = vertices[i];
                g_clr[vertsOffset + i] = clr[i];
                g_nrm[vertsOffset + i] = nrm[i];
                g_uv[vertsOffset + i] = uv[i];
                g_tan[vertsOffset + i] = tan[i];
            }
            for (int i = 0; i < vertices.size() / 3; ++i)
            {
                g_mat_indices[vertsOffset / 3 + i] = mat_indices[i];
                g_lightMark[vertsOffset / 3 + i] = lightMark[i];
            }
            for (int i = 0; i < meshPieces.size(); ++i)
            {
                g_meshPieces[meshPiecesOffset + i] = meshPieces[i];
            }

            vertsOffset += vertices.size();
            meshPiecesOffset += meshPieces.size();
        }
    }
}

void UpdateGasAndIas(bool staticNeedUpdate)
{
//#ifdef USING_20XX
    // no archieve inst func in using20xx
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
        for(int i=0;i<g_meshPieces.size();i++)
        {
            buildMeshAccelSplitMesh(state, g_meshPieces[i]);
        }
#define WXL 1
        std::cout << "begin copy\n";
        timer.tick();
        size_t vertices_size_in_bytes = g_vertices.size() * sizeof(Vertex);
        size_t static_vertices_size_in_bytes = g_staticVertNum * sizeof(Vertex);
        size_t dynamic_vertices_size_in_bytes = vertices_size_in_bytes - static_vertices_size_in_bytes;
        bool realloced;
        size_t offset = 0;
        size_t numBytes = vertices_size_in_bytes;
        auto updateRange = [&vertices_size_in_bytes, &dynamic_vertices_size_in_bytes, &realloced, &offset,
                            &numBytes]() {
            if (!realloced && WXL) {
                offset = g_staticVertNum * sizeof(Vertex);
                numBytes = dynamic_vertices_size_in_bytes;
            } else {
                offset = 0;
                numBytes = vertices_size_in_bytes;
            }
        };
#if WXL
        realloced = state.d_vertices.resize(vertices_size_in_bytes, dynamic_vertices_size_in_bytes);
        state.d_clr.resize(vertices_size_in_bytes, dynamic_vertices_size_in_bytes);
        state.d_uv.resize(vertices_size_in_bytes, dynamic_vertices_size_in_bytes);
        state.d_nrm.resize(vertices_size_in_bytes, dynamic_vertices_size_in_bytes);
        state.d_tan.resize(vertices_size_in_bytes, dynamic_vertices_size_in_bytes);
        size_t reservedCap = state.d_vertices.capacity - vertices_size_in_bytes;
        if (reservedCap > 0) {
            CUDA_CHECK(cudaMemset(reinterpret_cast<char *>((CUdeviceptr &)state.d_vertices) +
                                      vertices_size_in_bytes, 0, reservedCap));
            CUDA_CHECK(cudaMemset(reinterpret_cast<char *>((CUdeviceptr &)state.d_clr) +
                                      vertices_size_in_bytes, 0, reservedCap));
            CUDA_CHECK(cudaMemset(reinterpret_cast<char *>((CUdeviceptr &)state.d_uv) +
                                      vertices_size_in_bytes, 0, reservedCap));
            CUDA_CHECK(cudaMemset(reinterpret_cast<char *>((CUdeviceptr &)state.d_nrm) +
                                      vertices_size_in_bytes, 0, reservedCap));
            CUDA_CHECK(cudaMemset(reinterpret_cast<char *>((CUdeviceptr &)state.d_tan) +
                                      vertices_size_in_bytes, 0, reservedCap));
        }
#else
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_vertices.reset()), vertices_size_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_clr.reset()), vertices_size_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_uv.reset()), vertices_size_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_nrm.reset()), vertices_size_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_tan.reset()), vertices_size_in_bytes));
#endif
        updateRange();
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_vertices) + offset,
                              (char *)g_vertices.data() + offset, numBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_clr) + offset, (char *)g_clr.data() + offset,
                              numBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_uv) + offset, (char *)g_uv.data() + offset,
                              numBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_nrm) + offset, (char *)g_nrm.data() + offset,
                              numBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_tan) + offset, (char *)g_tan.data() + offset,
                              numBytes, cudaMemcpyHostToDevice));
        if (staticNeedUpdate && offset != 0) {
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_vertices),
                                  (char *)g_vertices.data(), static_vertices_size_in_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_clr), (char *)g_clr.data(),
                                  static_vertices_size_in_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_uv), (char *)g_uv.data(),
                                  static_vertices_size_in_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_nrm), (char *)g_nrm.data(),
                                  static_vertices_size_in_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_tan), (char *)g_tan.data(),
                                  static_vertices_size_in_bytes, cudaMemcpyHostToDevice));
        }

        const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof(uint32_t);
        const size_t light_mark_size_in_bytes = g_lightMark.size() * sizeof(unsigned short);
#if WXL
        state.d_mat_indices.resize(mat_indices_size_in_bytes);
        state.d_lightMark.resize(light_mark_size_in_bytes);
#else
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_mat_indices.reset()), mat_indices_size_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_lightMark.reset()), light_mark_size_in_bytes));
#endif
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>((CUdeviceptr)state.d_mat_indices), g_mat_indices.data(),
                              mat_indices_size_in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>((CUdeviceptr)state.d_lightMark), g_lightMark.data(),
                              light_mark_size_in_bytes, cudaMemcpyHostToDevice));
        timer.tock("done dynamic mesh update");
        std::cout << "end copy\n";
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
    //zeno::log_info("light clr after read: {} {} {}", ld.emission[0],ld.emission[1],ld.emission[2]);
    lightdats[key] = ld;
}
void update_hdr_sky(float sky_rot, zeno::vec3f sky_rot3d, float sky_strength) {
    state.params.sky_rot = sky_rot;
    state.params.sky_rot_x = sky_rot3d[0];
    state.params.sky_rot_y = sky_rot3d[1];
    state.params.sky_rot_z = sky_rot3d[2];
    state.params.sky_strength = sky_strength;
}

void using_hdr_sky(bool enable) {
    state.params.usingHdrSky = enable;
}

void update_procedural_sky(
    zeno::vec2f sunLightDir,
    float sunLightSoftness,
    zeno::vec2f windDir,
    float timeStart,
    float timeSpeed,
    float sunLightIntensity,
    float colorTemperatureMix,
    float colorTemperature
){

    auto &ud = zeno::getSession().userData();
    sunLightDir[1] = clamp(sunLightDir[1], -90.f, 90.f);
    state.params.sunLightDirY = sin(sunLightDir[1] / 180.f * M_PI);
    state.params.sunLightDirX = cos(sunLightDir[1] / 180.f * M_PI) * sin(sunLightDir[0] / 180.f * M_PI);
    state.params.sunLightDirZ = cos(sunLightDir[1] / 180.f * M_PI) * cos(sunLightDir[0] / 180.f * M_PI);

    windDir[1] = clamp(windDir[1], -90.f, 90.f);
    state.params.windDirY = sin(windDir[1] / 180.f * M_PI);
    state.params.windDirX = cos(windDir[1] / 180.f * M_PI) * sin(windDir[0] / 180.f * M_PI);
    state.params.windDirZ = cos(windDir[1] / 180.f * M_PI) * cos(windDir[0] / 180.f * M_PI);

    state.params.sunSoftness = clamp(sunLightSoftness, 0.01f, 1.0f);
    state.params.sunLightIntensity = sunLightIntensity;
    state.params.colorTemperatureMix = clamp(colorTemperatureMix, 0.00f, 1.0f);
    state.params.colorTemperature = clamp(colorTemperature, 1000.0f, 40000.0f);

    int frameid = ud.get2<int>("frameid", 0);
    state.params.elapsedTime = timeStart + timeSpeed * frameid;
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
static int uniformBufferInitialized = false;
void optixUpdateUniforms(std::vector<float4> & inConstants) {

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &state.d_uniforms.reset() ), sizeof(float4)*512));

    CUDA_CHECK(cudaMemset(reinterpret_cast<char *>((CUdeviceptr &)state.d_uniforms), 0, sizeof(float4)*512));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>((CUdeviceptr)state.d_uniforms), inConstants.data(),
                          sizeof(float4)*inConstants.size(), cudaMemcpyHostToDevice));

    uniformBufferInitialized = true;

}
void optixupdatelight() {
    camera_changed = true;

    //zeno::log_info("lights size {}", lightdats.size());

    g_lights.clear();
    g_lightMesh.clear();
    g_lightColor.clear();

    for (auto const &[key, dat]: lightdats) {
        auto &light = g_lights.emplace_back();
        light.emission = make_float3( (float)(dat.emission[0]), (float)dat.emission[1], (float)dat.emission[2] );
        //zeno::log_info("light clr after read: {} {} {}", light.emission.x,light.emission.y,light.emission.z);
        light.corner   = make_float3( dat.v0[0], dat.v0[1], dat.v0[2] );
        //zeno::log_info("light clr after read: {} {} {}", light.corner.x,light.corner.y,light.corner.z);
        light.v1       = make_float3( dat.v1[0], dat.v1[1], dat.v1[2] );
        //zeno::log_info("light clr after read: {} {} {}", light.v1.x,light.v1.y,light.v1.z);
        light.v2       = make_float3( dat.v2[0], dat.v2[1], dat.v2[2] );
        //zeno::log_info("light clr after read: {} {} {}", light.v2.x,light.v2.y,light.v2.z);
        light.normal   = make_float3( dat.normal[0], dat.normal[1], dat.normal[2] );
        //zeno::log_info("light clr after read: {} {} {}", light.normal.x,light.normal.y,light.normal.z);
        addLightMesh(light.corner, light.v2, light.v1, light.normal, light.emission);
    }

    if(g_lights.size()) {
        g_lights[0].cdf = length(cross(g_lights[0].v1, g_lights[0].v2));
        float a = g_lights[0].cdf;
        for (int l = 1; l < g_lights.size(); l++) {
            g_lights[l].cdf = g_lights[l - 1].cdf + length(cross(g_lights[l].v1, g_lights[l].v2));
        }
    }
//    for(int l=0;l<g_lights.size();l++)
//    {
//        g_lights[l].cdf /= g_lights[g_lights.size()-1].cdf;
//
//    }

    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &state.lightsbuf_p.reset() ),
                sizeof( ParallelogramLight ) * std::max(g_lights.size(),(size_t)1)
                ) );
    state.params.lights = (ParallelogramLight*)(CUdeviceptr)state.lightsbuf_p;
    if (g_lights.size())
        CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.lightsbuf_p ),
                g_lights.data(), sizeof( ParallelogramLight ) * g_lights.size(),
                cudaMemcpyHostToDevice
                ) );
}

void optixupdatematerial(std::vector<bool> const            &markers,
                         std::vector<std::string> const     &shaders, 
                         std::vector<std::vector<std::string>> &texs) 
{
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

        if (!markers[i]) {
            OptixUtil::rtMaterialShaders.emplace_back(shaders[i].c_str(), 
                                                    "__closesthit__radiance", 
                                                    "__anyhit__shadow_cutout");
        } else { 
            OptixUtil::rtMaterialShaders.emplace_back(shaders[i].c_str(), 
                                                    "__closesthit__radiance_volume", 
                                                    "__anyhit__occlusion_volume",
                                                    "__intersection__volume");
        }
        OptixUtil::rtMaterialShaders.back().has_vdb = markers[i]; 

        if(texs.size()>0){
            std::cout<<"texSize:"<<texs[i].size()<<std::endl;
            for(int j=0;j<texs[i].size();j++)
            {
                std::cout<<"texName:"<<texs[i][j]<<std::endl;
                OptixUtil::rtMaterialShaders[i].addTexture(j, texs[i][j]);
            }
        }
    }

    CppTimer theTimer;
    theTimer.tick();
    
    uint task_count = OptixUtil::rtMaterialShaders.size();
    //std::vector<tbb::task_group> task_groups(task_count);
    for(int i=0; i<task_count; ++i)
    {
        OptixUtil::_compile_group.run([&shaders, i] () {
            
            printf("now compiling %d'th shader \n", i);
            if(OptixUtil::rtMaterialShaders[i].loadProgram(i, nullptr)==false)
            {
                std::cout<<"program compile failed, using default"<<std::endl;
                
                OptixUtil::rtMaterialShaders[i].m_shaderFile     = shaders[0];
                OptixUtil::rtMaterialShaders[i].m_hittingEntry   = "";
                OptixUtil::rtMaterialShaders[i].m_shadingEntry   = "__closesthit__radiance";
                OptixUtil::rtMaterialShaders[i].m_occlusionEntry = "__anyhit__shadow_cutout";
                std::cout<<OptixUtil::rtMaterialShaders[i].loadProgram(i, nullptr)<<std::endl;
                std::cout<<"shader restored to default\n";
            }
        });
    }

    OptixUtil::_compile_group.wait();
    theTimer.tock("Done Optix Shader Compile:");

    OptixUtil::createRenderGroups(state.context, OptixUtil::ray_module);
    if (OptixUtil::sky_tex.has_value()) {
        state.params.sky_texture = OptixUtil::g_tex[OptixUtil::sky_tex.value()]->texture;
    }
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
    std::string instID;
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
std::vector<std::shared_ptr<smallMesh>> &oMeshes, int meshesStart, int vertsStart)
{
    size_t num_tri = (verts.size()-vertsStart)/3;
    oMeshes.resize(meshesStart);
    size_t tris_per_mesh = 1024;
    size_t num_iter = num_tri/tris_per_mesh + 1;
    for(int i=0; i<num_iter;i++)
    {
        auto m = std::make_shared<smallMesh>();
        for(int j=0;j<tris_per_mesh;j++)
        {
            size_t idx = i*tris_per_mesh + j;
            if(idx<num_tri){
                m->verts.emplace_back(verts[vertsStart + idx*3+0]);
                m->verts.emplace_back(verts[vertsStart + idx*3+1]);
                m->verts.emplace_back(verts[vertsStart + idx*3+2]);
                m->mat_idx.emplace_back(mat_idx[vertsStart/3 + idx]);
                m->idx.emplace_back(make_uint3(j*3+0,j*3+1,j*3+2));
            }
        }
        if(m->verts.size()>0)
            oMeshes.push_back(std::move(m));
    }
}
static void updateStaticDrawObjects() {
    g_vertices.clear();
    g_clr.clear();
    g_nrm.clear();
    g_uv.clear();
    g_tan.clear();
    g_mat_indices.clear();
    g_lightMark.clear();
    size_t n = 0;
    for (auto const &[key, dat]: drawdats) {
        if(key.find(":static:")!=key.npos && dat.instID == "Default")
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
        if(key.find(":static:")!=key.npos && dat.instID == "Default") {
            auto it = g_mtlidlut.find(dat.mtlid);
            int mtlindex = it != g_mtlidlut.end() ? it->second : 0;
            //zeno::log_error("{} {}", dat.mtlid, mtlindex);
            //#pragma omp parallel for
            for (size_t i = 0; i < dat.tris.size() / 3; i++) {
                g_mat_indices[n + i] = mtlindex;
                g_lightMark[n + i] = 0;
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
    }
}
static void updateDynamicDrawObjects() {
    size_t n = 0;
    for (auto const &[key, dat]: drawdats) {
        if(key.find(":static:")==key.npos && dat.instID == "Default")
            n += dat.tris.size()/3;
    }

    g_vertices.resize(g_staticVertNum + n * 3);
    g_clr.resize(g_staticVertNum + n*3);
    g_nrm.resize(g_staticVertNum + n*3);
    g_uv.resize(g_staticVertNum + n*3);
    g_tan.resize(g_staticVertNum + n*3);
    g_mat_indices.resize(g_staticVertNum/3 + n);
    g_lightMark.resize(g_staticVertNum/3 + n);
    n = 0;
    for (auto const &[key, dat]: drawdats) {
        if(key.find(":static:")==key.npos && dat.instID == "Default") {
            auto it = g_mtlidlut.find(dat.mtlid);
            int mtlindex = it != g_mtlidlut.end() ? it->second : 0;
            //zeno::log_error("{} {}", dat.mtlid, mtlindex);
            //#pragma omp parallel for
            for (size_t i = 0; i < dat.tris.size() / 3; i++) {
                g_mat_indices[g_staticVertNum/3 + n + i] = mtlindex;
                g_lightMark[g_staticVertNum/3 + n + i] = 0;
                g_vertices[g_staticVertNum + (n + i) * 3 + 0] = {
                    dat.verts[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.verts[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.verts[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                g_vertices[g_staticVertNum + (n + i) * 3 + 1] = {
                    dat.verts[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.verts[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.verts[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                g_vertices[g_staticVertNum + (n + i) * 3 + 2] = {
                    dat.verts[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.verts[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.verts[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };

                g_clr[g_staticVertNum + (n + i) * 3 + 0] = {
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                g_clr[g_staticVertNum + (n + i) * 3 + 1] = {
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                g_clr[g_staticVertNum + (n + i) * 3 + 2] = {
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };

                g_nrm[g_staticVertNum + (n + i) * 3 + 0] = {
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                g_nrm[g_staticVertNum + (n + i) * 3 + 1] = {
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                g_nrm[g_staticVertNum + (n + i) * 3 + 2] = {
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };

                g_uv[g_staticVertNum + (n + i) * 3 + 0] = {
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                g_uv[g_staticVertNum + (n + i) * 3 + 1] = {
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                g_uv[g_staticVertNum + (n + i) * 3 + 2] = {
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };

                g_tan[g_staticVertNum + (n + i) * 3 + 0] = {
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                g_tan[g_staticVertNum + (n + i) * 3 + 1] = {
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                g_tan[g_staticVertNum + (n + i) * 3 + 2] = {
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };
            }
            n += dat.tris.size() / 3;
        }
    }
//    size_t ltris = g_lightMesh.size()/3;
//    for(int l=0;l<ltris;l++)
//    {
//        g_vertices.push_back(g_lightMesh[l*3+0]);
//        g_vertices.push_back(g_lightMesh[l*3+1]);
//        g_vertices.push_back(g_lightMesh[l*3+2]);
//        auto l1 = g_lightMesh[l*3+1] - g_lightMesh[l*3+0];
//        auto l2 = g_lightMesh[l*3+2] - g_lightMesh[l*3+0];
//        auto zl1= zeno::vec3f(l1.x, l1.y, l1.z);
//        auto zl2= zeno::vec3f(l2.x, l2.y, l2.z);
//        auto normal = zeno::normalize(zeno::cross(zl1,zl2));
//        g_nrm.push_back(make_float4(normal[0], normal[1], normal[2],0));
//        g_nrm.push_back(make_float4(normal[0], normal[1], normal[2],0));
//        g_nrm.push_back(make_float4(normal[0], normal[1], normal[2],0));
//        g_clr.push_back(g_lightColor[l*3+0]);
//        g_clr.push_back(g_lightColor[l*3+1]);
//        g_clr.push_back(g_lightColor[l*3+2]);
//        g_tan.push_back(make_float4(0));
//        g_tan.push_back(make_float4(0));
//        g_tan.push_back(make_float4(0));
//        g_uv.push_back( make_float4(0));
//        g_uv.push_back( make_float4(0));
//        g_uv.push_back( make_float4(0));
//        g_mat_indices.push_back(0);
//        g_lightMark.push_back(1);
//    }


}
static void updateStaticDrawInstObjects()
{
    g_instLUT.clear();
    std::unordered_map<std::string, std::size_t> numVertsLUT;
    for (const auto &[key, dat] : drawdats)
    {
        if (key.find(":static:") != key.npos && dat.instID != "Default")
        {
            numVertsLUT[dat.instID] += dat.tris.size() / 3;
        }
    }
    for (auto &[instID, numVerts] : numVertsLUT)
    {
        auto& instData = g_instLUT[instID];
        auto &vertices = instData.vertices;
        auto &clr = instData.clr;
        auto &nrm = instData.nrm;
        auto &uv = instData.uv;
        auto &tan = instData.tan;
        auto &mat_indices = instData.mat_indices;
        auto &lightMark = instData.lightMark;

        vertices.resize(numVerts * 3);
        clr.resize(numVerts * 3);
        nrm.resize(numVerts * 3);
        uv.resize(numVerts * 3);
        tan.resize(numVerts * 3);
        mat_indices.resize(numVerts);
        lightMark.resize(numVerts);

        numVerts = 0;
    }
    for (const auto &[key, dat] : drawdats)
    {
        if (key.find(":static:") != key.npos && dat.instID != "Default")
        {
            auto &numVerts = numVertsLUT[dat.instID];
            auto &instData = g_instLUT[dat.instID];
            auto &vertices = instData.vertices;
            auto &clr = instData.clr;
            auto &nrm = instData.nrm;
            auto &uv = instData.uv;
            auto &tan = instData.tan;
            auto &mat_indices = instData.mat_indices;
            auto &lightMark = instData.lightMark;

            auto it = g_mtlidlut.find(dat.mtlid);
            int mtlindex = it != g_mtlidlut.end() ? it->second : 0;
            //zeno::log_error("{} {}", dat.mtlid, mtlindex);
            //#pragma omp parallel for
            for (std::size_t i = 0; i < dat.tris.size() / 3; ++i)
            {
                mat_indices[numVerts + i] = mtlindex;
                lightMark[numVerts + i] = 0;
                vertices[(numVerts + i) * 3 + 0] = {
                    dat.verts[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.verts[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.verts[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                vertices[(numVerts + i) * 3 + 1] = {
                    dat.verts[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.verts[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.verts[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                vertices[(numVerts + i) * 3 + 2] = {
                    dat.verts[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.verts[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.verts[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };

                clr[(numVerts + i) * 3 + 0] = {
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                clr[(numVerts + i) * 3 + 1] = {
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                clr[(numVerts + i) * 3 + 2] = {
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };

                nrm[(numVerts + i) * 3 + 0] = {
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                nrm[(numVerts + i) * 3 + 1] = {
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                nrm[(numVerts + i) * 3 + 2] = {
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };

                uv[(numVerts+ i) * 3 + 0] = {
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                uv[(numVerts + i) * 3 + 1] = {
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                uv[(numVerts + i) * 3 + 2] = {
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };

                tan[(numVerts + i) * 3 + 0] = {
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                tan[(numVerts + i) * 3 + 1] = {
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                tan[(numVerts + i) * 3 + 2] = {
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };
            }
            numVerts += dat.tris.size() / 3;
        }
    }
}
static void updateDynamicDrawInstObjects()
{
    std::unordered_map<std::string, std::size_t> numVertsLUT;
    for (const auto &[key, dat] : drawdats)
    {
        if (key.find(":static:") == key.npos && dat.instID != "Default")
        {
            numVertsLUT[dat.instID] += dat.tris.size() / 3;
        }
    }
    for (auto &[instID, numVerts] : numVertsLUT)
    {
        auto& instData = g_instLUT[instID];
        auto &vertices = instData.vertices;
        auto &clr = instData.clr;
        auto &nrm = instData.nrm;
        auto &uv = instData.uv;
        auto &tan = instData.tan;
        auto &mat_indices = instData.mat_indices;
        auto &lightMark = instData.lightMark;
        auto &staticVertNum = instData.staticVertNum;

        vertices.resize(staticVertNum + numVerts * 3);
        clr.resize(staticVertNum + numVerts * 3);
        nrm.resize(staticVertNum + numVerts * 3);
        uv.resize(staticVertNum + numVerts * 3);
        tan.resize(staticVertNum + numVerts * 3);
        mat_indices.resize(staticVertNum / 3 + numVerts);
        lightMark.resize(staticVertNum / 3 + numVerts);

        numVerts = 0;
    }
    for (const auto &[key, dat] : drawdats)
    {
        if (key.find(":static:") == key.npos && dat.instID != "Default")
        {
            auto &numVerts = numVertsLUT[dat.instID];
            auto &instData = g_instLUT[dat.instID];
            auto &vertices = instData.vertices;
            auto &clr = instData.clr;
            auto &nrm = instData.nrm;
            auto &uv = instData.uv;
            auto &tan = instData.tan;
            auto &mat_indices = instData.mat_indices;
            auto &lightMark = instData.lightMark;
            auto &staticVertNum = instData.staticVertNum;

            auto it = g_mtlidlut.find(dat.mtlid);
            int mtlindex = it != g_mtlidlut.end() ? it->second : 0;
            //zeno::log_error("{} {}", dat.mtlid, mtlindex);
            //#pragma omp parallel for
            for (std::size_t i = 0; i < dat.tris.size() / 3; ++i)
            {
                mat_indices[staticVertNum / 3 + numVerts + i] = mtlindex;
                lightMark[staticVertNum / 3 + numVerts + i] = 0;
                vertices[staticVertNum + (numVerts + i) * 3 + 0] = {
                    dat.verts[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.verts[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.verts[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                vertices[staticVertNum + (numVerts + i) * 3 + 1] = {
                    dat.verts[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.verts[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.verts[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                vertices[staticVertNum + (numVerts + i) * 3 + 2] = {
                    dat.verts[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.verts[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.verts[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };

                clr[staticVertNum + (numVerts + i) * 3 + 0] = {
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                clr[staticVertNum + (numVerts + i) * 3 + 1] = {
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                clr[staticVertNum + (numVerts + i) * 3 + 2] = {
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };

                nrm[staticVertNum + (numVerts + i) * 3 + 0] = {
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                nrm[staticVertNum + (numVerts + i) * 3 + 1] = {
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                nrm[staticVertNum + (numVerts + i) * 3 + 2] = {
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };

                uv[staticVertNum + (numVerts+ i) * 3 + 0] = {
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                uv[staticVertNum + (numVerts + i) * 3 + 1] = {
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                uv[staticVertNum + (numVerts + i) * 3 + 2] = {
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };

                tan[staticVertNum + (numVerts + i) * 3 + 0] = {
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                };
                tan[staticVertNum + (numVerts + i) * 3 + 1] = {
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                };
                tan[staticVertNum + (numVerts + i) * 3 + 2] = {
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                };
            }
            numVerts += dat.tris.size() / 3;
        }
    }
}

void load_object(std::string const &key, std::string const &mtlid, const std::string &instID, float const *verts, size_t numverts, int const *tris, size_t numtris, std::map<std::string, std::pair<float const *, size_t>> const &vtab) {
    DrawDat &dat = drawdats[key];
    //ZENO_P(mtlid);
    dat.mtlid = mtlid;
    dat.instID = instID;
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

struct InstTrs
{
    std::string instID;
    std::string onbType;

    std::vector<float> pos;
    std::vector<float> uv;
    std::vector<float> nrm;
    std::vector<float> clr;
    std::vector<float> tang;
};

static std::unordered_map<std::string, InstTrs> instTrsLUT;

void load_inst(const std::string &key, const std::string &instID, const std::string &onbType, std::size_t numInsts, const float *pos, const float *nrm, const float *uv, const float *clr, const float *tang)
{
    InstTrs &instTrs = instTrsLUT[key];
    instTrs.instID = instID;
    instTrs.onbType = onbType;

    instTrs.pos.assign(pos, pos + numInsts * 3);
    instTrs.nrm.assign(nrm, nrm + numInsts * 3);
    instTrs.uv.assign(uv, uv + numInsts * 3);
    instTrs.clr.assign(clr, clr + numInsts * 3);
    instTrs.tang.assign(tang, tang + numInsts * 3);
}

void unload_inst(const std::string &key)
{
    instTrsLUT.erase(key);
}

void UpdateInst()
{
    for (auto &[key, instTrs] : instTrsLUT)
    {
        const auto& instID = instTrs.instID;
        auto& instMat = g_instMatsLUT[instID];
        auto& instAttr = g_instAttrsLUT[instID];

        const auto& numInstMats = instTrs.pos.size() / 3;
        instMat.resize(numInstMats);
        instAttr.pos.resize(numInstMats);
        instAttr.nrm.resize(numInstMats);
        instAttr.uv.resize(numInstMats);
        instAttr.clr.resize(numInstMats);
        instAttr.tang.resize(numInstMats);
#pragma omp parallel for
        for (int i = 0; i < numInstMats; ++i)
        {
            auto translateMat = glm::translate(glm::vec3(instTrs.pos[3 * i + 0], instTrs.pos[3 * i + 1], instTrs.pos[3 * i + 2]));

            zeno::vec3f t0 = {instTrs.nrm[3 * i + 0], instTrs.nrm[3 * i + 1], instTrs.nrm[3 * i + 2]};
            zeno::vec3f t1 = {instTrs.clr[3 * i + 0], instTrs.clr[3 * i + 1], instTrs.clr[3 * i + 2]};
            t0 = normalizeSafe(t0);
            zeno::vec3f t2;
            zeno::guidedPixarONB(t0, t1, t2);
            glm::mat4x4 rotateMat(1);
            if (instTrs.onbType == "XYZ")
            {
                rotateMat[0][0] = t2[0];
                rotateMat[0][1] = t2[1];
                rotateMat[0][2] = t2[2];
                rotateMat[1][0] = t1[0];
                rotateMat[1][1] = t1[1];
                rotateMat[1][2] = t1[2];
                rotateMat[2][0] = t0[0];
                rotateMat[2][1] = t0[1];
                rotateMat[2][2] = t0[2];
            }
            else if (instTrs.onbType == "YXZ")
            {
                rotateMat[0][0] = t1[0];
                rotateMat[0][1] = t1[1];
                rotateMat[0][2] = t1[2];
                rotateMat[1][0] = t2[0];
                rotateMat[1][1] = t2[1];
                rotateMat[1][2] = t2[2];
                rotateMat[2][0] = t0[0];
                rotateMat[2][1] = t0[1];
                rotateMat[2][2] = t0[2];
            }
            else if (instTrs.onbType == "YZX")
            {
                rotateMat[0][0] = t1[0];
                rotateMat[0][1] = t1[1];
                rotateMat[0][2] = t1[2];
                rotateMat[1][0] = t0[0];
                rotateMat[1][1] = t0[1];
                rotateMat[1][2] = t0[2];
                rotateMat[2][0] = t2[0];
                rotateMat[2][1] = t2[1];
                rotateMat[2][2] = t2[2];
            }
            else if (instTrs.onbType == "ZYX")
            {
                rotateMat[0][0] = t0[0];
                rotateMat[0][1] = t0[1];
                rotateMat[0][2] = t0[2];
                rotateMat[1][0] = t1[0];
                rotateMat[1][1] = t1[1];
                rotateMat[1][2] = t1[2];
                rotateMat[2][0] = t2[0];
                rotateMat[2][1] = t2[1];
                rotateMat[2][2] = t2[2];
            }
            else if (instTrs.onbType == "ZXY")
            {
                rotateMat[0][0] = t0[0];
                rotateMat[0][1] = t0[1];
                rotateMat[0][2] = t0[2];
                rotateMat[1][0] = t2[0];
                rotateMat[1][1] = t2[1];
                rotateMat[1][2] = t2[2];
                rotateMat[2][0] = t1[0];
                rotateMat[2][1] = t1[1];
                rotateMat[2][2] = t1[2];
            }
            else
            {
                rotateMat[0][0] = t2[0];
                rotateMat[0][1] = t2[1];
                rotateMat[0][2] = t2[2];
                rotateMat[1][0] = t0[0];
                rotateMat[1][1] = t0[1];
                rotateMat[1][2] = t0[2];
                rotateMat[2][0] = t1[0];
                rotateMat[2][1] = t1[1];
                rotateMat[2][2] = t1[2];
            }

            auto scaleMat = glm::scale(glm::vec3(1, 1, 1));
            instMat[i] = translateMat * rotateMat * scaleMat;
            instAttr.pos[i].x = instTrs.pos[3 * i + 0];
            instAttr.pos[i].y = instTrs.pos[3 * i + 1];
            instAttr.pos[i].z = instTrs.pos[3 * i + 2];
            instAttr.nrm[i].x = instTrs.nrm[3 * i + 0];
            instAttr.nrm[i].y = instTrs.nrm[3 * i + 1];
            instAttr.nrm[i].z = instTrs.nrm[3 * i + 2];
            instAttr.uv[i].x = instTrs.uv[3 * i + 0];
            instAttr.uv[i].y = instTrs.uv[3 * i + 1];
            instAttr.uv[i].z = instTrs.uv[3 * i + 2];
            instAttr.clr[i].x = instTrs.clr[3 * i + 0];
            instAttr.clr[i].y = instTrs.clr[3 * i + 1];
            instAttr.clr[i].z = instTrs.clr[3 * i + 2];
            instAttr.tang[i].x = instTrs.tang[3 * i + 0];
            instAttr.tang[i].y = instTrs.tang[3 * i + 1];
            instAttr.tang[i].z = instTrs.tang[3 * i + 2];
        }
    }
}

void set_window_size(int nx, int ny) {
    state.params.width = nx;
    state.params.height = ny;
    camera_changed = true;
    resize_dirty = true;
}

void set_perspective(float const *U, float const *V, float const *W, float const *E, float aspect, float fov, float fpd, float aperture) {
    auto &cam = state.params.cam;
    //float c_aspect = fw/fh;
    //float u_aspect = aspect;
    //float r_fh = fh * 0.001;
    //float r_fw = fw * 0.001;
    //zeno::log_info("Camera film w {} film h {} aspect {} {}", fw, fh, u_aspect, c_aspect);

    cam.eye = make_float3(E[0], E[1], E[2]);
    cam.right = normalize(make_float3(U[0], U[1], U[2]));
    cam.up = normalize(make_float3(V[0], V[1], V[2]));
    cam.front = normalize(make_float3(W[0], W[1], W[2]));

    float radfov = fov * float(M_PI) / 180;
    float tanfov = std::tan(radfov / 2.0f);
    cam.front /= tanfov;
    cam.right *= aspect;

    camera_changed = true;
    //cam.aspect = aspect;
    //cam.fov = fov;
    //camera.setZxxViewMatrix(U, V, W);
    //camera.setAspectRatio(aspect);
    //camera.setFovY(fov * aspect * (float)M_PI / 180.0f);

    cam.focalPlaneDistance = fpd;
    cam.aperture = aperture;
}


void optixrender(int fbo, int samples, bool simpleRender) {
    samples = zeno::envconfig::getInt("SAMPLES", samples);
    // 张心欣老爷请添加环境变量：export ZENO_SAMPLES=256
    zeno::log_debug("rendering samples {}", samples);
    state.params.simpleRender = false;//simpleRender;
    if (!output_buffer_o) throw sutil::Exception("no output_buffer_o");
    if (!gl_display_o) throw sutil::Exception("no gl_display_o");
    updateState( *output_buffer_o, state.params );
    const int max_samples_once = 16;

    for (int f = 0; f < samples; f += max_samples_once) { // 张心欣不要改这里
        state.params.samples_per_launch = std::min(samples - f, max_samples_once);
        launchSubframe( *output_buffer_o, state );
        state.params.subframe_index++;
    }
    displaySubframe( *output_buffer_o, *gl_display_o, state, fbo );
    auto &ud = zeno::getSession().userData();
    if (ud.has("optix_image_path")) {
        auto path = ud.get2<std::string>("optix_image_path");
        auto p = (*output_buffer_o).getHostPointer();
        auto w = (*output_buffer_o).width();
        auto h = (*output_buffer_o).height();
        stbi_flip_vertically_on_write(true);
        stbi_write_jpg(path.c_str(), w, h, 4, p, 100);
        zeno::log_info("optix: saving screenshot {}x{} to {}", w, h, path);
        ud.erase("optix_image_path");
    }
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
    using namespace OptixUtil;
    try {
        CUDA_SYNC_CHECK();
        cleanupState( state );
        rtMaterialShaders.clear();
    } catch (sutil::Exception const &e) {
        std::cout << "OptixCleanupError: " << e.what() << std::endl;
        std::memset((void *)&state, 0, sizeof(state));
        std::memset((void *)&rtMaterialShaders[0], 0, sizeof(rtMaterialShaders[0]) * rtMaterialShaders.size());

             context                  .handle=0;
pipeline                 .handle=0;
ray_module               .handle=0;
raygen_prog_group        .handle=0;
radiance_miss_group      .handle=0;
occlusion_miss_group     .handle=0;

    }
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
