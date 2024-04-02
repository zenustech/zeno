// this part of code is modified from nvidia's optix example
#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <memory>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <sampleConfig.h>

#include <stdint.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>
#include <stb_image_write.h>
#include <tuple>
#ifdef __linux__
#include <unistd.h>
#include <stdio.h>
#endif

//#include <GLFW/glfw3.h>
#include "XAS.h"
#include "magic_enum.hpp"
#include "optixPathTracer.h"

#include <zeno/para/parallel_sort.h>
#include <zeno/para/parallel_scan.h>
#include <zeno/utils/log.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/UserData.h>
#include "optixSphere.h"
#include "optixVolume.h"
#include "zeno/core/Session.h"

#include <algorithm>
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
#include <zeno/utils/string.h>
#include <zeno/utils/envconfig.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/types/LightObject.h>

#include <tinygltf/json.hpp>
#include <unordered_map>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "LightBounds.h"
#include "LightTree.h"

#include "TypeCaster.h"

#include "LightBounds.h"
#include "LightTree.h"

#include "ChiefDesignerEXR.h"
using namespace zeno::ChiefDesignerEXR;

#include "zeno/utils/image_proc.h"

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

// Camera state
bool             camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// Mouse state
int32_t mouse_button = -1;

//int32_t samples_per_launch = 16;
//int32_t samples_per_launch = 16;

std::vector<std::shared_ptr<VolumeWrapper>> list_volume;
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

struct EmptyData {};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;

typedef Record<HitGroupData> HitGroupRecord;
typedef Record<EmptyData>   CallablesRecord;


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
#ifdef USE_SHORT_COMPACT
uchar4 toRGBA8(float4 in)
{
  return make_uchar4((unsigned char)(in.x*256.0),
                     (unsigned char)(in.y*256.0),
                     (unsigned char)(in.z*256.0),
                     (unsigned char)(in.w*256.0));
}
ushort2 rgba8ToUshort2(uchar4 in)
{
    uchar4 rgba = toRGBA8(in);
    unsigned short x = rgba.x;
    x = x<<16 + rgba.y;
    unsigned short y = rgba.z<<16 + rgba.w;
}
ushort2 toHalfColor(float4 in)
{
  return make_ushort3((unsigned short)(in.x*65536.0f),
                      (unsigned short)(in.y*65536.0f),
                      (unsigned short)(in.z*65536.0f));
}

ushort2 toHalf(float4 in)
{
  return make_ushort3((unsigned short)(in.x*65536.0f),
                      (unsigned short)(in.y*65536.0f),
                      (unsigned short)(in.z*65536.0f));
}

ushort2 halfNormal(float4 in)
{
  float3 val = make_float3((in.x + 1.0f)/2.0f,
                           (in.y + 1.0f)/2.0f,
                           (in.z + 1.0f)/2.0f);

  return make_ushort3((unsigned short)(val.x*65536.0f),
                      (unsigned short)(val.y*65536.0f),
                      (unsigned short)(val.z*65536.0f));
}
#else
  #ifdef USE_SHORT
    ushort3 toHalfColor(float4 in)
    {
        
      return make_ushort3((unsigned short)(in.x*65536.0f),
                          (unsigned short)(in.y*65536.0f),
                          (unsigned short)(in.z*65536.0f));
    }

    ushort3 halfNormal(float4 in)
    {
      float3 val = make_float3((in.x + 1.0f)/2.0f,
                               (in.y + 1.0f)/2.0f,
                               (in.z + 1.0f)/2.0f);
      //val = normalize(val);

      return make_ushort3((unsigned short)(val.x*65536.0f),
                          (unsigned short)(val.y*65536.0f),
                          (unsigned short)(val.z*65536.0f));
    }
  #else
    float4 toHalfColor(float4 in)
    {
      return in;
    }
    float4 toHalf(float4 in)
    {
      return in;
    }

    float4 halfNormal(float4 in)
    {
      return in;
    }
  #endif
#endif

std::optional<sutil::CUDAOutputBuffer<uchar4>> output_buffer_o;
std::optional<sutil::CUDAOutputBuffer<float3>> output_buffer_color;
std::optional<sutil::CUDAOutputBuffer<float3>> output_buffer_diffuse;
std::optional<sutil::CUDAOutputBuffer<float3>> output_buffer_specular;
std::optional<sutil::CUDAOutputBuffer<float3>> output_buffer_transmit;
std::optional<sutil::CUDAOutputBuffer<float3>> output_buffer_background;
std::optional<sutil::CUDAOutputBuffer<float3>> output_buffer_mask;
using Vertex = float4;

struct PathTracerState
{
    OptixDeviceContext context = 0;

    OptixTraversableHandle         rootHandleIAS;
    raii<CUdeviceptr>              rootBufferIAS;

    OptixTraversableHandle         meshHandleIAS;
    raii<CUdeviceptr>              meshBufferIAS;

    OptixTraversableHandle         gas_handle               = {};  // Traversable handle for triangle AS
    raii<CUdeviceptr>              d_gas_output_buffer;  // Triangle AS memory

    raii<CUdeviceptr>              d_vertices;
    raii<CUdeviceptr>              d_clr;
    raii<CUdeviceptr>              d_nrm;
    raii<CUdeviceptr>              d_uv;
    raii<CUdeviceptr>              d_tan;
    raii<CUdeviceptr>              d_lightMark;
    raii<CUdeviceptr>              d_mat_indices;

    raii<CUdeviceptr>              vertexAuxOffsetGlobal;
    
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
    raii<CUdeviceptr> albedo_buffer_p;
    raii<CUdeviceptr> normal_buffer_p;

    raii<CUdeviceptr> accum_buffer_d;
    raii<CUdeviceptr> accum_buffer_s;
    raii<CUdeviceptr> accum_buffer_t;
    raii<CUdeviceptr> accum_buffer_b;
    raii<CUdeviceptr> lightsbuf_p;
    raii<CUdeviceptr> sky_cdf_p;
    raii<CUdeviceptr> sky_start;
    Params                         params;
    raii<CUdeviceptr>                        d_params;
    CUdeviceptr                              d_params2=0;

    raii<CUdeviceptr>  d_raygen_record;
    raii<CUdeviceptr>  d_miss_records;
    raii<CUdeviceptr>  d_hitgroup_records;
    raii<CUdeviceptr>  d_callable_records;    

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
    ~smallMesh(){d_gas_output_buffer.reset(); dverts.reset(); dmats.reset(); 
        didx.reset();}
};

//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

//const int32_t TRIANGLE_COUNT = 32;
//const int32_t MAT_COUNT      = 5;
#ifdef USE_SHORT
    static std::vector<Vertex> g_vertices= // TRIANGLE_COUNT*3
        {
            {0,0,0},
            {0,0,0},
            {0,0,0},
    };
    static std::vector<ushort3> g_clr= // TRIANGLE_COUNT*3
        {
            {0,0,0},
            {0,0,0},
            {0,0,0},
    };
    static std::vector<ushort3> g_nrm= // TRIANGLE_COUNT*3
        {
            {0,0,0},
            {0,0,0},
            {0,0,0},
    };
    static std::vector<ushort3> g_uv= // TRIANGLE_COUNT*3
        {
            {0,0,0},
            {0,0,0},
            {0,0,0},
    };
    static std::vector<ushort3> g_tan= // TRIANGLE_COUNT*3
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
#else
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
#endif

static void compact_triangle_vertex_attribute(const std::vector<Vertex>& attrib, std::vector<Vertex>& compactAttrib, std::vector<unsigned int>& vertIdsPerTri) {
    using id_t = unsigned int;
    using kv_t = std::pair<Vertex, id_t>;
    
    std::vector<kv_t> kvs(attrib.size());
#pragma omp parallel for
    for (auto i = 0; i < attrib.size(); ++i)
        kvs[i] = std::make_pair(attrib[i], (id_t)i);

    // sort
    auto compOp = [](const kv_t &a, const kv_t &b) {
        if (a.first.x < b.first.x) return true;
        else if (a.first.x > b.first.x) return false;
        if (a.first.y < b.first.y) return true;
        else if (a.first.y > b.first.y) return false;
        if (a.first.z < b.first.z) return true;

        return false;
    };
    zeno::parallel_sort(std::begin(kvs), std::end(kvs), compOp);

    // excl scan
    std::vector<id_t> mark(kvs.size()), offset(kvs.size());
#pragma omp parallel for
    for (auto i = /*not 0*/1; i < kvs.size(); ++i)
        if (kvs[i].first.x == kvs[i - 1].first.x && 
            kvs[i].first.y == kvs[i - 1].first.y && 
            kvs[i].first.z == kvs[i - 1].first.z)
            mark[i] = 1;
    zeno::parallel_inclusive_scan_sum(std::begin(mark), std::end(mark), 
        std::begin(offset), [](const auto &v) { return v; });
    auto numNewAttribs = offset.back() + 1;
    mark[0] = 1;

    compactAttrib.resize(numNewAttribs);
    vertIdsPerTri.resize(attrib.size());
#pragma omp parallel for
    for (auto i = 0; i < kvs.size(); ++i) {
        auto originalIndex = kvs[i].second;
        auto newIndex = offset[i];
        vertIdsPerTri[originalIndex] = newIndex;
        if (mark[i]) 
            compactAttrib[offset[i]] = kvs[i].first;
    }
}

struct LightsWrapper {
    std::vector<Vertex> _planeLightGeo;
    std::vector<Vertex> _sphereLightGeo;
    std::vector<float3> _triangleLightGeo;
    std::vector<GenericLight> g_lights;

    OptixTraversableHandle   lightPlanesGas{};
    raii<CUdeviceptr>  lightPlanesGasBuffer{};

    OptixTraversableHandle  lightSpheresGas{};
    raii<CUdeviceptr> lightSpheresGasBuffer{};

    OptixTraversableHandle  lightTrianglesGas{};
    raii<CUdeviceptr> lightTrianglesGasBuffer{};

    raii<CUdeviceptr> lightBitTrailsPtr;
    raii<CUdeviceptr> lightTreeNodesPtr;
    raii<CUdeviceptr> lightTreeDummyPtr;

    raii<CUdeviceptr> triangleLightCoords;
    raii<CUdeviceptr> triangleLightNormals;

    void reset() { *this = {}; }

} lightsWrapper;

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
std::unordered_map<std::string, std::vector<float>> g_instScaleLUT;
struct InstAttr
{
    std::vector<float3> pos;
    std::vector<float3> nrm;
    std::vector<float3> uv;
    std::vector<float3> clr;
    std::vector<float3> tang;
};
std::unordered_map<std::string, InstAttr> g_instAttrsLUT;




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
    state.params.handle = state.rootHandleIAS;

    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &state.accum_buffer_p.reset() ),
                state.params.width * state.params.height * sizeof( float4 )
                ) );
    state.params.accum_buffer = (float4*)(CUdeviceptr)state.accum_buffer_p;

    auto& params = state.params;

    CUDA_CHECK( cudaMallocManaged(
            reinterpret_cast<void**>( &state.albedo_buffer_p.reset()),
            params.width * params.height * sizeof( float3 )
            ) );
    state.params.albedo_buffer = (float3*)(CUdeviceptr)state.albedo_buffer_p;
    
    CUDA_CHECK( cudaMallocManaged(
            reinterpret_cast<void**>( &state.normal_buffer_p.reset()),
            params.width * params.height * sizeof( float3 )
            ) );
    state.params.normal_buffer = (float3*)(CUdeviceptr)state.normal_buffer_p;
    
    state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    //state.params.samples_per_launch = samples_per_launch;
    state.params.subframe_index     = 0u;
}

static void handleCameraUpdate( Params& params )
{
    if( !camera_changed )
        return;
    camera_changed = false;
    updateRootIAS();
    //params.vp1 = cam_vp1;
    //params.vp2 = cam_vp2;
    //params.vp3 = cam_vp3;
    //params.vp4 = cam_vp4;

    camera.setAspectRatio( static_cast<float>( params.windowSpace.x ) / static_cast<float>( params.windowSpace.y ) );

    //params.eye = camera.eye();
    //camera.UVWFrame( params.U, params.V, params.W );
}


static void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );
    (*output_buffer_color).resize( params.width, params.height );
    (*output_buffer_diffuse).resize( params.width, params.height );
    (*output_buffer_specular).resize( params.width, params.height );
    (*output_buffer_transmit).resize( params.width, params.height );
    (*output_buffer_background).resize( params.width, params.height );
    (*output_buffer_mask).resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.accum_buffer_p .reset()),
        params.width * params.height * sizeof( float4 )
            ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.accum_buffer_d .reset()),
        params.width * params.height * sizeof( float4 )
            ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.accum_buffer_s .reset()),
        params.width * params.height * sizeof( float4 )
            ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.accum_buffer_t .reset()),
        params.width * params.height * sizeof( float4 )
            ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.accum_buffer_b .reset()),
        params.width * params.height * sizeof( float4 )
            ) );
    state.params.accum_buffer = (float4*)(CUdeviceptr)state.accum_buffer_p;

    CUDA_CHECK( cudaMallocManaged(
                reinterpret_cast<void**>( &state.albedo_buffer_p.reset()),
                params.width * params.height * sizeof( float3 )
                ) );
    state.params.albedo_buffer = (float3*)(CUdeviceptr)state.albedo_buffer_p;
    
    CUDA_CHECK( cudaMallocManaged(
                reinterpret_cast<void**>( &state.normal_buffer_p.reset()),
                params.width * params.height * sizeof( float3 )
                ) );
    state.params.normal_buffer = (float3*)(CUdeviceptr)state.normal_buffer_p;
    
    state.params.accum_buffer_D = (float4*)(CUdeviceptr)state.accum_buffer_d;
    state.params.accum_buffer_S = (float4*)(CUdeviceptr)state.accum_buffer_s;
    state.params.accum_buffer_T = (float4*)(CUdeviceptr)state.accum_buffer_t;
    state.params.accum_buffer_B = (float4*)(CUdeviceptr)state.accum_buffer_b;
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


static void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, PathTracerState& state, bool denoise)
{
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer  = result_buffer_data;
    state.params.frame_buffer_C = (*output_buffer_color     ).map();
    state.params.frame_buffer_D = (*output_buffer_diffuse   ).map();
    state.params.frame_buffer_S = (*output_buffer_specular  ).map();
    state.params.frame_buffer_T = (*output_buffer_transmit  ).map();
    state.params.frame_buffer_B = (*output_buffer_background).map();
    state.params.frame_buffer_M = (*output_buffer_mask      ).map();
    state.params.num_lights = lightsWrapper.g_lights.size();
    state.params.denoise = denoise;
    for(int j=0;j<1;j++){
      for(int i=0;i<1;i++){
        state.params.tile_i = i;
        state.params.tile_j = j;
        state.params.tile_w = state.params.windowSpace.x;
        state.params.tile_h = state.params.windowSpace.y;

        //CUDA_SYNC_CHECK();
        CUDA_CHECK( cudaMemcpy((void*)state.d_params2 ,
                    &state.params, sizeof( Params ),
                    cudaMemcpyHostToDevice
                    ) );

        //CUDA_SYNC_CHECK();

            /* printf("mama%d\n", std::this_thread::get_id()); */
            /* fflush(stdout); */
        OPTIX_CHECK( optixLaunch(
                    state.pipeline,
                    0,
                    (CUdeviceptr)state.d_params2,
                    sizeof( Params ),
                    &state.sbt,
                    state.params.tile_w,   // launch width
                    state.params.tile_h,  // launch height
                    1                     // launch depth
                    ) );
      }
    }
    output_buffer.unmap();
    (*output_buffer_color   ).unmap();
    (*output_buffer_diffuse   ).unmap();
    (*output_buffer_specular  ).unmap();
    (*output_buffer_transmit  ).unmap();
    (*output_buffer_background).unmap();
    (*output_buffer_mask      ).unmap();

    try {
        CUDA_SYNC_CHECK();
    } 
    catch(std::exception const& e)
    {
        std::cout << "Exception: " << e.what() << "\n";
    } 
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
    //output_buffer.getBuffer();
    //output_buffer_o.getHostPointer();
}

void updateSphereXAS() {
    timer.tick();
    //cleanupSpheresGPU();

    buildInstancedSpheresGAS(state.context, sphereInstanceGroupAgentList);

    if (uniformed_sphere_gas_handle == 0 && !xinxinoptix::SphereTransformedTable.empty()) { 
        buildUniformedSphereGAS(state.context, uniformed_sphere_gas_handle, uniformed_sphere_gas_buffer);
    }

    std::vector<OptixInstance> optix_instances; 
    optix_instances.reserve(sphereInstanceGroupAgentList.size() + SphereTransformedTable.size());
    const float mat3r4c[12] = {1,0,0,0,0,1,0,0,0,0,1,0};

    size_t instance_idx = 0u;
    size_t sbt_offset = 0u;

    for (auto& sphereAgent : sphereInstanceGroupAgentList) {

        if (sphereAgent->inst_sphere_gas_handle == 0) continue;

        OptixInstance inst{};
        ++instance_idx;

        auto combinedID = sphereAgent->base.materialID + ":" + std::to_string(ShaderMaker::Sphere);
        auto shader_index = OptixUtil::matIDtoShaderIndex[combinedID];

        sbt_offset = shader_index * RAY_TYPE_COUNT;

        inst.flags = OPTIX_INSTANCE_FLAG_NONE;
        inst.sbtOffset = sbt_offset;
        inst.instanceId = instance_idx;
        inst.visibilityMask = DefaultMatMask; 
        inst.traversableHandle = sphereAgent->inst_sphere_gas_handle;

        memcpy(inst.transform, mat3r4c, sizeof(float) * 12);
        optix_instances.push_back( inst );
    }

    if (uniformed_sphere_gas_handle != 0) {

        for(auto& [key, dsphere] : SphereTransformedTable) {

            auto combinedID = dsphere.materialID + ":" + std::to_string(ShaderMaker::Sphere);
            auto shader_index = OptixUtil::matIDtoShaderIndex[combinedID];

            sbt_offset = shader_index * RAY_TYPE_COUNT;
            
            OptixInstance inst{};
            ++instance_idx;
            
            inst.flags = OPTIX_INSTANCE_FLAG_NONE;
            inst.sbtOffset = sbt_offset;
            inst.instanceId = instance_idx;
            inst.visibilityMask = DefaultMatMask;
            inst.traversableHandle = uniformed_sphere_gas_handle;

            auto transform_ptr = glm::value_ptr( dsphere.optix_transform );
            memcpy(inst.transform, transform_ptr, sizeof(float) * 12);
            optix_instances.push_back( inst );
        }
    }

    OptixAccelBuildOptions accel_options{};
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;

    buildIAS(state.context, accel_options, optix_instances, sphereBufferXAS,  sphereHandleXAS);
    timer.tock("Build Sphere IAS");
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
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    buildXAS(state.context, accel_options, triangle_input, mesh->d_gas_output_buffer, mesh->gas_handle);
    
    mesh->dverts.reset();
    mesh->dmats.reset(); 
    mesh->didx.reset();
}

static size_t g_staticMeshNum = 0;
static size_t g_staticVertNum = 0;
static size_t g_staticAndDynamicMeshNum = 0;
static size_t g_staticAndDynamicVertNum = 0;

static void buildMeshIAS(PathTracerState& state, int rayTypeCount, std::vector<std::shared_ptr<smallMesh>> m_meshes) {
    std::cout<<"IAS begin"<<std::endl;
    timer.tick();

    const float mat3r4c[12] = {1,0,0,0,0,1,0,0,0,0,1,0};

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

    std::vector<uint32_t> vertexAuxOffsetGlobal(num_instances);
    uint32_t vertexAuxOffset = 0u;
#ifdef USE_SHORT
    std::vector<ushort3> instPos(num_instances);
    std::vector<ushort3> instNrm(num_instances);
    std::vector<ushort3> instUv(num_instances);
    std::vector<ushort3> instClr(num_instances);
    std::vector<ushort3> instTang(num_instances);
#else
    std::vector<float3> instPos(num_instances);
    std::vector<float3> instNrm(num_instances);
    std::vector<float3> instUv(num_instances);
    std::vector<float3> instClr(num_instances);
    std::vector<float3> instTang(num_instances);
#endif
    size_t sbt_offset = 0;

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
        memcpy( optix_instance.transform, mat3r4c, sizeof( float ) * 12 );

        vertexAuxOffsetGlobal[i] = vertexAuxOffset;
        vertexAuxOffset += mesh->verts.size(); 

        instPos[i] = toHalf({0,0,0,0});
        instNrm[i] = toHalf({0,0,0,0});;
        instUv[i] = toHalf({0,0,0,0});;
        instClr[i] = toHalf({0,0,0,0});;
        instTang[i] = toHalf({0,0,0,0});;
    }

    std::size_t instanceId = g_staticAndDynamicMeshNum;
    std::size_t meshesOffset = g_staticAndDynamicMeshNum;
    for (auto &[instID, instData] : g_instLUT)
    {
        auto it = g_instMatsLUT.find(instID);
        if (it != g_instMatsLUT.end())
        {
            const auto &instMats = it->second;
            const auto &instScales = g_instScaleLUT[instID];
            const auto &instAttrs = g_instAttrsLUT[instID];
            for (std::size_t i = 0; i < instData.meshPieces.size(); ++i)
            {
                auto mesh = m_meshes[meshesOffset];
                for (std::size_t k = 0; k < instMats.size(); ++k)
                {
                    const auto &instMat = instMats[k];
                    float scale = instScales[k];
                    float instMat3r4c[12] = {
                        instMat[0][0] * scale, instMat[1][0] * scale, instMat[2][0] * scale, instMat[3][0],
                        instMat[0][1] * scale, instMat[1][1] * scale, instMat[2][1] * scale, instMat[3][1],
                        instMat[0][2] * scale, instMat[1][2] * scale, instMat[2][2] * scale, instMat[3][2]};
                    auto& optix_instance = optix_instances[instanceId];
                    optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                    optix_instance.instanceId = static_cast<unsigned int>(instanceId);
                    optix_instance.visibilityMask = DefaultMatMask;
                    optix_instance.traversableHandle = mesh->gas_handle;
                    memcpy(optix_instance.transform, instMat3r4c, sizeof(float) * 12);

                    vertexAuxOffsetGlobal[instanceId] = vertexAuxOffset; 
                    
                    instPos[instanceId] = toHalf(instAttrs.pos[k]);
                    instNrm[instanceId] = toHalf(instAttrs.nrm[k]);
                    instUv[instanceId] = toHalf(instAttrs.uv[k]);
                    instClr[instanceId] = toHalf(instAttrs.clr[k]);
                    instTang[instanceId] = toHalf(instAttrs.tang[k]);

                    ++instanceId;
                }
                vertexAuxOffset += mesh->verts.size();
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
                memcpy(optix_instance.transform, mat3r4c, sizeof(float) * 12);

                vertexAuxOffsetGlobal[instanceId] = vertexAuxOffset; 

                instPos[instanceId] = toHalf({0,0,0,0});;
                instNrm[instanceId] = toHalf({0,0,0,0});;
                instUv[instanceId] = toHalf({0,0,0,0});;
                instClr[instanceId] = toHalf({0,0,0,0});;
                instTang[instanceId] = toHalf({0,0,0,0});;
                
                ++instanceId;
                ++meshesOffset;

                vertexAuxOffset += mesh->verts.size();
            }
        }
    }

    state.vertexAuxOffsetGlobal.resize(sizeof(vertexAuxOffsetGlobal[0]) * vertexAuxOffsetGlobal.size(), 0);
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.vertexAuxOffsetGlobal ),
                vertexAuxOffsetGlobal.data(),
                sizeof(vertexAuxOffsetGlobal[0]) * vertexAuxOffsetGlobal.size(),
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


    OptixAccelBuildOptions accel_options{};
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;

    buildIAS(state.context, accel_options, optix_instances, state.meshBufferIAS, state.meshHandleIAS);
    timer.tock("Build Mesh IAS");    
}

void updateRootIAS()
{
    timer.tick();
    const auto campos = state.params.cam.eye;
    const float mat3r4c[12] = {1,0,0,-campos.x,   
                               0,1,0,-campos.y,   
                               0,0,1,-campos.z};

    std::vector<OptixInstance> optix_instances{};
    uint optix_instance_idx = 0u;
    uint sbt_offset = 0u;

    if (state.meshHandleIAS != 0u) {
        OptixInstance inst{};

        inst.flags = OPTIX_INSTANCE_FLAG_NONE;
        inst.sbtOffset = 0;
        inst.instanceId = optix_instance_idx;
        inst.visibilityMask = DefaultMatMask;
        inst.traversableHandle = state.meshHandleIAS;

        memcpy(inst.transform, mat3r4c, sizeof(float) * 12);
        optix_instances.push_back( inst );
    }

    if (sphereHandleXAS != 0u) {
        OptixInstance opinstance {};
        ++optix_instance_idx;
        sbt_offset = 0u;

        opinstance.flags = OPTIX_INSTANCE_FLAG_NONE;
        opinstance.instanceId = optix_instance_idx;
        opinstance.sbtOffset = sbt_offset;
        opinstance.visibilityMask = DefaultMatMask;
        opinstance.traversableHandle = sphereHandleXAS;
        memcpy(opinstance.transform, mat3r4c, sizeof(float) * 12);
        optix_instances.push_back( opinstance );
    }

    // process volume
	for (uint i=0; i<OptixUtil::volumeBoxs.size(); ++i) {
        
		OptixInstance optix_instance {};
    	++optix_instance_idx;

		auto& [key, val] = OptixUtil::volumeBoxs.at(i);

		auto combinedID = key + ":" + std::to_string(ShaderMaker::Volume);
    	auto shader_index = OptixUtil::matIDtoShaderIndex[combinedID];

    	sbt_offset = shader_index * RAY_TYPE_COUNT;

		optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
		optix_instance.instanceId = optix_instance_idx;
		optix_instance.sbtOffset = sbt_offset;
		optix_instance.visibilityMask = VolumeMatMask; //VOLUME_OBJECT;
		optix_instance.traversableHandle = val->accel.handle;
        
		getOptixTransform( *(val), optix_instance.transform );
        
        if ( OptixUtil::g_vdb_list_for_each_shader.count(shader_index) > 0 ) {

            auto& vdb_list = OptixUtil::g_vdb_list_for_each_shader.at(shader_index);

            if (vdb_list.size() > 0)
            {
                auto vdb_key = vdb_list.front();
                auto vdb_ptr = OptixUtil::g_vdb_cached_map.at(vdb_key);

                auto volume_index_offset = list_volume_index_in_shader_list[0];
                optix_instance.traversableHandle = list_volume[shader_index-volume_index_offset]->accel.handle;

                auto ibox = vdb_ptr->grids.front()->indexedBox();

                auto imax = glm::vec3(ibox.max().x(), ibox.max().y(), ibox.max().z()); 
                auto imin = glm::vec3(ibox.min().x(), ibox.min().y(), ibox.min().z()); 

                auto diff = imax + 1.0f - imin;
                auto center = imin + diff / 2.0f;

                glm::mat4 dirtyMatrix(1.0f);
                dirtyMatrix = glm::scale(dirtyMatrix, 1.0f/diff);
                dirtyMatrix = glm::translate(dirtyMatrix, -center);
                
                dirtyMatrix = val->transform * dirtyMatrix;

                auto dummy = glm::transpose(dirtyMatrix);
                auto dummy_ptr = glm::value_ptr( dummy );
                for (size_t i=0; i<12; ++i) {   
                    //optix_instance.transform[i] = mat3r4c[i];
                    optix_instance.transform[i] = dummy_ptr[i];
                }
            }
        } // count  

		optix_instance.transform[3] -= campos.x;
		optix_instance.transform[7] -= campos.y;
		optix_instance.transform[11] -= campos.z;

		optix_instances.push_back( optix_instance );
	}

    uint32_t MAX_INSTANCE_ID;
    optixDeviceContextGetProperty( state.context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID, &MAX_INSTANCE_ID, sizeof(MAX_INSTANCE_ID) );
    state.params.maxInstanceID = MAX_INSTANCE_ID;

  	//process light
	if (lightsWrapper.lightTrianglesGas != 0)
	{
		OptixInstance opinstance {};

		auto combinedID = std::string("Light") + ":" + std::to_string(ShaderMaker::Mesh);
		auto shader_index = OptixUtil::matIDtoShaderIndex[combinedID];

		opinstance.flags = OPTIX_INSTANCE_FLAG_NONE;
		opinstance.instanceId = MAX_INSTANCE_ID-2;
		opinstance.sbtOffset = shader_index * RAY_TYPE_COUNT;
		opinstance.visibilityMask = LightMatMask;
		opinstance.traversableHandle = lightsWrapper.lightTrianglesGas;
		memcpy(opinstance.transform, mat3r4c, sizeof(float) * 12);

		optix_instances.push_back( opinstance );
	}

	if (lightsWrapper.lightPlanesGas != 0)
	{
		OptixInstance opinstance {};

		auto combinedID = std::string("Light") + ":" + std::to_string(ShaderMaker::Mesh);
		auto shader_index = OptixUtil::matIDtoShaderIndex[combinedID];

		opinstance.flags = OPTIX_INSTANCE_FLAG_NONE;
		opinstance.instanceId = MAX_INSTANCE_ID-1;
		opinstance.sbtOffset = shader_index * RAY_TYPE_COUNT;
		opinstance.visibilityMask = LightMatMask;
		opinstance.traversableHandle = lightsWrapper.lightPlanesGas;
		memcpy(opinstance.transform, mat3r4c, sizeof(float) * 12);

		optix_instances.push_back( opinstance );
	}

	if (lightsWrapper.lightSpheresGas != 0)
	{
		OptixInstance opinstance {};

		auto combinedID = std::string("Light") + ":" + std::to_string(ShaderMaker::Sphere);
		auto shader_index = OptixUtil::matIDtoShaderIndex[combinedID];

		opinstance.flags = OPTIX_INSTANCE_FLAG_NONE;
		opinstance.instanceId = MAX_INSTANCE_ID;
		opinstance.sbtOffset = shader_index * RAY_TYPE_COUNT;
		opinstance.visibilityMask = LightMatMask;
		opinstance.traversableHandle = lightsWrapper.lightSpheresGas;
		memcpy(opinstance.transform, mat3r4c, sizeof(float) * 12);

		optix_instances.push_back( opinstance );
	}

	OptixAccelBuildOptions accel_options{};
	accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;

	buildIAS(state.context, accel_options, optix_instances, state.rootBufferIAS, state.rootHandleIAS);  

	timer.tock("update Root IAS");
	state.params.handle = state.rootHandleIAS;
}

static void buildMeshAccel( PathTracerState& state )
{
    //
    // copy mesh data to device
    //
    const size_t vertices_size_in_bytes = g_vertices.size() * sizeof( Vertex );
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
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    buildXAS(state.context, accel_options, triangle_input, state.d_gas_output_buffer, state.d_gas_output_buffer);

    state.d_vertices.reset();
    state.d_mat_indices.reset();
}

static void createSBT( PathTracerState& state )
{
        state.d_raygen_record.reset();
        state.d_miss_records.reset();
        state.d_hitgroup_records.reset();
        state.d_callable_records.reset();

        state.d_gas_output_buffer.reset();
        state.accum_buffer_p.reset();
        state.albedo_buffer_p.reset();
        state.normal_buffer_p.reset();

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

    const auto shader_count = OptixUtil::rtMaterialShaders.size();

    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    const size_t hitgroup_record_count = shader_count * RAY_TYPE_COUNT;

    raii<CUdeviceptr>  &d_hitgroup_records = state.d_hitgroup_records;
    
    CUDA_CHECK(cudaMalloc((void**)&d_hitgroup_records.reset(),
                hitgroup_record_size * hitgroup_record_count
                ));

    std::vector<HitGroupRecord> hitgroup_records(hitgroup_record_count);
    std::vector<CallablesRecord> callable_records(shader_count);

    for( int j = 0; j < shader_count; ++j ) {

        auto& shader_ref = OptixUtil::rtMaterialShaders[j];
        const auto has_vdb = shader_ref.has_vdb;

        const uint sbt_idx = RAY_TYPE_COUNT * j;

        OPTIX_CHECK( optixSbtRecordPackHeader( shader_ref.callable_prog_group, &callable_records[j] ) );

        if (!has_vdb) {

            hitgroup_records[sbt_idx] = {};

            hitgroup_records[sbt_idx].data.uniforms        = reinterpret_cast<float4*>( (CUdeviceptr)state.d_uniforms );
#ifdef USE_SHORT
            hitgroup_records[sbt_idx].data.uv              = reinterpret_cast<ushort3*>( (CUdeviceptr)state.d_uv );
            hitgroup_records[sbt_idx].data.nrm             = reinterpret_cast<ushort3*>( (CUdeviceptr)state.d_nrm );
            hitgroup_records[sbt_idx].data.clr             = reinterpret_cast<ushort3*>( (CUdeviceptr)state.d_clr );
            hitgroup_records[sbt_idx].data.tan             = reinterpret_cast<ushort3*>( (CUdeviceptr)state.d_tan );
#else
            hitgroup_records[sbt_idx].data.uv              = reinterpret_cast<float4*>( (CUdeviceptr)state.d_uv );
            hitgroup_records[sbt_idx].data.nrm             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_nrm );
            hitgroup_records[sbt_idx].data.clr             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_clr );
            hitgroup_records[sbt_idx].data.tan             = reinterpret_cast<float4*>( (CUdeviceptr)state.d_tan );
#endif
            hitgroup_records[sbt_idx].data.lightMark       = reinterpret_cast<unsigned short*>( (CUdeviceptr)state.d_lightMark );
            hitgroup_records[sbt_idx].data.auxOffset       = reinterpret_cast<uint32_t*>( (CUdeviceptr)state.vertexAuxOffsetGlobal );
#ifdef USE_SHORT
            hitgroup_records[sbt_idx].data.instPos         = reinterpret_cast<ushort3*>( (CUdeviceptr)state.d_instPos );
            hitgroup_records[sbt_idx].data.instNrm         = reinterpret_cast<ushort3*>( (CUdeviceptr)state.d_instNrm );
            hitgroup_records[sbt_idx].data.instUv          = reinterpret_cast<ushort3*>( (CUdeviceptr)state.d_instUv );
            hitgroup_records[sbt_idx].data.instClr         = reinterpret_cast<ushort3*>( (CUdeviceptr)state.d_instClr );
            hitgroup_records[sbt_idx].data.instTang        = reinterpret_cast<ushort3*>( (CUdeviceptr)state.d_instTang );
#else
            hitgroup_records[sbt_idx].data.instPos         = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instPos );
            hitgroup_records[sbt_idx].data.instNrm         = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instNrm );
            hitgroup_records[sbt_idx].data.instUv          = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instUv );
            hitgroup_records[sbt_idx].data.instClr         = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instClr );
            hitgroup_records[sbt_idx].data.instTang        = reinterpret_cast<float3*>( (CUdeviceptr)state.d_instTang );
#endif
            for(int t=0;t<32;t++)
            {
                hitgroup_records[sbt_idx].data.textures[t] = shader_ref.getTexture(t);
            }

            hitgroup_records[sbt_idx+1] = hitgroup_records[sbt_idx]; // SBT for occlusion ray-type for ith material

            OPTIX_CHECK( optixSbtRecordPackHeader( shader_ref.core->m_radiance_hit_group, &hitgroup_records[sbt_idx] ) );
            OPTIX_CHECK( optixSbtRecordPackHeader( shader_ref.core->m_occlusion_hit_group, &hitgroup_records[sbt_idx+1] ) );

        } else {

            HitGroupRecord rec = {};

            rec.data.uniforms = reinterpret_cast<float4*>( (CUdeviceptr)state.d_uniforms );

            if (OptixUtil::g_vdb_list_for_each_shader.count(j) != 0) {

                auto& vdb_list = OptixUtil::g_vdb_list_for_each_shader.at(j);
				//if (OptixUtil::g_cached_vdb_map.count(key_vdb) == 0) continue;
				//auto& volumeWrapper = OptixUtil::g_vdb[key_vdb];

				for(uint t=0; t<min(vdb_list.size(), 8ull); ++t)
				{
					auto vdb_key = vdb_list[t];
					auto vdb_ptr = OptixUtil::g_vdb_cached_map.at(vdb_key);

					rec.data.vdb_grids[t] = vdb_ptr->grids.front()->deviceptr;
					rec.data.vdb_max_v[t] = vdb_ptr->grids.front()->max_value;
				}
            }

            for(uint t=0;t<32;t++)
            {
                rec.data.textures[t] = shader_ref.getTexture(t);
            }

            {
                auto j = nlohmann::json::parse(shader_ref.parameters);

                if (!j["vol_depth"].is_null()) {
                    rec.data.vol_depth = j["vol_depth"];
                }

                if (!j["vol_extinction"].is_null()) {
                    rec.data.vol_extinction = j["vol_extinction"];
                }

                if (!j["equiangular"].is_null()) {
                    rec.data.equiangular = j["equiangular"];
                }

                if (!j["multiscatter"].is_null()) {
                    rec.data.multiscatter = j["multiscatter"];
                }
            }

            hitgroup_records[sbt_idx] = rec;
            hitgroup_records[sbt_idx+1] = rec;

            OPTIX_CHECK(optixSbtRecordPackHeader( shader_ref.core->m_radiance_hit_group, &hitgroup_records[sbt_idx] ) );
            OPTIX_CHECK(optixSbtRecordPackHeader( shader_ref.core->m_occlusion_hit_group, &hitgroup_records[sbt_idx+1] ) );
            
        }
        

        hitgroup_records[sbt_idx].data.dc_index = j;
        hitgroup_records[sbt_idx+1].data.dc_index = j;
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
    //state.sbt.exceptionRecord;

    {
        raii<CUdeviceptr>& d_callable_records = state.d_callable_records;
        size_t      sizeof_callable_record = sizeof( CallablesRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_callable_records ), sizeof_callable_record * shader_count ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)d_callable_records ), callable_records.data(),
                                sizeof_callable_record * shader_count, cudaMemcpyHostToDevice ) );

        state.sbt.callablesRecordBase          = d_callable_records;
        state.sbt.callablesRecordCount         = shader_count;
        state.sbt.callablesRecordStrideInBytes = static_cast<unsigned int>( sizeof_callable_record );
    }
}

static void cleanupState( PathTracerState& state )
{
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group ));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_miss_group));

    OPTIX_CHECK(optixModuleDestroy(OptixUtil::ray_module));
    OPTIX_CHECK(optixModuleDestroy(OptixUtil::sphere_module));

    cleanupSpheresGPU();
    lightsWrapper.reset();
    
    for (auto& ele : list_volume) {
        cleanupVolume(*ele);
    }
    list_volume.clear();

    for (auto const& [key, val] : OptixUtil::g_vdb_cached_map) {
        cleanupVolume(*val);
    }
    OptixUtil::g_vdb_cached_map.clear();
    OptixUtil::g_ies.clear();

    std::cout << "optix cleanup" << std::endl;
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
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------


#ifdef OPTIX_BASE_GL
std::optional<sutil::GLDisplay> gl_display_o;
#endif

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
#ifndef OPTIX_BASE_GL
    output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
#endif

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
    if (!output_buffer_color) {
      output_buffer_color.emplace(
          output_buffer_type,
          state.params.width,
          state.params.height
      );
      output_buffer_color->setStream( 0 );
    }
    if (!output_buffer_diffuse) {
      output_buffer_diffuse.emplace(
          output_buffer_type,
          state.params.width,
          state.params.height
      );
      output_buffer_diffuse->setStream( 0 );
    }
    if (!output_buffer_specular) {
      output_buffer_specular.emplace(
          output_buffer_type,
          state.params.width,
          state.params.height
      );
      output_buffer_specular->setStream( 0 );
    }
    if (!output_buffer_transmit) {
      output_buffer_transmit.emplace(
          output_buffer_type,
          state.params.width,
          state.params.height
      );
      output_buffer_transmit->setStream( 0 );
    }
    if (!output_buffer_background) {
      output_buffer_background.emplace(
          output_buffer_type,
          state.params.width,
          state.params.height
      );
      output_buffer_background->setStream( 0 );
    }
    if (!output_buffer_mask) {
      output_buffer_mask.emplace(
          output_buffer_type,
          state.params.width,
          state.params.height
      );
      output_buffer_mask->setStream( 0 );
    }
#ifdef OPTIX_BASE_GL
        if (!gl_display_o) {
            gl_display_o.emplace(sutil::BufferImageFormat::UNSIGNED_BYTE4);
        }
#endif
    xinxinoptix::update_procedural_sky(zeno::vec2f(-60, 45), 1, zeno::vec2f(0, 0), 0, 0.1,
                                       1.0, 0.0, 6500.0);
    xinxinoptix::using_hdr_sky(true);
    xinxinoptix::show_background(false);
    std::string parent_path;
#ifdef __linux__
    char path[1024];
    getcwd(path, sizeof(path));
    auto cur_path = std::string(path);
#else
    auto cur_path = std::string(_pgmptr);
    cur_path = cur_path.substr(0, cur_path.find_last_of("\\"));
#endif
    OptixUtil::sky_tex = cur_path + "/hdr/Panorama.hdr";
    OptixUtil::addTexture(OptixUtil::sky_tex.value());
    xinxinoptix::update_hdr_sky(0, {0, 0, 0}, 0.8);
}


void updateVolume(uint volume_shader_offset) {

    if (OptixUtil::g_vdb_cached_map.empty()) { return; }

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

    OptixUtil::logInfoVRAM("Before Volume GAS");

    std::map<uint, std::vector<std::string>> tmp_map{};

    for (auto const& [index, val] : OptixUtil::g_vdb_list_for_each_shader) {
        auto base_key = val.front();

        if (OptixUtil::g_vdb_indice_visible.count(base_key) == 0) continue;

        list_volume.push_back( OptixUtil::g_vdb_cached_map[base_key] );
        list_volume_index_in_shader_list.push_back(index + volume_shader_offset);

        tmp_map[index + volume_shader_offset] = val;
   }

   OptixUtil::g_vdb_list_for_each_shader = tmp_map;

    for (uint i=0; i<list_volume.size(); ++i) {
        //VolumeAccel accel;
        buildVolumeAccel( list_volume[i]->accel, *(list_volume[i]), state.context );
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
    {
        splitMesh(g_vertices, g_mat_indices, g_meshPieces, 0, 0);
        g_staticMeshNum = g_meshPieces.size();

        size_t vertSize = TRI_PER_MESH * 3 * g_meshPieces.size();
        vertSize = std::min(g_vertices.size(), vertSize);

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
    {
        splitMesh(g_vertices, g_mat_indices, g_meshPieces, g_staticMeshNum, g_staticVertNum);
        g_staticAndDynamicMeshNum = g_meshPieces.size();

        size_t vertSize = TRI_PER_MESH * 3 * g_meshPieces.size();
        vertSize = std::min(g_vertices.size(), vertSize);

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

            std::size_t vertSize = TRI_PER_MESH * 3 * meshPieces.size();
            vertSize = std::min(vertices.size(), vertSize);
            
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
void UpdateDynamicInstMesh(std::map<std::string, int> const &mtlidlut)
{
    camera_changed = true;
    g_mtlidlut = mtlidlut;
    updateDynamicDrawInstObjects();
    
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
            
            std::size_t vertSize = TRI_PER_MESH * 3 * meshPieces.size();
            vertSize = std::min(vertices.size(), vertSize);

            vertices.resize(vertSize);
            clr.resize(vertSize);
            nrm.resize(vertSize);
            tan.resize(vertSize);
            uv.resize(vertSize);
            mat_indices.resize(vertSize / 3);
            lightMark.resize(vertSize / 3);
        }
}

void CopyInstMeshToGlobalMesh()
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

            for (size_t i = 0; i < vertices.size(); ++i)
            {
                g_vertices[vertsOffset + i] = vertices[i];
#ifdef USE_SHORT
                g_clr[vertsOffset + i] = toHalf(clr[i]);
                g_nrm[vertsOffset + i] = halfNormal(nrm[i]);
                g_uv[vertsOffset + i]  = toHalf(uv[i]);
                g_tan[vertsOffset + i] = halfNormal(tan[i]);
#else
                g_clr[vertsOffset + i] = clr[i];
                g_nrm[vertsOffset + i] = nrm[i];
                g_uv[vertsOffset + i] = uv[i];
                g_tan[vertsOffset + i] = tan[i];
#endif
            }
            for (size_t i = 0; i < vertices.size() / 3; ++i)
            {
                g_mat_indices[vertsOffset / 3 + i] = mat_indices[i];
                g_lightMark[vertsOffset / 3 + i] = lightMark[i];
            }
            for (size_t i = 0; i < meshPieces.size(); ++i)
            {
                g_meshPieces[meshPiecesOffset + i] = meshPieces[i];
            }

            vertsOffset += vertices.size();
            meshPiecesOffset += meshPieces.size();
        }
}

void UpdateMeshGasAndIas(bool staticNeedUpdate)
{
        for(int i=0;i<g_meshPieces.size();i++)
        {
            buildMeshAccelSplitMesh(state, g_meshPieces[i]);
        }
#define WXL 1
        std::cout << "begin copy\n";
        timer.tick();
#ifdef USE_SHORT
        size_t vertices_size_in_bytes = g_vertices.size() * sizeof(ushort3);
        size_t static_vertices_size_in_bytes = g_staticVertNum * sizeof(ushort3);
#else
        size_t vertices_size_in_bytes = g_vertices.size() * sizeof(Vertex);
        size_t static_vertices_size_in_bytes = g_staticVertNum * sizeof(Vertex);
#endif
        size_t dynamic_vertices_size_in_bytes = vertices_size_in_bytes - static_vertices_size_in_bytes;
        bool realloced;
        size_t offset = 0;
        size_t numBytes = vertices_size_in_bytes;
        auto updateRange = [&vertices_size_in_bytes, &dynamic_vertices_size_in_bytes, &realloced, &offset,
                            &numBytes]() {
            if (!realloced && WXL) {
#ifdef USE_SHORT
              offset = g_staticVertNum * sizeof(ushort3);
#else
                offset = g_staticVertNum * sizeof(Vertex);
#endif
                numBytes = dynamic_vertices_size_in_bytes;
            } else {
                offset = 0;
                numBytes = vertices_size_in_bytes;
            }
        };
#if WXL
        //realloced = state.d_vertices.resize(vertices_size_in_bytes, dynamic_vertices_size_in_bytes);
        state.d_clr.resize(vertices_size_in_bytes, dynamic_vertices_size_in_bytes);
        state.d_uv.resize(vertices_size_in_bytes, dynamic_vertices_size_in_bytes);
        state.d_nrm.resize(vertices_size_in_bytes, dynamic_vertices_size_in_bytes);
        state.d_tan.resize(vertices_size_in_bytes, dynamic_vertices_size_in_bytes);
        size_t reservedCap = state.d_clr.capacity - vertices_size_in_bytes;
        if (reservedCap > 0) {
            // CUDA_CHECK(cudaMemset(reinterpret_cast<char *>((CUdeviceptr &)state.d_vertices) +
            //                           vertices_size_in_bytes, 0, reservedCap));
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
        //CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_vertices.reset()), vertices_size_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_clr.reset()), vertices_size_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_uv.reset()), vertices_size_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_nrm.reset()), vertices_size_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_tan.reset()), vertices_size_in_bytes));
#endif
        updateRange();
        // CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_vertices) + offset,
        //                       (char *)g_vertices.data() + offset, numBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_clr) + offset, (char *)g_clr.data() + offset,
                              numBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_uv) + offset, (char *)g_uv.data() + offset,
                              numBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_nrm) + offset, (char *)g_nrm.data() + offset,
                              numBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_tan) + offset, (char *)g_tan.data() + offset,
                              numBytes, cudaMemcpyHostToDevice));
        if (staticNeedUpdate && offset != 0) {
            // CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr &)state.d_vertices),
            //                       (char *)g_vertices.data(), static_vertices_size_in_bytes, cudaMemcpyHostToDevice));
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
        buildMeshIAS(state, 2, g_meshPieces);
}

static std::map<std::string, LightDat> lightdats;
static std::vector<float2>  triangleLightCoords;
static std::vector<float3>  triangleLightNormals;

void unload_light(){

    lightdats.clear();
    triangleLightCoords.clear();
    triangleLightNormals.clear();

    std::cout << "Lights unload done. \n"<< std::endl;
}

void load_triangle_light(std::string const &key, LightDat& ld,
                        const zeno::vec3f &v0,  const zeno::vec3f &v1,  const zeno::vec3f &v2, 
                        const zeno::vec3f *pn0, const zeno::vec3f *pn1, const zeno::vec3f *pn2,
                        const zeno::vec3f *uv0, const zeno::vec3f *uv1, const zeno::vec3f *uv2) {

    ld.v0.assign(v0.begin(), v0.end());
    ld.v1.assign(v1.begin(), v1.end());
    ld.v2.assign(v2.begin(), v2.end());

    if (pn0 != nullptr && pn1 != nullptr, pn2 != nullptr) {
        ld.normalBufferOffset = triangleLightNormals.size();
        triangleLightNormals.push_back(*(float3*)pn0);
        triangleLightNormals.push_back(*(float3*)pn1);
        triangleLightNormals.push_back(*(float3*)pn2);
    }

    if (uv0 != nullptr && uv1 != nullptr && uv2 != nullptr) {
        ld.coordsBufferOffset = triangleLightCoords.size();
        triangleLightCoords.push_back(*(float2*)uv0);
        triangleLightCoords.push_back(*(float2*)uv1);
        triangleLightCoords.push_back(*(float2*)uv2);
    }

    lightdats[key] = ld;
}

void load_light(std::string const &key, LightDat& ld, float const*v0, float const*v1, float const*v2) {

    ld.v0.assign(v0, v0 + 3);
    ld.v1.assign(v1, v1 + 3);
    ld.v2.assign(v2, v2 + 3);
    
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

void show_background(bool enable) {
    state.params.show_background = enable;
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

static void addTriangleLightGeo(float3 p0, float3 p1, float3 p2) {
    auto& geo = lightsWrapper._triangleLightGeo;
    geo.push_back(p0); geo.push_back(p1); geo.push_back(p2);
}

static void addLightPlane(float3 p0, float3 v1, float3 v2, float3 normal)
{
    float3 vert0 = p0, vert1 = p0 + v1, vert2 = p0 + v2, vert3 = p0 + v1 + v2;

    auto& geo = lightsWrapper._planeLightGeo;

    geo.push_back(make_float4(vert0.x, vert0.y, vert0.z, 0.f));
    geo.push_back(make_float4(vert1.x, vert1.y, vert1.z, 0.f));
    geo.push_back(make_float4(vert3.x, vert3.y, vert3.z, 0.f));
   
    geo.push_back(make_float4(vert0.x, vert0.y, vert0.z, 0.f));
    geo.push_back(make_float4(vert3.x, vert3.y, vert3.z, 0.f));
    geo.push_back(make_float4(vert2.x, vert2.y, vert2.z, 0.f));
}

static void addLightSphere(float3 center, float radius) 
{
    Vertex vt {center.x, center.y, center.z, radius};
    lightsWrapper._sphereLightGeo.push_back(vt);
}

static int uniformBufferInitialized = false;
// void optixUpdateUniforms(std::vector<float4> & inConstants) 
void optixUpdateUniforms(void *inConstants, std::size_t size) {

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &state.d_uniforms.reset() ), sizeof(float4)*512));

    CUDA_CHECK(cudaMemset(reinterpret_cast<char *>((CUdeviceptr &)state.d_uniforms), 0, sizeof(float4)*512));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>((CUdeviceptr)state.d_uniforms), (float4*)inConstants,
                          sizeof(float4)*size, cudaMemcpyHostToDevice));

    uniformBufferInitialized = true;

}

static void buildLightPlanesGAS( PathTracerState& state, std::vector<Vertex>& lightMesh, raii<CUdeviceptr>& bufferGas, OptixTraversableHandle& handleGas)
{
    if (lightMesh.empty()) {
        handleGas = 0;
        bufferGas.reset();
        return;
    }

    const size_t vertices_size_in_bytes = lightMesh.size() * sizeof( Vertex );

    raii<CUdeviceptr> d_lightMesh;
    
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_lightMesh ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)d_lightMesh ),
                lightMesh.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice 
                ) );

    std::vector<uint32_t> triangle_input_flags(1, OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);

    OptixBuildInput triangle_input                           = {};
    triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes         = sizeof( Vertex );
    triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>( lightMesh.size() );
    triangle_input.triangleArray.vertexBuffers               = lightMesh.empty() ? nullptr : &d_lightMesh;
    triangle_input.triangleArray.flags                       = triangle_input_flags.data();
    triangle_input.triangleArray.numSbtRecords               = 1; // g_lightMesh.empty() ? 1 : g_mtlidlut.size();
    triangle_input.triangleArray.sbtIndexOffsetBuffer        = 0;//state.d_mat_indices;
    // triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
    // triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );

    OptixAccelBuildOptions accel_options {};
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
    
    buildXAS(state.context, accel_options, triangle_input, bufferGas, handleGas);
}

static void buildLightSpheresGAS( PathTracerState& state, std::vector<Vertex>& lightSpheres, raii<CUdeviceptr>& bufferGas, OptixTraversableHandle& handleGas) {

    if (lightSpheres.empty()) {
        handleGas = 0;
        bufferGas.reset();
        return;
    }

    const size_t sphere_count = lightSpheres.size(); 

    OptixAccelBuildOptions accel_options {};
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;

    raii<CUdeviceptr> d_vertex_buffer{}; 
   
    {
        auto data_length = sizeof( Vertex ) * sphere_count;

        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertex_buffer.reset() ), data_length) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)d_vertex_buffer ), lightSpheres.data(),
                                data_length, cudaMemcpyHostToDevice ) );
    }
    CUdeviceptr d_radius_buffer = (CUdeviceptr) ( (char*)d_vertex_buffer.handle + 12u );  

    OptixBuildInput sphere_input{};

    sphere_input.type                      = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    sphere_input.sphereArray.numVertices   = sphere_count;
    sphere_input.sphereArray.vertexBuffers = &d_vertex_buffer;
    sphere_input.sphereArray.radiusBuffers = &d_radius_buffer;
    sphere_input.sphereArray.singleRadius = false;
    sphere_input.sphereArray.vertexStrideInBytes = 16;
    sphere_input.sphereArray.radiusStrideInBytes = 16;
    //sphere_input.sphereArray.primitiveIndexOffset = 0;

    std::vector<uint> sphere_input_flags(1, OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);
    
    sphere_input.sphereArray.flags         = sphere_input_flags.data();
    sphere_input.sphereArray.numSbtRecords = 1;
    sphere_input.sphereArray.sbtIndexOffsetBuffer = 0;
    // sphere_input.sphereArray.sbtIndexOffsetSizeInBytes = sizeof(uint);
    // sphere_input.sphereArray.sbtIndexOffsetStrideInBytes = sizeof(uint);

    buildXAS(state.context, accel_options, sphere_input, bufferGas, handleGas);
}

static void buildLightTrianglesGAS( PathTracerState& state, std::vector<float3>& lightMesh, raii<CUdeviceptr>& bufferGas, OptixTraversableHandle& handleGas)
{
    if (lightMesh.empty()) {
        handleGas = 0;
        bufferGas.reset();
        return;
    }

    const size_t vertices_size_in_bytes = lightMesh.size() * sizeof( float3 );

    raii<CUdeviceptr> d_lightMesh;
    
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_lightMesh ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)d_lightMesh ),
                lightMesh.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice 
                ) );

    std::vector<uint32_t> triangle_input_flags(1, OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);

    OptixBuildInput triangle_input                           = {};
    triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes         = sizeof( float3 );
    triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>( lightMesh.size() );
    triangle_input.triangleArray.vertexBuffers               = lightMesh.empty() ? nullptr : &d_lightMesh;
    triangle_input.triangleArray.flags                       = triangle_input_flags.data();
    triangle_input.triangleArray.numSbtRecords               = 1;
    triangle_input.triangleArray.sbtIndexOffsetBuffer        = 0;
    // triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
    // triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );

    OptixAccelBuildOptions accel_options {};
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
    
    buildXAS(state.context, accel_options, triangle_input, bufferGas, handleGas);
}

void buildLightTree() {
    camera_changed = true;
    state.lightsbuf_p.reset();

    state.params.lightTreeSampler = 0llu;
    state.params.triangleLightCoordsBuffer = 0llu;
    state.params.triangleLightNormalBuffer = 0llu;
    
    state.params.firstRectLightIdx = UINT_MAX;
    state.params.firstSphereLightIdx = UINT_MAX;
    state.params.firstTriangleLightIdx = UINT_MAX;

    state.params.lights = 0llu;
    state.params.num_lights = 0u;

    lightsWrapper.reset();

    std::vector<LightDat*> sortedLights; 
    sortedLights.reserve(lightdats.size());

    for(const auto& [_, dat] : lightdats) {
        auto ldp = (LightDat*)&dat;
        sortedLights.push_back(std::move(ldp));
    }

    std::sort(sortedLights.begin(), sortedLights.end(), 
        [](const auto& a, const auto& b) {
            return a->shape < b->shape;
        });

    {
        auto byte_size = triangleLightNormals.size() * sizeof(float3);

        CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &lightsWrapper.triangleLightNormals.reset() ), byte_size) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)lightsWrapper.triangleLightNormals ),
                                triangleLightNormals.data(), byte_size, cudaMemcpyHostToDevice) );
        state.params.triangleLightNormalBuffer = lightsWrapper.triangleLightNormals.handle;
    }

    {
        auto byte_size = triangleLightCoords.size() * sizeof(float2);

        CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &lightsWrapper.triangleLightCoords.reset() ), byte_size) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)lightsWrapper.triangleLightCoords ),
                                triangleLightCoords.data(), byte_size, cudaMemcpyHostToDevice) );
        state.params.triangleLightCoordsBuffer = lightsWrapper.triangleLightCoords.handle;
    }

    uint32_t idx = 0u;

    uint32_t firstRectLightIdx = UINT_MAX;
    uint32_t firstSphereLightIdx = UINT_MAX;
    uint32_t firstTriangleLightIdx = UINT_MAX;

    for(uint32_t idx=0u; idx<sortedLights.size(); ++idx) {

        auto& dat = *sortedLights.at(idx);
        auto& light = lightsWrapper.g_lights.emplace_back();

        uint8_t config = zeno::LightConfigNull; 
        config |= dat.visible? zeno::LightConfigVisible: zeno::LightConfigNull; 
        config |= dat.doubleside? zeno::LightConfigDoubleside: zeno::LightConfigNull;
        light.config = config;

        light.color.x = fmaxf(dat.color.at(0), FLT_EPSILON);
        light.color.y = fmaxf(dat.color.at(1), FLT_EPSILON);
        light.color.z = fmaxf(dat.color.at(2), FLT_EPSILON);

        light.spreadMajor = clamp(dat.spreadMajor, 0.0f, 1.0f);
        light.spreadMinor = clamp(dat.spreadMinor, 0.0f, 1.0f);

        auto void_angle = 0.5f * (1.0f - light.spreadMajor) * M_PIf;
        light.spreadNormalize = 2.f / (2.f + (2.f * void_angle - M_PIf) * tanf(void_angle));

        light.mask = dat.mask;
        light.intensity  = dat.intensity;
        light.vIntensity = dat.vIntensity;
        light.maxDistance = dat.maxDistance <= 0.0? FLT_MAX:dat.maxDistance;
        light.falloffExponent = dat.falloffExponent;
        
        float3& v0 = *(float3*)dat.v0.data();
        float3& v1 = *(float3*)dat.v1.data();
        float3& v2 = *(float3*)dat.v2.data();

        light.N = *(float3*)dat.normal.data();
        light.N = normalize(light.N);
        light.T = normalize(v1);
        light.B = normalize(v2);

        const auto center = v0 + v1 * 0.5f + v2 * 0.5f;
        const auto radius = fminf(length(v1), length(v2)) * 0.5f;

        light.type  = magic_enum::enum_cast<zeno::LightType>(dat.type).value_or(zeno::LightType::Diffuse);
        light.shape = magic_enum::enum_cast<zeno::LightShape>(dat.shape).value_or(zeno::LightShape::Plane);

        if (light.spreadMajor < 0.005f) {
            light.type = zeno::LightType::Direction;
        }

        if (light.shape == zeno::LightShape::Plane || light.shape == zeno::LightShape::Ellipse) {

            firstRectLightIdx = min(idx, firstRectLightIdx);

            light.setRectData(v0, v1, v2, light.N);
            addLightPlane(v0, v1, v2, light.N);

            light.rect.isEllipse = (light.shape == zeno::LightShape::Ellipse);

            if (dat.fluxFixed > 0) {
                light.intensity = dat.fluxFixed / light.rect.Area(); 
            }

        } else if (light.shape == zeno::LightShape::Sphere) {

            firstSphereLightIdx = min(idx, firstSphereLightIdx);

            light.setSphereData(center, radius);       
            addLightSphere(center, radius);

            if (dat.fluxFixed > 0) {
                auto intensity = dat.fluxFixed / light.sphere.area;
                light.intensity = intensity;
            }

        } else if (light.shape == zeno::LightShape::Point) {
            light.point = {center};
            if (dat.fluxFixed > 0) {
                auto intensity = dat.fluxFixed / (4 * M_PIf);
                light.intensity = intensity;
            }

        } else if (light.shape == zeno::LightShape::TriangleMesh) {

            firstTriangleLightIdx = min(idx, firstTriangleLightIdx);
            light.setTriangleData(v0, v1, v2, light.N, dat.coordsBufferOffset, dat.normalBufferOffset);
            addTriangleLightGeo(v0, v1, v2);
        }

        if (light.type == zeno::LightType::Spot) {

            auto spread_major = clamp(light.spreadMajor, 0.01, 1.00);
            auto spread_inner = clamp(light.spreadMinor, 0.01, 0.99);

            auto major_angle = spread_major * 0.5f * M_PIf;
            major_angle = fmaxf(major_angle, 2 * FLT_EPSILON);

            auto inner_angle = spread_inner * major_angle;
            auto falloff_angle = major_angle - inner_angle;

            light.setConeData(center, light.N, 0.0f, major_angle, falloff_angle);
        }
        if (light.type == zeno::LightType::Projector) {
            light.point = {center};
        }

        if ( OptixUtil::g_ies.count(dat.profileKey) > 0 ) {

            auto& val = OptixUtil::g_ies.at(dat.profileKey);
            light.ies = val.ptr.handle;
            light.type = zeno::LightType::IES;
            //light.shape = zeno::LightShape::Point;
            light.setConeData(center, light.N, radius, val.coneAngle, FLT_EPSILON);

            if (dat.fluxFixed > 0) {
                auto scale = val.coneAngle / M_PIf;
                light.intensity = dat.fluxFixed * scale * scale;
            }
        } 
        
        if ( OptixUtil::g_tex.count(dat.textureKey) > 0 ) {

            auto& val = OptixUtil::g_tex.at(dat.textureKey);
            light.tex = val->texture;
            light.texGamma = dat.textureGamma;
        }
    }

    if (lightsWrapper.g_lights.empty()) { return; }

    state.params.firstRectLightIdx = firstRectLightIdx;
    state.params.firstSphereLightIdx = firstSphereLightIdx;
    state.params.firstTriangleLightIdx = firstTriangleLightIdx;

    buildLightPlanesGAS(state, lightsWrapper._planeLightGeo, lightsWrapper.lightPlanesGasBuffer, lightsWrapper.lightPlanesGas);
    buildLightSpheresGAS(state, lightsWrapper._sphereLightGeo, lightsWrapper.lightSpheresGasBuffer, lightsWrapper.lightSpheresGas);
    buildLightTrianglesGAS(state, lightsWrapper._triangleLightGeo, lightsWrapper.lightTrianglesGasBuffer, lightsWrapper.lightTrianglesGas);

    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.lightsbuf_p.reset() ),
        sizeof( GenericLight ) * std::max(lightsWrapper.g_lights.size(),(size_t)1)
        ) );

    state.params.lights = (GenericLight*)(CUdeviceptr)state.lightsbuf_p;
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.lightsbuf_p ),
                lightsWrapper.g_lights.data(), sizeof( GenericLight ) * lightsWrapper.g_lights.size(),
                cudaMemcpyHostToDevice
                ) );

    auto lsampler = pbrt::LightTreeSampler(lightsWrapper.g_lights);

    raii<CUdeviceptr>& lightBitTrailsPtr = lightsWrapper.lightBitTrailsPtr;
    raii<CUdeviceptr>& lightTreeNodesPtr = lightsWrapper.lightTreeNodesPtr;
    raii<CUdeviceptr>& lightTreeDummyPtr = lightsWrapper.lightTreeDummyPtr;

    lightBitTrailsPtr.reset(); lightTreeNodesPtr.reset(); 
    lsampler.upload(lightBitTrailsPtr.handle, lightTreeNodesPtr.handle);

    struct Dummy {
        unsigned long long bitTrails;
        unsigned long long treeNodes;
        pbrt::Bounds3f bounds;
    };

    Dummy dummy = { lightBitTrailsPtr.handle, lightTreeNodesPtr.handle, lsampler.bounds() };

    {
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &lightTreeDummyPtr.reset() ), sizeof( dummy )) );
        CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)lightTreeDummyPtr ),
                &dummy, sizeof( dummy ), cudaMemcpyHostToDevice) );
        state.params.lightTreeSampler = lightTreeDummyPtr.handle;
    }
}

inline std::map<std::tuple<std::string, ShaderMaker>, std::shared_ptr<OptixUtil::OptixShaderCore>> shaderCoreLUT {};

void optixupdatematerial(std::vector<std::shared_ptr<ShaderPrepared>> &shaders) 
{
    camera_changed = true;

    CppTimer theTimer;
    theTimer.tick();
    //static bool hadOnce = false;
    if (OptixUtil::ray_module.handle==0) {

        OptixUtil::_compile_group.run([&] () {

            if (!OptixUtil::createModule(
                OptixUtil::ray_module,
                state.context,
                sutil::lookupIncFile("PTKernel.cu"),
                "PTKernel.cu")) throw std::runtime_error("base ray module failed to compile");

            OptixUtil::createRenderGroups(state.context, OptixUtil::ray_module);
        });

        OptixUtil::_compile_group.run([&] () {
            auto shader_string = sutil::lookupIncFile("DeflMatShader.cu"); 
                
            auto shaderCore = std::make_shared<OptixUtil::OptixShaderCore>(shader_string, "__closesthit__radiance", "__anyhit__shadow_cutout");
            shaderCore->loadProgram(0);
            shaderCoreLUT.emplace(std::tuple{"DeflMatShader.cu", ShaderMaker::Mesh}, shaderCore);
        });

        OptixUtil::_compile_group.run([&] () {
            auto shader_string = sutil::lookupIncFile("DeflMatShader.cu"); 

            auto shaderCore = std::make_shared<OptixUtil::OptixShaderCore>(shader_string, "__closesthit__radiance", "__anyhit__shadow_cutout");
            shaderCore->moduleIS = &OptixUtil::sphere_module;
            shaderCore->loadProgram(0, "--define-macro=_SPHERE_");
            shaderCoreLUT.emplace(std::tuple{"DeflMatShader.cu", ShaderMaker::Sphere}, shaderCore);
        });

        OptixUtil::_compile_group.run([&] () {
            auto shader_string = sutil::lookupIncFile("Light.cu");

            auto shaderCore = std::make_shared<OptixUtil::OptixShaderCore>(shader_string, "__closesthit__radiance", "__anyhit__shadow_cutout");
            shaderCore->loadProgram(0);
            shaderCoreLUT.emplace(std::tuple{"Light.cu", ShaderMaker::Mesh}, shaderCore);
        });

        OptixUtil::_compile_group.run([&] () {
            auto shader_string = sutil::lookupIncFile("Light.cu");

            auto shaderCore = std::make_shared<OptixUtil::OptixShaderCore>(shader_string, "__closesthit__radiance", "__anyhit__shadow_cutout");
            shaderCore->moduleIS = &OptixUtil::sphere_module;
            shaderCore->loadProgram(0);
            shaderCoreLUT.emplace(std::tuple{"Light.cu", ShaderMaker::Sphere}, shaderCore);
        });

        OptixUtil::_compile_group.run([&] () {
            auto shader_string = sutil::lookupIncFile("volume.cu");

            auto shaderCore = std::make_shared<OptixUtil::OptixShaderCore>(shader_string, 
                                                    "__closesthit__radiance_volume", "__anyhit__occlusion_volume", "__intersection__volume");
            shaderCore->loadProgram(0);
            shaderCoreLUT.emplace(std::tuple{"volume.cu", ShaderMaker::Volume}, shaderCore);
        });

        OptixUtil::_compile_group.wait();
    } //hadOnce = true;

    OptixUtil::rtMaterialShaders.resize(0);
    OptixUtil::rtMaterialShaders.resize(shaders.size());

    for (int i = 0; i < shaders.size(); i++) {

OptixUtil::_compile_group.run([&shaders, i] () {

        auto marker = std::string("//PLACEHOLDER");
        auto marker_length = marker.length();

        auto& callable_string = shaders[i]->callable;
        auto start_marker = callable_string.find(marker);

        if (start_marker != std::string::npos) {
            auto end_marker = callable_string.find(marker, start_marker + marker_length);

            callable_string.replace(start_marker, marker_length, "/*PLACEHOLDER");
            callable_string.replace(end_marker, marker_length, "PLACEHOLDER*/");
        }

        std::shared_ptr<OptixUtil::OptixShaderCore> shaderCore = nullptr;
        auto key = std::tuple{shaders[i]->filename, shaders[i]->mark};

        if (shaderCoreLUT.count(key) > 0) {
            shaderCore = shaderCoreLUT.at(key);
        }

        OptixUtil::rtMaterialShaders[i].core = shaderCore;
        OptixUtil::rtMaterialShaders[i].parameters = shaders[i]->parameters;  
        OptixUtil::rtMaterialShaders[i].callable = shaders[i]->callable;

        switch(shaders[i]->mark) {
            case(ShaderMaker::Mesh): {          
                break;
            }
            case(ShaderMaker::Sphere): { 
                break;
            }
            case(ShaderMaker::Volume): {      
                OptixUtil::rtMaterialShaders[i].has_vdb = true; 
                break;
            }
            default: {}
        }

        auto& texs = shaders[i]->tex_names;

        if(texs.size()>0){
            std::cout<<"texSize:"<<texs.size()<<std::endl;
            for(int j=0;j<texs.size();j++)
            {
                std::cout<<"texName:"<<texs[j]<<std::endl;
                OptixUtil::rtMaterialShaders[i].addTexture(j, texs[j]);
            }
        }
}); //_compile_group
    } //for

OptixUtil::_compile_group.wait();
    
    uint task_count = OptixUtil::rtMaterialShaders.size();
    //std::vector<tbb::task_group> task_groups(task_count);
    for(int i=0; i<task_count; ++i)
    {
        OptixUtil::_compile_group.run([&shaders, i] () {
            
            printf("now compiling %d'th shader \n", i);
            if(OptixUtil::rtMaterialShaders[i].loadProgram(i)==false)
            {
                std::cout<<"shader compiling failed, using fallback shader instead"<<std::endl;
                OptixUtil::rtMaterialShaders[i].loadProgram(i, true);
                std::cout<<"shader restored to fallback\n";
            }
        });
    }

    OptixUtil::_compile_group.wait();
    theTimer.tock("Done Optix Shader Compile:");

    if (OptixUtil::sky_tex.has_value()) {
        state.params.sky_texture = OptixUtil::g_tex[OptixUtil::sky_tex.value()]->texture;
        state.params.skynx = OptixUtil::sky_nx_map[OptixUtil::sky_tex.value()];
        state.params.skyny = OptixUtil::sky_ny_map[OptixUtil::sky_tex.value()];
        state.params.envavg = OptixUtil::sky_avg_map[OptixUtil::sky_tex.value()];
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.sky_cdf_p.reset() ),
                              sizeof(float2)*OptixUtil::sky_cdf_map[OptixUtil::sky_tex.value()].size() ) );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.sky_start.reset() ),
                              sizeof(int)*OptixUtil::sky_start_map[OptixUtil::sky_tex.value()].size() ) );
        cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr)state.sky_cdf_p),
                   OptixUtil::sky_cdf_map[OptixUtil::sky_tex.value()].data(),
                   sizeof(float)*OptixUtil::sky_cdf_map[OptixUtil::sky_tex.value()].size(),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr)state.sky_cdf_p)+sizeof(float)*OptixUtil::sky_cdf_map[OptixUtil::sky_tex.value()].size(),
                   OptixUtil::sky_pdf_map[OptixUtil::sky_tex.value()].data(),
                   sizeof(float)*OptixUtil::sky_pdf_map[OptixUtil::sky_tex.value()].size(),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr)state.sky_start),
                   OptixUtil::sky_start_map[OptixUtil::sky_tex.value()].data(),
                   sizeof(int)*OptixUtil::sky_start_map[OptixUtil::sky_tex.value()].size(),
                   cudaMemcpyHostToDevice);
        state.params.skycdf = reinterpret_cast<float *>((CUdeviceptr)state.sky_cdf_p);
        state.params.sky_start = reinterpret_cast<int *>((CUdeviceptr)state.sky_start);

    } else {
        state.params.skynx = 0;
        state.params.skyny = 0;
    }

}

void optixupdateend() {
    camera_changed = true;
        OptixUtil::createPipeline();

    printf("Pipeline created \n");
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

    createSBT( state );
    printf("SBT created \n");

    initLaunchParams( state );
    printf("init params created \n");
}

struct DrawDat {
    std::vector<std::string>  mtlidList;
    std::string mtlid;
    std::string instID;
    std::vector<float> verts;
    std::vector<int> tris;
    std::vector<int> triMats;
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

std::set<std::string> uniqueMatsForMesh() {

    std::set<std::string> result;
    for (auto const &[key, dat]: drawdats) {
        for(auto s:dat.mtlidList) {
          result.insert(s);
        }
    }

    return result;
}

void splitMesh(std::vector<Vertex> & verts, std::vector<uint32_t> &mat_idx, 
std::vector<std::shared_ptr<smallMesh>> &oMeshes, int meshesStart, int vertsStart)
{
    size_t num_tri = (verts.size()-vertsStart)/3;
    oMeshes.resize(meshesStart);
    size_t tris_per_mesh = TRI_PER_MESH;
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
            //auto it = g_mtlidlut.find(dat.mtlid);
            // mtlindex = it != g_mtlidlut.end() ? it->second : 0;
            //zeno::log_error("{} {}", dat.mtlid, mtlindex);
            //#pragma omp parallel for
            for (size_t i = 0; i < dat.tris.size() / 3; i++) {
                int mtidx = dat.triMats[i];
                int mtlindex = 0;
                if(mtidx!=-1) {
                    auto matName = dat.mtlidList[mtidx];
                    auto it = g_mtlidlut.find(matName);
                    mtlindex = it != g_mtlidlut.end() ? it->second : 0;
                }
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

                g_clr[(n + i) * 3 + 0] = toHalf({
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                });
                g_clr[(n + i) * 3 + 1] = toHalf({
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                });
                g_clr[(n + i) * 3 + 2] = toHalf({
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                });

                g_nrm[(n + i) * 3 + 0] = halfNormal({
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                });
                g_nrm[(n + i) * 3 + 1] = halfNormal({
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                });
                g_nrm[(n + i) * 3 + 2] = halfNormal({
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                });

                g_uv[(n + i) * 3 + 0] = toHalf({
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                });
                g_uv[(n + i) * 3 + 1] = toHalf({
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                });
                g_uv[(n + i) * 3 + 2] = toHalf({
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                });

                g_tan[(n + i) * 3 + 0] = halfNormal({
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                });
                g_tan[(n + i) * 3 + 1] = halfNormal({
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                });
                g_tan[(n + i) * 3 + 2] = halfNormal({
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                });
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
    g_clr.resize(g_staticVertNum + n * 3);
    g_nrm.resize(g_staticVertNum + n * 3);
    g_uv.resize(g_staticVertNum + n * 3);
    g_tan.resize(g_staticVertNum + n * 3);
    g_mat_indices.resize(g_staticVertNum/3 + n);
    g_lightMark.resize(g_staticVertNum/3 + n);
    n = 0;
    for (auto const &[key, dat]: drawdats) {
        if(key.find(":static:")==key.npos && dat.instID == "Default") {
//            auto it = g_mtlidlut.find(dat.mtlid);
//            int mtlindex = it != g_mtlidlut.end() ? it->second : 0;
            //zeno::log_error("{} {}", dat.mtlid, mtlindex);
            //#pragma omp parallel for
            for (size_t i = 0; i < dat.tris.size() / 3; i++) {
                int mtidx = dat.triMats[i];
                int mtlindex = 0;
                if(mtidx!=-1) {
                    auto matName = dat.mtlidList[mtidx];
                    auto it = g_mtlidlut.find(matName);
                    mtlindex = it != g_mtlidlut.end() ? it->second : 0;
                }
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

                g_clr[g_staticVertNum + (n + i) * 3 + 0] = toHalf({
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                });
                g_clr[g_staticVertNum + (n + i) * 3 + 1] = toHalf({
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                });
                g_clr[g_staticVertNum + (n + i) * 3 + 2] = toHalf({
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("clr")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                });

                g_nrm[g_staticVertNum + (n + i) * 3 + 0] = halfNormal({
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                });
                g_nrm[g_staticVertNum + (n + i) * 3 + 1] = halfNormal({
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                });
                g_nrm[g_staticVertNum + (n + i) * 3 + 2] = halfNormal({
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("nrm")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                });

                g_uv[g_staticVertNum + (n + i) * 3 + 0] = toHalf({
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                });
                g_uv[g_staticVertNum + (n + i) * 3 + 1] = toHalf({
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                });
                g_uv[g_staticVertNum + (n + i) * 3 + 2] = toHalf({
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("uv")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                });

                g_tan[g_staticVertNum + (n + i) * 3 + 0] = halfNormal({
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 0] * 3 + 2],
                    0,
                });
                g_tan[g_staticVertNum + (n + i) * 3 + 1] = halfNormal({
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 1] * 3 + 2],
                    0,
                });
                g_tan[g_staticVertNum + (n + i) * 3 + 2] = halfNormal({
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 0],
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 1],
                    dat.getAttr("tang")[dat.tris[i * 3 + 2] * 3 + 2],
                    0,
                });
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

//            auto it = g_mtlidlut.find(dat.mtlid);
//            int mtlindex = it != g_mtlidlut.end() ? it->second : 0;
            //zeno::log_error("{} {}", dat.mtlid, mtlindex);
            //#pragma omp parallel for
            for (std::size_t i = 0; i < dat.tris.size() / 3; ++i)
            {
                int mtidx = dat.triMats[i];
                int mtlindex = 0;
                if(mtidx!=-1) {
                    auto matName = dat.mtlidList[mtidx];
                    auto it = g_mtlidlut.find(matName);
                    mtlindex = it != g_mtlidlut.end() ? it->second : 0;
                }
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

//            auto it = g_mtlidlut.find(dat.mtlid);
//            int mtlindex = it != g_mtlidlut.end() ? it->second : 0;
            //zeno::log_error("{} {}", dat.mtlid, mtlindex);
            //#pragma omp parallel for
            for (std::size_t i = 0; i < dat.tris.size() / 3; ++i)
            {
                int mtidx = dat.triMats[i];
                int mtlindex = 0;
                if(mtidx!=-1) {
                    auto matName = dat.mtlidList[mtidx];
                    auto it = g_mtlidlut.find(matName);
                    mtlindex = it != g_mtlidlut.end() ? it->second : 0;
                }
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

void load_object(std::string const &key, std::string const &mtlid, const std::string &instID,
                 float const *verts, size_t numverts, int const *tris, size_t numtris,
                 std::map<std::string, std::pair<float const *, size_t>> const &vtab,
                 int const *matids, std::vector<std::string> const &matNameList) {
    DrawDat &dat = drawdats[key];
    //ZENO_P(mtlid);
    dat.triMats.assign(matids, matids + numtris);
    dat.mtlidList = matNameList;
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
    sphereInstanceGroupAgentList.clear();
    sphereInstanceGroupAgentList.reserve(SpheresInstanceGroupMap.size());

    for (auto &[key, instTrs] : instTrsLUT)
    {
        if ( SpheresInstanceGroupMap.count(instTrs.instID) > 0 ) {

            auto& sphereInstanceBase = SpheresInstanceGroupMap[instTrs.instID];

            auto float_count = instTrs.pos.size();
            auto element_count = float_count/3u;

            auto sia = std::make_shared<SphereInstanceAgent>(sphereInstanceBase);

            sia->radius_list = std::vector<float>(element_count, sphereInstanceBase.radius);
            for (size_t i=0; i<element_count; ++i) {
                sia->radius_list[i] *= instTrs.tang[3*i +0];
            }

            sia->center_list.resize(element_count);
            memcpy(sia->center_list.data(), instTrs.pos.data(), sizeof(float) * float_count);

            sphereInstanceGroupAgentList.push_back(sia);
            continue;
        } // only for sphere

        const auto& instID = instTrs.instID;
        auto& instMat = g_instMatsLUT[instID];
        auto& instAttr = g_instAttrsLUT[instID];
        auto& instScale = g_instScaleLUT[instID];


        const auto& numInstMats = instTrs.pos.size() / 3;
        instMat.resize(numInstMats);
        instScale.resize(numInstMats);
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
            float pScale = instTrs.tang[3*i +0];
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
            instScale[i] = pScale;
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
void set_window_size_v2(int nx, int ny, zeno::vec2i bmin, zeno::vec2i bmax, zeno::vec2i target, bool keepRatio=true) {
  zeno::vec2i t;
  t[0] = target[0]; t[1] = target[1];
  int dx = bmax[0] - bmin[0];
  int dy = bmax[1] - bmin[1];
  if(keepRatio==true)
  {
    t[1] = (int)((float)dy/(float)dx*(float)t[0])+1;
  }
  state.params.width = t[0];
  state.params.height = t[1];
  float sx = (float)t[0]/(float)dx;
  float sy = (float)t[1]/(float)dy;

  state.params.windowSpace = make_int2(sx * (float)nx, sy * (float)ny);
  state.params.windowCrop_min = make_int2(sx * (float)bmin[0], sy * (float)bmin[1]);
  state.params.windowCrop_max = make_int2(sx * (float)bmax[0], sy * (float)bmax[1]);
}
void set_window_size(int nx, int ny) {
    state.params.width = nx;
    state.params.height = ny;
    state.params.windowSpace = make_int2(nx, ny);
    state.params.windowCrop_min = make_int2(0,0);
    state.params.windowCrop_max = make_int2(nx, ny);
    camera_changed = true;
    resize_dirty = true;
}

void set_physical_camera_param(float aperture, float shutter_speed, float iso, bool aces, bool exposure) {
    state.params.physical_camera_aperture = aperture;
    state.params.physical_camera_shutter_speed = shutter_speed;
    state.params.physical_camera_iso = iso;
    state.params.physical_camera_aces = aces;
    state.params.physical_camera_exposure = exposure;
}
void set_perspective(float const *U, float const *V, float const *W, float const *E, float aspect, float fov, float fpd, float aperture) {
    set_perspective_by_fov(U,V,W,E,aspect,fov,0,0.024f,fpd,aperture,0.0f,0.0f,0.0f,0.0f);
}
void set_perspective_by_fov(float const *U, float const *V, float const *W, float const *E, float aspect, float fov, int fov_type, float L, float focal_distance, float aperture, float pitch, float yaw, float h_shift, float v_shift) {
    auto &cam = state.params.cam;
    cam.eye = make_float3(E[0], E[1], E[2]);
    cam.right = normalize(make_float3(U[0], U[1], U[2]));
    cam.up = normalize(make_float3(V[0], V[1], V[2]));
    cam.front = normalize(make_float3(W[0], W[1], W[2]));

    float half_radfov = fov * float(M_PI) / 360.0f;
    float half_tanfov = std::tan(half_radfov);
    cam.focal_length = L / 2.0f / half_tanfov;
    cam.focal_length = std::max(0.0001f,cam.focal_length);
    cam.aperture = std::max(0.0f,aperture);
    

    // L = L/cam.focal_length;
    // cam.focal_length = 1.0f;

    switch (fov_type){
        case 0:
            cam.height = L;
            break;
        case 1:
            cam.height = L / aspect;
            break;
        case 2:
            cam.height = sqrtf(L * L / (1.0f + aspect *aspect));
            break;
    }            
    cam.width = cam.height * aspect;

    cam.pitch = pitch;
    cam.yaw = yaw;
    cam.horizontal_shift = h_shift;
    cam.vertical_shift = v_shift;
    cam.focal_distance = std::max(cam.focal_length, focal_distance);
    camera_changed = true;
}
void set_perspective_by_focal_length(float const *U, float const *V, float const *W, float const *E, float aspect, float focal_length, float w, float h, float focal_distance, float aperture, float pitch, float yaw, float h_shift, float v_shift) {
    auto &cam = state.params.cam;
    cam.eye = make_float3(E[0], E[1], E[2]);
    cam.right = normalize(make_float3(U[0], U[1], U[2]));
    cam.up = normalize(make_float3(V[0], V[1], V[2]));
    cam.front = normalize(make_float3(W[0], W[1], W[2]));

    cam.focal_length = std::max(0.0001f,focal_length);
    cam.aperture = std::max(0.0f,aperture);

    cam.width = w;
    cam.height = h;
    cam.pitch = pitch;
    cam.yaw = yaw;
    cam.horizontal_shift = h_shift;
    cam.vertical_shift = v_shift;
    cam.focal_distance = std::max(cam.focal_length, focal_distance);
    camera_changed = true;
}

void set_outside_random_number(int32_t outside_random_number) {
    state.params.outside_random_number = outside_random_number;
}
static void write_pfm(std::string& path, int w, int h, const float *rgb) {
    std::string header = zeno::format("PF\n{} {}\n-1.0\n", w, h);
    std::vector<char> data(header.size() + w * h * sizeof(zeno::vec3f));
    memcpy(data.data(), header.data(), header.size());
    memcpy(data.data() + header.size(), rgb, w * h * sizeof(zeno::vec3f));
    zeno::file_put_binary(data, path);
}

void *optixgetimg_extra(std::string name) {
    if (name == "diffuse") {
        return output_buffer_diffuse->getHostPointer();
    }
    else if (name == "specular") {
        return output_buffer_specular->getHostPointer();
    }
    else if (name == "transmit") {
        return output_buffer_transmit->getHostPointer();
    }
    else if (name == "background") {
        return output_buffer_background->getHostPointer();
    }
    else if (name == "mask") {
        return output_buffer_mask->getHostPointer();
    }
    else if (name == "color") {
        return output_buffer_color->getHostPointer();
    }
    throw std::runtime_error("invalid optixgetimg_extra name: " + name);
}
static void save_exr(float3* ptr, int w, int h, std::string path) {
    std::vector<float3> data(w * h);
    std::copy_n(ptr, w * h, data.data());
    zeno::image_flip_vertical(data.data(), w, h);
    const char *err = nullptr;
    int ret = SaveEXR((float *) data.data(), w, h, 3, 1, path.c_str(), &err);
    if (ret != 0) {
        if (err) {
            zeno::log_error("failed to perform SaveEXR to {}: {}", path, err);
            FreeEXRErrorMessage(err);
        }
    }
}
void optixrender(int fbo, int samples, bool denoise, bool simpleRender) {

    bool imageRendered = false;
    samples = zeno::envconfig::getInt("SAMPLES", samples);
    // 张心欣老爷请添加环境变量：export ZENO_SAMPLES=256
    zeno::log_debug("rendering samples {}", samples);
    state.params.simpleRender = false;//simpleRender;
    if (!output_buffer_o) throw sutil::Exception("no output_buffer_o");
#ifdef OPTIX_BASE_GL
    if (!gl_display_o) throw sutil::Exception("no gl_display_o");
#endif
    updateState( *output_buffer_o, state.params );
//    updateState( *output_buffer_diffuse, state.params);
//    updateState( *output_buffer_specular, state.params);
//    updateState( *output_buffer_transmit, state.params);
//    updateState( *output_buffer_background, state.params);

    const int max_samples_once = 1;
    for (int f = 0; f < samples; f += max_samples_once) { // 张心欣不要改这里

        state.params.samples_per_launch = std::min(samples - f, max_samples_once);
        launchSubframe( *output_buffer_o, state, denoise);
        state.params.subframe_index++;
    }

#ifdef OPTIX_BASE_GL
    displaySubframe( *output_buffer_o, *gl_display_o, state, fbo );
#endif
    auto &ud = zeno::getSession().userData();
    if (ud.has("optix_image_path")) {
        auto path = ud.get2<std::string>("optix_image_path");
        auto p = (*output_buffer_o).getHostPointer();
        auto w = (*output_buffer_o).width();
        auto h = (*output_buffer_o).height();
        stbi_flip_vertically_on_write(true);
        bool enable_output_aov = zeno::getSession().userData().get2<bool>("output_aov", true);
        bool enable_output_exr = zeno::getSession().userData().get2<bool>("output_exr", true);
        bool enable_output_mask = zeno::getSession().userData().get2<bool>("output_mask", false);
        auto exr_path = path.substr(0, path.size() - 4) + ".exr";
        if (enable_output_mask) {
            std::vector<uint8_t> data;
            data.reserve(w * h * 3);
            float* ptr = (float *)optixgetimg_extra("mask");
            for (auto i = 0; i < w * h * 3; i++) {
                data.push_back(int(ptr[i]));
            }
            std::string native_path = std::filesystem::u8path(path + "_mask.png").string();
            stbi_flip_vertically_on_write(1);
            stbi_write_png(native_path.c_str(), w, h, 3, data.data(),0);
        }
        // AOV
        if (enable_output_aov) {
            SaveMultiLayerEXR(
                    {
                            (float*)optixgetimg_extra("color"),
                            (float*)optixgetimg_extra("diffuse"),
                            (float*)optixgetimg_extra("specular"),
                            (float*)optixgetimg_extra("transmit"),
                            (float*)optixgetimg_extra("background"),
                            (float*)optixgetimg_extra("mask"),
                    },
                    w,
                    h,
                    {
                            "",
                            "diffuse.",
                            "specular.",
                            "transmit.",
                            "background.",
                            "mask.",
                    },
                    exr_path.c_str()
            );
        }
        else {
            if (enable_output_exr) {
                save_exr((float3 *)optixgetimg_extra("color"), w, h, exr_path);
            }
            else {
                std::string jpg_native_path = std::filesystem::u8path(path).string();
                stbi_write_jpg(jpg_native_path.c_str(), w, h, 4, p, 100);
                if (denoise) {
                    const float* _albedo_buffer = reinterpret_cast<float*>(state.albedo_buffer_p.handle);
                    //SaveEXR(_albedo_buffer, w, h, 4, 0, (path+".albedo.exr").c_str(), nullptr);
                    auto a_path = path + ".albedo.pfm";
                    std::string native_a_path = std::filesystem::u8path(a_path).string();
                    write_pfm(native_a_path, w, h, _albedo_buffer);

                    const float* _normal_buffer = reinterpret_cast<float*>(state.normal_buffer_p.handle);
                    //SaveEXR(_normal_buffer, w, h, 4, 0, (path+".normal.exr").c_str(), nullptr);
                    auto n_path = path + ".normal.pfm";
                    std::string native_n_path = std::filesystem::u8path(n_path).string();
                    write_pfm(native_n_path, w, h, _normal_buffer);
                }
            }
        }
        zeno::log_info("optix: saving screenshot {}x{} to {}", w, h, path);
        ud.erase("optix_image_path");

        imageRendered = true;
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

        OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
        OPTIX_CHECK(optixDeviceContextDestroy(state.context));
    }
    catch (sutil::Exception const& e) {
        std::cout << "OptixCleanupError: " << e.what() << std::endl;
    }
////    state.d_vertices.reset();
////    state.d_clr.reset();
////    state.d_mat_indices.reset();
////    state.d_nrm.reset();
////    state.d_tan.reset();
////    state.d_uv.reset();
//        std::memset((void *)&state, 0, sizeof(state));
//        //std::memset((void *)&rtMaterialShaders[0], 0, sizeof(rtMaterialShaders[0]) * rtMaterialShaders.size());
//
//
    context                  .handle=0;
    pipeline                 .handle=0;
    ray_module               .handle=0;
    sphere_module            .handle=0;
    raygen_prog_group        .handle=0;
    radiance_miss_group      .handle=0;
    occlusion_miss_group     .handle=0;

    OptixUtil::shaderCoreLUT.clear();

    output_buffer_o           .reset();
    output_buffer_diffuse     .reset();
    output_buffer_specular    .reset();
    output_buffer_transmit    .reset();
    output_buffer_background  .reset();
    output_buffer_mask        .reset();
    g_StaticMeshPieces        .clear();
    g_meshPieces              .clear();
    state = {};
    isPipelineCreated               = false;


            
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
