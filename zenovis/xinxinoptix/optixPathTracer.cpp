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
#include <zeno/utils/pfm.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/utils/fileio.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/UserData.h>
#include "optixVolume.h"
#include "zeno/core/Session.h"

#include <algorithm>
#include <array>
#include <optional>
#include <cstring>

#include <iostream>
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

#include "ShaderBuffer.h"
#include "TypeCaster.h"

#include "LightBounds.h"
#include "LightTree.h"
#include "Portal.h"
#include "Scene.h"

#include <curve/Hair.h>
#include <curve/optixCurve.h>

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

std::optional<sutil::CUDAOutputBuffer<uchar4>> output_buffer_o;
using Vertex = float3;

struct PathTracerState
{
    OptixTraversableHandle         rootHandleIAS;
    raii<CUdeviceptr>              rootBufferIAS;
    
    raii<CUdeviceptr>              d_uniforms;

    raii<CUstream>                       stream;
    raii<CUdeviceptr> accum_buffer_p;
    raii<CUdeviceptr> albedo_buffer_p;
    raii<CUdeviceptr> normal_buffer_p;

    raii<CUdeviceptr> accum_buffer_d;
    raii<CUdeviceptr> accum_buffer_s;
    raii<CUdeviceptr> accum_buffer_t;
    raii<CUdeviceptr> accum_buffer_b;
    raii<CUdeviceptr> frame_buffer_p;
    raii<CUdeviceptr> accum_buffer_m;

    raii<CUdeviceptr> finite_lights_ptr;

    PortalLightList  plights;
    DistantLightList dlights;

    //std::vector<Portal> portals; 
    
    raii<CUdeviceptr> sky_cdf_p;
    raii<CUdeviceptr> sky_start;
    Params                         params;
    raii<CUdeviceptr>                        d_params;

    OptixShaderBindingTable sbt {};
};

PathTracerState state;



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
    auto& params = state.params;
    params.handle = state.rootHandleIAS;
    
    auto byte_size = params.width * params.height * sizeof( float3 );

    state.accum_buffer_p.resize(byte_size);
    params.accum_buffer = (float3*)(CUdeviceptr)state.accum_buffer_p;
    
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
        params.width * params.height * sizeof( float3 )
            ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.accum_buffer_d .reset()),
        params.width * params.height * sizeof( float3 )
            ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.accum_buffer_s .reset()),
        params.width * params.height * sizeof( float3 )
            ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.accum_buffer_t .reset()),
        params.width * params.height * sizeof( float3 )
            ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.accum_buffer_m .reset()),
        params.width * params.height * sizeof( ushort3 )
            ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.frame_buffer_p .reset()),
        params.width * params.height * sizeof( float3 )
            ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.accum_buffer_b .reset()),
        params.width * params.height * sizeof( ushort1 )
            ) );
    state.params.accum_buffer = (float3*)(CUdeviceptr)state.accum_buffer_p;
    
    state.params.accum_buffer_D = (float3*)(CUdeviceptr)state.accum_buffer_d;
    state.params.accum_buffer_S = (float3*)(CUdeviceptr)state.accum_buffer_s;
    state.params.accum_buffer_T = (float3*)(CUdeviceptr)state.accum_buffer_t;
    state.params.frame_buffer_M = (ushort3*)(CUdeviceptr)state.accum_buffer_m;
    state.params.frame_buffer_P = (float3*)(CUdeviceptr)state.frame_buffer_p;
    state.params.accum_buffer_B = (ushort1*)(CUdeviceptr)state.accum_buffer_b;
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
    state.params.num_lights = defaultScene.lightsWrapper.g_lights.size();
    state.params.denoise = denoise;

        CUDA_CHECK( cudaMemcpy((void*)state.d_params.handle,
                    &state.params, sizeof( Params ),
                    cudaMemcpyHostToDevice
                    ) );
                    
        //timer.tick();
        
        OPTIX_CHECK( optixLaunch(
                    OptixUtil::pipeline,
                    0,
                    (CUdeviceptr)state.d_params.handle,
                    sizeof( Params ),
                    &state.sbt,
                    state.params.width,
                    state.params.height,
                    1
                    ) );

        //timer.tock("frame time");
        output_buffer.unmap();
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

void updateRootIAS()
{
    defaultScene.make_scene(OptixUtil::context, state.rootBufferIAS, state.rootHandleIAS, state.params.cam.eye);
    state.params.handle = state.rootHandleIAS;
    return;

    const auto campos = state.params.cam.eye;
    const float mat3r4c[12] = {1,0,0,-campos.x,   
                               0,1,0,-campos.y,   
                               0,0,1,-campos.z};

    std::vector<OptixInstance> optix_instances{};
    uint optix_instance_idx = 0u;
    uint sbt_offset = 0u;

    auto op_index = optix_instances.size();

    uint32_t MAX_INSTANCE_ID;
    optixDeviceContextGetProperty( OptixUtil::context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID, &MAX_INSTANCE_ID, sizeof(MAX_INSTANCE_ID) );
    state.params.maxInstanceID = MAX_INSTANCE_ID;

    for (auto& [key, val] : hair_yyy_cache) {

        OptixInstance opinstance {};
        auto& [filePath, mode, mtlid] = key;
        
        auto shader_mark = mode + 3;

		auto combinedID = std::tuple(mtlid, (ShaderMark)shader_mark);
		auto shader_index = defaultScene.shader_indice_table[combinedID];

        auto& hair_state = geo_hair_cache[ std::tuple(filePath, mode) ];

		opinstance.flags = OPTIX_INSTANCE_FLAG_NONE;
		//opinstance.instanceId = op_index++;
		opinstance.sbtOffset = shader_index * RAY_TYPE_COUNT;
		opinstance.visibilityMask = DefaultMatMask;
		opinstance.traversableHandle = hair_state->node->handle;

        // sutil::Matrix3x4 yUpTransform = {
        //     0.0f, 1.0f, 0.0f, -campos.x,
        //     0.0f, 0.0f, 1.0f, -campos.y,
        //     1.0f, 0.0f, 0.0f, -campos.z,
        // };

        for (auto& trans : val) {

            auto dummy = glm::transpose(trans);
            auto dummy_ptr = glm::value_ptr( dummy );

		    memcpy(opinstance.transform, dummy_ptr, sizeof(float) * 12);
            opinstance.transform[3]  += -campos.x;
            opinstance.transform[7]  += -campos.y;
            opinstance.transform[11] += -campos.z;

            opinstance.instanceId = op_index++;
		    optix_instances.push_back( opinstance );
        }
    }
}

static void createSBT( PathTracerState& state )
{
    raii<CUdeviceptr>  &d_raygen_record = OptixUtil::d_raygen_record;
    const size_t raygen_record_size = sizeof( RayGenRecord );
    d_raygen_record.resize(raygen_record_size);

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( OptixUtil::raygen_prog_group, &rg_sbt ) );

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)d_raygen_record ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ) );

    raii<CUdeviceptr>  &d_miss_records = OptixUtil::d_miss_records;
    const size_t miss_record_size = sizeof( MissRecord );
    d_miss_records.resize(miss_record_size * RAY_TYPE_COUNT);

    MissRecord ms_sbt[2];
    OPTIX_CHECK_LOG( optixSbtRecordPackHeader( OptixUtil::radiance_miss_group,  &ms_sbt[0] ) );
    ms_sbt[0].data.bg_color = make_float4( 0.0f );
    OPTIX_CHECK_LOG( optixSbtRecordPackHeader( OptixUtil::occlusion_miss_group, &ms_sbt[1] ) );
    ms_sbt[1].data.bg_color = make_float4( 0.0f );

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr&)d_miss_records ),
                ms_sbt,
                miss_record_size * RAY_TYPE_COUNT,
                cudaMemcpyHostToDevice
                ) );

    const auto shader_count = OptixUtil::rtMaterialShaders.size();

    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    const size_t hitgroup_record_count = shader_count * RAY_TYPE_COUNT;

    raii<CUdeviceptr>  &d_hitgroup_records = OptixUtil::d_hitgroup_records;
    d_hitgroup_records.resize(hitgroup_record_size * hitgroup_record_count);

    std::vector<HitGroupRecord> hitgroup_records(hitgroup_record_count);
    std::vector<CallablesRecord> callable_records(shader_count);

    for( int j = 0; j < shader_count; ++j ) {

        auto& shader_ref = OptixUtil::rtMaterialShaders[j];
        const auto has_vdb = shader_ref.has_vdb;

        const uint sbt_idx = RAY_TYPE_COUNT * j;

        OPTIX_CHECK( optixSbtRecordPackHeader( shader_ref.callable_prog_group, &callable_records[j] ) );

        if (!has_vdb) {

            hitgroup_records[sbt_idx] = {};
            hitgroup_records[sbt_idx].data.uniforms = reinterpret_cast<float4*>( (CUdeviceptr)state.d_uniforms );

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

                const auto& vdbs = shader_ref.vbds;
				for(uint t=0; t<min(vdbs.size(), 8ull); ++t)
				{
					auto vdb_key = vdbs[t];
                    if (defaultScene._vdb_grids_cached.count(vdb_key)==0) continue;

					auto vdb_ptr = defaultScene._vdb_grids_cached.at(vdb_key);
					rec.data.vdb_grids[t] = vdb_ptr->grids.front()->deviceptr;
					rec.data.vdb_max_v[t] = vdb_ptr->grids.front()->max_value;
				}

            for(uint t=0;t<32;t++)
            {
                rec.data.textures[t] = shader_ref.getTexture(t);
            }
            if (!shader_ref.parameters.empty())
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
        raii<CUdeviceptr>& d_callable_records = OptixUtil::d_callable_records;
        size_t      sizeof_callable_record = sizeof( CallablesRecord );

        d_callable_records.resize( sizeof_callable_record * shader_count );

        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)d_callable_records ), callable_records.data(),
                                sizeof_callable_record * shader_count, cudaMemcpyHostToDevice ) );

        state.sbt.callablesRecordBase          = d_callable_records;
        state.sbt.callablesRecordCount         = shader_count;
        state.sbt.callablesRecordStrideInBytes = static_cast<unsigned int>( sizeof_callable_record );
    }
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

    //CUDA_CHECK( cudaStreamCreate( &state.stream.reset() ) );
    state.d_params.resize( sizeof( Params ) );

    if (!output_buffer_o) {
      output_buffer_o.emplace(
          output_buffer_type,
          state.params.width,
          state.params.height
      );
      output_buffer_o->setStream( 0 );
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

    OptixUtil::default_sky_tex = cur_path + "/hdr/Panorama.hdr";
    OptixUtil::sky_tex = OptixUtil::default_sky_tex;

    OptixUtil::setSkyTexture(OptixUtil::sky_tex.value());
    xinxinoptix::update_hdr_sky(0, {0, 0, 0}, 0.8);
}

static std::map<std::string, LightDat> lightdats;
static std::vector<float2>  triangleLightCoords;
static std::vector<float3>  triangleLightNormals;

const std::map<std::string, LightDat> &get_lightdats() {
    return lightdats;
}

void unload_light(){

    lightdats.clear();
    triangleLightCoords.clear();
    triangleLightNormals.clear();

    state.dlights = {};
    state.plights = {};

    state.params.dlights_ptr = 0llu;
    state.params.plights_ptr = 0llu;

    OptixUtil::portal_delayed.reset();

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
    if (enable != state.params.show_background) {
        state.params.show_background = enable;
        state.params.subframe_index = 0;
    }
}

void updatePortalLights(const std::vector<Portal>& portals) {

    decltype(OptixUtil::tex_lut)::const_accessor tex_accessor;
    OptixUtil::tex_lut.find(tex_accessor, {OptixUtil::sky_tex.value(), false});
    
    auto &tex = tex_accessor->second;

    auto& pll = state.plights;
    auto& pls = pll.list;
    pls.clear();
    pls.reserve(std::max(portals.size(), size_t(0)) );

    glm::mat4 rotation = glm::mat4(1.0f);
    rotation = glm::rotate(rotation, glm::radians(state.params.sky_rot_y), glm::vec3(0,1,0));
    rotation = glm::rotate(rotation, glm::radians(state.params.sky_rot_x), glm::vec3(1,0,0));
    rotation = glm::rotate(rotation, glm::radians(state.params.sky_rot_z), glm::vec3(0,0,1));
    rotation = glm::rotate(rotation, glm::radians(state.params.sky_rot), glm::vec3(0,1,0));
    
    glm::mat4* rotation_ptr = nullptr;
    if ( glm::mat4(1.0f) != rotation ) {
        rotation_ptr = &rotation;
    }

    for (auto& portal : portals) {
        auto pl = PortalLight(portal, (float3*)tex->rawData.data(), tex->width, tex->height, rotation_ptr);
        pls.push_back(std::move(pl));
    }

    state.params.plights_ptr = (void*)pll.upload();
}

void updateDistantLights(std::vector<zeno::DistantLightData>& dldl) 
{
    if (dldl.empty()) {
        state.dlights = {};
        state.params.dlights_ptr = 0u;
        return;
    }

    float power = 0.0f;

    std::vector<float> cdf; cdf.reserve(dldl.size());

    for (auto& dld : dldl) {
        auto ppp = dld.color * dld.intensity;
        power += (ppp[0] + ppp[1] + ppp[2]) / 3.0f;
        cdf.push_back(power);
    }

    for(auto& c : cdf) {
        c /= power;
    }

    state.dlights = DistantLightList {dldl, cdf};
    state.params.dlights_ptr = (void*)state.dlights.upload();
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
    auto& geo = defaultScene.lightsWrapper._triangleLightGeo;
    geo.push_back(p0); geo.push_back(p1); geo.push_back(p2);
}

static void addLightPlane(float3 p0, float3 v1, float3 v2, float3 normal)
{
    float3 vert0 = p0, vert1 = p0 + v1, vert2 = p0 + v2, vert3 = p0 + v1 + v2;

    auto& geo = defaultScene.lightsWrapper._planeLightGeo;

    geo.push_back(make_float3(vert0.x, vert0.y, vert0.z));
    geo.push_back(make_float3(vert1.x, vert1.y, vert1.z));
    geo.push_back(make_float3(vert3.x, vert3.y, vert3.z));
   
    geo.push_back(make_float3(vert0.x, vert0.y, vert0.z));
    geo.push_back(make_float3(vert3.x, vert3.y, vert3.z));
    geo.push_back(make_float3(vert2.x, vert2.y, vert2.z));
}

static void addLightSphere(float3 center, float radius) 
{
    float4 vt {center.x, center.y, center.z, radius};
    defaultScene.lightsWrapper._sphereLightGeo.push_back(vt);
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
    
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &d_lightMesh ), vertices_size_in_bytes, 0 ) );
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
    triangle_input.triangleArray.sbtIndexOffsetBuffer        = 0;
    // triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
    // triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );

    OptixAccelBuildOptions accel_options {};
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
    
    buildXAS(OptixUtil::context, accel_options, triangle_input, bufferGas, handleGas);
}

static void buildLightSpheresGAS( PathTracerState& state, std::vector<float4>& lightSpheres, raii<CUdeviceptr>& bufferGas, OptixTraversableHandle& handleGas) {

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
        auto data_length = sizeof( float4 ) * sphere_count;

        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &d_vertex_buffer.reset() ), data_length, 0) );
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

    buildXAS(OptixUtil::context, accel_options, sphere_input, bufferGas, handleGas);
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
    
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &d_lightMesh ), vertices_size_in_bytes, 0 ) );
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
    
    buildXAS(OptixUtil::context, accel_options, triangle_input, bufferGas, handleGas);
}

void buildLightTree() {
    camera_changed = true;
    state.finite_lights_ptr.reset();

    state.params.lightTreeSampler = 0llu;
    state.params.triangleLightCoordsBuffer = 0llu;
    state.params.triangleLightNormalBuffer = 0llu;
    
    state.params.firstRectLightIdx = UINT_MAX;
    state.params.firstSphereLightIdx = UINT_MAX;
    state.params.firstTriangleLightIdx = UINT_MAX;

    state.params.lights = 0llu;
    state.params.num_lights = 0u;

    auto& lightsWrapper = defaultScene.lightsWrapper;
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

        CUDA_CHECK( cudaMallocAsync(reinterpret_cast<void**>( &lightsWrapper.triangleLightNormals.reset() ), byte_size, 0) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)lightsWrapper.triangleLightNormals ),
                                triangleLightNormals.data(), byte_size, cudaMemcpyHostToDevice) );
        state.params.triangleLightNormalBuffer = lightsWrapper.triangleLightNormals.handle;
    }

    {
        auto byte_size = triangleLightCoords.size() * sizeof(float2);

        CUDA_CHECK( cudaMallocAsync(reinterpret_cast<void**>( &lightsWrapper.triangleLightCoords.reset() ), byte_size, 0) );
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
            light.point = {center, radius};
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

        OptixUtil::TexKey tex_key {dat.textureKey, false};
        
        if ( OptixUtil::tex_lut.count( tex_key ) > 0 ) {

            decltype(OptixUtil::tex_lut)::const_accessor tex_accessor;
            OptixUtil::tex_lut.find(tex_accessor, tex_key);

            auto& val = tex_accessor->second;
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

    CUDA_CHECK( cudaMallocAsync(
        reinterpret_cast<void**>( &state.finite_lights_ptr.reset() ),
        sizeof( GenericLight ) * std::max(lightsWrapper.g_lights.size(),(size_t)1), 0 ) );

    state.params.lights = (GenericLight*)(CUdeviceptr)state.finite_lights_ptr;
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)state.finite_lights_ptr ),
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
        CUDA_CHECK( cudaMallocAsync(reinterpret_cast<void**>( &lightTreeDummyPtr.reset() ), sizeof( dummy ), 0) );
        CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( (CUdeviceptr)lightTreeDummyPtr ),
                &dummy, sizeof( dummy ), cudaMemcpyHostToDevice) );
        state.params.lightTreeSampler = lightTreeDummyPtr.handle;
    }

    defaultScene.prepare_light_ias(OptixUtil::context);
}

inline std::map<std::tuple<std::string, ShaderMark>, std::shared_ptr<OptixUtil::OptixShaderCore>> shaderCoreLUT {};

void updateShaders(std::vector<std::shared_ptr<ShaderPrepared>> &shaders, 
                   bool requireTriangObj, bool requireTriangLight, 
                   bool requireSphereObj, bool requireSphereLight, 
                   bool requireVolumeObj, uint usesCurveTypeFlags, bool refresh)
{
    camera_changed = true;

    CppTimer theTimer;
    theTimer.tick();
    
    if (refresh) {

        shaderCoreLUT = {};
        OptixUtil::_compile_group.run([&] () {

            if (!OptixUtil::createModule(
                OptixUtil::raygen_module.reset(),
                OptixUtil::context,
                sutil::lookupIncFile("PTKernel.cu"),
                "PTKernel.cu")) throw std::runtime_error("base ray module failed to compile");

            OptixUtil::createRenderGroups(OptixUtil::context, OptixUtil::raygen_module);
        });

        if (requireTriangObj) {

            OptixUtil::_compile_group.run([&] () {
                auto shader_string = sutil::lookupIncFile("DeflMatShader.cu"); 
                auto shaderCore = std::make_shared<OptixUtil::OptixShaderCore>(shader_string, "__closesthit__radiance", "__anyhit__shadow_cutout");

                std::vector<std::string> macros = {"--undefine-macro=_P_TYPE_", "--define-macro=_P_TYPE_=0"};
                shaderCore->loadProgram(0, macros);
                shaderCoreLUT[ std::tuple{"DeflMatShader.cu", ShaderMark::Mesh} ] = shaderCore;
            });    
        }

        if (requireSphereObj) {

            OptixUtil::_compile_group.run([&] () {
                auto shader_string = sutil::lookupIncFile("DeflMatShader.cu"); 
                auto shaderCore = std::make_shared<OptixUtil::OptixShaderCore>(shader_string, "__closesthit__radiance", "__anyhit__shadow_cutout");
                shaderCore->moduleIS = &OptixUtil::sphere_ism;

                std::vector<std::string> macros = {"--undefine-macro=_P_TYPE_", "--define-macro=_P_TYPE_=1"};
                shaderCore->loadProgram(1, macros);
                shaderCoreLUT[ std::tuple{"DeflMatShader.cu", ShaderMark::Sphere} ] = shaderCore;
            });
        }

        if (requireVolumeObj) {

            OptixUtil::_compile_group.run([&] () {
                auto shader_string = sutil::lookupIncFile("volume.cu");
                auto shaderCore = std::make_shared<OptixUtil::OptixShaderCore>(shader_string, 
                                                        "__closesthit__radiance_volume", "__anyhit__occlusion_volume", "__intersection__volume");
                shaderCore->loadProgram(4);
                shaderCoreLUT[ std::tuple{"volume.cu", ShaderMark::Volume} ] = shaderCore;
            });
        }

        if (usesCurveTypeFlags) {

            OptixUtil::_compile_group.run([&] () {
                auto shader_string = sutil::lookupIncFile("DeflMatShader.cu"); 
                std::vector<std::string> macros = {"--undefine-macro=_P_TYPE_", "--define-macro=_P_TYPE_=2"};

                const std::vector<std::tuple<OptixPrimitiveTypeFlags, OptixModule*, ShaderMark>> curveTypes {

                    { OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE, &OptixUtil::round_quadratic_ism, ShaderMark::CURVE_QUADRATIC },
                    { OPTIX_PRIMITIVE_TYPE_FLAGS_FLAT_QUADRATIC_BSPLINE,  &OptixUtil::flat_quadratic_ism,  ShaderMark::CURVE_RIBBON    },
                    { OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE,     &OptixUtil::round_cubic_ism,     ShaderMark::CURVE_CUBIC     },

                    { OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR,       &OptixUtil::round_linear_ism, ShaderMark::CURVE_LINEAR },
                    { OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM,   &OptixUtil::round_catrom_ism, ShaderMark::CURVE_CATROM },
                    { OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BEZIER, &OptixUtil::round_bezier_ism, ShaderMark::CURVE_BEZIER },
                };

                for (auto& [tflag, module_ptr, mark]: curveTypes) {

                    bool ignore = !(usesCurveTypeFlags & tflag);
                    if (ignore) continue;

                    auto shaderCore = std::make_shared<OptixUtil::OptixShaderCore>(shader_string, "__closesthit__radiance", "__anyhit__shadow_cutout");
                    shaderCore->moduleIS = module_ptr;

                    shaderCore->loadProgram(10, macros);
                    shaderCoreLUT[ std::tuple{"DeflMatShader.cu", mark} ] = shaderCore;
                }
            });
        } // usesCurveTypeFlags
        
    } // refresh

    OptixUtil::_compile_group.run([&] () {
        auto shader_string = sutil::lookupIncFile("Light.cu");

        if (requireTriangLight) {
            auto shaderCore = std::make_shared<OptixUtil::OptixShaderCore>(shader_string, "__closesthit__radiance", "__anyhit__shadow_cutout");
            shaderCore->loadProgram(2);
            shaderCoreLUT[ std::tuple{"Light.cu", ShaderMark::Mesh} ] = shaderCore;
        }

        if (requireSphereLight) {
            auto shaderCore = std::make_shared<OptixUtil::OptixShaderCore>(shader_string, "__closesthit__radiance", "__anyhit__shadow_cutout");
            shaderCore->moduleIS = &OptixUtil::sphere_ism;
            shaderCore->loadProgram(3);
            shaderCoreLUT[ std::tuple{"Light.cu", ShaderMark::Sphere} ] = shaderCore;
        }
    });

    OptixUtil::_compile_group.wait();

    //OptixUtil::rtMaterialShaders.clear();
    OptixUtil::rtMaterialShaders.resize(shaders.size());

    for (int i = 0; i < shaders.size(); i++) {
        if (!shaders[i]->dirty) continue;

        shaders[i]->dirty = false;
        OptixUtil::rtMaterialShaders[i].dirty = true;

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

        auto& rtShader = OptixUtil::rtMaterialShaders[i];

        rtShader.core = shaderCore;
        rtShader.callable = shaders[i]->callable;
        rtShader.parameters = shaders[i]->parameters;
        
        auto macro = globalShaderBufferGroup.code(callable_string);
        rtShader.macros = macro;

        if (ShaderMark::Volume == shaders[i]->mark) {
            rtShader.has_vdb = true; 
        }

        auto& texs = shaders[i]->texs;
        rtShader.texs = {};
        rtShader.texs.reserve(texs.size());
        for(int j=0; j<texs.size(); j++) {
            rtShader.texs.push_back(texs[j]->texture);
        }

        auto& vdbs = shaders[i]->vdb_keys;
        for (int j=0; j<shaders[i]->vdb_keys.size(); ++j)
        {
            rtShader.vbds.push_back(vdbs[j]);
        }
}); //_compile_group
    } //for

OptixUtil::_compile_group.wait();
    
    uint task_count = OptixUtil::rtMaterialShaders.size();
    //std::vector<tbb::task_group> task_groups(task_count);
    for(int i=0; i<task_count; ++i)
    {
        auto& shader_ref = OptixUtil::rtMaterialShaders[i];
        if (!refresh && !shader_ref.dirty) continue;
        shader_ref.dirty = false;

        OptixUtil::_compile_group.run([&shaders, i] () {

            auto fallback = shaders[i]->matid == "Default";
            fallback |= shaders[i]->matid == "Light";
            
            //("now compiling %d'th shader \n", i);
            if(OptixUtil::rtMaterialShaders[i].loadProgram(i, fallback)==false)
            {
                std::cout<<"shader compiling failed, using fallback shader instead"<<std::endl;
                OptixUtil::rtMaterialShaders[i].loadProgram(i, true);
                std::cout<<"shader restored to fallback\n";
            }
        });
    }

    OptixUtil::_compile_group.wait();
//    theTimer.tock("Done Optix Shader Compile:");

    if (OptixUtil::sky_tex.has_value() && OptixUtil::sky_tex_ptr!=nullptr) {

        auto& tex = OptixUtil::sky_tex_ptr;
        if (tex->texture == state.params.sky_texture) return;

        state.params.sky_texture = tex->texture;
        state.params.skynx = tex->width;
        state.params.skyny = tex->height;
        state.params.envavg = tex->average;

        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.sky_cdf_p.reset() ),
                            sizeof(float)*tex->cdf.size(), 0 ) );
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.sky_start.reset() ),
                              sizeof(int)*tex->start.size(), 0 ) );

        cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr)state.sky_cdf_p),
                   tex->cdf.data(),
                   sizeof(float)*tex->cdf.size(),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(reinterpret_cast<char *>((CUdeviceptr)state.sky_start),
                   tex->start.data(),
                   sizeof(int)*tex->start.size(),
                   cudaMemcpyHostToDevice);

        state.params.skycdf = reinterpret_cast<float *>((CUdeviceptr)state.sky_cdf_p);
        state.params.sky_start = reinterpret_cast<int*>((CUdeviceptr)state.sky_start);

    } else {
        state.params.skynx = 0;
        state.params.skyny = 0;
    }

}

void configPipeline(bool shaderDirty) {
    camera_changed = true;

    auto buffers = globalShaderBufferGroup.upload();
    state.params.global_buffers = (void*)buffers;

    CppTimer timer;

    if (shaderDirty) {
        timer.tick();
        createSBT( state );
        timer.tock("SBT created \n");
    }

    timer.tick();
    OptixUtil::createPipeline(defaultScene.maxNodeDepth, shaderDirty);
    timer.tock("Pipeline created \n");

    timer.tick();
    initLaunchParams( state );
    timer.tock("init params created \n");
}

void prepareScene()
{
    defaultScene.make_scene(OptixUtil::context, state.rootBufferIAS, state.rootHandleIAS);
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
}
void set_window_size(int nx, int ny) {
    state.params.width = nx;
    state.params.height = ny;
    camera_changed = true;
    resize_dirty = true;
}

void set_physical_camera_param(float aperture, float shutter_speed, float iso, bool aces, bool exposure, bool panorama_camera, bool panorama_vr180, float pupillary_distance) {
    state.params.physical_camera_aperture = aperture;
    state.params.physical_camera_shutter_speed = shutter_speed;
    state.params.physical_camera_iso = iso;
    state.params.physical_camera_aces = aces;
    state.params.physical_camera_exposure = exposure;
    state.params.physical_camera_panorama_camera = panorama_camera;
    state.params.physical_camera_panorama_vr180 = panorama_vr180;
    state.params.physical_camera_pupillary_distance = pupillary_distance;
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

std::vector<float> optixgetimg_extra2(std::string name, int w, int h) {
    std::vector<float> tex_data(w * h * 3);
    if (name == "diffuse") {
        cudaMemcpy(tex_data.data(), (void*)state.accum_buffer_d.handle, sizeof(float) * tex_data.size(), cudaMemcpyDeviceToHost);
    }
    else if (name == "specular") {
        cudaMemcpy(tex_data.data(), (void*)state.accum_buffer_s.handle, sizeof(float) * tex_data.size(), cudaMemcpyDeviceToHost);
    }
    else if (name == "transmit") {
        cudaMemcpy(tex_data.data(), (void*)state.accum_buffer_t.handle, sizeof(float) * tex_data.size(), cudaMemcpyDeviceToHost);
    }
    else if (name == "background") {
        std::vector<ushort1> temp_buffer(w * h);
        cudaMemcpy(temp_buffer.data(), (void*)state.accum_buffer_b.handle, sizeof(ushort1) * temp_buffer.size(), cudaMemcpyDeviceToHost);
        for (auto i = 0; i < temp_buffer.size(); i++) {
            float v = toFloat(temp_buffer[i]);
            tex_data[i * 3 + 0] = v;
            tex_data[i * 3 + 1] = v;
            tex_data[i * 3 + 2] = v;
        }
    }
    else if (name == "mask") {
        std::vector<ushort3> temp_buffer(w * h);
        cudaMemcpy(temp_buffer.data(), (void*)state.accum_buffer_m.handle, sizeof(ushort3) * temp_buffer.size(), cudaMemcpyDeviceToHost);
        for (auto i = 0; i < temp_buffer.size(); i++) {
            float3 v = toFloat(temp_buffer[i]);
            tex_data[i * 3 + 0] = v.x;
            tex_data[i * 3 + 1] = v.y;
            tex_data[i * 3 + 2] = v.z;
        }
    }
    else if (name == "pos") {
        cudaMemcpy(tex_data.data(), (void*)state.frame_buffer_p.handle, sizeof(float) * tex_data.size(), cudaMemcpyDeviceToHost);
    }
    else if (name == "color") {
        cudaMemcpy(tex_data.data(), (void*)state.accum_buffer_p.handle, sizeof(float) * tex_data.size(), cudaMemcpyDeviceToHost);
    }
    else {
        throw std::runtime_error("invalid optixgetimg_extra name: " + name);
    }
    return tex_data;
}

std::vector<half> optixgetimg_extra3(std::string name, int w, int h) {
    std::vector<half> tex_data(w * h * 3);
    if (name == "diffuse") {
        std::vector<float> temp_buffer(w * h * 3);
        cudaMemcpy(temp_buffer.data(), (void*)state.accum_buffer_d.handle, sizeof(temp_buffer[0]) * temp_buffer.size(), cudaMemcpyDeviceToHost);
        for (auto i = 0; i < temp_buffer.size(); i++) {
            tex_data[i] = temp_buffer[i];
        }
    }
    else if (name == "specular") {
        std::vector<float> temp_buffer(w * h * 3);
        cudaMemcpy(temp_buffer.data(), (void*)state.accum_buffer_s.handle, sizeof(temp_buffer[0]) * temp_buffer.size(), cudaMemcpyDeviceToHost);
        for (auto i = 0; i < temp_buffer.size(); i++) {
            tex_data[i] = temp_buffer[i];
        }
    }
    else if (name == "transmit") {
        std::vector<float> temp_buffer(w * h * 3);
        cudaMemcpy(temp_buffer.data(), (void*)state.accum_buffer_t.handle, sizeof(temp_buffer[0]) * temp_buffer.size(), cudaMemcpyDeviceToHost);
        for (auto i = 0; i < temp_buffer.size(); i++) {
            tex_data[i] = temp_buffer[i];
        }
    }
    else if (name == "background") {
        std::vector<half> temp_buffer(w * h);
        cudaMemcpy(temp_buffer.data(), (void*)state.accum_buffer_b.handle, sizeof(temp_buffer[0]) * temp_buffer.size(), cudaMemcpyDeviceToHost);
        for (auto i = 0; i < temp_buffer.size(); i++) {
            tex_data[i * 3 + 0] = temp_buffer[i];
            tex_data[i * 3 + 1] = temp_buffer[i];
            tex_data[i * 3 + 2] = temp_buffer[i];
        }
    }
    else if (name == "mask") {
        cudaMemcpy(tex_data.data(), (void*)state.accum_buffer_m.handle, sizeof(half) * tex_data.size(), cudaMemcpyDeviceToHost);
    }
    else if (name == "pos") {
        cudaMemcpy(tex_data.data(), (void*)state.frame_buffer_p.handle, sizeof(float) * tex_data.size(), cudaMemcpyDeviceToHost);
    }
    else if (name == "color") {
        std::vector<float> temp_buffer(w * h * 3);
        cudaMemcpy(temp_buffer.data(), (void*)state.accum_buffer_p.handle, sizeof(temp_buffer[0]) * temp_buffer.size(), cudaMemcpyDeviceToHost);
        for (auto i = 0; i < temp_buffer.size(); i++) {
            tex_data[i] = temp_buffer[i];
        }
    }
    else {
        throw std::runtime_error("invalid optixgetimg_extra name: " + name);
    }
    zeno::image_flip_vertical((ushort3*)tex_data.data(), w, h);
    return tex_data;
}

glm::vec3 get_click_pos(int x, int y) {
    int w = state.params.width;
    int h = state.params.height;
    auto frame_buffer_pos = optixgetimg_extra2("pos", w, h);
    auto index = x + (h - 1 - y) * w;
    auto posWS = ((glm::vec3*)frame_buffer_pos.data())[index];
    return posWS;
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
static void save_png_data(std::string path, int w, int h, float* ptr) {
    std::vector<uint8_t> data;
    data.reserve(w * h * 3);
    for (auto i = 0; i < w * h * 3; i++) {
        data.push_back(std::lround(ptr[i] * 255.0f));
    }
    std::string native_path = zeno::create_directories_when_write_file(path);
    stbi_flip_vertically_on_write(1);
    stbi_write_png(native_path.c_str(), w, h, 3, data.data(),0);
}
static void save_png_color(std::string path, int w, int h, float* ptr) {
    std::vector<uint8_t> data;
    data.reserve(w * h * 3);
    for (auto i = 0; i < w * h * 3; i++) {
        data.push_back(std::lround(pow(ptr[i], 1.0f/2.2f) * 255.0f));
    }
    std::string native_path = zeno::create_directories_when_write_file(path);
    stbi_flip_vertically_on_write(1);
    stbi_write_png(native_path.c_str(), w, h, 3, data.data(),0);
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

    if (denoise) {
        auto w = state.params.width;
        auto h = state.params.height;
        size_t byte_size = sizeof(float3) * w * h;
        state.albedo_buffer_p.resize(byte_size);
        state.normal_buffer_p.resize(byte_size);
    } else {
        state.albedo_buffer_p.reset();
        state.normal_buffer_p.reset();
    }
    state.params.albedo_buffer = (float3*)state.albedo_buffer_p.handle;
    state.params.normal_buffer = (float3*)state.normal_buffer_p.handle;

    auto &ud = zeno::getSession().userData();
    const int max_samples_once = 1;
    for (int f = 0; f < samples; f += max_samples_once) { // 张心欣不要改这里
        if (ud.get2<bool>("viewport-optix-pause", false)) {
            continue;
        }

        state.params.samples_per_launch = std::min(samples - f, max_samples_once);
        launchSubframe( *output_buffer_o, state, denoise);
        state.params.subframe_index++;
    }

#ifdef OPTIX_BASE_GL
    displaySubframe( *output_buffer_o, *gl_display_o, state, fbo );
#endif
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
            path = path.substr(0, path.size() - 4);
            save_png_data(path + "_mask.png", w, h,  optixgetimg_extra2("mask", w, h).data());
        }
        // AOV
        if (enable_output_aov) {
            if (enable_output_exr) {
                zeno::create_directories_when_write_file(exr_path);
                SaveMultiLayerEXR_half(
                        {
                                optixgetimg_extra3("color", w, h).data(),
                                optixgetimg_extra3("diffuse", w, h).data(),
                                optixgetimg_extra3("specular", w, h).data(),
                                optixgetimg_extra3("transmit", w, h).data(),
                                optixgetimg_extra3("background", w, h).data(),
                                optixgetimg_extra3("mask", w, h).data(),
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
                path = path.substr(0, path.size() - 4);
                save_png_color(path + ".aov.diffuse.png",   w, h,  optixgetimg_extra2("diffuse", w, h).data());
                save_png_color(path + ".aov.specular.png",  w, h,  optixgetimg_extra2("specular", w, h).data());
                save_png_color(path + ".aov.transmit.png",  w, h,  optixgetimg_extra2("transmit", w, h).data());
                save_png_data(path + ".aov.background.png", w, h,  optixgetimg_extra2("background", w, h).data());
                save_png_data(path + ".aov.mask.png",       w, h,  optixgetimg_extra2("mask", w, h).data());
            }
        }
        else {
            if (enable_output_exr) {
                zeno::create_directories_when_write_file(exr_path);
                save_exr((float3 *)optixgetimg_extra2("color", w, h).data(), w, h, exr_path);
            }
            else {
                std::string jpg_native_path = zeno::create_directories_when_write_file(path);
                stbi_write_jpg(jpg_native_path.c_str(), w, h, 4, p, 100);
                if (denoise) {
                    auto byte_size = state.albedo_buffer_p.size;
                    std::vector<std::byte> temp; temp.resize(byte_size); 

                    const float* _albedo_buffer = reinterpret_cast<float*>(state.albedo_buffer_p.handle);
                    cudaMemcpy(temp.data(), _albedo_buffer, byte_size, cudaMemcpyDeviceToHost);
                    
                    auto a_path = path + ".albedo.pfm";
                    std::string native_a_path = zeno::create_directories_when_write_file(a_path);
                    zeno::write_pfm(native_a_path.c_str(), w, h, (float*)temp.data());

                    const float* _normal_buffer = reinterpret_cast<float*>(state.normal_buffer_p.handle);
                    cudaMemcpy(temp.data(), _normal_buffer, byte_size, cudaMemcpyDeviceToHost);

                    auto n_path = path + ".normal.pfm";
                    std::string native_n_path = zeno::create_directories_when_write_file(n_path);
                    zeno::write_pfm(native_n_path.c_str(), w, h, (float*)temp.data());
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

void optixCleanup() {

    state.dlights = {};
    state.params.dlights_ptr = 0u;

    state.plights = {};
    state.params.plights_ptr = 0u;

    defaultScene.lightsWrapper.reset();
    state.finite_lights_ptr.reset();
    
    state.params.sky_strength = 1.0f;

    std::vector<OptixUtil::TexKey> keys;

    for (auto& [k, _] : OptixUtil::tex_lut) {
        if (k.path != OptixUtil::default_sky_tex) {
            keys.push_back(k);
        }
    }

    for (auto& k : keys) {
        OptixUtil::removeTexture(k);
    }
   
    OptixUtil::sky_tex = OptixUtil::default_sky_tex;

    for (auto const& [key, val] : defaultScene._vdb_grids_cached) {
        cleanupVolume(*val);
    }
    
    defaultScene = {};
    OptixUtil::g_ies.clear();

    cleanupHairs();
    globalShaderBufferGroup.reset();

    using namespace OptixUtil;

    pipelineMark = {};
    // raygen_module            .handle=0;
    // sphere_ism               .handle=0;
    // raygen_prog_group        .handle=0;

    try {
        CUDA_SYNC_CHECK();
    }
    catch(std::exception const& e)
    {
        std::cout << "Exception: " << e.what() << "\n";
    }
}

void optixDestroy() {
    using namespace OptixUtil;
    try {
        CUDA_SYNC_CHECK();
        optixCleanup();

        rtMaterialShaders.clear();
        shaderCoreLUT.clear();
        OptixUtil::resetAll();
    }
    catch (sutil::Exception const& e) {
        std::cout << "OptixCleanupError: " << e.what() << std::endl;
    }

    context                  .handle=0;
    pipeline                 .handle=0;
    raygen_module            .handle=0;
    sphere_ism               .handle=0;
    raygen_prog_group        .handle=0;
    radiance_miss_group      .handle=0;
    occlusion_miss_group     .handle=0;

    output_buffer_o           .reset();
    state = {};
    pipelineMark = {};         
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
