#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include "Light.h"
#include "volume.h"

#include "TraceStuff.h"
#include "zxxglslvec.h"
#include "math_constants.h"

#include <Octree.h>
#include <cuda_fp16.h>
// #include "nvfunctional"

using DataTypeNVDB0 = nanovdb::Float;
using GridTypeNVDB0 = nanovdb::NanoGrid<DataTypeNVDB0>;

#define HF0 __ushort_as_half((unsigned short)0x0000U)
#define HF1 __ushort_as_half((unsigned short)0x3C00U)

__inline__ __device__ half clamp( const half f, const half a=HF0, const half b=HF1 )
{
    return __hmax( a, __hmin( f, b ) );
}

__inline__ __device__ bool valid(const float3& vvv) {
    return vvv.x>0 || vvv.y>0 || vvv.z>0; // && (vvv.x>0 || vvv.y>0 || vvv.z>0);
}

__inline__ __device__ bool rayHit(const float3& ray_ori, const float3& ray_dir, const nanovdb::BBox<nanovdb::Vec3f>& box, 
                                uint8_t bounds, float& t0, float& t1) {

    auto iray = nanovdb::Ray<float>( reinterpret_cast<const nanovdb::Vec3f&>( ray_ori ),
                                     reinterpret_cast<const nanovdb::Vec3f&>( ray_dir ), t0, t1 );
    if (bounds == 0)
        return iray.intersects( box, t0, t1 );

    auto maxi = box.max();
    auto mini = box.min();
    auto diff = maxi - mini;
    auto center = mini + diff/2;
    
    if (bounds == 1)
    {
        iray.setEye(iray.eye() - center);
        iray.setEye(iray.eye() / diff );
        iray.setDir(iray.dir() / diff );
        return iray.intersects({0,0,0}, 0.5f, t0, t1);
    }
    if (bounds == 2) {

        iray.intersects( box, t0, t1 );

        auto b_t0 = t0;
        auto b_t1 = t1;

        nanovdb::Vec3f scale(2,1,2);
        iray.setEye( iray.eye() - center);
        iray.setEye( iray.eye() * scale / diff );
        iray.setDir( iray.dir() * scale / diff );
        bool hit = iray.intersects({0,-0.5f,0}, 1, t0, t1);

        t0 = max(b_t0, t0);
        t1 = min(b_t1, t1);

        return hit;
    }
}

struct AABB {
    float3 mini;
    float3 maxi;

    AABB() = default;
    __device__ AABB(float3 mi, float3 ma) {
        mini = mi, maxi = ma;
    }
    __device__ float3 ext() const {
        return maxi - mini;
    }
    __device__ float3 mid() const {
        return mini + ext()/2.0f;
    }
    __device__ bool valid() const {
        return maxi.x > mini.x && maxi.y > mini.y && maxi.z > mini.z;
    }
};

struct OcTrace {
    float t0 = -1;
    half min_d, max_d;
};

#define MAX_STACK_DEPTH 23
#define USE_STACK_DEPTH 8
struct OcFrame { 
    uint32_t node;
    float t_max;
};
struct OcStack {
    OcFrame storage[USE_STACK_DEPTH];

    __device__ OcFrame& read(uint8_t idx) {
        return storage[idx];
    }
    __device__ const OcFrame& read(uint8_t idx) const {
        return storage[idx];
    }
    __device__ void write(uint8_t idx, uint32_t node, float tmax)
    {
        storage[idx] = {node, tmax}; 
    }
};

__device__ OcTrace traverseSVO(const OcNode* __restrict__ buffer, uint32_t root_idx, const AABB& root_box, OcStack& stack, const float3& ray_o, const float3& ray_d, float thickness, float t_skip, uint8_t max_depth=8) {

    // OcStack stack {};
    OcTrace trace {};

    auto _root_scale = root_box.ext();
    auto _root_offset = root_box.mini;
    
    // scale root box to [0.f, 1.f]
    auto _inver_scale = 1.0f / _root_scale;
    auto _ray_o = (ray_o - _root_offset) * _inver_scale;
    auto _ray_d = ray_d * _inver_scale;

        auto rayScale = length( _ray_d );
        thickness *= rayScale;

    _ray_d /= rayScale;
    // move root box to [1.f, 2.f]
    _ray_o += float3 {1, 1, 1};

    const float epsilon = exp2f(-MAX_STACK_DEPTH);

    // Get rid of small ray direction components to avoid division by zero.
    if (fabsf(_ray_d.x) < epsilon) _ray_d.x = copysignf(epsilon, _ray_d.x);
    if (fabsf(_ray_d.y) < epsilon) _ray_d.y = copysignf(epsilon, _ray_d.y);
    if (fabsf(_ray_d.z) < epsilon) _ray_d.z = copysignf(epsilon, _ray_d.z);
    // Precompute the coefficients of tx(x), ty(y), and tz(z).
    // The octree is assumed to reside at coordinates [1, 2].
    float tx_coef = 1.0f / -fabs(_ray_d.x);
    float ty_coef = 1.0f / -fabs(_ray_d.y);
    float tz_coef = 1.0f / -fabs(_ray_d.z);
    float tx_bias = tx_coef * _ray_o.x;
    float ty_bias = ty_coef * _ray_o.y;
    float tz_bias = tz_coef * _ray_o.z;
    // Select octant mask to mirror the coordinate system so
    // that ray direction is negative along each axis.
    uint8_t octant_mask = 7;
    if (_ray_d.x > 0.0f) octant_mask ^= 1, tx_bias = 3.0f * tx_coef - tx_bias;
    if (_ray_d.y > 0.0f) octant_mask ^= 2, ty_bias = 3.0f * ty_coef - ty_bias;
    if (_ray_d.z > 0.0f) octant_mask ^= 4, tz_bias = 3.0f * tz_coef - tz_bias;

    // Initialize the active span of t-values.
    float plane[2] {2.0f, 1.0f}; // range [2.f, 1.f]
    float t_min = fmaxf(fmaxf(plane[0] * tx_coef - tx_bias, plane[0] * ty_coef - ty_bias), plane[0] * tz_coef - tz_bias);
    float t_max = fminf(fminf(plane[1] * tx_coef - tx_bias, plane[1] * ty_coef - ty_bias), plane[1] * tz_coef - tz_bias);
    float h = t_max;
    t_min = fmaxf(t_min, 0.0f);
    t_min = fmaxf(t_min, t_skip * rayScale);
    //t_max = fminf(t_max, 1.0f);

    uint32_t nodeIdx = 0;
    uint8_t idx = 0;
    float3 pos = {1.f, 1.f, 1.f}; // far plane, first level
    if (1.5f * tx_coef - tx_bias > t_min) idx ^= 1, pos.x = 1.5f;
    if (1.5f * ty_coef - ty_bias > t_min) idx ^= 2, pos.y = 1.5f;
    if (1.5f * tz_coef - tz_bias > t_min) idx ^= 4, pos.z = 1.5f;
    // test with center plane 1.5f

    uint8_t depth = 0u; //USE_STACK_DEPTH-1; 
    float voxel_len = 0.5f;

    while (true) {
        // pos is far plane for child node
        const float tx_corner = pos.x * tx_coef - tx_bias;
        const float ty_corner = pos.y * ty_coef - ty_bias;
        const float tz_corner = pos.z * tz_coef - tz_bias;
        const float tc_max = fminf(fminf(tx_corner, ty_corner), tz_corner); 

        const auto& node = buffer[nodeIdx];
        const uint8_t child_mask = node.childMask();

        const uint8_t child_shift = idx ^ octant_mask ^ 7; // mirroring
        const uint8_t test_mask = 1u << child_shift;

        if ( (child_mask & test_mask) && t_min <= t_max ) { // valid child
            uint8_t mask_lower = test_mask - 1u;
            mask_lower = child_mask & mask_lower;
            uint8_t child_offset = __popc(mask_lower);

            uint32_t childIdx = node.childOffset() + child_offset;

            float tv_max = fminf(t_max, tc_max);
            float half = voxel_len * 0.5f;
            float tx_center = half * tx_coef + tx_corner;
            float ty_center = half * ty_coef + ty_corner;
            float tz_center = half * tz_coef + tz_corner;

            if (t_min <= tv_max) {

                const auto& cnode = buffer[childIdx];                    
                if (max_depth<=(depth+1) || cnode.childMask()==0) { // leaf
                    const float node_tmax = tv_max;
                    const float len = node_tmax - t_min; 
                    const float max_d = __half2float(cnode.max_d);

                    auto thick = max_d * len;
                    if (thickness > thick) {
                        thickness -= thick;
                    } else {
                        // const float min_d = __half2float(cnode.min_d);
                        thick = thick - thickness;
                        float dt = thick / max_d;
                        // (tc_max - dt) is the sample distance
                        trace = {node_tmax-dt, cnode.min_d, cnode.max_d};
                        break;
                    }
                    //march to next child along ray, or pop stack
                } else {
                    if (tc_max < h) {
                        stack.write(depth, nodeIdx, t_max);
                    }
                    h = tc_max;
                    nodeIdx = childIdx;

                    idx = 0;
                    depth++;
                    voxel_len = half;

                    if (tx_center > t_min) idx ^= 1, pos.x += voxel_len;
                    if (ty_center > t_min) idx ^= 2, pos.y += voxel_len;
                    if (tz_center > t_min) idx ^= 4, pos.z += voxel_len;

                    t_max = tv_max;
                    continue;
                } // not leaf
                
            }
        }

        // Advance, forward march
        uint8_t step_mask = 0;
        if (tx_corner <= tc_max) step_mask ^= 1, pos.x -= voxel_len;
        if (ty_corner <= tc_max) step_mask ^= 2, pos.y -= voxel_len;
        if (tz_corner <= tc_max) step_mask ^= 4, pos.z -= voxel_len;

        t_min = tc_max;
        idx ^= step_mask;

        if ( (idx & step_mask) != 0 ) { // Pop
            uint32_t differing_bits = 0;
            if (step_mask & 1) differing_bits |= __float_as_int(pos.x) ^ __float_as_int(pos.x+voxel_len);
            if (step_mask & 2) differing_bits |= __float_as_int(pos.y) ^ __float_as_int(pos.y+voxel_len);
            if (step_mask & 4) differing_bits |= __float_as_int(pos.z) ^ __float_as_int(pos.z+voxel_len);

            auto hbit = 31 - __clz(differing_bits); // highest bit position
            depth = MAX_STACK_DEPTH - 1 - hbit;

            if (depth >= USE_STACK_DEPTH) { break; }

            // voxel_len = exp2f(hbit - MAX_STACK_DEPTH);
            voxel_len = ldexpf(1.0f, hbit - MAX_STACK_DEPTH); 
            {
                const auto& tmp= stack.read(depth);
                nodeIdx = tmp.node;
                t_max = tmp.t_max;
            }
            int shx = __float_as_int(pos.x) >> hbit;
            int shy = __float_as_int(pos.y) >> hbit;
            int shz = __float_as_int(pos.z) >> hbit; 
            pos.x = __int_as_float(shx << hbit);
            pos.y = __int_as_float(shy << hbit);
            pos.z = __int_as_float(shz << hbit);

            idx = (shx & 1) | ((shy & 1) << 1) | ((shz & 1) << 2);
            h = 0.0f;
        }
    }

    trace.t0 /= rayScale;
    return trace;
}

__device__ __forceinline__ auto EvalVolume(VolumeIn& vin, uint16_t dc_index, float3& pos, VolumeOut& out, bool shadowRay=false) {

    vin.pos_view = pos;
    optixDirectCall<void, void*, bool, VolumeOut&>( dc_index, (void*)&vin, shadowRay, out);
}

__device__ __forceinline__ int roundUpToMultiple(int v, int m) {
    return ((v + m - 1) / m) * m;
}

extern "C" __global__ void __intersection__volume()
{
    auto gas = optixGetGASTraversableHandle();
    auto gas_ptr = (void**)optixGetGASPointerFromHandle(gas);

    const auto& aabb = reinterpret_cast<const AABB&>(*(gas_ptr-4));
    const auto& box = reinterpret_cast<const nanovdb::BBox<nanovdb::Vec3f>&>(aabb);

    const auto* sbt_data = (HitGroupData*)( optixGetSbtDataPointer() );
    
    const float3 ray_ori = optixGetObjectRayOrigin();
          float3 ray_dir = optixGetObjectRayDirection(); // not normalized
    const auto dirlen = length(ray_dir);
    ray_dir = normalize(ray_dir);

    float t0 = optixGetRayTmin();
    float t1 = optixGetRayTmax();

    if (sbt_data->vol_depth==0)
        t0 = 0;

    { // world distance to object distance 
        t0 = t0 * dirlen; 
        t1 = t1 * dirlen;
    }

    uint8_t bounds = *((char*)gas_ptr-1);
    auto hitted = rayHit( ray_ori, ray_dir, box, bounds, t0, t1 );
    if (!hitted) { return; }

    const auto obj_t0 = t0;
    const auto obj_t1 = t1;

    const float len = 1.0f / dirlen;
    // object distance to world distance 
    t0 = t0 * len;
    t1 = t1 * len;

    t0 = fmaxf(t0, optixGetRayTmin());
    t1 = fmaxf(t1, t0);

    if (t1 < t0) {
        auto tmp = t1;
        t1 = t0; t0 = tmp;
    }
    if (optixGetRayTmin() >= t1) { 
        return;
    }

    auto flags = optixGetRayFlags();
    const bool anyhit = flags & OPTIX_RAY_FLAG_ENFORCE_ANYHIT;
    const bool hetero = 0 < sbt_data->vol_depth;

    auto prd = getPRD<CommonPRD>();
    t1 = fminf(prd->maxDistance, t1);

    if ( !hetero ) {
        auto& vol = prd->vol;
        if (anyhit) {
            vol.t0 = t0;
            vol.t1 = t1;
            optixReportIntersection(t0, 1);
            return;
        }

        bool cached = (vol.t1 > vol.t0);
        if (cached) {
            bool replace = 0==t0 && vol.t1>t1;
            if (!replace) return;
        }
        vol.t0 = t0;
        vol.t1 = t1;
        vol.homo_matid = sbt_data->dc_index;
        return;
    }

    const auto octree_ptr = reinterpret_cast<OcNode*>(*(gas_ptr-5));
    assert(octree_ptr != nullptr);

    const auto bbox = aabb;
    const auto dim = bbox.ext();

    const int OCTREE_DEPTH = 6;
    const int leafRes = 1 << OCTREE_DEPTH;
    int3 paddedDim {
            roundUpToMultiple(int(dim.x), leafRes),
            roundUpToMultiple(int(dim.y), leafRes),
            roundUpToMultiple(int(dim.z), leafRes) };
    const float3 minCoord = bbox.mini;
    const float3 maxCoord {
            minCoord.x + paddedDim.x,
            minCoord.y + paddedDim.y,
            minCoord.z + paddedDim.z };

    auto octbox = AABB(minCoord, maxCoord);

    if ( !anyhit ) {

        OcStack stack {};
        auto thickness = -logf(1.0f-prd->rndf())  / (len * sbt_data->vol_extinction);
        auto sek = traverseSVO(octree_ptr, 0, octbox, stack, ray_ori, ray_dir, thickness, obj_t0);

        if (sek.t0<0) {
            return; // empty
        }
        t0 = fmaxf(sek.t0 * len, t0);
        // t1 = fminf(sek.t1 * len, t1);
        if ( t0 < optixGetRayTmax() ) {

            auto& vol = prd->vol;
            vol.density_min = sek.min_d;
            vol.density_max = sek.max_d;

            vol.t0 = t0;
            vol.t1 = t1;
            optixReportIntersection(t0, 0);
        }
        return;
    }

    auto sprd = reinterpret_cast<ShadowPRD*>(prd);
    auto transmittance = sprd->attanuation;
    
    using GridTypeNVDB = nanovdb::NanoGrid<nanovdb::Fp16>;
    const auto* _grid = reinterpret_cast<const GridTypeNVDB*>(sbt_data->vdb_grids[0]);
    const auto& _acc = _grid->tree().getAccessor();
    
    using Sampler = nanovdb::SampleFromVoxels<GridTypeNVDB::AccessorType, 0, true>;
    auto sampler = Sampler(_acc);

    auto seed = prd->seed;
    auto scale = 1.0f / (len * sbt_data->vol_extinction);

    float t_progress = obj_t0;
    OcStack stack {};

    do {
        auto thickness = -logf(1.0f-rnd(seed)) * scale;
        auto sek = traverseSVO(octree_ptr, 0, octbox, stack, ray_ori, ray_dir, thickness, t_progress);

        if (sek.t0 >= obj_t1 || sek.t0 < 0) // empty zone
            break;

        t_progress = sek.t0;

        __half homo_prob = sek.min_d / sek.max_d;
        __half prob = __half( rnd(seed) );

        if (prob <= homo_prob) {
            transmittance = {};
            break;
        }
        prob = prob - homo_prob;
        
        auto test_point = ray_ori + t_progress * ray_dir;
        auto& pidx = reinterpret_cast<nanovdb::Vec3f&>(test_point); 
        auto dens = __half(sampler(pidx));
        
        __half density = clamp(dens, HF0, sek.max_d);
        __half ratio = (density-sek.min_d) / (sek.max_d-sek.min_d);
        ratio = clamp(ratio, HF0, HF1);

        if ( prob < ratio ) {
            transmittance = {};
            break;
        }
        // transmittance = transmittance * vec3(ratio);
    } while(true);

    sprd->seed = seed;
    sprd->attanuation = transmittance; 
    if ( valid(transmittance) ) return;

    // optixTerminateRay();
    auto& vol = prd->vol;
    vol.t0 = t0;
    vol.t1 = t1;
    optixReportIntersection(t0, 0);
}

extern "C" __global__ void __closesthit__radiance_volume()
{
    RadiancePRD* prd = getPRD();

    const HitGroupData* sbt_data = (HitGroupData*)( optixGetSbtDataPointer() );
    const auto dc_index = sbt_data->dc_index;

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

    float t0 = prd->vol.t0; // world space
    // float t1 = prd->vol.t1; // world space
    float3 new_orig = ray_orig + t0 * ray_dir;
    
    VolumeOut vout {};
    bool scatter = false;
    {
        VolumeIn vin {};
        vin.seed = &prd->seed;
        vin.sbt_ptr = (void*)sbt_data;
        // optixGetObjectToWorldTransformMatrix((float*)vin.objectToWorld);
        optixGetWorldToObjectTransformMatrix((float*)vin.worldToObject);

        const half min_d = prd->vol.density_min;
        const half max_d = prd->vol.density_max;

        __half homo_ratio = min_d / max_d;
        __half prob = __half( prd->rndf() );

        if (prob <= homo_ratio) {
            vout.density = -HF1; // ignore
            vout.albedo = {1,1,1};
            EvalVolume(vin, dc_index, new_orig, vout);
            vout.density = +HF1;
            scatter = true;

        } else {

            EvalVolume(vin, dc_index, new_orig, vout);

            half density = (vout.density - min_d) / (max_d - min_d);
                 density = clamp(density, HF0, HF1);
            
            prob = __hmax(prob-homo_ratio, HF0) / (HF1 - homo_ratio);
            scatter = prob <= density;
        }
    }

    
    prd->radiance = vout.emission;

    if (!scatter) {
        prd->_tmin_ = t0;
        prd->alphaHit = true;
        return;
    } 
    // else {
        prd->origin = new_orig;
        prd->depth += 1;
        prd->_tmax_ = t0;
        prd->updateAttenuation(vout.albedo);
        prd->geometryNormal = {};

        float3 new_dir  = ray_dir;

        pbrt::HenyeyGreenstein hg (vout.anisotropy);
        float2 uu = { prd->rndf(), prd->rndf() };
        auto pdf = hg.sample(-ray_dir, new_dir, uu);              
        //auto relative_prob = prob * (CUDART_PI_F * 4);
        prd->samplePdf = pdf;
        prd->direction = normalize(new_dir);
        if (prd->denoise) {
            prd->tmp_normal = normalize(-ray_dir + new_dir);
            prd->tmp_albedo = vout.albedo;
        }
    // }
    
    auto evalBxDF = [hg=hg, albedo=vout.albedo](const float3& _wi_, const float3& _wo_, float& thisPDF) -> float3 {
        // pbrt::HenyeyGreenstein hg(aniso);
        thisPDF = hg.p(_wo_, _wi_);
        return albedo * thisPDF;
    };

    ShadowPRD shadowPRD {};
    shadowPRD.seed = prd->seed ^ 0x9e3779b9u;
    shadowPRD.depth = prd->depth;
    shadowPRD.origin = new_orig; //camera sapce
    shadowPRD.attanuation = vec3(1.0f);

    DirectLighting<true, true>(shadowPRD, new_orig+params.cam.eye, ray_dir, evalBxDF);
    prd->radiance += shadowPRD.radiance;
}

extern "C" __global__ void __anyhit__occlusion_volume()
{
    const auto hk = optixGetHitKind();
    if (hk == 0) { // hetero
        optixTerminateRay();
        return;
    }

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

    const HitGroupData* sbt_data = (HitGroupData*)( optixGetSbtDataPointer() );
    const auto dc_index = sbt_data->dc_index;

    ShadowPRD* prd = getPRD<ShadowPRD>();
    const float t0 = prd->vol.t0;
    const float t1 = prd->vol.t1;
    //t1 = prd->maxDistance;

    const float t_max = t1 - t0; // world space
          float t_ele = 0;

    float3 test_point = ray_orig; 
    float3 transmittance = make_float3(1.0f);
    float hgp = 1.0f;
    pbrt::HenyeyGreenstein hg(9.0f);

    const float sigma_t = sbt_data->vol_extinction;

    VolumeIn vin {};
    vin.seed = &prd->seed;
    vin.sbt_ptr = (void*)sbt_data;
    optixGetObjectToWorldTransformMatrix((float*)vin.objectToWorld);
    optixGetWorldToObjectTransformMatrix((float*)vin.worldToObject);

    if (0 == sbt_data->vol_depth) { // Homogeneous

        test_point += ray_dir * 0.5f * (t0 + t1);

        VolumeOut homo_out;
        EvalVolume(vin, dc_index, test_point, homo_out, true);
        hg = pbrt::HenyeyGreenstein(homo_out.anisotropy);

        vec3& trans = *(vec3*)&transmittance;
        vec3 sa = homo_out.extinction * t_max;
        #pragma unroll
        for (char i=0; i<3; ++i) {
            auto& s = sa[i];
            if (s < 1e-4f)
                trans[i] = (1.0f - s + 0.5f * s * s);
            else
                trans[i] = expf(-s);
        }
        prd->attanuation *= transmittance;

        //transmittance *= vol_out.albedo * hg.p(-ray_dir, ray_dir);
        optixIgnoreIntersection();
        return;
    }

    auto level = sbt_data->vol_depth;
    while(--level > 0) {

        auto prob = prd->rndf();
        t_ele -= log(1.0f-prob) / (sigma_t);

        test_point = ray_orig + (t0+t_ele) * ray_dir;

        if (t_ele >= t_max) {
            break;
        } // over shoot, outside of volume

        VolumeOut vol_out;
        EvalVolume(vin, dc_index, test_point, vol_out, true);

        const auto v_density = vol_out.density / half(sigma_t);

        auto prob_scatter = clamp(v_density, HF0, HF1);
        auto prob_nulling = HF1 - prob_scatter;

        if (vol_out.anisotropy != half(hg.g) ) {
            hg = pbrt::HenyeyGreenstein(vol_out.anisotropy);
            hgp = hg.p(-ray_dir, ray_dir);
        }

        half prob_continue = half(hgp) * prob_scatter;
        prob_continue = clamp(prob_continue, HF0, prob_scatter);

        auto tr = transmittance * prob_nulling;
        tr += transmittance * prob_continue * vol_out.albedo;
        
        transmittance = clamp(tr, 0.0, 1.0f);

        auto avg = dot(transmittance, make_float3(1.0f/3.0f));
        if (avg < 0.1f) {
            float q = fmax(0.05f, 1 - avg);
            if (prd->rndf() < q) { 
                transmittance = vec3(0);
                break; 
            } else {
                transmittance /= 1-q;
            }
        }
    }

    if (0 == level) { transmittance = {}; }
    prd->attanuation *= transmittance;

    if ( valid(prd->attanuation) )
        optixIgnoreIntersection();
    else
        optixTerminateRay();
}
