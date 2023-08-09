#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <random>
#include <zeno/types/DummyObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/utils/parallel_reduce.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>

namespace zeno {

static constexpr const char *zs_bvh_tag = "zsbvh";

/// ref: zenovis/xinxinoptix/DisneyBSDF.h
using uint2 = zs::vec<int, 2>;
using float2 = zs::vec<float, 2>;
using v3 = zs::vec<float, 3>;
using i3 = zs::vec<int, 3>;

static __device__ __forceinline__ uint2 Sobol(unsigned int n) {
    uint2 p = uint2{0u, 0u};
    uint2 d = uint2{0x80000000u, 0x80000000u};

    for (; n != 0u; n >>= 1u) {
        if ((n & 1u) != 0u) {
            p[0] ^= d[0];
            p[1] ^= d[1];
        }

        d[0] >>= 1u;        // 1st dimension Sobol matrix, is same as base 2 Van der Corput
        d[1] ^= d[1] >> 1u; // 2nd dimension Sobol matrix
    }
    return p;
}

// adapted from: https://www.shadertoy.com/view/3lcczS
static __device__ __forceinline__ unsigned int ReverseBits(unsigned int x) {
    x = ((x & 0xaaaaaaaau) >> 1) | ((x & 0x55555555u) << 1);
    x = ((x & 0xccccccccu) >> 2) | ((x & 0x33333333u) << 2);
    x = ((x & 0xf0f0f0f0u) >> 4) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x & 0xff00ff00u) >> 8) | ((x & 0x00ff00ffu) << 8);
    return (x >> 16) | (x << 16);
}

// EDIT: updated with a new hash that fixes an issue with the old one.
// details in the post linked at the top.
static __device__ __forceinline__ unsigned int OwenHash(unsigned int x,
                                                        unsigned int seed) { // works best with random seeds
    x ^= x * 0x3d20adeau;
    x += seed;
    x *= (seed >> 16) | 1u;
    x ^= x * 0x05526c56u;
    x ^= x * 0x53a22864u;
    return x;
}
static __device__ __forceinline__ unsigned int OwenScramble(unsigned int p, unsigned int seed) {
    p = ReverseBits(p);
    p = OwenHash(p, seed);
    return ReverseBits(p);
}
static __device__ __forceinline__ float2 sobolRnd(unsigned int &seed) {

    uint2 ip = Sobol(seed);
    ip[0] = OwenScramble(ip[0], 0xe7843fbfu);
    ip[1] = OwenScramble(ip[1], 0x8d8fb1e0u);
    seed++;
    seed = seed & 0xffffffffu;
    return float2{float(ip[0]) / float(0xffffffffu), float(ip[1]) / float(0xffffffffu)};

    //return make_float2(rnd(seed), rnd(seed));
}

static __device__ __forceinline__ float2 sobolRnd2(unsigned int &seed) {

    uint2 ip = Sobol(seed);
    ip[0] = OwenScramble(ip[0], 0xe7843fbfu);
    ip[1] = OwenScramble(ip[1], 0x8d8fb1e0u);
    seed++;
    seed = seed & 0xffffffffu;
    return float2{float(ip[0]) / float(0xffffffffu), float(ip[1]) / float(0xffffffffu)};

    //return make_float2(rnd(seed), rnd(seed));
}

static __forceinline__ __device__ v3 UniformSampleHemisphere(float r1, float r2) {
    float r = zs::sqrt(zs::max(0.0, 1.0 - r1 * r1));
    float phi = 2.0f * 3.1415926f * r2;
    return v3(r * zs::cos(phi), r * zs::sin(phi), r1);
}

static __forceinline__ __device__ void CoordinateSystem(const v3 &a, v3 &b, v3 &c) {
    //    if (abs(a.x) > abs(a.y))
    //        b = float3{-a.z, 0, a.x} /
    //              sqrt(max(_FLT_EPL_, a.x * a.x + a.z * a.z));
    //    else
    //        b = float3{0, a.z, -a.y} /
    //              sqrt(max(_FLT_EPL_, a.y * a.y + a.z * a.z));

    if (zs::abs(a[0]) > zs::abs(a[1]))
        b = v3{-a[2], 0, a[0]};
    else
        b = v3{0, a[2], -a[1]};

    b = b.normalized();
    c = a.cross(b);
}

struct Onb {
    __forceinline__ __device__ Onb(const v3 &normal) {
        m_normal = normal;

        if (zs::abs(m_normal[0]) > zs::abs(m_normal[2])) {
            m_binormal[0] = -m_normal[1];
            m_binormal[1] = m_normal[0];
            m_binormal[2] = 0;
        } else {
            m_binormal[0] = 0;
            m_binormal[1] = -m_normal[2];
            m_binormal[2] = m_normal[1];
        }

        m_binormal = m_binormal.normalized();
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(v3 &p) const {
        p = p[0] * m_tangent + p[1] * m_binormal + p[2] * m_normal;
    }

    v3 m_tangent;
    v3 m_binormal;
    v3 m_normal;
};
static __forceinline__ __device__ void world2local(v3 &v, const v3 &T, const v3 &B, const v3 &N) {
    v = v3{T.dot(v), B.dot(v), N.dot(v)}.normalized();
}

namespace rtgems {

constexpr float origin() {
    return 1.0f / 32.0f;
}
constexpr float int_scale() {
    return 256.0f;
}
constexpr float float_scale() {
    return 1.0f / 65536.0f;
}

// Normal points outward for rays exiting the surface, else is flipped.
static __inline__ __device__ v3 offset_ray(const v3 &p, const v3 &n) {
    i3 of_i{(int)(int_scale() * n[0]), (int)(int_scale() * n[1]), (int)(int_scale() * n[2])};

    v3 p_i{__int_as_float(__float_as_int(p[0]) + ((p[0] < 0) ? -of_i[0] : of_i[0])),
           __int_as_float(__float_as_int(p[1]) + ((p[1] < 0) ? -of_i[1] : of_i[1])),
           __int_as_float(__float_as_int(p[2]) + ((p[2] < 0) ? -of_i[2] : of_i[2]))};

    return v3{fabsf(p[0]) < origin() ? p[0] + float_scale() * n[0] : p_i[0],
              fabsf(p[1]) < origin() ? p[1] + float_scale() * n[1] : p_i[1],
              fabsf(p[2]) < origin() ? p[2] + float_scale() * n[2] : p_i[2]};
}
} // namespace rtgems

template <typename VPosRange, typename VIndexRange, typename Bv, int codim = 3>
void retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, VPosRange &&posR, VIndexRange &&idxR,
                               zs::Vector<Bv> &ret) {
    using namespace zs;
    using bv_t = Bv;
    constexpr auto space = execspace_e::cuda;
    pol(range(range_size(idxR)),
        [tris = idxR.begin(), pos = posR.begin(), bvs = view<space>(ret)] ZS_LAMBDA(int ei) mutable {
            auto inds = tris[ei];
            auto x0 = pos[inds[0]];
            bv_t bv{x0, x0};
            for (int d = 1; d != 3; ++d)
                merge(bv, pos[inds[d]]);
            bvs[ei] = bv;
        });
}

struct ComputeVertexAO : INode {
    void apply() override {
        auto prim = get_input2<PrimitiveObject>("prim");
        auto scene = get_input2<PrimitiveObject>("scene");
        auto nrmTag = get_input2<std::string>("nrm_tag");
        auto niters = get_input2<int>("sample_iters");
        auto distCap = get_input2<float>("dist_cap");
        if (distCap < std::numeric_limits<float>::epsilon())
            distCap = std::numeric_limits<float>::max();

        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto pol = cuda_exec();
        using bvh_t = ZenoLinearBvh::lbvh_t;
        using bv_t = typename bvh_t::Box;
        // zs::AABBBox<3, float>

        auto allocator = get_temporary_memory_source(pol);
        const auto &sceneTris = scene->tris.values;
        const auto &scenePos = scene->verts.values;
        zs::Vector<v3> pos{allocator, scenePos.size()};
        zs::Vector<i3> indices{allocator, sceneTris.size()};
        zs::copy(mem_device, (void *)pos.data(), (void *)scenePos.data(), sizeof(v3) * pos.size());
        zs::copy(mem_device, (void *)indices.data(), (void *)sceneTris.data(), sizeof(i3) * indices.size());

        auto &sceneData = scene->userData();
        if (!sceneData.has<ZenoLinearBvh>(zs_bvh_tag)) {
            auto zsbvh = std::make_shared<ZenoLinearBvh>();
            zsbvh->thickness = 0;
            zsbvh->et = ZenoLinearBvh::surface;

            auto &bvh = zsbvh->bvh;
            zs::Vector<bv_t> bvs{indices.size(), memsrc_e::device, 0};
            retrieve_bounding_volumes(pol, range(pos), range(indices), bvs);
            bvh.build(pol, bvs);
            sceneData.set(zs_bvh_tag, zsbvh);
        } else {
            auto zsbvh = sceneData.get<ZenoLinearBvh>(zs_bvh_tag); // std::shared_ptr<>
            auto &bvh = zsbvh->bvh;
            zs::Vector<bv_t> bvs{allocator, indices.size()};
            retrieve_bounding_volumes(pol, range(pos), range(indices), bvs);
            bvh.refit(pol, bvs);
        }

        auto zsbvh = sceneData.get<ZenoLinearBvh>(zs_bvh_tag); // std::shared_ptr<>
        auto &bvh = zsbvh->bvh;

        const auto &hpos = prim->verts.values;
        const auto &hnrms = prim->attr<vec3f>(nrmTag);
        zs::Vector<v3> xs{allocator, hpos.size()}, nrms{allocator, hpos.size()};
        zs::copy(mem_device, (void *)xs.data(), (void *)hpos.data(), sizeof(v3) * xs.size());
        zs::copy(mem_device, (void *)nrms.data(), (void *)hnrms.data(), sizeof(v3) * nrms.size());
        zs::Vector<float> aos{allocator, pos.size()};
        pol(enumerate(xs, nrms, aos),
            [scenePos = view<space>(pos), sceneTris = view<space>(indices), bvh = view<space>(bvh), niters,
             distCap] ZS_LAMBDA(unsigned int i, const v3 &x, v3 nrm, float &ao) {
                unsigned int seed = i;
                // u64 sd = 1442695040888963407ull;
                u64 sd = i;
                ao = 0.f;
                nrm = nrm.normalized();
                Onb tbn = Onb(nrm);
#if 1
                v3 n0;
                v3 n1;
                CoordinateSystem(nrm, n0, n1);
#else
                auto n0 = nrm.orthogonal().normalized();
                auto n1 = nrm.cross(n0);
#endif
                tbn.m_tangent = n0;
                tbn.m_binormal = n1;
                int accum = 0;
                for (int k = 0; k < niters; ++k) {
#if 1
#if 0
                    auto r = sobolRnd(seed);
                    auto w = UniformSampleHemisphere(r[0], r[1]);
#else
                    auto r0 = 1.f * zs::PCG::pcg32_random_r(sd, 1442695040888963407ull) / std::numeric_limits<u32>::max();
                    auto r1 = 1.f * zs::PCG::pcg32_random_r(sd, 1442695040888963407ull) / std::numeric_limits<u32>::max();
                    auto w = UniformSampleHemisphere(r0, r1);
#endif
                    tbn.inverse_transform(w);
                    w = w.normalized();
                    if (w.dot(nrm) < 0)
                        w = -w;
#else
                    v3 w;
                    do {
                        auto r = sobolRnd(seed);
                        w = UniformSampleHemisphere(r[0], r[1]);
                        tbn.inverse_transform(w);
                        w = w.normalized();
                    } while (w.dot(nrm) < std::numeric_limits<float>::epsilon());
#endif
                    bool hit = false;
                    auto xx = rtgems::offset_ray(x, nrm);
                    // auto &xx = x;
                    bvh.ray_intersect(xx, w, [&x = xx, &w, &scenePos, &sceneTris, &hit, distCap](int triNo) {
                        if (hit)
                            return;
                        auto tri = sceneTris[triNo];
                        auto t0 = scenePos[tri[0]];
                        auto t1 = scenePos[tri[1]];
                        auto t2 = scenePos[tri[2]];
                        if (auto d = ray_tri_intersect(x, w, t0, t1, t2);
                            d < distCap && d > std::numeric_limits<float>::epsilon() * 10) {
                            hit = true;
#if 0
                            if (i < 2 && k < 3) {
                                printf("[%u] iter %d dir [%f] (%f, %f, %f), dist (%f)\n", i, (int)k, w.norm(), w[0],
                                       w[1], w[2], d);
                            }
#endif
                            return;
                        }
                    });
                    if (hit)
                        accum++;
#if 0
                    if (i < 2 && k < 3) {
                        printf("[%u] iter %d dir [%f] (%f, %f, %f), r (%f, %f)\n", i, (int)k, w.norm(), w[0], w[1],
                               w[2], r[0], r[1]);
                    }
#endif
                }
                ao = 1.f * accum / niters;
            });

        const auto &haos = prim->add_attr<float>(get_input2<std::string>("ao_tag"));
        zs::copy(mem_device, (void *)haos.data(), (void *)aos.data(), sizeof(float) * haos.size());

        set_output("prim", prim);
    }
};

ZENDEFNODE(ComputeVertexAO, {
                                {
                                    {"PrimitiveObject", "prim", ""},
                                    {"PrimitiveObject", "scene", ""},
                                    {"string", "nrm_tag", "nrm"},
                                    {"string", "ao_tag", "ao"},
                                    {"float", "dist_cap", "0"},
                                    {"int", "sample_iters", "512"},
                                },
                                {"prim"},
                                {},
                                {"tracing"},
                            });

} // namespace zeno