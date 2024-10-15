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

#include "Noise.cuh"

namespace zeno {

/// utilities
constexpr std::size_t count_warps(std::size_t n) noexcept {
    return (n + 31) / 32;
}
constexpr int warp_index(int n) noexcept {
    return n / 32;
}
constexpr auto warp_mask(int i, int n) noexcept {
    int k = n % 32;
    const int tail = n - k;
    if (i < tail)
        return zs::make_tuple(0xFFFFFFFFu, 32);
    return zs::make_tuple(((unsigned)(1ull << k) - 1), k);
}

template <typename T, typename Op>
__forceinline__ __device__ void reduce_to(int i, int n, T val, T &dst, Op op) {
    auto [mask, numValid] = warp_mask(i, n);
    __syncwarp(mask);
    auto locid = threadIdx.x & 31;
    for (int stride = 1; stride < 32; stride <<= 1) {
        auto tmp = __shfl_down_sync(mask, val, stride);
        if (locid + stride < numValid)
            val = op(val, tmp);
    }
    if (locid == 0)
        dst = val;
}

template <typename TransOp, typename ReduceOp>
float prim_reduce(typename ZenoParticles::particles_t &verts, float e, TransOp top, ReduceOp rop,
                  std::string attrToReduce) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using T = typename ZenoParticles::particles_t::value_type;
    auto nchn = verts.getPropertySize(attrToReduce);
    auto offset = verts.getPropertyOffset(attrToReduce);
    const auto nwarps = count_warps(verts.size());

    auto cudaPol = cuda_exec();

    Vector<float> res{verts.get_allocator(), nwarps};
    // cudaPol(res, [e] ZS_LAMBDA(auto &v) { v = e; });
    cudaPol(range(verts.size()), [res = proxy<space>(res), verts = proxy<space>({}, verts), offset, nwarps, nchn, top,
                                  rop] ZS_LAMBDA(int i) mutable {
        auto [mask, numValid] = warp_mask(i, nwarps);
        float v = top(verts(offset, i));
        while (--nchn) {
            v = rop(top(verts(offset++, i)), v);
        }
        reduce_to(i, nwarps, v, res[i / 32], rop);
    });

    Vector<float> ret{res.get_allocator(), 1};
    zs::reduce(cudaPol, std::begin(res), std::end(res), std::begin(ret), e, rop);
    return ret.getVal();
}

struct ZSParticlePerlinNoise : INode {
    virtual void apply() override {
        auto zspars = get_input<ZenoParticles>("zspars");
        auto attrTag = get_input2<std::string>("Attribute");
        auto opType = get_input2<std::string>("OpType");
        auto frequency = get_input2<zeno::vec3f>("Frequency");
        auto offset = get_input2<zeno::vec3f>("Offset");
        auto roughness = get_input2<float>("Roughness");
        auto turbulence = get_input2<int>("Turbulence");
        auto amplitude = get_input2<float>("Amplitude");
        auto attenuation = get_input2<float>("Attenuation");
        auto mean = get_input2<zeno::vec3f>("MeanNoise");

        bool isAccumulate = opType == "accumulate" ? true : false;

        zs::SmallString tag = attrTag;

        auto &tv = zspars->getParticles();

        if (!tv.hasProperty(tag))
            throw std::runtime_error(fmt::format("Attribute [{}] doesn't exist!", tag.asChars()));
        const int nchns = tv.getPropertySize(tag);

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        pol(zs::range(tv.size()),
            [tvv = zs::proxy<space>({}, tv), tag, nchns, isAccumulate,
             frequency = zs::vec<float, 3>::from_array(frequency), offset = zs::vec<float, 3>::from_array(offset),
             roughness, turbulence, amplitude, attenuation,
             mean = zs::vec<float, 3>::from_array(mean)] __device__(int no) mutable {
                auto wcoord = tvv.pack(zs::dim_c<3>, "x", no);
                auto pp = frequency * wcoord - offset;

                float scale = amplitude;

                if (nchns == 3) {
                    // fractal Brownian motion
                    auto fbm = zs::vec<float, 3>::uniform(0);
                    for (int i = 0; i < turbulence; ++i, pp *= 2.f, scale *= roughness) {
                        zs::vec<float, 3> pln{ZSPerlinNoise1::perlin(pp[0], pp[1], pp[2]),
                                              ZSPerlinNoise1::perlin(pp[1], pp[2], pp[0]),
                                              ZSPerlinNoise1::perlin(pp[2], pp[0], pp[1])};
                        fbm += scale * pln;
                    }
                    auto noise = zs::vec<float, 3>{zs::pow(fbm[0], attenuation), zs::pow(fbm[1], attenuation),
                                                   zs::pow(fbm[2], attenuation)} +
                                 mean;

                    if (isAccumulate)
                        tvv.tuple(zs::dim_c<3>, tag, no) =
                            tvv.pack(zs::dim_c<3>, tag, no) + noise;
                    else
                        tvv.tuple(zs::dim_c<3>, tag, no) = noise;

                } else if (nchns == 1) {
                    float fbm = 0;
                    for (int i = 0; i < turbulence; ++i, pp *= 2.f, scale *= roughness) {
                        float pln = ZSPerlinNoise1::perlin(pp[0], pp[1], pp[2]);
                        fbm += scale * pln;
                    }
                    auto noise = zs::pow(fbm, attenuation) + mean[0];

                    if (isAccumulate)
                        tvv(tag, no) += noise;
                    else
                        tvv(tag, no) = noise;
                }
            });

        set_output("zspars", zspars);
    }
};

ZENDEFNODE(ZSParticlePerlinNoise, {/* inputs: */
                               {"zspars",
                                {"string", "Attribute", "v"},
                                {"enum replace accumulate", "OpType", "accumulate"},
                                {"vec3f", "Frequency", "1, 1, 1"},
                                {"vec3f", "Offset", "0, 0, 0"},
                                {"float", "Roughness", "0.5"},
                                {"int", "Turbulence", "4"},
                                {"float", "Amplitude", "1.0"},
                                {"float", "Attenuation", "1.0"},
                                {"vec3f", "MeanNoise", "0, 0, 0"}},
                               /* outputs: */
                               {"zspars"},
                               /* params: */
                               {},
                               /* category: */
                               {"Eulerian"}});

struct ZSPrimitiveReduction : zeno::INode {
    struct pass_on {
        template <typename T>
        constexpr T operator()(T v) const noexcept {
            return v;
        }
    };
    struct getabs {
        template <typename T>
        constexpr T operator()(T v) const noexcept {
            return zs::abs(v);
        }
    };
    virtual void apply() override {
        using namespace zs;
        auto prim = get_input<ZenoParticles>("ZSParticles");
        auto &verts = prim->getParticles();
        auto attrToReduce = get_input2<std::string>("attr");
        if (attrToReduce == "pos")
            attrToReduce = "x";
        if (attrToReduce == "vel")
            attrToReduce = "v";

        if (!verts.hasProperty(attrToReduce))
            throw std::runtime_error(fmt::format("verts do not have property [{}]\n", attrToReduce));

        auto opStr = get_input2<std::string>("op");
        zeno::NumericValue result;
        if (opStr == "avg") {
            result = prim_reduce(verts, 0, pass_on{}, std::plus<float>{}, attrToReduce) / verts.size();
        } else if (opStr == "max") {
            result = prim_reduce(verts, detail::deduce_numeric_lowest<float>(), pass_on{}, getmax<float>{}, attrToReduce);
        } else if (opStr == "min") {
            result = prim_reduce(verts, detail::deduce_numeric_max<float>(), pass_on{}, getmin<float>{}, attrToReduce);
        } else if (opStr == "absmax") {
            result = prim_reduce(verts, 0, getabs{}, getmax<float>{}, attrToReduce);
        }

        auto out = std::make_shared<zeno::NumericObject>();
        out->set(result);
        set_output("result", std::move(out));
    }
};
ZENDEFNODE(ZSPrimitiveReduction, {/* inputs: */ {
                                      "ZSParticles",
                                      {"string", "attr", "pos"},
                                      {"enum avg max min absmax", "op", "avg"},
                                  },
                                  /* outputs: */
                                  {
                                      "result",
                                  },
                                  /* params: */
                                  {},
                                  /* category: */
                                  {
                                      "primitive",
                                  }});

struct ZSGetUserData : zeno::INode {
    virtual void apply() override {
        auto object = get_input<ZenoParticles>("object");
        auto key = get_param<std::string>("key");
        auto hasValue = object->zsUserData().has(key);
        auto data = hasValue ? object->zsUserData().get(key) : std::make_shared<DummyObject>();
        set_output2("hasValue", hasValue);
        set_output("data", std::move(data));
    }
};

ZENDEFNODE(ZSGetUserData, {
                              {"object"},
                              {"data", {"bool", "hasValue"}},
                              {{"string", "key", ""}},
                              {"lifecycle"},
                          });

struct ColoringSelected : INode {
    using tiles_t = typename ZenoParticles::particles_t;
    void markBoundaryVerts(zs::CudaExecutionPolicy &pol, ZenoParticles *prim) {
        using namespace zs;
        auto &vtemp = prim->getParticles();
        vtemp.append_channels(pol, std::vector<zs::PropertyTag>{{"on_boundary", 1}});
        auto markIter = vtemp.begin("on_boundary", dim_c<1>, int_c);
        auto markIterEnd = vtemp.end("on_boundary", dim_c<1>, int_c);
        pol(detail::iter_range(markIter, markIterEnd), [] ZS_LAMBDA(auto &mark) mutable { mark = 0; });

        if (prim->category == ZenoParticles::curve) {
            auto &eles = prim->getQuadraturePoints();
            mark_surface_boundary_verts(pol, eles, wrapv<2>{}, markIter, (size_t)0);
        } else if (prim->category == ZenoParticles::surface) {
            auto &eles = prim->getQuadraturePoints();
            mark_surface_boundary_verts(pol, eles, wrapv<3>{}, markIter, (size_t)0);
        } else if (prim->category == ZenoParticles::tet) {
            auto &surf = (*prim)[ZenoParticles::s_surfTriTag];
            mark_surface_boundary_verts(pol, surf, wrapv<3>{}, markIter, (size_t)0);
        }
    }
    template <typename LsView>
    void markVerts(zs::CudaExecutionPolicy &cudaPol, zs::SmallString tag, ZenoParticles *zsprim, LsView lsv,
                   bool boundaryWise) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto &vtemp = zsprim->getParticles();
        auto numVerts = vtemp.size();
        cudaPol(range(numVerts), [vtemp = proxy<space>({}, vtemp), tag, lsv, boundaryWise] ZS_LAMBDA(int i) mutable {
            if (boundaryWise && vtemp.hasProperty("on_boundary"))
                if (vtemp("on_boundary", i, int_c) == 0) // only operate on verts on boundary
                    return;
            auto x = vtemp.pack(dim_c<3>, "x", i);
            if (lsv.getSignedDistance(x) < 0) {
                vtemp(tag, i) = 1.f;
            }
        });
    }
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto zsprim = get_input<ZenoParticles>("ZSParticles");

        auto cudaPol = zs::cuda_exec().sync(true);

        auto zsls = get_input<ZenoLevelSet>("ZSLevelSet");
        bool boundaryWise = get_input2<bool>("boundary_wise");
        auto &vtemp = zsprim->getParticles();
        if (boundaryWise || vtemp.hasProperty("on_boundary"))
            markBoundaryVerts(cudaPol, zsprim.get());

        auto tag = get_input2<std::string>("markTag");
        vtemp.append_channels(cudaPol, std::vector<zs::PropertyTag>{{tag, 1}});
        cudaPol(range(vtemp, tag), [] ZS_LAMBDA(auto &mark) mutable { mark = 0; });

        match([&](const auto &ls) {
            using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
            using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
            using const_transition_ls_t = typename ZenoLevelSet::const_transition_ls_t;
            if constexpr (is_same_v<RM_CVREF_T(ls), basic_ls_t>) {
                match([&](const auto &lsPtr) {
                    auto lsv = get_level_set_view<execspace_e::cuda>(lsPtr);
                    markVerts(cudaPol, tag, zsprim.get(), lsv, boundaryWise);
                })(ls._ls);
            } else if constexpr (is_same_v<RM_CVREF_T(ls), const_sdf_vel_ls_t>) {
                match([&](auto lsv) { markVerts(cudaPol, tag, zsprim.get(), SdfVelFieldView{lsv}, boundaryWise); })(
                    ls.template getView<execspace_e::cuda>());
            } else if constexpr (is_same_v<RM_CVREF_T(ls), const_transition_ls_t>) {
                match([&](auto fieldPair) {
                    auto &fvSrc = zs::get<0>(fieldPair);
                    auto &fvDst = zs::get<1>(fieldPair);
                    markVerts(
                        cudaPol, tag, zsprim.get(),
                        TransitionLevelSetView{SdfVelFieldView{fvSrc}, SdfVelFieldView{fvDst}, ls._stepDt, ls._alpha},
                        boundaryWise);
                })(ls.template getView<zs::execspace_e::cuda>());
            }
        })(zsls->getLevelSet());

        set_output("ZSParticles", zsprim);
    }
};

ZENDEFNODE(ColoringSelected, {{
                                  "ZSParticles",
                                  "ZSLevelSet",
                                  {"bool", "boundary_wise", "0"},
                                  {"string", "markTag", "selected"},
                              },
                              {"ZSParticles"},
                              {},
                              {"geom"}});

} // namespace zeno