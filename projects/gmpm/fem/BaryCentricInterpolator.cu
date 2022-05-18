#include "../Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
// #include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "../../ZenoFX/LinearBvh.h"

namespace zeno{

using T = float;
using vec3 = zs::vec<T,3>;
using vec4 = zs::vec<T,4>;
using mat3 = zs::vec<T,3,3>;
using mat4 = zs::vec<T,4,4>;

struct ZSComputeBaryCentricWeights : INode {
    T ComputeVolume(
        const vec3& p0,
        const vec3& p1,
        const vec3& p2,
        const vec3& p3
    ) const {
        mat4 m{};
        for(size_t i = 0;i < 3;++i){
            m(i,0) = p0[i];
            m(i,1) = p1[i];
            m(i,2) = p2[i];
            m(i,4) = p3[i];
        }
        m(4,0) = m(4,1) = m(4,2) = m(4,3) = 1;
        return zs::determinant(m);
    }
    vec4 ComputeBaryCentricCoordinate(const vec3& p,
        const vec3& p0,
        const vec3& p1,
        const vec3& p2,
        const vec3& p3
    ) const {
        auto vol = ComputeVolume(p0,p1,p2,p3);
        auto vol0 = ComputeVolume(p,p1,p2,p3);
        auto vol1 = ComputeVolume(p0,p,p2,p3);      
        auto vol2 = ComputeVolume(p0,p1,p,p3);
        auto vol3 = ComputeVolume(p0,p1,p2,p);
        return vec4{vol0/vol,vol1/vol,vol2/vol,vol3/vol};
    }
    void apply() override {
        using namespace zs;
        auto zsvolume = get_input<ZenoParticles>("zsvolume");
        auto zssurf = get_input<ZenoParticles>("zssurf");
        // the bvh of zstets
        auto lbvh = get_input<zeno::LBvh>("lbvh");

        const auto& verts = zsvolume->getParticles();
        const auto& eles = zsvolume->getQuadraturePoints();

        const auto& everts = zssurf->getParticles();

        auto &bcw = (*zsvolume)["bcws"];
        bcw = typename ZenoParticles::particles_t({{"inds",1},{"w",4}},everts.size(),zs::memsrc_e::device);

        auto cudaExec = zs::cuda_exec();
        const auto numFEMVerts = verts.size();
        const auto numFEMEles = eles.size();
        const auto numEmbedVerts = bcw.size();
        
        constexpr auto space = zs::execspace_e::cuda;
        cudaExec(zs::range(numEmbedVerts),
            [verts = proxy<space>({},verts),eles = proxy<space>({},eles),bcw = proxy<space>({},bcw),everts = proxy<space>({},everts)] ZS_LAMBDA (int vi) mutable {
                const auto& p = everts.pack<3>("x",vi);
                // lbvh->iter_neighbors(p,[&](int ei){
                //     const auto& p0 = verts.pack<3>("x",zs::reinterpret_bits<int>(eles("inds",0,ei)));
                //     const auto& p1 = verts.pack<3>("x",zs::reinterpret_bits<int>(eles("inds",1,ei)));
                //     const auto& p2 = verts.pack<3>("x",zs::reinterpret_bits<int>(eles("inds",2,ei)));
                //     const auto& p3 = verts.pack<3>("x",zs::reinterpret_bits<int>(eles("inds",3,ei)));

                //     auto ws = ComputeBaryCentricCoordinate(p,p0,p1,p2,p3);
                //     float epsilon = 1e-6;
                //     if(ws[0] > -epsilon && ws[1] > -epsilon && ws[2] > -epsilon && ws[3] > -epsilon){
                //         bcw("inds",vi) = reinterpret_bits<float>(ei);
                //         bcw.tuple<4>("w",vi) = ws;
                //     }
                // });
            }
        );
        set_output("zsvolume", std::move(zsvolume));
    }
};

ZENDEFNODE(ZSComputeBaryCentricWeights, {{{"interpolator","zsvolume"}, {"embed surf", "zssurf"}},
                            {{"interpolator on gpu", "zsvolume"}},
                            {},
                            {"FEM"}});


struct ZSInterpolateEmbedPrim : zeno::INode {
    void apply() override {
        using namespace zs;
        auto zstets = get_input<ZenoParticles>("zsvolume");
        auto zssurf = get_input<ZenoParticles>("zssurf");

        auto outAttr = get_param<std::string>("outAttr");

        auto cudaExec = zs::cuda_exec();

        auto &everts = zssurf->getParticles();
        everts.append_channels(cudaExec, {{outAttr, 3}});
        
        const auto& verts = zstets->getParticles();
        const auto& eles = zstets->getQuadraturePoints();
        const auto& bcw = (*zstets)["bcws"];

        const auto nmEmbedVerts = bcw.size();

        if(everts.size() != nmEmbedVerts)
            throw std::runtime_error("INPUT SURF SIZE AND BCWS SIZE DOES NOT MATCH");


        constexpr auto space = zs::execspace_e::cuda;

        cudaExec(zs::range(nmEmbedVerts),
            [outAttr = zs::SmallString{outAttr},verts = proxy<space>({},verts),eles = proxy<space>({},eles),bcw = proxy<space>({},bcw),everts = proxy<space>({},everts)] ZS_LAMBDA (int vi) mutable {
                const auto& ei = bcw.pack<1>("inds",vi).reinterpret_bits<int>()[0];
                const auto& w = bcw.pack<4>("w",vi);

                everts.tuple<3>(outAttr,vi) = vec3::zeros();

                for(size_t i = 0;i < 4;++i){
                    // const auto& idx = eles.pack<4>("inds",ei).reinterpret_bits<int>()[i];
                    const auto idx = reinterpret_bits<int>(eles("inds", i, ei));
                    everts.tuple<3>(outAttr,vi) = everts.pack<3>(outAttr,vi) + w[i] * verts.pack<3>("x", idx);
                }
        });

        set_output("zssurf",zssurf);
    }
};

ZENDEFNODE(ZSInterpolateEmbedPrim, {{{"zsvolume"}, {"embed primitive", "zssurf"}},
                            {{"embed primitive", "zssurf"}},
                            {{}},
                            {"FEM"}});


} // namespace zeno