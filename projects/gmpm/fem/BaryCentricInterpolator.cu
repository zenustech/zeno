#include "../Structures.hpp"
#include "../Utils.hpp"
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
    static constexpr T ComputeVolume(
        const vec3& p0,
        const vec3& p1,
        const vec3& p2,
        const vec3& p3
    ) {
        mat4 m{};
        for(size_t i = 0;i < 3;++i){
            m(i,0) = p0[i];
            m(i,1) = p1[i];
            m(i,2) = p2[i];
            m(i,3) = p3[i];
        }
        m(3,0) = m(3,1) = m(3,2) = m(3,3) = 1;
        return zs::determinant(m);
    }

    static constexpr T ComputeArea(
        const vec3& p0,
        const vec3& p1,
        const vec3& p2
    ) {
        auto p01 = p0 - p1;
        auto p02 = p0 - p2;
        auto p12 = p1 - p2;
        T a = p01.length();
        T b = p02.length();
        T c = p12.length();
        T s = (a + b + c)/3;
        return zs::sqrt(s*(s-a)*(s-b)*(s-c));
    }

    constexpr vec4 ComputeBaryCentricCoordinate(const vec3& p,
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
        // auto lbvh = get_input<zeno::LBvh>("lbvh");
        auto thickness = get_param<float>("bvh_thickness");

        const auto& verts = zsvolume->getParticles();
        const auto& eles = zsvolume->getQuadraturePoints();

        const auto& everts = zssurf->getParticles();
        const auto& etris = zssurf->getQuadraturePoints();

        auto &bcw = (*zsvolume)["bcws"];
        bcw = typename ZenoParticles::particles_t({{"inds",1},{"w",4},{"area",1}},everts.size(),zs::memsrc_e::device,0);

        auto cudaExec = zs::cuda_exec();
        const auto numFEMVerts = verts.size();
        const auto numFEMEles = eles.size();
        const auto numEmbedVerts = bcw.size();
        const auto numEmbedTris = etris.size();

        auto bvs = retrieve_bounding_volumes(cudaExec,verts,eles,wrapv<4>{},thickness);
        auto tetsBvh = zsvolume->bvh(ZenoParticles::s_elementTag);

        tetsBvh.build(cudaExec,bvs);

        constexpr auto space = zs::execspace_e::cuda;
        cudaExec(zs::range(numEmbedVerts),
            [this, verts = proxy<space>({},verts),eles = proxy<space>({},eles),bcw = proxy<space>({},bcw),everts = proxy<space>({},everts),tetsBvh = proxy<space>(tetsBvh)] ZS_LAMBDA (int vi) mutable {
                const auto& p = everts.pack<3>("x",vi);
                tetsBvh.iter_neighbors(p,[&](int ei){
                    auto inds = eles.pack<4>("inds", ei).reinterpret_bits<int>();
                    const auto& p0 = verts.pack<3>("x",inds[0]);
                    const auto& p1 = verts.pack<3>("x",inds[1]);
                    const auto& p2 = verts.pack<3>("x",inds[2]);
                    const auto& p3 = verts.pack<3>("x",inds[3]);

                    auto ws = ComputeBaryCentricCoordinate(p,p0,p1,p2,p3);

                    float epsilon = 1e-6;
                    if(ws[0] > -epsilon && ws[1] > -epsilon && ws[2] > -epsilon && ws[3] > -epsilon){
                        bcw("inds",vi) = reinterpret_bits<float>(ei);
                        bcw.tuple<4>("w",vi) = ws;
                    }
                });
            }
        );

        cudaExec(zs::range(numEmbedVerts),
            [bcw = proxy<space>({},bcw)] ZS_LAMBDA (int vi) mutable {
                bcw("area",vi) = 0.;
        });

        cudaExec(zs::range(numEmbedTris),
            [everts = proxy<space>({},everts),etris = proxy<space>({},etris),bcw = proxy<space>({},bcw)]
                ZS_LAMBDA (int ti) mutable {
                    auto inds = etris.pack<3>("inds",ti).reinterpret_bits<int>(); 
                    auto p0 = everts.pack<3>("x",inds[0]);
                    auto p1 = everts.pack<3>("x",inds[1]);
                    auto p2 = everts.pack<3>("x",inds[2]);

                    auto aA = ComputeArea(p0,p1,p2)/3;

                    bcw.tuple<1>("area",inds[0]) = bcw.pack<1>("area",inds[0]) + aA;
                    bcw.tuple<1>("area",inds[1]) = bcw.pack<1>("area",inds[1]) + aA;
                    bcw.tuple<1>("area",inds[2]) = bcw.pack<1>("area",inds[2]) + aA;
        });

        set_output("zsvolume", std::move(zsvolume));
    }
};

ZENDEFNODE(ZSComputeBaryCentricWeights, {{{"interpolator","zsvolume"}, {"embed surf", "zssurf"}},
                            {{"interpolator on gpu", "zsvolume"}},
                            {{"float","bvh_thickness","0"}},
                            {"FEM"}});


struct ZSInterpolateEmbedPrim : zeno::INode {
    void apply() override {
        using namespace zs;
        auto zstets = get_input<ZenoParticles>("zsvolume");
        auto zssurf = get_input<ZenoParticles>("zssurf");

        auto outAttr = get_param<std::string>("outAttr");

        auto cudaExec = zs::cuda_exec();

        auto &everts = zssurf->getParticles();
        // everts.append_channels(cudaExec, {{outAttr, 3}});
        
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
#if 0
                if(vi == 100){
                    auto vert = everts.pack<3>(outAttr,vi);
                    printf("V<%d>->E<%d>(%f,%f,%f,%f) :\t%f\t%f\t%f\n",vi,ei,w[0],w[1],w[2],w[3],vert[0],vert[1],vert[2]);
                }
#endif
        });
        set_output("zssurf",zssurf);
    }
};

ZENDEFNODE(ZSInterpolateEmbedPrim, {{{"zsvolume"}, {"embed primitive", "zssurf"}},
                            {{"embed primitive", "zssurf"}},
                            {{"string","outAttr","x"}},
                            {"FEM"}});


} // namespace zeno