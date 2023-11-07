#include "Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>

#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "kernel/global_intersection_analysis.hpp"


namespace zeno {

struct Detangle : zeno::INode {
    virtual void apply () override {
        using namespace zs;
        auto cudaExec = cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;
        using T = float;
        using dtiles_t = zs::TileVector<T, 32>;
        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        auto exec_tag = wrapv<space>{};
        constexpr auto eps = (T)1e-6;

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        const auto& verts = zsparticles->getParticles();
        const auto& halfedges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
        const auto& tris = zsparticles->getQuadraturePoints();

        auto xtag = get_input2<std::string>("xtag");     
        // these buffer should be initialized during zcloth initialization
        zs::bht<int,2,int> csHT{verts.get_allocator(),GIA::DEFAULT_MAX_GIA_INTERSECTION_PAIR};

        auto nm_iters = get_input2<int>("nm_iters");

        dtiles_t icm_grad{verts.get_allocator(),{
            {"grad",3},
            {"inds",2}
        },0};

        auto maximum_correction = get_input2<float>("maximum_correction");
        auto progressive_slope = get_input2<float>("progressive_slope");

        dtiles_t vtemp{verts.get_allocator(),{
                {"grad",3},
                {"w",1}
            },verts.size()};

            

        for(int iter = 0;iter != nm_iters;++iter) {
            csHT.reset(cudaExec,true);
            GIA::eval_intersection_contour_minimization_gradient(cudaExec,
                verts,xtag,
                halfedges,
                tris,csHT,
                icm_grad);        

            TILEVEC_OPS::fill(cudaExec,vtemp,"grad",(T)0.0);
            TILEVEC_OPS::fill(cudaExec,vtemp,"w",(T)0.0);

            cudaExec(zs::range(icm_grad.size()),[
                exec_tag = exec_tag,
                h0 = maximum_correction,
                g02 = progressive_slope * progressive_slope,
                xtag = zs::SmallString(xtag),
                vtemp = proxy<space>({},vtemp),
                icm_grad = proxy<space>({},icm_grad),
                verts = proxy<space>({},verts),
                halfedges = proxy<space>({},halfedges),
                tris = proxy<space>({},tris)] ZS_LAMBDA(int ci) mutable {
                    auto pair = icm_grad.pack(dim_c<2>,"inds",ci,int_c);
                    auto hi = pair[0];
                    auto ti = pair[1];
                    auto G = icm_grad.pack(dim_c<3>,"grad",ci);

                    auto Gn = G.norm();
                    auto Gn2 = Gn * Gn;
                    auto impulse = h0 * G / zs::sqrt(Gn2 + g02);

                    auto hedge = half_edge_get_edge(hi,halfedges,tris);
                    auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                    auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);

                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                    vec3 halfedge_vertices[2] = {};
                    vec3 tri_vertices[3] = {};
                    vec3 htri_vertices[3] = {};

                    for(int i = 0;i != 2;++i)
                        halfedge_vertices[i] = verts.pack(dim_c<3>,xtag,hedge[i]);

                    for(int i = 0;i != 3;++i) {
                        tri_vertices[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                        htri_vertices[i] = verts.pack(dim_c<3>,xtag,htri[i]);
                    }

                    vec3 tri_bary{};
                    vec2 edge_bary{};

                    LSL_GEO::intersectionBaryCentric(halfedge_vertices[0],
                        halfedge_vertices[1],
                        tri_vertices[0],
                        tri_vertices[1],
                        tri_vertices[2],edge_bary,tri_bary);
                    
                    T cminv = (T)0;
                    for(int i = 0;i != 2;++i)
                        cminv += edge_bary[i] * edge_bary[i];
                    for(int i = 0;i != 3;++i)
                        cminv += tri_bary[i] * tri_bary[i];

                    for(int i = 0;i != 2;++i) {
                        auto beta = edge_bary[i] / cminv;
                        atomic_add(exec_tag,&vtemp("w",hedge[i]),(T)1.0);
                        for(int d = 0;d != 3;++d)
                            atomic_add(exec_tag,&vtemp("grad",d,hedge[i]),impulse[d] * beta);
                    }

                    for(int i = 0;i != 3;++i) {
                        auto beta = -tri_bary[i] / cminv;
                        atomic_add(exec_tag,&vtemp("w",tri[i]),(T)1.0);
                        for(int d = 0;d != 3;++d)
                            atomic_add(exec_tag,&vtemp("grad",d,tri[i]),impulse[d] * beta);                    
                    }
            });

            auto gradn = TILEVEC_OPS::dot<3>(cudaExec,vtemp,"grad","grad");
            std::cout << "gradn : " << gradn << std::endl;

            cudaExec(zs::range(verts.size()),[
                eps = eps,
                xtag = zs::SmallString(xtag),
                verts = proxy<space>({},verts),
                vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                    if(vtemp("w",vi) > eps)
                        verts.tuple(dim_c<3>,xtag,vi) = verts.pack(dim_c<3>,xtag,vi) + vtemp.pack(dim_c<3>,"grad",vi) / vtemp("w",vi);
                        verts.tuple(dim_c<3>,xtag,vi) = verts.pack(dim_c<3>,xtag,vi) + vtemp.pack(dim_c<3>,"grad",vi);
            });

        }

        set_output("zsparticles",zsparticles);
    }
}; 


ZENDEFNODE(Detangle, {
    {
        {"zsparticles"},
        {"string", "xtag", "x"},
        {"int","nm_iters","1"},
        {"float","maximum_correction","0.1"},
        {"float","progressive_slope","0.1"}
    },
    {
        {"zsparticles"}
    },
    {
    },
    {"GIA"},
});


};