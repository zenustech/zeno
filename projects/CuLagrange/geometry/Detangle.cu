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
                {"grad",3}
            },verts.size()};

        auto halfedges_host = halfedges.clone({zs::memsrc_e::host});
        auto tris_host = tris.clone({zs::memsrc_e::host});

        auto use_global_scheme = get_input2<bool>("use_global_scheme");

        auto tri_bvh = LBvh<3,int,T>{};

        for(int iter = 0;iter != nm_iters;++iter) {
            csHT.reset(cudaExec,true);

            GIA::retrieve_self_intersection_tri_halfedge_pairs(cudaExec,
                verts,xtag,
                tris,
                halfedges,
                tri_bvh,
                csHT,
                iter > 0); 

            if(csHT.size() == 0)
                break;

            GIA::eval_intersection_contour_minimization_gradient(cudaExec,
                verts,xtag,
                halfedges,
                tris,csHT,
                icm_grad,
                halfedges_host,
                tris_host,
                use_global_scheme);        

            TILEVEC_OPS::fill(cudaExec,vtemp,"grad",(T)0.0);
            // TILEVEC_OPS::fill(cudaExec,vtemp,"w",(T)0.0);

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
                    // auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                    // auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);

                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                    // vec3 halfedge_vertices[2] = {};
                    // vec3 tri_vertices[3] = {};
                    // vec3 htri_vertices[3] = {};

                    // for(int i = 0;i != 2;++i)
                    //     halfedge_vertices[i] = verts.pack(dim_c<3>,xtag,hedge[i]);

                    // for(int i = 0;i != 3;++i) {
                    //     tri_vertices[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                    //     htri_vertices[i] = verts.pack(dim_c<3>,xtag,htri[i]);
                    // }

                    // vec3 tri_bary{};
                    // vec2 edge_bary{};

                    // LSL_GEO::intersectionBaryCentric(halfedge_vertices[0],
                    //     halfedge_vertices[1],
                    //     tri_vertices[0],
                    //     tri_vertices[1],
                    //     tri_vertices[2],edge_bary,tri_bary);
                    
                    // T cminv = (T)0;
                    // for(int i = 0;i != 2;++i)
                    //     cminv += edge_bary[i] * edge_bary[i];
                    // for(int i = 0;i != 3;++i)
                    //     cminv += tri_bary[i] * tri_bary[i];

                    for(int i = 0;i != 2;++i) {
                        // auto beta = edge_bary[i] / cminv;
                        T beta = 1;
                        // atomic_add(exec_tag,&vtemp("w",hedge[i]),(T)1.0);
                        for(int d = 0;d != 3;++d)
                            atomic_add(exec_tag,&vtemp("grad",d,hedge[i]),impulse[d] * beta);
                    }

                    for(int i = 0;i != 3;++i) {
                        // auto beta = -tri_bary[i] / cminv;
                        T beta = -1;
                        // atomic_add(exec_tag,&vtemp("w",tri[i]),(T)1.0);
                        for(int d = 0;d != 3;++d)
                            atomic_add(exec_tag,&vtemp("grad",d,tri[i]),impulse[d] * beta);                    
                    }
            });

            // auto gradn = TILEVEC_OPS::dot<3>(cudaExec,vtemp,"grad","grad");
            // std::cout << "gradn : " << gradn << std::endl;

            cudaExec(zs::range(verts.size()),[
                eps = eps,
                h0 = maximum_correction,
                g02 = progressive_slope * progressive_slope,
                xtag = zs::SmallString(xtag),
                verts = proxy<space>({},verts),
                vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                    // if(vtemp("w",vi) > eps)
                    //     verts.tuple(dim_c<3>,xtag,vi) = verts.pack(dim_c<3>,xtag,vi) + vtemp.pack(dim_c<3>,"grad",vi) / vtemp("w",vi);
                    auto G = vtemp.pack(dim_c<3>,"grad",vi);
                    auto Gn = G.norm();
                    auto Gn2 = Gn * Gn;
                    auto impulse = h0 * G / zs::sqrt(Gn2 + g02 + 1e-6);
                    verts.tuple(dim_c<3>,xtag,vi) = verts.pack(dim_c<3>,xtag,vi) + impulse;
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
        {"bool","use_global_scheme","0"},
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

struct Detangle2 : zeno::INode {
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
        auto& verts = zsparticles->getParticles();
        const auto& halfedges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
        auto& edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
        auto& tris = zsparticles->getQuadraturePoints();

        auto xtag = get_input2<std::string>("xtag");     
        // these buffer should be initialized during zcloth initialization
        auto nm_iters = get_input2<int>("nm_iters");

        auto maximum_correction = get_input2<float>("maximum_correction");
        auto progressive_slope = get_input2<float>("progressive_slope");
        auto use_global_scheme = get_input2<bool>("use_global_scheme");

        // zs::bht<int,2,int> csET{verts.get_allocator(),GIA::DEFAULT_MAX_GIA_INTERSECTION_PAIR};
        if(!zsparticles->hasMeta(GIA::GIA_CS_ET_BUFFER_KEY)) {
            zsparticles->setMeta(GIA::GIA_CS_ET_BUFFER_KEY,
                zs::bht<int,2,int>{verts.get_allocator(),GIA::DEFAULT_MAX_GIA_INTERSECTION_PAIR});
        }
        auto& csET = zsparticles->readMeta<zs::bht<int,2,int>>(GIA::GIA_CS_ET_BUFFER_KEY);

        // if(!zsparticles->hasMeta(GIA::GIA_CS_EKT_BUFFER_KEY)) {
        //     zsparticles->setMeta(GIA::GIA_CS_EKT_BUFFER_KEY,
        //         zs::bht<int,2,int>{verts.get_allocator(),GIA::DEFAULT_MAX_GIA_INTERSECTION_PAIR});
        // }
        // auto& csEKT = zsparticles->readMeta<zs::bht<int,2,int>>(GIA::GIA_CS_EKT_BUFFER_KEY);

        auto has_bvh = zsparticles->hasBvh(GIA::GIA_TRI_BVH_BUFFER_KEY);
        if(!zsparticles->hasBvh(GIA::GIA_TRI_BVH_BUFFER_KEY))
            zsparticles->bvh(GIA::GIA_TRI_BVH_BUFFER_KEY) = LBvh<3,int,T>{};
        auto& tri_bvh = zsparticles->bvh(GIA::GIA_TRI_BVH_BUFFER_KEY);   

        if(!verts.hasProperty("grad") || verts.getPropertySize("grad") != 3)
            verts.append_channels(cudaExec,{{"grad",3}});

        if(!zsparticles->hasAuxData(GIA::GIA_IMC_GRAD_BUFFER_KEY)) {
            (*zsparticles)[GIA::GIA_IMC_GRAD_BUFFER_KEY] = dtiles_t{verts.get_allocator(),{
                {"grad",3}
            },GIA::DEFAULT_MAX_GIA_INTERSECTION_PAIR};
        }

        auto& icm_grad = (*zsparticles)[GIA::GIA_IMC_GRAD_BUFFER_KEY];

        zs::CppTimer timer;

        #define EDGE_NORMAL_DOF 3 

        if(!tris.hasProperty("nrm") || tris.getPropertySize("nrm") != 3) {
            tris.append_channels(cudaExec,{{"nrm",3}});
        }
        if(!tris.hasProperty("d") || tris.getPropertySize("d") != 1) {
            tris.append_channels(cudaExec,{{"d",1}});
        }

        dtiles_t kvtemp{verts.get_allocator(),{
            {"x",3}
        },0};
        dtiles_t kttemp{tris.get_allocator(),{
            {"inds",3},
            {"nrm",3},
            {"d",1}
        },0};

        // if(has_input<ZenoParticles>("kboundary")) {
        //     auto kboundary = get_input<ZenoParticles>("kboundary");
        //     auto substep_id = get_input2<int>("substep_id");
        //     auto nm_substeps = get_input2<int>("nm_substeps");
        //     auto w = (float)(substep_id + 1) / (float)nm_substeps;
        //     auto pw = (float)(substep_id) / (float)nm_substeps;  
        //     const auto& kverts = kboundary->getParticles();
        //     const auto& ktris = kboundary->getQuadraturePoints();  

        //     kvtemp.resize(kverts.size());
        //     auto kxtag = "x";
        //     auto kpxtag = kverts.hasProperty("px") ? "px" : "x";
        //     cudaExec(zs::range(kverts.size()),[
        //         w = w,
        //         kxtag = zs::SmallString(kxtag),
        //         kpxtag = zs::SmallString(kpxtag),
        //         kverts = proxy<space>({},kverts),
        //         kvtemp = proxy<space>({},kvtemp)] ZS_LAMBDA(int kvi) mutable {
        //             auto kvert = kverts.pack(dim_c<3>,kpxtag,kvi) * (1 - w) + kverts.pack(dim_c<3>,kxtag,kvi) * w;
        //             kvtemp.tuple(dim_c<3>,"x",kvi) = kvert;
        //     });
        //     kttemp.resize(ktris.size());    
        //     TILEVEC_OPS::copy<3>(cudaExec,ktris,"inds",kttemp,"inds");
        //     cudaExec(zs::range(kttemp.size()),[
        //         kttemp = proxy<space>({},kttemp),
        //         kvtemp = proxy<space>({},kvtemp)] ZS_LAMBDA(int kti) mutable {
        //             auto ktri = kttemp.pack(dim_c<3>,"inds",kti,int_c);
        //             zs::vec<T,3> ktvs[3] = {};
        //             for(int i = 0;i != 3;++i)
        //                 ktvs[i] = kvtemp.pack(dim_c<3>,"x",ktri[i]);
        //             kttemp.tuple(dim_c<3>,"nrm",kti) = LSL_GEO::facet_normal(ktvs[0],ktvs[1],ktvs[2]);
        //             kttemp("d",kti) = -kttemp.pack(dim_c<3>,"nrm",kti).dot(ktvs[0]);
        //     });         

        //     if(!kboundary->hasBvh(GIA::GIA_TRI_BVH_BUFFER_KEY)) {
        //         kboundary->bvh(GIA::GIA_TRI_BVH_BUFFER_KEY) = LBvh<3,int,T>{};
        //         auto& ktri_bvh = kboundary->bvh(GIA::GIA_TRI_BVH_BUFFER_KEY); 
        //         auto kbvs = retrieve_bounding_volumes(cudaExec,kvtemp,ktris,wrapv<3>{},(T)0,"x");
        //         ktri_bvh.build(cudaExec,kbvs);
        //     }else {
        //         auto need_refit_bvh = get_input2<bool>("refit_kboundary_bvh");
        //         if(need_refit_bvh) {
        //             auto& ktri_bvh = kboundary->bvh(GIA::GIA_TRI_BVH_BUFFER_KEY); 
        //             auto kbvs = retrieve_bounding_volumes(cudaExec,kvtemp,ktris,wrapv<3>{},(T)0,"x");
        //             ktri_bvh.refit(cudaExec,kbvs);
        //         }
        //     }     
        // }

        for(int iter = 0;iter != nm_iters;++iter) {
            TILEVEC_OPS::fill(cudaExec,verts,"grad",(T)0.0);

            csET.reset(cudaExec,true);
            auto use_dirty_bits = iter > 0;

            timer.tick();
            cudaExec(zs::range(tris.size()),[
                eps = eps,
                tris = proxy<space>({},tris),
                verts = proxy<space>({},verts),
                xtag = zs::SmallString(xtag)] ZS_LAMBDA(int ti) mutable {
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    zs::vec<T,3> tvs[3] = {};
                    for(int i = 0;i != 3;++i)
                        tvs[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                    auto tnrm = LSL_GEO::facet_normal(tvs[0],tvs[1],tvs[2]);
                    auto d = -tnrm.dot(tvs[0]);
                    tris.tuple(dim_c<3>,"nrm",ti) = tnrm;
                    tris("d",ti) = d;
            });
            timer.tock("eval triangle plane");

            GIA::retrieve_self_intersection_tri_edge_pairs(cudaExec,
                verts,xtag,
                tris,
                edges,
                tri_bvh,
                csET,
                has_bvh); 

            if(csET.size() > 0) {
                timer.tick();
                GIA::eval_self_intersection_contour_minimization_gradient(cudaExec,
                    verts,xtag,
                    edges,
                    halfedges,
                    tris,
                    csET,
                    icm_grad);      
                timer.tock("eval_self_intersection_contour_minimization_gradient");  


                timer.tick();
                cudaExec(zip(zs::range(csET.size()),csET._activeKeys),[
                    exec_tag = exec_tag,
                    h0 = maximum_correction,
                    g02 = progressive_slope * progressive_slope,
                    xtag = zs::SmallString(xtag),
                    // vtemp = proxy<space>({},vtemp),
                    icm_grad = proxy<space>({},icm_grad),
                    verts = proxy<space>({},verts),
                    edges = proxy<space>({},edges),
                    tris = proxy<space>({},tris)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                        // auto pair = icm_grad.pack(dim_c<2>,"inds",ci,int_c);
                        auto ei = pair[0];
                        auto ti = pair[1];
                        auto G = icm_grad.pack(dim_c<3>,"grad",ci);

                        auto Gn = G.norm();
                        auto Gn2 = Gn * Gn;
                        auto impulse = h0 * G / zs::sqrt(Gn2 + g02);

                        // edges("dirty",ei) = 1;
                        // tris("dirty",ti) = 1;

                        auto edge = edges.pack(dim_c<2>,"inds",ei,int_c);
                        auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                        for(int i = 0;i != 2;++i) {
                            T beta = 1;
                            for(int d = 0;d != 3;++d)
                                atomic_add(exec_tag,&verts("grad",d,edge[i]),impulse[d] * beta);
                        }

                        for(int i = 0;i != 3;++i) {
                            T beta = -1;
                            for(int d = 0;d != 3;++d)
                                atomic_add(exec_tag,&verts("grad",d,tri[i]),impulse[d] * beta);                    
                        }
                });
                timer.tock("assemble self icm gradient");
            }

            // if(has_input<ZenoParticles>("kboundary")) {
            //     csET.reset(cudaExec,true);
            //     auto kboundary = get_input<ZenoParticles>("kboundary");
            //     const auto& ktri_bvh = kboundary->bvh(GIA::GIA_TRI_BVH_BUFFER_KEY);
            //     GIA::retrieve_intersection_with_edge_ktri_pairs(cudaExec,
            //         verts,xtag,
            //         edges,
            //         kvtemp,"x",
            //         kttemp,
            //         ktri_bvh,
            //         csET);

            //     if(csET.size() > 0) {
            //         timer.tick();
            //         GIA::eval_intersection_contour_minimization_gradient_with_edge_and_ktri(cudaExec,
            //             verts,xtag,
            //             edges,
            //             halfedges,
            //             tris,
            //             kvtemp,"x",
            //             kttemp,
            //             csET,
            //             icm_grad);      
            //         timer.tock("eval_intersection_contour_minimization_gradient_with_edge_and_ktri");  

            //         timer.tick();
            //         cudaExec(zip(zs::range(csET.size()),csET._activeKeys),[
            //             exec_tag = exec_tag,
            //             h0 = maximum_correction,
            //             g02 = progressive_slope * progressive_slope,
            //             xtag = zs::SmallString(xtag),
            //             // vtemp = proxy<space>({},vtemp),
            //             icm_grad = proxy<space>({},icm_grad),
            //             verts = proxy<space>({},verts),
            //             edges = proxy<space>({},edges)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
            //                 // auto pair = icm_grad.pack(dim_c<2>,"inds",ci,int_c);
            //                 auto ei = pair[0];
            //                 auto G = icm_grad.pack(dim_c<3>,"grad",ci);

            //                 auto Gn = G.norm();
            //                 auto Gn2 = Gn * Gn;
            //                 auto impulse = h0 * G / zs::sqrt(Gn2 + g02);

            //                 auto edge = edges.pack(dim_c<2>,"inds",ei,int_c);
            //                 for(int i = 0;i != 2;++i) {
            //                     T beta = 1;
            //                     for(int d = 0;d != 3;++d)
            //                         atomic_add(exec_tag,&verts("grad",d,edge[i]),impulse[d] * beta);
            //                 }
            //         });
            //         timer.tock("assemble self icm gradient");                    
            //     } 
            // }

            timer.tick();
            cudaExec(zs::range(verts.size()),[
                eps = eps,
                h0 = maximum_correction,
                g02 = progressive_slope * progressive_slope,
                xtag = zs::SmallString(xtag),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
                    // if(vtemp("w",vi) > eps)
                    //     verts.tuple(dim_c<3>,xtag,vi) = verts.pack(dim_c<3>,xtag,vi) + vtemp.pack(dim_c<3>,"grad",vi) / vtemp("w",vi);
                    auto G = verts.pack(dim_c<3>,"grad",vi);
                    if(G.norm() < eps)
                        return;
                    // auto Gn = G.norm();
                    // auto Gn2 = Gn * Gn;
                    // auto impulse = h0 * G / zs::sqrt(Gn2 + g02 + 1e-6);
                    auto impulse = G;
                    verts.tuple(dim_c<3>,xtag,vi) = verts.pack(dim_c<3>,xtag,vi) + impulse;
            });
            timer.tock("write_back_to_displacement");
        }

        set_output("zsparticles",zsparticles);
    }
}; 


ZENDEFNODE(Detangle2, {
    {
        {"zsparticles"},
        {"string", "xtag", "x"},
        {"int","nm_iters","1"},
        {"bool","use_global_scheme","0"},
        {"float","maximum_correction","0.1"},
        {"float","progressive_slope","0.1"},
        {"kboudary"},
        {"int","substep_id","0"},
        {"int","nm_substeps","1"},
        {"bool","refit_kboundary_bvh","1"}
    },
    {
        {"zsparticles"}
    },
    {
    },
    {"GIA"},
});


};