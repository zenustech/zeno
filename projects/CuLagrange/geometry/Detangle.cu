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

            std::cout << "nm_self_intersections_HT : " << csHT.size() << std::endl;

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
                
            auto icm_gradn = TILEVEC_OPS::dot<3>(cudaExec,icm_grad,"grad","grad");
            std::cout << "icm_HT_gradn : " << icm_gradn << std::endl;

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
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

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

            auto gradn = TILEVEC_OPS::dot<3>(cudaExec,vtemp,"grad","grad");
            std::cout << "apply gradient : " << gradn << std::endl;

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
        auto& csET = zsparticles->readMeta<zs::bht<int,2,int> &>(GIA::GIA_CS_ET_BUFFER_KEY);

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
        if(!verts.hasProperty("icm_intersected") || verts.getPropertySize("icm_intersected") != 1)
            verts.append_channels(cudaExec,{{"icm_intersected",1}});

        if(!zsparticles->hasAuxData(GIA::GIA_IMC_GRAD_BUFFER_KEY)) {
            (*zsparticles)[GIA::GIA_IMC_GRAD_BUFFER_KEY] = dtiles_t{verts.get_allocator(),{
                {"grad",3},{"bary",4},{"inds",2}
            },GIA::DEFAULT_MAX_GIA_INTERSECTION_PAIR};
        }

        auto& icm_grad = (*zsparticles)[GIA::GIA_IMC_GRAD_BUFFER_KEY];
        TILEVEC_OPS::fill(cudaExec,icm_grad,"grad",(T)0.0);
        auto use_barycentric_interpolator = get_input2<bool>("use_barycentric_interpolator");
        // if(use_barycentric_interpolator && !icm_grad.hasProperty("bary")) {
        //     icm_grad.append_channels(cudaExec,{{"bary",4}});
        // }

        auto skip_animation_intersection = get_input2<bool>("skip_animation_intersection");

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

        LBvh<3,int,T> ktri_bvh{};

        zs::Vector<int> impulse_count{verts.get_allocator(),verts.size()};
        cudaExec(zs::range(impulse_count),[]ZS_LAMBDA(auto& count) mutable {count = 0;});

        auto relaxation_rate = get_input2<T>("relaxation_rate");
        bool mark_intersection = get_input2<bool>("mark_intersection");


        auto detangle_with_boundary = get_input2<bool>("detangle_with_boundary");
        auto do_self_detangle = get_input2<bool>("do_self_detangle");


        if(has_input<ZenoParticles>("kboundary") && detangle_with_boundary) {
            auto kboundary = get_input<ZenoParticles>("kboundary");
            auto substep_id = get_input2<int>("substep_id");
            auto nm_substeps = get_input2<int>("nm_substeps");
            auto w = (float)(substep_id + 1) / (float)nm_substeps;
            auto pw = (float)(substep_id) / (float)nm_substeps;  
            const auto& kverts = kboundary->getParticles();
            const auto& kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];
            const auto& ktris = kboundary->getQuadraturePoints();  

            kvtemp.resize(kverts.size());
            auto kxtag = "x";
            auto kpxtag = kverts.hasProperty("px") ? "px" : "x";
            auto use_cur_kine_configuration = get_input2<bool>("use_cur_kine_configuration");
            auto alpha = use_cur_kine_configuration ? w : pw;
            cudaExec(zs::range(kverts.size()),[
                alpha = alpha,
                kxtag = zs::SmallString(kxtag),
                kpxtag = zs::SmallString(kpxtag),
                kverts = proxy<space>({},kverts),
                kvtemp = proxy<space>({},kvtemp)] ZS_LAMBDA(int kvi) mutable {
                    auto kvert = kverts.pack(dim_c<3>,kpxtag,kvi) * (1 - alpha) + kverts.pack(dim_c<3>,kxtag,kvi) * alpha;
                    kvtemp.tuple(dim_c<3>,"x",kvi) = kvert;
            });
            kttemp.resize(ktris.size());    
            TILEVEC_OPS::copy<3>(cudaExec,ktris,"inds",kttemp,"inds");
            cudaExec(zs::range(kttemp.size()),[
                kttemp = proxy<space>({},kttemp),
                kvtemp = proxy<space>({},kvtemp)] ZS_LAMBDA(int kti) mutable {
                    auto ktri = kttemp.pack(dim_c<3>,"inds",kti,int_c);
                    zs::vec<T,3> ktvs[3] = {};
                    for(int i = 0;i != 3;++i)
                        ktvs[i] = kvtemp.pack(dim_c<3>,"x",ktri[i]);
                    kttemp.tuple(dim_c<3>,"nrm",kti) = LSL_GEO::facet_normal(ktvs[0],ktvs[1],ktvs[2]);
                    kttemp("d",kti) = -kttemp.pack(dim_c<3>,"nrm",kti).dot(ktvs[0]);
            });         

            auto kbvs = retrieve_bounding_volumes(cudaExec,kvtemp,ktris,wrapv<3>{},(T)0,"x");
            ktri_bvh.build(cudaExec,kbvs);     
        }

        for(int iter = 0;iter != nm_iters;++iter) {
            auto refit_tri_bvh = iter > 0;
            timer.tick();
            auto tri_bvs = retrieve_bounding_volumes(cudaExec,verts,tris,wrapv<3>{},0,xtag);
            timer.tock("retrieve_tri_bvh_bounding_volumes");

            timer.tick();
            if(refit_tri_bvh)
                tri_bvh.refit(cudaExec,tri_bvs);
            else
                tri_bvh.build(cudaExec,tri_bvs);   
            timer.tock("refit_tri_bvh");            


            cudaExec(zs::range(impulse_count),[]ZS_LAMBDA(auto& count) mutable {count = 0;});

            TILEVEC_OPS::fill(cudaExec,verts,"grad",(T)0.0);
            if(mark_intersection)
                TILEVEC_OPS::fill(cudaExec,verts,"icm_intersected",(T)0.0);


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


            bool has_kine_intersection = false;
            bool has_self_intersection = false;
            if(has_input<ZenoParticles>("kboundary") && detangle_with_boundary) {
                auto kboundary = get_input<ZenoParticles>("kboundary");
                const auto& kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];
                const auto& ktris = kboundary->getQuadraturePoints();
                // const auto& ktri_bvh = kboundary->bvh(GIA::GIA_TRI_BVH_BUFFER_KEY);
                const auto& khalfedges = (*kboundary)[ZenoParticles::s_surfHalfEdgeTag];
                int nm_kinematic_intersection = 0;
                {
                    timer.tick();


                    GIA::retrieve_intersection_with_edge_tri_pairs(cudaExec,
                        verts,xtag,
                        edges,
                        kvtemp,"x",
                        kttemp,
                        ktri_bvh,
                        csET,
                        icm_grad,
                        use_barycentric_interpolator);
                    timer.tock("retrieve_intersection_with_EKT_pairs");

                    if(csET.size() > 0)
                        has_kine_intersection = true;


                    nm_kinematic_intersection += csET.size();
                    auto enforce_boundary_normal = get_input2<bool>("enforce_boundary_normal");
                    timer.tick();
                    GIA::eval_intersection_contour_minimization_gradient_of_edgeA_with_triB(cudaExec,
                        verts,xtag,
                        edges,
                        halfedges,
                        tris,
                        kvtemp,"x",
                        kttemp,
                        maximum_correction,
                        progressive_slope,                        
                        csET,
                        icm_grad,
                        enforce_boundary_normal);      
                    timer.tock("eval_intersection_contour_minimization_gradient_with_EKT");  
                    timer.tick();
                    cudaExec(zip(zs::range(csET.size()),csET._activeKeys),[
                        impulse_count = proxy<space>(impulse_count),
                        exec_tag = exec_tag,
                        eps = eps,
                        use_barycentric_interpolator = use_barycentric_interpolator,
                        mark_intersection = mark_intersection,
                        xtag = zs::SmallString(xtag),
                        icm_grad = proxy<space>({},icm_grad),
                        relaxation_rate = relaxation_rate,
                        gradOffset = verts.getPropertyOffset("grad"),
                        verts = proxy<space>({},verts),
                        edges = proxy<space>({},edges)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                            auto ei = pair[0];
                            auto edge = edges.pack(dim_c<2>,"inds",ei,int_c);
                            if(mark_intersection) {
                                verts("icm_intersected",edge[0]) = (T)1.0;
                                verts("icm_intersected",edge[1]) = (T)1.0;
                            }

                            auto impulse = icm_grad.pack(dim_c<3>,"grad",ci) * relaxation_rate;

                            // printf("EKT impulse[%d] : %f\n",ci,(float)icm_grad.pack(dim_c<3>,"grad",ci).norm());

                            if(impulse.norm() < eps)
                                return;

                            T edge_cminv = 1;
                            zs::vec<T,2> edge_bary{};
                            if(use_barycentric_interpolator) {
                                auto bary = icm_grad.pack(dim_c<4>,"bary",ci);
                                edge_bary[0] = bary[0];
                                edge_bary[1] = 1 - bary[0];

                                edge_cminv = 0;
                                for(int i = 0;i != 2;++i)
                                    edge_cminv += edge_bary[i] * edge_bary[i] / verts("m",edge[i]);
                            }


                            for(int i = 0;i != 2;++i) {
                                T beta = 1;
                                if(use_barycentric_interpolator) {
                                    beta = verts("minv",edge[i]) * edge_bary[i] / edge_cminv;
                                    // printf("edge[%d][%d]_beta : %f\n",ei,edge[i],(float)beta);
                                }
                                atomic_add(exec_tag,&impulse_count[edge[i]],1);
                                for(int d = 0;d != 3;++d)
                                    atomic_add(exec_tag,&verts(gradOffset + d,edge[i]),impulse[d] * beta);
                            }
                    });

                    // std::cout << "EKT IMPULSE : " << TILEVEC_OPS::dot<3>(cudaExec,verts,"grad","grad") << std::endl;
                    timer.tock("assemble EKT icm gradient");   
                }

                {
                    timer.tick();
                    GIA::retrieve_intersection_with_edge_tri_pairs(cudaExec,
                        kvtemp,"x",
                        kedges,
                        verts,xtag,
                        tris,
                        tri_bvh,
                        csET,
                        icm_grad,
                        use_barycentric_interpolator);
                    timer.tock("retrieve_intersection_with_KET_pairs");

                    if(csET.size() > 0)
                        has_kine_intersection = true;


                    nm_kinematic_intersection += csET.size();



                    timer.tick();
                    GIA::eval_intersection_contour_minimization_gradient_of_edgeA_with_triB(cudaExec,
                        kvtemp,"x",
                        kedges,
                        khalfedges,
                        kttemp,
                        verts,xtag,
                        tris,
                        maximum_correction,
                        progressive_slope,                        
                        csET,
                        icm_grad);      
                    timer.tock("eval_intersection_contour_minimization_gradient_with_KET"); 

                    timer.tick();
                    cudaExec(zip(zs::range(csET.size()),csET._activeKeys),[
                        exec_tag = exec_tag,
                        impulse_count = proxy<space>(impulse_count),
                        eps = eps,
                        use_barycentric_interpolator = use_barycentric_interpolator,
                        mark_intersection = mark_intersection,
                        xtag = zs::SmallString(xtag),
                        gradOffset = verts.getPropertyOffset("grad"),
                        icm_grad = proxy<space>({},icm_grad),
                        relaxation_rate = relaxation_rate,
                        verts = proxy<space>({},verts),
                        tris = proxy<space>({},tris)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                            auto ti = pair[1];
                            auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                            if(mark_intersection) {
                                verts("icm_intersected",tri[0]) = (T)1.0;
                                verts("icm_intersected",tri[1]) = (T)1.0;
                                verts("icm_intersected",tri[2]) = (T)1.0;
                            }

                            auto impulse = icm_grad.pack(dim_c<3>,"grad",ci) * relaxation_rate;

                            // printf("KET impulse[%d] : %f\n",ci,(float)icm_grad.pack(dim_c<3>,"grad",ci).norm());
                            // if(impulse.norm() < eps)
                            //     return;


                            T tri_cminv = 1;
                            zs::vec<T,3> tri_bary{};

                            if(use_barycentric_interpolator) {
                                auto bary = icm_grad.pack(dim_c<4>,"bary",ci);
                                tri_bary[0] = bary[1];
                                tri_bary[1] = bary[2];
                                tri_bary[2] = bary[3];

                                tri_cminv = 0;
                                for(int i = 0;i != 3;++i)
                                    tri_cminv += tri_bary[i] * tri_bary[i] / verts("m",tri[i]);
                                // cminv = t * t / verts("m",edge[0]) + (1 - t) * (1 - t) / verts("m",edge[1]);
                            }


                            for(int i = 0;i != 3;++i) {
                                T beta = 1;
                                if(use_barycentric_interpolator) {
                                    beta = verts("minv",tri[i]) * tri_bary[i] / tri_cminv;
                                    // printf("tri[%d][%d]_beta : %f\n",ti,tri[i],(float)beta);
                                }
                                atomic_add(exec_tag,&impulse_count[tri[i]],1);
                                for(int d = 0;d != 3;++d)
                                    atomic_add(exec_tag,&verts(gradOffset + d,tri[i]),-impulse[d] * beta);
                            }
                    });

                    std::cout << "KET IMPULSE : " << TILEVEC_OPS::dot<3>(cudaExec,verts,"grad","grad") << std::endl;
                    timer.tock("assemble KET icm gradient");      


                }            

                // std::cout << "do kinematic boundary detangle with " << nm_kinematic_intersection << " intersections" << std::endl;
            }

            // csET.reset(cudaExec,true);

            if(do_self_detangle) {
                std::cout << "do self detangle" << std::endl;

                GIA::retrieve_self_intersection_tri_edge_pairs(cudaExec,
                    verts,xtag,
                    tris,
                    edges,
                    tri_bvh,
                    csET,
                    icm_grad,
                    // has_bvh,
                    use_barycentric_interpolator); 

                std::cout << "nm_self_intersections_ET : " << csET.size() << std::endl;

                has_self_intersection = csET.size() > 0;

                if(has_self_intersection) {
                    timer.tick();
                    GIA::eval_self_intersection_contour_minimization_gradient(cudaExec,
                        verts,xtag,
                        edges,
                        halfedges,
                        tris,
                        maximum_correction,
                        progressive_slope,
                        csET,
                        icm_grad);      
                    timer.tock("eval_self_intersection_contour_minimization_gradient");  
                    
                    auto icm_gradn = TILEVEC_OPS::dot<3>(cudaExec,icm_grad,"grad","grad");
                    std::cout << "icm_ET_gradn : " << icm_gradn << std::endl;

                    timer.tick();
                    cudaExec(zip(zs::range(csET.size()),csET._activeKeys),[
                        impulse_count = proxy<space>(impulse_count),
                        exec_tag = exec_tag,
                        mark_intersection = mark_intersection,
                        use_barycentric_interpolator = use_barycentric_interpolator,
                        eps = eps,
                        relaxation_rate = relaxation_rate,
                        h0 = maximum_correction,
                        g02 = progressive_slope * progressive_slope,
                        // xtagOffset = verts.getPropertyOffset(xtag),
                        icm_grad = proxy<space>({},icm_grad),
                        verts = proxy<space>({},verts),
                        gradOffset = verts.getPropertyOffset("grad"),
                        edges = edges.begin("inds", dim_c<2>, int_c),
                        tris = tris.begin("inds", dim_c<3>, int_c)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                            auto edge = edges[pair[0]];
                            auto tri = tris[pair[1]];

                            // auto G = icm_grad.pack(dim_c<3>,"grad",ci) * relaxation_rate;
                            // if(G.norm() < eps)
                            //     return;

                            // auto Gn = G.norm();
                            // auto Gn2 = Gn * Gn;
                            // auto impulse = h0 * G / zs::sqrt(Gn2 + g02);
                            auto impulse = icm_grad.pack(dim_c<3>,"grad",ci) * relaxation_rate;
                            

                            T tri_cminv = 1;
                            T edge_cminv = 1;

                            zs::vec<T,3> tri_bary{};
                            zs::vec<T,2> edge_bary{};

                            if(use_barycentric_interpolator) {
                                auto bary = icm_grad.pack(dim_c<4>,"bary",ci);
                                edge_bary[0] = bary[0];
                                edge_bary[1] = 1 - bary[0];
                                tri_bary[0] = bary[1];
                                tri_bary[1] = bary[2];
                                tri_bary[2] = bary[3];

                                tri_cminv = 0;
                                edge_cminv = 0;
                                for(int i = 0;i != 3;++i)
                                    tri_cminv += tri_bary[i] * tri_bary[i] / verts("m",tri[i]);
                                
                                for(int i = 0;i != 2;++i)
                                    edge_cminv += edge_bary[i] * edge_bary[i] / verts("m",edge[i]);
                                // cminv = t * t / verts("m",edge[0]) + (1 - t) * (1 - t) / verts("m",edge[1]);
                            }


                            for(int i = 0;i != 2;++i) { 
                                T beta = 1;
                                if(use_barycentric_interpolator) {
                                    // printf("verts[%d].minv = %f -> %f\n",edge[i],(float)verts("minv",edge[i]),(float)verts("m",edge[i]));
                                    beta = verts("minv",edge[i]) * edge_bary[i] / edge_cminv;
                                }
                                // atomic_add(exec_tag,&impulse_count[edge[i]],1);
                                for(int d = 0;d != 3;++d)
                                    atomic_add(exec_tag,&verts(gradOffset + d,edge[i]),impulse[d]  * beta);
                            }

                            for(int i = 0;i != 3;++i) {
                                T beta = 1;
                                if(use_barycentric_interpolator) {
                                    // printf("verts[%d].minv = %f -> %f\n",tri[i],(float)verts("minv",tri[i]),(float)verts("m",tri[i]));
                                    beta = verts("minv",tri[i]) * tri_bary[i] / tri_cminv;
                                }
                                // atomic_add(exec_tag,&impulse_count[tri[i]],1);
                                for(int d = 0;d != 3;++d)
                                    atomic_add(exec_tag,&verts(gradOffset + d,tri[i]),-impulse[d] * beta);             
                            }
                    });
                    timer.tock("assemble self icm gradient");
                }

            }

            // std::cout << "has_self_intersection : " << has_self_intersection << "\t" << "has_kine_intersection : " << has_kine_intersection << std::endl;

            if(!has_self_intersection && !has_kine_intersection)
                break;

            timer.tick();

            auto filter_the_update = get_input2<bool>("filter_the_update");

            auto gradn = TILEVEC_OPS::dot<3>(cudaExec,verts,"grad","grad");
            std::cout << "apply gradient : " << gradn <<  std::endl;
            cudaExec(zs::range(verts.size()),[
                impulse_count = proxy<space>(impulse_count),
                eps = eps,
                filter_the_update = filter_the_update,
                h0 = maximum_correction,
                g02 = progressive_slope * progressive_slope,
                xtagOffset = verts.getPropertyOffset(xtag),
                gradOffset = verts.getPropertyOffset("grad"),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {

                    // if(vtemp("w",vi) > eps)
                    //     verts.tuple(dim_c<3>,xtag,vi) = verts.pack(dim_c<3>,xtag,vi) + vtemp.pack(dim_c<3>,"grad",vi) / vtemp("w",vi);
                    auto G = verts.pack(dim_c<3>,gradOffset,vi);
                    // if(G.norm() < eps)
                    //     return;
                    if(filter_the_update) {
                        // if(impulse_count[vi] == 0)
                        //     return;
                        auto Gn = G.norm();
                        auto Gn2 = Gn * Gn;
                        G = h0 * G / zs::sqrt(Gn2 + g02 + 1e-6);
                        // G /= (T)impulse_count[vi];
                    }

                    // if(vi == 15 || vi == 27) {
                    //     printf("verts[%d] update : %f %f %f\n",
                    //         vi,
                    //         (float)verts("m",vi),
                    //         (float)verts("minv",vi),
                    //         (float)G.norm());
                    // }
                    verts.tuple(dim_c<3>,xtagOffset,vi) = verts.pack(dim_c<3>,xtagOffset,vi) + G;
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
        {"float","relaxation_rate","1"},
        {"kboundary"},
        {"int","substep_id","0"},
        {"int","nm_substeps","1"},
        {"bool","refit_kboundary_bvh","1"},
        {"bool","mark_intersection","0"},
        {"bool","use_cur_kine_configuration","1"},
        {"bool","use_barycentric_interpolator","0"},
        {"bool","filter_the_update","1"},
        {"bool","enforce_boundary_normal","0"},
        {"bool","detangle_with_boundary","1"},
        {"bool","do_self_detangle","1"},
        {"bool","skip_animation_intersection","1"}
    },
    {
        {"zsparticles"}
    },
    {
    },
    {"GIA"},
});



struct VisualizeDetanglePair : zeno::INode {
    virtual void apply() override {
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

        if(!zsparticles->hasMeta(GIA::GIA_CS_ET_BUFFER_KEY)) {
            zsparticles->setMeta(GIA::GIA_CS_ET_BUFFER_KEY,
                zs::bht<int,2,int>{verts.get_allocator(),GIA::DEFAULT_MAX_GIA_INTERSECTION_PAIR});
        }
        auto& csET = zsparticles->readMeta<zs::bht<int,2,int> &>(GIA::GIA_CS_ET_BUFFER_KEY);  
        
        auto has_bvh = zsparticles->hasBvh(GIA::GIA_TRI_BVH_BUFFER_KEY);
        if(!zsparticles->hasBvh(GIA::GIA_TRI_BVH_BUFFER_KEY))
            zsparticles->bvh(GIA::GIA_TRI_BVH_BUFFER_KEY) = LBvh<3,int,T>{};
        auto& tri_bvh = zsparticles->bvh(GIA::GIA_TRI_BVH_BUFFER_KEY);   

        if(!zsparticles->hasAuxData(GIA::GIA_IMC_GRAD_BUFFER_KEY)) {
            (*zsparticles)[GIA::GIA_IMC_GRAD_BUFFER_KEY] = dtiles_t{verts.get_allocator(),{
                {"grad",3},{"bary",4}
            },GIA::DEFAULT_MAX_GIA_INTERSECTION_PAIR};
        }


        dtiles_t self_icm_grad{verts.get_allocator(),{
            {"grad",3},{"bary",4},{"inds",2}},10000};
        dtiles_t ket_icm_grad{verts.get_allocator(),{
            {"grad",3},{"bary",4},{"inds",2}},10000};            
        dtiles_t ekt_icm_grad{verts.get_allocator(),{
            {"grad",3},{"bary",4},{"inds",2}},10000};            

        auto use_barycentric_interpolator = get_input2<bool>("use_barycentric_interpolator");
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
        
        LBvh<3,int,T> ktri_bvh{};        

        auto detangle_with_boundary = get_input2<bool>("detangle_with_boundary");
        auto do_self_detangle = get_input2<bool>("do_self_detangle");

        if(has_input<ZenoParticles>("kboundary") && detangle_with_boundary) {
            auto kboundary = get_input<ZenoParticles>("kboundary");
            auto substep_id = get_input2<int>("substep_id");
            auto nm_substeps = get_input2<int>("nm_substeps");
            auto w = (float)(substep_id + 1) / (float)nm_substeps;
            auto pw = (float)(substep_id) / (float)nm_substeps;  
            const auto& kverts = kboundary->getParticles();
            const auto& kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];
            const auto& ktris = kboundary->getQuadraturePoints();  

            kvtemp.resize(kverts.size());
            auto kxtag = "x";
            auto kpxtag = kverts.hasProperty("px") ? "px" : "x";
            auto use_cur_kine_configuration = get_input2<bool>("use_cur_kine_configuration");
            auto alpha = use_cur_kine_configuration ? w : pw;
            cudaExec(zs::range(kverts.size()),[
                alpha = alpha,
                kxtag = zs::SmallString(kxtag),
                kpxtag = zs::SmallString(kpxtag),
                kverts = proxy<space>({},kverts),
                kvtemp = proxy<space>({},kvtemp)] ZS_LAMBDA(int kvi) mutable {
                    auto kvert = kverts.pack(dim_c<3>,kpxtag,kvi) * (1 - alpha) + kverts.pack(dim_c<3>,kxtag,kvi) * alpha;
                    kvtemp.tuple(dim_c<3>,"x",kvi) = kvert;
            });
            kttemp.resize(ktris.size());    
            TILEVEC_OPS::copy<3>(cudaExec,ktris,"inds",kttemp,"inds");
            cudaExec(zs::range(kttemp.size()),[
                kttemp = proxy<space>({},kttemp),
                kvtemp = proxy<space>({},kvtemp)] ZS_LAMBDA(int kti) mutable {
                    auto ktri = kttemp.pack(dim_c<3>,"inds",kti,int_c);
                    zs::vec<T,3> ktvs[3] = {};
                    for(int i = 0;i != 3;++i)
                        ktvs[i] = kvtemp.pack(dim_c<3>,"x",ktri[i]);
                    kttemp.tuple(dim_c<3>,"nrm",kti) = LSL_GEO::facet_normal(ktvs[0],ktvs[1],ktvs[2]);
                    kttemp("d",kti) = -kttemp.pack(dim_c<3>,"nrm",kti).dot(ktvs[0]);
            });         


            auto kbvs = retrieve_bounding_volumes(cudaExec,kvtemp,ktris,wrapv<3>{},(T)0,"x");
            ktri_bvh.build(cudaExec,kbvs);  
        }
            
        auto tri_bvs = retrieve_bounding_volumes(cudaExec,verts,tris,wrapv<3>{},0,xtag);
        tri_bvh.build(cudaExec,tri_bvs);   

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


        size_t nm_ket_intersections = 0;
        size_t nm_ekt_intersections = 0;
        size_t nm_self_intersections = 0;


        dtiles_t ekt_verts_buffer{verts.get_allocator(),{
            {"xe",3},{"xt",3}
        },nm_ekt_intersections};

        dtiles_t ket_verts_buffer{verts.get_allocator(),{
            {"xe",3},{"xt",3}
        },nm_ekt_intersections};

        dtiles_t set_verts_buffer{verts.get_allocator(),{
            {"xe",3},{"xt",3}
        },nm_self_intersections};


        if(has_input<ZenoParticles>("kboundary") && detangle_with_boundary) {
            auto kboundary = get_input<ZenoParticles>("kboundary");
            const auto& kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];
            const auto& ktris = kboundary->getQuadraturePoints();
            // const auto& ktri_bvh = kboundary->bvh(GIA::GIA_TRI_BVH_BUFFER_KEY);
            const auto& khalfedges = (*kboundary)[ZenoParticles::s_surfHalfEdgeTag];
            {
                GIA::retrieve_intersection_with_edge_tri_pairs(cudaExec,
                    verts,xtag,
                    edges,
                    kvtemp,"x",
                    kttemp,
                    ktri_bvh,
                    csET,
                    ekt_icm_grad,
                    use_barycentric_interpolator);
                nm_ekt_intersections = csET.size();

                ekt_verts_buffer.resize(nm_ekt_intersections);
                cudaExec(zs::range(nm_ekt_intersections),[
                    verts = proxy<space>({},verts),
                    xtag = zs::SmallString(xtag),
                    edges = proxy<space>({},edges),
                    use_barycentric_interpolator = use_barycentric_interpolator,
                    ints_buffer = proxy<space>({},ekt_icm_grad),
                    et_verts_buffer = proxy<space>({},ekt_verts_buffer),
                    kvtemp = proxy<space>({},kvtemp),
                    kttemp = proxy<space>({},kttemp)] ZS_LAMBDA(int ci) mutable {
                        auto pair = ints_buffer.pack(dim_c<2>,"inds",ci,int_c);
                        auto bary = ints_buffer.pack(dim_c<4>,"bary",ci);
                        auto ei = pair[0];
                        auto kti = pair[1];

                        auto edge = edges.pack(dim_c<2>,"inds",ei,int_c);
                        auto ktri = kttemp.pack(dim_c<3>,"inds",kti,int_c);

                        zs::vec<T,2> edge_bary{bary[0],1 - bary[0]};
                        zs::vec<T,3> ktri_bary{bary[1],bary[2],bary[3]};

                        if(!use_barycentric_interpolator) {
                            edge_bary = zs::vec<T,2>::uniform((T)0.5);
                            ktri_bary = zs::vec<T,3>::uniform((T)1 / (T)3);
                        }

                        zs::vec<T,3> eps[2] = {};
                        zs::vec<T,3> ktps[3] = {};

                        for(int i = 0;i != 2;++i)
                            eps[i] = verts.pack(dim_c<3>,xtag,edge[i]);

                        for(int i = 0;i != 3;++i)
                            ktps[i] = kvtemp.pack(dim_c<3>,"x",ktri[i]);

                        auto xe = zs::vec<T,3>::zeros();
                        auto xt = zs::vec<T,3>::zeros();

                        for(int i = 0;i != 2;++i)
                            xe += edge_bary[i] * eps[i];
                        for(int i = 0;i != 3;++i)
                            xt += ktri_bary[i] * ktps[i];
                        
                        et_verts_buffer.tuple(dim_c<3>,"xe",ci) = xe;
                        et_verts_buffer.tuple(dim_c<3>,"xt",ci) = xt;
                });
            }

            {
                GIA::retrieve_intersection_with_edge_tri_pairs(cudaExec,
                    kvtemp,"x",
                    kedges,
                    verts,xtag,
                    tris,
                    tri_bvh,
                    csET,
                    ket_icm_grad,
                    use_barycentric_interpolator);
                nm_ket_intersections = csET.size();

                ket_verts_buffer.resize(nm_ket_intersections);
                cudaExec(zs::range(nm_ket_intersections),[
                    verts = proxy<space>({},verts),
                    xtag = zs::SmallString(xtag),
                    tris = proxy<space>({},tris),
                    use_barycentric_interpolator = use_barycentric_interpolator,
                    ints_buffer = proxy<space>({},ket_icm_grad),
                    et_verts_buffer = proxy<space>({},ket_verts_buffer),
                    kvtemp = proxy<space>({},kvtemp),
                    kedges = proxy<space>({},kedges)] ZS_LAMBDA(int ci) mutable {
                        auto pair = ints_buffer.pack(dim_c<2>,"inds",ci,int_c);
                        auto bary = ints_buffer.pack(dim_c<4>,"bary",ci);
                        auto kei = pair[0];
                        auto ti = pair[1];

                        auto kedge = kedges.pack(dim_c<2>,"inds",kei,int_c);
                        auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                        zs::vec<T,2> kedge_bary{bary[0],1 - bary[0]};
                        zs::vec<T,3> tri_bary{bary[1],bary[2],bary[3]};

                        if(!use_barycentric_interpolator) {
                            kedge_bary = zs::vec<T,2>::uniform((T)0.5);
                            tri_bary = zs::vec<T,3>::uniform((T)1 / (T)3);
                        }

                        zs::vec<T,3> keps[2] = {};
                        zs::vec<T,3> tps[3] = {};

                        for(int i = 0;i != 2;++i)
                            keps[i] = kvtemp.pack(dim_c<3>,"x",kedge[i]);

                        for(int i = 0;i != 3;++i)
                            tps[i] = verts.pack(dim_c<3>,xtag,tri[i]);

                        auto xe = zs::vec<T,3>::zeros();
                        auto xt = zs::vec<T,3>::zeros();

                        for(int i = 0;i != 2;++i)
                            xe += kedge_bary[i] * keps[i];
                        for(int i = 0;i != 3;++i)
                            xt += tri_bary[i] * tps[i];
                        
                        et_verts_buffer.tuple(dim_c<3>,"xe",ci) = xe;
                        et_verts_buffer.tuple(dim_c<3>,"xt",ci) = xt;
                });

            }            
            // std::cout << "do kinematic boundary detangle with " << nm_kinematic_intersection << " intersections" << std::endl;
        }

        if(do_self_detangle) {
            GIA::retrieve_self_intersection_tri_edge_pairs(cudaExec,
                verts,xtag,
                tris,
                edges,
                tri_bvh,
                csET,
                self_icm_grad,
                use_barycentric_interpolator);     
            nm_self_intersections = csET.size(); 
            set_verts_buffer.resize(nm_self_intersections);

            cudaExec(zs::range(nm_self_intersections),[
                verts = proxy<space>({},verts),
                xtag = zs::SmallString(xtag),
                edges = proxy<space>({},edges),
                use_barycentric_interpolator = use_barycentric_interpolator,
                tris = proxy<space>({},tris),
                ints_buffer = proxy<space>({},self_icm_grad),
                et_verts_buffer = proxy<space>({},set_verts_buffer)] ZS_LAMBDA(int ci) mutable {
                    auto pair = ints_buffer.pack(dim_c<2>,"inds",ci,int_c);
                    auto bary = ints_buffer.pack(dim_c<4>,"bary",ci);
                    auto ei = pair[0];
                    auto ti = pair[1];

                    auto edge = edges.pack(dim_c<2>,"inds",ei,int_c);
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                    zs::vec<T,2> edge_bary{bary[0],1 - bary[0]};
                    zs::vec<T,3> tri_bary{bary[1],bary[2],bary[3]};

                    if(!use_barycentric_interpolator) {
                        edge_bary = zs::vec<T,2>::uniform((T)0.5);
                        tri_bary = zs::vec<T,3>::uniform((T)1 / (T)3);
                    }

                    zs::vec<T,3> eps[2] = {};
                    zs::vec<T,3> tps[3] = {};

                    for(int i = 0;i != 2;++i)
                        eps[i] = verts.pack(dim_c<3>,xtag,edge[i]);

                    for(int i = 0;i != 3;++i)
                        tps[i] = verts.pack(dim_c<3>,xtag,tri[i]);

                    auto xe = zs::vec<T,3>::zeros();
                    auto xt = zs::vec<T,3>::zeros();

                    for(int i = 0;i != 2;++i)
                        xe += edge_bary[i] * eps[i];
                    for(int i = 0;i != 3;++i)
                        xt += tri_bary[i] * tps[i];
                    
                    et_verts_buffer.tuple(dim_c<3>,"xe",ci) = xe;
                    et_verts_buffer.tuple(dim_c<3>,"xt",ci) = xt;
            });
        }

        ekt_verts_buffer = ekt_verts_buffer.clone({memsrc_e::host});
        ket_verts_buffer = ket_verts_buffer.clone({memsrc_e::host});
        set_verts_buffer = set_verts_buffer.clone({memsrc_e::host});

        auto ompPol = omp_exec();
        constexpr auto omp_space = execspace_e::openmp;

        auto ekt_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& ekt_verts = ekt_vis->verts;
        auto& ekt_lines = ekt_vis->lines;
        ekt_verts.resize(nm_ekt_intersections * 2);
        ekt_lines.resize(nm_ekt_intersections);

        ompPol(zs::range(nm_ekt_intersections),[
            &ekt_verts,&ekt_lines,
            verts_buffer = proxy<omp_space>({},ekt_verts_buffer)] (int ci) mutable {
                auto xe = verts_buffer.pack(dim_c<3>,"xe",ci);
                auto xt = verts_buffer.pack(dim_c<3>,"xt",ci);

                ekt_verts[ci * 2 + 0] = xe.to_array();
                ekt_verts[ci * 2 + 1] = xt.to_array();
                ekt_lines[ci] = zeno::vec2i{ci * 2 + 0,ci * 2 + 1};
        });

        auto ket_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& ket_verts = ket_vis->verts;
        auto& ket_lines = ket_vis->lines;
        ket_verts.resize(nm_ket_intersections * 2);
        ket_lines.resize(nm_ket_intersections);

        ompPol(zs::range(nm_ket_intersections),[
            &ket_verts,&ket_lines,
            verts_buffer = proxy<omp_space>({},ket_verts_buffer)] (int ci) mutable {
                auto xe = verts_buffer.pack(dim_c<3>,"xe",ci);
                auto xt = verts_buffer.pack(dim_c<3>,"xt",ci);

                ket_verts[ci * 2 + 0] = xe.to_array();
                ket_verts[ci * 2 + 1] = xt.to_array();
                ket_lines[ci] = zeno::vec2i{ci * 2 + 0,ci * 2 + 1};
        });        

        auto set_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& set_verts = set_vis->verts;
        auto& set_lines = set_vis->lines;
        set_verts.resize(nm_self_intersections * 2);
        set_lines.resize(nm_self_intersections);

        ompPol(zs::range(nm_self_intersections),[
            &set_verts,&set_lines,
            verts_buffer = proxy<omp_space>({},set_verts_buffer)] (int ci) mutable {
                auto xe = verts_buffer.pack(dim_c<3>,"xe",ci);
                auto xt = verts_buffer.pack(dim_c<3>,"xt",ci);

                // printf("xe[%d] : %f %f %f\t xt[%d] : %f %f %f\n",
                //     ci,(float)xe[0],(float)xe[1],(float)xe[2],
                //     ci,(float)xt[0],(float)xt[1],(float)xt[2]);

                set_verts[ci * 2 + 0] = xe.to_array();
                set_verts[ci * 2 + 1] = xt.to_array();
                set_lines[ci] = zeno::vec2i{ci * 2 + 0,ci * 2 + 1};
        });    
        

        set_output("ekt_vis",std::move(ekt_vis));
        set_output("ket_vis",std::move(ket_vis));
        set_output("set_vis",std::move(set_vis));
    }
};


ZENDEFNODE(VisualizeDetanglePair, {
    {
        {"zsparticles"},
        {"string", "xtag", "x"},
        {"kboundary"},
        {"int","substep_id","0"},
        {"int","nm_substeps","1"},
        {"bool","use_cur_kine_configuration","1"},
        {"bool","use_barycentric_interpolator","0"},
        {"bool","detangle_with_boundary","1"},
        {"bool","do_self_detangle","1"},
        // {"bool","skip_animation_intersection","1"}
    },
    {
        {"ekt_vis"},{"ket_vis"},{"set_vis"}
    },
    {
    },
    {"GIA"},
});



};