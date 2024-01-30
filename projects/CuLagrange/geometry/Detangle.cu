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

#include "kernel/intersection.hpp"
#include "kernel/intersection_contour_minimization.hpp"


namespace zeno {

// #define TIMING_DETANGLE

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

        constexpr auto DETANGLE_CS_ET_BUFFER_KEY = "DETANGLE_CS_ET_BUFFER_KEY";
        constexpr auto DETANGLE_CS_EKT_BUFFER_KEY = "DETANGLE_CS_EKT_BUFFER_KEY";
        constexpr auto DETANGLE_CS_KET_BUFFER_KEY = "DETANGLE_CS_KET_BUFFER_KEY";
        constexpr auto DETANGLE_TRI_BVH_BUFFER_KEY = "DETANGLE_TRI_BVH_BUFFER_KEY";
        constexpr auto DETANGLE_ICM_GRADIENT_BUFFER_KEY = "DETANGLE_ICM_GRADIENT_BUFFER_KEY";
        constexpr auto DEFAULT_MAX_DETANGLE_INTERSECTION_PAIR = 10000;

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

        auto collision_group_name = get_input2<std::string>("collision_group");

        if(!zsparticles->hasMeta(DETANGLE_CS_ET_BUFFER_KEY)) {
            zsparticles->setMeta(DETANGLE_CS_ET_BUFFER_KEY,
                zs::bht<int,2,int>{verts.get_allocator(),DEFAULT_MAX_DETANGLE_INTERSECTION_PAIR});
        }
        if(!zsparticles->hasMeta(DETANGLE_CS_EKT_BUFFER_KEY)) {
            zsparticles->setMeta(DETANGLE_CS_EKT_BUFFER_KEY,
                zs::bht<int,2,int>{verts.get_allocator(),DEFAULT_MAX_DETANGLE_INTERSECTION_PAIR});            
        }
        if(!zsparticles->hasMeta(DETANGLE_CS_KET_BUFFER_KEY)) {
            zsparticles->setMeta(DETANGLE_CS_KET_BUFFER_KEY,
                zs::bht<int,2,int>{verts.get_allocator(),DEFAULT_MAX_DETANGLE_INTERSECTION_PAIR});            
        }

        auto& csET = zsparticles->readMeta<zs::bht<int,2,int> &>(DETANGLE_CS_ET_BUFFER_KEY);
        auto& csEKT = zsparticles->readMeta<zs::bht<int,2,int> &>(DETANGLE_CS_EKT_BUFFER_KEY);
        auto& csKET = zsparticles->readMeta<zs::bht<int,2,int> &>(DETANGLE_CS_KET_BUFFER_KEY);

        auto has_bvh = zsparticles->hasBvh(DETANGLE_TRI_BVH_BUFFER_KEY);
        if(!zsparticles->hasBvh(DETANGLE_TRI_BVH_BUFFER_KEY))
            zsparticles->bvh(DETANGLE_TRI_BVH_BUFFER_KEY) = LBvh<3,int,T>{};
        auto& tri_bvh = zsparticles->bvh(DETANGLE_TRI_BVH_BUFFER_KEY);   

        if(!verts.hasProperty("grad") || verts.getPropertySize("grad") != 3)
            verts.append_channels(cudaExec,{{"grad",3}});
        if(!verts.hasProperty("icm_intersected") || verts.getPropertySize("icm_intersected") != 1)
            verts.append_channels(cudaExec,{{"icm_intersected",1}});
        if(!zsparticles->hasAuxData(DETANGLE_ICM_GRADIENT_BUFFER_KEY)) {
            (*zsparticles)[DETANGLE_ICM_GRADIENT_BUFFER_KEY] = dtiles_t{verts.get_allocator(),{
                {"grad",3},{"bary",4},{"inds",2}
            },DEFAULT_MAX_DETANGLE_INTERSECTION_PAIR};
        }

        auto& icm_grad = (*zsparticles)[DETANGLE_ICM_GRADIENT_BUFFER_KEY];
        TILEVEC_OPS::fill(cudaExec,icm_grad,"grad",(T)0.0);
        auto use_barycentric_interpolator = get_input2<bool>("use_barycentric_interpolator");
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
            {"x",3},
            {"collision_cancel",1},
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
                hasKCollisionCancel = kverts.hasProperty("colllision_cancel"),
                kvtemp = proxy<space>({},kvtemp)] ZS_LAMBDA(int kvi) mutable {
                    auto kvert = kverts.pack(dim_c<3>,kpxtag,kvi) * (1 - alpha) + kverts.pack(dim_c<3>,kxtag,kvi) * alpha;
                    kvtemp.tuple(dim_c<3>,"x",kvi) = kvert;
                    if(hasKCollisionCancel)
                        kvtemp("collision_cancel",kvi) = kverts("collision_cancel",kvi);
                    else
                        kvtemp("collision_cancel",kvi) = 0;
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

                    auto knrm = LSL_GEO::facet_normal(ktvs[0],ktvs[1],ktvs[2]);
                    if(isnan(knrm.norm())) {
                        // printf("nan knrm detected at ktri[%d]\n",kti);
                        kttemp.tuple(dim_c<3>,"nrm",kti) = vec3::zeros();
                        kttemp("d",kti) = 0.;
                    } else {
                        kttemp.tuple(dim_c<3>,"nrm",kti) = knrm;
                        kttemp("d",kti) = -kttemp.pack(dim_c<3>,"nrm",kti).dot(ktvs[0]);
                    }
            });         

            auto kbvs = retrieve_bounding_volumes(cudaExec,kvtemp,ktris,wrapv<3>{},(T)0,"x");
            ktri_bvh.build(cudaExec,kbvs);     
        }

        int nm_intersections = 0;
        int nm_kinematic_intersection = 0;
        if(mark_intersection)
            TILEVEC_OPS::fill(cudaExec,verts,"icm_intersected",(T)0.0);

        auto nm_detangle_restart_iters = get_input2<int>("nm_detangle_restart_iters");

        for(int iter = 0;iter != nm_iters;++iter) {
            auto do_proximity_detection = ((iter % nm_detangle_restart_iters) == 0);
            auto do_proximity_recheck = !do_proximity_detection;
            
            auto refit_tri_bvh = iter > 0;
#ifdef TIMING_DETANGLE
            timer.tick();
#endif

            // if(do_proximity_detection) {
            auto tri_bvs = retrieve_bounding_volumes(cudaExec,verts,tris,wrapv<3>{},0,xtag);

            #ifdef TIMING_DETANGLE
            timer.tock("retrieve_tri_bvh_bounding_volumes");

            timer.tick();
            #endif
            if(refit_tri_bvh)
                tri_bvh.refit(cudaExec,tri_bvs);
            else
                tri_bvh.build(cudaExec,tri_bvs);
            #ifdef TIMING_DETANGLE   
            timer.tock("refit_tri_bvh");      
            #endif      
            // }

            cudaExec(zs::range(impulse_count),[]ZS_LAMBDA(auto& count) mutable {count = 0;});

            TILEVEC_OPS::fill(cudaExec,verts,"grad",(T)0.0);
            // auto use_dirty_bits = iter > 0;
            #ifdef TIMING_DETANGLE
            timer.tick();
            #endif
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
            #ifdef TIMING_DETANGLE
            timer.tock("eval triangle plane");
            #endif

            bool has_kine_intersection = false;
            bool has_self_intersection = false;
            if(has_input<ZenoParticles>("kboundary") && detangle_with_boundary) {
                auto kboundary = get_input<ZenoParticles>("kboundary");
                const auto& kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];
                const auto& ktris = kboundary->getQuadraturePoints();
                const auto& khalfedges = (*kboundary)[ZenoParticles::s_surfHalfEdgeTag];

                auto kine_icm_grad = (T)0;

                {
                    #ifdef TIMING_DETANGLE
                    timer.tick();
                    #endif

                    // std::cout << "retrive_intersections_between_edges_and_ktris" << std::endl;
                    if(do_proximity_detection) {
                        retrieve_intersection_with_edge_tri_pairs(cudaExec,
                            verts,xtag,
                            edges,
                            kvtemp,"x",
                            kttemp,
                            ktri_bvh,
                            csEKT,
                            icm_grad,
                            false);
                    }
                    if(iter == 0)
                        nm_kinematic_intersection += csEKT.size();

                    #ifdef TIMING_DETANGLE
                    timer.tock("retrieve_intersection_with_EKT_pairs");
                    #endif
                    // std::cout << "finish retrive_intersections_between_edges_and_ktris : " << csET.size() << std::endl;

                    if(csEKT.size() > 0)
                        has_kine_intersection = true;


                    // nm_kinematic_intersection += csET.size();
                    auto enforce_boundary_normal = get_input2<bool>("enforce_boundary_normal");

                    #ifdef TIMING_DETANGLE
                    timer.tick();
                    #endif



                    eval_intersection_contour_minimization_gradient_of_edgeA_with_triB(cudaExec,
                        verts,xtag,
                        edges,
                        halfedges,
                        tris,
                        kvtemp,"x",
                        kttemp,
                        maximum_correction,
                        progressive_slope,                        
                        csEKT,
                        icm_grad,
                        enforce_boundary_normal,
                        do_proximity_recheck);     


                    #ifdef TIMING_DETANGLE
                    timer.tock("eval_intersection_contour_minimization_gradient_with_EKT");  
                    timer.tick();
                    #endif

                    cudaExec(zip(zs::range(csEKT.size()),csEKT._activeKeys),[
                        vertsHasM = verts.hasProperty("m"),
                        vertsHasMinv = verts.hasProperty("minv"),
                        impulse_count = proxy<space>(impulse_count),
                        exec_tag = exec_tag,
                        iter = iter,
                        eps = eps,
                        use_barycentric_interpolator = false,
                        mark_intersection = mark_intersection,
                        xtag = zs::SmallString(xtag),
                        icm_grad = proxy<space>({},icm_grad),
                        relaxation_rate = relaxation_rate,
                        gradOffset = verts.getPropertyOffset("grad"),
                        verts = proxy<space>({},verts),
                        edges = proxy<space>({},edges)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                            auto ei = pair[0];
                            auto edge = edges.pack(dim_c<2>,"inds",ei,int_c);
                            if(mark_intersection && iter == 0) {
                                verts("icm_intersected",edge[0]) = (T)1.0;
                                verts("icm_intersected",edge[1]) = (T)1.0;
                            }

                            auto impulse = icm_grad.pack(dim_c<3>,"grad",ci) * relaxation_rate;

                            if(impulse.norm() < eps)
                                return;

                            T edge_cminv = 1;
                            zs::vec<T,2> edge_bary{};

                            zs::vec<T,2> ms{};
                            zs::vec<T,2> minvs{};
                            if(vertsHasM) {         
                                for(int i = 0;i != 2;++i)
                                    ms[i] = verts("m",edge[i]);
                            }else {
                                ms = zs::vec<T,2>::uniform(static_cast<T>(1.0));
                            }

                            if(vertsHasMinv) {
                                for(int i = 0;i != 2;++i)
                                    minvs[i] = verts("minv",edge[i]);
                            }else {
                                minvs = zs::vec<T,2>::uniform(static_cast<T>(1.0));
                            }

                            if(use_barycentric_interpolator) {
                                auto bary = icm_grad.pack(dim_c<4>,"bary",ci);
                                edge_bary[0] = bary[0];
                                edge_bary[1] = 1 - bary[0];

                                edge_cminv = 0;
                                
                                for(int i = 0;i != 2;++i)
                                    edge_cminv += edge_bary[i] * edge_bary[i] / ms[i];
                            }

                            for(int i = 0;i != 2;++i) {
                                T beta = 1;
                                if(use_barycentric_interpolator) {
                                    beta = minvs[i] * edge_bary[i] / edge_cminv;
                                    // printf("edge[%d][%d]_beta : %f\n",ei,edge[i],(float)beta);
                                }
                                atomic_add(exec_tag,&impulse_count[edge[i]],1);
                                for(int d = 0;d != 3;++d)
                                    atomic_add(exec_tag,&verts(gradOffset + d,edge[i]),impulse[d] * beta);
                            }
                    });

                    auto ekt_impulse_norm2 = TILEVEC_OPS::dot<3>(cudaExec,verts,"grad","grad");

                    std::cout << "ekt_impulse_norm2 : " << ekt_impulse_norm2 << std::endl;

                    kine_icm_grad += ekt_impulse_norm2;
                    // std::cout << "EKT IMPULSE : " << ekt_impulse_norm2 << std::endl;
                    #ifdef TIMING_DETANGLE
                    timer.tock("assemble EKT icm gradient");   
                    #endif
                }

                {
                    // std::cout << "retrive_intersections_between_kedges_and_tris" << std::endl;
                    #ifdef TIMING_DETANGLE
                    timer.tick();
                    #endif

                    if(do_proximity_detection) {
                        retrieve_intersection_with_edge_tri_pairs(cudaExec,
                            kvtemp,"x",
                            kedges,
                            verts,xtag,
                            tris,
                            tri_bvh,
                            csKET,
                            icm_grad,
                            false);
                    }
                    #ifdef TIMING_DETANGLE
                    timer.tock("retrieve_intersection_with_KET_pairs");
                    #endif


                    // nm_intersections += csET.size();
                    // std::cout << "finish retrive_intersections_between_kedges_and_tris" << std::endl;

                    if(csKET.size() > 0)
                        has_kine_intersection = true;

                    if(iter == 0)
                        nm_kinematic_intersection += csKET.size();

                    #ifdef TIMING_DETANGLE
                    timer.tick();
                    #endif

                    eval_intersection_contour_minimization_gradient_of_edgeA_with_triB(cudaExec,
                        kvtemp,"x",
                        kedges,
                        khalfedges,
                        kttemp,
                        verts,xtag,
                        tris,
                        maximum_correction,
                        progressive_slope,                        
                        csKET,
                        icm_grad,
                        false,
                        do_proximity_recheck);   
                        
                    #ifdef TIMING_DETANGLE
                    timer.tock("eval_intersection_contour_minimization_gradient_with_KET"); 
                    timer.tick();
                    #endif

                    cudaExec(zip(zs::range(csKET.size()),csKET._activeKeys),[
                        vertsHasM = verts.hasProperty("m"),
                        vertsHasMinv = verts.hasProperty("minv"),
                        exec_tag = exec_tag,
                        impulse_count = proxy<space>(impulse_count),
                        eps = eps,
                        iter = iter,
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
                            if(mark_intersection && iter == 0) {
                                verts("icm_intersected",tri[0]) = (T)1.0;
                                verts("icm_intersected",tri[1]) = (T)1.0;
                                verts("icm_intersected",tri[2]) = (T)1.0;
                            }

                            auto impulse = icm_grad.pack(dim_c<3>,"grad",ci) * relaxation_rate;
                            if(impulse.norm() < eps)
                                return;

                            T tri_cminv = 1;
                            zs::vec<T,3> tri_bary{};

                            zs::vec<T,3> ms{};
                            zs::vec<T,3> minvs{};
                            if(vertsHasM) {         
                                for(int i = 0;i != 3;++i)
                                    ms[i] = verts("m",tri[i]);
                            }else {
                                ms = zs::vec<T,3>::uniform(static_cast<T>(1.0));
                            }

                            if(vertsHasMinv) {
                                for(int i = 0;i != 3;++i)
                                    minvs[i] = verts("minv",tri[i]);
                            }else {
                                minvs = zs::vec<T,3>::uniform(static_cast<T>(1.0));
                            }


                            if(use_barycentric_interpolator) {
                                auto bary = icm_grad.pack(dim_c<4>,"bary",ci);
                                tri_bary[0] = bary[1];
                                tri_bary[1] = bary[2];
                                tri_bary[2] = bary[3];

                                tri_cminv = 0;
                                for(int i = 0;i != 3;++i)
                                    tri_cminv += tri_bary[i] * tri_bary[i] / ms[i];
                            }


                            for(int i = 0;i != 3;++i) {
                                T beta = 1;
                                if(use_barycentric_interpolator) {
                                    beta = minvs[i] * tri_bary[i] / tri_cminv;
                                }
                                atomic_add(exec_tag,&impulse_count[tri[i]],1);
                                for(int d = 0;d != 3;++d)
                                    atomic_add(exec_tag,&verts(gradOffset + d,tri[i]),-impulse[d] * beta);
                            }
                    });

                    auto ket_impulse_norm2 = TILEVEC_OPS::dot<3>(cudaExec,verts,"grad","grad");
                    
                    std::cout << "ket_impulse_norm2 : " << ket_impulse_norm2 << std::endl;

                    kine_icm_grad += ket_impulse_norm2;

                    #ifdef TIMING_DETANGLE
                    timer.tock("assemble KET icm gradient");   
                    #endif   
                }
                
                kine_icm_grad = kine_icm_grad;
                if(kine_icm_grad < 1e-5)
                    has_kine_intersection = false;
            }

            if(do_self_detangle) {
                // std::cout << "do self detangle" << std::endl;
                auto skip_too_close_pair_at_rest_shape = get_input2<bool>("skip_too_close_intersections_at_rest");
                auto use_collision_group = true;
                
                auto skip_distance = get_input2<float>("skip_distance");


                if(do_proximity_detection) {
                    std::cout << "do_self_detangle_detection" << std::endl;
                    retrieve_self_intersection_tri_edge_pairs(cudaExec,
                        verts,xtag,collision_group_name,
                        tris,
                        edges,
                        tri_bvh,
                        csET,
                        icm_grad,
                        skip_distance,
                        false,
                        skip_too_close_pair_at_rest_shape,
                        use_collision_group); 
                }

                std::cout << "nm_self_intersections_ET : " << csET.size() << std::endl;

                has_self_intersection = csET.size() > 0;
                if(iter == 0) {
                    nm_intersections += csET.size();
                }


                if(has_self_intersection) {

                    auto enforce_self_intersection_normal = get_input2<bool>("enforce_self_intersection_normal");
                    timer.tick();
                    eval_self_intersection_contour_minimization_gradient(cudaExec,
                        verts,xtag,
                        edges,
                        halfedges,
                        tris,
                        maximum_correction,
                        progressive_slope,
                        csET,
                        icm_grad,
                        enforce_self_intersection_normal,
                        do_proximity_recheck);   

                    // if(iter == 0)
                    //     nm_intersections += csET.size();


                        
                    #ifdef TIMING_DETANGLE
                    timer.tock("eval_self_intersection_contour_minimization_gradient"); 
                    #endif 
                    
                    auto icm_gradn = TILEVEC_OPS::dot<3>(cudaExec,icm_grad,"grad","grad",csET.size());

                    std::cout << "self_impulse_norm2 : " << icm_gradn << std::endl;

                    if(icm_gradn < 1e-5)
                        has_self_intersection = false;
                    else {
                        timer.tick();
                        cudaExec(zip(zs::range(csET.size()),csET._activeKeys),[
                            vertsHasM = verts.hasProperty("m"),
                            vertsHasMinv = verts.hasProperty("minv"),
                            impulse_count = proxy<space>(impulse_count),
                            exec_tag = exec_tag,
                            mark_intersection = mark_intersection,
                            use_barycentric_interpolator = false,
                            eps = eps,
                            iter = iter,
                            relaxation_rate = relaxation_rate,
                            h0 = maximum_correction,
                            g02 = progressive_slope * progressive_slope,
                            icm_grad = proxy<space>({},icm_grad),
                            verts = proxy<space>({},verts),
                            gradOffset = verts.getPropertyOffset("grad"),
                            edges = edges.begin("inds", dim_c<2>, int_c),
                            tris = tris.begin("inds", dim_c<3>, int_c)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                                auto edge = edges[pair[0]];
                                auto tri = tris[pair[1]];

                                if(mark_intersection && iter == 0) {
                                    verts("icm_intersected",tri[0]) = (T)1.0;
                                    verts("icm_intersected",tri[1]) = (T)1.0;
                                    verts("icm_intersected",tri[2]) = (T)1.0;

                                    verts("icm_intersected",edge[0]) = (T)1.0;
                                    verts("icm_intersected",edge[1]) = (T)1.0;                                
                                }

                                zs::vec<T,2> edge_ms{};
                                zs::vec<T,2> edge_minvs{};
                                if(vertsHasM) {         
                                    for(int i = 0;i != 2;++i)
                                        edge_ms[i] = verts("m",edge[i]);
                                }else {
                                    edge_ms = zs::vec<T,2>::uniform(static_cast<T>(1.0));
                                }

                                if(vertsHasMinv) {
                                    for(int i = 0;i != 2;++i)
                                        edge_minvs[i] = verts("minv",edge[i]);
                                }else {
                                    edge_minvs = zs::vec<T,2>::uniform(static_cast<T>(1.0));
                                }

                                zs::vec<T,3> tri_ms{};
                                zs::vec<T,3> tri_minvs{};
                                if(vertsHasM) {         
                                    for(int i = 0;i != 3;++i)
                                        tri_ms[i] = verts("m",tri[i]);
                                }else {
                                    tri_ms = zs::vec<T,3>::uniform(static_cast<T>(1.0));
                                }

                                if(vertsHasMinv) {
                                    for(int i = 0;i != 3;++i)
                                        tri_minvs[i] = verts("minv",tri[i]);
                                }else {
                                    tri_minvs = zs::vec<T,3>::uniform(static_cast<T>(1.0));
                                }

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
                                        tri_cminv += tri_bary[i] * tri_bary[i] / tri_ms[i];
                                    
                                    for(int i = 0;i != 2;++i)
                                        edge_cminv += edge_bary[i] * edge_bary[i] / edge_ms[i];
                                }


                                for(int i = 0;i != 2;++i) { 
                                    T beta = 1;
                                    if(use_barycentric_interpolator) {
                                        beta = edge_minvs[i] * edge_bary[i] / edge_cminv;
                                    }
                                    for(int d = 0;d != 3;++d)
                                        atomic_add(exec_tag,&verts(gradOffset + d,edge[i]),impulse[d]  * beta);
                                }

                                for(int i = 0;i != 3;++i) {
                                    T beta = 1;
                                    if(use_barycentric_interpolator) {
                                        beta = tri_minvs[i] * tri_bary[i] / tri_cminv;
                                    }
                                    for(int d = 0;d != 3;++d)
                                        atomic_add(exec_tag,&verts(gradOffset + d,tri[i]),-impulse[d] * beta);             
                                }
                        });
                        #ifdef TIMING_DETANGLE
                        timer.tock("assemble self icm gradient");
                        #endif
                    }
                }
            }

            if(iter == 0) {
                std::cout << "nm_intersections : " << nm_intersections << std::endl;
                std::cout << "nm_kin_intersections : " << nm_kinematic_intersection << std::endl;
            }

            auto gradInfNorm = TILEVEC_OPS::inf_norm<3>(cudaExec,verts,"grad");
            if(gradInfNorm < 1e-3)
                break;

            if(!has_self_intersection && !has_kine_intersection)
                break;

            #ifdef TIMING_DETANGLE
            timer.tick();
            #endif

            auto filter_the_update = get_input2<bool>("filter_the_update");

            auto gradn = TILEVEC_OPS::dot<3>(cudaExec,verts,"grad","grad");
            // std::cout << "apply gradient : " << gradn <<  std::endl;
            cudaExec(zs::range(verts.size()),[
                impulse_count = proxy<space>(impulse_count),
                eps = eps,
                filter_the_update = filter_the_update,
                h0 = maximum_correction,
                g02 = progressive_slope * progressive_slope,
                xtagOffset = verts.getPropertyOffset(xtag),
                gradOffset = verts.getPropertyOffset("grad"),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
                    auto G = verts.pack(dim_c<3>,gradOffset,vi);
                    if(filter_the_update) {
                        auto Gn = G.norm();
                        auto Gn2 = Gn * Gn;
                        G = h0 * G / zs::sqrt(Gn2 + g02 + 1e-6);
                    }
                    verts.tuple(dim_c<3>,xtagOffset,vi) = verts.pack(dim_c<3>,xtagOffset,vi) + G;
            });

            #ifdef TIMING_DETANGLE
            timer.tock("write_back_to_displacement");
            #endif
        }

        set_output("zsparticles",zsparticles);
    }
}; 


ZENDEFNODE(Detangle2, {
    {
        {"zsparticles"},
        {"string", "xtag", "x"},
        {"string","collision_group",""},
        {"bool","skip_too_close_intersections_at_rest","0"},
        {"int","nm_detangle_restart_iters","1"},
        {"float","skip_distance","0"},
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
        {"bool","enforce_self_intersection_normal","0"},
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

};