#pragma once

#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>

#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"


#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bvs.hpp"
#include "zensim/container/Bvtt.hpp"

namespace zeno { namespace COLLISION_UTILS {

using T = float;
using bvh_t = zs::LBvh<3,int,T>;
using bv_t = zs::AABBBox<3, T>;
using vec3 = zs::vec<T, 3>;
using vec4 = zs::vec<T,4>;
using vec4i = zs::vec<int,4>;

/*
    find the self intersection collision pair between triangle cell and vertex,
    the intersection test is not supported with global intersection analysis
    we should instead use an extra filtering technique to support gia, instead of code it  inside the collision detection
    we should exclude the in active pair using a filter
*/
template<typename Pol,
            typename PosTileVec,
            typename SurfPointTileVec,
            typename SurfTriTileVec,
            typename HalfEdgeTileVec> 
inline void do_facet_point_collision_detection(Pol& pol,
    const PosTileVec& verts,const zs::SmallString& xtag,
    const SurfPointTileVec& points,
    const SurfTriTileVec& tris,
    const HalfEdgeTileVec& halfedges,
    zs::Vector<zs::vec<int,4>>& csPT,
    // bool with_global_intersection_analysis,
    int& nm_collisions,T in_collisionEps,T out_collisionEps) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto avgl = compute_average_edge_length(pol,verts,xtag,tris);
        auto bvh_thickness = 3 * avgl;

        auto spBvh = bvh_t{};
        auto bvs = retrieve_bounding_volumes(pol,verts,points,wrapv<1>{},(T)bvh_thickness,xtag);
        spBvh.build(pol,bvs);

        zs::Vector<int> nm_csPT{points.get_allocator(),1};
        nm_csPT.setVal(0);

        pol(zs::range(tris.size()),[in_collisionEps = in_collisionEps,
            out_collisionEps = out_collisionEps,
            verts = proxy<space>({},verts),
            points = proxy<space>({},points),
            tris = proxy<space>({},tris),
            nm_csPT = proxy<space>(nm_csPT),
            xtag = xtag,
            // with_global_intersection_analysis = with_global_intersection_analysis,
            halfedges = proxy<space>({},halfedges),
            csPT = proxy<space>(csPT),
            spBvh = proxy<space>(spBvh),thickness = bvh_thickness] ZS_LAMBDA(int stI) {
                auto tri = tris.pack(dim_c<3>,"inds",stI,int_c);
                if(verts.hasProperty("active"))
                    for(int i = 0;i != 3;++i)
                        if(verts("active",tri[i]) < 1e-6)
                            return;
                
                auto cp = vec3::zeros();
                for(int i = 0;i != 3;++i)
                    cp += verts.pack(dim_c<3>,xtag,tri[i]) / (T)3.0;
                auto bv = bv_t{get_bounding_box(cp - thickness,cp + thickness)};

                vec3 tvs[3] = {};
                for(int i = 0;i != 3;++i)
                    tvs[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                auto tnrm = LSL_GEO::facet_normal(tvs[0],tvs[1],tvs[2]);

                auto hi = zs::reinterpret_bits<int>(tris("he_inds",stI));
                vec3 bnrms[3] = {};
                for(int i = 0;i != 3;++i){

                    auto edge_normal = tnrm;
                    auto opposite_hi = zs::reinterpret_bits<int>(halfedges("opposite_he",hi));
                    if(opposite_hi >= 0){
                        auto nti = zs::reinterpret_bits<int>(halfedges("to_face",opposite_hi));
                        auto ntri = tris.pack(dim_c<3>,"inds",nti,int_c);
                        auto ntnrm = LSL_GEO::facet_normal(
                            verts.pack(dim_c<3>,xtag,ntri[0]),
                            verts.pack(dim_c<3>,xtag,ntri[1]),
                            verts.pack(dim_c<3>,xtag,ntri[2]));
                        edge_normal = tnrm + ntnrm;
                        edge_normal = edge_normal/(edge_normal.norm() + (T)1e-6);
                    }
                    auto e01 = tvs[(i + 1) % 3] - tvs[(i + 0) % 3];
                    bnrms[i] = edge_normal.cross(e01).normalized();

                    hi = zs::reinterpret_bits<int>(halfedges("next_he",hi));
                }  

                T min_penertration_depth = 1e8;
                int min_vi = -1;

                auto process_vertex_face_collision_pairs = [&](int spI) {
                    auto vi = reinterpret_bits<int>(points("inds",spI));
                    if(tri[0] == vi || tri[1] == vi || tri[2] == vi)
                        return;
                    if(verts.hasProperty("active"))
                        if(verts("active",vi) < 1e-6)
                            return;

                    int embed_tet_id = -1;
                    if(verts.hasProperty("embed_tet_id")) {
                        embed_tet_id = zs::reinterpret_bits<int>(verts("embed_tet_id",vi));
                    }

                    auto p = verts.pack(dim_c<3>,xtag,vi);
                    auto seg = p - tvs[0];
                    auto dist = seg.dot(tnrm);

                    auto collisionEps = dist > 0 ? out_collisionEps : in_collisionEps;

                    auto barySum = (T)1.0;
                    T distance = LSL_GEO::get_vertex_triangle_distance(tvs[0],tvs[1],tvs[2],p,barySum);

                    if(distance > collisionEps)
                        return;

                    if(barySum > (T)(1.0 + 1e-6)) {
                        for(int i = 0;i != 3;++i){
                            seg = p - tvs[i];
                            if(bnrms[i].dot(seg) < 0)
                                return;
                        }
                    }

                    if(dist < 0 && embed_tet_id >= 0)
                        return;

                    if(dist < 0 && distance > min_penertration_depth)
                        return;

                    if(dist > 0) {
                        csPT[atomic_add(exec_cuda,&nm_csPT[0],(int)1)] = zs::vec<int,4>{vi,tri[0],tri[1],tri[2]};
                    }
                };
                spBvh.iter_neighbors(bv,process_vertex_face_collision_pairs);
                if(min_vi >= 0)
                    csPT[atomic_add(exec_cuda,&nm_csPT[0],(int)1)] = zs::vec<int,4>{min_vi,tri[0],tri[1],tri[2]};
        });
        nm_collisions = nm_csPT.getVal(0);
}


// deprecated
template<typename Pol,
    typename REAL,
    typename ZenoParticlePtr,
    typename PosTileVec,
    typename TetTileVec,
    typename SurfPointTileVec,
    typename SurfTriTileVec,
    typename HalfEdgeTileVec,
    typename HashMap>
inline int do_tetrahedra_surface_points_and_kinematic_boundary_collision_detection(Pol& pol,
    ZenoParticlePtr kinematic,
    PosTileVec& tet_verts,const zs::SmallString& tet_pos_attr_tag,
    const TetTileVec& tets,
    const SurfPointTileVec& tet_surf_points,const SurfTriTileVec& tet_surf_tris,
    HalfEdgeTileVec& tet_halfedges,
    REAL out_collisionEps,
    REAL in_collisionEps,
    HashMap& csPT,
    // zs::Vector<int>& csPTOffsets,
    bool collide_from_exterior = true,
    bool write_back_gia_res = false) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};
        using T = typename RM_CVREF_T(tet_verts)::value_type;   

        PosTileVec surf_verts_buffer{tet_surf_points.get_allocator(),{
            {tet_pos_attr_tag,3},
            {"active",1},
            {"nrm",3}
        },tet_surf_points.size()};

        SurfTriTileVec surf_tris_buffer{tet_surf_tris.get_allocator(),{
            {"inds",3},
            {"he_inds",1},   
        },tet_surf_tris.size()};    

        topological_sample(pol,tet_surf_points,tet_verts,tet_pos_attr_tag,surf_verts_buffer);
        if(tet_verts.hasProperty("active")) {
            topological_sample(pol,tet_surf_points,tet_verts,"active",surf_verts_buffer);
        }else {
            TILEVEC_OPS::fill(pol,surf_verts_buffer,"active",(T)1.0);
        }

        TILEVEC_OPS::copy(pol,tet_surf_tris,"inds",surf_tris_buffer,"inds");
        TILEVEC_OPS::copy(pol,tet_surf_tris,"he_inds",surf_tris_buffer,"he_inds");
        reorder_topology(pol,tet_surf_points,surf_tris_buffer); 

        TILEVEC_OPS::fill(pol,surf_verts_buffer,"nrm",(T)0.0);
        pol(zs::range(surf_tris_buffer.size()),[
            surf_verts_buffer = proxy<space>({},surf_verts_buffer),
            tet_pos_attr_tag = tet_pos_attr_tag,
            exec_tag,
            surf_tris_buffer = proxy<space>({},surf_tris_buffer)] ZS_LAMBDA(int ti) mutable {
                auto tri = surf_tris_buffer.pack(dim_c<3>,"inds",ti,int_c);
                zs::vec<T,3> tV[3] = {};
                for(int i = 0;i != 3;++i)
                    tV[i] = surf_verts_buffer.pack(dim_c<3>,tet_pos_attr_tag,tri[i]);
                auto tnrm = LSL_GEO::facet_normal(tV[0],tV[1],tV[2]); 
                for(int i = 0;i != 3;++i)
                    for(int d = 0;d != 3;++d)
                        atomic_add(exec_tag,&surf_verts_buffer("nrm",d,tri[i]),tnrm[d]);
        });
        TILEVEC_OPS::normalized_channel<3>(pol,surf_verts_buffer,"nrm");

        auto tet_tris_cnorm = 2 * compute_average_edge_length(pol,surf_verts_buffer,tet_pos_attr_tag,surf_tris_buffer);            
        auto tetBvh = bvh_t{};
        auto tetBvs = retrieve_bounding_volumes(pol,tet_verts,tets,wrapv<4>{},(T)0,tet_pos_attr_tag);
        tetBvh.build(pol,tetBvs);

        // int k_id = -1;
        csPT.reset(pol,true);

        if(write_back_gia_res) {
            if(!tet_verts.hasProperty("flood"))
                tet_verts.append_channels(pol,{{"flood",1}});
            TILEVEC_OPS::fill(pol,tet_verts,"flood",(T)0.0);
        }


        int kvoffset = 0;
        int ktoffset = 0;

        auto& kverts = kinematic->getParticles();
        const auto& ktris = kinematic->getQuadraturePoints();
        const auto& khalfedges = (*kinematic)[ZenoParticles::s_surfHalfEdgeTag];


        if(write_back_gia_res) {
            if(!kverts.hasProperty("flood"))
                kverts.append_channels(pol,{{"flood",1}});
            TILEVEC_OPS::fill(pol,kverts,"flood",(T)0.0);
        }


        zs::Vector<int> gia_res{surf_verts_buffer.get_allocator(),0};
        zs::Vector<int> tris_gia_res{surf_tris_buffer.get_allocator(),0};


        auto ring_mask_width = do_global_intersection_analysis_with_connected_manifolds(pol,
            surf_verts_buffer,tet_pos_attr_tag,surf_tris_buffer,tet_halfedges,collide_from_exterior,
            kverts,"x",ktris,khalfedges,true,
            gia_res,tris_gia_res);

        // as we only do the intersection detection between kverts and ptris, we only need to flood the two
        if(write_back_gia_res) {
            pol(zs::range(surf_verts_buffer.size()),[
                ring_mask_width = ring_mask_width,
                tet_surf_points = proxy<space>({},tet_surf_points),
                tet_verts = proxy<space>({},tet_verts),
                gia_res = proxy<space>(gia_res)] ZS_LAMBDA(int pi) mutable {
                    auto vi = zs::reinterpret_bits<int>(tet_surf_points("inds",pi));
                    for(int d = 0;d != ring_mask_width;++d) {
                        auto ring_mask = gia_res[pi * ring_mask_width + d];
                        if(ring_mask > 0){
                            tet_verts("flood",vi) = (T)1.0;
                            return;
                        }
                    }
            });
            pol(zs::range(ktris.size()),[
                t_offset = surf_tris_buffer.size(),
                ring_mask_width = ring_mask_width,
                // ktris_buffer = proxy<space>({},ktris_buffer),
                ktris = proxy<space>({},ktris),
                tris_gia_res = proxy<space>(tris_gia_res),
                kverts = proxy<space>({},kverts)] ZS_LAMBDA(int kti) mutable {
                    for(int d = 0;d != ring_mask_width;++d) {
                        auto ring_mask = tris_gia_res[(kti + t_offset) * ring_mask_width + d];
                        if(ring_mask > 0) {
                            auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);
                            for(int i = 0;i != 3;++i)
                                kverts("flood",ktri[i]) = (T)1.0;
                            return;
                        }
                    }
            });
        }
    
        // std::cout << "finish write back gia res" << std::endl;

        auto ktris_cnorm = 2 * compute_average_edge_length(pol,kverts,"x",ktris);
        // auto bvh_thickness = ktris_cnorm > in_collisionEps ? ktris_cnorm : in_collisionEps;
        auto bvh_thickness = ktris_cnorm; 

        auto ktriBvh = bvh_t{};
        auto ktriBvs = retrieve_bounding_volumes(pol,kverts,ktris,wrapv<3>{},(T)0,"x");
        ktriBvh.build(pol,ktriBvs);

        std::cout << "do csPT testing with k_active channel : " << kverts.hasProperty("k_active") << std::endl;

        bool colliding_from_inside = true;
        pol(zs::range(surf_verts_buffer.size()),[
            verts_buffer = proxy<space>({},surf_verts_buffer),
            ktriBvh = proxy<space>(ktriBvh),
            exec_tag = exec_tag,
            ring_mask_width = ring_mask_width,
            colliding_from_inside = colliding_from_inside,
            thickness = bvh_thickness,
            out_collisionEps = out_collisionEps,
            in_collisionEps = in_collisionEps,
            csPT = proxy<space>(csPT),
            ktoffset = ktoffset,
            kt_offset = surf_tris_buffer.size(),
            kv_offset = surf_verts_buffer.size(),
            khalfedges = proxy<space>({},khalfedges),
            kverts = proxy<space>({},kverts),
            ktris = proxy<space>({},ktris),
            pos_attr_tag = zs::SmallString(tet_pos_attr_tag),
            gia_res = proxy<space>(gia_res),
            tris_gia_res = proxy<space>(tris_gia_res)] ZS_LAMBDA(int vi) mutable {
                if(verts_buffer("active",vi) < (T)0.5)
                    return;
                auto p = verts_buffer.pack(dim_c<3>,pos_attr_tag,vi);
                auto bv = bv_t{get_bounding_box(p - thickness,p + thickness)};
                T min_penertration_distance = limits<T>::max();
                int min_kti = -1; 
                auto vnrm = verts_buffer.pack(dim_c<3>,"nrm",vi);

                auto process_vertex_kface_collision_pairs = [&](int kti) mutable {
                    auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);
                    // printf("testing pairs[%d %d] : %f %f %f\n",vi,kti,
                    //     (float)kverts("k_active",ktri[0]),
                    //     (float)kverts("k_active",ktri[1]),
                    //     (float)kverts("k_active",ktri[2]));
                    // if(kverts.hasProperty("k_active"))
                    //     for(int i = 0;i != 3;++i)
                    //         if(kverts("k_active",ktri[i]) < (T)0.5)
                    //             return;
                    vec3 ktvs[3] = {};
                    for(int i = 0;i != 3;++i)
                        ktvs[i] = kverts.pack(dim_c<3>,"x",ktri[i]);
                    auto ktnrm = LSL_GEO::facet_normal(ktvs[0],ktvs[1],ktvs[2]);
                    auto seg = p - ktvs[0];
                    auto dist = ktnrm.dot(seg);

                    if(!colliding_from_inside)
                        dist = -dist;

                    // if(dist < 0 && ktnrm.dot(vnrm) < 0.0)
                    //     return;

                    auto collisionEps = dist > 0 ? out_collisionEps : in_collisionEps;
                    auto barySum = (T)1.0;
                    T distance = LSL_GEO::get_vertex_triangle_distance(ktvs[0],ktvs[1],ktvs[2],p,barySum);       

                    // printf("testing pairs[%d %d] : %f %f\n",vi,kti,(float)distance,(float)collisionEps);

                    if(distance > collisionEps)
                        return;     

                    // printf("testing pairs[%d %d]\n",vi,kti);
                    int RING_MASK = 0;
                    for(int i = 0;i != ring_mask_width;++i) {
                        RING_MASK |= tris_gia_res[(kti + kt_offset) * ring_mask_width + i] & gia_res[vi * ring_mask_width + i];
                    }

                    // int RING_MASK = tris_gia_res[kti + kt_offset] & gia_res[vi];
                    if(dist < 0 && RING_MASK == 0) {
                        // printf("negative distance but ring mask not matched\n");
                        return;
                    }
                    if(dist > 0 && RING_MASK > 0) {
                        // printf("positive distance but ring mask matched\n");
                        return;
                    }

                    auto khi = zs::reinterpret_bits<int>(ktris("he_inds",kti));  
                    vec3 kbnrms[3] = {};
                    for(int i = 0;i != 3;++i) {
                        auto edge_normal = ktnrm;
                        auto opposite_he = zs::reinterpret_bits<int>(khalfedges("opposite_he",khi));
                        if(opposite_he >= 0) {
                            auto nkti = zs::reinterpret_bits<int>(khalfedges("to_face",opposite_he));
                            auto nktri = ktris.pack(dim_c<3>,"inds",nkti,int_c);
                            auto nktnrm = LSL_GEO::facet_normal(
                                kverts.pack(dim_c<3>,"x",nktri[0]),
                                kverts.pack(dim_c<3>,"x",nktri[1]),
                                kverts.pack(dim_c<3>,"x",nktri[2]));
                            edge_normal = ktnrm + nktnrm;
                            edge_normal = edge_normal/(edge_normal.norm() + (T)1e-6);
                        }
                        auto e01 = ktvs[(i + 1) % 3] - ktvs[(i + 0) % 3];
                        kbnrms[i] = edge_normal.cross(e01).normalized();
                        khi = zs::reinterpret_bits<int>(khalfedges("next_he",khi));
                    }   

                    if(barySum > (T)(1.2)) {
                        for(int i = 0;i != 3;++i){
                            seg = p - ktvs[i];
                            if(kbnrms[i].dot(seg) < 0)
                                return;
                        }
                    }   

                    if(dist < 0 && distance < min_penertration_distance){
                        min_penertration_distance = distance;
                        min_kti = kti;
                    }


                    if(dist > 0)
                        csPT.insert(zs::vec<int,2>{vi,kti + ktoffset});
                };
                ktriBvh.iter_neighbors(bv,process_vertex_kface_collision_pairs);

                if(min_kti >= 0)
                    csPT.insert(zs::vec<int,2>{vi,min_kti + ktoffset});
        });

        return csPT.size();
}

// TRI and Kpoint
template<typename Pol,
    typename REAL,
    typename PosTileVec,
    typename TetTileVec,
    typename SurfPointTileVec,
    typename SurfTriTileVec,
    typename HalfEdgeTileVec,
    typename HashMap>
inline int do_tetrahedra_surface_mesh_and_kinematic_boundary_collision_detection(Pol& pol,
    const std::vector<ZenoParticles*>& kinematics,
    PosTileVec& tet_verts,const zs::SmallString& tet_pos_attr_tag,
    const TetTileVec& tets,
    const SurfPointTileVec& tet_surf_points,const SurfTriTileVec& tet_surf_tris,
    HalfEdgeTileVec& tet_halfedges,
    REAL out_collisionEps,
    REAL in_collisionEps,
    HashMap& csPT,
    zs::Vector<int>& csPTOffsets,
    bool write_back_gia_res = false) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};
        using T = typename RM_CVREF_T(tet_verts)::value_type;

        PosTileVec surf_verts_buffer{tet_surf_points.get_allocator(),{
            {tet_pos_attr_tag,3},
            {"active",1}
        },tet_surf_points.size()};

        SurfTriTileVec surf_tris_buffer{tet_surf_tris.get_allocator(),{
            {"inds",3},
            {"he_inds",1},      
        },tet_surf_tris.size()};       

        topological_sample(pol,tet_surf_points,tet_verts,tet_pos_attr_tag,surf_verts_buffer);
        if(tet_verts.hasProperty("active")) {
            topological_sample(pol,tet_surf_points,tet_verts,"active",surf_verts_buffer);
        }else {
            TILEVEC_OPS::fill(pol,surf_verts_buffer,"active",(T)1.0);
        }

        TILEVEC_OPS::copy(pol,tet_surf_tris,"inds",surf_tris_buffer,"inds");
        TILEVEC_OPS::copy(pol,tet_surf_tris,"he_inds",surf_tris_buffer,"he_inds");
        reorder_topology(pol,tet_surf_points,surf_tris_buffer);  

        auto tet_tris_cnorm = 2 * compute_average_edge_length(pol,surf_verts_buffer,tet_pos_attr_tag,surf_tris_buffer);            
        auto tetBvh = bvh_t{};
        auto tetBvs = retrieve_bounding_volumes(pol,tet_verts,tets,wrapv<4>{},(T)0,tet_pos_attr_tag);
        tetBvh.build(pol,tetBvs);

        auto triBvh = bvh_t{};
        auto triBvs = retrieve_bounding_volumes(pol,tet_verts,tet_surf_tris,wrapv<3>{},(T)0,tet_pos_attr_tag);
        triBvh.build(pol,triBvs);

        int k_id = -1;
        csPT.reset(pol,true);

        if(write_back_gia_res) {
            if(!tet_verts.hasProperty("flood"))
                tet_verts.append_channels(pol,{{"flood",1}});
            TILEVEC_OPS::fill(pol,tet_verts,"flood",(T)0.0);
        }

        int kvoffset = 0;
        auto kinematic = kinematics[0];
        auto& kverts = kinematic->getParticles();
        const auto& ktris = kinematic->getQuadraturePoints();
        const auto& khalfedges = (*kinematic)[ZenoParticles::s_surfHalfEdgeTag];

        zs::Vector<int> gia_res{kverts.get_allocator(),surf_verts_buffer.size() + kverts.size()};
        zs::Vector<int> tris_gia_res{ktris.get_allocator(),surf_tris_buffer.size() + ktris.size()};

        SurfPointTileVec kverts_buffer{kverts.get_allocator(),{
            {"x",3},
            // {"ring_mask",1},
            {"active",1}
            // {"embed_tet_id",1},
            // {"mustExclude",1}
        },kverts.size()};
    
        // initialize the data
        {
            TILEVEC_OPS::copy(pol,kverts,"x",kverts_buffer,"x");
            // TILEVEC_OPS::fill(pol,kverts,"ring_mask",zs::reinterpret_bits<T>((int)0));
            if(kverts.hasProperty("active"))
                TILEVEC_OPS::copy(pol,kverts,"active",kverts_buffer,"active");
            else
                TILEVEC_OPS::fill(pol,kverts_buffer,"active",(T)1.0);
            // TILEVEC_OPS::fill(pol,kverts_buffer,"embed_tet_id",zs::reinterpret_bits<T>((int)-1));
        }

        if(write_back_gia_res) {
            if(!kverts.hasProperty("flood")) {
                kverts.append_channels(pol,{{"flood",1}});
            }
            TILEVEC_OPS::fill(pol,kverts,"flood",(T)0.0);
        }

        auto nm_rings = do_global_intersection_analysis_with_connected_manifolds(pol,
            surf_verts_buffer,tet_pos_attr_tag,surf_tris_buffer,tet_halfedges,true,
            kverts_buffer,"x",ktris,khalfedges,false,
            gia_res,tris_gia_res);

        // as we only do the intersection detection between kverts and ptris, we only need to flood the two
        if(write_back_gia_res) {
            pol(zs::range(surf_tris_buffer.size()),[
                tet_verts = proxy<space>({},tet_verts),
                tris_gia_res = proxy<space>(tris_gia_res),
                surf_tris_buffer = proxy<space>({},surf_tris_buffer),
                surf_tris = proxy<space>({},tet_surf_tris)] ZS_LAMBDA(int ti) mutable {
                    // auto vi = zs::reinterpret_bits<int>(surf_verts_buffer("inds",pi));
                    auto tri = surf_tris.pack(dim_c<3>,"inds",ti,int_c);
                    auto ring_mask = tris_gia_res[ti];
                    if(ring_mask > 0)
                        for(int i = 0;i != 3;++i)
                            tet_verts("flood",tri[i]) = (T)1.0;
            });
            pol(zs::range(kverts.size()),[
                kverts = proxy<space>({},kverts),
                gia_res = proxy<space>(gia_res),
                voffset = surf_verts_buffer.size(),
                kverts_buffer = proxy<space>({},kverts_buffer)] ZS_LAMBDA(int kvi) mutable {
                    auto ring_mask = gia_res[kvi + voffset];
                    if(ring_mask > 0)
                        kverts("flood",kvi) = (T)1.0;
            });
        }
        
        auto ktris_cnorm = 2 * compute_average_edge_length(pol,kverts,"x",ktris);    


        bool colliding_from_inside = true;
        pol(zs::range(kverts_buffer.size()),[
            exec_tag = exec_tag,
            out_collisionEps = out_collisionEps,
            in_collisionEps = in_collisionEps,
            kverts_buffer = proxy<space>({},kverts_buffer),
            csPT = proxy<space>(csPT),
            gia_res = proxy<space>(gia_res),
            tris_gia_res = proxy<space>(tris_gia_res),
            kvoffset = kvoffset,
            thickness = ktris_cnorm,
            halfedges = proxy<space>({},tet_halfedges),
            colliding_from_inside = colliding_from_inside,
            verts_buffer = proxy<space>({},surf_verts_buffer),
            pos_attr_tag = zs::SmallString(tet_pos_attr_tag),
            kv_offset = surf_verts_buffer.size(),
            kt_offset = surf_tris_buffer.size(),
            tri_buffer = proxy<space>({},surf_tris_buffer),
            triBvh = proxy<space>(triBvh)] ZS_LAMBDA(int kvi) mutable {
                    auto kp = kverts_buffer.pack(dim_c<3>,"x",kvi);
                    auto bv = bv_t{get_bounding_box(kp - thickness,kp + thickness)};
                    T min_penertration_distance = (T)1e8;
                    int min_ti = -1;     
                    auto process_vertex_face_collision_pairs = [&](int ti) {
                auto tri = tri_buffer.pack(dim_c<3>,"inds",ti,int_c);
                vec3 tvs[3] = {};
                for(int i = 0;i != 3;++i)
                    tvs[i] = verts_buffer.pack(dim_c<3>,pos_attr_tag,tri[i]);
                auto tnrm = LSL_GEO::facet_normal(tvs[0],tvs[1],tvs[2]);

                auto seg = kp - tvs[0];
                auto dist = tnrm.dot(seg);
                if(colliding_from_inside)
                    dist = -dist;
                
                auto collisionEps = dist > 0 ? out_collisionEps : in_collisionEps;
                auto barySum = (T)1.0;
                T distance = LSL_GEO::get_vertex_triangle_distance(tvs[0],tvs[1],tvs[2],kp,barySum);

                if(distance > collisionEps)
                    return;       

                auto hi = zs::reinterpret_bits<int>(tri_buffer("he_inds",ti));
                vec3 bnrms[3] = {};
                for(int i = 0;i != 3;++i) {
                    auto edge_normal = tnrm;
                    auto opposite_he = zs::reinterpret_bits<int>(halfedges("opposite_he",hi));
                    if(opposite_he >= 0) {
                        auto nti = zs::reinterpret_bits<int>(halfedges("to_face",opposite_he));
                        auto ntri = tri_buffer.pack(dim_c<3>,"inds",nti,int_c);
                        auto ntnrm = LSL_GEO::facet_normal(
                            verts_buffer.pack(dim_c<3>,pos_attr_tag,ntri[0]),
                            verts_buffer.pack(dim_c<3>,pos_attr_tag,ntri[1]),
                            verts_buffer.pack(dim_c<3>,pos_attr_tag,ntri[2]));
                        edge_normal = tnrm + ntnrm;
                        edge_normal = edge_normal/(edge_normal.norm() + (T)1e-6);
                    }
                    auto e01 = tvs[(i + 1) % 3] - tvs[(i + 0) % 3];
                    bnrms[i] = edge_normal.cross(e01).normalized();
                    hi = zs::reinterpret_bits<int>(halfedges("next_he",hi));
                }

                if(barySum > (T)(1.0 + 1e-6)) {
                    for(int i = 0;i != 3;++i){
                        seg = kp - tvs[i];
                        if(bnrms[i].dot(seg) < 0)
                            return;
                    }
                }

                if(dist < 0/* && distance < min_penertration_distance*/) {
                    int RING_MASK = gia_res[kvi + kv_offset] & tris_gia_res[ti];
                    if(RING_MASK == 0)
                        return;
                }

                csPT.insert(zs::vec<int,2>{kvi + kvoffset,ti});
            };
            triBvh.iter_neighbors(bv,process_vertex_face_collision_pairs);

        });

        return csPT.size();
}

};

};