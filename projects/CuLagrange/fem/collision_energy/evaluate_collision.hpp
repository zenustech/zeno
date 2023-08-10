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

#include "../../geometry/kernel/calculate_facet_normal.hpp"
#include "../../geometry/kernel/topology.hpp"
#include "../../geometry/kernel/compute_characteristic_length.hpp"
#include "../../geometry/kernel/calculate_bisector_normal.hpp"

#include "../../geometry/kernel/tiled_vector_ops.hpp"
#include "../../geometry/kernel/geo_math.hpp"


#include "../../geometry/kernel/calculate_edge_normal.hpp"

#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bvs.hpp"
#include "zensim/container/Bvtt.hpp"

#include "vertex_face_sqrt_collision.hpp"
#include "vertex_face_collision.hpp"
#include "../../geometry/kernel/topology.hpp"
#include "../../geometry/kernel/intersection.hpp"
// #include "edge_edge_sqrt_collision.hpp"
// #include "edge_edge_collision.hpp"

namespace zeno { namespace COLLISION_UTILS {

using T = float;
using bvh_t = zs::LBvh<3,int,T>;
using bv_t = zs::AABBBox<3, T>;
using vec3 = zs::vec<T, 3>;



template<typename Pol,
            typename PosTileVec,
            typename SurfPointTileVec,
            typename SurfTriTileVec,
            // typename TetraTileVec,
            // typename HalfFacetTileVec,
            typename HalfEdgeTileVec> 
inline void do_facet_point_collision_detection(Pol& cudaPol,
    const PosTileVec& verts,const zs::SmallString& xtag,
    const SurfPointTileVec& points,
    const SurfTriTileVec& tris,
    const HalfEdgeTileVec& halfedges,
    // const TetraTileVec& tets,
    // const HalfFacetTileVec& halffacets,
    zs::Vector<zs::vec<int,4>>& csPT,
    int& nm_collisions,T in_collisionEps,T out_collisionEps) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        auto avgl = compute_average_edge_length(cudaPol,verts,xtag,tris);
        auto bvh_thickness = 3 * avgl;

        auto spBvh = bvh_t{};
        auto bvs = retrieve_bounding_volumes(cudaPol,verts,points,wrapv<1>{},(T)bvh_thickness,xtag);
        spBvh.build(cudaPol,bvs);

        zs::Vector<int> nm_csPT{points.get_allocator(),1};
        nm_csPT.setVal(0);

        zs::vec<int,12> facets = {
            0,1,2,
            1,3,2,
            0,2,3,
            0,3,1
        };

        cudaPol(zs::range(tris.size()),[in_collisionEps = in_collisionEps,
            out_collisionEps = out_collisionEps,
            verts = proxy<space>({},verts),
            points = proxy<space>({},points),
            tris = proxy<space>({},tris),
            nm_csPT = proxy<space>(nm_csPT),
            xtag = xtag,
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
            
                // auto tnrm = triNrmBuffer.pack(dim_c<3>,"nrm",stI);
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

                    // bool can_be_penetrating = true;
                    int embed_tet_id = -1;
                    if(verts.hasProperty("embed_tet_id")) {
                        embed_tet_id = zs::reinterpret_bits<int>(verts("embed_tet_id",vi));
                    }

                    auto p = verts.pack(dim_c<3>,xtag,vi);
                    auto seg = p - tvs[0];
                    auto dist = seg.dot(tnrm);

                    auto collisionEps = dist > 0 ? out_collisionEps : in_collisionEps;

                    auto barySum = (T)1.0;
                    T distance = LSL_GEO::pointTriangleDistance(tvs[0],tvs[1],tvs[2],p,barySum);

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

                    if(dist < 0 && verts.hasProperty("ring_mask")) {
                        // do gia intersection test
                        int RING_MASK = zs::reinterpret_bits<int>(verts("ring_mask",vi));
                        if(RING_MASK == 0)
                            return;
                        // bool is_same_ring = false;
                        int TRING_MASK = 0;
                        for(int i = 0;i != 3;++i) {
                            // auto TGIA_TAG = reinterpret_bits<int>(verts("ring_mask",tri[i]));
                            TRING_MASK |= zs::reinterpret_bits<int>(verts("ring_mask",tri[i]));
                            // if((TGIA_TAG | GIA_TAG) > 0)
                            //     is_same_ring = true;
                        }
                        RING_MASK = RING_MASK & TRING_MASK;
                        // the point and the tri should belong to the same ring
                        if(RING_MASK == 0)
                            return;

                        // now the two pair belong to the same ring, check whether they belong black-white loop, and have different colors 
                        auto COLOR_MASK = reinterpret_bits<int>(verts("color_mask",vi));
                        auto TYPE_MASK = reinterpret_bits<int>(verts("type_mask",vi));

                        // only check the common type-1(white-black loop) rings
                        int TTYPE_MASK = 0;
                        for(int i = 0;i != 3;++i)
                            TTYPE_MASK |= reinterpret_bits<int>(verts("type_mask",tri[i]));
                        
                        RING_MASK &= (TYPE_MASK & TTYPE_MASK);
                        // int nm_common_rings = 0;
                        // while()
                        // as long as there is one ring in which the pair have different colors, neglect the pair
                        int curr_ri_mask = 1;
                        for(;RING_MASK > 0;RING_MASK = RING_MASK >> 1,curr_ri_mask = curr_ri_mask << 1) {
                            if(RING_MASK & 1) {
                                for(int i = 0;i != 3;++i) {
                                    auto TCOLOR_MASK = reinterpret_bits<int>(verts("color_mask",tri[i])) & curr_ri_mask;
                                    auto VCOLOR_MASK = reinterpret_bits<int>(verts("color_mask",vi)) & curr_ri_mask;
                                    if(TCOLOR_MASK == VCOLOR_MASK)
                                        return;
                                }
                            }
                        }

                        // if(dist < 0) {
                        min_vi = vi;
                        min_penertration_depth = distance;
                        // }
                        // do shortest path test
                        
                    }

                    // if(dist < 0 && verts.hasProperty("embed_tet_id")) {
                    //     auto tet_id = zs::reinterpret_bits<int>(verts("embed_tet_id",vi));
                    // }

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

// POINT and KTri
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
            exec_tag,
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
                    T distance = LSL_GEO::pointTriangleDistance(ktvs[0],ktvs[1],ktvs[2],p,barySum);       

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
        // TILEVEC_OPS::fill(pol,surf_verts_buffer,"ring_mask",zs::reinterpret_bits<T>((int)0));
        if(tet_verts.hasProperty("active")) {
            topological_sample(pol,tet_surf_points,tet_verts,"active",surf_verts_buffer);
        }else {
            TILEVEC_OPS::fill(pol,surf_verts_buffer,"active",(T)1.0);
        }

        TILEVEC_OPS::copy(pol,tet_surf_tris,"inds",surf_tris_buffer,"inds");
        TILEVEC_OPS::copy(pol,tet_surf_tris,"he_inds",surf_tris_buffer,"he_inds");
        reorder_topology(pol,tet_surf_points,surf_tris_buffer);
        // TILEVEC_OPS::fill(pol,surf_tris_buffer,"ring_mask",zs::reinterpret_bits<T>((int)0));   

        auto tet_tris_cnorm = 2 * compute_average_edge_length(pol,surf_verts_buffer,tet_pos_attr_tag,surf_tris_buffer);            
        auto tetBvh = bvh_t{};
        auto tetBvs = retrieve_bounding_volumes(pol,tet_verts,tets,wrapv<4>{},(T)0,tet_pos_attr_tag);
        tetBvh.build(pol,tetBvs);

        auto triBvh = bvh_t{};
        auto triBvs = retrieve_bounding_volumes(pol,tet_verts,tet_surf_tris,wrapv<3>{},(T)0,tet_pos_attr_tag);
        triBvh.build(pol,triBvs);

        int k_id = -1;
        // csPTOffsets.resize(kinematics.size());
        // zs::Vector<int> csPTSize{csPTOffsets.get_allocator(),kinematics.size()};
        csPT.reset(pol,true);

        if(write_back_gia_res) {
            if(!tet_verts.hasProperty("flood"))
                tet_verts.append_channels(pol,{{"flood",1}});
            TILEVEC_OPS::fill(pol,tet_verts,"flood",(T)0.0);
        }

        int kvoffset = 0;

        // for(auto kinematic : kinematics) {
        //     ++k_id;
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
        // pol(zs::range(gia_res),[] ZS_LAMBDA(auto& ring_mask) {ring_mask = 0;});
        // pol(zs::range(tris_gia_res),[] ZS_LAMBDA(auto& ring_mask) {ring_mask = 0;});

        if(write_back_gia_res) {
            if(!kverts.hasProperty("flood")) {
                kverts.append_channels(pol,{{"flood",1}});
            }
            TILEVEC_OPS::fill(pol,kverts,"flood",(T)0.0);
        }

        // zs::Vector<int> nmExcludedPoints{tets.get_allocator(),1};
        // nmExcludedPoints.setVal(0);
        // pol(zs::range(kverts_buffer.size()),[
        //     pos_attr_tag = zs::SmallString(tet_pos_attr_tag),
        //     tet_verts = proxy<space>({},tet_verts),
        //     tetBvh = proxy<space>(tetBvh),
        //     tets = proxy<space>({},tets),
        //     thickness = tet_tris_cnorm,
        //     exec_tag,
        //     nmExcludedPoints = proxy<space>(nmExcludedPoints),
        //     kverts_buffer = proxy<space>({},kverts_buffer)] ZS_LAMBDA(int kvi) mutable {
        //         auto kv = kverts_buffer.pack(dim_c<3>,"x",kvi);
        //         auto bv = bv_t{get_bounding_box(kv - thickness,kv + thickness)};
        //         auto mark_interior_verts = [&](int ei) {
        //             // printf("testing %d %d\n",pi,ei);
        //             auto tet = tets.pack(dim_c<4>,"inds",ei,int_c);
        //             for(int i = 0;i != 4;++i)
        //                 if(tet[i] == kvi)
        //                     return;
        //             zs::vec<T,3> tV[4] = {};
        //             for(int i = 0;i != 4;++i)
        //                 tV[i] = tet_verts.pack(dim_c<3>,pos_attr_tag,tet[i]);
        //             if(LSL_GEO::is_inside_tet(tV[0],tV[1],tV[2],tV[3],kv)){
        //                 kverts_buffer("embed_tet_id",kvi) = zs::reinterpret_bits<T>((int)ei);
                        
        //             }
        //         };
        //         tetBvh.iter_neighbors(bv,mark_interior_verts);
                
        //         auto embed_tet_id = zs::reinterpret_bits<int>(kverts_buffer("embed_tet_id",kvi));
        //         if(embed_tet_id == -1)
        //             atomic_add(exec_tag,&nmExcludedPoints[0],(int)1);
        // });  
        // std::cout << "nm_excluded_points[ " << k_id <<  "] :" << nmExcludedPoints.getVal(0) << "\t" << kverts.size() << std::endl;    
        // pol(zs::range(kverts_buffer.size()),[
        //     kverts_buffer = proxy<space>({},kverts_buffer)] ZS_LAMBDA(int kvi) mutable {
        //         auto embed_tet_id = zs::reinterpret_bits<int>(kverts_buffer("embed_tet_id",kvi));
        //         if(embed_tet_id == -1)
        //             kverts_buffer("mustExclude",kvi) = (T)1.0;
        //         else
        //             kverts_buffer("mustExclude",kvi) = (T)0.0;
        // });    

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
            exec_tag,
            out_collisionEps = out_collisionEps,
            in_collisionEps = in_collisionEps,
            kverts_buffer = proxy<space>({},kverts_buffer),
            csPT = proxy<space>(csPT),
            // nm_k_csPT = proxy<space>(nm_k_csPT),
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
            // gia_res = proxy<space>(gia_res),
            // tris_gia_res = proxy<space>(tris_gia_res),
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
                T distance = LSL_GEO::pointTriangleDistance(tvs[0],tvs[1],tvs[2],kp,barySum);

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
                    // int RING_MASK = zs::reinterpret_bits<int>(kv_gia_res("ring_mask",kvi)) & zs::reinterpret_bits<int>(pt_gia_res("ring_mask",ti));
                    int RING_MASK = gia_res[kvi + kv_offset] & tris_gia_res[ti];
                    if(RING_MASK == 0)
                        return;
                    // min_penertration_distance = distance;
                    // min_ti = ti;
                }

                // if(dist > 0) {
                    // atomic_add(exec_tag,&nm_k_csPT[0],(int)1);
                    csPT.insert(zs::vec<int,2>{kvi + kvoffset,ti});
                // }
            };
            triBvh.iter_neighbors(bv,process_vertex_face_collision_pairs);
            // if(min_ti >= 0){
            //     // atomic_add(exec_tag,&nm_k_csPT[0],(int)1);
            //     csPT.insert(zs::vec<int,2>{kvi + kvoffset,min_ti});
            // }

        });
        // kvoffset += kverts.size();
        return csPT.size();
}

template<typename Pol,
    typename REAL,
    typename PosTileVec,
    typename TetTileVec,
    typename SurfPointTileVec,
    typename SurfTriTileVec,
    typename HalfEdgeTileVec,
    typename HalfFacetTileVec,
    typename HashMap>
inline void do_tetrahedra_surface_tris_and_points_self_collision_detection(Pol& pol,
    PosTileVec& tet_verts,const zs::SmallString& pos_attr_tag,
    const TetTileVec& tets,
    const SurfPointTileVec& surf_points,const SurfTriTileVec& surf_tris,
    HalfEdgeTileVec& surf_halfedge,
    HalfFacetTileVec& tet_halffacet,
    REAL outCollisionEps,
    REAL inCollisionEps,
    HashMap& csPT,
    bool write_back_gia_res = false) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};
        using T = typename RM_CVREF_T(tet_verts)::value_type;

        PosTileVec surf_verts_buffer{surf_points.get_allocator(),{
            {pos_attr_tag,3},
            // {"ring_mask",1},
            // {"color_mask",1},
            // {"type_mask",1},
            {"embed_tet_id",1},
            // {"mustExclude",1},
            // {"is_loop_vertex",1},
            {"active",1}
        },surf_points.size()};
        // PosTileVec gia_res{surf_points.get_allocator(),{
        //     {"color_mask",1}
        // },(size_t)0};

        SurfTriTileVec surf_tris_buffer{surf_tris.get_allocator(),{
            {"inds",3},
            {"he_inds",1},
            // {"ring_mask",1},
            // {"color_mask",1},
            // {"type_mask",1}            
        },surf_tris.size()};

        PosTileVec gia_res{surf_points.get_allocator(),{
            // {pos_attr_tag,3},
            {"ring_mask",1},
            {"color_mask",1},
            {"type_mask",1},
            // {"mustExclude",1},
            {"is_loop_vertex",1},
            // {"active",1}
        },(size_t)0};

        SurfTriTileVec tris_gia_res{surf_tris.get_allocator(),{
            // {"inds",3},
            // {"he_inds",1},
            {"ring_mask",1},
            {"color_mask",1},
            {"type_mask",1}            
        },(size_t)0};



        topological_sample(pol,surf_points,tet_verts,pos_attr_tag,surf_verts_buffer);
        // TILEVEC_OPS::fill(pol,surf_verts_buffer,"ring_mask",zs::reinterpret_bits<T>((int)0));
        // TILEVEC_OPS::fill(pol,surf_verts_buffer,"color_mask",zs::reinterpret_bits<T>((int)0));
        // TILEVEC_OPS::fill(pol,surf_verts_buffer,"type_mask",zs::reinterpret_bits<T>((int)0));
        if(tet_verts.hasProperty("active")) {
            topological_sample(pol,surf_points,tet_verts,"active",surf_verts_buffer);
        }else {
            TILEVEC_OPS::fill(pol,surf_verts_buffer,"active",(T)1.0);
        }


        TILEVEC_OPS::copy(pol,surf_tris,"inds",surf_tris_buffer,"inds");
        TILEVEC_OPS::copy(pol,surf_tris,"he_inds",surf_tris_buffer,"he_inds");
        reorder_topology(pol,surf_points,surf_tris_buffer);
        // TILEVEC_OPS::fill(pol,surf_tris_buffer,"ring_mask",zs::reinterpret_bits<T>((int)0));
        // TILEVEC_OPS::fill(pol,surf_tris_buffer,"color_mask",zs::reinterpret_bits<T>((int)0));
        // TILEVEC_OPS::fill(pol,surf_tris_buffer,"type_mask",zs::reinterpret_bits<T>((int)0));

        auto cnorm = 5 * compute_average_edge_length(pol,surf_verts_buffer,pos_attr_tag,surf_tris_buffer);
        // evaluate_embed_tet_id
        auto tetBvh = bvh_t{};
        auto tetBvs = retrieve_bounding_volumes(pol,tet_verts,tets,wrapv<4>{},(T)0,pos_attr_tag);
        tetBvh.build(pol,tetBvs);
        TILEVEC_OPS::fill(pol,surf_verts_buffer,"embed_tet_id",zs::reinterpret_bits<T>((int)-1));

        // zs::Vector<int> nmExcludedPoints{tets.get_allocator(),1};
        // nmExcludedPoints.setVal(0);
        pol(zs::range(surf_verts_buffer.size()),[
            pos_attr_tag = zs::SmallString(pos_attr_tag),
            surf_verts_buffer = proxy<space>({},surf_verts_buffer),
            surf_points = proxy<space>({},surf_points),
            tetBvh = proxy<space>(tetBvh),
            tets = proxy<space>({},tets),
            thickness = cnorm,
            exec_tag,
            // nmExcludedPoints = proxy<space>(nmExcludedPoints),
            tet_verts = proxy<space>({},tet_verts)] ZS_LAMBDA(int pi) mutable {
                auto pv = surf_verts_buffer.pack(dim_c<3>,pos_attr_tag,pi);
                auto vi = zs::reinterpret_bits<int>(surf_points("inds",pi));
                auto bv = bv_t{get_bounding_box(pv - thickness,pv + thickness)};
                auto mark_interior_verts = [&, exec_tag](int ei) {
                    // printf("testing %d %d\n",pi,ei);
                    auto tet = tets.pack(dim_c<4>,"inds",ei,int_c);
                    for(int i = 0;i != 4;++i)
                        if(tet[i] == vi)
                            return;
                    zs::vec<T,3> tV[4] = {};
                    for(int i = 0;i != 4;++i)
                        tV[i] = tet_verts.pack(dim_c<3>,pos_attr_tag,tet[i]);
                    if(LSL_GEO::is_inside_tet(tV[0],tV[1],tV[2],tV[3],pv)){
                        surf_verts_buffer("embed_tet_id",pi) = zs::reinterpret_bits<T>((int)ei);
                        // atomic_add(exec_tag,&nmExcludedPoints[0],(int)1);
                    }
                };
                tetBvh.iter_neighbors(bv,mark_interior_verts);
        });        
        // std::cout << "nm_excluded_points :" << nmExcludedPoints.getVal(0) << "\t" << surf_verts_buffer.size() << std::endl;

        // pol(zs::range(surf_verts_buffer.size()),[
        //     surf_verts_buffer = proxy<space>({},surf_verts_buffer)] ZS_LAMBDA(int pi) mutable {
        //         auto embed_tet_id = zs::reinterpret_bits<int>(surf_verts_buffer("embed_tet_id",pi));
        //         if(embed_tet_id >= 0)
        //             surf_verts_buffer("mustExclude",pi) = (T)1.0;
        //         else
        //             surf_verts_buffer("mustExclude",pi) = (T)0.0;
        // });
        auto ring_mask_width = do_global_self_intersection_analysis(pol,
            surf_verts_buffer,pos_attr_tag,surf_tris_buffer,surf_halfedge,
            gia_res,tris_gia_res);
        std::cout << "ring_mask_width of GIA : " << ring_mask_width << std::endl;

        if(write_back_gia_res) {
            // if(!tet_verts.hasProperty("ring_mask")) {
            //     tet_verts.append_channels(pol,{{"ring_mask",1}});
            // }
            if(!tet_verts.hasProperty("flood")) {
                tet_verts.append_channels(pol,{{"flood",1}});
            }
            // TILEVEC_OPS::fill(pol,tet_verts,"ring_mask",zs::reinterpret_bits<T>((int)0));
            TILEVEC_OPS::fill(pol,tet_verts,"flood",(T)0);
            pol(zs::range(surf_points.size()),[
                tet_verts = proxy<space>({},tet_verts),
                surf_points = proxy<space>({},surf_points),
                ring_mask_width = ring_mask_width,
                gia_res = proxy<space>({},gia_res)] ZS_LAMBDA(int pi) mutable {
                    auto vi = zs::reinterpret_bits<int>(surf_points("inds",pi));
                    // tet_verts("ring_mask",vi) = surf_verts_buffer("ring_mask",pi);
                    for(int i = 0;i != ring_mask_width;++i) {
                        auto ring_mask = zs::reinterpret_bits<int>(gia_res("ring_mask",pi * ring_mask_width + i));
                        if(ring_mask > 0) {
                            tet_verts("flood",vi) = (T)1.0;
                            return;
                        }
                    }
            });
        }

        auto spBvh = bvh_t{};
        auto spBvs = retrieve_bounding_volumes(pol,tet_verts,surf_points,wrapv<1>{},(T)cnorm,pos_attr_tag);
        spBvh.build(pol,spBvs);

        csPT.reset(pol,true);
        pol(zs::range(surf_tris_buffer.size()),[
            outCollisionEps = outCollisionEps,
            inCollisionEps = inCollisionEps,
            surf_verts_buffer = proxy<space>({},surf_verts_buffer,"surf_verts_buffer_problem"),
            surf_tris_buffer = proxy<space>({},surf_tris_buffer),
            pos_attr_tag = zs::SmallString(pos_attr_tag),
            surf_halfedge = proxy<space>({},surf_halfedge),
            tets = proxy<space>({},tets),
            tet_verts = proxy<space>({},tet_verts),
            gia_res = proxy<space>({},gia_res),
            tris_gia_res= proxy<space>({},tris_gia_res),
            ring_mask_width = ring_mask_width,
            thickness = cnorm,
            csPT = proxy<space>(csPT),
            spBvh = proxy<space>(spBvh)] ZS_LAMBDA(int ti) mutable {
                auto tri = surf_tris_buffer.pack(dim_c<3>,"inds",ti,int_c);
                for(int i = 0;i != 3;++i)
                    if(surf_verts_buffer("active",tri[i]) < (T)0.5)
                        return;
                zs::vec<T,3> tvs[3] = {};
                for(int i = 0;i != 3;++i)
                    tvs[i] = surf_verts_buffer.pack(dim_c<3>,pos_attr_tag,tri[i]);
                auto tri_center = zs::vec<T,3>::zeros();
                for(int i = 0;i != 3;++i)
                    tri_center += tvs[i]/(T)3.0;
                auto bv = bv_t{get_bounding_box(tri_center - thickness,tri_center + thickness)};
                auto tnrm = LSL_GEO::facet_normal(tvs[0],tvs[1],tvs[2]);

                auto hi = zs::reinterpret_bits<int>(surf_tris_buffer("he_inds",ti));
                vec3 bnrms[3] = {};
                for(int i = 0;i != 3;++i){
                    auto edge_normal = tnrm;
                    auto opposite_hi = zs::reinterpret_bits<int>(surf_halfedge("opposite_he",hi));
                    if(opposite_hi >= 0){
                        auto nti = zs::reinterpret_bits<int>(surf_halfedge("to_face",opposite_hi));
                        auto ntri = surf_tris_buffer.pack(dim_c<3>,"inds",nti,int_c);
                        auto ntnrm = LSL_GEO::facet_normal(
                            surf_verts_buffer.pack(dim_c<3>,pos_attr_tag,ntri[0]),
                            surf_verts_buffer.pack(dim_c<3>,pos_attr_tag,ntri[1]),
                            surf_verts_buffer.pack(dim_c<3>,pos_attr_tag,ntri[2]));
                        edge_normal = tnrm + ntnrm;
                        edge_normal = edge_normal/(edge_normal.norm() + (T)1e-6);
                    }
                    auto e01 = tvs[(i + 1) % 3] - tvs[(i + 0) % 3];
                    bnrms[i] = edge_normal.cross(e01).normalized();
                    hi = zs::reinterpret_bits<int>(surf_halfedge("next_he",hi));
                } 

                T min_penertration_depth = zs::limits<T>::max();
                int min_vi = -1;
                auto process_vertex_face_collision_pairs = [&, pos_attr_tag](int vi) {
                    if(tri[0] == vi || tri[1] == vi || tri[2] == vi)
                        return;
                    if(surf_verts_buffer("active",vi) < (T)0.5)
                        return;
                    auto embed_tet_id = zs::reinterpret_bits<int>(surf_verts_buffer("embed_tet_id",vi));
                    auto p = surf_verts_buffer.pack(dim_c<3>,pos_attr_tag,vi);
                    auto seg = p - tri_center;
                    auto dist = seg.dot(tnrm);

                    auto collisionEps = dist > 0 ? outCollisionEps : inCollisionEps;

                    auto barySum = (T)1.0;
                    T distance = LSL_GEO::pointTriangleDistance(tvs[0],tvs[1],tvs[2],p,barySum);

                    if(distance > collisionEps)
                        return;

                    if(barySum > (T)(1.0 + 1e-6)) {
                        for(int i = 0;i != 3;++i){
                            seg = p - tvs[i];
                            if(bnrms[i].dot(seg) < 0)
                                return;
                        }
                    }

                    if(dist < 0 && embed_tet_id < 0)
                        return;

                    if(dist < 0 && distance > min_penertration_depth)
                        return;

                    // if(dist < 0) {
                    //     auto neighbor_tet = zs::reinterpret_bits<int>(halffacets(""))
                    // }
                    bool is_valid_inverted_pair = false;
                    for(int bi = 0;bi != ring_mask_width;++bi) {
                        if(dist < 0) {
                            // do gia intersection test
                            int RING_MASK = zs::reinterpret_bits<int>(gia_res("ring_mask",vi * ring_mask_width + bi));
                            if(RING_MASK == 0)
                                continue;
                            // bool is_same_ring = false;
                            int TRING_MASK = ~0;
                            for(int i = 0;i != 3;++i) {
                                TRING_MASK &= zs::reinterpret_bits<int>(gia_res("ring_mask",tri[i] * ring_mask_width + bi));
                            }
                            RING_MASK = RING_MASK & TRING_MASK;
                            // the point and the tri should belong to the same ring
                            if(RING_MASK == 0)
                                continue;

                            // now the two pair belong to the same ring, check whether they belong black-white loop, and have different colors 
                            auto COLOR_MASK = reinterpret_bits<int>(gia_res("color_mask",vi * ring_mask_width + bi));
                            auto TYPE_MASK = reinterpret_bits<int>(gia_res("type_mask",vi * ring_mask_width + bi));

                            // only check the common type-1(white-black loop) rings
                            int TTYPE_MASK = 0;
                            for(int i = 0;i != 3;++i)
                                TTYPE_MASK |= reinterpret_bits<int>(gia_res("type_mask",tri[i] * ring_mask_width + bi));
                            
                            RING_MASK &= (TYPE_MASK & TTYPE_MASK);
                            // type-0 ring
                            if(RING_MASK == 0) {
                                is_valid_inverted_pair = true;
                                break;
                            }
                            // int nm_common_rings = 0;
                            // while()
                            // as long as there is one ring in which the pair have different colors, neglect the pair
                            int curr_ri_mask = 1;
                            bool is_color_same = false;
                            for(;RING_MASK > 0;RING_MASK = RING_MASK >> 1,curr_ri_mask = curr_ri_mask << 1) {
                                if(RING_MASK & 1) {
                                    // int TCOLOR_MASK = ~0; 
                                    // for(int i = 0;i != 3;++i)
                                    //     TCOLOR_MASK &= reinterpret_bits<int>(gia_res("color_mask",tri[i] * ring_mask_width + bi)) & curr_ri_mask;
                                    // int auto VCOLOR_MASK = reinterpret_bits<int>(gia_res("color_mask",vi * ring_mask_width + bi)) & curr_ri_mask;
                                    for(int i = 0;i != 3;++i) {
                                        auto TCOLOR_MASK = reinterpret_bits<int>(gia_res("color_mask",tri[i] * ring_mask_width + bi)) & curr_ri_mask;
                                        auto VCOLOR_MASK = reinterpret_bits<int>(gia_res("color_mask",vi * ring_mask_width + bi)) & curr_ri_mask;
                                        if(TCOLOR_MASK == VCOLOR_MASK) {
                                            is_color_same = true;
                                            break;
                                        }
                                    }
                                    if(is_color_same)
                                        break;
                                }
                            }

                            if(!is_color_same) {
                                // type-1 ring with different color
                                is_valid_inverted_pair = true;
                                break;
                            }

                            // break;

                            // embed_tet_id >= 0
                            // do shortest path detection
                            // auto ei = embed_tet_id;



                            // is_valid_pair = true;
                        }
                        // if(!is_valid_inverted_pair)
                        //     return;
                    }


                    if(ring_mask_width == 0 && dist < 0) {
                        return;
                    }

                    if(dist < 0 && is_valid_inverted_pair){
                        min_vi = vi;
                        min_penertration_depth = distance;
                    }
                    if(dist > 0)
                        csPT.insert(zs::vec<int,2>{vi,ti});
                };
                spBvh.iter_neighbors(bv,process_vertex_face_collision_pairs);
                if(min_vi >= 0)
                    csPT.insert(zs::vec<int,2>{min_vi,ti});
        });

        auto stBvh = bvh_t{};
        auto stBvs = retrieve_bounding_volumes(pol,tet_verts,surf_tris,wrapv<3>{},(T)cnorm,pos_attr_tag);
        stBvh.build(pol,stBvs);

        // csPT.reset(pol,true);
        // for each verts, find the closest tri
        pol(zs::range(surf_verts_buffer.size()),[
            outCollisionEps = outCollisionEps,
            inCollisionEps = inCollisionEps,
            surf_verts_buffer = proxy<space>({},surf_verts_buffer,"surf_verts_buffer_problem"),
            surf_tris_buffer = proxy<space>({},surf_tris_buffer),
            pos_attr_tag = zs::SmallString(pos_attr_tag),
            surf_halfedge = proxy<space>({},surf_halfedge),
            tets = proxy<space>({},tets),
            tet_verts = proxy<space>({},tet_verts),
            gia_res = proxy<space>({},gia_res),
            tris_gia_res= proxy<space>({},tris_gia_res),
            ring_mask_width = ring_mask_width,
            thickness = cnorm,
            csPT = proxy<space>(csPT),
            stBvh = proxy<space>(stBvh)] ZS_LAMBDA(int pi) mutable {
                auto p = surf_verts_buffer.pack(dim_c<3>,pos_attr_tag,pi);
                auto bv = bv_t{get_bounding_box(p - thickness,p + thickness)};

                if(surf_verts_buffer("active",pi) < (T)0.5)
                    return;
                T min_penertration_depth = zs::limits<T>::max();
                int min_ti = -1;
                auto process_vertex_face_collision_pairs = [&, pos_attr_tag](int ti) {
                    auto tri = surf_tris_buffer.pack(dim_c<3>,"inds",ti,int_c);
                    for(int i = 0;i != 3;++i)
                        if(surf_verts_buffer("active",tri[i]) < (T)0.5)
                            return;

                    zs::vec<T,3> tvs[3] = {};
                    for(int i = 0;i != 3;++i)
                        tvs[i] = surf_verts_buffer.pack(dim_c<3>,pos_attr_tag,tri[i]);

                    if(tri[0] == pi || tri[1] == pi || tri[2] == pi)
                        return;

                    // auto tri_center = zs::vec<T,3>::zeros();
                    // for(int i = 0;i != 3;++i)
                    //     tri_center += tvs[i]/(T)3.0;

                    auto embed_tet_id = zs::reinterpret_bits<int>(surf_verts_buffer("embed_tet_id",pi));
                    // auto p = surf_verts_buffer.pack(dim_c<3>,pos_attr_tag,vi);
                    auto tnrm = LSL_GEO::facet_normal(tvs[0],tvs[1],tvs[2]);
                    auto seg = p - tvs[0];
                    auto dist = seg.dot(tnrm);

                    auto collisionEps = dist > 0 ? outCollisionEps : inCollisionEps;

                    auto barySum = (T)1.0;
                    T distance = LSL_GEO::pointTriangleDistance(tvs[0],tvs[1],tvs[2],p,barySum);

                    if(distance > collisionEps)
                        return;


                    if(dist < 0 && distance > min_penertration_depth)
                        return;

                    // if(dist < 0) {
                    //     auto neighbor_tet = zs::reinterpret_bits<int>(halffacets(""))
                    // }
                    if(dist < 0 && embed_tet_id < 0)
                        return;
                    // printf("testing pair %d %d\n",pi,ti);

                    bool is_valid_inverted_pair = false;
                    for(int bi = 0;bi != ring_mask_width;++bi) {
                        if(dist < 0) {
                            // do gia intersection test
                            int V_RING_MASK = zs::reinterpret_bits<int>(gia_res("ring_mask",pi * ring_mask_width + bi));
                            if(V_RING_MASK == 0)
                                continue;
                            // bool is_same_ring = false;
                            // int TRING_MASK = ~0;
                            int TRING_MASK = ~0;
                            for(int i = 0;i != 3;++i) {
                                TRING_MASK &= zs::reinterpret_bits<int>(gia_res("ring_mask",tri[i] * ring_mask_width + bi));
                            }
                            auto RING_MASK = V_RING_MASK & TRING_MASK;
                            // the point and the tri should belong to the same ring
                            if(RING_MASK == 0)
                                continue;
                            // else if(V_RING_MASK > 0 && TRING_MASK > 0){
                            //     is_valid_inverted_pair = true;
                            //     break;
                            // }else {
                            //     continue;
                            // }

                            // now the two pair belong to the same ring, check whether they belong black-white loop, and have different colors 
                            auto COLOR_MASK = reinterpret_bits<int>(gia_res("color_mask",pi * ring_mask_width + bi));
                            auto TYPE_MASK = reinterpret_bits<int>(gia_res("type_mask",pi * ring_mask_width + bi));

                            // only check the common type-1(white-black loop) rings
                            int TTYPE_MASK = 0;
                            for(int i = 0;i != 3;++i)
                                TTYPE_MASK |= reinterpret_bits<int>(gia_res("type_mask",tri[i] * ring_mask_width + bi));
                            
                            RING_MASK &= (TYPE_MASK & TTYPE_MASK);
                            // type-0 ring
                            if(RING_MASK == 0) {
                                is_valid_inverted_pair = true;
                                break;
                            }
                            // int nm_common_rings = 0;
                            // while()
                            // as long as there is one ring in which the pair have different colors, neglect the pair
                            int curr_ri_mask = 1;
                            bool is_color_same = false;
                            for(;RING_MASK > 0;RING_MASK = RING_MASK >> 1,curr_ri_mask = curr_ri_mask << 1) {
                                if(RING_MASK & 1) {
                                    // int TCOLOR_MASK = ~0; 
                                    // for(int i = 0;i != 3;++i)
                                    //     TCOLOR_MASK &= reinterpret_bits<int>(gia_res("color_mask",tri[i] * ring_mask_width + bi)) & curr_ri_mask;
                                    // int auto VCOLOR_MASK = reinterpret_bits<int>(gia_res("color_mask",vi * ring_mask_width + bi)) & curr_ri_mask;
                                    for(int i = 0;i != 3;++i) {
                                        auto TCOLOR_MASK = reinterpret_bits<int>(gia_res("color_mask",tri[i] * ring_mask_width + bi)) & curr_ri_mask;
                                        auto VCOLOR_MASK = reinterpret_bits<int>(gia_res("color_mask",pi * ring_mask_width + bi)) & curr_ri_mask;
                                        if(TCOLOR_MASK == VCOLOR_MASK) {
                                            is_color_same = true;
                                            break;
                                        }
                                    }
                                    if(is_color_same)
                                        break;
                                }
                            }

                            if(!is_color_same) {
                                // type-1 ring with different color
                                is_valid_inverted_pair = true;
                                break;
                            }
                            // else {
                            //     printf("skip with the same color %d %d\n",pi,ti);
                            // }

                            // break;

                            // embed_tet_id >= 0
                            // do shortest path detection
                            // auto ei = embed_tet_id;



                            // is_valid_pair = true;
                        }
                        // if(!is_valid_inverted_pair)
                        //     return;
                    }


                    if(barySum > (T)(1.0 + 0.1)) {
                        auto hi = zs::reinterpret_bits<int>(surf_tris_buffer("he_inds",ti));
                        vec3 bnrms[3] = {};
                        for(int i = 0;i != 3;++i){
                            auto edge_normal = tnrm;
                            auto opposite_hi = zs::reinterpret_bits<int>(surf_halfedge("opposite_he",hi));
                            if(opposite_hi >= 0){
                                auto nti = zs::reinterpret_bits<int>(surf_halfedge("to_face",opposite_hi));
                                auto ntri = surf_tris_buffer.pack(dim_c<3>,"inds",nti,int_c);
                                auto ntnrm = LSL_GEO::facet_normal(
                                    surf_verts_buffer.pack(dim_c<3>,pos_attr_tag,ntri[0]),
                                    surf_verts_buffer.pack(dim_c<3>,pos_attr_tag,ntri[1]),
                                    surf_verts_buffer.pack(dim_c<3>,pos_attr_tag,ntri[2]));
                                edge_normal = tnrm + ntnrm;
                                edge_normal = edge_normal/(edge_normal.norm() + (T)1e-6);
                            }
                            auto e01 = tvs[(i + 1) % 3] - tvs[(i + 0) % 3];
                            bnrms[i] = edge_normal.cross(e01).normalized();
                            hi = zs::reinterpret_bits<int>(surf_halfedge("next_he",hi));
                        } 


                        for(int i = 0;i != 3;++i){
                            seg = p - tvs[i];
                            if(bnrms[i].dot(seg) < 0) {
                                // printf("skip due to bisector normal check\n");
                                return;
                            }
                        }
                    }

                    if(ring_mask_width == 0 && dist < 0) {
                        // printf("empty_ring_mask and negative dist\n");
                        return;
                    }

                    if(dist < 0 && is_valid_inverted_pair){
                        min_ti = ti;
                        min_penertration_depth = distance;
                    }
                    else if(dist > 0){
                        csPT.insert(zs::vec<int,2>{pi,ti});
                        // printf("find_new_positive pair %d %d\n",pi,ti);
                    }
                    // else {
                    //     printf("invalid inverted pair\n");
                    // }
                };
                stBvh.iter_neighbors(bv,process_vertex_face_collision_pairs);
                if(min_ti >= 0) {
                    // printf("find_new_negative pair %d %d\n",pi,min_ti);
                    csPT.insert(zs::vec<int,2>{pi,min_ti});
                }
        });
}

template<typename Pol,
    typename REAL,
    typename PosTileVec,
    typename PointTileVec,
    typename TriTileVec,
    typename GradHessianTileVec,
    typename HashMap>
inline void evaluate_fp_self_collision_gradient_and_hessian(Pol& pol,
    const PosTileVec& verts,const zs::SmallString& pos_attr_tag,const zs::SmallString& nodal_area_tag,
    const PointTileVec& points,
    const TriTileVec& tris,const zs::SmallString& tri_area_tag,
    const HashMap& csPT,
    GradHessianTileVec& gh_buffer,REAL intensity,REAL ceps) {

        using namespace zs;
        using T = typename RM_CVREF_T(verts)::value_type;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        gh_buffer.resize(csPT.size());

        pol(zip(zs::range(csPT.size()),zs::range(csPT._activeKeys)),[
            verts = proxy<space>({},verts),
            pos_attr_tag = pos_attr_tag,
            points = proxy<space>({},points),
            nodal_area_tag = nodal_area_tag,
            tri_area_tag = tri_area_tag,
            tris = proxy<space>({},tris),
            gh_buffer = proxy<space>({},gh_buffer),
            ceps = ceps,
            intensity = intensity] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto pi = pair[0];
                auto ti = pair[1];

                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                zs::vec<T,3> cv[4] = {};
                auto vi = zs::reinterpret_bits<int>(points("inds",pi));
                cv[0] = verts.pack(dim_c<3>,pos_attr_tag,vi);
                for(int i = 0;i != 3;++i)   
                    cv[i + 1] = verts.pack(dim_c<3>,pos_attr_tag,tri[i]);

                auto area = verts(nodal_area_tag,vi) + tris(tri_area_tag,ti);
                
                auto mu = verts("mu",vi);
                auto lam = verts("lam",vi);

                auto cforce = -area * intensity * VERTEX_FACE_SQRT_COLLISION::gradient(cv,mu,lam,ceps,false);
                auto K = area * intensity * VERTEX_FACE_SQRT_COLLISION::hessian(cv,mu,lam,ceps,false);

                gh_buffer.tuple(dim_c<4>,"inds",ci) = zs::vec<int,4>{vi,tri[0],tri[1],tri[2]}.reinterpret_bits(float_c);
                gh_buffer.tuple(dim_c<12>,"grad",ci) = cforce;
                gh_buffer.tuple(dim_c<12 * 12>,"H",ci) = K;
        }); 
}

template<typename Pol,
    typename REAL,
    typename PosTileVec,
    typename TriTileVec,
    typename GradHessianTileVec,
    typename HashMap>
inline void evaluate_tri_kvert_collision_gradient_and_hessian(Pol& pol,
    const std::vector<ZenoParticles*>& kinematics,
    const PosTileVec& verts,const zs::SmallString& pos_attr_tag,const zs::SmallString& nodal_area_tag,
    const TriTileVec& tris,const zs::SmallString& tri_area_tag,
    const HashMap& csPT,
    GradHessianTileVec& gh_buffer,REAL intensity,REAL ceps,bool reverse = false) {
        using namespace zs;
        using T = typename RM_CVREF_T(verts)::value_type;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        int nm_kverts = 0;
        for(auto kinematic : kinematics)
            nm_kverts += kinematic->getParticles().size();
        
        PosTileVec all_kverts_buffer{verts.get_allocator(),{
            {"x",3},
            {"area",1}
        },nm_kverts};
        int voffset = 0;
        for(auto kinematic : kinematics) {
            const auto& kverts = kinematic->getParticles();
            TILEVEC_OPS::copy(pol,kverts,"x",all_kverts_buffer,"x",voffset);
            TILEVEC_OPS::copy(pol,kverts,nodal_area_tag,all_kverts_buffer,"area",voffset);
            voffset += kverts.size();
        }

        gh_buffer.resize(csPT.size());

        pol(zip(zs::range(csPT.size()),zs::range(csPT._activeKeys)),[
            all_kverts_buffer = proxy<space>({},all_kverts_buffer),
            verts = proxy<space>({},verts),
            pos_attr_tag = pos_attr_tag,
            tris = proxy<space>({},tris),
            nodal_area_tag = nodal_area_tag,
            tri_area_tag = tri_area_tag,
            ceps = ceps,
            intensity = intensity,
            reverse = reverse,
            gh_buffer = proxy<space>({},gh_buffer)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto kvi = pair[0];
                auto ti = pair[1];
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                zs::vec<T,3> cV[4] = {};
                cV[0] = all_kverts_buffer.pack(dim_c<3>,"x",kvi);
                for(int i = 0;i != 3;++i)
                    cV[i + 1] = verts.pack(dim_c<3>,pos_attr_tag,tri[i]);

                // auto kv = all_kverts_buffer.pack(dim_c<3>,"x",kvi);
                auto area = tris(tri_area_tag,ti) + all_kverts_buffer("area",kvi);

                auto mu = verts("mu",tri[0]);
                auto lam = verts("lam",tri[0]);

                auto cforce = -area * intensity * VERTEX_FACE_SQRT_COLLISION::gradient(cV,mu,lam,ceps,reverse);
                auto K = area * intensity * VERTEX_FACE_SQRT_COLLISION::hessian(cV,mu,lam,ceps,reverse);

                zs::vec<T,9> tforce{};
                zs::vec<T,9,9> tK{};
                for(int i = 0;i != 9;++i)
                    tforce[i] = cforce[i + 3];
                for(int i = 0;i != 9;++i)
                    for(int j = 0;j != 9;++j)
                        tK(i,j) = K(i + 3,j + 3);

                gh_buffer.tuple(dim_c<3>,"inds",ci) = tri.reinterpret_bits(float_c);
                gh_buffer.tuple(dim_c<9>,"grad",ci) = tforce;
                gh_buffer.tuple(dim_c<9 * 9>,"H",ci) = tK;
        });
}

template<typename Pol,
    typename REAL,
    typename PosTileVec,
    typename TriTileVec,
    typename PointTileVec,
    typename GradHessianTileVec,
    typename HashMap>
inline void evaluate_ktri_vert_collision_gradient_and_hessian(Pol& pol,
    const ZenoParticles* kinematic,
    const PosTileVec& verts,const zs::SmallString& pos_attr_tag,const zs::SmallString& nodal_area_tag,
    const TriTileVec& tris,const zs::SmallString& tri_area_tag,
    const PointTileVec& points,
    const HashMap& csPT,
    GradHessianTileVec& gh_buffer,REAL intensity,REAL ceps,bool reverse = false) {
        using namespace zs;
        using T = typename RM_CVREF_T(verts)::value_type;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};

        // int nm_kverts = 0;
        // int nm_ktris = 0;

        // for(auto kinematic : kinematics) {
        //     nm_kverts += kinematic->getParticles().size();
        //     nm_ktris += kinematic->getQuadraturePoints().size();
        // }

        // PosTileVec all_kverts_buffer{verts.get_allocator(),{
        //     {"x",3},
        // },nm_kverts};

        // TriTileVec all_ktris_buffer{tris.get_allocator(),{
        //     {"inds",3},
        //     {"area",1}
        // },nm_ktris};

        // int voffset = 0;
        // int toffset = 0;
        // for(auto kinematic : kinematics) {
        const auto& kverts = kinematic->getParticles();
        const auto& ktris = kinematic->getQuadraturePoints();

        // TILEVEC_OPS::copy(pol,kverts,"x",all_kverts_buffer,"x",voffset);
        // TILEVEC_OPS::copy(pol,ktris,"area",all_ktris_buffer,"area",toffset);

        // pol(zs::range(ktris.size()),[
        //     ktris = proxy<space>({},ktris),
        //     ktris = proxy<space>({},ktris),
        //     voffset = voffset,
        //     toffset = toffset] ZS_LAMBDA(int kti) mutable {
        //         auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);
        //         ktri += voffset;
        //         all_ktris_buffer.tuple(dim_c<3>,"inds",toffset + kti) = ktri.reinterpret_bits(float_c);
        // });

        // voffset += kverts.size();
        // toffset += ktris.size();
        // }

        gh_buffer.resize(csPT.size());
        pol(zip(zs::range(csPT.size()),zs::range(csPT._activeKeys)),[
            exec_tag,
            kverts = proxy<space>({},kverts),
            ktris = proxy<space>({},ktris),
            verts = proxy<space>({},verts),
            points = proxy<space>({},points),
            pos_attr_tag = pos_attr_tag,
            nodal_area_tag = nodal_area_tag,
            tri_area_tag = tri_area_tag,
            ceps = ceps,
            intensity = intensity,
            reverse = reverse,
            gh_buffer = proxy<space>({},gh_buffer)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto pi = pair[0];
                auto vi = zs::reinterpret_bits<int>(points("inds",pi));
                auto kti = pair[1];
                auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);

#if 1

                zs::vec<T,3> cV[4] = {};

                cV[0] = verts.pack(dim_c<3>,pos_attr_tag,vi);
                for(int i = 0;i != 3;++i)
                    cV[i + 1] = kverts.pack(dim_c<3>,"x",ktri[i]);

                auto area = ktris("area",kti) + verts("area",vi);
                auto mu = verts("mu",vi);
                auto lam = verts("lam",vi);

                auto cforce = -area * intensity * VERTEX_FACE_SQRT_COLLISION::gradient(cV,mu,lam,ceps,reverse);
                auto K = area * intensity * VERTEX_FACE_SQRT_COLLISION::hessian(cV,mu,lam,ceps,reverse);  

                zs::vec<T,3> tforce{};
                zs::vec<T,3,3> tK{};

                for(int i = 0;i != 3;++i)
                    tforce[i] = cforce[i];
                for(int i = 0;i != 3;++i)
                    for(int j = 0;j != 3;++j)
                        tK(i,j) = K(i,j);
                tK = (tK + tK.transpose()) * (T)0.5;
                
                // make_pd(tK);

                // if(zs::determinant(tK) < 0)
                //     printf("non-spd tK detected at %d %f %f %f %f %f\n %f %f %f\n%f %f %f\n%f %f %f\n",
                //         (int)ci,(float)zs::determinant(tK),(float)area,(float)intensity,mu,lam,
                //             (float)tK(0,0),(float)tK(0,1),(float)tK(0,2),
                //             (float)tK(1,0),(float)tK(1,1),(float)tK(1,2),
                //             (float)tK(2,0),(float)tK(2,1),(float)tK(2,2));

                // for(int i = 0;i != 3;++i)
                //     atomic_add(exec_ag,&gh_buffer("grad",vi * 3 + 0))
                gh_buffer.tuple(dim_c<3>,"grad",ci) = tforce;
                gh_buffer.tuple(dim_c<3,3>,"H",ci) = tK;
                // gh_buffer.tuple(dim_c<3>,"grad",ci) = zs::vec<T,3>::zeros();
                // gh_buffer.tuple(dim_c<3,3>,"H",ci) = zs::vec<T,3,3>::zeros();
                gh_buffer("inds",ci) = zs::reinterpret_bits<T>((int)vi);      
#else
                zs::vec<T,3> ktV[3] = {};
                for(int i = 0;i != 3;++i)
                    ktV[i] = kverts.pack(dim_c<3>,"x",ktri[i]);

                auto plane_root = kverts.pack(dim_c<3>,"x",ktri[0]);
                auto plane_nrm = LSL_GEO::facet_normal(ktV[0],ktV[1],ktV[2]);
                auto mu = verts("mu",vi);
                auto lam = verts("lam",vi);

                auto eps = ceps;
                auto p = verts.pack(dim_c<3>,pos_attr_tag,vi);
                auto seg = p - plane_root;

                auto area = ktris("area",kti) + verts("area",vi);

                auto fc = zs::vec<T,3>::zeros();
                auto Hc = zs::vec<T,3,3>::zeros();
                auto dist = seg.dot(plane_nrm) - eps;
                if(dist < (T)0){
                    fc = -dist * mu * intensity * plane_nrm;
                    Hc = mu * intensity * dyadic_prod(plane_nrm,plane_nrm);
                }

                gh_buffer.tuple(dim_c<3>,"grad",ci) = fc;
                gh_buffer.tuple(dim_c<3,3>,"H",ci) = Hc;
                // gh_buffer.tuple(dim_c<3>,"grad",ci) = zs::vec<T,3>::zeros();
                // gh_buffer.tuple(dim_c<3,3>,"H",ci) = zs::vec<T,3,3>::zeros();
                gh_buffer("inds",ci) = zs::reinterpret_bits<T>((int)vi);                

#endif
                       
        });
    }



template<int MAX_KINEMATIC_COLLISION_PAIRS,
    typename Pol,
    typename PosTileVec,
    typename SurfPointTileVec,
    typename SurfLineTileVec,
    typename SurfTriTileVec,
    typename SurfLineNrmTileVec,
    typename SurfTriNrmTileVec,
    typename KPosTileVec,
    typename KCollisionBuffer>
inline void do_kinematic_point_collision_detection(Pol& cudaPol,
    PosTileVec& verts,const zs::SmallString& xtag,
    const SurfPointTileVec& points,
    SurfLineTileVec& lines,
    SurfTriTileVec& tris,
    SurfLineNrmTileVec& nrmLines,
    SurfTriNrmTileVec& nrmTris,
    const KPosTileVec& kverts,
    KCollisionBuffer& kc_buffer,
    T in_collisionEps,T out_collisionEps,bool update_normal = true) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        auto stBvh = bvh_t{};
        auto bvs = retrieve_bounding_volumes(cudaPol,verts,tris,wrapv<3>{},(T)0.0,xtag);
        stBvh.build(cudaPol,bvs);

        auto avgl = compute_average_edge_length(cudaPol,verts,xtag,tris);
        auto bvh_thickness = 2 * avgl;    

        if(update_normal) {
            if(!calculate_facet_normal(cudaPol,verts,xtag,tris,nrmTris,"nrm")){
                throw std::runtime_error("fail updating kinematic facet normal");
            }       
            if(!COLLISION_UTILS::calculate_cell_bisector_normal(cudaPol,
                verts,xtag,
                lines,
                tris,
                nrmTris,"nrm",
                nrmLines,"nrm")){
                    throw std::runtime_error("fail calculate cell bisector normal");
            }    
        }

        TILEVEC_OPS::fill<2>(cudaPol,kc_buffer,"inds",zs::vec<int,2>::uniform(-1).template reinterpret_bits<T>());
        TILEVEC_OPS::fill(cudaPol,kc_buffer,"inverted",reinterpret_bits<T>((int)0));

        cudaPol(zs::range(kverts.size()),[in_collisionEps = in_collisionEps,
                out_collisionEps = out_collisionEps,
                verts = proxy<space>({},verts),xtag,
                lines = proxy<space>({},lines),
                tris = proxy<space>({},tris),
                nrmTris = proxy<space>({},nrmTris),
                nrmLines = proxy<space>({},nrmLines),
                kverts = proxy<space>({},kverts),
                kc_buffer = proxy<space>({},kc_buffer),
                stBvh = proxy<space>(stBvh),thickness = bvh_thickness] ZS_LAMBDA(int kvi) mutable {

                    auto kp = kverts.pack(dim_c<3>,"x",kvi);
                    auto bv = bv_t{get_bounding_box(kp - thickness,kp + thickness)};

                    int nm_collision_pairs = 0;
                    auto process_kinematic_vertex_face_collision_pairs = [&](int stI) {
                        if(nm_collision_pairs >= MAX_KINEMATIC_COLLISION_PAIRS)
                            return;
                        auto tri = tris.pack(dim_c<3>,"inds",stI).reinterpret_bits(int_c);
                        for(int i = 0;i != 3;++i)
                            if(verts("k_active",tri[i]) < 1e-6)
                                return;

                        auto average_thickness = (T)0.0;
                        if(verts.hasProperty("k_thickness")){
                            // average_thickness = (T)0.0;
                            for(int i = 0;i != 3;++i)
                                average_thickness += verts("k_thickness",tri[i])/(T)3.0;
                        }



                        if(verts.hasProperty("is_verted")) {

                            for(int i = 0;i != 3;++i)
                                if(reinterpret_bits<int>(verts("is_inverted",tri[i])))
                                    return;

                        }

                        T dist = (T)0.0;

                        // if(tri[0] > 5326 || tri[1] > 5326 || tri[2] > 5326){
                        //     printf("invalid tri detected : %d %d %d\n",tri[0],tri[1],tri[2]);
                        //     return;
                        // }

                        auto nrm = nrmTris.pack(dim_c<3>,"nrm",stI);
                        auto seg = kp - verts.pack(dim_c<3>,xtag,tri[0]);


                        auto t0 = verts.pack(dim_c<3>,xtag,tri[0]);
                        auto t1 = verts.pack(dim_c<3>,xtag,tri[1]);
                        auto t2 = verts.pack(dim_c<3>,xtag,tri[2]);

                        auto e01 = (t0 - t1).norm();
                        auto e02 = (t0 - t2).norm();
                        auto e12 = (t1 - t2).norm();

                        T barySum = (T)1.0;
                        T distance = LSL_GEO::pointTriangleDistance(t0,t1,t2,kp,barySum);

                        dist = seg.dot(nrm);
                        // increase the stability, the tri must already in collided in the previous frame before been penerated in the current frame
                        // if(dist > 0 && tris("collide",stI) < 0.5)
                        //     return;

                        auto collisionEps = dist < 0 ? out_collisionEps * ((T)1.0 + average_thickness) : in_collisionEps;

                        if(barySum > 1.1)
                            return;

                        if(distance > collisionEps)
                            return;

                        // if(dist < -(avge * inset_ratio + 1e-6) || dist > (outset_ratio * avge + 1e-6))
                        //     return;

                        // if the triangle cell is too degenerate
                        if(!LSL_GEO::pointProjectsInsideTriangle(t0,t1,t2,kp))
                            for(int i = 0;i != 3;++i) {
                                auto bisector_normal = get_bisector_orient(lines,tris,nrmLines,"nrm",stI,i);
                                // auto test = bisector_normal.cross(nrm).norm() < 1e-2;
                                seg = kp - verts.pack(dim_c<3>,xtag,tri[i]);
                                if(bisector_normal.dot(seg) < 0)
                                    return;
                            }

                        kc_buffer.template tuple<2>("inds",kvi * MAX_KINEMATIC_COLLISION_PAIRS + nm_collision_pairs) = zs::vec<int,2>(kvi,stI).template reinterpret_bits<T>();
                        auto vertexFaceCollisionAreas = /*tris("area",stI) + */kverts("area",kvi); 
                        kc_buffer("area",kvi * MAX_KINEMATIC_COLLISION_PAIRS + nm_collision_pairs) = vertexFaceCollisionAreas;   
                        // if(vertexFaceCollisionAreas < 0)
                        //     printf("negative face area detected\n");  
                        int is_inverted = dist > (T)0.0 ? 1 : 0;  
                        kc_buffer("inverted",kvi * MAX_KINEMATIC_COLLISION_PAIRS + nm_collision_pairs) = reinterpret_bits<T>(is_inverted);            
                        nm_collision_pairs++;  
                    };
                    stBvh.iter_neighbors(bv,process_kinematic_vertex_face_collision_pairs);
            });
}


template<typename Pol,
    typename PosTileVec,
    typename GradHessianTileVec>
inline void evaluate_fp_collision_grad_and_hessian(
    Pol& cudaPol,
    const PosTileVec& verts,const zs::SmallString& xtag,
    const zs::Vector<zs::vec<int,4>>& csPT,
    int nm_csPT,
    GradHessianTileVec& gh_buffer,
    T in_collisionEps,T out_collisionEps,
    T collisionStiffness,
    T mu,T lambda) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        gh_buffer.resize(nm_csPT);
        cudaPol(zs::range(nm_csPT),[
            gh_buffer = proxy<space>({},gh_buffer),
            in_collisionEps = in_collisionEps,
            out_collisionEps = out_collisionEps,
            verts = proxy<space>({},verts),
            csPT = proxy<space>(csPT),
            mu = mu,lam = lambda,
            stiffness = collisionStiffness,
            xtag = xtag] ZS_LAMBDA(int ci) mutable {
                auto inds = csPT[ci];
                gh_buffer.tuple(dim_c<4>,"inds",ci) = inds.reinterpret_bits(float_c);
                vec3 cv[4] = {};
                for(int i = 0;i != 4;++i)
                    cv[i] = verts.pack(dim_c<3>,xtag,inds[i]);

                auto ceps = out_collisionEps;
                auto alpha = stiffness;
                auto beta = (T)1.0;

                auto cforce = -alpha * beta * VERTEX_FACE_SQRT_COLLISION::gradient(cv,mu,lam,ceps);
                auto K = alpha * beta * VERTEX_FACE_SQRT_COLLISION::hessian(cv,mu,lam,ceps);

                gh_buffer.tuple(dim_c<12>,"grad",ci) = cforce/* + dforce*/;
                gh_buffer.tuple(dim_c<12 * 12>,"H",ci) = K/* + C/dt*/;
                if(isnan(K.norm())){
                    printf("nan cK detected : %d\n",ci);
                }
        });
}

template<typename Pol,
    typename PosTileVec,
    typename FPCollisionBuffer,
    typename GradHessianTileVec>
inline void evaluate_fp_collision_grad_and_hessian(
    Pol& cudaPol,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,T dt,
    const FPCollisionBuffer& fp_collision_buffer,// recording all the fp collision pairs
    GradHessianTileVec& gh_buffer,int offset,
    T in_collisionEps,T out_collisionEps,
    T collisionStiffness,
    T mu,T lambda,T kd_theta) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
 
        int start = offset;
        int fp_size = fp_collision_buffer.size(); 

        TILEVEC_OPS::fill_range(cudaPol,gh_buffer,"H",(T)0.0,start,fp_size);
        TILEVEC_OPS::fill_range(cudaPol,gh_buffer,"grad",(T)0.0,start,fp_size); 

        // std::cout << "inds size compair : " << fp_collision_buffer.getPropertySize("inds") << "\t" << gh_buffer.getPropertySize("inds") << std::endl;

        TILEVEC_OPS::copy(cudaPol,fp_collision_buffer,"inds",gh_buffer,"inds",start); 

        cudaPol(zs::range(fp_size),
            [verts = proxy<space>({},verts),xtag,vtag,dt,kd_theta,
                fp_collision_buffer = proxy<space>({},fp_collision_buffer),
                gh_buffer = proxy<space>({},gh_buffer),
                in_collisionEps = in_collisionEps,
                out_collisionEps = out_collisionEps,
                stiffness = collisionStiffness,
                mu = mu,lam = lambda,start = start] ZS_LAMBDA(int cpi) mutable {
                auto inds = fp_collision_buffer.template pack<4>("inds",cpi).reinterpret_bits(int_c);
                for(int j = 0;j != 4;++j)
                    if(inds[j] < 0)
                        return;
                vec3 cv[4] = {};
                for(int j = 0;j != 4;++j)
                    cv[j] = verts.template pack<3>(xtag,inds[j]);
            
                // auto is_inverted = reinterpret_bits<int>(fp_collision_buffer("inverted",cpi));
                // auto ceps = is_inverted ? in_collisionEps : out_collisionEps;

                auto ceps = out_collisionEps;
                // ceps += (T)1e-2 * ceps;

                auto alpha = stiffness;
                auto beta = fp_collision_buffer("area",cpi);
          
                auto cforce = -alpha * beta * VERTEX_FACE_SQRT_COLLISION::gradient(cv,mu,lam,ceps);
                auto K = alpha * beta * VERTEX_FACE_SQRT_COLLISION::hessian(cv,mu,lam,ceps);

                // gh_buffer.template tuple<12>("grad",cpi + start) = -alpha * beta * VERTEX_FACE_SQRT_COLLISION::gradient(cv,mu,lam,ceps);
                // gh_buffer.template tuple<12*12>("H",cpi + start) =  alpha * beta * VERTEX_FACE_SQRT_COLLISION::hessian(cv,mu,lam,ceps); 
                
                
                // adding rayleigh damping term
                // vec3 v0[4] = {verts.pack(dim_c<3>,vtag, inds[0]),
                // verts.pack(dim_c<3>,vtag, inds[1]),
                // verts.pack(dim_c<3>,vtag, inds[2]),
                // verts.pack(dim_c<3>,vtag, inds[3])}; 
                // auto vel = COLLISION_UTILS::flatten(v0); 

                // auto C = K * kd_theta;
                // auto dforce = -C * vel;
                gh_buffer.template tuple<12>("grad",cpi + start) = cforce/* + dforce*/;
                gh_buffer.template tuple<12*12>("H",cpi + start) = K/* + C/dt*/;
                if(isnan(K.norm())){
                    printf("nan cK detected : %d\n",cpi);
                }
        });
}

// TODO: add damping collision term
template<typename Pol,
    typename TetTileVec,
    typename PosTileVec,
    typename SurfTriTileVec,
    typename FPCollisionBuffer,
    typename GradHessianTileVec>
inline void evaluate_kinematic_fp_collision_grad_and_hessian(
    Pol& cudaPol,
    const TetTileVec& eles,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,T dt,
    const SurfTriTileVec& tris,
    const PosTileVec& kverts,
    const FPCollisionBuffer& kc_buffer,
    GradHessianTileVec& gh_buffer,int offset,
    T in_collisionEps,T out_collisionEps,
    T collisionStiffness,
    T mu,T lambda,T kd_theta) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        int start = offset;
        int fp_size = kc_buffer.size();

        // TILEVEC_OPS::fill_range(cudaPol,gh_buffer,"H",(T)0.0,start,fp_size);
        // TILEVEC_OPS::fill_range(cudaPol,gh_buffer,"grad",(T)0.0,start,fp_size);

        // get only the dynamic object's dofs
        // TILEVEC_OPS::copy(cudaPol,kc_buffer,"inds",gh_buffer,"inds",start);
        // cudaPol(zs::range(fp_size),
        //     [gh_buffer = proxy<space>({},gh_buffer),start = start] ZS_LAMBDA(int fpi) mutable {
        //         gh_buffer("inds",0,start + fpi) = gh_buffer("inds",1,start + fpi);
        //         auto tmp = gh_buffer("inds",2,start + fpi);
        //         gh_buffer("inds",2,start + fpi) = gh_buffer("inds",3,start + fpi);
        //         gh_buffer("inds",3,start + fpi) = tmp;
        // });


        cudaPol(zs::range(fp_size),
            [verts = proxy<space>({},verts),xtag,vtag,dt,kd_theta,
                eles = proxy<space>({},eles),
                tris = proxy<space>({},tris),
                kverts = proxy<space>({},kverts),
                kc_buffer = proxy<space>({},kc_buffer),
                gh_buffer = proxy<space>({},gh_buffer),start,
                in_collisionEps = in_collisionEps,
                out_collisionEps = out_collisionEps,
                stiffness = collisionStiffness,
                mu = mu,lam = lambda] ZS_LAMBDA(int cpi) mutable {
                auto inds = kc_buffer.pack(dim_c<2>,"inds",cpi).reinterpret_bits(int_c);
                // auto oinds = kc_buffer.pack(dim_c<4>,"inds",cpi).reinterpret_bits(int_c);
                for(int i = 0;i != 2;++i)
                    if(inds[i] < 0)
                        return;
                vec3 cv[4] = {};
                cv[0] = kverts.pack(dim_c<3>,"x",inds[0]);
                auto tri = tris.pack(dim_c<3>,"inds",inds[1]).reinterpret_bits(int_c);
                for(int j = 1;j != 4;++j)
                    cv[j] = verts.template pack<3>(xtag,tri[j-1]);
                
                // vec3 cvel[4] = {};
                // cvel[0] = vec3::zeros();
                // for(int j = 1;j != 4;++j)
                //     cvel[j] = verts.template pack<3>(vel_tag,inds[j]);

                // auto is_inverted = reinterpret_bits<int>(kc_buffer("inverted",cpi));
                auto average_thickness = (T)0.0;
                if(verts.hasProperty("k_thickness")){
                    // average_thickness = (T)0.0;
                    for(int i = 0;i != 3;++i)
                        average_thickness += verts("k_thickness",tri[i])/(T)3.0;
                }


                auto ceps = out_collisionEps * ((T)1.0 + average_thickness);
                auto alpha = stiffness;
                auto beta = kc_buffer("area",cpi);

                // change the 

                auto cgrad = -alpha * beta * VERTEX_FACE_SQRT_COLLISION::gradient(cv,mu,lam,ceps,true);
                auto cH = alpha * beta * VERTEX_FACE_SQRT_COLLISION::hessian(cv,mu,lam,ceps,true);

                auto ei = reinterpret_bits<int>(tris("ft_inds",inds[1]));
                if(ei < 0)
                    return;
                // auto cp = gh_buffer.pack(dim_c<2>,"inds",ei).reinterpret_bits(int_c);
                // auto pidx = cp[0];
                // auto tri = tris.pack(dim_c<3>,"inds",cp[1]).reinterpret_bits(int_c);
                auto tet = eles.pack(dim_c<4>,"inds",ei).reinterpret_bits(int_c);
                auto inds_reorder = zs::vec<int,3>::zeros();
                for(int i = 0;i != 3;++i){
                    auto idx = tri[i];
                    for(int j = 0;j != 4;++j)
                        if(idx == tet[j])
                            inds_reorder[i] = j;
                }

                vec3 v0[4] = {zs::vec<T,3>::zeros(),
                verts.pack(dim_c<3>,vtag, tri[0]),
                verts.pack(dim_c<3>,vtag, tri[1]),
                verts.pack(dim_c<3>,vtag, tri[2])}; 
                auto vel = COLLISION_UTILS::flatten(v0);

                auto C = cH * kd_theta;
                auto dforce = -C * vel;

                cgrad += dforce;
                cH += C/dt;

                // gh_buffer.template tuple<12>("grad",cpi + start) = cforce + dforce;
                // gh_buffer.template tuple<12*12>("H",cpi + start) = K + C/dt;

                for(int i = 3;i != 12;++i){
                    int d0 = i % 3;
                    int row = inds_reorder[i/3 - 1]*3 + d0;
                    atomic_add(exec_cuda,&gh_buffer("grad",row,ei),cgrad[i]);
                    for(int j = 3;j != 12;++j){
                        int d1 = j % 3;
                        int col = inds_reorder[j/3 - 1]*3 + d1;
                        if(row >= 12 || col >= 12){
                            printf("invalid row = %d and col = %d %d %d detected %d %d %d\n",row,col,i/3,j/3,
                                inds_reorder[0],
                                inds_reorder[1],
                                inds_reorder[2]);
                        }
                        atomic_add(exec_cuda,&gh_buffer("H",row*12 + col,ei),cH(i,j));
                    }                    
                }
                // for(int i = 1;i != 4;++i){ 
                //     auto idx = inds[i];
                //     for(int j = 0;j != 4;++j){
                //         if(idx == tet[j]) {
                //             for(int d = 0;d != 3;++d)
                //                 atomic_add(exec_cuda,&gh_buffer("grad",j*3 + d,ei),cgrad[i * 3 + d]);
                //         }
                //     }
                    
                //     gh_buffer("grad",i,cpi + start) = cgrad[i];

                // }
                // for(int i = 3;i != 12;++i)
                //     for(int j = 3;j != 12;++j)
                //         gh_buffer("H",i * 12 + j,cpi + start) = cH(i,j);
                // auto test_ind = gh_buffer.pack(dim_c<4>,"inds",start + cpi).reinterpret_bits(int_c);
                // auto cgrad_norm = cgrad.norm();
                // auto cH_norm = cH.norm();
                // printf("find_kinematic_collision[%d %d %d %d] : %f %f\n",inds[0],inds[1],inds[2],inds[3],(float)alpha,(float)beta);
        });
}

// template<typename Pol,
//     typename PosTileVec,
//     typename EECollisionBuffer,
//     typename GradHessianTileVec>
// void evaluate_ee_collision_grad_and_hessian(Pol& cudaPol,
//     const PosTileVec& verts,const zs::SmallString& xtag,
//     const EECollisionBuffer& ee_collision_buffer,
//     GradHessianTileVec& gh_buffer,int offset,
//     T in_collisionEps,T out_collisionEps,
//     T collisionStiffness,
//     T mu,T lambda) {
//         using namespace zs;
//         constexpr auto space = execspace_e::cuda;

//         int start = offset;
//         int ee_size = ee_collision_buffer.size();

//         TILEVEC_OPS::fill_range(cudaPol,gh_buffer,"H",(T)0.0,start,ee_size);
//         TILEVEC_OPS::fill_range(cudaPol,gh_buffer,"grad",(T)0.0,start,ee_size);
//         TILEVEC_OPS::copy(cudaPol,ee_collision_buffer,"inds",gh_buffer,"inds",start);

//         cudaPol(zs::range(ee_size),[
//             verts = proxy<space>({},verts),xtag,
//             in_collisionEps,out_collisionEps,
//             ee_collision_buffer = proxy<space>({},ee_collision_buffer),
//             gh_buffer = proxy<space>({},gh_buffer),
//             start = start,
//             stiffness = collisionStiffness,mu = mu,lam = lambda] ZS_LAMBDA(int eei) mutable {
//                 auto inds = ee_collision_buffer.template pack<4>("inds",eei).reinterpret_bits(int_c);
//                 for(int i = 0;i != 4;++i)
//                     if(inds[i] < 0)
//                         return;
//                 for(int j = 0;j != 4;++j){
//                     auto active = verts("active",inds[j]);
//                     if(active < 1e-6)
//                         return;
//                 }  
//                 vec3 cv[4] = {};
//                 for(int j = 0;j != 4;++j)
//                     cv[j] = verts.template pack<3>(xtag,inds[j]);       

//                 auto is_inverted = reinterpret_bits<int>(ee_collision_buffer("inverted",eei));
//                 auto ceps = is_inverted ? in_collisionEps : out_collisionEps;

//                 auto alpha = stiffness;
//                 auto beta = ee_collision_buffer("area",eei);

//                 auto a = ee_collision_buffer.template pack<2>("abary",eei);
//                 auto b = ee_collision_buffer.template pack<2>("bbary",eei);

//                 const T tooSmall = (T)1e-6;

//                 if(is_inverted) {
//                     gh_buffer.template tuple<12>("grad",eei + start) = -alpha * beta * EDGE_EDGE_SQRT_COLLISION::gradientNegated(cv,a,b,mu,lam,ceps,tooSmall);
//                     gh_buffer.template tuple<12*12>("H",eei + start) = alpha * beta * EDGE_EDGE_SQRT_COLLISION::hessianNegated(cv,a,b,mu,lam,ceps,tooSmall);
//                     // gh_buffer.template tuple<12>("grad",eei + start) = -alpha * beta * EDGE_EDGE_COLLISION::gradientNegated(cv,a,b,mu,lam,ceps);
//                     // gh_buffer.template tuple<12*12>("H",eei + start) = alpha * beta * EDGE_EDGE_COLLISION::hessianNegated(cv,a,b,mu,lam,ceps);
//                 }else {
//                     gh_buffer.template tuple<12>("grad",eei + start) = -alpha * beta * EDGE_EDGE_SQRT_COLLISION::gradient(cv,a,b,mu,lam,ceps,tooSmall);
//                     gh_buffer.template tuple<12*12>("H",eei + start) = alpha * beta * EDGE_EDGE_SQRT_COLLISION::hessian(cv,a,b,mu,lam,ceps,tooSmall);  
//                     // gh_buffer.template tuple<12>("grad",eei + start) = -alpha * beta * EDGE_EDGE_COLLISION::gradient(cv,a,b,mu,lam,ceps);
//                     // gh_buffer.template tuple<12*12>("H",eei + start) = alpha * beta * EDGE_EDGE_COLLISION::hessian(cv,a,b,mu,lam,ceps);                  
//                 }
//         });
//     }


};

};