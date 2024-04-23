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
// #include "../../geometry/kernel/calculate_bisector_normal.hpp"

#include "../../geometry/kernel/tiled_vector_ops.hpp"
#include "../../geometry/kernel/geo_math.hpp"


#include "../../geometry/kernel/calculate_edge_normal.hpp"

#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bvs.hpp"
#include "zensim/container/Bvtt.hpp"

#include "vertex_face_sqrt_collision.hpp"
#include "vertex_face_collision.hpp"
#include "../../geometry/kernel/topology.hpp"
// #include "../../geometry/kernel/intersection.hpp"
#include "../../geometry/kernel/global_intersection_analysis.hpp"

#include "collision_utils.hpp"

#include "../Ccds.hpp"
// #include "edge_edge_sqrt_collision.hpp"
// #include "edge_edge_collision.hpp"

namespace zeno { namespace COLLISION_UTILS {

// #define USE_INTERSECTION

using T = float;
using bvh_t = zs::LBvh<3,int,T>;
using bv_t = zs::AABBBox<3, T>;
using vec3 = zs::vec<T, 3>;
using vec4 = zs::vec<T,4>;
using vec4i = zs::vec<int,4>;

#define COLLISION_AMONG_SAME_GROUP 1
#define COLLISION_AMONG_DIFFERENT_GROUP 2

template<typename Pol,
    typename PosTileVec,
    typename TriTileVec,
    typename HalfEdgeTileVec,
    typename PTHashMap,
    typename TriBvh,
    typename CollisionBuffer,
    typename T = typename PosTileVec::value_type>
void calc_imminent_self_PT_collision_impulse(Pol& pol,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,
    const TriTileVec& tris,
    const HalfEdgeTileVec& halfedges,
    const REAL& thickness,
    size_t buffer_offset,
    const TriBvh& triBvh,
    CollisionBuffer& imminent_collision_buffer,
    PTHashMap& csPT) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6;
        csPT.reset(pol,true);

        pol(zs::range(verts.size()),[
            xtag = xtag,
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            thickness = thickness,
            eps = eps,
            halfedges = proxy<space>({},halfedges),
            triBvh = proxy<space>(triBvh),
            // imminent_collision_buffer = proxy<space>({},imminent_collision_buffer),
            csPT = proxy<space>(csPT)] ZS_LAMBDA(int vi) mutable {
                auto p = verts.pack(dim_c<3>,xtag,vi);
                auto bv = bv_t{get_bounding_box(p - thickness/(REAL)2,p + thickness/(REAL)2)};
                auto do_close_proximity_detection = [&](int ti) {
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    for(int i = 0;i != 3;++i)
                        if(tri[i] == vi)
                            return;

                    if(verts.hasProperty("collision_cancel") && verts("collision_cancel",vi) > 1e-3)
                        return;

                    bool has_dynamic_points = false;
                    if(verts("minv",vi) > eps)
                        has_dynamic_points = true;
                    for(int i = 0;i != 3;++i) {
                        if(verts("minv",tri[i]) > eps)
                            has_dynamic_points = true;
                        if(verts.hasProperty("collision_cancel") && verts("collision_cancel",tri[i]) > 1e-3)
                            return;
                    }
                    if(!has_dynamic_points)
                        return;

                    vec3 ts[3] = {};
                    for(int i = 0;i != 3;++i)
                        ts[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                    
                    auto tnrm = LSL_GEO::facet_normal(ts[0],ts[1],ts[2]);
                    auto seg = p - ts[0];
                    auto project_dist = zs::abs(seg.dot(tnrm));
                    if(project_dist > thickness)
                        return;
                    
                    vec3 bary{};

                    LSL_GEO::get_triangle_vertex_barycentric_coordinates(ts[0],ts[1],ts[2],p,bary);
                    for(int i = 0;i != 3;++i)
                        if(bary[i] > 1 + eps || bary[i] < -eps)
                            return;

                    auto pr = p;
                    for(int i = 0;i != 3;++i)
                        pr -= bary[i] * ts[i];

                    
                    if(pr.norm() > thickness)
                        return;

                    csPT.insert(zs::vec<int,2>{vi,ti});
                };

                triBvh.iter_neighbors(bv,do_close_proximity_detection);
        });

        pol(zip(zs::range(csPT.size()),csPT._activeKeys),[
            csPT = proxy<space>(csPT),
            imminent_collision_buffer = proxy<space>({},imminent_collision_buffer),
            xtag = xtag,vtag = vtag,
            eps = eps,
            buffer_offset = buffer_offset,
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto vi = pair[0];
                auto ti = pair[1];
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                vec4i inds{tri[0],tri[1],tri[2],vi};
                vec3 ps[4] = {};
                for(int i = 0;i != 4;++i)
                    ps[i] = verts.pack(dim_c<3>,xtag,inds[i]);
                vec3 vs[4] = {};
                for(int i = 0;i != 4;++i)
                    vs[i] = verts.pack(dim_c<3>,vtag,inds[i]);

                vec3 bary_centric{};
                LSL_GEO::get_triangle_vertex_barycentric_coordinates(ps[0],ps[1],ps[2],ps[3],bary_centric);

                for(int i = 0;i != 3;++i)
                    bary_centric[i] = bary_centric[i] < 0 ? 0 : bary_centric[i];
                auto barySum = bary_centric[0] + bary_centric[1] + bary_centric[2];
                bary_centric = bary_centric / barySum;  

                vec4 bary{-bary_centric[0],-bary_centric[1],-bary_centric[2],1};

                auto pr = vec3::zeros();
                auto vr = vec3::zeros();
                for(int i = 0;i != 4;++i) {
                    pr += bary[i] * ps[i];
                    vr += bary[i] * vs[i];
                }

                auto collision_nrm = LSL_GEO::facet_normal(ps[0],ps[1],ps[2]);
                auto align = collision_nrm.dot(pr);
                if(align < 0)
                    collision_nrm *= -1;
                // LSL_GEO::facet_normal(ps[0],ps[1],ps[2]);
                auto relative_normal_velocity = vr.dot(collision_nrm);

                imminent_collision_buffer.tuple(dim_c<4>,"bary",id + buffer_offset) = bary;
                imminent_collision_buffer.tuple(dim_c<4>,"inds",id + buffer_offset) = inds.reinterpret_bits(float_c);
                auto I = relative_normal_velocity < 0 ? relative_normal_velocity : (T)0;
                imminent_collision_buffer.tuple(dim_c<3>,"impulse",id + buffer_offset) = -collision_nrm * I;
                imminent_collision_buffer.tuple(dim_c<3>,"collision_normal",id + buffer_offset) = collision_nrm;
        });
}


template<typename Pol,
    typename PosTileVec,
    typename EdgeTileVec,
    // typename ProximityBuffer,
    typename EEHashMap,
    typename EdgeBvh,
    typename T = typename PosTileVec::value_type>
void detect_self_imminent_EE_close_proximity(Pol& pol,
    PosTileVec& verts,
    const zs::SmallString& xtag,const zs::SmallString& Xtag,const zs::SmallString& collision_group_name,
    const EdgeTileVec& edges,
    const T& thickness,
    const EdgeBvh& edgeBvh,
    EEHashMap& csEE,
    bool skip_too_close_pair_at_rest_configuration = false,
    bool use_collision_group = false) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec2i = zs::vec<int,2>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6;

        csEE.reset(pol,true);

        auto has_rest_shape = verts.hasProperty(Xtag);
        auto has_collision_group = verts.hasProperty(collision_group_name);
        auto has_collision_cancel = verts.hasProperty("collision_cancel");

        pol(zs::range(edges.size()),[
            has_collision_cancel = has_collision_cancel,
            has_rest_shape = has_rest_shape,
            has_collision_group = has_collision_group,
            skip_rest = skip_too_close_pair_at_rest_configuration,
            use_collision_group = use_collision_group,
            collision_group_name = collision_group_name,
            Xtag = Xtag,
            xtag = xtag,
            verts = proxy<space>({},verts),
            edges = proxy<space>({},edges),
            thickness = thickness,
            thickness2 = thickness * thickness,
            eps = eps,
            edgeBvh = proxy<space>(edgeBvh),
            csEE = proxy<space>(csEE)] ZS_LAMBDA(int ei) mutable {
                auto ea = edges.pack(dim_c<2>,"inds",ei,int_c);
                if(has_collision_cancel)
                    for(int i = 0;i != 2;++i)
                        if(verts("collision_cancel",ea[i]) > 1e-3)
                            return;

                vec3 pas[2] = {};
                for(int i = 0;i != 2;++i)
                    pas[i] = verts.pack(dim_c<3>,xtag,ea[i]);
                auto bv = bv_t{get_bounding_box(pas[0],pas[1])};
                bv._max += thickness/2;
                bv._min -= thickness/2;
                auto la = (pas[0] - pas[1]).norm();

                auto do_close_proximity_detection = [&](int nei) mutable {
                    if(ei >= nei)
                        return;
                    auto eb = edges.pack(dim_c<2>,"inds",nei,int_c);
                    for(int i = 0;i != 2;++i){
                        if(eb[i] == ea[0] || eb[i] == ea[1])
                            return;
                    }

                    if(has_collision_cancel)
                        for(int i = 0;i != 2;++i)
                            if(verts.hasProperty("collision_cancel") && verts("collision_cancel",eb[i]) > 1e-3)
                                return;

                    if(verts.hasProperty("minv")) {
                        bool has_dynamic_points = false;
                        for(int i = 0;i != 2;++i) {
                            if(verts("minv",ea[i]) > eps)
                                has_dynamic_points = true;
                            if(verts("minv",eb[i]) > eps)
                                has_dynamic_points = true;
                        }
                        if(!has_dynamic_points)
                            return;
                    }

                    vec3 pbs[2] = {};
                    for(int i = 0;i != 2;++i)
                        pbs[i] = verts.pack(dim_c<3>,xtag,eb[i]);

                    vec2 edge_bary{};

                    if((pas[0] - pas[1]).cross(pbs[0] - pbs[1]).norm() < eps)
                        return;


#ifdef USE_INTERSECTION

                    int type{};
                    if(LSL_GEO::get_edge_edge_intersection_barycentric_coordinates(pas[0],pas[1],pbs[0],pbs[1],edge_bary,type) > thickness2)
                        return;
#else
                    LSL_GEO::get_edge_edge_barycentric_coordinates(pas[0],pas[1],pbs[0],pbs[1],edge_bary);
                    for(int i = 0;i != 2;++i) {
                        if(edge_bary[i] < -eps || edge_bary[i] > 1 + eps)
                            return;
                        if(edge_bary[1] < -eps || edge_bary[1] > 1 + eps)
                            return;
                    }
#endif
                    vec4 bary{edge_bary[0] - 1,-edge_bary[0],1 - edge_bary[1],edge_bary[1]};

#ifndef USE_INTERSECTION
                    auto rp = bary[0] * pas[0] + bary[1] * pas[1] + bary[2] * pbs[0] + bary[3] * pbs[1];
                    if(rp.norm() > thickness) {
                        return;
                    }
#endif
                    vec4i inds{ea[0],ea[1],eb[0],eb[1]};
                    // for(int i = 0;i != 4;++i)
                    //     verts("dcd_collision_tag",inds[i]) = 1;

                    if(has_rest_shape && skip_rest) {
                        vec3 rpas[2] = {};
                        vec3 rpbs[2] = {};
                        for(int i = 0;i != 2;++i) {
                            rpas[i] = verts.pack(dim_c<3>,Xtag,ea[i]);
                            rpbs[i] = verts.pack(dim_c<3>,Xtag,eb[i]);
                        }

                        auto is_same_collision_group = true;
                        if(use_collision_group) 
                            is_same_collision_group = zs::abs(verts(collision_group_name,ea[0]) - verts(collision_group_name,eb[0])) < 0.1;


                        if(LSL_GEO::get_edge_edge_distance(rpas[0],rpas[1],rpbs[0],rpbs[1]) < thickness && is_same_collision_group)
                            return;
                    }

                    auto id = csEE.insert(vec2i{ei,nei});
                    // proximity_buffer.tuple(dim_c<4>,"bary",id + buffer_offset) = bary;
                    // proximity_buffer.tuple(dim_c<4>,"inds",id + buffer_offset) = inds.reinterpret_bits(float_c);
                    // proximity_buffer("type",id + buffer_offset) = zs::reinterpret_bits<float>((int)1);

                };
                edgeBvh.iter_neighbors(bv,do_close_proximity_detection);
        });
}


template<typename Pol,
    typename PosTileVec,
    typename TriTileVec,
    typename PTHashMap,
    typename TriBvh,
    typename T = typename PosTileVec::value_type>
void detect_imminent_PKT_close_proximity(Pol& pol,
    const PosTileVec& verts,const zs::SmallString& xtag,
    const PosTileVec& kverts,const zs::SmallString& kxtag,
    const TriTileVec& ktris,
    const T& thickness,
    const TriBvh& ktriBvh,
    PTHashMap& csPKT) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto eps = (T)1e-6;
        csPKT.reset(pol,true);
        pol(zs::range(verts.size()),[
            eps = eps,
            xtag = zs::SmallString(xtag),
            verts = proxy<space>({},verts),
            kxtag = zs::SmallString(kxtag),
            kverts = proxy<space>({},kverts),
            ktris = proxy<space>({},ktris),
            thickness = thickness,
            thickness2 = thickness * thickness,
            ktriBvh = proxy<space>(ktriBvh),
            csPKT = proxy<space>(csPKT)] ZS_LAMBDA(int vi) mutable {
                if(verts.hasProperty("collision_cancel") && verts("collision_cancel",vi) > 1e-3)
                        return;
                auto p = verts.pack(dim_c<3>,xtag,vi);
                auto bv = bv_t{get_bounding_box(p - thickness/(T)2,p + thickness/(T)2)};

                zs::vec<T,3> kts[3] = {};

                auto is_dynamic_vert = verts.hasProperty("minv") ? (verts("minv",vi) > eps) : true;
                auto do_close_proximity_detection = [&](int kti) {
                    auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);
                    for(int i = 0;i != 3;++i)
                        if(kverts.hasProperty("collision_cancel") && kverts("collision_cancel",ktri[i]) > eps)
                            return;

                    auto is_dynamic_ktri = true;
                    if(kverts.hasProperty("minv")) {
                        bool is_dynamic_ktri = false;
                        for(int i = 0;i != 3;++i) {
                            if(kverts("minv",ktri[i]) > eps)
                                is_dynamic_ktri = true;
                        }
                    }
                    if(!is_dynamic_vert && !is_dynamic_ktri)
                        return;

                    for(int i = 0;i != 3;++i)
                        kts[i] = kverts.pack(dim_c<3>,kxtag,ktri[i]);
                    
                    vec3 ktri_bary{};
#ifdef USE_INTERSECTION
                    if(LSL_GEO::get_vertex_triangle_intersection_barycentric_coordinates(p,ts[0],ts[1],ts[2],tri_bary) > thickness2)
                        return;
#else
                    LSL_GEO::get_triangle_vertex_barycentric_coordinates(kts[0],kts[1],kts[2],p,ktri_bary);
                    for(int i = 0;i != 3;++i)
                        if(ktri_bary[i] > 1 + eps || ktri_bary[i] < -eps)
                            return;
#endif
                    vec4 bary{-ktri_bary[0],-ktri_bary[1],-ktri_bary[2],1};
#ifndef USE_INTERSECTION
                    auto rp = kts[0] * bary[0] + kts[1] * bary[1] + kts[2] * bary[2] + p * bary[3];
                    if(rp.norm() > thickness)
                        return;
#endif
                    auto id = csPKT.insert(zs::vec<int,2>{vi,kti});
                };
                ktriBvh.iter_neighbors(bv,do_close_proximity_detection);
        });
}


template<typename Pol,
    typename PosTileVec,
    typename EdgeTileVec,
    typename EEHashMap,
    typename EdgeBvh,
    typename T = typename PosTileVec::value_type>
void detect_imminent_EKE_close_proximity(Pol& pol,
    const PosTileVec& verts,const zs::SmallString& xtag,
    const EdgeTileVec& edges,
    const PosTileVec& kverts,const zs::SmallString& kxtag,
    const EdgeTileVec& kedges,
    const T& thickness,
    const EdgeBvh& kedgeBvh,
    EEHashMap& csEKE) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec2i = zs::vec<int,2>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6;

        csEKE.reset(pol,true);

        pol(zs::range(edges.size()),[
            xtag = xtag,
            verts = proxy<space>({},verts),
            edges = proxy<space>({},edges),
            kxtag = kxtag,
            kverts = proxy<space>({},kverts),
            kedges = proxy<space>({},kedges),
            thickness = thickness,
            thickness2 = thickness * thickness,
            eps = eps,
            kedgeBvh = proxy<space>(kedgeBvh),
            csEKE = proxy<space>(csEKE)] ZS_LAMBDA(int ei) mutable {
                auto edge = edges.pack(dim_c<2>,"inds",ei,int_c);
                for(int i = 0;i != 2;++i)
                    if(verts.hasProperty("collision_cancel") && verts("collision_cancel",edge[i]) > 1e-3)
                        return;

                vec3 ps[2] = {};
                for(int i = 0;i != 2;++i)
                    ps[i] = verts.pack(dim_c<3>,xtag,edge[i]);
                auto bv = bv_t{get_bounding_box(ps[0],ps[1])};
                bv._max += thickness/2;
                bv._min -= thickness/2;
                auto edge_len = (ps[0] - ps[1]).norm();

                auto is_dynamic_edge = true;
                if(verts.hasProperty("minv")) {
                    is_dynamic_edge = false;
                    for(int i = 0;i != 2;++i) {
                        if(verts("minv",edge[i]) > eps)
                            is_dynamic_edge = true;
                    }
                }

                auto do_close_proximity_detection = [&](int kei) mutable {
                    auto kedge = kedges.pack(dim_c<2>,"inds",kei,int_c);
                    for(int i = 0;i != 2;++i)
                        if(kverts.hasProperty("collision_cancel") && kverts("collision_cancel",kedge[i]) > 1e-3)
                            return;

                    auto is_dynamic_kedge = true;
                    if(kverts.hasProperty("minv")) {
                        is_dynamic_kedge = false;
                        for(int i = 0;i != 2;++i) {
                            if(kverts("minv",kedge[i]) > eps)
                                is_dynamic_kedge = true;
                        }
                    }

                    if(!is_dynamic_edge && !is_dynamic_kedge)
                        return;

                    vec3 kps[2] = {};
                    for(int i = 0;i != 2;++i)
                        kps[i] = kverts.pack(dim_c<3>,kxtag,kedge[i]);

                    vec2 edge_bary{};

                    if((ps[0] - ps[1]).cross(kps[0] - kps[1]).norm() < eps)
                        return;

#ifdef USE_INTERSECTION

                    int type{};
                    if(LSL_GEO::get_edge_edge_intersection_barycentric_coordinates(pas[0],pas[1],pbs[0],pbs[1],edge_bary,type) > thickness2)
                        return;
#else
                    LSL_GEO::get_edge_edge_barycentric_coordinates(ps[0],ps[1],kps[0],kps[1],edge_bary);
                    for(int i = 0;i != 2;++i) {
                        if(edge_bary[i] < -eps || edge_bary[i] > 1 + eps)
                            return;
                        if(edge_bary[1] < -eps || edge_bary[1] > 1 + eps)
                            return;
                    }
#endif
                    vec4 bary{edge_bary[0] - 1,-edge_bary[0],1 - edge_bary[1],edge_bary[1]};

#ifndef USE_INTERSECTION
                    auto rp = bary[0] * ps[0] + bary[1] * ps[1] + bary[2] * kps[0] + bary[3] * kps[1];
                    if(rp.norm() > thickness) {
                        return;
                    }
#endif
                    auto id = csEKE.insert(vec2i{ei,kei});
                };
                kedgeBvh.iter_neighbors(bv,do_close_proximity_detection);
        });
}




template<typename Pol,
    typename PosTileVec,
    typename EdgeTileVec,
    typename ProximityBuffer,
    typename EEHashMap,
    typename EdgeBvh,
    typename T = typename PosTileVec::value_type>
void detect_self_imminent_EE_close_proximity(Pol& pol,
    PosTileVec& verts,
    const zs::SmallString& xtag,const zs::SmallString& Xtag,const zs::SmallString& collision_group_name,
    const EdgeTileVec& edges,
    const T& thickness,
    size_t buffer_offset,
    const EdgeBvh& edgeBvh,
    ProximityBuffer& proximity_buffer,
    EEHashMap& csEE,
    bool skip_too_close_pair_at_rest_configuration = false,
    bool use_collision_group = false,
    int collision_group_strategy = COLLISION_AMONG_SAME_GROUP | COLLISION_AMONG_DIFFERENT_GROUP) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec2i = zs::vec<int,2>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6;

        csEE.reset(pol,true);

        auto has_rest_shape = verts.hasProperty(Xtag);
        auto has_collision_group = verts.hasProperty(collision_group_name);
        auto has_collision_cancel = verts.hasProperty("collision_cancel");

        pol(zs::range(edges.size()),[
            collision_group_strategy = collision_group_strategy,
            has_collision_cancel = has_collision_cancel,
            has_rest_shape = has_rest_shape,
            has_collision_group = has_collision_group,
            skip_rest = skip_too_close_pair_at_rest_configuration,
            use_collision_group = use_collision_group,
            collision_group_name = collision_group_name,
            Xtag = Xtag,
            xtag = xtag,
            verts = proxy<space>({},verts),
            edges = proxy<space>({},edges),
            thickness = thickness,
            thickness2 = thickness * thickness,
            buffer_offset = buffer_offset,
            eps = eps,
            proximity_buffer = proxy<space>({},proximity_buffer),
            edgeBvh = proxy<space>(edgeBvh),
            csEE = proxy<space>(csEE)] ZS_LAMBDA(int ei) mutable {
                auto ea = edges.pack(dim_c<2>,"inds",ei,int_c);
                if(has_collision_cancel)
                    for(int i = 0;i != 2;++i)
                        if(verts("collision_cancel",ea[i]) > 1e-3)
                            return;

                vec3 pas[2] = {};
                for(int i = 0;i != 2;++i)
                    pas[i] = verts.pack(dim_c<3>,xtag,ea[i]);
                auto bv = bv_t{get_bounding_box(pas[0],pas[1])};
                bv._max += thickness/2;
                bv._min -= thickness/2;
                auto la = (pas[0] - pas[1]).norm();

                auto do_close_proximity_detection = [&](int nei) mutable {
                    if(ei >= nei)
                        return;
                    auto eb = edges.pack(dim_c<2>,"inds",nei,int_c);
                    for(int i = 0;i != 2;++i){
                        if(eb[i] == ea[0] || eb[i] == ea[1])
                            return;
                    }

                    if(has_collision_group) {
                        if(!(collision_group_strategy & COLLISION_AMONG_DIFFERENT_GROUP)) {
                            for(int i = 0;i != 2;++i) {
                                auto eaGroup = verts(collision_group_name,ea[i]);
                                for(int j = 0;j != 2;++j){
                                    auto ebGroup = verts(collision_group_name,eb[j]);
                                    if(zs::abs(eaGroup - ebGroup) > 0.1) {
                                        return;
                                    }
                                }
                            }
                        }
                        if(!(collision_group_strategy & COLLISION_AMONG_SAME_GROUP)) {
                            for(int i = 0;i != 2;++i) {
                                auto eaGroup = verts(collision_group_name,ea[i]);
                                for(int j = 0;j != 2;++j){
                                    auto ebGroup = verts(collision_group_name,eb[j]);
                                    if(zs::abs(eaGroup - ebGroup) < 0.1) {
                                        return;
                                    }
                                }
                            }
                        }
                    }


                    if(has_collision_cancel)
                        for(int i = 0;i != 2;++i)
                            if(verts("collision_cancel",eb[i]) > 1e-3)
                                return;

                    if(verts.hasProperty("minv")) {
                        bool has_dynamic_points = false;
                        for(int i = 0;i != 2;++i) {
                            if(verts("minv",ea[i]) > eps)
                                has_dynamic_points = true;
                            if(verts("minv",eb[i]) > eps)
                                has_dynamic_points = true;
                        }
                        if(!has_dynamic_points)
                            return;
                    }

                    vec3 pbs[2] = {};
                    for(int i = 0;i != 2;++i)
                        pbs[i] = verts.pack(dim_c<3>,xtag,eb[i]);

                    vec2 edge_bary{};

                    if((pas[0] - pas[1]).cross(pbs[0] - pbs[1]).norm() < eps)
                        return;


#ifdef USE_INTERSECTION

                    int type{};
                    if(LSL_GEO::get_edge_edge_intersection_barycentric_coordinates(pas[0],pas[1],pbs[0],pbs[1],edge_bary,type) > thickness2)
                        return;
#else
                    LSL_GEO::get_edge_edge_barycentric_coordinates(pas[0],pas[1],pbs[0],pbs[1],edge_bary);
                    for(int i = 0;i != 2;++i) {
                        if(edge_bary[i] < -eps || edge_bary[i] > 1 + eps)
                            return;
                        if(edge_bary[1] < -eps || edge_bary[1] > 1 + eps)
                            return;
                    }
#endif
                    vec4 bary{edge_bary[0] - 1,-edge_bary[0],1 - edge_bary[1],edge_bary[1]};

#ifndef USE_INTERSECTION
                    auto rp = bary[0] * pas[0] + bary[1] * pas[1] + bary[2] * pbs[0] + bary[3] * pbs[1];
                    if(rp.norm() > thickness) {
                        return;
                    }
#endif
                    vec4i inds{ea[0],ea[1],eb[0],eb[1]};
                    // for(int i = 0;i != 4;++i)
                    //     verts("dcd_collision_tag",inds[i]) = 1;

                    if(has_rest_shape && skip_rest) {
                        vec3 rpas[2] = {};
                        vec3 rpbs[2] = {};
                        for(int i = 0;i != 2;++i) {
                            rpas[i] = verts.pack(dim_c<3>,Xtag,ea[i]);
                            rpbs[i] = verts.pack(dim_c<3>,Xtag,eb[i]);
                        }

                        auto is_same_collision_group = true;
                        if(use_collision_group) 
                            is_same_collision_group = zs::abs(verts(collision_group_name,ea[0]) - verts(collision_group_name,eb[0])) < 0.1;


                        if(LSL_GEO::get_edge_edge_distance(rpas[0],rpas[1],rpbs[0],rpbs[1]) < thickness && is_same_collision_group)
                            return;
                    }
 
                    auto id = csEE.insert(vec2i{ei,nei});
                    proximity_buffer.tuple(dim_c<4>,"bary",id + buffer_offset) = bary;
                    proximity_buffer.tuple(dim_c<4>,"inds",id + buffer_offset) = inds.reinterpret_bits(float_c);
                    proximity_buffer("type",id + buffer_offset) = zs::reinterpret_bits<float>((int)1);

                };
                edgeBvh.iter_neighbors(bv,do_close_proximity_detection);
        });
}


template<typename Pol,
    typename PosTileVec,
    typename TriTileVec,
    typename PTHashMap,
    typename TriBvh,
    typename ProximityBuffer,
    typename T = typename PosTileVec::value_type>
void detect_self_imminent_PT_close_proximity(Pol& pol,
    PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& Xtag,const zs::SmallString& collisionGroupTag,
    const TriTileVec& tris,
    const T& thickness,
    size_t buffer_offset,
    const TriBvh& triBvh,
    ProximityBuffer& proximity_buffer,
    PTHashMap& csPT,
    bool skip_too_close_pair_at_rest_configuration = false,
    bool use_collision_group = false,
    int collision_group_strategy = COLLISION_AMONG_SAME_GROUP | COLLISION_AMONG_DIFFERENT_GROUP) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto eps = (T)1e-6;
        // constexpr auto exec_tag = wrapv<space>{};
        csPT.reset(pol,true);

        auto has_rest_shape = verts.hasProperty(Xtag);
        auto has_collision_group = verts.hasProperty(collisionGroupTag);
        pol(zs::range(verts.size()),[
            collisionGroupTag = collisionGroupTag,
            has_rest_shape = has_rest_shape,
            has_collision_group = has_collision_group,
            skip_rest = skip_too_close_pair_at_rest_configuration,
            use_collision_group = use_collision_group,
            eps = eps,
            XtagOffset = verts.getPropertyOffset(Xtag),
            xtag = xtag,
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            buffer_offset = buffer_offset,
            thickness = thickness,
            thickness2 = thickness * thickness,
            triBvh = proxy<space>(triBvh),
            collision_group_strategy = collision_group_strategy,
            proximity_buffer = proxy<space>({},proximity_buffer),
            csPT = proxy<space>(csPT)] ZS_LAMBDA(int vi) mutable {

                if(verts.hasProperty("collision_cancel") && verts("collision_cancel",vi) > 1e-3)
                        return;
                auto p = verts.pack(dim_c<3>,xtag,vi);
                auto bv = bv_t{get_bounding_box(p - thickness/(T)2,p + thickness/(T)2)};

                zs::vec<T,3> ts[3] = {};
                auto do_close_proximity_detection = [&](int ti) {
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    for(int i = 0;i != 3;++i)
                        if(tri[i] == vi)
                            return;

                    if(has_collision_group) {
                        if(!(collision_group_strategy & COLLISION_AMONG_DIFFERENT_GROUP)) {
                            auto vgroup = verts(collisionGroupTag,vi);
                            for(int i = 0;i != 3;++i) {
                                auto tgroup = verts(collisionGroupTag,tri[i]);
                                // if they belong to two different groups
                                if(zs::abs(vgroup - tgroup) > 0.1) {
                                    return;
                                }
                            }
                        }
                        if(!(collision_group_strategy & COLLISION_AMONG_SAME_GROUP)) {
                            auto vgroup = verts(collisionGroupTag,vi);
                            for(int i = 0;i != 3;++i) {
                                auto tgroup = verts(collisionGroupTag,tri[i]);
                                if(zs::abs(vgroup - tgroup) < 0.1)
                                    return;
                            }
                        }
                    }

                    for(int i = 0;i != 3;++i)
                        if(verts.hasProperty("collision_cancel") && verts("collision_cancel",tri[i]) > eps)
                            return;

                    if(verts.hasProperty("minv")) {
                        bool has_dynamic_points = false;
                        if(verts("minv",vi) > eps)
                            has_dynamic_points = true;
                        for(int i = 0;i != 3;++i) {
                            if(verts("minv",tri[i]) > eps)
                                has_dynamic_points = true;
                        }
                        if(!has_dynamic_points)
                            return;
                    }

                    for(int i = 0;i != 3;++i)
                        ts[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                    
                    vec3 tri_bary{};

#ifdef USE_INTERSECTION

                    if(LSL_GEO::get_vertex_triangle_intersection_barycentric_coordinates(p,ts[0],ts[1],ts[2],tri_bary) > thickness2)
                        return;
#else
                    LSL_GEO::get_triangle_vertex_barycentric_coordinates(ts[0],ts[1],ts[2],p,tri_bary);
                    for(int i = 0;i != 3;++i)
                        if(tri_bary[i] > 1 + eps || tri_bary[i] < -eps)
                            return;
#endif

                    vec4 bary{-tri_bary[0],-tri_bary[1],-tri_bary[2],1};

#ifndef USE_INTERSECTION
                    auto rp = ts[0] * bary[0] + ts[1] * bary[1] + ts[2] * bary[2] + p * bary[3];
                    if(rp.norm() > thickness)
                        return;
#endif
                
                    if(has_rest_shape && skip_rest) {
                        auto rp = verts.pack(dim_c<3>,XtagOffset,vi);
                        vec3 rts[3] = {};
                        for(int i = 0;i != 3;++i)
                            rts[i] = verts.pack(dim_c<3>,XtagOffset,tri[i]);

                        auto is_same_collision_group = true;
                        if(use_collision_group && has_collision_group) 
                            is_same_collision_group = zs::abs(verts(collisionGroupTag,vi) - verts(collisionGroupTag,tri[0])) < 0.1;

                        // if the collision pair are initialliy closed, and belong to the same collision group, skip it
                        if(LSL_GEO::get_vertex_triangle_distance(rts[0],rts[1],rts[2],rp) < thickness && is_same_collision_group)
                            return; 
                    }
                    
                    auto id = csPT.insert(zs::vec<int,2>{vi,ti});
                    vec4i inds{tri[0],tri[1],tri[2],vi};

                    proximity_buffer.tuple(dim_c<4>,"inds",id + buffer_offset) = inds.reinterpret_bits(float_c);
                    proximity_buffer.tuple(dim_c<4>,"bary",id + buffer_offset) = bary;
                    proximity_buffer("type",id + buffer_offset) = zs::reinterpret_bits<float>((int)0);
                };

                triBvh.iter_neighbors(bv,do_close_proximity_detection);
        });
}


template<typename Pol,
    typename PosTileVec,
    typename TriTileVec,
    typename PTHashMap,
    typename TriBvh,
    // typename ProximityBuffer,
    typename T = typename PosTileVec::value_type>
void detect_self_imminent_PT_close_proximity(Pol& pol,
    PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& Xtag,const zs::SmallString& collisionGroupTag,
    const TriTileVec& tris,
    const T& thickness,
    const TriBvh& triBvh,
    PTHashMap& csPT,
    bool skip_too_close_pair_at_rest_configuration = false,
    bool use_collision_group = false,
    int collision_group_strategy = COLLISION_AMONG_SAME_GROUP | COLLISION_AMONG_DIFFERENT_GROUP) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto eps = (T)1e-6;
        // constexpr auto exec_tag = wrapv<space>{};
        csPT.reset(pol,true);

        auto has_rest_shape = verts.hasProperty(Xtag);
        auto has_collision_group = verts.hasProperty(collisionGroupTag);
        auto has_collision_cancel = verts.hasProperty("collision_cancel");

        pol(zs::range(verts.size()),[
            collisionGroupTag = collisionGroupTag,
            collision_group_strategy = collision_group_strategy,
            has_collision_cancel = has_collision_cancel,
            has_rest_shape = has_rest_shape,
            has_collision_group = has_collision_group,
            skip_rest = skip_too_close_pair_at_rest_configuration,
            use_collision_group = use_collision_group,
            eps = eps,
            Xtag = Xtag,
            xtag = zs::SmallString(xtag),
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            thickness = thickness,
            thickness2 = thickness * thickness,
            triBvh = proxy<space>(triBvh),
            csPT = proxy<space>(csPT)] ZS_LAMBDA(int vi) mutable {

                if(has_collision_cancel)
                    if(verts("collision_cancel",vi) > 1e-3)
                        return;
                auto p = verts.pack(dim_c<3>,xtag,vi);
                auto bv = bv_t{ (p - thickness/(T)2,p + thickness/(T)2)};

                
                zs::vec<T,3> ts[3] = {};
                auto do_close_proximity_detection = [&](int ti) {
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    for(int i = 0;i != 3;++i)
                        if(tri[i] == vi)
                            return;

                    if(has_collision_group) {
                        if(!(collision_group_strategy & COLLISION_AMONG_DIFFERENT_GROUP)) {
                            auto vgroup = verts(collisionGroupTag,vi);
                            for(int i = 0;i != 3;++i) {
                                auto tgroup = verts(collisionGroupTag,tri[i]);
                                // if they belong to two different groups
                                if(zs::abs(vgroup - tgroup) > 0.1) {
                                    return;
                                }
                            }
                        }
                        if(!(collision_group_strategy & COLLISION_AMONG_SAME_GROUP)) {
                            auto vgroup = verts(collisionGroupTag,vi);
                            for(int i = 0;i != 3;++i) {
                                auto tgroup = verts(collisionGroupTag,tri[i]);
                                if(zs::abs(vgroup - tgroup) < 0.1)
                                    return;
                            }
                        }
                    }

                    if(has_collision_cancel)
                        for(int i = 0;i != 3;++i)
                            if(verts("collision_cancel",tri[i]) > eps)
                                return;

                    if(verts.hasProperty("minv")) {
                        bool has_dynamic_points = false;
                        if(verts("minv",vi) > eps)
                            has_dynamic_points = true;
                        for(int i = 0;i != 3;++i) {
                            if(verts("minv",tri[i]) > eps)
                                has_dynamic_points = true;
                        }
                        if(!has_dynamic_points)
                            return;
                    }

                    for(int i = 0;i != 3;++i)
                        ts[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                    
                    vec3 tri_bary{};

#ifdef USE_INTERSECTION

                    if(LSL_GEO::get_vertex_triangle_intersection_barycentric_coordinates(p,ts[0],ts[1],ts[2],tri_bary) > thickness2)
                        return;
#else
                    LSL_GEO::get_triangle_vertex_barycentric_coordinates(ts[0],ts[1],ts[2],p,tri_bary);
                    for(int i = 0;i != 3;++i)
                        if(tri_bary[i] > 1 + eps || tri_bary[i] < -eps)
                            return;
#endif

                    vec4 bary{-tri_bary[0],-tri_bary[1],-tri_bary[2],1};

#ifndef USE_INTERSECTION
                    auto rp = ts[0] * bary[0] + ts[1] * bary[1] + ts[2] * bary[2] + p * bary[3];
                    if(rp.norm() > thickness)
                        return;
#endif
             
                    if(has_rest_shape && skip_rest) {
                        auto rp = verts.pack(dim_c<3>,Xtag,vi);
                        vec3 rts[3] = {};
                        for(int i = 0;i != 3;++i)
                            rts[i] = verts.pack(dim_c<3>,Xtag,tri[i]);

                        auto is_same_collision_group = true;
                        if(use_collision_group && has_collision_group) 
                            is_same_collision_group = zs::abs(verts(collisionGroupTag,vi) - verts(collisionGroupTag,tri[0])) < 0.1;

                        // if the collision pair are initialliy closed, and belong to the same collision group, skip it
                        if(LSL_GEO::get_vertex_triangle_distance(rts[0],rts[1],rts[2],rp) < thickness && is_same_collision_group)
                            return; 
                    }             

                    auto id = csPT.insert(zs::vec<int,2>{vi,ti});
                };

                triBvh.iter_neighbors(bv,do_close_proximity_detection);
        });
}


template<typename Pol,
    typename PosTileVec,
    typename EdgeTileVec,
    typename CollisionBuffer,
    typename EEHashMap,
    typename EdgeBvh,
    typename T = typename PosTileVec::value_type>
void calc_imminent_self_EE_collision_impulse(Pol& pol,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,
    const EdgeTileVec& edges,
    const REAL& thickness,
    size_t buffer_offset,
    const EdgeBvh& edgeBvh,
    CollisionBuffer& imminent_collision_buffer,EEHashMap& csEE,
    int collision_group_strategy = COLLISION_AMONG_SAME_GROUP | COLLISION_AMONG_DIFFERENT_GROUP) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        // constexpr auto exec_tag = wrapv<space>{};
        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec2i = zs::vec<int,2>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6;
        // constexpr auto MAX_PT_COLLISION_PAIRS = 1000000;
        // zs::bht<int,2,int> csEE{verts.get_allocator(),MAX_PT_COLLISION_PAIRS};
        csEE.reset(pol,true);

        pol(zs::range(edges.size()),[
            collision_group_strategy = collision_group_strategy,
            xtag = zs::SmallString(xtag),
            verts = proxy<space>({},verts),
            edges = proxy<space>({},edges),
            thickness = thickness,
            eps = eps,
            edgeBvh = proxy<space>(edgeBvh),
            csEE = proxy<space>(csEE)] ZS_LAMBDA(int ei) mutable {
                auto ea = edges.pack(dim_c<2>,"inds",ei,int_c);
                for(int i = 0;i != 2;++i)
                    if(verts.hasProperty("collision_cancel") && verts("collision_cancel",ea[i]) > 1e-3)
                        return;

                vec3 pas[2] = {};
                for(int i = 0;i != 2;++i)
                    pas[i] = verts.pack(dim_c<3>,xtag,ea[i]);
                auto bv = bv_t{get_bounding_box(pas[0],pas[1])};
                bv._max += thickness/2;
                bv._min -= thickness/2;
                auto la = (pas[0] - pas[1]).norm();

                auto do_close_proximity_detection = [&](int nei) mutable {
                    if(ei >= nei)
                        return;
                    auto eb = edges.pack(dim_c<2>,"inds",nei,int_c);
                    for(int i = 0;i != 2;++i){
                        if(eb[i] == ea[0] || eb[i] == ea[1])
                            return;
                    }



                    for(int i = 0;i != 2;++i)
                        if(verts.hasProperty("collision_cancel") && verts("collision_cancel",eb[i]) > 1e-3)
                            return;

                    bool has_dynamic_points = false;
                    for(int i = 0;i != 2;++i) {
                        if(verts("minv",ea[i]) > eps)
                            has_dynamic_points = true;
                        if(verts("minv",eb[i]) > eps)
                            has_dynamic_points = true;
                    }
                    if(!has_dynamic_points)
                        return;

                    vec3 pbs[2] = {};
                    for(int i = 0;i != 2;++i)
                        pbs[i] = verts.pack(dim_c<3>,xtag,eb[i]);
                

                    vec2 bary{};
                    LSL_GEO::get_edge_edge_barycentric_coordinates(pas[0],pas[1],pbs[0],pbs[1],bary);

                    if(bary[0] < -eps || bary[0] > 1 + eps)
                        return;
                    if(bary[1] < -eps || bary[1] > 1 + eps)
                        return;

                    auto pr = (bary[0] - 1) * pas[0] + (-bary[0]) * pas[1] + (1 - bary[1]) * pbs[0] + (bary[1]) * pbs[1];
                    if(pr.norm() > thickness)
                        return;
                    // auto ref_bary = vec2{ra,rb};
                    // if((bary - ref_bary).norm() > eps)
                    //     printf("edge bary not aligned : %f %f %f %f\n",(float)bary[0],(float)bary[1],(float)ref_bary[0],(float)ref_bary[1]);

                    csEE.insert(vec2i{ei,nei});
                };
                edgeBvh.iter_neighbors(bv,do_close_proximity_detection);
        });

        // printf("nm_EE_collision_pairs : %d %d\n",(int)csEE.size(),(int)edges.size());
        // std::cout << "do EE collision impulse computation" << std::endl;

        // imminent_collision_buffer.resize(csEE.size() + buffer_offset);
        pol(zip(zs::range(csEE.size()),csEE._activeKeys),[
            verts = proxy<space>({},verts),
            edges = proxy<space>({},edges),
            buffer_offset  = buffer_offset,
            xtag = zs::SmallString(xtag),
            vtag = zs::SmallString(vtag),
            eps = eps,
            imminent_collision_buffer = proxy<space>({},imminent_collision_buffer)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto ei = pair[0];
                auto nei = pair[1];

                auto ea = edges.pack(dim_c<2>,"inds",ei,int_c);
                auto eb = edges.pack(dim_c<2>,"inds",nei,int_c);
                vec4i inds{ea[0],ea[1],eb[0],eb[1]};

                vec3 ps[4] = {};
                vec3 vs[4] = {};
                for(int i = 0;i != 4;++i) {
                    ps[i] = verts.pack(dim_c<3>,xtag,inds[i]);
                    vs[i] = verts.pack(dim_c<3>,vtag,inds[i]);
                }

                vec2 edge_bary{};
                LSL_GEO::get_edge_edge_barycentric_coordinates(ps[0],ps[1],ps[2],ps[3],edge_bary);
                vec4 bary{edge_bary[0] - 1,-edge_bary[0],1 - edge_bary[1],edge_bary[1]};

                auto pr = vec3::zeros();
                auto vr = vec3::zeros();
                for(int i = 0;i != 4;++i) {
                    pr += bary[i] * ps[i];
                    vr += bary[i] * vs[i];
                }

                auto collision_nrm = (ps[0] - ps[1]).cross(ps[2] - ps[3]).normalized();
                // if(collision_nrm.norm() < eps * 100) {

                // }
                auto align = collision_nrm.dot(pr);
                if(align < 0)
                    collision_nrm *= -1;
                auto relative_normal_velocity = vr.dot(collision_nrm);

                imminent_collision_buffer.tuple(dim_c<4>,"bary",ci + buffer_offset) = bary;
                imminent_collision_buffer.tuple(dim_c<4>,"inds",ci + buffer_offset) = inds.reinterpret_bits(float_c);
                auto I = relative_normal_velocity < 0 ? relative_normal_velocity : (T)0;
                imminent_collision_buffer.tuple(dim_c<3>,"impulse",ci + buffer_offset) = -collision_nrm * I; 
                imminent_collision_buffer.tuple(dim_c<3>,"collision_normal",ci + buffer_offset) = collision_nrm; 
        });
        // std::cout << "finish Eval EE collision imminent impulse" << std::endl;
        // printf("Eval EE collision impulse\n");
}


template<typename Pol,
    typename InverseMassTileVec,
    typename MassTileVec,
    // typename THICKNESS_REAL,
    typename PosTileVec,
    typename TrisTileVec,
    typename ImpulseBuffer,
    typename ImpulseCount,
    typename TriBvh,
    typename PTHashMap,
    typename T = typename PosTileVec::value_type>
void calc_continous_self_PT_collision_impulse(Pol& pol,
    const InverseMassTileVec& invMass,
    const MassTileVec& mass,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,
    const TrisTileVec& tris,
    const T& thickness,
    TriBvh& triCCDBvh,
    bool refit_bvh,
    PTHashMap& csPT,
    // PTHashMap& preCSPT,
    ImpulseBuffer& impulse_buffer,
    ImpulseCount& impulse_count,
    // bool recalc_collision_pairs = true,
    bool skip_too_close_pair_at_rest_configuration = false,
    bool use_collision_group = false,
    bool output_debug_inform = false,
    int collision_group_strategy = COLLISION_AMONG_SAME_GROUP | COLLISION_AMONG_DIFFERENT_GROUP) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        // constexpr auto exec_tag = wrapv<space>{};

        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec2i = zs::vec<int,2>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6;

        auto execTag = wrapv<space>{};

        // std::cout << "do continous PT collilsion detection" << std::endl;

        // std::cout << "build continous PT spacial structure" << std::endl;


        auto bvs = retrieve_bounding_volumes(pol,verts,tris,verts,wrapv<3>{},(T)1.0,(T)0,xtag,vtag);
        if(refit_bvh)
            triCCDBvh.refit(pol,bvs);
        else
            triCCDBvh.build(pol,bvs);

        // zs::bht<int,2,int> csPT{verts.get_allocator(),100000};
        csPT.reset(pol,true);

        auto has_collision_group = verts.hasProperty("collision_group");
        auto has_rest_shape = verts.hasProperty("X");

        pol(zs::range(verts.size()),[
            use_collision_group = use_collision_group,
            skip_rest = skip_too_close_pair_at_rest_configuration,
            has_collision_group = has_collision_group,
            has_rest_shape = has_rest_shape,
            invMass = proxy<space>({},invMass),
            xtag = xtag,
            vtag = vtag,
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            // csPT = proxy<space>(csPT),
            thickness = thickness,
            output_debug_inform = output_debug_inform,
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count),
            eps = eps,
            execTag = execTag,
            csPT = proxy<space>(csPT),
            bvh = proxy<space>(triCCDBvh)] ZS_LAMBDA(int vi) mutable {
                if(verts.hasProperty("collision_cancel") && verts("collision_cancel",vi) > 1e-3)
                    return;
                auto p = verts.pack(dim_c<3>,xtag,vi);
                auto v = verts.pack(dim_c<3>,vtag,vi);
                bv_t bv{p, p + v};

                auto do_close_proximity_detection = [&](int ti) mutable {
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    for(int i = 0;i != 3;++i) {
                        if(tri[i] == vi)
                            return;
                        if(verts.hasProperty("collision_cancel") && verts("collision_cancel",tri[i]) > 1e-3)
                            return;
                    }

                    bool has_dynamic_points = verts("minv",vi) > eps;
                    for(int i = 0;i != 3;++i) {
                        if(invMass("minv",tri[i]) > eps)
                            has_dynamic_points = true;
                    }
                    if(!has_dynamic_points)
                        return;

                    if(skip_rest && has_rest_shape) {
                        auto rp = verts.pack(dim_c<3>,"X",vi);
                        vec3 rts[3] = {};
                        for(int i = 0;i != 3;++i)
                            rts[i] = verts.pack(dim_c<3>,"X",tri[i]);

                        auto is_same_collision_group = true;
                        if(use_collision_group) 
                            is_same_collision_group = zs::abs(verts("collision_group",vi) - verts("collision_group",tri[0])) < 0.1;
                    
                        if(LSL_GEO::get_vertex_triangle_distance(rts[0],rts[1],rts[2],rp) < thickness && is_same_collision_group)
                            return; 
                    }

                    csPT.insert(vec2i{vi,ti});
                };  
                bvh.iter_neighbors(bv,do_close_proximity_detection);              
        });

        // std::cout << "nm close PT proxy : " << csPT.size() << std::endl;
        // std::cout << "compute continouse PT proxy impulse" << std::endl;
        pol(zip(zs::range(csPT.size()),csPT._activeKeys),[
            invMass = proxy<space>({},invMass),
            mass = proxy<space>({},mass),
            xtag = xtag,
            vtag = vtag,
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            csPT = proxy<space>(csPT),
            // thickness = thickness,
            output_debug_inform = output_debug_inform,
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count),
            eps = eps,
            execTag = execTag] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto vi = pair[0];
                auto ti = pair[1];
                auto p = verts.pack(dim_c<3>,xtag,vi);
                auto v = verts.pack(dim_c<3>,vtag,vi);

                bv_t bv{p, p + v};

                vec3 ps[4] = {};
                ps[3] = p;
                vec3 vs[4] = {};
                vs[3] = v;
                vec4 bary{0,0,0,1};
                vec4i inds{-1,-1,-1,vi};

                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                for(int i = 0;i != 3;++i) {
                    ps[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                    vs[i] = verts.pack(dim_c<3>,vtag,tri[i]);
                    inds[i] = tri[i];
                }

                auto alpha = (T)1.0;

                if(!accd::ptccd(ps[3],ps[0],ps[1],ps[2],vs[3],vs[0],vs[1],vs[2],(T)0.2,(T)0,alpha))
                    return;

                vec3 nps[4] = {};
                for(int i = 0;i != 4;++i)
                    nps[i] = ps[i] + vs[i] * alpha;

                vec3 bary_centric{};
                LSL_GEO::get_triangle_vertex_barycentric_coordinates(nps[0],nps[1],nps[2],nps[3],bary_centric);
                auto ori_bary = bary_centric;
                for(int i = 0;i != 3;++i)
                    bary_centric[i] = bary_centric[i] < 0 ? 0 : bary_centric[i];
                auto barySum = bary_centric[0] + bary_centric[1] + bary_centric[2];
                bary_centric = bary_centric / barySum;  

                for(int i = 0;i != 3;++i)
                    bary[i] = -bary_centric[i];

                auto rv = vec3::zeros();
                for(int i = 0;i != 4;++i) {
                    rv += vs[i] * bary[i];
                }
                        
                auto collision_nrm = LSL_GEO::facet_normal(nps[0],nps[1],nps[2]);

                // auto align = collision_nrm.dot(rv);
                // if(align > 0)
                //     collision_nrm *= -1;
                auto rv_nrm = collision_nrm.dot(rv);

                auto cm = (T).0;
                for(int i = 0;i != 4;++i) {
                    // cm += bary[i] * bary[i] * invMass("minv",inds[i]);
                    cm += bary[i] * bary[i] / mass("m",inds[i]);
                }
                if(cm < eps)
                    return;
                
                auto impulse = -collision_nrm * rv_nrm * ((T)1 - alpha);
                // if(output_debug_inform) {
                //     printf("find PT collision pairs[%d %d] with ccd : %f impulse : %f %f %f\n",ti,vi,(float)alpha,(float)impulse[0],(float)impulse[1],(float)impulse[2]);
                // }

                for(int i = 0;i != 4;++i) {
                    // if(invMass("minv",inds[i]) < eps)
                    //     continue;
                    auto beta = (bary[i] * invMass("minv",inds[i])) / cm;
                    atomic_add(execTag,&impulse_count[inds[i]],1);
                    for(int d = 0;d != 3;++d)
                        atomic_add(execTag,&impulse_buffer[inds[i]][d],impulse[d] * beta);
                }

        });

        // std::cout << "finish computing continouse PT impulse" << std::endl;

        // std::cout << "finish continous PT detection" << std::endl;
}


template<typename Pol,
    typename PosTileVec,
    typename TrisTileVec,
    typename kTriPathBvh,
    typename PKTHashMap,  
    typename CCDTocBuffer,
    typename T = typename PosTileVec::value_type>
void detect_continous_PKT_collision_pairs(Pol& pol,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,
    const PosTileVec& kverts,const zs::SmallString& kxtag,const zs::SmallString& kvtag,
    const TrisTileVec& ktris,
    const kTriPathBvh& ktriCCDBvh,
    CCDTocBuffer& tocs,
    PKTHashMap& csPKT) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec2i = zs::vec<int,2>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6; 

        csPKT.reset(pol,true);

        pol(zs::range(verts.size()),[
            xtag = xtag,vtag = vtag,
            verts = proxy<space>({},verts),
            tocs = proxy<space>(tocs),
            csPKT = proxy<space>(csPKT),
            kxtag = kxtag,
            kvtag = kvtag,
            kverts = proxy<space>({},kverts),
            ktris = proxy<space>({},ktris),
            eps = eps,
            ktriCCDBvh = proxy<space>(ktriCCDBvh)] ZS_LAMBDA(int vi) mutable {
                if(verts.hasProperty("minv") && verts("minv",vi) < eps)
                    return;
                // printf("testing verts[%d] for intersections\n",vi);
               int first_collided_kti = -1;
                T min_toc = std::numeric_limits<T>::max();
                auto p = verts.pack(dim_c<3>,xtag,vi);
                auto v = verts.pack(dim_c<3>,vtag,vi);
                bv_t bv{p,p + v};
                auto do_close_proximity_detection = [&](int kti) mutable {
                    // printf("testing verts[%d] and ktris[%d] for intersections\n",vi,kti);
                    auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);
                    vec3 ktps[3] = {};
                    vec3 ktvs[3] = {};
                    for(int i = 0;i != 3;++i) {
                        ktps[i] = kverts.pack(dim_c<3>,kxtag,ktri[i]);
                        ktvs[i] = kverts.pack(dim_c<3>,kvtag,ktri[i]);
                    }

                    auto toc = (T)1.0;
                    if(!accd::ptccd(p,ktps[0],ktps[1],ktps[2],v,ktvs[0],ktvs[1],ktvs[2],(T)0.2,(T)0.0,toc))
                        return;
                    if(toc < min_toc) {
                        min_toc = toc;
                        first_collided_kti = kti;
                    }
                };
                ktriCCDBvh.iter_neighbors(bv,do_close_proximity_detection);

                if(first_collided_kti >= 0) {
                    csPKT.insert(vec2i{vi,first_collided_kti});
                    tocs[vi] = min_toc;
                }
        });        
}


template<typename Pol,
    typename PosTileVec,
    typename TrisTileVec,
    typename ImpulseBuffer,
    typename ImpulseCount,
    typename kTriPathBvh,
    typename PKTHashMap,  
    typename T = typename PosTileVec::value_type>
void calc_continous_PKT_collision_impulse(Pol& pol,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,
    const PosTileVec& kverts,const zs::SmallString& kxtag,const zs::SmallString& kvtag,
    const TrisTileVec& ktris,
    const kTriPathBvh& ktriCCDBvh,
    PKTHashMap& csPKT,
    ImpulseBuffer& impulse_buffer,
    ImpulseCount& impulse_count) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec2i = zs::vec<int,2>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6;

        auto execTag = wrapv<space>{};     

        zs::Vector<T> tocs{verts.get_allocator(),verts.size()};
        pol(zs::range(tocs),[] ZS_LAMBDA(auto& toc) mutable {toc = std::numeric_limits<T>::max();});
        detect_continous_PKT_collision_pairs(pol,verts,xtag,vtag,kverts,kxtag,kvtag,ktris,ktriCCDBvh,tocs,csPKT);

        pol(zip(zs::range(csPKT.size()),csPKT._activeKeys),[
            xtag = xtag,vtag = vtag,
            tocs = proxy<space>(tocs),
            verts = proxy<space>({},verts),
            csPKT = proxy<space>(csPKT),
            kxtag = kxtag,kvtag = kvtag,
            kverts = proxy<space>({},kverts),
            ktris = proxy<space>({},ktris),
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto vi = pair[0];
                auto kti = pair[1];
                auto p = verts.pack(dim_c<3>,xtag,vi);
                auto v = verts.pack(dim_c<3>,vtag,vi);

                auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);
                vec3 ps[4] = {};ps[3] = p;
                vec3 vs[4] = {};vs[3] = v;

                for(int i = 0;i != 3;++i) {
                    ps[i] = kverts.pack(dim_c<3>,kxtag,ktri[i]);
                    vs[i] = kverts.pack(dim_c<3>,kvtag,ktri[i]);
                }

                auto toc = tocs[vi];
                vec3 nps[4] = {};
                for(int i = 0;i != 4;++i)
                    nps[i] = ps[i] + vs[i] * toc * 0.99;    

                auto collision_nrm = (nps[1] - nps[0]).cross(nps[2] - nps[0]);
                auto area = collision_nrm.norm();
                if(area < eps)
                    return;
                collision_nrm /= area;
                vec3 bary_centric{};
                LSL_GEO::get_triangle_vertex_barycentric_coordinates(nps[0],nps[1],nps[2],nps[3],bary_centric);
                vec4 bary{-bary_centric[0],-bary_centric[1],-bary_centric[2],1};

                auto rv = vec3::zeros();
                auto rp = vec3::zeros();
                for(int i = 0;i != 4;++i) {
                    rv += vs[i] * bary[i];
                    rp += nps[i] * bary[i];
                }
                if(rv.dot(rp) > 0)
                    return;

                auto rv_nrm = collision_nrm.dot(rv);       
                auto impulse = -collision_nrm * rv_nrm * ((T)1 - toc);      

                impulse_count[vi] += 1;
                impulse_buffer[vi] += impulse;                                               
        });
}

template<typename Pol,
    typename PosTileVec,
    typename TrisTileVec,
    typename ImpulseBuffer,
    typename ImpulseCount,
    typename kVertexPathBvh,
    typename PKTHashMap,  
    typename T = typename PosTileVec::value_type>
void calc_continous_KPT_collision_impulse(Pol& pol,
    const PosTileVec& kverts,const zs::SmallString& kxtag,const zs::SmallString& kvtag,
    const kVertexPathBvh& kVertexCCDBvh,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,
    const TrisTileVec& tris,
    PKTHashMap& csKPT,
    ImpulseBuffer& impulse_buffer,
    ImpulseCount& impulse_count) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec2i = zs::vec<int,2>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6;

        auto execTag = wrapv<space>{};     

        csKPT.reset(pol,true);
        zs::Vector<T> tocs{tris.get_allocator(),tris.size()};
        pol(zs::range(tris.size()),[tocs = proxy<space>(tocs)] ZS_LAMBDA(auto ti) mutable {tocs[ti] = std::numeric_limits<T>::max();});
        auto tri_bvs = retrieve_bounding_volumes(pol,verts,tris,verts,wrapv<3>{},(T)1.0,(T)0,xtag,vtag);
        
        pol(zs::range(tris.size()),[
            xtag = xtag,vtag = vtag,
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            tocs = proxy<space>(tocs),
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count),
            csKPT = proxy<space>(csKPT),
            kxtag = kxtag,
            kvtag = kvtag,
            kverts = proxy<space>({},kverts),
            eps = eps,
            tri_bvs = proxy<space>(tri_bvs),
            kVertexCCDBvh = proxy<space>(kVertexCCDBvh)] ZS_LAMBDA(int ti) mutable {
                bool is_dynamic_tri = true;

                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                // if(verts.hasProperty("minv")) {
                //     is_dynamic_tri = false;
                //     for(int i = 0;i != 3;++i)
                //         if(verts("minv",tri[i]) > eps)
                //             is_dynamic_tri = true;
                // }

                if(!is_dynamic_tri)
                    return;

               int first_collided_kvi = -1;
                T min_toc = std::numeric_limits<T>::max();

                vec3 tps[3] = {};
                vec3 tvs[3] = {};
                for(int i = 0;i != 3;++i) {
                    tps[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                    tvs[i] = verts.pack(dim_c<3>,vtag,tri[i]);
                }

                auto bv = tri_bvs[ti];
                // bv_t bv{tps[0],tps[1]};
                // merge(bv,tps[2]);

                // auto tri_vel = vec3::zeros();

                auto do_close_proximity_detection = [&](int kvi) mutable {
                    auto kp = kverts.pack(dim_c<3>,kxtag,kvi);
                    auto kv = kverts.pack(dim_c<3>,kvtag,kvi);
 
                    auto toc = (T)1.0;
                    if(!accd::ptccd(kp,tps[0],tps[1],tps[2],kv,tvs[0],tvs[1],tvs[2],(T)0.2,(T)0.0,toc))
                        return;
                    if(toc < min_toc) {
                        min_toc = toc;
                        first_collided_kvi = kvi;
                    }
                };
                kVertexCCDBvh.iter_neighbors(bv,do_close_proximity_detection);

                if(first_collided_kvi >= 0) {
                    csKPT.insert(vec2i{first_collided_kvi,ti});
                    tocs[ti] = min_toc;
                }
        });

        pol(zip(zs::range(csKPT.size()),csKPT._activeKeys),[
            xtag = xtag,vtag = vtag,
            execTag = execTag,
            tocs = proxy<space>(tocs),
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            csKPT = proxy<space>(csKPT),
            kxtag = kxtag,kvtag = kvtag,
            kverts = proxy<space>({},kverts),
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto kvi = pair[0];
                auto ti = pair[1];
                auto kp = kverts.pack(dim_c<3>,kxtag,kvi);
                auto kv = kverts.pack(dim_c<3>,kvtag,kvi);

                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                vec3 ps[4] = {};ps[3] = kp;
                vec3 vs[4] = {};vs[3] = kv;

                for(int i = 0;i != 3;++i) {
                    ps[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                    vs[i] = verts.pack(dim_c<3>,vtag,tri[i]);
                }

                auto toc = tocs[ti];
                vec3 nps[4] = {};
                for(int i = 0;i != 4;++i)
                    nps[i] = ps[i] + vs[i] * toc * 0.99;    

                auto collision_nrm = (nps[1] - nps[0]).cross(nps[2] - nps[0]);
                auto area = collision_nrm.norm();
                if(area < eps)
                    return;
                collision_nrm /= area;
                vec3 bary_centric{};
                LSL_GEO::get_triangle_vertex_barycentric_coordinates(nps[0],nps[1],nps[2],nps[3],bary_centric);
                vec4 bary{-bary_centric[0],-bary_centric[1],-bary_centric[2],1};

                auto rv = vec3::zeros();
                auto rp = vec3::zeros();
                for(int i = 0;i != 4;++i) {
                    rv += vs[i] * bary[i];
                    rp += nps[i] * bary[i];
                }
                if(rv.dot(rp) > 0)
                    return;

                auto rv_nrm = collision_nrm.dot(rv);       

                auto cm = (T).0;
                for(int i = 0;i != 3;++i)
                    cm += bary_centric[i] * bary_centric[i] / verts("m",tri[i]);
                if(cm < eps)
                    return;


                auto impulse = -collision_nrm * rv_nrm * ((T)1 - toc);      

                for(int i = 0;i != 3;++i) {
                    // if(verts("minv",tri[i]) < eps)
                    //     return;
                    auto beta = -(bary_centric[i] * verts("minv",tri[i])) / cm;
                    atomic_add(execTag,&impulse_count[tri[i]],1);
                    for(int d = 0;d != 3;++d)
                        atomic_add(execTag,&impulse_buffer[tri[i]][d],impulse[d] * beta);
                }                                             
        });
}


template<typename Pol,  
    typename PosTileVec,
    typename EdgeTileVec,
    typename ImpulseBuffer,
    typename ImpulseCountBuffer,
    typename kEdgePathBvh,
    typename EKEHashMap,
    typename T = typename PosTileVec::value_type>
void calc_continous_EKE_collision_impulse_with_toc(Pol& pol,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,
    const EdgeTileVec& edges,
    const PosTileVec& kverts,const zs::SmallString& kxtag,const zs::SmallString& kvtag,
    const EdgeTileVec& kedges,
    kEdgePathBvh& keBvh,
    EKEHashMap& csEKE,
    ImpulseBuffer& impulse_buffer,
    ImpulseCountBuffer& impulse_count) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        // constexpr auto exec_tag = wrapv<space>{};

        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec2i = zs::vec<int,2>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6;

        zs::Vector<T> tocs{edges.get_allocator(),edges.size()};
        pol(zs::range(tocs),[] ZS_LAMBDA(auto& toc) mutable {toc = std::numeric_limits<T>::max();});

        auto edge_bvs = retrieve_bounding_volumes(pol,verts,edges,verts,wrapv<2>{},(T)1.0,(T)0,xtag,vtag);        

        csEKE.reset(pol,true);

        auto execTag = wrapv<space>{};

        pol(zs::range(edges.size()),[
            xtag = xtag,vtag = vtag,
            edge_bvs = proxy<space>(edge_bvs),
            tocs = proxy<space>(tocs),
            verts = proxy<space>({},verts),
            edges = proxy<space>({},edges),
            kxtag = kxtag,
            kvtag = kvtag,
            kverts = proxy<space>({},kverts),
            kedges = proxy<space>({},kedges),
            keBvh = proxy<space>(keBvh),
            eps = eps,
            csEKE = proxy<space>(csEKE),
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count)] ZS_LAMBDA(int ei) mutable {
                bool is_dynamic_edge = true;
                auto edge = edges.pack(dim_c<2>,"inds",ei,int_c);
                // if(verts.hasProperty("minv")) {
                //     is_dynamic_edge = false;
                //     for(int i = 0;i != 2;++i)
                //         if(verts("minv",edge[i]) > eps)
                //             is_dynamic_edge = true;
                // }
                if(!is_dynamic_edge)
                    return;      

                vec3 eps[2] = {};
                for(int i = 0;i != 2;++i)
                    eps[i] = verts.pack(dim_c<3>,xtag,edge[i]);

                // auto bv = bv_t{get_bounding_box(eps[0],eps[1])};
                auto bv = edge_bvs[ei];

                int first_collided_kei = -1;
                T min_toc = std::numeric_limits<T>::max();

                vec3 evs[2] = {};
                for(int i = 0;i != 2;++i)
                    evs[i] = verts.pack(dim_c<3>,vtag,edge[i]);

                auto do_close_proximity_detection = [&](int kei) mutable {
                    auto kedge = kedges.pack(dim_c<2>,"inds",kei,int_c);
                    vec3 keps[2] = {};
                    vec3 kevs[2] = {};
                    for(int i = 0;i != 2;++i) {
                        keps[i] = kverts.pack(dim_c<3>,kxtag,kedge[i]);
                        kevs[i] = kverts.pack(dim_c<3>,kvtag,kedge[i]);
                    }

                    auto toc = (T)1.0;
                    if(!accd::eeccd(eps[0],eps[1],keps[0],keps[1],evs[0],evs[1],kevs[0],kevs[1],(T)0.2,(T)0,toc))
                        return;     

                    if(toc < min_toc) {
                        min_toc = toc;
                        first_collided_kei = kei;
                    }        
                };   
                keBvh.iter_neighbors(bv,do_close_proximity_detection);
                if(first_collided_kei >= 0) {
                    csEKE.insert(vec2i{ei,first_collided_kei});
                    tocs[ei] = min_toc;
                }
        });

    pol(zip(zs::range(csEKE.size()),csEKE._activeKeys),[
        execTag = execTag,
        xtag = xtag,vtag = vtag,
        tocs = proxy<space>(tocs),
        verts = proxy<space>({},verts),
        edges = proxy<space>({},edges),
        csEKE = proxy<space>(csEKE),
        kxtag = kxtag,kvtag = kvtag,
        impulse_count = proxy<space>(impulse_count),
        impulse_buffer = proxy<space>(impulse_buffer),
        kverts = proxy<space>({},kverts),
        kedges = proxy<space>({},kedges)] ZS_LAMBDA(auto id,const auto& pair) mutable {
            auto ei  = pair[0];
            auto kei = pair[1];
            auto edge = edges.pack(dim_c<2>,"inds",ei,int_c);
            auto kedge = kedges.pack(dim_c<2>,"inds",kei,int_c);

            vec3 ps[4] = {};
            vec3 vs[4] = {};

            for(int i = 0;i != 2;++i) {
                ps[i] = verts.pack(dim_c<3>,xtag,edge[i]);
                ps[i + 2] = kverts.pack(dim_c<3>,kxtag,kedge[i]);
                vs[i] = verts.pack(dim_c<3>,vtag,edge[i]);
                vs[i + 2] = kverts.pack(dim_c<3>,kvtag,kedge[i]);
            }

            vec3 nps[4] = {};
            auto toc = tocs[ei];
            for(int i = 0;i != 4;++i)
                nps[i] = ps[i] + toc * vs[i];

            vec2 edge_bary{};
            LSL_GEO::get_edge_edge_barycentric_coordinates(nps[0],nps[1],nps[2],nps[3],edge_bary);
                for(int i = 0;i != 2;++i) {
                edge_bary[i] = edge_bary[i] < 0 ? 0 : edge_bary[i];
                edge_bary[i] = edge_bary[i] > 1 ? 1 : edge_bary[i];
            }            
            vec4 bary{edge_bary[0] - 1,-edge_bary[0],1 - edge_bary[1],edge_bary[1]};

            auto rv = vec3::zeros();
            auto pr = vec3::zeros();
            for(int i = 0;i != 4;++i) {
                rv += vs[i] * bary[i];
                pr += nps[i] * bary[i];
            }

            auto collision_nrm = pr.normalized();
            auto rv_nrm = collision_nrm.dot(rv);
            auto cm = (T).0;
            cm += bary[0] * bary[0] / verts("m",edge[0]);
            cm += bary[1] * bary[1] / verts("m",edge[1]);

            if(cm < eps)
                return; 

            auto impulse = -collision_nrm * rv_nrm * ((T)1 - toc);     

            for(int i = 0;i != 2;++i) {
                auto beta = (bary[i] * verts("minv",edge[i])) / cm;
                atomic_add(execTag,&impulse_count[edge[i]],1);
                    for(int d = 0;d != 3;++d)
                        atomic_add(execTag,&impulse_buffer[edge[i]][d],impulse[d] * beta);
            }      
    });
}

template<typename Pol,
    typename InverseMassTileVec,
    typename MassTileVec,
    // typename THICKNESS_REAL,
    typename PosTileVec,
    typename TrisTileVec,
    typename ImpulseBuffer,
    typename ImpulseCount,
    typename TriBvh,
    typename PTHashMap,
    typename T = typename PosTileVec::value_type>
void calc_continous_self_PT_collision_impulse_with_toc(Pol& pol,
    const InverseMassTileVec& invMass,
    const MassTileVec& mass,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,
    const TrisTileVec& tris,
    const T& thickness,
    const T& igore_rest_shape_thickness,
    TriBvh& triCCDBvh,
    bool refit_bvh,
    PTHashMap& csPT,
    // PTHashMap& preCSPT,
    ImpulseBuffer& impulse_buffer,
    ImpulseCount& impulse_count,
    bool skip_too_close_pair_at_rest_configuration = false,
    bool use_collision_group = false,
    bool output_debug_inform = false,
    int collision_group_strategy = COLLISION_AMONG_SAME_GROUP | COLLISION_AMONG_DIFFERENT_GROUP) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        // constexpr auto exec_tag = wrapv<space>{};

        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec2i = zs::vec<int,2>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6;

        auto execTag = wrapv<space>{};

        auto bvs = retrieve_bounding_volumes(pol,verts,tris,verts,wrapv<3>{},(T)1.0,(T)thickness,xtag,vtag);
        if(refit_bvh)
            triCCDBvh.refit(pol,bvs);
        else
            triCCDBvh.build(pol,bvs);

        // zs::bht<int,2,int> csPT{verts.get_allocator(),100000};
        csPT.reset(pol,true);

        auto has_collision_group = verts.hasProperty("collision_group");
        auto has_rest_shape = verts.hasProperty("X");

        zs::Vector<float> tocs{verts.get_allocator(),verts.size()};
        pol(zs::range(verts.size()),[tocs = proxy<space>(tocs)] ZS_LAMBDA(auto vi) mutable {tocs[vi] = 1;});

        pol(zs::range(verts.size()),[
            collision_group_strategy = collision_group_strategy,
            use_collision_group = use_collision_group,
            skip_rest = skip_too_close_pair_at_rest_configuration,
            has_collision_group = has_collision_group,
            has_rest_shape = has_rest_shape,
            invMass = proxy<space>({},invMass),
            xtag = xtag,
            vtag = vtag,
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            thickness = thickness,
            igore_rest_shape_thickness = igore_rest_shape_thickness,
            tocs = proxy<space>(tocs),
            output_debug_inform = output_debug_inform,
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count),
            eps = eps,
            execTag = execTag,
            csPT = proxy<space>(csPT),
            bvh = proxy<space>(triCCDBvh)] ZS_LAMBDA(int vi) mutable {
                if(verts.hasProperty("collision_cancel") && verts("collision_cancel",vi) > 1e-3)
                    return;
                auto p = verts.pack(dim_c<3>,xtag,vi);
                auto v = verts.pack(dim_c<3>,vtag,vi);
                bv_t bv{p, p + v};

                int min_ti = -1;
                float min_alpha = 1; 

                auto do_close_proximity_detection = [&](int ti) mutable {
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    for(int i = 0;i != 3;++i) {
                        if(tri[i] == vi)
                            return;
                        if(verts.hasProperty("collision_cancel") && verts("collision_cancel",tri[i]) > 1e-3)
                            return;
                    }

                    bool has_dynamic_points = verts("minv",vi) > eps;
                    for(int i = 0;i != 3;++i) {
                        if(invMass("minv",tri[i]) > eps)
                            has_dynamic_points = true;
                    }
                    if(!has_dynamic_points)
                        return;

                    if(has_collision_group) {
                        if(!(collision_group_strategy & COLLISION_AMONG_DIFFERENT_GROUP)) {
                            auto vgroup = verts("collision_group",vi);
                            for(int i = 0;i != 3;++i) {
                                auto tgroup = verts("collision_group",tri[i]);
                                // if they belong to two different groups
                                if(zs::abs(vgroup - tgroup) > 0.1) {
                                    return;
                                }
                            }
                        }
                        if(!(collision_group_strategy & COLLISION_AMONG_SAME_GROUP)) {
                            auto vgroup = verts("collision_group",vi);
                            for(int i = 0;i != 3;++i) {
                                auto tgroup = verts("collision_group",tri[i]);
                                if(zs::abs(vgroup - tgroup) < 0.1)
                                    return;
                            }
                        }
                    }


                    if(skip_rest && has_rest_shape) {
                        auto rp = verts.pack(dim_c<3>,"X",vi);
                        vec3 rts[3] = {};
                        for(int i = 0;i != 3;++i)
                            rts[i] = verts.pack(dim_c<3>,"X",tri[i]);

                        auto is_same_collision_group = true;
                        if(use_collision_group) 
                            is_same_collision_group = zs::abs(verts("collision_group",vi) - verts("collision_group",tri[0])) < 0.1;
                    
                        if(LSL_GEO::get_vertex_triangle_distance(rts[0],rts[1],rts[2],rp) < igore_rest_shape_thickness && is_same_collision_group)
                            return; 
                    }

                    vec3 ps[4] = {};
                    vec3 vs[4] = {};
                    vec4 bary{0,0,0,1};
                    vec4i inds{tri[0],tri[1],tri[2],vi};

                    for(int i = 0;i != 4;++i) {
                        ps[i] = verts.pack(dim_c<3>,xtag,inds[i]);
                        vs[i] = verts.pack(dim_c<3>,vtag,inds[i]);
                    }

                    auto alpha = (T)1.0;
                    if(!accd::ptccd(ps[3],ps[0],ps[1],ps[2],vs[3],vs[0],vs[1],vs[2],(T)0.2,(T)thickness,alpha))
                        return;
                    if(alpha < min_alpha) {
                        min_alpha = alpha;
                        min_ti = ti;
                    }
                };  
                bvh.iter_neighbors(bv,do_close_proximity_detection);  

                if(min_ti >= 0) {
                    csPT.insert(vec2i{vi,min_ti});
                    tocs[vi] = min_alpha;
                }            
        });

        // std::cout << "nm close PT proxy : " << csPT.size() << std::endl;
        // std::cout << "compute continouse PT proxy impulse" << std::endl;
        pol(zip(zs::range(csPT.size()),csPT._activeKeys),[
            invMass = proxy<space>({},invMass),
            mass = proxy<space>({},mass),
            xtag = xtag,
            vtag = vtag,
            tocs = proxy<space>(tocs),
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            csPT = proxy<space>(csPT),
            thickness = thickness,
            output_debug_inform = output_debug_inform,
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count),
            eps = eps,
            execTag = execTag] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto vi = pair[0];
                auto ti = pair[1];
                auto p = verts.pack(dim_c<3>,xtag,vi);
                auto v = verts.pack(dim_c<3>,vtag,vi);

                bv_t bv{p, p + v};

                vec3 ps[4] = {};
                ps[3] = p;
                vec3 vs[4] = {};
                vs[3] = v;
                


                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                vec4i inds{tri[0],tri[1],tri[2],vi};

                for(int i = 0;i != 3;++i) {
                    ps[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                    vs[i] = verts.pack(dim_c<3>,vtag,tri[i]);
                    // inds[i] = tri[i];
                }

                // auto alpha = (T)1.0;
                auto alpha = tocs[vi];

                // TRY TO PREVENT SOME STICKY
                // if(alpha < (T)0.1) {
                //     printf("the ccd collision[%f] is too closed, skip it, dcd and detangle should solve it\n",(float)alpha);
                //     return;
                // }

                vec3 nps[4] = {};
                for(int i = 0;i != 4;++i)
                    nps[i] = ps[i] + vs[i] * alpha * 0.99;

                auto collision_nrm = (nps[1] - nps[0]).cross(nps[2] - nps[0]);
                auto area = collision_nrm.norm();
                if(area < eps)
                    return;
                collision_nrm /= area;
                // auto collision_nrm = LSL_GEO::facet_normal(nps[0],nps[1],nps[2]);

                vec3 bary_centric{};
                LSL_GEO::get_triangle_vertex_barycentric_coordinates(nps[0],nps[1],nps[2],nps[3],bary_centric);
                vec4 bary{-bary_centric[0],-bary_centric[1],-bary_centric[2],1};
                // for(int i = 0;i != 3;++i)
                //     bary[i] = -bary_centric[i];

                auto rv = vec3::zeros();
                auto rp = vec3::zeros();
                for(int i = 0;i != 4;++i) {
                    rv += vs[i] * bary[i];
                    rp += nps[i] * bary[i];
                }
                if(rv.dot(rp) > 0)
                    return;
                        

                auto rv_nrm = collision_nrm.dot(rv);

                auto cm = (T).0;
                for(int i = 0;i != 4;++i)
                    cm += bary[i] * bary[i] / mass("m",inds[i]);
                if(cm < eps)
                    return;
                
                auto impulse = -collision_nrm * rv_nrm * ((T)1 - alpha);
                // if(output_debug_inform) {
                //     printf("find PT collision pairs[%d %d] with ccd : %f impulse : %f %f %f\n",ti,vi,(float)alpha,(float)impulse[0],(float)impulse[1],(float)impulse[2]);
                // }

                for(int i = 0;i != 4;++i) {
                    if(invMass("minv",inds[i]) < eps)
                        continue;
                    auto beta = (bary[i] * invMass("minv",inds[i])) / cm;

                    // if((impulse * beta).norm() > 5 && output_debug_inform) {
                    //     printf("large CCD-PT-impulse detected %f : %f %f %f nrm :  %f\n",
                    //         (float)beta,
                    //         (float)impulse[0],
                    //         (float)impulse[1],
                    //         (float)impulse[2],
                    //         (float)collision_nrm.norm());
                    // }

                    atomic_add(execTag,&impulse_count[inds[i]],1);
                    for(int d = 0;d != 3;++d)
                        atomic_add(execTag,&impulse_buffer[inds[i]][d],impulse[d] * beta);
                }

        });
        // std::cout << "finish computing continouse PT impulse" << std::endl;
        // std::cout << "finish continous PT detection" << std::endl;
}



template<typename Pol,  
    typename InverseMassTileVec,
    typename MassTileVec,
    typename PosTileVec,
    typename EdgeTileVec,
    typename ImpulseBuffer,
    typename ImpulseCountBuffer,
    typename EdgeBvh,
    typename EEHashMap,
    typename T = typename PosTileVec::value_type>
void calc_continous_self_EE_collision_impulse(Pol& pol,
    const InverseMassTileVec& invMass,
    const MassTileVec& mass,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,
    const EdgeTileVec& edges,
    const T& thickness,
    const size_t& start_edge_id,
    const size_t& end_edge_id,
    EdgeBvh& edgeCCDBvh,
    bool refit_bvh,
    EEHashMap& csEE,
    ImpulseBuffer& impulse_buffer,
    ImpulseCountBuffer& impulse_count,
    bool skip_too_close_pair_at_rest_configuration = false,
    bool use_collision_group = false,
    bool output_debug_inform = false) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        // constexpr auto exec_tag = wrapv<space>{};

        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec2i = zs::vec<int,2>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6;

        // auto edgeCCDBvh = bvh_t{};
        // std::cout << "build continous EE structure" << std::endl;


        // ALLOCATION BOTTLENECK 1
        auto edgeBvs = retrieve_bounding_volumes(pol,verts,edges,verts,wrapv<2>{},(T)1.0,(T)0,xtag,vtag);
        if(refit_bvh)
            edgeCCDBvh.refit(pol,edgeBvs);
        else
            edgeCCDBvh.build(pol,edgeBvs);

        // std::cout << "do continous EE collilsion detection" << std::endl;


        // ALLOCATION BOTTLENECK 2
        // zs::bht<int,2,int> csEE{edges.get_allocator(),100000};
        csEE.reset(pol,true);


        auto has_collision_group = verts.hasProperty("collision_group");
        auto has_rest_shape = verts.hasProperty("X");

        auto execTag = wrapv<space>{};

        auto nm_test_edges = end_edge_id - start_edge_id;

        pol(zs::range(nm_test_edges),[
            use_collision_group = use_collision_group,
            skip_rest = skip_too_close_pair_at_rest_configuration,
            has_collision_group = has_collision_group,
            has_rest_shape = has_rest_shape,
            xtag = xtag,
            vtag = vtag,
            verts = proxy<space>({},verts),
            edges = proxy<space>({},edges),
            invMass = proxy<space>({},invMass),
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count),
            thickness = thickness,
            output_debug_inform = output_debug_inform,
            eps = eps,
            csEE = proxy<space>(csEE),
            execTag = execTag,
            edgeBvs = proxy<space>(edgeBvs),
            bvh = proxy<space>(edgeCCDBvh)] ZS_LAMBDA(int ei) mutable {
                auto ea = edges.pack(dim_c<2>,"inds",ei,int_c);
                // auto bv = edgeBvs[ei];
                for(int i = 0;i != 2;++i)
                    if(verts.hasProperty("collision_cancel") && verts("collision_cancel",ea[i]) > 1e-3)
                        return;

                auto v0 = verts.pack(dim_c<3>, xtag, ea[0]);
                auto v1 = verts.pack(dim_c<3>, xtag, ea[1]);
                auto dir0 = verts.pack(dim_c<3>, vtag, ea[0]);
                auto dir1 = verts.pack(dim_c<3>, vtag, ea[1]);   
                T alpha = (T)1.0;
                auto bv = bv_t{get_bounding_box(v0, v0 + alpha * dir0)};
                merge(bv, v1);
                merge(bv, v1 + alpha * dir1);
                // bv._min -= xi;
                // bv._max += xi;
   
                auto do_close_proximity_detection = [&](int nei) mutable {
                    if(ei >= nei)
                        return;
                    auto eb = edges.pack(dim_c<2>,"inds",nei,int_c);
                    for(int i = 0;i != 2;++i){
                        if(eb[i] == ea[0] || eb[i] == ea[1])
                            return;
                    }

                    for(int i = 0;i != 2;++i)
                        if(verts.hasProperty("collision_cancel") && verts("collision_cancel",eb[i]) > 1e-3)
                            return;

                    auto has_dynamic_points = false;
                    for(int i = 0;i != 2;++i) {
                        if(invMass("minv",ea[i]) > eps)
                            has_dynamic_points = true;
                        if(invMass("minv",eb[i]) > eps)
                            has_dynamic_points = true;
                    }
                    if(!has_dynamic_points)
                        return;


                    if(has_rest_shape && skip_rest) {
                        vec3 rpas[2] = {};
                        vec3 rpbs[2] = {};
                        for(int i = 0;i != 2;++i) {
                            rpas[i] = verts.pack(dim_c<3>,"X",ea[i]);
                            rpbs[i] = verts.pack(dim_c<3>,"X",eb[i]);
                        }

                        auto is_same_collision_group = true;
                        if(use_collision_group) 
                            is_same_collision_group = zs::abs(verts("collision_group",ea[0]) - verts("collision_group",eb[0])) < 0.1;

                        if(LSL_GEO::get_edge_edge_distance(rpas[0],rpas[1],rpbs[0],rpbs[1]) < thickness && is_same_collision_group)
                            return;
                    }

                    vec4i inds{ea[0],ea[1],eb[0],eb[1]};
                    vec3 ps[4] = {};
                    vec3 pps[4] = {};
                    for(int i = 0;i != 4;++i) {
                        pps[i] = verts.pack(dim_c<3>,"x",inds[i]);
                        ps[i] = pps[i] + verts.pack(dim_c<3>,"v",inds[i]);
                    }

                    auto ori = LSL_GEO::orient(ps[0],ps[1],ps[2],ps[3]);
                    auto pori = LSL_GEO::orient(pps[0],pps[1],pps[2],pps[3]);

                    if(ori * pori > 0)
                        return;

                    csEE.insert(vec2i{ei,nei});
                };
                bvh.iter_neighbors(bv,do_close_proximity_detection);
        });

        // std::cout << "nm close EE proxy : " << csEE.size() << std::endl;
        // std::cout << "compute continous EE proxy impulse" << std::endl;

        pol(zip(zs::range(csEE.size()),csEE._activeKeys),[
            xtag = xtag,
            vtag = vtag,
            verts = proxy<space>({},verts),
            edges = proxy<space>({},edges),
            invMass = proxy<space>({},invMass),
            mass = proxy<space>({},mass),
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count),
            // thickness = thickness,
            output_debug_inform = output_debug_inform,
            eps = eps,
            execTag = execTag,
            edgeBvs = proxy<space>(edgeBvs),
            bvh = proxy<space>(edgeCCDBvh)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto ei = pair[0];
                auto nei = pair[1];
                auto ea = edges.pack(dim_c<2>,"inds",ei,int_c);
                vec3 ps[4] = {};
                ps[0] = verts.pack(dim_c<3>,xtag,ea[0]);
                ps[1] = verts.pack(dim_c<3>,xtag,ea[1]);
                vec3 vs[4] = {};
                vs[0] = verts.pack(dim_c<3>,vtag,ea[0]);
                vs[1] = verts.pack(dim_c<3>,vtag,ea[1]);

                vec4i inds{ea[0],ea[1],-1,-1};

                T minvs[4] = {};
                minvs[0] = verts("minv",ea[0]);
                minvs[1] = verts("minv",ea[1]);

                vec3 imps[4] = {};

                // auto do_close_proximity_detection = [&](int nei) mutable {
                auto eb = edges.pack(dim_c<2>,"inds",nei,int_c);

                for(int i = 0;i != 2;++i) {
                    ps[i + 2] = verts.pack(dim_c<3>,xtag,eb[i]);
                    vs[i + 2] = verts.pack(dim_c<3>,vtag,eb[i]);
                    inds[i + 2] = eb[i];
                    minvs[i + 2] = verts("minv",eb[i]);
                }

                // if(!compute_continous_EE_collision_impulse(ps,vs,minvs,imps))
                //     return;    

                auto alpha = (T)1.0;

                auto edge_a = ps[0] - ps[1];
                auto edge_b = ps[2] - ps[3];
                auto ee_norm = edge_a.norm() + edge_b.norm();
                if((edge_a.cross(edge_b).norm() / ee_norm) < 1e-3)
                    return;

                if(!accd::eeccd(ps[0],ps[1],ps[2],ps[3],vs[0],vs[1],vs[2],vs[3],(T)0.2,(T)0,alpha))
                    return;                

                vec3 nps[4] = {};
                for(int i = 0;i != 4;++i)
                    nps[i] = ps[i] + vs[i] * alpha;

                vec2 edge_bary{};
                LSL_GEO::get_edge_edge_barycentric_coordinates(nps[0],nps[1],nps[2],nps[3],edge_bary);
                 for(int i = 0;i != 2;++i) {
                    if(edge_bary[i] < eps || edge_bary[i] > 1 - eps)
                        return;
                    // edge_bary[i] = edge_bary[i] > 1 ? 1 : edge_bary[i];
                 }
                vec4 bary{edge_bary[0] - 1,-edge_bary[0],1 - edge_bary[1],edge_bary[1]};

                auto rv = vec3::zeros();
                auto pr = vec3::zeros();
                for(int i = 0;i != 4;++i) {
                    rv += vs[i] * bary[i];
                    pr += nps[i] * bary[i];
                }

                auto collision_nrm = pr.normalized();
                auto rv_nrm = collision_nrm.dot(rv);
                auto cm = (T).0;
                for(int i = 0;i != 4;++i) {
                    // cm += bary[i] * bary[i] * invMass("minv",inds[i]);
                    cm += bary[i] * bary[i] / mass("m",inds[i]);
                }
                if(cm < eps)
                    return;

                // if(cm < 0) {
                //     printf("negative cm detected : %f\n",(float)cm);
                // }

                // cm = cm < 0.01 ? 0.01 : cm;
                auto impulse = -collision_nrm * rv_nrm * ((T)1 - alpha);
                // if(output_debug_inform)
                //     printf("find EE collision pairs[%d %d] with ccd\n",ei,nei);

                for(int i = 0;i != 4;++i) {
                    // if(invMass("minv",inds[i]) < eps)
                    //     continue;
                    auto beta = (bary[i] * invMass("minv",inds[i])) / cm;

                    // if((impulse * beta).norm() > 10 && output_debug_inform) {
                    //     printf("large CCD-EE-impulse detected %f : %f %f %f : %f %f : alpha : %f\n",
                    //         (float)beta,
                    //         (float)impulse[0],
                    //         (float)impulse[1],
                    //         (float)impulse[2],
                    //         (float)rv_nrm,
                    //         (float)collision_nrm.norm(),
                    //         (float)alpha);
                    // }

                    atomic_add(execTag,&impulse_count[inds[i]],1);
                    for(int d = 0;d != 3;++d)
                        atomic_add(execTag,&impulse_buffer[inds[i]][d],impulse[d] * beta);
                }   
                // };
                // bvh.iter_neighbors(bv,do_close_proximity_detection);
        });

        // std::cout << "finish computing continous EE proxy impulse" << std::endl;
}


template<typename Pol,
    typename InverseMassTileVec,
    typename PosTileVec,
    typename EdgeTileVec,
    typename EdgeBvh,
    typename TriTileVec,
    typename TriBvh,
    typename T = typename PosTileVec::value_type>
void find_closest_intersection_free_configuration(Pol& pol,
    const InverseMassTileVec& invMass,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,
    const EdgeTileVec& edges,
    bool refit_bvh,
    const size_t& start_edge_id,
    const size_t& end_edge_id,
    EdgeBvh& edgeCCDBvh,
    const TriTileVec& tris,
    TriBvh& triCCDBvh) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        // constexpr auto exec_tag = wrapv<space>{};

        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec2i = zs::vec<int,2>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6;

        zs::Vector<float> tocs{verts.get_allocator(),verts.size()};
        pol(zs::range(tocs.size()),[tocs = proxy<space>(tocs)] ZS_LAMBDA(int vi) mutable {tocs[vi] = (T)1;});
        auto nm_test_edges = end_edge_id - start_edge_id;
        auto execTag = wrapv<space>{};
        
        zs::Vector<int> nm_ccd_fail{verts.get_allocator(),1}; 
        int nm_iters = 0;

        do {

        pol(zs::range(tocs.size()),[tocs = proxy<space>(tocs)] ZS_LAMBDA(int vi) mutable {tocs[vi] = (T)1;});
        nm_ccd_fail.setVal(0);

        auto edgeBvs = retrieve_bounding_volumes(pol,verts,edges,verts,wrapv<2>{},(T)1.0,(T)0,xtag,vtag);
        if(refit_bvh)
            edgeCCDBvh.refit(pol,edgeBvs);
        else
            edgeCCDBvh.build(pol,edgeBvs);

        pol(zs::range(nm_test_edges),[
                xtag = xtag,
                vtag = vtag,
                nm_ccd_fail = proxy<space>(nm_ccd_fail),
                tocs = proxy<space>(tocs),
                verts = proxy<space>({},verts),
                edges = proxy<space>({},edges),
                invMass = proxy<space>({},invMass),
                eps = eps,
                execTag = execTag,
                edgeBvs = proxy<space>(edgeBvs),
                bvh = proxy<space>(edgeCCDBvh)] ZS_LAMBDA(int ei) mutable {
            auto ea = edges.pack(dim_c<2>,"inds",ei,int_c);
            // auto bv = edgeBvs[ei];
            for(int i = 0;i != 2;++i)
                if(verts.hasProperty("collision_cancel") && verts("collision_cancel",ea[i]) > 1e-3)
                    return;

            auto v0 = verts.pack(dim_c<3>, xtag, ea[0]);
            auto v1 = verts.pack(dim_c<3>, xtag, ea[1]);
            auto dir0 = verts.pack(dim_c<3>, vtag, ea[0]);
            auto dir1 = verts.pack(dim_c<3>, vtag, ea[1]);   
            // T alpha = (T)1.0;
            auto bv = bv_t{get_bounding_box(v0, v0 + dir0)};
            merge(bv, v1);
            merge(bv, v1 + dir1);
            // bv._min -= xi;
            // bv._max += xi;

            int min_nei = -1;
            float min_alpha = 1; 

            auto do_close_proximity_detection = [&](int nei) mutable {
                if(ei >= nei)
                    return;
                auto eb = edges.pack(dim_c<2>,"inds",nei,int_c);
                for(int i = 0;i != 2;++i){
                    if(eb[i] == ea[0] || eb[i] == ea[1])
                        return;
                }

                for(int i = 0;i != 2;++i)
                    if(verts.hasProperty("collision_cancel") && verts("collision_cancel",eb[i]) > 1e-3)
                        return;

                auto has_dynamic_points = false;
                for(int i = 0;i != 2;++i) {
                    if(invMass("minv",ea[i]) > eps)
                        has_dynamic_points = true;
                    if(invMass("minv",eb[i]) > eps)
                        has_dynamic_points = true;
                }
                if(!has_dynamic_points)
                    return;


                vec4i inds{ea[0],ea[1],eb[0],eb[1]};
                vec3 ps[4] = {};
                vec3 vs[4] = {};
                for(int i = 0;i != 4;++i) {
                    ps[i] = verts.pack(dim_c<3>,xtag,inds[i]);
                    vs[i] = verts.pack(dim_c<3>,vtag,inds[i]);
                }

                auto alpha = (T)1.0;
                if(!accd::eeccd(ps[0],ps[1],ps[2],ps[3],vs[0],vs[1],vs[2],vs[3],(T)0.2,(T)0,alpha))
                    return;     

                if(alpha < min_alpha) {
                    min_alpha = alpha;
                    min_nei = nei;
                }
            };
            bvh.iter_neighbors(bv,do_close_proximity_detection);

            if(min_nei >= 0) {
                auto eb = edges.pack(dim_c<2>,"inds",min_nei,int_c);
                for(int i = 0;i != 2;++i) {
                    atomic_min(execTag,&tocs[ea[i]],min_alpha);
                    atomic_min(execTag,&tocs[eb[i]],min_alpha);
                }
                atomic_add(execTag,&nm_ccd_fail[0],1);
            }
        });

        auto tribvs = retrieve_bounding_volumes(pol,verts,tris,verts,wrapv<3>{},(T)1.0,(T)0,xtag,vtag);
        if(refit_bvh)
            triCCDBvh.refit(pol,tribvs);
        else
            triCCDBvh.build(pol,tribvs);
        
        pol(zs::range(verts.size()),[
            invMass = proxy<space>({},invMass),
            xtag = xtag,
            vtag = vtag,
            nm_ccd_fail = proxy<space>(nm_ccd_fail),
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            tocs = proxy<space>(tocs),
            eps = eps,
            execTag = execTag,
            bvh = proxy<space>(triCCDBvh)] ZS_LAMBDA(int vi) mutable {
                if(verts.hasProperty("collision_cancel") && verts("collision_cancel",vi) > 1e-3)
                    return;
                auto p = verts.pack(dim_c<3>,xtag,vi);
                auto v = verts.pack(dim_c<3>,vtag,vi);
                bv_t bv{p, p + v};

                int min_ti = -1;
                float min_alpha = 1; 

                auto do_close_proximity_detection = [&](int ti) mutable {
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    for(int i = 0;i != 3;++i) {
                        if(tri[i] == vi)
                            return;
                        if(verts.hasProperty("collision_cancel") && verts("collision_cancel",tri[i]) > 1e-3)
                            return;
                    }

                    bool has_dynamic_points = verts("minv",vi) > eps;
                    for(int i = 0;i != 3;++i) {
                        if(invMass("minv",tri[i]) > eps)
                            has_dynamic_points = true;
                    }
                    if(!has_dynamic_points)
                        return;

                    vec3 ps[4] = {};
                    vec3 vs[4] = {};
                    vec4 bary{0,0,0,1};
                    vec4i inds{tri[0],tri[1],tri[2],vi};

                    for(int i = 0;i != 4;++i) {
                        ps[i] = verts.pack(dim_c<3>,xtag,inds[i]);
                        vs[i] = verts.pack(dim_c<3>,vtag,inds[i]);
                    }
                    auto alpha = (T)1.0;
                    if(!accd::ptccd(ps[3],ps[0],ps[1],ps[2],vs[3],vs[0],vs[1],vs[2],(T)0.2,(T)0,alpha))
                        return;
                    if(alpha < min_alpha) {
                        min_alpha = alpha;
                        min_ti = ti;
                    }
                };  
                bvh.iter_neighbors(bv,do_close_proximity_detection);  

                if(min_ti >= 0) {
                    auto tri = tris.pack(dim_c<3>,"inds",min_ti,int_c);
                    for(int i = 0;i != 3;++i) {
                        atomic_min(execTag,&tocs[tri[i]],min_alpha);
                    }
                    atomic_min(execTag,&tocs[vi],min_alpha);
                    atomic_add(execTag,&nm_ccd_fail[0],1);
                }            
        });        
        pol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            tocs = proxy<space>(tocs)] ZS_LAMBDA(int vi) mutable {
                verts.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,"v",vi) * tocs[vi] * 0.99;
        });
        ++nm_iters;
        std::cout << "CCD_COLLISION_ELIMATION_ITER[" << nm_iters << "] : nm_ccd_fail : " << nm_ccd_fail.getVal(0) << std::endl;
 
        }while(nm_ccd_fail.getVal(0) > 0);
}

template<typename Pol,  
    typename InverseMassTileVec,
    typename MassTileVec,
    typename PosTileVec,
    typename EdgeTileVec,
    typename ImpulseBuffer,
    typename ImpulseCountBuffer,
    typename EdgeBvh,
    typename EEHashMap,
    typename T = typename PosTileVec::value_type>
void calc_continous_self_EE_collision_impulse_with_toc(Pol& pol,
    const InverseMassTileVec& invMass,
    const MassTileVec& mass,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,
    const EdgeTileVec& edges,
    const T& thickness,
    const T& ignore_rest_shape_thickness,
    const size_t& start_edge_id,
    const size_t& end_edge_id,
    EdgeBvh& edgeCCDBvh,
    bool refit_bvh,
    EEHashMap& csEE,
    ImpulseBuffer& impulse_buffer,
    ImpulseCountBuffer& impulse_count,
    bool skip_too_close_pair_at_rest_configuration = false,
    bool use_collision_group = false,
    bool output_debug_inform = false,
    int collision_group_strategy = COLLISION_AMONG_SAME_GROUP | COLLISION_AMONG_DIFFERENT_GROUP) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        // constexpr auto exec_tag = wrapv<space>{};

        using vec2 = zs::vec<T,2>;
        using vec3 = zs::vec<T,3>;
        using vec4 = zs::vec<T,4>;
        using vec2i = zs::vec<int,2>;
        using vec4i = zs::vec<int,4>;
        constexpr auto eps = (T)1e-6;

        // auto edgeCCDBvh = bvh_t{};
        // std::cout << "build continous EE structure" << std::endl;


        // ALLOCATION BOTTLENECK 1
        auto edgeBvs = retrieve_bounding_volumes(pol,verts,edges,verts,wrapv<2>{},(T)1.0,(T)thickness,xtag,vtag);
        if(refit_bvh)
            edgeCCDBvh.refit(pol,edgeBvs);
        else
            edgeCCDBvh.build(pol,edgeBvs);

        // std::cout << "do continous EE collilsion detection" << std::endl;


        auto nm_test_edges = end_edge_id - start_edge_id;
        zs::Vector<float> tocs{edges.get_allocator(),nm_test_edges};
        pol(zs::range(nm_test_edges),[tocs = proxy<space>(tocs)] ZS_LAMBDA(auto ei) mutable {tocs[ei] = (T)1;});

        // ALLOCATION BOTTLENECK 2
        // zs::bht<int,2,int> csEE{edges.get_allocator(),100000};
        csEE.reset(pol,true);

        auto has_collision_group = verts.hasProperty("collision_group");
        auto has_rest_shape = verts.hasProperty("X");

        auto execTag = wrapv<space>{};

        pol(zs::range(nm_test_edges),[
            collision_group_strategy = collision_group_strategy,
            use_collision_group = use_collision_group,
            skip_rest = skip_too_close_pair_at_rest_configuration,
            has_collision_group = has_collision_group,
            has_rest_shape = has_rest_shape,
            xtag = xtag,
            vtag = vtag,
            tocs = proxy<space>(tocs),
            verts = proxy<space>({},verts),
            edges = proxy<space>({},edges),
            invMass = proxy<space>({},invMass),
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count),
            thickness = thickness,
            output_debug_inform = output_debug_inform,
            eps = eps,
            ignore_rest_shape_thickness = ignore_rest_shape_thickness,
            csEE = proxy<space>(csEE),
            execTag = execTag,
            edgeBvs = proxy<space>(edgeBvs),
            bvh = proxy<space>(edgeCCDBvh)] ZS_LAMBDA(int ei) mutable {
                auto ea = edges.pack(dim_c<2>,"inds",ei,int_c);
                // auto bv = edgeBvs[ei];
                for(int i = 0;i != 2;++i)
                    if(verts.hasProperty("collision_cancel") && verts("collision_cancel",ea[i]) > 1e-3)
                        return;

                auto v0 = verts.pack(dim_c<3>, xtag, ea[0]);
                auto v1 = verts.pack(dim_c<3>, xtag, ea[1]);
                auto dir0 = verts.pack(dim_c<3>, vtag, ea[0]);
                auto dir1 = verts.pack(dim_c<3>, vtag, ea[1]);   
                // T alpha = (T)1.0;
                auto bv = bv_t{get_bounding_box(v0, v0 + dir0)};
                merge(bv, v1);
                merge(bv, v1 + dir1);
                // bv._min -= xi;
                // bv._max += xi;
   
                int min_nei = -1;
                float min_alpha = 1; 

                auto do_close_proximity_detection = [&](int nei) mutable {
                    if(ei >= nei)
                        return;
                    auto eb = edges.pack(dim_c<2>,"inds",nei,int_c);
                    for(int i = 0;i != 2;++i){
                        if(eb[i] == ea[0] || eb[i] == ea[1])
                            return;
                    }

                    for(int i = 0;i != 2;++i)
                        if(verts.hasProperty("collision_cancel") && verts("collision_cancel",eb[i]) > 1e-3)
                            return;

                    if(has_collision_group) {
                        if(!(collision_group_strategy & COLLISION_AMONG_DIFFERENT_GROUP)) {
                            for(int i = 0;i != 2;++i) {
                                auto eaGroup = verts("collision_group",ea[i]);
                                for(int j = 0;j != 2;++j){
                                    auto ebGroup = verts("collision_group",eb[j]);
                                    if(zs::abs(eaGroup - ebGroup) > 0.1) {
                                        return;
                                    }
                                }
                            }
                        }
                        if(!(collision_group_strategy & COLLISION_AMONG_SAME_GROUP)) {
                            for(int i = 0;i != 2;++i) {
                                auto eaGroup = verts("collision_group",ea[i]);
                                for(int j = 0;j != 2;++j){
                                    auto ebGroup = verts("collision_group",eb[j]);
                                    if(zs::abs(eaGroup - ebGroup) < 0.1) {
                                        return;
                                    }
                                }
                            }
                        }
                    }
                        

                    auto has_dynamic_points = false;
                    for(int i = 0;i != 2;++i) {
                        if(invMass("minv",ea[i]) > eps)
                            has_dynamic_points = true;
                        if(invMass("minv",eb[i]) > eps)
                            has_dynamic_points = true;
                    }
                    if(!has_dynamic_points)
                        return;

                    if(has_rest_shape && skip_rest) {
                        vec3 rpas[2] = {};
                        vec3 rpbs[2] = {};
                        for(int i = 0;i != 2;++i) {
                            rpas[i] = verts.pack(dim_c<3>,"X",ea[i]);
                            rpbs[i] = verts.pack(dim_c<3>,"X",eb[i]);
                        }

                        auto is_same_collision_group = true;
                        if(use_collision_group) 
                            is_same_collision_group = zs::abs(verts("collision_group",ea[0]) - verts("collision_group",eb[0])) < 0.1;

                        if(LSL_GEO::get_edge_edge_distance(rpas[0],rpas[1],rpbs[0],rpbs[1]) < ignore_rest_shape_thickness && is_same_collision_group)
                            return;
                    }

                    vec4i inds{ea[0],ea[1],eb[0],eb[1]};
                    vec3 ps[4] = {};
                    vec3 vs[4] = {};
                    for(int i = 0;i != 4;++i) {
                        ps[i] = verts.pack(dim_c<3>,xtag,inds[i]);
                        vs[i] = verts.pack(dim_c<3>,vtag,inds[i]);
                    }

                    auto alpha = (T)1.0;
                    if(!accd::eeccd(ps[0],ps[1],ps[2],ps[3],vs[0],vs[1],vs[2],vs[3],(T)0.1,(T)thickness,alpha))
                        return;     

                    if(alpha < min_alpha) {
                        min_alpha = alpha;
                        min_nei = nei;
                    }
                };
                bvh.iter_neighbors(bv,do_close_proximity_detection);

                if(min_nei >= 0) {
                    csEE.insert(vec2i{ei,min_nei});
                    tocs[ei] = min_alpha;
                }
        });

        // std::cout << "nm close EE proxy : " << csEE.size() << std::endl;
        // std::cout << "compute continous EE proxy impulse" << std::endl;

        pol(zip(zs::range(csEE.size()),csEE._activeKeys),[
            xtag = xtag,
            vtag = vtag,
            tocs = proxy<space>(tocs),
            verts = proxy<space>({},verts),
            edges = proxy<space>({},edges),
            invMass = proxy<space>({},invMass),
            mass = proxy<space>({},mass),
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count),
            // thickness = thickness,
            output_debug_inform = output_debug_inform,
            eps = eps,
            execTag = execTag,
            edgeBvs = proxy<space>(edgeBvs),
            bvh = proxy<space>(edgeCCDBvh)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto ei = pair[0];
                auto nei = pair[1];
                auto ea = edges.pack(dim_c<2>,"inds",ei,int_c);
                vec3 ps[4] = {};
                ps[0] = verts.pack(dim_c<3>,xtag,ea[0]);
                ps[1] = verts.pack(dim_c<3>,xtag,ea[1]);
                vec3 vs[4] = {};
                vs[0] = verts.pack(dim_c<3>,vtag,ea[0]);
                vs[1] = verts.pack(dim_c<3>,vtag,ea[1]);

                vec4i inds{ea[0],ea[1],-1,-1};

                T minvs[4] = {};
                minvs[0] = verts("minv",ea[0]);
                minvs[1] = verts("minv",ea[1]);

                vec3 imps[4] = {};

                // auto do_close_proximity_detection = [&](int nei) mutable {
                auto eb = edges.pack(dim_c<2>,"inds",nei,int_c);

                for(int i = 0;i != 2;++i) {
                    ps[i + 2] = verts.pack(dim_c<3>,xtag,eb[i]);
                    vs[i + 2] = verts.pack(dim_c<3>,vtag,eb[i]);
                    inds[i + 2] = eb[i];
                    minvs[i + 2] = verts("minv",eb[i]);
                }

                // if(!compute_continous_EE_collision_impulse(ps,vs,minvs,imps))
                //     return;    

                // auto alpha = (T)1.0;
                // if(!accd::eeccd(ps[0],ps[1],ps[2],ps[3],vs[0],vs[1],vs[2],vs[3],(T)0.2,(T)0,alpha))
                //     return;                

                auto alpha = tocs[ei];

                vec3 nps[4] = {};
                for(int i = 0;i != 4;++i)
                    nps[i] = ps[i] + vs[i] * alpha;

                if((nps[1] - nps[0]).norm() < eps * 100 || (nps[3] - nps[2]).norm() < eps * 100)
                    return;

                if((nps[1] - nps[0]).cross(nps[3] - nps[2]).norm() < eps)
                    return;

                vec2 edge_bary{};
                LSL_GEO::get_edge_edge_barycentric_coordinates(nps[0],nps[1],nps[2],nps[3],edge_bary);
                 for(int i = 0;i != 2;++i) {
                    edge_bary[i] = edge_bary[i] < 0 ? 0 : edge_bary[i];
                    edge_bary[i] = edge_bary[i] > 1 ? 1 : edge_bary[i];
                 }
                vec4 bary{edge_bary[0] - 1,-edge_bary[0],1 - edge_bary[1],edge_bary[1]};

                auto rv = vec3::zeros();
                auto pr = vec3::zeros();
                for(int i = 0;i != 4;++i) {
                    rv += vs[i] * bary[i];
                    pr += nps[i] * bary[i];
                }

                auto collision_nrm = pr.normalized();
                auto rv_nrm = collision_nrm.dot(rv);
                auto cm = (T).0;
                for(int i = 0;i != 4;++i)
                    cm += bary[i] * bary[i] / mass("m",inds[i]);
                if(cm < eps)
                    return;
                auto impulse = -collision_nrm * rv_nrm * ((T)1 - alpha);
                // if(output_debug_inform)
                //     printf("find EE collision pairs[%d %d] with ccd\n",ei,nei);

                for(int i = 0;i != 4;++i) {
                    if(invMass("minv",inds[i]) < eps)
                        continue;
                    auto beta = (bary[i] * invMass("minv",inds[i])) / cm;
                    atomic_add(execTag,&impulse_count[inds[i]],1);
                    for(int d = 0;d != 3;++d)
                        atomic_add(execTag,&impulse_buffer[inds[i]][d],impulse[d] * beta);
                }   
                // };
                // bvh.iter_neighbors(bv,do_close_proximity_detection);
        });

        // std::cout << "finish computing continous EE proxy impulse" << std::endl;
}




template<typename Pol,
    typename PosTileVec,
    typename CollisionBuffer,
    typename T = typename PosTileVec::value_type>
void apply_impulse(Pol& pol,
    PosTileVec& verts,const zs::SmallString& vtag,
    const T& imminent_restitution_rate,
    const T& imminent_relaxation_rate,
    CollisionBuffer& imminent_collision_buffer,
    const size_t& nm_apply_imps) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        // constexpr auto exec_tag = wrapv<space>{};
        using vec3 = zs::vec<T,3>;
        constexpr auto eps = (T)1e-6;

        zs::Vector<vec3> impulse_buffer{verts.get_allocator(),verts.size()};   
        pol(zs::range(impulse_buffer),[] ZS_LAMBDA(auto& imp) mutable {imp = vec3::zeros();});
        zs::Vector<int> impulse_count{verts.get_allocator(),verts.size()};
        pol(zs::range(impulse_count),[] ZS_LAMBDA(auto& c) mutable {c = 0;});

        auto execTag = wrapv<space>{};
        pol(zs::range(imminent_collision_buffer.size()),[
            verts = proxy<space>({},verts),
            vtag = zs::SmallString(vtag),
            imminent_collision_buffer = proxy<space>({},imminent_collision_buffer),
            execTag = execTag,
            eps = eps,
            restitution_rate = imminent_restitution_rate,
            impulse_count = proxy<space>(impulse_count),
            impulse_buffer = proxy<space>(impulse_buffer)] ZS_LAMBDA(auto ci) mutable {
                auto inds = imminent_collision_buffer.pack(dim_c<4>,"inds",ci,int_c);
                auto bary = imminent_collision_buffer.pack(dim_c<4>,"bary",ci);
                auto impulse = imminent_collision_buffer.pack(dim_c<3>,"impulse",ci);
                if(impulse.norm() < eps)
                    return;

                T cminv = 0;
                for(int i = 0;i != 4;++i){
                    if(verts("minv",inds[i]) < eps)
                        continue;
                    cminv += bary[i] * bary[i] * verts("minv",inds[i]);
                }
                // impulse /= beta;

                for(int i = 0;i != 4;++i) {
                    if(verts("minv",inds[i]) < eps)
                        continue;
                    auto beta = verts("minv",inds[i]) * bary[i] / cminv;
                    atomic_add(execTag,&impulse_count[inds[i]],1);
                    for(int d = 0;d != 3;++d)
                        atomic_add(execTag,&impulse_buffer[inds[i]][d],impulse[d] * beta);
                }
                
        });
        pol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtag = zs::SmallString(vtag),
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count),
            relaxation_rate = imminent_relaxation_rate,
            eps = eps,
            execTag = execTag] ZS_LAMBDA(int vi) mutable {
            if(impulse_buffer[vi].norm() < eps || impulse_count[vi] == 0)
                return;
            auto impulse = relaxation_rate * impulse_buffer[vi] / (T)impulse_count[vi];
            // auto dp = impulse * verts("minv",vi);

            for(int i = 0;i != 3;++i)   
                atomic_add(execTag,&verts(vtag,i,vi),impulse[i]);
        }); 
}

template<typename Pol,
    typename PosTileVec,
    typename CollisionBuffer,
    typename NmImpulseCount,
    typename T = typename PosTileVec::value_type>
void apply_impulse(Pol& pol,
    PosTileVec& verts,const zs::SmallString& vtag,
    const T& imminent_restitution_rate,
    const T& imminent_relaxation_rate,
    const T& res_threshold,
    CollisionBuffer& imminent_collision_buffer,
    NmImpulseCount& nm_impulse_count,
    const size_t& nm_impulses) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        // constexpr auto exec_tag = wrapv<space>{};
        using vec3 = zs::vec<T,3>;
        constexpr auto eps = (T)1e-6;

        zs::Vector<vec3> impulse_buffer{verts.get_allocator(),verts.size()};   
        pol(zs::range(impulse_buffer),[] ZS_LAMBDA(auto& imp) mutable {imp = vec3::zeros();});
        zs::Vector<int> impulse_count{verts.get_allocator(),verts.size()};
        pol(zs::range(impulse_count),[] ZS_LAMBDA(auto& c) mutable {c = 0;});

        auto execTag = wrapv<space>{};
        pol(zs::range(nm_impulses),[
            verts = proxy<space>({},verts),
            vtag = zs::SmallString(vtag),
            imminent_collision_buffer = proxy<space>({},imminent_collision_buffer),
            execTag = execTag,
            res_threshold = res_threshold,
            eps = eps,
            restitution_rate = imminent_restitution_rate,
            impulse_count = proxy<space>(impulse_count),
            impulse_buffer = proxy<space>(impulse_buffer)] ZS_LAMBDA(auto ci) mutable {
                auto inds = imminent_collision_buffer.pack(dim_c<4>,"inds",ci,int_c);
                auto bary = imminent_collision_buffer.pack(dim_c<4>,"bary",ci);
                auto impulse = imminent_collision_buffer.pack(dim_c<3>,"impulse",ci);
                if(impulse.norm() < eps)
                    return;

                T cminv = 0;
                for(int i = 0;i != 4;++i) {
                    if(verts("minv",inds[i]) < eps)
                        continue;
                    cminv += bary[i] * bary[i] * verts("minv",inds[i]);
                }
                // impulse /= beta;

                for(int i = 0;i != 4;++i) {
                    if(verts("minv",inds[i]) < eps)
                        continue;
                    auto beta = verts("minv",inds[i]) * bary[i] / cminv;
                    atomic_add(execTag,&impulse_count[inds[i]],1);
                    for(int d = 0;d != 3;++d)
                        atomic_add(execTag,&impulse_buffer[inds[i]][d],impulse[d] * beta);
                }
                
        });
        nm_impulse_count.setVal(0);
        pol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtag = zs::SmallString(vtag),
            nm_impulse_count = proxy<space>(nm_impulse_count),
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count),
            relaxation_rate = imminent_relaxation_rate,
            eps = eps,
            res_threshold = res_threshold,
            execTag = execTag] ZS_LAMBDA(int vi) mutable {
            if(impulse_buffer[vi].norm() < eps || impulse_count[vi] == 0)
                return;
            auto impulse = relaxation_rate * impulse_buffer[vi] / (T)impulse_count[vi];
            // auto dp = impulse * verts("minv",vi);

            if(impulse_buffer[vi].norm() < res_threshold)
                return;

            for(int i = 0;i != 3;++i)   
                atomic_add(execTag,&verts(vtag,i,vi),impulse[i]);
            // if(impulse_buffer[vi].norm() > res_threshold)
            atomic_add(execTag,&nm_impulse_count[0],1);
        }); 
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
            exec_tag = exec_tag,
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
        auto ring_mask_width = GIA::do_global_self_intersection_analysis(pol,
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
                    T distance = LSL_GEO::get_vertex_triangle_distance(tvs[0],tvs[1],tvs[2],p,barySum);

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
            exec_tag = exec_tag,
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
        });
}

};

};