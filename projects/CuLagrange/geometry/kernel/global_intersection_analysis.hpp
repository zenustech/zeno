#pragma once


#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/Bvh.hpp"
#include "Utils.hpp"
#include "compute_characteristic_length.hpp"

#include <iostream>
#include "tiled_vector_ops.hpp"
#include "topology.hpp"

#include "geo_math.hpp"


namespace zeno {

namespace GIA {

#define USE_FAST_TRI_SEG_INTERSECTION

    
    constexpr int GIA_BACKGROUND_COLOR = 0;
    constexpr int GIA_TURNING_POINTS_COLOR = 1;
    constexpr int GIA_RED_COLOR = 2;
    constexpr int GIA_WHITE_COLOR = 3;
    constexpr int GIA_BLACK_COLOR = 4;
    constexpr int DEFAULT_MAX_GIA_INTERSECTION_PAIR = 100000;
    constexpr int DEFAULT_MAX_NM_TURNING_POINTS = 500;
    constexpr int DEFAULT_MAX_BOUNDARY_POINTS = 500;
    
    constexpr auto GIA_IMC_GRAD_BUFFER_KEY = "GIA_IMC_GRAD_BUFFER_KEY";
    constexpr auto GIA_CS_ET_BUFFER_KEY = "GIA_CS_ET_BUFFER_KEY";
    constexpr auto GIA_CS_EKT_BUFFER_KEY = "GIA_CS_EKT_BUFFER_KEY";
    constexpr auto GIA_VTEMP_BUFFER_KEY = "GIA_VTEMP_BUFFER_KEY";
    constexpr auto GIA_TRI_BVH_BUFFER_KEY = "GIA_TRI_BVH_BUFFER_KEY";
    constexpr auto GIA_KTRI_BVH_BUFFER_KEY = "GIA_TRI_BVH_BUFFER_KEY";

    template<typename HalfEdgeTileVecProxy>
    constexpr auto is_neighboring_halfedges(const HalfEdgeTileVecProxy& halfedges,const int& h0,const int& h1) {
        auto nh0 = h0;
        for(int i = 0;i != 2;++i) {
            nh0 = get_next_half_edge(nh0,halfedges,1);
            if(nh0 == h1)
                return true;
        }
        if(get_next_half_edge(h0,halfedges,0,true) == h1)
            return true;
        return false;
    }

    template<typename HalfEdgeTileVecProxy>
    constexpr auto is_boundary_halfedge(const HalfEdgeTileVecProxy& halfedges,const int& h) {
        if(get_next_half_edge(h,halfedges,0,true) < 0)
            return true;
        return false;
    }

    template<typename HalfEdgeTileVecProxy,typename TriTileVecProxy>
    constexpr int find_intersection_turning_point(const HalfEdgeTileVecProxy& halfedges,const TriTileVecProxy& tris,const zs::vec<int,2>& pair) {
        using namespace zs;
        
        auto hi = pair[0];
        auto ti = pair[1];
        auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));

        auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
        auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);

        for(int i = 0;i != 3;++i)
            for(int j = 0;j != 3;++j)
                if(tri[i] == htri[j])
                    return tri[i];

        return -1;
    }


    template<typename HalfEdgeTileVecProxy,typename TriTileVecProxy>
    constexpr auto is_closed_intersection_pair(const HalfEdgeTileVecProxy& halfedges,const TriTileVecProxy& tris,const zs::vec<int,2>& pair) {
        if(is_boundary_halfedge(halfedges,pair[0]))
            return false;
        if(find_intersection_turning_point(halfedges,tris,pair) >= 0)
            return false;
        return true;
    }

// retrieve all the intersection pairs and decide whether they are:
//      closed intersection
//      turning point intersection
//      boudary edge intersection
    template<typename Pol,
        typename PosTileVec,
        typename TriTileVec,
        typename HETileVec,
        typename TriBVH>
    void retrieve_self_intersection_tri_halfedge_pairs(Pol& pol,
        const PosTileVec& verts, const zs::SmallString& xtag,
        const TriTileVec& tris,
        const HETileVec& halfedges,
        TriBVH& tri_bvh,
        zs::bht<int,2,int>& res,
        bool refit_bvh = false) {
            using namespace zs;
            using vec2i = zs::vec<int,2>;
            using bv_t = typename ZenoParticles::lbvh_t::Box;
            using vec3 = zs::vec<T,3>;
            using table_vec2i_type = zs::bht<int,2,int>;

            constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
            constexpr auto exec_tag = wrapv<space>{};
            constexpr auto eps = 1e-6;

            auto tri_bvs = retrieve_bounding_volumes(pol,verts,tris,wrapv<3>{},0,xtag);
            
            if(refit_bvh)
                tri_bvh.refit(pol,tri_bvs);
            else
                tri_bvh.build(pol,tri_bvs);   

            res.reset(pol,true);
            pol(zs::range(halfedges.size()),[
                halfedges = proxy<space>({},halfedges),
                tris = proxy<space>({},tris),
                verts = proxy<space>({},verts),
                xtag = xtag,
                eps = eps,
                res = proxy<space>(res),
                tri_bvh = proxy<space>(tri_bvh)] ZS_LAMBDA(int hei) mutable {
                    auto ohei = zs::reinterpret_bits<int>(halfedges("opposite_he",hei));
                    if(hei > ohei && ohei >= 0)
                        return;
                    auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hei));
                    auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);
                    auto local_vert_id = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hei));
                    vec2i hedge{};
                    hedge[0] = htri[(local_vert_id + 0) % 3];
                    hedge[1] = htri[(local_vert_id + 1) % 3];

                    auto hv0 = verts.pack(dim_c<3>,xtag,hedge[0]);
                    auto hv1 = verts.pack(dim_c<3>,xtag,hedge[1]);
                    auto bv = bv_t{get_bounding_box(hv0,hv1)};

                    bool is_dynamic_edge = true;
                    if(verts.hasProperty("ani_mask")) {
                        is_dynamic_edge = false;
                        for(int i = 0;i != 2;++i)
                            if(verts("ani_mask",hedge[i]) < 0.99)
                                is_dynamic_edge = true;
                    }
                        
                    auto process_potential_he_tri_intersection_pairs = [&](int ti) mutable {
                        auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                        for(int i = 0;i != 3;++i)
                            if(tri[i] == hedge[0] || tri[i] == hedge[1])
                                return;

                        bool is_dynamic_tri = true;
                        if(verts.hasProperty("ani_mask")) {
                            is_dynamic_tri = false;
                            for(int i = 0;i != 3;++i)
                                if(verts("ani_mask",tri[i]) < 0.99)
                                    is_dynamic_tri = true;
                        }

                        if(!is_dynamic_tri && !is_dynamic_edge)
                            return;

                        vec3 tvs[3] = {};
                        for(int i = 0;i != 3;++i)
                            tvs[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                        if(!et_intersected(hv0,hv1,tvs[0],tvs[1],tvs[2]))
                            return;
                        res.insert(vec2i{hei,ti});
                        if(ohei >= 0)
                            res.insert(vec2i{ohei,ti});
                        // else
                        //     printf("find boundary edge intersected\n");
                    }; 

                    tri_bvh.iter_neighbors(bv,process_potential_he_tri_intersection_pairs);              
            });  
    }



// retrieve all the intersection pairs and decide whether they are:
//      closed intersection
//      turning point intersection
//      boudary edge intersection
    template<typename Pol,
        typename PosTileVec,
        typename TriTileVec,
        typename EdgeTileVec,
        typename TriBVH,
        typename BaryTileVec>
    void retrieve_self_intersection_tri_edge_pairs(Pol& pol,
        const PosTileVec& verts, const zs::SmallString& xtag,
        const TriTileVec& tris,
        const EdgeTileVec& edges,
        const TriBVH& tri_bvh,
        zs::bht<int,2,int>& res,
        BaryTileVec& bary_buffer,
        // bool refit_bvh = false,
        bool use_barycentric_interpolator = false) {
            using namespace zs;
            using vec2i = zs::vec<int,2>;
            using bv_t = typename ZenoParticles::lbvh_t::Box;
            using vec3 = zs::vec<T,3>;
            using table_vec2i_type = zs::bht<int,2,int>;

            constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
            constexpr auto exec_tag = wrapv<space>{};
            constexpr auto eps = 1e-6;


            zs::CppTimer timer;

            // timer.tick();
            // auto tri_bvs = retrieve_bounding_volumes(pol,verts,tris,wrapv<3>{},0,xtag);
            // timer.tock("retrieve_bounding_volumes");

            // timer.tick();
            // if(refit_bvh)
            //     tri_bvh.refit(pol,tri_bvs);
            // else
            //     tri_bvh.build(pol,tri_bvs);   
            // timer.tock("refit_bvh");

            res.reset(pol,true);
            timer.tick();


#ifdef USE_TMP_BUFFER_BEFORE_HASH
            constexpr size_t MAX_NM_INTERSECTION_PER_EDGE = 5;
            auto allocator = get_temporary_memory_source(pol);
            zs::Vector<int> tmp{allocator, edges.size() * MAX_NM_INTERSECTION_PER_EDGE};
            pol(zs::range(tmp),[] ZS_LAMBDA(auto& ti) mutable {ti = -1;});
            // zs::Vector<size_t> nm_ints{allocator,1};
            // nm_ints.setVal(0);
#endif
            pol(zs::range(edges.size()),[
                exec_tag = exec_tag,
                // use_dirty_bit = use_dirty_bit,
                // edges = proxy<space>({},edges),
                edges = edges.begin("inds", dim_c<2>, int_c),
                // edges = view<space>(edges), 
                // edgeIndsOffset = edges.getPropertyOffset("inds"),
                // tris = proxy<space>({},tris),
// #ifdef USE_FAST_TRI_SEG_INTERSECTION
                triNrmOffset = tris.getPropertyOffset("nrm"),
                triDOffset = tris.getPropertyOffset("d"),
                // triE0Offset = tris.getPropertyOffset("e0"),
                // triE1Offset = tris.getPropertyOffset("e1"),
                // triE2Offset = tris.getPropertyOffset("e2"),
                // triAlignAxisOffst = tris.getPropertyOffset("max_nrm_aligned_axis"),

                // triDirtyOffset = tris.getPropertyOffset("dirty"),
                // edgeDirtyOffset = edges.getPropertyOffset("dirty"),
// #endif
                triIndsOffset = tris.getPropertyOffset("inds"),
                tris = view<space>(tris),
                // verts = proxy<space>({},verts),
                verts = view<space>(verts),
                xOffset = verts.getPropertyOffset(xtag),
                aniMaskOffset = verts.getPropertyOffset("ani_mask"),
                hasAniMask = verts.hasProperty("ani_mask"),
                // xtag = xtag,
                eps = eps,
#ifdef USE_TMP_BUFFER_BEFORE_HASH    
                MAX_NM_INTERSECTION_PER_EDGE = MAX_NM_INTERSECTION_PER_EDGE,
                tmp = proxy<space>(tmp),
                // nm_ints = proxy<space>(nm_ints),
#else
                res = proxy<space>(res),
#endif
                use_barycentric_interpolator = use_barycentric_interpolator,
                bary_buffer = proxy<space>({},bary_buffer),
                tri_bvh = proxy<space>(tri_bvh)] ZS_LAMBDA(int ei) mutable {
                    // auto edge = edges.pack(dim_c<2>,edgeIndsOffset,ei,int_c);
                    auto edge = edges[ei];

                    auto vs = verts.pack(dim_c<3>, xOffset, edge[0]);
                    auto ve = verts.pack(dim_c<3>, xOffset, edge[1]);

                    auto bv = bv_t{get_bounding_box(vs,ve)};

                    bool edge_has_dynamic_points = true;
                    // if(hasAniMask) {
                    //     edge_has_dynamic_points = false;
                    //     for(int i = 0;i != 2;++i)
                    //         // if(verts("ani_mask",edge[i]) < 0.99)
                    //         if (verts(aniMaskOffset, edge[i]) < 0.99)
                    //             edge_has_dynamic_points = true;
                    // }
                        
                    int nm_intersect_tri = 0;
                    auto process_potential_ET_intersection_pairs = [&](int ti) mutable {
                        auto tri = tris.pack(dim_c<3>,triIndsOffset,ti,int_c);
                        for(int i = 0;i != 3;++i)
                            if(tri[i] == edge[0] || tri[i] == edge[1])
                                return;

                        auto tri_has_dynamic_points = true;
                        // if(hasAniMask) {
                        //     tri_has_dynamic_points = false;
                        //     for(int i = 0;i != 3;++i)
                        //         if(verts(aniMaskOffset,tri[i]) < 0.99)
                        //             tri_has_dynamic_points = true;
                        // }

                        if(!edge_has_dynamic_points && !tri_has_dynamic_points)
                            return;

                        vec3 tvs[3] = {};
                        for(int i = 0;i != 3;++i)
                            tvs[i] = verts.pack(dim_c<3>,xOffset,tri[i]);
#ifdef USE_FAST_TRI_SEG_INTERSECTION


                        // auto is_intersect = et_intersected(vs,ve,
                        //         verts.pack(dim_c<3>,xOffset,tri[0]),
                        //         verts.pack(dim_c<3>,xOffset,tri[1]),
                        //         verts.pack(dim_c<3>,xOffset,tri[2]));

                    // {
                        auto tnrm = tris.pack(dim_c<3>,triNrmOffset,ti);
                        auto d = tris(triDOffset,ti);

                        auto en = (ve - vs).norm();
                        if(en < eps)
                            return;

                        auto is_parallel = zs::abs(tnrm.dot(ve - vs) / en) < eps;
                        if(is_parallel)
                            return;

                        auto ts = vs.dot(tnrm) + d;
                        auto te = ve.dot(tnrm) + d;

                        if(ts * te > 0) {
                            // auto tnrm_recomp = LSL_GEO::facet_normal(tvs[0],tvs[1],tvs[2]);
                            // auto d_recomp = -tnrm_recomp.dot(tvs[0]);
                            // printf("is_intersect but ts * te > 0 : %f %f %f %f %f %f %f\n",(float)(ts * te),
                            //     (float)tnrm_recomp.dot(tnrm),(float)d,(float)d_recomp,
                            //     (float)(tnrm.dot(tvs[0]) + d),
                            //     (float)(tnrm.dot(tvs[1]) + d),
                            //     (float)(tnrm.dot(tvs[2]) + d));
                            return;
                        }

                        auto e0 = tvs[1] - tvs[0];
                        auto e1 = tvs[2] - tvs[0];
                        auto e2 = tvs[1] - tvs[2];

                        // auto ts_minus_te = ts - te;

                        auto t = ts / (ts - te);
                        auto p = vs + t * (ve - vs);
                        auto q = p - tvs[0];


                        auto n0 = e0.cross(q);
                        auto n1 = e1.cross(q);
                        if(n0.dot(n1) > 0) {
                            // if(is_intersect)
                            //     printf("should intersect but n0 * n1 = %f > 0\n",(float)(n0.dot(n1)));
                            return;
                        }

                        q = p - tvs[2];
                        n1 = e1.cross(q);
                        auto n2 = e2.cross(q);

                        if(n1.dot(n2) < 0) {
                            // if(is_intersect)
                            //     printf("should intersect but n1 * n2 = %f > 0\n",(float)(n0.dot(n1)));
                            return;
                        }

                        // we need a third test, for handling degenerate case
                        q = p - tvs[1];
                        n0 = e0.cross(q);
                        n2 = e2.cross(q);

                        if(n0.dot(n2) > 0) {
                            return;
                        }

                    // }
                    // if(!is_intersect) {
                    //     printf("should not pass here\n");
                    // }

                    // {
                    //     // if(!LSL_GEO::is_ray_triangle_intersection(vs,ve - vs,tvs[0],tvs[1],tvs[2],eps))
                    //     //     return;
                    // }
#else
                        if(!et_intersected(vs,ve,tvs[0],tvs[1],tvs[2]))
                            return;

                        auto tnrm = tris.pack(dim_c<3>,triNrmOffset,ti);
                        auto d = tris(triDOffset,ti);

                        auto ts = vs.dot(tnrm) + d;
                        auto te = ve.dot(tnrm) + d;

                        auto t = ts / (ts - te);
                        auto p = vs + t * (ve - vs);

#endif

#ifdef USE_TMP_BUFFER_BEFORE_HASH   
                        if(nm_intersect_tri < MAX_NM_INTERSECTION_PER_EDGE) {
                            tmp[ei *MAX_NM_INTERSECTION_PER_EDGE + nm_intersect_tri] = ti;
                            ++nm_intersect_tri;
                        }
                        // tmp[atomic_add(exec_tag,&nm_ints[0],(size_t)1)] = vec2i{ei,ti};   
#else
                        auto ci = res.insert(vec2i{ei,ti});
                        if(use_barycentric_interpolator) {
                            zs::vec<T,3> bary{};
                            LSL_GEO::get_vertex_triangle_barycentric_coordinates(tvs[0],tvs[1],tvs[2],p,bary);
                            bary_buffer.tuple(dim_c<4>,"bary",ci) = zs::vec<T,4>{1 - t,bary[0],bary[1],bary[2]};
                        }
                        bary_buffer.tuple(dim_c<2>,"inds",ci) = zs::vec<int,2>{ei,ti}.reinterpret_bits(float_c);
#endif
                    }; 
                    tri_bvh.iter_neighbors(bv,process_potential_ET_intersection_pairs);              
            });  
#ifdef USE_TMP_BUFFER_BEFORE_HASH  
            // auto ints_count = nm_ints.getVal(0);
            pol(zs::range(edges.size()),[
                tmp = proxy<space>(tmp),
                res = proxy<space>(res)] ZS_LAMBDA(int ei) mutable {
                    for(int i = 0;i != MAX_NM_INTERSECTION_PER_EDGE;++i) {
                        if(tmp[ei * MAX_NM_INTERSECTION_PER_EDGE + i] < 0)
                            break;
                        else
                            res.insert(zs::vec<int,2>{ei,tmp[ei * 5 + i]});
                    }
                    // res.insert(tmp[i]);
            });
#endif

            printf("nm_intersection_pair : %d\n",res.size());
            timer.tock("detect intersection pair");
    }



    template<typename Pol,
        typename PosTileVec,
        typename TriTileVec,
        typename EdgeTileVec,
        typename BaryTileVec,
        typename TriBVH>
    void retrieve_intersection_with_edge_tri_pairs(Pol& pol,
        const PosTileVec& verts, const zs::SmallString& xtag,
        const EdgeTileVec& edges,
        // const TriTileVec& tris,
        // const TriBVH& tri_bvh,
        const PosTileVec& kverts, const zs::SmallString& kxtag,
        // const EdgeTileVec& kedges,
        const TriTileVec& ktris,
        const TriBVH& ktri_bvh,
        zs::bht<int,2,int>& cs_ET,
        // zs::bht<int,2,int>& cs_KET,
        BaryTileVec& bary_buffer,
        bool use_barycentric_interpolator = false) {
            using namespace zs;
            using vec2i = zs::vec<int,2>;
            using bv_t = typename ZenoParticles::lbvh_t::Box;
            using vec3 = zs::vec<T,3>;
            using table_vec2i_type = zs::bht<int,2,int>;

            constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
            constexpr auto exec_tag = wrapv<space>{};
            constexpr auto eps = 1e-6;

            zs::CppTimer timer;
            cs_ET.reset(pol,true);

            timer.tick();

            pol(zs::range(edges.size()),[
                exec_tag = exec_tag,
                use_barycentric_interpolator = use_barycentric_interpolator,
                edges = edges.begin("inds", dim_c<2>, int_c),
                bary_buffer = proxy<space>({},bary_buffer),
                ktriNrmOffset = ktris.getPropertyOffset("nrm"),
                ktriDOffset = ktris.getPropertyOffset("d"),
                ktriIndsOffset = ktris.getPropertyOffset("inds"),
                ktris = view<space>(ktris),
                verts = view<space>(verts),
                xOffset = verts.getPropertyOffset(xtag),
                kverts = proxy<space>({},kverts),
                kxOffset = kverts.getPropertyOffset(kxtag),
                minvOffset = verts.getPropertyOffset("minv"),
                hasMinv = verts.hasProperty("minv"),
                kminvOffset = kverts.getPropertyOffset("minv"),
                hasKMinv = kverts.hasProperty("minv"),
                eps = eps,
                res = proxy<space>(cs_ET),
                ktri_bvh = proxy<space>(ktri_bvh)] ZS_LAMBDA(int ei) mutable {
                    auto edge = edges[ei];

                    auto vs = verts.pack(dim_c<3>, xOffset, edge[0]);
                    auto ve = verts.pack(dim_c<3>, xOffset, edge[1]);

                    auto bv = bv_t{get_bounding_box(vs,ve)};

                    auto process_potential_EKT_intersection_pairs = [&](int kti) mutable {
                        auto ktri = ktris.pack(dim_c<3>,ktriIndsOffset,kti,int_c);

                        vec3 ktvs[3] = {};
                        for(int i = 0;i != 3;++i)
                            ktvs[i] = kverts.pack(dim_c<3>,kxOffset,ktri[i]);

                        // bool ktri_has_dynamic_points = true;
                        // if(hasKMinv) {
                        //     ktri_has_dynamic_points = false;
                        //     for(int i = 0;i != 3;++i) {
                        //         if (kverts(kminvOffset, ktri[i]) > eps)
                        //             ktri_has_dynamic_points = true;
                        //     }
                        // }
                        // if(!edge_has_dynamic_points && !ktri_has_dynamic_points)
                        //     return;

#ifdef USE_FAST_TRI_SEG_INTERSECTION

                        auto ktnrm = ktris.pack(dim_c<3>,ktriNrmOffset,kti);
                        auto d = ktris(ktriDOffset,kti);

                        auto is_parallel = zs::abs(ktnrm.dot(ve - vs)) < eps;
                        if(is_parallel)
                            return;

                        auto ts = vs.dot(ktnrm) + d;
                        auto te = ve.dot(ktnrm) + d;

                        if(ts * te > 0) 
                            return;

                        auto e0 = ktvs[1] - ktvs[0];
                        auto e1 = ktvs[2] - ktvs[0];
                        auto e2 = ktvs[1] - ktvs[2];

                        auto t = ts / (ts - te);
                        auto p = vs + t * (ve - vs);
                        auto q = p - ktvs[0];


                        auto n0 = e0.cross(q);
                        auto n1 = e1.cross(q);
                        if(n0.dot(n1) > -eps) {
                            // if(is_intersect)
                            //     printf("should intersect but n0 * n1 = %f > 0\n",(float)(n0.dot(n1)));
                            return;
                        }


                        q = p - ktvs[2];
                        n1 = e1.cross(q);
                        auto n2 = e2.cross(q);

                        if(n1.dot(n2) < eps) {
                            return;
                        }

                        q = p - ktvs[1];
                        n0 = e0.cross(q);
                        n2 = e2.cross(q);

                        if(n0.dot(n2) > 0) {
                            return;
                        }

#else
                        if(!et_intersected(vs,ve,ktvs[0],ktvs[1],ktvs[2]))
                            return;

                        auto ktnrm = ktris.pack(dim_c<3>,ktriNrmOffset,kti);
                        auto d = ktris(ktriDOffset,kti);

                        auto ts = vs.dot(ktnrm) + d;
                        auto te = ve.dot(ktnrm) + d;

                        auto t = ts / (ts - te);
                        auto p = vs + t * (ve - vs);
#endif
                        auto ci = res.insert(vec2i{ei,kti});
                        if(use_barycentric_interpolator) {
                            zs::vec<T,3> bary{};
                            LSL_GEO::get_vertex_triangle_barycentric_coordinates(ktvs[0],ktvs[1],ktvs[2],p,bary);
                            bary_buffer.tuple(dim_c<4>,"bary",ci) = zs::vec<T,4>{1-t,bary[0],bary[1],bary[2]};
                        }
                        bary_buffer.tuple(dim_c<2>,"inds",ci) = zs::vec<int,2>{ei,kti}.reinterpret_bits(float_c);

                    }; 
                    ktri_bvh.iter_neighbors(bv,process_potential_EKT_intersection_pairs);              
            });  

//             cs_KET.reset(pol,true);
//             pol(zs::range(kedges.size()),[
//                 exec_tag = exec_tag,
//                 use_barycentric_interpolator = use_barycentric_interpolator,
//                 bary_buffer = proxy<space>({},bary_buffer),
//                 kedges = kedges.begin("inds", dim_c<2>, int_c),
//                 kverts = proxy<space>({},kverts),
//                 kxOffset = kverts.getPropertyOffset(kxtag),
//                 verts = view<space>(verts),
//                 xOffset = verts.getPropertyOffset(xtag),
//                 tris = view<space>(tris),
//                 triNrmOffset = tris.getPropertyOffset("nrm"),
//                 triDOffset = tris.getPropertyOffset("d"),
//                 triIndsOffset = tris.getPropertyOffset("inds"),
//                 aniMaskOffset = verts.getPropertyOffset("ani_mask"),
//                 hasAniMask = verts.hasProperty("ani_mask"),
//                 eps = eps,
//                 res = proxy<space>(cs_KET),
//                 tri_bvh = proxy<space>(tri_bvh)] ZS_LAMBDA(int kei) mutable {
//                     auto kedge = kedges[kei];
//                     auto kvs = kverts.pack(dim_c<3>, kxOffset, kedge[0]);
//                     auto kve = kverts.pack(dim_c<3>, kxOffset, kedge[1]);

//                     auto kbv = bv_t{get_bounding_box(kvs,kve)};

//                     auto process_potential_KET_intersection_pairs = [&](int ti) mutable {
//                         auto tri = tris.pack(dim_c<3>,triIndsOffset,ti,int_c);

//                         if(hasAniMask) {
//                             auto tri_has_dynamic_points = false;
//                             for(int i = 0;i != 3;++i) {
//                                 // if(verts("ani_mask",edge[i]) < 0.99)
//                                 if (verts(aniMaskOffset, tri[i]) < 0.99)
//                                     tri_has_dynamic_points = true;
//                             }
//                             if(!tri_has_dynamic_points)
//                                 return;
//                         }

//                         vec3 tvs[3] = {};
//                         for(int i = 0;i != 3;++i)
//                             tvs[i] = verts.pack(dim_c<3>,xOffset,tri[i]);

// #ifdef USE_FAST_TRI_SEG_INTERSECTION

//                         auto tnrm = tris.pack(dim_c<3>,triNrmOffset,ti);
//                         auto d = tris(triDOffset,ti);

//                         auto is_parallel = zs::abs(tnrm.dot(kve - kvs)) < eps;
//                         if(is_parallel)
//                             return;

//                         auto kts = kvs.dot(tnrm) + d;
//                         auto kte = kve.dot(tnrm) + d;

//                         if(kts * kte > 0) 
//                             return;

//                         auto e0 = tvs[1] - tvs[0];
//                         auto e1 = tvs[2] - tvs[0];
//                         auto e2 = tvs[1] - tvs[2];

//                         auto t = kts / (kts - kte);
//                         auto p = kvs + t * (kve - kvs);
//                         auto q = p - tvs[0];


//                         auto n0 = e0.cross(q);
//                         auto n1 = e1.cross(q);
//                         if(n0.dot(n1) > eps) {
//                             // if(is_intersect)
//                             //     printf("should intersect but n0 * n1 = %f > 0\n",(float)(n0.dot(n1)));
//                             return;
//                         }


//                         q = p - tvs[2];
//                         n1 = e1.cross(q);
//                         auto n2 = e2.cross(q);

//                         if(n1.dot(n2) < -eps) {
//                             return;
//                         }

// #else
//                         if(!et_intersected(kvs,kve,tvs[0],tvs[1],tvs[2]))
//                             return;
// #endif

//                         auto ci = res.insert(vec2i{kei,ti});
//                         // auto ci = res.insert(vec2i{ei,kti});
//                         if(use_barycentric_interpolator) {
//                             zs::vec<T,3> bary{};
//                             LSL_GEO::get_vertex_triangle_barycentric_coordinates(tvs[0],tvs[1],tvs[2],p,bary);
//                             bary_buffer.tuple(dim_c<4>,"bary",ci) = zs::vec<T,4>{t,bary[0],bary[1],bary[2]};
//                         }

//                     }; 
//                     tri_bvh.iter_neighbors(kbv,process_potential_KET_intersection_pairs);              
//             });  
            timer.tock("detect KET or EKT intersection pair");
    }

// extract all the intersection loops and decide whether they are:
//  a. closed loop               ->  without boundary or turning point intersection
//  b. eight path                ->  with with two turning point intersections who share a common turning point
//  c. LL path                   ->  with two turning point who have different turning point
//  d. BLI                       ->  with one turning point, and extent to a boundary point
//  e. cross path                ->  with one turning point, and extent to two boundary points
//  f. BB/II path                ->  with two boundary points at the same side
//  g. BI path                   ->  with two boundary points at different side      

    constexpr auto IS_BOUDARY = 1 << 0;
    constexpr auto IS_LOOP = 1 << 1;

    template<typename VertTileVecHost,
        typename TriTileVecHost,
        typename HalfEdgeTileVecHost>
    void compute_HT_intersection_barycentric(const VertTileVecHost& verts,const zs::SmallString& xtag,
            const TriTileVecHost& tris,
            const HalfEdgeTileVecHost& halfedges,
            const zs::vec<int,2>& pair,
            zs::vec<T,2>& edge_bary,
            zs::vec<T,3>& tri_bary) {
        using namespace zs;

        auto hi = pair[0];
        auto ti = pair[1];
        auto edge = half_edge_get_edge(hi,halfedges,tris);
        auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

        zs::vec<T,3> eps[2] = {};
        zs::vec<T,3> tps[3] = {};
        for(int i = 0;i != 2;++i)
            eps[i] = verts.pack(dim_c<3>,xtag,edge[i]);
        for(int i = 0;i != 3;++i)
            tps[i] = verts.pack(dim_c<3>,xtag,tri[i]);
        
        LSL_GEO::get_segment_triangle_intersection_barycentric_coordinates(eps[0],eps[1],tps[0],tps[1],tps[2],edge_bary,tri_bary);
    }

    template<typename TriTileVecHost,
        typename HalfEdgeTileVecHost,
        typename IntVector,
        typename HTHashMapHost>
    constexpr size_t group_intersections_within_the_same_contour(const TriTileVecHost& tris,
            const HalfEdgeTileVecHost& halfedges,
            const HTHashMapHost& csHT,
            IntVector& intersection_group_tag) {
        return 0;
    }



    template<typename TriTileVecHost,
        typename HalfEdgeTileVecHost,
        typename IntVector,
        typename Vec2iVector,
        typename HTHashMapHost>
    constexpr bool trace_intersection_loop(const TriTileVecHost& tris,
            const HalfEdgeTileVecHost& halfedges,
            const HTHashMapHost& csHT,
            const zs::vec<int,2>& trace_start,
            IntVector& intersection_mask,
            IntVector& loop_sides,
            Vec2iVector& loop_buffers,
            Vec2iVector& loop_buffers_0,
            Vec2iVector& loop_buffers_1) {
        using namespace zs;
        using vec2i = zs::vec<int,2>;        
        // trace from the boundary loop first, find all the BB\BLI\BI path
        loop_buffers_0.clear();
        loop_buffers_1.clear();
        loop_buffers.clear();
        loop_sides.clear();

        loop_buffers_0.push_back(trace_start);
        loop_buffers.push_back(trace_start);
        loop_sides.push_back(0);

        bool use_input_buffer0 = true;
        auto cur_trace = trace_start;

        auto id = csHT.query(cur_trace);
        if(id < 0)
            std::cout << "fail querying start_trace" << std::endl;
        intersection_mask[id] = 1;

        if(find_intersection_turning_point(halfedges,tris,cur_trace) >= 0) {
            cur_trace[0] = zs::reinterpret_bits<int>(halfedges("opposite_he",cur_trace[0]));
            if(cur_trace[0] < 0) {
                std::cout << "init turning pair is also a boundary pair" << std::endl;
                return use_input_buffer0;
            }
            loop_buffers_0.push_back(cur_trace);
            loop_buffers.push_back(cur_trace);
            loop_sides.push_back(0);
            auto id = csHT.query(cur_trace);
            // if(id < 0) {
            //     std::cout << "fail querying the opposite pair of init turning point" << std::endl;
            //     return use_input_buffer0;
            // }
            intersection_mask[id] = 1;
        }

        // std::cout << "entering the loop" << std::endl;

        do {
            auto hi = cur_trace[0];
            auto ti = cur_trace[1];
            // check whether the current hti intersect with ti
            if(find_intersection_turning_point(halfedges,tris,cur_trace) >= 0)
                break;

            bool find_intersection = false;
            auto nhi = hi;
            for(int i = 0;i != 2;++i) {
                // std::cout << "try_finding ajacent halfedge" << std::endl;
                nhi = get_next_half_edge(nhi,halfedges,1);
                if(auto id = csHT.query(vec2i{nhi,ti});id >= 0) {
                    intersection_mask[id] = 1;
                    cur_trace = vec2i{nhi,ti};
                    if(use_input_buffer0) {
                        loop_sides.push_back(0); 
                        loop_buffers_0.push_back(cur_trace);
                    }else {
                        loop_sides.push_back(1);
                        loop_buffers_1.push_back(cur_trace);
                    }
                    loop_buffers.push_back(cur_trace);
                    find_intersection = true;
                    break;
                }
            }
            if(!find_intersection) {
                // std::cout << "try_finding ajacent triangle" << std::endl;
                auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                auto thi = zs::reinterpret_bits<int>(tris("he_inds",ti));
                for(int i = 0;i != 3;++i) {
                    if(auto id = csHT.query(vec2i{thi,hti});id >= 0) {
                        intersection_mask[id] = 1;
                        cur_trace = vec2i{thi,hti};
                        use_input_buffer0 = !use_input_buffer0;
                        if(use_input_buffer0) {
                            loop_sides.push_back(0); 
                            loop_buffers_0.push_back(cur_trace);
                        }else {
                            loop_sides.push_back(1);
                            loop_buffers_1.push_back(cur_trace);
                        }
                        find_intersection = true;
                        loop_buffers.push_back(cur_trace);
                        break;
                    }
                    thi = get_next_half_edge(thi,halfedges,1);
                }
            }
            if(!find_intersection) {
                printf("abruptly end while the cur_trace[%d %d] is neither a turning point nor a boundary point\n",cur_trace[0],cur_trace[1]);
                // throw std::runtime_error("abruptly end while the cur_trace is neither a turning point nor a boundary point");
                return use_input_buffer0;
            }
            if(is_boundary_halfedge(halfedges,cur_trace[0])) {
                printf("find boundary halfedge %d\n",cur_trace[0]);
                break;
            }

            cur_trace[0] = zs::reinterpret_bits<int>(halfedges("opposite_he",cur_trace[0]));
            auto id = csHT.query(cur_trace);
            if(id < 0)
                std::cout << "fail querying the opposite pair of next point" << std::endl;
            
            if(intersection_mask[id] > 0) {
                return use_input_buffer0;
            }
            intersection_mask[id] = 1;
            if(use_input_buffer0) {
                loop_sides.push_back(0); 
                loop_buffers_0.push_back(cur_trace);
            }else {
                loop_sides.push_back(1);
                loop_buffers_1.push_back(cur_trace);
            }
            loop_buffers.push_back(cur_trace);
        }while(cur_trace[0] != trace_start[0] || cur_trace[1] != trace_start[1]);

        return use_input_buffer0;
    }


    template<typename TriTileVecHost,
        typename HalfEdgeTileVecHost,
        typename IntVector,
        typename HTHashMapHost>
    constexpr void trace_intersection_loop(const TriTileVecHost& tris,
            const HalfEdgeTileVecHost& halfedges,
            const HTHashMapHost& csHT,
            const zs::vec<int,2>& trace_start,
            IntVector& intersection_mask,
            IntVector& loop_sides,
            IntVector& loop_buffers) {
        using namespace zs;
        using vec2i = zs::vec<int,2>;        
        // trace from the boundary loop first, find all the BB\BLI\BI path
        loop_buffers.clear();
        loop_sides.clear();

        bool use_positive_side = true;
        auto cur_trace = trace_start;

        auto update_context = [&](int pair_id) {
            loop_buffers.push_back(pair_id);
            if(use_positive_side) {
                loop_sides.push_back(0); 
            }else {
                loop_sides.push_back(1);
            }
            intersection_mask[pair_id] = 1;
        };

        auto id = csHT.query(cur_trace);
        update_context(id);

        if(find_intersection_turning_point(halfedges,tris,cur_trace) >= 0) {
            cur_trace[0] = zs::reinterpret_bits<int>(halfedges("opposite_he",cur_trace[0]));
            if(cur_trace[0] < 0)
                return;
            auto id = csHT.query(cur_trace);
            update_context(id);
        }

        // std::cout << "search loop " << std::endl;

        do {
            auto hi = cur_trace[0];
            auto ti = cur_trace[1];
            // check whether the current hti intersect with ti
            if(find_intersection_turning_point(halfedges,tris,cur_trace) >= 0)
                break;

            bool find_intersection = false;
            auto nhi = hi;

            for(int i = 0;i != 2;++i) {
                nhi = get_next_half_edge(nhi,halfedges,1);
                if(auto id = csHT.query(vec2i{nhi,ti});id >= 0) {
                    update_context(id);
                    cur_trace = vec2i{nhi,ti};
                    find_intersection = true;
                    // std::cout << "find_neigh halfedge" << std::endl;
                    break;
                }
            }

            if(!find_intersection) {
                auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                auto thi = zs::reinterpret_bits<int>(tris("he_inds",ti));
                for(int i = 0;i != 3;++i) {
                    if(auto id = csHT.query(vec2i{thi,hti});id >= 0) {
                        use_positive_side = !use_positive_side;
                        update_context(id);
                        cur_trace = vec2i{thi,hti};
                        find_intersection = true;
                        // std::cout << "find_neigh triangle" << std::endl;
                        break;
                    }
                    thi = get_next_half_edge(thi,halfedges,1);
                }
            }
            if(!find_intersection) {
                printf("abruptly end while the cur_trace[%d %d] is neither a turning point nor a boundary point\n",cur_trace[0],cur_trace[1]);
                // throw std::runtime_error("abruptly end while the cur_trace is neither a turning point nor a boundary point");
                return;
            }
            if(is_boundary_halfedge(halfedges,cur_trace[0])) {
                printf("find boundary halfedge %d\n",cur_trace[0]);
                break;
            }

            cur_trace[0] = zs::reinterpret_bits<int>(halfedges("opposite_he",cur_trace[0]));
            // std::cout << "trace next halfedge " << cur_trace[0] << "\t" << cur_trace[1] << std::endl;
            auto id = csHT.query(cur_trace);

            if(intersection_mask[id] > 0) {
                return;
            }

            update_context(id);
        }while(cur_trace[0] != trace_start[0] || cur_trace[1] != trace_start[1]);

        std::cout << "finish tracing " << std::endl;
    }


    // NEW VERSION
    // instead of doing the coloring, we should refit the rbf function of each intersecting loop
    template<typename Pol,
        typename PosTileVec,
        typename TriTileVec,
        typename HalfEdgeTileVec,
        typename HTHashMap,
        auto space = Pol::exec_tag::value>
    int do_global_self_intersection_analysis(Pol& pol,
            const PosTileVec& verts,const zs::SmallString& xtag,
            const PosTileVec& verts_host,
            const TriTileVec& tris,
            const TriTileVec& tris_host,
            const HalfEdgeTileVec& halfedges,
            const HalfEdgeTileVec halfedges_host,
            HTHashMap& csHT,
            zs::Vector<int>& gia_res,zs::Vector<int>& tris_gia_res) {
        using namespace zs;
        using T = typename PosTileVec::value_type;
        using dtiles_t = zs::TileVector<T, 32>;
        using table_vec2i_type = zs::bht<int,2,int>;
        using table_int_type = zs::bht<int,1,int>;
        using vec2i = zs::vec<int,2>;

        constexpr auto exec_tag = wrapv<space>{};  
        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();

        // find the intersection pairs and clone the result to host backend
        auto tri_bvh = LBvh<3,int,T>{};
        retrieve_self_intersection_tri_halfedge_pairs(pol,verts,xtag,tris,halfedges,tri_bvh,csHT);
        bht<int,2,int> boundary_pairs_set{csHT.get_allocator(),DEFAULT_MAX_BOUNDARY_POINTS};
        boundary_pairs_set.reset(pol,true);
        bht<int,2,int> turning_pairs_set{csHT.get_allocator(),DEFAULT_MAX_NM_TURNING_POINTS};
        turning_pairs_set.reset(pol,true);

        pol(zip(zs::range(csHT.size()),csHT._activeKeys),[
            halfedges = proxy<space>({},halfedges),
            tris = proxy<space>({},tris),
            boundary_pairs_set = proxy<space>(boundary_pairs_set),
            turning_pairs_set = proxy<space>(turning_pairs_set)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                if(is_boundary_halfedge(halfedges,pair[0]))
                    boundary_pairs_set.insert(pair);
                if(find_intersection_turning_point(halfedges,tris,pair) >= 0)
                    turning_pairs_set.insert(pair);
        });

        auto boundary_pairs_vec = boundary_pairs_set._activeKeys.clone({zs::memsrc_e::host});
        auto turning_pairs_vec = turning_pairs_set._activeKeys.clone({zs::memsrc_e::host});

        std::vector<int> intersection_pairs_mask{csHT.size(),0};
        std::vector<int> gia_colors{verts.size(),GIA_BACKGROUND_COLOR};

        // for(int pi = 0;pi != turning_pairs_vec.size();++pi) {
        //     if(intersection_pairs_mask[pi])
        //         continue;

        //     std::fill(gia_colors.begin(),gia_colors.end(),GIA_BACKGROUND_COLOR);

        //     auto start_pair = turning_pairs_vec[pi];
        //     std::vector<vec2i> trace_0{},trace_1{};
        //     trace_intersection_loop(
        //         proxy<omp_space>({},tris_host),
        //         proxy<omp_space>({},halfedges_host),
        //         proxy<omp_space>(csHT_host),
        //         trace_0,trace_1);
            
        //     int nm_turning_pair = 0;
        //     vec2i turning_points{-1,-1};
        //     if(auto idx = find_intersection_turning_point(halfedges_host,tris_host,trace_0[0]);idx >= 0)
        //         turning_points[nm_turning_pair++] = idx;
        //     if(auto idx = find_intersection_turning_point(halfedges_host,tris_host,trace_0[trace_0.size() - 1]);idx >= 0)
        //         turning_points[nm_turning_pair++] = idx;
        //     if(auto idx = find_intersection_turning_point(halfedges_host,tris_host,trace_1[0]);idx >= 0)
        //         turning_points[nm_turning_pair++] = idx;
        //     if(auto idx = find_intersection_turning_point(halfedges_host,tris_host,trace_1[trace_1.size() - 1]);idx >= 0)
        //         turning_points[nm_turning_pair++] = idx;
        //     if(is_boundary_halfedge(proxy<omp_space>({},halfedges_host),trace_0[0][0]))
        //         continue;
        //     if(is_boundary_halfedge(proxy<omp_space>({},halfedges_host),trace_0[triace_0.size() - 1][0]))
        //         continue;
        //     if(is_boundary_halfedge(proxy<omp_space>({},halfedges_host),trace_1[0][0]))
        //         continue;
        //     if(is_boundary_halfedge(proxy<omp_space>({},halfedges_host),trace_1[trace_1.size() - 1][0]))
        //         continue;

        //     switch(nm_turning_pair) {
        //         case 2:  
        //             trace_0.extend(trace_1);
        //             flood_fill_the_smallest_path_region(
        //                 proxy<omp_space>({},verts_host),
        //                 xtag,
        //                 proxy<omp_space>({},tris_host),
        //                 proxy<omp_space>({},halfedges_host),
        //                 trace_0,
        //                 turning_points,
        //                 GIA_RED_COLOR,
        //                 GIA_BACKGROUND_COLOR,
        //                 GIA_TURNING_POINTS_COLOR,
        //                 gia_colors,
        //             );
        //             break;
        //         case 1: case 0:
        //             flood_fill_the_smallest_path_region(
        //                 proxy<omp_space>({},verts_host),
        //                 xtag,
        //                 proxy<omp_space>({},tris_host),
        //                 proxy<omp_space>({},halfedges_host),
        //                 trace_0,
        //                 turning_points,
        //                 GIA_BLACK_COLOR,
        //                 GIA_BACKGROUND_COLOR,
        //                 GIA_TURNING_POINTS_COLOR,
        //                 gia_colors,
        //             );
        //             flood_fill_the_smallest_path_region(
        //                 proxy<omp_space>({},verts_host),
        //                 xtag,
        //                 proxy<omp_space>({},tris_host),
        //                 proxy<omp_space>({},halfedges_host),
        //                 trace_1,
        //                 turning_points,
        //                 GIA_WHITE_COLOR,
        //                 GIA_BACKGROUND_COLOR,
        //                 GIA_TURNING_POINTS_COLOR,
        //                 gia_colors,
        //             );
        //             break;
        //         default:
        //             break;
        //     }
        // }
        
    }

    template<typename Pol,
        typename PosTileVec,
        typename TriTileVec,
        typename HalfEdgeTileVec,
        typename HTHashMap,
        typename ICMGradTileVec,
        auto space = Pol::exec_tag::value,
        typename T = typename PosTileVec::value_type>
    void eval_intersection_contour_minimization_gradient(Pol& pol,
            const PosTileVec& verts,const zs::SmallString& xtag,
            const HalfEdgeTileVec& halfedges,
            const TriTileVec& tris,
            HTHashMap& csHT,
            ICMGradTileVec& icm_grad,
            const HalfEdgeTileVec& halfedges_host,
            const TriTileVec& tris_host,
            bool use_global_scheme = false,
            bool use_barycentric_interpolator = false) {
        using namespace zs;
        auto exec_tag = wrapv<space>{};
        using vec2i = zs::vec<int,2>; 
        using vec3 = zs::vec<T,3>;
   
        icm_grad.resize(csHT.size());
        // local scheme
        pol(zip(zs::range(csHT.size()),csHT._activeKeys),[
            icm_grad = proxy<space>({},icm_grad),
            use_barycentric_interpolator = use_barycentric_interpolator,
            verts = proxy<space>({},verts),xtag = zs::SmallString(xtag),
            halfedges = proxy<space>({},halfedges),
            tris = proxy<space>({},tris)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto hi = pair[0];
                auto ti = pair[1];

                auto hedge = half_edge_get_edge(hi,halfedges,tris);

                auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);

                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                for(int i = 0;i != 3;++i)
                    if(tri[i] == hedge[0] || tri[i] == hedge[1])
                        return;

                vec3 halfedge_vertices[2] = {};
                vec3 tri_vertices[3] = {};
                vec3 htri_vertices[3] = {};

                for(int i = 0;i != 2;++i)
                    halfedge_vertices[i] = verts.pack(dim_c<3>,xtag,hedge[i]);

                for(int i = 0;i != 3;++i) {
                    tri_vertices[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                    htri_vertices[i] = verts.pack(dim_c<3>,xtag,htri[i]);
                }

                auto tri_normal = LSL_GEO::facet_normal(tri_vertices[0],tri_vertices[1],tri_vertices[2]);
                auto htri_normal = LSL_GEO::facet_normal(htri_vertices[0],htri_vertices[1],htri_vertices[2]);
                auto halfedge_orient = (halfedge_vertices[1] - halfedge_vertices[0]).normalized();

                auto R = tri_normal.cross(htri_normal).normalized();

                auto halfedge_local_vertex_id = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
                auto halfedge_opposite_vertex_id = htri[(halfedge_local_vertex_id + 2) % 3];
                auto halfedge_opposite_vertex = verts.pack(dim_c<3>,xtag,halfedge_opposite_vertex_id);

                auto in_plane_edge_normal = halfedge_opposite_vertex - halfedge_vertices[0];
                in_plane_edge_normal = in_plane_edge_normal - in_plane_edge_normal.dot(halfedge_orient) * halfedge_orient;
                in_plane_edge_normal = in_plane_edge_normal.normalized();

                if(R.dot(in_plane_edge_normal) < 0)
                    R = -R;
                
                const auto& E = halfedge_orient;
                const auto& N = tri_normal;

                auto ER = E.dot(R);
                auto EN = E.dot(N);

                auto G = R - static_cast<T>(2.0) * ER / EN * N;
                // https://diglib.eg.org/bitstream/handle/10.2312/SCA.SCA12.311-316/311-316.pdf?sequence=1
                // point it out the formula in the original paper by Nadia.etc is inaccurate, although the bad effect is limited
                // auto G = R - ER / EN * N;
                icm_grad.tuple(dim_c<3>,"grad",ci) = G;
                icm_grad.tuple(dim_c<2>,"inds",ci) = pair.reinterpret_bits(float_c);
        }); 
        // global scheme

        if(use_global_scheme) { // TODO: the nvcc compiler won't pass here, check later
            // constexpr auto omp_space = execspace_e::openmp;
            // auto ompPol = omp_exec();

            // icm_grad = icm_grad.clone({zs::memsrc_e::host});
            // auto icm_grad_proxy = proxy<omp_space>({},icm_grad);

            // std::vector<int> mask_host{};mask_host.resize(csHT.size());
            // ompPol(zs::range(mask_host),[] (auto& m) mutable {m = 0;});
            // std::vector<int> loop_sides{};
            // std::vector<int> loop_buffers{};
            // // std::vector<int> loop_buffers_0{};
            // // std::vector<int> loop_buffers_1{};

            // auto csHT_host = csHT.clone({memsrc_e::host});

            // // tackling L-L loop
            // auto globally_update_icm_gradient = [&](const auto& _loop_buffers,const auto& _loop_sides) {
            //     auto avg_icm_grad = zs::vec<T,3>::zeros();
            //     for(int i = 0;i != _loop_buffers.size();++i) {
            //         auto pair_id = _loop_buffers[i];
            //         auto grad = icm_grad_proxy.pack(dim_c<3>,"grad",pair_id);
            //         if(_loop_sides[i] > 0)
            //             grad = -grad;
            //         avg_icm_grad += grad;
            //     }
            //     avg_icm_grad /= (T)_loop_buffers.size();
            //     std::cout << "average icm grad : " << avg_icm_grad[0] << "\t" << avg_icm_grad[1] << "\t" << avg_icm_grad[2] << std::endl;
            //     ompPol(zs::range(_loop_buffers.size()),[icm_grad_proxy = icm_grad_proxy,
            //             &_loop_sides,
            //             &_loop_buffers,
            //             avg_icm_grad = avg_icm_grad] (const auto& li) mutable {
            //         auto pi = _loop_buffers[li];
            //         auto grad = _loop_sides[li] > 0 ? -avg_icm_grad : avg_icm_grad;
            //         icm_grad_proxy.tuple(dim_c<3>,"grad",pi) = grad;
            //     });                
            // };

            // while(true) {
            //     zs::vec<int,2> res{-1,-1};
            //     ompPol(zip(zs::range(csHT_host.size()),csHT_host._activeKeys),[
            //         &res,
            //         halfedges_host_proxy = proxy<omp_space>({},halfedges_host),
            //         tris_host_proxy = proxy<omp_space>({},tris_host),
            //         &mask_host] (auto ci,const auto& pair) mutable {
            //             if(mask_host[ci] == 1)
            //                 return;
            //             if(find_intersection_turning_point(halfedges_host_proxy,tris_host_proxy,pair) >= 0)
            //                 res = pair;
            //     });
            //     if(res[0] == -1)
            //         break;
            //     trace_intersection_loop(proxy<omp_space>({},tris_host),
            //         proxy<omp_space>({},halfedges_host),
            //         proxy<omp_space>(csHT_host),
            //         res,
            //         mask_host,
            //         loop_sides,
            //         loop_buffers);
            //     globally_update_icm_gradient(loop_buffers,loop_sides);
            // }
            // while(true){
            //     int bii = -1;
            //     ompPol(zip(zs::range(csHT_host.size()),csHT_host._activeKeys),[
            //         &bii,
            //         halfedges_host_proxy = proxy<omp_space>({},halfedges_host),
            //         &mask_host] (auto ci,const auto& pair) mutable {
            //             if(mask_host[ci] == 1)
            //                 return;
            //             if(is_boundary_halfedge(halfedges_host_proxy,pair[0]))
            //                 bii = ci;
            //     });
            //     if(bii == -1)
            //         break;

            //     std::cout << "find boundary pair : " << bii << std::endl;
            //     trace_intersection_loop(proxy<omp_space>({},tris_host),
            //         proxy<omp_space>({},halfedges_host),
            //         proxy<omp_space>(csHT_host),
            //         proxy<omp_space>(csHT_host._activeKeys)[bii],
            //         mask_host,
            //         loop_sides,
            //         loop_buffers);
            //     globally_update_icm_gradient(loop_buffers,loop_sides);
            // }

            // while(true) {
            //     int cii = -1;
            //     ompPol(zip(zs::range(csHT_host.size()),csHT_host._activeKeys),[
            //         halfedges_host_proxy = proxy<omp_space>({},halfedges_host),
            //         &cii,
            //         &mask_host] (auto ci,const auto& pair) mutable {
            //             if(mask_host[ci] == 1)
            //                 return;
            //             cii = ci;
            //     });
            //     if(cii == -1)
            //         break;
            //     std::cout << "find closed pair : " << cii << std::endl;
            //     trace_intersection_loop(proxy<omp_space>({},tris_host),
            //         proxy<omp_space>({},halfedges_host),
            //         proxy<omp_space>(csHT_host),
            //         proxy<omp_space>(csHT_host._activeKeys)[cii],
            //         mask_host,
            //         loop_sides,
            //         loop_buffers);
            //     globally_update_icm_gradient(loop_buffers,loop_sides);
            // }

            // icm_grad = icm_grad.clone({zs::memsrc_e::device,0});
        }
    }   


    template<typename VECTOR3d,typename SCALER>
    constexpr auto eval_HT_contour_minimization_gradient(const VECTOR3d& ht0,
        const VECTOR3d& ht1,
        const VECTOR3d& ht2,
        const VECTOR3d& htnrm,
        const VECTOR3d& tnrm,
        SCALER& normal_coeff) {
            // auto htri_normal = LSL_GEO::facet_normal(ht0,ht1,ht2);  
            // auto tri_normal = LSL_GEO::facet_normal(t0,t1,t2); 
            const auto& htri_normal = htnrm;
            const auto& tri_normal = tnrm;
            auto R = tri_normal.cross(htri_normal).normalized();

            auto E = (ht1 - ht0).normalized();
            
            auto in_plane_edge_normal = ht2 - ht0;
            in_plane_edge_normal = in_plane_edge_normal - in_plane_edge_normal.dot(E) * E;
            // in_plane_edge_normal = in_plane_edge_normal.normalized();
    
            if(R.dot(in_plane_edge_normal) < 0)
                R = -R;

            const auto& N = tri_normal;

            auto ER = E.dot(R);
            auto EN = E.dot(N);

        // https://diglib.eg.org/bitstream/handle/10.2312/SCA.SCA12.311-316/311-316.pdf?sequence=1
        // point it out the formula in the original paper by Nadia.etc is inaccurate, although the bad effect is limited
        // auto G = R - ER / EN * N;
            // auto G = R - static_cast<T>(2.0) * ER / EN * N;
            // G = R;
            normal_coeff = static_cast<T>(2.0) * ER / EN;
            // normal_coeff = ER / EN;

            return R;
    }

    template<typename Pol,
        typename PosTileVec,
        typename TriTileVec,
        typename EdgeTileVec,
        typename HalfEdgeTileVec,
        typename HTHashMap,
        typename ICMGradTileVec,
        auto space = Pol::exec_tag::value,
        typename T = typename PosTileVec::value_type>
    void eval_self_intersection_contour_minimization_gradient(Pol& pol,
            const PosTileVec& verts,const zs::SmallString& xtag,
            const EdgeTileVec& edges,
            const HalfEdgeTileVec& halfedges,
            const TriTileVec& tris,
            const T& maximum_correction,
            const T& progressive_slope,
            HTHashMap& csET,
            ICMGradTileVec& icm_grad) {
        using namespace zs;
        auto exec_tag = wrapv<space>{};
        using vec2i = zs::vec<int,2>; 
        using vec3 = zs::vec<T,3>;
   
        // icm_grad.resize(csET.size());
        // local scheme
        pol(zip(zs::range(csET.size()),csET._activeKeys),[
            icm_grad = proxy<space>({},icm_grad),
            h0 = maximum_correction,
            g02 = progressive_slope * progressive_slope,
            verts = proxy<space>({},verts),xtag = zs::SmallString(xtag),
            edges = proxy<space>({},edges),
            halfedges = proxy<space>({},halfedges),
            tris = proxy<space>({},tris)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto ei = pair[0];
                auto ti = pair[1];

                auto edge = edges.pack(dim_c<2>,"inds",ei,int_c);
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                for(int i = 0;i != 3;++i)
                    if(tri[i] == edge[0] || tri[i] == edge[1])
                        return;

                vec3 edge_vertices[2] = {};
                vec3 tri_vertices[3] = {};
                for(int i = 0;i != 2;++i)
                    edge_vertices[i] = verts.pack(dim_c<3>,xtag,edge[i]);
                for(int i = 0;i != 3;++i)
                    tri_vertices[i] = verts.pack(dim_c<3>,xtag,tri[i]);

                auto G = vec3::zeros();

                int his[2] = {};
                his[0] = zs::reinterpret_bits<int>(edges("he_inds",ei));
                his[1] = zs::reinterpret_bits<int>(halfedges("opposite_he",his[0]));

                auto nrm = tris.pack(dim_c<3>,"nrm",ti);


                T normal_coeff = 0;
                for(auto hi : his)
                {
                    if(hi < 0)
                        break;

                    auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                    auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);
                    auto halfedge_local_vertex_id = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
                    auto halfedge_opposite_vertex_id = htri[(halfedge_local_vertex_id + 2) % 3];
                    auto halfedge_opposite_vertex = verts.pack(dim_c<3>,xtag,halfedge_opposite_vertex_id);

                    auto hnrm = tris.pack(dim_c<3>,"nrm",hti);
                    T dnc{};
                    G += eval_HT_contour_minimization_gradient(edge_vertices[0],
                        edge_vertices[1],
                        halfedge_opposite_vertex,
                        hnrm,nrm,dnc);

                    normal_coeff += dnc;
                }

                // if(isnan(G.norm())) {
                //     printf("nan G detected: %f %f %f\n",(float)G[0],(float)G[1],(float)G[2]);
                // }
                G -= normal_coeff * nrm;

                // auto Gn = G.norm();
                // auto Gn2 = Gn * Gn;
                // G = h0 * G / zs::sqrt(Gn2 + g02);

                icm_grad.tuple(dim_c<3>,"grad",ci) = G;
                // icm_grad.tuple(dim_c<2>,"inds",ci) = pair.reinterpret_bits(float_c);
        }); 
    }


    template<typename Pol,
        typename PosTileVec,
        typename TriTileVec,
        typename EdgeTileVec,
        typename HalfEdgeTileVec,
        typename HTHashMap,
        typename ICMGradTileVec,
        auto space = Pol::exec_tag::value,
        typename T = typename PosTileVec::value_type>
    void eval_intersection_contour_minimization_gradient_of_edgeA_with_triB(Pol& pol,
            const PosTileVec& verts,const zs::SmallString& xtag,
            const EdgeTileVec& edges,
            const HalfEdgeTileVec& halfedges,
            const TriTileVec& tris,
            const PosTileVec& kverts,const zs::SmallString& kxtag,
            const TriTileVec& ktris,
            const T& maximum_correction,
            const T& progressive_slope,
            HTHashMap& csET,
            ICMGradTileVec& icm_grad,
            bool enforce_triangle_normal = false) {
        using namespace zs;
        auto exec_tag = wrapv<space>{};
        using vec2i = zs::vec<int,2>; 
        using vec3 = zs::vec<T,3>;
   
        // icm_grad.resize(csET.size());
        // local scheme
        pol(zip(zs::range(csET.size()),csET._activeKeys),[
            enforce_triangle_normal = enforce_triangle_normal,
            icm_grad = proxy<space>({},icm_grad),
            kxtag = zs::SmallString(kxtag),
            verts = proxy<space>({},verts),xtag = zs::SmallString(xtag),
            edges = proxy<space>({},edges),
            halfedges = proxy<space>({},halfedges),
            tris = proxy<space>({},tris),
            h0 = maximum_correction,
            g02 = progressive_slope * progressive_slope,
            kverts = proxy<space>({},kverts),
            ktris = proxy<space>({},ktris)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto ei = pair[0];
                auto kti = pair[1];

                auto edge = edges.pack(dim_c<2>,"inds",ei,int_c);
                auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);

                vec3 edge_vertices[2] = {};
                vec3 ktri_vertices[3] = {};
                for(int i = 0;i != 2;++i)
                    edge_vertices[i] = verts.pack(dim_c<3>,xtag,edge[i]);
                for(int i = 0;i != 3;++i)
                    ktri_vertices[i] = kverts.pack(dim_c<3>,kxtag,ktri[i]);

                auto G = vec3::zeros();

                int his[2] = {};
                his[0] = zs::reinterpret_bits<int>(edges("he_inds",ei));
                his[1] = zs::reinterpret_bits<int>(halfedges("opposite_he",his[0]));

                auto knrm = ktris.pack(dim_c<3>,"nrm",kti);

                T normal_coeff = 0;

                for(auto hi : his)
                {
                    if(hi < 0)
                        break;

                    auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                    auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);
                    auto halfedge_local_vertex_id = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
                    auto halfedge_opposite_vertex_id = htri[(halfedge_local_vertex_id + 2) % 3];
                    auto halfedge_opposite_vertex = verts.pack(dim_c<3>,xtag,halfedge_opposite_vertex_id);

                    auto hnrm = tris.pack(dim_c<3>,"nrm",hti);

                    T dnc{};

                    G += eval_HT_contour_minimization_gradient(edge_vertices[0],
                        edge_vertices[1],
                        halfedge_opposite_vertex,
                        hnrm,knrm,dnc);

                    if(enforce_triangle_normal && dnc > 0) {
                        dnc = -dnc;
                    }

                    normal_coeff += dnc;
                }


                
                // printf("normal_coeff : %f R : %f %f %f\n",(float)normal_coeff,
                //     (float)G[0],
                //     (float)G[1],
                //     (float)G[2]);

                G -= normal_coeff * knrm;



                auto Gn = G.norm();
                auto Gn2 = Gn * Gn;
                G = h0 * G / zs::sqrt(Gn2 + g02);

                icm_grad.tuple(dim_c<3>,"grad",ci) = G;
                // icm_grad.tuple(dim_c<2>,"inds",ci) = pair.reinterpret_bits(float_c);
        }); 
    }

};
};

