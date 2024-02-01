#pragma once


#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/Bvh.hpp"
#include "Utils.hpp"
#include "compute_characteristic_length.hpp"

#include <iostream>

#include "geo_math.hpp"
#include "tiled_vector_ops.hpp"
#include "topology.hpp"


namespace zeno {


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


template<typename Pol,typename PosTileVec,typename TriTileVec,typename HETileVec,typename InstTileVec>
size_t retrieve_self_intersection_tri_halfedge_list_info(Pol& pol,
    const PosTileVec& verts,
    const zs::SmallString& xtag,
    const TriTileVec& tris,
    const HETileVec& halfedges,
    InstTileVec& intersect_buffers) {
        using namespace zs;
        using vec2i = zs::vec<int,2>;
        using bv_t = typename ZenoParticles::lbvh_t::Box;
        using vec3 = zs::vec<T,3>;
        using table_vec2i_type = zs::bht<int,2,int>;

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};
        zs::Vector<int> nmIts{verts.get_allocator(),1};
        nmIts.setVal(0);

        auto bvs = retrieve_bounding_volumes(pol,verts,tris,wrapv<3>{},0,xtag);
        auto triBvh = LBvh<3,int,T>{};
        triBvh.build(pol,bvs);

        // auto cnorm = compute_average_edge_length(pol,verts,xtag,tris);
        // cnorm *= 3;

        auto max_intersections = intersect_buffers.size();

        pol(zs::range(halfedges.size()),[
            exec_tag,
            nmIts = proxy<space>(nmIts),
            max_intersections = max_intersections,
            halfedges = proxy<space>({},halfedges),/*'to_vertex' 'to_face' 'opposite_he' 'next_he'*/
            verts = proxy<space>({},verts),
            nm_verts = verts.size(),
            tris = proxy<space>({},tris),
            triBvh = proxy<space>(triBvh),
            // thickness = cnorm,
            intersect_buffers = proxy<space>({},intersect_buffers),
            xtag = zs::SmallString(xtag)] ZS_LAMBDA(int hei) mutable {
                vec2i edge{};
                auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hei));
                auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);

                auto local_vert_id = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hei));
                edge[0] = htri[local_vert_id];
                edge[1] = htri[(local_vert_id + 1) % 3];
                if(edge[0] > edge[1]) {
                    auto tmp = edge[0];
                    edge[0] = edge[1];
                    edge[1] = tmp;
                }


                vec3 eV[2] = {};
                // auto edgeCenter = vec3::zeros();
                for(int i = 0;i != 2;++i) {
                    eV[i] = verts.pack(dim_c<3>,xtag,edge[i]);
                    // edgeCenter += eV[i] / (T)2.0;
                }

                auto dir = eV[1] - eV[0];
                
                auto bv = bv_t{get_bounding_box(eV[0],eV[1])};
                auto process_potential_he_tri_intersection_pairs = [&, exec_tag](int ti) {
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                    int nm_combinatorial_coincidences = 0;
                    // int common_idx = 0;
                    for(int i = 0;i != 2;++i)
                        for(int j = 0;j != 3;++j)
                            if(edge[i] == tri[j]){
                                // common_idx = tri[j];
                                ++nm_combinatorial_coincidences;
                            }
                    
                    if(nm_combinatorial_coincidences > 0)
                        return;

                    // might need an accurate predicate here for floating-point intersection testing
                    {
                        vec3 tV[3] = {};
                        for(int i = 0;i != 3;++i)
                            tV[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                        // auto dir = eV[1] - tV[0];
                        double r{};
                        // LSL_GEO::tri_ray_intersect_d<double>(eV[0],eV[1],tV[0],tV[1],tV[2],r);
                        if(LSL_GEO::tri_ray_intersect_d<double>(eV[0],eV[1],tV[0],tV[1],tV[2],r)) {
                            auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                            if(offset >= max_intersections)
                                return;
                            auto intp = r * dir + eV[0];
                            intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{hei,ti}.reinterpret_bits(float_c);
                            intersect_buffers.tuple(dim_c<3>,"int_points",offset) = intp;
                            intersect_buffers("r",offset) = (T)r;
                        }
                    }
                };

                triBvh.iter_neighbors(bv,process_potential_he_tri_intersection_pairs);
        });

        // std::cout << "initialize corner_idx : " << nmIts.getVal(0) << std::endl;

        if(nmIts.getVal(0) >= max_intersections) {
            throw std::runtime_error("max_size_of_intersections buffer reach");
        }

        pol(zs::range(nmIts.getVal(0)),[
            intersect_buffers = proxy<space>({},intersect_buffers),
            tris = proxy<space>({},tris),
            nm_tris = tris.size(),
            halfedges = proxy<space>({},halfedges)] ZS_LAMBDA(int iti) mutable {
                auto pair = intersect_buffers.pack(dim_c<2>,"pair",iti,int_c);
                auto hi = pair[0];
                auto ti = pair[1];
                // if(ti >= nm_tris) {
                //     printf("invalid pair[%d] %d %d\n",iti,pair[0],pair[1]);
                //     return;
                // }
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                // if(hti >= nm_tris) {
                //     printf("invalid to_face : %d\n",hti);
                //     return;
                // }
                auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);
                int common_idx = -1;
                for(int i = 0;i != 3;++i)
                    for(int j = 0;j != 3;++j)   
                        if(tri[i] == htri[j]) {
                            common_idx = tri[i];
                        }      
                intersect_buffers("corner_idx",iti) = zs::reinterpret_bits<T>((int)common_idx);      
                
        });

        // std::cout << "finish initialize corner_idx" << std::endl;

        return nmIts.getVal(0);
}

template<typename Pol,typename PosTileVec,typename TriTileVec,typename HETileVec,typename InstTileVec>
int retrieve_intersection_tri_halfedge_info_of_two_meshes(Pol& pol,
    const PosTileVec& verts_A, const zs::SmallString& xtag_A,
    const TriTileVec& tris_A,
    const HETileVec& halfedges_A,
    const PosTileVec& verts_B, const zs::SmallString& xtag_B,
    const TriTileVec& tris_B,
    InstTileVec& he_A_and_tri_B_intersect_buffers) {
        using namespace zs;
        using vec2i = zs::vec<int,2>;
        using bv_t = typename ZenoParticles::lbvh_t::Box;
        using vec3 = zs::vec<T,3>;
        using table_vec2i_type = zs::bht<int,2,int>;

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};
        
        zs::Vector<int> nmIts{verts_A.get_allocator(),1};
        nmIts.setVal(0);

        auto cnorm_A = compute_average_edge_length(pol,verts_A,xtag_A,tris_A);
        auto cnorm_B = compute_average_edge_length(pol,verts_B,xtag_B,tris_B);
        auto cnorm = cnorm_A > cnorm_B ? cnorm_A : cnorm_B;
        cnorm *= 3;

        auto bvs = retrieve_bounding_volumes(pol,verts_B,tris_B,wrapv<3>{},cnorm,xtag_B);
        auto tri_B_bvh = LBvh<3,int,T>{};
        tri_B_bvh.build(pol,bvs);

        pol(zs::range(halfedges_A.size()),[
                halfedges_A = proxy<space>({},halfedges_A),
                tris_A = proxy<space>({},tris_A),
                verts_A = proxy<space>({},verts_A),
                verts_B = proxy<space>({},verts_B),
                tris_B = proxy<space>({},tris_B),
                xtag_A = xtag_A,
                xtag_B = xtag_B,
                exec_tag,
                intersect_buffers = proxy<space>({},he_A_and_tri_B_intersect_buffers),
                nmIts = proxy<space>(nmIts),
                thickness = cnorm,
                tri_B_bvh = proxy<space>(tri_B_bvh)] ZS_LAMBDA(int hei_A) mutable {
            vec2i edge_A{};
            auto ti_A = zs::reinterpret_bits<int>(halfedges_A("to_face",hei_A));
            auto tri_A = tris_A.pack(dim_c<3>,"inds",ti_A,int_c);

            auto local_vert_id_A = zs::reinterpret_bits<int>(halfedges_A("local_vertex_id",hei_A));
            edge_A[0] = tri_A[(local_vert_id_A + 0) % 3];
            edge_A[1] = tri_A[(local_vert_id_A + 1) % 3];

            auto ohei_A = zs::reinterpret_bits<int>(halfedges_A("opposite_he",hei_A));

            if(edge_A[0] > edge_A[1] && ohei_A >= 0)
                return;

            // if(edge_A[0] > edge_A[1]) {
            //     auto tmp = edge_A[0];
            //     edge_A[0] = edge_A[1];
            //     edge_A[1] = tmp;
            // }

            vec3 eV_A[2] = {}; 
            for(int i = 0;i != 2;++i)
                eV_A[i] = verts_A.pack(dim_c<3>,xtag_A,edge_A[i]);
            auto edgeCenter_A = (eV_A[0] + eV_A[1])/(T)2.0;

            auto dir_A = eV_A[1] - eV_A[0];
            auto bv_A = bv_t{get_bounding_box(edgeCenter_A - thickness,edgeCenter_A + thickness)};

            auto process_potential_he_tri_intersection_pairs = [&, exec_tag](int ti_B) {
                auto tri_B = tris_B.pack(dim_c<3>,"inds",ti_B,int_c);
                // might need an accurate predicate here for floating-point intersection testing
                {
                    vec3 tV_B[3] = {};
                    for(int i = 0;i != 3;++i)
                        tV_B[i] = verts_B.pack(dim_c<3>,xtag_B,tri_B[i]);
                    // auto dir = eV[1] - tV[0];
                    double r{};
                    if(LSL_GEO::tri_ray_intersect_d<double>(eV_A[0],eV_A[1],tV_B[0],tV_B[1],tV_B[2],r)) {
                        auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                        auto intp = r * dir_A + eV_A[0];
                        intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{hei_A,ti_B}.reinterpret_bits(float_c);
                        intersect_buffers.tuple(dim_c<3>,"int_points",offset) = intp;
                        intersect_buffers("r",offset) = (T)r;
                        // make sure the opposite he - tri pairs are also inserted
                        // auto opposite_hei_A = zs::reinterpret_bits<int>(halfedges_A("opposite_he",hei_A));
                        if(ohei_A >= 0) {
                            offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                            intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{ohei_A,ti_B}.reinterpret_bits(float_c);
                            intersect_buffers.tuple(dim_c<3>,"int_points",offset) = intp;
                            intersect_buffers("r",offset) = (T)(1 - r);
                        }
                    }
                }                    
            };
            tri_B_bvh.iter_neighbors(bv_A,process_potential_he_tri_intersection_pairs);
        });
        return nmIts.getVal(0);
        // return 0;
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
        const PosTileVec& verts, const zs::SmallString& xtag, const zs::SmallString& collision_group_name,
        const TriTileVec& tris,
        const EdgeTileVec& edges,
        const TriBVH& tri_bvh,
        zs::bht<int,2,int>& res,
        BaryTileVec& bary_buffer,
        const float& thickness,
        bool use_barycentric_interpolator = false,
        bool skip_too_close_pair_at_rest_configuration = false,
        bool use_collision_group = false) {
            using namespace zs;
            using vec2i = zs::vec<int,2>;
            using bv_t = typename ZenoParticles::lbvh_t::Box;
            using vec3 = zs::vec<T,3>;
            using table_vec2i_type = zs::bht<int,2,int>;

            constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
            constexpr auto exec_tag = wrapv<space>{};
            constexpr auto eps = 1e-6;


            // zs::CppTimer timer;

            res.reset(pol,true);
            // timer.tick();


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
                edges = edges.begin("inds", dim_c<2>, int_c),
                triNrmOffset = tris.getPropertyOffset("nrm"),
                triDOffset = tris.getPropertyOffset("d"),
                triIndsOffset = tris.getPropertyOffset("inds"),
                tris = view<space>(tris),
                // verts = proxy<space>({},verts),
                verts = view<space>(verts),
                xOffset = verts.getPropertyOffset(xtag),
                aniMaskOffset = verts.getPropertyOffset("ani_mask"),
                hasAniMask = verts.hasProperty("ani_mask"),
                colllisionCancelOffset = verts.getPropertyOffset("collision_cancel"),
                hasCollisionCancel = verts.hasProperty("collision_cancel"),
                hasMinv = verts.hasProperty("minv"),
                minvOffset = verts.getPropertyOffset("minv"),
                hasRestShape = verts.hasProperty("X"),
                restShapeOffset = verts.getPropertyOffset("X"),
                hasCollisionGroup = verts.hasProperty(collision_group_name),
                collisionGroupOffset = verts.getPropertyOffset(collision_group_name),
                thickness = thickness,
                use_collision_group = use_collision_group,
                skip_too_close_pair_at_rest_configuration = skip_too_close_pair_at_rest_configuration,
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
                    auto edge = edges[ei];

                    if(hasCollisionCancel){
                        for(int i = 0;i != 2;++i)
                            if(verts(colllisionCancelOffset,edge[i]) > 0.5)
                                return;
                    }

                    auto vs = verts.pack(dim_c<3>, xOffset, edge[0]);
                    auto ve = verts.pack(dim_c<3>, xOffset, edge[1]);

                    auto bv = bv_t{get_bounding_box(vs,ve)};

                    bool edge_has_dynamic_points = true;
                    if(hasMinv) {
                        for(int i = 0;i != 2;++i)
                            if(verts(minvOffset,edge[i]) < 0.0001)
                                edge_has_dynamic_points = false;
                    }

                    auto do_rest_shape_skip_in_the_same_collision_group = hasRestShape && skip_too_close_pair_at_rest_configuration;

                    auto Vs = vs;
                    auto Ve = ve;
                    if(do_rest_shape_skip_in_the_same_collision_group) {
                        Vs = verts.pack(dim_c<3>,restShapeOffset,edge[0]);
                        Ve = verts.pack(dim_c<3>,restShapeOffset,edge[1]);
                    }
 
                    int nm_intersect_tri = 0;
                    auto process_potential_ET_intersection_pairs = [&](int ti) mutable {
                        auto tri = tris.pack(dim_c<3>,triIndsOffset,ti,int_c);

                        if(hasCollisionCancel){
                            for(int i = 0;i != 3;++i)
                                if(verts(colllisionCancelOffset,tri[i]) > 0.5)
                                    return;
                        }

                        for(int i = 0;i != 3;++i)
                            if(tri[i] == edge[0] || tri[i] == edge[1])
                                return;

                        auto tri_has_dynamic_points = true;
                        if(hasMinv) {
                            for(int i = 0;i != 3;++i)
                                if(verts(minvOffset,tri[i]) < 0.0001)
                                    tri_has_dynamic_points = false;
                        }

                        if(!edge_has_dynamic_points && !tri_has_dynamic_points)
                            return;


                        if(do_rest_shape_skip_in_the_same_collision_group) {
                            vec3 tVs[3] = {};
                            for(int i = 0;i != 3;++i)
                                tVs[i] = verts.pack(dim_c<3>,restShapeOffset,tri[i]);
                            
                            auto is_same_collision_group = false;
                            if(use_collision_group) {
                                is_same_collision_group = zs::abs(verts(collisionGroupOffset,edge[0]) - verts(collisionGroupOffset,tri[0])) < 0.1;
                            }

                            if(is_same_collision_group) {
                                for(int i = 0;i != 3;++i) {
                                    if((tVs[i] - Vs).norm() < thickness)
                                        return;
                                    if((tVs[i] - Ve).norm() < thickness)
                                        return;    
                                }
                            }
                        }

                        vec3 tvs[3] = {};
                        for(int i = 0;i != 3;++i)
                            tvs[i] = verts.pack(dim_c<3>,xOffset,tri[i]);

                        auto tnrm = tris.pack(dim_c<3>,triNrmOffset,ti);
                        auto d = tris(triDOffset,ti);

                        auto en = (ve - vs).norm();
                        if(en < eps)
                            return;

                        auto is_parallel = zs::abs(tnrm.dot(ve - vs) / en) < 1e-3;
                        if(is_parallel)
                            return;

                        auto ts = vs.dot(tnrm) + d;
                        auto te = ve.dot(tnrm) + d;

                        if(ts * te > 0) {
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
                            return;
                        }

                        q = p - tvs[2];
                        n1 = e1.cross(q);
                        auto n2 = e2.cross(q);

                        if(n1.dot(n2) < 0) {
                            return;
                        }

                        q = p - tvs[1];
                        n0 = e0.cross(q);
                        n2 = e2.cross(q);

                        if(n0.dot(n2) > 0) {
                            return;
                        }

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
                            LSL_GEO::get_triangle_vertex_barycentric_coordinates(tvs[0],tvs[1],tvs[2],p,bary);
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

            // printf("nm_intersection_pair : %d\n",res.size());
            // timer.tock("detect intersection pair");
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

            // zs::CppTimer timer;
            cs_ET.reset(pol,true);

            // timer.tick();

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
                hasCollisionCancel = verts.hasProperty("collision_cancel"),
                collisionCancelOffset = verts.getPropertyOffset("collision_cancel"),
                hasKCollisionCancel = kverts.hasProperty("collision_cancel"),
                kCollisionCancelOffset = kverts.getPropertyOffset("collision_cancel"),
                eps = eps,
                res = proxy<space>(cs_ET),
                ktri_bvh = proxy<space>(ktri_bvh)] ZS_LAMBDA(int ei) mutable {
                    auto edge = edges[ei];

                    if(hasCollisionCancel) {
                        for(int i = 0;i != 2;++i)
                            if(verts(collisionCancelOffset,edge[i]) > 0.5)
                                return;
                    }

                    auto vs = verts.pack(dim_c<3>, xOffset, edge[0]);
                    auto ve = verts.pack(dim_c<3>, xOffset, edge[1]);

                    auto bv = bv_t{get_bounding_box(vs,ve)};

                    bool is_dynamic_edge = true;
                    if(hasMinv)
                        for(int i = 0;i != 2;++i)
                            if(verts(minvOffset,edge[i]) < 0.0001)
                                is_dynamic_edge = false;

                    auto process_potential_EKT_intersection_pairs = [&](int kti) mutable {
                        auto ktri = ktris.pack(dim_c<3>,ktriIndsOffset,kti,int_c);

                        if(hasKCollisionCancel) {
                            for(int i = 0;i != 3;++i)
                                if(kverts(kCollisionCancelOffset,ktri[i]) > 0.5)
                                    return;
                        }

                        bool is_dynamic_ktri = true;
                        if(hasKMinv)
                            for(int i = 0;i != 3;++i)
                                if(kverts(kminvOffset,ktri[i]) < 0.0001)
                                    is_dynamic_ktri = false;

                        if(!is_dynamic_edge || !is_dynamic_ktri)
                            return;      

                        vec3 ktvs[3] = {};
                        for(int i = 0;i != 3;++i)
                            ktvs[i] = kverts.pack(dim_c<3>,kxOffset,ktri[i]);

                        auto ktnrm = ktris.pack(dim_c<3>,ktriNrmOffset,kti);
                        if(ktnrm.norm() < 1e-6)
                            return;
                        auto d = ktris(ktriDOffset,kti);

                        auto en = (ve - vs).norm();
                        if(en < eps)
                            return;                        

                        auto is_parallel = zs::abs(ktnrm.dot(ve - vs) / en) < 1e-3;
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

                        auto ci = res.insert(vec2i{ei,kti});
                        if(use_barycentric_interpolator) {
                            zs::vec<T,3> bary{};
                            LSL_GEO::get_triangle_vertex_barycentric_coordinates(ktvs[0],ktvs[1],ktvs[2],p,bary);
                            bary_buffer.tuple(dim_c<4>,"bary",ci) = zs::vec<T,4>{1-t,bary[0],bary[1],bary[2]};
                        }
                        bary_buffer.tuple(dim_c<2>,"inds",ci) = zs::vec<int,2>{ei,kti}.reinterpret_bits(float_c);

                    }; 
                    ktri_bvh.iter_neighbors(bv,process_potential_EKT_intersection_pairs);              
            });  
            // timer.tock("detect KET or EKT intersection pair");
    }
};