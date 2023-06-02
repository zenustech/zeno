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

template<typename Pol,typename PosTileVec,typename TriTileVec,typename EdgeTileVec>
int mark_edge_tri_intersection(Pol& pol,
        const PosTileVec& verts,
        TriTileVec& tris,
        EdgeTileVec& edges,
        const zs::SmallString& xTag,
        const zs::SmallString& markTag,
        bool mark_edges,
        bool mark_tris) {
    using namespace zs;
    using vec2i = zs::vec<int,2>;
    using T = typename PosTileVec::value_type;
    using vec3 = zs::vec<T,3>;
    using bv_t = typename ZenoParticles::lbvh_t::Box;

    constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
    // check the channels
    if(!verts.hasProperty(xTag) || verts.getPropertySize(xTag) != 3) {
        fmt::print(fg(fmt::color::red),"the input verts has no specified xTag : {}\n",xTag);
        return -1;
    }

    if(!edges.hasProperty(markTag) && mark_edges) {
        edges.append_channels(pol,{{markTag,1}});
    }
    if(!tris.hasProperty(markTag) && mark_tris) {
        tris.append_channels(pol,{{markTag,1}});
    }

    if(mark_edges){
        TILEVEC_OPS::fill(pol,edges,markTag,reinterpret_bits<T>((int)0));
    }
    if(mark_tris){
        TILEVEC_OPS::fill(pol,tris,markTag,reinterpret_bits<T>((int)0));
    }

    auto bvs = retrieve_bounding_volumes(pol,verts,tris,wrapv<3>{},(T)0.0,xTag);
    auto triBvh = LBvh<3,int,T>{};
    triBvh.build(pol,bvs);

    auto cnorm = compute_average_edge_length(pol,verts,xTag,tris);
    cnorm *= 2;

    pol(zs::range(edges.size()),[
            verts = proxy<space>({},verts),
            edges = proxy<space>({},edges),
            tris = proxy<space>({},tris),
            triBvh = proxy<space>(triBvh),
            thickness = cnorm,
            xTag = xTag,
            markTag = markTag,
            mark_edges = mark_edges,
            mark_tris = mark_tris] ZS_LAMBDA(int ei) mutable {
        auto edgeCenter = vec3::zeros();
        auto edge = edges.pack(dim_c<2>,"inds",ei).reinterpret_bits(int_c);
        for(int i = 0;i != 2;++i)
            edgeCenter += verts.pack(dim_c<3>,xTag,edge[i]) / (T)2.0;
        
        auto bv = bv_t(get_bounding_box(edgeCenter - thickness,edgeCenter + thickness));
        auto process_potential_intersection_pairs = [&](int ti) mutable {
            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            int nm_match_indices = 0;
            for(int i = 0;i != 3;++i)
                for(int j = 0;j != 2;++j)
                    if(tri[i] == edge[j])
                        nm_match_indices++;
            // we need to deal with the coplanar case here
            if(nm_match_indices >= 1)
                return;
            
            auto ro = verts.pack(dim_c<3>,xTag,edge[0]);
            auto re = verts.pack(dim_c<3>,xTag,edge[1]);
            auto rd = re - ro;
            auto dist = rd.norm();
            rd = rd/((T)1e-7 + dist);
            
            vec3 vA[3] = {};
            for(int i = 0;i != 3;++i)
                vA[i] = verts.pack(dim_c<3>,xTag,tri[i]);

            auto r = LSL_GEO::tri_ray_intersect(ro,rd,vA[0],vA[1],vA[2]);
            if(r > dist)
                return;

            if(mark_edges)
                edges(markTag,ei) = (T)1;
            if(mark_tris)
                tris(markTag,ti) = (T)1;
        };

        triBvh.iter_neighbors(bv,process_potential_intersection_pairs);
    });

    return 0;
}

// constexpr bool 

// template<typename Pol>
// int retrieve_isolated_islands(Pol& pol,
//     int nm_nodes,
//     const zs::Vector<zs::vec<int,2>>& topos,
//     zs::Vector<int>& island_ids) {
//         zs::SparseMatrix<int,true> spmat{};
//         topological_incidence_matrix(pol,nm_nodes,topos,spmat);
//         island_ids
// }

// template<typename Pol,typename PosTileVec,typename TriTileVec,typename InstTable>
// int retrieve_triangulate_mesh_intersections(Pol& pol,
//     const PosTileVec& verts_A,
//     const zs::SmallString& xtag_A,
//     TriTileVec& tris_A,
//     const PosTileVec& verts_B,
//     const zs::SmallString& xtag_B,
//     TriTileVec& tris_B,
// )



// template<typename Pol,typename PosTileVec,typename TriTileVec>
// int retrieve_triangulate_mesh_intersection_list(Pol& pol,
//     const PosTileVec& verts_A,
//     const zs::SmallString& xtag_A,
//     TriTileVec& tris_A,
//     const PosTileVec& verts_B,
//     const zs::SmallString& xtag_B,
//     TriTileVec& tris_B,
//     zs::Vector<zs::vec<int,2>>& intersect_buffers,
//     zs::Vector<int>& intersect_types,
//     bool use_self_collision = false) {
//         using namespace zs;
//         using vec2i = zs::vec<int,2>;
//         using bv_t = typename ZenoParticles::lbvh_t::Box;
//         using vec3 = zs::vec<T,3>;

//         constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
//         constexpr auto exec_tag = wrapv<space>{};
//         zs::Vector<int> nmIts{verts_A.get_allocator(),1};
//         nmIts.setVal(0);

//         auto bvsA = retrieve_bounding_volumes(pol,verts_A,tris_A,wrapv<3>{},0,xtag_A);
//         auto triABvh = LBvh<3,int,T>{};
//         triABvh.build(pol,bvsA);

//         auto cnormA = compute_average_edge_length(pol,verts_A,xtag_A,tris_A);
//         auto cnormB = compute_average_edge_length(pol,verts_B,xtag_B,tris_B);        
//         auto cnorm = cnormA > cnormB ? cnormA : cnormB;
//         cnorm *= 3;

//         // if(!tris_B.hasProperty("nrm"))
//         //     throw std::runtime_error("the tris_B has no \'nrm\' channel");

//         pol(zs::range(tris_B.size()),[
//             exec_tag,
//             nmIts = proxy<space>(nmIts),
//             verts_A = proxy<space>({},verts_A),
//             tris_A = proxy<space>({},tris_A),
//             verts_B = proxy<space>({},verts_B),
//             tris_B = proxy<space>({},tris_B),
//             triABvh = proxy<space>(triABvh),
//             intersect_buffers = proxy<space>(intersect_buffers),
//             intersect_types = proxy<space>(intersect_types),
//             thickness = cnorm,
//             xtag_A = xtag_A,
//             xtag_B = xtag_B,
//             use_self_collision = use_self_collision] ZS_LAMBDA(int tb_i) mutable {
//                 auto triBCenter = vec3::zeros();
//                 auto triB = tris_B.pack(dim_c<3>,"inds",tb_i,int_c);
//                 vec3 vA[3] = {};
//                 vec3 vB[3] = {};
//                 for(int i = 0;i != 3;++i)
//                     vB[i] = verts_B.pack(dim_c<3>,xtag_B,triB[i]);

//                 auto vb01 = vB[1] - vB[0];
//                 auto vb02 = vB[2] - vB[0];
//                 auto nrmB = vb01.cross(vb02).normalized();
//                 // nrmB = nrmB / (nrmB.norm() + (T)1e-6);

//                 for(int i = 0;i != 3;++i)
//                     triBCenter += vB[i] / (T)3.0;
                
//                 auto bv = bv_t{get_bounding_box(triBCenter - thickness,triBCenter + thickness)};
//                 // auto nrmB = tris_B.pack(dim_c<3>,"nrm",tb_i);

//                 auto process_potential_intersection_pairs = [&](int ta_i) {
//                     auto triA = tris_A.pack(dim_c<3>,"inds",ta_i,int_c);
//                     for(int i = 0;i != 3;++i)
//                         vA[i] = verts_A.pack(dim_c<3>,xtag_A,triA[i]);

//                     if(use_self_collision){
//                         if(ta_i >= tb_i)
//                             return;
//                         int nm_topological_coincidences = 0;
//                         zs::vec<bool,3> triA_coincidences_flag = zs::vec<bool,3>::uniform(false_c);
//                         zs::vec<bool,3> triB_coincidences_flag = zs::vec<bool,3>::uniform(false_c);
//                         for(int i = 0;i != 3;++i)
//                             for(int j = 0;j != 3;++j)
//                                 if(triA[i] == triB[j]){
//                                     triA_coincidences_flag[i] = true_c;
//                                     triB_coincidences_flag[j] = true_c;
//                                     ++nm_topological_coincidences;
//                                 }
//                         if(nm_topological_coincidences >= 3)
//                             printf("invalid nm_topological_coincidences detected %d\n",nm_topological_coincidences);
//                         if(nm_topological_coincidences == 2){
//                             // should we neglect this sort of intersection?
//                             // return;
//                             int triA_different_idx = -1;
//                             int triB_different_idx = -1;
//                             for(int i = 0;i != 3;++i){
//                                 if(!triA_coincidences_flag[i])
//                                     triA_different_idx = i;
//                                 if(!triB_coincidences_flag[i])
//                                     triB_different_idx = i;
//                             }
//                             if(triA_different_idx < 0 || triB_different_idx < 0)
//                                 printf("invalid tri_different_idx detected : %d %d\n",triA_different_idx,triB_different_idx);
//                             auto e01 = vA[(triA_different_idx + 1) % 3] - vA[triA_different_idx];
//                             auto e02 = vA[(triA_different_idx + 2) % 3] - vA[triA_different_idx];
//                             auto e03 = vB[triB_different_idx] - vA[triA_different_idx];

//                             auto vol = zs::abs(e01.cross(e02).dot(e03));
//                             if(vol < 0)
//                                 printf("invalid vol evaluation detected %f\n",(float)vol);
//                             if(vol > 1e-4)
//                                 return;


//                             auto va01 = vA[1] - vA[0];
//                             auto va02 = vA[2] - vA[0];
//                             auto nrmA = va01.cross(va02).normalized();

//                             if(nrmA.dot(nrmB) > (T)0.0)
//                                 return;
//                             // now the two triangles are coplanar
//                             // check intersection

//                             printf("detected target type[0] of intersection : %d %d\n",ta_i,tb_i);

//                             auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
//                             intersect_buffers[offset][0] = ta_i;
//                             intersect_buffers[offset][1] = tb_i;
//                             intersect_types[offset] = 2;
//                             return;
//                         }
//                         if(nm_topological_coincidences == 1){
//                             // return;
//                             int triA_coincide_idx = -1;
//                             int triB_coincide_idx = -1;
//                             for(int i = 0;i != 3;++i){
//                                 if(triA_coincidences_flag[i])
//                                     triA_coincide_idx = i;
//                                 if(triB_coincidences_flag[i])
//                                     triB_coincide_idx = i;
//                             }
//                             if(triA_coincide_idx == -1 || triB_coincide_idx == -1)
//                                 printf("invalid triA_coincide_idx and triB_coincide_idx detected\n");

//                             auto ea = vA[(triA_coincide_idx + 2) % 3] - vA[(triA_coincide_idx + 1) % 3];
//                             auto eb = vB[(triB_coincide_idx + 2) % 3] - vB[(triB_coincide_idx + 1) % 3];

//                             auto ro = vA[(triA_coincide_idx + 1) % 3];
//                             auto r = LSL_GEO::tri_ray_intersect(ro,ea,vB[0],vB[1],vB[2]);
//                             if(r < (T)(1.0 + 1e-6)) {
//                                 // printf("detected target type[1] of intersection : %d %d\n",ta_i,tb_i);
//                                 auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
//                                 intersect_buffers[offset][0] = ta_i;
//                                 intersect_buffers[offset][1] = tb_i;
//                                 intersect_types[offset] = 1;
//                                 return;
//                             }

//                             ro = vB[(triB_coincide_idx + 1) % 3];
//                             r = LSL_GEO::tri_ray_intersect(ro,eb,vA[0],vA[1],vA[2]);
//                             if(r < (T)(1.0 + 1e-6)) {
//                                 // printf("detected target type[1] of intersection : %d %d\n",ta_i,tb_i);
//                                 auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
//                                 intersect_buffers[offset][0] = ta_i;
//                                 intersect_buffers[offset][1] = tb_i;
//                                 intersect_types[offset] = 1;
//                                 return;
//                             }
//                         } else {
//  // return;
//                             vec3 eas[3] = {};
//                             vec3 ebs[3] = {};

//                             for(int i = 0;i != 3;++i) {
//                                 eas[i] = vA[(i + 1) % 3] - vA[i];
//                                 ebs[i] = vB[(i + 1) % 3] - vB[i];
//                             }

//                             for(int i = 0;i != 3;++i){
//                                 auto r = LSL_GEO::tri_ray_intersect(vA[i],eas[i],vB[0],vB[1],vB[2]);
//                                 if(r < (T)(1.0 + 1e-6)) {
//                                     auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);

//                                     // if(offset == 0) {
//                                     //     printf("detected first intersection :\nvA : {%f %f %f}\neas : {%f %f %f}\nvB[0] : {%f %f %f}\nvB[1] : {%f %f %f}\nvB[2] : {%f %f %f}\n",
//                                     //         (float)vA[i][0],(float)vA[i][1],(float)vA[i][2],
//                                     //         (float)eas[i][0],(float)eas[i][1],(float)eas[i][2],
//                                     //         (float)vB[0][0],(float)vB[0][1],(float)vB[0][2],
//                                     //         (float)vB[1][0],(float)vB[1][1],(float)vB[1][2],
//                                     //         (float)vB[2][0],(float)vB[2][1],(float)vB[2][2]);
//                                     //     printf("the two : triA{%d %d %d},triB{%d %d %d}\n",
//                                     //         triA[0],triA[1],triA[2],triB[0],triB[1],triB[2]);
//                                     // }

//                                     intersect_buffers[offset][0] = ta_i;
//                                     intersect_buffers[offset][1] = tb_i;
//                                     intersect_types[offset] = 0;
//                                     return;
//                                 }

//                                 r = LSL_GEO::tri_ray_intersect(vB[i],ebs[i],vA[0],vA[1],vA[2]);
//                                 if(r < (T)(1.0 + 1e-6)) {
//                                     auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                    
//                                     // if(offset == 0) {
//                                     //     auto vBend = vB[i] + ebs[i];
//                                     //     printf("detected first intersection[%f] :\nvB : {%f %f %f}\nebs : {%f %f %f}\nvBend : {%f %f %f}\nvA[0] : {%f %f %f}\nvA[1] : {%f %f %f}\nvA[2] : {%f %f %f}\n",
//                                     //         (float)r,
//                                     //         (float)vB[i][0],(float)vB[i][1],(float)vB[i][2],
//                                     //         (float)ebs[i][0],(float)ebs[i][1],(float)ebs[i][2],
//                                     //         (float)vBend[0],(float)vBend[1],(float)vBend[2],
//                                     //         (float)vA[0][0],(float)vA[0][1],(float)vA[0][2],
//                                     //         (float)vA[1][0],(float)vA[1][1],(float)vA[1][2],
//                                     //         (float)vA[2][0],(float)vA[2][1],(float)vA[2][2]);
//                                     // }

//                                     intersect_buffers[offset][0] = ta_i;
//                                     intersect_buffers[offset][1] = tb_i;
//                                     intersect_types[offset] = 0;
//                                     return;
//                                 }
//                             }
//                         }
//                     }
//                     else{
//                         printf("the non-self-intersection algorithm not implemented yet");
//                         return;
//                     }
//                 };

//                 triABvh.iter_neighbors(bv,process_potential_intersection_pairs);
//         });

//         return nmIts.getVal(0);
// }

template<typename VecT, zs::enable_if_all<VecT::dim == 1, (VecT::extent <= 3), (VecT::extent > 1)> = 0>
constexpr auto project_onto_plane(const VecT& plane_root,const VecT& plane_nrm,const VecT& projected_point) {
    auto e = projected_point - plane_root;
    return e - e.dot(plane_nrm) * plane_nrm + projected_point;
}

// the two tris are <v0,v1,va> and <v0,v1,vb>
template<typename VecT, zs::enable_if_all<VecT::dim == 1, (VecT::extent <= 3), (VecT::extent > 1)> = 0>
constexpr bool is_intersecting_with_2_combinatorial_coincidences(const VecT& v0,const VecT& v1,const VecT& va,const VecT& vb) {
    using T = typename VecT::value_type;
    auto na = LSL_GEO::facet_normal(v0,v1,va);
    auto nb = LSL_GEO::facet_normal(v0,v1,vb);
    auto align = na.dot(nb);
    return align > (T)(1e-6 - 1) ? false : true;
}

// the two tris are <v0,va1,va2> and <v0,vb1,vb2>
template<typename VecT, zs::enable_if_all<VecT::dim == 1, (VecT::extent <= 3), (VecT::extent > 1)> = 0>
constexpr bool is_intersecting_with_1_combinatorial_coincidence(const VecT& v0,
        const VecT& va1,
        const VecT& va2,
        const VecT& vb1,
        const VecT& vb2) {
    using T = typename VecT::value_type;
    auto ea = va2 - va1;
    auto eb = vb2 - vb1;
    auto ra = LSL_GEO::tri_ray_intersect(va1,ea,v0,vb1,vb2);
    if(ra < (T)1.0) {
        return true;
    }
    auto rb = LSL_GEO::tri_ray_intersect(vb1,eb,v0,va1,va2);
    if(rb < (T)1.0){
        return true;
    }
    return false;
}

template<typename Vec3T, zs::enable_if_all<Vec3T::dim == 1, (Vec3T::extent <= 3), (Vec3T::extent > 1)> = 0>
constexpr bool is_intersecting(const Vec3T vA[3],const Vec3T vB[3]) {
    using T = typename Vec3T::value_type;
    Vec3T eas[3] = {};
    Vec3T ebs[3] = {};

    for(int i = 0;i != 3;++i) {
        eas[i] = vA[(i + 1) % 3] - vA[i];
        ebs[i] = vB[(i + 1) % 3] - vB[i];
    }

    auto ea_indices = zs::vec<int,3>::uniform(-1);
    auto eb_indices = zs::vec<int,3>::uniform(-1);
    int nm_ea_its = 0;
    int nm_eb_its = 0;
    Vec3T ea_its[3] = {};
    Vec3T eb_its[3] = {};

    auto nrmA = LSL_GEO::facet_normal(vA[0],vA[1],vA[2]);
    Vec3T ra{};
    Vec3T rb{};

    for(int i = 0;i != 3;++i){
        auto r = LSL_GEO::tri_ray_intersect(vA[i],eas[i],vB[0],vB[1],vB[2]);
        ra[i] = r;
        if(r < (T)(1.0)) {
            ea_indices[nm_ea_its] = i;
            ea_its[nm_ea_its] = vA[i] + eas[i] * r;
            ++nm_ea_its;
        }

        r = LSL_GEO::tri_ray_intersect(vB[i],ebs[i],vA[0],vA[1],vA[2]);
        rb[i] = r;
        if(r < (T)(1.0)) {
            eb_indices[nm_eb_its] = i;
            eb_its[nm_eb_its] = vB[i] + ebs[i] * r;
            ++nm_eb_its;
        } 
    }

    // return nm_eb_its + nm_ea_its == 2 ? true : false;
    if(nm_eb_its + nm_ea_its == 2)
        return true;

    // auto ori_nm_ea_its = nm_ea_its;
    // auto ori_nm_eb_its = nm_eb_its;

    // if(nm_ea_its + nm_eb_its == 1) {
    //     // return;
    //     if(nm_ea_its == 1) {
    //         auto ea_idx = ea_indices[0];
    //         auto v0 = vA[ea_idx];
    //         auto v1 = vA[(ea_idx + 1)  % 3];
    //         // auto avg_tb_edge_length = (ebs[0].norm() + ebs[1].norm() + ebs[2].norm()) / (T)3.0;
    //         // check if the end point of edge lies inside the the counter facet A, if so, push all the neighbored tris of  the end point 
    //         T d0 = (T)0;
    //         T d1 = (T)0;
    //         T b0 = (T)0;
    //         T b1 = (T)0;
    //         d0 = LSL_GEO::pointTriangleDistance(vB[0],vB[1],vB[2],v0,b0);
    //         d1 = LSL_GEO::pointTriangleDistance(vB[0],vB[1],vB[2],v1,b1);
    //         if(b0 < (T)(1 + 1e-6) && b1 < (T)(1 + 1e-6))
    //             return;
    //         d0 = b0 < (T)(1 + 1e-6) ? d0 : std::numeric_limits<T>::infinity();
    //         d1 = b1 < (T)(1 + 1e-6) ? d1 : std::numeric_limits<T>::infinity();
    //         // if(d0 < 0 || d1 < 0)
    //         //     printf("wrong zs::abs impl\n");
    //         if(d0 < d1) {
    //             nm_ea_its++;
    //             ea_indices[1] = (ea_idx - 1 + 3) % 3;
    //             ea_its[1] = ea_its[0];
    //         }else {
    //             nm_ea_its++;
    //             ea_indices[1] = (ea_idx + 1) % 3;
    //             ea_its[1] = ea_its[0];
    //         }
    //     }
    //     if(nm_eb_its == 1) {
    //         auto eb_idx = eb_indices[0];
    //         auto v0 = vB[eb_idx];
    //         auto v1 = vB[(eb_idx + 1)  % 3];

    //         T d0 = (T)0;
    //         T d1 = (T)0;
    //         T b0 = (T)0;
    //         T b1 = (T)0;
    //         d0 = LSL_GEO::pointTriangleDistance(vA[0],vA[1],vA[2],v0,b0);
    //         d1 = LSL_GEO::pointTriangleDistance(vA[0],vA[1],vA[2],v1,b1);
    //         if(b0 < (T)(1 + 1e-6) && b1 < (T)(1 + 1e-6))
    //             return;
    //         d0 = b0 < (T)(1 + 1e-6) ? d0 : std::numeric_limits<T>::infinity();
    //         d1 = b1 < (T)(1 + 1e-6) ? d1 : std::numeric_limits<T>::infinity();

    //         // if(d0 < 0 || d1 < 0)
    //         //     printf("wrong zs::abs impl\n");
    //         if(d0 < d1) {
    //             nm_eb_its++;
    //             eb_indices[1] = (eb_idx - 1 + 3) % 3;
    //             eb_its[1] = eb_its[0];
    //         }else {
    //             nm_eb_its++;
    //             eb_indices[1] = (eb_idx + 1) % 3;
    //             eb_its[1] = eb_its[0];
    //         }
    //     }
    //     if(nm_eb_its + nm_ea_its == 1)
    //         return false;
    //     else 
    //         return true;
    // }

    // if(nm_ea_its + nm_eb_its == 0) {
    //     // return;
    //     // check wheter vertex of A intersect with B 
    //     auto avg_ta_edge_length = (eas[0].norm() + eas[1].norm() + eas[2].norm()) / (T)3.0;
    //     auto avg_tb_edge_length = (ebs[0].norm() + ebs[1].norm() + ebs[2].norm()) / (T)3.0;
    //     for(int i = 0;i != 3;++i) {
    //         T barySum = (T)0;
    //         auto distance = LSL_GEO::pointTriangleDistance(vB[0],vB[1],vB[2],vA[i],barySum);
    //         if(barySum < (T)(1 + 1e-6) && distance < avg_ta_edge_length * 1e-5) {
    //             nm_ea_its = 2;
    //             ea_indices[0] = i;
    //             ea_indices[1] = (i + 2) % 3;
    //             ea_its[0] = ea_its[1] = vA[i];
    //             break;
    //         }

    //         distance = LSL_GEO::pointTriangleDistance(vA[0],vA[1],vA[2],vB[i],barySum);
    //         if(barySum < (T)(1 + 1e-6) && distance < avg_tb_edge_length * 1e-5) {
    //             nm_eb_its = 2;
    //             eb_indices[0] = i;
    //             eb_indices[1] = (i + 2) % 3;
    //             eb_its[0] = eb_its[1] = vB[i];
    //             break;
    //         }
    //     }

    //     return nm_ea_its + nm_eb_its == 0 ? false : true;
    // }

    // return false;
}

template<typename Pol,typename PosTileVec,typename TriTileVec,typename HETileVec,typename InstTileVec>
int retrieve_self_intersection_tri_halfedge_list_info(Pol& pol,
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

        auto cnorm = compute_average_edge_length(pol,verts,xtag,tris);
        cnorm *= 3;

        // std::cout << "retrieve halfedge and tri pairs" << std::endl;
        // std::cout << "halfedges : " << halfedges.size() << std::endl;
        // std::cout << "verts : " << verts.size() << std::endl;
        // std::cout << "tris : " << tris.size() << std::endl;


        pol(zs::range(halfedges.size()),[
            exec_tag,
            nmIts = proxy<space>(nmIts),
            halfedges = proxy<space>({},halfedges),/*'to_vertex' 'to_face' 'opposite_he' 'next_he'*/
            verts = proxy<space>({},verts),
            nm_verts = verts.size(),
            tris = proxy<space>({},tris),
            triBvh = proxy<space>(triBvh),
            thickness = cnorm,
            intersect_buffers = proxy<space>({},intersect_buffers),
            xtag = xtag] ZS_LAMBDA(int hei) mutable {
                vec2i edge{};
                auto ti = zs::reinterpret_bits<int>(halfedges("to_face",hei));
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                auto local_vert_id = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hei));
                edge[0] = tri[local_vert_id];
                edge[1] = tri[(local_vert_id + 1) % 3];

                auto ohei = zs::reinterpret_bits<int>(halfedges("opposite_he",hei));

                if(edge[0] > edge[1] && ohei >= 0)
                    return;

                if(edge[0] >= nm_verts || edge[1] >= nm_verts)
                    printf("invalid edge detected : %d %d\n",edge[0],edge[1]);

                vec3 eV[2] = {};
                auto edgeCenter = vec3::zeros();
                for(int i = 0;i != 2;++i) {
                    eV[i] = verts.pack(dim_c<3>,xtag,edge[i]);
                    edgeCenter += eV[i] / (T)2.0;
                }

                auto dir = eV[1] - eV[0];
                
                auto bv = bv_t{get_bounding_box(edgeCenter - thickness,edgeCenter + thickness)};
                auto process_potential_he_tri_intersection_pairs = [&](int ti) {
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
                        auto r = LSL_GEO::tri_ray_intersect(eV[0],dir,tV[0],tV[1],tV[2]);
                        if(r < (T)1.0) {
                            auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                            auto intp = r * dir + eV[0];
                            intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{hei,ti}.reinterpret_bits(float_c);
                            intersect_buffers.tuple(dim_c<3>,"int_points",offset) = intp;

                            // make sure the opposite he - tri pairs are also inserted
                            auto opposite_hei = zs::reinterpret_bits<int>(halfedges("opposite_he",hei));
                            if(opposite_hei >= 0) {
                                offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                                intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{opposite_hei,ti}.reinterpret_bits(float_c);
                                intersect_buffers.tuple(dim_c<3>,"int_points",offset) = intp;
                            }
                        }
                    }
                };

                triBvh.iter_neighbors(bv,process_potential_he_tri_intersection_pairs);
        });

        // std::cout << "initialize corner_idx : " << nmIts.getVal(0) << std::endl;

        pol(zs::range(nmIts.getVal(0)),[
            intersect_buffers = proxy<space>({},intersect_buffers),
            tris = proxy<space>({},tris),
            nm_tris = tris.size(),
            halfedges = proxy<space>({},halfedges)] ZS_LAMBDA(int iti) mutable {
                auto pair = intersect_buffers.pack(dim_c<2>,"pair",iti,int_c);
                auto hi = pair[0];
                auto ti = pair[1];
                if(ti >= nm_tris) {
                    printf("invalid pair[%d] %d %d\n",iti,pair[0],pair[1]);
                    return;
                }
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                if(hti >= nm_tris) {
                    printf("invalid to_face : %d\n",hti);
                    return;
                }
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


template<typename Pol,typename PosTileVec,typename TriTileVec,typename HalfEdgeTileVec,typename GIA_TILEVEC>
int do_global_self_intersection_analysis(Pol& pol,
    PosTileVec& verts,
    const zs::SmallString& xtag,
    const TriTileVec& tris,
    HalfEdgeTileVec& halfedges,
    GIA_TILEVEC& gia_res,zs::bht<int,2,int>& conn_of_first_ring,int max_nm_intersections = 1000) {
        using namespace zs;
        using T = typename PosTileVec::value_type;
        using index_type = std::make_signed_t<int>;
        using size_type = std::make_unsigned_t<int>;
        using IV = zs::vec<int,2>;
        // using table_vec2i_type = zs::bcht<IV,int,true,zs::universal_hash<IV>,16>;
        using table_vec2i_type = zs::bht<int,2,int>;
        // using table_int_type = zs::bcht<int,int,true,zs::universal_hash<int>,16>;
        using table_int_type = zs::bht<int,1,int>;
        using edge_topo_type = zs::Vector<zs::vec<int,2>>;
        using inst_buffer_type = zs::Vector<zs::vec<int,2>>;
        using inst_class_type = zs::Vector<int>;
        using vec3 = zs::vec<T,3>;
        using vec2i = zs::vec<int,2>;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};

        using dtiles_t = zs::TileVector<T, 32>;        
        dtiles_t ints_buffer{verts.get_allocator(),
            {
                {"corner_idx",1},
                {"pair",2},
                {"int_points",3}
            },max_nm_intersections};

        if(!halfedges.hasProperty("broken"))
            halfedges.append_channels(pol,{{"broken",1}});
        TILEVEC_OPS::fill(pol,halfedges,"broken",(T)0.0);
        
        // if(!verts.hasProperty("is_corner"))
        //     verts.append_channels(pol,{{"is_corner",1}});
        // TILEVEC_OPS::fill(pol,verts,"is_corner",(T)0.0);

        // std::cout << "retrive_self_intersection_tri_half_list_info" << std::endl;

        auto nm_insts = retrieve_self_intersection_tri_halfedge_list_info(pol,verts,xtag,tris,halfedges,ints_buffer);
        table_vec2i_type cftab{ints_buffer.get_allocator(),(size_t)nm_insts};
        cftab.reset(pol,true);
        zs::Vector<int> cfbuffer{ints_buffer.get_allocator(),(size_t)nm_insts};  

        // std::cout << "initialize cftab" << std::endl;

        pol(zs::range(nm_insts),[
            cftab = proxy<space>(cftab),
            cfbuffer = proxy<space>(cfbuffer),
            ints_buffer = proxy<space>({},ints_buffer)] ZS_LAMBDA(int isi) mutable {
                auto pair = ints_buffer.pack(dim_c<2>,"pair",isi,int_c);
                // auto type = zs::reinterpret_bits<int>(ints_buffer("type",isi));
                auto hi = pair[0];
                auto ti = pair[1];
                // auto type = zs::reinterpret_bits<int>(ints_buffer("type",isi));
                // if(output_intermediate_information)
                //     printf("pair[%d] : [%d %d]\n",type,pair[0],pair[1]);
                if(auto setNo = cftab.insert(zs::vec<int,2>{hi,ti});setNo != table_vec2i_type::sentinel_v)
                    cfbuffer[setNo] = isi;

        });

        // std::cout << "build incidentItsTab" << std::endl;

        table_vec2i_type incidentItsTab{tris.get_allocator(),nm_insts * 2};
        incidentItsTab.reset(pol,true);    
        pol(zs::range(nm_insts),[
            ints_buffer = proxy<space>({},ints_buffer),
            cftab = proxy<space>(cftab),
            cfbuffer = proxy<space>(cfbuffer),
            tris = proxy<space>({},tris),
            halfedges = proxy<space>({},halfedges),
            incidentItsTab = proxy<space>(incidentItsTab)] ZS_LAMBDA(int isi) mutable {
                auto pair = ints_buffer.pack(dim_c<2>,"pair",isi,int_c);
                auto hi = pair[0];
                auto ti = pair[1];
                auto ohi = zs::reinterpret_bits<int>(halfedges("opposite_he",hi));
                if(ohi >= 0) {
                    if(auto no = cftab.query(vec2i{ohi,ti});no >= 0) {
                        auto nisi = cfbuffer[no];
                        if(isi > nisi)
                            incidentItsTab.insert(vec2i{isi,nisi});
                    }else {
                        printf("impossible reaching here, the hi and ohi should both have been inserted\n");
                    }
                }
                auto corner_idx = zs::reinterpret_bits<int>(ints_buffer("corner_idx",isi));
                if(corner_idx >= 0)
                    return;
                
                auto nhi = hi;
                for(int i = 0;i != 2;++i) {
                    nhi = zs::reinterpret_bits<int>(halfedges("next_he",nhi));
                    if(auto no = cftab.query(vec2i{nhi,ti});no >= 0) {
                        auto nisi = cfbuffer[no];
                        if(isi > nisi) {
                            incidentItsTab.insert(vec2i{isi,nisi});
                        }
                        return;
                    }
                }

                auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                auto thi = zs::reinterpret_bits<int>(tris("he_inds",ti));
                for(int i = 0;i != 3;++i) {
                    if(auto no = cftab.query(vec2i{thi,hti});no >= 0) {
                        auto nisi = cfbuffer[no];
                        if(isi > nisi) {
                            incidentItsTab.insert(vec2i{isi,nisi});
                        }
                        return;
                    }
                    thi = zs::reinterpret_bits<int>(halfedges("next_he",thi));
                }

                printf("impossible reaching here with broken insertion ring\n");
        });

        auto nmEntries = incidentItsTab.size();
        zs::Vector<zs::vec<int,2>> conn_topo{tris.get_allocator(),nmEntries};
        pol(zip(conn_topo,range(incidentItsTab._activeKeys)),[] ZS_LAMBDA(zs::vec<int,2> &ij,const auto& key) mutable {ij = key;});
        zs::Vector<int> ringTag{tris.get_allocator(),nm_insts};

        // std::cout << "Mark disconnected island" << std::endl;


        auto nm_rings = mark_disconnected_island(pol,conn_topo,ringTag);

        // std::cout << "finish Mark disconnected island with nm_rings : " << nm_rings << std::endl;

        // return nm_rings;

        zs::Vector<int> ringSize(tris.get_allocator(),nm_rings);
        pol(zs::range(ringSize.size()),[
            ringSize = proxy<space>(ringSize)] ZS_LAMBDA(int ri) mutable {
                ringSize[ri] = 0;
        });
        pol(zs::range(nm_insts),[
            ints_buffer = proxy<space>({},ints_buffer),
            ringTag = proxy<space>(ringTag),
            exec_tag,
            ringSize = proxy<space>(ringSize)] ZS_LAMBDA(int isi) mutable {
                atomic_add(exec_tag,&ringSize[ringTag[isi]],(int)1);
        });

        zs::Vector<int> island_buffer{verts.get_allocator(),verts.size()};
        
        gia_res.resize(verts.size());
        pol(zs::range(verts.size()),[gia_res = proxy<space>({},gia_res)] ZS_LAMBDA(int ni) mutable {
            // nodal_colors[ni] = 0;
            gia_res("ring_mask",ni) = zs::reinterpret_bits<T>((int)0);
            gia_res("color_mask",ni) = zs::reinterpret_bits<T>((int)0);
            gia_res("type_mask",ni) = zs::reinterpret_bits<T>((int)0);
            gia_res("is_corner",ni) = zs::reinterpret_bits<T>((int)0);
        });

        zs::Vector<zs::vec<int,2>> edge_topos{tris.get_allocator(),tris.size() * 3};
        pol(range(tris.size()),[
            tris = proxy<space>({},tris),
            // tab = proxy<space>(tab),
            // topo_tag = zs::SmallString(topo_tag),
            edge_topos = proxy<space>(edge_topos)] ZS_LAMBDA(int ti) mutable {
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                for(int i = 0;i != 3;++i){
                    if(tri[i] < tri[(i + 1) % 3])
                        edge_topos[ti * 3 + i] = zs::vec<int,2>{tri[i],tri[(i + 1) % 3]};
                    else
                        edge_topos[ti * 3 + i] = vec2i{-1,-1};
                }
        }); 

        conn_of_first_ring.reset(pol,true);

        for(int ri = 0;ri != nm_rings;++ri) {
            auto rsize = ringSize.getVal(ri);
            // if(output_intermediate_information)
            //     printf("ring[%d] Size : %d\n",ri,rsize);

            // edge_topo_type dc_edge_topos{tris.get_allocator(),rsize * 6};
            table_int_type disable_points{tris.get_allocator(),rsize * 8};
            table_vec2i_type disable_lines{tris.get_allocator(),rsize * 6};
            disable_points.reset(pol,true);
            disable_lines.reset(pol,true);

            pol(zs::range(nm_insts),[
                ints_buffer = proxy<space>({},ints_buffer),
                ringTag = proxy<space>(ringTag),
                // output_intermediate_information,
                ri,
                halfedges = proxy<space>({},halfedges),
                // topo_tag = zs::SmallString(topo_tag),
                // dc_edge_topos = proxy<space>(dc_edge_topos),
                gia_res = proxy<space>({},gia_res),
                verts = proxy<space>({},verts),
                disable_points = proxy<space>(disable_points),
                disable_lines = proxy<space>(disable_lines),
                nm_insts,
                tris = proxy<space>({},tris)] ZS_LAMBDA(int isi) mutable {
                    if(ringTag[isi] != ri)
                        return;
                    
                    auto pair = ints_buffer.pack(dim_c<2>,"pair",isi,int_c);
                    auto hi = pair[0];
                    halfedges("broken",hi) = (T)1.0;
                    // auto ti = pair[1];
                    // auto type = zs::reinterpret_bits<int>(ints_buffer("type",isi));

                    // auto ta = pair[0];
                    // auto tb = pair[1];
                    // zs::vec<int,3> tri_pairs[2] = {};
                    // tri_pairs[0] = tris.pack(dim_c<3>,"inds",ta,int_c);
                    // tri_pairs[1] = tris.pack(dim_c<3>,"inds",tb,int_c);
                    auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                    auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);
                    // FIND A BUG HERE
                    auto local_vidx = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
                    auto a = htri[local_vidx];
                    auto b = htri[(local_vidx + 1) % 3];
                    if(a > b) {
                        auto tmp = a;
                        a = b;
                        b = tmp;
                    }
                    disable_lines.insert(zs::vec<int,2>{a,b});
                    // for(int t = 0;t != 2;++t) {
                    //     zs::vec<int,2> out_edges[3] = {};
                    //     elm_to_edges(tri_pairs[t],out_edges);
                    //     for(int i = 0;i != 3;++i) {
                    //         auto a = out_edges[i][0];
                    //         auto b = out_edges[i][1];
                    //         if(a > b) {
                    //             auto tmp = a;
                    //             a = b;
                    //             b = tmp;
                    //         }
                    //         disable_lines.insert(zs::vec<int,2>{a,b});
                    //     }
                    // }
                    auto corner_idx = zs::reinterpret_bits<int>(ints_buffer("corner_idx",isi));
                    if(corner_idx >= 0){
                        gia_res("is_corner",corner_idx) = (T)1.0;
                        disable_points.insert(corner_idx);
                    }

                    // if(type == 1) {
                    //     int coincident_idx = -1;
                    //     for(int i = 0;i != 3;++i)
                    //         for(int j = 0;j != 3;++j)
                    //             if(tri_pairs[0][i] == tri_pairs[1][j])
                    //                 coincident_idx = tri_pairs[0][i];
                    //     if(coincident_idx < 0){
                    //         // if(output_intermediate_information)
                    //         //     printf("invalid coincident_idx detected : %d\n",coincident_idx);
                    //     }else
                    //         disable_points.insert(coincident_idx);
                    // }   
    
            });
            // table_vec2i_type connected_topo{edge_topos.get_allocator(),edge_topos.size() * 2};
            int nm_islands = 0;
            // if(ri > 0)
            //     nm_islands = mark_disconnected_island(pol,edge_topos,disable_points,disable_lines,island_buffer,conn_of_first_ring);
            // else {
                nm_islands = mark_disconnected_island(pol,edge_topos,disable_points,disable_lines,island_buffer,conn_of_first_ring);
                auto nn = conn_of_first_ring.size();
                pol(zip(range(nn),zs::range(conn_of_first_ring._activeKeys)),[
                    disable_lines = proxy<space>(disable_lines)] ZS_LAMBDA(auto i,const auto& ij) mutable {
                        auto ji = vec2i{ij[1],ij[0]};
                        auto no = disable_lines.query(ij);
                        if(no >= 0) {
                            printf("invalid topo[%d %d] detected in disable_lines %d\n",ij[0],ij[1],no);
                        }
                        no = disable_lines.query(ji);
                        if(no >= 0) {
                            printf("invalid topo[%d %d] reverse detected in disable_lines %d\n",ij[1],ij[0],no);
                        }
                });
                // std::cout << "size of conn_of_first_ring : " << conn_of_first_ring.size() << std::endl;
            // }
            // std::cout << "ring[" << ri << "] : " << nm_islands << "\tnm_broken_edges : " << disable_lines.size() << "\tnm_broken_corners : " << disable_points.size() << std::endl;


            zs::Vector<int> nm_cmps_every_island_count{verts.get_allocator(),(size_t)nm_islands};
            // nm_cmps_every_island_count.setVal(0);
            pol(zs::range(nm_cmps_every_island_count.size()),[
                nm_cmps_every_island_count = proxy<space>(nm_cmps_every_island_count)] ZS_LAMBDA(int i) mutable {
                    nm_cmps_every_island_count[i] = 0;
            });
            pol(zs::range(verts.size()),[
                verts = proxy<space>({},verts),
                island_buffer = proxy<space>(island_buffer),
                nm_cmps_every_island_count = proxy<space>(nm_cmps_every_island_count)] ZS_LAMBDA(int vi) mutable {
                    auto island_idx = island_buffer[vi];
                    atomic_add(exec_cuda,&nm_cmps_every_island_count[island_idx],(int)1);
            });

            for(int i = 0;i != nm_islands;++i)
                std::cout << nm_cmps_every_island_count.getVal(i) << "\t";
            std::cout << std::endl;

            int max_size = 0;
            int max_island_idx = 0;
            for(int i = 0;i != nm_islands;++i) {
                auto size_of_island = nm_cmps_every_island_count.getVal(i);
                if(size_of_island > max_size){
                    max_size = size_of_island;
                    max_island_idx = i;
                }
            }
            int black_island_idx = -1;
            if(nm_islands == 3) {
                for(int i = 0;i != nm_islands;++i){
                    if(i == max_island_idx)
                        continue;
                    black_island_idx = i;
                    break;
                }
            }
            auto cur_ri_mask = (int)1 << ri;

            pol(zs::range(verts.size()),[
                gia_res = proxy<space>({},gia_res),
                nm_islands,
                cur_ri_mask,
                black_island_idx,
                exec_tag,
                // ints_types = proxy<space>(ints_types),
                max_island_idx = max_island_idx,
                island_buffer = proxy<space>(island_buffer)] ZS_LAMBDA(int vi) mutable {
                    auto island_idx = island_buffer[vi];

                    // might exceed the integer range
                    auto ring_mask = zs::reinterpret_bits<int>(gia_res("ring_mask",vi));
                    auto color_mask = zs::reinterpret_bits<int>(gia_res("color_mask",vi));
                    auto type_mask = zs::reinterpret_bits<int>(gia_res("type_mask",vi));
                    // ring_mask += ((int) << ri)
                    if(island_idx != max_island_idx/* || ints_types[island_idx] == 1*/){
                        ring_mask |= cur_ri_mask;
                    }
                    if(nm_islands == 3)
                        type_mask |= cur_ri_mask;
                    if(nm_islands == 3 && island_idx == black_island_idx)
                        color_mask |= cur_ri_mask;
                    gia_res("ring_mask",vi) = zs::reinterpret_bits<T>(ring_mask);
                    gia_res("color_mask",vi) = zs::reinterpret_bits<T>(color_mask);
                    gia_res("type_mask",vi) = zs::reinterpret_bits<T>(type_mask);
            });
        }

        pol(zs::range(gia_res.size()),[
            gia_res = proxy<space>({},gia_res)] ZS_LAMBDA(int vi) mutable {
                auto is_corner = gia_res("is_corner",vi);
                if(is_corner > (T)0.5)
                    gia_res("ring_mask",vi) = zs::reinterpret_bits<T>((int)0);
        });

        return nm_rings;
}

template<typename Pol,typename PosTileVec,typename TriTileVec,typename InstTileVec>
int retrieve_triangulate_mesh_self_intersection_list_info(Pol& pol,
    const PosTileVec& verts_A,
    const zs::SmallString& xtag_A,
    TriTileVec& tris_A,
    const PosTileVec& verts_B,
    const zs::SmallString& xtag_B,
    TriTileVec& tris_B,
    InstTileVec& intersect_buffers,
    bool use_self_collision = true) {
        using namespace zs;
        using vec2i = zs::vec<int,2>;
        using bv_t = typename ZenoParticles::lbvh_t::Box;
        using vec3 = zs::vec<T,3>;

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};
        zs::Vector<int> nmIts{verts_A.get_allocator(),1};
        nmIts.setVal(0);

        auto bvsA = retrieve_bounding_volumes(pol,verts_A,tris_A,wrapv<3>{},0,xtag_A);
        auto triABvh = LBvh<3,int,T>{};
        triABvh.build(pol,bvsA);

        auto cnormA = compute_average_edge_length(pol,verts_A,xtag_A,tris_A);
        auto cnormB = compute_average_edge_length(pol,verts_B,xtag_B,tris_B);        
        auto cnorm = cnormA > cnormB ? cnormA : cnormB;
        cnorm *= 3;

        // if(!tris_B.hasProperty("nrm"))
        //     throw std::runtime_error("the tris_B has no \'nrm\' channel");
        if(!intersect_buffers.hasProperty("int_points"))
            throw std::runtime_error("the input intersect_buffers has no \'int_points\' attribute");

        pol(zs::range(tris_B.size()),[
            exec_tag,
            nmIts = proxy<space>(nmIts),
            verts_A = proxy<space>({},verts_A),
            tris_A = proxy<space>({},tris_A),
            verts_B = proxy<space>({},verts_B),
            tris_B = proxy<space>({},tris_B),
            triABvh = proxy<space>(triABvh),
            intersect_buffers = proxy<space>({},intersect_buffers),
            thickness = cnorm,
            xtag_A = xtag_A,
            xtag_B = xtag_B,
            use_self_collision = use_self_collision] ZS_LAMBDA(int tb_i) mutable {
                auto triBCenter = vec3::zeros();
                auto triB = tris_B.pack(dim_c<3>,"inds",tb_i,int_c);
                vec3 vA[3] = {};
                vec3 vB[3] = {};
                for(int i = 0;i != 3;++i)
                    vB[i] = verts_B.pack(dim_c<3>,xtag_B,triB[i]);

                auto vb01 = vB[1] - vB[0];
                auto vb02 = vB[2] - vB[0];
                auto nrmB = vb01.cross(vb02).normalized();
                // nrmB = nrmB / (nrmB.norm() + (T)1e-6);

                for(int i = 0;i != 3;++i)
                    triBCenter += vB[i] / (T)3.0;
                
                auto bv = bv_t{get_bounding_box(triBCenter - thickness,triBCenter + thickness)};
                // auto nrmB = tris_B.pack(dim_c<3>,"nrm",tb_i);

                auto process_potential_intersection_pairs = [&](int ta_i) {
                    auto triA = tris_A.pack(dim_c<3>,"inds",ta_i,int_c);
                    for(int i = 0;i != 3;++i)
                        vA[i] = verts_A.pack(dim_c<3>,xtag_A,triA[i]);

                    if(use_self_collision){
                        if(ta_i >= tb_i)
                            return;
                        int nm_topological_coincidences = 0;
                        zs::vec<bool,3> triA_coincidences_flag = zs::vec<bool,3>::uniform(false_c);
                        zs::vec<bool,3> triB_coincidences_flag = zs::vec<bool,3>::uniform(false_c);
                        for(int i = 0;i != 3;++i)
                            for(int j = 0;j != 3;++j)
                                if(triA[i] == triB[j]){
                                    triA_coincidences_flag[i] = true_c;
                                    triB_coincidences_flag[j] = true_c;
                                    ++nm_topological_coincidences;
                                }
                        if(nm_topological_coincidences >= 3)
                            printf("invalid nm_topological_coincidences detected %d\n",nm_topological_coincidences);
                        else if(nm_topological_coincidences == 2){
                            // should we neglect this sort of intersection?
                            return;
                            int triA_different_idx = -1;
                            int triB_different_idx = -1;
                            for(int i = 0;i != 3;++i){
                                if(!triA_coincidences_flag[i])
                                    triA_different_idx = i;
                                if(!triB_coincidences_flag[i])
                                    triB_different_idx = i;
                            }
                            if(triA_different_idx < 0 || triB_different_idx < 0)
                                printf("invalid tri_different_idx detected : %d %d\n",triA_different_idx,triB_different_idx);
                            auto e01 = vA[(triA_different_idx + 1) % 3] - vA[triA_different_idx];
                            auto e02 = vA[(triA_different_idx + 2) % 3] - vA[triA_different_idx];
                            auto e03 = vB[triB_different_idx] - vA[triA_different_idx];

                            auto vol = zs::abs(e01.cross(e02).dot(e03));
                            // if(vol < 0)
                            //     printf("invalid vol evaluation detected %f\n",(float)vol);
                            if(vol > 1e-4)
                                return;


                            auto va01 = vA[1] - vA[0];
                            auto va02 = vA[2] - vA[0];
                            auto nrmA = va01.cross(va02).normalized();

                            if(nrmA.dot(nrmB) > (T)0.0)
                                return;
                            // now the two triangles are coplanar
                            // check intersection

                            // printf("detected target type[0] of intersection : %d %d\n",ta_i,tb_i);

                            // auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                            // intersect_buffers[offset][0] = ta_i;
                            // intersect_buffers[offset][1] = tb_i;
                            // intersect_types[offset] = 2;
                            return;
                        }
                        else if(nm_topological_coincidences == 1){
                            int triA_coincide_idx = -1;
                            int triB_coincide_idx = -1;
                            for(int i = 0;i != 3;++i){
                                if(triA_coincidences_flag[i])
                                    triA_coincide_idx = i;
                                if(triB_coincidences_flag[i])
                                    triB_coincide_idx = i;
                            }
                            if(triA_coincide_idx == -1 || triB_coincide_idx == -1)
                                printf("invalid triA_coincide_idx and triB_coincide_idx detected\n");

                            auto ea = vA[(triA_coincide_idx + 2) % 3] - vA[(triA_coincide_idx + 1) % 3];
                            auto eb = vB[(triB_coincide_idx + 2) % 3] - vB[(triB_coincide_idx + 1) % 3];

                            auto ro = vA[(triA_coincide_idx + 1) % 3];
                            auto r = LSL_GEO::tri_ray_intersect(ro,ea,vB[0],vB[1],vB[2]);
                            if(r < (T)(1.0)) {
                                auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                                intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{ta_i,tb_i}.reinterpret_bits(float_c);
                                intersect_buffers("type",offset) = zs::reinterpret_bits<T>((int)1);

                                intersect_buffers.tuple(dim_c<6>,"its_edge_mark",offset) = zs::vec<int,6>::uniform(0).reinterpret_bits(float_c);
                                intersect_buffers("its_edge_mark",(triA_coincide_idx + 1) % 3,offset) = zs::reinterpret_bits<T>((int)1);
                                auto its_p0 = ro + ea * r;
                                auto its_p1 = vA[triA_coincide_idx];
                                intersect_buffers.tuple(dim_c<6>,"int_points",offset) = zs::vec<T,6>{
                                    its_p0[0],its_p0[1],its_p0[2],
                                    its_p1[0],its_p1[1],its_p1[2]};
                                return;
                            }

                            ro = vB[(triB_coincide_idx + 1) % 3];
                            r = LSL_GEO::tri_ray_intersect(ro,eb,vA[0],vA[1],vA[2]);
                            if(r < (T)(1.0)) {
                                // printf("detected target type[1] of intersection : %d %d\n",ta_i,tb_i);
                                auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                                intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{ta_i,tb_i}.reinterpret_bits(float_c);
                                intersect_buffers("type",offset) = zs::reinterpret_bits<T>((int)1);
                                // intersect_buffers.tuple(dim_c<4>,"int_edges",offset) = zs::vec<int,4>{
                                //     -1,(triB_coincide_idx + 1) % 3,
                                //     -1,-1}.reinterpret_bits(float_c);
                                // intersect_buffers.tuple(dim_c<2>,"its_edges",offset) = zs::vec<int,2>{(triB_coincide_idx + 1) % 3,-1}.reinterpret_bits(float_c);
                                // intersect_buffers.tuple(dim_c<2>,"its_edge_types",offset) = zs::vec<int,2>{1,-1}.reinterpret_bits(float_c);
                                intersect_buffers.tuple(dim_c<6>,"its_edge_mark",offset) = zs::vec<int,6>::uniform(0).reinterpret_bits(float_c);
                                intersect_buffers("its_edge_mark",((triB_coincide_idx + 1) % 3) + 3,offset) = zs::reinterpret_bits<T>((int)1);

                                auto its_p0 = ro + eb * r;
                                auto its_p1 = vB[triB_coincide_idx];
                                intersect_buffers.tuple(dim_c<6>,"int_points",offset) = zs::vec<T,6>{
                                    its_p0[0],its_p0[1],its_p0[2],
                                    its_p1[0],its_p1[1],its_p1[2]};
                                return;
                            }

                        } else if(nm_topological_coincidences == 0){ 
                            // return;
                            vec3 eas[3] = {};
                            vec3 ebs[3] = {};

                            for(int i = 0;i != 3;++i) {
                                eas[i] = vA[(i + 1) % 3] - vA[i];
                                ebs[i] = vB[(i + 1) % 3] - vB[i];
                            }


                            auto ea_indices = zs::vec<int,3>::uniform(-1);
                            auto eb_indices = zs::vec<int,3>::uniform(-1);
                            int nm_ea_its = 0;
                            int nm_eb_its = 0;
                            vec3 ea_its[3] = {};
                            vec3 eb_its[3] = {};


                            auto va01 = vA[1] - vA[0];
                            auto va02 = vA[2] - vA[0];
                            auto nrmA = va01.cross(va02).normalized();
                            
                            // if(zs::abs(nrmA.dot(nrmB)) > (T)(1 - 1e-5))
                            //     return;
                            // auto avg_ta_edge_length = (eas[0].norm() + eas[1].norm() + eas[2].norm()) / (T)3.0;
                            // auto avg_tb_edge_length = (ebs[0].norm() + ebs[1].norm() + ebs[2].norm()) / (T)3.0;

                            vec3 ra{};
                            vec3 rb{};

                            for(int i = 0;i != 3;++i){
                                auto r = LSL_GEO::tri_ray_intersect(vA[i],eas[i],vB[0],vB[1],vB[2]);
                                ra[i] = r;
                                if(r < (T)(1.0)) {
                                    ea_indices[nm_ea_its] = i;
                                    ea_its[nm_ea_its] = vA[i] + eas[i] * r;
                                    ++nm_ea_its;
                                }

                                r = LSL_GEO::tri_ray_intersect(vB[i],ebs[i],vA[0],vA[1],vA[2]);
                                rb[i] = r;
                                if(r < (T)(1.0)) {
                                    eb_indices[nm_eb_its] = i;
                                    eb_its[nm_eb_its] = vB[i] + ebs[i] * r;
                                    ++nm_eb_its;
                                } 
                            }

                            if(nm_eb_its + nm_ea_its > 2) {
                                printf("more than 2 intersection detected\n");
                            }
// #if 0

                            auto ori_nm_ea_its = nm_ea_its;
                            auto ori_nm_eb_its = nm_eb_its;
                            if(nm_ea_its + nm_eb_its == 1) {
                                // return;
                                if(nm_ea_its == 1) {
                                    auto ea_idx = ea_indices[0];
                                    auto v0 = vA[ea_idx];
                                    auto v1 = vA[(ea_idx + 1)  % 3];
                                    // auto avg_tb_edge_length = (ebs[0].norm() + ebs[1].norm() + ebs[2].norm()) / (T)3.0;

                                    // check if the end point of edge lies inside the the counter facet A, if so, push all the neighbored tris of  the end point 
                                    T d0 = (T)0;
                                    T d1 = (T)0;
                                    T b0 = (T)0;
                                    T b1 = (T)0;
                                    d0 = LSL_GEO::pointTriangleDistance(vB[0],vB[1],vB[2],v0,b0);
                                    d1 = LSL_GEO::pointTriangleDistance(vB[0],vB[1],vB[2],v1,b1);
                                    if(b0 < (T)(1 + 1e-6) && b1 < (T)(1 + 1e-6))
                                        return;
                                    d0 = b0 < (T)(1 + 1e-6) ? d0 : std::numeric_limits<T>::infinity();
                                    d1 = b1 < (T)(1 + 1e-6) ? d1 : std::numeric_limits<T>::infinity();
                                    // if(d0 < 0 || d1 < 0)
                                    //     printf("wrong zs::abs impl\n");
                                    if(d0 < d1) {
                                        nm_ea_its++;
                                        ea_indices[1] = (ea_idx - 1 + 3) % 3;
                                        ea_its[1] = ea_its[0];
                                    }else {
                                        nm_ea_its++;
                                        ea_indices[1] = (ea_idx + 1) % 3;
                                        ea_its[1] = ea_its[0];
                                    }
                                    // if(zs::abs((v0 - vB[0]).dot(nrmB)) < avg_tb_edge_length * 1e-4) {
                                    //     nm_ea_its++;
                                    //     ea_indices[1] = (ea_idx - 1 + 3) % 3;
                                    //     ea_its[1] = ea_its[0];
                                    // }else if(zs::abs((v1 - vB[0]).dot(nrmB)) < avg_tb_edge_length * 1e-4) {
                                    //     nm_ea_its++;
                                    //     ea_indices[1] = (ea_idx + 1) % 3;
                                    //     ea_its[1] = ea_its[0];
                                    // }
                                    // else {
                                    //     printf("losing one potential collision\n");
                                    //     // return;
                                    // }
                                }
                                if(nm_eb_its == 1) {
                                    auto eb_idx = eb_indices[0];
                                    auto v0 = vB[eb_idx];
                                    auto v1 = vB[(eb_idx + 1)  % 3];

                                    T d0 = (T)0;
                                    T d1 = (T)0;
                                    T b0 = (T)0;
                                    T b1 = (T)0;
                                    d0 = LSL_GEO::pointTriangleDistance(vA[0],vA[1],vA[2],v0,b0);
                                    d1 = LSL_GEO::pointTriangleDistance(vA[0],vA[1],vA[2],v1,b1);
                                    if(b0 < (T)(1 + 1e-6) && b1 < (T)(1 + 1e-6))
                                        return;
                                    d0 = b0 < (T)(1 + 1e-6) ? d0 : std::numeric_limits<T>::infinity();
                                    d1 = b1 < (T)(1 + 1e-6) ? d1 : std::numeric_limits<T>::infinity();

                                    // if(d0 < 0 || d1 < 0)
                                    //     printf("wrong zs::abs impl\n");
                                    if(d0 < d1) {
                                        nm_eb_its++;
                                        eb_indices[1] = (eb_idx - 1 + 3) % 3;
                                        eb_its[1] = eb_its[0];
                                    }else {
                                        nm_eb_its++;
                                        eb_indices[1] = (eb_idx + 1) % 3;
                                        eb_its[1] = eb_its[0];
                                    }
                                    // auto avg_ta_edge_length = (eas[0].norm() + eas[1].norm() + eas[2].norm()) / (T)3.0;

                                    // if(zs::abs((v0 - vB[0]).dot(nrmA)) < avg_ta_edge_length * 1e-4) {
                                    //     nm_eb_its++;
                                    //     eb_indices[1] = (eb_idx - 1 + 3) % 3;
                                    //     eb_its[1] = eb_its[0];
                                    // }
                                    // else if(zs::abs((v1 - vB[0]).dot(nrmA)) < avg_ta_edge_length * 1e-4) {
                                    //     nm_eb_its++;
                                    //     eb_indices[1] = (eb_idx + 1) % 3;
                                    //     eb_its[1] = eb_its[0];
                                    // }
                                    // else {
                                    //     printf("losing one potential collision\n");
                                    //     // return;
                                    // }
                                }
                            }
// #else
                            if(nm_ea_its + nm_eb_its == 0) {
                                // return;
                                // check wheter vertex of A intersect with B 
                                auto avg_ta_edge_length = (eas[0].norm() + eas[1].norm() + eas[2].norm()) / (T)3.0;
                                auto avg_tb_edge_length = (ebs[0].norm() + ebs[1].norm() + ebs[2].norm()) / (T)3.0;
                                for(int i = 0;i != 3;++i) {
                                    T barySum = (T)0;
                                    auto distance = LSL_GEO::pointTriangleDistance(vB[0],vB[1],vB[2],vA[i],barySum);
                                    if(barySum < (T)(1 + 1e-6) && distance < avg_ta_edge_length * 1e-5) {
                                        nm_ea_its = 2;
                                        ea_indices[0] = i;
                                        ea_indices[1] = (i + 2) % 3;
                                        ea_its[0] = ea_its[1] = vA[i];
                                        break;
                                    }

                                    distance = LSL_GEO::pointTriangleDistance(vA[0],vA[1],vA[2],vB[i],barySum);
                                    if(barySum < (T)(1 + 1e-6) && distance < avg_tb_edge_length * 1e-5) {
                                        nm_eb_its = 2;
                                        eb_indices[0] = i;
                                        eb_indices[1] = (i + 2) % 3;
                                        eb_its[0] = eb_its[1] = vB[i];
                                        break;
                                    }
                                }
                                if(nm_ea_its + nm_eb_its > 0) {
                                    // printf("find point collision : %d\ntriA: %d %d %d\ntriB: %d %d %d\nvA: %f %f %f %f %f %f %f %f %f\nvB: %f %f %f %f %f %f %f %f %f",
                                    //     nm_ea_its + nm_eb_its,
                                    //     triA[0],triA[1],triA[2],
                                    //     triB[0],triB[1],triB[2],    
                                    //     (float)vA[0][0],(float)vA[0][1],(float)vA[0][2],
                                    //     (float)vA[1][0],(float)vA[1][1],(float)vA[1][2],
                                    //     (float)vA[2][0],(float)vA[2][1],(float)vA[2][2],
                                    //     (float)vB[0][0],(float)vB[0][1],(float)vB[0][2],
                                    //     (float)vB[1][0],(float)vB[1][1],(float)vB[1][2],
                                    //     (float)vB[2][0],(float)vB[2][1],(float)vB[2][2]);
                                }
                            }
// #endif
                            if(nm_ea_its + nm_eb_its == 0)
                                return;

                            if(nm_ea_its + nm_eb_its != 2){
                                // printf("only one collision edge impossible reaching here, check the code :[%d %d] -> [%d %d]!!!\nvA : %f %f %f %f %f %f %f %f %f\nvB : %f %f %f %f %f %f %f %f %f\n",
                                //     ori_nm_ea_its,ori_nm_eb_its,nm_ea_its,nm_eb_its,
                                //     (float)vA[0][0],(float)vA[0][1],(float)vA[0][2],
                                //     (float)vA[1][0],(float)vA[1][1],(float)vA[1][2],
                                //     (float)vA[2][0],(float)vA[2][1],(float)vA[2][2],
                                //     (float)vB[0][0],(float)vB[0][1],(float)vB[0][2],
                                //     (float)vB[1][0],(float)vB[1][1],(float)vB[1][2],
                                //     (float)vB[2][0],(float)vB[2][1],(float)vB[2][2]);
                                return;
                            }

                            // if(nm_ea_its + nm_eb_its == 2) {
                            auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                            intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{ta_i,tb_i}.reinterpret_bits(float_c);

                            intersect_buffers.tuple(dim_c<6>,"its_edge_mark",offset) = zs::vec<int,6>::uniform(0).reinterpret_bits(float_c);
                            for(int i = 0;i != nm_ea_its;++i)
                                intersect_buffers("its_edge_mark",ea_indices[i],offset) = zs::reinterpret_bits<T>((int)1);
                            for(int i = 0;i != nm_eb_its;++i)
                                intersect_buffers("its_edge_mark",eb_indices[i] + 3,offset) = zs::reinterpret_bits<T>((int)1);

                            vec3 its_ps[2];
                            for(int i = 0;i != nm_ea_its;++i)
                                its_ps[i] = ea_its[i];
                            for(int i = 0;i != nm_eb_its;++i)
                                its_ps[nm_ea_its + i] = eb_its[i];
                            intersect_buffers.tuple(dim_c<6>,"int_points",offset) = zs::vec<T,6>{
                                its_ps[0][0],its_ps[0][1],its_ps[0][2],
                                its_ps[1][0],its_ps[1][1],its_ps[1][2]};

                            intersect_buffers("type",offset) = zs::reinterpret_bits<T>((int)0);
                            return;                                
                            // }
                        }
                    }
                    else{
                        
                    }
                };

                triABvh.iter_neighbors(bv,process_potential_intersection_pairs);
        });

        auto nmIntersections = nmIts.getVal(0);
        

        return nmIntersections;
}

template<typename Pol,typename PosTileVec,typename TriTileVec,typename IntsTileVec,typename HalfEdgeTileVec,typename GIA_TILEVEC>
int do_global_self_intersection_analysis_on_surface_mesh_info(Pol& pol,
    const PosTileVec& verts,
    const zs::SmallString& xtag,
    const TriTileVec& tris,
    const HalfEdgeTileVec& halfedges,
    IntsTileVec& ints_buffer,
    GIA_TILEVEC& gia_res,bool output_intermediate_information = false) {
        using namespace zs;
        using index_type = std::make_signed_t<int>;
        using size_type = std::make_unsigned_t<int>;
        using IV = zs::vec<int,2>;
        // using table_vec2i_type = zs::bcht<IV,int,true,zs::universal_hash<IV>,16>;
        using table_vec2i_type = zs::bht<int,2,int>;
        // using table_int_type = zs::bcht<int,int,true,zs::universal_hash<int>,16>;
        using table_int_type = zs::bht<int,1,int>;
        using edge_topo_type = zs::Vector<zs::vec<int,2>>;
        using inst_buffer_type = zs::Vector<zs::vec<int,2>>;
        using inst_class_type = zs::Vector<int>;
        using vec3 = zs::vec<T,3>;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};

        // table_type tab{verts.get_allocator(),tris.size() * 8};
        // inst_buffer_type ints_buffer{tris.get_allocator(),tris.size() * 8};
        // inst_class_type ints_types{tris.get_allocator(),tris.size() * 8};

        auto nm_insts = retrieve_triangulate_mesh_self_intersection_list_info(pol,
            verts,xtag,tris,verts,xtag,tris,ints_buffer,true);

        // auto topo_tag = is_volume_surface ? "fp_inds" : "inds";
        std::string topo_tag{"inds"};
        // if(!tris.hasProperty(topo_tag))
        //     fmt::print(fg(fmt::color::red),"do_global_self_intersection_analysis::the input tris has no {} topo channel\n",topo_tag);

        table_vec2i_type cftab{ints_buffer.get_allocator(),(size_t)nm_insts};
        cftab.reset(pol,true);
        zs::Vector<int> cfbuffer{ints_buffer.get_allocator(),(size_t)nm_insts};

        // std::cout << "ALL_INTERSECTION PAIR: " << nm_insts << std::endl;
        // if(!ints_buffer.hasProperty("pair"))
        //     printf("the ints_buffer has no \'pair\' channel\n");
        pol(zs::range(nm_insts),[
            cftab = proxy<space>(cftab),output_intermediate_information,
            cfbuffer = proxy<space>(cfbuffer),
            ints_buffer = proxy<space>({},ints_buffer)] ZS_LAMBDA(int isi) mutable {
                auto pair = ints_buffer.pack(dim_c<2>,"pair",isi,int_c);
                auto type = zs::reinterpret_bits<int>(ints_buffer("type",isi));
                auto ta = pair[0];
                auto tb = pair[1];
                // auto type = zs::reinterpret_bits<int>(ints_buffer("type",isi));
                if(output_intermediate_information)
                    printf("pair[%d] : [%d %d]\n",type,pair[0],pair[1]);
                if(auto setNo = cftab.insert(zs::vec<int,2>{ta,tb});setNo != table_vec2i_type::sentinel_v)
                    cfbuffer[setNo] = isi;

        });


        table_vec2i_type incidentItsTab{tris.get_allocator(),nm_insts * 2};
        incidentItsTab.reset(pol,true);
        pol(zs::range(nm_insts),[
            ints_buffer = proxy<space>({},ints_buffer),
            cftab = proxy<space>(cftab),
            cfbuffer = proxy<space>(cfbuffer),
            tris = proxy<space>({},tris),
            verts = proxy<space>({},verts),
            xtag,
            output_intermediate_information,
            incidentItsTab = proxy<space>(incidentItsTab),
            halfedges = proxy<space>({},halfedges)] ZS_LAMBDA(int isi) mutable {
                auto tpair = ints_buffer.pack(dim_c<2>,"pair",isi,int_c);
                auto ta = tpair[0];
                auto tb = tpair[1];
                auto its_edge_mark = ints_buffer.pack(dim_c<6>,"its_edge_mark",isi,int_c);
                auto type = zs::reinterpret_bits<int>(ints_buffer("type",isi));

                // return;
                if(type == 1) {// intersection pair with one topological coincident vertex
                    for(int i = 0;i != 3;++i)
                        if(its_edge_mark[i] == 1){
                            // the edge of the first triangle intersect with the second triangle
                            auto he_idx = zs::reinterpret_bits<int>(tris("he_inds",ta));
                            // int he_idx = start_he_idx;
                            for(int j = 0;j != i;++j)
                                he_idx = zs::reinterpret_bits<int>(halfedges("next_he",he_idx));
                            auto opposite_he_idx = zs::reinterpret_bits<int>(halfedges("opposite_he",he_idx));
                            auto tn = zs::reinterpret_bits<int>(halfedges("to_face",opposite_he_idx));

                            auto t0 = tb;
                            auto t1 = tn;
                            if(t0 > t1) {
                                auto tmp = t1;
                                t1 = t0;
                                t0 = tmp;
                            }

                            if(auto nItsIdx = cftab.query(zs::vec<int,2>{t0,t1});nItsIdx != table_vec2i_type::sentinel_v) {
                                nItsIdx = cfbuffer[nItsIdx];
                                auto isi0 = isi;
                                auto isi1 = nItsIdx;
                                if(isi0 > isi1) {
                                    auto tmp = isi0;
                                    isi0 = isi1;
                                    isi1 = tmp;
                                }
                                incidentItsTab.insert(zs::vec<int,2>{isi0,isi1});                                
                            }else if(output_intermediate_information)
                                printf("invalid_0 nItsIdx[%d] query from [%d %d] : tpair[%d %d]\ntest incidence for pair-type[%d]-[%d %d] : edge_incidence[%d %d %d %d %d %d]\n",
                                    nItsIdx,t0,t1,tpair[0],tpair[1],
                                    type,tpair[0],tpair[1],
                                    its_edge_mark[0],
                                    its_edge_mark[1],
                                    its_edge_mark[2],
                                    its_edge_mark[3],
                                    its_edge_mark[4],
                                    its_edge_mark[5]);

                                    
                        }
                    for(int i = 3;i != 6;++i)
                        if(its_edge_mark[i] == 1) {
                            // the edge of the second triangle intersect with the first triangle
                            auto he_idx = zs::reinterpret_bits<int>(tris("he_inds",tb));
                            // int he_idx = start_he_idx;
                            for(int j = 0;j != i-3;++j)
                                he_idx = zs::reinterpret_bits<int>(halfedges("next_he",he_idx));
                            auto opposite_he_idx = zs::reinterpret_bits<int>(halfedges("opposite_he",he_idx));
                            auto tn = zs::reinterpret_bits<int>(halfedges("to_face",opposite_he_idx));

                            auto t0 = ta;
                            auto t1 = tn;
                            if(t0 > t1) {
                                auto tmp = t1;
                                t1 = t0;
                                t0 = tmp;
                            }

                            if(auto nItsIdx = cftab.query(zs::vec<int,2>{t0,t1});nItsIdx != table_vec2i_type::sentinel_v) {
                                nItsIdx = cfbuffer[nItsIdx];
                                auto isi0 = isi;
                                auto isi1 = nItsIdx;
                                if(isi0 > isi1) {
                                    auto tmp = isi0;
                                    isi0 = isi1;
                                    isi1 = tmp;
                                }
                                incidentItsTab.insert(zs::vec<int,2>{isi0,isi1});                                
                            }else if(output_intermediate_information)
                                printf("invalid_1 nItsIdx[%d] query from [%d %d] : tpair[%d %d]\n",nItsIdx,t0,t1,tpair[0],tpair[1]);
                        }
                }
                else if(type == 0) {// intersection pair without topological coincident vertex
                    for(int i = 0;i != 3;++i){
                        if(its_edge_mark[i] == 1){
                            // // the edge of the first triangle intersect with the second trianglegia_res
                            auto he_idx = zs::reinterpret_bits<int>(tris("he_inds",ta));
                            // int he_idx = start_he_idx;
                            for(int j = 0;j != i;++j)
                                he_idx = zs::reinterpret_bits<int>(halfedges("next_he",he_idx));
                            auto opposite_he_idx = zs::reinterpret_bits<int>(halfedges("opposite_he",he_idx));
                            auto tn = zs::reinterpret_bits<int>(halfedges("to_face",opposite_he_idx));

                            auto t0 = tb;
                            auto t1 = tn;
                            if(t0 > t1) {
                                auto tmp = t1;
                                t1 = t0;
                                t0 = tmp;
                            }

                            if(auto nItsIdx = cftab.query(zs::vec<int,2>{t0,t1});nItsIdx != table_vec2i_type::sentinel_v) {
                                nItsIdx = cfbuffer[nItsIdx];
                                auto isi0 = isi;
                                auto isi1 = nItsIdx;
                                if(isi0 > isi1) {
                                    auto tmp = isi0;
                                    isi0 = isi1;
                                    isi1 = tmp;
                                }
                                incidentItsTab.insert(zs::vec<int,2>{isi0,isi1});                                
                            }else{
                                auto tri_t0 = tris.pack(dim_c<3>,"inds",t0,int_c);
                                auto tri_t1 = tris.pack(dim_c<3>,"inds",t1,int_c);
                                auto tri_tp0 = tris.pack(dim_c<3>,"inds",tpair[0],int_c);
                                auto tri_tp1 = tris.pack(dim_c<3>,"inds",tpair[1],int_c);

                                // testing if t0 and t1 actually intersect
                                int nm_topological_coincidences = 0;
                                for(int j = 0;j != 3;++j)
                                    for(int k = 0;k != 3;++k)
                                        if(tri_t0[j] == tri_t1[k])
                                            ++nm_topological_coincidences;
                                
                                if(nm_topological_coincidences == 0) {
                                    vec3 v0[3] = {};
                                    vec3 v1[3] = {};
                                    vec3 e0s[3] = {};
                                    vec3 e1s[3] = {};
                                    for(int i = 0;i != 3;++i) {
                                        v0[i] = verts.pack(dim_c<3>,xtag,tri_t0[i]);
                                        v1[i] = verts.pack(dim_c<3>,xtag,tri_t1[i]);
                                    }

                                    for(int i = 0;i != 3;++i) {
                                        e0s[i] = v0[(i + 1) % 3] - v0[i];
                                        e1s[i] = v1[(i + 1) % 3] - v1[i];
                                    }

                                    auto nrm0 = (v0[1] - v0[0]).cross(v0[2] - v0[0]).normalized();
                                    auto nrm1 = (v1[1] - v1[0]).cross(v1[2] - v1[0]).normalized();

                                    if(zs::abs(nrm0.dot(nrm1)) > (T)(1 - 1e-5)) {
                                        if(output_intermediate_information)
                                            printf("due to normal check invalid_2 nItsIdx[%d] query from [%d %d] : tpair[%d %d]\nt0 : %d %d %d\nt1 : %d %d %d\ntpair[0]: %d %d %d\ntpair[1]: %d %d %d\nnormal_check : %f\n",
                                                nItsIdx,t0,t1,tpair[0],tpair[1],
                                                tri_t0[0],tri_t0[1],tri_t0[2],
                                                tri_t1[0],tri_t1[1],tri_t1[2],
                                                tri_tp0[0],tri_tp0[1],tri_tp0[2],
                                                tri_tp1[0],tri_tp1[1],tri_tp1[2],(float)zs::abs(nrm0.dot(nrm1)));
                                        return;
                                    }
                                    // return;

                                    auto e0_indices = zs::vec<int,3>::uniform(-1);
                                    auto e1_indices = zs::vec<int,3>::uniform(-1);
                                    int nm_e0_its = 0;
                                    int nm_e1_its = 0;
                                    vec3 e0_its[3] = {};
                                    vec3 e1_its[3] = {};

                                    for(int j = 0;j != 3;++j){
                                        auto r = LSL_GEO::tri_ray_intersect(v0[j],e0s[j],v1[0],v1[1],v1[2]);
                                        if(r < (T)(1.0 - 1e-6)) {
                                            e0_indices[nm_e0_its] = j;
                                            e0_its[nm_e0_its] = v0[j] + e0s[j] * r;
                                            ++nm_e0_its;
                                        }

                                        r = LSL_GEO::tri_ray_intersect(v1[j],e1s[j],v0[0],v0[1],v0[2]);
                                        if(r < (T)(1.0 - 1e-6)) {
                                            e1_indices[nm_e1_its] = j;
                                            e1_its[nm_e1_its] = v1[i] + e1s[j] * r;
                                            ++nm_e1_its;
                                        } 
                                    }

                                    if(nm_e0_its + nm_e1_its == 1){
                                        if(output_intermediate_information)
                                            printf("point geometrical coincidence invalid_2 nItsIdx[%d] query from [%d %d] : tpair[%d %d]\nt0 : %d %d %d\nt1 : %d %d %d\ntpair[0]: %d %d %d\ntpair[1]: %d %d %d\n",
                                            nItsIdx,t0,t1,tpair[0],tpair[1],
                                            tri_t0[0],tri_t0[1],tri_t0[2],
                                            tri_t1[0],tri_t1[1],tri_t1[2],
                                            tri_tp0[0],tri_tp0[1],tri_tp0[2],
                                            tri_tp1[0],tri_tp1[1],tri_tp1[2]); 
                                        return;
                                    }else {
                                        if(output_intermediate_information)
                                            printf("impossible reaching here invalid_2 nItsIdx[%d] query from [%d %d] : tpair[%d %d]\nt0 : %d %d %d\nt1 : %d %d %d\ntpair[0]: %d %d %d\ntpair[1]: %d %d %d\nnm_its : %d\n",
                                                nItsIdx,t0,t1,tpair[0],tpair[1],
                                                tri_t0[0],tri_t0[1],tri_t0[2],
                                                tri_t1[0],tri_t1[1],tri_t1[2],
                                                tri_tp0[0],tri_tp0[1],tri_tp0[2],
                                                tri_tp1[0],tri_tp1[1],tri_tp1[2],nm_e0_its + nm_e1_its); 
                                        return;
                                    }
                                    

                                }else {
                                    if(output_intermediate_information)
                                        printf("due to topological coincidence invalid_2 nItsIdx[%d] query from [%d %d] : tpair[%d %d]\nt0 : %d %d %d\nt1 : %d %d %d\ntpair[0]: %d %d %d\ntpair[1]: %d %d %d\n",
                                            nItsIdx,t0,t1,tpair[0],tpair[1],
                                            tri_t0[0],tri_t0[1],tri_t0[2],
                                            tri_t1[0],tri_t1[1],tri_t1[2],
                                            tri_tp0[0],tri_tp0[1],tri_tp0[2],
                                            tri_tp1[0],tri_tp1[1],tri_tp1[2]); 
                                    return;  
                                }
                            }
                        }
                    }
                    for(int i = 3;i != 6;++i){
                        if(its_edge_mark[i] == 1) {
                            auto he_idx = zs::reinterpret_bits<int>(tris("he_inds",tb));
                            // int he_idx = start_he_idx;
                            for(int j = 0;j != i-3;++j)
                                he_idx = zs::reinterpret_bits<int>(halfedges("next_he",he_idx));
                            auto opposite_he_idx = zs::reinterpret_bits<int>(halfedges("opposite_he",he_idx));
                            auto tn = zs::reinterpret_bits<int>(halfedges("to_face",opposite_he_idx));

                            // auto triB = tris.pack(dim_c<3>,"inds",tb,int_c);
                            // auto inter_b_edge = zs::vec<int,2>{tri_B[(i + 1) % ],tri_B[i]};

                            auto t0 = ta;
                            auto t1 = tn;
                            if(t0 > t1) {
                                auto tmp = t1;
                                t1 = t0;
                                t0 = tmp;
                            }

                            if(auto nItsIdx = cftab.query(zs::vec<int,2>{t0,t1});nItsIdx != table_vec2i_type::sentinel_v) {
                                nItsIdx = cfbuffer[nItsIdx];
                                auto isi0 = isi;
                                auto isi1 = nItsIdx;
                                if(isi0 > isi1) {
                                    auto tmp = isi0;
                                    isi0 = isi1;
                                    isi1 = tmp;
                                }
                                incidentItsTab.insert(zs::vec<int,2>{isi0,isi1});                                
                            }else if(output_intermediate_information){
                                auto tri_t0 = tris.pack(dim_c<3>,"inds",t0,int_c);
                                auto tri_t1 = tris.pack(dim_c<3>,"inds",t1,int_c);
                                auto tri_tp0 = tris.pack(dim_c<3>,"inds",tpair[0],int_c);
                                auto tri_tp1 = tris.pack(dim_c<3>,"inds",tpair[1],int_c);

                                // testing if t0 and t1 actually intersect
                                int nm_topological_coincidences = 0;
                                for(int j = 0;j != 3;++j)
                                    for(int k = 0;k != 3;++k)
                                        if(tri_t0[j] == tri_t1[k])
                                            ++nm_topological_coincidences;
                                
                                if(nm_topological_coincidences == 0) {
                                    vec3 v0[3] = {};
                                    vec3 v1[3] = {};
                                    vec3 e0s[3] = {};
                                    vec3 e1s[3] = {};
                                    for(int i = 0;i != 3;++i) {
                                        v0[i] = verts.pack(dim_c<3>,xtag,tri_t0[i]);
                                        v1[i] = verts.pack(dim_c<3>,xtag,tri_t1[i]);
                                    }

                                    for(int i = 0;i != 3;++i) {
                                        e0s[i] = v0[(i + 1) % 3] - v0[i];
                                        e1s[i] = v1[(i + 1) % 3] - v1[i];
                                    }

                                    auto nrm0 = (v0[1] - v0[0]).cross(v0[2] - v0[0]).normalized();
                                    auto nrm1 = (v1[1] - v1[0]).cross(v1[2] - v1[0]).normalized();

                                    if(zs::abs(nrm0.dot(nrm1)) > (T)(1 - 1e-5)) {
                                        if(output_intermediate_information)
                                            printf("due to normal check invalid_3 nItsIdx[%d] query from [%d %d] : tpair[%d %d]\nt0 : %d %d %d\nt1 : %d %d %d\ntpair[0]: %d %d %d\ntpair[1]: %d %d %d\nnormal_check : %f\n",
                                                nItsIdx,t0,t1,tpair[0],tpair[1],
                                                tri_t0[0],tri_t0[1],tri_t0[2],
                                                tri_t1[0],tri_t1[1],tri_t1[2],
                                                tri_tp0[0],tri_tp0[1],tri_tp0[2],
                                                tri_tp1[0],tri_tp1[1],tri_tp1[2],(float)nrm0.dot(nrm1));
                                        return;
                                    }
                                    // return;

                                    auto e0_indices = zs::vec<int,3>::uniform(-1);
                                    auto e1_indices = zs::vec<int,3>::uniform(-1);
                                    int nm_e0_its = 0;
                                    int nm_e1_its = 0;
                                    vec3 e0_its[3] = {};
                                    vec3 e1_its[3] = {};

                                    for(int j = 0;j != 3;++j){
                                        auto r = LSL_GEO::tri_ray_intersect(v0[j],e0s[j],v1[0],v1[1],v1[2]);
                                        if(r < (T)(1.0)) {
                                            e0_indices[nm_e0_its] = j;
                                            e0_its[nm_e0_its] = v0[j] + e0s[j] * r;
                                            ++nm_e0_its;
                                        }

                                        r = LSL_GEO::tri_ray_intersect(v1[j],e1s[j],v0[0],v0[1],v0[2]);
                                        if(r < (T)(1.0)) {
                                            e1_indices[nm_e1_its] = j;
                                            e1_its[nm_e1_its] = v1[i] + e1s[j] * r;
                                            ++nm_e1_its;
                                        } 
                                    }

                                    if(nm_e0_its + nm_e1_its == 1){
                                        if(output_intermediate_information)
                                            printf("point geometrical coincidence invalid_3 nItsIdx[%d] query from [%d %d] : tpair[%d %d]\nt0 : %d %d %d\nt1 : %d %d %d\ntpair[0]: %d %d %d\ntpair[1]: %d %d %d\nv0 : %f %f %f %f %f %f %f %f %f\nv1: %f %f %f %f %f %f %f %f %f\n",
                                                nItsIdx,t0,t1,tpair[0],tpair[1],
                                                tri_t0[0],tri_t0[1],tri_t0[2],
                                                tri_t1[0],tri_t1[1],tri_t1[2],
                                                tri_tp0[0],tri_tp0[1],tri_tp0[2],
                                                tri_tp1[0],tri_tp1[1],tri_tp1[2],
                                                (float)v0[0][0],(float)v0[0][1],(float)v0[0][2],
                                                (float)v0[1][0],(float)v0[1][1],(float)v0[1][2],
                                                (float)v0[2][0],(float)v0[2][1],(float)v0[2][2],
                                                (float)v1[0][0],(float)v1[0][1],(float)v1[0][2],
                                                (float)v1[1][0],(float)v1[1][1],(float)v1[1][2],
                                                (float)v1[2][0],(float)v1[2][1],(float)v1[2][2]); 
                                        return;
                                    }else{
                                        if(output_intermediate_information)  
                                            printf("impossible reaching here invalid_3 nItsIdx[%d] \nv0 : %f %f %f %f %f %f %f %f %f\nv1: %f %f %f %f %f %f %f %f %f\n",
                                                nItsIdx,
                                                // t0,t1,tpair[0],tpair[1],
                                                // tri_t0[0],tri_t0[1],tri_t0[2],
                                                // tri_t1[0],tri_t1[1],tri_t1[2],
                                                // tri_tp0[0],tri_tp0[1],tri_tp0[2],
                                                // tri_tp1[0],tri_tp1[1],tri_tp1[2],nm_e0_its + nm_e1_its,
                                                (float)v0[0][0],(float)v0[0][1],(float)v0[0][2],
                                                (float)v0[1][0],(float)v0[1][1],(float)v0[1][2],
                                                (float)v0[2][0],(float)v0[2][1],(float)v0[2][2],
                                                (float)v1[0][0],(float)v1[0][1],(float)v1[0][2],
                                                (float)v1[1][0],(float)v1[1][1],(float)v1[1][2],
                                                (float)v1[2][0],(float)v1[2][1],(float)v1[2][2]);  
                                        return;
                                    }
                                    

                                }else {
                                    if(output_intermediate_information)
                                        printf("due to topological coincidence invalid_3 nItsIdx[%d] query from [%d %d] : tpair[%d %d]\nt0 : %d %d %d\nt1 : %d %d %d\ntpair[0]: %d %d %d\ntpair[1]: %d %d %d\n",
                                            nItsIdx,t0,t1,tpair[0],tpair[1],
                                            tri_t0[0],tri_t0[1],tri_t0[2],
                                            tri_t1[0],tri_t1[1],tri_t1[2],
                                            tri_tp0[0],tri_tp0[1],tri_tp0[2],
                                            tri_tp1[0],tri_tp1[1],tri_tp1[2]);   
                                    return;
                                }
                            }           
                        }
                    }
                }

        });

        if(output_intermediate_information)
            std::cout << "FINISH INTERSECTION TOPO EVAL" << std::endl;

        auto nmEntries = incidentItsTab.size();
        zs::Vector<zs::vec<int,2>> conn_topo{tris.get_allocator(),nmEntries};
        pol(zip(conn_topo,range(incidentItsTab._activeKeys)),[] ZS_LAMBDA(zs::vec<int,2> &ij,const auto& key) mutable {
            ij = key;
        });

        edge_topo_type edge_topos{tris.get_allocator(),tris.size() * 3};
        pol(range(tris.size()),[
            tris = proxy<space>({},tris),
            // tab = proxy<space>(tab),
            topo_tag = zs::SmallString(topo_tag),
            edge_topos = proxy<space>(edge_topos)] ZS_LAMBDA(int ti) mutable {
                auto tri = tris.pack(dim_c<3>,topo_tag,ti,int_c);
                for(int i = 0;i != 3;++i){
                    edge_topos[ti * 3 + i] = zs::vec<int,2>{tri[i],tri[(i + 1) % 3]};
                }
        }); 


        zs::Vector<int> ringTag{tris.get_allocator(),nm_insts};
        auto nm_rings = mark_disconnected_island(pol,conn_topo,ringTag);
        zs::Vector<int> ringSize(tris.get_allocator(),nm_rings);
        pol(zs::range(ringSize.size()),[
            ringSize = proxy<space>(ringSize)] ZS_LAMBDA(int ri) mutable {
                ringSize[ri] = 0;
        });
        pol(zs::range(nm_insts),[
            ints_buffer = proxy<space>({},ints_buffer),
            ringTag = proxy<space>(ringTag),
            exec_tag,
            ringSize = proxy<space>(ringSize)] ZS_LAMBDA(int isi) mutable {
                atomic_add(exec_tag,&ringSize[ringTag[isi]],(int)1);
        });

        zs::Vector<int> island_buffer{verts.get_allocator(),verts.size()};
        
        gia_res.resize(verts.size());
        pol(zs::range(verts.size()),[gia_res = proxy<space>({},gia_res)] ZS_LAMBDA(int ni) mutable {
            // nodal_colors[ni] = 0;
            gia_res("ring_mask",ni) = zs::reinterpret_bits<T>((int)0);
            gia_res("color_mask",ni) = zs::reinterpret_bits<T>((int)0);
            gia_res("type_mask",ni) = zs::reinterpret_bits<T>((int)0);
        });

        if(output_intermediate_information)
            std::cout << "nm_rings " << nm_rings << std::endl;

        for(int ri = 0;ri != nm_rings;++ri) {
            auto rsize = ringSize.getVal(ri);
            if(output_intermediate_information)
                printf("ring[%d] Size : %d\n",ri,rsize);

            // edge_topo_type dc_edge_topos{tris.get_allocator(),rsize * 6};
            table_int_type disable_points{tris.get_allocator(),rsize * 8};
            table_vec2i_type disable_lines{tris.get_allocator(),rsize * 6};
            disable_points.reset(pol,true);
            disable_lines.reset(pol,true);

            pol(zs::range(nm_insts),[
                ints_buffer = proxy<space>({},ints_buffer),
                ringTag = proxy<space>(ringTag),
                output_intermediate_information,
                ri,
                topo_tag = zs::SmallString(topo_tag),
                // dc_edge_topos = proxy<space>(dc_edge_topos),
                disable_points = proxy<space>(disable_points),
                disable_lines = proxy<space>(disable_lines),
                nm_insts,
                tris = proxy<space>({},tris)] ZS_LAMBDA(int isi) mutable {
                    if(ringTag[isi] != ri)
                        return;
                    auto pair = ints_buffer.pack(dim_c<2>,"pair",isi,int_c);
                    auto type = zs::reinterpret_bits<int>(ints_buffer("type",isi));

                    auto ta = pair[0];
                    auto tb = pair[1];
                    zs::vec<int,3> tri_pairs[2] = {};
                    tri_pairs[0] = tris.pack(dim_c<3>,topo_tag,ta,int_c);
                    tri_pairs[1] = tris.pack(dim_c<3>,topo_tag,tb,int_c);

                    for(int t = 0;t != 2;++t) {
                        zs::vec<int,2> out_edges[3] = {};
                        elm_to_edges(tri_pairs[t],out_edges);
                        for(int i = 0;i != 3;++i) {
                            auto a = out_edges[i][0];
                            auto b = out_edges[i][1];
                            if(a > b) {
                                auto tmp = a;
                                a = b;
                                b = tmp;
                            }
                            disable_lines.insert(zs::vec<int,2>{a,b});
                        }
                    }

                    if(type == 1) {
                        int coincident_idx = -1;
                        for(int i = 0;i != 3;++i)
                            for(int j = 0;j != 3;++j)
                                if(tri_pairs[0][i] == tri_pairs[1][j])
                                    coincident_idx = tri_pairs[0][i];
                        if(coincident_idx < 0){
                            if(output_intermediate_information)
                                printf("invalid coincident_idx detected : %d\n",coincident_idx);
                        }else
                            disable_points.insert(coincident_idx);
                    }   
    
            });
            auto nm_islands = mark_disconnected_island(pol,edge_topos,disable_points,disable_lines,island_buffer);
            zs::Vector<int> nm_cmps_every_island_count{verts.get_allocator(),(size_t)nm_islands};
            // nm_cmps_every_island_count.setVal(0);
            pol(zs::range(nm_cmps_every_island_count.size()),[
                nm_cmps_every_island_count = proxy<space>(nm_cmps_every_island_count)] ZS_LAMBDA(int i) mutable {
                    nm_cmps_every_island_count[i] = 0;
            });
            pol(zs::range(verts.size()),[
                verts = proxy<space>({},verts),
                island_buffer = proxy<space>(island_buffer),
                nm_cmps_every_island_count = proxy<space>(nm_cmps_every_island_count)] ZS_LAMBDA(int vi) mutable {
                    auto island_idx = island_buffer[vi];
                    atomic_add(exec_cuda,&nm_cmps_every_island_count[island_idx],(int)1);
            });

            int max_size = 0;
            int max_island_idx = 0;
            for(int i = 0;i != nm_islands;++i) {
                auto size_of_island = nm_cmps_every_island_count.getVal(i);
                if(size_of_island > max_size){
                    max_size = size_of_island;
                    max_island_idx = i;
                }
            }
            int black_island_idx = -1;
            if(nm_islands == 3) {
                for(int i = 0;i != nm_islands;++i){
                    if(i == max_island_idx)
                        continue;
                    black_island_idx = i;
                    break;
                }
            }
            auto cur_ri_mask = (int)1 << ri;

            pol(zs::range(verts.size()),[
                gia_res = proxy<space>({},gia_res),
                nm_islands,
                cur_ri_mask,
                black_island_idx,
                exec_tag,
                // ints_types = proxy<space>(ints_types),
                max_island_idx = max_island_idx,
                island_buffer = proxy<space>(island_buffer)] ZS_LAMBDA(int vi) mutable {
                    auto island_idx = island_buffer[vi];

                    // might exceed the integer range
                    auto ring_mask = zs::reinterpret_bits<int>(gia_res("ring_mask",vi));
                    auto color_mask = zs::reinterpret_bits<int>(gia_res("color_mask",vi));
                    auto type_mask = zs::reinterpret_bits<int>(gia_res("type_mask",vi));
                    // ring_mask += ((int) << ri)
                    if(island_idx != max_island_idx/* || ints_types[island_idx] == 1*/){
                        ring_mask |= cur_ri_mask;
                    }
                    if(nm_islands == 3)
                        type_mask |= cur_ri_mask;
                    if(nm_islands == 3 && island_idx == black_island_idx)
                        color_mask |= cur_ri_mask;
                    gia_res("ring_mask",vi) = zs::reinterpret_bits<T>(ring_mask);
                    gia_res("color_mask",vi) = zs::reinterpret_bits<T>(color_mask);
                    gia_res("type_mask",vi) = zs::reinterpret_bits<T>(type_mask);
            });
        }




        // skip the intersection pair with only one coincident point? s
        pol(zs::range(nm_insts),[
            // ints_types = proxy<space>(ints_types),
            gia_res = proxy<space>({},gia_res),
            ints_buffer = proxy<space>({},ints_buffer),
            tris = proxy<space>({},tris)] ZS_LAMBDA(int isi) mutable {
                auto type = zs::reinterpret_bits<int>(ints_buffer("type",isi));
                if(type == 1) {
                    auto tpair = ints_buffer.pack(dim_c<2>,"pair",isi,int_c);
                    auto ta = tpair[0];
                    auto tb = tpair[1];
                    auto triA = tris.pack(dim_c<3>,"inds",ta,int_c);
                    auto triB = tris.pack(dim_c<3>,"inds",tb,int_c);


                    int coidx = 0;
                    for(int i = 0;i != 3;++i)
                        for(int j = 0;j != 3;++j)
                            if(triA[i] == triB[j])
                                coidx = triA[i];
                    // nodal_colors[coidx] = 0;
                    gia_res("ring_mask",coidx) = zs::reinterpret_bits<T>((int)0);
                    gia_res("color_mask",coidx) = zs::reinterpret_bits<T>((int)0);
                    gia_res("type_mask",coidx) = zs::reinterpret_bits<T>((int)0);
                }
        });

        return nm_insts;
}


template<typename Pol,typename PosTileVec,typename TriTileVec,typename HETileVec,typename InstTileVec>
int retrieve_intersection_tri_halfedge_info_of_two_meshes(Pol& pol,
    const PosTileVec& verts_A, const zs::SmallString& xtag_A,
    const TriTileVec& tris_A,
    const HETileVec& halfedges_A,
    const PosTileVec& verts_B, const zs::SmallString& xtag_B,
    const TriTileVec& tris_B,
    // const HETileVec& halfedges_B,
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

        auto bvs = retrieve_bounding_volumes(pol,verts_B,tris_B,wrapv<3>{},0,xtag_B);
        auto tri_B_bvh = LBvh<3,int,T>{};
        tri_B_bvh.build(pol,bvs);

        auto cnorm_A = compute_average_edge_length(pol,verts_A,xtag_A,tris_A);
        auto cnorm_B = compute_average_edge_length(pol,verts_B,xtag_B,tris_B);
        auto cnorm = cnorm_A > cnorm_B ? cnorm_A : cnorm_B;
        cnorm *= 3;

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

            vec3 eV_A[2] = {};
            auto edgeCenter_A = vec3::zeros();
            for(int i = 0;i != 2;++i) {
                eV_A[i] = verts_A.pack(dim_c<3>,xtag_A,edge_A[i]);
                edgeCenter_A += eV_A[i] / (T)2.0;
            }

            auto dir_A = eV_A[1] - eV_A[0];
            auto bv_A = bv_t{get_bounding_box(edgeCenter_A - thickness,edgeCenter_A + thickness)};

            auto process_potential_he_tri_intersection_pairs = [&](int ti_B) {
                auto tri_B = tris_B.pack(dim_c<3>,"inds",ti_B,int_c);
                // might need an accurate predicate here for floating-point intersection testing
                {
                    vec3 tV_B[3] = {};
                    for(int i = 0;i != 3;++i)
                        tV_B[i] = verts_B.pack(dim_c<3>,xtag_B,tri_B[i]);
                    // auto dir = eV[1] - tV[0];
                    auto r = LSL_GEO::tri_ray_intersect(eV_A[0],dir_A,tV_B[0],tV_B[1],tV_B[2]);
                    if(r < (T)1.0) {
                        auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                        auto intp = r * dir_A + eV_A[0];
                        intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{hei_A,ti_B}.reinterpret_bits(float_c);
                        intersect_buffers.tuple(dim_c<3>,"int_points",offset) = intp;
                        // make sure the opposite he - tri pairs are also inserted
                        auto opposite_hei_A = zs::reinterpret_bits<int>(halfedges_A("opposite_he",hei_A));
                        if(opposite_hei_A >= 0) {
                            offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                            intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{opposite_hei_A,ti_B}.reinterpret_bits(float_c);
                            intersect_buffers.tuple(dim_c<3>,"int_points",offset) = intp;
                        }
                    }
                }                    
            };
            tri_B_bvh.iter_neighbors(bv_A,process_potential_he_tri_intersection_pairs);
        });
        return nmIts.getVal(0);
        // return 0;
}

template<typename Pol,typename PosTileVec,typename TriTileVec,typename HalfEdgeTileVec,typename GIA_TILEVEC,typename IntsTileVec>
int do_global_intersection_analysis(Pol& pol,
    const PosTileVec& verts_A,const zs::SmallString& xtag_A,
    const TriTileVec& tris_A,
    const HalfEdgeTileVec& halfedges_A,
    const PosTileVec& verts_B,const zs::SmallString& xtag_B,
    const TriTileVec& tris_B,
    const HalfEdgeTileVec& halfedges_B,
    IntsTileVec& ints_buffer_A_2_B,int &nm_A_2_B_ints,
    IntsTileVec& ints_buffer_B_2_A,int &nm_B_2_A_ints,
    GIA_TILEVEC& gia_res,int max_nm_intersections = 1000) {
        using namespace zs;
        using T = typename PosTileVec::value_type;
        using dtiles_t = zs::TileVector<T, 32>;
        using table_vec2i_type = zs::bht<int,2,int>;
        using table_int_type = zs::bht<int,1,int>;
        using vec2i = zs::vec<int,2>;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};
        
        // dtiles_t ints_buffer_A_2_B{verts_A.get_allocator(),
        //     {
        //         {"pair",2},
        //         {"int_points",3}
        // },max_nm_intersections};

        // dtiles_t ints_buffer_B_2_A{verts_B.get_allocator(),
        //     {
        //         {"pair",2},
        //         {"int_points",3}
        // },max_nm_intersections};

        nm_A_2_B_ints = retrieve_intersection_tri_halfedge_info_of_two_meshes(pol,verts_A,xtag_A,tris_A,halfedges_A,verts_B,xtag_B,tris_B,ints_buffer_A_2_B);
        nm_B_2_A_ints = retrieve_intersection_tri_halfedge_info_of_two_meshes(pol,verts_B,xtag_B,tris_B,halfedges_B,verts_A,xtag_A,tris_A,ints_buffer_B_2_A);

        // std::cout << "nm_A_2_B_ints : " << nm_A_2_B_ints << "\t nm_B_2_A_ints : " << nm_B_2_A_ints << std::endl;

        auto total_nm_verts = verts_A.size() + verts_B.size();
        gia_res.resize(total_nm_verts);
        TILEVEC_OPS::fill(pol,gia_res,"ring_mask",zs::reinterpret_bits<T>((int)0));

        // return 0;

        auto total_nm_ints = nm_A_2_B_ints + nm_B_2_A_ints;
        if(total_nm_ints == 0)
            return 0;

        table_vec2i_type A_2_B_tab{ints_buffer_A_2_B.get_allocator(),(size_t)nm_A_2_B_ints};
        A_2_B_tab.reset(pol,true);
        zs::Vector<int> A_2_B_buffer{ints_buffer_A_2_B.get_allocator(),nm_A_2_B_ints};

        table_vec2i_type B_2_A_tab{ints_buffer_B_2_A.get_allocator(),(size_t)nm_B_2_A_ints};
        B_2_A_tab.reset(pol,true);
        zs::Vector<int> B_2_A_buffer{ints_buffer_B_2_A.get_allocator(),(size_t)nm_B_2_A_ints};

        pol(zs::range(nm_A_2_B_ints),[
            ints_buffer_A_2_B = proxy<space>({},ints_buffer_A_2_B),
            A_2_B_buffer = proxy<space>(A_2_B_buffer),
            A_2_B_tab = proxy<space>(A_2_B_tab)] ZS_LAMBDA(int iti) mutable {
                auto pair = ints_buffer_A_2_B.pack(dim_c<2>,"pair",iti,int_c);
                if(auto no = A_2_B_tab.insert(pair);no >= 0)
                    A_2_B_buffer[no] = iti;
                else
                    printf("the pair A_2_B[%d %d] has been inserted more than once\n",pair[0],pair[1]);
        });

        pol(zs::range(nm_B_2_A_ints),[
            ints_buffer_B_2_A = proxy<space>({},ints_buffer_B_2_A),
            B_2_A_buffer = proxy<space>(B_2_A_buffer),
            B_2_A_tab = proxy<space>(B_2_A_tab)] ZS_LAMBDA(int iti) mutable {
                auto pair = ints_buffer_B_2_A.pack(dim_c<2>,"pair",iti,int_c);
                if(auto no = B_2_A_tab.insert(pair);no >= 0)
                    B_2_A_buffer[no] = iti;
                else
                    printf("the pair B_2_A[%d %d] has been inserted more than once\n",pair[0],pair[1]);
        });

        //we order the all collision-pairs as [A_2_B,B_2_A]
        table_vec2i_type incidentItsTab{gia_res.get_allocator(),(size_t)total_nm_ints * 2};
        incidentItsTab.reset(pol,true);
        pol(zs::range(nm_A_2_B_ints),[
            ints_buffer_A_2_B = proxy<space>({},ints_buffer_A_2_B),
            A_2_B_tab = proxy<space>(A_2_B_tab),
            A_2_B_buffer = proxy<space>(A_2_B_buffer),
            halfedges_A = proxy<space>({},halfedges_A),
            offset = A_2_B_buffer.size(),
            incidentItsTab = proxy<space>(incidentItsTab),
            ints_buffer_B_2_A = proxy<space>({},ints_buffer_B_2_A),
            B_2_A_tab = proxy<space>(B_2_A_tab),
            B_2_A_buffer = proxy<space>(B_2_A_buffer),
            tris_B = proxy<space>({},tris_B),
            halfedges_B = proxy<space>({},halfedges_B)] ZS_LAMBDA(int a2b_id) mutable {
                auto a2b_pair = ints_buffer_A_2_B.pack(dim_c<2>,"pair",a2b_id,int_c);
                auto ha = a2b_pair[0];
                auto tb = a2b_pair[1];
                auto oha = zs::reinterpret_bits<int>(halfedges_A("opposite_he",ha));
                // insert the conn of the counter pair of reverse-helfedge
                if(oha >= 0) {
                    if(auto no = A_2_B_tab.query(vec2i{oha,tb});no >= 0) {
                        auto na2b_id = A_2_B_buffer[no];
                        // as it is symmtric, we only need to establish this connectivity once
                        if(a2b_id > na2b_id)
                            incidentItsTab.insert(vec2i{a2b_id,na2b_id});
                    }else {
                        printf("impossible reaching here, the hi and ohi should both have been inserted\n");
                    }
                }

                // finding a possible pair in the loop of current halfedge
                auto nha = ha;
                for(int i = 0;i != 2;++i) {
                    nha = zs::reinterpret_bits<int>(halfedges_A("next_he",nha));
                    if(auto no = A_2_B_tab.query(vec2i{nha,tb});no >= 0) {
                        auto na2b_id = A_2_B_buffer[no];
                        // as it is symmtric, we only need to establish this connectivity once
                        if(a2b_id > na2b_id)
                            incidentItsTab.insert(vec2i{a2b_id,na2b_id});
                        return;
                    }
                }

                auto ta = zs::reinterpret_bits<int>(halfedges_A("to_face",ha));
                auto hb = zs::reinterpret_bits<int>(tris_B("he_inds",tb));
                for(int i = 0;i != 3;++i) {
                    if(auto no = B_2_A_tab.query(vec2i{hb,ta});no >= 0) {
                        auto nb2a_id = B_2_A_buffer[no];
                        nb2a_id += offset;
                        incidentItsTab.insert(vec2i{a2b_id,nb2a_id});
                        return;
                    }
                    hb = zs::reinterpret_bits<int>(halfedges_B("next_he",hb));
                }

                printf("impossible reaching here, the intersection ring seems to be broken\n");
        });

        pol(zs::range(nm_B_2_A_ints),[
            ints_buffer_B_2_A = proxy<space>({},ints_buffer_B_2_A),
            halfedges_B = proxy<space>({},halfedges_B),
            B_2_A_tab = proxy<space>(B_2_A_tab),
            A_2_B_tab = proxy<space>(A_2_B_tab),
            tris_A = proxy<space>({},tris_A),
            halfedges_A = proxy<space>({},halfedges_A),
            offset = A_2_B_buffer.size(),
            B_2_A_buffer = proxy<space>(B_2_A_buffer),
            incidentItsTab = proxy<space>(incidentItsTab)] ZS_LAMBDA(int b2a_id) mutable {
                auto b2a_pair = ints_buffer_B_2_A.pack(dim_c<2>,"pair",b2a_id,int_c);
                auto hb = b2a_pair[0];
                auto ta = b2a_pair[1];
                auto ohb = zs::reinterpret_bits<int>(halfedges_B("opposite_he",hb));
                if(ohb >= 0) {
                    if(auto no = B_2_A_tab.query(vec2i{ohb,ta});no >= 0) {
                        auto nb2a_id = B_2_A_buffer[no];
                        if(b2a_id > nb2a_id)
                            incidentItsTab.insert(vec2i{b2a_id + offset,nb2a_id + offset});
                    }else {
                        printf("impossible reaching here, the hi and ohi should both have been inserted\n");
                    }
                }

                auto nhb = hb;
                for(int i = 0;i != 2;++i) {
                    nhb = zs::reinterpret_bits<int>(halfedges_B("next_he",nhb));
                    if(auto no = B_2_A_tab.query(vec2i{nhb,ta});no >= 0) {
                        auto nb2a_id = B_2_A_buffer[no];
                        if(b2a_id > nb2a_id)
                            incidentItsTab.insert(vec2i{b2a_id + offset,nb2a_id + offset});
                        return;
                    }
                }

                // just for testing
                auto tb = zs::reinterpret_bits<int>(halfedges_B("to_face",hb));
                auto ha = zs::reinterpret_bits<int>(tris_A("he_inds",ta));
                for(int i = 0;i != 3;++i) {
                    if(auto no = A_2_B_tab.query(vec2i{ha,tb});no >= 0)
                        return;
                    ha = zs::reinterpret_bits<int>(halfedges_A("next_he",ha));
                }
                printf("impossible reaching here, the intersection ring seems to be broken\n");
        });

        auto nmEmtries = incidentItsTab.size();

        zs::Vector<zs::vec<int,2>> conn_topo{gia_res.get_allocator(),nmEmtries};
        pol(zip(conn_topo,range(incidentItsTab._activeKeys)),[] ZS_LAMBDA(zs::vec<int,2> &ij,const auto& key) mutable {ij = key;});
        zs::Vector<int> ringTag{gia_res.get_allocator(),total_nm_ints};
        auto nm_rings = mark_disconnected_island(pol,conn_topo,ringTag);


        std::cout << "nmEmtris of incidentItsTab : " << nmEmtries << std::endl;
        std::cout << "nmRings = " << nm_rings << std::endl;
        // return 0;

        // if(ringTag.size() == 0)
        //     return 0;

        zs::Vector<int> ringSize(gia_res.get_allocator(),nm_rings);
        pol(zs::range(ringSize.size()),[
            ringSize = proxy<space>(ringSize)] ZS_LAMBDA(int ri) mutable {
                ringSize[ri] = 0;
        });
        pol(zs::range(ringTag.size()),[
            ringTag = proxy<space>(ringTag),
            exec_tag,
            ringSize = proxy<space>(ringSize)] ZS_LAMBDA(int isi) mutable {
                atomic_add(exec_tag,&ringSize[ringTag[isi]],(int)1);
        });        



        zs::Vector<vec2i> edges_A_topos{tris_A.get_allocator(),tris_A.size() * 3};
        zs::Vector<vec2i> edges_B_topos{tris_B.get_allocator(),tris_B.size() * 3};
        auto tris_2_edges_topo = [&](const TriTileVec& tris,zs::Vector<vec2i>& edge_topos) mutable {
            pol(zs::range(tris.size()),[
                tris = proxy<space>({},tris),
                edge_topos = proxy<space>(edge_topos)] ZS_LAMBDA(int ti) mutable {
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    for(int i = 0;i != 3;++i) {
                        if(tri[i] < tri[(i + 1) % 3])
                            edge_topos[ti * 3 + i] = vec2i{tri[i],tri[(i + 1) % 3]};
                        else
                            edge_topos[ti * 3 + i] = vec2i{-1,-1};
                    }
            });
        };
        tris_2_edges_topo(tris_A,edges_A_topos);
        tris_2_edges_topo(tris_B,edges_B_topos);

        dtiles_t gia_res_A{verts_A.get_allocator(),{
            {"ring_mask",1}
        },verts_A.size()};
        TILEVEC_OPS::fill(pol,gia_res_A,"ring_mask",zs::reinterpret_bits<T>((int)0));
        dtiles_t gia_res_B{verts_B.get_allocator(),{
            {"ring_mask",1}
        },verts_B.size()};
        TILEVEC_OPS::fill(pol,gia_res_B,"ring_mask",zs::reinterpret_bits<T>((int)0));

        if(nm_rings >= 32) {
            std::cout << "the nm_rings[" << nm_rings << "] exceeds the max_nm_rings[" << 32 << "]" << std::endl;
        }

        nm_rings = nm_rings >= 32 ? 31 : nm_rings;
        // return 0;

        for(int ri = 0;ri != nm_rings;++ri) {
            std::cout << "ring size[" << ri << "] : " << ringSize.getVal(ri) << std::endl;
            std::cout << "size of ringTag " << ringTag.size() << std::endl;
            std::cout << "total_nm_ints" << total_nm_ints << std::endl;
            std::cout << "nm_A_2_B : " << nm_A_2_B_ints << std::endl;
            std::cout << "nm_B_2_A : " << nm_B_2_A_ints << std::endl;
            auto mark_gia_tag = [&](const dtiles_t& ints_buffer,
                int ints_size,
                int ints_offset,
                const TriTileVec& tris,
                const PosTileVec& verts,
                // const zs::Vector<int>& island_buffer,
                const zs::Vector<vec2i>& edges_topo,
                const HalfEdgeTileVec& halfedges,
                GIA_TILEVEC& the_gia_res) mutable {
                    zs::Vector<int> island_buffer{the_gia_res.get_allocator(),the_gia_res.size()};
                    table_vec2i_type disable_lines{ints_buffer.get_allocator(),ringSize.getVal(ri) * 6};
                    disable_lines.reset(pol,true);
                    table_int_type disable_points{ints_buffer.get_allocator(),ringSize.getVal(ri) * 8};
                    disable_points.reset(pol,true);
                    // build disable_lines
                    pol(zs::range(ints_size),[
                        ints_buffer = proxy<space>({},ints_buffer),
                        ringTag = proxy<space>(ringTag),
                        ringTagSize = ringTag.size(),
                        intsBufferSize = ints_size,
                        ri = ri,
                        disable_lines = proxy<space>(disable_lines),
                        tris = proxy<space>({},tris),
                        offset = ints_offset,
                        halfedges = proxy<space>({},halfedges)] ZS_LAMBDA(int iti) mutable {
                            // if(iti + offset >= ringTagSize) {
                            //     printf("its + offset(%d) >= ringTag.size()(%d)\n",iti + offset,(int)ringTagSize);
                            //     return;
                            // }
                            if(ringTag[iti + offset] != ri)
                                return;
                            // if(iti >= intsBufferSize) {
                            //     printf("its(%d) >= ints_buffer.size()(%d)\n",iti,(int)intsBufferSize);
                            //     return;
                            // }
                            auto pair = ints_buffer.pack(dim_c<2>,"pair",iti,int_c);
                            auto hi = pair[0];
                            auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                            auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);
                            auto local_vidx = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
                            auto a = htri[local_vidx];
                            auto b = htri[(local_vidx + 1) % 3];
                            if(a > b) {
                                auto tmp = a;
                                a = b;
                                b = tmp;
                            }
                            disable_lines.insert(zs::vec<int,2>{a,b});                            
                    });

                    // return;

                    auto nm_islands = mark_disconnected_island(pol,edges_topo,disable_points,disable_lines,island_buffer);
                    std::cout << "nm_islands : " << nm_islands << std::endl;
                    if(nm_islands == 1)
                        return;

                    int valid_island_idx = 0;

                    zs::Vector<int> nm_cmps_every_island_count{the_gia_res.get_allocator(),(size_t)nm_islands};
                    // pol(zs::range(nm_cmps_every_island_count),[](auto& island_size) {island_size = 0;});
                    pol(zs::range(nm_cmps_every_island_count.size()),
                        [nm_cmps_every_island_count = proxy<space>(nm_cmps_every_island_count)] ZS_LAMBDA(int ci) mutable {
                            nm_cmps_every_island_count[ci] = 0;
                    });
                    // return;

                    // find the island with the most must excluded points or the largest
                    pol(zs::range(the_gia_res.size()),[
                        island_buffer = proxy<space>(island_buffer),
                        verts = proxy<space>({},verts),
                        nm_cmps_every_island_count = proxy<space>(nm_cmps_every_island_count)] ZS_LAMBDA(int vi) mutable {
                            auto island_idx = island_buffer[vi];
                            if(verts.hasProperty("GIA_must_exclude")){
                                if(verts("GIA_must_exclude",vi) < (T)0.5)
                                    return;
                            }
                            atomic_add(exec_cuda,&nm_cmps_every_island_count[island_idx],(int)1);
                    });


                
                    for(int i = 0;i != nm_islands;++i)
                        std::cout << nm_cmps_every_island_count.getVal(i) << "\t";
                    std::cout << std::endl;

                    // // return;
                    // if(nm_islands == 1)

                    int max_size = 0;

                    for(int i = 0;i != nm_islands;++i) {
                        auto size_of_island = nm_cmps_every_island_count.getVal(i);
                        if(size_of_island > max_size){
                            max_size = size_of_island;
                            valid_island_idx = i;
                        }
                    }
                    
                    auto cur_ri_mask = (int)1 << ri;

                    pol(zs::range(the_gia_res.size()),[
                        the_gia_res = proxy<space>({},the_gia_res),
                        nm_islands,
                        cur_ri_mask,
                        exec_tag,
                        valid_island_idx = valid_island_idx,
                        island_buffer = proxy<space>(island_buffer)] ZS_LAMBDA(int vi) mutable {
                            auto island_idx = island_buffer[vi];
                            auto ring_mask = zs::reinterpret_bits<int>(the_gia_res("ring_mask",vi));
                            if(island_idx != valid_island_idx/* || ints_types[island_idx] == 1*/){
                                ring_mask |= cur_ri_mask;
                            }
                            the_gia_res("ring_mask",vi) = zs::reinterpret_bits<T>(ring_mask);
                    });
            };
            mark_gia_tag(ints_buffer_A_2_B,nm_A_2_B_ints,0,tris_A,verts_A,edges_A_topos,halfedges_A,gia_res_A);
            mark_gia_tag(ints_buffer_B_2_A,nm_B_2_A_ints,nm_A_2_B_ints,tris_B,verts_B,edges_B_topos,halfedges_B,gia_res_B);
        }

        // a & b

        TILEVEC_OPS::copy(pol,gia_res_A,"ring_mask",gia_res,"ring_mask",0);
        TILEVEC_OPS::copy(pol,gia_res_B,"ring_mask",gia_res,"ring_mask",gia_res_A.size());

        return nm_rings;
}

};