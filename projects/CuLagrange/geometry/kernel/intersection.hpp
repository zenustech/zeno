#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/Bvh.hpp"
#include "Utils.hpp"
#include "compute_characteristic_length.hpp"

#include <iostream>
#include "tiled_vector_ops.hpp"

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

// structure of intersection Buffer output
// assuming there is no geometric or combitorial coincidence
// #tpairs : vec2i indexing the two indices of intersecting triangles
// #type : the type of intersection for triangles A and B
/*              (0) the intersection point is inside A
                (1) the intersection point is on the edge of triangle A
                (2) the intersection point is on the vertex of triangle A
                (3) the there is more than one intersection and is coplanar
*/
// #ID : depends on the type of intersection, return the vertex\edge index
/*              (0) return the local ID of the intersecting edge  of B
                (1) return the local ID of the intersecting edge  of A
                (2) return the local ID of the intersecting point of A
                (3) skip this sort of coplanar intersection
*/
// #ip : the position of intersection point
// return number of intersection pairs

// template<typename Vector3d>
// constexpr bool is_triangle_segment_intersect(const Vector3d tvs[3],const Vector3d svs[2]) {
//     return true;
// }

// calculate the surface normal of the two tribuffer before apply this function
template<typename Pol,typename PosTileVec,typename TriTileVec,typename IntersectionBuffer>
int triangulate_mesh_intersection(Pol& pol,
        const PosTileVec& verts_A,
        const zs::SmallString& xTagA,
        const TriTileVec& tris_A,
        const PosTileVec& verts_B,
        const zs::SmallString& xTagB,
        const TriTileVec& tris_B,
        IntersectionBuffer& intersects,
        const zs::SmallString& tpairsTag,
        const zs::SmallString& typeTag,
        const zs::SmallString& IDTag,
        const zs::SmallString& ipTag,
        bool update_normal = false) {
    using namespace zs;
    using vec2i = zs::vec<int,2>;
    using T = typename PosTileVec::value_type;
    using vec3 = zs::vec<T,3>;
    using bv_t = typename ZenoParticles::lbvh_t::Box;

    constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
    
    zs::Vector<int> nmIts{verts_A.get_allocator(),1};
    nmIts.setVal(0);

    // check the intersection channel
    if(!intersects.hasProperty(tpairsTag) || intersects.getPropertySize(tpairsTag) != 2){
        fmt::print(fg(fmt::color::red),"triangulate_mesh_intersection::intersects buffer contains no valid tpairsTag::{}\n",tpairsTag);
        return false;
    }
    if(!intersects.hasProperty(typeTag) || intersects.getPropertySize(typeTag) != 1){
        fmt::print(fg(fmt::color::red),"triangulate_mesh_intersection::intersects buffer contains no valid typeTag::{}\n",typeTag);
        return false;
    }
    if(!intersects.hasProperty(IDTag) || intersects.getPropertySize(IDTag) != 1){
        fmt::print(fg(fmt::color::red),"triangulate_mesh_intersection::intersects buffer contains no valid IDTag::{}\n",IDTag);
        return false;
    }
    if(!intersects.hasProperty(ipTag) || intersects.getPropertySize(ipTag) != 3){
        fmt::print(fg(fmt::color::red),"triangulate_mesh_intersection::intersects buffer contains no valid ipTag::{}\n",ipTag);
        return false;
    }

    auto bvsA = retrieve_bounding_volumes(pol,verts_A,tris_A,wrapv<3>{},0,xTagA);
    auto triABvh = LBvh<3, int,T>{};
    triABvh.build(pol,bvsA);

    auto cnormA = compute_average_edge_length(pol,verts_A,xTagA,tris_A);
    auto cnormB = compute_average_edge_length(pol,verts_B,xTagB,tris_B);

    auto cnorm = cnormA > cnormB ? cnormA : cnormB;
    cnorm *= 2;

    // retrieve all the intersection pairs
    pol(zs::range(tris_B.size()),[
            exec_tag = wrapv<space>{},
            nmIts = proxy<space>(nmIts),
            verts_A = proxy<space>({},verts_A),
            verts_B = proxy<space>({},verts_B),
            tris_A = proxy<space>({},tris_A),
            tris_B = proxy<space>({},tris_B),
            triABvh = proxy<space>(triABvh),
            intBuffer = proxy<space>({},intersects),
            xTagA = xTagA,
            xTagB = xTagB,
            tpairsTag = tpairsTag,
            typeTag = typeTag,
            IDTag = IDTag,
            ipTag = ipTag,
            thickness = cnorm] ZS_LAMBDA(int ti) mutable {
        auto triBCenter = vec3::zeros();
        auto triB = tris_B.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
        vec3 vB[3] = {};
        vec3 vA[3] = {};
        for(int i = 0;i != 3;++i)
            vB[i] = verts_B.pack(dim_c<3>,xTagB,triB[i]);
        for(int i = 0;i != 3;++i)
            triBCenter += vB[i] / (T)3.0; 
        auto bv = bv_t{get_bounding_box(triBCenter - thickness,triBCenter + thickness)};
        auto nrmB = tris_B.pack(dim_c<3>,"nrm",ti);

        auto process_potential_intersection_pairs = [&](int triA_idx) {
            auto triA = tris_A.pack(dim_c<3>,"inds",triA_idx).reinterpret_bits(int_c);
            auto nrmA = tris_A.pack(dim_c<3>,"nrm",triA_idx).reinterpret_bits(int_c);
            for(int i = 0;i != 3;++i)
                vA[i] = verts_A.pack(dim_c<3>,xTagA,triA[i]);
            auto r = std::numeric_limits<T>::infinity();
            vec3 p{};
            int edge_idx = 0;
            for(edge_idx = 0;edge_idx != 3;++edge_idx) {
                auto ro = vB[edge_idx];
                auto kv = vB[(edge_idx + 1) % 3] - vB[edge_idx];
                auto kn = kv.norm();
                auto rd = kv / ((T)kn + (T)1e-8);

                bool align = rd.dot(nrmA) > (T)0.0;
                if(!align)
                    continue;
                auto r = LSL_GEO::tri_ray_intersect(ro,rd,vA[0],vA[1],vA[2]);
                if(r < std::numeric_limits<T>::infinity()){
                    p = ro + r * rd;
                    break;
                }
            }
            // type(0) intersection
            if(r < (vB[(edge_idx + 1) % 3] - vB[edge_idx]).norm()) { 
                auto intID = atomic_add(exec_tag,&nmIts[0],1);
                intBuffer.pack(dim_c<2>,tpairsTag,intID) = vec2i(triA_idx,ti).reinterpret_bits(float_c);
                intBuffer(typeTag,intID) = reinterpret_bits<T>((int)0);
                intBuffer(IDTag,intID) = reinterpret_bits<T>((int)edge_idx);
                intBuffer.tuple(dim_c<3>,ipTag,intID) = p; 
            }

            for(edge_idx = 0;edge_idx != 3;++edge_idx) {
                auto ro = vA[edge_idx];
                auto kv = vA[(edge_idx + 1) % 3] - vA[edge_idx];
                auto kn = kv.norm();
                auto rd = kv / ((T)kn + (T)1e-8);

                bool align = rd.dot(nrmB);
            }
        };
    });
}

};