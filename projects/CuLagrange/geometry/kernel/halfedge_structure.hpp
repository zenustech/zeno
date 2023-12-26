#pragma once

#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "tiled_vector_ops.hpp"
#include "zensim/container/Bht.hpp"

namespace zeno {

    template<typename Pol,typename TetTileVec,typename HalfFacetTileVec>
    bool build_tetrahedra_half_facet(Pol& pol,
            TetTileVec& tets,
            HalfFacetTileVec& halfFacet) {
        using namespace zs;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;
        using T = typename TetTileVec::value_type;

        constexpr auto space = Pol::exec_tag::value;
        halfFacet.resize(tets.size() * 4);

        TILEVEC_OPS::fill(pol,halfFacet,"opposite_hf",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(pol,halfFacet,"next_hf",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(pol,halfFacet,"to_tet",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(pol,halfFacet,"local_idx",reinterpret_bits<T>((int)-1));

        bcht<vec3i,int,true,universal_hash<vec3i>,32> hftab{tets.get_allocator(),tets.size() * 4};

        zs::vec<int,3 * 4> facet_indices{
            1,3,2,
            0,2,3,
            0,3,1,
            0,1,2
        };
        pol(zs::range(tets.size()),[
            tets = proxy<space>({},tets),
            halfFacet = proxy<space>({},halfFacet),
            facet_indices = facet_indices,
            hftab = proxy<space>(hftab)] ZS_LAMBDA(int ti) mutable {
                auto tet = tets.pack(dim_c<4>,"inds",ti,int_c);
                vec4i hinds{};
                for(int i = 0;i != 4;++i) {
                    auto facet = vec3i{
                        tet[facet_indices[i * 3 + 0]],
                        tet[facet_indices[i * 3 + 1]],
                        tet[facet_indices[i * 3 + 2]]};
                    int min_idx = 0;
                    int min_id = facet[i];
                    for(int j = 1;j != 3;++j)
                        if(facet[j] < min_id){
                            min_idx = j;
                            min_id = facet[j];
                        }
                    for(int j = 0;j != min_idx;++j) {
                        auto tmp = facet[0];
                        facet[0] = facet[1];
                        facet[1] = facet[2];
                        facet[2] = tmp;
                    }

                    if(hinds[i] = hftab.insert(facet);hinds[i] >= 0){
                        halfFacet("to_tet",hinds[i]) = zs::reinterpret_bits<T>(ti);
                        halfFacet("local_idx",hinds[i]) = zs::reinterpret_bits<T>(i);
                    }

                    if(i == 0) {
                        tets("hf_inds",ti) = zs::reinterpret_bits<T>((int)hinds[i]);
                    }
                }
                for(int i = 0;i != 4;++i)
                    halfFacet("next_hf",hinds[i]) = zs::reinterpret_bits<T>((int)hinds[(i + 1) % 3]);
        });

        pol(zs::range(halfFacet.size()),[
            tets = proxy<space>({},tets),
            halfFacet = proxy<space>({},halfFacet),
            facet_indices = facet_indices,
            hftab = proxy<space>(hftab)] ZS_LAMBDA(int hi) mutable {
                auto ti = zs::reinterpret_bits<int>(halfFacet("to_tet",hi));
                auto tet = tets.pack(dim_c<4>,"inds",ti,int_c);
                auto local_idx = zs::reinterpret_bits<int>(halfFacet("local_idx",hi));
                auto facet = vec3i{
                        tet[facet_indices[local_idx * 3 + 0]],
                        tet[facet_indices[local_idx * 3 + 1]],
                        tet[facet_indices[local_idx * 3 + 2]]};
                int min_idx = 0;
                int min_id = facet[0];
                for(int i = 1;i != 3;++i)
                    if(facet[i] < min_id){
                        min_idx = i;
                        min_id = facet[i];
                    }
                for(int i = 0;i != min_idx;++i) {
                    auto tmp = facet[0];
                    facet[0] = facet[1];
                    facet[1] = facet[2];
                    facet[2] = tmp;
                }
                auto tmp = facet[1];
                facet[1] = facet[2];
                facet[2] = tmp;
                if(auto no = hftab.query(facet);no >= 0) {
                    halfFacet("opposite_hf",hi) = zs::reinterpret_bits<T>((int)no);
                }
        });

        return true;
    }

    // the input mesh should be a manifold
    template<typename Pol,typename TriTileVec,typename PointTileVec,typename HalfEdgeTileVec>
    bool build_half_edge_structure_for_triangle_mesh(Pol& pol,
            TriTileVec& tris,
            // EdgeTileVec& lines,
            PointTileVec& points,
            HalfEdgeTileVec& halfEdge) {
        using namespace zs;
        using vec2i = zs::vec<int, 2>;
		using vec3i = zs::vec<int, 3>;
        using T = typename TriTileVec::value_type;

        constexpr auto space = Pol::exec_tag::value;


        halfEdge.resize(tris.size() * 3);

        TILEVEC_OPS::fill(pol,halfEdge,"local_vertex_id",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(pol,halfEdge,"to_face",reinterpret_bits<T>((int)-1));
        // TILEVEC_OPS::fill(pol,halfEdge,"to_edge",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(pol,halfEdge,"opposite_he",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(pol,halfEdge,"next_he",reinterpret_bits<T>((int)-1));      
        // we might also need a space hash structure here, map from [i1,i2]->[ej]

        // surface tri edges' indexing the halfedge list
        bcht<vec2i,int,true,universal_hash<vec2i>,32> hetab{tris.get_allocator(),tris.size() * 3};
        // Vector<int> sfi{tris.get_allocator(),tris.size() * 3};

        // surface points' indexing one of the connected half-edge
        bcht<int,int,true,universal_hash<int>,32> ptab{points.get_allocator(),points.size()};
        Vector<int> spi{points.get_allocator(),points.size()};
        pol(range(points.size()),
            [ptab = proxy<space>(ptab),points = proxy<space>({},points),spi = proxy<space>(spi)] ZS_LAMBDA(int pi) mutable {
                auto pidx = reinterpret_bits<int>(points("inds",pi));
                if(int no = ptab.insert(pidx);no >= 0)
                    spi[no] = pi;
                else{
                    printf("same point [%d] has been inserted twice\n",pidx);
                }
        });

        pol(range(tris.size()),
            [hetab = proxy<space>(hetab),
                points = proxy<space>({},points),
                ptab = proxy<space>(ptab),
                spi = proxy<space>(spi),
                halfEdge = proxy<space>({},halfEdge),
                tris = proxy<space>({},tris)] ZS_LAMBDA(int ti) mutable {
                    auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                    vec3i hinds{};
                    for(int i = 0;i != 3;++i){
                        if(hinds[i] = hetab.insert(vec2i{tri[i],tri[(i+1)%3]});hinds[i] >= 0){
                            int no = hinds[i];
                            auto pno = spi[ptab.query(tri[i])];
                            halfEdge("local_vertex_id",no) = reinterpret_bits<T>((int)i);
                            points("he_inds",pno) = reinterpret_bits<T>(no);

                            // if(tri[i] < tri[(i+1)%3]){
                            auto a = tri[i];
                            auto b = tri[(i+1)%3];
                            if(a > b){
                                auto tmp = a;
                                a = b;
                                b = tmp;
                            }

                            halfEdge("to_face",no) = reinterpret_bits<T>(ti);
                            if(i == 0)
                                tris("he_inds",ti) = reinterpret_bits<T>(no);
                        }else {
                            auto no = hinds[i];
                            int hid = hetab.query(vec2i{tri[i],tri[(i+1)%3]});
                            int ori_ti = reinterpret_bits<int>(halfEdge("to_face",hid));
                            auto ori_tri = tris.pack(dim_c<3>,"inds",ori_ti,int_c);
                            printf("the same directed edge <%d %d> has been inserted twice! original hetab[%d], cur: %d <%d %d %d> ori: %d <%d %d %d>\n",
                                tri[i],tri[(i+1)%3],hid,ti,tri[0],tri[1],tri[2],ori_ti,ori_tri[0],ori_tri[1],ori_tri[2]);
                        }
                    }
                    for(int i = 0;i != 3;++i)
                        halfEdge("next_he",hinds[i]) = reinterpret_bits<T>((int)hinds[(i+1) % 3]);
        });

        pol(range(halfEdge.size()),
            [halfEdge = proxy<space>({},halfEdge),hetab = proxy<space>(hetab),tris = proxy<space>({},tris)] ZS_LAMBDA(int hi) mutable {
                auto ti = zs::reinterpret_bits<int>(halfEdge("to_face",hi));
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                auto local_vidx = reinterpret_bits<int>(halfEdge("local_vertex_id",hi));
                auto a = tri[local_vidx];
                // auto nxtHalfEdgeIdx = reinterpret_bits<int>(halfEdge("next_he",hi));
                auto b = tri[(local_vidx + 1) % 3];
                auto key = vec2i{b,a};

                // printf("half_edge[%d] = [%d %d]\n",hi,a,b);

                if(int hno = hetab.query(key);hno >= 0) {
                    halfEdge("opposite_he",hi) = reinterpret_bits<T>(hno);
                }else {
                    halfEdge("opposite_he",hi) = reinterpret_bits<T>((int)-1);
                }
                
        });

        return true;
    }


    // the input mesh should be a manifold
    template<typename Pol,typename TriTileVec,typename HalfEdgeTileVec>
    bool build_half_edge_structure_for_triangle_mesh_robust(Pol& pol,
            TriTileVec& tris,
            HalfEdgeTileVec& halfEdge) {
        using namespace zs;
        using vec2i = zs::vec<int, 2>;
		using vec3i = zs::vec<int, 3>;
        using T = typename TriTileVec::value_type;

        constexpr auto space = Pol::exec_tag::value;
        auto exec_tag = wrapv<space>{};

        halfEdge.resize(tris.size() * 3);

        TILEVEC_OPS::fill(pol,halfEdge,"local_vertex_id",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(pol,halfEdge,"to_face",reinterpret_bits<T>((int)-1));
        // TILEVEC_OPS::fill(pol,halfEdge,"to_edge",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(pol,halfEdge,"opposite_he",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(pol,halfEdge,"next_he",reinterpret_bits<T>((int)-1));      
        // we might also need a space hash structure here, map from [i1,i2]->[ej]

        // surface tri edges' indexing the halfedge list
        bcht<vec3i,int,true,universal_hash<vec3i>,32> heTritab{tris.get_allocator(),tris.size() * 3};
        bcht<vec2i,int,true,universal_hash<vec2i>,32> hetab{tris.get_allocator(),tris.size() * 3};

        // zs::Vector<int> he_inds_buffer{tris.get_allocator(),tris.size() * 3};
        zs::Vector<bool> he_is_manifold{tris.get_allocator(),tris.size() * 3};
        pol(zs::range(he_is_manifold.size()),[
            he_is_manifold = proxy<space>(he_is_manifold)] ZS_LAMBDA(int id) mutable {
                he_is_manifold[id] = true;
        });

        pol(range(tris.size()),
            [heTritab = proxy<space>(heTritab),
                halfEdge = proxy<space>({},halfEdge),
                tris = proxy<space>({},tris)] ZS_LAMBDA(int ti) mutable {
                    auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                    vec3i hinds{};
                    for(int i = 0;i != 3;++i){
                        if(hinds[i] = heTritab.insert(vec3i{tri[i],tri[(i+1)%3],ti});hinds[i] >= 0){
                            int no = hinds[i];
                            halfEdge("local_vertex_id",no) = reinterpret_bits<T>((int)i);
                            halfEdge("to_face",no) = reinterpret_bits<T>(ti);
                            if(i == 0)
                                tris("he_inds",ti) = reinterpret_bits<T>(no);
                        }else {
                            auto no = hinds[i];
                            int hid = heTritab.query(vec3i{tri[i],tri[(i+1)%3],ti});
                            int ori_ti = reinterpret_bits<int>(halfEdge("to_face",hid));
                            auto ori_tri = tris.pack(dim_c<3>,"inds",ori_ti,int_c);
                            printf("the same directed edge <%d %d %d> has been inserted twice! original heTritab[%d], cur: %d <%d %d %d> ori: %d <%d %d %d>\n",
                                tri[i],tri[(i+1)%3],ti,hid,ti,tri[0],tri[1],tri[2],ori_ti,ori_tri[0],ori_tri[1],ori_tri[2]);
                        }
                    }
                    for(int i = 0;i != 3;++i)
                        halfEdge("next_he",hinds[i]) = reinterpret_bits<T>((int)hinds[(i+1) % 3]);
        });

        zs::Vector<int> he_inds_buffer{tris.get_allocator(),tris.size() * 3};
        pol(zip(range(heTritab.size()),heTritab._activeKeys),
            [hetab = proxy<space>(hetab),
                he_inds_buffer = proxy<space>(he_inds_buffer),
                tris = proxy<space>({},tris)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto ti = pair[2];
                auto edge = vec2i{pair[0],pair[1]};
                if(auto no = hetab.insert(edge);no >= 0) {
                    he_inds_buffer[no] = id;
                }
        });

        pol(zip(range(heTritab.size()),heTritab._activeKeys),
            [hetab = proxy<space>(hetab),
                he_is_manifold = proxy<space>(he_is_manifold),
                he_inds_buffer = proxy<space>(he_inds_buffer),
                tris = proxy<space>({},tris)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto ti = pair[2];
                auto edge = vec2i{pair[0],pair[1]};
                
                auto no = hetab.query(edge);
                auto id_should_be = he_inds_buffer[no];
                if(id != id_should_be) {
                    he_is_manifold[id] = false;
                    he_is_manifold[id_should_be] = false;
                    auto oppo_no = hetab.query(vec2i{edge[1],edge[0]});
                    if(oppo_no >= 0) {
                        auto oppo_id = he_inds_buffer[oppo_no];
                        he_is_manifold[oppo_id] = false;
                    }
                }
        });

        zs::Vector<int> nm_non_manifold_edges{tris.get_allocator(),1};
        nm_non_manifold_edges.setVal(0);

        pol(range(halfEdge.size()),
            [halfEdge = proxy<space>({},halfEdge),
                    exec_tag = exec_tag,
                    nm_non_manifold_edges = proxy<space>(nm_non_manifold_edges),
                    he_inds_buffer = proxy<space>(he_inds_buffer),
                    he_is_manifold = proxy<space>(he_is_manifold),
                    hetab = proxy<space>(hetab),
                    tris = proxy<space>({},tris)] ZS_LAMBDA(int hi) mutable {
                if(!he_is_manifold[hi]) {
                    atomic_add(exec_tag,&nm_non_manifold_edges[0],1);
                    return;
                }

                auto ti = zs::reinterpret_bits<int>(halfEdge("to_face",hi));
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                auto local_vidx = reinterpret_bits<int>(halfEdge("local_vertex_id",hi));
                
                auto key = vec2i{tri[(local_vidx + 1) % 3],tri[local_vidx]};

                if(int hno = hetab.query(key);hno >= 0) {
                    auto oppo_hi = he_inds_buffer[hno];
                    halfEdge("opposite_he",hi) = reinterpret_bits<T>(oppo_hi);
                }
        });

        // std::cout << "number non_manifold edges detected : " << nm_non_manifold_edges.getVal(0) << std::endl;

        return true;
    }


    template<typename HalfEdgeTileVec>
    constexpr int get_next_half_edge(int hei,const HalfEdgeTileVec& half_edges,int step = 1,bool reverse = false) {
        using namespace zs;
        for(int i = 0;i != step;++i)
            hei = reinterpret_bits<int>(half_edges("next_he",hei));
        if(reverse)
            hei = reinterpret_bits<int>(half_edges("opposite_he",hei));
        return hei;
    }

    template<typename HalfEdgeTileVec,typename TriTileVec>
    constexpr int half_edge_get_another_vertex(int hei,const HalfEdgeTileVec& half_edges,const TriTileVec& tris) {
        using namespace zs;
        // hei = reinterpret_bits<int>(half_edges("next_he",hei));
        hei = get_next_half_edge(hei,half_edges,1,false);
        auto ti = zs::reinterpret_bits<int>(half_edges("to_face",hei));
        auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
        return tri[reinterpret_bits<int>(half_edges("local_vertex_id",hei))];
    }

    template<typename HalfEdgeTileVec,typename TriTileVec>
    constexpr zs::vec<int,2> half_edge_get_edge(int hei,const HalfEdgeTileVec& half_edges,const TriTileVec& tris) {
        using namespace zs;
        auto ti = zs::reinterpret_bits<int>(half_edges("to_face",hei));
        auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
        auto local_vertex_id = reinterpret_bits<int>(half_edges("local_vertex_id",hei));
        return zs::vec<int,2>{tri[local_vertex_id],tri[(local_vertex_id + 1) % 3]};
    }

    // some operation with half edge structure
    template<int MAX_NEIGHS,typename HalfEdgeTileVec,typename TriTileVec>
    constexpr zs::vec<int,MAX_NEIGHS> get_one_ring_neigh_points(int hei,const HalfEdgeTileVec& half_edges,const TriTileVec& tris) {
        using namespace zs;
        auto res = zs::vec<int,MAX_NEIGHS>::uniform(-1);
        auto hei0 = hei;
        int i = 0;
        // res[0] = half_edge_get_another_vertex(hei,half_edges);
        for(i = 0;i != MAX_NEIGHS;++i) {
            res[i] = half_edge_get_another_vertex(hei,half_edges,tris);
            auto nhei = get_next_half_edge(hei,half_edges,2,true);
            if(nhei == hei0)
                break;
            if(nhei < 0 && (i+1) < MAX_NEIGHS) {
                nhei = get_next_half_edge(hei,half_edges,2,false);
                if(nhei > 0){
                    auto ti = zs::reinterpret_bits<int>(half_edges("to_face",nhei));
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    res[i + 1] = tri[reinterpret_bits<int>(half_edges("local_vertex_id",nhei))];
                    break;
                }
            }
            hei = nhei;
        }
        if(i == MAX_NEIGHS)
            printf("the max_one_ring_neighbor limit exceeds");

        return res;
    }

    template<int MAX_NEIGHS,typename HalfEdgeTileVec>
    constexpr zs::vec<int,MAX_NEIGHS> get_one_ring_neigh_edges(int hei,const HalfEdgeTileVec& half_edges) {
        using namespace zs;
        auto res = zs::vec<int,MAX_NEIGHS>::uniform(-1);
        auto hei0 = hei;
        auto nhei = hei;
        int i = 0;
        for(i = 0;i != MAX_NEIGHS;++i) {
            res[i] = reinterpret_bits<int>(half_edges("to_edge",hei));
            nhei = get_next_half_edge(hei,half_edges,2,true);
            if(hei0 == nhei || nhei == -1)
                break;
            hei = nhei;
        }
        if(i < MAX_NEIGHS-1 && nhei == -1) {
            ++i;
            hei = get_next_half_edge(hei,half_edges,2,false);
            res[i] = reinterpret_bits<int>(half_edges("to_edge",hei));
        }
        return res;
    }

    template<int MAX_NEIGHS,typename HalfEdgeTileVec>
    constexpr zs::vec<int,MAX_NEIGHS> get_one_ring_neigh_tris(int hei,const HalfEdgeTileVec& half_edges) {
        using namespace zs;
        auto res = zs::vec<int,MAX_NEIGHS>::uniform(-1);
        auto hei0 = hei;
        int i = 0;
        res[0] = reinterpret_bits<int>(half_edges("to_face",hei));
        for(int i = 1;i != MAX_NEIGHS;++i) {
            hei = get_next_half_edge(hei,half_edges,1,true);
            if(hei == hei0 || hei < 0)
                break;
            res[i] = reinterpret_bits<int>(half_edges("to_face",hei));
        }

        if(i == MAX_NEIGHS)
            printf("the max_one_ring_neighbor limit exceeds");

        return res;

    }

};