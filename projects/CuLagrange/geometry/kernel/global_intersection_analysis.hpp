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

    constexpr int GIA_BACKGROUND_COLOR = 0;
    constexpr int GIA_TURNING_POINTS_COLOR = 1;
    constexpr int GIA_RED_COLOR = 2;
    constexpr int GIA_WHITE_COLOR = 3;
    constexpr int GIA_BLACK_COLOR = 4;
    constexpr int DEFAULT_MAX_GIA_INTERSECTION_PAIR = 100000;
    constexpr int DEFAULT_MAX_NM_TURNING_POINTS = 100;
    constexpr int DEFAULT_MAX_BOUNDARY_POINTS = 100;

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
    template<typename Pol,typename PosTileVec,typename TriTileVec,typename HETileVec>
    void retrieve_self_intersection_tri_halfedge_pairs(Pol& pol,
        const PosTileVec& verts, const zs::SmallString& xtag,
        const TriTileVec& tris,
        const HETileVec& halfedges,
        zs::bht<int,2,int>& res) {
            using namespace zs;
            using vec2i = zs::vec<int,2>;
            using bv_t = typename ZenoParticles::lbvh_t::Box;
            using vec3 = zs::vec<T,3>;
            using table_vec2i_type = zs::bht<int,2,int>;

            constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
            constexpr auto exec_tag = wrapv<space>{};
            constexpr auto eps = 1e-6;

            auto tri_bvs = retrieve_bounding_volumes(pol,verts,tris,wrapv<3>{},0,xtag);
            auto tri_bvh = LBvh<3,int,T>{};
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

                    auto process_potential_he_tri_intersection_pairs = [&](int ti) mutable {
                        auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                        for(int i = 0;i != 3;++i)
                            if(tri[i] == hedge[0] || tri[i] == hedge[1])
                                return;
                        vec3 tvs[3] = {};
                        for(int i = 0;i != 3;++i)
                            tvs[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                        if(!et_intersected(hv0,hv1,tvs[0],tvs[1],tvs[2]))
                            return;
                        res.insert(vec2i{hei,ti});
                        if(ohei >= 0)
                            res.insert(vec2i{ohei,ti});
                    }; 

                    tri_bvh.iter_neighbors(bv,process_potential_he_tri_intersection_pairs);              
            });  
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
        
        LSL_GEO::intersectionBaryCentric(eps[0],eps[1],tps[0],tps[1],tps[2],edge_bary,tri_bary);
    }

    template<typename TriTileVecHost,
        typename HalfEdgeTileVecHost,
        typename HTHashMapHost>
    bool trace_intersection_loop(const TriTileVecHost& tris,
            const HalfEdgeTileVecHost& halfedges,
            const HTHashMapHost& csHT,
            const zs::vec<int,2>& trace_start,
            std::vector<int>& intersection_mask,
            std::vector<int>& loop_sides,
            std::vector<zs::vec<int,2>>& loop_buffers,
            std::vector<zs::vec<int,2>>& loop_buffers_0,
            std::vector<zs::vec<int,2>>& loop_buffers_1) {
        using namespace zs;
        using vec2i = zs::vec<int,2>; 
        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();          
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
        intersection_mask[id] = 1;

        if(find_intersection_turning_point(halfedges,tris,cur_trace) >= 0) {
            cur_trace[0] = zs::reinterpret_bits<int>(halfedges("opposite_he",cur_trace[0]));
            loop_buffers_0.push_back(cur_trace);
            loop_buffers.push_back(cur_trace);
            loop_sides.push_back(0);
            auto id = csHT.query(cur_trace);
            intersection_mask[id] = 1;
        }

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
                throw std::runtime_error("abruptly end while the cur_trace is neither a turning point nor a boundary point");
            }
            if(is_boundary_halfedge(halfedges,cur_trace[0])) {
                printf("find boundary halfedge %d\n",cur_trace[0]);
                break;
            }

            cur_trace[0] = zs::reinterpret_bits<int>(halfedges("opposite_he",cur_trace[0]));
            auto id = csHT.query(cur_trace);
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



    // the input intersection loop must be one of the three types : CLOSED ring, L-L ring, B-B ring
    // for 
    // template<typename PosTileVecProxy,
    //     typename TriTileVecProxy,
    //     typename HalfEdgeTileVecProxy>
    // int flood_fill_the_smallest_path_region(const PosTileVecProxy& verts,const zs::SmallString& xtag,
    //     const TriTileVecProxy& tris,
    //     const HalfEdgeTileVecProxy& halfedges,
    //     const std::vector<vec2i>& path,
    //     const zs::vec<int,2> turning_point,
    //     int coloring_color,
    //     int background_color,
    //     int turning_point_color,
    //     std::vector<int>& colors) {
    
    //     using namespace zs;

    //     constexpr auto omp_space = execspace_e::openmp;
    //     auto ompPol = omp_exec();

    //     bool found_smaller_region = false;

    //     std::fill(colors.begin(),colors.end(),background_color);
    //     for(int i = 0;i != 2;++i)
    //         if(turning_point[i] >= 0)
    //             colors[turning_point[i]] = turning_point_color;
    
    //     std::vector<int> minimum_distance{verts.size()};
    //     std::fill(minimum_distance.begin(),minimum_distance.end(),std::numeric_limits<float>::max());

    //     std::vector<int> halfedges_mask{halfedges.size(),0};
    //     // coloring sparsely
    //     ompPol(zs::range(path),[&halfedges_mask] (const auto& pair) mutable {halfedges_mask[pair[0]] = 1;});

    //     for(const auto& pair : path) {
    //         auto hi = pair[0];
    //         auto ti = pair[1];

    //         auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
    //         auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);
    //         auto local_vert_id = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
    //         vec2i hedge{};
    //         hedge[0] = htri[(local_vert_id + 0) % 3];
    //         hedge[1] = htri[(local_vert_id + 1) % 3];

    //         auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

    //         zs::vec<T,3> tps[3] = {};
    //         zs::vec<T,3> eps[2] = {};

    //         for(int i = 0;i != 3;++i)
    //             tps[i] = verts.pack(dim_c<3>,xtag,tri[i]);
    //         for(int i = 0;i != 2;++i)
    //             eps[i] = verts.pack(dim_c<3>,xtag,hedge[i]);

    //         auto tnrm = LSL_GEO::facet_normal(tps[0],tps[1],tps[2]);
    //         zs::vec<T,2> ep_dist{};
    //         auto edge_length = (eps[0] - eps[1]).norm();
    //         auto edge_project_length = (eps[0] - eps[1]).dot(tnrm);
    //         for(int i = 0;i != 2;++i)
    //             ep_dist[i] = zs::abs(tnrm.dot(eps[i] - tps[0]) / edge_project_length * edge_length);

    //         zs::vec<int,2> new_color{};
    //         if(tnrm.dot(eps[0] - tps[0]) > 0)
    //             new_color = zs::vec<int,2>{coloring_color,-coloring_color};
    //         else if(tnrm.dot(eps[1] - tps[0]) > 0)
    //             new_color = zs::vec<int,2>{-coloring_color,coloring_color};
    //         else {
    //             std::cout << "invalid intersection pair detected during flood-fill" << std::endl;
    //             throw std::runtime_error("invalid intersection pair detected during flood-fill");
    //         }

    //         for(int i = 0;i != 2;++i) {
    //             if(ep_dist[i] < minimum_distance[hedge[i]]) {
    //                 colors[hedge[i]] = new_color[i];
    //                 flooding_fill_vertices.insert(hedge[i]);
    //                 minimum_distance[hedge[i]] = ep_dist[i];
    //             }
    //         }
    //     }

    //     // initialize the flooding front
    //     std::deque<int> flooding_front_positive{};
    //     std::deque<int> flooding_front_negative{};
    //     for(const auto& pair : path) {
    //         auto ti = pair[1];
    //         auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
    //         auto thi = zs::reinterpret_bits<int>(("he_inds",ti));
    //         for(int i = 0;i != 3;++i) {
    //             if(halfedges_mask[thi] == 0) {
    //                 // finding a front
    //                 auto edge = half_edge_get_edge(thi,halfedges,tris);
    //                 for(int i = 0;i != 2;++i) {
    //                     if(colors[edge[i]] == coloring_color) {
    //                         flooding_front_positive.push_back(thi);
    //                          break;
    //                     }
    //                     else if(colors[edge[i]] == -coloring_color) {
    //                         flooding_front_negative.push_back(thi);
    //                         break;
    //                     }
    //                 }
    //                 halfedges_mask[thi] = 1;
    //             }
    //             thi = zs::reinterpret_bits<int>(halfedges("next_he",thi));
    //         }            
    //     }
    //     // do the coloring and move forward the front
    //     bool has_positive_front_update = false;
    //     bool has_negative_front_update = false;

    //     auto paint_and_update_the_front = [&](int the_use_color,std::deque<int>& the_front,bool& get_updated) {
    //         auto size_of_front = the_front.size();
    //         get_updated = false;
    //         for(int i = 0;i != size_of_front;++i) {
    //             // coloring the current halfedge
    //             auto hi = the_front.pop_front();
    //             auto edge = half_edge_get_edge(hi,halfedges,tris);
    //             for(int i = 0;i != 2;++i)
    //                 if(colors[i] == background_color)
    //                     colors[i] = the_use_color;
                
    //             // push in new connected halfedges
    //             auto ohi = zs::reinterpret_bits<int>(halfedges("opposite_he",hi));
    //             if(ohi < 0 || halfedges_mask[ohi] == 1)
    //                 continue;
    //             auto nohi = ohi;
    //             for(int i = 0;i != 3;++i) {
    //                 if(halfedges_mask[nohi] == 0) {
    //                     halfedges_mask[nohi] = 1;
    //                     the_front.push_back(nohi);
    //                     get_updated = true;
    //                 }
    //                 nohi = get_next_half_edge(nohi,halfedges,1);
    //             }
    //         }
    //     };

    //     do{
    //         paint_and_update_the_front(coloring_color,flooding_front_positive,has_positive_front_update);
    //         paint_and_update_the_front(-coloring_color,flooding_front_negative,has_negative_front_update);
    //     } while(has_positive_front_update && has_negative_front_update);

    //     if(has_negative_front_update) {
    //         // use the positive coloring
    //         ompPol(zs::range(colors),[] (auto& clr) mutable {
    //             if(clr == -coloring_color)
    //                 clr = background_color;
    //         });
    //     } else if(has_positive_front_update) {
    //         ompPol(zs::range(colors),[] (auto& clr) mutable {
    //             if(clr == coloring_color)
    //                 clr = background_color;
    //             else if(clr == -coloring_color)
    //                 clr = coloring_color;
    //         });
    //     }
    // }


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
        retrieve_self_intersection_tri_halfedge_pairs(pol,verts,xtag,tris,halfedges,csHT);
        bht<int,2,int> boundary_pairs_set{csHT.get_allocator(),MAX_BOUNDARY_POINTS};
        boundary_pairs_set.reset(pol,true);
        bht<int,2,int> turning_pairs_set{csHT.get_allocator(),MAX_NM_TURNING_POINTS};
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


};
};

