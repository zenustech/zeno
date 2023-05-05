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



template<typename Pol,typename PosTileVec,typename TriTileVec>
int retrieve_triangulate_mesh_intersection_list(Pol& pol,
    const PosTileVec& verts_A,
    const zs::SmallString& xtag_A,
    TriTileVec& tris_A,
    const PosTileVec& verts_B,
    const zs::SmallString& xtag_B,
    TriTileVec& tris_B,
    zs::Vector<zs::vec<int,2>>& intersect_buffers,
    zs::Vector<int>& intersect_types,
    bool use_self_collision = false) {
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

        pol(zs::range(tris_B.size()),[
            exec_tag,
            nmIts = proxy<space>(nmIts),
            verts_A = proxy<space>({},verts_A),
            tris_A = proxy<space>({},tris_A),
            verts_B = proxy<space>({},verts_B),
            tris_B = proxy<space>({},tris_B),
            triABvh = proxy<space>(triABvh),
            intersect_buffers = proxy<space>(intersect_buffers),
            intersect_types = proxy<space>(intersect_types),
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
                        if(nm_topological_coincidences == 2){
                            // should we neglect this sort of intersection?
                            // return;
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
                            if(vol < 0)
                                printf("invalid vol evaluation detected %f\n",(float)vol);
                            if(vol > 1e-4)
                                return;


                            auto va01 = vA[1] - vA[0];
                            auto va02 = vA[2] - vA[0];
                            auto nrmA = va01.cross(va02).normalized();

                            if(nrmA.dot(nrmB) > (T)0.0)
                                return;
                            // now the two triangles are coplanar
                            // check intersection

                            printf("detected target type[0] of intersection : %d %d\n",ta_i,tb_i);

                            auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                            intersect_buffers[offset][0] = ta_i;
                            intersect_buffers[offset][1] = tb_i;
                            intersect_types[offset] = 2;
                            return;
                        }
                        if(nm_topological_coincidences == 1){
                            // return;
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
                            if(r < (T)(1.0 + 1e-6)) {
                                printf("detected target type[1] of intersection : %d %d\n",ta_i,tb_i);
                                auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                                intersect_buffers[offset][0] = ta_i;
                                intersect_buffers[offset][1] = tb_i;
                                intersect_types[offset] = 1;
                                return;
                            }

                            ro = vB[(triB_coincide_idx + 1) % 3];
                            r = LSL_GEO::tri_ray_intersect(ro,eb,vA[0],vA[1],vA[2]);
                            if(r < (T)(1.0 + 1e-6)) {
                                printf("detected target type[1] of intersection : %d %d\n",ta_i,tb_i);
                                auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                                intersect_buffers[offset][0] = ta_i;
                                intersect_buffers[offset][1] = tb_i;
                                intersect_types[offset] = 1;
                                return;
                            }
                        } else {
 // return;
                            vec3 eas[3] = {};
                            vec3 ebs[3] = {};

                            for(int i = 0;i != 3;++i) {
                                eas[i] = vA[(i + 1) % 3] - vA[i];
                                ebs[i] = vB[(i + 1) % 3] - vB[i];
                            }

                            for(int i = 0;i != 3;++i){
                                auto r = LSL_GEO::tri_ray_intersect(vA[i],eas[i],vB[0],vB[1],vB[2]);
                                if(r < (T)(1.0 + 1e-6)) {
                                    auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);

                                    // if(offset == 0) {
                                    //     printf("detected first intersection :\nvA : {%f %f %f}\neas : {%f %f %f}\nvB[0] : {%f %f %f}\nvB[1] : {%f %f %f}\nvB[2] : {%f %f %f}\n",
                                    //         (float)vA[i][0],(float)vA[i][1],(float)vA[i][2],
                                    //         (float)eas[i][0],(float)eas[i][1],(float)eas[i][2],
                                    //         (float)vB[0][0],(float)vB[0][1],(float)vB[0][2],
                                    //         (float)vB[1][0],(float)vB[1][1],(float)vB[1][2],
                                    //         (float)vB[2][0],(float)vB[2][1],(float)vB[2][2]);
                                    //     printf("the two : triA{%d %d %d},triB{%d %d %d}\n",
                                    //         triA[0],triA[1],triA[2],triB[0],triB[1],triB[2]);
                                    // }

                                    intersect_buffers[offset][0] = ta_i;
                                    intersect_buffers[offset][1] = tb_i;
                                    intersect_types[offset] = 0;
                                    return;
                                }

                                r = LSL_GEO::tri_ray_intersect(vB[i],ebs[i],vA[0],vA[1],vA[2]);
                                if(r < (T)(1.0 + 1e-6)) {
                                    auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                    
                                    // if(offset == 0) {
                                    //     auto vBend = vB[i] + ebs[i];
                                    //     printf("detected first intersection[%f] :\nvB : {%f %f %f}\nebs : {%f %f %f}\nvBend : {%f %f %f}\nvA[0] : {%f %f %f}\nvA[1] : {%f %f %f}\nvA[2] : {%f %f %f}\n",
                                    //         (float)r,
                                    //         (float)vB[i][0],(float)vB[i][1],(float)vB[i][2],
                                    //         (float)ebs[i][0],(float)ebs[i][1],(float)ebs[i][2],
                                    //         (float)vBend[0],(float)vBend[1],(float)vBend[2],
                                    //         (float)vA[0][0],(float)vA[0][1],(float)vA[0][2],
                                    //         (float)vA[1][0],(float)vA[1][1],(float)vA[1][2],
                                    //         (float)vA[2][0],(float)vA[2][1],(float)vA[2][2]);
                                    // }

                                    intersect_buffers[offset][0] = ta_i;
                                    intersect_buffers[offset][1] = tb_i;
                                    intersect_types[offset] = 0;
                                    return;
                                }
                            }
                        }
                    }
                    else{
                        printf("the non-self-intersection algorithm not implemented yet");
                        return;
                    }
                };

                triABvh.iter_neighbors(bv,process_potential_intersection_pairs);
        });

        return nmIts.getVal(0);
}


template<typename Pol,typename PosTileVec,typename TriTileVec>
int do_global_self_intersection_analysis_on_surface_mesh(Pol& pol,
    const PosTileVec& verts,
    const zs::SmallString& xtag,
    const TriTileVec& tris,
    // bool is_volume_surface,
    zs::Vector<zs::vec<int,2>>& ints_buffer,
    zs::Vector<int>& nodal_colors) {
        using namespace zs;
        using index_type = std::make_signed_t<int>;
        using size_type = std::make_unsigned_t<int>;
        using table_vec2i_type = zs::bcht<zs::vec<int,2>,int,true,zs::universal_hash<zs::vec<int,2>>,16>;
        using table_int_type = zs::bcht<int,int,true,zs::universal_hash<int>,16>;
        using edge_topo_type = zs::Vector<zs::vec<int,2>>;
        using inst_buffer_type = zs::Vector<zs::vec<int,2>>;
        using inst_class_type = zs::Vector<int>;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};

        // table_type tab{verts.get_allocator(),tris.size() * 8};
        // inst_buffer_type ints_buffer{tris.get_allocator(),tris.size() * 8};
        inst_class_type ints_types{tris.get_allocator(),tris.size() * 8};

        auto nm_insts = retrieve_triangulate_mesh_intersection_list(pol,
            verts,xtag,tris,verts,xtag,tris,ints_buffer,ints_types,true);
        
        edge_topo_type dc_edge_topos{tris.get_allocator(),nm_insts * 6};
        table_int_type tab{tris.get_allocator(),tris.size() * 8};

        // auto topo_tag = is_volume_surface ? "fp_inds" : "inds";
        std::string topo_tag{"inds"};
        // if(!tris.hasProperty(topo_tag))
        //     fmt::print(fg(fmt::color::red),"do_global_self_intersection_analysis::the input tris has no {} topo channel\n",topo_tag);

        pol(zs::range(nm_insts),[
            ints_buffer = proxy<space>(ints_buffer),
            ints_types = proxy<space>(ints_types),
            topo_tag = zs::SmallString(topo_tag),
            dc_edge_topos = proxy<space>(dc_edge_topos),
            tab = proxy<space>(tab),
            tris = proxy<space>({},tris)] ZS_LAMBDA(int isi) mutable {
                auto ta = ints_buffer[isi][0];
                auto tb = ints_buffer[isi][1];
                auto triA = tris.pack(dim_c<3>,topo_tag,ta,int_c);
                auto triB = tris.pack(dim_c<3>,topo_tag,tb,int_c);

                dc_edge_topos[isi * 6 + 0] = zs::vec<int,2>{triA[0],triA[1]};
                dc_edge_topos[isi * 6 + 1] = zs::vec<int,2>{triA[1],triA[2]};
                dc_edge_topos[isi * 6 + 2] = zs::vec<int,2>{triA[2],triA[0]};
                dc_edge_topos[isi * 6 + 3] = zs::vec<int,2>{triB[0],triB[1]};
                dc_edge_topos[isi * 6 + 4] = zs::vec<int,2>{triB[1],triB[2]};
                dc_edge_topos[isi * 6 + 5] = zs::vec<int,2>{triB[2],triB[0]};

                if(ints_types[isi] == 1) {
                    int coincident_idx = -1;
                    for(int i = 0;i != 3;++i)
                        for(int j = 0;j != 3;++j)
                            if(triA[i] == triB[j])
                                coincident_idx = triA[i];
                    tab.insert(coincident_idx);
                }                
        });

        edge_topo_type edge_topos{tris.get_allocator(),tris.size() * 3};
        pol(range(tris.size()),[
            tris = proxy<space>({},tris),
            tab = proxy<space>(tab),
            topo_tag = zs::SmallString(topo_tag),
            edge_topos = proxy<space>(edge_topos)] ZS_LAMBDA(int ti) mutable {
                auto tri = tris.pack(dim_c<3>,topo_tag,ti,int_c);
                auto is_coincident_idx = zs::vec<bool,3>::uniform(false);
                for(int i = 0;i != 3;++i)
                    if(auto qno = tab.query(tri[i]);qno >= 0)
                        is_coincident_idx[i] = true;

                for(int i = 0;i != 3;++i){
                    if(is_coincident_idx[i] || is_coincident_idx[(i + 1) % 3])
                        edge_topos[ti * 3 + i] = zs::vec<int,2>::uniform(-1);
                    else
                        edge_topos[ti * 3 + i] = zs::vec<int,2>{tri[i],tri[(i + 1) % 3]};
                }
        }); 

        zs::Vector<int> island_buffer{verts.get_allocator(),verts.size()};
		auto nm_islands = mark_disconnected_island(pol,edge_topos,dc_edge_topos,island_buffer);
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

        nodal_colors.resize(verts.size());
        pol(zs::range(verts.size()),[
            nodal_colors = proxy<space>(nodal_colors),
            // ints_types = proxy<space>(ints_types),
            max_island_idx = max_island_idx,
            island_buffer = proxy<space>(island_buffer)] ZS_LAMBDA(int vi) mutable {
                auto island_idx = island_buffer[vi];

                if(island_idx == max_island_idx/* || ints_types[island_idx] == 1*/)
                    nodal_colors[vi] = 0;
                else
                    nodal_colors[vi] = 1;
        });
        // skip the intersection pair with only one coincident point? s
        pol(zs::range(nm_insts),[
            ints_types = proxy<space>(ints_types),
            nodal_colors = proxy<space>(nodal_colors),
            ints_buffer = proxy<space>(ints_buffer),
            tris = proxy<space>({},tris)] ZS_LAMBDA(int isi) mutable {
                if(ints_types[isi] == 1) {
                    auto ta = ints_buffer[isi][0];
                    auto tb = ints_buffer[isi][1];
                    auto triA = tris.pack(dim_c<3>,"inds",ta,int_c);
                    auto triB = tris.pack(dim_c<3>,"inds",tb,int_c);

                    int coidx = 0;
                    for(int i = 0;i != 3;++i)
                        for(int j = 0;j != 3;++j)
                            if(triA[i] == triB[j])
                                coidx = triA[i];
                    nodal_colors[coidx] = 0;
                }
        });

        // pol(zs::range(nm_insts),[
        //     ints_buffer = proxy<space>(ints_buffer),
        //     tris = proxy<space>({},tris),
        //     topo_tag = zs::SmallString(topo_tag),
        //     nodal_colors = proxy<space>(nodal_colors)] ZS_LAMBDA(int isi) mutable {
        //         auto ta = ints_buffer[isi][0];
        //         auto tb = ints_buffer[isi][1];
        //         auto triA = tris.pack(dim_c<3>,topo_tag,ta,int_c);
        //         auto triB = tris.pack(dim_c<3>,topo_tag,tb,int_c);

        //         for(int i = 0;i != 3;++i) {
        //             nodal_colors[triA[i]] = 1;
        //             nodal_colors[triB[i]] = 1;
        //         }
        // });

        return nm_insts;
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
                r = LSL_GEO::tri_ray_intersect(ro,rd,vA[0],vA[1],vA[2]);
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