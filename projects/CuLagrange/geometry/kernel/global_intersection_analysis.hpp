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
#include "halfedge_structure.hpp"
#include "intersection.hpp"

namespace zeno {

namespace GIA {

    constexpr int GIA_BACKGROUND_COLOR = 0;
    constexpr int GIA_TURNING_POINTS_COLOR = 1;
    constexpr int GIA_RED_COLOR = 2;
    constexpr int GIA_WHITE_COLOR = 3;
    constexpr int GIA_BLACK_COLOR = 4;
    constexpr int DEFAULT_MAX_GIA_INTERSECTION_PAIR = 100000;
    constexpr int DEFAULT_MAX_NM_TURNING_POINTS = 500;
    constexpr int DEFAULT_MAX_BOUNDARY_POINTS = 500;

    constexpr auto GIA_CS_ET_BUFFER_KEY = "GIA_CS_ET_BUFFER_KEY";
    constexpr auto GIA_CS_EKT_BUFFER_KEY = "GIA_CS_EKT_BUFFER_KEY";
    constexpr auto GIA_VTEMP_BUFFER_KEY = "GIA_VTEMP_BUFFER_KEY";
    constexpr auto GIA_TRI_BVH_BUFFER_KEY = "GIA_TRI_BVH_BUFFER_KEY";
    constexpr auto GIA_KTRI_BVH_BUFFER_KEY = "GIA_TRI_BVH_BUFFER_KEY";


    template<typename Pol,typename PosTileVec,typename TriTileVec,typename HalfEdgeTileVec,typename GIA_TILEVEC>
    int do_global_self_intersection_analysis(Pol& pol,
        const PosTileVec& verts,
        const zs::SmallString& xtag,
        const TriTileVec& tris,
        HalfEdgeTileVec& halfedges,
        GIA_TILEVEC& gia_res,
        GIA_TILEVEC& tris_gia_res,
        size_t max_nm_intersections = 50000) {
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
                    {"int_points",3},
                    {"r",1},
                    {"is_broken",1}
                },max_nm_intersections};

            if(!halfedges.hasProperty("broken"))
                halfedges.append_channels(pol,{{"broken",1}});
            TILEVEC_OPS::fill(pol,halfedges,"broken",(T)0.0);
            
            auto nm_insts = retrieve_self_intersection_tri_halfedge_list_info(pol,verts,xtag,tris,halfedges,ints_buffer);

            std::cout << "nm_insts : " << nm_insts << std::endl;
            TILEVEC_OPS::fill(pol,ints_buffer,"is_broken",(T)0);
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
                    if(auto setNo = cftab.insert(zs::vec<int,2>{hi,ti});setNo != table_vec2i_type::sentinel_v)
                        cfbuffer[setNo] = isi;

            });

            // std::cout << "build incidentItsTab" << std::endl;
            zs::Vector<int> nmInvalid{gia_res.get_allocator(),1};
            nmInvalid.setVal(0);

            table_vec2i_type incidentItsTab{tris.get_allocator(),nm_insts * 2};
            incidentItsTab.reset(pol,true);    
            pol(zs::range(nm_insts),[
                exec_tag,
                verts = proxy<space>({},verts),
                xtag = zs::SmallString(xtag),
                ints_buffer = proxy<space>({},ints_buffer),
                cftab = proxy<space>(cftab),
                cfbuffer = proxy<space>(cfbuffer),
                nmInvalid = proxy<space>(nmInvalid),
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
                            vec3 tV[3] = {};
                            auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                            for(int i = 0;i != 3;++i)
                                tV[i] = verts.pack(dim_c<3>,xtag,tri[i],int_c);
                            
                            auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                            auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);
                            auto hlocal_idx = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
                            zs::vec<int,2> hedge{htri[hlocal_idx],htri[(hlocal_idx + 1) % 3]};
                            zs::vec<T,3> eV[2] = {};
                            for(int i = 0;i != 2;++i)
                                eV[i] = verts.pack(dim_c<3>,xtag,hedge[i]);
                            auto dir = eV[1] - eV[0];
                            double hr{};
                            LSL_GEO::tri_ray_intersect_d<double>(eV[0],eV[1],tV[0],tV[1],tV[2],hr);

                            auto ohti = zs::reinterpret_bits<int>(halfedges("to_face",ohi));
                            auto ohtri = tris.pack(dim_c<3>,"inds",ohti,int_c);
                            auto ohlocal_idx = zs::reinterpret_bits<int>(halfedges("local_vertex_id",ohi));
                            zs::vec<int,2> ohedge{ohtri[ohlocal_idx],ohtri[(ohlocal_idx + 1) % 3]};
                            for(int i = 0;i != 2;++i)
                                eV[i] = verts.pack(dim_c<3>,xtag,ohedge[i]);
                            dir = eV[1] - eV[0];
                            double ohr{};
                            LSL_GEO::tri_ray_intersect_d<double>(eV[0],eV[1],tV[0],tV[1],tV[2],ohr);


                            printf("do_global_self_intersection_analysis_error::impossible reaching here, the hi and ohi should both have been inserted %f %f %f\n",(float)hr,(float)ohr,(float)ints_buffer("r",isi));
                            ints_buffer("is_broken",isi) = (T)1.0;
                            atomic_add(exec_tag,&nmInvalid[0],(int)1);
                            return;
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

                    printf("do_global_self_intersection_analysis_error::impossible reaching here with broken insertion ring %f\n",(float)ints_buffer("r",isi));
                    ints_buffer("is_broken",isi) = (T)1.0;
                    atomic_add(exec_tag,&nmInvalid[0],(int)1);
            });

            auto nmInvalidCount = nmInvalid.getVal(0);
            if(nmInvalidCount > 0)
                printf("SELF GIA invalid state detected\n");
            // there might be some broken rings


            auto nmEntries = incidentItsTab.size();
            zs::Vector<zs::vec<int,2>> conn_topo{tris.get_allocator(),nmEntries};
            pol(zip(conn_topo,range(incidentItsTab._activeKeys)),[] ZS_LAMBDA(zs::vec<int,2> &ij,const auto& key) mutable {ij = key;});
            zs::Vector<int> ringTag{tris.get_allocator(),(size_t)nm_insts};

            // std::cout << "Mark disconnected island" << std::endl;


            auto nm_rings = mark_disconnected_island(pol,conn_topo,ringTag);

            std::cout << "nm_rings : " << nm_rings << std::endl;

            zs::Vector<int> is_broken_rings{ringTag.get_allocator(),(size_t)nm_rings};
            pol(zs::range(is_broken_rings),[] ZS_LAMBDA(auto& is_br) mutable {is_br = 0;});
            pol(zs::range(nm_insts),[ringTag = proxy<space>(ringTag),is_broken_rings = proxy<space>(is_broken_rings),ints_buffer = proxy<space>({},ints_buffer)] ZS_LAMBDA(int isi) mutable {
                if(ints_buffer("is_broken",isi) > (T)0.5) {
                    auto ring_id = ringTag[isi];
                    is_broken_rings[ring_id] = 1;
                }
            });

            std::cout << "broken_ring_tag : ";
            for(int i = 0;i != nm_rings;++i)
                std::cout << is_broken_rings.getVal(i) << "\t";
            std::cout << std::endl;

            // std::cout << "finish Mark disconnected island with nm_rings : " << nm_rings << std::endl;

            auto ring_mask_width = (nm_rings + 31) / 32;

            gia_res.resize(verts.size() * ring_mask_width);
            pol(zs::range(gia_res.size()),[gia_res = proxy<space>({},gia_res)] ZS_LAMBDA(int mi) mutable {
                // nodal_colors[ni] = 0;
                gia_res("ring_mask",mi) = zs::reinterpret_bits<T>((int)0);
                gia_res("color_mask",mi) = zs::reinterpret_bits<T>((int)0);
                gia_res("type_mask",mi) = zs::reinterpret_bits<T>((int)0);
                gia_res("is_loop_vertex",mi) = zs::reinterpret_bits<T>((int)0);
            });
            tris_gia_res.resize(tris.size() * ring_mask_width);
            pol(zs::range(tris_gia_res.size()),[tris_gia_res = proxy<space>({},tris_gia_res)] ZS_LAMBDA(int mi) mutable {
                // nodal_colors[ni] = 0;
                tris_gia_res("ring_mask",mi) = zs::reinterpret_bits<T>((int)0);
                tris_gia_res("color_mask",mi) = zs::reinterpret_bits<T>((int)0);
                tris_gia_res("type_mask",mi) = zs::reinterpret_bits<T>((int)0);
                // tris_gia_res("is_loop_vertex",ti) = zs::reinterpret_bits<T>((int)0);
            });

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

            // pol(zs::range(nm_rings),[ringSize = proxy<space>(ringSize),is_broken_rings = proxy<space>(is_broken_rings)] ZS_LAMBDA(int ri) mutable {
            //     if(is_broken_rings[ri])
            //         ringSize[ri] = 0;
            // });

            zs::Vector<int> island_buffer{verts.get_allocator(),verts.size()};
            

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

            // conn_of_first_ring.reset(pol,true);

            for(int ri = 0;ri != nm_rings;++ri) {
                auto rsize = (size_t)ringSize.getVal(ri);
                // if(rsize == 0)
                //     continue;
                auto is_broken_ring = is_broken_rings.getVal(ri);
                if(is_broken_ring)
                    continue;

                int cur_ri_mask = 1 << (ri % 32);
                int ri_offset = ri / 32;

                table_int_type disable_points{tris.get_allocator(),rsize * 8};
                table_vec2i_type disable_lines{tris.get_allocator(),rsize * 6};
                disable_points.reset(pol,true);
                disable_lines.reset(pol,true);

                pol(zs::range(nm_insts),[
                    ints_buffer = proxy<space>({},ints_buffer),
                    ringTag = proxy<space>(ringTag),
                    ri,
                    cur_ri_mask = cur_ri_mask,
                    ri_offset = ri_offset,
                    ring_mask_width = ring_mask_width,
                    halfedges = proxy<space>({},halfedges),
                    gia_res = proxy<space>({},gia_res),
                    tris_gia_res = proxy<space>({},tris_gia_res),
                    verts = proxy<space>({},verts),
                    disable_points = proxy<space>(disable_points),
                    disable_lines = proxy<space>(disable_lines),
                    nm_insts,
                    tris = proxy<space>({},tris)] ZS_LAMBDA(int isi) mutable {
                        if(ringTag[isi] != ri)
                            return;
                        
                        auto pair = ints_buffer.pack(dim_c<2>,"pair",isi,int_c);
                        auto hi = pair[0];
                        auto ti = pair[1];

                        int cur_ri_mask = 1 << ri;
                        auto tring_mask = zs::reinterpret_bits<int>(tris_gia_res("ring_mask",ti * ring_mask_width + ri_offset));
                        tring_mask |= cur_ri_mask;
                        tris_gia_res("ring_mask",ti * ring_mask_width + ri_offset) = zs::reinterpret_bits<T>(tring_mask);

                        halfedges("broken",hi) = (T)1.0;

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
                        auto corner_idx = zs::reinterpret_bits<int>(ints_buffer("corner_idx",isi));
                        if(corner_idx >= 0){
                            gia_res("is_loop_vertex",corner_idx * ring_mask_width + ri_offset) = (T)1.0;
                            disable_points.insert(corner_idx);
                        }
                });
                // table_vec2i_type connected_topo{edge_topos.get_allocator(),edge_topos.size() * 2};
                int nm_islands = 0;
                nm_islands = mark_disconnected_island(pol,edge_topos,disable_points,disable_lines,island_buffer);
                std::cout << "ring[" << ri << "] : " << nm_islands << "\tnm_broken_edges : " << disable_lines.size() << "\tnm_broken_corners : " << disable_points.size() << std::endl;


                zs::Vector<int> nm_cmps_every_island_count{verts.get_allocator(),(size_t)nm_islands};
                // nm_cmps_every_island_count.setVal(0);
                pol(zs::range(nm_cmps_every_island_count.size()),[
                    nm_cmps_every_island_count = proxy<space>(nm_cmps_every_island_count)] ZS_LAMBDA(int i) mutable {
                        nm_cmps_every_island_count[i] = 0;
                });

                // it is a really bad idea to use mustExclude here, as this tag might no be locally significant
                pol(zs::range(verts.size()),[
                    exec_tag,
                    verts = proxy<space>({},verts),
                    island_buffer = proxy<space>(island_buffer),
                    nm_cmps_every_island_count = proxy<space>(nm_cmps_every_island_count)] ZS_LAMBDA(int vi) mutable {
                        auto island_idx = island_buffer[vi];
                        // if(verts.hasProperty("mustExclude"))
                        //     if(verts("mustExclude",vi) > (T)0.5)
                        //         return; 
                        atomic_add(exec_tag,&nm_cmps_every_island_count[island_idx],(int)1);
                });



                int max_size = 0;
                int max_island_idx = -1;
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


                // auto cur_ri_mask = (int)1 << ri;


                for(int i = 0;i != nm_islands;++i)
                    std::cout << nm_cmps_every_island_count.getVal(i) << "\t";
                // std::cout << "max_island = " << max_island_idx << std::endl;
                // std::cout << std::endl;

                pol(zs::range(verts.size()),[
                    gia_res = proxy<space>({},gia_res),
                    nm_islands,
                    cur_ri_mask,
                    ri_offset = ri_offset,
                    ring_mask_width = ring_mask_width,
                    black_island_idx,
                    exec_tag,
                    // ints_types = proxy<space>(ints_types),
                    max_island_idx = max_island_idx,
                    island_buffer = proxy<space>(island_buffer)] ZS_LAMBDA(int vi) mutable {
                        auto island_idx = island_buffer[vi];
                        if(island_idx == max_island_idx)
                            return;
                        // might exceed the integer range
                        auto ring_mask = zs::reinterpret_bits<int>(gia_res("ring_mask",vi * ring_mask_width + ri_offset));
                        auto color_mask = zs::reinterpret_bits<int>(gia_res("color_mask",vi * ring_mask_width + ri_offset));
                        auto type_mask = zs::reinterpret_bits<int>(gia_res("type_mask",vi * ring_mask_width + ri_offset));
                        // ring_mask += ((int) << ri)

                        // if(island_idx != max_island_idx/* || ints_types[island_idx] == 1*/){
                        ring_mask |= cur_ri_mask;
                        // }
                        if(nm_islands == 3)
                            type_mask |= cur_ri_mask;
                        if(nm_islands == 3 && island_idx == black_island_idx)
                            color_mask |= cur_ri_mask;
                        gia_res("ring_mask",vi * ring_mask_width + ri_offset) = zs::reinterpret_bits<T>(ring_mask);
                        gia_res("color_mask",vi * ring_mask_width + ri_offset) = zs::reinterpret_bits<T>(color_mask);
                        gia_res("type_mask",vi * ring_mask_width + ri_offset) = zs::reinterpret_bits<T>(type_mask);
                });
            }

            pol(zs::range(gia_res.size()),[
                // ring_mask_width = ring_mask_width,
                gia_res = proxy<space>({},gia_res)] ZS_LAMBDA(int mi) mutable {
                    // for(int i = 0;i != ring_mask_width;++i) {
                        auto is_corner = gia_res("is_loop_vertex",mi);
                        if(is_corner > (T)0.5)
                            gia_res("ring_mask",mi) = zs::reinterpret_bits<T>((int)0);
                    // }
            });

            return ring_mask_width;
    }

    // NEW VERSION
    template<typename Pol,typename PosTileVec,typename TriTileVec,typename HalfEdgeTileVec, auto space = Pol::exec_tag::value>
    int do_global_intersection_analysis_with_connected_manifolds(Pol& pol,
        const PosTileVec& verts_A,const zs::SmallString& xtag_A,
        const TriTileVec& tris_A,
        HalfEdgeTileVec& halfedges_A,bool A_intersect_interior,
        const PosTileVec& verts_B,const zs::SmallString& xtag_B,
        const TriTileVec& tris_B,
        const HalfEdgeTileVec& halfedges_B,bool B_intersect_interior,
        zs::Vector<int>& gia_res,zs::Vector<int>& tris_gia_res) {
            using namespace zs;
            using T = typename PosTileVec::value_type;
            using dtiles_t = zs::TileVector<T, 32>;
            using table_vec2i_type = zs::bht<int,2,int>;
            using table_int_type = zs::bht<int,1,int>;
            using vec2i = zs::vec<int,2>;
            // constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
            constexpr auto exec_tag = wrapv<space>{};

            constexpr size_t MAX_NM_INTERSECTIONS = 50000;

            dtiles_t ints_buffer_A_2_B{verts_A.get_allocator(),{
                {"pair",2},
                {"int_points",3},
                {"r",1}
            },MAX_NM_INTERSECTIONS};
            dtiles_t ints_buffer_B_2_A{verts_B.get_allocator(),{
                {"pair",2},
                {"int_points",3},
                {"r",1}
            },MAX_NM_INTERSECTIONS};


            // using the buffer is much safer
            auto nm_A_2_B_ints = retrieve_intersection_tri_halfedge_info_of_two_meshes(pol,verts_A,xtag_A,tris_A,halfedges_A,verts_B,xtag_B,tris_B,ints_buffer_A_2_B);
            auto nm_B_2_A_ints = retrieve_intersection_tri_halfedge_info_of_two_meshes(pol,verts_B,xtag_B,tris_B,halfedges_B,verts_A,xtag_A,tris_A,ints_buffer_B_2_A);

            if(!halfedges_A.hasProperty("intersect")) {
                halfedges_A.append_channels(pol,{{"intersect",1}});
            }
            TILEVEC_OPS::fill(pol,halfedges_A,"intersect",(T)0.0);
            pol(zs::range(nm_A_2_B_ints),[
                ints_buffer_A_2_B = proxy<space>({},ints_buffer_A_2_B),halfedges_A = proxy<space>({},halfedges_A)] ZS_LAMBDA(int iti) mutable {
                    auto pair = ints_buffer_A_2_B.pack(dim_c<2>,"pair",iti,int_c);
                    halfedges_A("intersect",pair[0]) = (T)1.0;
            });

            auto total_nm_ints = nm_A_2_B_ints + nm_B_2_A_ints;

            std::cout << "total_nm_ints : " << total_nm_ints << " : " << nm_A_2_B_ints << " : " << nm_B_2_A_ints << std::endl;

            auto total_nm_verts = verts_A.size() + verts_B.size();
            auto total_nm_tris = tris_A.size() + tris_B.size();


            if(total_nm_ints == 0) {
                gia_res.resize(total_nm_verts);
                tris_gia_res.resize(total_nm_tris);
                pol(zs::range(gia_res),[] ZS_LAMBDA(auto& ring_mask) {ring_mask = 0;});
                pol(zs::range(tris_gia_res),[] ZS_LAMBDA(auto& ring_mask) {ring_mask = 0;});
                return 1;
            }

            table_vec2i_type A_2_B_tab{ints_buffer_A_2_B.get_allocator(),(size_t)nm_A_2_B_ints};
            A_2_B_tab.reset(pol,true);
            zs::Vector<int> A_2_B_buffer{ints_buffer_A_2_B.get_allocator(),(size_t)nm_A_2_B_ints};

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

            zs::Vector<int> nmInvalid{verts_A.get_allocator(),(size_t)1};
            nmInvalid.setVal(0);

            auto establish_connections = [&pol,&nmInvalid,&incidentItsTab,exec_tag](
                const zs::bht<int,2,int>& _A_2_B_tab,const Vector<int>& _A_2_B_buffer,const HalfEdgeTileVec& _A_halfedges,const TriTileVec& _A_tris,size_t _A_offset,
                const zs::bht<int,2,int>& _B_2_A_tab,const Vector<int>& _B_2_A_buffer,const HalfEdgeTileVec& _B_halfedges,const TriTileVec& _B_tris,size_t _B_offset) mutable {
                    auto nn = _A_2_B_tab.size();
                    pol(zip(zs::range(nn),_A_2_B_tab._activeKeys),[
                        exec_tag,
                        _A_2_B_tab = proxy<space>(_A_2_B_tab),
                        _A_2_B_buffer = proxy<space>(_A_2_B_buffer),
                        _A_halfedges = proxy<space>({},_A_halfedges),
                        _A_tris = proxy<space>({},_A_tris),
                        _A_offset = _A_offset,
                        _B_2_A_tab = proxy<space>(_B_2_A_tab),
                        _B_2_A_buffer = proxy<space>(_B_2_A_buffer),
                        _B_halfedges = proxy<space>({},_B_halfedges),
                        _B_tris = proxy<space>({},_B_tris),
                        _B_offset = _B_offset,
                        nmInvalid = proxy<space>(nmInvalid),
                        incidentItsTab = proxy<space>(incidentItsTab)] ZS_LAMBDA(auto,const auto& a2b) mutable {
                            auto a2b_no = _A_2_B_tab.query(a2b);
                            auto a2b_isi = _A_2_B_buffer[a2b_no];
                            auto ha = a2b[0];
                            auto tb = a2b[1];
                            auto oha = zs::reinterpret_bits<int>(_A_halfedges("opposite_he",ha));
                            if(oha >= 0) {
                                if(auto na2b_no = _A_2_B_tab.query(vec2i{oha,tb});na2b_no >= 0) {
                                    auto na2b_isi = _A_2_B_buffer[na2b_no];
                                    // as it is symmtric, we only need to establish this connectivity once
                                    if(a2b_isi + _A_offset > na2b_isi + _A_offset)
                                        incidentItsTab.insert(vec2i{a2b_isi + _A_offset,na2b_isi + _A_offset});
                                }else {
                                    printf("do_global_intersection_analysis_with_connected_manifolds_new::impossible reaching here, the hi and ohi should both have been inserted\n");
                                    atomic_add(exec_tag,&nmInvalid[0],(int)1);
                                }
                            }
                            // notice here the nb2a_no + _B_offset might be very likely inside the range of a2b_no which [0,...,MAX_NM_INTERSECTIONS]
                            // with some possibility, nb2a_no + _B_offset might equal some query of a2b_tab, which might be problematic
                            auto nha = ha;
                            for(int i = 0;i != 2;++i) {
                                nha = zs::reinterpret_bits<int>(_A_halfedges("next_he",nha));
                                if(auto na2b_no = _A_2_B_tab.query(vec2i{nha,tb});na2b_no >= 0) {
                                    // as it is symmtric, we only need to establish this connectivity once
                                    auto na2b_isi = _A_2_B_buffer[na2b_no]; 
                                    if(a2b_isi + _A_offset > na2b_isi + _A_offset)
                                        incidentItsTab.insert(vec2i{a2b_isi + _A_offset,na2b_isi + _A_offset});
                                    return;
                                }
                            }
                            auto ta = zs::reinterpret_bits<int>(_A_halfedges("to_face",ha));
                            auto hb = zs::reinterpret_bits<int>(_B_tris("he_inds",tb));
                            for(int i = 0;i != 3;++i) {
                                if(auto nb2a_no = _B_2_A_tab.query(vec2i{hb,ta});nb2a_no >= 0) {
                                    auto nb2a_isi = _B_2_A_buffer[nb2a_no];
                                    if(a2b_isi + _A_offset > nb2a_isi + _B_offset)
                                        incidentItsTab.insert(vec2i{a2b_isi + _A_offset,nb2a_isi + _B_offset});
                                    return;
                                }
                                hb = zs::reinterpret_bits<int>(_B_halfedges("next_he",hb));
                            }
                            printf("do_global_intersection_analysis_with_connected_manifolds_new::impossible reaching here, the intersection ring seems to be broken\n");
                            atomic_add(exec_tag,&nmInvalid[0],(int)1);
                    });
            };

            auto nmInvalidCounts = nmInvalid.getVal(0);
            if(nmInvalidCounts > 0)
                throw std::runtime_error("Invalid state for GIA detected");

            int A_offset = 0;
            int B_offset = nm_A_2_B_ints;
            establish_connections(A_2_B_tab,A_2_B_buffer,halfedges_A,tris_A,A_offset,B_2_A_tab,B_2_A_buffer,halfedges_B,tris_B,B_offset);
            establish_connections(B_2_A_tab,B_2_A_buffer,halfedges_B,tris_B,B_offset,A_2_B_tab,A_2_B_buffer,halfedges_A,tris_A,A_offset);

            auto nmEmtries = incidentItsTab.size();
            std::cout << "nm_incidentItsTab : " << nmEmtries << std::endl;

            zs::Vector<zs::vec<int,2>> conn_topo{gia_res.get_allocator(),nmEmtries};
            pol(zip(conn_topo,range(incidentItsTab._activeKeys)),[] ZS_LAMBDA(zs::vec<int,2> &ij,const auto& key) mutable {ij = key;});
            zs::Vector<int> ringTag{gia_res.get_allocator(),(size_t)total_nm_ints};
            auto nm_rings = mark_disconnected_island(pol,conn_topo,ringTag);

            // width of ring_mask
            auto ring_mask_width = (size_t)((nm_rings + 31) / 32);

            std::cout << "nm_rings : " << nm_rings << std::endl;
            std::cout << "ring_mask_width : " << ring_mask_width << std::endl;


            // if(total_nm_ints == 0) {
            gia_res.resize(total_nm_verts * ring_mask_width);
            tris_gia_res.resize(total_nm_tris * ring_mask_width);
            pol(zs::range(gia_res),[] ZS_LAMBDA(auto& ring_mask) {ring_mask = 0;});
            pol(zs::range(tris_gia_res),[] ZS_LAMBDA(auto& ring_mask) {ring_mask = 0;});
                // return 1;
            // }

            //
            

            zs::Vector<vec2i> edges_topos{tris_A.get_allocator(),tris_A.size() * 3 + tris_B.size() * 3};
            auto tris_2_edges_topo = [&pol](const TriTileVec& tris,zs::Vector<vec2i>& edge_topos,int v_offset,int e_offset) mutable {
                pol(zs::range(tris.size()),[
                    tris = proxy<space>({},tris),
                    e_offset = e_offset,
                    v_offset = v_offset,
                    edge_topos = proxy<space>(edge_topos)] ZS_LAMBDA(int ti) mutable {
                        auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                        for(int i = 0;i != 3;++i) {
                            edge_topos[ti * 3 + i + e_offset] = vec2i{tri[i],tri[(i + 1) % 3]} + (int)v_offset;
                        }
                });
            };
            tris_2_edges_topo(tris_A,edges_topos,0,0);
            tris_2_edges_topo(tris_B,edges_topos,verts_A.size(),tris_A.size() * 3);

            table_vec2i_type disable_lines{edges_topos.get_allocator(),(size_t)total_nm_ints};
            disable_lines.reset(pol,true);
            table_int_type disable_points{edges_topos.get_allocator(),(size_t)total_nm_ints};
            disable_points.reset(pol,true);

            // cut all the all the intersected edges
            auto collect_intersected_halfedges = [&](const HalfEdgeTileVec& halfedges,
                const TriTileVec& tris,
                const dtiles_t& ints_buffer,size_t nm_ints,int v_offset) mutable {
                    pol(zs::range(nm_ints),[
                        v_offset = v_offset,
                        halfedges = proxy<space>({},halfedges),
                        tris = proxy<space>({},tris),
                        disable_lines = proxy<space>(disable_lines),
                        ints_buffer = proxy<space>({},ints_buffer)] ZS_LAMBDA(int iti) mutable {
                            auto pair = ints_buffer.pack(dim_c<2>,"pair",iti,int_c);
                            auto hi = pair[0];
                            auto local_vertex_id = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
                            auto ti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                            auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                            auto a = tri[(local_vertex_id + 0) % 3];
                            auto b = tri[(local_vertex_id + 1) % 3];

                            // if(a > b){
                            //     auto tmp = a;
                            //     a = b;
                            //     b = tmp;    
                            // }
                            disable_lines.insert(zs::vec<int,2>{a + v_offset,b + v_offset});
                    });
            };
            collect_intersected_halfedges(halfedges_A,tris_A,ints_buffer_A_2_B,nm_A_2_B_ints,0);
            collect_intersected_halfedges(halfedges_B,tris_B,ints_buffer_B_2_A,nm_B_2_A_ints,verts_A.size());
            zs::Vector<int> island_buffer{gia_res.get_allocator(),(size_t)total_nm_verts};
            auto nm_islands = mark_disconnected_island(pol,edges_topos,disable_points,disable_lines,island_buffer);

            std::cout << "nm_islands : " << nm_islands << std::endl;

            // find detect the sparse coloring of islands
            auto mark_init_tris_gia = [&](const zs::bht<int,2,int> ints_buffer_tab,zs::Vector<int>& ints_buffer,int its_offset,
                const TriTileVec& tris,int t_offset) {
                    pol(zip(zs::range(ints_buffer_tab.size()),zs::range(ints_buffer_tab._activeKeys)),[
                        ints_buffer_tab = proxy<space>(ints_buffer_tab),
                        tris_gia_res = proxy<space>(tris_gia_res),
                        ints_buffer = proxy<space>(ints_buffer),
                        its_offset = its_offset,t_offset = t_offset,ring_mask_width = ring_mask_width,
                        ringTag = proxy<space>(ringTag)] ZS_LAMBDA(auto,const auto& pair) mutable {
                        auto hi = pair[0];
                        auto ti = pair[1];
                        auto no = ints_buffer_tab.query(pair);
                        auto iti = ints_buffer[no];
                        auto ri = ringTag[iti + its_offset];

                        int ring_mask = 1 << (ri % 32);
                        int ri_offset = ri / 32;

                        atomic_or(exec_cuda,&tris_gia_res[(ti + t_offset) * ring_mask_width + ri_offset],ring_mask);    
                });
            };
            mark_init_tris_gia(A_2_B_tab,A_2_B_buffer,0,tris_B,(int)tris_A.size());
            mark_init_tris_gia(B_2_A_tab,B_2_A_buffer,(int)nm_A_2_B_ints,tris_A,0);
            // for each halfedge find the closest tri
            auto collect_halfedges = [&](zs::bht<int,1,int>& halfedges_tab,const dtiles_t& ints_buffer,int nm_ints) {
                pol(zs::range(nm_ints),[
                    ints_buffer = proxy<space>({},ints_buffer),
                    halfedges_tab = proxy<space>(halfedges_tab)] ZS_LAMBDA(int iti) mutable {
                        auto pair = ints_buffer.pack(dim_c<2>,"pair",iti,int_c);
                        halfedges_tab.insert(pair[0]); 
                });
            };
            auto find_closest_tri = [&](const HalfEdgeTileVec& halfedges,const PosTileVec& hverts,const TriTileVec& htris,
                        const TriTileVec& tris,const PosTileVec& tverts,
                        const zs::bht<int,1,int>& halfedges_tab,
                        zs::Vector<int>& closestTriID,
                        const dtiles_t& ints_buffer,
                        int nm_ints) {
                zs::Vector<T> min_rs{closestTriID.get_allocator(),closestTriID.size()};
                pol(zs::range(min_rs),[] ZS_LAMBDA(auto& min_r) mutable {min_r = std::numeric_limits<T>::max();});
                pol(zs::range(closestTriID),[] ZS_LAMBDA(auto& tid) mutable {tid = -1;});
                pol(zs::range(nm_ints),[
                    ints_buffer = proxy<space>({},ints_buffer),
                    min_rs = proxy<space>(min_rs),
                    halfedges_tab = proxy<space>(halfedges_tab), exec_tag] ZS_LAMBDA(int iti) mutable {
                        auto pair = ints_buffer.pack(dim_c<2>,"pair",iti,int_c);
                        // auto no = halfedges_tab.insert(pair[0]);
                        // if(no < 0)
                        auto no = halfedges_tab.query(pair[0]);
                        if(no < 0) {
                            printf("impossibe reaching here\n");
                            return;
                        }
                        auto r = ints_buffer("r",iti);
                        atomic_min(exec_tag,&min_rs[no],r);
                });
                pol(zs::range(nm_ints),[
                    ints_buffer = proxy<space>({},ints_buffer),
                    closestTriID = proxy<space>(closestTriID),
                    min_rs = proxy<space>(min_rs),
                    halfedges_tab = proxy<space>(halfedges_tab)] ZS_LAMBDA(int iti) mutable {
                        auto pair = ints_buffer.pack(dim_c<2>,"pair",iti,int_c);
                        auto no = halfedges_tab.query(pair[0]);
                        auto r = ints_buffer("r",iti);
                        if(r == min_rs[no])
                            closestTriID[no] = pair[1];
                });
            };
            zs::bht<int,1,int> halfedges_A_tab{halfedges_A.get_allocator(),(size_t)nm_A_2_B_ints};
            halfedges_A_tab.reset(pol,true);
            zs::bht<int,1,int> halfedges_B_tab{halfedges_B.get_allocator(),(size_t)nm_B_2_A_ints};
            halfedges_B_tab.reset(pol,true);
            zs::Vector<int> halfedges_A_closest_tri{halfedges_A.get_allocator(),(size_t)nm_A_2_B_ints};
            zs::Vector<int> halfedges_B_closest_tri{halfedges_B.get_allocator(),(size_t)nm_B_2_A_ints};
            collect_halfedges(halfedges_A_tab,ints_buffer_A_2_B,nm_A_2_B_ints);
            collect_halfedges(halfedges_B_tab,ints_buffer_B_2_A,nm_B_2_A_ints);
            find_closest_tri(halfedges_A,verts_A,tris_A,tris_B,verts_B,halfedges_A_tab,halfedges_A_closest_tri,ints_buffer_A_2_B,nm_A_2_B_ints);
            find_closest_tri(halfedges_B,verts_B,tris_B,tris_A,verts_A,halfedges_B_tab,halfedges_B_closest_tri,ints_buffer_B_2_A,nm_B_2_A_ints);

            auto gia_flood_vertex_region = [&](const HalfEdgeTileVec& halfedges,const zs::bht<int,1,int>& halfedges_tab,
                const TriTileVec& htris,const PosTileVec& hverts,const zs::SmallString& hxtag,const TriTileVec& tris,const PosTileVec& tverts,const zs::SmallString& txtag,
                const zs::Vector<int>& closestTriID,int its_offset,int hv_offset,
                const zs::bht<int,2,int>& ints_tab,const zs::Vector<int>& ints_tab_buffer,
                bool mark_interior) {
                    pol(zip(zs::range(halfedges_tab.size()),zs::range(halfedges_tab._activeKeys)),[
                        halfedges_tab = proxy<space>(halfedges_tab),
                        halfedges = proxy<space>({},halfedges),
                        htris = proxy<space>({},htris),
                        ringTag = proxy<space>(ringTag),
                        hxtag = hxtag,txtag = txtag,
                        hverts = proxy<space>({},hverts),its_offset = its_offset,
                        ints_tab = proxy<space>(ints_tab),ints_tab_buffer = proxy<space>(ints_tab_buffer),
                        closestTriID = proxy<space>(closestTriID),
                        mark_interior = mark_interior,hv_offset = hv_offset,gia_res = proxy<space>(gia_res),
                        tris = proxy<space>({},tris),ring_mask_width = ring_mask_width,
                        tverts = proxy<space>({},tverts)] ZS_LAMBDA(auto,const auto& key) mutable {
                            auto hi = key[0];
                            auto no = halfedges_tab.query(hi);
                            if(no >= closestTriID.size()) {
                                printf("closestTriID overflow : %d %d\n",(int)no, (int)closestTriID.size());
                                return;
                            }
                            auto cti = closestTriID[no];

                            auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                            auto htri = htris.pack(dim_c<3>,"inds",hti,int_c);
                            auto h_local_vertex_id = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
                            auto hvi = htri[h_local_vertex_id];
                            auto hv = hverts.pack(dim_c<3>,hxtag,hvi);

                            auto ctri = tris.pack(dim_c<3>,"inds",cti,int_c);
                            zs::vec<T,3> ctV[3] = {};
                            for(int i = 0;i != 3;++i)
                                ctV[i] = tverts.pack(dim_c<3>,txtag,ctri[i]);

                            auto ctnrm = LSL_GEO::facet_normal(ctV[0],ctV[1],ctV[2]);  
                            ctnrm = !mark_interior ? ctnrm : -ctnrm;

                            auto seg = hv - ctV[0];
                            // printf("testing root : %d %d %f\n",hi,cti,(float)ctnrm.dot(seg));
                            if(ctnrm.dot(seg) > 0) {
                                // printf("find_root points\n");

                                auto iti_no = ints_tab.query(vec2i{hi,cti});
                                if(iti_no >= ints_tab_buffer.size()) {
                                    printf("ints_tab_buffer overflow %d %d\n",iti_no,ints_tab_buffer.size());
                                    return;
                                }
                                auto iti = ints_tab_buffer[iti_no];
                                
                                if(iti + its_offset >= ringTag.size()) {
                                    printf("ringTag overflow %d %d\n",iti + its_offset,(int)ringTag.size());
                                    return;
                                }
                                auto ri = ringTag[iti + its_offset];
                                int ring_mask = 1 << (ri % 32);
                                int ri_offset = ri / 32;

                                atomic_or(exec_cuda,&gia_res[(hvi + hv_offset) * ring_mask_width + ri_offset],ring_mask);    
                            }
                    });
            };
            gia_flood_vertex_region(halfedges_A,halfedges_A_tab,
                tris_A,verts_A,xtag_A,tris_B,verts_B,xtag_B,
                halfedges_A_closest_tri,0,0,
                A_2_B_tab,A_2_B_buffer,B_intersect_interior);
            gia_flood_vertex_region(halfedges_B,halfedges_B_tab,
                tris_B,verts_B,xtag_B,tris_A,verts_A,xtag_A,
                halfedges_B_closest_tri,nm_A_2_B_ints,(int)verts_A.size(),
                B_2_A_tab,B_2_A_buffer,A_intersect_interior);

            zs::Vector<int> island_ring_mask{gia_res.get_allocator(),(size_t)nm_islands * (size_t)ring_mask_width};
            pol(zs::range(island_ring_mask),[] ZS_LAMBDA(auto& ring_mask) mutable {ring_mask = 0;});

            std::cout << "nm_islands : " << nm_islands << std::endl;

            // decide the ring_mask of each island
            pol(zs::range(total_nm_verts),[
                gia_res = proxy<space>(gia_res),
                exec_tag,
                // ri_offset = ri_offset,
                ring_mask_width = ring_mask_width,
                island_ring_mask = proxy<space>(island_ring_mask),
                island_buffer = proxy<space>(island_buffer)] ZS_LAMBDA(int vi) mutable {
                    auto island_id = island_buffer[vi];
                    for(int i = 0;i != ring_mask_width;++i) {
                        auto ring_mask = gia_res[vi * ring_mask_width + i];
                        atomic_or(exec_tag,&island_ring_mask[island_id * ring_mask_width + i],ring_mask);
                    }
            });
            // flood the ring_mask of each island back to vertices
            pol(zs::range(total_nm_verts),[
                gia_res = proxy<space>(gia_res),
                // ri_offset = ri_offset,
                ring_mask_width = ring_mask_width,
                island_ring_mask = proxy<space>(island_ring_mask),
                island_buffer = proxy<space>(island_buffer)] ZS_LAMBDA(int vi) mutable {
                    auto island_id = island_buffer[vi];
                    for(int i = 0;i != ring_mask_width;++i) {
                        auto ring_mask = island_ring_mask[island_id * ring_mask_width + i];
                        gia_res[vi * ring_mask_width + i] = ring_mask;
                    }
            });

            zs::Vector<int> nm_flood_A{verts_A.get_allocator(),(size_t)1};
            nm_flood_A.setVal(0);
            pol(zs::range(verts_A.size()),[
                gia_res = proxy<space>(gia_res),
                exec_tag,
                nm_flood_A = proxy<space>(nm_flood_A),
                ring_mask_width = ring_mask_width] ZS_LAMBDA(int vi) mutable {
                    for(int d = 0;d != ring_mask_width;++d) {
                        if(gia_res[vi * ring_mask_width + d] > 0) {
                            atomic_add(exec_tag,&nm_flood_A[0],(int)1);
                            return;
                        }
                    }
            });
            std::cout << "nm_flood_A : " << nm_flood_A.getVal(0) << std::endl;

            zs::Vector<int> nm_flood_B{verts_B.get_allocator(),(size_t)1};
            nm_flood_B.setVal(0);
            pol(zs::range(verts_B.size()),[
                gia_res = proxy<space>(gia_res),
                exec_tag,
                voffset = verts_A.size(),
                nm_flood_B = proxy<space>(nm_flood_B),
                ring_mask_width = ring_mask_width] ZS_LAMBDA(int vi) mutable {
                    for(int d = 0;d != ring_mask_width;++d) {
                        if(gia_res[(vi + voffset) * ring_mask_width + d] > 0) {
                            atomic_add(exec_tag,&nm_flood_B[0],(int)1);
                            return;
                        }
                    }
            });
            std::cout << "nm_flood_B : " << nm_flood_B.getVal(0) << std::endl;


            pol(zs::range(total_nm_tris),[
                tris_gia_res = proxy<space>(tris_gia_res),
                exec_tag,
                ring_mask_width = ring_mask_width,
                gia_res = proxy<space>(gia_res),
                tris_A = proxy<space>({},tris_A),
                tris_B = proxy<space>({},tris_B),
                v_offset = verts_A.size(),
                t_offset = tris_A.size()] ZS_LAMBDA(int ti) mutable mutable {
                    auto tri = ti < t_offset ? tris_A.pack(dim_c<3>,"inds",ti,int_c) : tris_B.pack(dim_c<3>,"inds",ti - t_offset,int_c) + (int)v_offset;
                    for(int d = 0;d != ring_mask_width;++d) {
                        auto tring_mask = tris_gia_res[ti * ring_mask_width + d];
                        int vring_mask = 0;
                        for(int i = 0;i != 3;++i) {
                            vring_mask |= gia_res[tri[i] * ring_mask_width + d];
                        }
                        tris_gia_res[ti * ring_mask_width + d] = vring_mask | tring_mask;
                    }
            });
            // decide the ring_mask of every tris
            return ring_mask_width;
    }


};
};

