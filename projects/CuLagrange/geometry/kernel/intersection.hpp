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

        auto cnorm = compute_average_edge_length(pol,verts,xtag,tris);
        cnorm *= 3;

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
            thickness = cnorm,
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

                // auto ohei = zs::reinterpret_bits<int>(halfedges("opposite_he",hei));

                // if(edge[0] > edge[1] && ohei >= 0)
                //     return;

                // if(edge[0] >= nm_verts || edge[1] >= nm_verts)
                //     printf("invalid edge detected : %d %d\n",edge[0],edge[1]);

                vec3 eV[2] = {};
                auto edgeCenter = vec3::zeros();
                for(int i = 0;i != 2;++i) {
                    eV[i] = verts.pack(dim_c<3>,xtag,edge[i]);
                    edgeCenter += eV[i] / (T)2.0;
                }

                auto dir = eV[1] - eV[0];
                
                auto bv = bv_t{get_bounding_box(edgeCenter - thickness,edgeCenter + thickness)};
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

                            // make sure the opposite he - tri pairs are also inserted
                            // auto opposite_hei = zs::reinterpret_bits<int>(halfedges("opposite_he",hei));
                            // if(opposite_hei >= 0) {
                            //     offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                            //     intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{opposite_hei,ti}.reinterpret_bits(float_c);
                            //     intersect_buffers.tuple(dim_c<3>,"int_points",offset) = intp;
                            // }
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



template<typename Pol,typename PosTileVec,typename TriTileVec,typename HalfEdgeTileVec,typename GIA_TILEVEC>
int do_global_self_intersection_analysis(Pol& pol,
    const PosTileVec& verts,
    const zs::SmallString& xtag,
    const TriTileVec& tris,
    HalfEdgeTileVec& halfedges,
    GIA_TILEVEC& gia_res,
    GIA_TILEVEC& tris_gia_res,
    // zs::bht<int,2,int>& conn_of_first_ring,
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


                        printf("do_global_self_intersection_analysis::impossible reaching here, the hi and ohi should both have been inserted %f %f %f\n",(float)hr,(float)ohr,(float)ints_buffer("r",isi));
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

                printf("do_global_self_intersection_analysis::impossible reaching here with broken insertion ring %f\n",(float)ints_buffer("r",isi));
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
            // if(output_intermediate_information)
            // printf("ring[%d] Size : %d\n",ri,rsize);


            int cur_ri_mask = 1 << (ri % 32);
            int ri_offset = ri / 32;

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
                cur_ri_mask = cur_ri_mask,
                ri_offset = ri_offset,
                ring_mask_width = ring_mask_width,
                halfedges = proxy<space>({},halfedges),
                // topo_tag = zs::SmallString(topo_tag),
                // dc_edge_topos = proxy<space>(dc_edge_topos),
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
                        gia_res("is_loop_vertex",corner_idx * ring_mask_width + ri_offset) = (T)1.0;
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
            nm_islands = mark_disconnected_island(pol,edge_topos,disable_points,disable_lines,island_buffer);
                // auto nn = conn_of_first_ring.size();
                // pol(zip(range(nn),zs::range(conn_of_first_ring._activeKeys)),[
                //     disable_lines = proxy<space>(disable_lines)] ZS_LAMBDA(auto i,const auto& ij) mutable {
                //         auto ji = vec2i{ij[1],ij[0]};
                //         auto no = disable_lines.query(ij);
                //         if(no >= 0) {
                //             printf("invalid topo[%d %d] detected in disable_lines %d\n",ij[0],ij[1],no);
                //         }
                //         no = disable_lines.query(ji);
                //         if(no >= 0) {
                //             printf("invalid topo[%d %d] reverse detected in disable_lines %d\n",ij[1],ij[0],no);
                //         }
                // });
                // std::cout << "size of conn_of_first_ring : " << conn_of_first_ring.size() << std::endl;
            // }
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

                auto process_potential_intersection_pairs = [&, exec_tag](int ta_i) {
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


        table_vec2i_type incidentItsTab{tris.get_allocator(), (std::size_t)nm_insts * 2};
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
                                    for(int j = 0;j != 3;++j) {
                                        v0[j] = verts.pack(dim_c<3>,xtag,tri_t0[j]);
                                        v1[j] = verts.pack(dim_c<3>,xtag,tri_t1[j]);
                                    }

                                    for(int j = 0;j != 3;++j) {
                                        e0s[j] = v0[(j + 1) % 3] - v0[j];
                                        e1s[j] = v1[(j + 1) % 3] - v1[j];
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
                                    for(int j = 0;j != 3;++j) {
                                        v0[j] = verts.pack(dim_c<3>,xtag,tri_t0[j]);
                                        v1[j] = verts.pack(dim_c<3>,xtag,tri_t1[j]);
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
                                            /// @note both these should be 'j'
                                            e1_its[nm_e1_its] = v1[j] + e1s[j] * r;
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


        zs::Vector<int> ringTag{tris.get_allocator(),(std::size_t)nm_insts};
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
            table_int_type disable_points{tris.get_allocator(),(std::size_t)rsize * 8};
            table_vec2i_type disable_lines{tris.get_allocator(),(std::size_t)rsize * 6};
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
                        // zs::vec<int,2> out_edges[3] = {};
                        auto out_edges = elm_to_edges(tri_pairs[t]);
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

        // auto cnorm_A = compute_average_edge_length(pol,verts_A,xtag_A,tris_A);
        // auto cnorm_B = compute_average_edge_length(pol,verts_B,xtag_B,tris_B);
        // auto cnorm = cnorm_A > cnorm_B ? cnorm_A : cnorm_B;
        // cnorm *= 3;

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



template<typename Pol,typename PosTileVec,typename TriTileVec,typename HETileVec>
int retrieve_intersection_tri_halfedge_info_of_two_meshes(Pol& pol,
    const PosTileVec& verts_A, const zs::SmallString& xtag_A,
    const TriTileVec& tris_A,
    const HETileVec& halfedges_A,
    const PosTileVec& verts_B, const zs::SmallString& xtag_B,
    const TriTileVec& tris_B,
    // const HETileVec& halfedges_B,
    zs::bht<int,2,int>& res) {
        using namespace zs;
        using vec2i = zs::vec<int,2>;
        using bv_t = typename ZenoParticles::lbvh_t::Box;
        using vec3 = zs::vec<T,3>;
        using table_vec2i_type = zs::bht<int,2,int>;

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};
        
        // zs::Vector<int> nmIts{verts_A.get_allocator(),1};
        // nmIts.setVal(0);

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
                tab = proxy<space>(res),
                // nmIts = proxy<space>(nmIts),
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
            auto edgeCenter_A = vec3::zeros();
            for(int i = 0;i != 2;++i) {
                eV_A[i] = verts_A.pack(dim_c<3>,xtag_A,edge_A[i]);
                edgeCenter_A += eV_A[i] / (T)2.0;
            }

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
                    // LSL_GEO::tri_ray_intersect_d<double>(eV_A[0],eV_A[1],tV_B[0],tV_B[1],tV_B[2],r);
                    if(LSL_GEO::tri_ray_intersect_d<double>(eV_A[0],eV_A[1],tV_B[0],tV_B[1],tV_B[2],r)) {
                        // auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                        auto intp = r * dir_A + eV_A[0];
                        tab.insert(vec2i{hei_A,ti_B});
                        // intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{hei_A,ti_B}.reinterpret_bits(float_c);
                        // intersect_buffers.tuple(dim_c<3>,"int_points",offset) = intp;
                        // make sure the opposite he - tri pairs are also inserted
                        // auto opposite_hei_A = zs::reinterpret_bits<int>(halfedges_A("opposite_he",hei_A));
                        if(ohei_A >= 0) {
                            // offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                            tab.insert(vec2i{ohei_A,ti_B});
                            // intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{opposite_hei_A,ti_B}.reinterpret_bits(float_c);
                            // intersect_buffers.tuple(dim_c<3>,"int_points",offset) = intp;
                        }
                    }
                }                    
            };
            tri_B_bvh.iter_neighbors(bv_A,process_potential_he_tri_intersection_pairs);
        });
        return res.size();
        // return 0;
}


template<typename DREAL,typename Pol,typename PosTileVec,typename TriTileVec,typename HETileVec>
size_t retrieve_intersection_tri_halfedge_info_of_two_meshes(Pol& pol,
    const PosTileVec& verts_A, const zs::SmallString& xtag_A,
    const TriTileVec& tris_A,
    const HETileVec& halfedges_A,
    const PosTileVec& verts_B, const zs::SmallString& xtag_B,
    const TriTileVec& tris_B,
    // const HETileVec& halfedges_B,
    zs::bht<int,2,int>& res,
    zs::Vector<DREAL>& r_res) {
        using namespace zs;
        using vec2i = zs::vec<int,2>;
        using bv_t = typename ZenoParticles::lbvh_t::Box;
        using vec3 = zs::vec<T,3>;
        using table_vec2i_type = zs::bht<int,2,int>;

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};
        
        // zs::Vector<int> nmIts{verts_A.get_allocator(),1};
        // nmIts.setVal(0);

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
                r_res = proxy<space>(r_res),
                tab = proxy<space>(res),
                // nmIts = proxy<space>(nmIts),
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
            auto edgeCenter_A = vec3::zeros();
            for(int i = 0;i != 2;++i) {
                eV_A[i] = verts_A.pack(dim_c<3>,xtag_A,edge_A[i]);
                edgeCenter_A += eV_A[i] / (T)2.0;
            }

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
                    DREAL r{};
                    // LSL_GEO::tri_ray_intersect_d<double>(eV_A[0],eV_A[1],tV_B[0],tV_B[1],tV_B[2],r);
                    if(LSL_GEO::tri_ray_intersect_d<DREAL>(eV_A[0],eV_A[1],tV_B[0],tV_B[1],tV_B[2],r)) {
                        // auto offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                        auto intp = r * dir_A + eV_A[0];
                        auto no = tab.insert(vec2i{hei_A,ti_B});
                        r_res[no] = r;
                        // intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{hei_A,ti_B}.reinterpret_bits(float_c);
                        // intersect_buffers.tuple(dim_c<3>,"int_points",offset) = intp;
                        // make sure the opposite he - tri pairs are also inserted
                        // auto opposite_hei_A = zs::reinterpret_bits<int>(halfedges_A("opposite_he",hei_A));
                        if(ohei_A >= 0) {
                            // offset = atomic_add(exec_tag,&nmIts[0],(int)1);
                            auto ono = tab.insert(vec2i{ohei_A,ti_B});
                            r_res[ono] = (DREAL)1 - r;
                            // intersect_buffers.tuple(dim_c<2>,"pair",offset) = zs::vec<int,2>{opposite_hei_A,ti_B}.reinterpret_bits(float_c);
                            // intersect_buffers.tuple(dim_c<3>,"int_points",offset) = intp;
                        }
                    }
                }                    
            };
            tri_B_bvh.iter_neighbors(bv_A,process_potential_he_tri_intersection_pairs);
        });
        return res.size();
        // return 0;
}

#if 0


#endif

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