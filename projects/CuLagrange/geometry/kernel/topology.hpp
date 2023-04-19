#pragma once

#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bcht.hpp"
#include "zensim/zpc_tpls/fmt/format.h"
#include "tiled_vector_ops.hpp"

#include "zensim/math/matrix/SparseMatrix.hpp"
#include "zensim/graph/ConnectedComponents.hpp"

namespace zeno {

    constexpr int is_edge_edge_match(const zs::vec<int,2>& e0,const zs::vec<int,2>& e1) {
        if(e0[0] == e1[0] && e0[1] == e1[1])
            return 0;
        if(e0[0] == e1[1] && e0[1] == e1[0])
            return 1;
        return -1;
    }

    constexpr int is_tri_edge_match(const zs::vec<int,3>& tri,const zs::vec<int,2>& edge) {
        for(int i = 0;i != 3;++i)
            if(is_edge_edge_match(zs::vec<int,2>(tri[(i+0)%3],tri[(i+1)%3]),edge) != -1)
                return i;
        return -1;
    }

    constexpr int is_tri_point_match(const zs::vec<int,3>& tri,int pidx) {
        for(int i = 0;i != 3;++i)
            if(tri[i] == pidx)
                return i;
        return -1;
    }

    constexpr int is_edge_point_match(const zs::vec<int,2>& edge,int pidx) {
        for(int i = 0;i != 2;++i)
            if(edge[i] == pidx)
                return i;
        return -1;
    }

    template<typename Pol,typename VTileVec,typename TriTileVec,typename TetTileVec>
    bool compute_ft_neigh_topo(Pol& pol,const VTileVec& verts,TriTileVec& tris,const TetTileVec& tets,const zs::SmallString& neighTag,float bvh_thickness) {
        using namespace zs;
        using T = typename VTileVec::value_type;
        using bv_t = AABBBox<3,T>;

        if(!tris.hasProperty(neighTag) || tris.getChannelSize(neighTag) != 1)
            return false;
        
        constexpr auto space = zs::execspace_e::cuda;
        auto tetsBvh = LBvh<3,int,T>{};
        
        auto bvs = retrieve_bounding_volumes(pol,verts,tets,wrapv<4>{},bvh_thickness,"x");
        tetsBvh.build(pol,bvs);

        size_t nmTris = tris.size();
        pol(zs::range(nmTris),
            [tets = proxy<space>({},tets),
                verts = proxy<space>({},verts),
                tris = proxy<space>({},tris),
                tetsBvh = proxy<space>(tetsBvh),
                neighTag] ZS_LAMBDA(int ti) mutable {
                    auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                    tris(neighTag,ti) = zs::reinterpret_bits<float>((int)-1);
                    int nm_found = 0;
                    auto cv = zs::vec<T, 3>::zeros();
                    for(int i = 0;i != 3;++i)
                        cv += verts.pack(dim_c<3>,"x",tri[i])/(T)3.0;
                    tetsBvh.iter_neighbors(cv,[&](int ntet) {
                        // if(ti == 0)
                        //     printf("test tet[%d] and tri[%d]\n",ntet,ti);
                        if(nm_found > 0)
                            return;
                        auto tet = tets.pack(dim_c<4>,"inds",ntet).reinterpret_bits(int_c);
                        for(int i = 0;i != 3;++i){
                            bool found_idx = false;
                            for(int j = 0;j != 4;++j)
                                if(tet[j] == tri[i]){
                                    found_idx = true;
                                    break;
                                }
                            if(!found_idx)
                                return;
                        }

                        nm_found++;
                        tris(neighTag,ti) = reinterpret_bits<float>(ntet);
                    });

                    if(nm_found == 0)
                        printf("found no neighbored tet for tri[%d]\n",ti);

        });

        return true;
    }

    template<typename Pol,typename VTileVec,typename TTileVec>
    bool compute_ff_neigh_topo(Pol& pol,const VTileVec& verts,TTileVec& tris,const zs::SmallString neighTag,float bvh_thickness) {
        using namespace zs;
        using T = typename VTileVec::value_type;
        using bv_t = AABBBox<3,T>;

        if(!tris.hasProperty(neighTag) || (tris.getChannelSize(neighTag) != 3)){
            return false;
        }

        constexpr auto space = zs::execspace_e::cuda;
        auto trisBvh = LBvh<3,int,T>{};
        auto bvs = retrieve_bounding_volumes(pol,verts,tris,wrapv<3>{},bvh_thickness,"x");
        trisBvh.build(pol,bvs);

        size_t nmTris = tris.size();
        // std::cout << "CALCULATE INCIDENT TRIS " << nmTris << std::endl;
        if(!tris.hasProperty("non_manfold"))
            tris.append_channels(pol,{{"non_manifold",1}});
        

        pol(zs::range(nmTris),
            [tris = proxy<space>({},tris),
                    verts = proxy<space>({},verts),
                    trisBvh = proxy<space>(trisBvh),
                    neighTag] ZS_LAMBDA(int ti) mutable {
                tris("non_manifold",ti) = (T)0;
                auto tri = tris.template pack<3>("inds",ti).template reinterpret_bits<int>();
                tris.template tuple<3>(neighTag,ti) = zs::vec<int,3>{-1,-1,-1}.template reinterpret_bits<float>();
                for(int i = 0;i < 3; ++i) {
                    int a0 = tri[(i + 0) % 3];
                    int a1 = tri[(i + 1) % 3];

                    auto v0 = verts.template pack<3>("x",a0);
                    auto v1 = verts.template pack<3>("x",a1);
                    auto cv = (v0 + v1) / 2;

                    // if(ti == 13759)
                    //     printf("T<%d> V: %d %d %d\n",ti,tri[0],tri[1],tri[2]);

                    bool found = false;
                    int nm_found = 0;
                    trisBvh.iter_neighbors(cv,[&](int nti) {
                        if(found || nti == ti)
                            return;
                        auto ntri = tris.template pack<3>("inds",nti).template reinterpret_bits<int>();
                        for(int j = 0;j < 3;++j) {
                            int b0 = ntri[(j + 0) % 3];
                            int b1 = ntri[(j + 1) % 3];

                            if(is_edge_edge_match(zs::vec<int,2>(a0,a1),zs::vec<int,2>(b0,b1)) != -1){
                                // found = true;
                                // if(ti == 844)
                                //     printf("found incident tri<%d>[%d %d %d] : ntr<%d>[%d %d %d]\n",
                                //         ti,tri[0],tri[1],tri[2],
                                //         nti,ntri[0],ntri[1],ntri[2]);
                                nm_found++;
                                tris(neighTag,i,ti) = reinterpret_bits<T>(nti);
                                return;
                            }
                        }
                    });
                    if(nm_found > 1) {
                        // printf("found a non-manifold facet %d %d\n",ti,nm_found);
                        tris("non_manifold",ti) = (T)1.0;
                    }
                    if(nm_found == 0) {
                        // printf("found boundary facet %d\n",ti);
                    }

                }
        });

        return true;
    }

    template<typename Pol,typename VTileVec,typename PTileVec,typename TTileVec>
    bool compute_fp_neigh_topo(Pol& pol,const VTileVec& verts,const PTileVec& points,TTileVec& tris,const zs::SmallString& neighTag,float bvh_thickness) {
        using namespace zs;
        using T = typename VTileVec::value_type;
        using bv_t = AABBBox<3,T>;

        if(!tris.hasProperty(neighTag) || tris.getChannelSize(neighTag) != 3) 
            return false;

        constexpr auto space = zs::execspace_e::cuda;
        auto trisBvh = LBvh<3,int,T>{};
        auto bvs = retrieve_bounding_volumes(pol,verts,tris,wrapv<3>{},bvh_thickness,"x");
        trisBvh.build(pol,bvs);

        size_t nm_points = points.size();
        // std::cout << "nm_points : " << nm_points << std::endl;
        pol(range(nm_points),
            [   trisBvh = proxy<space>(trisBvh),
                verts = proxy<space>({},verts),
                points = proxy<space>({},points),
                tris = proxy<space>({},tris),neighTag] ZS_LAMBDA(int pi) mutable {
                    auto pidx = reinterpret_bits<int>(points("inds",pi));
                    auto p = verts.template pack<3>("x",pidx);

                    trisBvh.iter_neighbors(p,[&](int nti){
                        auto ntri = tris.template pack<3>("inds",nti).reinterpret_bits(int_c);
                        int match_idx = is_tri_point_match(ntri,pidx);

                        if(match_idx != -1) {
                            tris(neighTag,match_idx,nti) = reinterpret_bits<T>(pi);
                        }
                    });
        });

        return true;
    }

    // template<typename Pol,typename VTileVec,typename ETileVec,typename PTileVec>
    // bool compute_ep_neigh_topo(Pol& pol,const VTileVec& verts,PTileVec& points,ETileVec& edges,const zs::SmallString& neighTag,float bvh_thickness) {
    //     using namespace zs;
    //     using T = typename VTileVec::value_type;
    //     using bv_t = AABBBox<3,T>;

    //     if(!edges.hasProperty(neighTag) || edges.getChannelSize(neighTag) != 2)
    //         return false;

    //     constexpr auto space = zs::execspace_e::cuda;
    //     auto edgesBvh LBvh<3,int,T>{};
    //     auto bvs = retrieve_bounding_volumes(pol,verts,tris,wrapv<2>{},bvh_thickness,"x");
    //     edgesBvh.build(pol,bvs);

    //     pol(range(points.size()),[
    //             verts = proxy<space>({},verts),
    //             points = proxy<space>({},points),
    //             edges = proxy<space>({},edges),
    //             edgesBvh = proxy<space>(edgesBvh),
    //             neighTag,thickness = bvh_thickness] ZS_LAMBDA(int pi) mutable {
    //                 auto pidx = reinterpret_bits<int>(points("inds",pi));
    //                 auto v = verts.pack(dim_c<3>,"x",pidx);
                    

    //     });

    // }


    template<typename Pol,typename VTileVec,typename ETileVec,typename TTileVec>
    bool compute_fe_neigh_topo(Pol& pol,const VTileVec& verts,ETileVec& edges,TTileVec& tris,const zs::SmallString& neighTag,float bvh_thickness) {
        using namespace zs;
        using T = typename VTileVec::value_type;
        using bv_t = AABBBox<3,T>;

        if(!edges.hasProperty(neighTag) || edges.getChannelSize(neighTag) != 2)
            return false;

        if(!tris.hasProperty(neighTag) || tris.getChannelSize(neighTag) != 3)
            return false;

        constexpr auto space = zs::execspace_e::cuda;
        auto trisBvh = LBvh<3,int,T>{};
        auto bvs = retrieve_bounding_volumes(pol,verts,tris,wrapv<3>{},bvh_thickness,"x");
        trisBvh.build(pol,bvs);

        size_t nm_edges = edges.size();
        pol(range(nm_edges),
            [   trisBvh = proxy<space>(trisBvh),
                verts = proxy<space>({},verts),
                edges = proxy<space>({},edges),
                tris = proxy<space>({},tris),neighTag] ZS_LAMBDA(int ei) mutable {
                    auto einds = edges.template pack<2>("inds",ei).template reinterpret_bits<int>();
                    auto a0 = einds[0];
                    auto a1 = einds[1];
                    auto v0 = verts.template pack<3>("x",a0);
                    auto v1 = verts.template pack<3>("x",a1);

                    auto ec = (v0 + v1)/(T)2.0;
                    int found_nm = 0;

                    trisBvh.iter_neighbors(ec,[&](int nti) {
                        if(found_nm == 2)
                            return;
                        // if(ei == 0)
                        //     printf("incident facet : %d\n",nti);
                        auto ntri = tris.template pack<3>("inds",nti).template reinterpret_bits<int>();
                        auto match_idx = is_tri_edge_match(ntri,einds);


                        if(match_idx < 0)
                            return;

                        auto t0 = verts.template pack<3>("x",ntri[0]);
                        auto t1 = verts.template pack<3>("x",ntri[1]);
                        auto t2 = verts.template pack<3>("x",ntri[2]);
                        auto tc = (t0 + t1 + t2) / (T)3.0;
                        auto dist = (tc - ec).norm();
                        if(dist > 10.0)
                            printf("the pair<%d %d> : [%d %d], [%d %d %d] is too far with dist = %f\n",
                                ei,nti,einds[0],einds[1],ntri[0],ntri[1],ntri[2],(float)dist);

                        // if(ei == 0) {
                        //     printf("do match on[%d] : %d %d : %d %d %d\n",match_idx,
                        //         einds[0],einds[1],ntri[0],ntri[1],ntri[2]);
                        // }
                        tris(neighTag,match_idx,nti) = reinterpret_bits<T>(ei);

                        int b0 = ntri[(match_idx + 0) % 3];
                        int b1 = ntri[(match_idx + 1) % 3];

                        // printf("pair edge[%d] and facet[%d]\n",ei,nti);

                        match_idx = is_edge_edge_match(zs::vec<int,2>(a0,a1),zs::vec<int,2>(b0,b1));
                        if(match_idx < 0)
                            printf("INVALID EDGE NEIGHBOR DETECTED, CHECK ALGORITHM\n");
                        edges(neighTag,match_idx,ei) = reinterpret_bits<T>(nti); 
                        found_nm++;
                        return;
                    });  


                    if(found_nm > 2) {
                        printf("found non-manifold edge : %d %d\n",ei,found_nm);
                    }  
                    if(found_nm == 1){
                        printf("found boundary edge : %d\n",ei);
                    }                
        });

        return true;
    }

    template<typename Pol>
    void mark_disconnected_island(Pol& pol,
            const zs::Vector<zs::vec<int,2>>& topo,
            // const zs::Vector<bool>& topo_disable_buffer,
            zs::Vector<int>& fasBuffer) {
        using namespace zs;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;

        using IV = zs::vec<int,2>;

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        // setup the incident matrix
        // auto simplex_size = topo.getPropertySize("inds");
        constexpr int simplex_size = 2;

        zs::bcht<IV,int,true,zs::universal_hash<IV>,16> tab{topo.get_allocator(),topo.size() * simplex_size};
        zs::Vector<int> is{topo.get_allocator(),0},js{topo.get_allocator(),0};
        // bool use_disable = topo_disable_buffer.size() == fasBuffer.size();
        // auto build_topo = [&](const auto& eles) mutable {
        // fmt::print("initialize incident matrix topo\n");
        std::cout << "initialize incident matrix topo" << std::endl;
        pol(range(topo.size()),[
            topo = proxy<space>(topo),
            // use_disable = use_disable,
            // topo_disable_buffer = proxy<space>(topo_disable_buffer),
            // simplex_size = simplex_size,
            tab = proxy<space>(tab)] ZS_LAMBDA(int ei) mutable {
                // if(use_disable)
                //     if(topo_disable_buffer[ei])
                //         return;
                // for(int i = 0;i != simplex_size;++i) {
                //     auto a = reinterpret_bits<int>(eles("inds",(i + 0) % simplex_size,ei));
                //     auto b = reinterpret_bits<int>(eles("inds",(i + 1) % simplex_size,ei));
                //     if(a > b){
                //         auto tmp = a;
                //         a = b;
                //         b = tmp;
                //     }

                //     tab.insert(IV{a,b});
                // }
                auto a = topo[ei][0];
                auto b = topo[ei][1];
                if(a > b){
                    auto tmp = a;
                    a = b;
                    b = tmp;
                }

                tab.insert(IV{a,b});                    
        });

        auto nmEntries = tab.size();
        std::cout << "nmEntries of Topo : " << nmEntries << std::endl;
        is.resize(nmEntries);
        js.resize(nmEntries);

        pol(zip(is,js,range(tab._activeKeys)),[] ZS_LAMBDA(int &i,int &j,const auto& ij){
            i = ij[0];
            j = ij[1];
        });
        {
            int offset = is.size();
            is.resize(offset + fasBuffer.size());
            js.resize(offset + fasBuffer.size());
            pol(range(fasBuffer.size()),[is = proxy<space>(is),js = proxy<space>(js),offset] ZS_LAMBDA(int i) mutable {
                is[offset + i] = i;
                js[offset + i] = i;
            });
        }

        zs::SparseMatrix<int,true> spmat{topo.get_allocator(),(int)fasBuffer.size(),(int)fasBuffer.size()};
        spmat.build(pol,(int)fasBuffer.size(),(int)fasBuffer.size(),range(is),range(js),true_c);

        union_find(pol,spmat,range(fasBuffer));
        zs::bcht<int, int, true, zs::universal_hash<int>, 16> vtab{fasBuffer.get_allocator(),fasBuffer.size()};        
        pol(range(fasBuffer.size()),[
            vtab = proxy<space>(vtab),
            fasBuffer = proxy<space>(fasBuffer)] ZS_LAMBDA(int vi) mutable {
                auto fa = fasBuffer[vi];
                while(fa != fasBuffer[fa])
                    fa = fasBuffer[fa];
                fasBuffer[vi] = fa;
                vtab.insert(fa);
        });

        pol(range(fasBuffer.size()),[
            fasBuffer = proxy<space>(fasBuffer),vtab = proxy<space>(vtab)] ZS_LAMBDA(int vi) mutable {
                auto ancestor = fasBuffer[vi];
                auto setNo = vtab.query(ancestor);
                fasBuffer[vi] = setNo;
        });

        auto nmSets = vtab.size();
        fmt::print("{} disjoint sets in total.\n",nmSets);
    }

    template<typename Pol,typename SurfTriTileVec,typename PosTileVec>
    void eval_intersection_ring_of_surf_tris(Pol& pol,
            const SurfTriTileVec& tris0,
            const PosTileVec& verts0,
            const SurfTriTileVec& tris1,
            const PosTileVec& verts1,
            zs::Vector<zs::vec<int,2>>& intersect_buffer,
            zs::Vector<int>& ring_tag) {
        
    }

    template<typename Pol,typename SurfTriTileVec,typename PosTileVec>
    void eval_self_intersection_ring_of_surf_tris(Pol& pol,
            const SurfTriTileVec& tris,
            const PosTileVec& verts) {

    }

    // the input mesh should be a manifold
    template<typename Pol,typename TriTileVec,typename EdgeTileVec,typename PointTileVec,typename HalfEdgeTileVec>
    bool build_surf_half_edge(Pol& pol,
            TriTileVec& tris,
            EdgeTileVec& lines,
            PointTileVec& points,
            HalfEdgeTileVec& halfEdge) {
        using namespace zs;
        using vec2i = zs::vec<int, 2>;
		using vec3i = zs::vec<int, 3>;
        using T = typename TriTileVec::value_type;

        constexpr auto space = Pol::exec_tag::value;

        halfEdge.resize(tris.size() * 3);

        TILEVEC_OPS::fill(pol,halfEdge,"to_vertex",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(pol,halfEdge,"to_face",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(pol,halfEdge,"to_edge",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(pol,halfEdge,"opposite_he",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(pol,halfEdge,"next_he",reinterpret_bits<T>((int)-1));      
        // we might also need a space hash structure here, map from [i1,i2]->[ej]

        // surface tri edges' indexing the halfedge list
        bcht<vec2i,int,true,universal_hash<vec2i>,32> hetab{tris.get_allocator(),tris.size() * 3};
        // Vector<int> sfi{tris.get_allocator(),tris.size() * 3};


        bcht<vec2i,int,true,universal_hash<vec2i>,32> etab{halfEdge.get_allocator(),halfEdge.size()};
        Vector<int> sei(lines.get_allocator(),lines.size());
        pol(range(lines.size()),[etab = proxy<space>(etab),
            lines = proxy<space>({},lines),
            sei = proxy<space>(sei)] ZS_LAMBDA(int ei) mutable {
                auto edge = lines.pack(dim_c<2>,"inds",ei,int_c);
                auto a = edge[0];
                auto b = edge[1];
                if(a > b){
                    auto tmp = a;
                    a = b;
                    b = tmp;
                }
                if(int no = etab.insert(vec2i{a,b});no >= 0)
                    sei[no] = ei;
                else{
                    printf("same edge [%d %d] has been inserted twice\n",a,b);
                }
        });

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
                lines = proxy<space>({},lines),
                etab = proxy<space>(etab),
                sei = proxy<space>(sei),
                halfEdge = proxy<space>({},halfEdge),
                tris = proxy<space>({},tris)] ZS_LAMBDA(int ti) mutable {
                    auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                    vec3i hinds{};
                    for(int i = 0;i != 3;++i){
                        if(hinds[i] = hetab.insert(vec2i{tri[i],tri[(i+1)%3]});hinds[i] >= 0){
                            int no = hinds[i];
                            auto pno = spi[ptab.query(tri[i])];
                            halfEdge("to_vertex",no) = reinterpret_bits<T>((int)tri[i]);
                            points("he_inds",pno) = reinterpret_bits<T>(no);

                            // if(tri[i] < tri[(i+1)%3]){
                            auto a = tri[i];
                            auto b = tri[(i+1)%3];
                            if(a > b){
                                auto tmp = a;
                                a = b;
                                b = tmp;
                            }
                            int eno = sei[etab.query(vec2i{a,b})];
                            halfEdge("to_edge",no) = reinterpret_bits<T>(eno);
                            lines("he_inds",eno) = reinterpret_bits<T>(no);
                            // }
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
            [halfEdge = proxy<space>({},halfEdge),hetab = proxy<space>(hetab)] ZS_LAMBDA(int hi) mutable {
                auto a = reinterpret_bits<int>(halfEdge("to_vertex",hi));
                auto nxtHalfEdgeIdx = reinterpret_bits<int>(halfEdge("next_he",hi));
                auto b = reinterpret_bits<int>(halfEdge("to_vertex",nxtHalfEdgeIdx));
                auto key = vec2i{b,a};

                // printf("half_edge[%d] = [%d %d]\n",hi,a,b);

                if(int hno = hetab.query(key);hno >= 0) {
                    halfEdge("opposite_he",hi) = reinterpret_bits<T>(hno);
                }else {
                    halfEdge("opposite_he",hi) = reinterpret_bits<T>((int)-1);
                }
                
        });

        // std::cout << "nm_tris : " << tris.size() << std::endl;
        // std::cout << "nm_edges : " << lines.size() << std::endl;
        // std::cout << "nm_points : " << points.size() << std::endl;
        // std::cout << "nm_half_edges : " << halfEdge.size() << std::endl;

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

    template<typename HalfEdgeTileVec>
    constexpr int half_edge_get_another_vertex(int hei,const HalfEdgeTileVec& half_edges) {
        using namespace zs;
        // hei = reinterpret_bits<int>(half_edges("next_he",hei));
        hei = get_next_half_edge(hei,half_edges,1,false);
        return reinterpret_bits<int>(half_edges("to_vertex",hei));
    }

    // some operation with half edge structure
    template<int MAX_NEIGHS,typename HalfEdgeTileVec>
    constexpr zs::vec<int,MAX_NEIGHS> get_one_ring_neigh_points(int hei,const HalfEdgeTileVec& half_edges) {
        using namespace zs;
        auto res = zs::vec<int,MAX_NEIGHS>::uniform(-1);
        auto hei0 = hei;
        int i = 0;
        // res[0] = half_edge_get_another_vertex(hei,half_edges);
        for(i = 0;i != MAX_NEIGHS;++i) {
            res[i] = half_edge_get_another_vertex(hei,half_edges);
            auto nhei = get_next_half_edge(hei,half_edges,2,true);
            if(nhei == hei0)
                break;
            if(nhei < 0 && (i+1) < MAX_NEIGHS) {
                nhei = get_next_half_edge(hei,half_edges,2,false);
                if(nhei > 0){
                    res[i + 1] = reinterpret_bits<int>(half_edges("to_vertex",nhei));
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



    // template<typename Pol,typename VecTI,zs::enable_if_all<VecTI::dim == 1, VecTI::extent >= 2, VecTI::etent <= 4> = 0>
    // void surface_topological_coloring(Pol& pol,
    //         const zs::Vector<VecTI>& topo,
    //         zs::Vector<int>& coloring) {
    //     using namespace zs;
    //     constexpr auto simplex_degree = VecTI::extent;
    //     constexpr auto space = Pol::exec_tag::value;

    //     coloring.resize(topo.size());
    //     if(simplex_degree == 2)
    //         wire_frame_coloring(pol,topo,coloring);
    //     else if(simplex_degree == 3)
    //         triangle_mesh_coloring(pol,topo,coloring);
    //     else if(simplex_degree == 4)
    //         tet_mesh_coloring(pol,topo)
        
    // }

};