#pragma once

#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bcht.hpp"
#include "zensim/zpc_tpls/fmt/format.h"
#include "tiled_vector_ops.hpp"

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
                        printf("found a non-manifold facet %d %d\n",ti,nm_found);
                        tris("non_manifold",ti) = (T)1.0;
                    }
                    if(nm_found == 0) {
                        printf("found boundary facet %d\n",ti);
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


    // void c

    // the input mesh should be a manifold
    template<typename Pol,typename SurfTriTileVec,typename SurfEdgeTileVec,typename SurfPointTileVec,typename HalfEdgeTileVec>
    bool build_surf_half_edge(Pol& cudaPol,SurfTriTileVec& tris,SurfEdgeTileVec& lines,SurfPointTileVec& points,HalfEdgeTileVec& halfEdge) {
        using namespace zs;
        using vec2i = zs::vec<int, 2>;
		using vec3i = zs::vec<int, 3>;
        using T = typename SurfTriTileVec::value_type;

        constexpr auto space = zs::execspace_e::cuda;

        TILEVEC_OPS::fill(cudaPol,halfEdge,"to_vertex",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(cudaPol,halfEdge,"to_face",reinterpret_bits<T>((int)-1));
        // TILEVEC_OPS::fill(cudaPol,halfEdge,"edge",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(cudaPol,halfEdge,"opposite_he",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(cudaPol,halfEdge,"next_he",reinterpret_bits<T>((int)-1));      
        // we might also need a space hash structure here, map from [i1,i2]->[ej]

        // surface tri edges' indexing the halfedge list
        bcht<vec2i,int,true,universal_hash<vec2i>,32> hetab{tris.get_allocator(),tris.size() * 3};
        // bcht<vec2i,int,true,universal_hash<vec2i>,32> etab{lines.get_allocator(),lines.size()};
        Vector<int> sfi{tris.get_allocator(),tris.size() * 3};
        // surface points' indexing one of the connected half-edge
        bcht<int,int,true,universal_hash<int>,32> ptab{points.get_allocator(),points.size()};
        Vector<int> spi{points.get_allocator(),points.size()};

        bcht<vec2i,int,true,universal_hash<vec2i>,32> de2fi{halfEdge.get_allocator(),halfEdge.size()};
        Vector<int> sei(lines.get_allocator(),lines.size());

        cudaPol(range(points.size()),
            [ptab = proxy<space>(ptab),points = proxy<space>({},points),spi = proxy<space>(spi)] ZS_LAMBDA(int pi) mutable {
                auto pidx = reinterpret_bits<int>(points("inds",pi));
                if(int no = ptab.insert(pidx);no >= 0)
                    spi[no] = pi;
        });
        // cudaPol(range(lines.size()),
        //     [estab = proxy<space>(estab),lines = proxy<space>({},lines),sei = proxy<space>(sei)] ZS_LAMBDA(int li) mutable {
        //         auto l = lines.pack(dim_c<2>,"inds",li).reinterpret_bits(int_c);
        //         if(no = estab.insert(vec2i{l[0],l[1]});no >= 0)
        //             sei[no] = li;
        // });
        // initialize surface tri <-> halfedge connectivity
        cudaPol(range(tris.size()),
            [hetab = proxy<space>(hetab),
                ptab = proxy<space>(ptab),
                spi = proxy<space>(spi),
                points = proxy<space>({},points),
                halfEdge = proxy<space>({},halfEdge),
                sfi = proxy<space>(sfi),
                tris = proxy<space>({},tris)] ZS_LAMBDA(int ti) mutable {
                    auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                    vec3i hinds{};
                    for(int i = 0;i != 3;++i){
                        if(hinds[i] = hetab.insert(vec2i{tri[i],tri[(i+1)%3]});hinds[i] >= 0){
                            auto no = hinds[i];
                            if(i == 0)
                                tris("he_inds",ti) = reinterpret_bits<T>(no);
                            auto pno = ptab.query(tri[i]);
                            halfEdge("to_vertex",no) = reinterpret_bits<T>(spi[pno]);
                            halfEdge("to_face",no) = reinterpret_bits<T>(ti);
                            points("he_inds",spi[pno]) = reinterpret_bits<T>(no);
                        }else {
                            auto no = hinds[i];
                            int pid = hetab.query(vec2i{tri[i],tri[(i+1)%3]});
                            int oti = sfi[pid];
                            printf("the same directed edge <%d %d> has been inserted twice! original sfi[%d %d] = %d, cur: %d <%d %d %d>\n",
                                tri[i],tri[(i+1)%3],no,pid,oti,ti,tri[0],tri[1],tri[2]);
                        }
                    }

                    for(int i = 0;i != 3;++i)
                        halfEdge("next_he",hinds[i]) = hinds[(i+1) % 3];
        });

        cudaPol(range(halfEdge.size()),
            [halfEdge = proxy<space>({},halfEdge),hetab = proxy<space>(hetab)] ZS_LAMBDA(int hi) mutable {
                auto curPIdx = reinterpret_bits<int>(halfEdge("to_vertex",hi));
                auto nxtHalfEdgeIdx = reinterpret_bits<int>(halfEdge("next_he",hi));
                auto nxtPIdx = reinterpret_bits<int>(halfEdge("to_vertex",reinterpret_bits<int>(halfEdge("to_vertex",nxtHalfEdgeIdx))));
                auto key = vec2i{nxtPIdx,curPIdx};

                if(auto hno = hetab.query(key);hno >= 0) {
                    halfEdge("opposite_he",hi) = reinterpret_bits<T>(hno);
                }else {
                    halfEdge("opposite_he",hi) = reinterpret_bits<T>(-1);
                }
                
        });



        // // building the de2fi hash map
        // cudaPol(zs::range(tris.size()), [
		// 		tris = proxy<space>({},tris,"tris_access_fe_fp_inds"),de2fi = proxy<space>(de2fi),halfEdge = proxy<space>({},halfEdge)] ZS_LAMBDA(int ti) mutable {
		// 			auto fe_inds = tris.pack(dim_c<3>,"fe_inds",ti).reinterpret_bits(int_c);
		// 			auto fp_inds = tris.pack(dim_c<3>,"fp_inds",ti).reinterpret_bits(int_c);

		// 			vec3i nos{};
		// 			for(int i = 0;i != 3;++i) {
		// 				if(auto no = de2fi.insert(vec2i{fp_inds[i],fp_inds[(i+1) % 3]});no >= 0 && no < halfEdge.size()){
		// 					nos[i] = no;
		// 					halfEdge("to_vertex",no) = reinterpret_bits<T>(fp_inds[i]);
		// 					halfEdge("face",no) = reinterpret_bits<T>(ti);
		// 					halfEdge("edge",no) = reinterpret_bits<T>(fe_inds[i]);
		// 					// halfEdge("next_he",no) = ti * 3 + (i+1) % 3;
		// 				} else
        //                     printf("invalid de2fi query : %d\n",no);				
		// 			}
		// 			for(int i = 0;i != 3;++i){
        //                 if(nos[i] >= 0 && nos[i] < halfEdge.size())
		// 				    halfEdge("next_he",nos[i]) = reinterpret_bits<T>(nos[(i+1) % 3]);
        //                 else
        //                     printf("invalid de2fi query : %d\n",nos[i]);
        //             }
		// });
        // fmt::print("build success state: {}\n", de2fi._buildSuccess.getVal());
        // cudaPol(zs::range(halfEdge.size()),
        //     [halfEdge = proxy<space>({},halfEdge),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int hei) mutable {
        //         auto idx0 = reinterpret_bits<int>(halfEdge("to_vertex",hei));
        //         auto nexthei = reinterpret_bits<int>(halfEdge("next_he",hei));
        //         auto idx1 = reinterpret_bits<int>(halfEdge("to_vertex",nexthei));
        //         if(auto no = de2fi.query(vec2i{idx1,idx0});no >= 0)
        //             halfEdge("opposite_he",hei) = reinterpret_bits<T>(no);
        //         else{	
        //             printf("detected boundary half edge : he[%d] : %d %d\n",hei,idx0,idx1);
        //             halfEdge("opposite_he",hei) = reinterpret_bits<T>((int)-1);
        //         }
        // });

        // cudaPol(zs::range(lines.size()),[
        //     lines = proxy<space>({},lines,"halfedge::line_set_he_inds"),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int li) mutable {
        //         auto ep_inds = lines.pack(dim_c<2>,"ep_inds",li).reinterpret_bits(int_c);
        //         if(auto no = de2fi.query(vec2i{ep_inds[0],ep_inds[1]});no >= 0){
        //             lines("he_inds",li) = reinterpret_bits<T>((int)no);
        //         }else {
        //             // some algorithm bug
        //         }
        // });

        // // std::cout << "problematic_fp_inds_size : " << tris.getPropertySize("fp_inds") << std::endl;

        // cudaPol(zs::range(tris.size()),[
        //     points = proxy<space>({},points),tris = proxy<space>({},tris,"tris_access_fp_inds"),de2fi = proxy<space>(de2fi)] __device__(int ti) mutable {
        //         auto fp_inds = tris.pack(dim_c<3>,"fp_inds",ti).reinterpret_bits(int_c);
        //         // if(auto no = de2fi.query(vec2i{fp_inds[0],fp_inds[1]});no >= 0){
        //         //     tris("he_inds",ti) = reinterpret_bits<T>((int)no);
        //         // }else {
        //         //     // some algorithm bug
        //         //     printf("invalid de2fi query %d\n",no);
        //         //     return;
        //         // }

        //         // for(int i = 0;i != 3;++i) {
        //         //     if(auto no = de2fi.query(vec2i{fp_inds[i],fp_inds[(i+1) % 3]});no >= 0){
        //         // //         if(fp_inds[i] >= 0 && fp_inds[i] < points.size()){
        //         // //             // points("he_inds",fp_inds[i]) = reinterpret_bits<T>((int)no);
        //         // //         }else
        //         // //             printf("invalid fp_inds[%d] = %d with points.size() = %d\n",i,fp_inds[i],(int)points.size());
        //         //     }else {
        //         // //         // some algorithm bug
        //         //     }						
        //         // }

        //         // {
        //         //     auto tmp = vec2i{fp_inds[0],fp_inds[1]};
        //         //     auto no_test = de2fi.query(tmp);
        //         // }
        //         // {
        //             for(int i = 0;i != 3;++i) {
        //                 if(auto no = de2fi.query(vec2i{fp_inds[i],fp_inds[(i+1) % 3]});no >= 0){
        //                     if(i == 0) {
        //                         tris("he_inds",ti) = reinterpret_bits<T>((int)no);
        //                     }
        //                     if(fp_inds[i] >= 0 && fp_inds[i] < points.size()){
        //                         points("he_inds",fp_inds[i]) = reinterpret_bits<T>((int)no);
        //                     }else
        //                         printf("invalid fp_inds[%d] = %d with points.size() = %d\n",i,fp_inds[i],(int)points.size());

        //                 }else {

        //                 }
        //             }
        //         // }
        // });

        // // handle the boundary points
        // cudaPol(zs::range(halfEdge.size()),
        //     [points = proxy<space>({},points),halfEdge = proxy<space>({},halfEdge)] ZS_LAMBDA(int hei) mutable {
        //         auto opposite_idx = reinterpret_bits<int>(halfEdge("opposite_he",hei));
        //         if(opposite_idx >= 0)
        //             return;
        //         // now the halfEdge is a boundary edge
        //         auto v_idx = reinterpret_bits<int>(halfEdge("to_vertex",hei));
        //         points("he_inds",v_idx) = reinterpret_bits<T>((int)hei);
        // });

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
            res[i] = reinterpret_bits<int>(half_edges("edge",hei));
            nhei = get_next_half_edge(hei,half_edges,2,true);
            if(hei0 == nhei || nhei == -1)
                break;
            hei = nhei;
        }
        if(i < MAX_NEIGHS-1 && nhei == -1) {
            ++i;
            hei = get_next_half_edge(hei,half_edges,2,false);
            res[i] = reinterpret_bits<int>(half_edges("edge",hei));
        }
        return res;
    }

    template<int MAX_NEIGHS,typename HalfEdgeTileVec>
    constexpr zs::vec<int,MAX_NEIGHS> get_one_ring_neigh_tris(int hei,const HalfEdgeTileVec& half_edges) {
        using namespace zs;
        auto res = zs::vec<int,MAX_NEIGHS>::uniform(-1);
        auto hei0 = hei;
        int i = 0;
        res[0] = reinterpret_bits<int>(half_edges("face",hei));
        for(int i = 1;i != MAX_NEIGHS;++i) {
            hei = get_next_half_edge(hei,half_edges,1,true);
            if(hei == hei0 || hei < 0)
                break;
            res[i] = reinterpret_bits<int>(half_edges("face",hei));
        }

        if(i == MAX_NEIGHS)
            printf("the max_one_ring_neighbor limit exceeds");

        return res;

    }

};