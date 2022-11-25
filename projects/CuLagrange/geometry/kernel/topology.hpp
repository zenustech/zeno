#pragma once

#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bcht.hpp"

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

        pol(zs::range(nmTris),
            [tris = proxy<space>({},tris),
                    verts = proxy<space>({},verts),
                    trisBvh = proxy<space>(trisBvh),
                    neighTag] ZS_LAMBDA(int ti) mutable {
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

    // the input mesh should be a manifold
    template<typename Pol,typename SurfTriTileVec,typename SurfEdgeTileVec,typename SurfPointTileVec,typename HalfEdgeTileVec>
    bool build_surf_half_edge(Pol& cudaPol,SurfTriTileVec& tris,SurfEdgeTileVec& lines,SurfPointTileVec& points,HalfEdgeTileVec& halfEdge) {
        using namespace zs;
        using vec2i = zs::vec<int, 2>;
		using vec3i = zs::vec<int, 3>;
        using T = typename SurfTriTileVec::value_type;

        constexpr auto space = zs::execspace_e::cuda;

        TILEVEC_OPS::fill(cudaPol,halfEdge,"to_vertex",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(cudaPol,halfEdge,"face",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(cudaPol,halfEdge,"edge",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(cudaPol,halfEdge,"opposite_he",reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(cudaPol,halfEdge,"next_he",reinterpret_bits<T>((int)-1));      
        // we might also need a space hash structure here, map from [i1,i2]->[ej]
        bcht<vec2i,int,true,universal_hash<vec2i>,32> de2fi{halfEdge.get_allocator(),halfEdge.size()};
        cudaPol(zs::range(tris.size()), [
				tris = proxy<space>({},tris),de2fi = proxy<space>(de2fi),halfEdge = proxy<space>({},halfEdge)] ZS_LAMBDA(int ti) mutable {
					auto fe_inds = tris.pack(dim_c<3>,"fe_inds",ti).reinterpret_bits(int_c);
					auto tri = tris.pack(dim_c<3>,"fp_inds",ti).reinterpret_bits(int_c);

					vec3i nos{};
					for(int i = 0;i != 3;++i) {
						if(auto no = de2fi.insert(vec2i{tri[i],tri[(i+1) % 3]});no >= 0){
							nos[i] = no;
							halfEdge("to_vertex",no) = reinterpret_bits<T>(tri[i]);
							halfEdge("face",no) = reinterpret_bits<T>(ti);
							halfEdge("edge",no) = reinterpret_bits<T>(fe_inds[i]);
							// halfEdge("next_he",no) = ti * 3 + (i+1) % 3;
						} else {
							// some error happen

						}						
					}
					for(int i = 0;i != 3;++i)
						halfEdge("next_he",nos[i]) = reinterpret_bits<T>(nos[(i+1) % 3]);
			});
			cudaPol(zs::range(halfEdge.size()),
				[halfEdge = proxy<space>({},halfEdge),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int hei) mutable {
					auto idx0 = reinterpret_bits<int>(halfEdge("to_vertex",hei));
					auto nexthei = reinterpret_bits<int>(halfEdge("next_he",hei));
					auto idx1 = reinterpret_bits<int>(halfEdge("to_vertex",nexthei));
					if(auto no = de2fi.query(vec2i{idx1,idx0});no >= 0)
						halfEdge("opposite_he",hei) = reinterpret_bits<T>(no);
					else	
						halfEdge("opposite_he",hei) = reinterpret_bits<T>((int)-1);
			});

			points.append_channels(cudaPol,{{"he_inds",1}});
			lines.append_channels(cudaPol,{{"he_inds",1}});
			tris.append_channels(cudaPol,{{"he_inds",1}});

			cudaPol(zs::range(lines.size()),[
				lines = proxy<space>({},lines,"halfedge::line_set_he_inds"),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int li) mutable {
					auto linds = lines.pack(dim_c<2>,"ep_inds",li).reinterpret_bits(int_c);
					if(auto no = de2fi.query(vec2i{linds[0],linds[1]});no >= 0){
						lines("he_inds",li) = reinterpret_bits<T>((int)no);
					}else {
						// some algorithm bug
					}
			});

			cudaPol(zs::range(tris.size()),[
				points = proxy<space>({},points),tris = proxy<space>({},tris),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int ti) mutable {
					auto tinds = tris.pack(dim_c<3>,"fp_inds",ti).reinterpret_bits(int_c);
					if(auto no = de2fi.query(vec2i{tinds[0],tinds[1]});no >= 0){
						tris("he_inds",ti) = reinterpret_bits<T>((int)no);
					}else {
						// some algorithm bug
					}

					for(int i = 0;i != 3;++i) {
						if(auto no = de2fi.query(vec2i{tinds[i],tinds[(i+1) % 3]});no >= 0){
							points("he_inds",tinds[i]) = reinterpret_bits<T>((int)no);
						}else {
							// some algorithm bug
						}						
					}
			});

            // handle the boundary points
            cudaPol(zs::range(halfEdge.size()),
                [points = proxy<space>({},points),halfEdge = proxy<space>({},halfEdge)] ZS_LAMBDA(int hei) mutable {
                    auto opposite_idx = reinterpret_bits<int>(halfEdge("opposite_he",hei));
                    if(opposite_idx >= 0)
                        return;
                    // now the halfEdge is a boundary edge
                    auto v_idx = reinterpret_bits<int>(halfEdge("to_vertex",hei));
                    points("he_inds",v_idx) = reinterpret_bits<T>((int)hei);
            });

            return true;

    }

    template<typename HalfEdgeTileVec>
    const int get_next_half_edge(int hei,const HalfEdgeTileVec& half_edges,int step = 1,bool reverse = false) {
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
        hei = reinterpret_bits<int>(half_edges("next_he",hei));
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
        res[0] = half_edge_get_another_vertex(hei,half_edges);
        for(i = 1;i != MAX_NEIGHS;++i) {
            hei = get_next_half_edge(hei,half_edges,1,false);
            res[i] = reinterpret_bits<int>(half_edges("to_vertex",hei));
            hei = reinterpret_bits<int>(half_edges("opposite_he",hei));
            if(hei == hei0 || hei < 0)
                break;
        }
        if(i == MAX_NEIGHS)
            printf("the max_one_ring_neighbor limit exceeds");

        return res;
    }

    // template<int MAX_NEIGHS,typename HalfEdgeTileVec>
    // constexpr zs::vec<int,MAX_NEIGHS> get_one_ring_neigh

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