#pragma once

#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/Bvh.hpp"


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
                auto tri = tris.template pack<3>("inds",ti).template reinterpret_bits(int_c);
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
                    auto einds = edges.template pack<2>("inds",ei).template reinterpret_bits(int_c);
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

};