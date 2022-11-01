#pragma once

#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zeno {
    using T = float;

    template<typename Pol,typename PosTileVec,typename SurfTriTileVec,typename SurfNrmTileVec>
    constexpr bool calculate_facet_normal(Pol& pol,const PosTileVec& verts,const zs::SmallString& xTag,SurfTriTileVec& tris,SurfNrmTileVec& tri_nrm_buffer,const zs::SmallString& nrmTag) {
        using namespace zs;
        if(!tris.hasProperty("inds") || tris.getChannelSize("inds") != 3) 
            return false;

        constexpr auto space = execspace_e::cuda;
        pol(zs::range(tris.size()),
            [verts = proxy<space>({},verts),tris = proxy<space>({},tris),tri_nrm_buffer = proxy<space>({},tri_nrm_buffer),xTag,nrmTag] ZS_LAMBDA(int ti) mutable {
                auto tri = tris.template pack<3>("inds",ti).template reinterpret_bits<int>();
                auto v0 = verts.template pack<3>(xTag,tri[0]);
                auto v1 = verts.template pack<3>(xTag,tri[1]);
                auto v2 = verts.template pack<3>(xTag,tri[2]);

                auto e01 = v1 - v0;
                auto e02 = v2 - v0;

                auto nrm = e02.cross(e01);
                auto nrm_norm = nrm.norm();
                if(nrm_norm < 1e-8)
                    nrm = zs::vec<T,3>::zeros();
                else
                    nrm = nrm / nrm_norm;

                tri_nrm_buffer.template tuple<3>(nrmTag,ti) = nrm;
        });

        return true;
    } 


    // template<typename Pol,typename VTileVec,typename TTileVec>
    // constexpr bool calculate_point_normal(Pol& pol,const VTileVec& verts,const TTileVec& tris,const zs::SmallString& nrmTag) {
    //     using namespace zs;
    //     if(!tris.hasProperty("inds") || tris.getChannelSize("inds") != 3) 
    //         return false;

    //     constexpr auto space = execspace_e::cuda;
    //     pol(zs::range(tris.size()),
    //         [verts = proxy<space>({},verts),tris = proxy<space>({},tris),nrmTag] ZS_LAMBDA(int ti) mutable {
    //             auto tri = tris.template pack<3>("inds",ti).template reinterpret_bits<int>();
    //             auto v0 = verts.template pack<3>("x",tris[0]);
    //             auto v1 = verts.template pack<3>("x",tris[1]);
    //             auto v2 = verts.template pack<3>("x",tris[2]);

    //             auto e01 = v1 - v0;
    //             auto e02 = v2 - v0;

    //             auto nrm = e01.cross(e02);
    //             auto nrm_norm = nrm.norm();
    //             if(nrm_norm < 1e-8)
    //                 nrm = vec<T,3>::zeros();
    //             else
    //                 nrm = nrm / nrm_norm;

    //             tris.template tuple<3>(nrmTag,ti) = nrm;
    //     });

    // }



};