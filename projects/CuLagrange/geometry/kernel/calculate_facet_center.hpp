#pragma once

#include "../../Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zeno {
    template<typename Pol,typename PosTileVec,typename SurfTriTileVec,typename SurfCenterTileVec,typename T = PosTileVec::value_type>
    void calculate_facet_center(Pol& pol,const PosTileVec& verts,const zs::SmallString& xTag,SurfTriTileVec& tris,SurfCenterTileVec& tri_center_buffer,const zs::SmallString& centerTag) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        pol(zs::range(tris.size()),[
                    verts = verts.begin(xTag,dim_c<3>),
                    tris = tris.begin("inds",dim_c<3>,int_c),
                    centerTagOffset = tri_center_buffer.getPropertyOffset(centerTag),
                    tri_center_buffer = proxy<space>({},tri_center_buffer)] ZS_LAMBDA(int ti) mutable {
                auto tri = tris[ti];
                tri_center_buffer.tuple(dim_c<3>,centerTagOffset,ti) = (verts[tri[0]] + verts[tri[1]] + verts[tri[2]])/static_cast<T>(3.0);
        });
    } 
};