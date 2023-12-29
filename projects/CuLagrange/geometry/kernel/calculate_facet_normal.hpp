#pragma once

#include "../../Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "geo_math.hpp"

namespace zeno {
    template<typename Pol,typename PosTileVec,typename SurfTriTileVec,typename SurfNrmTileVec>
    void calculate_facet_normal(Pol& pol,const PosTileVec& verts,const zs::SmallString& xTag,const SurfTriTileVec& tris,SurfNrmTileVec& tri_nrm_buffer,const zs::SmallString& nrmTag) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        pol(zs::range(tris.size()),[
                    verts = verts.begin(xTag,dim_c<3>),
                    tris = tris.begin("inds",dim_c<3>,int_c),
                    nrmTagOffset = tri_nrm_buffer.getPropertyOffset(nrmTag),
                    tri_nrm_buffer = proxy<space>({},tri_nrm_buffer)] ZS_LAMBDA(int ti) mutable {
                auto tri = tris[ti];
                tri_nrm_buffer.tuple(dim_c<3>,nrmTagOffset,ti) = LSL_GEO::facet_normal(verts[tri[0]],verts[tri[1]],verts[tri[2]]);
        });
    }
};