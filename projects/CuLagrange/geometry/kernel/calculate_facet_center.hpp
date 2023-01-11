#pragma once

#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zeno {
    using T = float;

    template<typename Pol,typename PosTileVec,typename SurfTriTileVec,typename SurfCenterTileVec>
    bool calculate_facet_center(Pol& pol,const PosTileVec& verts,const zs::SmallString& xTag,SurfTriTileVec& tris,SurfCenterTileVec& tri_center_buffer,const zs::SmallString& centerTag) {
        using namespace zs;
        if(!tris.hasProperty("inds") || tris.getPropertySize("inds") != 3) {
            if(!tris.hasProperty("inds"))
                fmt::print(fg(fmt::color::red),"the tris has no 'inds' channel\n");
            else if(tris.getPropertySize("inds") != 3)
                fmt::print(fg(fmt::color::red),"the tris has invalid 'inds' channel size {}\n",tris.getPropertySize("inds"));
            return false;
        }
        if(tris.size() != tri_center_buffer.size()) {
            fmt::print(fg(fmt::color::red),"the tris's size {} does not match that of tri_center_buffer {}\n",
                tris.size(),tri_center_buffer.size());
            return false;
        }

        constexpr auto space = execspace_e::cuda;
        pol(zs::range(tris.size()),
            [verts = proxy<space>({},verts),tris = proxy<space>({},tris),tri_center_buffer = proxy<space>({},tri_center_buffer),xTag,centerTag] ZS_LAMBDA(int ti) mutable {
                auto tri = tris.template pack<3>("inds",ti).reinterpret_bits(int_c);
                auto v0 = verts.template pack<3>(xTag,tri[0]);
                auto v1 = verts.template pack<3>(xTag,tri[1]);
                auto v2 = verts.template pack<3>(xTag,tri[2]);
                tri_center_buffer.template tuple<3>(centerTag,ti) = (v0 + v1 + v2)/(T)3.0;
        });

        return true;
    } 

};