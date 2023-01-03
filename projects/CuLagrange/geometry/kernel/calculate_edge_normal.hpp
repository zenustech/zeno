#pragma once

#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "tiled_vector_ops.hpp"

namespace zeno {
    using T = float;

    template<typename Pol,typename LineTileVec,typename SurfTriNrmTileVec,typename SurfLineNrmTileVec>
    bool calculate_edge_normal_from_facet_normal(Pol& pol,const LineTileVec& lines,
        const SurfTriNrmTileVec& ttemp,const zs::SmallString& srcTag,
        SurfLineNrmTileVec& etemp,const zs::SmallString& dstTag) {
            using namespace zs;
            if(!ttemp.hasProperty(srcTag) || ttemp.getPropertySize(srcTag) != 3){
                fmt::print(fg(fmt::color::red),"the input triNrmTileVec has no valid {} normal channel\n",srcTag);
                return false;
            }
            if(!etemp.hasProperty(dstTag) || etemp.getPropertySize(dstTag) != 3) {
                fmt::print(fg(fmt::color::red),"the input lineNrmTileVec has no valid {} normal channel\n",dstTag);
                return false;
            }

            TILEVEC_OPS::assemble_from<3,2>(pol,ttemp,srcTag,etemp,dstTag,"fe_inds");
            TILEVEC_OPS::normalized_channel<3>(pol,etemp,dstTag);
    }

};