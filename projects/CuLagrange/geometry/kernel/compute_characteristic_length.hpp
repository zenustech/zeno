#pragma once

#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zeno {
    using T = float;

    template<typename Pol,typename VTileVec,typename TTileVec>
    constexpr T compute_average_edge_length(Pol& pol,const VTileVec& verts,const TTileVec& elms) {
        using namespace zs;

        static_assert(is_same_v<typename VTileVec::value_type,T>,"precision not match");
        static_assert(is_same_v<typename TTileVec::value_type,T>,"precision not match");

        constexpr auto space = execspace_e::cuda;
        Vector<T> length_sum{verts.get_allocator(),1};
        length_sum.setVal((T)0);
        
        auto elm_dim = elms.getChannelSize("inds");
        auto nm_elms = elms.size();
        auto nm_edges = (elm_dim * nm_elms); 

        pol(zs::range(elms.size()),
            [length_sum = proxy<space>(length_sum),
                verts = proxy<space>({},verts),
                elms = proxy<space>({},elms),
                elm_dim] ZS_LAMBDA(int ei) mutable {
            for(int i = 0;i < elm_dim;++i) {
                int e0 = reinterpret_bits<int>(elms("inds",(i + 0) % elm_dim,ei));
                int e1 = reinterpret_bits<int>(elms("inds",(i + 1) % elm_dim,ei));

                auto a0 = verts.template pack<3>("x",e0);
                auto a1 = verts.template pack<3>("x",e1);

                auto e01 = (a0 - a1).length();
                atomic_add(exec_cuda,&length_sum[0],e01);
            }
        });

        auto res = length_sum.getVal() / (T)nm_edges;
        return res;
    }
}