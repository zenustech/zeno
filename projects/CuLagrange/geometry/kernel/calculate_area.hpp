#pragma once

#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "../../geometry/kernel/tiled_vector_ops.hpp"
#include "../../geometry/kernel/geo_math.hpp"


namespace zeno {
    using T = float;

    template<typename Pol,typename TTileVec,typename PTileVec>
    bool calculate_points_one_ring_area(Pol& pol,const TTileVec& tris,PTileVec& points,
            const zs::SmallString& pointAreaTag,const zs::SmallString& triAreaTag) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        if(!tris.hasProperty("fp_inds"))
            return false;
        if(!tris.hasProperty(triAreaTag))
            return false;
        if(!points.hasProperty(pointAreaTag))
            return false;

        TILEVEC_OPS::fill(pol,points,pointAreaTag,(T)0.0);

        pol(zs::range(tris.size()),
            [tris = proxy<space>({},tris),points = proxy<space>({},points),triAreaTag,pointAreaTag] __device__(int ti) mutable {
                auto subarea = tris(triAreaTag,ti) / (T)3.0;
                for(int i = 0;i != 3;++i) {
                    auto pidx = reinterpret_bits<int>(tris("fp_inds",i,ti));
                    atomic_add(exec_cuda,&points(pointAreaTag,pidx),subarea);
                }
        });
        return true;
        
    }

    template<typename Pol,typename VTileVec,typename ETileVec,typename TTileVec>
    bool calculate_lines_area(Pol& pol,const VTileVec& verts,ETileVec& lines,const TTileVec& tris,const zs::SmallString& triAreaTag,const zs::SmallString& lineAreaTag) {
        using namespace zs;
        constexpr auto space =  execspace_e::cuda;
        if(!lines.hasProperty(lineAreaTag))
            return false;
        if(!lines.hasProperty("fe_inds"))
            return false;
        if(!tris.hasProperty(triAreaTag))
            return false;

        TILEVEC_OPS::fill(pol,lines,lineAreaTag,(T)0.0);

        pol(zs::range(lines.size()),
            [verts = proxy<space>({},verts),lines = proxy<space>({},lines),tris = proxy<space>({},tris),triAreaTag,lineAreaTag] __device__(int li) mutable {
                for(int i = 0;i != 2;++i){
                    auto tidx = reinterpret_bits<int>(lines("fe_inds",i,li));
                    auto subArea = tris(triAreaTag,tidx) / (T)2.0;
                    lines(lineAreaTag,li) += subArea;
                }
        });
        return true;
    }

    template<typename Pol,typename VTileVec,typename TTileVec>
    bool calculate_tris_area(Pol& pol,const VTileVec& verts,TTileVec& tris,const zs::SmallString& xTag,const zs::SmallString& areaTag) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        if(!verts.hasProperty(xTag))
            return false;
        if(!tris.hasProperty(areaTag))
            return false;

        // std::cout << "try calculate tris area" << std::endl;
        pol(zs::range(tris.size()),
            [verts = proxy<space>({},verts),tris = proxy<space>({},tris),xTag,areaTag] __device__(int ti) mutable {
                auto inds = tris.template pack<3>("inds",ti).reinterpret_bits(int_c);
                auto a0 = verts.template pack<3>(xTag,inds[0]);
                auto a1 = verts.template pack<3>(xTag,inds[1]);
                auto a2 = verts.template pack<3>(xTag,inds[2]);
                tris(areaTag,ti) = LSL_GEO::area(a0,a1,a2);
                // tris(areaTag,ti) = (T)1.0;
        });
        // std::cout << "try calculate tris area" << std::endl;
        return true;
    }

};