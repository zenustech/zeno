#pragma once

#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "topology.hpp"

namespace zeno { namespace COLLISION_UTILS { 

    using T = float;

    template<typename Pol,typename PosTileVec,typename SurfLineTileVec,typename BisectorTileVec,typename SurfNormalTileVec>
    bool calculate_cell_bisector_normal(Pol& pol,
            const PosTileVec& verts,const zs::SmallString& xTag,
            const SurfLineTileVec& lines,
            const SurfNormalTileVec& tri_nrm_buffer,const zs::SmallString& triNrmTag,
            BisectorTileVec& bis_nrm_buffer,const zs::SmallString& biNrmTag) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        if(!verts.hasProperty(xTag))
            return false;
        if(!lines.hasProperty("fe_inds"))
            return false;
        if(!tri_nrm_buffer.hasProperty(triNrmTag))
            return false;
        if(!bis_nrm_buffer.hasProperty(biNrmTag))
            return false;

        pol(range(lines.size()),
            [verts = proxy<space>({},verts),
                tri_nrm_buffer = proxy<space>({},tri_nrm_buffer),
                bis_nrm_buffer = proxy<space>({},bis_nrm_buffer),
                lines = proxy<space>({},lines),
                xTag,triNrmTag,biNrmTag] ZS_LAMBDA(int ei) mutable {
                    auto e_inds = lines.template pack<2>("inds",ei).template reinterpret_bits<int>();
                    auto fe_inds = lines.template pack<2>("fe_inds",ei).template reinterpret_bits<int>();
                    auto n0 = tri_nrm_buffer.template pack<3>(triNrmTag,fe_inds[0]);
                    auto n1 = tri_nrm_buffer.template pack<3>(triNrmTag,fe_inds[1]);

                    auto ne = (n0 + n1).normalized();
                    auto e0 = verts.template pack<3>(xTag,e_inds[0]);
                    auto e1 = verts.template pack<3>(xTag,e_inds[1]);
                    auto e10 = e1 - e0;

                    bis_nrm_buffer.template tuple<3>(biNrmTag,ei) = e10.cross(ne).normalized();
        });        

        return true;
    }

    template<typename SurfLineTileVec,typename SurfTriTileVec,typename BisectorTileVec>
    constexpr zs::vec<T,3> get_bisector_orient(const SurfLineTileVec& lines,const SurfTriTileVec& tris,
        const BisectorTileVec& bis_nrm_buffer,const zs::SmallString& bisector_normal_tag,
        int cell_id,int bisector_id) {
            using namespace zs;
            auto inds = tris.template pack<3>("inds",cell_id).reinterpret_bits(int_c);
            auto eidx = reinterpret_bits<int>(tris("fe_inds",bisector_id,cell_id));

            auto line = lines.template pack<2>("inds",eidx).reinterpret_bits(int_c);
            auto tline = zs::vec<int,2>(inds[bisector_id],inds[(bisector_id+1)%3]);

            auto res = bis_nrm_buffer.template pack<3>(bisector_normal_tag,eidx);
            if(is_edge_edge_match(line,tline) == 1)
                res =  (T)-1 * res;
            return res;
    }

    template<typename PosTileVec,typename SurfLineTileVec,typename SurfTriTileVec,typename BisectorTileVec,typename SurfNrmTileVec>
    constexpr int is_inside_the_cell(const PosTileVec& verts,const zs::SmallString& x_tag,
            const SurfLineTileVec& lines,const SurfTriTileVec& tris,
            const SurfNrmTileVec& surf_nrm_buffer,const zs::SmallString& triNrmTag,
            const BisectorTileVec& bis_nrm_buffer,const zs::SmallString& bisector_normal_tag,
            int cell_id,const zs::vec<T,3>& p,T inset,T offset,T& dist) {
        using namespace zs;

        auto fe_inds = tris.template pack<3>("fe_inds",cell_id).reinterpret_bits(int_c);
        auto nrm = surf_nrm_buffer.template pack<3>("nrm",cell_id);
        
        auto inds = tris.template pack<3>("inds",cell_id).reinterpret_bits(int_c);
        auto seg = p - verts.template pack<3>(x_tag,inds[0]);    

        dist = seg.dot(nrm);
        if(dist < -(inset + 1e-6) || dist > (offset + 1e-6))
            return 1;
        
        for(int i = 0;i != 3;++i) {
            auto bisector_normal = get_bisector_orient(lines,tris,bis_nrm_buffer,bisector_normal_tag,cell_id,i);
            seg = p - verts.template pack<3>(x_tag,inds[i]);
            if(bisector_normal.dot(seg) < 0)
                return 2;
        }

        return 0;
    } 

    template<typename PosTileVec,typename SurfLineTileVec,typename SurfTriTileVec,typename BisectorTileVec,typename SurfNrmTileVec>
    constexpr int is_inside_the_cell(const PosTileVec& verts,const zs::SmallString& x_tag,
            const SurfLineTileVec& lines,const SurfTriTileVec& tris,
            const SurfNrmTileVec& surf_nrm_buffer,const zs::SmallString& triNrmTag,
            const BisectorTileVec& bis_nrm_buffer,const zs::SmallString& bisector_normal_tag,
            int cell_id,const zs::vec<T,3>& p,T inset,T offset) {
        using namespace zs;

        T dist{};
        return is_inside_the_cell(verts,x_tag,lines,tris,surf_nrm_buffer,triNrmTag,bis_nrm_buffer,bisector_normal_tag,cell_id,p,inset,offset,dist);
    } 


};
};