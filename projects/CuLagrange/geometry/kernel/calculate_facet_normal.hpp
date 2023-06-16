#pragma once

#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "geo_math.hpp"

namespace zeno {
    using T = float;

    template<typename Pol,typename PosTileVec,typename SurfTriTileVec,typename SurfNrmTileVec>
    bool calculate_facet_normal(Pol& pol,const PosTileVec& verts,const zs::SmallString& xTag,const SurfTriTileVec& tris,SurfNrmTileVec& tri_nrm_buffer,const zs::SmallString& nrmTag) {
        // std::cout << "calculate facet normal" << std::endl;
        
        using namespace zs;

        if(!tris.hasProperty("inds")){
            std::cout << "the tris has no 'inds' channel\n" << std::endl;
            fmt::print(fg(fmt::color::red),"the tris has no 'inds' channel\n");
            return false;
        }
        if(tris.getPropertySize("inds") != 3){
            std::cout << "the tris has invalid 'inds' channel size {}\n" << std::endl;
            fmt::print(fg(fmt::color::red),"the tris has invalid 'inds' channel size {}\n",tris.getPropertySize("inds"));
            return false;
        }
        if(tris.size() != tri_nrm_buffer.size()) {
            std::cout << "invalid tris and triNrms" << std::endl;
            fmt::print(fg(fmt::color::red),"the tris's size {} does not match that of tri_nrm_buffer {}\n",
                tris.size(),tri_nrm_buffer.size());
            return false;
        }

        if(!tri_nrm_buffer.hasProperty(nrmTag)) {
            // std::cout << "the tri_nrm_buffer has no " << nrmTag  << " channel" << std::endl;
            fmt::print(fg(fmt::color::red),"the tri_nrm_buffer has no {} channel\n",nrmTag);
            return false;
        }

        if(tri_nrm_buffer.getPropertySize(nrmTag) != 3) {
            // std::cout << "the tri_nrm_buffer has no " << nrmTag  << " channel" << std::endl;
            fmt::print(fg(fmt::color::red),"the tri_nrm_buffer has invalid {} channel, which should be vec3\n",nrmTag);
            return false;
        }


        constexpr auto space = execspace_e::cuda;
        pol(zs::range(tris.size()),
            [verts = proxy<space>({},verts),tris = proxy<space>({},tris),tri_nrm_buffer = proxy<space>({},tri_nrm_buffer),xTag,nrmTag] ZS_LAMBDA(int ti) mutable {
                auto tri = tris.template pack<3>("inds",ti).reinterpret_bits(int_c);
                auto v0 = verts.template pack<3>(xTag,tri[0]);
                auto v1 = verts.template pack<3>(xTag,tri[1]);
                auto v2 = verts.template pack<3>(xTag,tri[2]);

                // auto e01 = v1 - v0;
                // auto e02 = v2 - v0;

                // auto nrm = e01.cross(e02);
                // auto nrm_norm = nrm.norm();
                // if(nrm_norm < 1e-8)
                //     nrm = zs::vec<T,3>::zeros();
                // else
                //     nrm = nrm / nrm_norm;

                tri_nrm_buffer.template tuple<3>(nrmTag,ti) = LSL_GEO::facet_normal(v0,v1,v2);
        });

        return true;
    } 



};