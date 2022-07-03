#pragma once

#include "../../Structures.hpp"

namespace zeno{
    template<int simplex_size,typename T,typename Pol,int dim = 3>
    constexpr void compute_gradient(Pol &pol,const typename ZenoParticles::particles_t &eles,
        const typename ZenoParticles::particles_t &verts,const zs::SmallString& xTag,
        const zs::TileVector<T,32>& vtemp,const zs::SmallString& tTag,
        zs::TileVector<T,32>& etemp,const zs::SmallString& gTag,zs::wrapv<simplex_size> = {}) {
            using namespace zs;
            using mat_sm1xd = zs::vec<T,simplex_size - 1,dim>;
            using mat_sm1xsm1 = zs::vec<T,simplex_size - 1,simplex_size - 1>;
            using vec_d = zs::vec<T,dim>;
            using vec_sm1 = zs::vec<T,simplex_size - 1>;

            static_assert(dim == 3 || dim == 2,"invalid dimension!\n");

            constexpr auto space = Pol::exec_tag::value;
            #if ZS_ENABLE_CUDA && defined(__CUDACC__)
                static_assert(space == execspace_e::cuda,"specified policy and compiler not match");
            #else
                static_assert(space != execspace_e::cuda,"specified policy and compiler not match");
            #endif
            if(!verts.hasProperty(xTag)){
                throw std::runtime_error("verts buffer does not contain specified channel");
            }
            if(!vtemp.hasProperty(tTag)){
                throw std::runtime_error("vtemp buffer does not contain specified channel");
            }
            if(!etemp.hasProperty(gTag)){
                throw std::runtime_error("etemp buffer does not contain specified channel");
            }
            if(eles.getChannelSize("inds") != simplex_size){
                throw std::runtime_error("the specified dimension does not match the input simplex size");
            }

            pol(range(eles.size()),
                [eles = proxy<space>({},eles),verts = proxy<space>({},verts),etemp = proxy<space>({},etemp),
                    vtemp = proxy<space>({},vtemp),xTag,tTag,gTag]
                    ZS_LAMBDA(int ei) mutable {
                        // constexpr int dim = RM_CVREF_T(dim_v)::value;
                        mat_sm1xd Dm{};
                        mat_sm1xsm1 Dm2{};
                        auto inds = eles.template pack<simplex_size>("inds",ei).template reinterpret_bits<int>();
                        vec_sm1 g{};
                        for(size_t i = 0;i != simplex_size - 1;++i){
                            // zs::row(Dm,i) = verts.pack<dim>(xTag,inds[i+1]) - verts.pack<dim>(xTag,inds[0]);
                            zs::tie(Dm(i, 0), Dm(i, 1), Dm(i, 2)) = verts.pack<dim>(xTag,inds[i+1]) - verts.pack<dim>(xTag,inds[0]);
                            g[i] = vtemp(tTag,inds[i+1]) - vtemp(tTag,inds[0]);
                        }
                        Dm2 = Dm * Dm.transpose();
                        Dm2 = zs::inverse(Dm2);
                        etemp.template tuple<dim>(gTag,ei) = Dm.transpose() * Dm2 * g;
            });
    }
};