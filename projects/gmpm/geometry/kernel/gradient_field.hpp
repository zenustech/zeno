#pragma once

#include "../../Structures.hpp"

namespace zeno{
    template<typename T,typename Pol,int dim = 3>
    constexpr void compute_gradient(Pol &pol,const typename ZenoParticles::particles_t &eles,
        const typename ZenoParticles::particles_t &verts,const zs::SmallString& xTag,
        const zs::TileVector<T,32>& vtemp,const zs::SmallString& tTag,
        zs::TileVector<T,32>& etemp,const zs::SmallString& gtag,zs::wrapv<dim> = {}) {
            using namespace zs;
            using mat = zs::vec<T,dim,dim>;
            using vec = zs::vec<T,dim>;

            static_assert(dim == 3 || dim == 2,"invalid dimension!\n");
            constexpr auto dimp1 = dim + 1;

            constexpr auto space = Pol::exec_tag::value;
            #if ZS_ENABLE_CUDA && defined(__CUDACC__)
                static_assert(space == execspace_e::cuda,"specified policy and compiler not match");
            #else
                static_assert(space != execspace_e::cuda,"specified policy and compiler not match");
            #endif

            if(!verts.hasProperty(xTag)){
                // printf("verts buffer does not contain specified channel\n");
                throw std::runtime_error("verts buffer does not contain specified channel");
            }
            if(!vtemp.hasProperty(tTag)){
                throw std::runtime_error("vtemp buffer does not contain specified channel");
            }
            if(!etemp.hasProperty(gtag)){
                // printf("etemp buffer does not contain specified channel\n");
                throw std::runtime_error("etemp buffer does not contain specified channel");
            }
            if(eles.getChannelSize() != dimp1){
                throw std::runtime_error("the specified dimension does not match the input simplex size");
            }
            etemp.append_channels(pol,{{gtag,dim}});

            pol(range(eles.size()),
                [eles = proxy<space>({},eles),verts = proxy<space>({},verts),etemp = proxy<space>({},etemp),
                    vtemp = proxy<space>({},vtemp),dim_v = wrapv<dim>{},xTag,tTag,gTag]
                    ZS_LAMBDA(int ei) mutable {
                        constexpr int dim = RM_CVREF_T(dim_v)::value;
                        constexpr int dimp1 = dim + 1;
                        mat Dm = mat{(T)0.0};
                        auto inds = eles.template pack<dimp1>("inds",ei).template reinterpret_bits<int>();
                        vec g{};
                        for(size_t i = 0;i < dim;++i){
                            zs::row(Dm,i) = verts.pack<dim>(xTag,inds[i+1]) - verts.pack<dim>(xTag,inds[0]);
                            g[i] = vtemp(tTag,inds[i+1]) - vtemp(tTag,inds[0]);
                        }
                        Dm = zs::inverse(Dm);
                        etemp.tuple<dim>(gTag,ei) = Dm * g;
            });
    }
};