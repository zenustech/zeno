#include "../../Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>



namespace zeno {

struct ZSSetupSkinningWeight : zeno::INode {
    using T = float;

    virtual void apply() override {
        using namespace zs;
        auto zspars = get_input<ZenoParticles>("ZSParticles");
        auto prefix = get_param<std::string>("prefix");
        auto weight_dim = get_input<zeno::NumericObject>("weight_dim")->get<int>();

        auto filter_threshold = get_param<float>("filter_threshold");

        auto preserve_dim = get_param<int>("preserve_dim");

        if(preserve_dim > weight_dim)
            throw std::runtime_error("the preserve dimension should be smaller than weight dimension");


        auto prim = *zspars->prim;
        std::vector<zs::PropertyTag> tags{{prefix,preserve_dim},{"inds",preserve_dim}};
        auto skinning_buffer = typename ZenoParticles::particles_t(tags,prim.size(),zs::memsrc_e::host);

        // std::vector<int> tmp_inds(weight_dim);
        // std::vector<T> tmp_weight(weight_dim);

        constexpr auto space = zs::execspace_e::openmp;
        auto ompExec = zs::omp_exec();
        ompExec(Collapse{prim.size()},
            [prim,&prefix,skinning_buffer = proxy<space>({},skinning_buffer),weight_dim,preserve_dim] (int vi) mutable {
                std::vector<int> tmp_inds(weight_dim);
                std::vector<T> tmp_weight(weight_dim);

                // copy the buffer
                for(int i = 0;i != weight_dim;++i){
                    auto attr_name = prefix + "_" + std::to_string(i);
                    tmp_inds[i] = i;
                    tmp_weight[i] = prim.attr<float>(attr_name)[vi];
                }
                // sort the buffer using bubble sort using descending order
                for(int i = 0;i != weight_dim;++i)
                    for(int j = weight_dim-1;j != i;--j)
                        if(tmp_weight[j] > tmp_weight[j-1]){
                            // if(vi == 0)
                            //     printf("do_swap\n");
                            std::swap(tmp_weight[j],tmp_weight[j-1]);
                            std::swap(tmp_inds[j],tmp_inds[j-1]);
                            // auto tmpw = tmp_weight[j];
                            // tmp_weight[j] = tmp_weight[j+1];
                            // tmp_weight[j+1] = tmpw;

                            // auto tmpidx = tmp_inds[j];
                            // tmp_inds[j] = tmp_inds[j+1];
                            // tmp_inds[j+1] = tmpidx;
                        }
                // assign the first preserve_dim weight into skinning_buffer
                for(int i = 0;i != preserve_dim;++i){
                    skinning_buffer(prefix,i,vi) = tmp_weight[i];
                    skinning_buffer("inds",i,vi) = reinterpret_bits<float>(tmp_inds[i]);
                }   

                // if(vi == 100){
                //     printf("skinning_buffer:\n");
                //     for(int i = 0;i < preserve_dim;++i){
                //         printf("%f\t%d\n",(float)skinning_buffer(prefix,i,vi),reinterpret_bits<int>(skinning_buffer("inds",i,vi)));
                //     }
                // }             
            });

        // kind of evaluate skinning information loss
        std::vector<T> nodal_weight_sum(prim.size());
        std::fill(nodal_weight_sum.begin(),nodal_weight_sum.end(),0);
        ompExec(Collapse{prim.size()},
            [&nodal_weight_sum,skinning_buffer = proxy<space>({},skinning_buffer),preserve_dim,prefix] (int vi) mutable {
                for(int i = 0;i < preserve_dim;++i)
                    nodal_weight_sum[vi] += skinning_buffer(prefix,i,vi);
        });

        // ompExec(Collapse{prim.size()},
        //     [&nodal_weight_sum,skinning_buffer = proxy<space>({},skinning_buffer),preserve_dim,prefix] (int vi) mutable {
        //         if(vi == 0){
        //             printf("skinning_buffer:\n");
        //             for(int i = 0;i != preserve_dim;++i)
        //                 printf("%f %d\n",skinning_buffer(prefix,i,vi),reinterpret_bits<int>(skinning_buffer("inds",i,vi)));
        //         }
        // });

        // fmt::print("nodal_sum : \n");
        // for(int i = 0;i < prim.size();++i)
        //     if((nodal_weight_sum[i] - 1) > 1e-4)
        //         fmt::print("v<{}> : {}\n",i,nodal_weight_sum[i]);


        skinning_buffer = skinning_buffer.clone({zs::memsrc_e::device, 0});

        const auto& eles = zspars->getQuadraturePoints();
        auto& pars = zspars->getParticles();

        constexpr auto cspace = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        pars.append_channels(cudaPol,tags);

        cudaPol(zs::range(pars.size()),
            [pars = proxy<cspace>({},pars),skinning_buffer = proxy<cspace>({},skinning_buffer),prefix = zs::SmallString{prefix},preserve_dim] __device__(int vi) mutable {
                for(int i = 0;i != preserve_dim;++i){
                    pars(prefix,i,vi) = skinning_buffer(prefix,i,vi);
                    pars("inds",i,vi) = skinning_buffer("inds",i,vi);
                }
            });
        
        set_output("ZSParticles",zspars);
    }
};

ZENDEFNODE(ZSSetupSkinningWeight, {{"ZSParticles","weight_dim"},
                         {"ZSParticles"},
                         {{"string", "prefix", "RENAME_ME"},{"float","filter_threshold","1e-5"},{"int","preserve_dim","5"}},
                         {"ZSGeometry"}});

};