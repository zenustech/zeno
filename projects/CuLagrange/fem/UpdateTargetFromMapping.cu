#include "Structures.hpp"
#include "zensim/math/Vec.h"
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include "zensim/cuda/execution/ExecutionPolicy.cuh"

namespace zeno {
struct UpdateTargetFromMapping : INode {
    using T = float; 

    void apply() override {
        auto particles = get_input<ZenoParticles>("ZSParticles"); 
        auto targetPrim = get_input<PrimitiveObject>("TargetPrim"); 
        auto mappingPrim = get_input<PrimitiveObject>("MappingPrim"); 

        auto pol = zs::cuda_exec();
        using namespace zs; 
        constexpr auto space = execspace_e::cuda;

        std::size_t indLen = mappingPrim->size();  
        auto& verts = particles->getParticles(); 
        zs::Vector<int> src {indLen, memsrc_e::um, 0}; 
        zs::Vector<zs::vec<T, 3>> targets {indLen, memsrc_e::um, 0};
        auto toZSVec3 = [](auto x)
        {
            return zs::vec<T, 3> {x[0], x[1], x[2]}; 
        }; 
        for (int i = 0; i < indLen; i++)    
        {
            src[i] = mappingPrim->verts.attr<int>("src_ind")[i]; 
            int dst_ind = mappingPrim->verts.attr<int>("dst_ind")[i]; 
            targets[i] = toZSVec3(targetPrim->verts[dst_ind]); 
        }
        pol(range(indLen), 
            [src = proxy<space>(src), 
            targets = proxy<space>(targets), 
            verts = proxy<space>({}, verts)] __device__ (int i) mutable {
                verts.tuple(dim_c<3>, "BCtarget", src[i]) = targets[i]; 
            }); 
    }
}; 

ZENDEFNODE(UpdateTargetFromMapping, {{
                                {"ZSParticles"}, 
                                {"TargetPrim"}, 
                                {"MappingPrim"},
                                "ZSParticles"
                           },
                           {},
                           {},
                           {"FEM"}});

} // namespace zeno