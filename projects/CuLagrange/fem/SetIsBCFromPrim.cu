#include "Structures.hpp"
#include "zensim/math/Vec.h"
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include "zensim/cuda/execution/ExecutionPolicy.cuh"

namespace zeno {
struct SetIsBCFromPrim : INode {
    using T = float; 
    using Ti = zs::conditional_t<zs::is_same_v<T, double>, zs::i64, zs::i32>;

    void apply() override {
        auto particles = get_input<ZenoParticles>("ZSParticles"); 
        auto mappingPrim = get_input<PrimitiveObject>("MappingPrim"); 

        auto pol = zs::cuda_exec();
        using namespace zs; 
        constexpr auto space = execspace_e::cuda;

        std::size_t indLen = mappingPrim->size();  
        auto& verts = particles->getParticles(); 
        auto propTags = verts.getPropertyTags(); 
        propTags.emplace_back(PropertyTag{"isBC", 1});
        propTags.emplace_back(PropertyTag{"BCtarget", 3});
        typename ZenoParticles::particles_t newVerts{verts.get_allocator(), propTags, verts.size()};  
        pol(range(verts.size()), 
            [verts = proxy<space>({}, verts), 
            newVerts = proxy<space>({}, newVerts)] __device__ (int i) mutable {
                for (int propid = 0; propid != verts._N; propid++)
                {
                    auto propOff = verts._tagOffsets[propid]; 
                    for (int chn = 0; chn < verts._tagSizes[propid]; ++chn)
                        newVerts(propOff + chn, i) = verts(propOff + chn, i); 
                }
            }); 
        zs::Vector<int> src {indLen, memsrc_e::um, 0}; 
        auto toZSVec3 = [](auto x)
        {
            return zs::vec<T, 3> {x[0], x[1], x[2]}; 
        }; 
        for (int i = 0; i < indLen; i++)    
        {
            src[i] = mappingPrim->verts.attr<int>("src_ind")[i]; 
            fmt::print("src[{}] = {}\n", i, src[i]); 
        }
        pol(range(newVerts.size()), 
            [newVerts = proxy<space>({}, newVerts)] __device__ (int i) mutable {
                newVerts("isBC", i) = 0; 
            }); 
        pol(range(indLen), 
            [src = proxy<space>(src), 
            newVerts = proxy<space>({}, newVerts)] __device__ (int i) mutable {
                newVerts("isBC", src[i]) = 1; 
            }); 
        verts = newVerts; 
    }
}; 

ZENDEFNODE(SetIsBCFromPrim, {{
                                {"ZSParticles"}, 
                                {"MappingPrim"},
                                "ZSParticles"
                           },
                           {},
                           {},
                           {"FEM"}});

} // namespace zeno