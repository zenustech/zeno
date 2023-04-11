#include "Structures.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/LevelSet.h"
#include "zensim/math/Vec.h"
#include "zensim/math/matrix/SparseMatrix.hpp"
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>

#define RETRIEVE_OBJECT_PTRS(T, STR)                                                  \
    ([this](const std::string_view str) {                                             \
        std::vector<T *> objPtrs{};                                                   \
        if (has_input<T>(str.data()))                                                 \
            objPtrs.push_back(get_input<T>(str.data()).get());                        \
        else if (has_input<zeno::ListObject>(str.data())) {                           \
            auto &objSharedPtrLists = *get_input<zeno::ListObject>(str.data());       \
            for (auto &&objSharedPtr : objSharedPtrLists.get())                       \
                if (auto ptr = dynamic_cast<T *>(objSharedPtr.get()); ptr != nullptr) \
                    objPtrs.push_back(ptr);                                           \
        }                                                                             \
        return objPtrs;                                                               \
    })(STR);

namespace zeno {

struct ZSParticlesFixVertices : INode
{
    template<class Policy, class TileVec, class LSView>
    void fixVerts(Policy pol, TileVec& verts, LSView& lsv)
    {
        using namespace zs; 
        constexpr auto space = execspace_e::cuda;
 
        pol(range(verts.size()), 
            [verts = proxy<execspace_e::cuda>({}, verts), lsv] __device__ (int vi) mutable {
                auto x = verts.pack(dim_c<3>, "x", vi);
                if (lsv.getSignedDistance(x) < 0)
                    verts("BCorder", vi) = 3; 
            });   
    }

    void apply() override {
        using namespace zs; 
        constexpr auto space = execspace_e::cuda;
        auto pol = zs::cuda_exec().sync(true); 

        auto zspars = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles"); 
        auto zsls = get_input<ZenoLevelSet>("ZSLevelSet"); 

        match([&](const auto &ls) {
            using basic_ls_t = typename ZenoLevelSet::basic_ls_t; 
            if constexpr (is_same_v<RM_CVREF_T(ls), basic_ls_t>) {
                match([&](const auto& lsPtr) {
                    auto lsv = get_level_set_view<execspace_e::cuda>(lsPtr);
                    for (auto par : zspars)    
                        fixVerts(pol, par->getParticles(), lsv);    
                })(ls._ls); 
            } else {
                throw std::runtime_error(fmt::format("received levelset of type [{}]\n", get_var_type_str(ls)));
            }
        })(zsls->getLevelSet()); 
    }
}; 

ZENDEFNODE(ZSParticlesFixVertices, {{
                                        "ZSParticles", 
                                        "ZSLevelSet"
                                    },
                                    {},
                                    {},
                                    {"FEM"}});
}