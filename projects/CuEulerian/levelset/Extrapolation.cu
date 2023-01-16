#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

#include <zeno/utils/log.h>

#include "../utils.cuh"

namespace zeno {

struct ZSExtrapolateGridAttr : INode {
    void apply() override {
        auto zsGrid = get_input<ZenoSparseGrid>("Grid");
        auto sdfGrid = get_input<ZenoSparseGrid>("SDF");
        auto tag = get_input2<std::string>("Attribute");
        auto isStaggered = get_input2<bool>("Staggered");
        auto direction = get_input2<std::string>("Direction");
        auto maxIter = get_input2<int>("Iterations");

        auto &spg = zsGrid->spg;
        auto &sdf = sdfGrid->spg;

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        if (!spg.hasProperty(tag)) {
            throw std::runtime_error("the Grid doesn't have the Attribute!");
        } else {
            int m_nchns = spg.getPropertySize(tag);
            if (isStaggered && m_nchns != 3)
                throw std::runtime_error("the size of the Attribute is not 3!");
        }

        

        set_output("Grid", zsGrid);
    }
};

ZENDEFNODE(ZSExtrapolateGridAttr, {/* inputs: */
                                   {"Grid",
                                    "SDF",
                                    {"string", "Attribute", ""},
                                    {"bool", "Staggered", "0"},
                                    {"enum both positive negative", "Direction", "both"},
                                    {"int", "Iterations", "3"}},
                                   /* outputs: */
                                   {"Grid"},
                                   /* params: */
                                   {},
                                   /* category: */
                                   {"Eulerian"}});

} // namespace zeno