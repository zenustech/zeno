#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/LevelSetUtils.tpp"
#include "zensim/geometry/SparseGrid.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {
struct ZSRenormalizeSDF : INode {
    void apply() override {
        int nIter = get_param<int>("iterations");
        auto sdfGrid = get_input<ZenoSparseGrid>("SDF");

        auto &sdf = sdfGrid->spg;
        auto block_cnt = sdf.numBlocks();

        ZenoSparseGrid::spg_t sdf_tmp{{{"sdf", 1}}, block_cnt, zs::memsrc_e::device, 0};
        sdf_tmp._table = sdf._table;
        sdf_tmp._transform = sdf._transform;
        sdf_tmp._background = sdf._background;
        
        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;
        pol(zs::range(block_cnt * 512), [spgv = zs::proxy<space>(sdf)] __device__(int cellno) mutable {
            
        });

        set_output("SDF", sdfGrid);
    }
};

ZENDEFNODE(ZSRenormalizeSDF, {/* inputs: */
                              {"SDF"},
                              /* outputs: */
                              {"SDF"},
                              /* params: */
                              {{"int", "iterations", "5"}},
                              /* category: */
                              {"Volume"}});

} // namespace zeno