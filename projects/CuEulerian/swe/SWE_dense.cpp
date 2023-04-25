#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include "../scheme.hpp"

namespace zeno {

struct SolveShallowWaterHeight : INode {
    void apply() override {
        auto grid = get_input<PrimitiveObject>("SWGrid");
        int nx, nz, halo;
        auto &ud = grid->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")) || (!ud.has<int>("halo")))
            zeno::log_error("no such UserData named '{}', '{}' or '{}'.", "nx", "nz", "halo");
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");
        halo = ud.get2<int>("halo");
        auto &pos = grid->verts;
        float dx = std::abs(pos[0][0] - pos[1][0]);
        auto dt = get_input2<float>("dt");

        auto &height = grid->verts.attr<float>("height");

        auto pol = zs::omp_exec();
        pol(zs::Collapse{nx, nz}, [&](int i, int j) {
            ;
            ;
        });

        set_output("SWGrid", std::move(grid));
    }
};

ZENDEFNODE(SolveShallowWaterHeight, {/* inputs: */
                                     {
                                         "SWGrid",
                                         {"float", "dt", "0.04"},
                                     },
                                     /* outputs: */
                                     {
                                         "SWGrid",
                                     },
                                     /* params: */
                                     {},
                                     /* category: */
                                     {"Eulerian"}});

} // namespace zeno