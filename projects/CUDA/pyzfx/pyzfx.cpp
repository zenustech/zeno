#include <zeno/core/INode.h>
#include <zeno/core/defNode.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include <cstdlib>
#include <zeno/utils/log.h>

namespace zeno {

struct PyZpcLite : INode {
    void apply() override {
        char *p;
        p = getenv("PATH");
        fmt::print("path: {}\n", p);
    }
};

ZENDEFNODE(PyZpcLite, {/* inputs: */
                       {},
                       /* outputs: */
                       {},
                       /* params: */
                       {},
                       /* category: */
                       {"PyZfx"}});

} // namespace zeno