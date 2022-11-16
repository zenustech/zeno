#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

struct FilterTopology : INode {
    void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto filTopo = get_param<std::string>("topo");

        auto primOut = std::static_pointer_cast<zeno::PrimitiveObject>(prim->clone());
        if(filTopo == "lines"){
            primOut->tris.resize(0);
            primOut->quads.resize(0);
        }
        if(filTopo == "tris"){
            primOut->lines.resize(0);
            primOut->quads.resize(0);
        }
        if(filTopo == "quads"){
            primOut->lines.resize(0);
            primOut->tris.resize(0);
        }

        set_output("primOut",std::move(primOut));
    }
};

ZENDEFNODE(FilterTopology, {/* inputs: */ {
                            {"prim"},
                        },
                        /* outputs: */
                        {
                            {"primOut"},
                        },
                        /* params: */
                        {
                            {"enum lines tris quads","topo","tris"}
                        },
                        /* category: */
                        {
                            "ZSGEOMETRY",
                        }});

};