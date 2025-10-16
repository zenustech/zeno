#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/types/ListObject_impl.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DummyObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/utils/cppdemangle.h>
#include <zeno/core/Graph.h>

namespace zeno {
namespace {

struct MakeDummy : zeno::INode {
    virtual void apply() override {
        ZImpl(set_output("dummy", std::make_unique<DummyObject>()));
    }
};

ZENDEFNODE(MakeDummy, {
    {},
    {{"object", "dummy"}},
    {},
    {"layout"},
});

}
}

