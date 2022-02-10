#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/extra/GlobalState.h>

namespace zeno {
namespace {

struct ToView : zeno::INode {
    virtual void apply() override {
        auto p = get_input("object");
        getGlobalState()->addViewObject(p);
        set_output("object", std::move(p));
    }
};

ZENDEFNODE(ToView, {
    {"object"},
    {"object"},
    {},
    {"frame"},
});

}
}
