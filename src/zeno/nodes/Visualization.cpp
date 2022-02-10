#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/cppdemangle.h>

namespace zeno {
namespace {

struct ToView : zeno::INode {
    virtual void complete() override {
        graph->nodesToExec.insert(myname);
    }

    virtual void apply() override {
        auto p = get_input("object");
        if (!p) {
            log_error("ToView: given object is nullptr");
        } else {
            auto pp = p->clone();
            if (!pp) {
                log_warn("ToView: given object doesn't support clone");
            } else {
                log_info("ToView: added view object of type {}", cppdemangle(typeid(*p)));
                getGlobalState()->addViewObject(pp);
            }
        }
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
