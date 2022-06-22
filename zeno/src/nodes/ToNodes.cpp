#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DummyObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/utils/cppdemangle.h>
#include <zeno/core/Graph.h>

namespace zeno {
namespace {

struct ToView : zeno::INode {
    virtual void complete() override {
        log_debug("ToView: {}", myname);
        graph->nodesToExec.insert(myname);
    }

    bool hasViewed = false;

    virtual void apply() override {
        auto p = get_input("object");
        bool isStatic = has_input("isStatic") ? get_input2<bool>("isStatic") : false;
        auto addtoview = [&] (zany const &p, auto extraOps) {
            if (!p) {
                log_error("ToView: given object is nullptr");
            } else {
                auto pp = isStatic && hasViewed ? std::make_shared<DummyObject>() : p->clone();
                hasViewed = true;
                if (!pp) {
                    log_error("ToView: given object doesn't support clone, giving up");
                } else {
                    log_debug("ToView: added view object of type {}", cppdemangle(typeid(*p)));
                    auto key = this->myname;
                    key.push_back(':');
                    if (isStatic)
                        key.append("static");
                    else
                        key.append(std::to_string(getThisSession()->globalState->frameid));
                    key.push_back(':');
                    key.append(std::to_string(getThisSession()->globalState->sessionid));
                    extraOps(key);
                    getThisSession()->globalComm->addViewObject(key, std::move(pp));
                    set_output2("viewid", std::move(key));
                }
            }
        };
        // TODO: what about list of lists?
        if (auto *lst = dynamic_cast<ListObject *>(p.get())) {
            for (size_t i = 0; i < lst->arr.size(); i++) {
                addtoview(lst->arr[i], [i] (auto &key) {
                    key.append("lst-" + std::to_string(i));
                });
            }
        } else {
            addtoview(p, [] (auto &key) {});
        }
        set_output("object", std::move(p));
    }
};

ZENDEFNODE(ToView, {
    {"object", {"bool", "isStatic", "0"}},
    {"object", {"string", "viewid"}},
    {},
    {"graphtool"},
});

struct HelperMute : zeno::INode {
    virtual void apply() override {
        for (auto const &[name, _]: this->inputs) {
            if (name == "SRC") continue;//sk
            set_output(name, get_input(name));
        }
    }
};

ZENDEFNODE(HelperMute, {
    {},
    {},
    {{"string", "NOTE", "Dont-use-this-node-directly"}},
    {"graphtool"},
});

struct HelperOnce : zeno::INode {
    bool m_done = false;

    virtual void preApply() override {
        if (!m_done) {
            INode::preApply();
            m_done = true;
        }
    }

    virtual void apply() override {
        for (auto const &[name, _]: this->inputs) {
            if (name == "SRC") continue;//sk
            set_output(name, get_input(name));
        }
    }
};

ZENDEFNODE(HelperOnce, {
    {},
    {},
    {{"string", "NOTE", "Dont-use-this-node-directly"}},
    {"graphtool"},
});

struct MakeDummy : zeno::INode {
    virtual void apply() override {
        set_output("dummy", std::make_shared<DummyObject>());
    }
};

ZENDEFNODE(MakeDummy, {
    {},
    {"dummy"},
    {},
    {"graphtool"},
});

}
}
