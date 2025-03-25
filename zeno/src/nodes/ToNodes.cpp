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
        std::string mode = has_input("mode:") ? get_input2<std::string>("mode:") : "";
        std::string name = has_input("name:") ? get_input2<std::string>("name:") : "";

        //auto pp = isStatic && hasViewed ? std::make_shared<DummyObject>() : p->clone();
        auto addtoview = [&] (auto const &addtoview, zany const &p, std::string const &postfix, 
            std::string const &mode, std::string const &name) -> void {
            if (auto *lst = dynamic_cast<ListObject *>(p.get())) {
                log_info("ToView got ListObject (size={}), expanding", lst->arr.size());
                for (size_t i = 0; i < lst->arr.size(); i++) {
                    zany const &lp = lst->arr[i];
                    addtoview(addtoview, lp, postfix + ":LIST" + std::to_string(i), mode, name);
                }
                return;
            }
            auto previewclone = [&] (zany const &p) {
                if (auto methview = p->method_node("view"); methview.empty()) {
                    return p->clone();
                } else {
                    return safe_at(getThisGraph()->callTempNode(methview, {{"arg0", p}}),
                                   "ret0", "method node output");
                }
            };
            if (!p) {
                log_error("ToView: given object is nullptr");
            } else {
                auto pp = isStatic && hasViewed ? std::make_shared<DummyObject>() : previewclone(p);
                if (!pp) {
                    log_error("ToView: given object doesn't support clone, giving up");
                } else {
                    auto key = this->myname;
                    key.append(postfix);
                    key.push_back(':');
                    if (isStatic)
                        key.append("static");
                    else
                        key.append(std::to_string(getThisSession()->globalState->frameid));
                    key.push_back(':');
                    key.append(std::to_string(getThisSession()->globalState->sessionid));

                    if (!name.empty()) {
                        key = name;
                    }
                    if (!mode.empty()) {
                        auto& ud = pp->userData();
                        ud.set2("stamp_mode", mode);
                    }

                    log_debug("ToView: add view object [{}] of type {}", key, cppdemangle(typeid(*p)));
                    getThisSession()->globalComm->addViewObject(key, std::move(pp));
                    set_output2("viewid", std::move(key));
                }
            }
        };

        //在计算端，没法addViewObject，就相当于没法导cache
        if (mode != "UnChanged")
            addtoview(addtoview, p, {}, mode, name);
        hasViewed = true;
        set_output("object", std::move(p));
    }
};

ZENDEFNODE(ToView, {
    {"object", {"bool", "isStatic", "0"}},
    {"object", {"string", "viewid"}},
    {{"string", "mode", "TotalChange"},
     {"string", "name", ""}},
    {"layout"},
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
    {"deprecated"}, // internal
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
    {"deprecated"}, // internal
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
    {"layout"},
});

}
}
