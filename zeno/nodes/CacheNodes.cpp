#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/extra/evaluate_condition.h>


namespace zeno {

struct CachedByKey : zeno::INode {
    std::map<std::string, std::shared_ptr<IObject>> cache;

    virtual void doApply() override {
        requireInput("key");
        auto key = get_input<zeno::StringObject>("key")->get();
        if (auto it = cache.find(key); it != cache.end()) {
            auto value = it->second;
            set_output("output", std::move(value));
        } else {
            requireInput("input");
            auto value = get_input("input");
            cache[key] = value;
            set_output("output", std::move(value));
        }
    }

    virtual void apply() override {}
};

ZENDEFNODE(CachedByKey, {
    {"input", "key"},
    {"output"},
    {},
    {"control"},
});

struct CachedOnce : zeno::INode {
    bool m_done = false;

    virtual void doApply() override {
        if (!m_done) {
            zeno::INode::doApply();
            m_done = true;
        }
    }

    virtual void apply() override {
        auto ptr = get_input("input");
        set_output("output", std::move(ptr));
    }
};

ZENDEFNODE(CachedOnce, {
    {"input"},
    {"output"},
    {},
    {"control"},
});


struct CachedIf : zeno::INode {
    bool m_done = false;

    virtual void doApply() override {
        if (has_input("keepCache")) {
            requireInput("keepCache");
            bool keep = evaluate_condition(get_input("keepCache").get());
            if (!keep) {
                m_done = false;
            }
        }
        if (!m_done) {
            zeno::INode::doApply();
            m_done = true;
        }
    }

    virtual void apply() override {
        auto ptr = get_input("input");
        set_output("output", std::move(ptr));
    }
};

ZENDEFNODE(CachedIf, {
    {"input", "keepCache"},
    {"output"},
    {},
    {"control"},
});

}
