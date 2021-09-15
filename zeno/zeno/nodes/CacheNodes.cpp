#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/extra/evaluate_condition.h>
#include <zeno/types/MutableObject.h>


namespace zeno {

struct CachedByKey : zeno::INode {
    std::map<std::string, std::shared_ptr<IObject>> cache;

    virtual void preApply() override {
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


struct CachedIf : zeno::INode {
    bool m_done = false;

    virtual void preApply() override {
        if (has_input("keepCache")) {
            requireInput("keepCache");
            bool keep = evaluate_condition(get_input("keepCache").get());
            if (!keep) {
                m_done = false;
            }
        }
        if (!m_done) {
            INode::preApply();
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


struct CachedOnce : zeno::INode {
    bool m_done = false;

    virtual void preApply() override {
        if (!m_done) {
            INode::preApply();
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


struct MakeMutable : zeno::INode {
    virtual void apply() override {
        auto obj = get_input2("anyobj");
        auto ptr = std::make_shared<MutableObject>();
        ptr->set(std::move(obj));
        set_output("mutable", std::move(ptr));
    }
};

ZENDEFNODE(MakeMutable, {
    {"anyobj"},
    {"mutable"},
    {},
    {"control"},
});


struct UpdateMutable : zeno::INode {
    virtual void apply() override {
        auto obj = get_input2("anyobj");
        auto ptr = get_input<MutableObject>("mutable");
        ptr->set(std::move(obj));
        set_output("mutable", std::move(ptr));
    }
};

ZENDEFNODE(UpdateMutable, {
    {"mutable", "anyobj"},
    {"mutable"},
    {},
    {"control"},
});


struct ReadMutable : zeno::INode {
    virtual void apply() override {
        auto ptr = get_input<MutableObject>("mutable");
        auto obj = ptr->value;
        set_output2("anyobj", std::move(obj));
    }
};

ZENDEFNODE(ReadMutable, {
    {"mutable"},
    {"anyobj"},
    {},
    {"control"},
});


}
