#include <zen/zen.h>
#include <zen/ListObject.h>
#include <zen/NumericObject.h>
#include <cassert>


struct ListLength : zen::INode {
    virtual void apply() override {
        auto list = get_input<zen::ListObject>("list");
        auto ret = std::make_shared<zen::NumericObject>();
        ret->set<int>(list->arr.size());
        set_output("length", std::move(ret));
    }
};

ZENDEFNODE(ListLength, {
    {"list"},
    {"index"},
    {},
    {"list"},
});


struct BeginFor;

struct BeginForObject : zen::IObject {
    BeginFor *that;
};

struct BeginFor : zen::INode {
    int m_index = 0, m_count = 1;

    bool isContinue() const {
        return m_index < m_count;
    }

    virtual void apply() override {
        auto count = get_input<zen::NumericObject>("count")->get<int>();
        auto list = get_input<zen::ListObject>("list");
        auto fore = std::make_shared<BeginForObject>();
        fore->that = this;
        set_output("FOR", std::move(fore));

        auto ret = std::make_shared<zen::NumericObject>();
        ret->set(m_index);
        set_output("index", std::move(ret));

        m_count = count;
        m_index++;
    }
};

ZENDEFNODE(BeginFor, {
    {"count"},
    {"index", "FOR"},
    {},
    {"list"},
});


struct EndFor : zen::INode {
    std::unique_ptr<zen::Context> m_ctx = nullptr;

    void push_context() {
        assert(!m_ctx);
        m_ctx = std::move(sess->ctx);
        sess->ctx = std::make_unique<zen::Context>(*m_ctx);
    }

    void pop_context() {
        assert(m_ctx);
        sess->ctx = std::move(m_ctx);
    }

    virtual void doApply() override {
        auto [sn, ss] = inputBounds.at("FOR");
        sess->applyNode(sn);
        auto ref = sess->getNodeOutput(sn, ss);
        auto fore = std::dynamic_pointer_cast<BeginForObject>(
            sess->getObject(ref));
        while (fore->that->isContinue()) {
            push_context();
            zen::INode::doApply();
            pop_context();
        }
    }

    virtual void apply() override {}
};

ZENDEFNODE(EndFor, {
    {"FOR"},
    {},
    {},
    {"list"},
});


struct ExtractList : zen::INode {
    virtual void apply() override {
        auto list = get_input<zen::ListObject>("list");
        auto index = get_input<zen::NumericObject>("index")->get<int>();
        auto obj = list->arr[index];
        set_output("object", std::move(obj));
    }
};

ZENDEFNODE(ExtractList, {
    {"list", "index"},
    {"object"},
    {},
    {"list"},
});


struct EmptyList : zen::INode {
    virtual void apply() override {
        auto list = std::make_shared<zen::ListObject>();
        set_output("list", std::move(list));
    }
};

ZENDEFNODE(EmptyList, {
    {},
    {"list"},
    {},
    {"list"},
});


struct AppendList : zen::INode {
    virtual void apply() override {
        auto list = get_input<zen::ListObject>("list");
        auto obj = get_input("object");
        list->arr.push_back(std::move(obj));
        set_output_ref("list", get_input_ref("list"));
    }
};

ZENDEFNODE(AppendList, {
    {"list", "object"},
    {"list"},
    {},
    {"list"},
});
