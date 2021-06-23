#include <zen/zen.h>
#include <zen/ListObject.h>
#include <zen/NumericObject.h>
#include <zen/ConditionObject.h>
#include <cassert>


template <class F>
struct RAII {
    F dtor;
    RAII(F &&dtor) : dtor(dtor) {}
    ~RAII() { dtor(); }
};


struct ContextManagedNode : zen::INode {
    std::unique_ptr<zen::Context> m_ctx = nullptr;

    void push_context() {
        assert(!m_ctx);
        m_ctx = std::move(graph->ctx);
        graph->ctx = std::make_unique<zen::Context>(*m_ctx);
    }

    void pop_context() {
        assert(m_ctx);
        graph->ctx = std::move(m_ctx);
    }

    auto scoped_push_context() {
        push_context();
        return RAII([this]() {
            pop_context();
        });
    }
};


struct IBeginFor : zen::INode {
    virtual bool isContinue() const = 0;
    virtual void update() = 0;
};


struct BeginFor : IBeginFor {
    int m_index;
    int m_count;

    virtual bool isContinue() const override {
        return m_index < m_count;
    }

    virtual void apply() override {
        m_index = 0;
        m_count = get_input<zen::NumericObject>("count")->get<int>();
        set_output("FOR", std::make_shared<zen::ConditionObject>());
    }

    virtual void update() override {
        auto ret = std::make_shared<zen::NumericObject>();
        ret->set(m_index);
        set_output("index", std::move(ret));
        m_index++;
    }
};

ZENDEFNODE(BeginFor, {
    {"count"},
    {"index", "FOR"},
    {},
    {"control"},
});


struct EndFor : ContextManagedNode {
    virtual void doApply() override {
        auto [sn, ss] = inputBounds.at("FOR");
        auto fore = dynamic_cast<IBeginFor *>(graph->nodes.at(sn).get());
        if (!fore) {
            printf("EndFor::FOR must be conn to BeginFor::FOR!\n");
            abort();
        }
        graph->applyNode(sn);
        while (fore->isContinue()) {
            fore->update();
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
    {"control"},
});


struct BeginForEach : IBeginFor {
    int m_index;
    std::shared_ptr<zen::ListObject> m_list;

    virtual bool isContinue() const override {
        return m_index < m_list->arr.size();
    }

    virtual void apply() override {
        m_index = 0;
        m_list = get_input<zen::ListObject>("list");
        set_output("FOR", std::make_shared<zen::ConditionObject>());
    }

    virtual void update() override {
        auto ret = std::make_shared<zen::NumericObject>();
        ret->set(m_index);
        set_output("index", std::move(ret));
        auto obj = m_list->arr[m_index];
        set_output("object", std::move(obj));
        m_index++;
    }
};

ZENDEFNODE(BeginForEach, {
    {"list"},
    {"object", "index", "FOR"},
    {},
    {"control"},
});
