#include <zen/zen.h>
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
        m_ctx = std::move(sess->ctx);
        sess->ctx = std::make_unique<zen::Context>(*m_ctx);
    }

    void pop_context() {
        assert(m_ctx);
        sess->ctx = std::move(m_ctx);
    }

    auto scoped_push_context() {
        push_context();
        return RAII([this]() {
            pop_context();
        });
    }
};


struct BeginFor : zen::INode {
    int m_index;
    int m_count;

    bool isContinue() const {
        return m_index < m_count;
    }

    virtual void apply() override {
        m_index = 0;
        m_count = get_input<zen::NumericObject>("count")->get<int>();
        set_output("FOR", std::make_shared<zen::ConditionObject>());
    }

    void update() {
        auto ret = std::make_shared<zen::NumericObject>();
        ret->set(m_index++);
        set_output("index", std::move(ret));
    }
};

ZENDEFNODE(BeginFor, {
    {"count"},
    {"index", "FOR"},
    {},
    {"list"},
});


struct EndFor : ContextManagedNode {
    virtual void doApply() override {
        auto [sn, ss] = inputBounds.at("FOR");
        auto fore = dynamic_cast<BeginFor *>(sess->nodes.at(sn).get());
        if (!fore) {
            printf("EndFor::FOR must be conn to BeginFor::FOR!\n");
            abort();
        }
        sess->applyNode(sn);
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
    {"list"},
});
