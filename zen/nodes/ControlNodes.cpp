#include <zen/zen.h>
#include <zen/NumericObject.h>
#include <zen/ConditionObject.h>
#include <cassert>


struct BeginFor : zen::INode {
    int m_index;
    int m_count;

    void initState() {
        m_index = 0;
        m_count = 1;
    }

    bool isContinue() const {
        return m_index < m_count;
    }

    virtual void apply() override {
        auto count = get_input<zen::NumericObject>("count")->get<int>();

        auto fore = std::make_shared<zen::ConditionObject>();
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
        auto fore = dynamic_cast<BeginFor *>(sess->nodes.at(sn).get());
        if (!fore) {
            printf("EndFor::FOR must be conn to BeginFor::FOR!\n");
            abort();
        }
        fore->initState();
        while (fore->isContinue()) {
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
