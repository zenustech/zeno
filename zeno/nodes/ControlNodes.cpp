#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/extra/ContextManaged.h>
#include <zeno/extra/evaluate_condition.h>

namespace zeno {

struct IBeginFor : zeno::INode {
    bool is_break = false;

    virtual bool isContinue() const = 0;
    virtual void update() = 0;
};


struct BeginFor : IBeginFor {
    int m_index = 0;
    int m_count = 0;
    
    virtual bool isContinue() const override {
        return m_index < m_count && !is_break;
    }

    virtual void apply() override {
        m_index = 0;
        is_break = false;
        m_count = get_input<zeno::NumericObject>("count")->get<int>();
        set_output("FOR", std::make_shared<zeno::ConditionObject>());
    }

    virtual void update() override {
        auto ret = std::make_shared<zeno::NumericObject>();
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


struct EndFor : zeno::ContextManagedNode {
    virtual void doApply() override {
        auto [sn, ss] = inputBounds.at("FOR");
        auto fore = dynamic_cast<IBeginFor *>(graph->nodes.at(sn).get());
        if (!fore) {
            printf("EndFor::FOR must be conn to BeginFor::FOR!\n");
            abort();
        }
        graph->applyNode(sn);
        std::unique_ptr<zeno::Context> old_ctx = nullptr;
        while (fore->isContinue()) {
            fore->update();
            push_context();
            zeno::INode::doApply();
            old_ctx = pop_context();
        }
        if (old_ctx) {
            // auto-valid the nodes in last iteration when refered from outside
            graph->ctx->mergeVisited(*old_ctx);
            old_ctx = nullptr;
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


struct BreakFor : zeno::INode {
    virtual void doApply() override {
        auto [sn, ss] = inputBounds.at("FOR");
        auto fore = dynamic_cast<IBeginFor *>(graph->nodes.at(sn).get());
        fore->is_break = true;  // will still keep going the rest of loop body?
    }

    virtual void apply() override {}
};

ZENDEFNODE(BreakFor, {
    {"FOR"},
    {},
    {},
    {"control"},
});

struct BeginForEach : IBeginFor {
    int m_index = 0;
    std::shared_ptr<zeno::ListObject> m_list;

    virtual bool isContinue() const override {
        return m_index < m_list->arr.size() && !is_break;
    }

    virtual void apply() override {
        m_index = 0;
        is_break = false;
        m_list = get_input<zeno::ListObject>("list");
        set_output("FOR", std::make_shared<zeno::ConditionObject>());
    }

    virtual void update() override {
        auto ret = std::make_shared<zeno::NumericObject>();
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

struct BeginSubstep : IBeginFor {
    float m_elapsed = 0;
    float m_total = 0;
    bool m_ever_called = false;

    virtual bool isContinue() const override {
        return m_elapsed < m_total && !is_break;
    }

    virtual void apply() override {
        m_elapsed = 0;
        is_break = false;
        m_ever_called = false;
        m_total = get_input<zeno::NumericObject>("total_dt")->get<float>();
        set_output("FOR", std::make_shared<zeno::ConditionObject>());
    }

    virtual void update() override {
        if (!m_ever_called) {
            printf("WARNING: SubstepDt never called for BeginSubstep!\n");
            is_break = true;
        }
        m_ever_called = false;
        auto ret = std::make_shared<zeno::NumericObject>();
        ret->set(m_elapsed);
        set_output("elapsed_time", std::move(ret));
    }
};

ZENDEFNODE(BeginSubstep, {
    {"total_dt"},
    {"FOR", "elapsed_time"},
    {},
    {"control"},
});

struct SubstepDt : zeno::INode {
    void apply() override {
        auto [sn, ss] = inputBounds.at("FOR");
        auto fore = dynamic_cast<BeginSubstep *>(graph->nodes.at(sn).get());
        if (!fore) {
            printf("SubstepDt::FOR must be conn to BeginSubstep::FOR!\n");
            abort();
        }
        fore->m_ever_called = true;
        float dt = get_input<zeno::NumericObject>("desired_dt")->get<float>();
        if (fore->m_elapsed + dt >= fore->m_total) {
            dt = std::max(0.f, fore->m_total - fore->m_elapsed);
            fore->m_elapsed = fore->m_total;
        }
        auto ret = std::make_shared<zeno::NumericObject>();
        ret->set(dt);
        set_output("actual_dt", std::move(ret));
    }
};

ZENDEFNODE(SubstepDt, {
    {"FOR", "desired_dt"},
    {"actual_dt"},
    {},
    {"control"},
});




struct IfElse : zeno::INode {
    virtual void doApply() override {
        requireInput("cond");
        auto cond = get_input("cond");
        if (has_option("MUTE")) {
            requireInput("true");
        } else if (evaluate_condition(cond.get())) {
            if (has_input("true")) {
                requireInput("true");
                set_output("result", get_input("true"));
            }
        } else {
            if (has_input("false")) {
                requireInput("false");
                set_output("result", get_input("false"));
            }
        }

        coreApply();
    }

    virtual void apply() override {}
};

ZENDEFNODE(IfElse, {
    {"true", "false", "cond"},
    {"result"},
    {},
    {"control"},
});


/*** Start Of - ZHXX Control Flow ***

struct IF : zeno::INode {

    bool m_condition;
    virtual void apply() override {
        m_condition = (get_input<zeno::NumericObject>("condition")->get<int>());
        set_output("Then", std::make_shared<zeno::ConditionObject>());
    }
    bool getCondition() const
    {
        return m_condition==1;
    }

};

ZENDEFNODE(IF, {
    {"condition"},
    {"Then"},
    {},
    {"control"},
});

struct EndIF : zeno::ContextManagedNode {
    virtual void doApply() override {
        auto [sn, ss] = inputBounds.at("IF");
        auto true_exp = dynamic_cast<IF *>(graph->nodes.at(sn).get());
        if (!true_exp) {
            printf("please connect true and false execution tree\n");
            abort();
        }
        graph->applyNode(sn);
        if(true_exp->getCondition()){
            push_context();
            zeno::INode::doApply();
            pop_context();
        }
    }

    virtual void apply() override {}
};

ZENDEFNODE(EndIF, {
    {"IF"},
    {},
    {},
    {"control"},
});

struct IBranch : zeno::INode {
    virtual bool getCondition() const = 0;
};

struct TrueBranch : IBranch {
    bool m_execute;
    virtual void apply() override {
        auto innum = (get_input<zeno::NumericObject>("condition")->get<int>());
        m_execute = (innum == 1);
        set_output("Branch", std::make_shared<zeno::ConditionObject>());
    }
    virtual bool getCondition() const override
    {
        return m_execute;
    }

};

ZENDEFNODE(TrueBranch, {
    {"condition"},
    {"Branch"},
    {},
    {"control"},
});
struct FalseBranch : IBranch {
    bool m_execute;
    virtual void apply() override {
        auto innum = (get_input<zeno::NumericObject>("condition")->get<int>());
        m_execute = (innum == 1);
        set_output("Branch", std::make_shared<zeno::ConditionObject>());
    }
    virtual bool getCondition() const override
    {
        return !m_execute;
    }

};

ZENDEFNODE(FalseBranch, {
    {"condition"},
    {"Branch"},
    {},
    {"control"},
});

struct EndBranch : zeno::ContextManagedNode {
    virtual void doApply() override {
        auto [sn, ss] = inputBounds.at("BranchIn");
        auto exec = dynamic_cast<IBranch *>(graph->nodes.at(sn).get());
        graph->applyNode(sn);
        if(exec->getCondition()){
            push_context();
            zeno::INode::doApply();
            pop_context();
        }
    }

    virtual void apply() override {
        set_output("Branch", std::make_shared<zeno::ConditionObject>());
    }
};

ZENDEFNODE(EndBranch, {
    {"BranchIn"},
    {"Branch"},
    {},
    {"control"},
});


struct ConditionedDo : zeno::INode {
    bool m_which;
    virtual void doApply() override {
        
        auto [sn, ss] = inputBounds.at("True");
        auto [sn1, ss1] = inputBounds.at("False");
        
        auto exec = dynamic_cast<EndBranch *>(graph->nodes.at(sn).get());
        exec->doApply();

        auto exec1 = dynamic_cast<EndBranch *>(graph->nodes.at(sn1).get());
        exec1->doApply();

    }

    virtual void apply() override {
    }
};

ZENDEFNODE(ConditionedDo, {
    {"True", "False"},
    {},
    {},
    {"control"},
});

*** End Of - ZHXX Control Flow ***/




}
