#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DummyObject.h>
#include <zeno/extra/ContextManaged.h>
#include <zeno/extra/evaluate_condition.h>
#include <zeno/utils/safe_at.h>

namespace zeno {

struct IBeginFor : zeno::INode {
    virtual bool isContinue() const = 0;
    virtual void updateIndex() = 0;
    virtual void resetIndex() = 0;
};


struct BeginFor : IBeginFor {
    int m_index = 0;
    int m_count = 0;

    void apply() override {
        set_output("index", std::make_shared<NumericObject>(m_index));
    }

    bool isContinue() const override {
        return m_index < getCount();
    }

    void updateIndex() override {
        m_index++;
        if (m_index < getCount())
            mark_dirty(true);
    }

    void resetIndex() override {
        m_index = 0;
        mark_dirty(true);
    }

    int getCount() const {
        int count = get_input<zeno::NumericObject>("count")->get<int>();
        return count;
    }
};

ZENDEFNODE(BeginFor, {
    {{"int", "count"}},
    {{"int", "index"}},
    {{"string", "For End", ""}},
    {"control"},
});


struct BeginForEach : IBeginFor {
    int m_index = 0;

    bool isContinue() const override final {
        return m_index < getCount();
    }

    void apply() override {
        if (isContinue())
        {
            std::shared_ptr<zeno::ListObject> list = get_input<zeno::ListObject>("objects");
            set_output("object", list->arr.at(m_index));
            set_output("index", std::make_shared<NumericObject>(m_index));
        }
    }

    void resetIndex() override {
        m_index = 0;
        mark_dirty(true);
    }

    virtual void updateIndex() override final {
        m_index++;
        if (m_index < getCount())
            mark_dirty(true);
    }

    int getCount() const {
        std::shared_ptr<zeno::ListObject> list = get_input<zeno::ListObject>("objects");
        return list->arr.size();
    }

};

ZENDEFNODE(BeginForEach, {
    {
        {"list", "objects", "", PrimarySocket},
    },
    {"object", {"int", "index"}},
    {{"string", "For End", ""}},
    {"control"},
});



struct EndFor : INode {

    virtual void preApply() override {
        //do nothing.
    }

    virtual void apply() override {
        requireInput("For Begin");
        std::string forbegin = get_input<zeno::StringObject>("For Begin")->get();

        graph->applyNode(forbegin);

        std::shared_ptr<IBeginFor> spBegin = std::dynamic_pointer_cast<IBeginFor>(graph->getNode(forbegin));
        if (!spBegin) {
            throw makeError<KeyError>("No matched For Begin", "");
        }

        spBegin->resetIndex();

        while (spBegin->isContinue()) {
            requireInput("object");
            //do something else
            spBegin->updateIndex();
        }
    }
};

ZENDEFNODE(EndFor, {
    {{"", "object", "", PrimarySocket}},
    {},
    {{"string", "For Begin", ""}},
    {"control"},
});


struct BreakFor : zeno::INode {
    virtual void apply() override {
        auto [sn, ss] = getinputbound("FOR", "input socket of BreakFor");
        auto fore = dynamic_cast<IBeginFor *>(graph->m_nodes.at(sn).get());
        if (!fore) {
            throw Exception("BreakFor::FOR must be conn to BeginFor::FOR!\n");
        }
        if (!has_input("breaks") || get_input2<bool>("breaks")) {
            //fore->breakThisFor();  // will still keep going the rest of loop body? yes
        }
    }

    //virtual void apply() override {}
};

ZENDEFNODE(BreakFor, {
    {{"", "FOR", "", PrimarySocket}, {"bool", "breaks", "1"}},
    {},
    {},
    {"control"},
});



struct EndForEach : INode {
    std::vector<zany> result;
    std::vector<zany> dropped_result;

    void preApply() override {
        //do nothing.
    }

    void apply() override {
        requireInput("For Begin");
        std::string forbegin = get_input<zeno::StringObject>("For Begin")->get();
        std::shared_ptr<IBeginFor> spBegin = std::dynamic_pointer_cast<IBeginFor>(graph->getNode(forbegin));
        if (!spBegin) {
            throw makeError<KeyError>("No matched For Begin", "");
        }

        graph->applyNode(forbegin);

        std::vector<zany> result;
        std::vector<zany> dropped_result;

        spBegin->resetIndex();

        while (spBegin->isContinue()) {
            requireInput("object");
            requireInput("objects");

            bool accept = true;
            if (requireInput("accept")) {
                accept = evaluate_condition(get_input("accept").get());
            }

            if (auto obj = get_input("object")) {
                if (accept)
                    result.push_back(std::move(obj));
                else
                    dropped_result.push_back(std::move(obj));
            }

            if (has_input("objects")) {
                auto listObj = get_input<zeno::ListObject>("objects");
                if (accept) {
                    for (auto obj : listObj->arr)
                        result.push_back(std::move(obj));
                }
                else {
                    for (auto obj : listObj->arr)
                        dropped_result.push_back(std::move(obj));
                }
            }

            spBegin->updateIndex();
        }

        auto list = std::make_shared<ListObject>();
        list->arr = std::move(result);
        set_output("list", std::move(list));

        auto dropped_list = std::make_shared<ListObject>();
        dropped_list->arr = std::move(dropped_result);
        set_output("droppedList", std::move(dropped_list));
    }

};

ZENDEFNODE(EndForEach, {
    {
        {"", "object", "", PrimarySocket},
        {"list", "objects", "", PrimarySocket},
        {"bool", "accept", "1"}
    },
    {"list", "droppedList"},
    {{"bool", "doConcat", "0"},
     {"string", "For Begin", ""}},
    {"control"},
});


struct BeginSubstep : INode {
    float m_total = 0;
    float m_mindt = 0;
    float m_elapsed = 0;
    bool m_ever_called = false;

    void apply() override {

    }

    //virtual bool isContinue() const override final {
    //    return m_elapsed < m_total;
    //}

    //virtual void execute() override final {
    //    m_elapsed = 0;
    //    m_ever_called = false;
    //    m_total = get_input<zeno::NumericObject>("total_dt")->get<float>();
    //    auto min_scale = has_input("min_scale") ?
    //        get_input<zeno::NumericObject>("min_scale")->get<float>() : 0.05f;
    //    m_mindt = m_total * min_scale;
    //    set_output("FOR", std::make_shared<zeno::DummyObject>());
    //}

    //virtual void update() override final {
    //    auto ret = std::make_shared<zeno::NumericObject>();
    //    ret->set(m_elapsed);
    //    set_output("elapsed_time", std::move(ret));
    //}
};

ZENDEFNODE(BeginSubstep, {
    {{"float", "total_dt"}, {"float", "min_scale", "0.05"}},
    {"FOR", {"float", "elapsed_time"}},
    {},
    {"control"},
});

struct SubstepDt : zeno::INode {
    void apply() override {
        auto [sn, ss] = getinputbound("FOR", "input socket of SubstepDt");
        auto fore = dynamic_cast<BeginSubstep *>(graph->m_nodes.at(sn).get());
        if (!fore) {
            throw Exception("SubstepDt::FOR must be conn to BeginSubstep::FOR!\n");
        }
        fore->m_ever_called = true;
        float dt = get_input<zeno::NumericObject>("desired_dt")->get<float>();
        if (fore->m_elapsed + dt >= fore->m_total) {
            dt = std::max(0.f, fore->m_total - fore->m_elapsed);
            fore->m_elapsed = fore->m_total;
        } else {
            if (dt < fore->m_mindt) {
                dt = fore->m_mindt;
            }
            fore->m_elapsed += dt;
        }
        float portion = fore->m_total ? fore->m_elapsed / fore->m_total : 0.0f;
        printf("** actual_dt: %f\n", dt);
        printf("** portion: %f\n", portion);
        auto ret_dt = std::make_shared<zeno::NumericObject>();
        ret_dt->set(dt);
        set_output("actual_dt", std::move(ret_dt));
        auto ret_portion = std::make_shared<zeno::NumericObject>();
        ret_portion->set(portion);
        set_output("portion", std::move(ret_portion));
    }
};

ZENDEFNODE(SubstepDt, {
    {{"", "FOR", "", PrimarySocket}, {"float", "desired_dt", "0.04"}},
    {{"float", "actual_dt"}, {"float", "portion"}},
    {},
    {"control"},
});




struct IfElse : zeno::INode {
    virtual void preApply() override {
        requireInput("cond");
        auto cond = get_input("cond");
        /*if (has_option("MUTE")) {
            requireInput("true");
        } else*/ if (evaluate_condition(cond.get())) {
            if (requireInput("true")) {
                set_output("result", get_input("true"));
            } else {
                set_output("result", std::make_shared<DummyObject>());
            }
        } else {
            if (requireInput("false")) {
                set_output("result", get_input("false"));
            } else {
                set_output("result", std::make_shared<DummyObject>());
            }
        }

        apply();
    }

    virtual void apply() override {}
};

ZENDEFNODE(IfElse, {
    {
        {"", "true", "", PrimarySocket},
        {"", "false", "", PrimarySocket},
        {"bool", "cond"},
    },
    {"result"},
    {},
    {"control"},
});


/*** Start Of - ZHXX Control Flow ***

struct IF : zeno::INode {

    bool m_condition;
    virtual void apply() override {
        m_condition = (get_input<zeno::NumericObject>("condition")->get<int>());
        set_output("Then", std::make_shared<zeno::DummyObject>());
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
    virtual void preApply() override {
        auto [sn, ss] = inputBounds.at("IF");
        auto true_exp = dynamic_cast<IF *>(graph->m_nodes.at(sn).get());
        if (!true_exp) {
            throw Exception("please connect true and false execution tree\n");
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
        set_output("Branch", std::make_shared<zeno::DummyObject>());
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
        set_output("Branch", std::make_shared<zeno::DummyObject>());
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
    virtual void preApply() override {
        auto [sn, ss] = inputBounds.at("BranchIn");
        auto exec = dynamic_cast<IBranch *>(graph->m_nodes.at(sn).get());
        graph->applyNode(sn);
        if(exec->getCondition()){
            push_context();
            zeno::INode::doApply();
            pop_context();
        }
    }

    virtual void apply() override {
        set_output("Branch", std::make_shared<zeno::DummyObject>());
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
    virtual void preApply() override {
        
        auto [sn, ss] = inputBounds.at("True");
        auto [sn1, ss1] = inputBounds.at("False");
        
        auto exec = dynamic_cast<EndBranch *>(graph->m_nodes.at(sn).get());
        exec->doApply();

        auto exec1 = dynamic_cast<EndBranch *>(graph->m_nodes.at(sn1).get());
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
