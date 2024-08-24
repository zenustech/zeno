#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DummyObject.h>
#include <zeno/extra/ContextManaged.h>
#include <zeno/extra/evaluate_condition.h>
#include <zeno/utils/safe_at.h>
#include <zeno/core/GlobalVariable.h>


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
    {{gParamType_Int, "count"}},
    {{gParamType_Int, "index"}},
    {{gParamType_String, "For End", ""}},
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
            set_output("object", list->get(m_index));
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
        return list->size();
    }

};

ZENDEFNODE(BeginForEach, {
    {
        {gParamType_List, "objects", "", zeno::Socket_ReadOnly},
    },
    {{gParamType_IObject, "object"}, {gParamType_Int, "index"}},
    {{gParamType_String, "For End", ""}},
    {"control"},
});



struct EndFor : INode {

    virtual void preApply() override {
        //do nothing.
    }

    virtual void apply() override {
        requireInput("For Begin");
        std::string forbegin = get_input<zeno::StringObject>("For Begin")->get();

        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);

        spGraph->applyNode(forbegin);

        std::shared_ptr<IBeginFor> spBegin = std::dynamic_pointer_cast<IBeginFor>(spGraph->getNode(forbegin));
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
    {{gParamType_FOR, "object", "", zeno::Socket_ReadOnly}},
    {},
    {{gParamType_String, "For Begin", ""}},
    {"control"},
});


struct BreakFor : zeno::INode {
    virtual void apply() override {
        zeno::ParamObject paramobj = get_input_obj_param("FOR");
        if (paramobj.links.empty())
            throw makeError("BreakFor::FOR must be conn to BeginFor::FOR");

        auto link = paramobj.links[0];

        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);
        auto fore = dynamic_cast<IBeginFor *>(spGraph->m_nodes.at(link.outNode).get());
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
    {{gParamType_FOR, "FOR", "", zeno::Socket_ReadOnly}, {gParamType_Bool, "breaks", "1"}},
    {},
    {},
    {"control"},
});



struct EndForEach : INode {

    void preApply() override {
        //do nothing.
    }

    void apply() override {
        requireInput("For Begin");
        std::string forbegin = get_input<zeno::StringObject>("For Begin")->get();
        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);
        std::shared_ptr<IBeginFor> spBegin = std::dynamic_pointer_cast<IBeginFor>(spGraph->getNode(forbegin));
        if (!spBegin) {
            throw makeError<KeyError>("No matched For Begin", "");
        }

        spGraph->applyNode(forbegin);

        auto list = std::make_shared<ListObject>();
        auto dropped_list = std::make_shared<ListObject>();

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
                    list->push_back(std::move(obj));
                else
                    dropped_list->push_back(std::move(obj));
            }

            if (has_input("objects")) {
                auto listObj = get_input<zeno::ListObject>("objects");
                if (accept) {
                    for (auto obj : listObj->get())
                        list->push_back(std::move(obj));
                }
                else {
                    for (auto obj : listObj->get())
                        dropped_list->push_back(std::move(obj));
                }
            }

            spBegin->updateIndex();
        }

        set_output("list", std::move(list));
        set_output("droppedList", std::move(dropped_list));
    }

};

ZENDEFNODE(EndForEach, {
    {
        {gParamType_IObject, "object", "", zeno::Socket_ReadOnly},
        {gParamType_List, "objects", "", zeno::Socket_ReadOnly},
        {gParamType_Bool, "accept", "1"}
    },
    {{gParamType_List, "droppedList"}},
    {{gParamType_Bool, "doConcat", "0"},
     {gParamType_String, "For Begin", ""}},
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
    {{gParamType_Float, "total_dt"}, {gParamType_Float, "min_scale", "0.05"}},
    {{gParamType_FOR, "FOR"}, {gParamType_Float, "elapsed_time"}},
    {},
    {"control"},
});

struct SubstepDt : zeno::INode {
    void apply() override {
        zeno::ParamObject paramobj = get_input_obj_param("FOR");
        if (paramobj.links.empty())
            throw makeError("BreakFor::FOR must be conn to BeginFor::FOR");

        auto link = paramobj.links[0];

        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);
        auto fore = dynamic_cast<BeginSubstep *>(spGraph->m_nodes.at(link.outNode).get());
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
    {{gParamType_FOR, "FOR", "", zeno::Socket_ReadOnly}, {gParamType_Float, "desired_dt", "0.04"}},
    {{gParamType_Float, "actual_dt"}, {gParamType_Float, "portion"}},
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
        {gParamType_IObject, "true", "", Socket_WildCard, NullControl, "wildCard"},
        {gParamType_IObject, "false", "", Socket_WildCard, NullControl, "wildCard"},
        {gParamType_Bool, "cond"},
    },
    {{gParamType_IObject, "result","",Socket_WildCard, NullControl, "wildCard"}},
    {},
    {"control"},
});

//test
struct TimeShift : zeno::INode {
    virtual void preApply() override {
        ParamPrimitive param = get_input_prim_param("offset");
        int offset = zeno_get<int>(param.defl);
        //∏≤∏«$F
        zeno::reflect::Any frame = getSession().globalVariableManager->getVariable("$F");
        float currFrame = (frame.has_value() ? zeno::reflect::any_cast<float>(frame) : 0) + offset;
        auto globalOverride = GlobalVariableOverride(shared_from_this(), "$F", zeno::reflect::make_any<float>(currFrame));
        //º∆À„…œ”Œ
        INode::preApply();
    }
    virtual void apply() override {
        auto prim = get_input2<zeno::PrimitiveObject>("prim");
        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(TimeShift, {
    {
        {gParamType_Primitive, "prim", "", Socket_ReadOnly},
        {gParamType_Int, "offset", "0", Socket_Primitve, Lineedit},
    },
    {{gParamType_Primitive, "prim"}},
    {},
    {"control"},
    });

struct Duplicate : INode {
    virtual void trigger_update_params(const std::string& param, bool changed, params_change_info changinfo) {
        if (param == "duplicate") {
            ParamsUpdateInfo params;
            auto const& duplicatePrimitiveParam = get_input_prim_param("duplicate");
            params.push_back({ duplicatePrimitiveParam , "duplicate" });
            auto const& inputObjParam = get_input_obj_param("input");
            params.push_back({ inputObjParam , "input" });
            auto const& outputObjParam = get_output_obj_param("owning");
            params.push_back({ outputObjParam , "owning" });

            int val = 0;
            zeno_get_if(duplicatePrimitiveParam.defl, val);
            if (val) {
                zeno::ParamObject objOutput;
                objOutput.bInput = false;
                objOutput.name = "duplicateObj";
                objOutput.socketType = zeno::Socket_Output;
                objOutput.type = gParamType_IObject;
                params.push_back({ objOutput , "" });
            }
            params_change_info changes = update_editparams(params);
            INode::trigger_update_params(param, true, changes);
        }
    }
    virtual void apply() override {
        auto inputObj = get_input2<zeno::IObject>("input");

        bool hasDuplicateOutput;
        get_output_obj_param("duplicateObj", &hasDuplicateOutput);
        if (hasDuplicateOutput) {
            auto objClone = inputObj->clone();
            objClone->update_key(get_uuid() + "_clone");
            set_output("duplicateObj", objClone);
        }
        set_output("owning", inputObj->move_clone());
    }
};
ZENDEFNODE(Duplicate,
    {
        {
            {gParamType_IObject, "input", "", Socket_Owning},
        },
        {
            {gParamType_IObject, "owning", "", Socket_Output},
        },
        {
            {gParamType_Bool, "duplicate", "0"},
        },
        {"control"}
    }
)

struct TestWildCard : INode {};

ZENDEFNODE(TestWildCard,
    {
        {
            {Param_Wildcard, "wilcard1", "", Socket_WildCard, Lineedit, "wildCardGroup1"},
            {Param_Wildcard, "wilcard2", "", Socket_WildCard, Lineedit, "wildCardGroup1"},
            {Obj_Wildcard, "wilcardObj1", "", Socket_WildCard, NullControl, "wildCardGroup2"},
            {Obj_Wildcard, "wilcardObj2", "", Socket_WildCard, NullControl, "wildCardGroup2"},
            {gParamType_Int, "int1", "0", Socket_Primitve},
        },
        {
            {Param_Wildcard, "wilcard3", "", Socket_WildCard, NullControl, "wildCardGroup1"},
            {Obj_Wildcard, "wilcardObj3", "", Socket_WildCard, NullControl, "wildCardGroup2"},
            //{"object", "owning", "", Socket_Output},
        },
        {
        },
        {"control"}
    }
)

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
