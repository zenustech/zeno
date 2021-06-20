#include <zen/zen.h>
#include <zen/NumericObject.h>
#include <zen/ConditionObject.h>


struct ListRange : zen::INode {
    virtual void listapply() override {
        int start = 0, stop = 0, step = 1;
        if (has_input("start"))
            start = get_input<zen::NumericObject>("start")->get<int>();
        if (has_input("stop"))
            stop = get_input<zen::NumericObject>("stop")->get<int>();
        if (has_input("step"))
            step = get_input<zen::NumericObject>("step")->get<int>();

        auto &ret = set_output_list("list");
        ret.m_isList = true;
        ret.m_arr.clear();
        for (int i = start; i < stop; i++) {
            auto val = std::make_unique<zen::NumericObject>();
            val->set(i);
            ret.m_arr.push_back(std::move(val));
        }
    }
};

ZENDEFNODE(ListRange, {
    {"start", "stop", "step"},
    {"list"},
    {},
    {"list"},
});


struct CloneFillList : zen::INode {
    virtual void listapply() override {
        int count = get_input<zen::NumericObject>("count")->get<int>();
        auto obj = get_input("object");

        auto &ret = set_output_list("list");
        ret.m_isList = true;
        ret.m_arr.clear();
        for (int i = 0; i < count; i++) {
            auto newobj = obj->clone();
            if (!newobj) {
                printf("ERROR: requested object doesn't support clone\n");
                return;
            }
            ret.m_arr.push_back(std::move(newobj));
        }
    }
};

ZENDEFNODE(CloneFillList, {
    {"count", "object"},
    {"list"},
    {},
    {"list"},
});


struct TestIsList : zen::INode {
    virtual void listapply() override {
        auto &inp = get_input_list("input");
        auto ret = std::make_unique<zen::ConditionObject>();
        ret->set(inp.m_isList);
        set_output("isList", std::move(ret));
    }
};

ZENDEFNODE(TestIsList, {
    {"input"},
    {"isList"},
    {},
    {"list"},
});


struct EmptyList : zen::INode {
    virtual void listapply() override {
        auto &ret = set_output_list("list");
        ret.m_isList = true;
        ret.m_arr.clear();
    }
};

ZENDEFNODE(EmptyList, {
    {},
    {"list"},
    {},
    {"list"},
});


struct ListAppendClone : zen::INode {
    virtual void listapply() override {
        auto &inp = get_input_list("input");
        auto obj = get_input("object");
        auto newobj = obj->clone();
        if (!newobj) {
            printf("ERROR: requested object doesn't support clone\n");
            return;
        }
        inp.m_isList = true;
        inp.m_arr.push_back(std::move(newobj));
        set_output_ref("list", get_input_ref("list"));
    }
};

ZENDEFNODE(ListAppendClone, {
    {"list", "object"},
    {"list"},
    {},
    {"list"},
});
