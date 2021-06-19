#include <zen/zen.h>
#include <zen/NumericObject.h>
#include <zen/ConditionObject.h>


struct ListRange : zen::INode {
    virtual void apply() override {
        int start = 0, stop = 0, step = 1;
        if (has_input("start"))
            start = get_input<zen::NumericObject>("start")->get<int>();
        if (has_input("stop"))
            stop = get_input<zen::NumericObject>("stop")->get<int>();
        if (has_input("step"))
            step = get_input<zen::NumericObject>("step")->get<int>();

        auto &ret = set_output_list("list");
        ret.m_isScalar = false;
        ret.m_arr.clear();
        for (int i = start; i < stop; i++) {
            auto val = std::make_unique<zen::NumericObject>();
            val->set(i);
            ret.m_arr.push_back(std::move(val));
        }
        printf("%d\n", ret.m_arr.size());
    }
};

ZENDEFNODE(ListRange, {
    {"start", "stop", "step"},
    {"list"},
    {},
    {"list"},
});


struct CloneFillList : zen::INode {
    virtual void apply() override {
        int count = get_input<zen::NumericObject>("count")->get<int>();
        auto obj = get_input("object");

        auto &ret = set_output_list("list");
        ret.m_isScalar = false;
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
