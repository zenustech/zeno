#include <zeno/zeno.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>


struct ListLength : zeno::INode {
    virtual void apply() override {
        auto list = get_input<zeno::ListObject>("list");
        auto ret = std::make_shared<zeno::NumericObject>();
        ret->set<int>(list->arr.size());
        set_output("length", std::move(ret));
    }
};

ZENDEFNODE(ListLength, {
    {"list"},
    {"length"},
    {},
    {"list"},
});


struct ExtractList : zeno::INode {
    virtual void apply() override {
        auto list = get_input<zeno::ListObject>("list");
        auto index = get_input<zeno::NumericObject>("index")->get<int>();
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


struct EmptyList : zeno::INode {
    virtual void apply() override {
        auto list = std::make_shared<zeno::ListObject>();
        set_output("list", std::move(list));
    }
};

ZENDEFNODE(EmptyList, {
    {},
    {"list"},
    {},
    {"list"},
});


struct AppendList : zeno::INode {
    virtual void apply() override {
        auto list = get_input<zeno::ListObject>("list");
        auto obj = get_input("object");
        list->arr.push_back(std::move(obj));
        set_output("list", get_input("list"));
    }
};

ZENDEFNODE(AppendList, {
    {"list", "object"},
    {"list"},
    {},
    {"list"},
});
