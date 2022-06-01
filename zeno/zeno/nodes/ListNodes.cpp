#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/string.h>
#include <sstream>

namespace zeno {

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


struct ListGetItem : zeno::INode {
    virtual void apply() override {
        auto list = get_input<zeno::ListObject>("list");
        auto index = get_input<zeno::NumericObject>("index")->get<int>();
        auto obj = list->arr.at(index);
        set_output2("object", std::move(obj));
    }
};

ZENDEFNODE(ListGetItem, {
    {"list", {"int", "index"}},
    {"object"},
    {},
    {"list"},
});

struct ExtractList : zeno::INode {
    virtual void apply() override {
        auto inkeys = get_param<std::string>("_KEYS");
        auto keys = zeno::split_str(inkeys, '\n');
        auto list = get_input<zeno::ListObject>("list");
        for (auto const& key : keys) {
            int index = std::stoi(key);
            if (list->arr.size() > index) {
                auto obj = list->arr[index];
                set_output2(key, std::move(obj));
            }
        }
    }
};

ZENDEFNODE(ExtractList, {
    {"list"},
    {},
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

struct ExtendList : zeno::INode {
    virtual void apply() override {
        auto list1 = get_input<zeno::ListObject>("list1");
        auto list2 = get_input<zeno::ListObject>("list2");
        for (auto const &ptr: list2->arr) {
            list1->arr.push_back(ptr);
        }
        set_output("list1", std::move(list1));
    }
};

ZENDEFNODE(ExtendList, {
    {"list1", "list2"},
    {"list1"},
    {},
    {"list"},
});


struct ResizeList : zeno::INode {
    virtual void apply() override {
        auto list = get_input<zeno::ListObject>("list");
        auto newSize = get_input<zeno::NumericObject>("newSize")->get<int>();
        list->arr.resize(newSize);
        set_output("list", std::move(list));
    }
};

ZENDEFNODE(ResizeList, {
    {"list", {"int", "newSize"}},
    {"list"},
    {},
    {"list"},
});


struct MakeSmallList : zeno::INode {
    virtual void apply() override {
        auto list = std::make_shared<zeno::ListObject>();
        auto doConcat = get_param<bool>("doConcat");
        for (int i = 0; i < 6; i++) {
            std::stringstream namess;
            namess << "obj" << i;
            auto name = namess.str();
            if (!has_input(name)) break;
            if (doConcat && has_input<ListObject>(name)) {
                auto objlist = get_input<ListObject>(name);
                for (auto const &obj: objlist->arr) {
                    list->arr.push_back(std::move(obj));
                }
            } else {
                auto obj = get_input2(name);
                list->arr.push_back(std::move(obj));
            }
        }
        set_output("list", std::move(list));
    }
};

ZENDEFNODE(MakeSmallList, {
    {"obj0", "obj1", "obj2", "obj3", "obj4", "obj5"},
    {"list"},
    {{"bool", "doConcat", "1"}},
    {"list"},
});

struct MakeList : zeno::INode {
    virtual void apply() override {
        auto list = std::make_shared<zeno::ListObject>();

        int max_input_index = 0;
        for (auto& pair : inputs) {
            if (std::isdigit(pair.first.back())) {
                max_input_index = std::max<int>(max_input_index, std::stoi(pair.first.substr(3)));
            }
        }
        for (int i = 0; i <= max_input_index; ++i) {
            std::stringstream namess;
            namess << "obj" << i;
            auto name = namess.str();
            if (!has_input(name)) continue;
            auto obj = get_input(name);
            list->arr.push_back(std::move(obj));
        }
        set_output("list", std::move(list));
    }
};

ZENDEFNODE(MakeList, {
    {},
    {"list"},
    {},
    {"list"},
    });


struct NumericRangeList : zeno::INode {
    virtual void apply() override {
        auto list = std::make_shared<zeno::ListObject>();
        auto start = get_input2<int>("start");
        auto stop = get_input2<int>("stop");
        auto skip = get_input2<int>("skip");
        for (int i = start; i < stop; i += skip) {
            list->arr.emplace_back(i);
        }
        set_output("list", std::move(list));
    }
};

ZENDEFNODE(NumericRangeList, {
    {{"int","start","0"},{"int","end","1"},{"int","skip","1"}},
    {"list"},
    {},
    {"list"},
    });


struct IsList : zeno::INode {
    virtual void apply() override {
        auto result = std::make_shared<zeno::NumericObject>();
        result->value = 0;
        if (has_input<zeno::ListObject>("list")) 
            result->value = 1;
        set_output("result", std::move(result));
    } 
};

ZENDEFNODE(IsList, {
    {"list"},
    {"result"},
    {},
    {"list"},
});


#ifdef ZENO_VISUALIZATION
struct ToVisualize_ListObject : zeno::INode {
    virtual void apply() override {
        auto list = get_input<ListObject>("list");
        auto path = get_param<std::string>("path");
        for (int i = 0; i < list->arr.size(); i++) {
            auto const &obj = list->arr[i];
            std::stringstream ss;
            ss << path << "." << i;
            if (auto o = silent_any_cast<std::shared_ptr<IObject>>(obj); o.has_value()) {
                if (auto node = graph->getOverloadNode("ToVisualize", {o.value()}); node) {
                    node->inputs["path:"] = ss.str();
                    node->doApply();
                }
            }
        }
    }
};

ZENO_DEFOVERLOADNODE(ToVisualize, _ListObject, typeid(ListObject).name())({
        {"list"},
        {},
        {{"string", "path", ""}},
        {"list"},
});
#endif

}
