#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/string.h>
#include <sstream>

namespace zeno {

struct ListLength : zeno::INode {
    virtual void apply() override {
        auto list = get_input<zeno::ListObject>("list");
        auto ret = std::make_shared<zeno::NumericObject>();
        ret->set<int>(list->size());
        set_output("length", std::move(ret));
    }
};

ZENDEFNODE(ListLength, {
    {{gParamType_List, "list", "", zeno::Socket_ReadOnly, NoMultiSockPanel}},
    {{gParamType_Int,"length"}},
    {},
    {"list"},
});


struct ListGetItem : zeno::INode {
    virtual void apply() override {
        auto index = get_input<zeno::NumericObject>("index")->get<int>();
        if (has_input<DictObject>("list")) {
            auto dict = get_input<zeno::DictObject>("list");
            if (index < 0 || index >= dict->lut.size())
                throw makeError<IndexError>(index, dict->lut.size(), "ListGetItem (for dict)");
            auto obj = std::next(dict->lut.begin(), index)->second;
            set_output("object", std::move(obj));
        } else {
            auto list = get_input<zeno::ListObject>("list");
            if (index < 0 || index >= list->size())
                throw makeError<IndexError>(index, list->size(), "ListGetItem");
            auto obj = list->get(index);
            set_output("object", std::move(obj));
        }
    }
};

ZENDEFNODE(ListGetItem, {
    {{gParamType_List, "list", "", zeno::Socket_ReadOnly, NoMultiSockPanel},
     {gParamType_Int, "index"}},
    {{gParamType_IObject, "object"}},
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
            if (list->size() > index) {
                auto obj = list->get(index);
                set_output(key, std::move(obj));
            }
        }
    }
};

ZENDEFNODE(ExtractList, {
    {{gParamType_List, "list", "", zeno::Socket_ReadOnly, NoMultiSockPanel}},
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
    {{gParamType_List, "list"}},
    {},
    {"list"},
});


struct AppendList : zeno::INode {
    virtual void apply() override {
        auto list = get_input<zeno::ListObject>("list");
        auto obj = get_input("object");
        list->push_back(std::move(obj));
        set_output("list", get_input("list"));
    }
};

ZENDEFNODE(AppendList, {
    {
        {gParamType_List, "list", "", zeno::Socket_ReadOnly, NoMultiSockPanel},
        {gParamType_IObject, "object", "", zeno::Socket_ReadOnly}
    },
    {{gParamType_List, "list"}},
    {},
    {"list"},
});

struct ExtendList : zeno::INode {
    virtual void apply() override {
        auto list1 = get_input<zeno::ListObject>("list1");
        auto list2 = get_input<zeno::ListObject>("list2");
        for (auto const &ptr: list2->get()) {
            list1->push_back(ptr);
        }
        set_output("list1", std::move(list1));
    }
};

ZENDEFNODE(ExtendList, {
    {
        {gParamType_List, "list1", "", zeno::Socket_ReadOnly, NoMultiSockPanel},
        {gParamType_List, "list2", "", zeno::Socket_ReadOnly, NoMultiSockPanel}
    },
    {{gParamType_List, "list1"}},
    {},
    {"list"},
});


struct ResizeList : zeno::INode {
    virtual void apply() override {
        auto list = get_input<zeno::ListObject>("list");
        auto newSize = get_input<zeno::NumericObject>("newSize")->get<int>();
        list->resize(newSize);
        set_output("list", std::move(list));
    }
};

ZENDEFNODE(ResizeList, {
    {{gParamType_List, "list"}, {gParamType_Int, "newSize"}},
    {{gParamType_List, "list"}},
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
                for (auto const &obj: objlist->get()) {
                    list->push_back(std::move(obj));
                }
            } else {
                auto obj = get_input(name);
                list->push_back(std::move(obj));
            }
        }
        set_output("list", std::move(list));
    }
};

ZENDEFNODE(MakeSmallList, {
    {
        {"object", "obj0"},
        {"object", "obj1"},
        {"object", "obj2"},
        {"object", "obj3"},
        {"object", "obj4"},
        {"object", "obj5"}
    },
    {{gParamType_List, "list"}},
    {{gParamType_Bool, "doConcat", "1"}},
    {"list"},
});

struct MakeList : zeno::INode {
    virtual void apply() override {
        auto list = get_input<zeno::ListObject>("objs");
        set_output("list", std::move(list));
    }
};

ZENDEFNODE(MakeList, {
    {{gParamType_List, "objs", "", zeno::Socket_ReadOnly}},
    {{gParamType_List, "list"}},
    {},
    {"list"},
    });


struct NumericRangeList : zeno::INode {
    virtual void apply() override {
        auto list = std::make_shared<zeno::ListObject>();
        auto start = get_input2<int>("start");
        auto end = get_input2<int>("end");
        auto skip = get_input2<int>("skip");
        for (int i = start; i < end; i += skip) {
            list->emplace_back(std::make_shared<NumericObject>(i));
        }
        set_output("list", std::move(list));
    }
};

ZENDEFNODE(NumericRangeList, {
    {{gParamType_Int,"start","0"},{gParamType_Int,"end","1"},{gParamType_Int,"skip","1"}},
    {{gParamType_List, "list"}},
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
    {
        {gParamType_List, "list", "", zeno::Socket_ReadOnly},
    },
    {{gParamType_Int,"result"}},
    {},
    {"list"},
});

/*#ifdef ZENO_VISUALIZATION
struct ToVisualize_ListObject : zeno::INode {
    virtual void apply() override {
        auto list = get_input<ListObject>("list");
        auto path = get_param<std::string>("path");
        for (int i = 0; i < list->size(); i++) {
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
        {{gParamType_String, "path", ""}},
        {"list"},
});
#endif*/

}
