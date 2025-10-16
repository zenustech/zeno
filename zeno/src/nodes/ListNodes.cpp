#include <zeno/zeno.h>
#include <zeno/types/ListObject_impl.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/string.h>
#include <sstream>

namespace zeno {

struct ListLength : zeno::INode {
    virtual void apply() override {
        auto list = ZImpl(get_input<zeno::ListObject>("list"));
        auto ret = std::make_unique<zeno::NumericObject>();
        ret->set<int>(list->m_impl->size());
        ZImpl(set_output("length", std::move(ret)));
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
        auto index = ZImpl(get_input<zeno::NumericObject>("index"))->get<int>();
        if (ZImpl(has_input<DictObject>("list"))) {
            auto dict = ZImpl(get_input<zeno::DictObject>("list"));
            if (index < 0) {
                index += dict->lut.size();
            }
            if (index < 0 || index >= dict->lut.size())
                throw makeError<IndexError>(index, dict->lut.size(), "ListGetItem (for dict)");
            auto obj = std::next(dict->lut.begin(), index)->second->clone();
            ZImpl(set_output("object", std::move(obj)));
        } else {
            auto list = ZImpl(get_input<zeno::ListObject>("list"));
            if (index < 0) {
                index += list->size();
            }
            if (index < 0 || index >= list->m_impl->size())
                throw makeError<IndexError>(index, list->m_impl->size(), "ListGetItem");
            auto obj = list->m_impl->get(index)->clone();
            ZImpl(set_output("object", std::move(obj)));
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
        auto inkeys = ZImpl(get_param<std::string>("_KEYS"));
        auto keys = zeno::split_str(inkeys, '\n');
        auto list = ZImpl(get_input<zeno::ListObject>("list"));
        for (auto const& key : keys) {
            int index = std::stoi(key);
            if (list->m_impl->size() > index) {
                auto obj = list->m_impl->get(index)->clone();
                ZImpl(set_output(key, std::move(obj)));
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
        auto list = std::make_unique<zeno::ListObject>();
        ZImpl(set_output("list", std::move(list)));
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
        auto list = ZImpl(get_input<zeno::ListObject>("list"));
        auto obj = ZImpl(clone_input("object"));
        list->m_impl->push_back(std::move(obj));
        ZImpl(set_output("list", ZImpl(clone_input("list"))));
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
        auto list1 = ZImpl(get_input<zeno::ListObject>("list1"));
        auto list2 = ZImpl(get_input<zeno::ListObject>("list2"));
        for (auto const &ptr: list2->m_impl->get()) {
            list1->m_impl->push_back(ptr->clone());
        }
        ZImpl(set_output("list1", std::move(list1)));
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
        auto list = ZImpl(get_input<zeno::ListObject>("list"));
        auto newSize = ZImpl(get_input<zeno::NumericObject>("newSize")->get<int>());
        list->m_impl->resize(newSize);
        ZImpl(set_output("list", std::move(list)));
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
        auto list = std::make_unique<zeno::ListObject>();
        auto doConcat = ZImpl(get_param<bool>("doConcat"));
        for (int i = 0; i < 6; i++) {
            std::stringstream namess;
            namess << "obj" << i;
            auto name = namess.str();
            if (!ZImpl(has_input(name))) break;
            if (doConcat && ZImpl(has_input<ListObject>(name))) {
                auto objlist = ZImpl(get_input<ListObject>(name));
                for (auto const &obj: objlist->m_impl->get()) {
                    list->m_impl->push_back(std::move(obj->clone()));
                }
            } else {
                auto obj = ZImpl(clone_input(name));
                list->m_impl->push_back(std::move(obj));
            }
        }
        ZImpl(set_output("list", std::move(list)));
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
        auto list = ZImpl(get_input<zeno::ListObject>("objs"));
        ZImpl(set_output("list", std::move(list)));
    }
};

ZENDEFNODE(MakeList, {
    {{gParamType_List, "objs", "", zeno::Socket_ReadOnly}},
    {{gParamType_List, "list"}},
    {},
    {"list"},
    });


struct MergeList : zeno::INode {
    virtual void apply() override {
        auto list1 = ZImpl(get_input<zeno::ListObject>("list1"));
        auto list2 = ZImpl(get_input<zeno::ListObject>("list2"));
        auto lst = create_ListObject();
        for (const auto& inobj : list1->get()) {
            lst->push_back(inobj->clone());
        }
        for (const auto& inobj : list2->get()) {
            lst->push_back(inobj->clone());
        }
        set_output("list", std::move(lst));
    }
};

ZENDEFNODE(MergeList, {
    {
        {gParamType_List, "list1"},
        {gParamType_List, "list2"}
    },
    {
        {gParamType_List, "list"}
    },
    {},
    {"list"},
    });


struct NumericRangeList : zeno::INode {
    virtual void apply() override {
        auto list = std::make_unique<zeno::ListObject>();
        auto start = ZImpl(get_input2<int>("start"));
        auto end = ZImpl(get_input2<int>("end"));
        auto skip = ZImpl(get_input2<int>("skip"));
        for (int i = start; i < end; i += skip) {
            list->m_impl->emplace_back(std::make_unique<NumericObject>(i));
        }
        ZImpl(set_output("list", std::move(list)));
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
        auto result = std::make_unique<zeno::NumericObject>();
        result->value = 0;
        if (ZImpl(has_input<zeno::ListObject>("list")))
            result->value = 1;
        ZImpl(set_output("result", std::move(result)));
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
        auto list = ZImpl(get_input<ListObject>("list");
        auto path = ZImpl(get_param<std::string>("path");
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

