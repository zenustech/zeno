#include <zeno/zeno.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject_impl.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/safe_at.h>
#include <iostream>
namespace zeno {
namespace {

struct DictSize : zeno::INode {
    virtual void apply() override {
        auto dict = ZImpl(get_input<zeno::DictObject>("dict"));
        auto ret = std::unique_ptr<zeno::NumericObject>();
        ret->set<int>(dict->lut.size());
        ZImpl(set_output("size", std::move(ret)));
    }
};

ZENDEFNODE(DictSize, {
    {{gParamType_Dict,"dict", "", zeno::Socket_ReadOnly}},
    {{gParamType_Int, "size"}},
    {},
    {"dict"},
});


struct DictGetItem : zeno::INode {
    virtual void apply() override {
        auto dict = ZImpl(get_input<zeno::DictObject>("dict"));
        auto key = ZImpl(get_input<zeno::StringObject>("key"))->get();
        if (ZImpl(has_input("defl") && dict->lut.find(key) == dict->lut.end())) {
            auto obj = ZImpl(clone_input("defl"));
            ZImpl(set_output("object", std::move(obj)));
        } else {
            auto obj = safe_at(dict->lut, key, "DictGetItem")->clone();
            ZImpl(set_output("object", std::move(obj)));
        }
    }
};

ZENDEFNODE(DictGetItem, {
    {{gParamType_Dict,"dict", "", zeno::Socket_Owning}, {gParamType_String, "key"}, {gParamType_IObject, "defl", "", zeno::Socket_ReadOnly}},
    {{gParamType_IObject, "object"}},
    {},
    {"dict"},
});


struct EmptyDict : zeno::INode {
    virtual void apply() override {
        auto dict = std::unique_ptr<zeno::DictObject>();
        ZImpl(set_output("dict", std::move(dict)));
    }
};

ZENDEFNODE(EmptyDict, {
    {},
    {{gParamType_Dict,"dict"}},
    {},
    {"dict"},
});


struct DictSetItem : zeno::INode {
    virtual void apply() override {
        auto dict = ZImpl(get_input<zeno::DictObject>("dict"));
        auto key = ZImpl(get_input<zeno::StringObject>("key"))->get();
        auto obj = ZImpl(clone_input("object"));
        dict->lut[key] = std::move(obj);
        ZImpl(set_output("dict", std::move(dict)));
    }
};

ZENDEFNODE(DictSetItem, {
    {{gParamType_Dict,"dict", "", zeno::Socket_ReadOnly}, {gParamType_String, "key"}, {gParamType_IObject, "object", "", zeno::Socket_ReadOnly}},
    {{gParamType_Dict,"dict"}},
    {},
    {"dict"},
});


struct MakeDict : zeno::INode {
    virtual void apply() override {
        auto dict = ZImpl(get_input<zeno::DictObject>("objs"));
        ZImpl(set_output("dict", std::move(dict)));
    }
};

ZENDEFNODE(MakeDict, {
    {{gParamType_Dict, "objs", "", zeno::Socket_Owning}},
    {{gParamType_Dict,"dict"}},
    {},
    {"dict"},
});

struct MocDictAsOutput : zeno::INode {
    virtual void apply() override {
    
    }
};

ZENDEFNODE(MocDictAsOutput, {
    {},
    {{gParamType_Dict,"dict"}},
    {},
    {"dict2"},
});

struct MakeSmallDict : zeno::INode {
    virtual void apply() override {
        auto dict = std::make_unique<zeno::DictObject>();
        for (int i = 0; i < 6; i++) {
            auto si = std::to_string(i);
            if (!ZImpl(has_input("obj" + si))) break;
            auto obj = ZImpl(clone_input("obj" + si));
            auto key = ZImpl(get_input2<std::string>("key" + si));
            dict->lut.emplace(std::move(key), std::move(obj));
        }
        ZImpl(set_output("dict", std::move(dict)));
    }
};

ZENDEFNODE(MakeSmallDict, {
    {
        {gParamType_String, "key0"}, {"IObject", "obj0"},
        {gParamType_String, "key1"}, {"IObject", "obj1"},
        {gParamType_String, "key2"}, {"IObject", "obj2"},
        {gParamType_String, "key3"}, {"IObject", "obj3"},
        {gParamType_String, "key4"}, {"IObject", "obj4"},
        {gParamType_String, "key5"}, {"IObject", "obj5"},
    },
    {{gParamType_Dict, "dict"}},
    {},
    {"dict"},
});


struct DictUnion : zeno::INode {
    virtual void apply() override {
        auto dict1 = ZImpl(get_input<zeno::DictObject>("dict1"));
        auto dict2 = ZImpl(get_input<zeno::DictObject>("dict2"));
        auto dict = std::make_unique<zeno::DictObject>();
        dict->lut = std::move(dict1->lut);
        dict->lut.merge(dict2->lut);
        ZImpl(set_output("dict", std::move(dict)));
    }
};

ZENDEFNODE(DictUnion, {
    {{gParamType_Dict,"dict1", "", zeno::Socket_ReadOnly},
     {gParamType_Dict,"dict2", "", zeno::Socket_ReadOnly}},
    {{gParamType_Dict,"dict"}},
    {},
    {"dict"},
});



struct ExtractDict : zeno::INode {
    virtual void apply() override {
        auto dict = ZImpl(get_input<zeno::DictObject>("dict"));
        for (auto paramobj : ZImpl(get_output_object_params()))
        {
            const std::string& key = paramobj.name;
            auto it = dict->lut.find(key);
            if (it == dict->lut.end())
                continue;
            auto obj = dict->lut.at(key)->clone();
            ZImpl(set_output(key, std::move(obj)));
        }
    }
};

ZENDEFNODE(ExtractDict, {
    {{gParamType_Dict,"dict"}},
    {},
    {},
    {"dict"},
});


struct ExtractLegacyDict : ExtractDict {
    virtual void apply() override {
        auto dict = ZImpl(get_input<zeno::DictObject>("dict"));
        for (auto const &[key, val]: dict->lut) {
            //outputs[key];
        }
        ExtractDict::apply();
    }
};

ZENDEFNODE(ExtractLegacyDict, {
    {{gParamType_Dict,"dict"}},
    {},
    {},
    {"deprecated"},
});


struct DictGetKeyList : zeno::INode {
    virtual void apply() override {
        auto dict = ZImpl(get_input<zeno::DictObject>("dict"));
        auto keys = std::make_unique<zeno::ListObject>();
        for (auto const &[k, v]: dict->lut) {
            auto so = std::make_unique<zeno::StringObject>();
            so->set(k);
            keys->m_impl->push_back(std::move(so));
        }
        ZImpl(set_output("keys", std::move(keys)));
    }
};

ZENDEFNODE(DictGetKeyList, {
    {{gParamType_Dict,"dict"}},
    {{gParamType_List, "keys"}},
    {},
    {"dict"},
});

/*
struct DictHasKey : zeno::INode {
    virtual void apply() override {
        auto dict = ZImpl(get_input<zeno::DictObject>("dict"));
        auto key = ZImpl(get_input2<std::string>("key"));
        int count = dict->lut.count(key);
        ZImpl(set_output("hasKey", bool(count)));
    }
};

ZENDEFNODE(DictHasKey, {
    {{gParamType_Dict,"dict"}, {gParamType_String, "key"}},
    {{gParamType_Bool, "hasKey"}},
    {},
    {"dict"},
});
*/

}
}
