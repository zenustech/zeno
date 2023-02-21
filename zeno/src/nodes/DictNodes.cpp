#include <zeno/zeno.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/safe_at.h>
#include <iostream>
namespace zeno {
namespace {

struct DictSize : zeno::INode {
    virtual void apply() override {
        auto dict = get_input<zeno::DictObject>("dict");
        auto ret = std::make_shared<zeno::NumericObject>();
        ret->set<int>(dict->lut.size());
        set_output("size", std::move(ret));
    }
};

ZENDEFNODE(DictSize, {
    {{"DictObject", "dict"}},
    {{"int", "size"}},
    {},
    {"dict"},
});


struct DictGetItem : zeno::INode {
    virtual void apply() override {
        auto dict = get_input<zeno::DictObject>("dict");
        auto key = get_input<zeno::StringObject>("key")->get();
        if (has_input("defl") && dict->lut.find(key) == dict->lut.end()) {
            auto obj = get_input("defl");
            set_output("object", std::move(obj));
        } else {
            auto obj = safe_at(dict->lut, key, "DictGetItem");
            set_output("object", std::move(obj));
        }
    }
};

ZENDEFNODE(DictGetItem, {
    {{"DictObject", "dict"}, {"string", "key"}, {"IObject", "defl"}},
    {{"zany", "object"}},
    {},
    {"dict"},
});


struct EmptyDict : zeno::INode {
    virtual void apply() override {
        auto dict = std::make_shared<zeno::DictObject>();
        set_output("dict", std::move(dict));
    }
};

ZENDEFNODE(EmptyDict, {
    {},
    {{"DictObject", "dict"}},
    {},
    {"dict"},
});


struct DictSetItem : zeno::INode {
    virtual void apply() override {
        auto dict = get_input<zeno::DictObject>("dict");
        auto key = get_input<zeno::StringObject>("key")->get();
        auto obj = get_input("object");
        dict->lut[key] = std::move(obj);
        set_output("dict", std::move(dict));
    }
};

ZENDEFNODE(DictSetItem, {
    {{"DictObject", "dict"}, {"string", "key"}, {"zany", "object"}},
    {{"DictObject", "dict"}},
    {},
    {"dict"},
});


struct MakeDict : zeno::INode {
    virtual void apply() override {
        auto dict = std::make_shared<zeno::DictObject>();
        for (auto pair : inputs)
        {
            const std::string &key = pair.first;
            if (key != "SRC")
            {
                auto obj = get_input(key);
                dict->lut[key] = std::move(obj);
            }
        }
        set_output("dict", std::move(dict));
    }
};

ZENDEFNODE(MakeDict, {
    {},
    {{"DictObject", "dict"}},
    {},
    {"dict"},
});

struct MocDictAsInput : zeno::INode {
    virtual void apply() override {
    
    }
};

ZENDEFNODE(MocDictAsInput, {
    {{"DictObject", "dict"}},
    {},
    {},
    {"dict2"},
});

struct MocDictAsOutput : zeno::INode {
    virtual void apply() override {
    
    }
};

ZENDEFNODE(MocDictAsOutput, {
    {},
    {{"DictObject", "dict"}},
    {},
    {"dict2"},
});

struct MakeSmallDict : zeno::INode {
    virtual void apply() override {
        auto dict = std::make_shared<zeno::DictObject>();
        for (int i = 0; i < 6; i++) {
            auto si = std::to_string(i);
            if (!has_input("obj" + si)) break;
            auto obj = get_input("obj" + si);
            auto key = get_input2<std::string>("key" + si);
            dict->lut.emplace(std::move(key), std::move(obj));
        }
        set_output("dict", std::move(dict));
    }
};

ZENDEFNODE(MakeSmallDict, {
    {
        {"string", "key0"}, {"IObject", "obj0"},
        {"string", "key1"}, {"IObject", "obj1"},
        {"string", "key2"}, {"IObject", "obj2"},
        {"string", "key3"}, {"IObject", "obj3"},
        {"string", "key4"}, {"IObject", "obj4"},
        {"string", "key5"}, {"IObject", "obj5"},
    },
    {"dict"},
    {},
    {"dict"},
});


struct ZipListAsDict : zeno::INode {
    virtual void apply() override {
        auto dict = std::make_shared<zeno::DictObject>();
        auto keys = get_input<ListObject>("values")->get2<std::string>();
        auto values = get_input<ListObject>("values")->get();
        for (int i = 0; i < values.size(); i++) {
            dict->lut.emplace(i < keys.size() ? keys[i] : std::to_string(i),
                              std::move(values[i]));
        }
        set_output("dict", std::move(dict));
    }
};

ZENDEFNODE(ZipListAsDict, {
    {
        {"ListObject", "keys"},
        {"ListObject", "values"},
    },
    {{"DictObject", "dict"}},
    {},
    {"dict"},
});


struct DictUnion : zeno::INode {
    virtual void apply() override {
        auto dict1 = get_input<zeno::DictObject>("dict1");
        auto dict2 = get_input<zeno::DictObject>("dict2");
        auto dict = std::make_shared<zeno::DictObject>();
        dict->lut = dict1->lut;
        dict->lut.merge(dict2->lut);
        set_output("dict", std::move(dict));
    }
};

ZENDEFNODE(DictUnion, {
    {{"DictObject", "dict1"}, {"DictObject", "dict2"}},
    {{"DictObject", "dict"}},
    {},
    {"dict"},
});



struct ExtractDict : zeno::INode {
    virtual void apply() override {
        auto dict = get_input<zeno::DictObject>("dict");
        for (auto pair : outputs)
        {
            const std::string& key = pair.first;
            auto it = dict->lut.find(key);
            if (it == dict->lut.end())
                continue;
            auto obj = dict->lut.at(key);
            set_output(key, std::move(obj));
        }
    }
};

ZENDEFNODE(ExtractDict, {
    {{"DictObject", "dict"}},
    {},
    {},
    {"dict"},
});


struct ExtractLegacyDict : ExtractDict {
    virtual void apply() override {
        auto dict = get_input<zeno::DictObject>("dict");
        for (auto const &[key, val]: dict->lut) {
            outputs[key];
        }
        ExtractDict::apply();
    }
};

ZENDEFNODE(ExtractLegacyDict, {
    {{"DictObject", "dict"}},
    {},
    {},
    {"deprecated"},
});


struct DictGetKeyList : zeno::INode {
    virtual void apply() override {
        auto dict = get_input<zeno::DictObject>("dict");
        auto keys = std::make_shared<zeno::ListObject>();
        for (auto const &[k, v]: dict->lut) {
            auto so = std::make_shared<zeno::StringObject>();
            so->set(k);
            keys->arr.push_back(std::move(so));
        }
        set_output("keys", std::move(keys));
    }
};

ZENDEFNODE(DictGetKeyList, {
    {{"DictObject", "dict"}},
    {{"ListObject", "keys"}},
    {},
    {"dict"},
});

struct DictHasKey : zeno::INode {
    virtual void apply() override {
        auto dict = get_input<zeno::DictObject>("dict");
        auto key = get_input2<std::string>("key");
        int count = dict->lut.count(key);
        set_output2("hasKey", bool(count));
    }
};

ZENDEFNODE(DictHasKey, {
    {{"DictObject", "dict"}, {"string", "key"}},
    {{"bool", "hasKey"}},
    {},
    {"dict"},
});

}
}
