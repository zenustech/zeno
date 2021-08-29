#include <zeno/zeno.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/string.h>

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
    {{"NumericObject:int", "size"}},
    {},
    {"dict"},
});


struct DictGetItem : zeno::INode {
    virtual void apply() override {
        auto dict = get_input<zeno::DictObject>("dict");
        auto key = get_input<zeno::StringObject>("key")->get();
        auto obj = dict->lut.at(key);
        set_output2("object", std::move(obj));
    }
};

ZENDEFNODE(DictGetItem, {
    {{"DictObject", "dict"}, {"StringObject", "key"}},
    {{"Any", "object"}},
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
        auto obj = get_input2("object");
        dict->lut[key] = std::move(obj);
        set_output("dict", std::move(dict));
    }
};

ZENDEFNODE(DictSetItem, {
    {{"DictObject", "dict"}, {"StringObject", "key"}, {"Any", "object"}},
    {{"DictObject", "dict"}},
    {},
    {"dict"},
});


struct MakeDict : zeno::INode {
    virtual void apply() override {
        auto inkeys = get_param<std::string>("_KEYS");
        auto keys = zeno::split_str(inkeys, '\n');
        auto dict = std::make_shared<zeno::DictObject>();
        for (auto const &key: keys) {
            if (has_input2(key)) {
                auto obj = get_input2(key);
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
    {{"dict1", "dict"}, {"dict2", "dict"}},
    {{"DictObject", "dict"}},
    {},
    {"dict"},
});



struct ExtractDict : zeno::INode {
    virtual void apply() override {
        auto inkeys = get_param<std::string>("_KEYS");
        auto keys = zeno::split_str(inkeys, '\n');
        auto dict = get_input<zeno::DictObject>("dict");
        for (auto const &key: keys) {
            auto it = dict->lut.find(key);
            if (it == dict->lut.end())
                continue;
            auto obj = it->second;
            set_output2(key, std::move(obj));
        }
    }
};

ZENDEFNODE(ExtractDict, {
    {{"DictObject", "dict"}},
    {},
    {},
    {"dict"},
});

}
