#include <zeno/zeno.h>
#include <zeno/DictObject.h>
#include <zeno/StringObject.h>
#include <zeno/NumericObject.h>
#include <zeno/safe_at.h>


struct DictSize : zeno::INode {
    virtual void apply() override {
        auto dict = get_input<zeno::DictObject>("dict");
        auto ret = std::make_shared<zeno::NumericObject>();
        ret->set<int>(dict->lut.size());
        set_output("size", std::move(ret));
    }
};

ZENDEFNODE(DictSize, {
    {"dict"},
    {"size"},
    {},
    {"dict"},
});


struct ExtractDict : zeno::INode {
    virtual void apply() override {
        auto dict = get_input<zeno::DictObject>("dict");
        auto key = get_input<zeno::StringObject>("key")->get();
        auto obj = dict->lut.at(key);
        set_output("object", std::move(obj));
    }
};

ZENDEFNODE(ExtractDict, {
    {"dict", "key"},
    {"object"},
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
    {"dict"},
    {},
    {"dict"},
});


struct UpdateDict : zeno::INode {
    virtual void apply() override {
        auto dict = get_input<zeno::DictObject>("dict");
        auto key = get_input<zeno::StringObject>("key")->get();
        auto obj = get_input("object");
        dict->lut[key] = std::move(obj);
        set_output("dict", get_input("dict"));
    }
};

ZENDEFNODE(UpdateDict, {
    {"dict", "key", "object"},
    {"dict"},
    {},
    {"dict"},
});


struct MakeSmallDict : zeno::INode {
    virtual void apply() override {
        auto inkeys = get_param<std::string>("_KEYS");
        auto keys = zeno::split_str(inkeys, '\n');
        auto dict = std::make_shared<zeno::DictObject>();
        for (auto const &key: keys) {
            if (has_input(key)) {
                auto obj = get_input(key);
                dict->lut[key] = std::move(obj);
            }
        }
        set_output("dict", std::move(dict));
    }
};

ZENDEFNODE(MakeSmallDict, {
    {},
    {"dict"},
    {},
    {"dict"},
});


struct ExtractSmallDict : zeno::INode {
    virtual void apply() override {
        auto dict = get_input<zeno::DictObject>("dict");
        for (int i = 0; i < 4; i++) {
            std::stringstream namess;
            namess << "name" << i;
            auto name = namess.str();
            auto key = get_param<std::string>(name);
            if (key.size() == 0) break;
            auto obj = safe_at(dict->lut, key, "ExtractSmallDict key");
            std::stringstream namess2;
            namess2 << "obj" << i;
            name = namess2.str();
            set_output(name, std::move(obj));
        }
        set_output("dict", std::move(dict));
    }
};

ZENDEFNODE(ExtractSmallDict, {
    {"dict"},
    {},
    {},
    {"dict"},
});
