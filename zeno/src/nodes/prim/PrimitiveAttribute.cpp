#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <cassert>

namespace zeno {

struct PrimitiveAddAttr : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_param<std::string>("name");
        auto type = get_param<std::string>("type");
        if (type == "float") {
            if (has_input("fillValue")) {
                auto fillvalue = get_input<NumericObject>("fillValue")->get<float>();
                prim->add_attr<float>(name, fillvalue);
            }
            else {
                prim->add_attr<float>(name);
            }
        }
        else if (type == "float3") {
            if (has_input("fillValue")) {
                auto fillvalue = get_input<NumericObject>("fillValue")->get<vec3f>();
                prim->add_attr<vec3f>(name, fillvalue);
            }
            else {
                prim->add_attr<vec3f>(name);
            }
        }
        else {
            throw Exception("Bad attribute type: " + type);
        }

        set_output("prim", get_input("prim"));
    }
};

ZENDEFNODE(PrimitiveAddAttr,
    { /* inputs: */ {
    "prim","fillValue",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"string", "name", "pos"},
    {"enum float float3", "type", "float3"},
    }, /* category: */ {
    "primitive",
    } });

struct PrimitiveGetAttrValue : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_param<std::string>(("name"));
        auto type = get_param<std::string>(("type"));
        auto index = get_input<zeno::NumericObject>("index")->get<int>();

        auto value = std::make_shared<zeno::NumericObject>();
        auto &it = prim->attr(name);

        if (type == "float") {
            value->set<float>(0);
                std::vector<float>& attr_arr = std::get<std::vector<float>>(it);
                if (index < attr_arr.size()) {
                    value->set<float>(attr_arr[index]);
                }
        }
        else if (type == "float3") {
            value->set<vec3f>(vec3f(0, 0, 0));
                std::vector<vec3f>& attr_arr = std::get<std::vector<vec3f>>(it);
                if (index < attr_arr.size()) {
                    value->set<vec3f>(attr_arr[index]);
                }
        }
        else {
            throw Exception("Bad attribute type: " + type);
        }
        set_output("value", std::move(value));
    }
};

ZENDEFNODE(PrimitiveGetAttrValue,
    { /* inputs: */ {
    "prim",{"int","index","0"},
    }, /* outputs: */ {
    "value",
    }, /* params: */ {
    {"string", "name", "pos"},
    {"enum float float3", "type", "float3"},
    }, /* category: */ {
    "primitive",
    } });

struct PrimitiveSetAttrValue : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_param<std::string>(("name"));
        auto type = get_param<std::string>(("type"));
        auto index = get_input<zeno::NumericObject>("index")->get<int>();
        auto &it = prim->attr(name);

        if (type == "float") {
            auto value = get_input<zeno::NumericObject>("value")->get<float>();
                std::vector<float>& attr_arr = std::get<std::vector<float>>(it);
                if (index < attr_arr.size()) {
                    attr_arr[index] = value;
                }
        }
        else if (type == "float3") {
            auto value = get_input<zeno::NumericObject>("value")->get<vec3f>();
                std::vector<vec3f>& attr_arr = std::get<std::vector<vec3f>>(it);
                if (index < attr_arr.size()) {
                    attr_arr[index] = value;
                }
        }
        else {
            throw Exception("Bad attribute type: " + type);
        }
    }
};

ZENDEFNODE(PrimitiveSetAttrValue,
    { /* inputs: */ {
    "prim",{"int","index","0"},"value",
    }, /* outputs: */ {
    }, /* params: */ {
    {"string", "name", "pos"},
    {"enum float float3", "type", "float3"},
    }, /* category: */ {
    "primitive",
    } });
}
