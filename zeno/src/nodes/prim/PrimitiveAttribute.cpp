#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <cassert>

namespace zeno {

struct PrimitiveAddAttr : zeno::INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto name = ZImpl(get_param<std::string>("name"));
        auto type = ZImpl(get_param<std::string>("type"));
        if (type == "float") {
            if (ZImpl(has_input("fillValue"))) {
                auto fillvalue = ZImpl(get_input<NumericObject>("fillValue"))->get<float>();
                prim->add_attr<float>(name, fillvalue);
            }
            else {
                prim->add_attr<float>(name);
            }
        }
        else if (type == "float3") {
            if (ZImpl(has_input("fillValue"))) {
                auto fillvalue = ZImpl(get_input<NumericObject>("fillValue"))->get<zeno::vec3f>();
                prim->add_attr<zeno::vec3f>(name, fillvalue);
            }
            else {
                prim->add_attr<zeno::vec3f>(name);
            }
        }
        else {
            throw Exception("Bad attribute type: " + type);
        }

        ZImpl(set_output("prim", ZImpl(clone_input("prim"))));
    }
};

ZENDEFNODE(PrimitiveAddAttr, 
    { 
        {
            {gParamType_Vec3f, "fillValue", "", zeno::Socket_ReadOnly},
            {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        }, 
    {{gParamType_Primitive, "prim"}},
    {
        {gParamType_String, "name", "clr"},
        {"enum float float3", "type", "float3"},
        {gParamType_String, "pybisgreat", "DEPRECATED! USE PrimFillAttr INSTEAD"},
    },
    {
    "deprecated",
    } });

struct PrimitiveDelAttr : zeno::INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto name = ZImpl(get_param<std::string>("name"));


        prim->verts.attrs.erase(name);
        prim->tris.attrs.erase(name);
        prim->lines.attrs.erase(name);
        prim->polys.attrs.erase(name);
        prim->loops.attrs.erase(name);

        ZImpl(set_output("prim", ZImpl(clone_input("prim"))));
    }
};

ZENDEFNODE(PrimitiveDelAttr,
    {
        {
            {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        },
    {{gParamType_Primitive, "prim"}},
    {
        {gParamType_String, "name", "nrm"},
    },
    {
    "deprecated",
    } });

struct PrimitiveGetAttrValue : zeno::INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto name = ZImpl(get_param<std::string>(("name")));
        auto type = ZImpl(get_param<std::string>(("type")));
        auto index = ZImpl(get_input<zeno::NumericObject>("index"))->get<int>();

        auto value = std::make_unique<zeno::NumericObject>();

        if (type == "float") {
            value->set<float>(0);
                std::vector<float>& attr_arr = prim->attr<float>(name);
                if (index < attr_arr.size()) {
                    value->set<float>(attr_arr[index]);
                }
        }
        else if (type == "float3") {
            value->set<vec3f>(vec3f(0, 0, 0));
                std::vector<vec3f>& attr_arr = prim->attr<zeno::vec3f>(name);
                if (index < attr_arr.size()) {
                    value->set<vec3f>(attr_arr[index]);
                }
        }
        else {
            throw Exception("Bad attribute type: " + type);
        }
        ZImpl(set_output("value", std::move(value)));
    }
};

ZENDEFNODE(PrimitiveGetAttrValue, {
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {gParamType_Int,"index","0"},
    }, 
    {
        {"NumericObject","value"},
    },
    {
        {gParamType_String, "name", "pos"},
        {"enum float float3", "type", "float3"},
    },
    {"deprecated"} 
});

struct PrimitiveSetAttrValue : zeno::INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto name = ZImpl(get_param<std::string>(("name")));
        auto type = ZImpl(get_param<std::string>(("type")));
        auto index = ZImpl(get_input<zeno::NumericObject>("index"))->get<int>();

        if (type == "float") {
            auto value = ZImpl(get_input<zeno::NumericObject>("value"))->get<float>();
                std::vector<float>& attr_arr = prim->add_attr<float>(name);
                if (index < attr_arr.size()) {
                    attr_arr[index] = value;
                }
        }
        else if (type == "float3") {
            auto value = ZImpl(get_input<zeno::NumericObject>("value"))->get<zeno::vec3f>();
                std::vector<vec3f>& attr_arr = prim->add_attr<zeno::vec3f>(name);
                if (index < attr_arr.size()) {
                    attr_arr[index] = value;
                }
        }
        else {
            throw Exception("Bad attribute type: " + type);
        }
        ZImpl(set_output("prim", std::move(prim)));
    }
};

ZENDEFNODE(PrimitiveSetAttrValue,{ 
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {gParamType_Int,"index","0"},
        {gParamType_Float,"value"},
    }, 
    {
{gParamType_Primitive, "prim"},
},
    {
        {gParamType_String, "name", "pos"},
        {"enum float float3", "type", "float3"},
    },
    {
    "deprecated",
    }
});

}
