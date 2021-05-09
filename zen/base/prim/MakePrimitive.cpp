#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/NumericObject.h>
#include <Hg/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zenbase {


struct MakePrimitive : zen::INode {
  virtual void apply() override {
    auto prim = zen::IObject::make<PrimitiveObject>();
    set_output("prim", prim);
  }
};

static int defMakePrimitive = zen::defNodeClass<MakePrimitive>("MakePrimitive",
    { /* inputs: */ {
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveGetSize : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto size = zen::IObject::make<NumericObject>();
    size->set<int>(prim->size());
    set_output("size", size);
  }
};

static int defPrimitiveGetSize = zen::defNodeClass<PrimitiveGetSize>("PrimitiveGetSize",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "size",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveResize : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto size = get_input("size")->as<NumericObject>()->get<int>();
    prim->resize(size);

    set_output_ref("prim", get_input_ref("prim"));
  }
};

static int defPrimitiveResize = zen::defNodeClass<PrimitiveResize>("PrimitiveResize",
    { /* inputs: */ {
    "prim",
    "size",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveAddAttr : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto name = std::get<std::string>(get_param("name"));
    auto type = std::get<std::string>(get_param("type"));
    if (type == "float") {
        prim->add_attr<float>(name);
    } else if (type == "float3") {
        prim->add_attr<hg::vec3f>(name);
    } else {
        printf("%s\n", type.c_str());
        assert(0 && "Bad attribute type");
    }

    set_output_ref("prim", get_input_ref("prim"));
  }
};

static int defPrimitiveAddAttr = zen::defNodeClass<PrimitiveAddAttr>("PrimitiveAddAttr",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"string", "name", "pos"},
    {"string", "type", "float3"},
    }, /* category: */ {
    "primitive",
    }});


}
