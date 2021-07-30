#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {


struct MakePrimitive : zeno::INode {
  virtual void apply() override {
    auto prim = std::make_shared<PrimitiveObject>();
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(MakePrimitive,
    { /* inputs: */ {
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveGetSize : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto size = std::make_shared<NumericObject>();
    size->set<int>(prim->size());
    set_output("size", std::move(size));
  }
};

ZENDEFNODE(PrimitiveGetSize,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "size",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveResize : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto size = get_input<NumericObject>("size")->get<int>();
    prim->resize(size);

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveResize,
    { /* inputs: */ {
    "prim",
    "size",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveAddAttr : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto name = std::get<std::string>(get_param("name"));
    auto type = std::get<std::string>(get_param("type"));
    if (type == "float") {
      if(has_input("fillValue")){
        auto fillvalue = get_input<NumericObject>("fillValue")->get<float>();
        prim->add_attr<float>(name, fillvalue);
      }
      else {
        prim->add_attr<float>(name);
      }
    } else if (type == "float3") {
        if(has_input("fillValue")){
          auto fillvalue = get_input<NumericObject>("fillValue")->get<vec3f>();
          prim->add_attr<vec3f>(name, fillvalue);
        }
        else {
          prim->add_attr<vec3f>(name);
        }
    } else {
        printf("%s\n", type.c_str());
        assert(0 && "Bad attribute type");
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
    {"string", "type", "float3"},
    }, /* category: */ {
    "primitive",
    }});


}
