#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/random.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>

namespace zeno {

struct PrimitiveFillAttr : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto value = get_input<NumericObject>("value")->value;
    auto attrName = get_param<std::string>("attrName"));
    auto attrType = get_param<std::string>("attrType"));
    if (std::holds_alternative<vec3f>(value)) {
        attrType = "float3";
    }
    if (!prim->has_attr(attrName)) {
        if (attrType == "float3") prim->add_attr<vec3f>(attrName);
        else if (attrType == "float") prim->add_attr<float>(attrName);
    }
    auto &arr = prim->attr(attrName);
    std::visit([](auto &arr, auto const &value) {
        if constexpr (is_vec_castable_v<decltype(arr[0]), decltype(value)>) {
            #pragma omp parallel for
            for (int i = 0; i < arr.size(); i++) {
                arr[i] = decltype(arr[i])(value);
            }
        } else {
            throw Exception((std::string)"Failed to promote variant type from " + typeid(value).name() + " to " + typeid(arr[0]).name());
        }
    }, arr, value);

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveFillAttr,
    { /* inputs: */ {
    "prim",
    {"NumericObject", "value"},
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"string", "attrName", "pos"},
    {"enum float float3 none", "attrType", "none"},
    }, /* category: */ {
    "primitive",
    }});


void print_cout(float x) {
    printf("%f\n", x);
}

void print_cout(vec3f const &a) {
    printf("%f %f %f\n", a[0], a[1], a[2]);
}


struct PrimitivePrintAttr : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto attrName = get_param<std::string>("attrName"));
    prim->attr_visit(attrName, [attrName](auto const &arr) {
        printf("attribute `%s`, length %zd:\n", attrName.c_str(), arr.size());
        for (int i = 0; i < arr.size(); i++) {
            print_cout(arr[i]);
        }
        if (arr.size() == 0) {
            printf("(no data)\n");
        }
        printf("\n");
    });

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitivePrintAttr,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"string", "attrName", "pos"},
    }, /* category: */ {
    "primitive",
    }});


// deprecated: use PrimitiveRandomAttr instead
struct PrimitiveRandomizeAttr : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto min = get_param<float>("min"));
    auto minY = get_param<float>("minY"));
    auto minZ = get_param<float>("minZ"));
    auto max = get_param<float>("max"));
    auto maxY = get_param<float>("maxY"));
    auto maxZ = get_param<float>("maxZ"));
    auto attrName = get_param<std::string>("attrName"));
    auto attrType = get_param<std::string>("attrType"));
    if (!prim->has_attr(attrName)) {
        if (attrType == "float3") prim->add_attr<vec3f>(attrName);
        else if (attrType == "float") prim->add_attr<float>(attrName);
    }
    prim->attr_visit(attrName, [min, minY, minZ, max, maxY, maxZ](auto &arr) {
        for (int i = 0; i < arr.size(); i++) {
            if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {
                // note: can't parallelize cuz frand() uses drand48() or rand()
                vec3f f(frand(), frand(), frand());
                vec3f a(min, minY, minZ);
                vec3f b(max, maxY, maxZ);
                arr[i] = mix(a, b, f);
            } else {
                arr[i] = mix(min, max, (float)frand());
            }
        }
    });

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveRandomizeAttr,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"string", "attrName", "pos"},
    {"enum float float3", "attrType", "float3"},
    {"float", "min", "-1"},
    {"float", "minY", "-1"},
    {"float", "minZ", "-1"},
    {"float", "max", "1"},
    {"float", "maxY", "1"},
    {"float", "maxZ", "1"},
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveRandomAttr : INode {
  virtual void apply() override {
    auto prim = has_input("prim") ?
        get_input<PrimitiveObject>("prim") :
        std::make_shared<PrimitiveObject>();
    auto min = get_input<NumericObject>("min");
    auto max = get_input<NumericObject>("max");
    auto attrName = get_param<std::string>("attrName"));
    auto attrType = get_param<std::string>("attrType"));
    if (!prim->has_attr(attrName)) {
        if (attrType == "float3") prim->add_attr<vec3f>(attrName);
        else if (attrType == "float") prim->add_attr<float>(attrName);
    }
    prim->attr_visit(attrName, [&](auto &arr) {
        for (int i = 0; i < arr.size(); i++) {
            if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {
                vec3f f(frand(), frand(), frand());
                auto a = min->is<float>() ? (vec3f)min->get<float>() : min->get<vec3f>();
                auto b = max->is<float>() ? (vec3f)max->get<float>() : max->get<vec3f>();
                arr[i] = mix(a, b, f);
            } else {
                float f(frand());
                auto a = min->get<float>();
                auto b = max->get<float>();
                arr[i] = mix(a, b, f);
            }
        }
    });

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveRandomAttr,
    { /* inputs: */ {
    "prim",
    {"NumericObject", "min", "-1"},
    {"NumericObject", "max", "1"},
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"string", "attrName", "pos"},
    {"enum float float3", "attrType", ""},
    }, /* category: */ {
    "primitive",
    }});

}
