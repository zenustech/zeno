#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

#ifdef _MSC_VER
static inline double drand48() {
	return rand() / (double)RAND_MAX;
}
#endif

namespace zeno {

template <class T, class S>
inline constexpr bool is_decay_same_v = std::is_same_v<std::decay_t<T>, std::decay_t<S>>;


struct PrimitiveFillAttr : zeno::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto const &value = get_input("value")->as<NumericObject>()->value;
    auto attrName = std::get<std::string>(get_param("attrName"));
    auto &arr = prim->attr(attrName);
    std::visit([](auto &arr, auto const &value) {
        if constexpr (zeno::is_vec_castable_v<decltype(arr[0]), decltype(value)>) {
            #pragma omp parallel for
            for (int i = 0; i < arr.size(); i++) {
                arr[i] = decltype(arr[i])(value);
            }
        } else {
            assert(0 && "Failed to promote variant type");
        }
    }, arr, value);

    set_output_ref("prim", get_input_ref("prim"));
  }
};

ZENDEFNODE(PrimitiveFillAttr,
    { /* inputs: */ {
    "prim",
    "value",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"string", "attrName", "pos"},
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveRandomizeAttr : zeno::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto min = std::get<float>(get_param("min"));
    auto minY = std::get<float>(get_param("minY"));
    auto minZ = std::get<float>(get_param("minZ"));
    auto max = std::get<float>(get_param("max"));
    auto maxY = std::get<float>(get_param("maxY"));
    auto maxZ = std::get<float>(get_param("maxZ"));
    auto attrName = std::get<std::string>(get_param("attrName"));
    auto &arr = prim->attr(attrName);
    std::visit([min, minY, minZ, max, maxY, maxZ](auto &arr) {
        for (int i = 0; i < arr.size(); i++) {
            if constexpr (is_decay_same_v<decltype(arr[i]), zeno::vec3f>) {
                zeno::vec3f f(drand48(), drand48(), drand48());
                zeno::vec3f a(min, minY, minZ);
                zeno::vec3f b(max, maxY, maxZ);
                arr[i] = zeno::mix(a, b, f);
            } else {
                arr[i] = zeno::mix(min, max, (float)drand48());
            }
        }
    }, arr);

    set_output_ref("prim", get_input_ref("prim"));
  }
};

ZENDEFNODE(PrimitiveRandomizeAttr,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"string", "attrName", "pos"},
    {"float", "min", "-1"},
    {"float", "minY", "-1"},
    {"float", "minZ", "-1"},
    {"float", "max", "1"},
    {"float", "maxY", "1"},
    {"float", "maxZ", "1"},
    }, /* category: */ {
    "primitive",
    }});


void print_cout(float x) {
    printf("%f\n", x);
}

void print_cout(zeno::vec3f const &a) {
    printf("%f %f %f\n", a[0], a[1], a[2]);
}


struct PrimitivePrintAttr : zeno::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto attrName = std::get<std::string>(get_param("attrName"));
    auto const &arr = prim->attr(attrName);
    std::visit([attrName](auto const &arr) {
        printf("attribute `%s`, length %d:\n", attrName.c_str(), arr.size());
        for (int i = 0; i < arr.size(); i++) {
            print_cout(arr[i]);
        }
        if (arr.size() == 0) {
            printf("(no data)\n");
        }
        printf("\n");
    }, arr);

    set_output_ref("prim", get_input_ref("prim"));
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

}
