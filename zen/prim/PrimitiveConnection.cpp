#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/NumericObject.h>
#include <zen/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zenbase {


struct PrimitiveSimplePoints : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    size_t points_count = prim->size();
    prim->points.resize(points_count);
    for (int i = 0; i < points_count; i++) {
      prim->points[i] = i;
    }

    set_output_ref("prim", get_input_ref("prim"));
  }
};

static int defPrimitiveSimplePoints = zen::defNodeClass<PrimitiveSimplePoints>("PrimitiveSimplePoints",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveSimpleLines : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    size_t lines_count = prim->size() / 2;
    prim->lines.resize(lines_count);
    for (int i = 0; i < lines_count; i++) {
      prim->lines[i] = zen::vec2i(2 * i, 2 * i + 1);
    }

    set_output_ref("prim", get_input_ref("prim"));
  }
};

static int defPrimitiveSimpleLines = zen::defNodeClass<PrimitiveSimpleLines>("PrimitiveSimpleLines",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveSimpleTris : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    size_t tris_count = prim->size() / 3;
    prim->tris.resize(tris_count);
    for (int i = 0; i < tris_count; i++) {
      prim->tris[i] = zen::vec3i(3 * i, 3 * i + 1, 3 * i + 2);
    }

    set_output_ref("prim", get_input_ref("prim"));
  }
};

static int defPrimitiveSimpleTris = zen::defNodeClass<PrimitiveSimpleTris>("PrimitiveSimpleTris",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveSimpleQuads : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    size_t quads_count = prim->size() / 4;
    prim->quads.resize(quads_count);
    for (int i = 0; i < quads_count; i++) {
      prim->quads[i] = zen::vec4i(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3);
    }

    set_output_ref("prim", get_input_ref("prim"));
  }
};

static int defPrimitiveSimpleQuads = zen::defNodeClass<PrimitiveSimpleQuads>("PrimitiveSimpleQuads",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});

}
