#include <cassert>
#include <cstdlib>
#include <cstring>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>

namespace zeno {

struct Make2DGridPrimitive : INode {
    virtual void apply() override {
        size_t nx = get_input<NumericObject>("nx")->get<int>();
        size_t ny = has_input("ny") ?
            get_input<NumericObject>("ny")->get<int>() : nx;
        float dx = 1.f / std::max(nx - 1, (size_t)1);
        float dy = 1.f / std::max(ny - 1, (size_t)1);
        vec3f ax = has_input("sizeX") ?
            get_input<NumericObject>("sizeX")->get<vec3f>()
            : vec3f(1, 0, 0);
        vec3f ay = has_input("sizeY") ?
            get_input<NumericObject>("sizeY")->get<vec3f>()
            : vec3f(0, 1, 0);
        vec3f o = has_input("origin") ?
            get_input<NumericObject>("origin")->get<vec3f>() : vec3f(0);
        if (has_input("scale")) {
            auto scale = get_input<NumericObject>("scale")->get<float>();
            ax *= scale;
            ay *= scale;
        }


    if (get_param<bool>("isCentered"))
      o -= (ax + ay) / 2;
    ax *= dx; ay *= dy;

    auto prim = std::make_shared<PrimitiveObject>();
    prim->resize(nx * ny);
    auto &pos = prim->add_attr<vec3f>("pos");
#pragma omp parallel for
    // for (size_t y = 0; y < ny; y++) {
    //     for (size_t x = 0; x < nx; x++) {
    for (int index = 0; index < nx * ny; index++) {
      int x = index % nx;
      int y = index / nx;
      vec3f p = o + x * ax + y * ay;
      size_t i = x + y * nx;
      pos[i] = p;
      // }
    }
    if (get_param<bool>("hasFaces")) {
        prim->tris.resize((nx - 1) * (ny - 1) * 2);
#pragma omp parallel for
        for (int index = 0; index < (nx - 1) * (ny - 1); index++) {
          int x = index % (nx - 1);
          int y = index / (nx - 1);
          prim->tris[index * 2][0] = y * nx + x;
          prim->tris[index * 2][1] = y * nx + x + 1;
          prim->tris[index * 2][2] = (y + 1) * nx + x + 1;
          prim->tris[index * 2 + 1][0] = (y + 1) * nx + x + 1;
          prim->tris[index * 2 + 1][1] = (y + 1) * nx + x;
          prim->tris[index * 2 + 1][2] = y * nx + x;
        }
    }
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(Make2DGridPrimitive,
        { /* inputs: */ {
        {"NumericObject:int", "nx"},
        {"NumericObject:int", "ny"},
        {"NumericObject:vec3f", "sizeX"},
        {"NumericObject:vec3f", "sizeY"},
        {"NumericObject:float", "scale"},
        {"NumericObject:vec3f", "origin"},
        }, /* outputs: */ {
        {"PrimitiveObject", "prim"},
        }, /* params: */ {
        {"bool", "isCentered", "0"},
        {"bool", "hasFaces", "1"},
        }, /* category: */ {
        "primitive",
        }});

struct Make3DGridPrimitive : INode {
    virtual void apply() override {
        size_t nx = get_input<NumericObject>("nx")->get<int>();
        size_t ny = has_input("ny") ?
            get_input<NumericObject>("ny")->get<int>() : nx;
        size_t nz = has_input("nz") ?
            get_input<NumericObject>("nz")->get<int>() : nx;
        float dx = 1.f / std::max(nx - 1, (size_t)1);
        float dy = 1.f / std::max(ny - 1, (size_t)1);
        float dz = 1.f / std::max(nz - 1, (size_t)1);
        vec3f ax = has_input("sizeX") ?
            get_input<NumericObject>("sizeX")->get<vec3f>()
            : vec3f(1, 0, 0);
        vec3f ay = has_input("sizeY") ?
            get_input<NumericObject>("sizeY")->get<vec3f>()
            : vec3f(0, 1, 0);
        vec3f az = has_input("sizeZ") ?
            get_input<NumericObject>("sizeZ")->get<vec3f>()
            : vec3f(0, 0, 1);
        vec3f o = has_input("origin") ?
            get_input<NumericObject>("origin")->get<vec3f>() : vec3f(0);
        if (has_input("scale")) {
            auto scale = get_input<NumericObject>("scale")->get<float>();
            ax *= scale;
            ay *= scale;
            az *= scale;
        }


    if (get_param<bool>("isCentered"))
      o -= (ax + ay + az) / 2;
    ax *= dx; ay *= dy; az *= dz;

    auto prim = std::make_shared<PrimitiveObject>();
    prim->resize(nx * ny * nz);
    auto &pos = prim->add_attr<vec3f>("pos");
#pragma omp parallel for
    // for (size_t y = 0; y < ny; y++) {
    //     for (size_t x = 0; x < nx; x++) {
    for (int index = 0; index < nx * ny * nz; index++) {
      int x = index % nx;
      int y = index / nx % ny;
      int z = index / nx / ny;
      vec3f p = o + x * ax + y * ay + z * az;
      pos[index] = p;
      // }
    }
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(Make3DGridPrimitive,
        { /* inputs: */ {
        "nx", "ny", "nz", "sizeX", "sizeY", "sizeZ", "scale", "origin",
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        {"bool", "isCentered", "0"},
        }, /* category: */ {
        "primitive",
        }});

struct MakeCubePrimitive : INode {
    virtual void apply() override {
        float spacing = get_input<NumericObject>("spacing")->get<float>();
        size_t nx = get_input<NumericObject>("nx")->get<int>();
        size_t ny = has_input("ny") ?
            get_input<NumericObject>("ny")->get<int>() : nx;
        size_t nz = has_input("nz") ?
            get_input<NumericObject>("nz")->get<int>() : nx;

        vec3f o = has_input("origin") ?
            get_input<NumericObject>("origin")->get<vec3f>() : vec3f(0);
    
    auto prim = std::make_shared<PrimitiveObject>();
    prim->resize(nx * ny * nz);
    auto &pos = prim->add_attr<vec3f>("pos");
#pragma omp parallel for
    // for (size_t y = 0; y < ny; y++) {
    //     for (size_t x = 0; x < nx; x++) {
    for (int index = 0; index < nx * ny * nz; index++) {
      int x = index % nx;
      int y = index / nx % ny;
      int z = index / nx / ny;
      vec3f p = o + vec3f(x * spacing, y * spacing, z * spacing);
      pos[index] = p;
      // }
    }
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(MakeCubePrimitive,
        { /* inputs: */ {
        "spacing", "nx", "ny", "nz", "origin",
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        {},
        }, /* category: */ {
        "primitive",
        }});
} // namespace zeno
