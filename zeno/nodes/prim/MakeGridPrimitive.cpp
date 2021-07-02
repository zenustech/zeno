#include <cassert>
#include <cstdlib>
#include <cstring>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/vec.h>
#include <zeno/zeno.h>

namespace zeno {

struct Make2DGridPrimitive : INode {
  virtual void apply() override {
    size_t nx = (size_t)(get_input<NumericObject>("nx")->get<float>());
    size_t ny = has_input("ny")
                    ? (size_t)(get_input<NumericObject>("ny")->get<float>())
                    : nx;
    float dx = has_input("dx") ? get_input<NumericObject>("dx")->get<float>()
                               : 1.f / std::max(nx - 1, (size_t)1);
    float dy = has_input("dy")
                   ? get_input<NumericObject>("dy")->get<float>()
                   : (has_input("dx") ? dx : 1.f / std::max(ny - 1, (size_t)1));
    vec3f ax = vec3f(1, 0, 0) *
               dx; // get_input<NumericObject>("sizeX")->get<vec3f>() * dx;
    vec3f ay = vec3f(0, 1, 0) *
               dy; // get_input<NumericObject>("sizeY")->get<vec3f>() * dy;
    vec3f o = has_input("origin")
                  ? get_input<NumericObject>("origin")->get<vec3f>()
                  : vec3f(0);

    if (get_param<int>("isCentered"))
      o -= (ax * (nx - 1) + ay * (ny - 1)) / 2;

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

    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(Make2DGridPrimitive, {/* inputs: */ {
                                     "nx",
                                     "ny",
                                     "dx",
                                     "dy",
                                     "sizeX",
                                     "sizeY",
                                     "origin",
                                 },
                                 /* outputs: */
                                 {
                                     "prim",
                                 },
                                 /* params: */
                                 {
                                     {"int", "isCentered", "0"},
                                 },
                                 /* category: */
                                 {
                                     "primitive",
                                 }});

} // namespace zeno
