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
            get_input<NumericObject>("ny")->get<int>() : 0;
        if (!ny) ny = nx;
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
            auto obj = get_input<NumericObject>("scale");
            auto scale = obj->is<int>() ? obj->get<int>() : obj->get<float>();
            ax *= scale;
            ay *= scale;
        }
    auto dir = get_param<std::string>("Direction");
    if(dir == "YZ")
    {
        ax = zeno::vec3f(0,ax[0],0);
        ay = zeno::vec3f(0, 0, ay[1]);
    }
    if(dir == "XZ")
    {
        ay = zeno::vec3f(0,0,ay[1]);
    }

    if (get_param<bool>("isCentered"))
      o -= (ax + ay) / 2;
    ax *= dx; ay *= dy;

    auto prim = std::make_shared<PrimitiveObject>();
    prim->resize(nx * ny);
    auto &pos = prim->add_attr<vec3f>("pos");
    // for (size_t y = 0; y < ny; y++) {
    //     for (size_t x = 0; x < nx; x++) {
#pragma omp parallel for
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
    prim->userData.get("nx") = std::make_shared<NumericObject>((int)nx);
    prim->userData.get("ny") = std::make_shared<NumericObject>((int)ny);
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(Make2DGridPrimitive,
        { /* inputs: */ {
        {"int", "nx", "2"},
        {"int", "ny", "0"},
        {"vec3f", "sizeX", "1,0,0"},
        {"vec3f", "sizeY", "0,1,0"},
        {"float", "scale", "1"},
        {"vec3f", "origin", "0,0,0"},
        }, /* outputs: */ {
        {"PrimitiveObject", "prim"},
        }, /* params: */ {
        {"enum XZ XY YZ", "Direction", "XZ"},
        {"bool", "isCentered", "0"},
        {"bool", "hasFaces", "1"},
        }, /* category: */ {
        "primitive",
        }});

struct Make3DGridPrimitive : INode {
    virtual void apply() override {
        size_t nx = get_input<NumericObject>("nx")->get<int>();
        size_t ny = has_input("ny") ?
            get_input<NumericObject>("ny")->get<int>() : 0;
        if (!ny) ny = nx;
        size_t nz = has_input("nz") ?
            get_input<NumericObject>("nz")->get<int>() : 0;
        if (!nz) nz = nx;
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
    // for (size_t y = 0; y < ny; y++) {
    //     for (size_t x = 0; x < nx; x++) {
#pragma omp parallel for
    for (int index = 0; index < nx * ny * nz; index++) {
      int x = index % nx;
      int y = index / nx % ny;
      int z = index / nx / ny;
      vec3f p = o + x * ax + y * ay + z * az;
      pos[index] = p;
    }
      // }
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(Make3DGridPrimitive,
        { /* inputs: */ {
        {"int", "nx", "2"},
        {"int", "ny", "0"},
        {"int", "nz", "0"},
        {"vec3f", "sizeX", "1,0,0"},
        {"vec3f", "sizeY", "0,1,0"},
        {"vec3f", "sizeZ", "0,0,1"},
        {"float", "scale", "1"},
        {"vec3f", "origin", "0,0,0"},
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        {"bool", "isCentered", "0"},
        }, /* category: */ {
        "primitive",
        }});


struct Make3DGridPointsInAABB : INode {
    virtual void apply() override {
        size_t nx = get_input<NumericObject>("nx")->get<int>();
        size_t ny = has_input("ny") ?
            get_input<NumericObject>("ny")->get<int>() : 0;
        if (!ny) ny = nx;
        size_t nz = has_input("nz") ?
            get_input<NumericObject>("nz")->get<int>() : 0;
        if (!nz) nz = nx;
        float dx = 1.f / std::max(nx - 1, (size_t)1);
        float dy = 1.f / std::max(ny - 1, (size_t)1);
        float dz = 1.f / std::max(nz - 1, (size_t)1);

        vec3f bmin = has_input("bmin") ?
            get_input<NumericObject>("bmin")->get<vec3f>() : vec3f(-1);
        vec3f bmax = has_input("bmax") ?
            get_input<NumericObject>("bmax")->get<vec3f>() : vec3f(1);
        auto delta = (bmax - bmin) * vec3f(dx, dy, dz);

        if (get_param<bool>("isStaggered")) {
            nx--, ny--, nz--;
            bmin += 0.5f * delta;
        }

        auto prim = std::make_shared<PrimitiveObject>();
        prim->resize(nx * ny * nz);
        auto &pos = prim->add_attr<vec3f>("pos");
#pragma omp parallel for
        for (int index = 0; index < nx * ny * nz; index++) {
            int x = index % nx;
            int y = index / nx % ny;
            int z = index / nx / ny;
            vec3f p = bmin + vec3f(x, y, z) * delta;
            pos[index] = p;
        }
        set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(Make3DGridPointsInAABB,
        { /* inputs: */ {
        {"int", "nx", "4"},
        {"int", "ny", "0"},
        {"int", "nz", "0"},
        {"vec3f", "bmin", "-1,-1,-1"},
        {"vec3f", "bmax", "1,1,1"},
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        {"bool", "isStaggered", "1"},
        }, /* category: */ {
        "primitive",
        }});


// TODO: deprecate this xuben-happy node
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

struct MakeBoxPrimitive : INode {
    virtual void apply() override {
        float size_x = get_input<NumericObject>("size_x")->get<float>();
        float size_y = get_input<NumericObject>("size_y")->get<float>();
        float size_z = get_input<NumericObject>("size_z")->get<float>();
        vec3f o = get_input<NumericObject>("origin")->get<vec3f>();
        auto prim = std::make_shared<PrimitiveObject>();
        prim->resize(8);
        auto& pos = prim->add_attr<vec3f>("pos");

        for (int index = 0; index < 8; index++) {
            int x = index / 2 / 2;
            int y = index / 2 % 2;
            int z = index % 2;
            vec3f p = o + vec3f(size_x * (x - 0.5), size_y * (y - 0.5), size_z * (z - 0.5));
            pos[index] = p;
        }

        if (get_param<bool>("use_quads")) {
            std::vector<vec4i> cubeQuads{
            vec4i(0, 4, 5, 1),
            vec4i(4, 6, 7, 5),
            vec4i(2, 3, 7, 6),
            vec4i(0, 1, 3, 2),
            vec4i(1, 5, 7, 3),
            vec4i(0, 2, 6, 4), };
            prim->quads.values = cubeQuads;
            set_output("prim", std::move(prim));
        }
        else {
            std::vector<vec3i> cubeTris{
            vec3i(4, 2, 0),
            vec3i(2, 7, 3),
            vec3i(6, 5, 7),
            vec3i(1, 7, 5),
            vec3i(0, 3, 1),
            vec3i(4, 1, 5),
            vec3i(4, 6, 2),
            vec3i(2, 6, 7),
            vec3i(6, 4, 5),
            vec3i(1, 3, 7),
            vec3i(0, 2, 3),
            vec3i(4, 0, 1) };
            prim->tris.values = cubeTris;
            set_output("prim", std::move(prim));
        }
    }
};
ZENO_DEFNODE(MakeBoxPrimitive)(
    { /* inputs: */ {
        {"float","size_x","2.0"},
        {"float","size_y","2.0"},
        {"float","size_z","2.0"},
        {"vec3f","origin","0,0,0"},
    }, /* outputs: */ {
        "prim",
    }, /* params: */ {
        {"bool","use_quads","0"},
    }, /* category: */ {
        "primitive",
    } }
);

} // namespace zeno
