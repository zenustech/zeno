#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/vec.h>
#define _USE_MATH_DEFINES
#include <math.h>
//#include <spdlog/spdlog.h>

namespace zeno {
struct CreateCube : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();

        auto &pos = prim->verts;
        pos.push_back(vec3f( 1,  1,  1));
        pos.push_back(vec3f( 1,  1, -1));
        pos.push_back(vec3f(-1,  1, -1));
        pos.push_back(vec3f(-1,  1,  1));
        pos.push_back(vec3f( 1, -1,  1));
        pos.push_back(vec3f( 1, -1, -1));
        pos.push_back(vec3f(-1, -1, -1));
        pos.push_back(vec3f(-1, -1,  1));

        auto &tris = prim->tris;
        // Top 0, 1, 2, 3
        tris.push_back(vec3i(0, 1, 2));
        tris.push_back(vec3i(0, 2, 3));
        // Right 0, 4, 5, 1
        tris.push_back(vec3i(0, 4, 5));
        tris.push_back(vec3i(0, 5, 1));
        // Front 0, 3, 7, 4
        tris.push_back(vec3i(0, 3, 7));
        tris.push_back(vec3i(0, 7, 4));
        // Left 2, 6, 7, 3
        tris.push_back(vec3i(2, 6, 7));
        tris.push_back(vec3i(2, 7, 3));
        // Back 1, 5, 6, 2
        tris.push_back(vec3i(1, 5, 6));
        tris.push_back(vec3i(1, 6, 2));
        // Bottom 4, 7, 6, 5
        tris.push_back(vec3i(4, 7, 6));
        tris.push_back(vec3i(4, 6, 5));
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateCube, {
    {},
    {"prim"},
    {},
    {"create"},
});

struct CreateCone : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();

        auto &pos = prim->verts;
        size_t seg = 32;
        for (size_t i = 0; i < seg; i++) {
            float rad = 2 * M_PI * i / 32;
            pos.push_back(vec3f(cos(rad), -1, -sin(rad)));
        }
        // top
        pos.push_back(vec3i(0, 1, 0));
        // bottom
        pos.push_back(vec3i(0, -1, 0));

        auto &tris = prim->tris;
        for (size_t i = 0; i < seg; i++) {
            tris.push_back(vec3i(seg, i, (i + 1) % seg));
            tris.push_back(vec3i(i, seg + 1, (i + 1) % seg));
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateCone, {
    {},
    {"prim"},
    {},
    {"create"},
});

struct CreateDisk : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();

        auto &pos = prim->verts;
        size_t seg = 32;
        for (size_t i = 0; i < seg; i++) {
            float rad = 2 * M_PI * i / 32;
            pos.push_back(vec3f(cos(rad), 0, -sin(rad)));
        }
        pos.push_back(vec3i(0, 0, 0));

        auto &tris = prim->tris;
        for (size_t i = 0; i < seg; i++) {
            tris.push_back(vec3i(seg, i, (i + 1) % seg));
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateDisk, {
    {},
    {"prim"},
    {},
    {"create"},
});

struct CreatePlane : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();

        auto &pos = prim->verts;
        pos.push_back(vec3f( 1, 0,  1));
        pos.push_back(vec3f( 1, 0, -1));
        pos.push_back(vec3f(-1, 0, -1));
        pos.push_back(vec3f(-1, 0,  1));

        auto &tris = prim->tris;
        tris.push_back(vec3i(0, 1, 2));
        tris.push_back(vec3i(0, 2, 3));

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreatePlane, {
    {},
    {"prim"},
    {},
    {"create"},
});

struct CreateCylinder : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();

        auto &pos = prim->verts;
        size_t seg = 32;
        for (size_t i = 0; i < seg; i++) {
            float rad = 2 * M_PI * i / 32;
            pos.push_back(vec3f(cos(rad), 1, -sin(rad)));
        }
        for (size_t i = 0; i < seg; i++) {
            float rad = 2 * M_PI * i / 32;
            pos.push_back(vec3f(cos(rad), -1, -sin(rad)));
        }
        pos.push_back(vec3i(0, 1, 0));
        pos.push_back(vec3i(0, -1, 0));

        auto &tris = prim->tris;
        // Top
        for (size_t i = 0; i < seg; i++) {
            tris.push_back(vec3i(seg * 2, i, (i + 1) % seg));
        }
        // Bottom
        for (size_t i = 0; i < seg; i++) {
            tris.push_back(vec3i(i + seg, seg * 2 + 1, (i + 1) % seg + seg));
        }
        // Side
        for (size_t i = 0; i < seg; i++) {
            size_t _0 = i;
            size_t _1 = (i + 1) % seg;
            size_t _2 = (i + 1) % seg + seg;
            size_t _3 = i + seg;
            tris.push_back(vec3i(_1, _0, _2));
            tris.push_back(vec3i(_2, _0, _3));
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateCylinder, {
    {},
    {"prim"},
    {},
    {"create"},
});

struct CreateSphere : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();

        size_t seg = 32;

        std::vector<vec3f> uvs;
        uvs.reserve(19 * 33);
        auto &pos = prim->verts;
        auto &nrm = prim->add_attr<zeno::vec3f>("nrm");
        for (auto i = -90; i <= 90; i += 10) {
            float r = cos(i / 180.0 * M_PI);
            float h = sin(i / 180.0 * M_PI);
            for (size_t j = 0; j <= seg; j++) {
                float rad = 2 * M_PI * j / 32;
                pos.push_back(vec3f(cos(rad) * r, h, -sin(rad) * r));
                uvs.push_back(vec3f(j / 32.0, i / 90.0 * 0.5 + 0.5, 0));
                nrm.push_back(zeno::normalize(pos[pos.size()-1]));
            }
        }

        auto &tris = prim->tris;
        auto &uv0  = tris.add_attr<zeno::vec3f>("uv0");
        auto &uv1  = tris.add_attr<zeno::vec3f>("uv1");
        auto &uv2  = tris.add_attr<zeno::vec3f>("uv2");
        size_t count = 0;
        for (auto i = -90; i < 90; i += 10) {
            for (size_t i = 0; i < seg; i++) {
                size_t _0 = i + (seg + 1) * count;
                size_t _1 = i + 1 + (seg + 1) * count;
                size_t _2 = i + 1 + (seg + 1) * (count + 1);
                size_t _3 = i + (seg + 1) * (count + 1);
                tris.push_back(vec3i(_1, _0, _2));
                tris.attr<zeno::vec3f>("uv0").push_back(uvs[_1]);
                tris.attr<zeno::vec3f>("uv1").push_back(uvs[_0]);
                tris.attr<zeno::vec3f>("uv2").push_back(uvs[_2]);

                tris.push_back(vec3i(_2, _0, _3));
                tris.attr<zeno::vec3f>("uv0").push_back(uvs[_2]);
                tris.attr<zeno::vec3f>("uv1").push_back(uvs[_0]);
                tris.attr<zeno::vec3f>("uv2").push_back(uvs[_3]);
            }
            count += 1;
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateSphere, {
    {},
    {"prim"},
    {},
    {"create"},
});

}
