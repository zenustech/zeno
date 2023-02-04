#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/log.h>

#include <vector>
#include <iostream>
#include <fstream>
using std::vector;
using zeno::vec3f;

namespace zeno {

static void write_vat(vector<vector<vec3f>> &v, const std::string &path) {
    std::ofstream file(path, std::ios::out | std::ios::binary);
    int height = v.size();
    file.write((char*)&height, sizeof(int));

    for (auto i = 0; i < height; i++) {
        int width = v[i].size();
        file.write((char*)&width, sizeof(int));
        for (auto j = 0; j < width; j++) {
            auto vec = v[i][j];
            file.write((char*)&vec[0], sizeof(float));
            file.write((char*)&vec[1], sizeof(float));
            file.write((char*)&vec[2], sizeof(float));
        }
        zeno::log_info("VAT: write frame {} done ({} face vec)!", i, width);
    }
}

static vector<vector<vec3f>> read_vat(const std::string &path) {
    vector<vector<vec3f>> v;
    std::ifstream file(path, std::ios::in | std::ios::binary);

    int height = 0;
    file.read((char*)&height, sizeof (int));
    v.resize(height);
    zeno::log_info("VAT: frames {}", height);

    for (auto i = 0; i < height; i++) {
        int width = 0;
        file.read((char*)&width, sizeof (int));
        zeno::log_info("VAT: read frame {} ({} face vec)!", i, width);

        v[i].resize(width);
        for (auto j = 0; j < width; j++) {
            zeno::vec3f vec;
            file.read((char*)&vec[0], sizeof (float));
            file.read((char*)&vec[1], sizeof (float));
            file.read((char*)&vec[2], sizeof (float));
            v[i][j] = vec;
        }
        zeno::log_info("VAT: read frame {} done ({} face vec)!", i, width);
    }
    return v;
}

struct WriteCustomVAT : INode {
    std::vector<std::shared_ptr<PrimitiveObject>> prims;
    virtual void apply() override {
        int frameid;
        if (has_input("frameid")) {
            frameid = get_param<int>("frameid");
        } else {
            frameid = getGlobalState()->frameid;
        }
        int frameStart = get_param<int>("frameStart");
        int frameEnd = get_param<int>("frameEnd");
        int frameCount = frameEnd - frameStart + 1;
        if (frameid == frameStart) {
            prims.resize(frameCount);
        }
        auto prim = std::dynamic_pointer_cast<PrimitiveObject>(get_input<PrimitiveObject>("prim")->clone());
        if (frameStart <= frameid && frameid <= frameEnd) {
            prims[frameid - frameStart] = prim;
        }
        if (frameid == frameEnd) {
            vector<vector<vec3f>> v;
            v.resize(prims.size());
            for (auto i = 0; i < prims.size(); i++) {
                auto prim = prims[i];
                v[i].resize(prim->tris.size() * 3);
                for (auto j = 0; j < prim->tris.size(); j++) {
                    const auto & tri = prim->tris[j];
                    v[i][j * 3 + 0] = prim->verts[tri[0]];
                    v[i][j * 3 + 1] = prim->verts[tri[1]];
                    v[i][j * 3 + 2] = prim->verts[tri[2]];
                }
            }
            std::string path = get_param<std::string>("path");
            write_vat(v, path);
            zeno::log_info("VAT: save success!");
        }
    }
};

ZENDEFNODE(WriteCustomVAT, {
    {
        {"prim"},
        {"frameid"},
    },
    {},
    {
        {"writepath", "path", ""},
        {"int", "frameStart", "0"},
        {"int", "frameEnd", "100"},
    },
    {"primitive"},
});

struct ReadCustomVAT : INode {
    vector<vector<vec3f>> v;
    virtual void apply() override {
        if (v.empty()) {
            std::string path = get_param<std::string>("path");
            v = read_vat(path);
        }

        int frameid;
        if (has_input("frameid")) {
            frameid = get_param<int>("frameid");
        } else {
            frameid = getGlobalState()->frameid;
        }
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        if (frameid < v.size()) {
            auto & f = v[frameid];
            prim->verts.resize(f.size());
            for (auto i = 0; i < prim->verts.size(); i++) {
                prim->verts[i] = f[i];
            }
            prim->tris.resize(f.size() / 3);
            for (auto i = 0; i < prim->tris.size(); i++) {
                prim->tris[i][0] = 3 * i + 0;
                prim->tris[i][1] = 3 * i + 1;
                prim->tris[i][2] = 3 * i + 2;
            }
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ReadCustomVAT, {
    {
        {"frameid"},
    },
    {
        {"prim"},
    },
    {
        {"readpath", "path", ""},
    },
    {"primitive"},
});

} // namespace zeno
