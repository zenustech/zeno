#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/log.h>
#include "zeno/para/parallel_reduce.h"

#include <vector>
#include <iostream>
#include <fstream>
using std::vector;
using zeno::vec3f;

namespace zeno {

static int16_t f32_to_i16(float v) {
    return v * std::numeric_limits<int16_t>::max();
}

static float i16_to_f32(int16_t v) {
    return (float)v / std::numeric_limits<int16_t>::max();
}

static void write_vec3f(std::ofstream &file, zeno::vec3f vec) {
    file.write((char*)&vec[0], sizeof(float));
    file.write((char*)&vec[1], sizeof(float));
    file.write((char*)&vec[2], sizeof(float));
}

static zeno::vec3f read_vec3f(std::ifstream &file) {
    zeno::vec3f vec;
    file.read((char*)&vec[0], sizeof (float));
    file.read((char*)&vec[1], sizeof (float));
    file.read((char*)&vec[2], sizeof (float));
    return vec;
}

static void write_normalized_vec3f(std::ofstream &file, vec3f vec, vec3f _min, vec3f _max) {
    vec = (vec - _min) / (_max - _min);

    int16_t _0 = f32_to_i16(vec[0]);
    int16_t _1 = f32_to_i16(vec[1]);
    int16_t _2 = f32_to_i16(vec[2]);

    file.write((char*)&_0, sizeof(int16_t));
    file.write((char*)&_1, sizeof(int16_t));
    file.write((char*)&_2, sizeof(int16_t));
}

static zeno::vec3f read_normalized_vec3f(std::ifstream &file, vec3f _min, vec3f _max) {
    int16_t _0;
    int16_t _1;
    int16_t _2;
    file.read((char*)&_0, sizeof (int16_t));
    file.read((char*)&_1, sizeof (int16_t));
    file.read((char*)&_2, sizeof (int16_t));

    zeno::vec3f vec = {
            i16_to_f32(_0),
            i16_to_f32(_1),
            i16_to_f32(_2)
    };
    vec = vec * (_max - _min) + _min;
    return vec;
}

static void write_vat(vector<vector<vec3f>> &v, const std::string &path) {
    std::ofstream file(path, std::ios::out | std::ios::binary);
    vector<vec3f> temp_bboxs;
    for (const auto& i: v) {
        auto bbox = parallel_reduce_minmax(i.begin(), i.end());
        temp_bboxs.push_back(bbox.first);
        temp_bboxs.push_back(bbox.second);
    }
    auto bbox = parallel_reduce_minmax(temp_bboxs.begin(), temp_bboxs.end());
    zeno::log_info("{} {}", bbox.first, bbox.second);
    write_vec3f(file, bbox.first);
    write_vec3f(file, bbox.second);

    int height = v.size();
    file.write((char*)&height, sizeof(int));
    int maxWidth = 0;
    for (auto i = 0; i < height; i++) {
        int width = v[i].size();
        maxWidth = std::max(maxWidth, width);
    }
    file.write((char*)&maxWidth, sizeof(int));

    for (auto i = 0; i < height; i++) {
        int width = v[i].size();
        file.write((char*)&width, sizeof(int));
    }

    for (auto i = 0; i < height; i++) {
        int width = v[i].size();
        for (auto j = 0; j < width; j++) {
            write_normalized_vec3f(file, v[i][j], bbox.first, bbox.second);
        }
        zeno::log_info("VAT: write frame {} done ({} face vec)!", i, width);
    }
}

static vector<vector<vec3f>> read_vat(const std::string &path) {
    vector<vector<vec3f>> v;
    std::ifstream file(path, std::ios::in | std::ios::binary);
    auto _min = read_vec3f(file);
    auto _max = read_vec3f(file);

    int height = 0;
    file.read((char*)&height, sizeof (int));
    v.resize(height);
    zeno::log_info("VAT: frames {}", height);
    int maxWidth = 0;
    file.read((char *) &maxWidth, sizeof(int));

    std::vector<int> widths = {};
    for (auto i = 0; i < height; i++) {
        int width = 0;
        file.read((char *) &width, sizeof(int));
        widths.push_back(width);
    }

    for (auto i = 0; i < height; i++) {
        int width = widths[i];

        v[i].resize(width);
        for (auto j = 0; j < width; j++) {
            v[i][j] = read_normalized_vec3f(file, _min, _max);
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
