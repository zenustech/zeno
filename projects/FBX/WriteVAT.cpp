#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/log.h>
#include "zeno/para/parallel_reduce.h"

#include <vector>
#include <iostream>
#include <fstream>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using std::vector;
using zeno::vec3f;

namespace zeno {

static uint16_t f32_to_u16(float v) {
    return v * std::numeric_limits<uint16_t>::max();
}

static float u16_to_f32(uint16_t v) {
    return (float)v / std::numeric_limits<uint16_t>::max();
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

    uint16_t _0 = f32_to_u16(vec[0]);
    uint16_t _1 = f32_to_u16(vec[1]);
    uint16_t _2 = f32_to_u16(vec[2]);

    file.write((char*)&_0, sizeof(uint16_t));
    file.write((char*)&_1, sizeof(uint16_t));
    file.write((char*)&_2, sizeof(uint16_t));
}

static int align_to(int count, int align) {
    int remainder = count % align;
    if (remainder == 0) {
        return count;
    }
    else {
        return count + (align - remainder);
    }
}

static zeno::vec3f read_normalized_vec3f(std::ifstream &file, vec3f _min, vec3f _max) {
    uint16_t _0;
    uint16_t _1;
    uint16_t _2;
    file.read((char*)&_0, sizeof (uint16_t));
    file.read((char*)&_1, sizeof (uint16_t));
    file.read((char*)&_2, sizeof (uint16_t));

    zeno::vec3f vec = {
            u16_to_f32(_0),
            u16_to_f32(_1),
            u16_to_f32(_2)
    };
    vec = vec * (_max - _min) + _min;
    return vec;
}

static void write_vat(vector<vector<vec3f>> &v, const std::string &path) {
    std::ofstream file(path, std::ios::out | std::ios::binary);
    file << 'Z';
    file << 'E';
    file << 'N';
    file << 'O';
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

    int frames = v.size();
    file.write((char*)&frames, sizeof(int));
    int maxWidth = 0;
    for (auto i = 0; i < frames; i++) {
        int width = v[i].size();
        maxWidth = std::max(maxWidth, width);
    }
    file.write((char*)&maxWidth, sizeof(int));
    int maxWidthAlign = align_to(maxWidth, 8192);
    int height = frames * (maxWidthAlign / 8192);
    file.write((char*)&height, sizeof(int));

    for (auto i = 0; i < frames; i++) {
        int width = v[i].size();
        file.write((char*)&width, sizeof(int));
    }

    for (auto i = 0; i < frames; i++) {
        int width = v[i].size();
        v[i].resize(maxWidthAlign);
        for (auto j = 0; j < maxWidthAlign; j++) {
            write_normalized_vec3f(file, v[i][j], bbox.first, bbox.second);
        }
        zeno::log_info("VAT: write frame {} done ({} face vec)!", i, width);
    }
}


static void write_vat_nrm(vector<vector<vec3f>> &v, const std::string &path) {
    int frames = v.size();
    int maxWidth = 0;
    for (auto i = 0; i < frames; i++) {
        int width = v[i].size();
        maxWidth = std::max(maxWidth, width);
    }
    int maxWidthAlign = align_to(maxWidth, 8192);
    int height = frames * (maxWidthAlign / 8192);

    std::vector<uint8_t> nrms;

    for (auto i = 0; i < frames; i++) {
        int width = v[i].size();
        v[i].resize(maxWidthAlign);
        for (auto j = 0; j < maxWidthAlign; j++) {
            nrms.push_back(uint8_t((v[i][j][0] * 0.5 + 0.5) * 255.99));
            nrms.push_back(uint8_t((v[i][j][1] * 0.5 + 0.5) * 255.99));
            nrms.push_back(uint8_t((v[i][j][2] * 0.5 + 0.5) * 255.99));
        }
        zeno::log_info("VAT: write frame {} done ({} face vec)!", i, width);
    }

    stbi_flip_vertically_on_write(false);
    stbi_write_png(path.c_str(), 8192, height, 3, nrms.data(), 0);
}

// VAT format
// bbox_min : (float, float, float)
// bbox_max : (float, float, float)
// frame_count : int32_t
// max_vertex_count : int32_t
// image_height : int32_t (image_width always 8192, not save)
// vertex_count_per_frame: frame_count * int32_t
// data: (w * h) * (uint16_t, uint16_t, uint16_t)

struct VATTexture {
    int frame_count;
    int max_triangles_per_frame;
    int height; // image width always is 8192
    std::vector<int> triangle_per_frame;
    std::vector<zeno::vec3f> data;
};

static VATTexture read_vat_texture(const std::string &path) {
    VATTexture vat = {};
    std::ifstream file(path, std::ios::in | std::ios::binary);
    char x;
    file >> x >> x >> x >> x;
    auto _min = read_vec3f(file);
    auto _max = read_vec3f(file);

    file.read((char*)&vat.frame_count, sizeof (int));
    zeno::log_info("VAT: frames {}", vat.frame_count);
    file.read((char *) &vat.max_triangles_per_frame, sizeof(int));
    file.read((char *) &vat.height, sizeof(int));
    zeno::log_info("VAT: height {}", vat.height);

    vat.triangle_per_frame.resize(vat.frame_count);
    for (auto i = 0; i < vat.frame_count; i++) {
        file.read((char *) &vat.triangle_per_frame[i], sizeof(int));
    }

    vat.data.resize(int64_t (vat.height) * 8192);
    for (int64_t i = 0; i < vat.height * 8192; i++) {
        vat.data[i] = read_normalized_vec3f(file, _min, _max);
    }
    return vat;
}
static vector<vector<vec3f>> read_vat(const std::string &path) {
    vector<vector<vec3f>> v;
    std::ifstream file(path, std::ios::in | std::ios::binary);
    auto _min = read_vec3f(file);
    auto _max = read_vec3f(file);

    int frames = 0;
    file.read((char*)&frames, sizeof (int));
    v.resize(frames);
    zeno::log_info("VAT: frames {}", frames);
    int maxWidth = 0;
    file.read((char *) &maxWidth, sizeof(int));
    int maxWidthAlign = align_to(maxWidth, 8192);
    int height = 0;
    file.read((char *) &height, sizeof(int));
    zeno::log_info("VAT: height {}", height);

    std::vector<int> widths = {};
    for (auto i = 0; i < frames; i++) {
        int width = 0;
        file.read((char *) &width, sizeof(int));
        widths.push_back(width);
    }

    for (auto i = 0; i < frames; i++) {
        int width = widths[i];

        v[i].resize(maxWidthAlign);
        for (auto j = 0; j < maxWidthAlign; j++) {
            v[i][j] = read_normalized_vec3f(file, _min, _max);
        }
        zeno::log_info("VAT: read frame {} done ({} face vec)!", i, width);
    }
    return v;
}

struct WriteCustomVAT : INode {
    std::vector<std::shared_ptr<PrimitiveObject>> prims;
    virtual void apply() override {
        std::string path = get_param<std::string>("path");
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
        auto raw_prim = get_input<PrimitiveObject>("prim");
        auto prim = std::dynamic_pointer_cast<PrimitiveObject>(raw_prim->clone());
        if (frameStart <= frameid && frameid <= frameEnd) {
            prims[frameid - frameStart] = prim;
        }
        if (frameid == frameEnd) {
            // face overflow check
            {
                int max_face_per_vat = 8192 / frameCount * 8192 / 3;
                int max_face_in_prims = 0;
                for (const auto & prim : prims) {
                    max_face_in_prims = std::max(max_face_in_prims, (int)prim->tris.size());
                }

                if (max_face_in_prims > max_face_per_vat) {
                    zeno::log_error("max_face_in_prims: {} > max_face_per_vat: {}", max_face_in_prims, max_face_per_vat);
                    set_output("prim", raw_prim);
                    return;
                }
            }
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

            vector<vector<vec3f>> nrms;
            nrms.resize(prims.size());
            for (auto i = 0; i < prims.size(); i++) {
                auto prim = prims[i];
                auto& nrm_ref = prim->verts.attr<vec3f>("nrm");
                nrms[i].resize(prim->tris.size() * 3);
                for (auto j = 0; j < prim->tris.size(); j++) {
                    const auto & tri = prim->tris[j];
                    nrms[i][j * 3 + 0] = nrm_ref[tri[0]];
                    nrms[i][j * 3 + 1] = nrm_ref[tri[1]];
                    nrms[i][j * 3 + 2] = nrm_ref[tri[2]];
                }
            }
            write_vat_nrm(nrms, path + ".png");

            {
                std::string obj_path = path + ".obj";
                auto prim = std::make_shared<zeno::PrimitiveObject>();
                {
                    auto & f = v.front();
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
                {
                    int total_frame = v.size();
                    int one_prim_line = align_to(v[0].size(), 8192) / 8192;
                    int total_lines = total_frame * one_prim_line;
                    zeno::log_info("total_frame: {}", total_frame);
                    zeno::log_info("one_prim_line: {}", one_prim_line);
                    zeno::log_info("total_lines: {}", total_lines);
                    std::ofstream fout(obj_path);
                    for (auto const &[x, y, z]: prim->verts) {
                        fout << zeno::format("v {} {} {}\n", x, y, z);
                    }
                    for (auto i = 0; i < prim->verts.size(); i++) {
                        auto index = (float(i) + 0.5f) / 8192.f;
                        auto u = index - zeno::floor(index);
                        auto v = (zeno::floor(index) + 0.5f) / float(total_lines);

                        fout << zeno::format("vt {} {}\n", u, v);
                    }

                    for (auto const &[x, y, z]: prim->tris) {
                        fout << zeno::format("f {}/{} {}/{} {}/{}\n", x + 1, x + 1, y + 1, y + 1, z + 1, z + 1);
                    }
                    fout << std::flush;
                }
            }
            zeno::log_info("VAT: save success!");
        }
        set_output("prim", raw_prim);
    }
};

ZENDEFNODE(WriteCustomVAT, {
    {
        {"prim"},
        {"frameid"},
    },
    {
        {"prim"},
    },
    {
        {"writepath", "path", ""},
        {"int", "frameStart", "0"},
        {"int", "frameEnd", "100"},
    },
    {"VAT"},
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
    {"VAT"},
});

struct ReadVATFile : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        auto vat = read_vat_texture(path);
        auto img = std::make_shared<PrimitiveObject>();
        img->verts.resize(vat.height * 8192);
        for (int64_t i = 0; i < vat.height * 8192; i++) {
            img->verts[i] = vat.data[i];
        }

        img->userData().set2("isImage", 1);
        img->userData().set2("w", 8192);
        img->userData().set2("h", vat.height);
        set_output("image", img);
    }
};

ZENDEFNODE(ReadVATFile, {
    {
        {"readpath", "path"},
    },
    {
        {"PrimitiveObject", "image"},
    },
    {},
    {"VAT"},
});

} // namespace zeno
