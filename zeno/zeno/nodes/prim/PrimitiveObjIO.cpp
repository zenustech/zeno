#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <Hg/StrUtils.h>
#include <iostream>
#include <fstream>


namespace {


static zeno::vec3i read_index(std::string str) {
    zeno::vec3i face(0, 0, 0);
    auto items = hg::split_str(str, '/');
    for (auto i = 0; i < items.size(); i++) {
        if (items[i].empty()) {
            continue;
        }
        face[i] = std::stoi(items[i]);
    }
    return face - 1;

}
static zeno::vec3f read_vec3f(std::vector<std::string> items) {
    zeno::vec3f vec(0, 0, 0);
    for (auto i = 0; i < items.size(); i++) {
        vec[i] = std::stof(items[i]);
    }
    return vec;
}

static void readobj(
        std::vector<zeno::vec3f> &vertices,
        std::vector<zeno::vec3i> &indices,
        const char *path)
{

    std::vector<zeno::vec3f> normals;
    std::vector<zeno::vec3f> uvs;
    std::vector<zeno::vec3i> loop_indices;

    auto is = std::ifstream(path);
    while (!is.eof())
    {
        std::string line;
        std::getline(is, line);
        line = hg::trim(line);
        if (line.empty()) {
            continue;
        }
        auto items = hg::split_str(line, ' ');
        items.erase(items.begin());

        if (hg::starts_with(line, "v ")) {
            vertices.push_back(read_vec3f(items));
        }
        else if (hg::starts_with(line, "vt")) {
            uvs.push_back(read_vec3f(items));
        }
        else if (hg::starts_with(line, "vn")) {
            normals.push_back(read_vec3f(items));
        }
        else if (hg::starts_with(line, "f")) {
            zeno::vec3i first_index = read_index(items[0]);
            zeno::vec3i last_index = read_index(items[1]);

            for (auto i = 2; i < items.size(); i++) {
                zeno::vec3i index = read_index(items[i]);
                zeno::vec3i face(first_index[0], last_index[0], index[0]);
                indices.push_back(face);

                loop_indices.push_back(first_index);
                loop_indices.push_back(last_index);
                loop_indices.push_back(index);

                last_index = index;
            }
        }
    }
}


struct ReadObjPrimitive : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto &pos = prim->add_attr<zeno::vec3f>("pos");
        readobj(pos, prim->tris, path.c_str());
        prim->resize(pos.size());
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ReadObjPrimitive,
        { /* inputs: */ {
        {"readpath", "path"},
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        }, /* category: */ {
        "primitive",
        }});

struct ImportObjPrimitive : ReadObjPrimitive {
};

ZENDEFNODE(ImportObjPrimitive,
        { /* inputs: */ {
        {"readpath", "path"},
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        }, /* category: */ {
        "primitive",
        }});



static void writeobj(
        std::vector<zeno::vec3f> const &vertices,
        std::vector<zeno::vec3i> const &indices,
        const char *path)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror(path);
        abort();
    }

    for (auto const &vert: vertices) {
        fprintf(fp, "v %f %f %f\n", vert[0], vert[1], vert[2]);
    }

    for (auto const &ind: indices) {
        fprintf(fp, "f %d %d %d\n", ind[0] + 1, ind[1] + 1, ind[2] + 1);
    }
    fclose(fp);
}


struct WriteObjPrimitive : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto &pos = prim->attr<zeno::vec3f>("pos");
        writeobj(pos, prim->tris, path.c_str());
    }
};

ZENDEFNODE(WriteObjPrimitive,
        { /* inputs: */ {
        {"writepath", "path"},
        "prim",
        }, /* outputs: */ {
        }, /* params: */ {
        }, /* category: */ {
        "primitive",
        }});

struct ExportObjPrimitive : WriteObjPrimitive {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto &pos = prim->attr<zeno::vec3f>("pos");
        writeobj(pos, prim->tris, path.c_str());
    }
};

ZENDEFNODE(ExportObjPrimitive,
        { /* inputs: */ {
        {"writepath", "path"},
        "prim",
        }, /* outputs: */ {
        }, /* params: */ {
        }, /* category: */ {
        "primitive",
        }});

}
