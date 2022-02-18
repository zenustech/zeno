#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/prim_ops.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <fstream>


namespace zeno {


static zeno::vec3i read_index(std::string str) {
    zeno::vec3i face(0, 0, 0);
    auto items = zeno::split_str(str, '/');
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
    int i = 0;
    for (auto item: items) {
        if (item.size() != 0) {
            vec[i] = std::stof(item);
            i += 1;
        }
    }
    return vec;
}

void read_obj_file(
        std::vector<zeno::vec3f> &vertices,
        //std::vector<zeno::vec3f> &uvs,
        //std::vector<zeno::vec3f> &normals,
        std::vector<zeno::vec3i> &indices,
        //std::vector<zeno::vec3i> &uv_indices,
        //std::vector<zeno::vec3i> &normal_indices,
        const char *path)
{


    auto is = std::ifstream(path);
    while (!is.eof()) {
        std::string line;
        std::getline(is, line);
        line = zeno::trim_string(line);
        if (line.empty()) {
            continue;
        }
        auto items = zeno::split_str(line, ' ');
        items.erase(items.begin());

        if (zeno::starts_with(line, "v ")) {
            vertices.push_back(read_vec3f(items));

        /*
        } else if (zeno::starts_with(line, "vt ")) {
            uvs.push_back(read_vec3f(items));

        } else if (zeno::starts_with(line, "vn ")) {
            normals.push_back(read_vec3f(items));
        */

        } else if (zeno::starts_with(line, "f ")) {
            zeno::vec3i first_index = read_index(items[0]);
            zeno::vec3i last_index = read_index(items[1]);

            for (auto i = 2; i < items.size(); i++) {
                zeno::vec3i index = read_index(items[i]);
                zeno::vec3i face(first_index[0], last_index[0], index[0]);
                //zeno::vec3i face_uv(first_index[1], last_index[1], index[1]);
                //zeno::vec3i face_normal(first_index[2], last_index[2], index[2]);
                indices.push_back(face);
                //uv_indices.push_back(face_uv);
                //normal_indices.push_back(face_normal);
                last_index = index;
            }
        }
    }
}


struct ReadObjPrimitive : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto &pos = prim->verts;
        //auto &uv = prim->verts.add_attr<zeno::vec3f>("uv");
        //auto &norm = prim->verts.add_attr<zeno::vec3f>("nrm");
        auto &tris = prim->tris;
        //auto &triuv = prim->tris.add_attr<zeno::vec3i>("uv");
        //auto &trinorm = prim->tris.add_attr<zeno::vec3i>("nrm");
        read_obj_file(pos, /*uv, norm,*/ tris, /*triuv, trinorm,*/ path.c_str());
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
