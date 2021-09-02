#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>

namespace {

static void readobj(
        std::vector<zeno::vec3f> &vertices,
        std::vector<zeno::vec3i> &indices,
        const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror(path);
        abort();
    }

    //printf("o %s\n", path);

    char hdr[128];
    while (EOF != fscanf(fp, "%s", hdr)) 
    {
        if (!strcmp(hdr, "v")) {
            zeno::vec3f vertex;
            fscanf(fp, "%f %f %f\n", &vertex[0], &vertex[1], &vertex[2]);
            //printf("v %f %f %f\n", vertex[0], vertex[1], vertex[2]);
            vertices.push_back(vertex);

        } else if (!strcmp(hdr, "f")) {
            zeno::vec3i last_index, first_index, index;

            fscanf(fp, "%d/%d/%d", &index[0], &index[1], &index[2]);
            first_index = index;

            fscanf(fp, "%d/%d/%d", &index[0], &index[1], &index[2]);
            last_index = index;

            while (fscanf(fp, "%d/%d/%d", &index[0], &index[1], &index[2]) > 0) {
                zeno::vec3i face(first_index[0], last_index[0], index[0]);
                //printf("f %d %d %d\n", face[0], face[1], face[2]);
                indices.push_back(face - 1);
                last_index = index;
            }
        }
    }
    fclose(fp);
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
