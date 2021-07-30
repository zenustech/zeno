#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>


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
}


struct ImportObjPrimitive : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path");
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto &pos = prim->add_attr<zeno::vec3f>("pos");
        readobj(pos, prim->tris, path->get().c_str());
        prim->resize(pos.size());
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ImportObjPrimitive,
        { /* inputs: */ {
        "path",
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        }, /* category: */ {
        "primitive",
        }});
