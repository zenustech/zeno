#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/logger.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <fstream>


namespace {

void writeobj(
        std::vector<zeno::vec3f> const &vertices,
        std::vector<zeno::vec3f> const &uvs,
        std::vector<zeno::vec3i> const &indices,
        const char *path)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror(path);
        abort();
    }
    
    fprintf(fp, "mtllib male02.mtl\n");
    
    for (auto const &vert: vertices) {
        fprintf(fp, "v %f %f %f\n", vert[0], vert[1], vert[2]);
    }

    for (auto const &uv: uvs) {
        fprintf(fp, "vt %f %f\n", uv[0], uv[1]);
    }

    for (auto const &ind: indices) {
        fprintf(fp, "f %d/%d %d/%d %d/%d\n",
            ind[0] + 1, ind[0] + 1,
            ind[1] + 1, ind[1] + 2,
            ind[2] + 1, ind[2] + 3
        );
    }
    fclose(fp);
}


struct ExportUVObjPrimitive :  zeno::INode  {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto &pos = prim->attr<zeno::vec3f>("pos");
        auto &uvs = prim->attr<zeno::vec3f>("uv");
        writeobj(pos, uvs, prim->tris, path.c_str());
    }
};

ZENDEFNODE(ExportUVObjPrimitive,
{   /* inputs: */ 
    {
        {"writepath", "path"},
        "prim",
    }, 
    /* outputs: */ 
    {}, 
    /* params: */ 
    {}, 
    /* category: */ 
    {"math",}
});

}
