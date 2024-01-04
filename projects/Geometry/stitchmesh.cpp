#include <cctype>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>
#include <string>
#include "./stitchMeshing/api.h"

namespace zeno {

struct StitchMeshing : INode {
    virtual void apply() override {
        // TODO(@seeegull):
        // 1. change quad mesh extraction part to instant meshes
        // 2. render arrow texture in zeno

        auto prim = get_input<PrimitiveObject>("prim");
        float scale = get_input2<float>("scale");
        bool flip = get_input2<bool>("flip_course_wale");

        int argc = 1;
        char* argv[10];
        argv[0] = (char*)malloc(sizeof("./stitch-meshing\0"));
        strcpy(argv[0], "./stitch-meshing\0");
        if (scale > 0) {
            argv[argc] = (char*)malloc(sizeof("-s\0"));
            strcpy(argv[argc], "-s\0");
            std::string scale_str = to_string(scale);
            argv[argc+1] = (char*)malloc((scale_str.size()+1)*sizeof(char));
            for (int i = 0; i < scale_str.size(); ++i)
                argv[argc+1][i] = scale_str[i];
            argv[argc+1][scale_str.size()] = '\0';
            argc += 2;
        }
        if (flip) {
            argv[argc] = (char*)malloc(sizeof("--flip\0"));
            strcpy(argv[argc], "--flip\0");
            argc += 1;
        }

        std::vector<std::vector<int>> faces(prim->tris->size(), std::vector<int>{});
        std::vector<std::vector<float>> verts(prim->verts->size(), std::vector<float>{});
        auto &pos = prim->verts;
        for (int i = 0; i < prim->tris->size(); ++i)
            for (int j = 0; j < 3; ++j)
                faces[i].push_back(prim->tris[i][j]);
        for (int i = 0; i < prim->verts->size(); ++i)
            for (int j = 0; j < 3; ++j)
                verts[i].push_back(pos[i][j]);

        runStitchMeshing(faces, verts, argc, argv);

        pos.resize(verts.size());
        for (int i = 0; i < pos.size(); ++i)
            for (int j = 0; j < 3; ++j)
                pos[i][j] = verts[i][j];
        prim->verts.foreach_attr<zeno::AttrAcceptAll>([&] (auto const &key, auto &arr) {
            arr.resize(verts.size());
        });
        prim->lines.clear();
        prim->tris.clear();
        prim->loops.clear();
        prim->polys.clear();
        for (int i = 0; i < faces.size(); ++i) {
            int beg = prim->loops.size();
            for (auto j : faces[i])
                prim->loops.push_back(j);
            prim->polys.emplace_back(beg, faces[i].size());
        }
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(StitchMeshing)
({
    {{"prim"},
    {"float", "scale", "0"},
    {"bool", "flip_course_wale", "0"}},
    {("prim")},
    {},
    {"primitive"},
});


} // namespace zeno
