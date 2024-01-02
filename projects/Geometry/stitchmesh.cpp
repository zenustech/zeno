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
        auto prim = get_input<PrimitiveObject>("prim");
        auto input_dir = get_input2<std::string>("input_dir");
        input_dir.erase(input_dir.find_last_not_of(" ")+1);
        auto output_dir = get_input2<std::string>("output_dir");
        float scale = get_input2<float>("scale");

        int argc = 1;
        char* argv[10];
        argv[0] = (char*)malloc(sizeof("./stitch-meshing\0"));
        strcpy(argv[0], "./stitch-meshing\0");
        if (input_dir.size() > 0) {
            argv[argc] = (char*)malloc(sizeof("-i\0"));
            strcpy(argv[argc], "-i\0");
            argv[argc+1] = (char*)malloc((input_dir.size()+1)*sizeof(char));
            for (int i = 0; i < input_dir.size(); ++i)
                argv[argc+1][i] = input_dir[i];
            argv[argc+1][input_dir.size()] = '\0';
            argc += 2;
        }
        if (output_dir.size() > 0) {
            argv[argc] = (char*)malloc(sizeof("-o\0"));
            strcpy(argv[argc], "-o\0");
            argv[argc+1] = (char*)malloc((output_dir.size()+1)*sizeof(char));
            for (int i = 0; i < output_dir.size(); ++i)
                argv[argc+1][i] = output_dir[i];
            argv[argc+1][output_dir.size()] = '\0';
            argc += 2;
        }
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

        std::vector<std::vector<int>> faces(prim->tris->size(), std::vector<int>{});
        std::vector<std::vector<float>> verts(prim->verts->size(), std::vector<float>{});
        for (int i = 0; i < prim->tris->size(); ++i)
            for (int j = 0; j < 3; ++j)
                faces[i].push_back(prim->tris[i][j]);
        for (int i = 0; i < prim->verts->size(); ++i)
            for (int j = 0; j < 3; ++j)
                verts[i].push_back(prim->verts[i][j]);

        runStitchMeshing(faces, verts, argc, argv);

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(StitchMeshing)
({
    {{"prim"},
    {"readpath", "input_dir", " "},
    {"string", "output_dir"},
    {"float", "scale", "0"},},
    {("prim")},
    {},
    {"primitive"},
});


} // namespace zeno
