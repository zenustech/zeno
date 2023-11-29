#include <cctype>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>
#include "instant_meshes/api.h"

namespace zeno {

struct QuadMesh : INode {
    virtual void apply() override {
        float crease = get_input2<float>("crease");
        bool deterministic = get_input2<bool>("deterministic");
        int smooth = get_input2<int>("smooth_iter");
        bool dominant = get_input2<bool>("quad_dominant");
        bool intrinsic = get_input2<bool>("intrinsic");
        bool boundary = get_input2<bool>("boundary");
        float scale = get_input2<float>("scale");
        int vert_num = get_input2<int>("vert_num");
        int face_num = get_input2<int>("face_num");
        int knn = get_input2<int>("knn");
        auto input_dir = get_input2<std::string>("input_mesh_dir");
        auto output_dir = get_input2<std::string>("output_mesh_dir");

        int argc = 1;
        char* argv[10];
        argv[0] = (char*)malloc(sizeof("./InstantMeshes\0"));
        strcpy(argv[0], "./InstantMeshes\0");
        if (output_dir.size() > 0) {
            argv[argc] = (char*)malloc(sizeof("--output\0"));
            strcpy(argv[argc], "--output\0");
            argv[argc+1] = (char*)malloc((output_dir.size()+1)*sizeof(char));
            for (int i = 0; i < output_dir.size(); ++i)
                argv[argc+1][i] = output_dir[i];
            argv[argc+1][output_dir.size()] = '\0';
            argc += 2;
        }
        if (crease > 0) {
            argv[argc] = (char*)malloc(sizeof("--crease\0"));
            strcpy(argv[argc], "--crease\0");
            std::string crease_str = to_string(crease);
            argv[argc+1] = (char*)malloc((crease_str.size()+1)*sizeof(char));
            for (int i = 0; i < crease_str.size(); ++i)
                argv[argc+1][i] = crease_str[i];
            argv[argc+1][crease_str.size()] = '\0';
            argc += 2;
        }
        if (deterministic == 1) {
            argv[argc] = (char*)malloc(sizeof("--deterministic\0"));
            strcpy(argv[argc], "--deterministic\0");
            argc += 1;
        }
        if (smooth > 0) {
            argv[argc] = (char*)malloc(sizeof("--smooth\0"));
            strcpy(argv[argc], "--smooth\0");
            std::string smooth_str = to_string(smooth);
            argv[argc+1] = (char*)malloc((smooth_str.size()+1)*sizeof(char));
            for (int i = 0; i < smooth_str.size(); ++i)
                argv[argc+1][i] = smooth_str[i];
            argv[argc+1][smooth_str.size()] = '\0';
            argc += 2;
        }
        if (dominant == 1) {
            argv[argc] = (char*)malloc(sizeof("--dominant\0"));
            strcpy(argv[argc], "--dominant\0");
            argc += 1;
        }
        if (intrinsic == 1) {
            argv[argc] = (char*)malloc(sizeof("--intrinsic\0"));
            strcpy(argv[argc], "--intrinsic\0");
            argc += 1;
        }
        if (boundary == 1) {
            argv[argc] = (char*)malloc(sizeof("--boundaries\0"));
            strcpy(argv[argc], "--boundaries\0");
            argc += 1;
        }
        if (scale > 0) {
            argv[argc] = (char*)malloc(sizeof("--scale\0"));
            strcpy(argv[argc], "--scale\0");
            std::string scale_str = to_string(scale);
            argv[argc+1] = (char*)malloc((scale_str.size()+1)*sizeof(char));
            for (int i = 0; i < scale_str.size(); ++i)
                argv[argc+1][i] = scale_str[i];
            argv[argc+1][scale_str.size()] = '\0';
            argc += 2;
        }
        if (vert_num > 0) {
            argv[argc] = (char*)malloc(sizeof("--vertices\0"));
            strcpy(argv[argc], "--vertices\0");
            std::string vert_num_str = to_string(vert_num);
            argv[argc+1] = (char*)malloc((vert_num_str.size()+1)*sizeof(char));
            for (int i = 0; i < vert_num_str.size(); ++i)
                argv[argc+1][i] = vert_num_str[i];
            argv[argc+1][vert_num_str.size()] = '\0';
            argc += 2;
        }
        if (face_num > 0) {
            argv[argc] = (char*)malloc(sizeof("--faces\0"));
            strcpy(argv[argc], "--faces\0");
            std::string face_num_str = to_string(face_num);
            argv[argc+1] = (char*)malloc((face_num_str.size()+1)*sizeof(char));
            for (int i = 0; i < face_num_str.size(); ++i)
                argv[argc+1][i] = face_num_str[i];
            argv[argc+1][face_num_str.size()] = '\0';
            argc += 2;
        }
        if (knn > 0) {
            argv[argc] = (char*)malloc(sizeof("--knn\0"));
            strcpy(argv[argc], "--knn\0");
            std::string knn_str = to_string(knn);
            argv[argc+1] = (char*)malloc((knn_str.size()+1)*sizeof(char));
            for (int i = 0; i < knn_str.size(); ++i)
                argv[argc+1][i] = knn_str[i];
            argv[argc+1][knn_str.size()] = '\0';
            argc += 2;
        }
        argv[argc] = (char*)malloc((input_dir.size()+1)*sizeof(char));
        for (int i = 0; i < input_dir.size(); ++i)
            argv[argc][i] = input_dir[i];
        argv[argc][input_dir.size()] = '\0';
        ++argc;
        runInstantMeshes(argc, argv);
    }
};

ZENO_DEFNODE(QuadMesh)
({
    {{"readpath", "input_mesh_dir"},
    {"readpath", "output_mesh_dir"},
    {"bool", "deterministic", "0"},
    {"float", "crease", "0"},
    {"int", "smooth_iter", "0"},
    {"bool", "quad_dominant", "0"},
    {"bool", "intrinsic", "0"},
    {"bool", "boundary", "0"},
    {"float", "scale", "0"},
    {"int", "vert_num", "0"},
    {"int", "face_num", "0"},
    {"int", "knn", "0"}},
    {},
    {},
    {"primitive"},
});


} // namespace zeno
