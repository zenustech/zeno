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
#include "./fTetWild/api.h"

namespace zeno {

struct FTetWild : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto input_dir = get_input2<std::string>("input_dir");
        input_dir.erase(input_dir.find_last_not_of(" ")+1);
        auto output_dir = get_input2<std::string>("output_dir");
        auto tag = get_input2<std::string>("tag");
        tag.erase(tag.find_last_not_of(" ")+1);
        auto operation = get_input2<std::string>("operation");
        float edge_length = get_input2<float>("edge_length");
        float epsilon = get_input2<float>("epsilon");
        float stop_energy = get_input2<float>("stop_energy");
        bool skip_simplify = get_input2<bool>("skip_simplify");
        bool no_binary = get_input2<bool>("no_binary");
        bool no_color = get_input2<bool>("no_color");
        bool smooth = get_input2<bool>("smooth_open_boundary");
        bool export_raw = get_input2<bool>("export_raw");
        bool manifold = get_input2<bool>("manifold_surface");
        bool coarsen = get_input2<bool>("coarsen");
        auto csg = get_input2<std::string>("csg");
        csg.erase(csg.find_last_not_of(" ")+1);
        bool disable_filter = get_input2<bool>("disable_filtering");
        bool floodfill = get_input2<bool>("use_floodfill");
        bool general_wn = get_input2<bool>("use_general_wn");
        bool input_wn = get_input2<bool>("use_input_for_wn");
        auto bg_mesh = get_input2<std::string>("bg_mesh");
        bg_mesh.erase(bg_mesh.find_last_not_of(" ")+1);

        int argc = 1;
        char* argv[40];
        argv[0] = (char*)malloc(sizeof("./FloatTetwild_bin\0"));
        strcpy(argv[0], "./FloatTetwild_bin\0");
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
        if (tag.size() > 0) {
            argv[argc] = (char*)malloc(sizeof("--tag\0"));
            strcpy(argv[argc], "--tag\0");
            argv[argc+1] = (char*)malloc((tag.size()+1)*sizeof(char));
            for (int i = 0; i < tag.size(); ++i)
                argv[argc+1][i] = tag[i];
            argv[argc+1][tag.size()] = '\0';
            argc += 2;
        }
        if (operation == "union") {
            argv[argc] = (char*)malloc(sizeof("--op\0"));
            strcpy(argv[argc], "--op\0");
            argv[argc+1] = (char*)malloc(sizeof("0\0"));
            strcpy(argv[argc+1], "0\0");
            argc += 2;
        } else if (operation == "intersection") {
            argv[argc] = (char*)malloc(sizeof("--op\0"));
            strcpy(argv[argc], "--op\0");
            argv[argc+1] = (char*)malloc(sizeof("1\0"));
            strcpy(argv[argc+1], "1\0");
            argc += 2;
        } else if (operation == "difference") {
            argv[argc] = (char*)malloc(sizeof("--op\0"));
            strcpy(argv[argc], "--op\0");
            argv[argc+1] = (char*)malloc(sizeof("2\0"));
            strcpy(argv[argc+1], "2\0");
            argc += 2;
        }
        if (edge_length > 0) {
            argv[argc] = (char*)malloc(sizeof("-l\0"));
            strcpy(argv[argc], "-l\0");
            std::string edge_length_str = to_string(edge_length);
            argv[argc+1] = (char*)malloc((edge_length_str.size()+1)*sizeof(char));
            for (int i = 0; i < edge_length_str.size(); ++i)
                argv[argc+1][i] = edge_length_str[i];
            argv[argc+1][edge_length_str.size()] = '\0';
            argc += 2;
        }
        if (epsilon > 0) {
            argv[argc] = (char*)malloc(sizeof("-e\0"));
            strcpy(argv[argc], "-e\0");
            std::string epsilon_str = to_string(epsilon);
            argv[argc+1] = (char*)malloc((epsilon_str.size()+1)*sizeof(char));
            for (int i = 0; i < epsilon_str.size(); ++i)
                argv[argc+1][i] = epsilon_str[i];
            argv[argc+1][epsilon_str.size()] = '\0';
            argc += 2;
        }
        if (stop_energy > 0) {
            argv[argc] = (char*)malloc(sizeof("--stop-energy\0"));
            strcpy(argv[argc], "--stop-energy\0");
            std::string stop_energy_str = to_string(stop_energy);
            argv[argc+1] = (char*)malloc((stop_energy_str.size()+1)*sizeof(char));
            for (int i = 0; i < stop_energy_str.size(); ++i)
                argv[argc+1][i] = stop_energy_str[i];
            argv[argc+1][stop_energy_str.size()] = '\0';
            argc += 2;
        }
        if (skip_simplify) {
            argv[argc] = (char*)malloc(sizeof("--skip-simplify\0"));
            strcpy(argv[argc], "--skip-simplify\0");
            argc += 1;
        }
        if (no_binary) {
            argv[argc] = (char*)malloc(sizeof("--no-binary\0"));
            strcpy(argv[argc], "--no-binary\0");
            argc += 1;
        }
        if (no_color) {
            argv[argc] = (char*)malloc(sizeof("--no-color\0"));
            strcpy(argv[argc], "--no-color\0");
            argc += 1;
        }
        if (smooth) {
            argv[argc] = (char*)malloc(sizeof("--smooth-open-boundary\0"));
            strcpy(argv[argc], "--smooth-open-boundary\0");
            argc += 1;
        }
        if (export_raw) {
            argv[argc] = (char*)malloc(sizeof("--export-raw\0"));
            strcpy(argv[argc], "--export-raw\0");
            argc += 1;
        }
        if (manifold) {
            argv[argc] = (char*)malloc(sizeof("--manifold-surface\0"));
            strcpy(argv[argc], "--manifold-surface\0");
            argc += 1;
        }
        if (coarsen) {
            argv[argc] = (char*)malloc(sizeof("--coarsen\0"));
            strcpy(argv[argc], "--coarsen\0");
            argc += 1;
        }
        if (csg.size() > 0) {
            argv[argc] = (char*)malloc(sizeof("--csg\0"));
            strcpy(argv[argc], "--csg\0");
            argv[argc+1] = (char*)malloc((csg.size()+1)*sizeof(char));
            for (int i = 0; i < csg.size(); ++i)
                argv[argc+1][i] = csg[i];
            argv[argc+1][csg.size()] = '\0';
            argc += 2;
        }
        if (disable_filter) {
            argv[argc] = (char*)malloc(sizeof("--disable-filtering\0"));
            strcpy(argv[argc], "--disable-filtering\0");
            argc += 1;
        }
        if (floodfill) {
            argv[argc] = (char*)malloc(sizeof("--use-floodfill\0"));
            strcpy(argv[argc], "--use-floodfill\0");
            argc += 1;
        }
        if (general_wn) {
            argv[argc] = (char*)malloc(sizeof("--use-general-wn\0"));
            strcpy(argv[argc], "--use-general-wn\0");
            argc += 1;
        }
        if (input_wn) {
            argv[argc] = (char*)malloc(sizeof("--use-input-for-wn\0"));
            strcpy(argv[argc], "--use-input-for-wn\0");
            argc += 1;
        }
        if (bg_mesh.size() > 0) {
            argv[argc] = (char*)malloc(sizeof("--bg-mesh\0"));
            strcpy(argv[argc], "--bg-mesh\0");
            argv[argc+1] = (char*)malloc((bg_mesh.size()+1)*sizeof(char));
            for (int i = 0; i < bg_mesh.size(); ++i)
                argv[argc+1][i] = bg_mesh[i];
            argv[argc+1][bg_mesh.size()] = '\0';
            argc += 2;
        }
        argv[argc] = (char*)malloc(sizeof("--level\0"));
        strcpy(argv[argc], "--level\0");
        argv[argc+1] = (char*)malloc(sizeof("3\0"));
        strcpy(argv[argc+1], "3\0");
        argc += 2;

        std::vector<std::vector<int>> faces(prim->tris->size(), std::vector<int>{});
        std::vector<std::vector<float>> verts(prim->verts->size(), std::vector<float>{});
        for (int i = 0; i < prim->tris->size(); ++i)
            for (int j = 0; j < 3; ++j)
                faces[i].push_back(prim->tris[i][j]);
        for (int i = 0; i < prim->verts->size(); ++i)
            for (int j = 0; j < 3; ++j)
                verts[i].push_back(prim->verts[i][j]);

        runFTetWild(faces, verts, argc, argv);

        prim->quads.clear();
        prim->quads.resize(faces.size());
        prim->verts.clear();
        prim->verts.resize(verts.size());
        for (int i = 0; i < faces.size(); ++i)
            for (int j = 0; j < 4; ++j)
                prim->quads[i][j] = faces[i][j];
        for (int i = 0; i < verts.size(); ++i)
            for (int j = 0; j < 3; ++j)
                prim->verts[i][j] = verts[i][j];
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(FTetWild)
({
    {{"prim"},
    {"readpath", "input_dir", " "},
    {"string", "output_dir"},
    {"readpath", "tag", " "},
    {"enum none union intersection difference", "operation", "none"},
    {"float", "edge_length", "0"},
    {"float", "epsilon", "0"},
    {"float", "stop_energy", "0"},
    {"bool", "skip_simplify", "0"},
    {"bool", "no_binary", "0"},
    {"bool", "no_color", "0"},
    {"bool", "smooth_open_boundary", "0"},
    {"bool", "export_raw", "0"},
    {"bool", "manifold_surface", "0"},
    {"bool", "coarsen", "0"},
    {"readpath", "csg", " "},
    {"bool", "disable_filtering", "0"},
    {"bool", "use_floodfill", "0"},
    {"bool", "use_general_wn", "0"},
    {"bool", "use_input_for_wn", "0"},
    {"readpath", "bg_mesh", " "}},
    {("prim")},
    {},
    {"primitive"},
});


} // namespace zeno
