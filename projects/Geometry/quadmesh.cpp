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
        auto prim = get_input<PrimitiveObject>("prim");
        float crease = get_input2<float>("crease");
        bool deterministic = get_input2<bool>("deterministic");
        int smooth = get_input2<int>("smooth_iter");
        bool dominant = get_input2<bool>("quad_dominant");
        bool intrinsic = get_input2<bool>("intrinsic");
        bool boundary = get_input2<bool>("boundary");
        float scale = get_input2<float>("scale");
        int vert_num = get_input2<int>("vert_num");
        int face_num = get_input2<int>("face_num");
        // int knn = get_input2<int>("knn");
        auto line_pick_tag = get_input<zeno::StringObject>("line_pick_tag")->get();

        if (prim->verts.size() < 4) {
            set_output("prim", std::move(prim));
            return;
        }

        std::set<std::pair<int, int>> marked_lines{};
        if (has_input("marked_lines")) {
            const auto &markedLines = get_input<PrimitiveObject>("marked_lines")->lines.values;
            for (vec2i line : markedLines) {
                marked_lines.insert(std::make_pair(line[0], line[1]));
            }
        }
        auto &lines = prim->lines;
        if (lines.has_attr(line_pick_tag)) {
            auto &efeature = lines.attr<int>(line_pick_tag);
            for (int i = 0; i < lines.size(); ++i) {
                if (efeature[i] == 1)
                    marked_lines.insert(std::make_pair(lines[i][0], lines[i][1]));
            }
        }
        if (marked_lines.size() > 0)
            boundary = true;

        int scale_constraints = 0;
        scale_constraints += scale > 0 ? 1 : 0;
        scale_constraints += face_num > 0 ? 1 : 0;
        scale_constraints += vert_num > 0 ? 1 : 0;
        if (scale_constraints > 1) {
            zeno::log_error("Only one of the \"scale\", \"vert_num\" and \"face_num\" parameters can be used at once.");
            set_output("prim", std::move(prim));
            return;
        }

        int argc = 1;
        char* argv[20];
        argv[0] = (char*)malloc(sizeof("./InstantMeshes\0"));
        strcpy(argv[0], "./InstantMeshes\0");
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
            if (dominant == 0) {
                scale *= 2;
            }
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
            if (dominant == 0) {
                vert_num /= 4;
            }
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
            if (dominant == 0) {
                face_num /= 4;
            }
            std::string face_num_str = to_string(face_num);
            argv[argc+1] = (char*)malloc((face_num_str.size()+1)*sizeof(char));
            for (int i = 0; i < face_num_str.size(); ++i)
                argv[argc+1][i] = face_num_str[i];
            argv[argc+1][face_num_str.size()] = '\0';
            argc += 2;
        }
        // if (knn > 0) {
        //     argv[argc] = (char*)malloc(sizeof("--knn\0"));
        //     strcpy(argv[argc], "--knn\0");
        //     std::string knn_str = to_string(knn);
        //     argv[argc+1] = (char*)malloc((knn_str.size()+1)*sizeof(char));
        //     for (int i = 0; i < knn_str.size(); ++i)
        //         argv[argc+1][i] = knn_str[i];
        //     argv[argc+1][knn_str.size()] = '\0';
        //     argc += 2;
        // }
        
        std::vector<std::vector<int>> faces(prim->tris->size(), std::vector<int>{});
        std::vector<std::vector<float>> verts(prim->verts->size(), std::vector<float>{});
        std::vector<std::vector<int>> features(prim->tris->size(), std::vector<int>{});
        auto &pos = prim->verts;
        for (int i = 0; i < prim->tris->size(); ++i) {
            for (int j = 0; j < 3; ++j) {
                int i0 = prim->tris[i][j], i1 = prim->tris[i][(j+1)%3];
                faces[i].push_back(i0);
                if (marked_lines.count(std::make_pair(i0, i1)) > 0 ||
                    marked_lines.count(std::make_pair(i1, i0)) > 0) {
                    features[i].push_back(1);
                } else {
                    features[i].push_back(0);
                }
            }
        }
        for (int i = 0; i < pos.size(); ++i)
            for (int j = 0; j < 3; ++j)
                verts[i].push_back(prim->verts[i][j]);

        runInstantMeshes(faces, verts, features, argc, argv);

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

ZENO_DEFNODE(QuadMesh)
({
    {{"prim"},
    {"bool", "deterministic", "0"},
    {"float", "crease", "0"},
    {"int", "smooth_iter", "0"},
    {"bool", "quad_dominant", "0"},
    {"bool", "intrinsic", "0"},
    {"bool", "boundary", "0"},
    {"float", "scale", "0"},
    {"int", "vert_num", "0"},
    {"int", "face_num", "0"},
    // {"int", "knn", "0"},
    {"string", "line_pick_tag", "line_selected"},
    {"marked_lines"}},
    {{"prim"}},
    {},
    {"primitive"},
});


} // namespace zeno
