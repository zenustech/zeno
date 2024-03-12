#include <cctype>
#include <filesystem>
#include <sstream>
#include <queue>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

#include "./BoundingBox.h"
#include "./SurfaceMesh.h"
#include "./algorithms/SurfaceRemeshing.h"
#include "./algorithms/SurfaceCurvature.h"
#include "./algorithms/TriangleKdTree.h"

#include "zensim/container/Bht.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zeno {

void splitNonManifoldEdges(std::shared_ptr<PrimitiveObject> prim,
                           std::map<std::pair<int, int>, int>& lines_map,
                           std::set<std::pair<int, int>>& marked_lines,
                           std::vector<int>& efeature) {
        // handle non-manifold edges
        auto &pos = prim->attr<vec3f>("pos");
        auto &lines = prim->lines;
        auto &vduplicate = prim->verts.attr<int>("v_duplicate");
        int vert_size = prim->verts.size();
        int line_size = lines.size();
        for (auto &it : prim->tris) {
            std::vector<bool> new_vert(3, false);
            std::vector<bool> new_line(3, false);
            std::vector<bool> marked(3, false);
            auto edge = std::make_pair(it[0], it[1]);
            new_line[0] = (lines_map.count(edge) > 0);
            marked[0] = (marked_lines.count(edge) > 0 || marked_lines.count(std::make_pair(it[1], it[0])) > 0);
            if (new_line[0]) {
                new_vert[0] = new_vert[1] = true;
                efeature[lines_map[edge]] = 1;
            }
            edge = std::make_pair(it[1], it[2]);
            new_line[1] = (lines_map.count(edge) > 0);
            marked[1] = (marked_lines.count(edge) > 0 || marked_lines.count(std::make_pair(it[2], it[1])) > 0);
            if (new_line[1]) {
                new_vert[1] = new_vert[2] = true;
                efeature[lines_map[edge]] = 1;
            }
            edge = std::make_pair(it[2], it[0]);
            new_line[2] = (lines_map.count(edge) > 0);
            marked[2] = (marked_lines.count(edge) > 0 || marked_lines.count(std::make_pair(it[0], it[2])) > 0);
            if (new_line[2]) {
                new_vert[2] = new_vert[0] = true;
                efeature[lines_map[edge]] = 1;
            }

            for (int j = 0; j < 3; ++j) {
                if (new_vert[j]) {
                    pos.push_back(pos[it[j]]);
                    prim->verts.foreach_attr<zeno::AttrAcceptAll>([&] (auto const &key, auto &arr) {
                        arr.push_back(arr[it[j]]);
                    });
                    it[j] = vert_size++;
                }
            }

            for (int j = 0; j < 3; ++j) {
                int line_id;
                bool flag = false;
                if (lines_map.count(std::make_pair(it[j], it[(j + 1) % 3])) > 0) {
                    line_id = lines_map[std::make_pair(it[j], it[(j + 1) % 3])];
                } else if (lines_map.count(std::make_pair(it[(j + 1) % 3], it[j])) > 0) {
                    line_id = lines_map[std::make_pair(it[(j + 1) % 3], it[j])];
                } else {
                    line_id = line_size;
                    flag = true;
                    ++line_size;
                }
                lines_map[std::make_pair(it[j], it[(j + 1) % 3])] = line_id;
                if (flag) {
                    lines.push_back(vec2i(it[j], it[(j + 1) % 3]));
                    if (new_line[j] || marked[j]) {
                        efeature.push_back(1);
                    } else {
                        efeature.push_back(0);
                    }
                }
            }
        }
}

void splitNonManifoldVertices(std::shared_ptr<PrimitiveObject> prim,
                              std::map<std::pair<int, int>, int>& lines_map) {
    // handle non-manifold vertices
    auto &lines = prim->lines;
    auto &faces = prim->tris;
    auto &pos = prim->attr<vec3f>("pos");
    auto &vduplicate = prim->verts.attr<int>("v_duplicate");
    int vert_size = prim->verts.size();
    int line_size = lines.size();
    int tri_size = faces.size();
    auto ef_adj = std::vector<vec2i>(line_size, vec2i(-1));
    auto vf_adj = std::vector<std::set<int>>(vert_size, std::set<int>());
    auto ff_adj = std::vector<vec3i>(tri_size, vec3i(-1));
    for (int f = 0; f < tri_size; ++f) {
        auto face = faces[f];
        for (int i = 0; i < 3; ++i) {
            vf_adj[face[i]].insert(f);
            int line_id = lines_map[std::make_pair(face[i], face[(i + 1) % 3])];
            if (ef_adj[line_id][0] == -1) {
                ef_adj[line_id][0] = f;
            } else {
                ef_adj[line_id][1] = f;
            }
        }
    }
    for (int l = 0; l < line_size; ++l) {
        int f0 = ef_adj[l][0];
        int f1 = ef_adj[l][1];
        if (f1 == -1)
            continue;
        if (ff_adj[f0][0] == -1) {
            ff_adj[f0][0] = f1;
        } else if (ff_adj[f0][1] == -1) {
            ff_adj[f0][1] = f1;
        } else {
            ff_adj[f0][2] = f1;
        }
        if (ff_adj[f1][0] == -1) {
            ff_adj[f1][0] = f0;
        } else if (ff_adj[f1][1] == -1) {
            ff_adj[f1][1] = f0;
        } else {
            ff_adj[f1][2] = f0;
        }
    }
    std::queue<int> q{};
    for (int v = vert_size - 1; v >= 0; --v) {
        int f = -1, next_f = -1, vid = v;
        bool flag = true;
        while (!vf_adj[v].empty()) {
            q.push(*(vf_adj[v].begin()));
            // split a vert
            if (!flag) {
                vid = vert_size++;
                pos.push_back(pos[v]);
                prim->verts.foreach_attr<zeno::AttrAcceptAll>([&] (auto const &key, auto &arr) {
                    arr.push_back(arr[v]);
                });
            }
            while (!q.empty()) {
                next_f = q.front();
                q.pop();
                vf_adj[v].erase(next_f);
                f = next_f;
                next_f = -1;
                if (vid != v) {
                    // modify vert index in faces and lines
                    for (int i = 0, i1 = 1, i2 = 2; i < 3; ++i, i1 = (i1 + 1) % 3, i2 = (i2 + 1) % 3) {
                        if (faces[f][i] == v) {
                            faces[f][i] = vid;
                            int l0 = lines_map[std::make_pair(v, faces[f][i1])];
                            int l1 = lines_map[std::make_pair(faces[f][i2], v)];
                            if (lines[l0][0] == v) {
                                lines[l0][0] = vid;
                            } else if (lines[l0][1] == v) {
                                lines[l0][1] = vid;
                            }
                            if (lines[l1][0] == v) {
                                lines[l1][0] = vid;
                            } else if (lines[l1][1] == v) {
                                lines[l1][1] = vid;
                            }
                            lines_map[std::make_pair(vid, faces[f][i1])] = l0;
                            lines_map[std::make_pair(faces[f][i2], vid)] = l1;
                            break;
                        }
                    }
                }
                for (int i = 0; i < 3; ++i) {
                    if (vf_adj[v].count(ff_adj[f][i]) > 0) {
                        q.push(ff_adj[f][i]);
                    }
                }
            }
            flag = false;
        }
    }
}

void returnNonManifold(std::shared_ptr<PrimitiveObject> prim) {
    // delete duplicate vertices
    auto& pos = prim->attr<vec3f>("pos");
    auto &vduplicate = prim->verts.attr<int>("v_duplicate");
    int vert_size = prim->verts.size();
    int line_size = prim->lines.size();
    int tri_size = prim->tris.size();
    for (int i = 0; i < tri_size; ++i) {
        for (int j = 0; j < 3; ++j) {
            int v = prim->tris[i][j];
            if (vduplicate[v] != v) {
                prim->tris[i][j] = vduplicate[v];
            }
        }
    }
    for (int i = 0; i < line_size; ++i) {
        for (int j = 0; j < 2; ++j) {
            int v = prim->lines[i][j];
            if (vduplicate[v] != v) {
                prim->lines[i][j] = vduplicate[v];
            }
        }
    }

    auto& vmap = prim->verts.add_attr<int>("v_garbage_collection");
    for (int i = 0; i < vert_size; ++i)
        vmap[i] = i;

    int i0 = 0;
    int i1 = prim->verts.size() - 1;
    while (1) {
        while (i0 < i1 && vduplicate[i0] == i0)
            ++i0;
        while (i0 < i1 && vduplicate[i1] != i1)
            --i1;
        if (i0 >= i1)
            break;
        prim->verts.foreach_attr<zeno::AttrAcceptAll>([&] (auto const &key, auto &arr) {
            std::swap(arr[i0], arr[i1]);
        });
        std::swap(pos[i0], pos[i1]);
        ++i0;
        --i1;
    };

    for (int v = 0; v < vert_size; ++v) {
        vduplicate[v] = vmap[vduplicate[v]];
    }
    for (int e = 0; e < line_size; ++e) {
        vec2i old = prim->lines[e];
        prim->lines[e] = vec2i(vmap[old[0]], vmap[old[1]]);
    }
    for (int f = 0; f < tri_size; ++f) {
        vec3i old = prim->tris[f];
        prim->tris[f] = vec3i(vmap[old[0]], vmap[old[1]], vmap[old[2]]);
    }

    prim->verts.erase_attr("v_garbage_collection");

    vert_size = (vduplicate[i0] != i0) ? i0 : i0 + 1;
    prim->verts.foreach_attr<zeno::AttrAcceptAll>([&] (auto const &key, auto &arr) {
        arr.resize(vert_size);
    });
    pos.resize(vert_size);
    
}

struct UniformRemeshing : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto iterations = get_input2<int>("iterations");
        float edge_length = get_input2<float>("edge_length");
        int vert_num = get_input2<int>("vert_num");
        int face_num = get_input2<int>("face_num");
        bool use_min_length = get_input2<bool>("use_min_length");
        auto line_pick_tag = get_input<zeno::StringObject>("line_pick_tag")->get();
        auto &pos = prim->attr<vec3f>("pos");
        zeno::log_info("before remeshing: verts num = {}, face num = {}", prim->verts.size(), prim->tris.size());

        int scale_constraints = 0;
        scale_constraints += (edge_length > 1e-10 || use_min_length) ? 1 : 0;
        scale_constraints += face_num > 0 ? 1 : 0;
        scale_constraints += vert_num > 0 ? 1 : 0;
        if (scale_constraints > 1) {
            zeno::log_error("Only one of the \"edge_length(/use_min_length)\", \"vert_num\" and \"face_num\" parameters can be used at once.");
            set_output("prim", std::move(prim));
            return;
        }

        // init line_pick attribute
        if (!prim->lines.has_attr(line_pick_tag)) {
            prim->lines.add_attr<int>(line_pick_tag, 0);
        }
        auto &efeature = prim->lines.attr<int>(line_pick_tag);

        // init v_duplicate attribute
        auto &vduplicate = prim->verts.add_attr<int>("v_duplicate", 0);
        int vert_size = prim->verts.size();
        for (int i = 0; i < vert_size; ++i) {
            vduplicate[i] = i;
        }

        // if there exist marked lines
        std::set<std::pair<int, int>> marked_lines{};
#if 1
        if (has_input("marked_lines")) {
            const auto &markedLines = get_input<PrimitiveObject>("marked_lines")->lines.values;
            for (vec2i line : markedLines) {
                marked_lines.insert(std::make_pair(line[0], line[1]));
            }
        }
        auto &lines = prim->lines;
        lines.clear();
        efeature.clear();
#elif 0
        /// DEBUG, all perimeter lines preserved
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();
        bht<int, 2, int> tab{prim->tris.size()};
        tab.reset(pol, true);
        pol(prim->tris.values, [tab = proxy<space>(tab)](auto tri) mutable {
            for (int d = 0; d != 3; ++d) {
                auto a = tri[d];
                auto b = tri[(d + 1) % 3];
                tab.insert(zs::vec<int, 2>{a, b});
            }
        });
        auto ne = tab.size();
        std::vector<int> marks(ne);
        pol(zip(range(ne), tab._activeKeys), [&marks, tab = proxy<space>(tab)](int no, auto line) mutable {
            if (tab.query(zs::vec<int, 2>{line[1], line[0]}) < 0) {
                marks[no] = 1;
            }
        });
        for (int i = 0; i != ne; ++i)
            if (marks[i]) {
                auto line = tab._activeKeys[i];
                marked_lines.insert(std::make_pair(line[0], line[1]));
            }
        auto &lines = prim->lines;
        lines.clear();
        efeature.clear();
#else
        auto &lines = prim->lines;
        if (lines.size() > 0) {
            int siz = lines.size();
            for (int i = 0; i < siz; ++i) {
                if (efeature[i] == 1) {
                    marked_lines.insert(std::make_pair(lines[i][0], lines[i][1]));
                }
            }
            lines.clear();
            efeature.clear();
        }
#endif
        std::map<std::pair<int, int>, int> lines_map{};
        splitNonManifoldEdges(prim, lines_map, marked_lines, efeature);
        splitNonManifoldVertices(prim, lines_map);

        auto mesh = new zeno::pmp::SurfaceMesh(prim, line_pick_tag);

        if (vert_num > 0) {
            face_num = vert_num * 2;
        }
        if (face_num > 0) {
            float tot_surface = 0;
            for (auto &it: prim->tris) {
                vec3f l1 = pos[it[0]] - pos[it[1]];
                vec3f l2 = pos[it[0]] - pos[it[2]];
                float s = 0.5f * length(cross(l1, l2));
                tot_surface += s;
            }
            edge_length = std::sqrt(tot_surface / face_num * 4 / std::sqrt(3));
            zeno::log_info("default edge_length: {}", edge_length);
        } else if (edge_length < 1e-10) {
            // If no edge length input,
            // take the average of all edges as default.
            auto &lines = prim->lines;
            for (auto eit : lines) {
                if (use_min_length) {
                    float len = length(pos[eit[0]] - pos[eit[1]]);
                    if (edge_length < 1e-10 || edge_length > len) {
                        edge_length = len;
                    }
                } else {
                    edge_length += length(pos[eit[0]] - pos[eit[1]]);
                }
            }
            if (!use_min_length) {
                edge_length /= (double)lines.size();
            }
            zeno::log_info("default edge_length: {}", edge_length);
        }
        zeno::pmp::SurfaceRemeshing(mesh, line_pick_tag, "v_sizing").uniform_remeshing(edge_length, iterations);

        returnNonManifold(prim);

        // delete v_duplicate at last
        prim->verts.erase_attr("v_duplicate");
        prim->verts.erase_attr("v_normal");
        prim->verts.erase_attr("v_deleted");
        prim->lines.erase_attr("e_deleted");
        prim->tris.erase_attr("f_deleted");
        prim->verts.erase_attr("curv_min");
        prim->verts.erase_attr("curv_max");
        prim->verts.erase_attr("curv_gaussian");
        prim->verts.update();
        prim->lines.clear();
        zeno::log_info("after remeshing: verts num = {}, face num = {}", prim->verts.size(), prim->tris.size());

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(UniformRemeshing)
({
    {{"prim"},
     {"int", "iterations", "10"},
     {"float", "edge_length", "0"},
     {"int", "vert_num", "0"},
     {"int", "face_num", "0"},
     {"bool", "use_min_length", "0"},
     {"string", "line_pick_tag", "line_selected"},
     {"marked_lines"}},
    {"prim"},
    {},
    {"primitive"},
});

struct AdaptiveRemeshing : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto iterations = get_input2<int>("iterations");
        float max_length = get_input2<float>("max_length");
        float min_length = get_input2<float>("min_length");
        float approximation_tolerance = get_input2<float>("approximation_tolerance");
        auto line_pick_tag = get_input<zeno::StringObject>("line_pick_tag")->get();
        auto length_tag = get_input<zeno::StringObject>("length_tag")->get();
        auto &pos = prim->attr<vec3f>("pos");
        zeno::log_info("before remeshing: verts num = {}, face num = {}", prim->verts.size(), prim->tris.size());

        // init line_pick attribute
        if (!prim->lines.has_attr(line_pick_tag)) {
            prim->lines.add_attr<int>(line_pick_tag, 0);
        }
        auto &efeature = prim->lines.attr<int>(line_pick_tag);

        // init v_duplicate attribute
        auto &vduplicate = prim->verts.add_attr<int>("v_duplicate", 0);
        int vert_size = prim->verts.size();
        for (int i = 0; i < vert_size; ++i) {
            vduplicate[i] = i;
        }

        zs::CppTimer timer;

        std::set<std::pair<int, int>> marked_lines{};
#if 1
        if (has_input("marked_lines")) {
            const auto &markedLines = get_input<PrimitiveObject>("marked_lines")->lines.values;
            for (vec2i line : markedLines) {
                marked_lines.insert(std::make_pair(line[0], line[1]));
            }
        }
        auto &lines = prim->lines;
        lines.clear();
        efeature.clear();
#else
        // if there exist marked lines
        auto &lines = prim->lines;
        if (lines.size() > 0) {
            int siz = lines.size();
            for (int i = 0; i < siz; ++i) {
                if (efeature[i] == 1) {
                    marked_lines.insert(std::make_pair(lines[i][0], lines[i][1]));
                }
            }
            lines.clear();
            efeature.clear();
        }
#endif

#if PMP_ENABLE_PROFILE
        timer.tick();
#endif
        std::map<std::pair<int, int>, int> lines_map{};
        splitNonManifoldEdges(prim, lines_map, marked_lines, efeature);
        splitNonManifoldVertices(prim, lines_map);

#if PMP_ENABLE_PROFILE
        timer.tock("handle non-manifold edges");
#endif

        auto mesh = new zeno::pmp::SurfaceMesh(prim, line_pick_tag);
        auto bb = mesh->bounds().size();
        if (max_length < 1e-10) {
            max_length = 0.0500 * bb;
            zeno::log_info("default max_length: {}", max_length);
        }
        if (min_length < 1e-10) {
            min_length = 0.0010 * bb;
            zeno::log_info("default min_length: {}", min_length);
        }
        if (approximation_tolerance < 1e-10) {
            approximation_tolerance = 0.0005 * bb;
            zeno::log_info("default approximation_tolerance: {}", approximation_tolerance);
        }
        zeno::pmp::SurfaceRemeshing(mesh, line_pick_tag, length_tag)
            .adaptive_remeshing(min_length, max_length, approximation_tolerance, iterations);

        returnNonManifold(prim);

        prim->verts.erase_attr("v_duplicate");
        prim->verts.erase_attr("v_normal");
        prim->verts.erase_attr("v_deleted");
        prim->lines.erase_attr("e_deleted");
        prim->tris.erase_attr("f_deleted");
        prim->verts.erase_attr("curv_min");
        prim->verts.erase_attr("curv_max");
        prim->verts.erase_attr("curv_gaussian");
        prim->verts.update();
        prim->lines.clear();
        zeno::log_info("after remeshing: verts num = {}, face num = {}", prim->verts.size(), prim->tris.size());

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(AdaptiveRemeshing)
({
    {{"prim"},
     {"int", "iterations", "10"},
     {"float", "max_length", "0"},
     {"float", "min_length", "0"},
     {"float", "approximation_tolerance", "0"},
     {"string", "line_pick_tag", "line_selected"},
     {"string", "length_tag", "length"},
     {"marked_lines"}},
    {"prim"},
    {},
    {"primitive"},
});

struct RepairDegenerateTriangle : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto iterations = get_input2<int>("iterations");
        float edge_length = get_input2<float>("min_edge_length");
        float area = get_input2<float>("min_area");
        float angle = get_input2<float>("max_angle");
        auto degenerate_tag = get_input<zeno::StringObject>("degenerate_tag")->get();
        bool color = get_input2<float>("color");
        auto &pos = prim->attr<vec3f>("pos");
        auto &efeature = prim->lines.add_attr<int>("e_feature");

        // init v_duplicate attribute
        auto &vduplicate = prim->verts.add_attr<int>("v_duplicate", 0);
        int vert_size = prim->verts.size();
        for (int i = 0; i < vert_size; ++i) {
            vduplicate[i] = i;
        }

        auto &lines = prim->lines;
        lines.clear();
        efeature.clear();

        std::set<std::pair<int, int>> marked_lines{};
        std::map<std::pair<int, int>, int> lines_map{};
        splitNonManifoldEdges(prim, lines_map, marked_lines, efeature);
        splitNonManifoldVertices(prim, lines_map);

        auto mesh = new pmp::SurfaceMesh(prim, "e_feature");

        pmp::SurfaceRemeshing(mesh, "e_feature", "v_sizing")
            .remove_degenerate_triangles(edge_length, area, angle, degenerate_tag, iterations, color);

        returnNonManifold(prim);

        // delete v_duplicate at last
        prim->verts.erase_attr("v_duplicate");
        prim->verts.erase_attr("v_normal");
        prim->verts.erase_attr("v_deleted");
        prim->lines.erase_attr("e_deleted");
        prim->tris.erase_attr("f_deleted");
        prim->verts.update();
        lines.clear();

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(RepairDegenerateTriangle)
({
    {{"prim"},
    {"int", "iterations", "10"},
    {"float", "min_edge_length", "0.001"},
    {"float", "min_area", "0.000001"},
    {"float", "max_angle", "170"},
    {"string", "degenerate_tag", "degenerate"},
    {"bool", "color", "0"}},
    {"prim"},
    {},
    {"primitive"},
});

struct CalcCurvature : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto min_curv_tag = get_input<zeno::StringObject>("min_curv_tag")->get();
        auto max_curv_tag = get_input<zeno::StringObject>("max_curv_tag")->get();
        auto gaussian_curv_tag = get_input<zeno::StringObject>("gaussian_curv_tag")->get();
        auto &pos = prim->attr<vec3f>("pos");
        auto &efeature = prim->lines.add_attr<int>("e_feature", 0);
        auto &vfeature = prim->verts.add_attr<int>("v_feature", 0);

        // init v_duplicate attribute
        auto &vduplicate = prim->verts.add_attr<int>("v_duplicate", 0);
        int vert_size = prim->verts.size();
        for (int i = 0; i < vert_size; ++i) {
            vduplicate[i] = i;
        }

        // if there exist marked lines
        std::set<std::pair<int, int>> marked_lines{};
        auto &lines = prim->lines;
        lines.clear();
        efeature.clear();

        std::map<std::pair<int, int>, int> lines_map{};
        splitNonManifoldEdges(prim, lines_map, marked_lines, efeature);
        splitNonManifoldVertices(prim, lines_map);

        auto mesh = new zeno::pmp::SurfaceMesh(prim, "e_feature");
        zeno::pmp::SurfaceCurvature curv(mesh, min_curv_tag, max_curv_tag, gaussian_curv_tag);
        curv.analyze_tensor(1);

        auto &min_curv = prim->verts.attr<float>(min_curv_tag);
        auto &max_curv = prim->verts.attr<float>(max_curv_tag);
        auto &gaussian_curv = prim->verts.attr<float>(gaussian_curv_tag);
        for (int v = 0; v < vert_size; ++v) {
            if (vduplicate[v] != v) {
                min_curv[v] = min_curv[vduplicate[v]] = -1e5;
                max_curv[v] = max_curv[vduplicate[v]] = 1e5;
                gaussian_curv[v] = gaussian_curv[vduplicate[v]] = -1e7;
            }
        }

        returnNonManifold(prim);

        // delete v_duplicate at last
        prim->verts.erase_attr("v_duplicate");
        prim->verts.erase_attr("v_normal");
        prim->verts.erase_attr("v_deleted");
        prim->lines.erase_attr("e_deleted");
        prim->tris.erase_attr("f_deleted");
        prim->verts.update();

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(CalcCurvature)
({
    {{"prim"},
     {"string", "min_curv_tag", "curv_min"},
     {"string", "max_curv_tag", "curv_max"},
     {"string", "gaussian_curv_tag", "curv_gaussian"}},
    {"prim"},
    {},
    {"primitive"},
});

struct MarkBoundary : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &pos = prim->attr<vec3f>("pos");
        auto &efeature = prim->lines.add_attr<int>("e_feature");
        auto vert_boundary_tag = get_input<zeno::StringObject>("vert_boundary_tag")->get();
        auto edge_boundary_tag = get_input<zeno::StringObject>("edge_boundary_tag")->get();

        // init v_duplicate attribute
        auto &vduplicate = prim->verts.add_attr<int>("v_duplicate", 0);
        int vert_size = prim->verts.size();
        for (int i = 0; i < vert_size; ++i) {
            vduplicate[i] = i;
        }

        // if there exist marked lines
        std::set<std::pair<int, int>> marked_lines{};
        auto &lines = prim->lines;
        lines.clear();
        efeature.clear();

        std::map<std::pair<int, int>, int> lines_map{};
        splitNonManifoldEdges(prim, lines_map, marked_lines, efeature);
        splitNonManifoldVertices(prim, lines_map);

        auto mesh = new zeno::pmp::SurfaceMesh(prim, "e_feature");
        auto &vboundary = prim->verts.add_attr<int>(vert_boundary_tag, 0);
        for (int line_size = lines.size(), e = 0; e < line_size; ++e) {
            if (mesh->is_boundary_e(e)) {
                vboundary[lines[e][0]] = vboundary[lines[e][1]] = 1;
                if (vduplicate[lines[e][0]] != lines[e][0]) {
                    vboundary[vduplicate[lines[e][0]]] = 1;
                }
                if (vduplicate[lines[e][1]] != lines[e][1]) {
                    vboundary[vduplicate[lines[e][1]]] = 1;
                }
            }
        }

        returnNonManifold(prim);
        
        auto &eboundary = prim->lines.add_attr<int>(edge_boundary_tag, 0);
        for (int line_size = lines.size(), e = 0; e < line_size; ++e) {
            eboundary[e] = vboundary[lines[e][0]] && vboundary[lines[e][1]];
        }

        // delete v_duplicate at last
        prim->verts.erase_attr("v_duplicate");
        prim->verts.erase_attr("v_normal");
        prim->verts.erase_attr("v_deleted");
        prim->lines.erase_attr("e_deleted");
        prim->tris.erase_attr("f_deleted");
        prim->verts.update();

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(MarkBoundary)
({
    {{"prim"},
     {"string", "vert_boundary_tag", "v_boundary"},
     {"string", "edge_boundary_tag", "e_boundary"}},
    {"prim"},
    {},
    {"primitive"},
});

bool segTriIntersection(const std::pair<vec3f, vec3f>& seg, const std::vector<vec3f>& tri, float& a, float& b) {
    #define EPSIL 0.000001
    pmp::BoundingBox bb0(std::vector<vec3f>{seg.first, seg.second}), bb1(tri);
    vec3f inter;
    if (!bb0.collide(bb1))
        return false;
    else {
        vec3f dir = seg.second - seg.first;
        dir = normalize(dir);

        enum DIRECTION{RIGHT, LEFT, MIDDLE};
        bool inside = true;
        char quadrant[3];
        vec3f candidatePlane;
        for (int i = 0; i < 3; i++) {
            if(seg.first[i] < bb1.min()[i]) {
                quadrant[i] = LEFT;
                candidatePlane[i] = bb1.min()[i];
                inside = false;
            } else if (seg.first[i] > bb1.max()[i]) {
                quadrant[i] = RIGHT;
                candidatePlane[i] = bb1.max()[i];
                inside = false;
            } else {
                quadrant[i] = MIDDLE;
            }
        }

        if(!inside) {
            vec3f maxT;
            for (int i = 0; i < 3; i++) {
                if (quadrant[i] != MIDDLE && dir[i] !=0.)
                    maxT[i] = (candidatePlane[i]-seg.first[i]) / dir[i];
                else
                    maxT[i] = -1.;
            }
            int whichPlane = 0;
            for (int i = 1; i < 3; i++)
                if (maxT[whichPlane] < maxT[i])
                    whichPlane = i;

            if (maxT[whichPlane] < 0.) return false;
        }

        vec3f edge1 = tri[1] - tri[0];
        vec3f edge2 = tri[2] - tri[0];
        vec3f pvec = cross(dir, edge2);
        float det = dot(edge1, pvec);
        vec3f tvec = seg.first - tri[0];
        float inv_det = 1.0 / det;
        vec3f qvec = cross(tvec, edge1);

        bool intersect = true;
        if (det > EPSIL) {
            a = dot(tvec, pvec);
            if (a < 0.0 || a > det)
            intersect = false;
            b = dot(dir, qvec);
            if (b < 0.0 || a + b > det)
            intersect = false;
        } else if(det < -EPSIL) {
            a = dot(tvec, pvec);
            if (a > 0.0 || a < det)
            intersect = false;
            b = dot(dir, qvec);
            if (b > 0.0 || a + b < det)
            intersect = false;
        } else 
            intersect = false;

        float orig_dist = dot(edge2, qvec) * inv_det;
        a *= inv_det;
        b *= inv_det;

        if(intersect)
            return (orig_dist>=0 && orig_dist<=distance(seg.second, seg.first));
        return false;
    }
}

bool triTriIntersection(const std::vector<vec3f>& p0, const std::vector<vec3f>& p1) {
    vec3f E1 = p0[1] - p0[0];
    vec3f E2 = p0[2] - p0[0];
    vec3f N1 = normalize(cross(E1, E2));
    float d1 = -dot(N1, p0[0]);

    float du0 = dot(N1, p1[0]) + d1;
    float du1 = dot(N1, p1[1]) + d1;
    float du2 = dot(N1, p1[2]) + d1;
    float du0du1 = du0 * du1;
    float du0du2 = du0 * du2;
    if(du0du1>0.0f && du0du2>0.0f)
        return 0;

    E1 = p1[1] - p1[0];
    E2 = p1[2] - p1[0];
    vec3f N2 = normalize(cross(E1, E2));
    float d2 = -dot(N2, p1[0]);

    float dv0 = dot(N2, p0[0]) + d2;
    float dv1 = dot(N2, p0[1]) + d2;
    float dv2 = dot(N2, p0[2]) + d2;
    float dv0dv1 = dv0 * dv1;
    float dv0dv2 = dv0 * dv2;
    if(dv0dv1>0.0f && dv0dv2>0.0f)
        return 0;

    vec3f D = cross(N1, N2);
    short index = 0;
    float max = std::fabs(D[0]);
    float bb = std::fabs(D[1]);
    float cc = std::fabs(D[2]);
    if(bb>max) max=bb,index=1;
    if(cc>max) max=cc,index=2;

    float vp0 = p0[0][index];
    float vp1 = p0[1][index];
    float vp2 = p0[2][index];
    float up0 = p1[0][index];
    float up1 = p1[1][index];
    float up2 = p1[2][index];

    float a,b,c,x0,x1;
    if(dv0dv1>0.0f) {
        a=vp2; b=(vp0-vp2)*dv2; c=(vp1-vp2)*dv2; x0=dv2-dv0; x1=dv2-dv1;
    } else if(dv0dv2>0.0f) {
        a=vp1; b=(vp0-vp1)*dv1; c=(vp2-vp1)*dv1; x0=dv1-dv0; x1=dv1-dv2;
    } else if(dv1*dv2>0.0f || dv0!=0.0f) {
        a=vp0; b=(vp1-vp0)*dv0; c=(vp2-vp0)*dv0; x0=dv0-dv1; x1=dv0-dv2;
    } else if(dv1!=0.0f) {
        a=vp1; b=(vp0-vp1)*dv1; c=(vp2-vp1)*dv1; x0=dv1-dv0; x1=dv1-dv2;
    } else if(dv2!=0.0f) {
        a=vp2; b=(vp0-vp2)*dv2; c=(vp1-vp2)*dv2; x0=dv2-dv0; x1=dv2-dv1;
    } else {
        // shouldn't reach here
    }

    float d,e,f,y0,y1;
    if(du0du1>0.0f) {
        d=up2; e=(up0-up2)*du2; f=(up1-up2)*du2; y0=du2-du0; y1=du2-du1;
    } else if(du0du2>0.0f) {
        d=up1; e=(up0-up1)*du1; f=(up2-up1)*du1; y0=du1-du0; y1=du1-du2;
    } else if(du1*du2>0.0f || du0!=0.0f) {
        d=up0; e=(up1-up0)*du0; f=(up2-up0)*du0; y0=du0-du1; y1=du0-du2;
    } else if(du1!=0.0f) {
        d=up1; e=(up0-up1)*du1; f=(up2-up1)*du1; y0=du1-du0; y1=du1-du2;
    } else if(du2!=0.0f) {
        d=up2; e=(up0-up2)*du2; f=(up1-up2)*du2; y0=du2-du0; y1=du2-du1;
    } else {
        // shouldn't reach here
    }

    float xx = x0 * x1;
    float yy = y0 * y1;
    float xxyy = xx * yy;

    float isect1[2], isect2[2];
    float tmp = a * xxyy;
    isect1[0] = tmp + b * x1 * yy;
    isect1[1] = tmp + c * x0 * yy;

    tmp = d * xxyy;
    isect2[0] = tmp + e * xx * y1;
    isect2[1] = tmp + f * xx * y0;

    if (isect1[0] > isect1[1])
        std::swap(isect1[0], isect1[1]);
    if (isect2[0] > isect2[1])
        std::swap(isect2[0], isect2[1]);

    if(isect1[1]<isect2[0] || isect2[1]<isect1[0]) return 0;
    return 1;
}

bool faceFaceIntersection(const vec3i& f0, const vec3i& f1, const std::vector<vec3f>& p0, const std::vector<vec3f>& p1) {
    int sv = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (f0[i] == f1[j])
                    sv++;
    if(sv==3) return true;
    if(sv==0) return (triTriIntersection(p0, p1));
    if(sv==1) {
        int i0, i1;
        bool flag = false;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                if (f0[i] == f1[j]) {
                    i0 = i, i1 = j;
                    flag = true;
                    break;
                }
            if (flag) break;
        }
        float a, b;
        vec3f ps[5] = {p0[i0], p0[(i0+1)%3], p0[(i0+2)%3], p1[(i1+1)%3], p1[(i1+2)%3]};
        for (int i = 0; i < 5; ++i) {
            ps[i] = 0.5f * ps[i];
            if (i > 0)
                ps[i] = ps[i] + ps[0];
        }
        if(segTriIntersection(std::make_pair(ps[1], ps[2]), p1, a, b)) {
            if(a+b>=1 || a<=EPSIL || b<=EPSIL ) return false;
            return true;
        }
        if(segTriIntersection(std::make_pair(ps[3], ps[4]), p0, a, b)) {
            if(a+b>=1 || a<=EPSIL || b<=EPSIL ) return false;
            return true;
        }

    }
    return false;
}

struct SelectIntersectingFaces : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto intersect_tag = get_input<zeno::StringObject>("intersecting_tag")->get();
        bool color = get_input2<float>("color");
        auto &verts = prim->verts;
        auto &faces = prim->tris;
        auto &referred = prim->tris.add_attr<int>("referred", 0);
        auto &intersecting = prim->tris.add_attr<int>(intersect_tag, 0);

        auto kdtree = pmp::TriangleKdTree(faces, verts);
        for (int i = 0; i < faces->size(); ++i) {
            auto fi = faces[i];
            referred[i] = 1;
            std::vector<vec3f> pfi{verts[fi[0]], verts[fi[1]], verts[fi[2]]};
            pmp::BoundingBox bbox(pfi);
            std::vector<int> inBox{};
            kdtree.faces_in_box(bbox, inBox);
            for(auto fib: inBox) {
                std::vector<vec3f> pfib{verts[faces[fib][0]], verts[faces[fib][1]], verts[faces[fib][2]]};
                if(referred[fib] == 0 && (fib != i) )
                    if(faceFaceIntersection(fi, faces[fib], pfi, pfib))
                        intersecting[i] = intersecting[fib] = 1;
            }
            inBox.clear();
        }
        prim->tris.erase_attr("referred");

        if (color) {
            auto &clr = prim->verts.add_attr<vec3f>("clr", vec3f(0.1, 0.6, 0.4));
            for (int i = 0; i < faces->size(); ++i) {
                if (intersecting[i] == 1) {
                    clr[faces[i][0]] = clr[faces[i][1]] = clr[faces[i][2]] = vec3f(0.8, 0.3, 0.1);
                }
            }
        }

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(SelectIntersectingFaces)
({
    {{"prim"},
    {"string", "intersecting_tag", "intersecting"},
    {"bool", "color", "0"}},
    {("prim")},
    {},
    {"primitive"},
});


} // namespace zeno
